"""Layered recursive digests over the memory store (day → … → year).

Adds a lossless "summary layer" on top of the raw fragments: every finished
natural period gets one extra memory row that narrates it. Source fragments are
NEVER touched — no digested flag, no deletion, no rewrite. That is the whole
difference from consolidate_sessions.py, which merges lossily and marks its
sources digested=1 (and those sources get hard-deleted after DIGESTED_PRUNE_DAYS).

Layers and keys:
    [日记] 2026-08-29    local calendar day
    [周记] 2026-W35      ISO week (Monday-start, America/Toronto local time)
    [半月记] 2026-08上   1st–15th; 下 = 16th–end of month
    [月记] 2026-08
    [季记] 2026-Q3
    [半年记] 2026-H2
    [年记] 2026

Two separate relationships run between the layers, and they deliberately do not
line up:

  * INPUT_CHILD — who reads whom. half-month reads days, month reads weeks,
    quarter reads months, half reads quarters, year reads halves. Weeks read raw
    fragments only, and months keep reading weeks rather than half-months: a
    week straddles the 15th/16th boundary, so feeding half-months into the month
    layer would double-count the straddling days and produce cross-boundary
    nonsense. Half-months hang off the day layer as their own short branch.
  * ABSORB_CHILD — who archives whom once written. Same pairs, plus week→day
    (a week archives its 7 day digests even though it does not read them), minus
    half-month (which absorbs nothing and simply ages out of watch on its own).

Anti-telephone-game mixing: a parent digest reads its children's full text *plus*
that period's top-importance raw entries, so drift introduced by one summarisation
round gets corrected against the originals.

Absorb = archive: tier='archive' (searchable, no longer surfacing in breath),
tier_until cleared. Raw fragments are never re-tiered.

Usage:
    python digest_rollup.py --dry-run              # what would be generated
    python digest_rollup.py                        # daily run: yesterday's 日记 + whatever else came due
    python digest_rollup.py --backfill-all         # no per-layer cap (day/year still windowed, see below)
    python digest_rollup.py --since 2026-06-01     # ignore anything older
    python digest_rollup.py --force --since 2026-08-01   # rewrite existing keys

Three layers carry their own enumeration window on top of --backfill, because
"every finished period since the store began" is the wrong default for them:

    day        only the last --day-lookback days (default 3). NOT lifted by
               --backfill-all — backfilling a year of days is hundreds of LLM
               calls for material the week digests already cover. Raise
               --day-lookback deliberately if you really want older ones.
    halfmonth  only the last --halfmonth-recent finished periods (default 1);
               --backfill-all lifts this and walks the whole history.
    year       only years >= --year-min (default 2026): 2025 is a partial year in
               this store, and 2026 will generate itself on 2027-01-01.

Safe to re-run: existing keys are skipped unless --force. Periods still in
progress are never generated. A period whose LLM call fails is left missing and
picked up by the next run.

Requires the `claude` CLI on PATH (override with --claude-cmd). Standard library only.
"""
from __future__ import annotations

import argparse
import random
import shutil
import sqlite3
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone, tzinfo
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Every row this script writes carries this session_id. It is the marker used to
# (a) find our own digests for idempotency and cascading, and (b) keep digests
# out of their own input sets — otherwise a --force re-run would feed a period's
# digest back into itself.
DIGEST_SESSION_ID = "digest_rollup"

# Appended by *code*, never by the model — an LLM asked to echo memory ids will
# happily invent them.
SOURCE_PREFIX = "\n——源记忆："
SOURCE_MAX_SHOWN = 8

# Measured on this box: haiku 25-45s on a 20-90k-character week, sonnet 150-240s
# on a month, opus ~70s on a small quarter. 300s (the original design figure) sits
# right on top of the sonnet numbers, so a normal month would start timing out and
# burning its retry; 600 leaves headroom. Lower it with --llm-timeout if a stuck
# call should be abandoned sooner.
LLM_TIMEOUT = 600.0          # seconds per claude call
LLM_ATTEMPTS = 2             # initial call + one retry

# Layer configuration. `cn` doubles as the key prefix: "[周记] ".
# `unit` fills "这一{unit}" / "一{unit}"; `ref` fills the "本X" possessive slots
# ("本周的周记") — kept separate because "本天" is not Chinese.
# LAYERS is also the *execution order*: every layer must come after everything it
# reads and everything it archives.
LAYERS = ("day", "week", "halfmonth", "month", "quarter", "half", "year")
LAYER_CFG = {
    "day":       {"cn": "日记",   "unit": "天",   "ref": "这一天", "importance": 0.45, "max_chars": 400,  "model": "haiku",  "watch_days": 2},
    "week":      {"cn": "周记",   "unit": "周",   "ref": "本周",   "importance": 0.55, "max_chars": 600,  "model": "haiku",  "watch_days": 8},
    "halfmonth": {"cn": "半月记", "unit": "半月", "ref": "本半月", "importance": 0.60, "max_chars": 700,  "model": "sonnet", "watch_days": 10},
    "month":     {"cn": "月记",   "unit": "月",   "ref": "本月",   "importance": 0.65, "max_chars": 800,  "model": "sonnet", "watch_days": 12},
    "quarter":   {"cn": "季记",   "unit": "季度", "ref": "本季度", "importance": 0.75, "max_chars": 1000, "model": "opus",   "watch_days": 16},
    "half":      {"cn": "半年记", "unit": "半年", "ref": "本半年", "importance": 0.85, "max_chars": 1200, "model": "opus",   "watch_days": 20},
    "year":      {"cn": "年记",   "unit": "年",   "ref": "本年",   "importance": 0.95, "max_chars": 1500, "model": "opus",   "watch_days": 30},
}
# Which layer feeds which — quarter reads months, half reads quarters. Layers
# absent as keys (day, week) read raw fragments only.
INPUT_CHILD = {"halfmonth": "day", "month": "week", "quarter": "month",
               "half": "quarter", "year": "half"}
# Which layer archives which once written. Same as INPUT_CHILD except: week
# archives the days it covers without reading them (the week is written from raw
# fragments, but the day digests underneath it have served their purpose), and
# halfmonth archives nothing — its own watch window expiring is its exit.
ABSORB_CHILD = {"week": "day", "month": "week", "quarter": "month",
                "half": "quarter", "year": "half"}
# Layers that postpone themselves while a child digest is still missing. The
# half-month is excluded on purpose: day digests are not backfilled, so an older
# half-month would wait forever instead of falling back to raw fragments.
WAIT_CHILDREN = frozenset({"month", "quarter", "half", "year"})

DEFAULT_MIN_ITEMS = 5        # below this, skip the period instead of padding it
DEFAULT_MIN_ITEMS_DAY = 3    # a quiet day is still worth one paragraph
DEFAULT_TOP_RAW = 10         # top-importance raw entries mixed into parent layers
DEFAULT_TOP_RAW_YEAR = 15    # a year gets a few more, it has only 2 children to mix against
DEFAULT_BACKFILL = 4         # missing periods generated per layer per run
DEFAULT_MAX_ENTRY_CHARS = 1200
DEFAULT_MAX_INPUT_CHARS = 120000

# Per-layer enumeration windows — see the module docstring for why each exists.
DEFAULT_DAY_LOOKBACK = 3     # finished days considered per run (catches up a 2-day outage)
DEFAULT_HALFMONTH_RECENT = 1 # finished half-months considered per run unless --backfill-all
DEFAULT_YEAR_MIN = 2026      # 2025 is a partial year in this store


# ---------------------------------------------------------------------------
# Timezone — America/Toronto, with a stdlib-only fallback
# ---------------------------------------------------------------------------

class _TorontoFallback(tzinfo):
    """US/Canada Eastern DST rules as of 2007: EST = UTC-5, EDT = UTC-4,
    DST from the 2nd Sunday of March 02:00 local to the 1st Sunday of
    November 02:00 local.

    Only used if zoneinfo cannot load America/Toronto (Windows needs the
    `tzdata` package; it is present on Sol's box, but a fresh Python would not
    have it and this script must not die over a missing wheel). Wrong for dates
    before 2007, which predate the memory store by two decades.
    """

    _STD = timedelta(hours=-5)
    _DST = timedelta(hours=-4)

    @staticmethod
    def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
        d = date(year, month, 1)
        offset = (weekday - d.weekday()) % 7
        return d + timedelta(days=offset + 7 * (n - 1))

    def _is_dst(self, dt: datetime) -> bool:
        y = dt.year
        start = datetime.combine(self._nth_weekday(y, 3, 6, 2), datetime.min.time()) + timedelta(hours=2)
        end = datetime.combine(self._nth_weekday(y, 11, 6, 1), datetime.min.time()) + timedelta(hours=2)
        naive = dt.replace(tzinfo=None)
        return start <= naive < end

    def utcoffset(self, dt):
        if dt is None:
            return self._STD
        return self._DST if self._is_dst(dt) else self._STD

    def dst(self, dt):
        if dt is None:
            return timedelta(0)
        return timedelta(hours=1) if self._is_dst(dt) else timedelta(0)

    def tzname(self, dt):
        if dt is None:
            return "EST"
        return "EDT" if self._is_dst(dt) else "EST"


def load_tz(name: str = "America/Toronto") -> tzinfo:
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(name)
    except Exception as exc:  # noqa: BLE001 — any failure means "no tz database"
        sys.stderr.write(
            f"[warn] zoneinfo 加载 {name} 失败（{type(exc).__name__}: {exc}），"
            f"改用内置 Eastern 时区规则（2007 年后有效）。装 `pip install tzdata` 可消除本警告。\n"
        )
        return _TorontoFallback()


# ---------------------------------------------------------------------------
# Period model
# ---------------------------------------------------------------------------

class Period:
    """One natural time window, resolved to UTC string bounds for SQL.

    created_at in the store is UTC ISO8601 (`2026-08-29T21:07:03.950601+00:00`),
    with a handful of legacy rows lacking microseconds or the offset. Comparing
    against a bare 19-char `YYYY-MM-DDTHH:MM:SS` bound is correct for all three
    shapes: any longer timestamp with the same prefix sorts *after* the bound,
    so a row sitting exactly on a boundary lands in the later period, and the
    idx_memories_created index still gets used.
    """

    __slots__ = ("anchor", "end_s", "end_utc", "key", "label", "layer", "start_s", "start_utc")

    def __init__(self, layer: str, label: str, start_utc: datetime, end_utc: datetime, anchor: date):
        self.layer = layer
        self.label = label                       # e.g. "2026-W35"
        self.key = f"[{LAYER_CFG[layer]['cn']}] {label}"
        self.start_utc = start_utc
        self.end_utc = end_utc
        self.start_s = start_utc.strftime("%Y-%m-%dT%H:%M:%S")
        self.end_s = end_utc.strftime("%Y-%m-%dT%H:%M:%S")
        self.anchor = anchor                     # a local date inside the period

    def __repr__(self) -> str:
        return f"<Period {self.key} {self.start_s}..{self.end_s}>"

    def local_range_text(self, tz: tzinfo) -> str:
        a = self.start_utc.astimezone(tz).date()
        b = (self.end_utc - timedelta(seconds=1)).astimezone(tz).date()
        if a == b:
            return f"{a.isoformat()}，多伦多本地时间"
        return f"{a.isoformat()} 至 {b.isoformat()}，多伦多本地时间"

    def stamp_utc(self) -> str:
        """created_at for the digest row: the last second of the period it covers.

        Anchoring to the period (rather than to generation time) keeps a
        backfilled 2025 digest aged like 2025 material instead of erupting into
        breath as if it were written today.
        """
        return (self.end_utc - timedelta(seconds=1)).isoformat()


def _local_midnight(d: date, tz: tzinfo) -> datetime:
    """Local 00:00 of date *d*, as an aware UTC datetime.

    Period boundaries are always local midnight on a Monday / the 1st of a
    month, and North American DST transitions happen at 02:00 on a Sunday, so
    no boundary here is ever ambiguous or nonexistent.
    """
    return datetime(d.year, d.month, d.day, tzinfo=tz).astimezone(timezone.utc)


def _month_add(y: int, m: int, delta: int) -> tuple[int, int]:
    idx = (y * 12 + (m - 1)) + delta
    return idx // 12, idx % 12 + 1


def day_period(d: date, tz: tzinfo) -> Period:
    """Day = local midnight → next local midnight."""
    return Period("day", d.isoformat(),
                  _local_midnight(d, tz), _local_midnight(d + timedelta(days=1), tz),
                  d)


def week_period(monday: date, tz: tzinfo) -> Period:
    """Week = local Monday 00:00 → next Monday 00:00.

    Labelled by ISO 8601 week number. The week's Thursday decides its ISO year
    (and, further down, which month/quarter/half it is filed under) — that is
    the ISO rule for a week straddling a boundary: a week belongs to whichever
    year/month contains the majority of its days, and the Thursday is exactly
    the majority marker.
    """
    thursday = monday + timedelta(days=3)
    iso = thursday.isocalendar()
    return Period(
        "week", f"{iso[0]}-W{iso[1]:02d}",
        _local_midnight(monday, tz), _local_midnight(monday + timedelta(days=7), tz),
        thursday,
    )


def halfmonth_period(y: int, m: int, h: int, tz: tzinfo) -> Period:
    """Half-month: h=1 is the 1st–15th, h=2 is the 16th to the end of the month.

    Split on a fixed day-of-month rather than on the midpoint so the label stays
    predictable: 上 is always 15 days, 下 is 13/14/15/16 depending on the month.
    """
    if h == 1:
        start, end = date(y, m, 1), date(y, m, 16)
    else:
        ny, nm = _month_add(y, m, 1)
        start, end = date(y, m, 16), date(ny, nm, 1)
    return Period("halfmonth", f"{y}-{m:02d}{'上' if h == 1 else '下'}",
                  _local_midnight(start, tz), _local_midnight(end, tz), start)


def month_period(y: int, m: int, tz: tzinfo) -> Period:
    ny, nm = _month_add(y, m, 1)
    return Period("month", f"{y}-{m:02d}",
                  _local_midnight(date(y, m, 1), tz), _local_midnight(date(ny, nm, 1), tz),
                  date(y, m, 1))


def quarter_period(y: int, q: int, tz: tzinfo) -> Period:
    start_m = (q - 1) * 3 + 1
    ny, nm = _month_add(y, start_m, 3)
    return Period("quarter", f"{y}-Q{q}",
                  _local_midnight(date(y, start_m, 1), tz), _local_midnight(date(ny, nm, 1), tz),
                  date(y, start_m, 1))


def half_period(y: int, h: int, tz: tzinfo) -> Period:
    start_m = 1 if h == 1 else 7
    ny, nm = _month_add(y, start_m, 6)
    return Period("half", f"{y}-H{h}",
                  _local_midnight(date(y, start_m, 1), tz), _local_midnight(date(ny, nm, 1), tz),
                  date(y, start_m, 1))


def year_period(y: int, tz: tzinfo) -> Period:
    return Period("year", f"{y}",
                  _local_midnight(date(y, 1, 1), tz), _local_midnight(date(y + 1, 1, 1), tz),
                  date(y, 1, 1))


def period_containing(layer: str, d: date, tz: tzinfo) -> Period:
    """The period of *layer* that the local date *d* falls inside.

    Single source of truth for "which bucket does this day belong to" — both the
    enumeration walk and the parent→child mapping go through it, so a week can
    never be enumerated one way and looked up another.
    """
    if layer == "day":
        return day_period(d, tz)
    if layer == "week":
        return week_period(d - timedelta(days=d.weekday()), tz)
    if layer == "halfmonth":
        return halfmonth_period(d.year, d.month, 1 if d.day <= 15 else 2, tz)
    if layer == "month":
        return month_period(d.year, d.month, tz)
    if layer == "quarter":
        return quarter_period(d.year, (d.month - 1) // 3 + 1, tz)
    if layer == "half":
        return half_period(d.year, 1 if d.month <= 6 else 2, tz)
    if layer == "year":
        return year_period(d.year, tz)
    raise ValueError(f"unknown layer: {layer}")


def next_period(period: Period, tz: tzinfo) -> Period:
    """The period immediately after *period*.

    period.end_utc is, by construction, local midnight of the first day of the
    next period — so converting it back to a local date and re-bucketing lands
    exactly one period forward, for every layer, without per-layer arithmetic.
    """
    return period_containing(period.layer, period.end_utc.astimezone(tz).date(), tz)


def enumerate_periods(layer: str, since_local: date, now_utc: datetime, tz: tzinfo) -> list[Period]:
    """All *finished* periods of *layer* from the one containing since_local
    onwards. A period counts as finished once its end instant has passed."""
    out: list[Period] = []
    p = period_containing(layer, since_local, tz)
    while p.end_utc <= now_utc:
        out.append(p)
        p = next_period(p, tz)
    return out


def sub_periods(period: Period, child_layer: str, tz: tzinfo) -> list[Period]:
    """The *child_layer* periods filed under *period*, chronologically.

    Membership is decided by the child's anchor, not by overlap. For a week the
    anchor is its Thursday, which is the ISO rule for a week straddling a month
    boundary (the week belongs to whichever month holds the majority of its days,
    and the Thursday marks that majority). Every other layer anchors on its own
    first day, so nothing straddles: days, months, quarters and halves nest
    exactly inside their parents.
    """
    first_local = period.start_utc.astimezone(tz).date()
    last_local = (period.end_utc - timedelta(seconds=1)).astimezone(tz).date()
    # Start a week early so a week whose Thursday lands in this period but whose
    # Monday sits in the previous one still gets considered.
    p = period_containing(child_layer, first_local - timedelta(days=7), tz)
    out: list[Period] = []
    while p.anchor <= last_local + timedelta(days=7):
        if first_local <= p.anchor <= last_local:
            out.append(p)
        p = next_period(p, tz)
    return out


# ---------------------------------------------------------------------------
# DB access
# ---------------------------------------------------------------------------

def connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=10000")
    return conn


def earliest_created(conn: sqlite3.Connection) -> str:
    row = conn.execute(
        "SELECT MIN(created_at) FROM memories WHERE memory_kind='long_term'"
    ).fetchone()
    return row[0] or ""


def fetch_raw(conn: sqlite3.Connection, period: Period, *, order: str, limit: int = 0,
              exclude_tiers: tuple[str, ...] = ()) -> list[dict]:
    """Raw (non-digest) entries whose created_at falls inside the period.

    Digest rows are excluded by session_id so a period can never summarise
    itself; consolidate merge-products ARE included — they are legitimate
    memories of that period, just already-merged ones.
    """
    sql = ("SELECT id, key, content, category, importance, valence, arousal, created_at, tier "
           "FROM memories WHERE memory_kind='long_term' AND digested=0 "
           "AND session_id != ? AND created_at >= ? AND created_at < ?")
    params: list = [DIGEST_SESSION_ID, period.start_s, period.end_s]
    if exclude_tiers:
        sql += " AND tier NOT IN ({})".format(",".join("?" * len(exclude_tiers)))
        params.extend(exclude_tiers)
    sql += " ORDER BY importance DESC, created_at ASC" if order == "importance" else " ORDER BY created_at ASC"
    if limit > 0:
        sql += f" LIMIT {int(limit)}"
    rows = conn.execute(sql, params).fetchall()
    return [
        {"id": r[0], "key": r[1], "content": r[2], "category": r[3],
         "importance": float(r[4] or 0.5), "valence": float(r[5] or 0.5),
         "arousal": float(r[6] or 0.3), "created_at": r[7], "tier": r[8] or ""}
        for r in rows
    ]


def raw_count(conn: sqlite3.Connection, period: Period, exclude_tiers: tuple[str, ...] = ()) -> int:
    sql = ("SELECT COUNT(*) FROM memories WHERE memory_kind='long_term' AND digested=0 "
           "AND session_id != ? AND created_at >= ? AND created_at < ?")
    params: list = [DIGEST_SESSION_ID, period.start_s, period.end_s]
    if exclude_tiers:
        sql += " AND tier NOT IN ({})".format(",".join("?" * len(exclude_tiers)))
        params.extend(exclude_tiers)
    return int(conn.execute(sql, params).fetchone()[0] or 0)


def fetch_digests(conn: sqlite3.Connection, keys: list[str]) -> list[dict]:
    """Our own digest rows for the given keys, in the order the keys were given.

    Deliberately does not filter on tier: a week already archived by an earlier
    month run is still the right input when that month is regenerated.
    """
    if not keys:
        return []
    placeholders = ",".join("?" * len(keys))
    rows = conn.execute(
        f"SELECT id, key, content, importance, created_at FROM memories "
        f"WHERE session_id = ? AND key IN ({placeholders}) ORDER BY key ASC",
        (DIGEST_SESSION_ID, *keys),
    ).fetchall()
    by_key = {r[1]: {"id": r[0], "key": r[1], "content": r[2],
                     "importance": float(r[3] or 0.5), "created_at": r[4]} for r in rows}
    return [by_key[k] for k in keys if k in by_key]


def existing_digest_id(conn: sqlite3.Connection, key: str) -> str:
    row = conn.execute(
        "SELECT id FROM memories WHERE session_id = ? AND key = ? ORDER BY created_at ASC LIMIT 1",
        (DIGEST_SESSION_ID, key),
    ).fetchone()
    return row[0] if row else ""


def period_emotion(conn: sqlite3.Connection, period: Period) -> tuple[float, float]:
    """Importance-weighted valence/arousal over the period's raw entries —
    the same aggregation consolidate_sessions.write_merged does over its
    fragments, so digests carry a comparable emotional colour."""
    rows = conn.execute(
        "SELECT importance, valence, arousal FROM memories "
        "WHERE memory_kind='long_term' AND digested=0 AND session_id != ? "
        "AND created_at >= ? AND created_at < ?",
        (DIGEST_SESSION_ID, period.start_s, period.end_s),
    ).fetchall()
    if not rows:
        return 0.5, 0.3
    weights = [max(0.01, float(r[0] or 0.5)) for r in rows]
    wsum = sum(weights) or 1.0
    v = sum(float(r[1] or 0.5) * w for r, w in zip(rows, weights)) / wsum
    a = sum(float(r[2] or 0.3) * w for r, w in zip(rows, weights)) / wsum
    return round(v, 3), round(a, 3)


def new_digest_id() -> str:
    return (f"mem_digest_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
            f"_{random.randint(1000, 9999)}")


def write_digest(conn: sqlite3.Connection, period: Period, content: str, *,
                 category: str, valence: float, arousal: float,
                 absorb_ids: list[str], existing_id: str = "") -> str:
    """Insert (or, with --force, rewrite) the digest row and archive what it
    absorbed — one short transaction, so a crash mid-way leaves neither a
    digest without its archiving nor archived children without their digest."""
    now = datetime.now(timezone.utc).isoformat()
    cfg = LAYER_CFG[period.layer]
    stamp = period.stamp_utc()
    mem_id = existing_id or new_digest_id()
    # Fresh digests surface in breath via the watch tier for a while (expiry is
    # anchored to the period end, so a backfilled old period never surfaces —
    # its watch window is already over and it goes straight to the plain tier).
    # The MCP maintenance tick auto-demotes expired watch rows to archive, and
    # absorption by a parent digest archives them early — both paths already exist.
    watch_until = period.end_utc + timedelta(days=cfg["watch_days"])
    if watch_until > datetime.now(timezone.utc):
        tier, tier_until = "watch", watch_until.isoformat()
    else:
        tier, tier_until = "", ""
    with conn:
        if existing_id:
            # Rewrite in place, keeping the id so anything that already cites this
            # digest stays valid. tier / tier_until / last_active / activation_count
            # / pinned / resolved are deliberately left alone: a --force rewrite
            # changes the text, not the row's place in the tier system — un-archiving
            # a month whose quarter already absorbed it would resurrect it into
            # breath, and a digest the user pinned or resolved should stay that way.
            #
            # embedding IS cleared: the new text is a different text, and
            # reindex_embeddings.py only refills rows where length(embedding)=0,
            # so keeping the old vector would leave semantic search matching the
            # previous version of this digest forever.
            conn.execute(
                "UPDATE memories SET key=?, content=?, memory_kind='long_term', category=?, "
                "importance=?, session_id=?, created_at=?, updated_at=?, embedding=X'', "
                "valence=?, arousal=?, "
                "digested=0, consolidated=1 WHERE id=?",
                (period.key, content, category, cfg["importance"], DIGEST_SESSION_ID,
                 stamp, now, valence, arousal, existing_id),
            )
        else:
            conn.execute(
                """
                INSERT INTO memories(
                    id, key, content, memory_kind, category, importance, session_id,
                    created_at, updated_at, embedding,
                    valence, arousal, pinned, resolved, digested, activation_count,
                    last_active, last_breath_at, consolidated, tier, tier_until
                ) VALUES (?, ?, ?, 'long_term', ?, ?, ?, ?, ?, X'',
                          ?, ?, 0, 0, 0, 1.0, ?, '', 1, ?, ?)
                """,
                # embedding left empty exactly as consolidate_sessions does —
                # reindex_embeddings.py backfills it; last_active tracks the period,
                # not the generation moment, so decay ages the row correctly.
                (mem_id, period.key, content, category, cfg["importance"], DIGEST_SESSION_ID,
                 stamp, now, valence, arousal, stamp, tier, tier_until),
            )
        if absorb_ids:
            placeholders = ",".join("?" * len(absorb_ids))
            conn.execute(
                f"UPDATE memories SET tier='archive', tier_until='', updated_at=? "
                f"WHERE session_id = ? AND id IN ({placeholders})",
                (now, DIGEST_SESSION_ID, *absorb_ids),
            )
    return mem_id


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def strip_source_line(text: str) -> str:
    """Drop the code-appended provenance tail before feeding a digest upward —
    otherwise the parent model sees memory ids and starts quoting them.

    Splits on the LAST marker, not the first: if a model ever echoes the marker
    mid-body despite being told not to, cutting at the first one would silently
    swallow real narrative.
    """
    return (text or "").rsplit(SOURCE_PREFIX, 1)[0].rstrip()


def source_line(ids: list[str]) -> str:
    if not ids:
        return ""
    if len(ids) > SOURCE_MAX_SHOWN:
        shown = ", ".join(ids[:SOURCE_MAX_SHOWN])
        return f"{SOURCE_PREFIX}{shown}, 等 {len(ids)} 条"
    return f"{SOURCE_PREFIX}{', '.join(ids)}"


def _local_stamp(created_at: str, tz: tzinfo) -> str:
    try:
        dt = datetime.fromisoformat((created_at or "").replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(tz).strftime("%Y-%m-%d %H:%M")
    except (ValueError, TypeError):
        return (created_at or "")[:16]


def _clip(text: str, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit] + "……（本条已截断）"


def trim_to_budget(items: list[dict], budget: int, entry_limit: int) -> tuple[list[dict], int]:
    """Keep the input under a character budget by dropping the least important
    entries first, then restoring chronological order.

    A busy week here runs to 180k characters of Chinese, which is both slow and
    close enough to the model's window to risk a silent truncation upstream.
    Dropping deliberately — lowest importance first, and telling the model how
    many were dropped — beats letting the context window decide for us.
    """
    sized = [(it, min(len(it["content"] or ""), entry_limit) + len(it["key"] or "") + 40) for it in items]
    total = sum(s for _, s in sized)
    if total <= budget:
        return items, 0
    order = sorted(range(len(sized)), key=lambda i: (sized[i][0]["importance"], sized[i][0]["created_at"]))
    dropped = set()
    for i in order:
        if total <= budget:
            break
        dropped.add(i)
        total -= sized[i][1]
    kept = [it for i, it in enumerate(items) if i not in dropped]
    return kept, len(dropped)


def _raw_block(items: list[dict], tz: tzinfo, entry_limit: int) -> str:
    out = []
    for i, it in enumerate(items, 1):
        out.append(f"[{i}] {_local_stamp(it['created_at'], tz)} | {it['key']}\n{_clip(it['content'], entry_limit)}")
    return "\n\n".join(out)


def _digest_block(items: list[dict]) -> str:
    return "\n\n".join(f"【{it['key']}】\n{strip_source_line(it['content'])}" for it in items)


def build_prompt(period: Period, tz: tzinfo, *, raw_items: list[dict], digest_items: list[dict],
                 dropped: int, entry_limit: int) -> str:
    cfg = LAYER_CFG[period.layer]
    cn, unit, ref, max_chars = cfg["cn"], cfg["unit"], cfg["ref"], cfg["max_chars"]

    header = (
        f"你是一名记忆整理员。下面是 {period.key}（{period.local_range_text(tz)}）"
        f"这一{unit}的记忆材料，请把它们收拢成一篇{cn}。\n\n"
        f"### 硬性规则\n"
        f"1. 只可以提取输入里已经出现过的信息。禁止推断、禁止演绎、禁止补充输入之外的任何说法；"
        f"材料没写的因果关系不要替它补上。宁可写少，不可编造。\n"
        f"2. 具体的人名、项目名、地名、日期、数字、专有名词一律照抄保留，不要泛化成「某人」「某个项目」。\n"
        f"3. 材料互相矛盾时以时间较晚的为准，必要时注明存在分歧。\n"
        f"4. 输出纯正文。不要 markdown 标题、不要代码块、不要分隔线，不要复述本提示。"
        f"第一行就是正文第一句：正文前后都不要写开场白、字数统计、「以上是我的输出」之类的交代。\n"
        f"5. 不要写出任何 mem_ 开头的记忆编号，来源清单由程序另行附加。\n"
        f"6. 全文中文，使用全角标点，引号一律用「」，总长度不超过 {max_chars} 字。\n\n"
        f"### 输出结构（三段，段间空一行，不加小标题）\n"
    )
    if period.layer in ("day", "week"):
        header += (f"第一段：主线叙事。开头第一句先用一句话概括这一{unit}的基调，"
                   f"然后按时间顺序展开这一{unit}实际发生的事。\n")
    else:
        header += f"第一段：主线叙事。开头第一句先用一句话概括这一{unit}的整体走向，然后展开主要脉络与转折。\n"
    header += (
        "第二段：关键事实短列，一行一条，行首用「· 」，只放日期、数字、名称、结论这类可以核对的硬信息。\n"
        "第三段：悬而未决的事项——输入里提出了却没有结论的问题、待办、没有下文的计划。确实没有就写「无」。\n\n"
    )

    parts = [header]
    if digest_items:
        child_cn = LAYER_CFG[INPUT_CHILD[period.layer]]["cn"]
        parts.append(f"### 一、{ref}的{child_cn}（{len(digest_items)} 篇，已是二手总结）\n\n{_digest_block(digest_items)}\n\n")
        parts.append(
            f"### 二、{ref}最重要的原始记忆（按 importance 取前 {len(raw_items)} 条，"
            f"用来校正上面二手总结里的失真）\n\n{_raw_block(raw_items, tz, entry_limit)}\n"
        )
    else:
        note = ""
        if period.layer not in ("day", "week"):
            note = f"（{ref}没有可用的下层总结，直接依据原始记忆书写）"
        parts.append(f"### 原始记忆（{len(raw_items)} 条，按时间升序）{note}\n\n{_raw_block(raw_items, tz, entry_limit)}\n")
    if dropped:
        parts.append(f"\n注：{ref}另有 {dropped} 条 importance 较低的记忆因篇幅所限未纳入，"
                     f"总结时不必宣称已覆盖全部内容。\n")
    # Restating the limit after the material measurably helps — by the time the
    # model reaches the end of a 70k-character prompt the header has faded.
    parts.append(f"\n以上是全部材料。现在直接开始输出{cn}正文，三段式，全文不超过 {max_chars} 字。\n")
    return "".join(parts)


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _resolve_cmd(name: str) -> str:
    """Resolve through PATH/PATHEXT.

    On Windows the CLI is `claude.cmd`; CreateProcess does not do PATHEXT
    lookup, so a bare ["claude", ...] argv raises FileNotFoundError there.
    shutil.which does the lookup and hands back a path CreateProcess accepts.
    """
    found = shutil.which(name)
    return found or name


_RULE_CHARS = {"-", "*", "_"}


def _is_rule(line: str) -> bool:
    s = line.strip()
    return len(s) >= 3 and len(set(s)) == 1 and s[0] in _RULE_CHARS


def clean_output(text: str) -> str:
    """Strip a wrapping code fence, then any preamble the model put above a
    horizontal rule.

    Observed in testing: sonnet answered with `793 characters, under the 800
    limit. This is my final output.` then `---` then the actual digest. Rule 4 of
    the prompt forbids that, and the prompt now says so more bluntly, but a model
    that ignores the instruction should not get its bookkeeping stored as memory.
    Only a rule inside the first few lines counts, so a divider used legitimately
    further down survives.
    """
    out = (text or "").strip()
    if out.startswith("```"):
        first_nl = out.find("\n")
        if first_nl != -1:
            body = out[first_nl + 1:]
            if body.rstrip().endswith("```"):
                body = body.rstrip()[:-3]
            out = body.strip()
    lines = out.split("\n")
    for i, line in enumerate(lines[:4]):
        if _is_rule(line):
            out = "\n".join(lines[i + 1:]).strip()
            break
    # A second observed leak shape (three real cases in the 2026-08 backfill,
    # all sonnet): a standalone first line of self-audit bookkeeping followed by
    # a blank line, no divider. Only strip when the first line carries an
    # explicit word-count marker AND sits alone above a blank line — a digest
    # that legitimately opens with narrative never matches.
    lines = out.split("\n")
    if len(lines) >= 3 and lines[1].strip() == "" and len(lines[0]) <= 120:
        first = lines[0]
        low = first.lower()
        meta = ("character limit" in low or "final output" in low
                or "字以内" in first or "最终版本" in first
                or (first[:1].isdigit() and "字" in first[:12]))
        if meta:
            out = "\n".join(lines[2:]).strip()
    return out.strip()


def call_llm(prompt: str, model: str, claude_cmd: str, *, timeout: float = LLM_TIMEOUT,
             verbose: bool = False) -> str:
    """One `claude -p` call, prompt on stdin, one retry on failure.

    stdin keeps the prompt off the command line — Windows caps a command line at
    ~32k characters and a week's material is many times that.
    """
    # --disallowedTools: claude -p runs agentic by default — during the 2026-08
    # backfill sonnet wrote a draft_digest.txt into the working directory and
    # counted characters before answering (hence the meta-talk leaks). Denying
    # the mutating tools keeps it a pure text generator.
    cmd = [_resolve_cmd(claude_cmd), "-p", "--model", model, "--output-format", "text",
           "--disallowedTools", "Bash,Write,Edit,NotebookEdit,Task,Agent,WebFetch,WebSearch"]
    last_err = ""
    for attempt in range(1, LLM_ATTEMPTS + 1):
        try:
            proc = subprocess.run(
                cmd, input=prompt, capture_output=True, text=True, check=False,
                encoding="utf-8", errors="replace", timeout=timeout,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"找不到 claude 命令（{claude_cmd}）。用 --claude-cmd 传全路径，"
                f"Windows 上通常是 ...\\npm\\claude.cmd。原始错误：{exc}"
            ) from exc
        except subprocess.TimeoutExpired:
            last_err = f"超时 {timeout:.0f}s"
        else:
            if proc.returncode == 0 and (proc.stdout or "").strip():
                return clean_output(proc.stdout)
            tail = ((proc.stderr or "") + (proc.stdout or "")).strip().replace("\n", " ")[:300]
            last_err = f"returncode={proc.returncode} 输出={tail or '(空)'}"
        if attempt < LLM_ATTEMPTS:
            sys.stderr.write(f"[warn] {model} 调用失败（{last_err}），重试一次\n")
        elif verbose:
            sys.stderr.write(f"[warn] {model} 重试后仍失败（{last_err}）\n")
    raise RuntimeError(f"claude -p 调用失败：{last_err}")


# ---------------------------------------------------------------------------
# Per-period work
# ---------------------------------------------------------------------------

class Skipped(Exception):
    """Period deliberately not generated — threshold not met. Cheap: no LLM call,
    so it does not consume the run's backfill budget."""


class Deferred(Exception):
    """Parent period postponed because a child digest is still missing. Also
    cheap, also does not consume the backfill budget."""


def layer_min_items(layer: str, args) -> int:
    """The --min-items threshold for one layer. Days get their own, lower one:
    a day with 4 fragments is a perfectly ordinary day, while a week with 4 is
    a week barely worth narrating."""
    return args.min_items_day if layer == "day" else args.min_items


def layer_top_raw(layer: str, args) -> int:
    return args.top_raw_year if layer == "year" else args.top_raw


def unsettled_children(conn: sqlite3.Connection, period: Period, tz: tzinfo, args) -> list[str]:
    """Child periods that ought to have a digest but do not have one yet.

    Without this gate a first backfill run would build [月记] 2025-07 out of zero
    week digests (the weeks having been capped away earlier in the same run),
    write it, and then never revisit it — idempotency would lock the bad version
    in and the quarter above would inherit it. A child counts as settled when it
    already has a digest, or when its own raw count is below that child layer's
    --min-items so it would be skipped anyway and no digest will ever appear.
    """
    child_layer = INPUT_CHILD.get(period.layer)
    if not child_layer:
        return []
    exclude = tuple(args.exclude_tier or ())
    threshold = layer_min_items(child_layer, args)
    missing = []
    for child in sub_periods(period, child_layer, tz):
        if existing_digest_id(conn, child.key):
            continue
        if raw_count(conn, child, exclude) < threshold:
            continue          # that child will never be generated — not a blocker
        missing.append(child.key)
    return missing


def gather_inputs(conn: sqlite3.Connection, period: Period, tz: tzinfo, args) -> dict:
    """Collect this period's material. Raises Skipped/Deferred when unusable."""
    exclude = tuple(args.exclude_tier or ())
    child_layer = INPUT_CHILD.get(period.layer)
    min_items = layer_min_items(period.layer, args)
    if not child_layer:
        # day and week: the whole period's fragments, chronologically.
        raw = fetch_raw(conn, period, order="created_at", exclude_tiers=exclude)
        digests: list[dict] = []
    else:
        if period.layer in WAIT_CHILDREN and not args.no_wait_children:
            missing = unsettled_children(conn, period, tz, args)
            if missing:
                raise Deferred(f"下层还缺 {len(missing)} 篇（{', '.join(missing[:4])}"
                               f"{'…' if len(missing) > 4 else ''}），等下层补齐后再生成"
                               f"（--no-wait-children 可强行生成）")
        # A half-month simply takes whatever day digests happen to exist — usually
        # none for a backfilled period, in which case it is written from raw
        # fragments alone, same as any parent whose children were all skipped.
        digests = fetch_digests(conn, [c.key for c in sub_periods(period, child_layer, tz)])
        raw = fetch_raw(conn, period, order="importance", limit=layer_top_raw(period.layer, args),
                        exclude_tiers=exclude)
        raw.sort(key=lambda r: r["created_at"])

    n_items = len(raw) + len(digests)
    if n_items < min_items:
        raise Skipped(f"输入仅 {n_items} 条（下层总结 {len(digests)} + 原始 {len(raw)}），"
                      f"少于门槛 {min_items}")

    kept, dropped = trim_to_budget(raw, args.max_input_chars, args.max_entry_chars)
    return {"raw": kept, "digests": digests, "dropped": dropped, "n_items": n_items}


def absorbed_digest_ids(conn: sqlite3.Connection, period: Period, tz: tzinfo,
                        digests: list[dict]) -> list[str]:
    """Digest rows this period archives on write.

    Usually exactly the children it just read, so no second query. The week is
    the exception: it is written from raw fragments but still retires the day
    digests underneath it, so those have to be looked up separately. Layers
    absent from ABSORB_CHILD (day, halfmonth) archive nothing.
    """
    child_layer = ABSORB_CHILD.get(period.layer)
    if not child_layer:
        return []
    if child_layer == INPUT_CHILD.get(period.layer):
        return [d["id"] for d in digests]
    rows = fetch_digests(conn, [c.key for c in sub_periods(period, child_layer, tz)])
    return [r["id"] for r in rows]


def process_period(conn: sqlite3.Connection, period: Period, tz: tzinfo, args, models: dict) -> dict:
    """Returns a result dict; raises Skipped or RuntimeError."""
    existing = existing_digest_id(conn, period.key)
    if existing and not args.force:
        raise Skipped("已存在（--force 可重写）")

    data = gather_inputs(conn, period, tz, args)
    raw, digests, dropped = data["raw"], data["digests"], data["dropped"]

    if args.dry_run:
        return {"dry": True, "raw": len(raw), "digests": len(digests), "dropped": dropped,
                "existing": bool(existing)}

    prompt = build_prompt(period, tz, raw_items=raw, digest_items=digests,
                          dropped=dropped, entry_limit=args.max_entry_chars)
    if args.verbose:
        sys.stderr.write(f"[info] {period.key} prompt {len(prompt)} 字符 → {models[period.layer]}\n")
    body = call_llm(prompt, models[period.layer], args.claude_cmd,
                    timeout=args.llm_timeout, verbose=args.verbose)
    if not body:
        raise RuntimeError("模型返回空内容")

    src_ids = [d["id"] for d in digests] + [r["id"] for r in raw]
    content = body.rstrip() + source_line(src_ids)
    valence, arousal = period_emotion(conn, period)
    absorb = absorbed_digest_ids(conn, period, tz, digests)  # our own sub-digests only, never raw fragments
    mem_id = write_digest(conn, period, content, category=args.category,
                          valence=valence, arousal=arousal,
                          absorb_ids=absorb, existing_id=existing)
    return {"dry": False, "id": mem_id, "chars": len(content), "raw": len(raw),
            "digests": len(digests), "dropped": dropped, "absorbed": len(absorb),
            "rewritten": bool(existing)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def layer_window(layer: str, periods: list[Period], now_utc: datetime, tz: tzinfo,
                 args) -> tuple[list[Period], str]:
    """Narrow a layer's enumerated periods to the ones it is allowed to generate,
    and return a human note explaining the cut.

    --backfill caps how many are *generated* per run but always takes the oldest
    pending ones first, which is right for a fully backfilled layer (the oldest
    missing one is the newest period) and badly wrong for the three layers whose
    history is deliberately empty: with no window, day would spend every run
    generating three days from early 2025 forever. See the module docstring.
    """
    if layer == "day":
        floor = now_utc.astimezone(tz).date() - timedelta(days=max(1, args.day_lookback))
        kept = [p for p in periods if p.anchor >= floor]
        return kept, f"（只看 {floor.isoformat()} 起的最近 {max(1, args.day_lookback)} 天）"
    if layer == "halfmonth" and not args.backfill_all:
        n = max(1, args.halfmonth_recent)
        if len(periods) > n:
            return periods[-n:], f"（只看最近 {n} 个，--backfill-all 可全补）"
        return periods, ""
    if layer == "year":
        kept = [p for p in periods if p.anchor.year >= args.year_min]
        if len(kept) != len(periods):
            return kept, f"（只生成 {args.year_min} 年及以后）"
        return kept, ""
    return periods, ""


def parse_since(value: str, tz: tzinfo, conn: sqlite3.Connection) -> date:
    if value:
        try:
            return date.fromisoformat(value.strip()[:10])
        except ValueError as exc:
            raise SystemExit(f"[ERROR] --since 日期格式不对（要 YYYY-MM-DD）：{value!r}") from exc
    earliest = earliest_created(conn)
    if not earliest:
        return datetime.now(tz).date()
    dt = datetime.fromisoformat(earliest.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(tz).date()


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:  # noqa: BLE001, S110 — console encoding is best-effort
                pass

    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(
        description="分层递归总结：日 → 周 → 半月 → 月 → 季 → 半年 → 年，原始碎片一律不动",
    )
    ap.add_argument("--db", default=str(here / "memory.db"), help="memory.db 路径（默认脚本同目录）")
    ap.add_argument("--dry-run", action="store_true",
                    help="只打印将要生成的周期与输入条数，不调 LLM、不写库")
    ap.add_argument("--force", action="store_true", help="重写已存在的同 key 总结")
    ap.add_argument("--since", default="", help="从该日期起枚举周期（YYYY-MM-DD，默认取库中最早记忆）")
    ap.add_argument("--backfill-all", action="store_true", help="解除每层最多回补 N 个周期的限制")
    ap.add_argument("--backfill", type=int, default=DEFAULT_BACKFILL,
                    help=f"每层每次最多回补的缺失周期数（默认 {DEFAULT_BACKFILL}）")
    ap.add_argument("--layers", default=",".join(LAYERS),
                    help="只跑其中几层，逗号分隔（" + ",".join(LAYERS) + "）")
    ap.add_argument("--claude-cmd", default="claude", help="claude CLI 命令名或全路径")
    ap.add_argument("--llm-timeout", type=float, default=LLM_TIMEOUT,
                    help=f"单次 claude 调用超时秒数（默认 {LLM_TIMEOUT:.0f}；opus 层材料多时可调大）")
    ap.add_argument("--model-day", default=LAYER_CFG["day"]["model"])
    ap.add_argument("--model-week", default=LAYER_CFG["week"]["model"])
    ap.add_argument("--model-halfmonth", default=LAYER_CFG["halfmonth"]["model"])
    ap.add_argument("--model-month", default=LAYER_CFG["month"]["model"])
    ap.add_argument("--model-quarter", default=LAYER_CFG["quarter"]["model"])
    ap.add_argument("--model-half", default=LAYER_CFG["half"]["model"])
    ap.add_argument("--model-year", default=LAYER_CFG["year"]["model"])
    ap.add_argument("--min-items", type=int, default=DEFAULT_MIN_ITEMS,
                    help=f"当期输入少于该条数则跳过（默认 {DEFAULT_MIN_ITEMS}，日记层除外）")
    ap.add_argument("--min-items-day", type=int, default=DEFAULT_MIN_ITEMS_DAY,
                    help=f"日记层的跳过门槛（默认 {DEFAULT_MIN_ITEMS_DAY}）")
    ap.add_argument("--top-raw", type=int, default=DEFAULT_TOP_RAW,
                    help=f"半月/月/季/半年额外混入的高 importance 原始条数（默认 {DEFAULT_TOP_RAW}）")
    ap.add_argument("--top-raw-year", type=int, default=DEFAULT_TOP_RAW_YEAR,
                    help=f"年记额外混入的高 importance 原始条数（默认 {DEFAULT_TOP_RAW_YEAR}）")
    ap.add_argument("--day-lookback", type=int, default=DEFAULT_DAY_LOOKBACK,
                    help=f"日记层只看最近这么多天（默认 {DEFAULT_DAY_LOOKBACK}）。"
                         f"--backfill-all 不会解除这个窗口，要补更早的日记必须显式调大")
    ap.add_argument("--halfmonth-recent", type=int, default=DEFAULT_HALFMONTH_RECENT,
                    help=f"半月记层只看最近这么多个已完结周期（默认 {DEFAULT_HALFMONTH_RECENT}），"
                         f"--backfill-all 解除")
    ap.add_argument("--year-min", type=int, default=DEFAULT_YEAR_MIN,
                    help=f"年记层只生成这一年及以后（默认 {DEFAULT_YEAR_MIN}，更早的年份数据不全）")
    ap.add_argument("--max-entry-chars", type=int, default=DEFAULT_MAX_ENTRY_CHARS,
                    help=f"单条记忆送进 prompt 的字符上限（默认 {DEFAULT_MAX_ENTRY_CHARS}）")
    ap.add_argument("--max-input-chars", type=int, default=DEFAULT_MAX_INPUT_CHARS,
                    help=f"单次 prompt 材料区字符预算，超出按 importance 从低往高丢（默认 {DEFAULT_MAX_INPUT_CHARS}）")
    ap.add_argument("--no-wait-children", action="store_true",
                    help="月/季/半年/年即使下层总结还没补齐也照常生成（默认等下层；半月记本就不等日记）")
    ap.add_argument("--exclude-tier", action="append", default=[],
                    help="排除某个 tier 的原始记忆（可重复，例如 --exclude-tier seabed）")
    ap.add_argument("--category", default="digest", help="新条目的 category（默认 digest）")
    ap.add_argument("--tz", default="America/Toronto", help="周期边界所用时区（默认 America/Toronto）")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    db_path = Path(args.db).resolve()
    if not db_path.exists():
        sys.stderr.write(f"[ERROR] 找不到数据库：{db_path}\n")
        return 1

    layers = [x.strip() for x in args.layers.split(",") if x.strip()]
    bad = [x for x in layers if x not in LAYERS]
    if bad:
        sys.stderr.write(f"[ERROR] 未知层级：{bad}，可选 {list(LAYERS)}\n")
        return 1

    models = {"day": args.model_day, "week": args.model_week,
              "halfmonth": args.model_halfmonth, "month": args.model_month,
              "quarter": args.model_quarter, "half": args.model_half,
              "year": args.model_year}

    tz = load_tz(args.tz)
    now_utc = datetime.now(timezone.utc)
    conn = connect(db_path)
    try:
        since_local = parse_since(args.since, tz, conn)
        sys.stderr.write(
            f"库：{db_path}\n起点：{since_local.isoformat()}（{args.tz}）  "
            f"现在：{now_utc.astimezone(tz).strftime('%Y-%m-%d %H:%M')}  "
            f"层级：{','.join(layers)}  "
            f"回补上限：{'不限' if args.backfill_all else args.backfill}/层\n"
        )
        if args.dry_run:
            sys.stderr.write("--- DRY-RUN：不调 LLM，不写库 ---\n")

        totals = {"generated": 0, "skipped": 0, "deferred": 0, "failed": 0}
        # Walked in LAYERS order (day → week → halfmonth → month → quarter →
        # half → year) so every layer sees the children this very run produced:
        # the week can archive today's fresh day digests, the month can read the
        # week it just wrote, and so on up.
        for layer in [x for x in LAYERS if x in layers]:
            periods, window_note = layer_window(
                layer, enumerate_periods(layer, since_local, now_utc, tz), now_utc, tz, args)
            pending = [p for p in periods if args.force or not existing_digest_id(conn, p.key)]
            budget = 10 ** 9 if args.backfill_all else max(0, args.backfill)
            sys.stderr.write(
                f"\n== {LAYER_CFG[layer]['cn']} == 视野内已完结 {len(periods)} 个周期{window_note}，"
                f"缺失 {len(pending)} 个，本次最多生成 {'不限' if args.backfill_all else budget} 个\n"
            )
            spent = 0
            deferred_keys: list[str] = []
            for p in pending:
                if spent >= budget:
                    break
                try:
                    res = process_period(conn, p, tz, args, models)
                except Skipped as exc:
                    # A below-threshold period never gets a key, so it would sit at
                    # the head of `pending` forever. Not charging it against the
                    # budget is what keeps a handful of empty weeks from stalling
                    # the whole backfill permanently.
                    totals["skipped"] += 1
                    sys.stderr.write(f"  [skip] {p.key}：{exc}\n")
                    continue
                except Deferred as exc:
                    totals["deferred"] += 1
                    deferred_keys.append(p.key)
                    if args.verbose:
                        sys.stderr.write(f"  [defer] {p.key}：{exc}\n")
                    continue
                except Exception as exc:  # noqa: BLE001 — one bad period must not kill the run
                    totals["failed"] += 1
                    spent += 1          # a failure did burn an LLM call; don't loop on it
                    sys.stderr.write(f"  [fail] {p.key}：{type(exc).__name__}: {exc}\n")
                    continue
                spent += 1
                drop_note = f"，超预算丢弃 {res['dropped']} 条" if res["dropped"] else ""
                if res["dry"]:
                    rewrite_note = "（已存在，将重写）" if res["existing"] else ""
                    sys.stderr.write(
                        f"  [dry] {p.key}  {p.start_s}..{p.end_s}  "
                        f"下层总结 {res['digests']} + 原始 {res['raw']}"
                        f"{drop_note}{rewrite_note}\n"
                    )
                else:
                    totals["generated"] += 1
                    rewrite_note = "  [重写]" if res["rewritten"] else ""
                    sys.stderr.write(
                        f"  [ok] {p.key}  {res['chars']} 字  "
                        f"源 {res['digests']}+{res['raw']}{drop_note}  "
                        f"归档下层 {res['absorbed']}  id={res['id']}{rewrite_note}\n"
                    )
                sys.stderr.flush()
            if deferred_keys and not args.verbose:
                head = "，".join(deferred_keys[:3])
                more = f" 等 {len(deferred_keys)} 个" if len(deferred_keys) > 3 else ""
                sys.stderr.write(
                    f"  [defer] {head}{more} 等下层总结补齐后再生成（--verbose 看明细，"
                    f"--no-wait-children 可强行生成）\n"
                )

        sys.stderr.write(
            f"\n完成：生成 {totals['generated']}，跳过 {totals['skipped']}，"
            f"延后 {totals['deferred']}，失败 {totals['failed']}\n"
        )
        if totals["generated"] and not args.dry_run:
            sys.stderr.write("提醒：新条目的 embedding 留空，跑 `python reindex_embeddings.py` 补齐语义检索。\n")
    finally:
        conn.close()
    # Non-zero exit when any period failed, so Task Scheduler shows a real
    # error code instead of silently reporting success on a broken run.
    return 1 if totals["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
