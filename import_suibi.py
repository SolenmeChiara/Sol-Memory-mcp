"""One-shot import of the backstage Claude 随笔 notes into the memory store.

Reads D:\\internet_explore\\notes\\随笔.md (a hand-written journal), splits it into
per-entry blocks, and writes each block verbatim into the biography layer
(tier='archive') of the SQLite memory store. Each entry becomes exactly one
memory row, back-dated to the entry's real date.

Design notes
------------
- Entry boundaries: the journal separates entries with a bare dash line ('—' or
  '---'). A '【…】' header line starts a new entry when it either follows such a
  separator OR carries a parseable date (YYYY年M月D日). Non-dated sub-headers that
  do NOT follow a separator (e.g. 【观察】/【跳过】/【今日浏览小结】) stay inside the
  current entry — they are sub-sections, not standalone entries. This keeps long
  session logs intact while still splitting the tail section, whose distinct
  dated entries (including the 2026-07-12 后台随笔) lack separators between them.
- Dates that fail to parse inherit the previous entry's date (e.g. 【更新】).
- Content that belongs to no entry (the file title) is reported, never dropped.
- category/importance/valence/arousal come from a cloud LLM (reusing
  consolidate_sessions.call_llm / LLM_BACKEND / .env loading). A parse failure
  falls back to conservative defaults per entry and never aborts the run.
- embedding is produced locally via Ollama (reusing reindex_embeddings.
  fetch_embedding + pack_embedding, little-endian float32). A failure leaves the
  blob empty so a later `python reindex_embeddings.py` fills it in.

Usage
-----
    python import_suibi.py --dry-run                 # parse + preview, no LLM, no writes
    python import_suibi.py --dry-run --limit 2       # preview first 2 entries
    python import_suibi.py --db test.db --limit 2    # real import into a test db
    python import_suibi.py --db test.db --no-backup  # skip the pre-write backup

NEVER run a write mode against the production memory.db without explicit sign-off.
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_FILE = Path(r"D:\internet_explore\notes\随笔.md")
DEFAULT_DB = HERE / "memory.db"

# Cloud-LLM tag fallbacks (used when the LLM is absent or its output won't parse).
DEFAULT_CATEGORY = "other"
DEFAULT_IMPORTANCE = 0.45
DEFAULT_VALENCE = 0.55
DEFAULT_AROUSAL = 0.35
VALID_CATEGORIES = frozenset(
    {"preference", "promise", "event", "anniversary", "emotion", "habit", "boundary", "other"}
)

# A line is a separator when, stripped, it is one-or-more dash-family characters
# and nothing else. Covers '—' (U+2014), '---', and a few look-alikes so the
# exact glyph the author used never matters.
_DASH_CHARS = frozenset("-‐‑‒–—―⸻─－")
_DATE_RE = re.compile(r"(\d{4})年(\d{1,2})月(\d{1,2})日")
# A date sitting immediately after the '】' of a header — stripped from the key
# so it isn't duplicated next to the ISO date. A date *inside* '【…】'
# (e.g. 【2025年12月19日的发现】) is left alone.
_TRAILING_DATE_RE = re.compile(r"^(【[^】]*】)\s*\d{4}年\d{1,2}月\d{1,2}日\s*(.*)$")

KEY_MAXLEN = 40
SESSION_PREFIX = "suibi_import_"
LABEL_BATCH = 8          # entries per LLM tagging request
LABEL_EXCERPT_CHARS = 700  # per-entry content sent to the tagger


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _is_separator(line: str) -> bool:
    s = line.strip()
    return bool(s) and all(ch in _DASH_CHARS for ch in s)


def _parse_date(text: str) -> "str | None":
    m = _DATE_RE.search(text)
    if not m:
        return None
    year, month, day = (int(g) for g in m.groups())
    try:
        return datetime(year, month, day).strftime("%Y-%m-%d")
    except ValueError:
        return None


def _clean_title(header: str) -> str:
    """Header text for the key: drop a date that trails the '】', keep everything
    else. Collapses the leftover dash/space glue between bracket and remainder."""
    m = _TRAILING_DATE_RE.match(header)
    if not m:
        return header.strip()
    bracket, rest = m.group(1), m.group(2).strip()
    rest = rest.lstrip("".join(_DASH_CHARS) + " 　")
    return f"{bracket} {rest}".strip() if rest else bracket


def parse_entries(text: str) -> "tuple[list[dict], list[str]]":
    """Split the journal into entries.

    Returns (entries, orphan_lines). Each entry dict has: header, lines (raw),
    date (parsed from its own header, or None if it must inherit). orphan_lines
    holds non-empty content that precedes the first entry (the file title).
    """
    entries: list[dict] = []
    orphans: list[str] = []
    cur: "dict | None" = None
    saw_sep = True  # start of file behaves as if a separator just occurred

    for raw in text.splitlines():
        line = raw.rstrip("\r")
        if _is_separator(line):
            saw_sep = True
            continue
        stripped = line.strip()
        if not stripped:
            # Blank line: content of the current entry; preserve saw_sep so the
            # blank(s) between a separator and its header don't reset it.
            if cur is not None:
                cur["lines"].append(line)
            continue

        is_header = stripped.startswith("【")
        date_here = _parse_date(stripped) if is_header else None
        start_new = is_header and (saw_sep or date_here is not None)

        if start_new:
            if cur is not None:
                entries.append(cur)
            cur = {"header": stripped, "lines": [line], "date": date_here}
        elif cur is None:
            orphans.append(line)
        else:
            cur["lines"].append(line)
        saw_sep = False

    if cur is not None:
        entries.append(cur)

    # Inherit dates for entries whose own header carried none.
    last_date: "str | None" = None
    for e in entries:
        if e["date"] is not None:
            last_date = e["date"]
            e["inherited"] = False
        else:
            e["date"] = last_date          # may stay None if nothing precedes it
            e["inherited"] = last_date is not None
    return entries, orphans


def _entry_content(entry: dict) -> str:
    """Verbatim content: header + body, with leading/trailing blank lines trimmed
    but every internal line preserved exactly."""
    lines = list(entry["lines"])
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def make_key(date_iso: str, header: str, seen: set) -> str:
    """[随笔] YYYY-MM-DD <title>, total length <= KEY_MAXLEN chars, deduped."""
    prefix = f"[随笔] {date_iso} "
    budget = KEY_MAXLEN - len(prefix)
    title = _clean_title(header)
    base = prefix + (title[:budget] if len(title) > budget else title)
    if base not in seen:
        seen.add(base)
        return base
    # Deterministic disambiguation (stable across re-runs since entry order is).
    for n in range(2, 100):
        suffix = f" #{n}"
        trimmed = title[: max(0, budget - len(suffix))]
        cand = prefix + trimmed + suffix
        if cand not in seen:
            seen.add(cand)
            return cand
    seen.add(base)
    return base


def build_records(entries: list[dict]) -> list[dict]:
    """Attach content, char count and a deduped key to every dated entry.
    Entries that never resolved a date are flagged unclassifiable."""
    seen_keys: set = set()
    records: list[dict] = []
    for idx, e in enumerate(entries, 1):
        content = _entry_content(e)
        rec = {
            "index": idx,
            "date": e["date"],
            "header": e["header"],
            "content": content,
            "chars": len(content),
            "inherited": e.get("inherited", False),
            "unclassifiable": e["date"] is None,
        }
        rec["key"] = None if e["date"] is None else make_key(e["date"], e["header"], seen_keys)
        records.append(rec)
    return records


# ---------------------------------------------------------------------------
# Cloud-LLM tagging (category / importance / valence / arousal)
# ---------------------------------------------------------------------------

def _default_tags() -> dict:
    return {
        "category": DEFAULT_CATEGORY,
        "importance": DEFAULT_IMPORTANCE,
        "valence": DEFAULT_VALENCE,
        "arousal": DEFAULT_AROUSAL,
    }


def _coerce_tags(item: object) -> dict:
    tags = _default_tags()
    if not isinstance(item, dict):
        return tags
    cat = str(item.get("category", "")).strip().lower()
    if cat in VALID_CATEGORIES:
        tags["category"] = cat
    for field, lo, hi in (("importance", 0.0, 1.0), ("valence", 0.0, 1.0), ("arousal", 0.0, 1.0)):
        try:
            tags[field] = max(lo, min(hi, float(item[field])))
        except (KeyError, TypeError, ValueError):
            pass
    return tags


def _build_label_prompt(chunk: list[dict]) -> str:
    blocks = []
    for i, rec in enumerate(chunk, 1):
        excerpt = " ".join(rec["content"].split())[:LABEL_EXCERPT_CHARS]
        blocks.append(f"[{i}] key: {rec['key']}\n    内容摘录: {excerpt}")
    body = "\n\n".join(blocks)
    return (
        "你是记忆打标助手。下面是 " + str(len(chunk)) + " 条随笔记忆，请为每一条给出情感与"
        "分类标签。\n"
        "\n### 字段\n"
        "- category：只能是 preference / promise / event / anniversary / emotion / "
        "habit / boundary / other 之一\n"
        "- importance：0.0~1.0，这条记忆的重要程度\n"
        "- valence：0.0~1.0，情感效价（0极消极 0.5中性 1积极）\n"
        "- arousal：0.0~1.0，情绪唤醒度（0平静 1激动）\n"
        "\n### 输出（严格）\n"
        "只输出一个纯 JSON 数组，长度必须是 " + str(len(chunk)) + "，顺序与输入一致，"
        "不要解释、不要 markdown 代码块。每个元素形如：\n"
        '{"category":"emotion","importance":0.6,"valence":0.7,"arousal":0.4}\n'
        "\n### 待打标记忆\n\n" + body + "\n"
    )


def label_records(records: list[dict], call_llm, parse_json_list) -> list[dict]:
    """Tag every record in batches. Any failure in a batch → conservative
    defaults for that whole batch (never raises)."""
    tags: list[dict] = []
    for start in range(0, len(records), LABEL_BATCH):
        chunk = records[start : start + LABEL_BATCH]
        try:
            raw = call_llm(_build_label_prompt(chunk))
            items = parse_json_list(raw)
        except Exception as exc:  # LLM/network/JSON — degrade, don't abort
            sys.stderr.write(f"[label] batch {start // LABEL_BATCH} fell back to defaults: {exc}\n")
            items = []
        for i in range(len(chunk)):
            tags.append(_coerce_tags(items[i] if i < len(items) else None))
    return tags


# ---------------------------------------------------------------------------
# DB write
# ---------------------------------------------------------------------------

def _verify_schema(conn: sqlite3.Connection) -> None:
    cols = {r[1] for r in conn.execute("PRAGMA table_info(memories)").fetchall()}
    if not cols:
        raise SystemExit("[ERROR] no 'memories' table in this db — is it a memory store?")
    if "tier" not in cols:
        raise SystemExit(
            "[ERROR] 'tier' column missing — run migrate_tiers.py on this db first."
        )


def _backup(db_path: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    target = db_path.parent / f"{db_path.name}.bak-{stamp}"
    src = sqlite3.connect(str(db_path), timeout=15)
    dst = sqlite3.connect(str(target))
    try:
        with dst:
            src.backup(dst)
    finally:
        dst.close()
        src.close()
    size_mb = target.stat().st_size // 1048576
    sys.stderr.write(f"[backup] -> {target.name} ({size_mb} MB)\n")
    return target


def _new_id() -> str:
    return (
        f"mem_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
        f"_{random.randint(1000, 9999)}"
    )


def _already_imported(conn: sqlite3.Connection, key: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM memories WHERE session_id LIKE ? AND key = ? LIMIT 1",
        (f"{SESSION_PREFIX}%", key),
    ).fetchone()
    return row is not None


def insert_record(
    conn: sqlite3.Connection,
    rec: dict,
    tags: dict,
    embedding: bytes,
    session_id: str,
) -> str:
    """Insert one archive-tier memory. created_at/updated_at/last_active are the
    entry's real date at UTC noon (noon avoids a timezone nudging the calendar
    day). memories_fts stays in sync via the AFTER INSERT trigger — no manual
    FTS work. Returns the new row id."""
    created_at = f"{rec['date']}T12:00:00+00:00"
    # consolidated=1: verbatim chronicle rows must never be picked up by
    # consolidate_sessions (merging would destroy the original text)
    for _ in range(6):  # id is a PK; retry on the astronomically rare collision
        mem_id = _new_id()
        try:
            conn.execute(
                """
                INSERT INTO memories(
                    id, key, content, memory_kind, category, importance, session_id,
                    created_at, updated_at, embedding,
                    valence, arousal, pinned, resolved, digested, activation_count,
                    last_active, consolidated, tier, tier_until
                ) VALUES (?, ?, ?, 'long_term', ?, ?, ?, ?, ?, ?, ?, ?,
                          0, 0, 0, 1.0, ?, 1, 'archive', '')
                """,
                (
                    mem_id, rec["key"], rec["content"], tags["category"], tags["importance"],
                    session_id, created_at, created_at, embedding,
                    tags["valence"], tags["arousal"], created_at,
                ),
            )
            return mem_id
        except sqlite3.IntegrityError:
            continue
    raise RuntimeError("could not mint a unique memory id after 6 tries")


# ---------------------------------------------------------------------------
# Dry-run preview
# ---------------------------------------------------------------------------

def print_dry_run(records: list[dict], orphans: list[str], limit: int) -> None:
    shown = records[:limit] if limit > 0 else records
    print("=" * 78)
    print("DRY-RUN 预览（不调用任何 LLM、不写库）")
    print("=" * 78)
    print(f"{'序号':<4} {'日期':<12} {'字数':>6}  key")
    print("-" * 78)
    for rec in shown:
        date = rec["date"] or "?? 无法解析"
        mark = " *继承" if rec["inherited"] else ""
        key = rec["key"] or "(未生成 — 无日期)"
        print(f"{rec['index']:<4} {date:<12}{mark:<4} {rec['chars']:>6}  {key}")
    print("-" * 78)

    total = len(records)
    inherited = [r for r in records if r["inherited"]]
    unclassifiable = [r for r in records if r["unclassifiable"]]
    by_date: dict = {}
    for r in records:
        by_date[r["date"] or "无法解析"] = by_date.get(r["date"] or "无法解析", 0) + 1

    print(f"条目总数：{total}" + (f"（预览只显示前 {len(shown)} 条）" if limit > 0 else ""))
    print("按日期分布：")
    for date in sorted(by_date):
        print(f"    {date}: {by_date[date]} 条")
    print(f"继承上一条日期的条目（如【更新】）：{len(inherited)} 条")
    if inherited:
        for r in inherited:
            print(f"    #{r['index']} {r['key']}")
    print(f"无法归类（解析不出且无可继承日期）的条目：{len(unclassifiable)} 条")
    for r in unclassifiable:
        print(f"    #{r['index']} header={r['header'][:50]!r} chars={r['chars']}")
    print(f"落单的块（不属于任何条目，如文件标题，不导入）：{len(orphans)} 行")
    for line in orphans:
        print(f"    {line!r}")
    print("=" * 78)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_import(records: list[dict], db_path: Path, no_backup: bool) -> dict:
    """Real import path. Assumes records already limited by the caller."""
    # Reuse the sibling maintenance scripts (import first, per spec).
    sys.path.insert(0, str(HERE))
    import consolidate_sessions as cs
    import reindex_embeddings as ri

    # .env loading exactly as consolidate does, then refresh its module globals
    # (mirrors run_consolidate) so call_llm sees the freshly loaded key/model.
    cs._load_dotenv(HERE)
    cs.LLM_BACKEND = os.environ.get("LLM_BACKEND", cs.LLM_BACKEND).lower()
    cs.OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", cs.OPENROUTER_API_KEY)
    cs.OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", cs.OPENROUTER_MODEL)
    cs.OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", cs.OLLAMA_MODEL)

    session_id = f"{SESSION_PREFIX}{datetime.now(timezone.utc).strftime('%Y%m%d')}"
    stats = {"total": len(records), "inserted": 0, "skipped": 0, "no_embedding": 0, "session_id": session_id}

    conn = sqlite3.connect(str(db_path), timeout=15)
    conn.execute("PRAGMA busy_timeout=15000")
    conn.execute("PRAGMA journal_mode=WAL")
    _verify_schema(conn)

    # Idempotency: drop entries already imported under any suibi_import_* session.
    pending = [r for r in records if not _already_imported(conn, r["key"])]
    stats["skipped"] = len(records) - len(pending)
    if not pending:
        conn.close()
        sys.stderr.write("[import] nothing new to insert (all keys already present).\n")
        return stats

    if not no_backup:
        _backup(db_path)

    backend = cs.LLM_BACKEND
    if backend == "openrouter" and not cs.OPENROUTER_API_KEY:
        sys.stderr.write("[label] OPENROUTER_API_KEY not set — every entry uses default tags.\n")
        tag_list = [_default_tags() for _ in pending]
    else:
        sys.stderr.write(f"[label] tagging {len(pending)} entries via {backend} ...\n")
        tag_list = label_records(pending, cs.call_llm, cs._parse_json_list)

    for rec, tags in zip(pending, tag_list):
        try:
            emb = ri.pack_embedding(ri.fetch_embedding(rec["content"]))
        except Exception as exc:  # Ollama down / data error — leave empty, reindex later
            sys.stderr.write(f"[embed] #{rec['index']} empty ({type(exc).__name__}); reindex later\n")
            emb = b""
        if not emb:
            stats["no_embedding"] += 1
        insert_record(conn, rec, tags, emb, session_id)
        stats["inserted"] += 1
        sys.stderr.write(f"[import] +{rec['key']}  cat={tags['category']} imp={tags['importance']:.2f}\n")

    conn.commit()
    conn.close()
    return stats


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

    ap = argparse.ArgumentParser(description="Import 随笔.md into the memory archive tier")
    ap.add_argument("--file", default=str(DEFAULT_FILE), help="source journal (default: 随笔.md)")
    ap.add_argument("--db", default=str(DEFAULT_DB), help="target sqlite db (default: ./memory.db)")
    ap.add_argument("--dry-run", action="store_true", help="parse + preview only; no LLM, no writes")
    ap.add_argument("--limit", type=int, default=0, help="process only the first N entries (0 = all)")
    ap.add_argument("--no-backup", action="store_true", help="skip the pre-write db backup")
    args = ap.parse_args()

    file_path = Path(args.file)
    if not file_path.exists():
        raise SystemExit(f"[ERROR] source file not found: {file_path}")

    text = file_path.read_text(encoding="utf-8", errors="replace")
    entries, orphans = parse_entries(text)
    records = build_records(entries)

    if args.dry_run:
        print_dry_run(records, orphans, args.limit)
        return

    db_path = Path(args.db).resolve()
    if not db_path.exists():
        raise SystemExit(f"[ERROR] db not found: {db_path}")

    to_process = records[: args.limit] if args.limit > 0 else records
    bad = [r for r in to_process if r["unclassifiable"]]
    if bad:
        sys.stderr.write(
            f"[import] refusing to insert {len(bad)} unclassifiable entr(y/ies) with no date "
            f"(indices {[r['index'] for r in bad]}); fix the journal or exclude them.\n"
        )
        to_process = [r for r in to_process if not r["unclassifiable"]]

    stats = run_import(to_process, db_path, args.no_backup)
    sys.stderr.write(
        f"\nDone. session_id={stats['session_id']} "
        f"inserted={stats['inserted']} skipped(existing)={stats['skipped']} "
        f"empty_embedding={stats['no_embedding']} (run reindex_embeddings.py to fill those)\n"
    )


if __name__ == "__main__":
    main()
