"""One-shot migration for the layered-memory (tier) architecture (2026-07-16).

Brings an existing memory.db up to the tier schema and seeds the initial tier
assignments. Idempotent: safe to run twice, and safe to run either before OR
after the upgraded memory_mcp.py first touches the db (the column/index steps
mirror _init_db, so whoever runs first wins and the other is a no-op).

Steps:
  1. Back up the target db via SQLite's online backup API to
     memory.db.bak-<YYYYMMDD-HHMM> (skipped in --dry-run).
  2. Add columns tier / tier_until + indexes idx_memories_tier / _key / _created
     (same logic as memory_mcp._init_db).
  3. Seabed backfill: tag the 2026-04-21..23 batch-import flood (non-pinned,
     session_id LIKE 'import_conversations_%') as tier='seabed' so it never
     surfaces in breath again. Same window as rescore_importance.FLOOD_WHERE.
     --include-uploads also tags session_id LIKE 'upload_%' rows (default off).
  4. Nudge backfill: rows whose key starts with '[nudge]' get category='nudge'.

Usage:
    python migrate_tiers.py --dry-run              # count only, no writes
    python migrate_tiers.py                        # back up + migrate
    python migrate_tiers.py --include-uploads      # also seabed the upload_% rows
    python migrate_tiers.py --db PATH              # target a different db (tests)

Every step prints the number of affected rows.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_DB = HERE / "memory.db"

# Kept in lock-step with rescore_importance.FLOOD_WHERE (minus the tier guard,
# which is added dynamically so a --dry-run works before the column exists).
FLOOD_WHERE = (
    "pinned = 0 AND session_id LIKE 'import_conversations_%' "
    "AND created_at >= '2026-04-21' AND created_at < '2026-04-24'"
)
UPLOAD_WHERE = "pinned = 0 AND session_id LIKE 'upload_%'"


def connect(db: Path, readonly: bool) -> sqlite3.Connection:
    if readonly:
        conn = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True, timeout=15)
    else:
        conn = sqlite3.connect(str(db), timeout=15)
        conn.execute("PRAGMA busy_timeout=15000")
    conn.row_factory = sqlite3.Row
    return conn


def has_column(conn: sqlite3.Connection, table: str, col: str) -> bool:
    return any(r[1] == col for r in conn.execute(f"PRAGMA table_info({table})"))


def _tier_guard(conn: sqlite3.Connection) -> str:
    """Only constrain on tier if the column already exists (so --dry-run works
    on a db that hasn't been migrated yet — there every row is untiered)."""
    return "COALESCE(tier,'') = '' AND " if has_column(conn, "memories", "tier") else ""


def backup(db: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    target = db.parent / f"{db.name}.bak-{stamp}"
    src = sqlite3.connect(str(db), timeout=15)
    dst = sqlite3.connect(str(target))
    try:
        with dst:
            src.backup(dst)
    finally:
        dst.close()
        src.close()
    size_mb = target.stat().st_size // 1048576
    print(f"[1] backup -> {target.name} ({size_mb} MB)")
    return target


def add_schema(conn: sqlite3.Connection) -> None:
    added = []
    for col, ddl in (("tier", "TEXT DEFAULT ''"), ("tier_until", "TEXT DEFAULT ''")):
        if not has_column(conn, "memories", col):
            conn.execute(f"ALTER TABLE memories ADD COLUMN {col} {ddl}")
            added.append(col)
    for idx, cols in (
        ("idx_memories_tier", "tier"),
        ("idx_memories_key", "key"),
        ("idx_memories_created", "created_at"),
    ):
        conn.execute(f"CREATE INDEX IF NOT EXISTS {idx} ON memories({cols})")
    conn.commit()
    print(f"[2] schema: columns added={added or 'none (already present)'}; "
          f"indexes ensured (tier/key/created)")


def count(conn: sqlite3.Connection, where: str) -> int:
    return conn.execute(f"SELECT COUNT(*) FROM memories WHERE {where}").fetchone()[0]


def run(db: Path, dry_run: bool, include_uploads: bool) -> None:
    if not db.exists():
        print(f"error: db not found: {db}", file=sys.stderr)
        sys.exit(1)

    if dry_run:
        conn = connect(db, readonly=True)
        print("[1] backup: skipped (--dry-run)")
        # Schema step can't run read-only; report intent.
        missing = [c for c in ("tier", "tier_until")
                   if not has_column(conn, "memories", c)]
        print(f"[2] schema: would add columns={missing or 'none'}; ensure 3 indexes")
    else:
        backup(db)
        conn = connect(db, readonly=False)
        add_schema(conn)  # must run before the tier UPDATEs below

    # Guard computed after add_schema so a real run filters on the freshly
    # added tier column (idempotent); a dry-run before migration has no tier
    # column and simply counts all window matches.
    guard = _tier_guard(conn)

    # --- Step 3: seabed backfill (flood window) ---
    flood_where = guard + FLOOD_WHERE
    if dry_run:
        print(f"[3] seabed (flood): would set tier='seabed' on {count(conn, flood_where)} rows")
    else:
        n = conn.execute(f"UPDATE memories SET tier='seabed' WHERE {flood_where}").rowcount
        conn.commit()
        print(f"[3] seabed (flood): set tier='seabed' on {n} rows")

    # --- Step 3b: optional uploads backfill ---
    if include_uploads:
        up_where = guard + UPLOAD_WHERE
        if dry_run:
            print(f"[3b] seabed (uploads): would set tier='seabed' on {count(conn, up_where)} rows")
        else:
            n = conn.execute(f"UPDATE memories SET tier='seabed' WHERE {up_where}").rowcount
            conn.commit()
            print(f"[3b] seabed (uploads): set tier='seabed' on {n} rows")
    else:
        print("[3b] seabed (uploads): skipped (pass --include-uploads to enable)")

    # --- Step 4: nudge category backfill ---
    # category != 'nudge' guard keeps re-runs at 0 changed rows.
    nudge_where = "key LIKE '[nudge]%' AND category != 'nudge'"
    if dry_run:
        print(f"[4] nudge: would set category='nudge' on {count(conn, nudge_where)} rows")
    else:
        n = conn.execute(f"UPDATE memories SET category='nudge' WHERE {nudge_where}").rowcount
        conn.commit()
        print(f"[4] nudge: set category='nudge' on {n} rows")

    conn.close()
    print("done." + (" (dry-run — nothing written)" if dry_run else ""))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true",
                    help="count-only, write nothing (read-only connection)")
    ap.add_argument("--include-uploads", action="store_true",
                    help="also tag session_id LIKE 'upload_%%' rows as seabed")
    ap.add_argument("--db", default=str(DEFAULT_DB),
                    help="target database path (default: ./memory.db next to this script)")
    args = ap.parse_args()
    run(Path(args.db).resolve(), args.dry_run, args.include_uploads)


if __name__ == "__main__":
    main()
