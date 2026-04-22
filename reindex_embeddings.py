"""Backfill bge-m3 embeddings for memories whose embedding blob is empty.

Usage:
    python reindex_embeddings.py [--db memory.db] [--workers 4] [--batch 50] [--limit 0]

Behaviour:
- Only processes rows where length(embedding) = 0 (safe to re-run; resumes automatically).
- Parallel workers hit Ollama; main thread batches the SQLite writes.
- Progress to stderr every 100 rows.
- Aborts loudly if 10 consecutive Ollama calls fail (assumes server unreachable
  or model unloaded — surface the error rather than silently skipping).
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import struct
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "bge-m3")
OLLAMA_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", "180"))
MAX_CHARS = 2000
MAX_CONSECUTIVE_FAILURES = 10


def pack_embedding(values):
    return struct.pack(f"<{len(values)}f", *values) if values else b""


def fetch_embedding(text: str) -> list[float]:
    """Call Ollama /api/embed; raise on any failure (caller tracks consecutive failures)."""
    body = json.dumps({"model": OLLAMA_EMBED_MODEL, "input": (text or "")[:MAX_CHARS]}).encode()
    req = urllib.request.Request(
        f"{OLLAMA_BASE_URL.rstrip('/')}/api/embed",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as resp:
        data = json.loads(resp.read())
    embeddings = data.get("embeddings") or []
    if not embeddings or not embeddings[0]:
        raise RuntimeError(f"empty embeddings field in response: keys={list(data.keys())}")
    return embeddings[0]


def main() -> None:
    # Windows console encoding
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

    parser = argparse.ArgumentParser(
        description="Backfill bge-m3 embeddings for memories with empty embedding blobs"
    )
    parser.add_argument("--db", default="memory.db", help="SQLite database path (default: memory.db)")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent ollama workers (default: 4)")
    parser.add_argument("--batch", type=int, default=50, help="Commit every N successful writes (default: 50)")
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Process at most N rows (0 = all; useful for debugging)",
    )
    args = parser.parse_args()

    db_path = Path(args.db).resolve()
    if not db_path.exists():
        print(f"[ERROR] db not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    rows = conn.execute(
        "SELECT id, content FROM memories WHERE length(embedding) = 0 ORDER BY rowid"
    ).fetchall()
    if args.limit > 0:
        rows = rows[: args.limit]
    total = len(rows)

    if total == 0:
        sys.stderr.write("Nothing to do — every memory already has an embedding.\n")
        conn.close()
        return

    sys.stderr.write(
        f"Backfilling {total} memories  workers={args.workers}  batch={args.batch}  "
        f"model={OLLAMA_EMBED_MODEL}  timeout={OLLAMA_TIMEOUT}s\n"
    )
    sys.stderr.flush()

    success = 0
    failed = 0
    consecutive = 0
    consecutive_lock = threading.Lock()
    pending: list[tuple[str, bytes]] = []
    t0 = time.monotonic()

    def commit_pending() -> None:
        nonlocal pending
        if not pending:
            return
        for mid, blob in pending:
            conn.execute("UPDATE memories SET embedding=? WHERE id=?", (blob, mid))
        conn.commit()
        pending = []

    def worker(item: tuple[str, str]) -> tuple:
        nonlocal consecutive
        mid, content = item
        try:
            emb = fetch_embedding(content)
            with consecutive_lock:
                consecutive = 0
            return ("ok", mid, pack_embedding(emb))
        except Exception as exc:
            with consecutive_lock:
                consecutive += 1
                cf = consecutive
            return ("fail", mid, f"{type(exc).__name__}: {exc}", cf)

    aborted_msg = ""
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for result in pool.map(worker, rows):
                kind = result[0]
                if kind == "ok":
                    _, mid, blob = result
                    pending.append((mid, blob))
                    success += 1
                    if len(pending) >= args.batch:
                        commit_pending()
                else:
                    _, mid, err_msg, cf = result
                    failed += 1
                    sys.stderr.write(
                        f"[FAIL #{failed}] id={mid[:8]} {err_msg}  (consecutive={cf})\n"
                    )
                    if cf >= MAX_CONSECUTIVE_FAILURES:
                        aborted_msg = (
                            f"{cf} consecutive failures — aborting. "
                            f"Likely cause: ollama unreachable or {OLLAMA_EMBED_MODEL} unloaded/hung."
                        )
                        break

                done = success + failed
                if done % 100 == 0 or done == total:
                    elapsed = time.monotonic() - t0
                    rate = done / elapsed if elapsed > 0 else 0
                    eta_min = (total - done) / rate / 60 if rate > 0 else float("inf")
                    sys.stderr.write(
                        f"[{done}/{total}] ok={success} fail={failed} "
                        f"{rate:.1f}/s ETA {eta_min:.1f}min\n"
                    )
                    sys.stderr.flush()
    finally:
        commit_pending()
        conn.close()

    elapsed = time.monotonic() - t0
    sys.stderr.write(
        f"\nDone. success={success} failed={failed} elapsed={elapsed/60:.1f}min\n"
    )

    if aborted_msg:
        sys.stderr.write(f"\n[ABORTED] {aborted_msg}\n")
        sys.exit(2)


if __name__ == "__main__":
    main()
