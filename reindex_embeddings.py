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


class EmbedDataError(Exception):
    """bge-m3 produced NaN / non-serialisable output for this specific input.
    Distinct from service-level errors (timeout, 502, connection refused) so
    the caller can skip the row without aborting the whole run."""


def fetch_embedding(text: str) -> list[float]:
    """Call Ollama /api/embed.

    Raises:
      EmbedDataError — ollama returned 500 with "NaN" / "unsupported value" in
        the body. The embedding model choked on this specific input. Skip the row.
      Other exceptions — genuine service problems (timeout, unreachable, etc).
        Caller counts these toward the consecutive-failure abort threshold.
    """
    body = json.dumps({"model": OLLAMA_EMBED_MODEL, "input": (text or "")[:MAX_CHARS]}).encode()
    req = urllib.request.Request(
        f"{OLLAMA_BASE_URL.rstrip('/')}/api/embed",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        err_body = ""
        try:
            err_body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        if exc.code == 500 and ("NaN" in err_body or "unsupported value" in err_body):
            raise EmbedDataError(f"bge-m3 produced NaN for this input: {err_body[:120]}")
        raise
    embeddings = data.get("embeddings") or []
    if not embeddings or not embeddings[0]:
        raise RuntimeError(f"empty embeddings field in response: keys={list(data.keys())}")
    return embeddings[0]


def run_reindex(
    db_path: Path,
    *,
    workers: int = 4,
    batch: int = 50,
    limit: int = 0,
    progress_cb=None,
    stop_event: "threading.Event | None" = None,
) -> dict:
    """Backfill bge-m3 embeddings in-place. Reusable by CLI and web server.

    progress_cb: optional callable(dict) invoked every `batch`-worth of results
        and once at completion. Dict keys: processed, total, success, failed,
        rate_per_s, eta_min, last_error (str, optional), aborted (bool).

    stop_event: optional threading.Event — set it to request graceful stop
        (current in-flight workers still finish, then loop exits cleanly).

    Returns the final progress dict.
    """
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    # Skip digested rows — they're archived merge-source fragments and don't
    # participate in search anymore, so spending LLM cycles on their embeddings
    # would be pure waste.
    rows = conn.execute(
        "SELECT id, content FROM memories "
        "WHERE length(embedding) = 0 AND digested = 0 "
        "ORDER BY rowid"
    ).fetchall()
    if limit > 0:
        rows = rows[:limit]
    total = len(rows)

    state = {
        "processed": 0, "total": total, "success": 0, "failed": 0,
        "data_errors": 0,                      # NaN / non-serialisable (skipped, not abort)
        "rate_per_s": 0.0, "eta_min": 0.0,
        "aborted": False, "aborted_reason": "",
        "last_error": "",
        "stage": "running",
    }

    def _emit() -> None:
        if progress_cb:
            try:
                progress_cb(dict(state))
            except Exception as exc:
                sys.stderr.write(f"[run_reindex] progress_cb error: {exc}\n")

    if total == 0:
        state["stage"] = "done"
        conn.close()
        _emit()
        return state

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
        if stop_event is not None and stop_event.is_set():
            return ("skip", item[0])
        mid, content = item
        try:
            emb = fetch_embedding(content)
            with consecutive_lock:
                consecutive = 0
            return ("ok", mid, pack_embedding(emb))
        except EmbedDataError as exc:
            # Model produced NaN for this input — skip the row, DON'T count
            # toward the consecutive-failure threshold (it would trip on any
            # data-heavy repo where a handful of inputs always trigger it).
            with consecutive_lock:
                consecutive = 0
            return ("data_err", mid, f"EmbedDataError: {exc}")
        except Exception as exc:
            with consecutive_lock:
                consecutive += 1
                cf = consecutive
            return ("fail", mid, f"{type(exc).__name__}: {exc}", cf)

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for result in pool.map(worker, rows):
                kind = result[0]
                if kind == "skip":
                    continue
                if kind == "ok":
                    _, mid, blob = result
                    pending.append((mid, blob))
                    state["success"] += 1
                    if len(pending) >= batch:
                        commit_pending()
                elif kind == "data_err":
                    _, mid, err_msg = result
                    state["data_errors"] += 1
                    state["last_error"] = f"id={mid[:8]} {err_msg}"
                    sys.stderr.write(f"[DATA-ERR #{state['data_errors']}] {state['last_error']}\n")
                else:
                    _, mid, err_msg, cf = result
                    state["failed"] += 1
                    state["last_error"] = f"id={mid[:8]} {err_msg}"
                    sys.stderr.write(f"[FAIL #{state['failed']}] {state['last_error']} (consecutive={cf})\n")
                    if cf >= MAX_CONSECUTIVE_FAILURES:
                        state["aborted"] = True
                        state["aborted_reason"] = (
                            f"{cf} consecutive failures — aborting. "
                            f"Likely ollama unreachable or {OLLAMA_EMBED_MODEL} unloaded/hung."
                        )
                        break

                state["processed"] = state["success"] + state["failed"] + state["data_errors"]
                if state["processed"] % batch == 0 or state["processed"] == total:
                    elapsed = time.monotonic() - t0
                    state["rate_per_s"] = state["processed"] / elapsed if elapsed > 0 else 0.0
                    state["eta_min"] = (
                        (total - state["processed"]) / state["rate_per_s"] / 60
                        if state["rate_per_s"] > 0 else 0.0
                    )
                    _emit()
    finally:
        commit_pending()
        conn.close()

    # Mark dirty on the importing store if one is reachable — here we don't hold
    # the MemoryStore reference, but the web server wrapper will handle it.
    state["stage"] = "aborted" if state["aborted"] else "done"
    elapsed = time.monotonic() - t0
    state["elapsed_s"] = elapsed
    _emit()
    return state


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

    # First, log the scan so CLI users see what's happening before progress_cb fires
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    pending = conn.execute(
        "SELECT COUNT(*) FROM memories WHERE length(embedding)=0 AND digested=0"
    ).fetchone()[0]
    conn.close()
    if args.limit > 0:
        pending = min(pending, args.limit)
    if pending == 0:
        sys.stderr.write("Nothing to do — every memory already has an embedding.\n")
        return
    sys.stderr.write(
        f"Backfilling {pending} memories  workers={args.workers}  batch={args.batch}  "
        f"model={OLLAMA_EMBED_MODEL}  timeout={OLLAMA_TIMEOUT}s\n"
    )
    sys.stderr.flush()

    def cli_progress(state: dict) -> None:
        if state["stage"] != "running":
            return
        de = state.get("data_errors", 0)
        de_part = f" nan={de}" if de else ""
        sys.stderr.write(
            f"[{state['processed']}/{state['total']}] "
            f"ok={state['success']} fail={state['failed']}{de_part} "
            f"{state['rate_per_s']:.1f}/s ETA {state['eta_min']:.1f}min\n"
        )
        sys.stderr.flush()

    final = run_reindex(
        db_path, workers=args.workers, batch=args.batch, limit=args.limit,
        progress_cb=cli_progress,
    )

    de = final.get("data_errors", 0)
    de_part = f" nan_skipped={de}" if de else ""
    sys.stderr.write(
        f"\nDone. success={final['success']} failed={final['failed']}{de_part} "
        f"elapsed={final.get('elapsed_s', 0)/60:.1f}min\n"
    )
    if de > 0:
        sys.stderr.write(
            f"注：有 {de} 条记忆让 bge-m3 输出 NaN，已跳过（保持 length(embedding)=0）。\n"
            f"这类内容通常是 bge-m3 在某些 input 上的数值不稳定 bug，重跑不会修复。\n"
            f"如果量大且介意，可以手动检查那些 id 然后 resolved=true 归档。\n"
        )
    if final["aborted"]:
        sys.stderr.write(f"\n[ABORTED] {final['aborted_reason']}\n")
        sys.exit(2)


if __name__ == "__main__":
    main()
