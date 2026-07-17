"""Rescore importance for the 2026-04 batch-import flood (2026-07-02 audit).

Background: the ~48k rows imported on 2026-04-21..23 were scored by gemma4:e4b,
which rated 96% of the whole library >= 0.7 — inflation that erases the
meaning of importance. Organic rows written since May average 0.63 and are
noticeably better calibrated.

Default (statistical) mode — instant, deterministic:
    importance_new = max(0.15, round(importance * 0.65, 3))
  applied to non-pinned rows in the flood window (created_at 2026-04-21..23,
  session_id LIKE 'import_conversations_%'). Recoverable from backups/.

--llm mode — slow, per-row re-scoring via local Ollama:
    python rescore_importance.py --llm [--model qwen3:8b] [--limit 500]
  Resumable (progress checkpoint in rescore_progress.json), commits every 50.
  Only bothers with rows the statistical pass already touched, and only
  lowers-or-keeps unless --allow-raise is given.

Both modes print a before/after distribution and change nothing else.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DB = HERE / "memory.db"
PROGRESS = HERE / "rescore_progress.json"

FLOOD_WHERE = (
    "created_at >= '2026-04-21' AND created_at < '2026-04-24' "
    "AND session_id LIKE 'import_conversations_%' AND pinned = 0"
)

SCORE_PROMPT = (
    "你在为一个个人记忆库校准单条记忆的重要度。评分标准：\n"
    "0.9-1.0 核心事实（身份、健康、边界、长期承诺）\n"
    "0.7-0.8 重要事件与关系变化\n"
    "0.5-0.6 一般事件、日常近况\n"
    "0.3-0.4 琐碎细节、单次玩笑、场景性闲聊\n"
    "0.1-0.2 无长期价值的碎片\n"
    "只输出一个 0.1 到 1.0 之间的小数，不要任何其他文字。\n\n"
    "记忆标签：{key}\n记忆内容：{content}\n"
)


def connect(readonly: bool = False) -> sqlite3.Connection:
    if readonly:
        conn = sqlite3.connect(f"file:{DB.as_posix()}?mode=ro", uri=True, timeout=15)
    else:
        conn = sqlite3.connect(DB, timeout=15)
        conn.execute("PRAGMA busy_timeout=15000")
    conn.row_factory = sqlite3.Row
    return conn


def show_distribution(conn: sqlite3.Connection, label: str) -> None:
    buckets = conn.execute(
        f"""SELECT
            SUM(CASE WHEN importance >= 0.9 THEN 1 ELSE 0 END),
            SUM(CASE WHEN importance >= 0.7 AND importance < 0.9 THEN 1 ELSE 0 END),
            SUM(CASE WHEN importance >= 0.5 AND importance < 0.7 THEN 1 ELSE 0 END),
            SUM(CASE WHEN importance >= 0.3 AND importance < 0.5 THEN 1 ELSE 0 END),
            SUM(CASE WHEN importance < 0.3 THEN 1 ELSE 0 END),
            ROUND(AVG(importance), 3), COUNT(*)
        FROM memories WHERE {FLOOD_WHERE}"""
    ).fetchone()
    print(f"{label}: >=0.9: {buckets[0]}  0.7-0.9: {buckets[1]}  "
          f"0.5-0.7: {buckets[2]}  0.3-0.5: {buckets[3]}  <0.3: {buckets[4]}  "
          f"avg={buckets[5]}  total={buckets[6]}")


def statistical(apply: bool) -> None:
    conn = connect(readonly=not apply)
    show_distribution(conn, "before")
    if not apply:
        n = conn.execute(f"SELECT COUNT(*) FROM memories WHERE {FLOOD_WHERE}").fetchone()[0]
        print(f"dry-run: would rescale {n} rows (importance * 0.65, floor 0.15). "
              f"Run with --apply.")
        conn.close()
        return
    t0 = time.time()
    cur = conn.execute(
        f"UPDATE memories SET importance = MAX(0.15, ROUND(importance * 0.65, 3)) "
        f"WHERE {FLOOD_WHERE}"
    )
    conn.commit()
    print(f"rescaled {cur.rowcount} rows in {time.time()-t0:.1f}s")
    show_distribution(conn, "after")
    conn.close()


def llm(model: str | None, limit: int, allow_raise: bool) -> None:
    sys.path.insert(0, str(HERE))
    import memory_mcp as m

    # Importing memory_mcp does NOT auto-load .env (it only loads in its own
    # main()), so refresh the ollama settings from .env here before we call the
    # model. Priority: CLI --model > process env > .env > code default.
    try:
        from consolidate_sessions import _load_dotenv
        _load_dotenv(HERE)
    except Exception as exc:
        print(f"[rescore] .env load skipped: {exc}")
    m.OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", m.OLLAMA_BASE_URL)
    m.OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", m.OLLAMA_MODEL)
    m.OLLAMA_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", str(m.OLLAMA_TIMEOUT)))
    if model:
        m.OLLAMA_MODEL = model
    print(f"LLM rescoring with model={m.OLLAMA_MODEL} (limit {limit} rows this run)")

    done: set[str] = set()
    if PROGRESS.exists():
        done = set(json.loads(PROGRESS.read_text(encoding="utf-8")).get("done", []))
        print(f"resuming: {len(done)} rows already rescored")

    conn = connect()
    rows = conn.execute(
        f"SELECT id, key, content, importance FROM memories WHERE {FLOOD_WHERE} "
        f"ORDER BY created_at, id"
    ).fetchall()
    todo = [r for r in rows if r["id"] not in done][:limit]
    print(f"{len(todo)} rows to score this run ({len(rows)} in flood window)")

    changed = 0
    for i, r in enumerate(todo):
        try:
            raw = m._call_ollama(SCORE_PROMPT.format(key=r["key"], content=r["content"][:500]))
            match = re.search(r"(0?\.\d+|1\.0|1)", raw)
            if not match:
                done.add(r["id"])
                continue
            score = max(0.1, min(1.0, float(match.group(1))))
            if not allow_raise:
                score = min(score, r["importance"])
            if abs(score - r["importance"]) >= 0.05:
                conn.execute("UPDATE memories SET importance=? WHERE id=?", (score, r["id"]))
                changed += 1
        except Exception as exc:
            print(f"  row {r['id']}: {type(exc).__name__}: {exc} — stopping (resumable)")
            break
        done.add(r["id"])
        if (i + 1) % 50 == 0:
            conn.commit()
            PROGRESS.write_text(json.dumps({"done": sorted(done)}), encoding="utf-8")
            print(f"  {i+1}/{len(todo)} scored, {changed} changed")
    conn.commit()
    PROGRESS.write_text(json.dumps({"done": sorted(done)}), encoding="utf-8")
    show_distribution(conn, "after")
    conn.close()
    print(f"done: {changed} rows changed this run")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--apply", action="store_true", help="statistical mode: actually write (default dry-run)")
    ap.add_argument("--llm", action="store_true", help="per-row LLM re-scoring instead of statistical")
    ap.add_argument("--model", default=None, help="Ollama model override for --llm")
    ap.add_argument("--limit", type=int, default=500, help="max rows per --llm run")
    ap.add_argument("--allow-raise", action="store_true", help="--llm may raise scores, not only lower")
    args = ap.parse_args()
    if args.llm:
        llm(args.model, args.limit, args.allow_raise)
    else:
        statistical(args.apply)


if __name__ == "__main__":
    main()
