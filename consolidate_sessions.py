"""Consolidate per-session memory fragments into continuous narratives.

Runs a remote (or local) LLM over each session's fragments and merges them
into a smaller set of narrative records, preserving chronology via created_at
and session_id. Original fragments are marked digested=1 (not deleted).

Usage:
    # OpenRouter (recommended — Gemini 3.1 Flash Lite)
    set OPENROUTER_API_KEY=sk-or-...
    python consolidate_sessions.py

    # Ollama fallback (local, weaker, slower)
    set LLM_BACKEND=ollama
    python consolidate_sessions.py --workers 1

Safe to re-run: skips sessions whose non-digested fragment count is < min-fragments.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sqlite3
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Config (all from env)
# ---------------------------------------------------------------------------

LLM_BACKEND = os.environ.get("LLM_BACKEND", "openrouter").lower()

# OpenRouter
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "google/gemini-3.1-flash-lite-preview")

# Ollama fallback
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:e4b")

LLM_TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "120"))
MAX_CONSECUTIVE_FAILURES = 10

# call_llm failover / retry tuning
_LLM_RETRY_MAX = 2          # same-side retries for 429/408 before failing over
_LLM_RETRY_BASE = 2.0       # seconds — exponential backoff base
_LLM_RETRY_CAP = 120.0      # cap any single wait (incl. Retry-After) at 2 min

# Count of calls silently served by ollama after the cloud side fell over, so a
# batch run can surface how much quietly degraded to the local model.
_llm_fallback_lock = threading.Lock()
_llm_fallback_to_ollama = 0


# ---------------------------------------------------------------------------
# LLM backends (OpenAI-compatible chat/completions schema for both paths)
# ---------------------------------------------------------------------------

def _call_openrouter(prompt: str) -> str:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set in environment")
    url = f"{OPENROUTER_BASE_URL.rstrip('/')}/chat/completions"
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "stream": False,
        # Gemini 2.5/3.x Flash Lite caps output at ~64k. Large sessions with 20
        # merged records × ~500 Chinese chars need ~15-20k tokens room — default
        # caps on OpenRouter can be as low as 2-4k and silently truncate.
        "max_tokens": 32000,
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": "https://github.com/SolenmeChiara/Sol-Memory-mcp",
            "X-Title": "Sol-Memory-mcp consolidate",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=LLM_TIMEOUT) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    parsed = json.loads(body)
    # Some OpenAI-compatible gateways (OpenRouter seen doing this) answer HTTP 200
    # with a top-level {"error": ...} instead of an HTTP status. Classify by the
    # embedded code so failover treats it like the equivalent HTTP status — and so
    # we never blow up on a bare KeyError('choices'). Mirrors reextract 93f508e.
    err = parsed.get("error")
    if err:
        emsg = (err.get("message") if isinstance(err, dict) else str(err)) or str(err)
        ecode = err.get("code") if isinstance(err, dict) else None
        try:
            ecode = int(ecode)
        except (TypeError, ValueError):
            ecode = None
        if ecode in (429, 408) or (ecode is not None and ecode >= 500):
            raise urllib.error.HTTPError(url, ecode, emsg, None, None)  # transient → retry/failover
        raise RuntimeError(f"OpenRouter 内容/策略拒绝（code={ecode}）：{emsg}")  # content → no failover
    if not parsed.get("choices"):
        raise RuntimeError(f"OpenRouter 返回无 choices：{str(parsed)[:200]}")
    choice = parsed["choices"][0]
    finish_reason = choice.get("finish_reason", "stop")
    content = choice["message"]["content"]
    if finish_reason == "length":
        raise RuntimeError(
            "LLM output truncated by max_tokens (session too large to merge in one shot). "
            "Consider lowering --max-fragments or bumping max_tokens further."
        )
    return content


def _call_ollama_local(prompt: str) -> str:
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "stream": False,
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=LLM_TIMEOUT) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    parsed = json.loads(body)
    return parsed["choices"][0]["message"]["content"]


def _is_infra_error(exc: Exception) -> bool:
    """True for 'can't reach the backend' errors that another backend might survive.

    Infra: connection refused / DNS / timeout (any OSError incl. URLError), plus
    HTTP 401/403/404/408/429/5xx (auth, wrong URL/model, rate-limit, server down).
    NOT infra: HTTP 400 (a content/policy rejection is a per-prompt problem — the
    other backend would reject it too), and RuntimeError (truncation / bad JSON /
    unknown backend). Those propagate without a pointless failover.
    """
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code != 400
    # URLError, socket.timeout(=TimeoutError), ConnectionError all subclass OSError
    return isinstance(exc, OSError)


def _backend_configured(name: str) -> bool:
    """Config-level availability (not live reachability)."""
    if name == "openrouter":
        return bool(OPENROUTER_API_KEY)
    if name == "ollama":
        return True  # local, always has a default URL + model
    return False


def _dispatch(name: str, prompt: str) -> str:
    return _call_openrouter(prompt) if name == "openrouter" else _call_ollama_local(prompt)


def _http_code(exc: Exception) -> "int | None":
    return exc.code if isinstance(exc, urllib.error.HTTPError) else None


def _retry_after_seconds(exc: Exception) -> "float | None":
    """Integer-seconds Retry-After header if present, capped at _LLM_RETRY_CAP."""
    if isinstance(exc, urllib.error.HTTPError) and exc.headers:
        ra = exc.headers.get("Retry-After")
        if ra:
            try:
                return min(float(int(ra)), _LLM_RETRY_CAP)
            except (TypeError, ValueError):
                return None
    return None


def _call_with_same_side_retry(name: str, prompt: str) -> str:
    """Call one backend; on 429/408 back off same-side (Retry-After honoured, cap
    120s, up to _LLM_RETRY_MAX) before giving up. Re-raises the last error so the
    caller can then decide whether to fail over to the other side."""
    attempt = 0
    while True:
        try:
            return _dispatch(name, prompt)
        except Exception as exc:
            code = _http_code(exc)
            if code in (429, 408) and attempt < _LLM_RETRY_MAX:
                ra = _retry_after_seconds(exc)
                delay = ra if ra is not None else min(_LLM_RETRY_BASE * (2 ** attempt), _LLM_RETRY_CAP)
                attempt += 1
                sys.stderr.write(
                    f"[consolidate] {name} HTTP {code}，本侧退避 {delay:.0f}s 重试"
                    f"（{attempt}/{_LLM_RETRY_MAX}）\n"
                )
                time.sleep(delay)
                continue
            raise


def _note_ollama_fallback() -> None:
    global _llm_fallback_to_ollama
    with _llm_fallback_lock:
        _llm_fallback_to_ollama += 1


def fallback_count() -> int:
    with _llm_fallback_lock:
        return _llm_fallback_to_ollama


def call_llm(prompt: str) -> str:
    """Call the LLM_BACKEND-preferred side, auto-failing-over to the other side on
    infrastructure errors (not on content/policy 400s). 429/408 are first retried
    same-side with backoff (Retry-After honoured); only if they persist do we
    switch. Logs any switch."""
    order = {
        "openrouter": ("openrouter", "ollama"),
        "ollama": ("ollama", "openrouter"),
    }.get(LLM_BACKEND)
    if order is None:
        raise RuntimeError(f"unknown LLM_BACKEND: {LLM_BACKEND}")
    primary, secondary = order

    # If the preferred side isn't even configured (e.g. openrouter selected but no
    # key), go straight to the other side when IT is configured.
    if not _backend_configured(primary):
        if _backend_configured(secondary):
            sys.stderr.write(f"[consolidate] {primary} 未配置，改用 {secondary}\n")
            primary, secondary = secondary, None
        else:
            raise RuntimeError(
                "没有可用的解析通道：openrouter 没配 key，ollama 也不可用。"
                "请检查 .env 里的 URL 和 key。"
            )

    try:
        return _call_with_same_side_retry(primary, prompt)
    except Exception as exc:
        if not _is_infra_error(exc):
            raise  # content / policy / truncation — switching backend won't help
        if secondary and _backend_configured(secondary):
            sys.stderr.write(
                f"[consolidate] {primary} 基建性故障（{type(exc).__name__}: {exc}）→ "
                f"自动切到 {secondary} 重试本次调用\n"
            )
            try:
                out = _call_with_same_side_retry(secondary, prompt)
            except Exception as exc2:
                raise RuntimeError(
                    f"两侧解析通道都拉不到（{primary}: {exc}；{secondary}: {exc2}）。"
                    f"请检查 URL 和 key。"
                ) from exc2
            if secondary == "ollama":
                _note_ollama_fallback()
            return out
        raise RuntimeError(
            f"{primary} 拉不到（{type(exc).__name__}: {exc}），且没有可切换的备用通道。"
            f"请检查 URL 和 key。"
        ) from exc


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def _parse_json_list(raw: str) -> list:
    import re as _re
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        parsed = json.loads(cleaned)
        return [parsed] if not isinstance(parsed, list) else parsed
    except json.JSONDecodeError:
        m = _re.search(r"\[.*\]", cleaned, _re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group())
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass
        raise RuntimeError(f"LLM output is not valid JSON array. First 200 chars: {cleaned[:200]}")


def build_merge_prompt(fragments: list[dict], target_count: int) -> str:
    frag_block = []
    for i, f in enumerate(fragments, 1):
        ts = f["created_at"]
        frag_block.append(
            f"[{i}] {ts} | category={f['category']} | importance={f['importance']:.2f}\n"
            f"    key: {f['key']}\n"
            f"    content: {f['content']}"
        )
    frag_text = "\n\n".join(frag_block)

    return (
        f"你是一个记忆整理助手。下面是来自**同一次对话**的 {len(fragments)} 条记忆碎片，"
        f"按时间升序排列。这些碎片之前被 LLM 拆得太碎——你的任务是把它们合并成 "
        f"**{target_count} 条**连续叙事记忆。\n"
        f"\n"
        f"### 合并规则\n"
        f"1. 围绕同一主题/事件的细节**合并到同一条 content 里**，保留时间顺序、因果、情感。\n"
        f"2. content 可以较长（数百字到上千字），优先完整性而非简洁。\n"
        f"3. 只有**真正不同主题**才应分开成多条（同一段对话里既聊了约会又聊了工作那种）。\n"
        f"4. 重复信息去重；互相矛盾的细节保留时间较晚的。\n"
        f"5. 输出条数必须是 **{target_count} 条**（不能多也不能少）。\n"
        f"\n"
        f"### 输出格式（严格）\n"
        f"只输出纯 JSON 数组，不加说明、不加 markdown 代码块。每条记忆包含：\n"
        f"- key：简短标题（≤20 字）\n"
        f"- content：连续叙事\n"
        f"- category：preference / promise / event / anniversary / emotion / habit / boundary / other\n"
        f"- importance：0.0~1.0\n"
        f"\n"
        f'[{{"key":"...","content":"...","category":"...","importance":0.7}}, ...]\n'
        f"\n"
        f"### 原始碎片（{len(fragments)} 条，时序升序）\n"
        f"\n"
        f"{frag_text}\n"
    )


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

_DB_LOCK = threading.RLock()


def _target_count(n: int) -> int:
    """Dynamic merge target: N/3 clamped to [3, 20]."""
    return max(3, min(20, round(n / 3)))


def _majority_category(cats: list[str]) -> str:
    from collections import Counter
    if not cats:
        return "other"
    return Counter(cats).most_common(1)[0][0]


def _weighted_emotion(fragments: list[dict]) -> tuple[float, float]:
    weights = [max(0.01, f["importance"]) for f in fragments]
    wsum = sum(weights) or 1.0
    v = sum(f["valence"] * w for f, w in zip(fragments, weights)) / wsum
    a = sum(f["arousal"] * w for f, w in zip(fragments, weights)) / wsum
    return round(v, 3), round(a, 3)


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Add the consolidated column if it doesn't exist (safe in either direction —
    memory_mcp.py's MemoryStore also does this on startup, whichever runs first wins).

    Also back-fills consolidated=1 on any existing merge-products (id like
    'mem_merged_%') so earlier test runs don't get re-consolidated.
    """
    with _DB_LOCK:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(memories)").fetchall()}
        if "consolidated" not in cols:
            conn.execute("ALTER TABLE memories ADD COLUMN consolidated INTEGER DEFAULT 0")
            sys.stderr.write("[info] added 'consolidated' column to memories table\n")
        cur = conn.execute(
            "UPDATE memories SET consolidated = 1 WHERE id LIKE 'mem_merged_%' AND consolidated = 0"
        )
        # Always commit — even if rowcount was 0, the DML started an implicit
        # transaction that must be closed before any caller tries `with conn:`.
        conn.commit()
        if cur.rowcount > 0:
            sys.stderr.write(f"[info] back-filled consolidated=1 on {cur.rowcount} existing merge-products\n")


def fetch_sessions_to_process(conn: sqlite3.Connection, min_fragments: int) -> list[tuple[str, int]]:
    """Return list of (session_id, non_digested_non_consolidated_count) where count >= min_fragments.

    Excluding consolidated=1 records ensures:
      1. The merge-products we just created aren't re-consolidated.
      2. Re-running consolidate only picks up truly new fragment accumulations,
         never re-shuffles already-merged material.
    """
    rows = conn.execute(
        "SELECT session_id, COUNT(*) AS n FROM memories "
        "WHERE session_id != '' AND digested = 0 AND consolidated = 0 "
        "AND memory_kind='long_term' "
        "GROUP BY session_id HAVING n >= ? ORDER BY n DESC",
        (min_fragments,),
    ).fetchall()
    return [(r[0], r[1]) for r in rows]


def fetch_fragments(conn: sqlite3.Connection, session_id: str) -> list[dict]:
    # sqlite3 connections aren't fully thread-safe for concurrent execute();
    # worker threads share one connection, so serialise reads too.
    with _DB_LOCK:
        rows = conn.execute(
            "SELECT id, key, content, category, importance, valence, arousal, created_at "
            "FROM memories WHERE session_id = ? AND digested = 0 AND consolidated = 0 "
            "AND memory_kind='long_term' ORDER BY created_at ASC",
            (session_id,),
        ).fetchall()
    return [
        {
            "id": r[0], "key": r[1], "content": r[2], "category": r[3],
            "importance": float(r[4] or 0.5),
            "valence": float(r[5] or 0.5), "arousal": float(r[6] or 0.3),
            "created_at": r[7],
        }
        for r in rows
    ]


def write_merged(
    conn: sqlite3.Connection,
    session_id: str,
    fragments: list[dict],
    merged_items: list[dict],
) -> int:
    """Insert merged records, mark original fragments digested. Returns number inserted."""
    if not merged_items:
        return 0

    earliest_ts = fragments[0]["created_at"]
    v_agg, a_agg = _weighted_emotion(fragments)
    cat_agg = _majority_category([f["category"] for f in fragments])
    max_imp = max((f["importance"] for f in fragments), default=0.5)
    now = datetime.now(timezone.utc).isoformat()
    frag_ids = [f["id"] for f in fragments]

    inserted = 0
    with _DB_LOCK:
        # Use sqlite3's context manager for transactions — commits on normal
        # exit, rolls back on exception. Avoids "cannot start a transaction
        # within a transaction" errors on Python 3.12+ (explicit BEGIN is
        # brittle once any prior DML has left an implicit txn open).
        with conn:
            for item in merged_items:
                if not isinstance(item, dict):
                    continue
                content = str(item.get("content", "")).strip()
                if not content:
                    continue
                key = str(item.get("key", ""))[:60] or "untitled"
                category = str(item.get("category", "")).strip() or cat_agg
                try:
                    importance = max(0.0, min(1.0, float(item.get("importance", max_imp))))
                except (TypeError, ValueError):
                    importance = max_imp
                new_id = (
                    f"mem_merged_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
                    f"_{random.randint(1000, 9999)}"
                )
                conn.execute(
                    """
                    INSERT INTO memories(
                        id, key, content, memory_kind, category, importance, session_id,
                        created_at, updated_at, embedding,
                        valence, arousal, pinned, resolved, digested, activation_count, last_active,
                        consolidated
                    ) VALUES (?, ?, ?, 'long_term', ?, ?, ?, ?, ?, X'',
                              ?, ?, 0, 0, 0, 1.0, ?, 1)
                    """,
                    (new_id, key, content, category, importance, session_id,
                     earliest_ts, now, v_agg, a_agg, now),
                )
                inserted += 1
            placeholders = ",".join("?" * len(frag_ids))
            conn.execute(
                f"UPDATE memories SET digested = 1, updated_at = ? WHERE id IN ({placeholders})",
                (now, *frag_ids),
            )
    return inserted


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_dotenv(root: Path) -> None:
    """Minimal .env loader — no external dependency."""
    env_path = root / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def run_consolidate(
    db_path: Path,
    *,
    min_fragments: int = 5,
    max_fragments: int = 0,
    workers: int = 2,
    limit: int = 0,
    dry_run: bool = False,
    backend: "str | None" = None,
    progress_cb=None,
    stop_event: "threading.Event | None" = None,
) -> dict:
    """Consolidate session fragments. Reusable by CLI and web server.

    progress_cb: optional callable(dict) invoked every 25 sessions and at completion.
    stop_event: threading.Event — set to request graceful stop after the current batch.

    Returns state dict with: total, processed, success, skipped, failed,
    frags_in, records_out, elapsed_s, aborted, aborted_reason, stage, last_error.
    """
    # Honour .env that the caller may have set already; re-read in case web server didn't.
    _load_dotenv(db_path.parent)
    # Module-level config may have been set by .env — pick up latest.
    # An explicit `backend` arg (e.g. the web server forcing ollama when there's
    # no cloud key) wins over the env default.
    global LLM_BACKEND, OPENROUTER_API_KEY, OPENROUTER_MODEL, OLLAMA_MODEL
    LLM_BACKEND = (backend or os.environ.get("LLM_BACKEND", LLM_BACKEND)).lower()
    OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", OPENROUTER_API_KEY)
    OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", OPENROUTER_MODEL)
    OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", OLLAMA_MODEL)

    if LLM_BACKEND == "openrouter" and not OPENROUTER_API_KEY and not dry_run:
        raise RuntimeError("OPENROUTER_API_KEY not set (put it in .env or export it)")

    # Snapshot the fallback counter so we can report how many of THIS run's calls
    # quietly degraded to the local ollama model after the cloud side failed.
    _fb_start = fallback_count()

    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
    ensure_schema(conn)

    sessions = fetch_sessions_to_process(conn, min_fragments)
    if max_fragments > 0:
        sessions = [s for s in sessions if s[1] <= max_fragments]
    if limit > 0:
        sessions = sessions[:limit]
    total = len(sessions)
    total_frags = sum(n for _, n in sessions)

    state = {
        "total": total, "total_frags": total_frags,
        "processed": 0, "success": 0, "skipped": 0, "failed": 0,
        "frags_in": 0, "records_out": 0,
        "rate_per_s": 0.0, "eta_min": 0.0,
        "elapsed_s": 0.0,
        "aborted": False, "aborted_reason": "",
        "last_error": "", "last_session": "",
        "backend": LLM_BACKEND,
        "model": OPENROUTER_MODEL if LLM_BACKEND == "openrouter" else OLLAMA_MODEL,
        "dry_run": dry_run,
        "stage": "running",
    }

    def _emit() -> None:
        if progress_cb:
            try:
                progress_cb(dict(state))
            except Exception as exc:
                sys.stderr.write(f"[run_consolidate] progress_cb error: {exc}\n")

    if total == 0:
        state["stage"] = "done"
        state["fallback_ollama"] = 0
        conn.close()
        _emit()
        return state

    consecutive = 0
    consecutive_lock = threading.Lock()
    t0 = time.monotonic()

    def worker(session_info: tuple[str, int]) -> tuple:
        nonlocal consecutive
        if stop_event is not None and stop_event.is_set():
            return ("skip", session_info[0], 0, 0)
        sid, frag_count = session_info
        if dry_run:
            return ("dry", sid, frag_count, _target_count(frag_count))
        fragments = fetch_fragments(conn, sid)
        if len(fragments) < min_fragments:
            return ("skip", sid, len(fragments), 0)
        tgt = _target_count(len(fragments))
        prompt = build_merge_prompt(fragments, tgt)
        try:
            raw = call_llm(prompt)
            items = _parse_json_list(raw)
            inserted = write_merged(conn, sid, fragments, items)
            with consecutive_lock:
                consecutive = 0
            return ("ok", sid, len(fragments), inserted)
        except Exception as exc:
            with consecutive_lock:
                consecutive += 1
                cf = consecutive
            return ("fail", sid, f"{type(exc).__name__}: {exc}", cf)

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i, result in enumerate(pool.map(worker, sessions), 1):
                kind = result[0]
                if kind == "ok":
                    _, sid, fin, inserted = result
                    state["success"] += 1
                    state["frags_in"] += fin
                    state["records_out"] += inserted
                    state["last_session"] = f"{sid}: {fin} → {inserted}"
                elif kind == "dry":
                    _, sid, fin, tgt = result
                    state["frags_in"] += fin
                    state["records_out"] += tgt
                    state["last_session"] = f"{sid}: {fin} → {tgt}"
                elif kind == "skip":
                    state["skipped"] += 1
                else:
                    _, sid, err_msg, cf = result
                    state["failed"] += 1
                    state["last_error"] = f"{sid} {err_msg}"
                    sys.stderr.write(f"[FAIL {state['last_error']}] (consecutive={cf})\n")
                    if cf >= MAX_CONSECUTIVE_FAILURES:
                        state["aborted"] = True
                        state["aborted_reason"] = f"{cf} consecutive LLM failures"
                        break
                state["processed"] = i
                if i % 25 == 0 or i == total:
                    elapsed = time.monotonic() - t0
                    state["rate_per_s"] = i / elapsed if elapsed > 0 else 0.0
                    state["eta_min"] = (
                        (total - i) / state["rate_per_s"] / 60
                        if state["rate_per_s"] > 0 else 0.0
                    )
                    _emit()
    finally:
        conn.close()

    state["elapsed_s"] = time.monotonic() - t0
    state["stage"] = "aborted" if state["aborted"] else "done"
    state["fallback_ollama"] = fallback_count() - _fb_start
    if state["fallback_ollama"] > 0:
        sys.stderr.write(
            f"[consolidate] 本地兜底了 {state['fallback_ollama']} 条"
            f"（云端拉不到时自动切到 ollama 完成的调用）\n"
        )
    _emit()
    return state


def main() -> None:
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

    parser = argparse.ArgumentParser(
        description="Consolidate per-session memory fragments into continuous narratives"
    )
    parser.add_argument("--db", default="memory.db")
    parser.add_argument("--min-fragments", type=int, default=5,
                        help="Only process sessions with at least N non-digested fragments (default 5)")
    parser.add_argument("--max-fragments", type=int, default=0,
                        help="Skip sessions with MORE than N fragments (0 = no upper bound)")
    parser.add_argument("--workers", type=int, default=2, help="Concurrent LLM calls (default 2)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process at most N sessions (0 = all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="List sessions that would be processed, no LLM calls, no DB writes")
    args = parser.parse_args()

    db_path = Path(args.db).resolve()
    if not db_path.exists():
        print(f"[ERROR] db not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    # Preamble — show what we're about to do (run_consolidate itself is quiet)
    conn_preview = sqlite3.connect(str(db_path), check_same_thread=False)
    ensure_schema(conn_preview)
    sessions_preview = fetch_sessions_to_process(conn_preview, args.min_fragments)
    if args.max_fragments > 0:
        sessions_preview = [s for s in sessions_preview if s[1] <= args.max_fragments]
    if args.limit > 0:
        sessions_preview = sessions_preview[: args.limit]
    total_preview = len(sessions_preview)
    total_frags_preview = sum(n for _, n in sessions_preview)
    conn_preview.close()

    if total_preview == 0:
        print("No sessions to consolidate (all below --min-fragments or already digested).", file=sys.stderr)
        return

    _backend_label = os.environ.get("LLM_BACKEND", LLM_BACKEND).lower()
    _model_label = (
        os.environ.get("OPENROUTER_MODEL", OPENROUTER_MODEL)
        if _backend_label == "openrouter"
        else os.environ.get("OLLAMA_MODEL", OLLAMA_MODEL)
    )
    sys.stderr.write(
        f"Consolidating {total_preview} sessions ({total_frags_preview} fragments total)  "
        f"backend={_backend_label}  model={_model_label}  workers={args.workers}\n"
    )
    if args.dry_run:
        sys.stderr.write("--- DRY-RUN: no LLM calls, no DB writes ---\n")
    sys.stderr.flush()

    def cli_progress(state: dict) -> None:
        if state["stage"] != "running":
            return
        if args.dry_run and state.get("last_session"):
            # In dry-run we show each session line as in the old main()
            sys.stderr.write(f"[{state['processed']}/{state['total']}] {state['last_session']}\n")
        else:
            sys.stderr.write(
                f"[{state['processed']}/{state['total']}] "
                f"ok={state['success']} skip={state['skipped']} fail={state['failed']} "
                f"frags_in={state['frags_in']} records_out={state['records_out']} "
                f"{state['rate_per_s']:.2f}/s ETA {state['eta_min']:.1f}min\n"
            )
        sys.stderr.flush()

    final = run_consolidate(
        db_path,
        min_fragments=args.min_fragments,
        max_fragments=args.max_fragments,
        workers=args.workers,
        limit=args.limit,
        dry_run=args.dry_run,
        progress_cb=cli_progress,
    )

    sys.stderr.write(
        f"\nDone. processed={final['success']} skipped={final['skipped']} failed={final['failed']} "
        f"frags_consumed={final['frags_in']} records_created={final['records_out']} "
        f"elapsed={final['elapsed_s']/60:.1f}min\n"
    )
    if not args.dry_run and final["success"] > 0:
        sys.stderr.write(
            "\n提醒：新合并记忆的 embedding 留空，跑 `python reindex_embeddings.py` 补齐。\n"
        )
    if final["aborted"]:
        sys.stderr.write(f"\n[ABORTED] {final['aborted_reason']}\n")
        sys.exit(2)


if __name__ == "__main__":
    main()
