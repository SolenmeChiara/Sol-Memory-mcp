#!/usr/bin/env python3
"""海床换血 — re-extract the historical conversation export into the seabed tier.

The April 2026 batch import used a weak local model (gemma4:e4b) and produced
~45,932 fragmented, mistranslated seabed rows. This pipeline re-reads the same
official claude.ai export (2,339 conversations) with a cloud model (OpenRouter
Gemini 3.1 Flash Lite) and a hardened prompt, writing a much smaller set of
faithful, chronologically-anchored memories back into the seabed tier.

It reuses the sibling scripts rather than duplicating their logic:
- batch_import.py  — conversation parsing (_conv_to_text / _raw_items /
  _normalize_iso_ts / _parse_json_list). batch_import.py is left byte-for-byte
  unchanged; we only import from it.
- consolidate_sessions.py — .env loader (_load_dotenv).
- reindex_embeddings.py — local Ollama embedding (fetch_embedding / pack_embedding,
  qwen3-embedding:4b, 2560-dim little-endian float32).

Two independent modes:

  1. IMPORT (default) — re-extract + write seabed rows. Long-running (~2k API
     calls). Resumable, concurrent, billed-metered, single-conversation-atomic.

  2. RETIRE (--retire) — soft-retire the OLD gemma seabed rows by flipping
     digested=1, resolved=1 so they leave search / vector index / list / random
     and are eventually hard-deleted by the store's 90-day prune daemon. Runs
     independently of the import.

Usage:
    # dry-run: parse + count + cost, no LLM, no writes
    python reextract_seabed.py --dry-run

    # real import into a TEST db (never the production memory.db without sign-off)
    python reextract_seabed.py --db test.db --limit 3

    # full run into production (only after explicit sign-off)
    python reextract_seabed.py --workers 4

    # resume: already-written conversations are skipped automatically
    python reextract_seabed.py --workers 4

    # retry only the conversations recorded in reextract_failed.json
    python reextract_seabed.py --retry-failed

    # soft-retire the old gemma seabed (read-only count first)
    python reextract_seabed.py --retire --dry-run
    python reextract_seabed.py --retire            # asks for confirmation
    python reextract_seabed.py --retire --rollback # undo a prior retire
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
from typing import Any, Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
# Import the sibling maintenance scripts (spec: reuse, don't fork).
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import batch_import as bi          # noqa: E402  (path insert must precede import)
import consolidate_sessions as cs  # noqa: E402
import reindex_embeddings as ri    # noqa: E402

DEFAULT_SRC = (
    r"E:\AI\CLAUDE"
    r"\data-4ca35ff9-5803-4b8a-b631-70649982853f-1776724981-29abb56a-batch-0000"
    r"\conversations.json"
)
DEFAULT_DB = HERE / "memory.db"
# The failed-idx ledger and retire snapshot live next to the script by default;
# tests override the directory via REEXTRACT_STATE_DIR to stay out of the repo.
_STATE_DIR = Path(os.environ.get("REEXTRACT_STATE_DIR", str(HERE)))
FAILED_FILE = _STATE_DIR / "reextract_failed.json"
# Conversations that were successfully processed but yielded 0 memories (the LLM
# legitimately returned [] or every item was sanitised away). They leave no DB
# row, so "session exists -> skip" can never catch them; this ledger stops a
# resume from re-calling (re-paying for) them. Distinct from reextract_failed.json
# (real errors, retried by --retry-failed): a done-empty idx is COMPLETE, not failed.
DONE_EMPTY_FILE = _STATE_DIR / "reextract_done_empty.json"
RETIRE_SNAPSHOT = _STATE_DIR / "reextract_retire_snapshot.json"

SESSION_PREFIX = "reextract_conversations_"

# Gemini 3.1 Flash Lite has a 1M-token context. Keep every conversation whole so
# the model sees the full narrative — only split truly gigantic ones. The largest
# conversation in this export is ~456k chars (~223k tokens), so 500k keeps all of
# them whole; the split is a safety valve for pathological inputs. Overridable.
SESSION_HARD_CHUNK_CHARS = int(os.environ.get("REEXTRACT_HARD_CHUNK_CHARS", "500000"))
MAX_MEMORIES_PER_CHUNK = 8

# OpenRouter output cap. 8 memories x ~130 completion tok ≈ 1k tok, so 16k is
# generous headroom even for a big multi-chunk conversation.
OPENROUTER_MAX_TOKENS = int(os.environ.get("REEXTRACT_MAX_TOKENS", "16000"))
LLM_TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "180"))

# Retry/backoff for the preview model (it rate-limits). Exponential with jitter.
MAX_RETRIES = int(os.environ.get("REEXTRACT_MAX_RETRIES", "3"))
RETRY_BASE = float(os.environ.get("REEXTRACT_RETRY_BASE", "2.0"))
RETRYABLE_HTTP = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

# Live pricing reverse-engineered from real usage (2 sample calls).
IN_PRICE_PER_TOK = 0.25 / 1_000_000
OUT_PRICE_PER_TOK = 1.50 / 1_000_000
SEC_PER_CALL_EST = 4.0  # measured ~4s/call for ETA

RETIRE_CUTOFF = "2026-04-24"
RETIRE_PREDICATE = (
    "tier='seabed' AND session_id LIKE 'import_conversations_%' "
    f"AND updated_at <= '{RETIRE_CUTOFF}'"
)

VALID_CATEGORIES = frozenset(
    {"preference", "promise", "event", "anniversary", "emotion", "habit", "boundary", "other"}
)

# gemma sin #1 guard: reject a "memory" whose content is really the extractor
# describing itself / the assistant's role, not something about Sol. Kept tight
# so it never nukes Sol's own words (e.g. "Sol 渴望拥有语言模型的能力" is legit and
# contains none of these role-leak phrases).
BANNED_CONTENT_SUBSTRINGS = (
    "记忆整理助手",
    "记忆管理者",
    "作为记忆",
    "作为AI助手",
    "作为 AI 助手",
    "作为一个AI",
    "作为一个 AI",
    "作为你的AI",
    "作为你的 AI",
    "作为人工智能",
    "身为记忆",
)

# The hardened extraction prompt (validated on samples 1042 / 2087: 27->5 and
# 30->4 memories, importance spread 0.40-0.95, subject re-centred on Sol).
_EXTRACT_TMPL = """你是一位温和细致的记忆管理者，在为用户 Sol 整理一本私人记忆日记。你面对的是 Sol 与 AI 助手的一段对话记录。

请从中提取值得长期珍藏的记忆。铁律如下：

【只记 Sol 本人】——最重要
- 只提取关于 Sol 这个人的：他的经历、情绪、想法、偏好、关系、决定、生活片段、创作。
- 绝对不要把 AI 助手的功能介绍、能力说明、"我可以帮你……"、通用知识科普、模型/技术八卦本身当作记忆。除非 Sol 对这些表达了个人态度或情感，只有那份"Sol 的反应"才值得记。
- 判断标准：一年后 Sol 重读这条，会觉得"这是关于我的事"，而不是"这是一段百科或客服说明"。

【反碎片化】
- 把围绕同一主题/事件的细节合并进同一条 content，保留时间顺序、因果、情感变化。宁可少而完整，不要拆成孤立的事实碎片。
- content 可较长（数百字无妨），优先完整性。只有真正互不相关的主题才拆成多条。

【importance 要有区分度】
- 用满 0.0–1.0 整个区间，不要都挤在 0.5–0.6。
- 日常闲聊/一次性的小事：0.25–0.45；值得回味的经历或明确的偏好/习惯：0.55–0.7；关乎身份认同、重要关系、人生节点、强烈情感的核心记忆：0.8–0.95。

【情感标注】valence 0.0(极消极)–1.0(极积极)，0.5中性；arousal 0.0(平静)–1.0(激动)，0.3日常。

一段对话通常整理 1–8 条，视内容丰富度而定。若整段几乎没有关于 Sol 本人的实质内容，可以只返回 1 条甚至空数组 []。

只输出纯 JSON 数组，不要任何额外说明或 markdown 代码块：
[{{"key":"简短标题(≤20字)","content":"完整叙事","category":"preference|promise|event|anniversary|emotion|habit|boundary|other","importance":0.6,"valence":0.7,"arousal":0.4}}]

对话记录：
{chunk}"""


# ---------------------------------------------------------------------------
# Environment / LLM backend
# ---------------------------------------------------------------------------

OPENROUTER_API_KEY = ""
OPENROUTER_MODEL = "google/gemini-3.1-flash-lite-preview"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def ensure_env() -> None:
    """Load .env (same loader consolidate uses) and refresh module globals."""
    global OPENROUTER_API_KEY, OPENROUTER_MODEL, OPENROUTER_BASE_URL
    cs._load_dotenv(HERE)
    OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", OPENROUTER_API_KEY)
    OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", OPENROUTER_MODEL)
    OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", OPENROUTER_BASE_URL)


class RetryableError(Exception):
    """A transient LLM failure (429 / 5xx / network) worth retrying.

    retry_after: seconds parsed from a Retry-After response header (429/503),
    honoured in preference to the computed exponential backoff when present."""

    def __init__(self, message: str, retry_after: Optional[float] = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


def _call_openrouter(prompt: str) -> Tuple[str, Dict[str, int]]:
    """One OpenRouter chat/completions call. Returns (text, usage_dict).

    Raises RetryableError for rate-limits / 5xx / network blips, and a plain
    RuntimeError for permanent failures (auth, bad request, truncation).
    """
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not set (put it in .env or export it)")
    url = f"{OPENROUTER_BASE_URL.rstrip('/')}/chat/completions"
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "stream": False,
        "max_tokens": OPENROUTER_MAX_TOKENS,
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": "https://github.com/SolenmeChiara/Sol-Memory-mcp",
            "X-Title": "Sol-Memory-mcp reextract-seabed",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=LLM_TIMEOUT) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        raw = ""
        try:
            raw = exc.read().decode("utf-8", "replace")[:300]
        except Exception:
            pass
        if exc.code in RETRYABLE_HTTP:
            retry_after: Optional[float] = None
            try:
                hdr = exc.headers.get("Retry-After") if exc.headers else None
                if hdr is not None:
                    retry_after = float(int(str(hdr).strip()))
            except (ValueError, TypeError):
                retry_after = None  # HTTP-date form or garbage — fall back to backoff
            raise RetryableError(f"HTTP {exc.code}: {raw}", retry_after=retry_after) from None
        raise RuntimeError(f"HTTP {exc.code}: {raw}") from None
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
        raise RetryableError(f"network: {type(exc).__name__}: {exc}") from None

    parsed = json.loads(body)
    choice = parsed["choices"][0]
    finish_reason = choice.get("finish_reason", "stop")
    content = choice["message"]["content"]
    usage = parsed.get("usage", {}) or {}
    usage = {
        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
    }
    if finish_reason == "length":
        raise RuntimeError(
            "LLM output truncated by max_tokens — conversation produced more than "
            f"{OPENROUTER_MAX_TOKENS} tokens of memories; raise REEXTRACT_MAX_TOKENS."
        )
    return content, usage


def call_llm_with_retry(prompt: str) -> Tuple[str, Dict[str, int]]:
    """_call_openrouter with exponential backoff on RetryableError."""
    last_exc: Optional[Exception] = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            return _call_openrouter(prompt)
        except RetryableError as exc:
            last_exc = exc
            if attempt >= MAX_RETRIES:
                break
            if exc.retry_after is not None and exc.retry_after >= 0:
                # Honour the server's Retry-After, but cap it so a bogus huge
                # value can't stall a worker indefinitely.
                sleep_s = min(exc.retry_after, 120.0)
            else:
                sleep_s = RETRY_BASE * (2 ** attempt) + random.uniform(0, 1)
            sys.stderr.write(
                f"[retry] {exc} — attempt {attempt + 1}/{MAX_RETRIES}, "
                f"backing off {sleep_s:.1f}s\n"
            )
            sys.stderr.flush()
            time.sleep(sleep_s)
    raise RuntimeError(f"exhausted {MAX_RETRIES} retries: {last_exc}")


# ---------------------------------------------------------------------------
# Conversation -> memory items
# ---------------------------------------------------------------------------

def conv_timestamps(conv: Dict[str, Any], created_ts: Optional[str]) -> Tuple[str, str]:
    """Return (created_at, updated_at) as ISO-UTC strings for one conversation.

    created_ts comes from batch_import._conv_to_text. updated_at is pulled from
    the conversation's own field; both fall back to each other, then to now().
    """
    updated_ts = bi._normalize_iso_ts(conv.get("updated_at"))
    now = datetime.now(timezone.utc).isoformat()
    created = created_ts or updated_ts or now
    updated = updated_ts or created_ts or now
    return created, updated


def _chunk_text(text: str) -> List[str]:
    if len(text) <= SESSION_HARD_CHUNK_CHARS:
        return [text]
    return [
        text[i : i + SESSION_HARD_CHUNK_CHARS]
        for i in range(0, len(text), SESSION_HARD_CHUNK_CHARS)
    ]


def _content_is_meta_leak(content: str) -> bool:
    return any(bad in content for bad in BANNED_CONTENT_SUBSTRINGS)


def sanitize_items(raw_items: List[Any]) -> Tuple[List[Dict[str, Any]], int]:
    """Coerce/validate LLM items. Returns (clean_items, rejected_count).

    Rejects: non-dicts, empty content, extractor/AI-role meta leaks, and exact
    duplicate content within the same conversation (whitespace-normalised)."""
    clean: List[Dict[str, Any]] = []
    seen: set = set()
    rejected = 0
    for it in raw_items[:MAX_MEMORIES_PER_CHUNK]:
        if not isinstance(it, dict):
            rejected += 1
            continue
        content = str(it.get("content", "")).strip()
        if not content or _content_is_meta_leak(content):
            rejected += 1
            continue
        norm = " ".join(content.split())
        if norm in seen:
            rejected += 1
            continue
        seen.add(norm)
        category = str(it.get("category", "other")).strip().lower()
        if category not in VALID_CATEGORIES:
            category = "other"
        try:
            importance = max(0.0, min(1.0, float(it.get("importance", 0.5))))
        except (TypeError, ValueError):
            importance = 0.5
        try:
            valence = max(0.0, min(1.0, float(it.get("valence", 0.5))))
        except (TypeError, ValueError):
            valence = 0.5
        try:
            arousal = max(0.0, min(1.0, float(it.get("arousal", 0.3))))
        except (TypeError, ValueError):
            arousal = 0.3
        clean.append(
            {
                "key": str(it.get("key", ""))[:60] or "untitled",
                "content": content,
                "category": category,
                "importance": importance,
                "valence": valence,
                "arousal": arousal,
            }
        )
    return clean, rejected


# ---------------------------------------------------------------------------
# DB
# ---------------------------------------------------------------------------

_DB_LOCK = threading.RLock()


def verify_schema(conn: sqlite3.Connection) -> None:
    cols = {r[1] for r in conn.execute("PRAGMA table_info(memories)").fetchall()}
    if not cols:
        raise SystemExit("[ERROR] no 'memories' table — is this a memory store db?")
    for needed in ("tier", "tier_until", "consolidated"):
        if needed not in cols:
            raise SystemExit(
                f"[ERROR] '{needed}' column missing — run migrate_tiers.py / start the "
                f"MCP once on this db first."
            )


def backup_db(db_path: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    # Backups go in a backups/ subdir (gitignored) rather than the repo root, so
    # the working tree stays clean.
    backup_dir = db_path.parent / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    target = backup_dir / f"{db_path.name}.bak-{stamp}"
    src = sqlite3.connect(str(db_path), timeout=30)
    dst = sqlite3.connect(str(target))
    try:
        with dst:
            src.backup(dst)
    finally:
        dst.close()
        src.close()
    size_mb = target.stat().st_size // 1048576
    sys.stderr.write(f"[backup] -> {target.name} ({size_mb} MB)\n")
    sys.stderr.flush()
    return target


def _new_id(created_at: str) -> str:
    """mem_ id anchored to the conversation's real created_at so ids sort by
    the true timeline (matches batch_import's convention)."""
    try:
        anchor = datetime.fromisoformat(created_at)
    except (ValueError, TypeError):
        anchor = datetime.now(timezone.utc)
    return f"mem_{anchor.strftime('%Y%m%d%H%M%S%f')}_{random.randint(1000, 9999)}"


def session_done(conn: sqlite3.Connection, session_id: str) -> bool:
    with _DB_LOCK:
        row = conn.execute(
            "SELECT 1 FROM memories WHERE session_id = ? LIMIT 1", (session_id,)
        ).fetchone()
    return row is not None


def existing_sessions(conn: sqlite3.Connection) -> set:
    with _DB_LOCK:
        rows = conn.execute(
            "SELECT DISTINCT session_id FROM memories WHERE session_id LIKE ?",
            (f"{SESSION_PREFIX}%",),
        ).fetchall()
    return {r[0] for r in rows}


def write_memories(
    conn: sqlite3.Connection,
    session_id: str,
    items: List[Dict[str, Any]],
    embeddings: List[bytes],
    created_at: str,
    updated_at: str,
) -> int:
    """Insert one conversation's memories in a single transaction.

    Seabed retrieval-layer row: tier='seabed', digested=0 (stays in FTS + vector
    matrix), consolidated=1 (never re-merged by consolidate_sessions, never nags
    the dashboard), pinned=0, resolved=0. created_at/updated_at/last_active are
    the conversation's real timestamps — this is what restores the true timeline.
    """
    inserted = 0
    with _DB_LOCK:
        with conn:
            for item, emb in zip(items, embeddings):
                mem_id = ""
                for _ in range(6):  # retry the astronomically rare PK collision
                    candidate = _new_id(created_at)
                    exists = conn.execute(
                        "SELECT 1 FROM memories WHERE id = ?", (candidate,)
                    ).fetchone()
                    if not exists:
                        mem_id = candidate
                        break
                if not mem_id:
                    continue
                conn.execute(
                    """
                    INSERT INTO memories(
                        id, key, content, memory_kind, category, importance, session_id,
                        created_at, updated_at, embedding,
                        valence, arousal, pinned, resolved, digested, activation_count,
                        last_active, consolidated, tier, tier_until
                    ) VALUES (?, ?, ?, 'long_term', ?, ?, ?, ?, ?, ?, ?, ?,
                              0, 0, 0, 1.0, ?, 1, 'seabed', '')
                    """,
                    (
                        mem_id, item["key"], item["content"], item["category"],
                        item["importance"], session_id, created_at, updated_at, emb,
                        item["valence"], item["arousal"], updated_at,
                    ),
                )
                inserted += 1
    return inserted


# ---------------------------------------------------------------------------
# Failed-idx tracking
# ---------------------------------------------------------------------------

def load_failed() -> Dict[str, str]:
    if not FAILED_FILE.exists():
        return {}
    try:
        data = json.loads(FAILED_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except (json.JSONDecodeError, OSError):
        pass
    return {}


def save_failed(failed: Dict[str, str]) -> None:
    try:
        FAILED_FILE.write_text(
            json.dumps(failed, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError as exc:
        sys.stderr.write(f"[warn] could not write {FAILED_FILE.name}: {exc}\n")


def load_done_empty() -> set:
    if not DONE_EMPTY_FILE.exists():
        return set()
    try:
        data = json.loads(DONE_EMPTY_FILE.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return {int(x) for x in data}
    except (json.JSONDecodeError, OSError, ValueError, TypeError):
        pass
    return set()


def save_done_empty(done_empty: set) -> None:
    try:
        DONE_EMPTY_FILE.write_text(
            json.dumps(sorted(done_empty), ensure_ascii=False), encoding="utf-8"
        )
    except OSError as exc:
        sys.stderr.write(f"[warn] could not write {DONE_EMPTY_FILE.name}: {exc}\n")


# ---------------------------------------------------------------------------
# Import mode
# ---------------------------------------------------------------------------

def load_conversations(src: Path) -> List[Dict[str, Any]]:
    """Load the export and materialise conversations in file order.

    The index into this list is the canonical idx: reextract_conversations_<idx>
    lines up 1:1 with the old import_conversations_<idx> naming (same file order).
    """
    size_mb = src.stat().st_size / 1024 / 1024
    sys.stderr.write(f"[load] {src.name} ({size_mb:.1f} MB) — this can take a minute…\n")
    sys.stderr.flush()
    with open(src, "r", encoding="utf-8") as f:
        data = json.load(f)
    fmt = bi.detect_format(data)
    convs = list(bi._raw_items(data))
    sys.stderr.write(f"[load] format={fmt}  conversations={len(convs)}\n")
    sys.stderr.flush()
    return convs


def parse_all(convs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Turn each conversation into a lightweight parsed record (no LLM)."""
    parsed: List[Dict[str, Any]] = []
    for idx, conv in enumerate(convs):
        title, text, created_ts = bi._conv_to_text(conv)
        created_at, updated_at = conv_timestamps(conv, created_ts)
        chunks = _chunk_text(text) if text.strip() else []
        parsed.append(
            {
                "idx": idx,
                "title": title,
                "text": text,
                "chars": len(text),
                "chunks": len(chunks),
                "created_at": created_at,
                "updated_at": updated_at,
                "empty": not text.strip(),
            }
        )
    return parsed


def dry_run_report(parsed: List[Dict[str, Any]]) -> None:
    total = len(parsed)
    empty = sum(1 for p in parsed if p["empty"])
    nonempty = total - empty
    total_chars = sum(p["chars"] for p in parsed)
    calls = sum(p["chunks"] for p in parsed)
    big = [p for p in parsed if p["chars"] > SESSION_HARD_CHUNK_CHARS]
    in_tokens = total_chars * 0.49 + calls * 600
    est_mem = sum(min(8, max(1, round(p["chars"] / 6000))) for p in parsed if not p["empty"])
    out_tokens = est_mem * 130
    in_cost = in_tokens * IN_PRICE_PER_TOK
    out_cost = out_tokens * OUT_PRICE_PER_TOK
    print("=" * 74)
    print("DRY-RUN — parse + count only, no LLM, no DB writes")
    print("=" * 74)
    print(f"conversations total         : {total}")
    print(f"  empty (skipped)           : {empty}")
    print(f"  non-empty (will extract)  : {nonempty}")
    print(f"total text chars            : {total_chars:,}")
    print(f"hard-chunk size             : {SESSION_HARD_CHUNK_CHARS:,} chars")
    print(f"conversations that split    : {len(big)}")
    print(f"estimated LLM calls         : {calls}")
    print(f"estimated input tokens      : {in_tokens:,.0f}")
    print(f"estimated new memories      : {est_mem:,}")
    print(f"estimated output tokens     : {out_tokens:,.0f}")
    print("pricing: $0.25/M in, $1.50/M out (from live usage)")
    print(f"estimated input cost        : ${in_cost:.2f}")
    print(f"estimated output cost       : ${out_cost:.2f}")
    print(f"ESTIMATED TOTAL COST        : ${in_cost + out_cost:.2f}")
    for w in (1, 4, 8):
        secs = calls * SEC_PER_CALL_EST / w
        print(f"  ETA @ {w} workers          : {secs / 60:.0f} min")
    print("=" * 74)


class Billing:
    """Thread-safe running totals for the progress/billing line."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.done = 0
        self.ok = 0
        self.ok_empty = 0
        self.skipped = 0
        self.failed = 0
        self.empty = 0
        self.memories = 0
        self.rejected = 0
        self.no_embedding = 0
        self.in_tok = 0
        self.out_tok = 0

    def cost(self) -> float:
        return self.in_tok * IN_PRICE_PER_TOK + self.out_tok * OUT_PRICE_PER_TOK


def process_conversation(
    conn: Optional[sqlite3.Connection],
    parsed: Dict[str, Any],
    conv_text: str,
    dry_run: bool,
) -> Dict[str, Any]:
    """Extract + (optionally) embed + write ONE conversation. Atomic: either all
    of its memories land or none do, so a resume can re-do it cleanly."""
    idx = parsed["idx"]
    session_id = f"{SESSION_PREFIX}{idx}"
    result: Dict[str, Any] = {
        "idx": idx, "status": "", "memories": 0, "rejected": 0,
        "no_embedding": 0, "in_tok": 0, "out_tok": 0, "error": "",
    }
    if parsed["empty"]:
        result["status"] = "empty"
        return result

    all_items: List[Dict[str, Any]] = []
    try:
        for chunk in _chunk_text(conv_text):
            raw, usage = call_llm_with_retry(_EXTRACT_TMPL.format(chunk=chunk))
            result["in_tok"] += usage["prompt_tokens"]
            result["out_tok"] += usage["completion_tokens"]
            clean, rejected = sanitize_items(bi._parse_json_list(raw))
            result["rejected"] += rejected
            all_items.extend(clean)
    except Exception as exc:
        result["status"] = "fail"
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    if dry_run:
        result["status"] = "ok"
        result["memories"] = len(all_items)
        return result

    # Zero memories from a non-empty conversation (LLM returned [] or everything
    # was sanitised away). Nothing to write, but it IS complete — flag ok_empty so
    # the caller records it in the done-empty ledger and never re-calls the LLM.
    if not all_items:
        result["status"] = "ok_empty"
        return result

    # DB phase — guarded so a per-conversation DB failure (busy_timeout exhausted,
    # disk full, etc.) becomes a recorded 'fail' instead of an exception that
    # escapes the worker, tears down pool.map, and loses the whole ledger.
    assert conn is not None
    try:
        # Belt-and-suspenders idempotency: skip if a prior run already wrote this idx.
        if session_done(conn, session_id):
            result["status"] = "skip"
            return result

        embeddings: List[bytes] = []
        for item in all_items:
            try:
                emb = ri.pack_embedding(ri.fetch_embedding(item["content"]))
            except Exception:
                emb = b""
            if not emb:
                result["no_embedding"] += 1
            embeddings.append(emb)

        inserted = write_memories(
            conn, session_id, all_items, embeddings,
            parsed["created_at"], parsed["updated_at"],
        )
        result["status"] = "ok"
        result["memories"] = inserted
    except Exception as exc:
        result["status"] = "fail"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["no_embedding"] = 0
    return result


def run_import(
    src: Path,
    db_path: Path,
    *,
    workers: int,
    limit: int,
    dry_run: bool,
    no_backup: bool,
    retry_failed: bool,
) -> None:
    ensure_env()
    if not dry_run and not OPENROUTER_API_KEY:
        raise SystemExit("[ERROR] OPENROUTER_API_KEY not set (put it in .env)")

    convs = load_conversations(src)
    parsed = parse_all(convs)

    if dry_run:
        dry_run_report(parsed)
        return

    conn = sqlite3.connect(str(db_path), check_same_thread=False, timeout=30)
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    verify_schema(conn)

    # Candidate idxs: all, or only the recorded failures with --retry-failed.
    failed = load_failed()
    done_empty = load_done_empty()
    if retry_failed:
        if not failed:
            print("[retry-failed] reextract_failed.json is empty — nothing to retry.")
            conn.close()
            return
        candidate_idxs = sorted(int(k) for k in failed if k.isdigit())
    else:
        candidate_idxs = list(range(len(parsed)))

    done = existing_sessions(conn)
    # Skip a conversation if it already has rows in the db (session_done), if it
    # was recorded as completed-but-empty last time (done_empty ledger — Fix #1),
    # or if it has no text body at all.
    todo = [
        i for i in candidate_idxs
        if f"{SESSION_PREFIX}{i}" not in done
        and i not in done_empty
        and not parsed[i]["empty"]
    ]
    pre_skipped = len(candidate_idxs) - len(todo)
    if limit > 0:
        todo = todo[:limit]

    print(
        f"[plan] {len(todo)} conversation(s) to process "
        f"(already-done/empty skipped: {pre_skipped})  "
        f"model={OPENROUTER_MODEL}  workers={workers}"
    )
    sys.stdout.flush()
    if not todo:
        conn.close()
        return

    if not no_backup:
        backup_db(db_path)

    bill = Billing()
    total = len(todo)
    t0 = time.monotonic()
    run_failures: Dict[str, str] = {}

    def flush_ledgers() -> None:
        """Persist both ledgers mid-run so a hard crash (OOM, power loss) loses at
        most one checkpoint of progress rather than the entire run's bookkeeping."""
        merged_failed = dict(failed)
        for idx in todo[:bill.done]:
            merged_failed.pop(str(idx), None)
        merged_failed.update(run_failures)
        save_failed(merged_failed)
        save_done_empty(done_empty)

    def worker(idx: int) -> Dict[str, Any]:
        return process_conversation(conn, parsed[idx], parsed[idx]["text"], False)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for res in pool.map(worker, todo):
            with bill.lock:
                bill.done += 1
                bill.in_tok += res["in_tok"]
                bill.out_tok += res["out_tok"]
                bill.rejected += res["rejected"]
                status = res["status"]
                if status == "ok":
                    bill.ok += 1
                    bill.memories += res["memories"]
                    bill.no_embedding += res["no_embedding"]
                elif status == "ok_empty":
                    bill.ok_empty += 1
                    done_empty.add(int(res["idx"]))
                elif status == "skip":
                    bill.skipped += 1
                elif status == "empty":
                    bill.empty += 1
                else:
                    bill.failed += 1
                    run_failures[str(res["idx"])] = res["error"]
                    sys.stderr.write(f"[FAIL idx={res['idx']}] {res['error']}\n")
                    sys.stderr.flush()
                done_n = bill.done
            if done_n % 50 == 0 or done_n == total:
                flush_ledgers()  # crash-resilient checkpoint (Fix #2)
                elapsed = time.monotonic() - t0
                rate = done_n / elapsed if elapsed > 0 else 0.0
                eta_min = (total - done_n) / rate / 60 if rate > 0 else 0.0
                print(
                    f"[{done_n}/{total}] ok={bill.ok} empty={bill.ok_empty} "
                    f"skip={bill.skipped} fail={bill.failed} mem={bill.memories} "
                    f"rej={bill.rejected} tok_in={bill.in_tok:,} tok_out={bill.out_tok:,} "
                    f"cost=${bill.cost():.2f} {rate:.2f}/s ETA {eta_min:.1f}min"
                )
                sys.stdout.flush()

    conn.close()

    # Final ledger update: drop this run's successes/empties from failed, keep new
    # fails; persist the done-empty set.
    for idx in todo:
        failed.pop(str(idx), None)
    failed.update(run_failures)
    save_failed(failed)
    save_done_empty(done_empty)

    elapsed = time.monotonic() - t0
    print(
        f"\nDone. processed={bill.done} ok={bill.ok} zero_memory={bill.ok_empty} "
        f"skipped={bill.skipped} failed={bill.failed} memories={bill.memories} "
        f"rejected(meta/dup)={bill.rejected} empty_embedding={bill.no_embedding}\n"
        f"tokens: in={bill.in_tok:,} out={bill.out_tok:,}  cost=${bill.cost():.2f}  "
        f"elapsed={elapsed / 60:.1f}min"
    )
    if bill.failed:
        print(
            f"{bill.failed} conversation(s) failed — recorded in {FAILED_FILE.name}. "
            f"Re-run with --retry-failed."
        )
    if bill.no_embedding:
        print(
            f"{bill.no_embedding} memory/ies got no embedding (Ollama hiccup) — "
            f"run `python reindex_embeddings.py` to backfill."
        )


# ---------------------------------------------------------------------------
# Retire mode
# ---------------------------------------------------------------------------

def _retire_counts(conn: sqlite3.Connection) -> Dict[str, int]:
    def c(where: str) -> int:
        return conn.execute(f"SELECT COUNT(*) FROM memories WHERE {where}").fetchone()[0]

    return {
        "seabed_total": c("tier='seabed'"),
        "seabed_import": c("tier='seabed' AND session_id LIKE 'import_conversations_%'"),
        "target": c(RETIRE_PREDICATE),
        "exempt_pearls_in_seabed": c(
            "tier='seabed' AND session_id LIKE 'import_conversations_%' "
            f"AND updated_at > '{RETIRE_CUTOFF}'"
        ),
        "exempt_promoted": c(
            "tier!='seabed' AND session_id LIKE 'import_conversations_%'"
        ),
        "already_retired": c(f"{RETIRE_PREDICATE} AND digested=1 AND resolved=1"),
    }


def run_retire(
    db_path: Path, *, dry_run: bool, rollback: bool, assume_yes: bool, no_backup: bool
) -> None:
    if rollback:
        run_retire_rollback(db_path, dry_run=dry_run, assume_yes=assume_yes, no_backup=no_backup)
        return

    ro = f"file:{db_path.as_posix()}?mode=ro"
    conn = sqlite3.connect(ro, uri=True, timeout=15) if dry_run else sqlite3.connect(str(db_path), timeout=30)
    if not dry_run:
        conn.execute("PRAGMA busy_timeout=30000")
    verify_schema(conn)
    counts = _retire_counts(conn)

    print("=" * 74)
    print("RETIRE — soft-retire the OLD gemma seabed rows")
    print("=" * 74)
    print(f"predicate: {RETIRE_PREDICATE}")
    print(f"seabed rows total                    : {counts['seabed_total']}")
    print(f"  of which import_conversations_%     : {counts['seabed_import']}")
    print(f"WOULD RETIRE (digested=1,resolved=1) : {counts['target']}")
    print(f"exempt: pearls edited after window   : {counts['exempt_pearls_in_seabed']}")
    print(f"exempt: promoted out of seabed       : {counts['exempt_promoted']}")
    print(
        f"  check: {counts['target']} + {counts['exempt_pearls_in_seabed']} = "
        f"{counts['target'] + counts['exempt_pearls_in_seabed']} "
        f"(should equal seabed_import={counts['seabed_import']})"
    )
    print(f"already retired (idempotent re-run)  : {counts['already_retired']}")
    print(
        "\nnote: retired rows are hidden immediately (out of search / vector index / "
        "list / random)."
    )
    print(
        "The store's prune daemon hard-deletes digested rows once "
        "COALESCE(NULLIF(last_active,''),updated_at,created_at) is > 90 days old. "
        "seabed last_active currently runs up to *today* (breath keeps touching them), "
        "so real deletion happens ~90 days AFTER retire — once digested=1 stops breath "
        "from touching them. That is the intended delayed auto-cleanup."
    )
    print("=" * 74)

    if dry_run:
        conn.close()
        return

    to_change = counts["target"] - counts["already_retired"]
    if to_change <= 0:
        print("Nothing to change — all target rows already retired.")
        conn.close()
        return

    # Refuse to clobber an existing snapshot — that would strand the earlier
    # batch's exact rollback. Rollback (or move the file) first.
    if RETIRE_SNAPSHOT.exists() and RETIRE_SNAPSHOT.stat().st_size > 0:
        print(
            f"\n[ERROR] a retire snapshot already exists: {RETIRE_SNAPSHOT.name}. "
            f"Refusing to overwrite it. Run `--retire --rollback` first (it consumes "
            f"the snapshot), or move/remove the file, then retire again."
        )
        conn.close()
        return

    if not assume_yes:
        print(
            f"\nAbout to set digested=1, resolved=1 on {to_change} row(s) in "
            f"{db_path}."
        )
        reply = input("Type 'yes' to proceed: ").strip().lower()
        if reply != "yes":
            print("Aborted — no changes made.")
            conn.close()
            return

    if not no_backup:
        backup_db(db_path)

    # Snapshot exact prior state of every target row so --rollback is precise
    # (a plain predicate-reverse would wrongly clear digested on rows that were
    # already digested before retire).
    rows = conn.execute(
        f"SELECT id, digested, resolved FROM memories WHERE {RETIRE_PREDICATE}"
    ).fetchall()
    snapshot = {r[0]: {"digested": int(r[1] or 0), "resolved": int(r[2] or 0)} for r in rows}
    RETIRE_SNAPSHOT.write_text(
        json.dumps({"cutoff": RETIRE_CUTOFF, "rows": snapshot}, ensure_ascii=False),
        encoding="utf-8",
    )
    sys.stderr.write(f"[retire] snapshot of {len(snapshot)} rows -> {RETIRE_SNAPSHOT.name}\n")

    # Do NOT touch updated_at: (a) prune keys off last_active, so it's useless
    # here; (b) the snapshot doesn't store updated_at, so a bump would make
    # rollback unable to restore the original; (c) bumping it would push rows
    # out of the retire predicate (updated_at > cutoff), so a rollback→re-retire
    # cycle could never re-match them. Idempotency is the predicate guard alone.
    with conn:
        n = conn.execute(
            f"UPDATE memories SET digested=1, resolved=1 "
            f"WHERE {RETIRE_PREDICATE} AND NOT (digested=1 AND resolved=1)"
        ).rowcount
    conn.close()
    print(f"Retired {n} row(s). Snapshot saved for exact rollback.")


def run_retire_rollback(
    db_path: Path, *, dry_run: bool, assume_yes: bool, no_backup: bool
) -> None:
    snap = None
    if RETIRE_SNAPSHOT.exists():
        try:
            snap = json.loads(RETIRE_SNAPSHOT.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            snap = None

    print("=" * 74)
    print("RETIRE --rollback — undo a prior soft-retire")
    print("=" * 74)
    if snap and isinstance(snap.get("rows"), dict):
        n = len(snap["rows"])
        print(f"exact snapshot found ({n} rows) — will restore each row's prior "
              f"digested/resolved values.")
        if dry_run:
            print("(dry-run — nothing written)")
            return
        if not assume_yes:
            reply = input(f"Type 'yes' to restore {n} rows: ").strip().lower()
            if reply != "yes":
                print("Aborted.")
                return
        conn = sqlite3.connect(str(db_path), timeout=30)
        conn.execute("PRAGMA busy_timeout=30000")
        if not no_backup:
            backup_db(db_path)
        # Restore only digested/resolved — updated_at was never touched by retire,
        # so leaving it alone preserves each row's original real timestamp.
        changed = 0
        with conn:
            for mem_id, prior in snap["rows"].items():
                cur = conn.execute(
                    "UPDATE memories SET digested=?, resolved=? WHERE id=?",
                    (int(prior["digested"]), int(prior["resolved"]), mem_id),
                )
                changed += cur.rowcount
        conn.close()
        # Consume the snapshot so a later retire can write a fresh one (the
        # overwrite guard would otherwise block a legitimate re-retire).
        try:
            RETIRE_SNAPSHOT.unlink()
        except OSError:
            pass
        print(f"Restored {changed} row(s) from snapshot; snapshot consumed.")
        return

    # No snapshot: fall back to the spec's predicate-reverse, with a loud warning.
    print(
        "WARNING: no snapshot file found. Falling back to predicate-reverse "
        "(digested=0, resolved=0 on the retire predicate). This also clears "
        "digested on rows that were ALREADY digested before retire (the ~33k "
        "previously-consolidated seabed fragments) — it is NOT a perfect inverse."
    )
    ro = f"file:{db_path.as_posix()}?mode=ro"
    conn = sqlite3.connect(ro, uri=True, timeout=15) if dry_run else sqlite3.connect(str(db_path), timeout=30)
    n = conn.execute(f"SELECT COUNT(*) FROM memories WHERE {RETIRE_PREDICATE}").fetchone()[0]
    print(f"predicate matches {n} rows.")
    if dry_run:
        conn.close()
        print("(dry-run — nothing written)")
        return
    if not assume_yes:
        reply = input(f"Type 'yes' to clear digested/resolved on {n} rows: ").strip().lower()
        if reply != "yes":
            print("Aborted.")
            conn.close()
            return
    if not no_backup:
        backup_db(db_path)
    with conn:
        changed = conn.execute(
            f"UPDATE memories SET digested=0, resolved=0 WHERE {RETIRE_PREDICATE}"
        ).rowcount
    conn.close()
    print(f"Reverted {changed} row(s) via predicate.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

    ap = argparse.ArgumentParser(
        description="Re-extract the historical export into the seabed tier (海床换血)."
    )
    ap.add_argument("--src", default=DEFAULT_SRC, help="conversations.json export path")
    ap.add_argument("--db", default=str(DEFAULT_DB), help="target sqlite db (default: ./memory.db)")
    ap.add_argument("--workers", type=int, default=4, help="concurrent LLM calls (default 4)")
    ap.add_argument("--limit", type=int, default=0, help="process at most N conversations (0=all)")
    ap.add_argument("--dry-run", action="store_true", help="parse + count + cost; no LLM, no writes")
    ap.add_argument("--no-backup", action="store_true", help="skip the pre-write db backup")
    ap.add_argument("--retry-failed", action="store_true",
                    help="process only the idxs recorded in reextract_failed.json")
    ap.add_argument("--retire", action="store_true",
                    help="RETIRE mode: soft-retire the old gemma seabed rows")
    ap.add_argument("--rollback", action="store_true",
                    help="with --retire: undo a prior retire")
    ap.add_argument("--yes", action="store_true",
                    help="with --retire: skip the interactive confirmation")
    args = ap.parse_args()

    db_path = Path(args.db).resolve()

    if args.retire:
        if not db_path.exists():
            raise SystemExit(f"[ERROR] db not found: {db_path}")
        run_retire(
            db_path, dry_run=args.dry_run, rollback=args.rollback,
            assume_yes=args.yes, no_backup=args.no_backup,
        )
        return

    src = Path(args.src)
    if not src.exists():
        raise SystemExit(f"[ERROR] source export not found: {src}")
    if not args.dry_run and not db_path.exists():
        raise SystemExit(f"[ERROR] db not found: {db_path} (create it via the MCP or migrate_tiers.py)")

    run_import(
        src, db_path,
        workers=args.workers, limit=args.limit, dry_run=args.dry_run,
        no_backup=args.no_backup, retry_failed=args.retry_failed,
    )


if __name__ == "__main__":
    main()
