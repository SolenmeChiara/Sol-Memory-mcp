#!/usr/bin/env python3
"""Batch import conversation exports into the memory SQLite database.

Usage:
    python batch_import.py "path/to/export.json" [--db memory.db] [--start 0] [--limit 10] [--dry-run]
    python batch_import.py export.zip --provider gemini --model gemini-3.5-flash-lite

--provider/--model default to IMPORT_PROVIDER/IMPORT_MODEL from .env (see README);
an explicit flag always wins.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import random
import re
import sqlite3
import struct
import sys
import threading
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Ollama configuration
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.environ.get("OLLAMA_MODEL", "gemma4:e4b")
OLLAMA_EMBED_MODEL: str = os.environ.get("OLLAMA_EMBED_MODEL", "qwen3-embedding:4b")
OLLAMA_TIMEOUT: float = float(os.environ.get("OLLAMA_TIMEOUT", "180"))
N_EMBED_WORKERS = 2

# ---------------------------------------------------------------------------
# Gemini configuration
# ---------------------------------------------------------------------------

GEMINI_DEFAULT_MODEL = "gemini-3.5-flash-lite"
GEMINI_TIMEOUT: float = float(os.environ.get("GEMINI_TIMEOUT", "180"))
# 8192 leaves enough headroom for thinking models (gemini-3.5-flash, gemini-3-*)
# to reason AND still emit a multi-memory JSON array. Non-thinking lite models
# stop well below this cap on their own.
GEMINI_MAX_OUTPUT_TOKENS = 20000

# ---------------------------------------------------------------------------
# OpenRouter configuration
# ---------------------------------------------------------------------------

OPENROUTER_BASE_URL_DEFAULT = "https://openrouter.ai/api/v1"
OPENROUTER_DEFAULT_MODEL = "google/gemini-3.5-flash-lite"
OPENROUTER_TIMEOUT: float = float(os.environ.get("OPENROUTER_TIMEOUT", "180"))

PROVIDERS = ("ollama", "gemini", "openrouter")

# Session-mode chunking. Gemini handles 1M-token context so we treat each
# conversation as a single chunk; only split if total chars exceed this.
SESSION_HARD_CHUNK_CHARS = 100_000


def _load_env_var(name: str) -> Optional[str]:
    """Look in process env first, then nearby .env files (script dir up 3 levels)."""
    val = os.environ.get(name)
    if val:
        return val.strip()
    for path in [
        SCRIPT_DIR / ".env",
        SCRIPT_DIR.parent / ".env",
        SCRIPT_DIR.parent.parent / ".env",
        SCRIPT_DIR.parent.parent.parent / ".env",
    ]:
        if not path.is_file():
            continue
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("set "):
                line = line[4:].lstrip()
            if "=" not in line:
                continue
            k, _, v = line.partition("=")
            if k.strip() == name:
                return v.strip().strip('"').strip("'")
    return None

# ---------------------------------------------------------------------------
# Ollama helpers (logic from memory_mcp.py)
# ---------------------------------------------------------------------------

def _call_ollama(prompt: str, model: str = OLLAMA_MODEL) -> str:
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "temperature": 0.3,
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    return json.loads(body)["choices"][0]["message"]["content"]


def _call_gemini(prompt: str, api_key: str, model: str = GEMINI_DEFAULT_MODEL) -> str:
    """Call Google AI Studio's Gemini generateContent endpoint.

    Returns the model's text response (single candidate). Raises RuntimeError
    on safety blocks / empty outputs so the caller can log + continue.
    """
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}"
        f":generateContent?key={api_key}"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": GEMINI_MAX_OUTPUT_TOKENS,
            "temperature": 0.3,
        },
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=GEMINI_TIMEOUT) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")[:500]
        raise RuntimeError(f"Gemini HTTP {e.code}: {raw}") from None

    parsed = json.loads(body)
    candidates = parsed.get("candidates") or []
    if not candidates:
        # Could be a top-level promptFeedback.blockReason
        feedback = parsed.get("promptFeedback") or {}
        raise RuntimeError(f"Gemini returned no candidates: {feedback}")
    cand = candidates[0]
    finish_reason = cand.get("finishReason")
    parts = (cand.get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
    if not text:
        # Thinking models can exhaust the output budget on internal reasoning
        # and emit zero content parts. Surface that distinctly so the caller
        # knows to raise maxOutputTokens or set thinkingConfig.thinkingBudget=0.
        if finish_reason == "MAX_TOKENS":
            usage = parsed.get("usageMetadata") or {}
            think_tok = usage.get("thoughtsTokenCount", "?")
            raise RuntimeError(
                f"Gemini MAX_TOKENS with empty content — thinking ate the budget "
                f"(thoughtsTokens={think_tok}, max={GEMINI_MAX_OUTPUT_TOKENS}). "
                f"Bump GEMINI_MAX_OUTPUT_TOKENS or disable thinking."
            )
        raise RuntimeError(f"Gemini returned empty text (finishReason={finish_reason})")
    return text


def _call_openrouter(prompt: str, api_key: str,
                     model: str = OPENROUTER_DEFAULT_MODEL) -> str:
    """Call OpenRouter's OpenAI-compatible chat/completions endpoint.

    Deliberately a standalone twin of memory_mcp._call_openrouter — batch_import
    is imported *by* memory_mcp, so it must not import back.
    """
    base = (_load_env_var("OPENROUTER_BASE_URL") or OPENROUTER_BASE_URL_DEFAULT).rstrip("/")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "temperature": 0.3,
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=OPENROUTER_TIMEOUT) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")[:500]
        raise RuntimeError(f"OpenRouter HTTP {e.code}: {raw}") from None
    parsed = json.loads(body)
    choices = parsed.get("choices") or []
    if not choices:
        raise RuntimeError(f"OpenRouter returned no choices: {str(parsed)[:300]}")
    text = (choices[0].get("message") or {}).get("content") or ""
    if not text.strip():
        raise RuntimeError("OpenRouter returned empty content")
    return text


def _call_llm(provider: str, prompt: str, model: str, api_key: Optional[str]) -> str:
    """Provider dispatcher: route to Gemini, OpenRouter or Ollama."""
    if provider == "gemini":
        if not api_key:
            raise RuntimeError("GOOGLE_AI_STUDIO_KEY not set (env or .env)")
        return _call_gemini(prompt, api_key, model)
    if provider == "openrouter":
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set (env or .env)")
        return _call_openrouter(prompt, api_key, model)
    return _call_ollama(prompt, model)


def _call_ollama_embedding(text: str) -> List[float]:
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/embed"
    payload = {"model": OLLAMA_EMBED_MODEL, "input": text[:2000]}
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as resp:
            body = resp.read().decode("utf-8")
        parsed = json.loads(body)
        embeddings = parsed.get("embeddings", [])
        return embeddings[0] if embeddings else []
    except Exception:
        return []


def _pack_embedding(values: List[float]) -> bytes:
    values = list(values)
    return struct.pack(f"<{len(values)}f", *values) if values else b""


# ---------------------------------------------------------------------------
# Prompt template and text helpers (from memory_mcp.py)
# ---------------------------------------------------------------------------

_SESSION_EXTRACT_TMPL = """你是一位温和细致的记忆管理者。你对经手的每一段记忆都带着安心感和珍视，像是在整理一本私人日记。

你的性格决定了你的工作方式：
- 你实事求是，从不在笔记中添加不客观的新内容。对话里说了什么就记什么，没说的绝不臆造。
- 你尊重完整性。一件事的来龙去脉不会被你拆散，相关的细节你会自然地归拢在一起。
- 你有分寸感。闲聊、调试代码、重复的问答，你会安静地略过，只留下真正有温度或有份量的东西。
- 你对情感很敏感。记录时你会自然地感受到这段对话的情绪色彩，并如实标注。
- 你惜字但不吝啬。每条记忆 50-300 字，足够还原现场，又不至于啰嗦。

现在请整理以下对话，提取出值得长期珍藏的记忆。一段对话通常整理出 1-8 条记忆，视内容丰富程度而定。

输出格式（纯 JSON 数组，不要加任何额外说明）：
[
  {{
    "key": "简短标题（15字以内）",
    "content": "完整的事件或主题描述",
    "category": "preference|promise|event|anniversary|emotion|habit|boundary|other",
    "importance": 0.5,
    "valence": 0.7,
    "arousal": 0.4
  }}
]

importance: 0.0-1.0，日常琐事 0.2-0.4，重要事件 0.6-0.8，核心记忆 0.9-1.0
valence: 0.0（极度消极）到 1.0（极度积极），0.5 为中性
arousal: 0.0（非常平静）到 1.0（非常激动），0.3 为日常状态

对话记录：
{chunk}"""


_IMPORT_EXTRACT_TMPL = (
    "从以下对话片段中提取 1-3 条值得记忆的条目。\n"
    "\n"
    "**重要：鼓励连续叙事，反对碎片化**\n"
    "- 把围绕同一主题/事件的细节**合并到一条 content 里**，保留时间顺序、因果关系、情感变化。\n"
    "- 宁可少而完整，也不要拆成孤立的事实碎片。\n"
    "- 反面示例（禁止）：拆成「去了公交站」「坐了 1ce 路」「到了 UTM」三条。\n"
    "- 正面示例（推荐）：合并成「今天从家出门坐 1ce 路去 UTM，路上讨论了 X」一条完整叙事。\n"
    "- content 可以较长（数百字无妨），优先完整性而非简洁。\n"
    "- 只有真正**互不相关**的主题才应拆成多条（例如同一段对话里既聊了约会又聊了工作）。\n"
    "\n"
    "每条记忆包含：\n"
    "- key：简短标题（≤20 字）\n"
    "- content：完整叙事（可跨越多轮对话）\n"
    "- category：preference / promise / event / anniversary / emotion / habit / boundary / other 之一\n"
    "- importance：0.0~1.0\n"
    "\n"
    "如果本片段没有值得记忆的内容，返回空数组 []。\n"
    "只输出纯 JSON 数组，不加任何说明、不加 markdown 代码块：\n"
    '[{{"key":"...","content":"...","category":"...","importance":0.7}}]\n'
    "\n"
    "对话片段：\n{chunk}"
)


def _chunk_conversation(text: str, window: int = 8000) -> List[str]:
    return [text[i : i + window] for i in range(0, len(text), window)]


def _parse_json_list(raw: str) -> List[Any]:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        parsed = json.loads(cleaned)
        return [parsed] if not isinstance(parsed, list) else parsed
    except json.JSONDecodeError:
        m = re.search(r"\[.*?\]", cleaned, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group())
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass
        return []


# ---------------------------------------------------------------------------
# Format detection and conversation iteration
# ---------------------------------------------------------------------------

def _is_conversation_obj(item: dict) -> bool:
    return any(k in item for k in ("chat_messages", "mapping", "messages"))


def detect_format(data: Any) -> str:
    if isinstance(data, list):
        first = next((x for x in data if isinstance(x, dict)), None)
        if first is None:
            return "empty"
        if "chat_messages" in first:
            return "claude_official"
        if "mapping" in first:
            return "chatgpt_list"
        if _is_conversation_obj(first):
            return "plugin_list"
        return "messages_list"  # raw list of turns — treat as one conversation
    if isinstance(data, dict):
        if "mapping" in data:
            return "chatgpt_single"
        if "messages" in data:
            return "plugin_single"
        if "conversations" in data:
            return "wrapped"
    return "unknown"


def _raw_items(data: Any) -> Iterator[Dict[str, Any]]:
    """Yield one conversation dict at a time without materialising the whole list."""
    if isinstance(data, list):
        first = next((x for x in data if isinstance(x, dict)), None)
        if first is not None and _is_conversation_obj(first):
            for item in data:
                if isinstance(item, dict):
                    yield item
        elif first is not None:
            # Whole list is one conversation's messages
            yield {"messages": [m for m in data if isinstance(m, dict)]}
    elif isinstance(data, dict):
        if "conversations" in data:
            for conv in data["conversations"]:
                if isinstance(conv, dict):
                    yield conv
        else:
            yield data


def _quick_count(data: Any) -> int:
    if isinstance(data, list):
        first = next((x for x in data if isinstance(x, dict)), None)
        if first is None:
            return 0
        if _is_conversation_obj(first):
            return sum(1 for x in data if isinstance(x, dict))
        return 1
    if isinstance(data, dict):
        if "conversations" in data:
            return len(data.get("conversations", []))
        return 1
    return 0


def _normalize_iso_ts(raw: Any) -> Optional[str]:
    """Coerce a Claude/ChatGPT export timestamp into the project's ISO+UTC form
    (e.g. `2026-05-07T20:10:10.126945+00:00`). Returns None if it can't parse.
    Accepts `...Z`, `...+00:00`, naive ISO, or a unix epoch number."""
    if raw is None or raw == "":
        return None
    if isinstance(raw, (int, float)):
        try:
            return datetime.fromtimestamp(float(raw), tz=timezone.utc).isoformat()
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(raw, str):
        s = raw.strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(s)
        except ValueError:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    return None


def _conv_to_text(conv: Dict[str, Any]) -> Tuple[str, str, Optional[str]]:
    """Return (title, plain_text, original_ts_iso) for one conversation dict.

    The third item is the conversation's own created_at (when available)
    normalised to ISO-8601 with UTC offset — used as the memory's
    `created_at` so imported history retains its original timeline.
    """
    parts: List[str] = []
    title = ""
    original_ts: Optional[str] = None

    if "chat_messages" in conv:
        # Claude official bulk export
        title = str(conv.get("name", "") or "")
        original_ts = (
            _normalize_iso_ts(conv.get("created_at"))
            or (_normalize_iso_ts(conv.get("chat_messages", [{}])[0].get("created_at"))
                if conv.get("chat_messages") else None)
        )
        if title:
            parts.append(f"--- 对话：{title} ---")
        for msg in conv.get("chat_messages", []):
            role = msg.get("sender", "")
            text = msg.get("text", "")
            if not text:
                for block in msg.get("content", []):
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "")
                        break
            if text:
                parts.append(f"{role}: {text}")

    elif "messages" in conv:
        # Plugin export (say field) or generic single conversation (content field)
        meta = conv.get("metadata", {})
        title = str(
            (meta.get("title", "") if isinstance(meta, dict) else "")
            or conv.get("title", "")
            or ""
        )
        original_ts = (
            _normalize_iso_ts(conv.get("created_at"))
            or _normalize_iso_ts(conv.get("create_time"))
            or (isinstance(meta, dict) and _normalize_iso_ts(meta.get("created_at")))
            or None
        )
        if title:
            parts.append(f"--- 对话：{title} ---")
        for m in conv.get("messages", []):
            role = m.get("role", "")
            mc = m.get("say") or m.get("content", "")
            if isinstance(mc, str) and mc:
                parts.append(f"{role}: {mc}")
            elif isinstance(mc, list):
                for p in mc:
                    if isinstance(p, dict) and p.get("type") == "text":
                        t = p.get("text", "")
                        if t:
                            parts.append(f"{role}: {t}")

    elif "mapping" in conv:
        # ChatGPT conversation
        title = str(conv.get("title", "") or "")
        original_ts = (
            _normalize_iso_ts(conv.get("create_time"))
            or _normalize_iso_ts(conv.get("created_at"))
        )
        if title:
            parts.append(f"--- 对话：{title} ---")
        mapping: Dict[str, Any] = conv.get("mapping", {})
        root_id: Optional[str] = None
        for nid, node in mapping.items():
            if node.get("parent") is None:
                root_id = nid
                break
        current_id: Optional[str] = root_id
        visited: set = set()
        while current_id and current_id not in visited:
            visited.add(current_id)
            node = mapping.get(current_id, {})
            msg = node.get("message") or {}
            author = msg.get("author", {}) or {}
            role = author.get("role", "")
            if role and role not in ("system", "tool"):
                content = msg.get("content", {})
                text = ""
                if isinstance(content, dict) and content.get("content_type") == "text":
                    text = "\n".join(
                        p if isinstance(p, str) else ""
                        for p in content.get("parts", [])
                    ).strip()
                elif isinstance(content, str):
                    text = content.strip()
                if text:
                    parts.append(f"{role}: {text}")
            children = node.get("children", [])
            current_id = children[0] if children else None

    return title, "\n".join(parts), original_ts


# ---------------------------------------------------------------------------
# Database (schema identical to memory_mcp.py MemoryStore)
# ---------------------------------------------------------------------------

_DB_LOCK = threading.RLock()

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS memories (
  id TEXT PRIMARY KEY,
  key TEXT NOT NULL,
  content TEXT NOT NULL,
  memory_kind TEXT NOT NULL DEFAULT 'long_term',
  category TEXT NOT NULL,
  importance REAL DEFAULT 0.5,
  session_id TEXT DEFAULT '',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  embedding BLOB DEFAULT X''
);
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
  key, content, content='memories', content_rowid='rowid'
);
CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
  INSERT INTO memories_fts(rowid, key, content) VALUES (new.rowid, new.key, new.content);
END;
CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
  INSERT INTO memories_fts(memories_fts, rowid, key, content)
  VALUES ('delete', old.rowid, old.key, old.content);
END;
CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
  INSERT INTO memories_fts(memories_fts, rowid, key, content)
  VALUES ('delete', old.rowid, old.key, old.content);
  INSERT INTO memories_fts(rowid, key, content) VALUES (new.rowid, new.key, new.content);
END;
"""

_NEW_COLS = [
    ("valence",          "REAL DEFAULT 0.5"),
    ("arousal",          "REAL DEFAULT 0.3"),
    ("pinned",           "INTEGER DEFAULT 0"),
    ("resolved",         "INTEGER DEFAULT 0"),
    ("digested",         "INTEGER DEFAULT 0"),
    ("activation_count", "REAL DEFAULT 1.0"),
    ("last_active",      "TEXT DEFAULT ''"),
]


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.executescript(_SCHEMA_SQL)
    columns = {r[1] for r in conn.execute("PRAGMA table_info(memories)").fetchall()}
    for col_name, col_def in _NEW_COLS:
        if col_name not in columns:
            conn.execute(f"ALTER TABLE memories ADD COLUMN {col_name} {col_def}")
    conn.commit()


def insert_memory(
    conn: sqlite3.Connection,
    *,
    memory_id: str,
    key: str,
    content: str,
    category: str = "other",
    importance: float = 0.5,
    session_id: str = "",
    created_at_override: Optional[str] = None,
    valence: float = 0.5,
    arousal: float = 0.3,
) -> None:
    """Insert one memory. If `created_at_override` is provided (ISO UTC string),
    use it for both `created_at` and `updated_at`; otherwise default to now."""
    now = datetime.now(timezone.utc).isoformat()
    created_at = created_at_override or now
    updated_at = created_at_override or now
    val = max(0.0, min(1.0, float(valence)))
    aro = max(0.0, min(1.0, float(arousal)))
    with _DB_LOCK:
        conn.execute(
            """
            INSERT INTO memories(id, key, content, memory_kind, category, importance, session_id,
                                 created_at, updated_at, embedding,
                                 valence, arousal, pinned, resolved, digested, activation_count, last_active)
            VALUES (?, ?, ?, 'long_term', ?, ?, ?, ?, ?, X'', ?, ?, 0, 0, 0, 1.0, ?)
            ON CONFLICT(id) DO NOTHING
            """,
            (memory_id, key, content, category, importance, session_id,
             created_at, updated_at, val, aro, now),
        )
        conn.commit()


def update_embedding(conn: sqlite3.Connection, memory_id: str, embedding: List[float]) -> None:
    if not embedding:
        return
    with _DB_LOCK:
        conn.execute(
            "UPDATE memories SET embedding=? WHERE id=?",
            (_pack_embedding(embedding), memory_id),
        )
        conn.commit()


# ---------------------------------------------------------------------------
# Background embedding worker pool
# ---------------------------------------------------------------------------

def _start_embed_workers(conn: sqlite3.Connection) -> "queue.Queue":
    q: queue.Queue = queue.Queue()

    def worker() -> None:
        while True:
            item = q.get()
            if item is None:
                q.task_done()
                return
            mid, content = item
            try:
                emb = _call_ollama_embedding(content)
                if emb:
                    update_embedding(conn, mid, emb)
            except Exception:
                pass
            q.task_done()

    for _ in range(N_EMBED_WORKERS):
        threading.Thread(target=worker, daemon=True).start()
    return q


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Windows consoles default to GBK/CP936 and crash on emoji in titles.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

    # Load .env (same loader consolidate uses) BEFORE anything reads the model
    # constants, then refresh the module globals — the OLLAMA_* constants at the
    # top of this file were evaluated at import time, so without this refresh a
    # standalone `python batch_import.py` run would ignore .env entirely and fall
    # back to the code defaults (e.g. embedding model). Priority stays
    # process-env > .env > code default because _load_dotenv uses setdefault and
    # os.environ.get() below reads the already-populated environment.
    global OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_EMBED_MODEL, OLLAMA_TIMEOUT, GEMINI_TIMEOUT
    global OPENROUTER_TIMEOUT
    try:
        from consolidate_sessions import _load_dotenv
        _load_dotenv(SCRIPT_DIR)
    except Exception as exc:  # never let a .env hiccup block the import
        sys.stderr.write(f"[batch_import] .env load skipped: {exc}\n")
    OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", OLLAMA_BASE_URL)
    OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", OLLAMA_MODEL)
    OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", OLLAMA_EMBED_MODEL)
    OLLAMA_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", str(OLLAMA_TIMEOUT)))
    GEMINI_TIMEOUT = float(os.environ.get("GEMINI_TIMEOUT", str(GEMINI_TIMEOUT)))
    OPENROUTER_TIMEOUT = float(os.environ.get("OPENROUTER_TIMEOUT", str(OPENROUTER_TIMEOUT)))

    parser = argparse.ArgumentParser(
        description="Batch import conversation exports into the memory SQLite database"
    )
    parser.add_argument("file", help="Path to the JSON or .zip export file")
    parser.add_argument("--db", default="memory.db", help="SQLite database path (default: ./memory.db)")
    parser.add_argument(
        "--start", type=int, default=0,
        help="Skip first N conversations — use for checkpoint resume (default: 0)",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Process at most N conversations, 0 = all (default: 0)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Call LLM and show what would be inserted, but skip all DB writes",
    )
    parser.add_argument(
        "--provider", choices=list(PROVIDERS), default=None,
        help="LLM backend for extraction. Default: IMPORT_PROVIDER from env/.env, "
             "else ollama",
    )
    parser.add_argument(
        "--model", default=None,
        help="Model name. Default: IMPORT_MODEL from env/.env, else the provider's "
             f"own default ({OLLAMA_MODEL} / {GEMINI_DEFAULT_MODEL} / "
             f"OPENROUTER_MODEL or {OPENROUTER_DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--session-mode", action=argparse.BooleanOptionalAction, default=True,
        help="Process each conversation as one chunk (only split if > 100k chars). "
             "Default ON. Use --no-session-mode for legacy 8k chunk-by-chunk extraction.",
    )
    args = parser.parse_args()

    # ---- Provider / model resolution ----
    # Priority for both: explicit CLI flag > env|.env (IMPORT_PROVIDER /
    # IMPORT_MODEL) > per-provider code default. argparse defaults are None so
    # "was it passed?" is unambiguous.
    if args.provider is None:
        env_provider = (_load_env_var("IMPORT_PROVIDER") or "").strip().lower()
        if env_provider and env_provider not in PROVIDERS:
            print(
                f"Error: invalid IMPORT_PROVIDER={env_provider!r} in env/.env "
                f"(expected one of {'|'.join(PROVIDERS)})",
                file=sys.stderr,
            )
            sys.exit(1)
        args.provider = env_provider or "ollama"

    if args.model is None:
        env_model = (_load_env_var("IMPORT_MODEL") or "").strip()
        if env_model:
            args.model = env_model
        elif args.provider == "gemini":
            args.model = GEMINI_DEFAULT_MODEL
        elif args.provider == "openrouter":
            args.model = (
                _load_env_var("OPENROUTER_MODEL") or ""
            ).strip() or OPENROUTER_DEFAULT_MODEL
        else:
            args.model = OLLAMA_MODEL

    # Cloud auth (ollama needs none)
    api_key: Optional[str] = None
    if args.provider == "gemini":
        api_key = _load_env_var("GOOGLE_AI_STUDIO_KEY")
        if not api_key:
            print("Error: GOOGLE_AI_STUDIO_KEY not in env or .env files", file=sys.stderr)
            sys.exit(1)
    elif args.provider == "openrouter":
        api_key = _load_env_var("OPENROUTER_API_KEY")
        if not api_key:
            print("Error: OPENROUTER_API_KEY not in env or .env files", file=sys.stderr)
            sys.exit(1)

    extract_tmpl = _SESSION_EXTRACT_TMPL if args.session_mode else _IMPORT_EXTRACT_TMPL
    print(f"Provider: {args.provider}  |  Model: {args.model}  |  "
          f"Mode: {'session' if args.session_mode else 'chunk-8k'}")

    json_path = Path(args.file)
    if not json_path.exists():
        print(f"Error: file not found: {json_path}", file=sys.stderr)
        sys.exit(1)

    size_mb = json_path.stat().st_size / 1024 / 1024
    print(f"Loading {json_path.name} ({size_mb:.1f} MB)…")
    sys.stdout.flush()
    if json_path.suffix.lower() == ".zip":
        # Claude.ai bulk exports ship as a zip — read conversations.json directly
        # without unpacking to disk.
        with zipfile.ZipFile(json_path) as zf:
            inner_name = next(
                (n for n in zf.namelist() if n.endswith("conversations.json")),
                None,
            )
            if inner_name is None:
                print(f"Error: no conversations.json inside {json_path}", file=sys.stderr)
                sys.exit(1)
            with zf.open(inner_name) as f:
                data = json.load(f)
    else:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

    fmt = detect_format(data)
    total = _quick_count(data)
    print(f"Format: {fmt}  |  Total conversations: {total}")

    end_idx = total if args.limit == 0 else min(args.start + args.limit, total)
    to_process = max(0, end_idx - args.start)
    print(
        f"Processing [{args.start}, {end_idx}) = {to_process} conversation(s)"
        + (" [dry-run — LLM enabled, DB writes skipped]" if args.dry_run else "")
    )
    sys.stdout.flush()

    conn: Optional[sqlite3.Connection] = None
    embed_q: Optional[queue.Queue] = None

    if not args.dry_run:
        db_path = Path(args.db).resolve()
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        init_db(conn)
        embed_q = _start_embed_workers(conn)
        print(f"Database: {db_path}")

    total_memories = 0
    total_chunks = 0
    total_embed_queued = 0
    displayed = 0

    for raw_idx, item in enumerate(_raw_items(data)):
        if raw_idx < args.start:
            continue
        if args.limit > 0 and displayed >= args.limit:
            break

        displayed += 1
        progress = f"[{displayed}/{to_process}]"

        title, text, original_ts = _conv_to_text(item)
        label = title[:60] if title else f"(untitled #{raw_idx})"
        ts_note = f"  @ {original_ts[:19]}" if original_ts else "  @ (no original ts — using now)"

        if not text.strip():
            print(f"{progress} 对话 \"{label}\"{ts_note} → skipped (empty)")
            sys.stdout.flush()
            continue

        # Session mode: keep the conversation whole; only split if it's huge.
        if args.session_mode:
            if len(text) > SESSION_HARD_CHUNK_CHARS:
                chunks = [text[i : i + SESSION_HARD_CHUNK_CHARS]
                          for i in range(0, len(text), SESSION_HARD_CHUNK_CHARS)]
            else:
                chunks = [text]
        else:
            chunks = _chunk_conversation(text)
        total_chunks += len(chunks)

        session_id = f"batch_{json_path.stem}_{raw_idx}"
        conv_memories = 0
        errors = 0

        for chunk in chunks:
            try:
                raw_resp = _call_llm(args.provider, extract_tmpl.format(chunk=chunk),
                                     args.model, api_key)
                items = _parse_json_list(raw_resp)
            except Exception as exc:
                errors += 1
                print(f"  [warn] LLM error: {exc}", file=sys.stderr)
                continue

            # Session-mode allows up to 8 memories/chunk; chunk-mode keeps the old 5 cap.
            item_cap = 8 if args.session_mode else 5
            for it in items[:item_cap]:
                if not isinstance(it, dict):
                    continue
                item_content = str(it.get("content", "")).strip()
                if not item_content:
                    continue
                # Anchor the memory id to the conversation's original timestamp
                # when available so re-imports stay chronologically sortable.
                id_anchor: datetime
                if original_ts:
                    try:
                        id_anchor = datetime.fromisoformat(original_ts)
                    except ValueError:
                        id_anchor = datetime.now(timezone.utc)
                else:
                    id_anchor = datetime.now(timezone.utc)
                mid = (
                    f"mem_{id_anchor.strftime('%Y%m%d%H%M%S%f')}"
                    f"_{random.randint(1000, 9999)}"
                )
                m_key = str(it.get("key", ""))[:60] or "untitled"
                m_cat = str(it.get("category", "other"))
                m_imp = max(0.0, min(1.0, float(it.get("importance", 0.5))))
                m_val = max(0.0, min(1.0, float(it.get("valence", 0.5))))
                m_aro = max(0.0, min(1.0, float(it.get("arousal", 0.3))))

                if args.dry_run:
                    snippet = item_content[:80].replace("\n", " ")
                    print(f"    [dry] {m_cat:10s} imp={m_imp:.2f} val={m_val:.2f} "
                          f"aro={m_aro:.2f} | {m_key} | {snippet}")
                else:
                    assert conn is not None and embed_q is not None
                    insert_memory(
                        conn,
                        memory_id=mid,
                        key=m_key,
                        content=item_content,
                        category=m_cat,
                        importance=m_imp,
                        session_id=session_id,
                        created_at_override=original_ts,
                        valence=m_val,
                        arousal=m_aro,
                    )
                    embed_q.put((mid, item_content))
                    total_embed_queued += 1
                conv_memories += 1

        total_memories += conv_memories
        err_note = f"  [{errors} error(s)]" if errors else ""
        tag = "[dry-run] 提取" if args.dry_run else "创建"
        print(f"{progress} 对话 \"{label}\"{ts_note} → {len(chunks)} chunks → {tag} {conv_memories} 条记忆{err_note}")
        sys.stdout.flush()

    # Drain embedding queue before exit
    if embed_q is not None:
        if total_embed_queued > 0:
            print(f"\n等待 {total_embed_queued} 条 embedding 后台完成…", flush=True)
        embed_q.join()

    if conn is not None:
        conn.close()

    if args.dry_run:
        print(
            f"\nDry-run complete: {displayed} conversation(s), "
            f"{total_chunks} chunk(s) → {total_memories} memories extracted (NOT written to DB)."
        )
    else:
        print(
            f"\nDone: {total_memories} memories created from {total_chunks} chunk(s) "
            f"across {displayed} conversation(s)."
        )


if __name__ == "__main__":
    main()
