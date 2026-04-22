#!/usr/bin/env python3
"""Batch import conversation exports into the memory SQLite database.

Usage:
    python batch_import.py "path/to/export.json" [--db memory.db] [--start 0] [--limit 10] [--dry-run]
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
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Ollama configuration
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.environ.get("OLLAMA_MODEL", "gemma4:e4b")
OLLAMA_EMBED_MODEL: str = os.environ.get("OLLAMA_EMBED_MODEL", "bge-m3")
OLLAMA_TIMEOUT: float = float(os.environ.get("OLLAMA_TIMEOUT", "180"))
N_EMBED_WORKERS = 2

# ---------------------------------------------------------------------------
# Ollama helpers (logic from memory_mcp.py)
# ---------------------------------------------------------------------------

def _call_ollama(prompt: str) -> str:
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": OLLAMA_MODEL,
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


def _conv_to_text(conv: Dict[str, Any]) -> Tuple[str, str]:
    """Return (title, plain_text) for one conversation dict."""
    parts: List[str] = []
    title = ""

    if "chat_messages" in conv:
        # Claude official bulk export
        title = str(conv.get("name", "") or "")
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

    return title, "\n".join(parts)


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
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _DB_LOCK:
        conn.execute(
            """
            INSERT INTO memories(id, key, content, memory_kind, category, importance, session_id,
                                 created_at, updated_at, embedding,
                                 valence, arousal, pinned, resolved, digested, activation_count, last_active)
            VALUES (?, ?, ?, 'long_term', ?, ?, ?, ?, ?, X'', 0.5, 0.3, 0, 0, 0, 1.0, ?)
            ON CONFLICT(id) DO NOTHING
            """,
            (memory_id, key, content, category, importance, session_id, now, now, now),
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

    parser = argparse.ArgumentParser(
        description="Batch import conversation exports into the memory SQLite database"
    )
    parser.add_argument("file", help="Path to the JSON export file")
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
        help="Parse and chunk only — no LLM calls, no DB writes",
    )
    args = parser.parse_args()

    json_path = Path(args.file)
    if not json_path.exists():
        print(f"Error: file not found: {json_path}", file=sys.stderr)
        sys.exit(1)

    size_mb = json_path.stat().st_size / 1024 / 1024
    print(f"Loading {json_path.name} ({size_mb:.1f} MB)…")
    sys.stdout.flush()
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    fmt = detect_format(data)
    total = _quick_count(data)
    print(f"Format: {fmt}  |  Total conversations: {total}")

    end_idx = total if args.limit == 0 else min(args.start + args.limit, total)
    to_process = max(0, end_idx - args.start)
    print(
        f"Processing [{args.start}, {end_idx}) = {to_process} conversation(s)"
        + (" [dry-run — no LLM, no DB]" if args.dry_run else "")
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

        title, text = _conv_to_text(item)
        label = title[:60] if title else f"(untitled #{raw_idx})"

        if not text.strip():
            print(f"{progress} 对话 \"{label}\" → skipped (empty)")
            sys.stdout.flush()
            continue

        chunks = _chunk_conversation(text)
        total_chunks += len(chunks)

        if args.dry_run:
            print(f"{progress} 对话 \"{label}\" → {len(chunks)} chunks → [dry-run]")
            sys.stdout.flush()
            continue

        assert conn is not None and embed_q is not None
        session_id = f"batch_{json_path.stem}_{raw_idx}"
        conv_memories = 0
        errors = 0

        for chunk in chunks:
            try:
                raw_resp = _call_ollama(_IMPORT_EXTRACT_TMPL.format(chunk=chunk))
                items = _parse_json_list(raw_resp)
            except Exception as exc:
                errors += 1
                print(f"  [warn] LLM error: {exc}", file=sys.stderr)
                continue

            for it in items[:5]:
                if not isinstance(it, dict):
                    continue
                item_content = str(it.get("content", "")).strip()
                if not item_content:
                    continue
                mid = (
                    f"mem_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
                    f"_{random.randint(1000, 9999)}"
                )
                insert_memory(
                    conn,
                    memory_id=mid,
                    key=str(it.get("key", ""))[:60] or "untitled",
                    content=item_content,
                    category=str(it.get("category", "other")),
                    importance=max(0.0, min(1.0, float(it.get("importance", 0.5)))),
                    session_id=session_id,
                )
                embed_q.put((mid, item_content))
                conv_memories += 1
                total_embed_queued += 1

        total_memories += conv_memories
        err_note = f"  [{errors} error(s)]" if errors else ""
        print(f"{progress} 对话 \"{label}\" → {len(chunks)} chunks → 创建 {conv_memories} 条记忆{err_note}")
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
            f"{total_chunks} chunk(s) parsed — no memories written."
        )
    else:
        print(
            f"\nDone: {total_memories} memories created from {total_chunks} chunk(s) "
            f"across {displayed} conversation(s)."
        )


if __name__ == "__main__":
    main()
