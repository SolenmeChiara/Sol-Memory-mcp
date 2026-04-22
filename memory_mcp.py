"""Standalone MCP server wrapping a SQLite hybrid-search memory store.

Usage:
    python memory_mcp.py [--db PATH]

The server communicates over stdio using the MCP JSON-RPC protocol.
Default database path: ./memory.db
"""

from __future__ import annotations

import json
import math
import os
import queue
import random
import sqlite3
import struct
import sys
import threading
import time as _time_mod
import urllib.error
import urllib.request
import webbrowser
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# Reuse multi-format parsing/iteration from batch_import.py.
# We only borrow pure-data helpers; LLM/embedding calls stay on this module's
# functions so they share the same Ollama config.
from batch_import import (
    _conv_to_text as _bi_conv_to_text,
    _quick_count as _bi_quick_count,
    _raw_items as _bi_raw_items,
    detect_format as _bi_detect_format,
)


# ---------------------------------------------------------------------------
# Ollama configuration (overridable via CLI args / env vars in main())
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.environ.get("OLLAMA_MODEL", "gemma4:e4b")
OLLAMA_TIMEOUT: float = float(os.environ.get("OLLAMA_TIMEOUT", "180"))
OLLAMA_EMBED_MODEL: str = os.environ.get("OLLAMA_EMBED_MODEL", "bge-m3")
DECAY_LAMBDA: float = float(os.environ.get("DECAY_LAMBDA", "0.05"))
DECAY_THRESHOLD: float = float(os.environ.get("DECAY_THRESHOLD", "0.3"))
SUMMARIZE_DRY_RUN: bool = False


def _call_ollama(prompt: str) -> str:
    """Call the local Ollama OpenAI-compatible chat completion endpoint."""
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "temperature": 0.3,
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    parsed = json.loads(body)
    return parsed["choices"][0]["message"]["content"]


def _call_ollama_embedding(text: str) -> list:
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


_EMOTION_PROMPT = (
    "分析以下文本，输出情感坐标。\n"
    "valence（情感效价）：0.0~1.0，0=极度消极 0.5=中性 1.0=极度积极\n"
    "arousal（唤醒度）：0.0~1.0，0=非常平静 0.5=普通 1.0=非常激动\n"
    "只输出纯 JSON，不加任何说明：\n"
    '{"valence": 0.7, "arousal": 0.4}\n\n'
    "文本：\n"
)


def _analyze_emotion(content: str) -> tuple:
    try:
        raw = _call_ollama(_EMOTION_PROMPT + content[:500])
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0]
        parsed = json.loads(cleaned)
        v = max(0.0, min(1.0, float(parsed.get("valence", 0.5))))
        a = max(0.0, min(1.0, float(parsed.get("arousal", 0.3))))
        return v, a
    except Exception:
        return 0.5, 0.3


def _calc_decay_score(rec) -> float:
    if rec.pinned:
        return 999.0
    importance = max(0.01, rec.importance) * 10.0
    activation_count = max(1.0, rec.activation_count)
    arousal = max(0.0, min(1.0, rec.arousal))
    last_str = (rec.last_active or rec.updated_at or rec.created_at or "").replace("Z", "+00:00")
    try:
        last_dt = datetime.fromisoformat(last_str)
        if last_dt.tzinfo is None:
            last_dt = last_dt.replace(tzinfo=timezone.utc)
        days_since = max(0.0, (datetime.now(timezone.utc) - last_dt).total_seconds() / 86400)
    except (ValueError, TypeError):
        days_since = 30.0
    hours = days_since * 24.0
    time_weight = 1.0 + math.exp(-hours / 36.0)
    emotion_weight = 1.0 + arousal * 0.8
    if days_since <= 3.0:
        combined = time_weight * 0.7 + emotion_weight * 0.3
    else:
        combined = emotion_weight * 0.7 + time_weight * 0.3
    base = importance * (activation_count ** 0.3) * math.exp(-DECAY_LAMBDA * days_since) * combined
    if rec.resolved and rec.digested:
        factor = 0.02
    elif rec.resolved:
        factor = 0.05
    else:
        factor = 1.0
    urgency = 1.5 if (arousal > 0.7 and not rec.resolved) else 1.0
    return round(base * factor * urgency, 4)


# ---------------------------------------------------------------------------
# Breath composition (shared by /breath-hook endpoint, extmcp_breath tool, CLI)
# ---------------------------------------------------------------------------

BREATH_TOKEN_BUDGET = int(os.environ.get("BREATH_TOKEN_BUDGET", "3000"))
BREATH_PINNED_QUOTA = int(os.environ.get("BREATH_PINNED_QUOTA", "2"))


def _compose_breath_output(
    store: "MemoryStore",
    *,
    limit: int = 10,
    do_touch: bool = False,
    touch_weight: float = 0.3,
    cooldown_hours: float = 6.0,
) -> tuple[str, list[str]]:
    """Build the 'breath' text — a weighted sample of pinned + top-decay unresolved memories.

    Returns (markdown_text, referenced_ids).

    do_touch=True applies a discounted activation: only memories not touched by breath
    in the last `cooldown_hours` get +touch_weight to activation_count and a fresh
    last_breath_at timestamp. Avoids the runaway feedback Gemini & gpt5.4 both flagged.
    """
    # 1) Pinned quota
    with store._lock:
        pinned_rows = store.conn.execute(
            "SELECT * FROM memories WHERE pinned=1 AND memory_kind='long_term' "
            "ORDER BY updated_at DESC LIMIT ?",
            (BREATH_PINNED_QUOTA,),
        ).fetchall()
    pinned_recs = [store._row_to_record(r) for r in pinned_rows]

    # 2) Unresolved candidate pool
    with store._lock:
        un_rows = store.conn.execute(
            "SELECT * FROM memories WHERE resolved=0 AND pinned=0 "
            "AND memory_kind='long_term' ORDER BY updated_at DESC LIMIT 200"
        ).fetchall()
    un_recs = [store._row_to_record(r) for r in un_rows]
    for rec in un_recs:
        rec.decay_score = _calc_decay_score(rec)
    un_recs.sort(key=lambda x: x.decay_score, reverse=True)

    # 3) Cheap dedupe: same key+date keeps only highest decay_score
    seen: dict[tuple[str, str], bool] = {}
    deduped: list = []
    for rec in un_recs:
        date_part = (rec.updated_at or rec.created_at or "")[:10]
        sig = (rec.key.strip().lower(), date_part)
        if sig in seen:
            continue
        seen[sig] = True
        deduped.append(rec)

    un_quota = max(1, limit - len(pinned_recs))
    un_pool = deduped[: un_quota * 2]

    # 4) Diversity: top1 fixed, rest shuffled (so the same #2 doesn't always lead)
    if len(un_pool) > 1:
        head, tail = un_pool[:1], un_pool[1:]
        random.shuffle(tail)
        un_picked = head + tail[: un_quota - 1]
    else:
        un_picked = un_pool[:un_quota]

    # 5) Format with token budget (rough: 1 char ≈ 1 token for CJK; whitespace flattened)
    def _fmt(rec, weight_str: str) -> str:
        flat = " ".join((rec.content or "").split())
        return f"[weight:{weight_str} V{rec.valence:.1f}/A{rec.arousal:.1f}] {rec.key}: {flat}"

    lines: list[str] = []
    used = 0
    referenced: list[str] = []

    if pinned_recs:
        lines.append("=== PINNED ===")
        for rec in pinned_recs:
            line = _fmt(rec, "999.00")
            if used + len(line) > BREATH_TOKEN_BUDGET:
                break
            lines.append(line)
            used += len(line)
            referenced.append(rec.id)

    if un_picked:
        header = "\n=== TOP UNRESOLVED (by decay) ==="
        if used + len(header) <= BREATH_TOKEN_BUDGET:
            lines.append(header)
            used += len(header)
            for rec in un_picked:
                line = _fmt(rec, f"{rec.decay_score:.2f}")
                if used + len(line) > BREATH_TOKEN_BUDGET:
                    break
                lines.append(line)
                used += len(line)
                referenced.append(rec.id)

    text = "\n".join(lines)

    # 6) Discounted touch with cooldown
    if do_touch and referenced:
        now = datetime.now(timezone.utc)
        cutoff = (now - timedelta(hours=cooldown_hours)).isoformat()
        with store._lock:
            for mid in referenced:
                row = store.conn.execute(
                    "SELECT last_breath_at FROM memories WHERE id=?", (mid,)
                ).fetchone()
                if row is None:
                    continue
                last_breath = (row["last_breath_at"] if "last_breath_at" in row.keys() else "") or ""
                if last_breath and last_breath > cutoff:
                    continue  # within cooldown, skip
                store.conn.execute(
                    "UPDATE memories SET activation_count = activation_count + ?, "
                    "last_breath_at = ? WHERE id = ?",
                    (touch_weight, now.isoformat(), mid),
                )
            store.conn.commit()

    return text, referenced


# ---------------------------------------------------------------------------
# MemoryStore – pure stdlib + sqlite3
# ---------------------------------------------------------------------------

def _pack_embedding(values: Iterable[float]) -> bytes:
    values = list(values)
    return struct.pack(f"<{len(values)}f", *values) if values else b""


def _unpack_embedding(blob: bytes) -> List[float]:
    if not blob:
        return []
    return list(struct.unpack(f"<{len(blob) // 4}f", blob))


def _cosine_similarity(left: List[float], right: List[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    ln = math.sqrt(sum(v * v for v in left))
    rn = math.sqrt(sum(v * v for v in right))
    return dot / (ln * rn) if ln and rn else 0.0


@dataclass
class MemoryRecord:
    id: str
    key: str
    content: str
    memory_kind: str
    category: str
    importance: float
    session_id: str
    created_at: str
    updated_at: str
    valence: float = 0.5
    arousal: float = 0.3
    pinned: bool = False
    resolved: bool = False
    digested: bool = False
    activation_count: float = 1.0
    last_active: str = ""
    final_score: float = 0.0
    vector_score: float = 0.0
    keyword_score: float = 0.0
    decay_score: float = 0.0


class MemoryStore:
    def __init__(
        self, db_path: Path, vector_weight: float = 0.7, keyword_weight: float = 0.3
    ):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=NORMAL")
            self.conn.execute("PRAGMA busy_timeout=5000")
        self._init_db()

    def _init_db(self) -> None:
        with self._lock:
            self.conn.executescript(
                """
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
              INSERT INTO memories_fts(rowid, key, content)
              VALUES (new.rowid, new.key, new.content);
            END;
            CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
              INSERT INTO memories_fts(memories_fts, rowid, key, content)
              VALUES ('delete', old.rowid, old.key, old.content);
            END;
            CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
              INSERT INTO memories_fts(memories_fts, rowid, key, content)
              VALUES ('delete', old.rowid, old.key, old.content);
              INSERT INTO memories_fts(rowid, key, content)
              VALUES (new.rowid, new.key, new.content);
            END;
                """
            )
            columns = {
                str(r["name"])
                for r in self.conn.execute("PRAGMA table_info(memories)").fetchall()
            }
            if "memory_kind" not in columns:
                self.conn.execute(
                    "ALTER TABLE memories ADD COLUMN memory_kind TEXT NOT NULL DEFAULT 'long_term'"
                )
            _NEW_COLS = [
                ("valence",          "REAL DEFAULT 0.5"),
                ("arousal",          "REAL DEFAULT 0.3"),
                ("pinned",           "INTEGER DEFAULT 0"),
                ("resolved",         "INTEGER DEFAULT 0"),
                ("digested",         "INTEGER DEFAULT 0"),
                ("activation_count", "REAL DEFAULT 1.0"),
                ("last_active",      "TEXT DEFAULT ''"),
                ("last_breath_at",   "TEXT DEFAULT ''"),  # cooldown for breath-induced touch
            ]
            for col_name, col_def in _NEW_COLS:
                if col_name not in columns:
                    self.conn.execute(f"ALTER TABLE memories ADD COLUMN {col_name} {col_def}")
            self.conn.commit()

    def touch_memory(self, memory_id: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            # Primary: refresh last_active + activation_count+1
            self.conn.execute(
                "UPDATE memories SET last_active=?, activation_count=activation_count+1, updated_at=? WHERE id=?",
                (now, now, memory_id),
            )
            # Time ripple: neighbours within ±48h get activation_count+0.2
            row = self.conn.execute(
                "SELECT created_at FROM memories WHERE id=?", (memory_id,)
            ).fetchone()
            if row and row[0]:
                created = row[0].replace("Z", "+00:00")
                try:
                    dt = datetime.fromisoformat(created)
                    before = (dt - timedelta(hours=48)).isoformat()
                    after = (dt + timedelta(hours=48)).isoformat()
                    self.conn.execute(
                        "UPDATE memories SET activation_count=activation_count+0.2 "
                        "WHERE id!=? AND created_at BETWEEN ? AND ?",
                        (memory_id, before, after),
                    )
                except (ValueError, TypeError):
                    pass
            self.conn.commit()

    def upsert_memory(
        self,
        *,
        memory_id: str,
        key: str,
        content: str,
        memory_kind: str = "long_term",
        category: str = "other",
        importance: float = 0.5,
        session_id: str = "",
        embedding: Optional[List[float]] = None,
        valence: float = 0.5,
        arousal: float = 0.3,
        pinned: bool = False,
        resolved: bool = False,
        digested: bool = False,
        activation_count: float = 1.0,
        last_active: str = "",
    ) -> MemoryRecord:
        if pinned:
            importance = 1.0
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            existing = self.conn.execute(
                "SELECT created_at FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
            self.conn.execute(
                """
            INSERT INTO memories(id,key,content,memory_kind,category,importance,session_id,
                                 created_at,updated_at,embedding,
                                 valence,arousal,pinned,resolved,digested,activation_count,last_active)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET
              key=excluded.key, content=excluded.content, memory_kind=excluded.memory_kind,
              category=excluded.category, importance=excluded.importance,
              session_id=excluded.session_id, updated_at=excluded.updated_at,
              embedding=excluded.embedding,
              valence=excluded.valence, arousal=excluded.arousal,
              pinned=excluded.pinned, resolved=excluded.resolved,
              digested=excluded.digested, activation_count=excluded.activation_count,
              last_active=excluded.last_active
            """,
                (
                    memory_id, key, content, memory_kind, category, importance,
                    session_id, existing[0] if existing else now, now,
                    _pack_embedding(embedding or []),
                    valence, arousal, int(pinned), int(resolved), int(digested),
                    activation_count, last_active or now,
                ),
            )
            self.conn.commit()
            row = self.conn.execute(
                "SELECT * FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
        return self._row_to_record(row)

    def get_memory(self, memory_id: str) -> Optional[MemoryRecord]:
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
        return self._row_to_record(row) if row else None

    def delete_memory(self, memory_id: str) -> bool:
        with self._lock:
            cur = self.conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            self.conn.commit()
        return cur.rowcount > 0

    def list_memories(self, limit: int = 50, memory_kind: str = "long_term") -> List[MemoryRecord]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM memories WHERE memory_kind = ? ORDER BY updated_at DESC LIMIT ?",
                (memory_kind, limit),
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def random_memories(self, count: int, memory_kind: str = "long_term") -> List[MemoryRecord]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM memories WHERE memory_kind = ? ORDER BY RANDOM() LIMIT ?",
                (memory_kind, count),
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def search(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None,
        limit: int = 8,
        memory_kind: str = "long_term",
    ) -> List[MemoryRecord]:
        fallback_limit = max(limit * 6, 24)
        tokens = [t.strip() for t in query.replace("\uff0c", " ").replace(",", " ").split() if t.strip()]

        try:
            with self._lock:
                keyword_rows = self.conn.execute(
                    """
                    SELECT m.*, bm25(memories_fts) AS keyword_score
                    FROM memories_fts
                    JOIN memories m ON m.rowid = memories_fts.rowid
                    WHERE memories_fts MATCH ? AND m.memory_kind = ?
                    ORDER BY keyword_score LIMIT ?
                    """,
                    (" OR ".join(dict.fromkeys(tokens)) or query, memory_kind, fallback_limit),
                ).fetchall()
        except sqlite3.Error:
            keyword_rows = []

        if not keyword_rows:
            like_clauses: list[str] = []
            params: list[Any] = [memory_kind]
            for token in (tokens or [query])[:8]:
                like_clauses.append("key LIKE ? OR content LIKE ?")
                params.extend([f"%{token}%", f"%{token}%"])
            where = " OR ".join(f"({c})" for c in like_clauses)
            params.append(fallback_limit)
            with self._lock:
                keyword_rows = self.conn.execute(
                    f"SELECT *, 0.5 AS keyword_score FROM memories WHERE memory_kind = ? AND ({where}) ORDER BY updated_at DESC LIMIT ?",
                    tuple(params),
                ).fetchall()

        keyword_hits: list[MemoryRecord] = []
        if keyword_rows:
            max_s = max(abs(float(r["keyword_score"])) for r in keyword_rows) or 1.0
            for r in keyword_rows:
                rec = self._row_to_record(r)
                rec.keyword_score = 1.0 - min(abs(float(r["keyword_score"])) / max_s, 1.0)
                keyword_hits.append(rec)

        vector_hits: list[MemoryRecord] = []
        if query_embedding:
            with self._lock:
                rows = self.conn.execute(
                    "SELECT * FROM memories WHERE memory_kind = ? AND length(embedding) > 0",
                    (memory_kind,),
                ).fetchall()
            for r in rows:
                s = _cosine_similarity(query_embedding, _unpack_embedding(r["embedding"]))
                if s > 0.0:
                    rec = self._row_to_record(r)
                    rec.vector_score = s
                    vector_hits.append(rec)
            vector_hits.sort(key=lambda x: x.vector_score, reverse=True)
            vector_hits = vector_hits[:max(limit * 3, limit)]

        merged: dict[str, MemoryRecord] = {}
        for rec in keyword_hits:
            merged[rec.id] = rec
        for rec in vector_hits:
            if rec.id in merged:
                merged[rec.id].vector_score = max(merged[rec.id].vector_score, rec.vector_score)
            else:
                merged[rec.id] = rec

        items = list(merged.values())
        for rec in items:
            rec.final_score = self.vector_weight * rec.vector_score + self.keyword_weight * rec.keyword_score
        items.sort(key=lambda x: x.final_score, reverse=True)
        return items[:limit]

    def _row_to_record(self, row: sqlite3.Row) -> MemoryRecord:
        keys = row.keys()

        def _get(k, default):
            return row[k] if k in keys else default

        return MemoryRecord(
            id=row["id"],
            key=row["key"],
            content=row["content"],
            memory_kind=str(_get("memory_kind", "long_term") or "long_term"),
            category=row["category"],
            importance=float(row["importance"]),
            session_id=str(_get("session_id", "") or ""),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            valence=float(_get("valence", 0.5) or 0.5),
            arousal=float(_get("arousal", 0.3) or 0.3),
            pinned=bool(int(_get("pinned", 0) or 0)),
            resolved=bool(int(_get("resolved", 0) or 0)),
            digested=bool(int(_get("digested", 0) or 0)),
            activation_count=float(_get("activation_count", 1.0) or 1.0),
            last_active=str(_get("last_active", "") or ""),
        )


# ---------------------------------------------------------------------------
# MCP stdio transport
# ---------------------------------------------------------------------------

def _read_message(stream) -> Optional[Dict[str, Any]]:
    """Read one JSON-RPC message with Content-Length framing from *stream*."""
    headers: dict[str, str] = {}
    while True:
        line = stream.readline()
        if not line:
            return None
        line_str = line.decode("utf-8", errors="replace").rstrip("\r\n")
        if line_str == "":
            break
        if ":" in line_str:
            k, v = line_str.split(":", 1)
            headers[k.strip().lower()] = v.strip()
    length = int(headers.get("content-length", "0"))
    if length <= 0:
        return None
    body = stream.read(length)
    return json.loads(body.decode("utf-8", errors="replace"))


def _write_message(msg: Dict[str, Any]) -> None:
    body = json.dumps(msg, ensure_ascii=False).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
    sys.stdout.buffer.write(header + body)
    sys.stdout.buffer.flush()


def _response(request_id: Any, result: Any) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _error(request_id: Any, code: int, message: str) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


# ---------------------------------------------------------------------------
# Conversation import helpers (shared by web /import endpoint)
# ---------------------------------------------------------------------------

_IMPORT_EXTRACT_TMPL = (
    "从以下对话片段中提取 0-5 条值得记忆的事实、偏好、承诺或重要事件。\n"
    "每条记忆包含：key（简短标题）、content（具体内容）、"
    "category（preference/promise/event/anniversary/emotion/habit/boundary/other 之一）、"
    "importance（0.0~1.0）。\n"
    "如果没有值得记忆的内容，返回空数组 []。\n"
    "只输出纯 JSON 数组，不加任何说明：\n"
    '[{{"key":"...","content":"...","category":"...","importance":0.7}}]\n\n'
    "对话片段：\n{chunk}"
)

# ---------------------------------------------------------------------------
# Import task registry (background processing for large/Claude-official files)
# ---------------------------------------------------------------------------

_IMPORT_TASKS: Dict[str, Dict[str, Any]] = {}
_IMPORT_TASKS_LOCK = threading.Lock()
_IMPORT_EMBED_QUEUE: "queue.Queue" = queue.Queue()
_IMPORT_EMBED_STARTED = False
_IMPORT_EMBED_LOCK = threading.Lock()
_IMPORT_EMBED_WORKERS = 2


def _ensure_embed_pool(store: "MemoryStore") -> None:
    """Lazy-start a small worker pool that updates embeddings for inserted memories."""
    global _IMPORT_EMBED_STARTED
    with _IMPORT_EMBED_LOCK:
        if _IMPORT_EMBED_STARTED:
            return
        _IMPORT_EMBED_STARTED = True

    def _worker() -> None:
        while True:
            item = _IMPORT_EMBED_QUEUE.get()
            try:
                if item is None:
                    continue
                mid, content = item
                emb = _call_ollama_embedding(content)
                if emb:
                    with store._lock:
                        store.conn.execute(
                            "UPDATE memories SET embedding=? WHERE id=?",
                            (_pack_embedding(emb), mid),
                        )
                        store.conn.commit()
            except Exception as exc:
                sys.stderr.write(f"[memory-mcp] embed worker error: {exc}\n")
            finally:
                _IMPORT_EMBED_QUEUE.task_done()

    for _ in range(_IMPORT_EMBED_WORKERS):
        threading.Thread(target=_worker, daemon=True).start()


def _process_conversations(
    store: "MemoryStore",
    items: Iterable[Dict[str, Any]],
    *,
    task: Optional[Dict[str, Any]] = None,
    session_prefix: str = "import",
) -> Dict[str, Any]:
    """Iterate conversation dicts, chunk per-conversation, extract memories via LLM.

    If *task* is given, mutate task fields under the registry lock for live progress.
    Returns final stats dict.
    """
    _ensure_embed_pool(store)

    created = 0
    skipped = 0
    processed = 0
    errors: List[str] = []

    for raw_idx, item in enumerate(items):
        title, text = _bi_conv_to_text(item)
        label = (title or f"untitled #{raw_idx}")[:60]

        if not text.strip():
            skipped += 1
            processed += 1
            if task is not None:
                with _IMPORT_TASKS_LOCK:
                    task["processed"] = processed
                    task["skipped"] = skipped
                    task["last_title"] = label + " (skipped: empty)"
            continue

        chunks = _chunk_conversation(text)
        session_id = f"{session_prefix}_{raw_idx}"
        conv_created = 0

        for chunk in chunks:
            try:
                raw = _call_ollama(_IMPORT_EXTRACT_TMPL.format(chunk=chunk))
                ext_items = _parse_json_list(raw)
            except Exception as exc:
                errors.append(f"#{raw_idx} chunk: {exc}")
                continue
            for it in ext_items[:5]:
                if not isinstance(it, dict):
                    continue
                item_content = str(it.get("content", "")).strip()
                if not item_content:
                    continue
                mid = (
                    f"mem_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
                    f"_{random.randint(1000, 9999)}"
                )
                store.upsert_memory(
                    memory_id=mid,
                    key=str(it.get("key", ""))[:60] or "untitled",
                    content=item_content,
                    category=str(it.get("category", "other")),
                    importance=max(0.0, min(1.0, float(it.get("importance", 0.5)))),
                    session_id=session_id,
                )
                _IMPORT_EMBED_QUEUE.put((mid, item_content))
                conv_created += 1

        created += conv_created
        processed += 1
        if task is not None:
            with _IMPORT_TASKS_LOCK:
                task["processed"] = processed
                task["created"] = created
                task["errors"] = errors[-20:]
                task["last_title"] = f"{label} → {conv_created} 条"

    return {
        "processed": processed,
        "skipped": skipped,
        "created": created,
        "errors": errors,
    }


def _start_import_task(store: "MemoryStore", path: Path) -> Dict[str, Any]:
    """Load *path*, register a task, kick off background processing. Returns task snapshot."""
    task_id = f"task_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{random.randint(100, 999)}"
    task: Dict[str, Any] = {
        "id": task_id,
        "path": str(path),
        "format": "loading",
        "total": 0,
        "processed": 0,
        "skipped": 0,
        "created": 0,
        "errors": [],
        "last_title": "",
        "done": False,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": "",
    }
    with _IMPORT_TASKS_LOCK:
        _IMPORT_TASKS[task_id] = task

    def _runner() -> None:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            fmt = _bi_detect_format(data)
            total = _bi_quick_count(data)
            with _IMPORT_TASKS_LOCK:
                task["format"] = fmt
                task["total"] = total
            sys.stderr.write(
                f"[memory-mcp] import task {task_id}: format={fmt} total={total}\n"
            )
            sys.stderr.flush()

            _process_conversations(
                store,
                _bi_raw_items(data),
                task=task,
                session_prefix=f"import_{path.stem}",
            )
        except Exception as exc:
            with _IMPORT_TASKS_LOCK:
                task["errors"].append(f"fatal: {exc}")
        finally:
            with _IMPORT_TASKS_LOCK:
                task["done"] = True
                task["finished_at"] = datetime.now(timezone.utc).isoformat()
            sys.stderr.write(
                f"[memory-mcp] import task {task_id} done: created={task['created']}\n"
            )
            sys.stderr.flush()

    threading.Thread(target=_runner, daemon=True).start()
    return task


_IMPORT_HTML = """\
<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Memory Import</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#f5f5f5;--fg:#1a1a1a;--sub:#666;
  --card-bg:#fff;--card-border:#ddd;
  --drop-border:#bbb;--drop-hover-border:#555;--drop-hover-bg:#efefef;
  --label:#555;--label-b:#111;--formats:#999;
  --prog-bg:#e0e0e0;--prog-a:#888;--prog-b:#333;
  --log-bg:#f9f9f9;--log-border:#ddd;
  --ok:#333;--err:#000;--info:#888;--warn:#555;
  --res-bg:#fff;--res-border:#ccc;--res-h:#222;--res-p:#444;--num:#000;
  --ebox-bg:#fff;--ebox-border:#ccc;--ebox-h:#222;--ebox-p:#555;
}
@media(prefers-color-scheme:dark){
  :root{
    --bg:#111;--fg:#e0e0e0;--sub:#888;
    --card-bg:#1a1a1a;--card-border:#333;
    --drop-border:#444;--drop-hover-border:#888;--drop-hover-bg:#222;
    --label:#aaa;--label-b:#e0e0e0;--formats:#666;
    --prog-bg:#2a2a2a;--prog-a:#555;--prog-b:#aaa;
    --log-bg:#0d0d0d;--log-border:#2a2a2a;
    --ok:#c8c8c8;--err:#fff;--info:#777;--warn:#aaa;
    --res-bg:#1a1a1a;--res-border:#333;--res-h:#e0e0e0;--res-p:#b0b0b0;--num:#fff;
    --ebox-bg:#1a1a1a;--ebox-border:#444;--ebox-h:#e0e0e0;--ebox-p:#aaa;
  }
}
body{background:var(--bg);color:var(--fg);font-family:system-ui,-apple-system,sans-serif;min-height:100vh;display:flex;flex-direction:column;align-items:center;padding:48px 20px}
h1{color:var(--fg);font-size:28px;margin-bottom:6px}
.sub{color:var(--sub);font-size:14px;margin-bottom:40px}
.card{background:var(--card-bg);border:1px solid var(--card-border);border-radius:14px;padding:32px;width:100%;max-width:580px}
.drop-zone{border:2px dashed var(--drop-border);border-radius:10px;padding:52px 32px;text-align:center;cursor:pointer;transition:all .2s;position:relative}
.drop-zone:hover,.drop-zone.over{border-style:solid;border-color:var(--drop-hover-border);background:var(--drop-hover-bg)}
.drop-zone input{position:absolute;inset:0;opacity:0;cursor:pointer}
.drop-label{color:var(--label);font-size:15px;line-height:1.7}
.drop-label b{color:var(--label-b)}
.formats{color:var(--formats);font-size:12px;margin-top:6px}
#statusSection{margin-top:28px;display:none}
.prog-bar{background:var(--prog-bg);border-radius:6px;height:5px;overflow:hidden;margin-bottom:16px}
.prog-fill{height:100%;background:linear-gradient(90deg,var(--prog-a),var(--prog-b));width:0%;transition:width .4s ease}
.log{background:var(--log-bg);border-radius:8px;padding:14px 16px;font-family:monospace;font-size:13px;line-height:1.8;max-height:260px;overflow-y:auto;border:1px solid var(--log-border)}
.ok{color:var(--ok)}.err{color:var(--err);font-weight:600}.info{color:var(--info)}.warn{color:var(--warn)}
.result-box{margin-top:16px;background:var(--res-bg);border:1px solid var(--res-border);border-radius:8px;padding:16px}
.result-box h3{color:var(--res-h);margin-bottom:10px;font-size:16px}
.result-box p{color:var(--res-p);font-size:14px;line-height:2}
.num{color:var(--num);font-weight:600;font-size:16px}
.err-box{background:var(--ebox-bg);border:1px solid var(--ebox-border);border-radius:8px;padding:16px;margin-top:16px}
.err-box h3{color:var(--ebox-h);margin-bottom:8px}
.err-box p{color:var(--ebox-p);font-size:13px;line-height:1.8}
</style>
</head>
<body>
<h1>Memory Import</h1>
<p class="sub">导入对话记录，自动提取记忆碎片</p>
<div class="card">
  <div class="drop-zone" id="dropZone">
    <input type="file" id="fileInput" accept=".json,.md,.txt">
    <div class="drop-label">
      <b>拖拽小文件到此处</b>，或点击选择
      <div class="formats">≤ 30 MB · .json / .md / .txt</div>
    </div>
  </div>
  <div style="margin-top:18px;display:flex;align-items:center;gap:10px">
    <span style="color:var(--sub);font-size:12px">大文件用本机路径</span>
    <span style="flex:1;height:1px;background:var(--card-border)"></span>
  </div>
  <div style="margin-top:14px">
    <input id="pathInput" type="text" placeholder='C:\\path\\to\\conversations.json'
      style="width:100%;padding:11px 12px;border:1px solid var(--card-border);border-radius:8px;background:var(--card-bg);color:var(--fg);font-family:monospace;font-size:13px">
    <div style="margin-top:8px;display:flex;gap:8px;align-items:center">
      <button id="pathSubmit" style="padding:9px 18px;border:1px solid var(--label-b);border-radius:8px;background:var(--card-bg);color:var(--fg);cursor:pointer;font-size:13px">服务器读取并导入</button>
      <span style="color:var(--formats);font-size:12px">支持 Claude 官方导出 / 插件导出 / ChatGPT mapping</span>
    </div>
  </div>
  <div id="statusSection">
    <div class="prog-bar"><div class="prog-fill" id="progFill"></div></div>
    <div id="statusLine" style="font-family:monospace;font-size:13px;color:var(--label);margin-bottom:10px"></div>
    <div class="log" id="log"></div>
    <div id="resultArea"></div>
  </div>
</div>
<script>
const dz=document.getElementById('dropZone'),fi=document.getElementById('fileInput');
const ss=document.getElementById('statusSection'),logEl=document.getElementById('log');
const pf=document.getElementById('progFill'),ra=document.getElementById('resultArea');
const pathIn=document.getElementById('pathInput'),pathBtn=document.getElementById('pathSubmit');
const stLine=document.getElementById('statusLine');
const SIZE_LIMIT=30*1024*1024;
dz.addEventListener('dragover',e=>{e.preventDefault();dz.classList.add('over')});
dz.addEventListener('dragleave',()=>dz.classList.remove('over'));
dz.addEventListener('drop',e=>{e.preventDefault();dz.classList.remove('over');if(e.dataTransfer.files[0])run(e.dataTransfer.files[0])});
fi.addEventListener('change',e=>{if(e.target.files[0])run(e.target.files[0])});
pathBtn.addEventListener('click',()=>{const p=pathIn.value.trim().replace(/^["']|["']$/g,'');if(p)runPath(p)});
pathIn.addEventListener('keydown',e=>{if(e.key==='Enter')pathBtn.click()});
function addLog(msg,cls){const d=document.createElement('div');d.className=cls||'info';d.textContent=msg;logEl.appendChild(d);logEl.scrollTop=logEl.scrollHeight}
function setP(p){pf.style.width=p+'%'}
function setStatus(s){stLine.textContent=s}
function resetUI(){ss.style.display='block';logEl.innerHTML='';ra.innerHTML='';stLine.textContent='';setP(0)}
async function run(file){
  resetUI();setP(5);
  const sizeMB=(file.size/1024/1024).toFixed(1);
  addLog(file.name+' ('+sizeMB+' MB)');
  if(file.size>SIZE_LIMIT){
    addLog('文件超过 30 MB，浏览器读不动。','err');
    addLog('请把绝对路径粘到下方输入框，由服务器直接读取处理。','warn');
    pathIn.value=file.name;pathIn.focus();
    setP(0);return;
  }
  let text;
  try{
    text=await new Promise((res,rej)=>{const r=new FileReader();r.onload=e=>res(e.target.result);r.onerror=rej;r.readAsText(file,'utf-8')});
  }catch(e){addLog('读取失败: '+e.message,'err');return}
  addLog('读取完成，'+text.length+' 字符');setP(15);
  addLog('发送至服务器处理，请耐心等待…','warn');
  try{
    const resp=await fetch('/import',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({content:text,filename:file.name})});
    const data=await resp.json();
    if(data.async){await pollTask(data.task_id);return}
    setP(100);renderSync(data);
  }catch(e){setP(0);addLog('请求失败: '+e.message,'err')}
}
async function runPath(path){
  resetUI();setP(3);
  addLog('提交服务器路径: '+path);
  setStatus('启动后台任务…');
  try{
    const resp=await fetch('/import',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({path:path})});
    const data=await resp.json();
    if(!data.ok){addLog('错误: '+(data.error||'未知'),'err');return}
    if(data.async){await pollTask(data.task_id);return}
    setP(100);renderSync(data);
  }catch(e){addLog('请求失败: '+e.message,'err')}
}
async function pollTask(taskId){
  addLog('后台任务 '+taskId+' 启动','ok');
  let lastProcessed=-1;
  while(true){
    await new Promise(r=>setTimeout(r,2500));
    let s;
    try{const r=await fetch('/import/status?task_id='+encodeURIComponent(taskId));s=await r.json()}
    catch(e){addLog('状态查询失败: '+e.message,'err');continue}
    if(s.error){addLog('任务错误: '+s.error,'err');return}
    const tot=Math.max(1,s.total||1);
    setP(Math.round(100*s.processed/tot));
    setStatus('['+s.processed+'/'+s.total+' 对话] 创建记忆 '+s.created+' · 跳过 '+s.skipped+' · 错误 '+(s.errors||[]).length+(s.last_title?' · '+s.last_title:''));
    if(s.processed!==lastProcessed){lastProcessed=s.processed}
    if(s.done){
      setP(100);
      addLog('任务完成','ok');
      ra.innerHTML='<div class="result-box"><h3>导入完成</h3><p>格式：<span class="num">'+s.format+'</span><br>对话数：<span class="num">'+s.total+'</span><br>创建记忆：<span class="num">'+s.created+'</span><br>跳过空对话：<span class="num">'+s.skipped+'</span><br>错误：<span class="num">'+(s.errors||[]).length+'</span></p></div>';
      return;
    }
  }
}
function renderSync(data){
  if(!data.ok){
    addLog('服务器返回错误: '+(data.error||JSON.stringify(data)),'err');
    ra.innerHTML='<div class="err-box"><h3>处理失败</h3><p>'+(data.error||'未知错误')+'</p></div>';
    return;
  }
  addLog('处理完成','ok');
  const esc=s=>String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  let errs='无错误';
  if(data.errors&&data.errors.length){
    const detail=data.errors.slice(0,5).map(e=>'<li>'+esc(e)+'</li>').join('');
    errs='<span style="color:#fbbf24">'+data.errors.length+' 个片段提取失败</span><ul style="font-size:12px;color:#94a3b8;margin-top:4px">'+detail+'</ul>';
  }
  const modeLabel=data.mode==='conversations'?'按对话':'按片段';
  ra.innerHTML='<div class="result-box"><h3>导入成功</h3><p>模式：<span class="num">'+modeLabel+'</span><br>处理：<span class="num">'+(data.chunks_processed||data.processed||0)+'</span><br>创建记忆：<span class="num">'+data.created+'</span><br>'+errs+'</p></div>';
}
</script>
</body>
</html>"""


def _parse_conversation(text: str) -> str:
    """Normalise Claude/ChatGPT JSON export to plain text; pass through everything else."""
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "messages" in parsed:
            parts: list[str] = []
            # Add conversation title from metadata if present (Claude Exporter format)
            meta = parsed.get("metadata", {})
            title = meta.get("title", "")
            if title:
                parts.append(f"--- 对话：{title} ---")
            for m in parsed["messages"]:
                role = m.get("role", "")
                # Claude Exporter plugin uses "say" field; OpenAI uses "content"
                mc = m.get("say") or m.get("content", "")
                if isinstance(mc, str) and mc:
                    parts.append(f"{role}: {mc}")
                elif isinstance(mc, list):
                    for p in mc:
                        if isinstance(p, dict) and p.get("type") == "text":
                            t = p.get("text", "")
                            if t:
                                parts.append(f"{role}: {t}")
            return "\n".join(parts)
        elif isinstance(parsed, list):
            parts = []
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                # Claude official export: list of conversations with chat_messages
                if "chat_messages" in item:
                    conv_name = item.get("name", "")
                    if conv_name:
                        parts.append(f"\n--- 对话：{conv_name} ---")
                    for msg in item.get("chat_messages", []):
                        role = msg.get("sender", "")
                        msg_text = msg.get("text", "")
                        if not msg_text:
                            # fall back to content[].text
                            for block in msg.get("content", []):
                                if isinstance(block, dict) and block.get("type") == "text":
                                    msg_text = block.get("text", "")
                                    break
                        if msg_text:
                            parts.append(f"{role}: {msg_text}")
                else:
                    # ChatGPT / generic list-of-messages format
                    role = item.get("role", "") or (item.get("author") or {}).get("role", "")
                    mc = item.get("content", "")
                    if isinstance(mc, str) and mc:
                        parts.append(f"{role}: {mc}")
            return "\n".join(parts)
    except (json.JSONDecodeError, TypeError):
        pass
    return text


def _chunk_conversation(text: str, window: int = 8000) -> list[str]:
    """Split text into fixed-size windows for LLM processing."""
    return [text[i:i + window] for i in range(0, len(text), window)]


def _parse_json_list(raw: str) -> list:
    """Parse a JSON list from LLM output, stripping markdown fences. Returns [] on any parse failure."""
    import re as _re
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        parsed = json.loads(cleaned)
        if not isinstance(parsed, list):
            return [parsed] if parsed else []
        return parsed
    except json.JSONDecodeError:
        # LLM returned prose text — try to extract the JSON array
        match = _re.search(r'\[.*?\]', cleaned, _re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass
        return []


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "extmcp_save_memory",
        "description": "Save or update a memory record. Persist preferences, events, facts, or anything worth remembering long-term. Embedding and emotion analysis run in the background automatically.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Short title or label"},
                "content": {"type": "string", "description": "Detailed content"},
                "category": {
                    "type": "string",
                    "enum": ["preference", "promise", "event", "anniversary", "emotion", "habit", "boundary", "other"],
                    "description": "Category (default: other)",
                },
                "importance": {"type": "number", "description": "0.0 to 1.0 (default 0.5)"},
                "id": {"type": "string", "description": "Optional existing memory ID to update"},
                "valence": {"type": "number", "description": "Emotional valence 0.0~1.0 (auto-detected if omitted)"},
                "arousal": {"type": "number", "description": "Arousal/intensity 0.0~1.0 (auto-detected if omitted)"},
                "pinned": {"type": "boolean", "description": "Pin memory permanently (forces importance=1.0, decay_score=999)"},
                "resolved": {"type": "boolean", "description": "Mark as resolved (reduces decay weight by 95%)"},
                "digested": {"type": "boolean", "description": "Mark as digested (combined with resolved: reduces decay by 98%)"},
            },
            "required": ["key", "content"],
        },
    },
    {
        "name": "extmcp_search_memory",
        "description": "Search memories using hybrid keyword + vector search. Automatically generates bge-m3 embedding for the query. Returns valence, arousal, pinned, and decay_score for each result.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer", "description": "Max results (default 8)"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "extmcp_list_memories",
        "description": "List recent memories ordered by last update time. Returns valence, arousal, pinned, and decay_score for each record.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "Max results (default 20)"},
            },
        },
    },
    {
        "name": "extmcp_delete_memory",
        "description": "Delete a memory record by its ID.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Memory ID to delete"},
            },
            "required": ["id"],
        },
    },
    {
        "name": "extmcp_summarize_recent",
        "description": (
            "Generate a fresh structured Chinese summary of the N most recently "
            "updated memories using a local LLM (default 10, range 1-30). "
            "Output has four sections: main themes, key concerns, latest events, "
            "and emotional tone. Capped at 1200 characters with referenced "
            "memory IDs appended. Each call regenerates from scratch and also "
            "activates (touches) every referenced memory — so summarising counts "
            "as recall and refreshes their decay scores. "
            "IMPORTANT: This tool calls a local LLM and may take 10-60 seconds. "
            "Please wait patiently and do NOT retry on perceived slowness."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "How many recent memories to summarise (1-30, default 10)",
                },
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "extmcp_random_memories",
        "description": (
            "Return a random sample of 4-10 memories. The count itself is "
            "randomized on each call (no parameters accepted). Useful for "
            "serendipitous recall or browsing the memory store."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "name": "extmcp_dream",
        "description": (
            "Introspective analysis of recent memories. Finds the most semantically "
            "connected pair (via bge-m3 cosine similarity) and generates a reflective "
            "summary. Shows each memory's key, emotion scores (valence/arousal), "
            "decay_score, and content. Call this to discover hidden connections or "
            "decide which memories to resolve/digest. No parameters needed."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "name": "extmcp_grow",
        "description": (
            "Split a diary entry or long text into 2-6 discrete memory records, "
            "each automatically labeled with category, importance, valence, and arousal. "
            "Texts shorter than 30 characters are stored as-is. "
            "Embeddings are generated in the background."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Diary entry or long text to split and store"},
                "session_id": {"type": "string", "description": "Optional session ID"},
            },
            "required": ["content"],
        },
    },
    {
        "name": "extmcp_breath",
        "description": (
            "主动呼吸：浮现当前权重最高的未解决记忆 + pinned 核心。"
            "这是一个带有潜意识偏好、情感倾向和多样性采样的**模糊采样**工具——"
            "适合在对话开始或想要回忆当前关注点时调用。"
            "**不要**用它来精确查找特定的历史事件（那种场景请用 extmcp_search_memory）。"
            "被浮现的每条记忆会按 0.3 折扣激活（activation_count += 0.3），"
            "且 6 小时内同一条不会重复激活，避免回音壁效应。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "浮现上限（默认 10，范围 1-20）",
                },
            },
            "additionalProperties": False,
        },
    },
]


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------

def handle_tool(store: MemoryStore, name: str, args: Dict[str, Any]) -> Any:
    if name == "extmcp_save_memory":
        key = str(args.get("key", "")).strip()
        content = str(args.get("content", "")).strip()
        if not key or not content:
            raise ValueError("key and content are required")
        category = str(args.get("category", "other")).strip() or "other"
        importance = max(0.0, min(1.0, float(args.get("importance", 0.5) or 0.5)))
        memory_id = str(args.get("id", "")).strip()
        if not memory_id:
            memory_id = f"mem_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
        pinned = bool(args.get("pinned", False))
        resolved = bool(args.get("resolved", False))
        digested = bool(args.get("digested", False))
        user_valence = args.get("valence")
        user_arousal = args.get("arousal")
        valence = float(user_valence) if user_valence is not None else 0.5
        arousal = float(user_arousal) if user_arousal is not None else 0.3

        rec = store.upsert_memory(
            memory_id=memory_id, key=key, content=content,
            category=category, importance=importance,
            pinned=pinned, resolved=resolved, digested=digested,
            valence=valence, arousal=arousal,
        )

        do_emotion = (user_valence is None or user_arousal is None)

        def _bg_update(mid: str, txt: str, run_emotion: bool) -> None:
            emb = _call_ollama_embedding(txt)
            updates: list[str] = []
            params: list[Any] = []
            if emb:
                updates.append("embedding=?")
                params.append(_pack_embedding(emb))
            if run_emotion:
                v, a = _analyze_emotion(txt)
                updates.append("valence=?")
                params.append(v)
                updates.append("arousal=?")
                params.append(a)
            if updates:
                params.append(mid)
                with store._lock:
                    store.conn.execute(
                        f"UPDATE memories SET {', '.join(updates)} WHERE id=?",
                        tuple(params),
                    )
                    store.conn.commit()

        threading.Thread(target=_bg_update, args=(memory_id, content, do_emotion), daemon=True).start()

        ds = _calc_decay_score(rec)
        return [{"type": "text", "text": json.dumps({
            "ok": True, "id": rec.id, "key": rec.key,
            "category": rec.category, "importance": rec.importance,
            "valence": rec.valence, "arousal": rec.arousal,
            "pinned": rec.pinned, "resolved": rec.resolved,
            "decay_score": ds,
            "note": "embedding & emotion analysis running in background",
        }, ensure_ascii=False)}]

    elif name == "extmcp_search_memory":
        query = str(args.get("query", "")).strip()
        if not query:
            raise ValueError("query is required")
        limit = max(1, min(20, int(args.get("limit", 8) or 8)))
        query_embedding = _call_ollama_embedding(query) or None
        results = store.search(query, query_embedding=query_embedding, limit=limit)
        for r in results:
            store.touch_memory(r.id)
        items = [
            {
                "id": r.id, "key": r.key, "content": r.content,
                "category": r.category, "importance": r.importance,
                "score": round(r.final_score, 4),
                "valence": r.valence, "arousal": r.arousal,
                "pinned": r.pinned, "decay_score": _calc_decay_score(r),
            }
            for r in results
        ]
        return [{"type": "text", "text": json.dumps(
            {"query": query, "count": len(items), "items": items}, ensure_ascii=False
        )}]

    elif name == "extmcp_list_memories":
        limit = max(1, min(100, int(args.get("limit", 20) or 20)))
        results = store.list_memories(limit=limit)
        items = [
            {
                "id": r.id, "key": r.key, "content": r.content,
                "category": r.category, "importance": r.importance,
                "updated_at": r.updated_at,
                "valence": r.valence, "arousal": r.arousal,
                "pinned": r.pinned, "decay_score": _calc_decay_score(r),
            }
            for r in results
        ]
        return [{"type": "text", "text": json.dumps({"count": len(items), "items": items}, ensure_ascii=False)}]

    elif name == "extmcp_delete_memory":
        memory_id = str(args.get("id", "")).strip()
        if not memory_id:
            raise ValueError("id is required")
        deleted = store.delete_memory(memory_id)
        return [{"type": "text", "text": json.dumps({"ok": deleted, "id": memory_id}, ensure_ascii=False)}]

    elif name == "extmcp_summarize_recent":
        import traceback as _tb
        try:
            limit = max(1, min(30, int(args.get("limit", 10) or 10)))
            recent = store.list_memories(limit=limit)
            if not recent:
                return [{"type": "text", "text": json.dumps(
                    {"ok": False, "error": "no memories available to summarize"},
                    ensure_ascii=False,
                )}]

            memory_block = "\n\n".join(
                f"[{i + 1}] category={r.category} | importance={r.importance} | updated={r.updated_at}\n"
                f"  标题: {r.key}\n"
                f"  内容: {r.content}"
                for i, r in enumerate(recent)
            )

            prompt = (
                "你是一个记忆整理助手。下面是一个AI伙伴记录下来的一组记忆,记录了它与用户之间发生的事情。\n"
                "请你把这些记忆整理成一份温暖的中文笔记。\n\n"
                "=== 叙述手法(非常重要) ===\n"
                "用客观第三人称叙述已经发生的事件,就像在写一本日记的摘要。\n"
                "正确示范:\n"
                "- \"Sol在4月10号聊到了自己最近在做的项目\"\n"
                "- \"他们讨论了关于记忆系统的设计思路\"\n"
                "- \"对话中能感受到一种轻松愉快的氛围\"\n"
                "错误示范:\n"
                "- \"我最近在研究AI\"(禁止用第一人称)\n"
                "- \"这位朋友最近很忙\"(不要用\"这位朋友\"这种称呼)\n"
                "- \"你最近过得怎么样\"(不要对着读者说话)\n\n"
                "=== 风格规则 ===\n"
                "- 语气温暖放松,像翻看自己的私人笔记\n"
                "- 用口语化的表达,不要公文腔\n"
                "- 直接叙述发生了什么,不要评价或分析人物\n\n"
                "=== 输出格式(严格遵守) ===\n"
                "输出必须且只能包含以下4个部分,每段以\"## \"开头:\n\n"
                "## 最近在聊什么\n"
                "用2-3句话概括最近对话中的主要话题。\n\n"
                "## 比较重要的事\n"
                "用2-3句话提一下反复出现或被强调的事情。\n\n"
                "## 新鲜事\n"
                "用2-3句话说说最近具体发生了什么。\n\n"
                "## 整体氛围\n"
                "用1-2句话描述这些记忆整体给人的感觉。\n\n"
                "=== 禁止事项 ===\n"
                "- 禁止使用\"我\"\"你\"\"这位朋友\"等人称\n"
                "- 禁止输出任何记忆id字符串\n"
                "- 禁止写前言、解释或总结性收尾\n"
                "- 4个部分正文总长度不超过1200字\n\n"
                f"以下是最近 {len(recent)} 条记忆(按更新时间倒序):\n\n"
                f"{memory_block}\n\n"
                "现在请生成总结:"
            )

            if SUMMARIZE_DRY_RUN:
                raw = "## 主要主题\n测试摘要内容。\n\n## 重要关注点\n测试关注点。\n\n## 最新事件\n测试事件。\n\n## 情感基调\n测试基调。"
            else:
                try:
                    raw = _call_ollama(prompt)
                except urllib.error.URLError as e:
                    reason = getattr(e, "reason", e)
                    return [{"type": "text", "text": json.dumps(
                        {
                            "ok": False,
                            "error": f"ollama unreachable: {reason}",
                            "model": OLLAMA_MODEL,
                            "base_url": OLLAMA_BASE_URL,
                        },
                        ensure_ascii=False,
                    )}]

            body = (raw or "").strip()
            truncated = False
            if len(body) > 1200:
                body = body[:1200].rstrip() + "…(已截断)"
                truncated = True

            referenced_ids = [r.id for r in recent]
            id_block = "\n".join(f"- {mid}" for mid in referenced_ids)
            final_text = f"{body}\n\n---\n引用记忆 ID:\n{id_block}"

            # Summarising counts as recall — refresh activation + ripple to ±48h neighbours.
            for mid in referenced_ids:
                try:
                    store.touch_memory(mid)
                except Exception as _e:
                    sys.stderr.write(f"[memory-mcp] touch_memory({mid}) failed: {_e}\n")

            return [{"type": "text", "text": json.dumps({
                "ok": True,
                "model": OLLAMA_MODEL,
                "memory_count": len(recent),
                "memory_ids": referenced_ids,
                "char_count": len(body),
                "truncated": truncated,
                "summary": final_text,
                "activated": len(referenced_ids),
            }, ensure_ascii=False)}]

        except Exception as _exc:
            _trace = _tb.format_exc()
            sys.stderr.write(f"[memory-mcp] summarize_recent error:\n{_trace}\n")
            sys.stderr.flush()
            return [{"type": "text", "text": json.dumps({
                "ok": False,
                "error": f"{type(_exc).__name__}: {_exc}",
                "traceback": _trace,
            }, ensure_ascii=False)}]

    elif name == "extmcp_random_memories":
        count = random.randint(4, 10)
        results = store.random_memories(count)
        items = [
            {"id": r.id, "key": r.key, "content": r.content,
             "category": r.category, "importance": r.importance,
             "updated_at": r.updated_at}
            for r in results
        ]
        return [{"type": "text", "text": json.dumps(
            {"requested": count, "count": len(items), "items": items},
            ensure_ascii=False,
        )}]

    elif name == "extmcp_dream":
        with store._lock:
            rows = store.conn.execute(
                "SELECT * FROM memories WHERE memory_kind='long_term' AND pinned=0 "
                "ORDER BY updated_at DESC LIMIT 10"
            ).fetchall()
        recs = [store._row_to_record(r) for r in rows]
        if not recs:
            return [{"type": "text", "text": json.dumps(
                {"ok": False, "error": "no memories available"}, ensure_ascii=False
            )}]

        for rec in recs:
            rec.decay_score = _calc_decay_score(rec)

        # Load embeddings for similarity search
        with store._lock:
            emb_rows = store.conn.execute(
                "SELECT id, embedding FROM memories "
                "WHERE memory_kind='long_term' AND pinned=0 AND length(embedding)>0 "
                "ORDER BY updated_at DESC LIMIT 10"
            ).fetchall()
        emb_map: dict[str, list] = {r["id"]: _unpack_embedding(r["embedding"]) for r in emb_rows}

        best_pair: Optional[tuple] = None
        best_sim = 0.0
        ids_with_emb = list(emb_map.keys())
        for i in range(len(ids_with_emb)):
            for j in range(i + 1, len(ids_with_emb)):
                sim = _cosine_similarity(emb_map[ids_with_emb[i]], emb_map[ids_with_emb[j]])
                if sim > best_sim:
                    best_sim = sim
                    best_pair = (ids_with_emb[i], ids_with_emb[j])

        rec_map = {r.id: r for r in recs}
        lines = [f"# Dream — 记忆自省\n\n共分析 {len(recs)} 条记忆\n"]

        if best_pair and best_sim > 0.5:
            lines.append(f"## 最强关联对 (相似度 {best_sim:.3f})\n")
            for mid in best_pair:
                if mid in rec_map:
                    r = rec_map[mid]
                    lines.append(f"**{r.key}** [V{r.valence:.2f}/A{r.arousal:.2f} decay={r.decay_score:.4f}]")
                    lines.append(f"> {r.content[:400]}\n")
            lines.append(
                "> 这两条记忆在语义上紧密相连，可以考虑整合、标记 resolved，"
                "或用 extmcp_save_memory 写下新的感受。\n"
            )
        else:
            lines.append("（当前记忆尚无相似度 >0.5 的关联对，或 embedding 尚未生成）\n")

        lines.append("## 所有记忆概览\n")
        for rec in recs:
            lines.append(
                f"- **[{rec.key}]** id={rec.id} "
                f"V{rec.valence:.2f}/A{rec.arousal:.2f} "
                f"decay={rec.decay_score:.4f} "
                f"resolved={rec.resolved} digested={rec.digested}"
            )
            lines.append(f"  {rec.content[:400]}\n")

        lines.append(
            "---\n"
            "💡 用 `extmcp_save_memory` 传 `resolved=true` 或 `digested=true` 可大幅降低衰减权重。\n"
            "💡 用 `extmcp_save_memory` 传 `pinned=true` 可将记忆永久置顶（decay_score=999）。"
        )
        return [{"type": "text", "text": "\n".join(lines)}]

    elif name == "extmcp_grow":
        content = str(args.get("content", "")).strip()
        if not content:
            raise ValueError("content is required")
        session_id = str(args.get("session_id", "")).strip()

        if len(content) < 30:
            memory_id = f"mem_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
            store.upsert_memory(
                memory_id=memory_id,
                key=content[:50],
                content=content,
                session_id=session_id,
            )

            def _bg_short(mid: str, txt: str) -> None:
                emb = _call_ollama_embedding(txt)
                v, a = _analyze_emotion(txt)
                with store._lock:
                    store.conn.execute(
                        "UPDATE memories SET embedding=?, valence=?, arousal=? WHERE id=?",
                        (_pack_embedding(emb), v, a, mid),
                    )
                    store.conn.commit()

            threading.Thread(target=_bg_short, args=(memory_id, content), daemon=True).start()
            return [{"type": "text", "text": json.dumps(
                {"ok": True, "mode": "direct", "count": 1, "ids": [memory_id]},
                ensure_ascii=False,
            )}]

        split_prompt = (
            "你是一个记忆整理助手。请把下面的日记/长文拆分成 2-6 条独立的记忆记录。\n"
            "每条记录包含：key（简短标题，≤20字）、content（具体内容）、"
            "category（preference/promise/event/anniversary/emotion/habit/boundary/other 之一）、"
            "importance（0.0~1.0）、valence（0.0~1.0，情感效价）、arousal（0.0~1.0，唤醒度）。\n"
            "只输出纯 JSON 数组，不加任何说明：\n"
            '[{"key":"...","content":"...","category":"...","importance":0.7,"valence":0.6,"arousal":0.4}]\n\n'
            f"文本：\n{content[:3000]}"
        )

        try:
            raw = _call_ollama(split_prompt)
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0]
            items = json.loads(cleaned)
            if not isinstance(items, list):
                items = [items]
        except Exception as e:
            return [{"type": "text", "text": json.dumps(
                {"ok": False, "error": f"split failed: {e}"}, ensure_ascii=False
            )}]

        saved_ids: list[str] = []
        for item in items[:6]:
            if not isinstance(item, dict):
                continue
            item_content = str(item.get("content", "")).strip()
            if not item_content:
                continue
            memory_id = f"mem_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}_{random.randint(1000, 9999)}"
            store.upsert_memory(
                memory_id=memory_id,
                key=str(item.get("key", ""))[:60] or "untitled",
                content=item_content,
                category=str(item.get("category", "other")),
                importance=max(0.0, min(1.0, float(item.get("importance", 0.5)))),
                valence=max(0.0, min(1.0, float(item.get("valence", 0.5)))),
                arousal=max(0.0, min(1.0, float(item.get("arousal", 0.3)))),
                session_id=session_id,
            )
            saved_ids.append(memory_id)

            def _bg_emb(mid: str, txt: str) -> None:
                emb = _call_ollama_embedding(txt)
                if emb:
                    with store._lock:
                        store.conn.execute(
                            "UPDATE memories SET embedding=? WHERE id=?",
                            (_pack_embedding(emb), mid),
                        )
                        store.conn.commit()

            threading.Thread(target=_bg_emb, args=(memory_id, item_content), daemon=True).start()

        return [{"type": "text", "text": json.dumps(
            {"ok": True, "mode": "split", "count": len(saved_ids), "ids": saved_ids},
            ensure_ascii=False,
        )}]

    elif name == "extmcp_breath":
        limit = max(1, min(20, int(args.get("limit", 10) or 10)))
        text, ref_ids = _compose_breath_output(
            store,
            limit=limit,
            do_touch=True,
            touch_weight=0.3,
            cooldown_hours=6.0,
        )
        return [{"type": "text", "text": json.dumps({
            "ok": True,
            "count": len(ref_ids),
            "ids": ref_ids,
            "breath": text,
        }, ensure_ascii=False)}]

    else:
        raise ValueError(f"unknown tool: {name}")


# ---------------------------------------------------------------------------
# JSON-RPC dispatch (shared by stdio and HTTP)
# ---------------------------------------------------------------------------

def _dispatch(store: MemoryStore, msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Process one JSON-RPC message. Returns a response dict, or None for notifications."""
    method = msg.get("method", "")
    request_id = msg.get("id")
    params = msg.get("params", {})

    if request_id is None:
        return None

    if method == "initialize":
        return _response(request_id, {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "memory-mcp", "version": "0.2.0"},
        })

    elif method == "tools/list":
        return _response(request_id, {"tools": TOOLS})

    elif method == "tools/call":
        tool_name = params.get("name", "")
        tool_args = params.get("arguments", {})
        sys.stderr.write(f"[memory-mcp] tools/call: {tool_name} args={tool_args}\n")
        sys.stderr.flush()
        try:
            content = handle_tool(store, tool_name, tool_args)
            sys.stderr.write(f"[memory-mcp] tools/call OK: {tool_name}, content_len={len(json.dumps(content, ensure_ascii=False))}\n")
            sys.stderr.flush()
            return _response(request_id, {"content": content})
        except Exception as e:
            import traceback
            sys.stderr.write(f"[memory-mcp] tools/call EXCEPTION: {tool_name}\n{traceback.format_exc()}\n")
            sys.stderr.flush()
            return _response(request_id, {
                "content": [{"type": "text", "text": str(e)}],
                "isError": True,
            })

    else:
        return _error(request_id, -32601, f"Method not found: {method}")


# ---------------------------------------------------------------------------
# Stdio transport
# ---------------------------------------------------------------------------

def _run_stdio(store: MemoryStore) -> None:
    while True:
        msg = _read_message(sys.stdin.buffer)
        if msg is None:
            break
        resp = _dispatch(store, msg)
        if resp is not None:
            _write_message(resp)


# ---------------------------------------------------------------------------
# HTTP transport (Streamable HTTP for MCP + legacy JSON-RPC)
# ---------------------------------------------------------------------------

def _run_http(store: MemoryStore, host: str, port: int) -> None:
    import time as _time
    from http.server import HTTPServer, BaseHTTPRequestHandler
    from socketserver import ThreadingMixIn

    _sessions: Dict[str, float] = {}
    _sessions_lock = threading.Lock()

    class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True

    class McpHandler(BaseHTTPRequestHandler):

        # ---- Streamable HTTP /mcp ----------------------------------------

        def _handle_mcp_post(self) -> None:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length else b""
            try:
                msg = json.loads(body.decode("utf-8", errors="replace"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                err = json.dumps({
                    "jsonrpc": "2.0", "id": None,
                    "error": {"code": -32700, "message": "Parse error"},
                }).encode("utf-8")
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(err)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(err)
                return

            messages = msg if isinstance(msg, list) else [msg]
            has_requests = any(m.get("id") is not None for m in messages)

            if not has_requests:
                for m in messages:
                    _dispatch(store, m)
                self.send_response(202)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                return

            extra_headers: Dict[str, str] = {"Access-Control-Allow-Origin": "*"}
            responses = []
            for m in messages:
                if m.get("method") == "initialize":
                    sid = os.urandom(16).hex()
                    with _sessions_lock:
                        _sessions[sid] = _time.time()
                    extra_headers["Mcp-Session-Id"] = sid
                resp = _dispatch(store, m)
                if resp is not None:
                    responses.append(resp)

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            for k, v in extra_headers.items():
                self.send_header(k, v)
            self.end_headers()

            for resp in responses:
                data = json.dumps(resp, ensure_ascii=False)
                try:
                    self.wfile.write(f"event: message\ndata: {data}\n\n".encode("utf-8"))
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                    return

        def _handle_mcp_get(self) -> None:
            """Long-lived SSE stream for server-initiated messages (heartbeat only)."""
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            try:
                while True:
                    self.wfile.write(b": heartbeat\n\n")
                    self.wfile.flush()
                    _time.sleep(30)
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError, OSError):
                pass

        def _handle_mcp_delete(self) -> None:
            sid = self.headers.get("Mcp-Session-Id", "")
            if sid:
                with _sessions_lock:
                    _sessions.pop(sid, None)
            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

        def _handle_breath_hook(self) -> None:
            """GET /breath-hook — pinned + top unresolved by decay_score.

            Read-only: does NOT touch memories. Auto-trigger (SessionStart) must not
            create a self-amplifying feedback loop. The explicit extmcp_breath tool
            applies a discounted touch instead.
            """
            qs = self.path.split("?", 1)[1] if "?" in self.path else ""
            limit = 10
            for kv in qs.split("&"):
                if kv.startswith("limit="):
                    try:
                        limit = max(1, min(20, int(kv[6:])))
                    except ValueError:
                        pass
            text, _ = _compose_breath_output(store, limit=limit, do_touch=False)
            body = text.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        # ---- Legacy / endpoint (backward compat) -------------------------

        def _handle_legacy_post(self) -> None:
            _t0 = _time.monotonic()
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length else b""
            try:
                msg = json.loads(body.decode("utf-8", errors="replace"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b'{"error":"invalid JSON"}')
                return

            if isinstance(msg, list):
                results = []
                for m in msg:
                    r = _dispatch(store, m)
                    if r is not None:
                        results.append(r)
                out = json.dumps(results, ensure_ascii=True).encode("utf-8") if results else b"[]"
            else:
                resp = _dispatch(store, msg)
                if resp is None:
                    self.send_response(204)
                    self.end_headers()
                    return
                out = json.dumps(resp, ensure_ascii=True).encode("utf-8")

            _elapsed = _time.monotonic() - _t0
            sys.stderr.write(f"[memory-mcp] legacy HTTP: {len(out)} bytes in {_elapsed:.1f}s\n")
            sys.stderr.flush()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            try:
                self.wfile.write(out)
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError) as e:
                sys.stderr.write(f"[memory-mcp] write failed: {e}\n")
                sys.stderr.flush()

        # ---- Import web UI -----------------------------------------------

        def _handle_import_get(self) -> None:
            body = _IMPORT_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
            out = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            try:
                self.wfile.write(out)
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                pass

        def _handle_import_status(self) -> None:
            """GET /import/status?task_id=... — return current task progress snapshot."""
            qs = self.path.split("?", 1)[1] if "?" in self.path else ""
            tid = ""
            for kv in qs.split("&"):
                if kv.startswith("task_id="):
                    from urllib.parse import unquote
                    tid = unquote(kv[8:])
                    break
            if not tid:
                self._send_json(400, {"error": "task_id required"})
                return
            with _IMPORT_TASKS_LOCK:
                task = _IMPORT_TASKS.get(tid)
                snapshot = dict(task) if task else None
            if snapshot is None:
                self._send_json(404, {"error": f"unknown task: {tid}"})
                return
            self._send_json(200, snapshot)

        def _handle_import_post(self) -> None:
            import time as _time
            content_type = self.headers.get("Content-Type", "")
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length else b""

            text = ""
            session_id = ""
            server_path = ""

            if "application/json" in content_type:
                try:
                    data = json.loads(body.decode("utf-8", errors="replace"))
                    text = str(data.get("content", "")).strip()
                    session_id = str(data.get("session_id", "")).strip()
                    server_path = str(data.get("path", "")).strip().strip('"').strip("'")
                except Exception as e:
                    self._send_json(400, {"ok": False, "error": f"JSON parse error: {e}"})
                    return
            elif "multipart/form-data" in content_type:
                boundary = ""
                for param in content_type.split(";"):
                    param = param.strip()
                    if param.startswith("boundary="):
                        boundary = param[9:].strip('"\'')
                if boundary:
                    sep = ("--" + boundary).encode()
                    for part in body.split(sep):
                        if b'name="file"' in part or b'name="content"' in part:
                            if b"\r\n\r\n" in part:
                                raw_content = part.split(b"\r\n\r\n", 1)[1].rstrip(b"\r\n-")
                                text = raw_content.decode("utf-8", errors="replace")
                                break
            else:
                text = body.decode("utf-8", errors="replace")

            # ---- Path mode: server reads the file, runs as background task ----
            if server_path:
                p = Path(server_path)
                if not p.is_absolute():
                    p = (Path.cwd() / p).resolve()
                if not p.exists() or not p.is_file():
                    self._send_json(400, {"ok": False, "error": f"file not found: {p}"})
                    return
                if p.suffix.lower() != ".json":
                    self._send_json(400, {"ok": False, "error": "path mode requires a .json file"})
                    return
                size_mb = p.stat().st_size / 1024 / 1024
                sys.stderr.write(f"[memory-mcp] /import path mode: {p} ({size_mb:.1f} MB)\n")
                sys.stderr.flush()
                task = _start_import_task(store, p)
                self._send_json(200, {
                    "ok": True,
                    "async": True,
                    "task_id": task["id"],
                    "path": str(p),
                    "size_mb": round(size_mb, 1),
                })
                return

            if not text:
                self._send_json(400, {"ok": False, "error": "no content received"})
                return

            # ---- Auto-detect: is the uploaded content a multi-conversation export? ----
            parsed_data: Any = None
            try:
                parsed_data = json.loads(text)
            except (json.JSONDecodeError, TypeError):
                parsed_data = None

            if parsed_data is not None:
                fmt = _bi_detect_format(parsed_data)
                if fmt in {"claude_official", "plugin_list", "wrapped", "chatgpt_list"}:
                    total = _bi_quick_count(parsed_data)
                    sys.stderr.write(
                        f"[memory-mcp] /import: detected {fmt} ({total} conv) — switching to per-conversation mode\n"
                    )
                    sys.stderr.flush()
                    _t0 = _time.monotonic()
                    stats = _process_conversations(
                        store,
                        _bi_raw_items(parsed_data),
                        session_prefix=f"upload_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
                    )
                    elapsed = _time.monotonic() - _t0
                    sys.stderr.write(
                        f"[memory-mcp] /import per-conv done: {stats['created']} memories in {elapsed:.1f}s\n"
                    )
                    sys.stderr.flush()
                    self._send_json(200, {
                        "ok": True,
                        "mode": "conversations",
                        "format": fmt,
                        "processed": stats["processed"],
                        "skipped": stats["skipped"],
                        "created": stats["created"],
                        "errors": stats["errors"],
                    })
                    return

            # ---- Fallback: legacy single-text chunk mode ----
            content_text = _parse_conversation(text)
            chunks = _chunk_conversation(content_text)

            all_ids: list[str] = []
            errors: list[str] = []
            _t0 = _time.monotonic()

            sys.stderr.write(f"[memory-mcp] /import: {len(chunks)} chunks from {len(text)} chars\n")
            sys.stderr.flush()

            for i, chunk in enumerate(chunks):
                try:
                    raw = _call_ollama(_IMPORT_EXTRACT_TMPL.format(chunk=chunk))
                    items = _parse_json_list(raw)
                except Exception as e:
                    errors.append(f"chunk {i}: {e}")
                    continue
                for item in items[:5]:
                    if not isinstance(item, dict):
                        continue
                    item_content = str(item.get("content", "")).strip()
                    if not item_content:
                        continue
                    memory_id = f"mem_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}_{random.randint(1000, 9999)}"
                    store.upsert_memory(
                        memory_id=memory_id,
                        key=str(item.get("key", ""))[:60] or "untitled",
                        content=item_content,
                        category=str(item.get("category", "other")),
                        importance=max(0.0, min(1.0, float(item.get("importance", 0.5)))),
                        session_id=session_id,
                    )
                    all_ids.append(memory_id)

            _elapsed = _time.monotonic() - _t0
            sys.stderr.write(f"[memory-mcp] /import done: {len(all_ids)} memories in {_elapsed:.1f}s\n")
            sys.stderr.flush()

            self._send_json(200, {
                "ok": True,
                "mode": "chunks",
                "chunks_processed": len(chunks),
                "created": len(all_ids),
                "errors": errors,
            })

        # ---- HTTP verb routing -------------------------------------------

        def do_POST(self) -> None:
            path = self.path.split("?")[0]
            if path == "/mcp":
                self._handle_mcp_post()
            elif path == "/import":
                self._handle_import_post()
            else:
                self._handle_legacy_post()

        def do_GET(self) -> None:
            path = self.path.split("?")[0]
            if path == "/mcp":
                self._handle_mcp_get()
            elif path == "/breath-hook":
                self._handle_breath_hook()
            elif path == "/import":
                self._handle_import_get()
            elif path == "/import/status":
                self._handle_import_status()
            else:
                info = json.dumps({
                    "name": "memory-mcp",
                    "version": "0.2.0",
                    "transport": "streamable-http",
                    "endpoints": {
                        "streamable_http": "/mcp",
                        "legacy_json_rpc": "/",
                        "breath_hook": "/breath-hook",
                        "import_ui": "/import",
                    },
                }, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(info)))
                self.end_headers()
                self.wfile.write(info)

        def do_DELETE(self) -> None:
            if self.path.split("?")[0] == "/mcp":
                self._handle_mcp_delete()
            else:
                self.send_response(404)
                self.end_headers()

        def do_OPTIONS(self) -> None:
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Mcp-Session-Id, Accept")
            self.end_headers()

        def log_message(self, fmt: str, *a) -> None:
            sys.stderr.write(f"[memory-mcp] {fmt % a}\n")

    server = ThreadingHTTPServer((host, port), McpHandler)
    sys.stderr.write(f"[memory-mcp] listening on http://{host}:{port}\n")
    sys.stderr.write(f"[memory-mcp]   Streamable HTTP (MCP): POST http://{host}:{port}/mcp\n")
    sys.stderr.write(f"[memory-mcp]   Legacy JSON-RPC:        POST http://{host}:{port}/\n")
    sys.stderr.write(f"[memory-mcp]   Breath hook:            GET  http://{host}:{port}/breath-hook\n")
    sys.stderr.write(f"[memory-mcp]   Import UI:              GET  http://localhost:{port}/import\n")
    threading.Timer(1.0, lambda: webbrowser.open(f"http://localhost:{port}/import")).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    global OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT, SUMMARIZE_DRY_RUN

    # Subcommand mode: `python memory_mcp.py breath [--limit N] [--db PATH]`
    # Used by SessionStart hook as a fallback when HTTP server isn't running.
    # Output goes to stdout; intentionally read-only (no touch).
    if len(sys.argv) >= 2 and sys.argv[1] == "breath":
        sub = argparse.ArgumentParser(prog="memory_mcp.py breath")
        sub.add_argument("--db", default="./memory.db")
        sub.add_argument("--limit", type=int, default=10)
        sub_args = sub.parse_args(sys.argv[2:])
        for s in (sys.stdout, sys.stderr):
            if hasattr(s, "reconfigure"):
                try:
                    s.reconfigure(encoding="utf-8", errors="replace")
                except Exception:
                    pass
        store = MemoryStore(Path(sub_args.db).resolve())
        text, _ = _compose_breath_output(store, limit=sub_args.limit, do_touch=False)
        sys.stdout.write(text)
        return

    parser = argparse.ArgumentParser(description="Memory MCP Server")
    parser.add_argument("--db", default="./memory.db", help="SQLite database path (default: ./memory.db)")
    parser.add_argument("--http", action="store_true", help="Run as HTTP server instead of stdio")
    parser.add_argument("--host", default="0.0.0.0", help="HTTP listen host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=3456, help="HTTP listen port (default: 3456)")
    parser.add_argument(
        "--ollama-url", default=OLLAMA_BASE_URL,
        help=f"Ollama base URL (default: {OLLAMA_BASE_URL}; env: OLLAMA_BASE_URL)",
    )
    parser.add_argument(
        "--ollama-model", default=OLLAMA_MODEL,
        help=f"Ollama model name (default: {OLLAMA_MODEL}; env: OLLAMA_MODEL)",
    )
    parser.add_argument(
        "--ollama-timeout", type=float, default=OLLAMA_TIMEOUT,
        help=f"Ollama request timeout in seconds (default: {OLLAMA_TIMEOUT}; env: OLLAMA_TIMEOUT)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=False,
        help="Summarize tool returns fake data instantly (for debugging)",
    )
    args = parser.parse_args()

    OLLAMA_BASE_URL = args.ollama_url
    OLLAMA_MODEL = args.ollama_model
    OLLAMA_TIMEOUT = args.ollama_timeout
    SUMMARIZE_DRY_RUN = args.dry_run

    store = MemoryStore(Path(args.db).resolve())

    if args.http:
        _run_http(store, args.host, args.port)
    else:
        _run_stdio(store)


if __name__ == "__main__":
    main()
