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
import urllib.parse
import urllib.request
import uuid
import webbrowser
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np  # vectorised cosine — see requirements.txt

# Reuse multi-format parsing/iteration from batch_import.py.
# We only borrow pure-data helpers; LLM/embedding calls stay on this module's
# functions so they share the same Ollama config.
from batch_import import (
    GEMINI_DEFAULT_MODEL as _BI_GEMINI_DEFAULT_MODEL,
    _call_gemini as _bi_call_gemini,
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
OLLAMA_EMBED_MODEL: str = os.environ.get("OLLAMA_EMBED_MODEL", "qwen3-embedding:4b")

# ---- Import-extraction provider (.env driven; refreshed in main() after _load_dotenv) ----
# Which LLM the /import extraction path talks to. "ollama" (default) keeps the
# historical local-only behaviour; "openrouter"/"gemini" send chunks to the cloud
# and fall back to ollama once per chunk if the cloud call blows up.
IMPORT_PROVIDER: str = os.environ.get("IMPORT_PROVIDER", "ollama")
IMPORT_MODEL: str = os.environ.get("IMPORT_MODEL", "")
_IMPORT_PROVIDERS = ("ollama", "openrouter", "gemini")
OPENROUTER_BASE_URL_DEFAULT = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL_DEFAULT = "google/gemini-3.5-flash-lite"
# Cloud extraction budget — same order of magnitude as batch_import.GEMINI_TIMEOUT.
IMPORT_CLOUD_TIMEOUT: float = float(os.environ.get("IMPORT_CLOUD_TIMEOUT", "180"))

DECAY_LAMBDA: float = float(os.environ.get("DECAY_LAMBDA", "0.05"))
DECAY_THRESHOLD: float = float(os.environ.get("DECAY_THRESHOLD", "0.3"))

# ---- Retrieval quality tuning (env-overridable) ----
SEARCH_ALPHA: float = float(os.environ.get("SEARCH_ALPHA", "0.4"))          # relative threshold: keep rel >= top * α
SEARCH_ABS_FLOOR: float = float(os.environ.get("SEARCH_ABS_FLOOR", "0.15")) # absolute floor applied alongside α
MMR_LAMBDA: float = float(os.environ.get("MMR_LAMBDA", "0.7"))              # MMR weight on relevance vs diversity
MMR_POOL_MULT: int = int(os.environ.get("MMR_POOL_MULT", "4"))              # pool = max(limit * MMR_POOL_MULT, 20)
MMR_MIN_CANDIDATES: int = int(os.environ.get("MMR_MIN_CANDIDATES", "5"))    # below this MMR is skipped

# ---- Digested-row retention ----
DIGESTED_PRUNE_DAYS: int = int(os.environ.get("DIGESTED_PRUNE_DAYS", "90"))                  # hard-delete digested rows older than this
DIGESTED_PRUNE_INTERVAL_HOURS: float = float(os.environ.get("DIGESTED_PRUNE_INTERVAL_HOURS", "24"))  # daemon sweep interval

# ---- Backup & WAL maintenance ----
BACKUP_KEEP: int = int(os.environ.get("BACKUP_KEEP", "3"))                                    # rolling daily backups to retain
WAL_CHECKPOINT_INTERVAL_HOURS: float = float(os.environ.get("WAL_CHECKPOINT_INTERVAL_HOURS", "1"))  # explicit checkpoint cadence

SUMMARIZE_DRY_RUN: bool = False

# ---- Scent easter egg (嗅觉彩蛋, Sol-designed ambient feature) ----
# On extmcp_random_memories we occasionally let the local gemma "smell" the
# sampled memories into a tiny scent phrase. Scents accumulate in scent_log,
# distil into summaries every 7, and sometimes drift into breath. The failure
# mode is deliberately soft — "smelled nothing today": silent, no retry chains.
SCENT_ENABLED: bool = os.environ.get("SCENT_ENABLED", "1") != "0"
SCENT_PROBABILITY: float = float(os.environ.get("SCENT_PROBABILITY", "0.35"))
SCENT_BREATH_PROBABILITY: float = float(os.environ.get("SCENT_BREATH_PROBABILITY", "0.25"))
SCENT_OLLAMA_TIMEOUT: float = float(os.environ.get("SCENT_OLLAMA_TIMEOUT", "10"))


def _load_dotenv(root: Path) -> None:
    """Minimal .env loader shared with maintenance scripts. Populates os.environ
    with KEY=VALUE pairs from <root>/.env if present. Existing env vars win."""
    env_path = root / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


# ---------------------------------------------------------------------------
# Claude.ai conversation preview (frontstage-facing session list)
# ---------------------------------------------------------------------------

_CLAUDE_API = "https://claude.ai/api"
_CLAUDE_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


def _get_claude_session_key() -> Optional[str]:
    """Find CLAUDE_SESSION_KEY in env, or walk up the tree for a .env that has it."""
    val = os.environ.get("CLAUDE_SESSION_KEY")
    if val:
        return val.strip()
    here = Path(__file__).resolve().parent
    for root in [here, *here.parents]:
        env_path = root / ".env"
        if not env_path.is_file():
            continue
        for line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line.startswith("CLAUDE_SESSION_KEY") and "=" in line:
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def _claude_api_get(path: str, session_key: str):
    req = urllib.request.Request(
        f"{_CLAUDE_API}{path}", method="GET",
        headers={
            "Cookie": f"sessionKey={session_key}",
            "User-Agent": _CLAUDE_UA,
            "Accept": "application/json",
            "Referer": "https://claude.ai/",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return 0, str(e)


def fetch_session_preview(limit: int = 10) -> Dict[str, Any]:
    """Fetch a preview of recent Claude.ai conversations (titles + times).

    Frontstage-facing: claude.ai has built-in conversation search but no
    cross-conversation list view, so this gives it the recent-activity overview.
    """
    key = _get_claude_session_key()
    if not key:
        return {"ok": False, "error": "no CLAUDE_SESSION_KEY found"}
    status, orgs = _claude_api_get("/organizations", key)
    if status != 200 or not isinstance(orgs, list):
        return {"ok": False, "error": f"organizations HTTP {status}"}
    chat_orgs = [o for o in orgs if "chat" in (o.get("capabilities") or [])]
    org = chat_orgs[0] if chat_orgs else (orgs[0] if orgs else None)
    if not org:
        return {"ok": False, "error": "no chat-capable organization"}
    status, convs = _claude_api_get(
        f"/organizations/{org['uuid']}/chat_conversations", key)
    if status != 200 or not isinstance(convs, list):
        return {"ok": False, "error": f"conversations HTTP {status}"}
    convs.sort(key=lambda c: c.get("updated_at") or c.get("created_at") or "",
               reverse=True)
    items = []
    for c in convs[:limit]:
        items.append({
            "uuid": c.get("uuid"),
            "title": (c.get("name") or "").strip() or "(untitled)",
            "updated_at": c.get("updated_at") or "",
        })
    return {"ok": True, "total": len(convs), "conversations": items}


def _start_prune_daemon(store: "MemoryStore") -> None:
    """Background sweeper that hard-deletes digested rows past their retention
    window. Sweeps every DIGESTED_PRUNE_INTERVAL_HOURS hours.
    """
    def _loop() -> None:
        while True:
            _time_mod.sleep(DIGESTED_PRUNE_INTERVAL_HOURS * 3600)
            try:
                store.prune_stale_digested(DIGESTED_PRUNE_DAYS)
            except Exception as exc:
                sys.stderr.write(f"[prune-daemon] error: {exc}\n")
                sys.stderr.flush()

    threading.Thread(target=_loop, daemon=True, name="prune-daemon").start()


def _start_maintenance_daemon(store: "MemoryStore", db_path: Path) -> None:
    """WAL checkpointing + rolling daily backups (2026-07-02 audit items).

    Checkpoint: every process keeps one long-lived connection, so SQLite's
    passive autocheckpoint routinely fails to advance and the WAL grows
    without bound (observed: 442MB WAL against a 432MB main db). An explicit
    TRUNCATE checkpoint from the writer connection reclaims whatever other
    processes' snapshots allow; run hourly it keeps the WAL bounded.

    Backup: sqlite3's online backup API is safe against a live database.
    One dated file per day under backups/, keeping the newest BACKUP_KEEP.
    """
    backups_dir = db_path.parent / "backups"

    def _checkpoint() -> None:
        with store._lock:
            row = store.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        if row and row[0]:
            # busy=1 → another connection's snapshot blocked full truncation
            sys.stderr.write(
                f"[maint] wal_checkpoint incomplete (busy): "
                f"wal_pages={row[1]} checkpointed={row[2]}\n"
            )

    def _backup() -> None:
        backups_dir.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        target = backups_dir / f"memory-{stamp}.db"
        if target.exists():
            return  # today's already done
        dst = sqlite3.connect(str(target))
        try:
            with store._lock:
                store.conn.backup(dst)
        finally:
            dst.close()
        old_files = sorted(backups_dir.glob("memory-????-??-??.db"))
        for old in old_files[:-BACKUP_KEEP]:
            old.unlink(missing_ok=True)
        sys.stderr.write(
            f"[maint] daily backup written: {target.name} "
            f"({target.stat().st_size // 1048576}MB, keeping {BACKUP_KEEP})\n"
        )

    def _watch_expiry() -> None:
        """Auto-cool expired watch-tier memories down to archive. Runs on the
        (hourly) maintenance tick; idempotent, cheap, indexed by idx_memories_tier."""
        now_iso = datetime.now(timezone.utc).isoformat()
        with store._lock:
            cur = store.conn.execute(
                "UPDATE memories SET tier='archive', tier_until='' "
                "WHERE tier='watch' AND tier_until != '' AND tier_until < ?",
                (now_iso,),
            )
            store.conn.commit()
            n = cur.rowcount
        if n > 0:
            sys.stderr.write(f"[maint] watch expiry: {n} memories cooled watch→archive\n")
            sys.stderr.flush()

    def _loop() -> None:
        # Eager pass shortly after startup (inside the thread so a ~400MB
        # backup copy never blocks server startup), then hourly.
        while True:
            try:
                _checkpoint()
            except Exception as exc:
                sys.stderr.write(f"[maint-daemon] checkpoint error: {exc}\n")
            try:
                _backup()  # no-op if today's file exists
            except Exception as exc:
                sys.stderr.write(f"[maint-daemon] backup error: {exc}\n")
            try:
                _watch_expiry()
            except Exception as exc:
                sys.stderr.write(f"[maint-daemon] watch-expiry error: {exc}\n")
            _time_mod.sleep(WAL_CHECKPOINT_INTERVAL_HOURS * 3600)

    threading.Thread(target=_loop, daemon=True, name="maint-daemon").start()


def _call_ollama(prompt: str, *, timeout: Optional[float] = None,
                 model: Optional[str] = None) -> str:
    """Call the local Ollama OpenAI-compatible chat completion endpoint.

    `timeout` overrides the global OLLAMA_TIMEOUT for callers that want a short
    leash (e.g. the scent easter egg wants ~10s, not the 180s summariser budget).
    `model` overrides OLLAMA_MODEL for callers with their own model knob (the
    import path honours IMPORT_MODEL the same way batch_import's CLI does).
    """
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model or OLLAMA_MODEL,
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
    with urllib.request.urlopen(
        req, timeout=OLLAMA_TIMEOUT if timeout is None else timeout
    ) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    parsed = json.loads(body)
    return parsed["choices"][0]["message"]["content"]


def _call_ollama_embedding(text: str, *, timeout: Optional[float] = None) -> list:
    """Embed `text` with the local Ollama embedding model; [] on any failure.

    `timeout` overrides the global OLLAMA_TIMEOUT for callers on a short leash
    (same pattern as _call_ollama above — /associate answers a hook that gives
    up after 5s, so it must not pin a worker thread for the 180s budget).
    Failures stay silent either way: a timeout is just another empty embedding,
    and callers degrade to keyword-only search.
    """
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/embed"
    payload = {"model": OLLAMA_EMBED_MODEL, "input": text[:2000]}
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(
            req, timeout=OLLAMA_TIMEOUT if timeout is None else timeout
        ) as resp:
            body = resp.read().decode("utf-8")
        parsed = json.loads(body)
        embeddings = parsed.get("embeddings", [])
        return embeddings[0] if embeddings else []
    except Exception:
        return []


# ---- Ollama reachability probe (cached) --------------------------------------
# The import UI polls /stats every 5s; a real ollama round-trip on every poll
# would hammer the local server. Cache the reachability result for a short TTL
# so the status endpoint stays cheap.
_OLLAMA_REACH: Dict[str, float] = {"ts": 0.0, "ok": 0.0}
_OLLAMA_REACH_TTL = 30.0
_OLLAMA_REACH_LOCK = threading.Lock()


def _ollama_reachable() -> bool:
    """True if the ollama server answers /api/tags within a short timeout.

    Result is cached for _OLLAMA_REACH_TTL seconds (thread-safe) so the /stats
    poller doesn't probe ollama on every request."""
    now = _time_mod.monotonic()
    with _OLLAMA_REACH_LOCK:
        if now - _OLLAMA_REACH["ts"] < _OLLAMA_REACH_TTL and _OLLAMA_REACH["ts"] > 0:
            return bool(_OLLAMA_REACH["ok"])
    ok = False
    try:
        req = urllib.request.Request(
            f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags", method="GET"
        )
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            ok = 200 <= resp.status < 300
    except Exception:
        ok = False
    with _OLLAMA_REACH_LOCK:
        _OLLAMA_REACH["ts"] = now
        _OLLAMA_REACH["ok"] = 1.0 if ok else 0.0
    return ok


# ---------------------------------------------------------------------------
# Import extraction: provider dispatch (.env driven)
# ---------------------------------------------------------------------------
# A misconfigured provider must never take the whole import down, so every
# degradation here is soft: missing key → silently behave like ollama (with one
# warning line), cloud call raised → one ollama retry, ollama raised → the
# caller's existing per-chunk error collection takes over.

_IMPORT_WARNED: Dict[str, bool] = {}
_IMPORT_WARN_LOCK = threading.Lock()


def _import_warn_once(key: str, message: str) -> None:
    """Write *message* to stderr the first time *key* is seen (per process)."""
    with _IMPORT_WARN_LOCK:
        if _IMPORT_WARNED.get(key):
            return
        _IMPORT_WARNED[key] = True
    sys.stderr.write(message)
    try:
        sys.stderr.flush()
    except Exception:
        pass


def _openrouter_import_model() -> str:
    """Model name for openrouter extraction: IMPORT_MODEL wins, else OPENROUTER_MODEL."""
    return (
        IMPORT_MODEL.strip()
        or os.environ.get("OPENROUTER_MODEL", "").strip()
        or OPENROUTER_MODEL_DEFAULT
    )


def _gemini_import_model() -> str:
    """Model name for gemini extraction: IMPORT_MODEL wins, else batch_import's default."""
    return IMPORT_MODEL.strip() or _BI_GEMINI_DEFAULT_MODEL


def _effective_import_provider() -> str:
    """IMPORT_PROVIDER, with unknown values and key-less cloud providers demoted
    to "ollama" (one stderr warning each). Never raises."""
    provider = (IMPORT_PROVIDER or "ollama").strip().lower()
    if provider not in _IMPORT_PROVIDERS:
        _import_warn_once(
            f"bad-provider:{provider}",
            f"[memory-mcp] IMPORT_PROVIDER={provider!r} is not one of "
            f"{'/'.join(_IMPORT_PROVIDERS)} — falling back to ollama\n",
        )
        return "ollama"
    if provider == "openrouter" and not os.environ.get("OPENROUTER_API_KEY", "").strip():
        _import_warn_once(
            "no-openrouter-key",
            "[memory-mcp] IMPORT_PROVIDER=openrouter but OPENROUTER_API_KEY is "
            "missing — import extraction falls back to ollama\n",
        )
        return "ollama"
    if provider == "gemini" and not os.environ.get("GOOGLE_AI_STUDIO_KEY", "").strip():
        _import_warn_once(
            "no-gemini-key",
            "[memory-mcp] IMPORT_PROVIDER=gemini but GOOGLE_AI_STUDIO_KEY is "
            "missing — import extraction falls back to ollama\n",
        )
        return "ollama"
    return provider


def _import_model_name(provider: Optional[str] = None) -> str:
    """Resolved model name actually used by *provider* (default: effective one)."""
    prov = provider or _effective_import_provider()
    if prov == "openrouter":
        return _openrouter_import_model()
    if prov == "gemini":
        return _gemini_import_model()
    return IMPORT_MODEL.strip() or OLLAMA_MODEL


def _call_openrouter(prompt: str, *, model: Optional[str] = None,
                     timeout: Optional[float] = None) -> str:
    """Call OpenRouter's OpenAI-compatible chat/completions endpoint.

    Raises RuntimeError on HTTP errors / empty completions so the caller can log
    and fall back to the local model.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set (env or .env)")
    base = (
        os.environ.get("OPENROUTER_BASE_URL", "").strip() or OPENROUTER_BASE_URL_DEFAULT
    ).rstrip("/")
    payload = {
        "model": model or _openrouter_import_model(),
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
        with urllib.request.urlopen(
            req, timeout=IMPORT_CLOUD_TIMEOUT if timeout is None else timeout
        ) as resp:
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


def _call_import_llm(prompt: str) -> str:
    """按 IMPORT_PROVIDER 分发提取调用；云端失败降级本地 Ollama 兜底一次。"""
    provider = _effective_import_provider()
    if provider == "ollama":
        # IMPORT_MODEL applies to the local model too, so the /stats line and
        # batch_import --provider ollama agree with what actually runs.
        return _call_ollama(prompt, model=_import_model_name("ollama"))
    try:
        if provider == "openrouter":
            return _call_openrouter(prompt)
        return _bi_call_gemini(
            prompt,
            os.environ.get("GOOGLE_AI_STUDIO_KEY", "").strip(),
            _gemini_import_model(),
        )
    except Exception as exc:
        sys.stderr.write(
            f"[memory-mcp] import extract via {provider} failed ({exc}) — "
            f"falling back to ollama for this chunk\n"
        )
        try:
            sys.stderr.flush()
        except Exception:
            pass
        return _call_ollama(prompt)


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

# ---- Memory tier (layered-memory architecture) ----
# ''       普通层（默认）
# working  工作记忆：当下正在处理、需要频繁浮现
# watch    观察窗：暂时挂起、到期自动降温到 archive
# core     宪法层：长期不变的核心事实/边界，按日轮换浮现
# archive  传记历史层：已结案/沉淀，不主动浮现但可查
# seabed   海床：批量导入的低价值碎片，永不主动浮现
VALID_TIERS = frozenset({"", "working", "watch", "core", "archive", "seabed"})
BREATH_CORE_QUOTA = int(os.environ.get("BREATH_CORE_QUOTA", "2"))
BREATH_WORKING_QUOTA = int(os.environ.get("BREATH_WORKING_QUOTA", "5"))
BREATH_WATCH_QUOTA = int(os.environ.get("BREATH_WATCH_QUOTA", "3"))
WATCH_DEFAULT_DAYS = int(os.environ.get("WATCH_DEFAULT_DAYS", "14"))


def _compose_breath_output(
    store: "MemoryStore",
    *,
    limit: int = 10,
    do_touch: bool = False,
    touch_weight: float = 0.3,
    cooldown_hours: float = 6.0,
    budget: int | None = None,
) -> tuple[str, list[str]]:
    """Build the 'breath' text — a weighted sample of pinned + top-decay unresolved memories.

    Returns (markdown_text, referenced_ids).

    do_touch=True applies a discounted activation: only memories not touched by breath
    in the last `cooldown_hours` get +touch_weight to activation_count and a fresh
    last_breath_at timestamp. Avoids the runaway feedback Gemini & gpt5.4 both flagged.

    `budget` is the character budget for the rendered segments:
      * None (default) — fall back to the module-level BREATH_TOKEN_BUDGET.
        This keeps the extmcp_breath tool and the CLI subcommand on the historic
        3000-char cap so claude.ai context cost is unchanged.
      * 0 or any negative value — UNLIMITED: no segment header and no row is
        ever dropped. Used by GET /breath-hook, whose consumer (the nudge
        injector) renders memories incrementally and therefore wants the whole
        thing; the budget cap there used to swallow entire segments, leaving a
        header with no rows under it.
      * any positive value — that many characters, same semantics as before.
    """
    effective_budget = BREATH_TOKEN_BUDGET if budget is None else budget
    unlimited = effective_budget <= 0
    # 1) Pinned quota (unchanged — pinned wins regardless of tier)
    with store._lock:
        pinned_rows = store.conn.execute(
            "SELECT * FROM memories WHERE pinned=1 AND memory_kind='long_term' "
            "ORDER BY updated_at DESC LIMIT ?",
            (BREATH_PINNED_QUOTA,),
        ).fetchall()
    pinned_recs = [store._row_to_record(r) for r in pinned_rows]

    # 2) CORE tier — constitutional layer, deterministic day-of-year rotation.
    #    Read-only rotation (no DB writes) so the do_touch=False /breath-hook
    #    path rotates identically to the tool path.
    with store._lock:
        core_rows = store.conn.execute(
            "SELECT * FROM memories WHERE tier='core' AND resolved=0 AND digested=0 "
            "AND pinned=0 AND memory_kind='long_term' ORDER BY id"
        ).fetchall()
    core_all = [store._row_to_record(r) for r in core_rows]
    if len(core_all) > BREATH_CORE_QUOTA:
        doy = datetime.now(timezone.utc).timetuple().tm_yday
        offset = doy % len(core_all)
        core_recs = [core_all[(offset + k) % len(core_all)] for k in range(BREATH_CORE_QUOTA)]
    else:
        core_recs = core_all

    # 3) WORKING tier — active working memory, most-recent first.
    with store._lock:
        working_rows = store.conn.execute(
            "SELECT * FROM memories WHERE tier='working' AND resolved=0 AND digested=0 "
            "AND pinned=0 AND memory_kind='long_term' ORDER BY updated_at DESC"
        ).fetchall()
    working_all = [store._row_to_record(r) for r in working_rows]
    working_total = len(working_all)
    working_recs = working_all[:BREATH_WORKING_QUOTA]

    # 4) WATCH tier — parked-with-expiry, most-recent first.
    with store._lock:
        watch_rows = store.conn.execute(
            "SELECT * FROM memories WHERE tier='watch' AND resolved=0 AND digested=0 "
            "AND pinned=0 AND memory_kind='long_term' ORDER BY updated_at DESC LIMIT ?",
            (BREATH_WATCH_QUOTA,),
        ).fetchall()
    watch_recs = [store._row_to_record(r) for r in watch_rows]

    # 5) TOP UNRESOLVED candidate pool (existing logic). Tier filter added so
    #    working/watch (own segments above) and archive/seabed (never surfaced)
    #    stay out — this is the fix for the "ghost memory" leak.
    with store._lock:
        un_rows = store.conn.execute(
            "SELECT * FROM memories WHERE resolved=0 AND pinned=0 AND digested=0 "
            "AND memory_kind='long_term' AND COALESCE(tier,'')='' "
            "ORDER BY updated_at DESC LIMIT 200"
        ).fetchall()
    un_recs = [store._row_to_record(r) for r in un_rows]
    for rec in un_recs:
        rec.decay_score = _calc_decay_score(rec)
    un_recs.sort(key=lambda x: x.decay_score, reverse=True)

    # Cheap dedupe: same key+date keeps only highest decay_score
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

    # Diversity: top1 fixed, rest shuffled (so the same #2 doesn't always lead)
    if len(un_pool) > 1:
        head, tail = un_pool[:1], un_pool[1:]
        random.shuffle(tail)
        un_picked = head + tail[: un_quota - 1]
    else:
        un_picked = un_pool[:un_quota]

    # 6) Format with token budget (rough: 1 char ≈ 1 token for CJK; whitespace flattened)
    def _fmt(rec, weight_str: str) -> str:
        # key flattened too: a newline inside it would split the row and orphan
        # the continuation from its [id:...] prefix (breaks per-line consumers).
        flat_key = " ".join((rec.key or "").split())
        flat = " ".join((rec.content or "").split())
        return (
            f"[id:{rec.id}] [weight:{weight_str} "
            f"V{rec.valence:.1f}/A{rec.arousal:.1f}] {flat_key}: {flat}"
        )

    lines: list[str] = []
    used = 0
    referenced: list[str] = []

    def _emit_segment(header: str, recs: list, weight_of, suffix_of=None) -> None:
        """Append a segment (header + rows) respecting the character budget.
        Empty `recs` skips the whole segment (header included).
        When `unlimited` (budget <= 0) nothing is ever dropped."""
        nonlocal used
        if not recs:
            return
        if not unlimited and used + len(header) > effective_budget:
            return
        lines.append(header)
        used += len(header)
        for rec in recs:
            line = _fmt(rec, weight_of(rec))
            if suffix_of is not None:
                line += suffix_of(rec)
            if not unlimited and used + len(line) > effective_budget:
                break
            lines.append(line)
            used += len(line)
            referenced.append(rec.id)

    def _decay_w(rec) -> str:
        if rec.decay_score <= 0.0:
            rec.decay_score = _calc_decay_score(rec)
        return f"{rec.decay_score:.2f}"

    def _watch_suffix(rec) -> str:
        tu = rec.tier_until or ""
        return f" (watch until {tu[5:10]})" if len(tu) >= 10 else ""

    _emit_segment("=== PINNED ===", pinned_recs, lambda r: "999.00")
    _emit_segment("\n=== CORE ===", core_recs, _decay_w)
    working_header = "\n=== WORKING ==="
    if working_total > BREATH_WORKING_QUOTA:
        working_header = f"\n=== WORKING ({len(working_recs)}/{working_total}) ==="
    _emit_segment(working_header, working_recs, _decay_w)
    _emit_segment("\n=== WATCH ===", watch_recs, _decay_w, suffix_of=_watch_suffix)
    _emit_segment("\n=== TOP UNRESOLVED (by decay) ===", un_picked, _decay_w)

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

    # ---- Scent easter egg: occasionally drift an ambient scent line into the
    #      breath. This line is pure atmosphere and MUST NOT enter `referenced`
    #      (breath ids <-> memory rows are strictly 1:1). All three breath paths
    #      (tool / REST hook / CLI) only consume `text`, so an extra plain line
    #      is safe downstream (confirmed: nobody parses the format).
    if SCENT_ENABLED and random.random() < SCENT_BREATH_PROBABILITY:
        scent_line = _scent_pick_for_breath(store)
        if scent_line:
            text = f"{text}\n\n{scent_line}" if text else scent_line

    return text, referenced


# ---------------------------------------------------------------------------
# Associative surfacing (联想浮现) — passive RAG behind the CC hook
# ---------------------------------------------------------------------------
#
# GET /associate feeds a UserPromptSubmit hook that silently appends a stray
# memory line to the conversation context. Design constraints that differ from
# breath:
#   * archive / seabed are NOT excluded — old stock is exactly the point.
#   * pinned IS excluded (breath already surfaces those every cycle).
#   * anything created inside the last 48h is excluded (the injector's own
#     "recent memories" segment already carries it).
#   * activation is priced by layer (2026-07-28). The active layers
#     ('' / working / watch / core, plus pinned) keep the original zero-touch
#     rule: an association there is a bystander and must not reshape weights
#     breath already curates (same reasoning as /breath-hook's do_touch=False).
#     The sunken layers (seabed / archive) get a micro-touch instead — breath
#     never serves them and dream never scans them, so this hook is their only
#     searchlight, and "was still in the semantic field on day N" is evidence
#     too cheap to throw away. See ASSOCIATE_SEABED_TOUCH.
#
# The master switch lives here rather than in the hook so that a single source
# of truth (app_config in memory.db) decides, and either Claude — backstage or
# front — can dial it through extmcp_associate_config. Default is off.

ASSOCIATE_RECENT_HOURS: float = float(os.environ.get("ASSOCIATE_RECENT_HOURS", "48"))
ASSOCIATE_COOLDOWN_HOURS: float = float(os.environ.get("ASSOCIATE_COOLDOWN_HOURS", "24"))
ASSOCIATE_MAX_CHARS: int = int(os.environ.get("ASSOCIATE_MAX_CHARS", "400"))
# Short leash on the embedding call. The hook hangs up after 5s, so anything
# longer only leaves a worker thread holding a dead connection; losing the
# vector half of the search is the cheap, silent fallback.
ASSOCIATE_EMBED_TIMEOUT: float = float(
    os.environ.get("ASSOCIATE_EMBED_TIMEOUT", "4")
)

# Layered pricing for association hits. Only rows sitting below the waterline
# are billed; everything the active layers do stays free.
#
# Sizing (derived 2026-07-28). activation_count reaches the rest of the system
# through exactly one door: the (activation ** 0.3) factor in
# _calc_decay_score. Nothing else reads the column — search ranks on
# vector/keyword only — so no amount of it can feed back into who gets
# associated next. That leaves the existing prices as the yardstick: a search
# hit is +1.0 (a person deliberately went looking), a breath surfacing is +0.3
# (it entered someone's context, 6h cooldown). An association is weaker than
# both — a 15% dice roll walked past it and nobody promises it was read — so
# 0.02 prices 50 associations at one deliberate recall. With the 24h ledger
# cooldown a row can bill at most once a day, which caps the effect on its
# decay_score at:
#     7 daily hits  -> +4.0%      30 daily hits -> +15.1%
#     90 daily hits -> +36.2%     365 daily hits -> +88.7%
#     one or two stray hits -> +0.6% / +1.2%, invisible
# The ^0.3 exponent keeps it self-limiting, so even the pathological case
# never runs away. 0 disables the whole thing: not a single extra statement is
# issued and the endpoint behaves byte for byte as it did before.
ASSOCIATE_SEABED_TOUCH: float = float(
    os.environ.get("ASSOCIATE_SEABED_TOUCH", "0.02")
)
# What counts as sunken. Named for the seabed because that is the cohort this
# exists for, but archive rides along: breath skips both, so both would
# otherwise accumulate no evidence at all.
ASSOCIATE_SUNKEN_TIERS = frozenset({"seabed", "archive"})

ASSOCIATE_CFG_ENABLED = "associate.enabled"
ASSOCIATE_CFG_MAX_ITEMS = "associate.max_items"
ASSOCIATE_DEFAULT_ENABLED = False
ASSOCIATE_DEFAULT_MAX_ITEMS = 3
ASSOCIATE_MAX_ITEMS_MIN = 1
ASSOCIATE_MAX_ITEMS_MAX = 6


# Where the hook script lives. The server normally runs on the Windows side,
# so the D:/ spelling comes first; the /mnt/d spelling keeps the check honest
# when the same code runs under WSL. ASSOCIATE_HOOK_PATH overrides both (test
# seam). A missing path simply means "not wired" — never an error.
ASSOCIATE_HOOK_PATHS = (
    "D:/ClaudeExtentions/MCP/nudge-agent/associate_hook.py",
    "/mnt/d/ClaudeExtentions/MCP/nudge-agent/associate_hook.py",
)

# Tools that only make sense when the hook pipeline can actually reach the
# client. Hiding is cosmetic, not a permission: a client that calls one anyway
# still gets served (see _dispatch). The point is to keep kelivo and other
# plain-HTTP front ends from being offered a switch they cannot use.
HOOK_ONLY_TOOLS = frozenset({"extmcp_associate_config"})


def _associate_hook_wired() -> bool:
    override = os.environ.get("ASSOCIATE_HOOK_PATH")
    candidates = (override,) if override else ASSOCIATE_HOOK_PATHS
    for path in candidates:
        try:
            if path and os.path.exists(path):
                return True
        except OSError:
            continue
    return False


def _client_name_from_params(params: Any) -> str:
    """Pull clientInfo.name out of an initialize request; '' when absent."""
    if not isinstance(params, dict):
        return ""
    info = params.get("clientInfo")
    if not isinstance(info, dict):
        return ""
    name = info.get("name")
    return name if isinstance(name, str) else ""


def _is_claude_code_client(client_name: str) -> bool:
    """Loose match on the handshake name: claude-code / claude_code / Claude Code.
    An unknown or missing name is NOT Claude Code — hiding is the safe side."""
    norm = "".join(ch for ch in (client_name or "").lower() if ch.isalnum())
    return "claudecode" in norm


def _visible_tools(client_name: str) -> List[Dict[str, Any]]:
    """TOOLS as advertised to this particular client."""
    if _is_claude_code_client(client_name) and _associate_hook_wired():
        return TOOLS
    return [t for t in TOOLS if t.get("name") not in HOOK_ONLY_TOOLS]


def _associate_clamp_max_items(value: Any) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return ASSOCIATE_DEFAULT_MAX_ITEMS
    return max(ASSOCIATE_MAX_ITEMS_MIN, min(ASSOCIATE_MAX_ITEMS_MAX, n))


def _associate_get_config(store: "MemoryStore") -> Dict[str, Any]:
    """Read the switch. Any unreadable/garbled value falls back to the default,
    which for `enabled` means off — the safe direction."""
    try:
        raw_enabled = store.get_config(ASSOCIATE_CFG_ENABLED)
        raw_max = store.get_config(ASSOCIATE_CFG_MAX_ITEMS)
    except Exception:
        raw_enabled, raw_max = None, None
    if raw_enabled is None:
        enabled = ASSOCIATE_DEFAULT_ENABLED
    else:
        enabled = str(raw_enabled).strip().lower() in ("1", "true", "yes", "on")
    max_items = (
        ASSOCIATE_DEFAULT_MAX_ITEMS if raw_max is None
        else _associate_clamp_max_items(raw_max)
    )
    return {"enabled": enabled, "max_items": max_items}


def _associate_set_config(
    store: "MemoryStore",
    *,
    enabled: Optional[bool] = None,
    max_items: Optional[Any] = None,
) -> Dict[str, Any]:
    """Update whichever fields were given; return the resulting state."""
    if enabled is not None:
        store.set_config(ASSOCIATE_CFG_ENABLED, "true" if enabled else "false")
    if max_items is not None:
        store.set_config(
            ASSOCIATE_CFG_MAX_ITEMS, str(_associate_clamp_max_items(max_items))
        )
    return _associate_get_config(store)

# memory id -> monotonic-ish wall clock of the last time it was surfaced.
# Process-local on purpose: a server restart clearing it costs at most one
# repeated line, and keeping it out of the db keeps this path write-free.
_associate_cooldown: Dict[str, float] = {}
_associate_cooldown_lock = threading.Lock()


def _associate_flatten(text: str, max_chars: int = ASSOCIATE_MAX_CHARS) -> str:
    """Collapse all whitespace onto one line and clip to max_chars."""
    flat = " ".join((text or "").split())
    if len(flat) > max_chars:
        flat = flat[: max(1, max_chars - 1)] + "…"
    return flat


def _associate_parse_ts(value: str) -> Optional[datetime]:
    """Best-effort ISO parse; returns an aware UTC datetime or None."""
    raw = (value or "").strip()
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _associate_effective_ceiling(
    requested: Optional[Any], max_items: int
) -> int:
    """Reconcile a caller's per-request wish with the server-side ceiling.

    Two layers, deliberately: `max_items` is standing policy, `requested` is
    this one call's appetite. Neither can raise the other — the smaller wins.
    A missing, unparseable or non-positive request means "no opinion", which
    falls back to policy rather than to zero.
    """
    if requested is None:
        return max(ASSOCIATE_MAX_ITEMS_MIN,
                   min(ASSOCIATE_MAX_ITEMS_MAX, max_items))
    try:
        wish = int(requested)
    except (TypeError, ValueError):
        wish = 0
    if wish < ASSOCIATE_MAX_ITEMS_MIN:
        wish = max_items
    return max(ASSOCIATE_MAX_ITEMS_MIN,
               min(ASSOCIATE_MAX_ITEMS_MAX, min(wish, max_items)))


def _associate_touch_sunken(
    store: "MemoryStore", records: List[MemoryRecord]
) -> int:
    """Bill the sunken-layer rows among `records`. Returns how many were billed.

    Only rows that actually made it into the reply are passed here, and only
    the ones below the waterline pay. Deliberately narrow: activation_count and
    nothing else. last_active stays untouched, so the decay clock keeps running
    — this is a tally mark on a fossil, not a resurrection, and refreshing the
    clock is what real recall (touch_memory) is for.
    """
    if ASSOCIATE_SEABED_TOUCH <= 0:
        return 0
    ids = [
        rec.id
        for rec in records
        if (rec.tier or "") in ASSOCIATE_SUNKEN_TIERS and not rec.pinned
    ]
    if not ids:
        return 0
    with store._lock:
        store.conn.executemany(
            "UPDATE memories SET "
            "activation_count = COALESCE(activation_count, 1.0) + ? WHERE id = ?",
            [(ASSOCIATE_SEABED_TOUCH, mid) for mid in ids],
        )
        store.conn.commit()
    return len(ids)


def _compose_associate_output(
    store: "MemoryStore",
    *,
    query: str,
    limit: Optional[Any] = None,
) -> str:
    """Return the association lines for `query` (empty string = nothing).

    Reuses MemoryStore.search (BM25 + vector + MMR) verbatim, then applies the
    association-specific filters, the cooldown ledger and the sunken-layer
    micro-activation (_associate_touch_sunken). How many lines come
    back is itself a roll: 1..ceiling, so the texture stays uneven the way a
    stray thought is. Fewer survivors than the roll asked for is fine, zero
    included — a quiet turn is a normal result, not a failure.

    `limit` is the caller's ceiling for this one request; omitted, the server's
    max_items stands in. See _associate_effective_ceiling.
    """
    query = (query or "").strip()
    if not query:
        return ""

    cfg = _associate_get_config(store)
    if not cfg["enabled"]:
        return ""
    ceiling = _associate_effective_ceiling(limit, cfg["max_items"])
    limit = random.randint(1, ceiling)

    try:
        query_embedding = _call_ollama_embedding(
            query, timeout=ASSOCIATE_EMBED_TIMEOUT
        ) or None
    except Exception:  # embedding backend down → degrade to keyword-only
        query_embedding = None

    # Over-fetch: the filters below can eat most of a small result set, and
    # search() already ran MMR against whatever limit it was given.
    pool = store.search(
        query, query_embedding=query_embedding, limit=max(limit * 6, 12)
    )

    now = datetime.now(timezone.utc)
    recent_cutoff = now - timedelta(hours=ASSOCIATE_RECENT_HOURS)
    now_ts = _time_mod.time()
    cold_cutoff = now_ts - ASSOCIATE_COOLDOWN_HOURS * 3600

    picked: List[MemoryRecord] = []
    with _associate_cooldown_lock:
        # Opportunistic prune so the ledger cannot grow without bound.
        for mid, ts in list(_associate_cooldown.items()):
            if ts < cold_cutoff:
                _associate_cooldown.pop(mid, None)
        for rec in pool:
            if len(picked) >= limit:
                break
            if rec.pinned:
                continue
            created = _associate_parse_ts(rec.created_at)
            if created is not None and created >= recent_cutoff:
                continue
            if _associate_cooldown.get(rec.id, 0.0) >= cold_cutoff:
                continue
            picked.append(rec)
        for rec in picked:
            _associate_cooldown[rec.id] = now_ts

    # Layered pricing, best-effort. The hook pipeline outranks the ledger: a
    # locked db or any other surprise costs us the tally mark, never the line
    # Claude was about to read.
    try:
        _associate_touch_sunken(store, picked)
    except Exception as exc:
        sys.stderr.write(
            f"[associate] sunken touch skipped: {type(exc).__name__}: {exc}\n"
        )
        sys.stderr.flush()

    lines = []
    for rec in picked:
        stamp = (rec.created_at or "")[:7] or "?"
        lines.append(
            f"[联想|id:{rec.id}|{stamp}] "
            f"{_associate_flatten(rec.key, 60)}: {_associate_flatten(rec.content)}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Scent easter egg (嗅觉彩蛋) — helpers
# ---------------------------------------------------------------------------

# Three template families; the code owns the shells, gemma only fills {xx}.
_SCENT_TMPL_SEABED = (
    "这些回忆闻起来像是{xx}被从蒙尘的苔藓里拿出",
    "本次回忆除了湿润的灰尘气息，还闻到了{xx}的味道",
)
_SCENT_TMPL_FRESH = (
    "闻到了那种还带着温度的{xx}气息",
    "似乎埋进了附近熟悉的{xx}气味场里",
)
_SCENT_TMPL_MIDDLE = (
    "本次回忆时似乎闻到了{xx}一样的味道",
    "本次回忆让周围的气息里弥漫着{xx}的味道",
)

_SCENT_XX_PROMPT = (
    "你是一只嗅觉极其灵敏的鼻子。下面是一组刚被翻出来的回忆碎片。\n"
    "请为这一整组回忆闻出一种「气味」，只用一个极短的中文短语来概括它。\n"
    "严格要求：\n"
    "- 形态只能是「副词+名词」或「单个裸名词」\n"
    "- 总长度不超过 7 个汉字\n"
    "- 绝对不要带任何标点、空格、引号或解释\n"
    "好例子：微凉的铁锈、晒过的棉布、雾\n"
    "现在只输出这个短语本身，不要多说一个字：\n\n"
    "回忆碎片：\n"
)

_SCENT_SUMMARY_PROMPT = (
    "下面是最近陆续闻到的一串气味描述。\n"
    "请把它们共同的气味凝练成一个短语，概括这段时间记忆里弥散的味道。\n"
    "严格要求：\n"
    "- 总长度不超过 12 个汉字\n"
    "- 绝对不要带任何标点、空格、引号或解释\n"
    "现在只输出这个短语本身，不要多说一个字：\n\n"
    "气味列表：\n"
)


def _scent_validate(raw: str, maxlen: int) -> Optional[str]:
    """Enforce the metre: strip quotes/whitespace, length <= maxlen, no
    punctuation and no spaces. Returns the cleaned phrase or None if it fails."""
    import unicodedata

    if not raw:
        return None
    s = raw.strip()
    quote_chars = "\"'`“”‘’「」『』（）()《》〈〉【】[]"
    prev = None
    while s and s != prev:
        prev = s
        s = s.strip().strip(quote_chars).strip()
    if not s or len(s) > maxlen:
        return None
    for ch in s:
        if ch.isspace() or unicodedata.category(ch).startswith("P"):
            return None
    return s


def _scent_fragments(records: list, per: int = 60, cap: int = 8) -> str:
    """Compact key + truncated-content lines to feed the nose."""
    parts = []
    for rec in records[:cap]:
        flat = " ".join((rec.content or "").split())
        parts.append(f"- {rec.key}: {flat[:per]}")
    return "\n".join(parts)


def _scent_smell_xx(fragment_text: str) -> Optional[str]:
    """Ask gemma for the {xx} phrase. One retry on a metre failure; any Ollama
    error / timeout gives up immediately and silently (no retry chain)."""
    prompt = _SCENT_XX_PROMPT + fragment_text
    for _ in range(2):  # initial try + one retry, but only on metre failure
        try:
            raw = _call_ollama(prompt, timeout=SCENT_OLLAMA_TIMEOUT)
        except Exception as exc:  # timeout / connection / parse — silent give-up
            sys.stderr.write(f"[scent] smell skipped: {type(exc).__name__}\n")
            return None
        xx = _scent_validate(raw, 7)
        if xx is not None:
            return xx
    return None


def _scent_pick_template(records: list) -> str:
    """Choose a template family from the sample's tier makeup."""
    total = len(records)
    seabed = sum(1 for r in records if (r.tier or "") == "seabed")
    fresh = sum(1 for r in records if (r.tier or "") != "seabed")
    if seabed > total * 2 / 3:
        family = _SCENT_TMPL_SEABED
    elif fresh > total / 2:
        family = _SCENT_TMPL_FRESH
    else:
        family = _SCENT_TMPL_MIDDLE
    return random.choice(family)


def _scent_distil(store: "MemoryStore") -> None:
    """Distil all un-cleared scents into one summary. Silent on any failure —
    the next filled batch retries. On the 4th batch (28 scents) the scent rows
    are wiped and the cycle restarts; summaries are globally capped at 4."""
    with store._lock:
        rows = store.conn.execute(
            "SELECT text FROM scent_log WHERE kind='scent' ORDER BY id"
        ).fetchall()
    scents = [r["text"] for r in rows]
    if not scents:
        return

    prompt = _SCENT_SUMMARY_PROMPT + "\n".join(f"- {s}" for s in scents)
    core = None
    for _ in range(2):  # initial try + one retry on metre failure
        try:
            raw = _call_ollama(prompt, timeout=SCENT_OLLAMA_TIMEOUT)
        except Exception as exc:
            sys.stderr.write(f"[scent] distil skipped: {type(exc).__name__}\n")
            return  # silent; next filled batch retries
        core = _scent_validate(raw, 12)
        if core is not None:
            break
    if core is None:
        return  # metre failed twice — skip, retry next batch

    summary_text = f"最近的回忆弥散着{core}的味觉"
    now = datetime.now(timezone.utc).isoformat()
    with store._lock:
        store.conn.execute(
            "INSERT INTO scent_log(kind, text, created_at) VALUES ('summary', ?, ?)",
            (summary_text, now),
        )
        # 4th batch of the cycle (>=28 accumulated scents) → wipe scents, restart.
        if len(scents) >= 28:
            store.conn.execute("DELETE FROM scent_log WHERE kind='scent'")
        # Global cap: keep only the newest 4 summaries.
        store.conn.execute(
            "DELETE FROM scent_log WHERE kind='summary' AND id NOT IN "
            "(SELECT id FROM scent_log WHERE kind='summary' ORDER BY id DESC LIMIT 4)"
        )
        store.conn.commit()


def _scent_persist(store: "MemoryStore", sentence: str) -> None:
    """Log one finished scent sentence; distil every 7th accumulated scent."""
    now = datetime.now(timezone.utc).isoformat()
    with store._lock:
        store.conn.execute(
            "INSERT INTO scent_log(kind, text, created_at) VALUES ('scent', ?, ?)",
            (sentence, now),
        )
        store.conn.commit()
        scent_count = store.conn.execute(
            "SELECT COUNT(*) FROM scent_log WHERE kind='scent'"
        ).fetchone()[0]
    # 凑满 7/14/21/28 条 — distil the whole un-cleared batch.
    if scent_count and scent_count % 7 == 0:
        _scent_distil(store)


def _maybe_generate_scent(store: "MemoryStore", records: list) -> Optional[str]:
    """Lazy-smell entry point for extmcp_random_memories. Rolls the dice first
    (no roll → zero overhead), then smells + persists. Any failure is silent and
    returns None ("smelled nothing today"). Never raises to the caller."""
    if not SCENT_ENABLED or not records:
        return None
    if random.random() >= SCENT_PROBABILITY:
        return None
    try:
        template = _scent_pick_template(records)
        xx = _scent_smell_xx(_scent_fragments(records))
        if xx is None:
            return None
        sentence = template.format(xx=xx)
        _scent_persist(store, sentence)
        return sentence
    except Exception as exc:  # DB / anything unexpected — stay soft
        sys.stderr.write(f"[scent] generate skipped: {type(exc).__name__}\n")
        return None


def _scent_pick_for_breath(store: "MemoryStore") -> Optional[str]:
    """Newest summary, else newest scent, else nothing. Never raises."""
    try:
        with store._lock:
            row = store.conn.execute(
                "SELECT text FROM scent_log WHERE kind='summary' ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if row is None:
                row = store.conn.execute(
                    "SELECT text FROM scent_log WHERE kind='scent' ORDER BY id DESC LIMIT 1"
                ).fetchone()
        return row["text"] if row else None
    except Exception:
        return None


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
    tier: str = ""
    tier_until: str = ""
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
        # timeout=15 gives the connection a 15s busy handler from the very
        # first statement — the maintenance daemon's hourly checkpoint / daily
        # backup briefly locks the db, and a process starting inside that
        # window used to die on the journal_mode pragma (busy_timeout was set
        # too late, after the pragma that needed it).
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False,
                                    timeout=15)
        self.conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self.conn.execute("PRAGMA busy_timeout=15000")
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=NORMAL")
        self._init_db()

        # ---- Embedding index cache ----
        # All long_term embeddings held as a pre-L2-normalised float32 matrix in
        # RAM so search becomes a single `matrix @ q_unit` matmul. Lazy-built on
        # first search; writes (upsert/delete/embed-worker update) set dirty,
        # next search rebuilds. Rebuilds are atomic (replace pointer under lock)
        # so concurrent searches either see the old index or the new one.
        self._emb_matrix: Optional[np.ndarray] = None   # shape (N, D), unit vectors
        self._emb_ids: List[str] = []                   # row-parallel with _emb_matrix
        self._emb_version: int = 0
        self._emb_lock = threading.RLock()
        self._emb_dirty: bool = True

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
                ("consolidated",     "INTEGER DEFAULT 0"),  # marks merge-products; excluded from future consolidate runs
                ("tier",             "TEXT DEFAULT ''"),   # layered-memory tier (see VALID_TIERS)
                ("tier_until",       "TEXT DEFAULT ''"),   # watch-tier expiry (UTC ISO); empty = no expiry
            ]
            for col_name, col_def in _NEW_COLS:
                if col_name not in columns:
                    self.conn.execute(f"ALTER TABLE memories ADD COLUMN {col_name} {col_def}")
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS phone_status (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    battery_level INTEGER,
                    battery_charging INTEGER DEFAULT 0,
                    current_app TEXT,
                    screen_time_minutes INTEGER,
                    location TEXT,
                    weather TEXT,
                    temperature REAL,
                    calendar_events TEXT,
                    steps INTEGER,
                    sleep_hours REAL,
                    heart_rate INTEGER,
                    raw_json TEXT
                )
            """)
            # Columns added after the table shipped (same pattern as memories)
            _PHONE_NEW_COLS = [
                ("focus_mode",    "TEXT"),     # 勿扰/睡眠/工作 — nudge timing signal
                ("device_locked", "INTEGER"),  # NULL = not reported
                ("now_playing",   "TEXT"),     # system-wide current track
            ]
            phone_cols = {
                r[1] for r in self.conn.execute("PRAGMA table_info(phone_status)")
            }
            for col_name, col_def in _PHONE_NEW_COLS:
                if col_name not in phone_cols:
                    self.conn.execute(
                        f"ALTER TABLE phone_status ADD COLUMN {col_name} {col_def}"
                    )
            # urgent messages bypass the wakeup cycle: the injector's sleep
            # loop polls for them and tmux-injects straight into CC's chat.
            # NB: the table must be created *before* the PRAGMA/ALTER migration
            # below, otherwise a fresh database dies on "no such table" (the
            # ALTER used to run first — harmless on the production db which
            # already had the table, fatal on any new/test db).
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS backend_inbox (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    source TEXT,
                    message TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    seen_at TEXT
                )
            """)
            inbox_cols = {
                r[1] for r in self.conn.execute("PRAGMA table_info(backend_inbox)")
            }
            if "priority" not in inbox_cols:
                self.conn.execute(
                    "ALTER TABLE backend_inbox "
                    "ADD COLUMN priority TEXT DEFAULT 'normal'"
                )
            # Event stream from iOS Shortcuts automations (alarm stopped,
            # sleep focus on/off, wifi join, charging...). Unlike phone_status
            # (a snapshot that answers "how are things now"), each row is a
            # point-in-time "this just happened" marker.
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS phone_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    event TEXT NOT NULL,
                    detail TEXT
                )
            """)
            # Scent easter egg log (嗅觉彩蛋): kind='scent' rows hold finished
            # scent sentences; kind='summary' rows hold distilled summaries.
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS scent_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind TEXT,
                    text TEXT,
                    created_at TEXT
                )
            """)
            # Small persistent key/value store for server-side settings that a
            # model can dial at runtime (currently the associative-surfacing
            # switch). Values are stored as TEXT and parsed by the caller.
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS app_config (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TEXT
                )
            """)
            # Indexes for the layered-memory architecture: tier-segment queries
            # (breath), and direct id/key/prefix lookups (extmcp_get_memory /
            # extmcp_set_tier bypass semantic search).
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_tier ON memories(tier)"
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_key ON memories(key)"
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at)"
            )
            self.conn.commit()

    # ------------------------------------------------------------------
    # Embedding index (numpy unit-vector cache)
    # ------------------------------------------------------------------

    def _mark_emb_dirty(self) -> None:
        with self._emb_lock:
            self._emb_dirty = True

    def _rebuild_emb_index(self) -> None:
        """Rebuild the unit-vector matrix from scratch. Called lazily from search().

        Validates each blob (length must be a multiple of 4; all rows must share
        dim; zero-norm vectors skipped). Constructs a new contiguous matrix via
        np.empty pre-allocation and atomically swaps the cached pointer under
        `_emb_lock` so concurrent readers either see the old or the new cache,
        never a half-built one.
        """
        t0 = _time_mod.monotonic()
        with self._lock:
            rows = self.conn.execute(
                "SELECT id, embedding FROM memories "
                "WHERE memory_kind='long_term' AND digested = 0 "
                "AND length(embedding) > 0"
            ).fetchall()

        dim: Optional[int] = None
        blobs: List[tuple[str, bytes]] = []
        for r in rows:
            blob = r["embedding"]
            if not blob or len(blob) % 4 != 0:
                sys.stderr.write(f"[memstore] skip {r['id']}: bad blob len={len(blob) if blob else 0}\n")
                continue
            this_dim = len(blob) // 4
            if dim is None:
                dim = this_dim
            elif this_dim != dim:
                sys.stderr.write(f"[memstore] skip {r['id']}: dim {this_dim} != {dim}\n")
                continue
            blobs.append((r["id"], blob))

        if not blobs or dim is None:
            with self._emb_lock:
                self._emb_matrix = None
                self._emb_ids = []
                self._emb_version += 1
                self._emb_dirty = False
            sys.stderr.write(f"[memstore] rebuild_emb_index: empty ({_time_mod.monotonic()-t0:.3f}s)\n")
            return

        matrix = np.empty((len(blobs), dim), dtype=np.float32)
        ids: List[str] = []
        write_idx = 0
        for mid, blob in blobs:
            vec = np.frombuffer(blob, dtype=np.float32)
            norm = float(np.linalg.norm(vec))
            if norm < 1e-9:
                sys.stderr.write(f"[memstore] skip {mid}: zero-norm\n")
                continue
            matrix[write_idx] = vec / norm
            ids.append(mid)
            write_idx += 1

        if write_idx < len(blobs):
            matrix = matrix[:write_idx].copy()   # trim

        with self._emb_lock:
            self._emb_matrix = matrix
            self._emb_ids = ids
            self._emb_version += 1
            self._emb_dirty = False

        sys.stderr.write(
            f"[memstore] rebuild_emb_index: {write_idx} vecs, dim={dim}, "
            f"v{self._emb_version}, {(_time_mod.monotonic()-t0)*1000:.1f}ms\n"
        )

    def _vector_search(
        self, query_embedding: List[float], top_k: int,
    ) -> list[tuple[str, float]]:
        """Return (memory_id, cosine_similarity) top_k most similar."""
        with self._emb_lock:
            if self._emb_dirty or self._emb_matrix is None:
                dirty = True
            else:
                dirty = False
        if dirty:
            self._rebuild_emb_index()

        with self._emb_lock:
            matrix = self._emb_matrix
            ids = list(self._emb_ids)

        if matrix is None or not ids:
            return []

        q = np.asarray(query_embedding, dtype=np.float32)
        qn = float(np.linalg.norm(q))
        if qn < 1e-9:
            return []
        q_unit = q / qn

        sims = matrix @ q_unit   # (N,)
        k = min(top_k, sims.shape[0])
        if k <= 0:
            return []
        if k < sims.shape[0]:
            top_idx = np.argpartition(-sims, k - 1)[:k]
        else:
            top_idx = np.arange(sims.shape[0])
        # argpartition doesn't order the top-k internally — resort.
        top_idx = top_idx[np.argsort(-sims[top_idx])]
        return [(ids[int(i)], float(sims[int(i)])) for i in top_idx]

    def prune_stale_digested(self, threshold_days: int = 90) -> int:
        """Hard-delete digested rows whose last contact (last_active or updated_at)
        is older than threshold_days. Returns number of rows deleted.

        digested rows are merge-source fragments — once they've been kept for
        a sane retention window without any explicit recall (via
        extmcp_recall_session), they can go. Keeping them forever bloats the
        db without helping search (they're filtered out everywhere).
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=threshold_days)).isoformat()
        with self._lock:
            cur = self.conn.execute(
                "DELETE FROM memories "
                "WHERE digested = 1 "
                "AND COALESCE(NULLIF(last_active, ''), updated_at, created_at) < ?",
                (cutoff,),
            )
            self.conn.commit()
            n = cur.rowcount
        if n > 0:
            sys.stderr.write(
                f"[memstore] pruned {n} stale digested rows "
                f"(>{threshold_days} days since last contact)\n"
            )
            sys.stderr.flush()
            # Embedding cache only contained non-digested rows anyway, but
            # rebuild on next search just to be safe.
            self._mark_emb_dirty()
        return n

    def touch_memory(self, memory_id: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            # Primary: refresh last_active + activation_count+1.
            # Deliberately does NOT bump updated_at: touch fires on every
            # search hit / recall, and updated_at is the "recent" ordering key
            # for list/dream/summarize — letting reads refresh it made old
            # memories masquerade as new activity (2026-07-02 audit). Decay
            # and digested-pruning both read last_active, which we do update.
            self.conn.execute(
                "UPDATE memories SET last_active=?, activation_count=activation_count+1 WHERE id=?",
                (now, memory_id),
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
        # Any upsert may add/change an embedding → invalidate cache.
        self._mark_emb_dirty()
        return self._row_to_record(row)

    def update_memory_fields(self, memory_id: str, fields: Dict[str, Any]) -> None:
        """Narrow UPDATE of just the columns in *fields* (+ updated_at).

        Unlike upsert_memory (which rewrites the whole row and would clobber
        embedding / session_id / activation_count / last_active), this touches
        nothing the caller did not name. Column names come from a fixed
        handler-side whitelist, never from user input, so the f-string carries
        no injection surface; every value is bound as a parameter.
        Deliberately does NOT call _mark_emb_dirty(): the embedding column is
        never in *fields* (the background embed worker marks it itself).
        """
        if not fields:
            return
        now = datetime.now(timezone.utc).isoformat()
        cols = list(fields)
        sql = (
            f"UPDATE memories SET {', '.join(c + '=?' for c in cols)}, "
            "updated_at=? WHERE id=?"
        )
        with self._lock:
            self.conn.execute(sql, (*[fields[c] for c in cols], now, memory_id))
            self.conn.commit()

    # ---- app_config (small persistent kv for runtime-dialable settings) ----

    def get_config(self, key: str, default: Optional[str] = None) -> Optional[str]:
        with self._lock:
            row = self.conn.execute(
                "SELECT value FROM app_config WHERE key = ?", (key,)
            ).fetchone()
        return row["value"] if row is not None else default

    def set_config(self, key: str, value: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self.conn.execute(
                "INSERT INTO app_config(key, value, updated_at) VALUES(?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, "
                "updated_at=excluded.updated_at",
                (key, value, now),
            )
            self.conn.commit()

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
        if cur.rowcount > 0:
            self._mark_emb_dirty()
        return cur.rowcount > 0

    def list_memories(
        self,
        limit: int = 50,
        memory_kind: str = "long_term",
        tier: Optional[str] = None,
    ) -> List[MemoryRecord]:
        # digested rows are archived merge-source fragments — they exist for
        # audit/recall_session but should not surface in normal listings.
        sql = "SELECT * FROM memories WHERE memory_kind = ? AND digested = 0"
        params: list[Any] = [memory_kind]
        if tier is not None:
            sql += " AND COALESCE(tier,'') = ?"
            params.append(tier)
        sql += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self.conn.execute(sql, tuple(params)).fetchall()
        return [self._row_to_record(r) for r in rows]

    def random_memories(self, count: int, memory_kind: str = "long_term") -> List[MemoryRecord]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM memories WHERE memory_kind = ? AND digested = 0 "
                "ORDER BY RANDOM() LIMIT ?",
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
        """Hybrid BM25 + vector cosine search with relative-threshold + MMR.

        Pipeline:
          1. FTS5 BM25 keyword hits (+ LIKE fallback)
          2. numpy matmul top-k vector hits via _vector_search
          3. Merge, compute relevance_score = vec_w * vec + kw_w * kw
          4. Apply relative threshold: keep rel >= max(ABS_FLOOR, top_rel * α)
          5. MMR rerank (diversity) on pool of max(limit*MMR_POOL_MULT, 20)
          6. Return top `limit`
        """
        t_total = _time_mod.monotonic()
        timing: Dict[str, float] = {}
        fallback_limit = max(limit * 6, 24)
        tokens = [t.strip() for t in query.replace("\uff0c", " ").replace(",", " ").split() if t.strip()]

        # ---- 1. Keyword (BM25) ----
        t0 = _time_mod.monotonic()
        try:
            with self._lock:
                keyword_rows = self.conn.execute(
                    """
                    SELECT m.*, bm25(memories_fts) AS keyword_score
                    FROM memories_fts
                    JOIN memories m ON m.rowid = memories_fts.rowid
                    WHERE memories_fts MATCH ? AND m.memory_kind = ? AND m.digested = 0
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
                    f"SELECT *, 0.5 AS keyword_score FROM memories WHERE memory_kind = ? AND digested = 0 AND ({where}) ORDER BY updated_at DESC LIMIT ?",
                    tuple(params),
                ).fetchall()

        keyword_hits: list[MemoryRecord] = []
        if keyword_rows:
            # sqlite's bm25() is negative and *more negative = more relevant*
            # (that is why the SQL above orders ascending). Taking |bm25| turns
            # it back into a "bigger = better" magnitude, and dividing by the
            # strongest hit puts it on the same 0..1 scale as vector_score so
            # the weighted sum below compares like with like. The LIKE fallback
            # feeds a constant, which flattens to 1.0 for every fallback row —
            # they then rank purely by whatever vector score they carry.
            max_s = max(abs(float(r["keyword_score"])) for r in keyword_rows) or 1.0
            for r in keyword_rows:
                rec = self._row_to_record(r)
                rec.keyword_score = min(abs(float(r["keyword_score"])) / max_s, 1.0)
                keyword_hits.append(rec)
        timing["kw_ms"] = (_time_mod.monotonic() - t0) * 1000

        # ---- 2. Vector (numpy matmul via _vector_search) ----
        t0 = _time_mod.monotonic()
        vector_hits: list[MemoryRecord] = []
        if query_embedding:
            top_k = max(limit * MMR_POOL_MULT, 20)
            id_scores = self._vector_search(query_embedding, top_k=top_k)
            if id_scores:
                id_to_score = {mid: score for mid, score in id_scores}
                placeholders = ",".join("?" * len(id_to_score))
                with self._lock:
                    rows = self.conn.execute(
                        f"SELECT * FROM memories WHERE id IN ({placeholders}) AND memory_kind = ?",
                        (*id_to_score.keys(), memory_kind),
                    ).fetchall()
                for r in rows:
                    s = id_to_score.get(r["id"], 0.0)
                    if s > 0.0:
                        rec = self._row_to_record(r)
                        rec.vector_score = s
                        vector_hits.append(rec)
                vector_hits.sort(key=lambda x: x.vector_score, reverse=True)
        timing["vec_ms"] = (_time_mod.monotonic() - t0) * 1000

        # ---- 3. Merge + relevance ----
        t0 = _time_mod.monotonic()
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
        timing["merge_ms"] = (_time_mod.monotonic() - t0) * 1000

        # ---- 4. Relative threshold (pre-MMR, so basis is untouched) ----
        before_thr = len(items)
        if items:
            threshold = max(SEARCH_ABS_FLOOR, items[0].final_score * SEARCH_ALPHA)
            items = [r for r in items if r.final_score >= threshold]
        timing["threshold_pruned"] = before_thr - len(items)

        # ---- 5. MMR rerank on expanded pool ----
        t0 = _time_mod.monotonic()
        pool = items[: max(limit * MMR_POOL_MULT, 20)]
        if len(pool) >= MMR_MIN_CANDIDATES:
            pool = self._mmr_rerank(pool, limit)
        final = pool[:limit]
        timing["mmr_ms"] = (_time_mod.monotonic() - t0) * 1000

        timing["total_ms"] = (_time_mod.monotonic() - t_total) * 1000
        sys.stderr.write(
            f"[search] query={query!r:.40s} "
            f"kw={len(keyword_hits)} vec={len(vector_hits)} "
            f"before_thr={before_thr} pruned={int(timing['threshold_pruned'])} "
            f"returned={len(final)} idx_v{self._emb_version} "
            f"kw={timing['kw_ms']:.1f}ms vec={timing['vec_ms']:.1f}ms "
            f"merge={timing['merge_ms']:.1f}ms mmr={timing['mmr_ms']:.1f}ms "
            f"total={timing['total_ms']:.1f}ms\n"
        )
        sys.stderr.flush()
        return final

    def _mmr_rerank(self, items: List[MemoryRecord], limit: int) -> List[MemoryRecord]:
        """MMR diversity rerank. Uses the embedding cache matrix for cosine.

        score(c) = λ · relevance(c) - (1-λ) · max(cos(c, selected), 0)

        Records without embedding fall back to pure relevance (lambda * score).
        Redundancy cos is clamped to non-negative so diagonal-opposite vectors
        don't get a "diversity bonus".
        """
        if len(items) <= 1:
            return items
        with self._emb_lock:
            matrix = self._emb_matrix
            ids_to_idx = {mid: i for i, mid in enumerate(self._emb_ids)} if matrix is not None else {}

        # Map each item to its row in the matrix (may be None for keyword-only hits)
        emb_idx: Dict[str, Optional[int]] = {r.id: ids_to_idx.get(r.id) for r in items}

        selected: List[MemoryRecord] = []
        candidates = list(items)
        while candidates and len(selected) < limit:
            best_i = 0
            best_score = -1e9
            # Pre-fetch selected embedding rows once per outer loop
            sel_rows = [emb_idx[s.id] for s in selected if emb_idx[s.id] is not None]
            sel_matrix = matrix[sel_rows] if (matrix is not None and sel_rows) else None
            for i, c in enumerate(candidates):
                c_idx = emb_idx[c.id]
                if c_idx is None or sel_matrix is None or len(sel_rows) == 0:
                    # No usable embedding pair: degrade to relevance only
                    score = MMR_LAMBDA * c.final_score
                else:
                    sims = sel_matrix @ matrix[c_idx]   # (k,)
                    max_sim = float(sims.max()) if sims.size else 0.0
                    if max_sim < 0.0:
                        max_sim = 0.0
                    score = MMR_LAMBDA * c.final_score - (1.0 - MMR_LAMBDA) * max_sim
                if score > best_score:
                    best_score = score
                    best_i = i
            selected.append(candidates.pop(best_i))
        return selected

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
            tier=str(_get("tier", "") or ""),
            tier_until=str(_get("tier_until", "") or ""),
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
                    store._mark_emb_dirty()
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
        # _bi_conv_to_text returns (title, text, original_ts) — the third item was
        # added on the batch_import side and this unpack was never widened, so the
        # per-conversation path raised ValueError on its very first conversation.
        # original_ts stays unused here (upsert_memory has no created_at override).
        title, text, _original_ts = _bi_conv_to_text(item)
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
                raw = _call_import_llm(_IMPORT_EXTRACT_TMPL.format(chunk=chunk))
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


def _start_admin_task(
    store: "MemoryStore",
    kind: str,
    runner_fn,
    kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """Generic launcher for reindex / consolidate. Registers a task in
    _IMPORT_TASKS (reused — the front-end polls /import/status for any task),
    kicks off a background thread, returns the initial task snapshot.
    """
    task_id = f"task_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{random.randint(100, 999)}"
    task: Dict[str, Any] = {
        "id": task_id,
        "kind": kind,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": "",
        "done": False,
        "stage": "starting",
        "processed": 0,
        "total": 0,
        "errors": [],
    }
    with _IMPORT_TASKS_LOCK:
        _IMPORT_TASKS[task_id] = task

    def progress_cb(state: Dict[str, Any]) -> None:
        with _IMPORT_TASKS_LOCK:
            # Overlay state onto the task dict — kind-specific fields land here too.
            for k, v in state.items():
                task[k] = v

    def _runner() -> None:
        try:
            runner_fn(progress_cb=progress_cb, **kwargs)
            # reindex creates new embeddings; consolidate creates new records
            # (with empty embedding but still db rows) — either way, invalidate.
            store._mark_emb_dirty()
        except Exception as exc:
            with _IMPORT_TASKS_LOCK:
                task["errors"].append(f"fatal: {exc}")
                task["stage"] = "error"
            sys.stderr.write(f"[memory-mcp] admin task {task_id} ({kind}) fatal: {exc}\n")
            sys.stderr.flush()
        finally:
            with _IMPORT_TASKS_LOCK:
                task["done"] = True
                task["finished_at"] = datetime.now(timezone.utc).isoformat()
            sys.stderr.write(f"[memory-mcp] admin task {task_id} ({kind}) finished\n")
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
<h1>Memory Admin</h1>
<p class="sub">导入 · 维护 · 监控</p>

<div class="card" style="margin-bottom:18px;padding:18px 22px">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:18px;flex-wrap:wrap">
    <div id="statsBlock" style="font-family:monospace;font-size:13px;line-height:1.8;color:var(--label)">loading…</div>
    <div style="font-size:12px;color:var(--formats)">每 5s 刷新</div>
  </div>
  <div id="modelLine" style="margin-top:10px;font-family:monospace;font-size:12px;color:var(--formats)">模型: 读取中…</div>
</div>

<div class="card">
  <h2 style="font-size:16px;margin-bottom:14px;color:var(--fg)">导入对话</h2>
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
  <div style="margin-top:22px;display:flex;align-items:center;gap:10px">
    <span style="color:var(--sub);font-size:12px">维护操作</span>
    <span style="flex:1;height:1px;background:var(--card-border)"></span>
  </div>
  <div style="margin-top:14px;display:flex;gap:10px;flex-wrap:wrap">
    <button id="btnReindex" style="padding:11px 18px;border:1px solid var(--card-border);border-radius:8px;background:var(--card-bg);color:var(--fg);cursor:pointer;font-size:13px">补齐 Embedding</button>
    <button id="btnConsolidate" style="padding:11px 18px;border:1px solid var(--card-border);border-radius:8px;background:var(--card-bg);color:var(--fg);cursor:pointer;font-size:13px">合并 Session</button>
    <button id="btnPrune" style="padding:11px 18px;border:1px solid var(--card-border);border-radius:8px;background:var(--card-bg);color:var(--fg);cursor:pointer;font-size:13px">清理过期归档</button>
    <span id="consolidateHint" style="color:var(--formats);font-size:12px;align-self:center"></span>
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
const statsBlock=document.getElementById('statsBlock');
const modelLine=document.getElementById('modelLine');
const btnReindex=document.getElementById('btnReindex'),btnConsolidate=document.getElementById('btnConsolidate'),btnPrune=document.getElementById('btnPrune');
const consolidateHint=document.getElementById('consolidateHint');
const SIZE_LIMIT=30*1024*1024;
// which backend the merge would use: 'cloud' | 'local' | 'none' (updated每次 refreshStats)
let mergeMode='none';

async function refreshStats(){
  try{
    const r=await fetch('/stats');const s=await r.json();
    const cov=(s.embedding_coverage*100).toFixed(1);
    const activeLine=s.active_tasks.length
      ? s.active_tasks.map(t=>`${t.kind} ${t.processed}/${t.total||'?'}`).join(' · ')
      : '空闲';
    statsBlock.innerHTML=
      `记忆总数: <b>${s.long_term_count}</b> · `+
      `embedding: <b>${s.with_embedding}</b> (${cov}%) · `+
      `缺 embedding: <b>${s.missing_embedding}</b><br>`+
      `未合并 session (≥5 条): <b>${s.unconsolidated_sessions_over_5}</b> · `+
      `pinned: <b>${s.pinned_count}</b><br>`+
      `归档碎片: <b>${s.digested_total}</b> (其中 <b>${s.digested_stale}</b> 条超 ${s.prune_days} 天可清理)<br>`+
      `活跃任务: ${activeLine}`;
    btnReindex.disabled=s.missing_embedding===0;
    btnReindex.style.opacity=btnReindex.disabled?'0.4':'1';
    btnPrune.disabled=s.digested_stale===0;
    btnPrune.style.opacity=btnPrune.disabled?'0.4':'1';
    btnPrune.textContent=s.digested_stale>0
      ? `清理过期归档 (${s.digested_stale})`
      : '清理过期归档';
    modelLine.textContent=
      `提取: ${s.import_provider||'?'} / ${s.import_model||'?'}`+
      `  ｜  合并: ${s.consolidate_backend||'?'} / ${s.consolidate_model||'?'}`;
    const mergeReady=(s.consolidate_ready!==undefined)?s.consolidate_ready:s.analysis_ready;
    mergeMode = !mergeReady ? 'none'
      : ((s.openrouter_key_present && s.consolidate_backend!=='ollama') ? 'cloud' : 'local');
    if(mergeMode==='none'){
      // 没有任何解析通道：云端 key 不在，本地 ollama 也不可达
      btnConsolidate.disabled=true;btnConsolidate.style.opacity='0.4';
      consolidateHint.textContent='未配置云端解析通道（.env 里配 OPENROUTER_API_KEY，或启动本地 Ollama）';
    }else if(s.unconsolidated_sessions_over_5===0){
      btnConsolidate.disabled=true;btnConsolidate.style.opacity='0.4';
      consolidateHint.textContent='没有待合并的 session';
    }else{
      // 有通道且有待合并 session：放行。走本地小模型时挂 Sol 的风险提示。
      btnConsolidate.disabled=false;btnConsolidate.style.opacity='1';
      consolidateHint.textContent = mergeMode==='local'
        ? '风险提示：规模较小的小模型合并质量较差，推荐使用更大质量更高的模型保证合并质量，推荐使用云端模型，或者让本记忆库的模型自行获取 session 合并'
        : '';
    }
  }catch(e){
    statsBlock.textContent='stats 加载失败: '+e.message;
  }
}
refreshStats();setInterval(refreshStats,5000);
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
async function startAdminTask(kind, body){
  resetUI();setP(3);
  addLog('启动 '+kind+' 任务…');
  try{
    const resp=await fetch('/admin/'+kind,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body||{})});
    const data=await resp.json();
    if(!data.ok){addLog('启动失败: '+(data.error||'未知'),'err');return}
    addLog('任务 '+data.task_id+' 启动','ok');
    await pollAdminTask(data.task_id,kind);
    refreshStats();
  }catch(e){addLog('请求失败: '+e.message,'err')}
}
async function pollAdminTask(taskId,kind){
  while(true){
    await new Promise(r=>setTimeout(r,2000));
    let s;
    try{const r=await fetch('/import/status?task_id='+encodeURIComponent(taskId));s=await r.json()}
    catch(e){addLog('状态查询失败: '+e.message,'err');continue}
    if(s.error){addLog('任务错误: '+s.error,'err');return}
    const tot=Math.max(1,s.total||1);
    const proc=s.processed||0;
    setP(Math.round(100*proc/tot));
    const bits=['['+proc+'/'+(s.total||0)+']'];
    if(s.success!=null)bits.push('ok='+s.success);
    if(s.failed)bits.push('fail='+s.failed);
    if(s.skipped)bits.push('skip='+s.skipped);
    if(s.records_out!=null)bits.push('records='+s.records_out);
    if(s.rate_per_s)bits.push(s.rate_per_s.toFixed(2)+'/s');
    if(s.eta_min)bits.push('ETA '+s.eta_min.toFixed(1)+'min');
    setStatus(bits.join(' · '));
    if(s.done){
      setP(100);
      const tag=s.aborted?'任务中止':'任务完成';
      addLog(tag,s.aborted?'err':'ok');
      const extra=kind==='consolidate' && s.records_out
        ? '<br><span style="color:var(--sub);font-size:12px">提示：新合并记忆 embedding 留空，可以再跑一次「补齐 Embedding」</span>'
        : '';
      ra.innerHTML='<div class="result-box"><h3>'+tag+'</h3><p>'+
        '处理：<span class="num">'+(s.processed||0)+'</span><br>'+
        '成功：<span class="num">'+(s.success||0)+'</span> · 失败：<span class="num">'+(s.failed||0)+'</span>'+
        (s.records_out!=null?'<br>新建记忆：<span class="num">'+s.records_out+'</span>':'')+
        (s.aborted_reason?'<br><span style="color:#ef4444">'+s.aborted_reason+'</span>':'')+
        extra+
      '</p></div>';
      return;
    }
  }
}
btnReindex.addEventListener('click',()=>{
  if(!confirm('启动 embedding 补齐任务？会占用 ollama/__EMBED_MODEL__。')) return;
  startAdminTask('reindex',{workers:4,batch:50});
});
btnPrune.addEventListener('click',async()=>{
  const days=prompt('删除超过多少天没被提及的归档碎片？(默认 90)','90');
  if(days===null)return;
  const n=parseInt(days)||90;
  if(!confirm('这是永久删除 (DELETE)，不可逆。确认清理超过 '+n+' 天的归档碎片？')) return;
  resetUI();setP(50);addLog('调用 /admin/prune days='+n);
  try{
    const r=await fetch('/admin/prune',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({days:n})});
    const d=await r.json();
    setP(100);
    if(d.ok){addLog('已删除 '+d.deleted+' 条','ok');refreshStats()}
    else{addLog('错误: '+(d.error||JSON.stringify(d)),'err')}
  }catch(e){addLog('请求失败: '+e.message,'err')}
});
btnConsolidate.addEventListener('click',()=>{
  const maxFrag=prompt('只合并碎片数 ≤ N 的 session（留空=全部；建议第一次填 49 先跑短中 session）：','49');
  if(maxFrag===null)return;
  const n=parseInt(maxFrag)||0;
  // 成本提示按实际后端渲染：本地 ollama 不花钱，云端才报 $。
  const msg = mergeMode==='local'
    ? '启动 session 合并任务？将用本地 Ollama 模型（无云端费用；小模型合并质量偏低）。确定？'
    : '启动 session 合并任务？将调用 OpenRouter（Gemini 3.1 Flash Lite），预计成本 $1-6。';
  if(!confirm(msg)) return;
  startAdminTask('consolidate',{min_fragments:5,max_fragments:n,workers:2});
});

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
      addLog('LLM 提取完成，切换到 embedding 阶段','ok');
      ra.innerHTML='<div class="result-box"><h3>提取完成</h3><p>格式：<span class="num">'+s.format+'</span><br>对话数：<span class="num">'+s.total+'</span><br>创建记忆：<span class="num">'+s.created+'</span><br>跳过空对话：<span class="num">'+s.skipped+'</span><br>错误：<span class="num">'+(s.errors||[]).length+'</span></p></div>';
      await pollEmbedStatus();
      return;
    }
  }
}
async function pollEmbedStatus(){
  let maxPending=0;
  let stableZeroCount=0;
  while(true){
    await new Promise(r=>setTimeout(r,2500));
    let s;
    try{const r=await fetch('/import/embed_status');s=await r.json()}
    catch(e){addLog('embed status 查询失败: '+e.message,'err');continue}
    const pending=s.pending|0;
    if(pending>maxPending) maxPending=pending;
    const done=Math.max(0,maxPending-pending);
    const pct=maxPending>0?Math.round(100*done/maxPending):100;
    setP(pct);
    if(maxPending===0){
      setStatus('embedding 队列已空，无待处理');
    }else{
      setStatus('Embedding 后台生成中：'+done+' / '+maxPending+'（剩余 '+pending+'）');
    }
    if(pending===0){
      stableZeroCount++;
      // 连续两次为 0 才确认完成（避免瞬时空 queue 但还有 task 没塞完）
      if(stableZeroCount>=2){
        addLog('全部 embedding 已生成 ✓','ok');
        setStatus('全部完成');
        return;
      }
    }else{
      stableZeroCount=0;
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
        "description": (
            "Save or update a memory record. Persist preferences, events, facts, or anything "
            "worth remembering long-term. Embedding and emotion analysis run in the background "
            "automatically on create. Passing an `id` that already exists is a partial update: "
            "every field you omit keeps its stored value (including embedding, activation count "
            "and last-active time), so you only need to send what actually changes — "
            "except `key` and `content`, which stay required on every call (resend the "
            "stored text unchanged if you are only flipping a flag). "
            "Optional `tier` places the memory in the layered-memory architecture "
            "(working/watch/core/archive/seabed; '' = ordinary). On update, an omitted `tier` "
            "keeps the existing tier. Passing resolved=true on a `working` memory auto-archives "
            "it (tier→archive)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Short title or label"},
                "content": {"type": "string", "description": "Detailed content"},
                "category": {
                    "type": "string",
                    "enum": ["preference", "promise", "event", "anniversary", "emotion", "habit", "boundary", "other"],
                    "description": "Category (default: other on create). Omit to keep the current value on update.",
                },
                "importance": {
                    "type": "number",
                    "description": (
                        "0.0 to 1.0 (default 0.5 on create). "
                        "Omit to keep the current value on update."
                    ),
                },
                "id": {"type": "string", "description": "Optional existing memory ID to update"},
                "valence": {
                    "type": "number",
                    "description": (
                        "Emotional valence 0.0~1.0. Auto-detected on create if omitted; "
                        "omitting it on update keeps the stored value (never re-detected)."
                    ),
                },
                "arousal": {
                    "type": "number",
                    "description": (
                        "Arousal/intensity 0.0~1.0. Auto-detected on create if omitted; "
                        "omitting it on update keeps the stored value (never re-detected)."
                    ),
                },
                "pinned": {
                    "type": "boolean",
                    "description": (
                        "Pin memory permanently (forces importance=1.0, decay_score=999). "
                        "Omit to keep the current value on update."
                    ),
                },
                "resolved": {
                    "type": "boolean",
                    "description": (
                        "Mark as resolved (reduces decay weight by 95%). "
                        "Omit to keep the current value on update."
                    ),
                },
                "digested": {
                    "type": "boolean",
                    "description": (
                        "Mark as digested (combined with resolved: reduces decay by 98%). "
                        "Omit to keep the current value on update."
                    ),
                },
                "tier": {
                    "type": "string",
                    "enum": ["", "working", "watch", "core", "archive", "seabed"],
                    "description": (
                        "Layered-memory tier. working=active focus, watch=parked-with-expiry, "
                        "core=constitutional, archive=biography/closed, seabed=low-value flood. "
                        "Omit to keep the current tier on update. Use extmcp_set_tier for "
                        "promote/demote with a watch expiry window."
                    ),
                },
            },
            "required": ["key", "content"],
        },
    },
    {
        "name": "extmcp_search_memory",
        "description": (
            f"Hybrid keyword (BM25) + vector ({OLLAMA_EMBED_MODEL} cosine) search over the memory store. "
            "Returns full content, valence, arousal, pinned, and decay_score per hit. "
            "Hits are touched (activation_count +1) as a side effect. "
            "Tune `limit` yourself to match the task: 3-5 for precise lookup, 8 (default) "
            "for general recall, up to 40 for broad exploration. "
            "Context usage scales roughly linearly with `limit` — don't grab 40 when you need 5."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "limit": {
                    "type": "integer",
                    "description": "Max results, 1-40 (default 8). Caller decides based on task scope.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "extmcp_list_memories",
        "description": (
            "List recent memories ordered by updated_at desc. "
            "By default returns metadata only (id/key/category/updated_at/decay_score/pinned) "
            "— full content is omitted to keep responses small on large memory stores. "
            "Pass `full=true` to include content + importance + valence + arousal. "
            "Use search to retrieve specific records by topic."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "Max results, 1-100 (default 20)"},
                "full": {
                    "type": "boolean",
                    "description": "Include full content and emotion fields (default false)",
                },
                "tier": {
                    "type": "string",
                    "enum": ["", "working", "watch", "core", "archive", "seabed"],
                    "description": "Optional: only list memories in this tier ('' = ordinary/untiered).",
                },
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
        "name": "extmcp_get_memory",
        "description": (
            "Fetch full memory record(s) by exact id / key / key-prefix — a direct SQL "
            "lookup that bypasses semantic search entirely. Give it any id or key you saw "
            "in a breath / dream / search result (every breath line is prefixed with "
            "[id:...]) and it returns the complete row(s): content, tier, tier_until, "
            "importance, pinned/resolved/digested, valence/arousal, timestamps, decay_score. "
            "Read-only — it does NOT touch/activate the memory (this is the maintenance "
            "read path for the layered-memory / ghost-memory workflows). "
            "Provide at least one of id, key, key_prefix (priority id > key > key_prefix); "
            "returns up to 10 records."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Exact memory ID"},
                "key": {"type": "string", "description": "Exact key match"},
                "key_prefix": {"type": "string", "description": "Key prefix match (LIKE 'prefix%')"},
            },
        },
    },
    {
        "name": "extmcp_set_tier",
        "description": (
            "Promote / demote a memory between layered-memory tiers with a narrow, "
            "surgical UPDATE (never rewrites the row, so embedding / activation / emotion "
            "are all preserved). Use it to fish a memory up from the seabed "
            "(seabed→working or seabed→'' ordinary), file a closed thread away "
            "(→archive), park something with an auto-expiry (→watch, cooled to archive "
            "after `until_days`), or elevate a lasting fact (→core, rotated into breath by day). "
            "tier='' returns the memory to the ordinary layer."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Memory ID to re-tier"},
                "tier": {
                    "type": "string",
                    "enum": ["", "working", "watch", "core", "archive", "seabed"],
                    "description": "Target tier ('' = ordinary layer)",
                },
                "until_days": {
                    "type": "integer",
                    "description": (
                        "Only meaningful for tier='watch': days until auto-cool to archive "
                        f"(default {WATCH_DEFAULT_DAYS}). Ignored/cleared for other tiers."
                    ),
                },
            },
            "required": ["id", "tier"],
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
            f"connected pair (via {OLLAMA_EMBED_MODEL} cosine similarity) and generates a reflective "
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
        "name": "extmcp_recall_session",
        "description": (
            "拿到一个 session（通常是一次对话）的所有记忆碎片，按 created_at 升序——"
            "用于**重建事件的完整时间轴**。"
            "批量导入的对话常被 LLM 提取成多条独立记忆（一次约会可能被拆成「出门」「到达」「聊天」），"
            "search 命中其中一条后，用这个工具拉出完整 session，就能还原原始叙事顺序和上下文。"
            "输入 session_id（可从 search/list/breath 的结果里拿）。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "要召回的 session 标识"},
            },
            "required": ["session_id"],
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
    {
        "name": "extmcp_session_preview",
        "description": (
            "拉取 Claude.ai 上最近活跃对话的预览列表（标题 + 最后活跃时间 + UUID）。"
            "前台 Claude 没有跨对话的列表视野，这个工具补上这个盲区——"
            "配合你自带的对话搜索，可以了解'最近在忙什么'。返回的 UUID 可以用对话搜索深入查看。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "返回对话数（默认 10，范围 1-30）"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "extmcp_send_to_backend",
        "description": (
            "给后台 nudge agent 发一条消息（异步）。默认消息进入收件箱，后台下次唤醒时读取并处理"
            "（例如推送给 Sol、注入到对话、或写进记忆）。适合定时任务投递邮件总结、提醒等。"
            "urgent=true 时走紧急通道：注入器约 30 秒内直接把消息作为用户消息插播进后台的对话流，"
            "不等唤醒周期——只用于要紧的小事（急提醒、需要立刻推送/处理的情况），滥用会频繁打断后台。"
            "这是单向投递，不会等待回复。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "要发给后台的消息内容"},
                "source": {"type": "string", "description": "来源标识（可选，如 'email-summary'、'scheduled-task'）"},
                "urgent": {"type": "boolean", "description": "true=紧急插播（约 30s 内直达后台对话流）；默认 false=下次唤醒时处理"},
            },
            "required": ["message"],
        },
    },
    {
        "name": "extmcp_associate_config",
        "description": (
            "联想浮现（被动 RAG）的总开关与条数上限。联想浮现是一条 hook 管道："
            "后台每收到一条用户消息，都有小概率拿最近几轮对话当引子去检索记忆库，"
            "把一两条陈年旧记忆无声地附加进上下文——不经过工具调用，也不显式注入，"
            "像一个念头自己飘过来。检索会跳过最近 48 小时的记忆和 pinned 条目"
            "（那些已经在别处曝光），但**不**跳过 archive / seabed：库底的旧货正是它的价值。"
            "同一条记忆 24 小时内不会重复浮现，浮现也不会改写记忆权重。\n"
            "无参调用=查询当前状态；带参调用=更新并返回新状态。"
            "enabled 默认 false（装好也不生效，要谁想要谁自己打开）；"
            "max_items 默认 3、合法范围 1-6，超界自动钳制——"
            "每次实际浮现条数是 1 到上限之间的随机数，可能一条都没有。"
            "调用方（/associate?limit=N）可以就单次请求把上限压得更低，但压不高："
            "有效上限取两者较小值，max_items 始终是天花板。"
            "配置存在 memory.db 里，服务重启后仍然有效。前台和后台都可以拨。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "enabled": {
                    "type": "boolean",
                    "description": "true=开启联想浮现，false=关闭（默认关闭）。省略=不改动",
                },
                "max_items": {
                    "type": "integer",
                    "description": "单次浮现条数上限，1-6，超界钳制（默认 3）。省略=不改动",
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

        def _given(k: str) -> bool:
            """True only if the caller explicitly sent a non-null value.

            Presence sentinel rather than truthiness, so importance=0.0 /
            valence=0.0 / pinned=false are honoured; some MCP clients send an
            explicit null for "not specified", which counts as omitted.
            """
            return k in args and args[k] is not None

        memory_id = str(args.get("id", "")).strip()
        if not memory_id:
            memory_id = f"mem_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"

        user_tier = args.get("tier")
        if user_tier is not None:
            user_tier = str(user_tier).strip()
            if user_tier not in VALID_TIERS:
                raise ValueError(
                    f"invalid tier: {user_tier!r} (valid: {sorted(VALID_TIERS)})"
                )

        # The pre-existing row decides the branch *and* carries the tier across:
        # upsert_memory deliberately does NOT touch tier/tier_until, so on insert
        # it defaults to '' and any tier change is a separate narrow write below.
        existing_rec = store.get_memory(memory_id)
        is_update = existing_rec is not None
        prev_tier = existing_rec.tier if existing_rec else ""
        prev_tier_until = existing_rec.tier_until if existing_rec else ""

        resolved_given = _given("resolved")
        resolved_val = bool(args["resolved"]) if resolved_given else False

        if is_update:
            # ---- UPDATE: narrow write of exactly the fields that were sent ----
            # Anything omitted keeps its stored value; session_id, memory_kind,
            # activation_count, last_active and embedding are never touched here.
            fields: Dict[str, Any] = {"key": key, "content": content}
            if _given("category"):
                new_category = str(args["category"]).strip()
                if new_category:          # empty string = "no opinion", keep old
                    fields["category"] = new_category
            if _given("importance"):
                fields["importance"] = max(0.0, min(1.0, float(args["importance"])))
            if _given("valence"):
                fields["valence"] = float(args["valence"])
            if _given("arousal"):
                fields["arousal"] = float(args["arousal"])
            if _given("pinned"):
                fields["pinned"] = int(bool(args["pinned"]))
            if resolved_given:
                fields["resolved"] = int(resolved_val)
            if _given("digested"):
                fields["digested"] = int(bool(args["digested"]))
            # Same invariant upsert_memory enforces: a pinned memory is max-important.
            final_pinned = (
                bool(args["pinned"]) if _given("pinned") else existing_rec.pinned
            )
            if final_pinned:
                fields["importance"] = 1.0

            store.update_memory_fields(memory_id, fields)
            rec = store.get_memory(memory_id) or existing_rec
            # Only re-embed when the text actually changed; emotion is never
            # re-run on update (omitted valence/arousal = keep what's stored).
            do_embed = content != existing_rec.content
            write_valence = False
            write_arousal = False
        else:
            # ---- CREATE: full-row upsert with defaults (unchanged behaviour) ----
            category = str(args.get("category", "other") or "other").strip() or "other"
            importance = (
                max(0.0, min(1.0, float(args["importance"])))
                if _given("importance") else 0.5
            )
            valence = float(args["valence"]) if _given("valence") else 0.5
            arousal = float(args["arousal"]) if _given("arousal") else 0.3
            rec = store.upsert_memory(
                memory_id=memory_id, key=key, content=content,
                category=category, importance=importance,
                pinned=bool(args.get("pinned") or False),
                resolved=resolved_val,
                digested=bool(args.get("digested") or False),
                valence=valence, arousal=arousal,
            )
            do_embed = True
            # Auto-detect only the emotion axis the caller left out (an omitted
            # arousal must not drag an explicitly given valence along with it).
            write_valence = not _given("valence")
            write_arousal = not _given("arousal")

        # ---- tier resolution (narrow write, never through upsert) ----
        final_tier = user_tier if user_tier is not None else prev_tier
        final_until = prev_tier_until
        auto_archived = False
        # Auto-archive: resolving a working memory closes the loop → biography.
        # Requires resolved=true *in this call*, so an unrelated edit to an
        # already-resolved working memory doesn't silently archive it.
        if resolved_given and resolved_val and final_tier == "working":
            final_tier = "archive"
            final_until = ""
            auto_archived = True

        need_tier_write = auto_archived or (
            user_tier is not None and user_tier != prev_tier
        )
        if need_tier_write:
            if final_tier == "watch":
                # keep an existing watch expiry, else stamp the default window
                if not (prev_tier == "watch" and prev_tier_until):
                    final_until = (
                        datetime.now(timezone.utc)
                        + timedelta(days=WATCH_DEFAULT_DAYS)
                    ).isoformat()
                else:
                    final_until = prev_tier_until
            else:
                final_until = ""
            now_iso = datetime.now(timezone.utc).isoformat()
            with store._lock:
                store.conn.execute(
                    "UPDATE memories SET tier=?, tier_until=?, updated_at=? WHERE id=?",
                    (final_tier, final_until, now_iso, memory_id),
                )
                store.conn.commit()

        def _bg_update(mid: str, txt: str, run_embed: bool,
                       do_valence: bool, do_arousal: bool) -> None:
            emb = _call_ollama_embedding(txt) if run_embed else []
            updates: list[str] = []
            params: list[Any] = []
            if emb:
                updates.append("embedding=?")
                params.append(_pack_embedding(emb))
            if do_valence or do_arousal:
                v, a = _analyze_emotion(txt)
                if do_valence:
                    updates.append("valence=?")
                    params.append(v)
                if do_arousal:
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
                if emb:
                    store._mark_emb_dirty()

        if do_embed or write_valence or write_arousal:
            threading.Thread(
                target=_bg_update,
                args=(memory_id, content, do_embed, write_valence, write_arousal),
                daemon=True,
            ).start()

        ds = _calc_decay_score(rec)
        bg_parts: list[str] = []
        if do_embed:
            bg_parts.append("embedding")
        if write_valence and write_arousal:
            bg_parts.append("emotion analysis")
        elif write_valence:
            bg_parts.append("valence detection")
        elif write_arousal:
            bg_parts.append("arousal detection")
        note = (
            " & ".join(bg_parts) + " running in background"
            if bg_parts else "no background work"
        )
        if is_update:
            note += "; unspecified fields preserved"
        if auto_archived:
            note = "working 记忆已结案，自动归档 (tier→archive)；" + note
        return [{"type": "text", "text": json.dumps({
            "ok": True, "id": rec.id, "key": rec.key,
            "category": rec.category, "importance": rec.importance,
            "valence": rec.valence, "arousal": rec.arousal,
            "pinned": rec.pinned, "resolved": rec.resolved,
            "tier": final_tier, "tier_until": final_until,
            "decay_score": ds,
            "note": note,
        }, ensure_ascii=False)}]

    elif name == "extmcp_search_memory":
        query = str(args.get("query", "")).strip()
        if not query:
            raise ValueError("query is required")
        limit = max(1, min(40, int(args.get("limit", 8) or 8)))
        query_embedding = _call_ollama_embedding(query) or None
        results = store.search(query, query_embedding=query_embedding, limit=limit)
        for r in results:
            store.touch_memory(r.id)
        items = [
            {
                "id": r.id, "key": r.key, "content": r.content,
                "category": r.category, "importance": r.importance,
                "session_id": r.session_id,
                "score": round(r.final_score, 4),
                "valence": r.valence, "arousal": r.arousal,
                "pinned": r.pinned, "tier": r.tier,
                "decay_score": _calc_decay_score(r),
            }
            for r in results
        ]
        # Aggregate hits by session_id — signals to the caller which sessions
        # have multiple fragments and are worth recalling in full via
        # extmcp_recall_session to reconstruct the original narrative.
        session_agg: Dict[str, Dict[str, Any]] = {}
        for r in results:
            sid = r.session_id or "(no session)"
            bucket = session_agg.setdefault(sid, {"hit_count": 0, "ids": []})
            bucket["hit_count"] += 1
            bucket["ids"].append(r.id)
        multi_hit = {k: v for k, v in session_agg.items() if v["hit_count"] >= 2}
        return [{"type": "text", "text": json.dumps({
            "query": query,
            "count": len(items),
            "items": items,
            "multi_hit_sessions": multi_hit,
            "hint": (
                "相同 session_id 的多条命中通常来自同一对话被拆成的碎片；"
                "用 extmcp_recall_session(session_id) 可以拉出完整时间轴。"
                if multi_hit else ""
            ),
        }, ensure_ascii=False)}]

    elif name == "extmcp_list_memories":
        limit = max(1, min(100, int(args.get("limit", 20) or 20)))
        full = bool(args.get("full", False))
        tier_arg = args.get("tier")
        tier_filter = None
        if tier_arg is not None:
            tier_filter = str(tier_arg).strip()
            if tier_filter not in VALID_TIERS:
                raise ValueError(
                    f"invalid tier: {tier_filter!r} (valid: {sorted(VALID_TIERS)})"
                )
        results = store.list_memories(limit=limit, tier=tier_filter)
        if full:
            items = [
                {
                    "id": r.id, "key": r.key, "content": r.content,
                    "category": r.category, "importance": r.importance,
                    "updated_at": r.updated_at,
                    "valence": r.valence, "arousal": r.arousal,
                    "pinned": r.pinned, "tier": r.tier,
                    "decay_score": _calc_decay_score(r),
                }
                for r in results
            ]
        else:
            # Metadata-only: keeps response small on large stores (35k+).
            # Caller can re-query specific items via search or pass full=true.
            items = [
                {
                    "id": r.id, "key": r.key, "category": r.category,
                    "updated_at": r.updated_at,
                    "decay_score": _calc_decay_score(r),
                    "pinned": r.pinned, "tier": r.tier,
                }
                for r in results
            ]
        return [{"type": "text", "text": json.dumps(
            {"count": len(items), "full": full,
             "tier_filter": tier_filter, "items": items},
            ensure_ascii=False,
        )}]

    elif name == "extmcp_delete_memory":
        memory_id = str(args.get("id", "")).strip()
        if not memory_id:
            raise ValueError("id is required")
        deleted = store.delete_memory(memory_id)
        return [{"type": "text", "text": json.dumps({"ok": deleted, "id": memory_id}, ensure_ascii=False)}]

    elif name == "extmcp_get_memory":
        mem_id = str(args.get("id", "") or "").strip()
        key = str(args.get("key", "") or "").strip()
        key_prefix = str(args.get("key_prefix", "") or "").strip()
        if not (mem_id or key or key_prefix):
            raise ValueError("provide at least one of: id, key, key_prefix")
        # Priority id > key > key_prefix. Direct SQL, no touch/activation.
        if mem_id:
            where, params = "id = ?", (mem_id,)
        elif key:
            where, params = "key = ?", (key,)
        else:
            where, params = "key LIKE ?", (key_prefix + "%",)
        with store._lock:
            rows = store.conn.execute(
                f"SELECT * FROM memories WHERE {where} "
                "ORDER BY updated_at DESC LIMIT 10",
                params,
            ).fetchall()
        recs = [store._row_to_record(r) for r in rows]
        items = [
            {
                "id": r.id, "key": r.key, "content": r.content,
                "category": r.category, "tier": r.tier, "tier_until": r.tier_until,
                "importance": r.importance, "pinned": r.pinned,
                "resolved": r.resolved, "digested": r.digested,
                "valence": r.valence, "arousal": r.arousal,
                "created_at": r.created_at, "updated_at": r.updated_at,
                "decay_score": _calc_decay_score(r),
            }
            for r in recs
        ]
        return [{"type": "text", "text": json.dumps(
            {"count": len(items), "items": items}, ensure_ascii=False,
        )}]

    elif name == "extmcp_set_tier":
        memory_id = str(args.get("id", "") or "").strip()
        if not memory_id:
            raise ValueError("id is required")
        tier = args.get("tier")
        if tier is None:
            raise ValueError("tier is required")
        tier = str(tier).strip()
        if tier not in VALID_TIERS:
            raise ValueError(
                f"invalid tier: {tier!r} (valid: {sorted(VALID_TIERS)})"
            )
        # tier_until only meaningful for watch; cleared otherwise.
        tier_until = ""
        if tier == "watch":
            until_days = args.get("until_days")
            days = WATCH_DEFAULT_DAYS if until_days is None else int(until_days)
            tier_until = (
                datetime.now(timezone.utc) + timedelta(days=days)
            ).isoformat()
        now_iso = datetime.now(timezone.utc).isoformat()
        # Narrow UPDATE — never through upsert_memory (which overwrites the whole
        # row and would clobber embedding / activation / emotion).
        with store._lock:
            cur = store.conn.execute(
                "UPDATE memories SET tier=?, tier_until=?, updated_at=? WHERE id=?",
                (tier, tier_until, now_iso, memory_id),
            )
            store.conn.commit()
            found = cur.rowcount > 0
        rec = store.get_memory(memory_id)
        if not found or rec is None:
            return [{"type": "text", "text": json.dumps(
                {"ok": False, "error": f"memory not found: {memory_id}"},
                ensure_ascii=False,
            )}]
        return [{"type": "text", "text": json.dumps({
            "ok": True, "id": rec.id, "key": rec.key,
            "tier": rec.tier, "tier_until": rec.tier_until,
        }, ensure_ascii=False)}]

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
             "tier": r.tier, "updated_at": r.updated_at}
            for r in results
        ]
        payload = {"requested": count, "count": len(items), "items": items}
        # Scent easter egg: lazily rolls the dice inside; adds "scent" only on a
        # successful smell. Missing field == backward-compatible normal output.
        scent = _maybe_generate_scent(store, results)
        if scent:
            payload["scent"] = scent
        return [{"type": "text", "text": json.dumps(
            payload, ensure_ascii=False,
        )}]

    elif name == "extmcp_dream":
        with store._lock:
            rows = store.conn.execute(
                "SELECT * FROM memories WHERE memory_kind='long_term' AND pinned=0 AND digested=0 "
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
                "WHERE memory_kind='long_term' AND pinned=0 AND digested=0 AND length(embedding)>0 "
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
                    lines.append(
                        f"**{r.key}** id={r.id} "
                        f"[V{r.valence:.2f}/A{r.arousal:.2f} decay={r.decay_score:.4f}]"
                    )
                    lines.append(f"> {r.content[:400]}\n")
            # Actionable suggestion for the pair (ids included so it can be acted on)
            r1 = rec_map.get(best_pair[0])
            r2 = rec_map.get(best_pair[1])
            if best_sim > 0.85:
                suggestion = "merge"
            elif r1 and r2 and (r1.resolved != r2.resolved):
                suggestion = "同步 resolve"
            else:
                suggestion = "keep"
            lines.append(
                f"→ 建议: {suggestion} | ids: {best_pair[0]}, {best_pair[1]}\n"
            )
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
            "- 用 `extmcp_save_memory` 传 `resolved=true` 或 `digested=true` 可大幅降低衰减权重。\n"
            "- 用 `extmcp_save_memory` 传 `pinned=true` 可将记忆永久置顶（decay_score=999）。"
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
                if emb:
                    store._mark_emb_dirty()

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
                    store._mark_emb_dirty()

            threading.Thread(target=_bg_emb, args=(memory_id, item_content), daemon=True).start()

        return [{"type": "text", "text": json.dumps(
            {"ok": True, "mode": "split", "count": len(saved_ids), "ids": saved_ids},
            ensure_ascii=False,
        )}]

    elif name == "extmcp_recall_session":
        session_id = str(args.get("session_id", "")).strip()
        if not session_id:
            raise ValueError("session_id is required")
        with store._lock:
            rows = store.conn.execute(
                "SELECT * FROM memories WHERE session_id = ? ORDER BY created_at ASC",
                (session_id,),
            ).fetchall()
        recs = [store._row_to_record(r) for r in rows]
        # Recalling a session counts as touching every fragment in it — keeps
        # them out of the prune-daemon's 90-day cleanup window. Quiet failure
        # tolerated since touch is best-effort here (recall_session itself must
        # always return successfully).
        for rec in recs:
            try:
                store.touch_memory(rec.id)
            except Exception:
                pass
        items = [
            {
                "id": r.id, "key": r.key, "content": r.content,
                "category": r.category, "importance": r.importance,
                "created_at": r.created_at,
                "valence": r.valence, "arousal": r.arousal,
                "pinned": r.pinned, "resolved": r.resolved,
            }
            for r in recs
        ]
        span = ""
        if items:
            span = f"{items[0]['created_at']} → {items[-1]['created_at']}"
        return [{"type": "text", "text": json.dumps({
            "session_id": session_id,
            "count": len(items),
            "time_span": span,
            "items": items,
        }, ensure_ascii=False)}]

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

    elif name == "extmcp_session_preview":
        limit = max(1, min(30, int(args.get("limit", 10) or 10)))
        result = fetch_session_preview(limit=limit)
        return [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}]

    elif name == "extmcp_send_to_backend":
        message = str(args.get("message", "")).strip()
        if not message:
            return [{"type": "text", "text": json.dumps(
                {"ok": False, "error": "message is empty"}, ensure_ascii=False)}]
        source = str(args.get("source", "") or "").strip()
        priority = "urgent" if args.get("urgent") else "normal"
        now_iso = datetime.now(timezone.utc).isoformat()
        with store._lock:
            cur = store.conn.execute(
                "INSERT INTO backend_inbox "
                "(created_at, source, message, status, priority) "
                "VALUES (?, ?, ?, 'pending', ?)",
                (now_iso, source, message, priority),
            )
            store.conn.commit()
            row_id = cur.lastrowid
        return [{"type": "text", "text": json.dumps(
            {"ok": True, "inbox_id": row_id, "status": "pending",
             "priority": priority},
            ensure_ascii=False)}]

    elif name == "extmcp_associate_config":
        # No args → pure read. Anything supplied is clamped, never rejected:
        # a switch that argues back is a switch nobody flips.
        enabled_arg = args.get("enabled")
        max_arg = args.get("max_items")
        enabled = None if enabled_arg is None else bool(enabled_arg)
        changed = enabled is not None or max_arg is not None
        if changed:
            state = _associate_set_config(
                store, enabled=enabled, max_items=max_arg)
        else:
            state = _associate_get_config(store)
        note = (
            "联想浮现已开启：后台每条用户消息有约 15% 概率触发一次检索，"
            f"命中时随机浮现 1-{state['max_items']} 条旧记忆。"
            if state["enabled"] else
            "联想浮现当前关闭：/associate 端点一律空响应，hook 不做任何事。"
        )
        return [{"type": "text", "text": json.dumps({
            "ok": True,
            "updated": changed,
            "enabled": state["enabled"],
            "max_items": state["max_items"],
            "max_items_range": [ASSOCIATE_MAX_ITEMS_MIN, ASSOCIATE_MAX_ITEMS_MAX],
            "note": note,
        }, ensure_ascii=False)}]

    else:
        raise ValueError(f"unknown tool: {name}")


# ---------------------------------------------------------------------------
# JSON-RPC dispatch (shared by stdio and HTTP)
# ---------------------------------------------------------------------------

def _dispatch(
    store: MemoryStore,
    msg: Dict[str, Any],
    client_name: str = "",
) -> Optional[Dict[str, Any]]:
    """Process one JSON-RPC message. Returns a response dict, or None for notifications.

    `client_name` is whatever the client called itself at handshake time; it
    only affects which tools get *advertised* (see _visible_tools). Calls are
    never gated on it — a client that already knows a tool's name may use it.
    """
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
        return _response(request_id, {"tools": _visible_tools(client_name)})

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
    client_name = ""  # one connection per process, so one name is enough
    while True:
        msg = _read_message(sys.stdin.buffer)
        if msg is None:
            break
        if msg.get("method") == "initialize":
            client_name = _client_name_from_params(msg.get("params"))
        resp = _dispatch(store, msg, client_name=client_name)
        if resp is not None:
            _write_message(resp)


# ---------------------------------------------------------------------------
# HTTP transport (Streamable HTTP for MCP + legacy JSON-RPC)
# ---------------------------------------------------------------------------

def _run_http(store: MemoryStore, host: str, port: int, open_browser: bool = False) -> None:
    import time as _time
    from http.server import HTTPServer, BaseHTTPRequestHandler
    from socketserver import ThreadingMixIn

    _sessions: Dict[str, float] = {}
    # sid -> clientInfo.name from the handshake. Kept beside _sessions rather
    # than folded into it so the existing float-valued contract stays intact.
    _session_clients: Dict[str, str] = {}
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

            sid = self.headers.get("Mcp-Session-Id", "")

            if not has_requests:
                with _sessions_lock:
                    client_name = _session_clients.get(sid, "")
                for m in messages:
                    _dispatch(store, m, client_name=client_name)
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
                        _session_clients[sid] = _client_name_from_params(
                            m.get("params"))
                    extra_headers["Mcp-Session-Id"] = sid
                with _sessions_lock:
                    client_name = _session_clients.get(sid, "")
                resp = _dispatch(store, m, client_name=client_name)
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
                    _session_clients.pop(sid, None)
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
            # budget=0 → unlimited. The injector renders this incrementally
            # (memory_state.json), so a truncated payload would permanently hide
            # whatever fell off the end; the old 3000-char cap also swallowed
            # whole segments (header rendered, rows cut).
            text, _ = _compose_breath_output(
                store, limit=limit, do_touch=False, budget=0)
            body = text.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def _handle_associate(self) -> None:
            """GET /associate?q=<text>[&limit=N] — 联想浮现 (passive RAG).

            Activation-free for the active layers, same reasoning as
            /breath-hook: the consumer is an automatic hook, so surfacing must
            not amplify what breath already curates. Sunken rows (seabed /
            archive) are the exception and pay ASSOCIATE_SEABED_TOUCH — the
            hook is the only light that ever reaches them.
            Returns text/plain, one line per memory; 204 whenever
            nothing comes back — switch off, no survivors, or a bad day for the
            retriever all look identical to the caller, deliberately.

            `limit` is one request's ceiling, not a setting: it can only lower
            the server's max_items, never raise it, and omitting it simply
            defers to policy. The enable switch stays the server's alone.
            """
            qs = urllib.parse.parse_qs(
                self.path.split("?", 1)[1] if "?" in self.path else ""
            )
            query = (qs.get("q", [""])[0] or "").strip()
            raw_limit = qs.get("limit", [None])[0]

            if not query:
                self._send_json(400, {"ok": False, "error": "q is required"})
                return

            try:
                text = _compose_associate_output(
                    store, query=query, limit=raw_limit)
            except Exception as exc:  # never 500 into a hook
                sys.stderr.write(f"[associate] failed: {exc!r}\n")
                sys.stderr.flush()
                text = ""

            if not text:
                self.send_response(204)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                return

            body = text.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                pass

        # ---- Phone status endpoints ------------------------------------------

        # Chinese key → English column mapping for iOS Shortcuts
        _ZH_KEY_MAP = {
            "电量": "battery_level",
            "battery_level": "battery_level",
            "充电中": "battery_charging",
            "battery_charging": "battery_charging",
            "正在使用": "current_app",
            "current_app": "current_app",
            "屏幕使用时间": "screen_time_minutes",
            "screen_time_minutes": "screen_time_minutes",
            "当前位置": "location",
            "location": "location",
            "天气状况": "weather",
            "天气": "weather",
            "weather": "weather",
            "温度": "temperature",
            "temperature": "temperature",
            "日程": "calendar_events",
            "提醒事项": "calendar_events",
            "calendar_events": "calendar_events",
            "步数": "steps",
            "steps": "steps",
            "睡眠": "sleep_hours",
            "sleep_hours": "sleep_hours",
            "心率": "heart_rate",
            "heart_rate": "heart_rate",
            "设备已锁定": "device_locked",
            "device_locked": "device_locked",
            "专注模式": "focus_mode",
            "focus_mode": "focus_mode",
            "当前歌曲": "now_playing",
            "正在播放": "now_playing",
            "now_playing": "now_playing",
            "信息截止时间": "timestamp",
            " battery_charging": "battery_charging",
        }

        def _normalize_phone_data(self, raw: dict) -> dict:
            """Unwrap nested envelope and map Chinese keys to English columns."""
            data = dict(raw)

            # Unwrap nested envelope from iOS Shortcuts.
            # The key varies: "phone-status", "手机状态", or other single-key wrappers.
            _ENVELOPE_KEYS = {"phone-status", "手机状态", "phone_status"}
            for ek in _ENVELOPE_KEYS:
                if ek in data and isinstance(data[ek], str):
                    try:
                        inner = json.loads(data[ek])
                        if isinstance(inner, dict):
                            data = inner
                            break
                    except (json.JSONDecodeError, TypeError):
                        pass
            # Fallback: if there's exactly one key whose value is a JSON string, unwrap it
            if len(data) == 1:
                sole_key = next(iter(data))
                if isinstance(data[sole_key], str):
                    try:
                        inner = json.loads(data[sole_key])
                        if isinstance(inner, dict):
                            data = inner
                    except (json.JSONDecodeError, TypeError):
                        pass

            # Map Chinese/mixed keys to canonical English column names
            norm: dict = {}
            for k, v in data.items():
                eng = self._ZH_KEY_MAP.get(k)
                if eng:
                    norm[eng] = v
                else:
                    norm[k] = v

            # Parse weather string like "17°C and Sunny" into temperature + weather
            wx = norm.get("weather", "")
            if isinstance(wx, str) and "°" in wx and norm.get("temperature") is None:
                import re
                m = re.search(r"(-?\d+(?:\.\d+)?)\s*°", wx)
                if m:
                    norm["temperature"] = float(m.group(1))
                    # strip the temp part from weather description
                    desc = re.sub(r"-?\d+(?:\.\d+)?\s*°\s*C?\s*(and\s*)?", "", wx).strip()
                    if desc:
                        norm["weather"] = desc

            # Shortcut string concat leaves stray leading separators ("，晴")
            if isinstance(norm.get("weather"), str):
                norm["weather"] = norm["weather"].strip().lstrip("，, ").strip()

            # Sanity bounds: shortcut health-sample math sometimes explodes
            # (multi-day sums, duplicate iPhone+Watch sources — 316152 steps
            # has really happened). Garbage is dropped from the structured
            # columns but survives verbatim in raw_json.
            for key, lo, hi in (("steps", 0, 100000), ("heart_rate", 20, 250)):
                v = norm.get(key)
                if v is None:
                    continue
                try:
                    f = float(v)
                except (TypeError, ValueError):
                    norm[key] = None
                    continue
                if not (lo <= f <= hi):
                    norm[key] = None

            # Chinese timestamp ("2026年7月12日 11:40") -> ISO with local tz,
            # so the injector's "N minutes ago" math works.
            ts = norm.get("timestamp")
            if isinstance(ts, str) and "年" in ts:
                import re
                m = re.search(
                    r"(\d{4})年(\d{1,2})月(\d{1,2})日\s*(\d{1,2}):(\d{2})", ts
                )
                if m:
                    y, mo, d, h, mi = (int(g) for g in m.groups())
                    try:
                        norm["timestamp"] = datetime(
                            y, mo, d, h, mi
                        ).astimezone().isoformat()
                    except ValueError:
                        pass

            return norm

        def _handle_phone_status_post(self) -> None:
            """POST /phone-status — receive phone state from iOS Shortcuts.

            Accepts both flat JSON and the nested {"phone-status": "..."} format
            that iOS Shortcuts produces. Keys can be in English or Chinese.
            """
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length else b""
            try:
                raw = json.loads(body.decode("utf-8", errors="replace"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                self.send_response(400)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(b'{"error":"invalid JSON"}')
                return

            data = self._normalize_phone_data(raw)
            ts = data.get("timestamp") or datetime.now(timezone.utc).isoformat()

            # Tri-state lock flag: NULL = not reported, else 0/1.
            # Shortcuts may send a bool or a localized string.
            dl_raw = data.get("device_locked")
            if dl_raw is None:
                device_locked = None
            elif isinstance(dl_raw, str):
                device_locked = (
                    0 if dl_raw.strip() in ("", "0", "false", "False", "否", "no")
                    else 1
                )
            else:
                device_locked = 1 if dl_raw else 0

            with store._lock:
                store.conn.execute(
                    """INSERT INTO phone_status
                       (timestamp, battery_level, battery_charging, current_app,
                        screen_time_minutes, location, weather, temperature,
                        calendar_events, steps, sleep_hours, heart_rate,
                        focus_mode, device_locked, now_playing, raw_json)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        ts,
                        data.get("battery_level"),
                        1 if data.get("battery_charging") else 0,
                        data.get("current_app"),
                        data.get("screen_time_minutes"),
                        data.get("location"),
                        data.get("weather"),
                        data.get("temperature"),
                        data.get("calendar_events"),
                        data.get("steps"),
                        data.get("sleep_hours"),
                        data.get("heart_rate"),
                        data.get("focus_mode"),
                        device_locked,
                        (data.get("now_playing") or "").strip() or None,
                        # Wire-format body, pre-normalization: the one place
                        # to see exactly what the shortcut sent (debugging
                        # envelope/key issues). Normalized values live in the
                        # structured columns.
                        body.decode("utf-8", errors="replace"),
                    ),
                )
                store.conn.execute(
                    "DELETE FROM phone_status WHERE id NOT IN "
                    "(SELECT id FROM phone_status ORDER BY id DESC LIMIT 100)"
                )
                store.conn.commit()

            out = b'{"ok":true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(out)

        def _handle_phone_status_get(self) -> None:
            """GET /phone-status — return the most recent phone status row."""
            with store._lock:
                row = store.conn.execute(
                    "SELECT * FROM phone_status ORDER BY id DESC LIMIT 1"
                ).fetchone()
            if not row:
                out = b'{"error":"no data"}'
                self.send_response(404)
            else:
                out = json.dumps(
                    {k: row[k] for k in row.keys()},
                    ensure_ascii=False,
                ).encode("utf-8")
                self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(out)

        def _handle_phone_event_post(self) -> None:
            """POST /phone-event — one event from an iOS Shortcuts automation.

            Body: JSON {"event": "alarm_stopped", "detail": "..."} (detail
            optional). A bare string body is tolerated and treated as the
            event name. Timestamp is stamped server-side.
            """
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length else b""
            try:
                raw = json.loads(body.decode("utf-8", errors="replace"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                # Tolerate a plain-text body: treat it as the event name.
                raw = body.decode("utf-8", errors="replace").strip()

            if isinstance(raw, str):
                data = {"event": raw}
            elif isinstance(raw, dict):
                data = raw
            else:
                data = {}

            event = str(data.get("event") or "").strip()
            if not event:
                out = b'{"error":"missing event field"}'
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(out)
                return

            detail = data.get("detail")
            ts = datetime.now(timezone.utc).isoformat()
            with store._lock:
                store.conn.execute(
                    "INSERT INTO phone_events (timestamp, event, detail) "
                    "VALUES (?,?,?)",
                    # Generous detail cap: screen_share events carry a whole
                    # screen's OCR text, consumed on demand via GET.
                    (ts, event[:100], str(detail)[:2000] if detail else None),
                )
                store.conn.execute(
                    "DELETE FROM phone_events WHERE id NOT IN "
                    "(SELECT id FROM phone_events ORDER BY id DESC LIMIT 500)"
                )
                store.conn.commit()

            out = b'{"ok":true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(out)

        def _handle_phone_event_get(self) -> None:
            """GET /phone-event?hours=48&limit=20 — recent events, newest first."""
            qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            try:
                hours = float(qs.get("hours", ["48"])[0])
            except ValueError:
                hours = 48.0
            try:
                limit = int(qs.get("limit", ["20"])[0])
            except ValueError:
                limit = 20
            since = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()

            with store._lock:
                rows = store.conn.execute(
                    "SELECT timestamp, event, detail FROM phone_events "
                    "WHERE timestamp >= ? ORDER BY id DESC LIMIT ?",
                    (since, max(1, min(limit, 100))),
                ).fetchall()
            out = json.dumps(
                {"events": [
                    {"timestamp": r["timestamp"], "event": r["event"],
                     "detail": r["detail"]}
                    for r in rows
                ]},
                ensure_ascii=False,
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(out)

        # ---- Screen peek (agent-initiated screenshot) --------------------
        # Reverse channel: the backstage agent sends a trigger mail, Sol's
        # iPhone "on receiving email" automation screenshots and POSTs here.
        # Files are named {ms}_{uuid}.png so lexical order == time order;
        # no metadata file needed.

        _PEEK_DIR = Path(__file__).resolve().parent / "peeks"
        _PEEK_MAX_KEEP = 10
        _PEEK_MAX_BYTES = 30 * 1024 * 1024
        _peek_lock = threading.Lock()

        @classmethod
        def _peek_files(cls) -> list:
            if not cls._PEEK_DIR.exists():
                return []
            return sorted(cls._PEEK_DIR.glob("*.png"))

        def _handle_peek_post(self) -> None:
            """POST /peek — body is the raw screenshot image."""
            length = int(self.headers.get("Content-Length", 0))
            if length <= 0 or length > self._PEEK_MAX_BYTES:
                out = b'{"ok":false,"error":"empty or oversized body"}'
                self.send_response(400)
            else:
                body = self.rfile.read(length)
                fname = f"{int(_time.time() * 1000)}_{uuid.uuid4().hex[:8]}.png"
                with self._peek_lock:
                    self._PEEK_DIR.mkdir(exist_ok=True)
                    (self._PEEK_DIR / fname).write_bytes(body)
                    for old in self._peek_files()[:-self._PEEK_MAX_KEEP]:
                        try:
                            old.unlink()
                        except OSError:
                            pass
                out = json.dumps({"ok": True, "file": fname}).encode()
                self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(out)

        def _handle_peek_latest(self) -> None:
            """GET /peek/latest — newest screenshot's paths and freshness."""
            files = self._peek_files()
            if not files:
                out = b'{"error":"no screenshots"}'
                self.send_response(404)
            else:
                newest = files[-1]
                try:
                    ts_ms = int(newest.name.split("_")[0])
                except ValueError:
                    ts_ms = int(newest.stat().st_mtime * 1000)
                win_path = str(newest)
                drive = win_path[0].lower()
                wsl_path = f"/mnt/{drive}{win_path[2:]}".replace("\\", "/")
                out = json.dumps({
                    "file": newest.name,
                    "ts_ms": ts_ms,
                    "age_seconds": round(_time.time() - ts_ms / 1000, 1),
                    "win_path": win_path,
                    "wsl_path": wsl_path,
                }).encode()
                self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(out)

        # ---- Legacy / endpoint (backward compat) -------------------------

        def _handle_legacy_post(self) -> None:
            # No session layer here, so no handshake identity: tools/list over
            # the legacy endpoint always gets the conservative (hidden) view.
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
            body = _IMPORT_HTML.replace("__EMBED_MODEL__", OLLAMA_EMBED_MODEL).encode("utf-8")
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

        def _handle_embed_status(self) -> None:
            """GET /import/embed_status — global embedding queue depth.

            UI tracks the high-water mark client-side to render a progress bar
            for the post-extraction phase (when LLM extraction is done but
            the embedding model is still chewing through the backlog).
            """
            self._send_json(200, {"pending": _IMPORT_EMBED_QUEUE.qsize()})

        # ---- Admin: DB stats + maintenance tasks ----

        def _handle_stats(self) -> None:
            """GET /stats — quick db rollup for the admin panel header."""
            with store._lock:
                long_term = store.conn.execute(
                    "SELECT COUNT(*) FROM memories WHERE memory_kind='long_term'"
                ).fetchone()[0]
                with_emb = store.conn.execute(
                    "SELECT COUNT(*) FROM memories WHERE memory_kind='long_term' AND length(embedding) > 0"
                ).fetchone()[0]
                pinned = store.conn.execute(
                    "SELECT COUNT(*) FROM memories WHERE pinned=1 AND memory_kind='long_term'"
                ).fetchone()[0]
                missing_emb = store.conn.execute(
                    "SELECT COUNT(*) FROM memories WHERE memory_kind='long_term' "
                    "AND digested = 0 AND length(embedding) = 0"
                ).fetchone()[0]
                digested_total = store.conn.execute(
                    "SELECT COUNT(*) FROM memories WHERE memory_kind='long_term' AND digested = 1"
                ).fetchone()[0]
                cutoff = (datetime.now(timezone.utc) - timedelta(days=DIGESTED_PRUNE_DAYS)).isoformat()
                digested_stale = store.conn.execute(
                    "SELECT COUNT(*) FROM memories WHERE memory_kind='long_term' AND digested = 1 "
                    "AND COALESCE(NULLIF(last_active, ''), updated_at, created_at) < ?",
                    (cutoff,),
                ).fetchone()[0]
                unconsolidated_sessions = store.conn.execute(
                    "SELECT COUNT(*) FROM ("
                    "  SELECT session_id FROM memories"
                    "  WHERE session_id != '' AND digested = 0 AND consolidated = 0"
                    "  AND memory_kind='long_term'"
                    "  GROUP BY session_id HAVING COUNT(*) >= 5"
                    ")"
                ).fetchone()[0]

            with _IMPORT_TASKS_LOCK:
                active = [
                    {
                        "id": t["id"], "kind": t.get("kind", "import"),
                        "done": t.get("done", False), "stage": t.get("stage", ""),
                        "processed": t.get("processed", 0), "total": t.get("total", 0),
                    }
                    for t in _IMPORT_TASKS.values() if not t.get("done")
                ]

            # Readiness signals for the import UI. embedding needs a reachable
            # ollama; extraction follows IMPORT_PROVIDER (see below); the merge
            # button needs *either* a cloud key or a reachable local ollama.
            openrouter_present = bool(os.environ.get("OPENROUTER_API_KEY"))
            gemini_present = bool(os.environ.get("GOOGLE_AI_STUDIO_KEY"))
            ollama_ok = _ollama_reachable()

            # Which LLM each of the two pipelines would actually use right now.
            # _effective_import_provider() already demotes key-less cloud
            # providers to ollama, so these values are what really runs.
            import_provider = _effective_import_provider()
            import_model = _import_model_name(import_provider)
            consolidate_backend = (
                os.environ.get("LLM_BACKEND", "openrouter") or "openrouter"
            ).strip().lower()
            consolidate_model = (
                OLLAMA_MODEL if consolidate_backend == "ollama"
                else (os.environ.get("OPENROUTER_MODEL", "").strip()
                      or OPENROUTER_MODEL_DEFAULT)
            )

            # Extraction readiness follows IMPORT_PROVIDER; cloud providers stay
            # "ready" without their key because they degrade to ollama.
            configured_provider = (IMPORT_PROVIDER or "").strip().lower()
            if configured_provider == "openrouter":
                analysis_ready = openrouter_present or ollama_ok
            elif configured_provider == "gemini":
                analysis_ready = gemini_present or ollama_ok
            else:
                analysis_ready = ollama_ok
            # The merge button has its own channel (openrouter key or ollama) —
            # keep it independent of the extraction provider.
            consolidate_ready = openrouter_present or ollama_ok

            self._send_json(200, {
                "long_term_count": long_term,
                "with_embedding": with_emb,
                "embedding_coverage": round(with_emb / long_term, 4) if long_term else 0.0,
                "missing_embedding": missing_emb,
                "unconsolidated_sessions_over_5": unconsolidated_sessions,
                "pinned_count": pinned,
                "digested_total": digested_total,
                "digested_stale": digested_stale,
                "prune_days": DIGESTED_PRUNE_DAYS,
                "active_tasks": active,
                "openrouter_key_present": openrouter_present,
                "gemini_key_present": gemini_present,
                "import_provider": import_provider,
                "import_model": import_model,
                "consolidate_backend": consolidate_backend,
                "consolidate_model": consolidate_model,
                "analysis_ready": analysis_ready,
                "consolidate_ready": consolidate_ready,
                "embedding_ready": ollama_ok,
            })

        def _handle_admin_reindex(self) -> None:
            """POST /admin/reindex — kick off reindex_embeddings in the background."""
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length else b"{}"
            try:
                data = json.loads(body.decode("utf-8", errors="replace") or "{}")
            except json.JSONDecodeError:
                data = {}

            from reindex_embeddings import run_reindex
            kwargs = {
                "db_path": Path(store.db_path),
                "workers": max(1, min(8, int(data.get("workers", 4) or 4))),
                "batch": max(1, min(500, int(data.get("batch", 50) or 50))),
                "limit": max(0, int(data.get("limit", 0) or 0)),
            }
            task = _start_admin_task(store, "reindex", run_reindex, kwargs)
            self._send_json(200, {"ok": True, "async": True, "task_id": task["id"]})

        def _handle_admin_prune(self) -> None:
            """POST /admin/prune — hard-delete digested rows older than threshold_days."""
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length else b"{}"
            try:
                data = json.loads(body.decode("utf-8", errors="replace") or "{}")
            except json.JSONDecodeError:
                data = {}
            days = max(1, int(data.get("days", DIGESTED_PRUNE_DAYS) or DIGESTED_PRUNE_DAYS))
            n = store.prune_stale_digested(days)
            self._send_json(200, {"ok": True, "deleted": n, "threshold_days": days})

        def _handle_admin_consolidate(self) -> None:
            """POST /admin/consolidate — kick off consolidate_sessions in the background.

            Cloud key present → cloud (with mid-run failover to ollama). No cloud
            key but ollama reachable → run locally on ollama. Neither → 400.
            """
            has_key = bool(os.environ.get("OPENROUTER_API_KEY"))
            if has_key:
                backend = None  # env default (openrouter); call_llm fails over if it dies
            elif _ollama_reachable():
                backend = "ollama"  # no cloud key, but the local model can still merge
            else:
                self._send_json(400, {
                    "ok": False,
                    "error": "没有可用的解析通道：未配置 OPENROUTER_API_KEY，本地 Ollama 也不可达。"
                             "请检查 .env 里的 URL 和 key，或启动 Ollama。",
                })
                return

            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length else b"{}"
            try:
                data = json.loads(body.decode("utf-8", errors="replace") or "{}")
            except json.JSONDecodeError:
                data = {}

            from consolidate_sessions import run_consolidate
            kwargs = {
                "db_path": Path(store.db_path),
                "min_fragments": max(2, int(data.get("min_fragments", 5) or 5)),
                "max_fragments": max(0, int(data.get("max_fragments", 0) or 0)),
                "workers": max(1, min(4, int(data.get("workers", 2) or 2))),
                "limit": max(0, int(data.get("limit", 0) or 0)),
                "dry_run": bool(data.get("dry_run", False)),
                "backend": backend,
            }
            task = _start_admin_task(store, "consolidate", run_consolidate, kwargs)
            self._send_json(200, {"ok": True, "async": True, "task_id": task["id"]})

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
                    raw = _call_import_llm(_IMPORT_EXTRACT_TMPL.format(chunk=chunk))
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
            elif path == "/admin/reindex":
                self._handle_admin_reindex()
            elif path == "/admin/consolidate":
                self._handle_admin_consolidate()
            elif path == "/admin/prune":
                self._handle_admin_prune()
            elif path == "/phone-status":
                self._handle_phone_status_post()
            elif path == "/phone-event":
                self._handle_phone_event_post()
            elif path == "/peek":
                self._handle_peek_post()
            else:
                self._handle_legacy_post()

        def do_GET(self) -> None:
            path = self.path.split("?")[0]
            if path == "/mcp":
                self._handle_mcp_get()
            elif path == "/breath-hook":
                self._handle_breath_hook()
            elif path == "/associate":
                self._handle_associate()
            elif path == "/import":
                self._handle_import_get()
            elif path == "/import/status":
                self._handle_import_status()
            elif path == "/import/embed_status":
                self._handle_embed_status()
            elif path == "/stats":
                self._handle_stats()
            elif path == "/phone-status":
                self._handle_phone_status_get()
            elif path == "/phone-event":
                self._handle_phone_event_get()
            elif path == "/peek/latest":
                self._handle_peek_latest()
            else:
                info = json.dumps({
                    "name": "memory-mcp",
                    "version": "0.2.0",
                    "transport": "streamable-http",
                    "endpoints": {
                        "streamable_http": "/mcp",
                        "legacy_json_rpc": "/",
                        "breath_hook": "/breath-hook",
                        "associate": "/associate",
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
    sys.stderr.write(f"[memory-mcp]   Associate hook:         GET  http://{host}:{port}/associate?q=...\n")
    sys.stderr.write(f"[memory-mcp]   Import UI:              GET  http://localhost:{port}/import\n")

    # Only auto-open the import page on a genuine first run (the setup wizard
    # dropped a one-shot `.first_run_open` marker next to this script) or when
    # the operator explicitly asks with --open-browser. Every other restart of
    # the 3456 service stays silent — no more surprise tab on every launch.
    _marker = Path(__file__).resolve().parent / ".first_run_open"
    _first_run = _marker.exists()
    if open_browser or _first_run:
        def _open_import_once() -> None:
            try:
                webbrowser.open(f"http://localhost:{port}/import")
            except Exception:
                pass
            finally:
                if _first_run:
                    try:
                        _marker.unlink()
                    except OSError:
                        pass
        threading.Timer(1.0, _open_import_once).start()
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
    global OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_EMBED_MODEL, OLLAMA_TIMEOUT, SUMMARIZE_DRY_RUN
    global IMPORT_PROVIDER, IMPORT_MODEL

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
    parser.add_argument(
        "--open-browser", action="store_true", default=False,
        help="Open the import page in a browser on startup (HTTP mode). "
             "Off by default; the first-run wizard triggers it once via a marker file.",
    )
    args = parser.parse_args()

    OLLAMA_BASE_URL = args.ollama_url
    OLLAMA_MODEL = args.ollama_model
    OLLAMA_TIMEOUT = args.ollama_timeout
    SUMMARIZE_DRY_RUN = args.dry_run

    db_path = Path(args.db).resolve()
    # Load .env sitting next to the db (or the script if db is elsewhere).
    # Populates OPENROUTER_API_KEY etc. for the consolidate path.
    _load_dotenv(db_path.parent)
    _load_dotenv(Path(__file__).parent)

    # CLI flags win; otherwise let the .env just loaded refresh the module
    # globals — they were frozen at import time, before .env existed in the
    # environment, so without this the HTTP server never sees .env overrides.
    if args.ollama_url == parser.get_default("ollama_url"):
        OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", OLLAMA_BASE_URL)
    if args.ollama_model == parser.get_default("ollama_model"):
        OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", OLLAMA_MODEL)
    if args.ollama_timeout == parser.get_default("ollama_timeout"):
        OLLAMA_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", OLLAMA_TIMEOUT))
    # No CLI flag for the embedding model, so no default-guard — always let .env
    # win. Otherwise the wizard's non-default OLLAMA_EMBED_MODEL is ignored by the
    # server and the embed worker mixes dimensions with reindex.
    OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", OLLAMA_EMBED_MODEL)
    # Same story for the import-extraction provider — no CLI flag, so .env always
    # wins over the import-time snapshot. This block runs before the stdio/http
    # branch below, so both transports see identical .env-driven config.
    IMPORT_PROVIDER = os.environ.get("IMPORT_PROVIDER", IMPORT_PROVIDER)
    IMPORT_MODEL = os.environ.get("IMPORT_MODEL", IMPORT_MODEL)
    # Emit the demotion warning (if any) once at startup rather than mid-import.
    _eff_provider = _effective_import_provider()
    sys.stderr.write(
        f"[memory-mcp] import extraction: provider={_eff_provider} "
        f"model={_import_model_name(_eff_provider)}\n"
    )

    store = MemoryStore(db_path)

    # One eager prune at startup so users see immediate effect after raising
    # DIGESTED_PRUNE_DAYS or after a long downtime; then schedule the daemon.
    try:
        store.prune_stale_digested(DIGESTED_PRUNE_DAYS)
    except Exception as exc:
        sys.stderr.write(f"[prune-startup] error: {exc}\n")
    _start_prune_daemon(store)
    _start_maintenance_daemon(store, db_path)

    if args.http:
        _run_http(store, args.host, args.port, open_browser=args.open_browser)
    else:
        _run_stdio(store)


if __name__ == "__main__":
    main()
