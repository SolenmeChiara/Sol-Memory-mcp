#!/usr/bin/env python3
"""SessionStart hook: surface highest-weight memories at the start of every session.

Strategy:
1. Try the running HTTP server at /breath-hook (fast, no DB lock contention).
2. If that fails, fall back to the CLI subcommand `python memory_mcp.py breath`,
   which opens its own SQLite connection.
3. If both paths fail, emit valid (empty-context) JSON so Claude Code's session
   doesn't hang — but log the failure to stderr so the user can see what went wrong.
   No silent swallowing.

Skip entirely with env var SOL_MEMORY_SKIP_BREATH=1.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path


HOOK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = HOOK_DIR.parent.parent  # .claude/hooks/ -> .claude/ -> project root
DEFAULT_URL = os.environ.get("SOL_MEMORY_URL", "http://localhost:3456")
LIMIT = int(os.environ.get("SOL_MEMORY_BREATH_LIMIT", "10"))
HTTP_TIMEOUT = float(os.environ.get("SOL_MEMORY_BREATH_TIMEOUT", "3"))
CLI_TIMEOUT = float(os.environ.get("SOL_MEMORY_BREATH_CLI_TIMEOUT", "30"))


def emit(context: str) -> None:
    """Always emit a valid JSON payload to stdout, even when context is empty."""
    payload = {
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": context,
        }
    }
    print(json.dumps(payload, ensure_ascii=False))


def try_http() -> str | None:
    url = f"{DEFAULT_URL.rstrip('/')}/breath-hook?limit={LIMIT}"
    try:
        with urllib.request.urlopen(url, timeout=HTTP_TIMEOUT) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
        sys.stderr.write(f"[session_breath] HTTP {url} failed: {exc} — falling back to CLI\n")
        return None


def try_cli() -> str | None:
    script = PROJECT_ROOT / "memory_mcp.py"
    db = PROJECT_ROOT / "memory.db"
    if not script.exists():
        sys.stderr.write(f"[session_breath] CLI fallback unavailable: {script} not found\n")
        return None
    if not db.exists():
        sys.stderr.write(f"[session_breath] CLI fallback unavailable: {db} not found\n")
        return None
    try:
        result = subprocess.run(
            [sys.executable, str(script), "breath", "--limit", str(LIMIT), "--db", str(db)],
            capture_output=True,
            text=True,
            timeout=CLI_TIMEOUT,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired:
        sys.stderr.write(f"[session_breath] CLI timed out after {CLI_TIMEOUT}s\n")
        return None
    if result.returncode != 0:
        sys.stderr.write(
            f"[session_breath] CLI exited {result.returncode}: {result.stderr.strip()[:300]}\n"
        )
        return None
    return result.stdout


def main() -> None:
    # Windows console defaults to GBK; force UTF-8 so Claude Code reads JSON correctly.
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

    if os.environ.get("SOL_MEMORY_SKIP_BREATH"):
        emit("")
        return

    body = try_http()
    if body is None:
        body = try_cli()

    if body is None:
        sys.stderr.write(
            "[session_breath] both HTTP and CLI failed — emitting empty context. "
            "Check that memory_mcp HTTP server is running, or that memory.db is accessible.\n"
        )
        emit("")
        return

    text = (body or "").strip()
    if not text:
        emit("")
        return

    # Wrap so Claude Code knows where it came from
    wrapped = (
        "<sol-memory-breath>\n"
        "以下是当前权重最高的记忆，作为本次对话开始时的上下文参考。\n"
        "（自动浮现，未触发激活；想主动激活请用 extmcp_breath 工具）\n\n"
        f"{text}\n"
        "</sol-memory-breath>"
    )
    emit(wrapped)


if __name__ == "__main__":
    main()
