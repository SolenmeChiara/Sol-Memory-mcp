#!/usr/bin/env python3
"""First-run setup wizard for Sol-Memory-mcp.

Behaviour
---------
- If a ``.env`` already exists next to this script → exit 0 immediately and
  silently (no output at all). This makes it safe to call unconditionally from
  start_http.bat: on every normal launch it's a no-op.
- Otherwise, walk the user through a tiny interactive setup and generate a
  commented ``.env``. Everything can be skipped with Enter; the minimal .env it
  produces still lets the server run fully locally (embedding defaults to the
  local ollama model, no cloud key required).

Design notes
------------
- Stdlib only. This file must never be imported by the stdio server path
  (memory_mcp.py) — it blocks on input() and would hang a stdio client.
- Windows consoles default to GBK/CP936; we reconfigure the streams to UTF-8
  with errors="replace" so the Chinese prompts never crash the wizard.
- Ctrl+C exits cleanly WITHOUT leaving a half-written .env: the file is only
  written once, at the very end, after every question has been answered.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ENV_PATH = SCRIPT_DIR / ".env"
FIRST_RUN_MARKER = SCRIPT_DIR / ".first_run_open"

# Cloud-parse presets. Every provider here is an OpenAI-compatible
# (chat/completions) endpoint — the whole codebase talks that one dialect via
# the OPENROUTER_* variables, so AI Studio / OpenAI / Anthropic all get written
# into those same three vars, just with a different base_url + model.
PRESETS = {
    "1": {
        "label": "OpenRouter（默认，聚合各家模型）",
        "base_url": "https://openrouter.ai/api/v1",
        "model": "google/gemini-3.1-flash-lite-preview",
    },
    "2": {
        "label": "Google AI Studio（免过路费，直连 Gemini）",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-3.1-flash-lite-preview",
    },
    "3": {
        "label": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
    },
    "4": {
        "label": "Anthropic（Claude，OpenAI 兼容层）",
        "base_url": "https://api.anthropic.com/v1/",
        "model": "claude-haiku-4-5",
    },
}

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_EMBED_MODEL = "qwen3-embedding:4b"

# Substrings that mark an ollama model as embedding-capable.
EMBED_HINTS = ("embed", "bge-m3", "nomic-embed", "mxbai", "arctic-embed", "gte", "e5")


def _reconfigure_streams() -> None:
    for stream in (sys.stdout, sys.stderr, sys.stdin):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def _p(text: str = "") -> None:
    """Print that survives a GBK console (falls back to replacement chars)."""
    try:
        print(text)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "utf-8"
        sys.stdout.write(text.encode(enc, "replace").decode(enc, "replace") + "\n")


def _ask(prompt: str, default: str = "") -> str:
    """Prompt once. Enter → default. EOF (piped/closed stdin) → default."""
    suffix = f" [{default}]" if default else ""
    try:
        raw = input(f"{prompt}{suffix}: ")
    except EOFError:
        return default
    raw = raw.strip()
    return raw if raw else default


def _mask(secret: str) -> str:
    if not secret:
        return "(未填写)"
    if len(secret) <= 8:
        return "****"
    return f"{secret[:4]}…{secret[-4:]}"


def _probe_ollama(base_url: str) -> "tuple[bool, list]":
    """(reachable, [model names]) via GET /api/tags. Short timeout, never raises."""
    import json as _json
    import urllib.request as _u
    try:
        req = _u.Request(base_url.rstrip("/") + "/api/tags", method="GET")
        with _u.urlopen(req, timeout=3.0) as resp:
            data = _json.loads(resp.read().decode("utf-8", "replace"))
    except Exception:
        return False, []
    names = [str(m.get("name", "")) for m in (data.get("models") or []) if m.get("name")]
    return True, names


def _find_embed_model(names: "list") -> "str | None":
    for name in names:
        low = name.lower()
        if any(h in low for h in EMBED_HINTS):
            return name
    return None


def _build_env(
    ollama_url: str,
    cloud: "dict | None",
    embed_model: "str | None" = None,
    embed_note: str = "",
) -> str:
    """Render the .env text.

    cloud is None when the user skipped cloud parse. embed_model is written only
    when it differs from the code default (qwen3-embedding:4b); embed_note is a
    human-readable line describing what the wizard found."""
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# Sol-Memory-mcp 配置（由 first_run_setup.py 生成于 {stamp}）",
        "# 变量优先级：进程环境变量 > 本文件 > 代码默认值。",
        "# 删掉本文件并重跑 start_http.bat 会再次触发本向导。",
        "",
        "# ---- 本地 Ollama（embedding 向量 + 轻量本地模型）----",
        "# 本记忆库必须自备 embedding 模型：没有它，语义检索 / breath 排序都不可用。",
        f"# embedding：{embed_note}" if embed_note else "# embedding 默认用 qwen3-embedding:4b。",
        f"OLLAMA_BASE_URL={ollama_url}",
    ]
    if embed_model:
        lines.append(f"OLLAMA_EMBED_MODEL={embed_model}")
    else:
        lines.append("# OLLAMA_EMBED_MODEL=qwen3-embedding:4b  # 代码默认，如需换模型取消注释")
    lines.append("")
    if cloud:
        lines += [
            "# ---- 云端解析通道（OpenAI 格式 chat/completions 端点）----",
            "# 会话提炼 / session 合并 / 记忆再提取都读这一套变量。",
            "LLM_BACKEND=openrouter",
            f"OPENROUTER_BASE_URL={cloud['base_url']}",
            f"OPENROUTER_API_KEY={cloud['api_key']}",
            f"OPENROUTER_MODEL={cloud['model']}",
            "",
        ]
    else:
        lines += [
            "# ---- 云端解析通道（未配置，只跑本地）----",
            "# 只用本地 ollama 时不需要云端 key。要启用云端解析，",
            "# 取消下面注释并填入 key（任意 OpenAI 兼容端点均可）：",
            "# LLM_BACKEND=openrouter",
            "# OPENROUTER_BASE_URL=https://openrouter.ai/api/v1",
            "# OPENROUTER_API_KEY=",
            "# OPENROUTER_MODEL=google/gemini-3.1-flash-lite-preview",
            "",
        ]
    return "\n".join(lines)


def _run_wizard() -> int:
    _p("=" * 60)
    _p(" Sol-Memory-mcp 初次启动向导")
    _p("=" * 60)
    _p("没检测到 .env，先做个极简配置。全程可直接回车用默认值。")
    _p("")

    # 1) Ollama URL
    _p("① 本地 Ollama 地址（跑 embedding 向量和本地轻量模型用）")
    ollama_url = _ask("   回车用默认", DEFAULT_OLLAMA_URL)
    _p("")

    # 1b) Embedding pre-check — this library CANNOT do semantic recall without one.
    embed_model = None   # None => keep code default qwen3-embedding:4b (don't write it)
    _p("   正在检测 Ollama 和 embedding 模型…")
    reachable, models = _probe_ollama(ollama_url)
    if reachable:
        found = _find_embed_model(models)
        if found:
            _p(f"   ✓ Ollama 可达，检测到 embedding 模型：{found}")
            chosen = _ask("   用它？回车确认，或输入别的模型名", found)
            embed_note = f"使用检测到的 {chosen}"
            if chosen != DEFAULT_EMBED_MODEL:
                embed_model = chosen
        else:
            _p("   ⚠ Ollama 可达，但没发现 embedding 模型。")
            _p("     本记忆库必须自备 embedding 模型（语义检索 / breath 排序全靠它）。")
            _p("     推荐先拉一个：ollama pull qwen3-embedding:4b")
            chosen = _ask("   要用的 embedding 模型名（回车用推荐默认，稍后自己 pull）", DEFAULT_EMBED_MODEL)
            embed_note = f"计划使用 {chosen}（记得 `ollama pull {chosen}`）"
            if chosen != DEFAULT_EMBED_MODEL:
                embed_model = chosen
    else:
        _p(f"   ⚠ 连不上 Ollama（{ollama_url}）。")
        _p("     没有 embedding，语义检索 / breath 排序等核心功能都用不了。")
        _p("     建议装好 Ollama + embedding 模型后重跑本向导，或稍后手改 .env。")
        embed_note = "向导运行时 Ollama 不可达——装好后请确认 OLLAMA_EMBED_MODEL（默认 qwen3-embedding:4b）"
    _p("")

    # 2) Cloud-parse preset
    _p("② 云端解析 API（会话提炼 / 记忆合并用；只支持 OpenAI 格式端点）")
    _p("   提醒：这里的模型会处理大量文本，请务必选择足够经济的模型或本地模型来运行。")
    for key in ("1", "2", "3", "4"):
        _p(f"   [{key}] {PRESETS[key]['label']}")
    _p("   [5] 自定义 URL")
    _p("   [0] 跳过（只跑本地，之后想加再改 .env）")
    choice = _ask("   选择", "1")
    _p("")

    cloud: "dict | None" = None
    if choice in PRESETS:
        preset = PRESETS[choice]
        _p(f"   已选：{preset['label']}")
        api_key = _ask("   API key（回车留空跳过）", "")
        model = _ask("   模型名（回车用推荐默认）", preset["model"])
        cloud = {
            "base_url": preset["base_url"],
            "api_key": api_key,
            "model": model,
        }
    elif choice == "5":
        base_url = _ask("   自定义 base_url（OpenAI 格式，如 https://.../v1）", "")
        if base_url:
            api_key = _ask("   API key（回车留空跳过）", "")
            model = _ask("   模型名", "gpt-4o-mini")
            cloud = {"base_url": base_url, "api_key": api_key, "model": model}
        else:
            _p("   未填 URL，按跳过处理。")
    else:
        _p("   跳过云端解析，只跑本地。")
    _p("")

    # Summary (key masked)
    _p("-" * 60)
    _p("即将写入 .env：")
    _p(f"  OLLAMA_BASE_URL   = {ollama_url}")
    _p(f"  embedding 模型     = {embed_model or DEFAULT_EMBED_MODEL}"
       + ("" if embed_model else "（代码默认）"))
    if cloud:
        _p(f"  OPENROUTER_BASE_URL = {cloud['base_url']}")
        _p(f"  OPENROUTER_MODEL    = {cloud['model']}")
        _p(f"  OPENROUTER_API_KEY  = {_mask(cloud['api_key'])}")
    else:
        _p("  云端解析          = 未配置（只跑本地）")
    _p("-" * 60)

    ENV_PATH.write_text(
        _build_env(ollama_url, cloud, embed_model, embed_note), encoding="utf-8"
    )
    # One-shot marker: tells the server to open the import page once, this launch.
    try:
        FIRST_RUN_MARKER.write_text("1", encoding="utf-8")
    except OSError:
        pass

    _p(f"✓ 已生成 {ENV_PATH}")
    if cloud and not cloud["api_key"]:
        _p("  提示：云端 key 留空了，云端解析暂不可用；本地 embedding 照常。")
    _p("  正在启动服务…")
    return 0


def main() -> int:
    # Already configured → silent no-op. This is the common case on every launch.
    if ENV_PATH.exists():
        return 0

    _reconfigure_streams()
    try:
        return _run_wizard()
    except KeyboardInterrupt:
        _p("")
        _p("已取消，未写入任何文件。下次启动会再问一遍。")
        return 130


if __name__ == "__main__":
    sys.exit(main())
