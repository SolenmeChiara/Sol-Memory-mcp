# Sol-Memory-mcp

**English** | [中文](#中文)

A lightweight SQLite-backed MCP memory server with hybrid retrieval (keyword + vector), Ebbinghaus decay, emotion coordinates, lifecycle management (pinned / resolved / digested) and introspection tools. It also doubles as an always-on HTTP hub: the box an iPhone posts its status, events and screenshots to, and the box a background agent reads them back out of.

## Features

- Save, search, list and delete memory entries
- BM25 keyword search fused with bge-m3 vector cosine similarity
- Ebbinghaus decay formula weighted by emotional arousal
- Chinese summarization / sentiment analysis / memory extraction via a local Ollama model
- Two transports: stdio (Claude Desktop) and Streamable HTTP (phones, remote clients)
- Drag-and-drop web UI for importing conversation history, with automatic format detection (Claude official export / plugin export / ChatGPT mapping)
- Phone sense endpoints: status snapshots, event timeline, screenshot drop-box — all forgiving about payload shape, all self-cleaning
- Backend inbox: leave messages for a background agent, with an urgent flag for express delivery

## The companion agent

[nudge-agent](https://github.com/SolenmeChiara/nudge-agent) is this project's companion process — a persistent background Claude Code instance that consumes what lands here. The dependency is deliberately one-directional:

- **This repo standalone is a perfectly fine diary.** Every memory feature works with no agent anywhere. The agent-facing extension tables (phone events, screen peeks, backend inbox) keep accepting and storing whatever the phone posts; rolling cleanup is self-contained here, so data never grows unbounded waiting for a reader that doesn't exist.
- **The agent without this repo is a cripple, by design.** Memory continuity, the inbox, the urgent lane and every phone sense live on this side of the bond.
- Planned: agent-facing tools (`extmcp_send_to_backend`, backend-oriented session recall) get hidden from MCP clients when no agent is detected, so a standalone install never shows dead switches.

## Quick start

```bash
# stdio mode (for Claude Desktop)
python memory_mcp.py --db ./memory.db

# HTTP mode (port 3456)
start_http.bat
# or
python memory_mcp.py --http --port 3456 --db ./memory.db
```

> **Security note**: the HTTP server has no authentication. Keep it on localhost / a private overlay network (Tailscale etc.); do not expose port 3456 to the public internet.

> **WSL / Windows warning**: never let processes on both sides of the WSL boundary open the SQLite file directly — WAL shared memory does not survive the 9P filesystem, whichever side opens the DB first locks the other out with `disk I/O error`. Run this server on the side that owns the file and let everything else talk to the HTTP port.

## MCP tools

| Tool | Description |
|---|---|
| `extmcp_save_memory` | Save/update a memory; embedding + sentiment run in a background worker |
| `extmcp_search_memory` | Hybrid keyword + vector search (hits bump `activation_count`) |
| `extmcp_list_memories` | List by update time, newest first |
| `extmcp_delete_memory` | Delete one entry |
| `extmcp_summarize_recent` | Chinese summary of the last N memories (`limit` 1-30, default 10); activates what it cites |
| `extmcp_random_memories` | Draw 4-10 entries at random |
| `extmcp_dream` | Introspection: find the most similar memory pairs, suggest what to resolve / digest |
| `extmcp_grow` | Split a journal / long text into 2-6 standalone memories |
| `extmcp_breath` | Active recall: surface high-weight unresolved memories + pinned cores, 0.3-discounted activation, 6h dedup |
| `extmcp_recall_session` | Pull the full memory timeline of one session by `session_id` |
| `extmcp_session_preview` | Peek at the last few messages of recent conversations |
| `extmcp_send_to_backend` | Leave a message in the backend inbox; `urgent=true` requests express delivery (the agent's injector polls every 30s and types it straight into the agent's chat) |

## HTTP endpoints

| Method & path | What it does |
|---|---|
| `POST /mcp`, `GET /mcp` | Streamable HTTP MCP transport |
| `GET /breath-hook` | Read-only breath block (pinned + top unresolved by decay score) — never activates memories, safe for hooks |
| `POST /import`, `GET /import` | Conversation-history import (web UI on GET; JSON body or local path on POST) |
| `GET /import/status`, `GET /import/embed_status` | Import task progress / global embedding queue depth |
| `POST /phone-status` | Phone state snapshot from iOS Shortcuts. Forgiving: flat or enveloped JSON, Chinese-locale keys mapped server-side, impossible values sanity-bounded to NULL, raw wire body kept in `raw_json` for debugging |
| `GET /phone-status` | Most recent phone status row |
| `POST /phone-event` | One timestamped event from an iOS automation (`{"event": "alarm_stopped", "detail": "…"}`); rolling 500 rows |
| `GET /phone-event?hours=48&limit=20` | Recent events, newest first |
| `POST /peek` | Raw screenshot bytes from the phone (the see-screen channel); rolling 10 files under `peeks/`, gitignored |
| `GET /peek/latest` | Newest screenshot's path and freshness |

## Bulk-importing conversation history

For 700MB-scale official Claude exports, use the CLI:

```bash
# dry-run first: detect format + conversation count (no LLM calls)
python batch_import.py "path/to/conversations.json" --dry-run

# try the first 5
python batch_import.py "path/to/conversations.json" --limit 5

# full run (may take hours)
python batch_import.py "path/to/conversations.json"

# resume from conversation N after an interruption
python batch_import.py "path/to/conversations.json" --start 500
```

Or open [http://localhost:3456/import](http://localhost:3456/import): drop small files (≤30 MB) in, paste a local absolute path for big ones — the server starts a background task and the browser polls progress.

## Backfilling missing embeddings

When the background worker falls behind or the service is interrupted, new memories are left with an empty embedding, which hurts vector recall. Backfill manually:

```bash
# 4 parallel workers by default
python reindex_embeddings.py

# gentler on ollama
python reindex_embeddings.py --workers 2

# debug with 5 rows
python reindex_embeddings.py --limit 5
```

Only touches rows with `length(embedding)=0`, safe to re-run. Exits after 10 consecutive failures rather than silently skipping.

## SessionStart hook (optional)

Auto-inject high-weight memories into Claude Code's context at session start. Add to your `.claude/settings.local.json` (or `~/.claude/settings.local.json` for global effect):

```json
{
  "hooks": {
    "SessionStart": [
      {
        "matcher": "startup|resume|clear",
        "hooks": [
          {
            "type": "command",
            "command": "python \"${CLAUDE_PROJECT_DIR}/.claude/hooks/session_breath.py\""
          }
        ]
      }
    ]
  }
}
```

The hook script `.claude/hooks/session_breath.py` ships with this repo. It:
1. Tries `GET http://localhost:3456/breath-hook` first (fastest when the HTTP server is up)
2. Falls back to the `python memory_mcp.py breath` CLI subcommand (reads SQLite directly, no server needed)
3. Never blocks the session if both fail, but errors go to stderr in full (no silent swallowing)

Environment variables:

| Variable | Description | Default |
|---|---|---|
| `SOL_MEMORY_URL` | server address | `http://localhost:3456` |
| `SOL_MEMORY_BREATH_LIMIT` | entries to surface | `10` |
| `SOL_MEMORY_BREATH_TIMEOUT` | HTTP timeout (s) | `3` |
| `SOL_MEMORY_BREATH_CLI_TIMEOUT` | CLI fallback timeout (s) | `30` |
| `SOL_MEMORY_SKIP_BREATH=1` | disable the hook temporarily | - |

The `/breath-hook` endpoint itself is **read-only** and never activates memories — this avoids a self-excitation feedback loop. To activate memories deliberately, have the LLM call the `extmcp_breath` tool.

## Configuration

| Env var | Description | Default |
|---|---|---|
| `OLLAMA_BASE_URL` | Ollama address | `http://localhost:11434` |
| `OLLAMA_MODEL` | summarization / extraction model | `gemma4:e4b` |
| `OLLAMA_EMBED_MODEL` | embedding model | `bge-m3` |
| `OLLAMA_TIMEOUT` | request timeout (s) | `180` |
| `DECAY_LAMBDA` | decay coefficient | `0.05` |
| `DECAY_THRESHOLD` | decay threshold | `0.3` |
| `BREATH_TOKEN_BUDGET` | breath output length budget | `3000` |
| `BREATH_PINNED_QUOTA` | pinned quota within breath | `2` |

## Database schema

Key columns of the `memories` table:

- Content: `id`, `key`, `content`, `category`, `importance`, `session_id`
- Time: `created_at`, `updated_at`, `last_active`, `last_breath_at`
- Emotion: `valence` (0-1), `arousal` (0-1)
- Lifecycle: `pinned`, `resolved`, `digested`
- Retrieval: `embedding` (BLOB, bge-m3 1024-dim float32)
- Activation: `activation_count` (REAL, bumped on retrieval/breath)

`memories_fts` is an FTS5 virtual table auto-maintaining the keyword index over `key + content`.

Companion tables for the agent side: `phone_status` (latest-N snapshots), `phone_events` (rolling 500), `backend_inbox` (`status` × `priority`, urgent rows are express-delivered by the agent's injector). Screenshots live on disk under `peeks/`, not in the DB.

The decay score formula lives in `_calc_decay_score()` in [memory_mcp.py](memory_mcp.py): a blend of importance, activation count, days since last activation, arousal, resolved/digested status and pinned state.

---

# 中文

[English](#sol-memory-mcp) | **中文**

一个轻量级的 MCP 记忆服务器，基于 SQLite 实现混合检索（关键词 + 向量），带衰减、情感坐标、生命周期管理（pinned / resolved / digested）和自省工具。它同时兼任一个常驻 HTTP 枢纽：iPhone 把状态、事件、截图投递到这里，后台 agent 再从这里读走。

## 功能

- 保存、搜索、列出、删除记忆条目
- BM25 关键词搜索 + bge-m3 向量余弦相似度融合排序
- Ebbinghaus 衰减公式 + 情感唤醒度加权
- 通过本地 Ollama 模型生成中文摘要 / 情感分析 / 记忆提取
- 支持 stdio（Claude Desktop）和 Streamable HTTP（手机远程访问）两种传输
- 拖拽式 Web UI 导入对话记录，自动按格式（Claude 官方 / 插件 / ChatGPT mapping）切换处理模式
- 手机感官端点：状态快照、事件时间线、截图投递箱——对 payload 形状全部宽容，全部自带滚动清理
- 后台收件箱：给后台 agent 留言，urgent 标记走即时插播

## 伴生项目

[nudge-agent](https://github.com/SolenmeChiara/nudge-agent) 是本项目的伴生进程——一个常驻后台的 Claude Code 实例，消费落在这里的一切。依赖关系刻意做成单向：

- **本仓库单独跑，就是一本好好的日记本。** 所有记忆功能不需要任何 agent。面向 agent 的扩展表（手机事件、截图、后台收件箱）照常接收和存储手机投递的数据；滚动清理在本侧自含，数据不会因为没有读者而无限膨胀。
- **agent 离开本仓库则会残掉，这是预期内的设计。** 记忆连续性、收件箱、紧急插播、全部手机感官，都长在这根纽带上。
- 计划中：检测不到 agent 时，面向 agent 的工具（`extmcp_send_to_backend`、后台向的 session 回溯）将对 MCP 客户端隐藏，单机安装永远不会暴露没有作用的开关。

## 快速启动

```bash
# stdio 模式（供 Claude Desktop 使用）
python memory_mcp.py --db ./memory.db

# HTTP 模式（端口 3456）
start_http.bat
# 或
python memory_mcp.py --http --port 3456 --db ./memory.db
```

> **安全提示**：HTTP 服务没有鉴权。请保持在 localhost / 私有组网（Tailscale 等）内使用，不要把 3456 端口暴露到公网。

> **WSL / Windows 警告**：绝不要让 WSL 边界两侧的进程直接打开同一个 SQLite 文件——WAL 共享内存跨不过 9P 文件系统，谁先开库谁独占，另一侧一律报 `disk I/O error`。让本服务跑在文件所在的那一侧，其他一切走 HTTP 端口。

## MCP 工具列表

| 工具 | 说明 |
|---|---|
| `extmcp_save_memory` | 保存/更新记忆，自动后台生成 embedding + 情感分析 |
| `extmcp_search_memory` | 关键词 + 向量混合搜索（命中后激活 activation_count） |
| `extmcp_list_memories` | 按更新时间倒序列出 |
| `extmcp_delete_memory` | 删除一条 |
| `extmcp_summarize_recent` | 生成最近 N 条记忆的中文摘要（`limit` 1-30，默认 10），同时激活引用记忆 |
| `extmcp_random_memories` | 随机抽取 4-10 条 |
| `extmcp_dream` | 自省，找出最相似的记忆对，提示该 resolve / digest 哪些 |
| `extmcp_grow` | 把日记 / 长文拆成 2-6 条独立记忆 |
| `extmcp_breath` | 主动呼吸：浮现高权重未解决记忆 + pinned 核心，按 0.3 折扣激活，6h 内同一条不重复 |
| `extmcp_recall_session` | 按 `session_id` 拉出该会话的完整记忆时间轴 |
| `extmcp_session_preview` | 速览最近几个对话的最后几条消息 |
| `extmcp_send_to_backend` | 给后台收件箱留言；`urgent=true` 请求即时投递（agent 的注入器每 30 秒轮询，直接打进 agent 的对话流） |

## HTTP 端点

| 方法与路径 | 用途 |
|---|---|
| `POST /mcp`、`GET /mcp` | Streamable HTTP 的 MCP 传输 |
| `GET /breath-hook` | 只读 breath 块（pinned + 衰减分靠前的未解决记忆）——不激活记忆，hook 安全 |
| `POST /import`、`GET /import` | 对话历史导入（GET 出 Web UI；POST 收 JSON 或本地路径） |
| `GET /import/status`、`GET /import/embed_status` | 导入任务进度 / 全局 embedding 队列深度 |
| `POST /phone-status` | iOS 快捷指令投递的手机状态快照。宽容解析：平铺或包裹的 JSON 都收，中文键名服务端映射，离谱数值置 NULL，原始报文存 `raw_json` 供调试 |
| `GET /phone-status` | 最近一条手机状态 |
| `POST /phone-event` | iOS 自动化的单条带时间戳事件（`{"event": "alarm_stopped", "detail": "…"}`）；滚动保留 500 行 |
| `GET /phone-event?hours=48&limit=20` | 近期事件，新的在前 |
| `POST /peek` | 手机上传的原始截图字节（see-screen 通道）；`peeks/` 下滚动保留 10 张，已 gitignore |
| `GET /peek/latest` | 最新截图的路径与新鲜度 |

## 批量导入对话历史

700MB 级 Claude 官方导出走命令行：

```bash
# 先 dry-run 看格式 + 对话数（不调 LLM）
python batch_import.py "path/to/conversations.json" --dry-run

# 跑前 5 个试水
python batch_import.py "path/to/conversations.json" --limit 5

# 全量（可能几小时）
python batch_import.py "path/to/conversations.json"

# 中断后从第 N 个对话续跑
python batch_import.py "path/to/conversations.json" --start 500
```

或者打开 [http://localhost:3456/import](http://localhost:3456/import)，小文件（≤30 MB）拖入，大文件粘贴本地绝对路径——服务器会自动启动后台任务，浏览器轮询进度。

## 补齐缺失的 embedding

后台 worker 来不及处理、或服务中断时，新插入的记忆 embedding 字段会留空，影响向量检索。手动补齐：

```bash
# 默认 4 worker 并行
python reindex_embeddings.py

# 想温柔点，少抢 ollama 资源
python reindex_embeddings.py --workers 2

# 只跑 5 条调试
python reindex_embeddings.py --limit 5
```

只处理 `length(embedding)=0` 的行，可以反复跑。连续 10 次失败会直接退出（避免静默 skip 掩盖问题）。

## SessionStart hook（可选）

让新会话开始时自动把高权重记忆注入 Claude Code 的上下文。在你的 `.claude/settings.local.json`（或 `~/.claude/settings.local.json` 全局生效）添加：

```json
{
  "hooks": {
    "SessionStart": [
      {
        "matcher": "startup|resume|clear",
        "hooks": [
          {
            "type": "command",
            "command": "python \"${CLAUDE_PROJECT_DIR}/.claude/hooks/session_breath.py\""
          }
        ]
      }
    ]
  }
}
```

Hook 脚本 `.claude/hooks/session_breath.py` 已随仓库提供。它会：
1. 先 GET `http://localhost:3456/breath-hook`（HTTP server 跑着时最快）
2. 失败时 fallback 到 `python memory_mcp.py breath` CLI 子命令（直接读 SQLite，不依赖 server）
3. 两条路径都失败也不卡住会话，但错误**完整**打到 stderr（不静默吞）

环境变量：

| 变量 | 说明 | 默认 |
|---|---|---|
| `SOL_MEMORY_URL` | server 地址 | `http://localhost:3456` |
| `SOL_MEMORY_BREATH_LIMIT` | 浮现条数 | `10` |
| `SOL_MEMORY_BREATH_TIMEOUT` | HTTP 超时（秒） | `3` |
| `SOL_MEMORY_BREATH_CLI_TIMEOUT` | CLI fallback 超时（秒） | `30` |
| `SOL_MEMORY_SKIP_BREATH=1` | 临时禁用 hook | - |

`/breath-hook` 端点本身**只读**，不会激活记忆——避免自激反馈。要主动激活记忆请让 LLM 调用 `extmcp_breath` 工具。

## 配置

| 环境变量 | 说明 | 默认 |
|---|---|---|
| `OLLAMA_BASE_URL` | Ollama 服务地址 | `http://localhost:11434` |
| `OLLAMA_MODEL` | 摘要 / 提取用模型 | `gemma4:e4b` |
| `OLLAMA_EMBED_MODEL` | embedding 模型 | `bge-m3` |
| `OLLAMA_TIMEOUT` | 请求超时（秒） | `180` |
| `DECAY_LAMBDA` | 衰减系数 | `0.05` |
| `DECAY_THRESHOLD` | 衰减阈值 | `0.3` |
| `BREATH_TOKEN_BUDGET` | breath 输出字数预算 | `3000` |
| `BREATH_PINNED_QUOTA` | breath 中 pinned 配额 | `2` |

## 数据库 schema

`memories` 表关键字段：

- 内容：`id`, `key`, `content`, `category`, `importance`, `session_id`
- 时间：`created_at`, `updated_at`, `last_active`, `last_breath_at`
- 情感：`valence` (0-1), `arousal` (0-1)
- 生命周期：`pinned`, `resolved`, `digested`
- 检索：`embedding` (BLOB, bge-m3 1024 维 float32)
- 激活：`activation_count` (REAL, 被检索/呼吸时累加)

`memories_fts` 是 FTS5 虚表，自动维护 `key + content` 的关键词索引。

agent 侧的伴生表：`phone_status`（近 N 条快照）、`phone_events`（滚动 500 行）、`backend_inbox`（`status` × `priority`，urgent 行由 agent 的注入器即时投递）。截图存磁盘 `peeks/` 目录，不进库。

衰减分数公式见 [memory_mcp.py](memory_mcp.py) 的 `_calc_decay_score()`：综合 importance、activation_count、距上次激活的天数、arousal、resolved/digested、pinned 状态。
