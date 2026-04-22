# Sol-Memory-mcp

一个轻量级的 MCP 记忆服务器，基于 SQLite 实现混合检索（关键词 + 向量），带衰减、情感坐标、生命周期管理（pinned / resolved / digested）和自省工具。

## 功能

- 保存、搜索、列出、删除记忆条目
- BM25 关键词搜索 + bge-m3 向量余弦相似度融合排序
- Ebbinghaus 衰减公式 + 情感唤醒度加权
- 通过本地 Ollama 模型生成中文摘要 / 情感分析 / 记忆提取
- 支持 stdio（Claude Desktop）和 Streamable HTTP（手机远程访问）两种传输
- 拖拽式 Web UI 导入对话记录，自动按格式（Claude 官方 / 插件 / ChatGPT mapping）切换处理模式

## 快速启动

```bash
# stdio 模式（供 Claude Desktop 使用）
python memory_mcp.py --db ./memory.db

# HTTP 模式（端口 3456，开放 /mcp、/、/import、/breath-hook）
start_http.bat
# 或
python memory_mcp.py --http --port 3456 --db ./memory.db
```

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

衰减分数公式见 [memory_mcp.py](memory_mcp.py) 的 `_calc_decay_score()`：综合 importance、activation_count、距上次激活的天数、arousal、resolved/digested、pinned 状态。
