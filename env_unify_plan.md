# .env 统一管理导入/合并模型 —— 施工档案

日期：2026-07-30 ｜ 决策人：Sol（完整档）｜ 状态：已完工入库（实现+独立审查 PASS_WITH_NOTES，182+28+15 断言全绿）
后续追加：审查发现并修复 IMPORT_MODEL 对 ollama 不生效（_call_ollama 加 model 参数）、consolidate 默认模型对齐 3.5-flash-lite；额外收获 = 修复 /import 逐对话路径的三元组解包死亡 bug（单独 commit）。已知留档：.env 值后不可写行内注释（loader 不剥离）；IMPORT_CLOUD_TIMEOUT/OPENROUTER_TIMEOUT/GEMINI_TIMEOUT 未进 README；云端提取无条数上限，大文件导入=每 chunk 一次付费调用

## 背景

三条 LLM 链路各自为政，Sol 无法从一处看清/控制"自动导入用的是哪个模型"：

| 链路 | 现状 | file:line |
|---|---|---|
| 前端 `/import` 提取 | 固定本地 Ollama（OLLAMA_MODEL，默认 gemma4:e4b），云端不参与 | memory_mcp.py:2095, 4847 |
| 整理/合并 consolidate | LLM_BACKEND（默认 openrouter）+ OPENROUTER_MODEL，失败 failover ollama | consolidate_sessions.py:38-43, 220 |
| batch_import.py CLI | --provider ollama\|gemini 只认命令行；gemini 直连 key=GOOGLE_AI_STUDIO_KEY 读 .env | batch_import.py:643, 664 |

.env 加载机制已存在：memory_mcp.py HTTP 启动时 `_load_dotenv`（5062-5074，CLI 参数优先、否则 .env 赢）；batch_import.py `_load_env_var`（55-77，进程 env 优先、逐级向上找 .env）。

## 目标（完整档）

一切模型/provider 选择集中到 `D:\ClaudeExtentions\MCP\Sol-Memory-mcp\.env`：

```ini
# ===== 云端 key =====
OPENROUTER_API_KEY=<已有>
GOOGLE_AI_STUDIO_KEY=<Sol 自填；缺省则 gemini 直连不可用>

# ===== 导入提取（前端 /import 与 batch_import.py CLI 共用默认）=====
IMPORT_PROVIDER=ollama        # ollama | openrouter | gemini
IMPORT_MODEL=                 # 空 = 按 provider 取各自默认

# ===== 整理/合并 =====
LLM_BACKEND=openrouter        # openrouter | ollama（既有变量，照旧）
OPENROUTER_MODEL=google/gemini-3.5-flash-lite

# ===== 本地 Ollama（既有变量，照旧）=====
OLLAMA_MODEL=gemma4:e4b
OLLAMA_EMBED_MODEL=<既有>
```

## 改动清单

1. **memory_mcp.py**：`/import` 提取处（2095、4847）由直调 `_call_ollama` 改为 provider 分发：
   - ollama → 现有 `_call_ollama`（默认路径，行为不变）
   - openrouter → 新增 `_call_openrouter`（OpenAI 兼容 chat/completions + Bearer 头，~30 行，模型取 IMPORT_MODEL 或回落 OPENROUTER_MODEL）
   - gemini → 复用 `batch_import._call_gemini`（已 import batch_import，模型取 IMPORT_MODEL 或 GEMINI_DEFAULT_MODEL）
   - 云端单 chunk 失败 → 该 chunk 降级 `_call_ollama` 兜底一次，再失败记 errors 照旧
   - IMPORT_PROVIDER/IMPORT_MODEL 进 `_load_dotenv` 后的刷新段（5062-5074 同款模式）
2. **memory_mcp.py 前端**：stats 端点（4647 一带）新增 extract_provider / extract_model / consolidate_backend / consolidate_model 字段；`/import` 页面显示"提取模型：xxx ｜ 合并模型：xxx"；analysis_ready 就绪判定按 provider 对应检查（gemini → GOOGLE_AI_STUDIO_KEY，openrouter → OPENROUTER_API_KEY，ollama → 可达性）
3. **batch_import.py**：--provider/--model 缺省时读 IMPORT_PROVIDER/IMPORT_MODEL（`_load_env_var`），命令行显式传参仍最高优先；provider choices 增加 openrouter（新增 `_call_openrouter` 同款）
4. **.env**：重写为上述分区注释版（保留现有真实 key 值；.env 在 gitignore 里，不入库）
5. **README.md**：配置一节同步；顺带补 save_memory 部分更新语义一句
6. **consolidate 链路不动**（已是 env 驱动，openrouter/ollama + failover 保持）

## 红线

- save_memory 修复（工作区 memory_mcp.py 既有 diff）先提交，本期改动绝不混入同一 commit
- batch_import.py 遗留改动（GEMINI_DEFAULT_MODEL→3.5-flash-lite、8192→20000）Sol 拍板保留、随本期一起提交
- .env 真实 key 不进任务书/报告/commit
- 验证照旧：临时库 + 桩函数 + 裸 SQL；云端 provider 用桩 HTTP 或 monkeypatch，不实际烧配额

## 流水线

主循环写任务书 → opus 实现（后台）→ 主循环亲自审查 diff + 实测 → 独立 opus 验证 → 主循环提交推送 → 提示 Sol 重启 3456 服务
