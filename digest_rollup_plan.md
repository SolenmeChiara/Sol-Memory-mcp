# digest_rollup 施工档案（2026-08-29）

## 需求（Sol 原话摘要）
记忆碎片太碎。要一个分层递归总结：每周据本周记忆写周总结；每月/每季/每半年据下层总结写上层总结。跨度越大权重越高。总结不影响原碎片，纯增量梗概层。LLM 用 `claude -p` 走套餐（haiku/sonnet/opus）。

## 与现有 consolidate 的区别
- `consolidate_sessions.py`：按 session 有损合并，源碎片标 digested=1，90 天后物理删除。
- 本工程：按自然时间窗口无损总结，碎片一律不动，总结另起条目。两者共存。

## 设计定稿
- 新文件 `digest_rollup.py`（Windows 侧运行，直连 SQLite，参照 consolidate_sessions.py 骨架）。
- 层级：周（[周记] 2026-W35）→ 月（[月记] 2026-08）→ 季（[季记] 2026-Q3）→ 半年（[半年记] 2026-H2）。
- 级联输入 + 防传话混合：月记 = 当月周记全文 + 当月 importance top10 原始条目；季、半年同理。
- 权重：importance 周 0.55 / 月 0.65 / 季 0.75 / 半年 0.85；tier 全部留空进普通层。
- 吸收即归档：月记生成成功后把被覆盖的周记 set tier='archive'（同时清 tier_until）；碎片永不动。
- 总结条目标记：consolidated=1（防被 consolidate 二次抓取）、digested=0（防 90 天清理）、session_id='digest_rollup'。
- 幂等：按 key 查重跳过；--force 原地重写（保留 created_at/tier/tier_until/last_active/activation_count/pinned/resolved，清 embedding）。
- 补漏：枚举已完结缺失周期，默认回补 4 个（只计消耗 LLM 的），--backfill-all 全补；父层等下层补齐（--no-wait-children 可关）。
- 门槛：当期活跃记忆 < 5 条跳过。
- LLM：`claude -p --model <haiku|sonnet|opus>`，prompt 走 stdin，600s 超时重试一次。源 id 由代码拼接（防幻觉），content 尾部最多列 8 个。
- 输入预算：--max-input-chars 120000 / --max-entry-chars 1200，超预算按 importance 从低到高丢并在 prompt 声明。
- 周界：America/Toronto 周一起始；created_at 用 19 字符前缀字符串比较（三种库内时间戳形态均正确，吃 idx_memories_created 索引）。
- 全失败退出码非 0（计划任务能看到真实错误）。
- breath 曝光（2026-08-29 晚 Sol 提出）：新总结生成时进 watch 层（周 8 / 月 12 / 季 16 / 半年 20 天），到期锚在周期结束时刻——历史回补的旧周期窗口已过，直接进普通层不打扰；到期自动降 archive 与被上层吸收归档两条既有路径都通。临时库已验证（新鲜 W34 进 watch until 9/1、过期 W32 进普通层）。注意：本次历史回补进程用的是改动前代码，回补完成后需手动把 W34 提 watch。

## 实现（opus 代理，2026-08-29）
951 行纯标准库。关键侦察结论：created_at 全库 UTC 三形态；FTS 由触发器自动维护；embedding 留空待 reindex_embeddings.py（无后台 worker 误捡）；tier='archive' 须同步清 tier_until。测试：离线周期数学、真实三档模型调用、注入安全 13 载荷、Windows 真机、生产库零写入核对。

## 审查（独立 opus 代理，2026-08-29）：PASS_WITH_NOTES
- 数据安全字节级验证：完整级联 78 条后，11785 条原始记忆全部 21 列 SHA-256 前后一致；FTS 零重影；归档 UPDATE 双保险（session_id+id）不误伤同名手写条目。
- breath 实测：总结条目 activation_count=1 使 decay 垫底（201 条池中排 198-201），正常不浮现——总结层对 breath 是只写不读，检索可达。
- 审查顺手修：--force 后 embedding 不清导致语义检索永远命中旧文本 → UPDATE 加 embedding=X''。
- 契约核对全过：digested 清理（memory_mcp.py:1768）只删 digested=1；consolidate 候选（consolidate_sessions.py:393）被 consolidated=1 挡；DDL 21 列逐列对照无误。

## 主循环裁决与收尾小改（审查后）
- 全失败退出码 0 → `return 1 if totals["failed"] else 0`（修）。
- --force 重置 pinned/resolved → 从 UPDATE SET 摘除，保留用户手动状态（修）。
- 源记忆尾巴占正文六成 → SOURCE_MAX_SHOWN 30→8（修）。
- period_emotion 不吃 --exclude-tier → 不改，遗留（情绪反映全期 vs 喂入材料是设计争议，影响面小）。
- 历史回补策略：--exclude-tier seabed（海床周自动落门槛跳过，真实记忆从 2026-W09 起步），计划任务同样带此参数。

## 部署记录（2026-08-29，Sol 批准：回补现在跑 + 注册周一任务）
- [x] 即时备份：backups/memory-pre-digest-20260829.db（sqlite backup API 在线快照，11785 行，quick_check ok，496MB，亲眼验证）
- [x] Windows 侧生产库 dry-run：23 周 + ~8 月 + 3 季 + 2 半年待生成，层级闸门正常
- [x] 回补启动：PowerShell Start-Process 脱离进程，日志 logs/backfill-20260829.log，W09 已 [ok]
- [x] 计划任务注册：SolMemoryDigestRollup，每周一 12:15，调 digest_task.bat（python/claude 路径写死，--db 显式，日志追加 logs/digest.log；仅登录时运行）
- [x] 回补完成（19:54）：36 篇全部生成零失败（30 archive + 6 在途），层级闸门单次运行内全串起；FTS 计数与主表一致；总行数差值吻合
- [x] 三篇 sonnet 月记（04/06/07）开头有字数自审元话语，手术剥除（正文完好）；clean_output 加精确匹配兜底规则（单测三剥三留全过）
- [x] 发现 claude -p 默认 agentic：sonnet 生成月记时往生产目录写了 draft_digest.txt 草稿（已删）——命令加 --disallowedTools 禁掉全部改环境工具，真实调用验证通过
- [x] reindex_embeddings.py：37 条全部成功（qwen3-embedding，5 条/秒）
- [x] W34 手动提 watch（until 2026-09-01T04:00:00Z，与新代码逻辑一致）
- [x] breath 实测：WATCH 段完整曝光 W34 带到期尾巴，其余各段无挤占
- [ ] 一周后确认计划任务真实产出（2026-08-31 周一 12:15 首跑，应生成 [周记] 2026-W35 并自动进 watch）

## 扩层：补上日 / 半月 / 年（2026-08-30，Sol 拍板七级链）
- 新键：`[日记] 2026-08-29`（imp 0.45 / haiku / watch 2 天 / 门槛 3）、`[半月记] 2026-08上|下`（1–15 与 16–月末，imp 0.60 / sonnet / watch 10 天）、`[年记] 2026`（imp 0.95 / opus / watch 30 天）。其余标记（category=digest、session_id、consolidated=1、digested=0、tier 规则）全部沿用。
- 两套父子关系从此分家：`INPUT_CHILD`（谁读谁）半月←日、月←周、季←月、半年←季、年←半年；`ABSORB_CHILD`（谁归档谁）在此基础上加周←日，减半月（半月谁也不吸，靠 watch 到期自己退场）。月记继续吃周记不吃半月记——周跨 15/16 边界，塞进链里会重复计账。
- 周期数学重构成 `period_containing` + `next_period` 两个原语，`enumerate_periods` 与 `sub_periods` 都由它们生成；旧的四层逐层分支删掉，离线单测逐条比对确认与 HEAD 输出完全一致。
- 三个新层各带枚举窗口，因为「库里有史以来每个已完结周期」对它们是错的默认：日只看最近 `--day-lookback`（3）天且**不被 --backfill-all 解除**（回补一年日记是几百次调用）；半月只看最近 `--halfmonth-recent`（1）个，--backfill-all 解除；年只生成 `--year-min`（2026）及以后，2026 年到 2027-01-01 自然生成。
- 计划任务从每周一改每天 12:15（`digest_task.bat` 同步改注释并加时间戳分隔行）。一个时段够全链：12:15 时昨天已完结，周一那天上周也已完结。
- 生产库快照 dry-run 预测首跑：3 篇日记（8/27 13 条、8/28 24 条、8/29 149 条，prompt 最大 5.9 万字符）＋ 1 篇半月记（2026-08上，0 篇日记 + 10 条碎片），其余层全是历史低于门槛的 skip，零 LLM 调用。

## 遗留风险（如实）
- 真实模型字数超标（周记要求 600 实出 1000+），prompt 压不住；观察 breath 实际表现再决定是否两段式压缩。
- 真实 LLM 元话语越界形态不止一种，clean_output 只兜住分隔线形态。
- 与常驻 MCP 并发写未实测（WAL + busy_timeout 理论够用）。
- W17 等重周超预算丢弃条目，周记是残卷（prompt 内已声明丢弃数）。
- dry-run 看不到父层预览（需 --dry-run --no-wait-children）。
- 扩层后新增：周日（以及周六）的日记会在周记生成的同一轮里被立刻归档，2 天 watch 窗被削掉——规格里「周吃日归档日」与「日 watch 2 天」本身冲突，按规格保留归档语义，材料由进 watch 的周记继续覆盖，一周有 5 天能拿满 2 天曝光。
- 日记层实测输出超标（要求 400 字，haiku 实出 600+），与周记同一个老毛病；且模型仍不完全服从「全角标点」。
- 未提交 git：digest_rollup.py、digest_task.bat、本档案（回补验证后统一提交）。

## 2026-08-30 七层扩层部署记录

- 审查：独立 opus 审查 PASS_WITH_NOTES（键名契约咬合、两级 kill switch 逐字节还原 HEAD、周期数学 6158 断言全等、热路径 0.047ms/call；审查中修复 digest_task.bat CRLF 丢失）
- 备份：memory-pre-digest7-20260830.db（11841 行，474MB，backup API）
- 提交：Sol-Memory-mcp fb96c89 / nudge-agent 987bed8，均已 push
- 部署：3456 重启（旧 8332 → 新实例）→ 计划任务 schtasks 改 DAILY 12:15 → rollup 首跑 → 注入器重启（旧 16120 → 16588，48765 锁单实例验证）
- rollup 首跑：生成 9 篇 0 失败——3 日记（8/27-29）+ 4 周记 2025（W27/29/30/31，W28 空窗跳过）+ 1 半月记 2026-08上 + 1 月记 2025-07（同轮级联：吃掉刚生成的 4 篇周记并 absorb 归档，实战验证 absorb 链）。reindex_embeddings 补齐 10/10
- 部署后验证：CHRONICLE 三行齐（日记 8/29 + 周记 W34 + 月记 2026-07，年记 2027 才有）；日记 tier 按周期锚精确落位（8/29、8/28 watch，8/27 过窗进普通层）；dream 抽样 0 digest；历史回补条目进普通层不占曝光
- 遗留：注入器侧 CHRONICLE 进 context 的验证等下一轮唤醒；2025 历史周记还缺 ~33 篇，每日任务按 4 篇/日节奏自动补齐（约 8 天）；审查遗留意见（半月/季/半年记占 WATCH 槽、_row_to_record 裸调用）记录在 commit fb96c89 正文，未动手
- 新工具：nudge-agent/restart_injector.sh——WSL 一条命令重启 Windows 注入器（注入器不可迁入 WSL：drvfs WAL + pc_status Windows API 两条硬依赖）
