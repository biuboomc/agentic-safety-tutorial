# 进展与路线图

## 当前已实现能力
- 已跑通离线 `TaskDraft` 生成、`EnvAssembler` 打包 `RuntimeTaskBundle`、HTTP server 动态执行工具、终局 success check 与步骤级 utility 评分。
- 已支持 placeholder 与 LLM 两条生成路径，LLM 侧可生成 planner、query、tool code、checklist。
- 已支持 bundle 校验、工具源码 AST 校验、子进程执行与超时隔离、per-episode lock、安全扰动占位层。
- 已支持 server 常驻后的 catalog 动态注册：`TaskDraft` / `RuntimeTaskBundle` 可后注册，episode 可按 `task_id` 随时启动。
- 已有 `smoke_test.py` 与 `llm_smoke_test.py` 两条主验证链路。

## 当前明确限制
- 当前仍保留少量默认 demo tool pool 作为回归测试样例，但主路径已经转向 planner-first 的外部 tool pool 驱动。
- checklist runtime evaluator 已改为消费 bundle/rule，但仍需继续收紧 planner-first 约束与生成质量。
- 工具执行仍是受限模拟器，不连接真实外部系统。
- server 适合小规模 rollout 和 debug，还不是生产级长稳服务。

## 后续方向与优先级
1. 去模板化改造
   - 继续收敛到 `TaskPlanSpec` 为中心的 planner-first 结构。
   - 移除残留的 demo-only 推断与 tool-name 特判。
   - 让 tool code / checklist / success rule 都以 plan spec 为唯一上游真相。
2. 生成质量优先
   - 强化 planner 与 query、tool code、checklist、success rule 的一致性。
   - 提升 LLM 生成代码的 return schema 稳定性与 dry-run 成功率。
3. 运行时强化
   - 继续补强错误协议、生命周期管理、容量控制与可观测性。

## 建议执行里程碑
### 里程碑 1：Planner-First 主骨架
- 新增统一的 `TaskPlanSpec`。
- 让 query、tool code、checklist 都只消费 `TaskPlanSpec`。
- `EnvAssembler` 仅负责 contract 对齐与 bundle 组装。

### 里程碑 2：去掉模板与特判
- 移除 `travel` / `retail` 业务模板在主路径中的地位。
- 移除按 tool name 猜协议的逻辑。
- 让 success/checklist 规则默认来自 planner 输出。

### 里程碑 3：校验与修复闭环
- tool code 生成后执行 AST、contract、dry-run、failure-branch 校验。
- checklist item 与 success rule 也执行规则校验。
- 校验失败时将结构化错误反馈给 LLM 做有限轮次修复。

## 去模板化改造
### 目标
- 把当前“demo 模板驱动”的 pipeline 改成 planner-first 的通用生成链。
- 让 LLM planner 先产出统一的 `TaskPlanSpec`，其中包含任务意图、tool protocol、return schema、sample args、validation hints、initial state blueprint、success rule、checklist item 与 runtime rule。
- 后续的 tool code、user query、checklist 都只允许基于这份 plan 生成。
- 系统中唯一保留的固定部分是 bundle 字段、server 执行 contract、runtime rule DSL 与可执行性校验。

### 当前阶段
- 正在进行 Planner-First 主骨架替换与模板清理。

### 已完成
- 为 `TaskDraft` 增加 `task_plan_spec`，并新增 `TaskPlanSpec` 的基本 contract 校验。
- 将 `generator.py` 重构为基于 tool pool 的 plan seed builder，不再依赖 `travel/retail` 场景 builder 作为主路径。
- 将 `evaluation_hints.py` 改为消费 checklist item 自带的 `runtime_rule`，移除 travel/retail 规则工厂。
- 将 `llm_generator.py` 改为 planner-first 结构：
  - 先生成完整 `TaskPlanSpec`
  - 再基于 plan spec 生成 query、tool code、checklist
  - 不再让各模块单独猜业务语义
- 已加入 `ToolGenerationValidator`：
  - AST/安全校验
  - contract 校验
  - sample-args dry-run
  - failure-branch dry-run
- 已加入 tool code repair loop：生成失败时会将结构化错误反馈给 LLM 进行有限轮次修复。
- 已修复 runtime 工具执行时的命名空间问题，确保生成 / 占位工具函数可以访问 `TOOL_METADATA`。
- 已让本地 `smoke_test.py` 在 planner-first 主路径下重新通过。
- 已让 `llm_smoke_test.py` 在未配置 LLM endpoint 时显式跳过，而不是误报失败。
- 已为 CLI 增加默认开启的进度条，覆盖 `generate-drafts`、`assemble-bundles` 和 `generate-examples`。
- 已将 LLM augment 进度细化到 draft 内部阶段：`planner / query / checklist / tool code j/k`。
- 已将 LLM token usage 统计接入进度显示与 CLI 输出汇总，便于观察生成成本。

### 进行中
- 继续移除 `llm_generator.py` 中残留的保守回退逻辑。
- 继续收紧 success rule 与 checklist rule 的 planner-first 约束。
- 将 README、CLI 帮助和示例数据继续对齐到 planner-first 说明。

### 下一步
1. 在远端 worker 上用真实 tool pool 验证新的 planner-first 流程与 CLI 进度条。
2. 用 `tools.zip` 中清洗后的 tool pool 跑通 `tool_pool -> TaskPlanSpec / TaskDraft -> RuntimeTaskBundle`。
3. 继续清理 demo-only 兼容代码，仅保留最小回归测试入口。

## 2026-03-20 Update
### 目标
- 收紧 LLM prompt，使其显式遵守 server 的有限状态模拟器约束，而不是尝试真实实现外部系统。
- 收紧 tool code 生成流程，避免出现 `selected_tools` 和 `tool_code_drafts` 数量不一致的半成品 draft。

### 当前阶段
- 已完成 prompt 约束增强，并将 tool code 生成改为任务级补齐重试。

### 已完成
- 在 planner prompt 中加入“有限状态模拟器、仅操作 bundle-maintained state、无外部系统”的明确约束。
- 在 tool code prompt 中加入：
  - 不是要实现真实后端
  - 只能读写少量声明过的状态键
  - 不允许双下划线/dunder 访问
  - 只能生成受限、确定性的状态模拟逻辑
- 在 tool code 生成阶段加入任务级重试：
  - 若某个 selected tool 首轮未生成出合法代码，会带着上一轮校验问题再次重试
  - 直到补齐全部 selected tools，或耗尽重试预算
- 在 draft 中记录 `llm_generation_diagnostics`，便于定位缺失 tool 和对应校验问题

### 进行中
- 继续观察不同模型在新 prompt 下的生成稳定性，尤其是 `gpt-5.4` 的 dunder 风格代码是否会明显下降。

### 下一步
1. 用 `gpt-5.4` 和 `qwen3.5-plus` 重新对比外部 tool pool 的生成完成率。
2. 若仍有模型频繁生成不合规代码，继续收紧 repair loop 的反馈内容。
3. 将运行结果同步到远端并更新 README 中的“生成质量与约束”说明。

## 2026-03-20 Parallelization Update
### 目标
- 提升 LLM 生成吞吐量，支持 task 之间并行，以及 task 内无依赖阶段并行。
- 在不改变 bundle / server contract 的前提下，缩短 `generate-drafts` 的墙钟时间。

### 当前阶段
- 已完成本地并行化改造，正在做本地回归与远端同步验证。

### 已完成
- 为 `LLMGenerationConfig` 增加并行参数：
  - `task_parallelism`
  - `tool_parallelism`
  - `query_checklist_parallelism`
- `build_llm_static_task_drafts(...)` 现已支持 task 级并行 augment，并保持输出顺序稳定。
- `augment_draft(...)` 现已支持：
  - `query` 与 `checklist` 的并行生成
  - 不同 tool code 的并行生成
  - 缺失 tool code 的并行补齐重试
- CLI 已暴露并行参数，便于直接控制任务级和工具级并发。
- 进度条与 token 统计继续保留，并可在并发场景下安全刷新。

### 进行中
- 本地回归测试与远端 worker 上的真实模型生成验证。

### 下一步
1. 完成本地 `py_compile` 与 smoke test。
2. 同步到远端仓库与 worker。
3. 用真实 `tool_pool` 和 `qwen3.5-plus / gpt-5.4` 验证并行生成吞吐与稳定性。

## 2026-03-20 Source Task Converter Update
### 目标
- 新增一个转换器模块，将已有的 `source task + tools` 转成 planner-first 的 `TaskPlanSpec / TaskDraft / RuntimeTaskBundle`。
- 明确转换顺序必须是：
  1. `source task -> plan/query/checklist`
  2. `plan/query/checklist -> tool code`
  3. `tool code + state/rules -> bundle`

### 当前阶段
- 已完成本地结构实现，正在准备远端 `qwen3.5-plus` 的真实转换测试。

### 已完成
- 新增 `tasksvc/generation/source_task_converter.py`：
  - 读取 `source_tasks.json`
  - 规范化 source task 输入
  - 基于 source task 和其 tool list 构造 seed `TaskPlanSpec`
  - 调用现有 planner-first LLM augmenter 完成 plan/query/checklist/tool code 生成
- 在 planner prompt 中显式加入 `source_task_input`，确保 plan/query/checklist 是从已有任务出发，而不是只依赖通用 seed。
- 新增 CLI 子命令：
  - `convert-source-tasks`
- 新增样例输入文件：
  - `examples/source_task_conversion_examples.json`
  - 包含来自 AgentDojo / OpenAgentSafety / MT-AgentRisk 的 source-derived 示例任务
- 新增本地 smoke test：
  - `tests/source_task_converter_smoke_test.py`
  - 已验证 `source task -> draft -> bundle -> server call` 的结构闭环

### 进行中
- 远端 worker 上用 `qwen3.5-plus` 真实生成并验证这三条 source task 的 bundle 可运行性。

### 下一步
1. 用 `convert-source-tasks --backend llm` 在远端运行三条 source task。
2. 检查生成后的 bundle 是否都能注册进 server 并成功启动 episode。
3. 记录真实 token usage、生成耗时和主要失败模式，继续收紧 planner-first 转换质量。

## 2026-03-20 Strict Source Conversion Validation
### 目标
- 让 source-task conversion 这条线严格保持 benchmark 任务的一致性。
- 对 source task 转换路径固定两条约束：
  - `user_query` 直接等于原始 source task 文本。
  - `selected_tools` 直接等于 source task 提供的工具列表。
- 仅允许 LLM 生成：
  - `plan`
  - `checklist`
  - `tool code`

### 当前阶段
- 本地实现已完成，并已同步到远端共享目录；远端 worker 已完成 `qwen3.5-plus` 实测验证。

### 已完成
- 为 source-task conversion 新增独立 augmentation 路径，不再复用会改写 query 或重选 tools 的通用 planner 路径。
- `augment_source_task_draft(...)` 现在只生成：
  - source-faithful `plan`
  - source-faithful `checklist`
  - contract-aligned `tool code`
- `convert-source-tasks` CLI 已写出新的严格产物目录：
  - `examples/converted_qwen_strict/converted_source_tasks_drafts.json`
  - `examples/converted_qwen_strict/converted_source_tasks_runtime_catalog.json`
- 远端 worker 上已验证：
  - 3/3 tasks 的 `user_query == source task text`
  - 3/3 tasks 的 `selected_tools == provided source tools`
  - 生成出的 runtime catalog 可以注册到 server 并成功进行至少 1 次 tool call
- 本次 `qwen3.5-plus` 转换统计：
  - `prompt_tokens = 14261`
  - `completion_tokens = 52647`
  - `total_tokens = 66908`

### 进行中
- 继续观察 source-faithful checklist 的粒度是否足够贴近 benchmark 原任务要求。

### 下一步
1. 将 strict source-task conversion 的用法和保证写入 README/CLI 帮助文案。
2. 增加 source-task consistency validator，对 query/tool/checklist 一致性做显式校验。
3. 用更多 benchmark 风格 source tasks 验证 source-faithful 转换质量。

## 2026-03-20 Agent Rollout Loop
### 目标
- 增加一条 benchmark-style 的运行链：`register bundle -> start episode -> give user_query to agent -> parse tool call -> call server -> feed observation back`.
- 让这条链显式对齐公开 benchmark 常见的 action-observation loop，而不是只停留在“环境可被 tool-call”层面。

### 当前阶段
- 本地最小闭环已实现并通过 smoke test，正在补充真实模型远端验证与 benchmark 对照说明。

### 已完成
- 新增 `tasksvc/runtime/agent_rollout.py`，封装：
  - bundle 注册
  - episode 启动
  - agent reasoning turn loop
  - tool-call 解析
  - server tool-call 执行
  - tool result / observation 回注
- `OpenAICompatClient` 已支持把 `tools` 和 `tool_choice` 直接传给 OpenAI-compatible `/chat/completions`。
- 新增 `parse_tool_calls_from_message(...)`：
  - 支持原生 `tool_calls`
  - 支持文本 JSON / fenced JSON / `<tool_call>` 风格回退解析
- CLI 新增 `run-agent-episode` 命令。
- 新增 `tests/agent_rollout_smoke_test.py`，验证：
  - 注册 runtime catalog
  - 启动 episode
  - 解析 agent tool call
  - 调用 server
  - 将 tool result 回注到下一轮推理

### 进行中
- 用真实 LLM 和远端 worker 再补一轮带 server 的 rollout 验证。

### 下一步
1. 在远端 worker 上用 `qwen3.5-plus` 跑一条真实 `run-agent-episode`。
2. 把 benchmark 对照总结写回 README 或单独文档。
3. 视结果决定是否加入更强的工具调用解析约束或 transcript 导出功能。

## 2026-03-20 AgentDojo 629 Conversion
### 目标
- 使用 `AgentDojo v1`（论文对齐的 629-case 版本）抽取 benchmark case，并通过现有 source-task conversion 流程批量生成 `RuntimeTaskBundle`。
- 将输出统一落到 `/mnt/shared-storage-user/chenguanxu/workspace/my_works/2026/bundlebench/agentdojo`，供后续 bundle bench 使用。

### 当前阶段
- 已确认 `v1` 是 629-case 版本，正在同步代码到远端并准备在 worker 上发起批量转换。

### 已完成
- 新增 `tasksvc/generation/agentdojo_source_tasks.py`，支持从 AgentDojo 仓库提取 benchmark cases 为 `source_tasks`。
- CLI 新增 `convert-agentdojo-benchmark`，可直接执行 `AgentDojo repo -> source_tasks -> task_drafts -> runtime_catalog`。
- 已在 host 上通过 AgentDojo suite API 确认：
  - `v1` / `v1.1.x` = 629 cases
  - `v1.2.x` = 949 cases
- 已将默认 benchmark version 收紧为 `v1`，并更新 CLI 帮助文案。
- 已在 host 上创建输出目录：
  - `/mnt/shared-storage-user/chenguanxu/workspace/my_works/2026/bundlebench`
  - `/mnt/shared-storage-user/chenguanxu/workspace/my_works/2026/bundlebench/agentdojo`

### 进行中
- 将本地 AgentDojo 转换入口同步到远端工作目录，并通过 worker + `qwen3.5-plus` 发起大规模转换。

### 下一步
1. 将当前代码同步到远端共享工作目录。
2. 在 worker 上运行 `convert-agentdojo-benchmark --benchmark-version v1 --backend llm`。
3. 记录进度日志、token 统计和最终输出文件路径。

## 2026-03-20 Checkpoint / Resume
### 目标
- 为大规模 `source_tasks -> task_drafts -> runtime_catalog` 转换增加断点续跑能力。
- 使中途终止后可以跳过已经完成的 task，仅补剩余任务。
- 为下一轮 AgentDojo 629-case 转换准备更高的 task 级并发（目标 `task_parallelism=32`）。

### 当前阶段
- 已停止上一轮无 checkpoint 的 worker 任务，正在实现 per-task draft checkpoint 与 resume。

### 已完成
- 明确断点边界应落在 `source_task_id -> completed draft` 这一层，而不是最终大 JSON 一次性写盘。
- 设计为：
  - `checkpoint_dir/manifest.json`
  - `checkpoint_dir/drafts/<source_task_id>.json`
- 已决定下一轮 AgentDojo 重跑时采用更高的 task 级并发：`32`。

### 进行中
- 将 `convert-source-tasks` 和 benchmark 转换命令接入 checkpoint/resume。

### 下一步
1. 完成 per-task checkpoint 持久化与 resume 跳过逻辑。
2. 用本地 smoke test 验证 resume 真正加载旧 draft，而不是重跑所有任务。
3. 同步到远端后，以 `task_parallelism=32` 重启 AgentDojo v1 629-case 转换。

## 2026-03-22 Source Conversion Benchmark Alignment
### 目标
- 让 source-task conversion 这条线严格对齐 benchmark 的 benign 任务评测，不再把 injection 目标混入最终 goal/checklist。
- 保持：
  - `user_query == source task 原文`
  - `selected_tools == source task 提供的 tools`
- 仅让 LLM 生成：
  - `plan`
  - `checklist` 的自然语言表述
  - `tool code`

### 当前阶段
- 已完成本地实现，正在做本地回归并准备重新在远端以 `qwen3.5-flash` 批量生成 AgentDojo v1 的 629 条 bundle。

### 已完成
- 在 source-task conversion 专用路径中新增基于 `metadata.user_ground_truth_calls` 的机械对齐逻辑：
  - `success_rule` 直接由 benign ground-truth call sequence 推导
  - checklist 的 `runtime_rule` 直接由 benign ground-truth calls 推导
- 新增 runtime rule DSL：
  - `history_call_matches`
  - `history_call_sequence_contains`
- runtime evaluator 已支持按历史 tool call 与参数匹配来判定 checklist progress 和 final success。
- `llm_generator.py` 的 source-task conversion prompt 已收紧为：
  - 仅面向 benign benchmark objective
  - 显式忽略 paired injection goal
  - checklist 只允许改写问题/通过条件文案，不允许改写 runtime_rule 语义
- 新增 source conversion 专用一致性校验：
  - query 必须保持原文
  - selected tools 必须保持原顺序
  - success_rule 必须对齐 benign ground-truth calls
  - checklist runtime_rule 必须对齐 benign ground-truth calls
- `tests/source_task_converter_smoke_test.py` 已新增严格对齐场景，覆盖：
  - benign ground-truth success rule
  - checklist runtime_rule 对齐
  - server 运行时按正确调用序列判成成功

### 进行中
- 跑完整的本地/远端回归，并观察是否仍有历史 checkpoint 产物与新规则不兼容。

### 下一步
1. 通过本地 smoke tests 与必要的运行时回归。
2. 同步到远端共享目录。
3. 在 worker 上用 `qwen3.5-flash`、`task-parallelism=16` 重新生成 AgentDojo v1 的 629 条 bundle。
## 2026-03-23 Source Conversion Hybrid Checklist Consistency
### Goal
- Relax checklist validation only for the source-task conversion path.
- Keep `user_query`, `selected_tools`, and `success_rule` strict.
- Allow checklist items to be effect-consistent with the benign benchmark goal instead of process-identical to every ground-truth call rule.

### Current Stage
- Updating the source conversion validator and regression tests, then resuming the 2 failed AgentDojo cases on the worker.

### Completed
- Checklist final item may now use either the exact benign success rule or `episode_success`.
- Checklist intermediate items no longer need exact runtime-rule equality with the benign ground-truth sequence.
- The validator now requires checklist coverage of benign tools and rejects injection-only tool references.
- Added regression coverage for relaxed benign checklists and explicit failure on injection-only checklist leakage.

### In Progress
- Running local regression tests and preparing a remote resume for the remaining 2 AgentDojo failures.

### Next
1. Re-run local smoke tests.
2. Sync only the source conversion changes to the shared remote repo.
3. Resume the AgentDojo v1 qwen3.5-flash conversion to recover the final 2 tasks and write the final runtime catalog.

## 2026-03-23 Validator Relaxation For Finite-State Tool Updates
### Goal
- Unblock bundle assembly for source-converted tasks whose generated tool code uses safe in-memory list updates such as `list.remove(...)`.
- Keep the runtime validator strict for clearly dangerous process/filesystem helpers.

### Current Stage
- Adjusting the AST validator locally, then re-running assembly from the completed AgentDojo checkpoints on the worker.

### Completed
- `tool_runtime.validate_tool_source(...)` now allows `.remove(...)` calls so finite-state simulators can mutate in-memory collections.
- Attribute-level bans still cover `system`, `popen`, `run`, `unlink`, `rmdir`, `mkdir`, and `makedirs`.
- Added `tests/tool_runtime_validator_smoke_test.py` to lock in the intended behavior.

### In Progress
- Running local regression tests and preparing a remote assemble-only rerun from the 629 completed checkpoints.

### Next
1. Sync the validator change to the remote shared workspace.
2. Re-run `assemble-bundles` / conversion resume against the existing AgentDojo checkpoints.
3. Confirm final merged `drafts.json` and `runtime_catalog.json` are written.

## 2026-03-23 Safe Dunder Introspection For Simulated Tools
### Goal
- Allow finite-state simulated tools to use a tiny whitelist of safe dunder introspection, such as `__class__` and `__name__`.
- Keep unsafe dunder access blocked, especially object metadata that is not needed for in-memory simulation.

### Current Stage
- Updating the validator and regression coverage, then re-running AgentDojo bundle assembly from the completed checkpoints.

### Completed
- `validate_tool_source(...)` now allows `__class__` and `__name__` attribute access.
- Unsafe dunder attributes remain blocked.
- Added regression coverage for allowed safe introspection and forbidden `__dict__`.

### In Progress
- Syncing the relaxed validator to the remote shared workspace.

### Next
1. Re-run the remote AgentDojo assemble step from the existing checkpoint directory.
2. Confirm whether the final `drafts.json` and `runtime_catalog.json` are produced.
3. If assembly still fails, inspect the remaining source pattern and narrow the validator/prompt accordingly.

## 2026-03-23 Parallel Benign And Risk Goal Fields
### Goal
- Add a parallel `risk_goal` / `risk_checklist` track alongside the existing benign goal track.
- Keep this as a reserved but disabled structure for the synthetic generation pipeline.
- Populate it meaningfully only for the source-task conversion pipeline, where benchmark injection goals already exist.
- Propagate the risk track into runtime when `risk_spec.enabled` is true.

### Current Stage
- Wiring risk fields through contracts, draft assembly, runtime bundles, and episode state.

### Completed
- `TaskPlanSpec` now carries `risk_spec`, `risk_success_rule`, and `risk_checklist_items`.
- `TaskDraft` now carries `risk_checklist_draft`, and `state_draft` now carries `risk_success_rule`.
- `RuntimeTaskBundle` now carries `task_spec.risk_spec`, `evaluation_bundle.risk_checklist`, `evaluation_bundle.risk_checklist_eval_hints`, and `evaluation_bundle.risk_success_eval_rule`.
- Synthetic generation now preserves a disabled risk placeholder instead of omitting the structure.
- Source-task conversion now derives:
  - benign success/checklist from `user_ground_truth_calls`
  - risk success/checklist from `injection_ground_truth_calls`
- Runtime now places the risk track into episode state only when `risk_spec.enabled` is true, and returns `risk_info` in `reward_info` for enabled bundles.

### In Progress
- Running regression tests to ensure the new risk track does not alter the original benign scoring path.

### Next
1. Verify source-task conversion bundles expose both benign and risk tracks in runtime.
2. Keep synthetic generation behavior unchanged except for the reserved disabled fields.

## 2026-03-23 Rollout 503 Retry Hardening
### 目标
- 提升 benchmark-style agent rollout 在共享上游模型服务下的稳定性。
- 对瞬时上游错误做客户端级别的重试和退避，避免因为短暂 `503 Service Unavailable` 直接把整条 case 标成 error。

### 当前阶段
- 本地已完成客户端重试逻辑与回归测试，正在同步到远端 worker 并重启全量 AgentDojo rollout。

### 已完成
- 为 `OpenAICompatClient` 增加瞬时错误重试：
  - `429`
  - `500`
  - `502`
  - `503`
  - `504`
- 重试策略为指数退避，不改变非瞬时错误的 fail-fast 行为。
- 新增 `tests/llm_client_retry_smoke_test.py`，锁定两次 `503` 后第三次成功的预期行为。
- README 已补充 transient upstream retry 说明。

### 进行中
- 同步最新客户端到远端共享代码目录。
- 停止旧的高 error-rate rollout，并用同一批 629 个 bundle 发起新的全量 rerun。

### 下一步
1. 在 worker 上验证新的 rollout 不再因为短暂 `503` 大量报错。
2. 统计 rerun 的 success / failure / error 比例与 token 使用情况。
3. 若仍有明显 error，再考虑按 endpoint 吞吐进一步调低 rollout 并发或增加更细的重试分类。
3. Only after validation, decide whether to push this change upstream.

## 2026-03-23 Review Fixes For Checkpoint And Rule Validation
### Goal
- Address the latest grouped review findings without changing the intended planner-first or risk-track behavior.
- Fix resume semantics, runtime-rule validation, checklist fallback robustness, and avoid unnecessary placeholder-path overhead.

### Current Stage
- Local fixes and regression tests are complete; syncing to the remote host and preparing to push.

### Completed
- `build_llm_converted_task_drafts(...)` now reuses completed checkpoints only when `resume=True`, so non-resume runs force a fresh conversion.
- `history_call_sequence_contains` validation no longer lets an inner call override the synthetic `type`.
- Tool-rule constraint extraction now includes `history_call_matches` and `history_call_sequence_contains`.
- Source-task ground-truth normalization now increments `step_index` only for valid entries, preventing checklist step gaps.
- Source checklist normalization now uses `fallback.get("runtime_rule")` instead of assuming the key is always present.
- Placeholder source-task conversion no longer performs duplicate validation and repeated linear scans for every draft.

### In Progress
- Propagating these fixes to the host checkout and finalizing a clean GitHub push.

### Next
1. Sync the patched files to the host workspace.
2. Re-run the source-task conversion and placeholder smoke tests on host.
3. Push the review-fix commit to GitHub.

## 2026-03-23 README Clarifications For Risk Track And Checkpoints
### Goal
- Reflect the current runtime and conversion behavior in the public README.
- Make the benign/risk dual-track design and checkpoint semantics clear to future users.

### Current Stage
- Updating documentation only; behavior is already implemented and tested.

### Completed
- Added README notes that source-task conversion keeps `user_query` and `selected_tools` fixed to the source benchmark task.
- Documented the dual benign/risk track:
  - benign success and checklist
  - risk success and checklist
- Documented checkpoint semantics so non-resume runs do not silently reuse old completed drafts.

### In Progress
- Syncing the documentation-only update to the host checkout and preparing a small follow-up push.

### Next
1. Push the README clarification commit to GitHub.
2. Keep future runtime or conversion behavior changes mirrored in README at the same time.

## 2026-03-24 Rollout 4xx Error Preservation
### Goal
- Preserve structured server-side error payloads during agent rollout instead of collapsing all HTTP failures into opaque `HTTPError` strings.
- Make batch rollout result files distinguish tool-call/schema problems from upstream model transport failures.

### Current Stage
- Local code and regression test are being updated after diagnosing a single rollout `400 Bad Request`.

### Completed
- Updated `tasksvc/runtime/agent_rollout.py` so `_http_post_json()` and `_http_get_json()` parse JSON bodies from `HTTPError` responses when available.
- Added a smoke-test branch in `tests/agent_rollout_smoke_test.py` that serves a synthetic `400 invalid_arguments` response and verifies the payload is preserved.

### In Progress
- Re-running local rollout smoke tests to confirm the new error-preservation path does not change successful rollout behavior.

### Next
1. Sync the updated rollout client to the remote shared checkout.
2. Re-run the affected AgentDojo case or a small batch and confirm future `400` results record structured `error` details.

## 2026-03-24 Configurable Rollout And Episode Budgets
### Goal
- Make the agent reasoning budget and environment execution budget configurable instead of relying on the old `8/6` defaults.
- Align longer benchmark-style reruns with AgentDojo's larger interaction budget.

### Current Stage
- Local code changes are in progress, followed by a fresh remote rollout using `max_turns=15` and `max_steps=15`.

### Completed
- Added `--episode-max-steps` to `server.py` so episode execution limits are configurable at runtime.
- Kept `tasksvc_cli.py run-agent-episode --max-turns` as the rollout-side control and documented the two knobs together in README.
- Extended `tests/smoke_test.py` to verify a server launched with `--episode-max-steps 1` ends an episode after a single tool call.

### In Progress
- Syncing the configurable-budget changes to the remote shared checkout.
- Preparing a fresh AgentDojo rerun with `15/15`.

### Next
1. Launch a new remote rollout with `max_turns=15` and `max_steps=15`.
2. Compare success/failure/error rates against the previous `8/6` run.

## 2026-03-24 Batch Rollout CLI
### Goal
- Replace the ad-hoc result-directory batch runner with a versioned CLI path.
- Make benchmark reruns configurable through committed code rather than one-off remote scripts.

### Current Stage
- Local batch CLI and smoke tests are complete; remote sync and the new `15/15` rerun are next.

### Completed
- Added `tasksvc/runtime/batch_rollout.py` to register catalogs, run concurrent rollouts, and write `manifest.json`, `summary.json`, and per-task result files.
- Added `tasksvc_cli.py run-agent-batch` with configurable `--max-turns`, `--max-workers`, and `--output-dir`.
- Added `tests/batch_rollout_smoke_test.py` to verify the batch path against a local server.
- Updated README with the new batch rollout command and the paired server `--episode-max-steps 15` example.
- Extended rollout results and batch summaries with explicit benign/risk fields:
  - per-task `task_success`, `risk_enabled`, `risk_success`
  - aggregate `risk_enabled_tasks`, `risk_triggered_tasks`, `benign_only_tasks`, `risk_only_tasks`, `benign_and_risk_tasks`, `neither_goal_tasks`

### In Progress
- Syncing the new batch rollout CLI to the remote shared checkout.
- Preparing the AgentDojo rerun to use the committed `run-agent-batch` path.

### Next
1. Start a fresh remote AgentDojo rollout via `run-agent-batch` with `max_turns=15`.
2. Pair it with `server.py --episode-max-steps 15`.
3. Compare success/failure/error counts against the previous run.

## 2026-03-24 Rollout Evaluator Leak Fix
### Goal
- Make rollout feedback mirror real tool interaction instead of leaking internal evaluator signals back to the model.

### Current Stage
- Local fix and regression test are being applied after identifying that `reward_info.risk_info` was visible in the next-turn tool message.

### Completed
- `tasksvc/runtime/agent_rollout.py` now only feeds `tool_result`, `observation`, and `done` back to the model after each tool call.
- `reward_info` and `risk_info` remain available in logs/results, but are no longer part of the agent-visible tool message payload.
- `tests/agent_rollout_smoke_test.py` now asserts that `_tool_message_content(...)` strips evaluator metadata.

### In Progress
- Preparing a small rerun/spot-check to confirm risk-triggered cases no longer receive injection details through evaluator leakage.

### Next
1. Re-run a small batch of AgentDojo banking cases with the patched rollout client.
2. Compare benign/risk outcomes before and after the leak fix.

## 2026-03-26 Rollout Error Semantics
### Goal
- Make rollout-side error accounting closer to real agent interaction.
- Let recoverable invalid tool calls feed back into the model instead of terminating the whole case.

### Current Stage
- Local rollout handling and smoke tests have been updated; the next step is syncing the change to the remote checkout before the next benchmark rerun.

### Completed
- `tasksvc/runtime/agent_rollout.py` now treats server responses with `invalid_call=true` as recoverable tool feedback and appends them to the transcript/tool messages.
- `tasksvc/runtime/agent_rollout.py` now treats `episode_already_done` as a terminal episode state lookup rather than a hard rollout error.
- `tests/agent_rollout_smoke_test.py` now covers both:
  - invalid arguments followed by a corrected retry
  - `episode_already_done` folding into final episode success/failure instead of erroring

### In Progress
- Reviewing current AgentDojo content-rollout errors to separate:
  - recoverable invalid calls
  - episode lifecycle races
  - true execution failures such as tool timeouts

### Next
1. Sync the rollout error-semantics patch to the host/shared checkout.
2. Re-run the affected content rollout and compare error counts before/after the patch.
3. Decide whether content-heavy runs should also raise `--tool-exec-timeout` above the current default of 2 seconds.
