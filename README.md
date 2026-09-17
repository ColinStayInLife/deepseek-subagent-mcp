# DeepSeek Subagent MCP

An opt-in MCP server for delegating bounded tasks to DeepSeek agents, with a durable background queue, evidence tracking, and explicit host-tool handoffs.

Version: **1.7.1**. Python standard library only. Designed for Linux: process identity, file locks, signals, and cancellation use Linux facilities. Use a recent Python 3 release (3.10+ syntax; tested locally with Python 3.13).

## 功能

- 主控获得用户明确授权后，自行拆分任务并交给 DeepSeek；普通对话不自动调用。
- 默认 `deepseek-flash`、`max`；后台最多 3 个执行任务，最多 24 个待处理任务。
- 异步提交、批次、执行依赖、短等待、取消及跨连接任务去重。
- 文件 SHA 快照、明确的读写契约、固定命令、验收检查及执行记录。
- 显式传递必要会话材料和带 SHA 的图片；浏览器/连接器请求交回主控执行。
- 可准备宿主原生代理委派，但实际启动、可用模型、工具和计费由宿主决定。

## v1.7.1：统一目录权限，改善工具预算收尾

目录契约缺少末尾 `/` 时，旧版可能拒绝子文件读取，却允许目录搜索。新版在启动模型前拒绝这种歧义规则，并统一读取、列表和搜索的范围检查。文件规则后来变成目录，也不能自动获得递归访问权限。

- 契约草稿新增 `read_dirs/write_dirs`，显式生成带 `/` 的目录规则；`read_paths/write_paths` 保留原有精确语义。文件后误加 `/` 也会被拒绝。
- `prepare.py check-contract` 离线检查契约、范围和 SHA，列出尚不存在的读取路径；不执行命令或模型，也不代表任务验收通过。
- `grep` 在搜索前后核对固定证据 SHA；修复没有 `rg` 时的单文件搜索，目录搜索不跟随子项符号链接。
- `list_dir` 支持 `offset/limit`，默认每页最多 80 项、可设最多 200 项，按输出预算返回完整条目和 `next_offset`；分页不是文件系统事务快照。
- 每轮显示剩余工具数，使用 38/40 次后，下一轮在剩余轮数、token 和时间内转为只报告。预算不足时保留本地交接；不会额外调用模型补收尾，`tool_limit` 也不会因拿到报告变成 `completed`。
- 一批调用超过上限时，仅执行预算内的前缀，确定未派发的尾部记录为 `not_executed` 并补齐调用配对，使已有产物可进入只读契约检查。执行中断或副作用未知仍保持未知状态。`receipt/handoff/structuredContent` 提供工具预算和未执行清单，后台状态显示剩余工具数。

示例草稿：

```json
{
  "objective": "审查指定模块和验收证据",
  "acceptance": ["结论附实际证据和未确定项"],
  "read_paths": ["spec.md"],
  "read_dirs": ["src", "results"],
  "write_paths": [],
  "commands": []
}
```

```bash
python3 prepare.py contract --draft draft.json --cwd /absolute/project --output contract.json
python3 prepare.py check-contract --contract contract.json --cwd /absolute/project
```

更新后重新连接 MCP。旧契约和旧任务不自动修改、增加预算或重跑；目录规则不合要求时，由主控按原授权明确修订。Flash max、v1.7 token 预算和默认 3 个并发保持不变。路径检查属于工具层约束，不能替代操作系统沙箱。离线测试验证这些机制，实际完成率、耗时及费用改善尚未通过付费对照测量。

## v1.7 更新：给 Flash max 留足实现和验证预算

| 参数 | 默认值 | 本 MCP 可配置上限 |
|---|---:|---:|
| `input_budget`：任务累计输入，含缓存 | 500,000 | 500,000 |
| `max_output_tokens`：单轮输出，含思考和工具参数 | 65,536 | 131,072 |
| `output_budget`：任务累计输出 | 128,000 | 500,000 |
| `timeout_sec`：任务累计执行时间 | 1,200 秒 | 3,600 秒 |

参数优先级为：显式任务参数 > 最近祖先目录 `.deepseek-subagent.json` 的 `defaults` > 服务默认值。一般省略预算参数即可；旧调用中显式写入的 32768、40000 等数值仍会生效。项目配置只覆盖执行预算，不授予文件或命令权限。

长任务在提交时显式选择更高预算，例如：

```json
{
  "action": "submit",
  "task_id": "long-review-001",
  "task": "按已确定的接口和验收标准，只读审查指定模块。",
  "cwd": "/absolute/project",
  "max_output_tokens": 131072,
  "output_budget": 500000,
  "timeout_sec": 3600
}
```

实际单轮输出取配置值与剩余累计输出的较小值；使用 131072 单轮档位时应同时设置足够的累计输出。输入是多轮累计软阈值，首轮没有实际用量校准，预测不能保证硬计费封顶。上限不是预付用量，也不强制模型生成到上限。

文本请求和显式宿主上下文文件由 192000 字节放宽至 **4 MiB**；包含图片的完整请求仍限制为 24 MiB，图片数量/单张大小限制保留。另按 1M 上下文窗口估算输入，并预留实际单轮输出和 16384 token 余量：首轮使用保守的 UTF-8 字节启发式，后续用实际输入用量和请求增长校准。`receipt.json` 的 `context_projection` 记录估算方法与图片未校准状态；这是估算保护，**不是精确 tokenizer，也不保证任意材料都能用满 1M**。累计输入预算仍独立生效。较早工具结果仍只在有完整证据 pins 时裁剪。

任务默认最多执行 20 分钟，长任务最多 60 分钟；单个获准命令仍遵守原契约与 600 秒限制。优先用后台 `submit/batch`，主控用原有短 `wait/status` 获取结果。同步兼容路径需宿主 `tool_timeout_sec=3650`；本地 `client.py` 看门狗为 3660 秒。

新后台任务在提交时冻结所有有效预算，避免排队期间服务默认值变化。`followup/resume` 沿用检查点中的原预算和累计用量，不为旧失败任务自动提额、重试或补跑；修改项目配置后，旧任务仍可能因原配置 SHA 不匹配而拒绝续接，需先核对已有结果。

Flash 保持 `max`，并发仍为 3；本次更新仅调整预算及保护机制，没有新增付费调用。离线 fixture 验证较长推理之后的写入/收尾、剩余预算约束、配置优先级和续接；实际完成率与耗时仍需真实授权任务积累证据。

依据：[DeepSeek 模型与价格](https://api-docs.deepseek.com/zh-cn/quick_start/pricing/)、[Responses 输出包含思考 token](https://api-docs.deepseek.com/api/create-response/)、[Codex MCP 超时配置](https://learn.chatgpt.com/docs/config-file/config-reference)。

## v1.6.1 更新：预算停止后的验收与推理耗尽诊断

### 已有产物时，用本地证据收尾

`input_budget`、`input_budget_prediction`、`output_budget`、`context_budget`、`step_limit`、`tool_limit` 停止时，若工具调用配对完整、无未知操作或待处理宿主请求、用量完整，服务会只读执行契约声明的 `checks` 和 `deliverables` 检查。它不会导入生成的程序、运行命令、重跑仿真或调用模型。检查最多占用剩余墙钟预算中的 5 秒；无法完成则明确记录 `deferred`。

`receipt.json`、`handoff.json`、工具结果及后台状态新增：

```json
{
  "status": "input_budget_prediction",
  "completion": {
    "execution": "input_budget_prediction",
    "verification": "passed",
    "model_report": "not_received",
    "controller_acceptance": "pending"
  },
  "local_verification": {
    "status": "passed",
    "checked": 3,
    "scope": "configured_contract_checks_only",
    "model_calls": 0,
    "commands_run": 0
  }
}
```

这里的 `passed` 只表示声明的机器检查通过。未配置检查为 `not_configured`，缺失或不符合要求为 `failed`，不满足安全边界或时间不足为 `deferred`；后两者不等于通过。模型正常收尾但本地检查未完成时，状态为 `acceptance_deferred`。

预算停止仍保留原 `status` 和错误标记，不自动变成 `completed`，也不自动启动依赖任务。主控根据原目标验收；已满足时，不必仅为补模型文字报告而重跑。`api_incomplete`、未知副作用和超时路径不会进行这类自动验收。

### 写入前推理耗尽时，交接原因和已读证据

API 因 `max_output_tokens` 截断时仍返回 `api_incomplete`，新增 `diagnostic`：

- `reasoning_output_exhausted`：API 用量明确显示该轮全部输出 token 都计入推理。
- `output_limit_exhausted`：其他输出上限截断；缺少推理计数时不猜测原因。
- `limit_source` 区分单轮配置限制与剩余累计输出预算限制；同时记录本轮上限、实际输出、推理、非推理输出、剩余累计预算。
- 记录该截断轮丢弃的工具调用数量，以及此前是否有写入/命令尝试或待处理宿主请求。该轮不执行任何工具，不代表此前或外部程序没有副作用。
- `handoff.json` 的 `inspection_index` 给出已读文件、工具错误及完整结果 pins，主控按需复用证据；不包含隐藏推理正文。

同一单轮上限同时容纳推理和可用输出；MCP 没有另一个参数能保证为代码保留指定数量的输出 token。实现提示已提前至首次读取后，提醒按固定接口实现可独立验证的部分，并列出任务开始时不存在的必需目标文件。提示不能保证避免长推理。

复杂任务由主控明确阶段和验收边界，例如先实现纯数据校验，再实现依赖原生库的封装。不是强制增加 scout/reviewer 调用，也不允许占位代码、跳过负控或无理由拆分简单任务。涉及模型、科学假设、关键接口的阶段仍由主控先验收；同一文件的修改保持串行。发生输出截断后先核对证据，再缩小任务或接管，不原样自动重试。

Flash `max`、模型、默认预算、单轮上限、并发数均未调整。此更新没有新增付费收尾调用，也不自动重放旧失败任务。

## v1.6 更新：纠偏、返工续接和上下文管理

- `steer`：主控向后台 `queued/running` 任务发送补充指令。Flash 在下一次模型请求前、模型返回后、各工具之间接收；跳过尚未执行的过时工具，已经执行的内容不回滚，正在运行的单个命令不会被这条消息中断。
- `followup`：主控验收后继续同一个任务，保留 Flash 历史、已执行命令记录及累计预算。独立 `followup_id` 保证请求重复时不重跑；异步任务仍进入原队列并重新检查文件冲突。
- `handoff.json`：自动整理简短报告、改动文件、验收检查、用量及证据索引，主控按需读取，不必重读完整执行过程。
- 上下文压力较高时，先裁剪较早且已成功返回的长工具输出。保留最近 4 个工具结果、完整任务指令、契约、图片和调用配对；被裁剪项必须有通过 SHA 校验的完整证据。裁剪前历史另存私有文件，不调用摘要模型。
- `reconcile`：未知任务结果经主控核对后登记证据并释放队列范围；不执行工具、不重新计费、不回滚修改，也不把任务标记为完成。

仍只在用户明确启用 `deepseek_subagent` 的任务内调用模型。Flash 默认 `max`，原预算与并发上限不变。

### 执行中纠偏

```json
{"action":"steer","task_id":"example-review-001","message_id":"correction-001","instruction":"新证据表明当前输入不足。停止后续写入，整理缺失参数及来源，不猜测数值。"}
```

同一 `message_id`、同一内容只接收一次；同 ID 不同内容拒绝。每任务最多 32 条、合计 65536 UTF-8 字节，每条最多 8192 字节。`status` 返回消息 ID 和 `applied`，不返回消息正文；`applied` 表示已交给工具循环，不表示模型已经执行或验收通过。停止前仍未应用的消息可在状态中核对。

### 验收后返工

```json
{"action":"followup","task_id":"example-review-001","followup_id":"repair-001","instruction":"按原契约补齐报告缺失的单位和验证结果，保留已确认正确的内容。"}
```

只允许从有私有检查点、用量完整、工具调用结果配对齐全的安全停止点续接。典型场景是 `completed` 后主控发现遗漏，或 `acceptance_failed` 后修复。不会改变 cwd、契约、权限、模型、强度、轮数上限或累计预算。原预算耗尽时拒绝续接；改变科学模型或契约时，先由主控核对现有产物，再明确创建新任务。

`resume` 仍专用于 `needs_host` 的 claim → 实际宿主工具 → resume 交接，不能用来代替返工。崩溃或副作用未知时不能用新 `followup_id` 绕过核对。重复 followup 不新增执行；通过 `status` 查询任务当前进展。历史 v1.5 任务没有新检查点，不能自动补出历史后续接。

### 未知结果核对

后台任务为 `outcome_unknown` 且 worker 已退出时，主控先检查 receipt、实际文件、命令进程与已认领的宿主动作，再准备 `DEEPSEEK_RECONCILE_V1` JSON：

```json
{
  "schema": "DEEPSEEK_RECONCILE_V1",
  "task_id": "example-review-001",
  "job_pin": {"path": "/absolute/runs/async/jobs/TASK_HASH/job.json", "sha256": "ACTUAL_SHA256", "bytes": 123},
  "receipt_pin": {"path": "/absolute/runs/RUN_ID/receipt.json", "sha256": "ACTUAL_SHA256", "bytes": 456},
  "note": "记录实际检查过的副作用、文件和剩余工作。",
  "effects_reviewed": true,
  "no_processes_running": true,
  "evidence": []
}
```

路径、字节数和 SHA 必须来自实际状态；只有 receipt 确实不存在时才填 `null`。`evidence` 填最多 40 个已检查文件的实际 pins。确认字段是主控的核对声明，程序不能证明所有外部进程或宿主动作已经停止。调用：

```json
{"action":"reconcile","task_id":"example-review-001","reconciliation_path":"/absolute/review.json"}
```

成功后状态为 `reconciled`，不满足下游 `depends_on` 对 `completed` 的要求。原命令去重记录继续保留，未知命令不会因此获得重跑权限。

### 上下文与科研判断

本版实现的是可追溯的工具输出裁剪，尚未自动生成语义摘要。裁剪保留首尾，不能保证中间内容不重要；模型应通过完整证据路径回查。若裁剪后仍超预算则停止，不自动加预算或调用额外模型。物理化学模型、单位、工况、来源、边界条件及验收要求应由主控明确放入任务/契约中，不能只埋在日志里。

`runs/` 新增 0600 权限的 `followup.json` 和裁剪前历史，可能包含 Flash 自身会话、必要推理和显式传入的材料；不包含隐式抓取的宿主历史，不能公开上传。检查点保留至下一段执行覆盖或任务状态目录由用户清理；公共状态只提供路径/校验值和短交接报告。

设计参考：[PsChina/deepseek-as-subagent](https://github.com/PsChina/deepseek-as-subagent) 的运行中纠偏、[PanGucheng/codex-deepseek-delegate-mcp](https://github.com/PanGucheng/codex-deepseek-delegate-mcp) 的任务续接与精选交接、[DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) 的工具结果裁剪。本实现保留原有标准库架构，未引入这些项目的运行依赖。

## v1.5 更新

- `clone_file`：按源文件 SHA 克隆 UTF-8 文本并做最多 32 项精确替换，全部成功才原子发布，已有目标不会被覆盖。
- 成功读取的内容重新验证后若完全相同，返回已有结果的短引用，减少重复内容进入模型历史。
- 根据上一轮真实用量及请求大小估算下一轮输入，预测超过剩余软预算时用本地记录收尾，不额外调用模型；该估算不是硬计费上限。
- V2 契约支持 `deliverables`：检查必需文件非空，`must_change=true` 时验证文件新建或内容发生变化；不满足则 `acceptance_failed`。
- `status/wait` 的 `execution` 元数据提供已记录的用量和写入次数，不披露提示词或推理文本。

模型思考强度、默认执行预算和并发配置保持不变。实际节省比例和模型成功率尚未做付费对照实验。

## 安装

克隆仓库后，让 MCP 宿主以 `python3 /absolute/path/to/deepseek-subagent-mcp/server.py` 启动 stdio 服务。将 `DEEPSEEK_API_KEY` 安全注入服务进程环境；**不要把密钥写入仓库或任务文件**。程序不会自动读取个人凭证文件。

Codex 配置示例（替换路径）：

```toml
[mcp_servers.deepseek-subagent]
command = "python3"
args = ["/absolute/path/to/deepseek-subagent-mcp/server.py"]
env_vars = ["DEEPSEEK_API_KEY"]
startup_timeout_sec = 20
tool_timeout_sec = 3650
```

修改工具定义后重新连接 MCP。`AGENTS.md` 提供本仓库的开发约束和可移植的主控使用规则；需要其他工作目录也采用这些规则时，将相关规则加入对应宿主指令。

可选环境变量：

| 变量 | 默认值 |
|---|---|
| `DEEPSEEK_API_BASE` | `https://api.deepseek.com` |
| `DEEPSEEK_SUBAGENT_MODEL` | `deepseek-flash` |
| `DEEPSEEK_SUBAGENT_EFFORT` | `max` |
| `DEEPSEEK_SUBAGENT_RUNS_DIR` | 本仓库的 `runs/` |

所有共享同一 `runs` 目录的后台任务共用 `scheduler.json` 的并发限制。不要将不同账户的状态目录混用。

## 使用

对主控说：

> 调用 deepseek_subagent 完成这项任务。请自行拆分适合并行或后台执行的部分，保留关键判断，最后汇总验收。

单项后台提交：

```json
{
  "action": "submit",
  "task_id": "example-review-001",
  "task": "只读检查指定代码，给出问题、证据和未确定项。",
  "cwd": "/absolute/project",
  "allow_write": false,
  "allow_shell": false
}
```

多个任务用 `examples/batch.json` 的格式准备文件，替换路径和任务描述，再提交：

```json
{"action":"batch","batch_path":"/absolute/project/batch.json"}
```

提交后主控继续独立工作，需要结果时使用：

```json
{"action":"wait","batch_id":"example-batch-001","wait_sec":15}
```

```json
{"action":"status","task_id":"example-review-001"}
```

```json
{"action":"cancel","task_id":"example-review-001"}
```

`status/wait/cancel` 用 `task_id` 或 `batch_id` 二选一。`wait` 最多等待 20 秒，状态变化提前返回。批次返回短元数据和证据路径；单项状态附短报告。

工具还保留同步 `run`、宿主请求认领 `claim`、续跑 `resume`、原生委派计划 `native_request` 和报告登记 `native_result`。

## 契约与证据

异步写文件或执行命令必须提供 `contract_path`。用 `prepare.py contract` 将简短草案转换为 `DEEPSEEK_TASK_V2`，补齐绝对路径及文件 SHA：

```bash
python3 prepare.py contract --cwd /absolute/project --draft draft.json --output task-contract.json
```

草案格式见 `examples/contract-draft.json`。实施任务可增加 `"deliverables": [{"path": "src/module.py", "must_change": true}]`，该路径也须在允许写入范围内。读取路径尽量精确，避免宽泛范围使不同任务被保守串行。目录规则以 `/` 结尾。

批次任务可包含 `depends_on` 任务 ID 数组；必须引用已有任务或同批任务，且不能成环。前置任务 `completed` 后才启动下游。主控需在下游任务中说明产物路径；队列不会自动注入前置报告。

已有证据、任务参数、报告、日志和累计用量保存在 `runs/`。该目录可能包含私有代码、路径、会话材料及图片，已加入 `.gitignore`，不要公开上传。

## 宿主工具和图片

用 `prepare.py host` 创建 `DEEPSEEK_HOST_V1`，显式提供 user/assistant 会话片段、图片及本任务真实可用的能力声明，然后传入 `host_context_path`。示例见 `examples/host.json`。

收到 `needs_host` 后，主控按以下步骤处理：

1. 在既有任务授权范围内，用同一 `task_id` 和匹配的 `request_id` 执行 `claim`，只认领一次。
2. 使用宿主实际工具执行获准请求；不把模型参数直接当代码执行。
3. 用 `prepare.py host-result` 保存真实结果与来源。
4. 以同一 `task_id`、`action=resume` 和 `host_result_path` 回传，不改变预算。

异步任务续跑仍在后台进行。`needs_host` 释放执行名额但保留文件范围。图片使用本地文件 SHA 和真实图片字节；浏览器登录态、连接器凭证留在宿主。

## 运行边界

- 主控保留架构、困难根因、科学模型与假设、关键结论及最终验收。机械检查通过不等于科学结论正确。
- `isError=false` 在状态查询中仅表示查询成功；逐项检查 `structuredContent.tasks` 的 `status` 和 `result`。排队、执行中及等待宿主都不是完成。
- 契约和队列冲突检查不是操作系统沙箱。获准命令拥有服务进程权限；文件协调不约束主控或外部程序。重型任务仍须遵守项目自己的资源管理。
- 取消不回滚文件、撤销宿主动作或消除已发生的 API 用量。未知结果先核对记录，不换 ID 盲目重试。
- 后台任务可在 stdio 连接结束后继续运行，但没有原生子代理卡片或完成推送，也不能自行唤醒已结束的主控会话。主控通过 `wait/status` 收取结果。
- 原生委派仅准备参数，不自动启动宿主代理。原生模式不通过本服务的 DeepSeek API key 计费；不能保证继承完整会话或所有工具。

## 离线验证

```bash
python3 -m unittest -v test_server.py test_project_support.py test_upgrade.py test_async_jobs.py test_execution_support.py test_workflow_support.py test_budget_completion.py test_budget_settings.py test_contract_efficiency.py test_public_config.py
```

模型响应均由离线 fixture 替代，并发测试使用真实本地进程和文件锁。测试不需要 API key，不进行付费模型、真实浏览器或连接器调用。

公开副本仅包含程序、合成测试和通用示例，不包含运行记录、会话数据库、私有项目参数、个人配置或历史备份。

也可运行离线协议及克隆产物验证：

```bash
python3 verify_offline.py --output runs/verification/check-001
```

输出目录必须尚不存在；其中可能包含本机绝对路径，保持本地保存。需要比较旧版默认上限时，额外指定 `--baseline /absolute/path/to/previous/server.py`；未指定时该比较字段为 `null`。
