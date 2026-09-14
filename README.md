# DeepSeek Subagent MCP

An opt-in MCP server for delegating bounded tasks to DeepSeek agents, with a durable background queue, evidence tracking, and explicit host-tool handoffs.

Version: **1.4.0**. Python standard library only. Designed for Linux: process identity, file locks, signals, and cancellation use Linux facilities. Use a recent Python 3 release (3.10+ syntax; tested locally with Python 3.13).

## 功能

- 主控获得用户明确授权后，自行拆分任务并交给 DeepSeek；普通对话不自动调用。
- 默认 `deepseek-flash`、`max`；后台最多 3 个执行任务，最多 24 个待处理任务。
- 异步提交、批次、执行依赖、短等待、取消及跨连接任务去重。
- 文件 SHA 快照、明确的读写契约、固定命令、验收检查及执行记录。
- 显式传递必要会话材料和带 SHA 的图片；浏览器/连接器请求交回主控执行。
- 可准备宿主原生代理委派，但实际启动、可用模型、工具和计费由宿主决定。

## 安装

克隆仓库后，让 MCP 宿主以 `python3 /absolute/path/to/deepseek-subagent-mcp/server.py` 启动 stdio 服务。将 `DEEPSEEK_API_KEY` 安全注入服务进程环境；**不要把密钥写入仓库或任务文件**。程序不会自动读取个人凭证文件。

Codex 配置示例（替换路径）：

```toml
[mcp_servers.deepseek-subagent]
command = "python3"
args = ["/absolute/path/to/deepseek-subagent-mcp/server.py"]
env_vars = ["DEEPSEEK_API_KEY"]
startup_timeout_sec = 20
tool_timeout_sec = 650
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

草案格式见 `examples/contract-draft.json`。读取路径尽量精确，避免宽泛范围使不同任务被保守串行。目录规则以 `/` 结尾。

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
python3 -m unittest -v test_server.py test_project_support.py test_upgrade.py test_async_jobs.py test_public_config.py
```

模型响应均由离线 fixture 替代，并发测试使用真实本地进程和文件锁。测试不需要 API key，不进行付费模型、真实浏览器或连接器调用。

公开副本仅包含程序、合成测试和通用示例，不包含运行记录、会话数据库、私有项目参数、个人配置或历史备份。
