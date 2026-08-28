# 执行与资源安全规范

本文档适用于本仓库的 mask、depth、pose、mesh、review、视频渲染和
Isaac 资产导出任务。完整事故证据见
[2026-08-20 pose_solver mesh-split 内存耗尽事故](incidents/2026-08-20-mesh-split-memory-exhaustion.md)。

## 为什么必须遵守

2026-08-20，两个相隔约 36 秒启动的临时 Python mesh 诊断分别在
Object-3 和 Object-4/Object-9 的高密度重建 mesh 上调用
`trimesh.Trimesh.split(only_watertight=False)`。它们分别增长到约
284.5 GiB 和 538.7 GiB 匿名内存，触发了两次 Linux global OOM。

这不是正常的缓慢内存泄漏。`split()` 会计算连通分量并为每个分量
物化、复制几何；对于高面数、碎片化或非流形重建 mesh，内存开销可能
远大于 GLB 文件大小。无输出不代表任务已经退出，也不能据此重复启动。

## 强制规则

1. 生产任务只能通过统一入口运行：

   ```bash
   .venv/bin/python -u -m pose_solver run --config configs/my_object.json
   ```

   除 `preflight` 和 `dry-run` 外，统一入口会自动进入进程组内存守护器。
   不得关闭或绕过 `runtime.memory_guard`。

2. 禁止用 `python -`、heredoc 或 `python -c` 临时执行 mesh、pose、render、
   review 或视频诊断。需要新诊断时，应新增参数有界、输入有界、输出可审计
   的 CLI，并通过 `tools/diagnostics/run_with_memory_guard.py` 启动。

3. 禁止对 reconstruction mesh 调用 `Trimesh.split()`。只统计分量时优先
   使用不物化子 mesh 的计数或稀疏标签方法，并仍需在守护器内运行。只有
   明确降面后的 collision proxy 才允许物化分量；当前代码对超过
   100,000 faces 的非 raw proxy 会直接拒绝。

4. 同一用户、同一数据或同一输出目录不得从多个终端、agent session 或
   后台任务同时启动重任务。若命令没有按期返回，先查进程和内存，不得
   直接重试。

5. Object-3 只允许使用 GPU 6、7，任何任务最多使用两张卡。其他数据也
   必须严格遵守配置中的 `runtime.devices`，不得临时扩展设备范围。

6. 默认安全线为：

   - 完整子进程组 RSS 不超过 32 GiB；
   - 系统 `MemAvailable` 不低于 128 GiB；
   - 每秒采样一次；
   - 超限后停止整个进程组，而不是只停止父进程。

## 启动前检查

- 先执行 `--stage preflight`，确认输入、帧范围、部件和 GPU 配置。
- 确认 `runtime.memory_guard.enabled` 为 `true`，并保留 32/128 GiB 安全线。
- 使用 `free -h` 查看 `available`，不要只看 `free`。
- 使用精确的进程查询确认没有相同 config、stage 或 output root 的任务。
- GPU 任务确认只暴露配置允许的设备；Object-3 必须严格为 6、7，最多两张。
- 为每次运行使用明确的输出目录和 guard 日志，避免不同运行互相覆盖。

推荐配置：

```json
"runtime": {
  "devices": [6, 7],
  "memory_guard": {
    "enabled": true,
    "minimum_available_gib": 128,
    "maximum_process_rss_gib": 32,
    "poll_seconds": 1,
    "report_seconds": 10,
    "stop_grace_seconds": 2
  }
}
```

## 运行中检查

守护日志位于：

```text
<output.root>/runtime/memory_guard.jsonl
```

重点观察：

- `process_group_rss_bytes` 是否持续单调快速增长；
- `mem_available_bytes` 是否接近 128 GiB；
- 是否出现 `limit_exceeded`、`start_rejected` 或 GPU 查询连续失败；
- 子进程数量是否异常增长；
- 终端无输出时，任务是否仍然存在并持续占用 CPU/RAM。

守护器退出码含义：

- `0`：任务正常结束；
- `125`：启动被拒绝或运行中达到资源限制；必须调查，不能原样重试；
- `130`：守护器收到中断并清理了子进程组；
- `137`：进程收到 SIGKILL，可能发生 OOM；按事故处理，不得重试。

## 无输出、超限或疑似 OOM 时

1. 立即停止启动任何新任务，不要用相同命令重试。
2. 保存完整命令、配置路径、输出目录、guard 日志、开始时间和退出码。
3. 使用 `free -h`、精确的 `ps`/`pgrep` 查询和 `nvidia-smi` 确认资源与
   进程归属。不要把其他用户的任务当作本项目进程处理。
4. 若 guard 仍在，等待它清理完整进程组；若 guard 已异常退出，只处理
   已核实的项目 PID/PGID，禁止宽泛匹配后批量杀进程。
5. 确认没有孤儿进程后再分析算法。必须改变算法、输入规模或资源边界，
   才能重新运行。
6. 不要擅自清空共享服务器 swap、重启服务或终止其他用户进程；需要时
   联系管理员。

## Mesh 诊断的安全替代方案

- 先读取文件大小和 mesh 顶点/面数量，不物化连通分量。
- 对需要统计的标签使用稀疏 component labels，并尽早释放临时数组。
- 可视化时先确定性抽样顶点/面，禁止把完整 mesh 复制到每个候选或视角。
- 物理碰撞分量必须从降面 proxy 生成；raw visual mesh 保持为单一资产。
- 新诊断 CLI 必须提供输入数量、最大帧数、最大 faces/points、最大视角数
  等硬上限，并将峰值内存纳入测试。

## 事故复盘结论

本次两次 OOM 的直接原因是无守护的临时 `python -` 诊断对重建 mesh
调用 `Trimesh.split()`；不是 pose 优化、视频编码或 Isaac 本身。后续即使
目标只是打印 mesh 分量数量，也必须遵守相同的资源安全边界。
