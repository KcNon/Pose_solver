# Optional tools

`scripts/` 只放正式 pipeline 入口。本目录保存不会定义另一条主流程的辅助工具：

- `diagnostics/`：depth 稳定性、自动状态和多视角 pose 评审；
- `highfps/`：把低帧率 depth gauge 重采样到高帧率片段，并将专项跟踪结果合回主轨迹。
- `stages/`：由正式 runner 调用的 Qwen、SAM、DA3、点云和 pose solver 内部阶段。

正式 pose runner 会按需调用 diagnostics；high-FPS 工具只在对应配置存在时手动运行。
