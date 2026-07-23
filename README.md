# pose_solver

`pose_solver` 将六视角同步 RGB、palette mask、DA3 depth/camera 和三个部件 mesh，
转换为逐帧 body-relative 6D pose、渲染视频以及可导入 Isaac Sim 的 URDF/USD 资产。

当前正式对象包含：

- `body`
- `inner_pot`
- `lid`

## 正式数据流

```text
六视角 RGB
   ↓ Qwen + SAM3
palette masks
   ↓ DA3 depth gauge + backprojection
分部件世界系点云
   ↓ 多视角 pose solve + body/lid refinement
trajectory.json
   ├─→ overlay / mesh-only / mesh+axes 视频
   └─→ canonical mesh → URDF → Isaac Sim 插入验证
```

各阶段只通过文件和 JSON 配置连接，不应从一个 CLI 脚本导入另一个 CLI 脚本。
可复用算法位于 `common/`，`scripts/` 只负责参数解析和流程编排。

## 目录

```text
common/       可复用的数据加载、几何、位姿、渲染和仿真资产逻辑
configs/      数据集与实验配置；绝对路径和人工先验只允许放在这里
scripts/      可直接运行的正式流程阶段
tests/        不依赖 GPU 的核心约定和资产测试
docs/         架构、仿真和专项说明
mesh/         输入 mesh（被 .gitignore 忽略）
experiments/  生成结果（被 .gitignore 忽略）
```

详细模块边界和维护规则见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

## 当前推荐入口

- Mask 全流程：`scripts/run_temporal_mask_pipeline.py`
- 深度 gauge：`scripts/calibrate_depth_gauge.py`
- 点云反投影：`scripts/backproject_normalized.py`
- 状态诊断：`scripts/detect_part_states.py`
- 初始位姿求解：`scripts/solve_multiview_pose.py`
- V6 全局精修与验收：`scripts/refine_multiview_pose.py`
- 视频渲染：`scripts/render_multiview_pose.py`
- 六视角评审：`scripts/export_multiview_pose_review.py`
- URDF 导出：`scripts/export_simulation_assets.py`
- Isaac Sim 验证：`scripts/run_isaac_insertion.py`

当前 rice-cooker 数据使用 `configs/pose_multiview_111*.json`；这些文件包含当前数据的
路径和先验，不是通用算法的一部分。批量处理新对象时应生成新配置，而不是修改
`common/` 中的算法。

V6 内部的 body、inner_pot、lid 和验收阶段仍可单独恢复执行，但批处理只应调用
`refine_multiview_pose.py`。它会检查各阶段产物并默认断点续跑；使用 `--force` 才会重算。

## 验证

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m compileall -q common scripts tests
```

Isaac Sim 脚本必须使用 Isaac Sim 自带的 `python.sh`，不能用项目 `.venv` 直接运行。
具体命令和结果解释见
[docs/SIMULATION_ASSET_PIPELINE.md](docs/SIMULATION_ASSET_PIPELINE.md)。

## 专项文档

- [MASK_PIPELINE.md](MASK_PIPELINE.md)：Qwen + SAM3 六视角 mask 流程
- [CURRENT_BOTTLENECKS_AND_ROADMAP.md](CURRENT_BOTTLENECKS_AND_ROADMAP.md)：
  当前精度瓶颈与路线
- [docs/SIMULATION_ASSET_PIPELINE.md](docs/SIMULATION_ASSET_PIPELINE.md)：
  URDF 与 Isaac Sim 插入验证
