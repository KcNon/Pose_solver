# pose_solver

`pose_solver` 将多视角同步 RGB（当前数据为八视角）、palette mask、DA3 depth/camera 和三个部件 mesh，
转换为逐帧 body-relative 6D pose、渲染视频以及可导入 Isaac Sim 的 URDF/USD 资产。

当前正式对象包含：

- `body`
- `inner_pot`
- `lid`

## 正式数据流

```text
多视角同步 RGB
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
tools/        可选诊断、评审和特定采样率专项工具
tests/        不依赖 GPU 的核心约定和资产测试
docs/         架构、仿真和专项说明
mesh/         输入 mesh（被 .gitignore 忽略）
experiments/  生成结果（被 .gitignore 忽略）
```

详细模块边界和维护规则见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

## 当前推荐入口

- Mask 全流程：`scripts/run_mask_pipeline.py`
- ReconViaGen 全流程：`scripts/run_reconviagen_pipeline.py`
- 深度与点云：`scripts/run_depth_pipeline.py`
- Pose 全流程：`scripts/run_pose_pipeline.py`
- 视频渲染：`scripts/render_multiview_pose.py`
- URDF 导出：`scripts/export_simulation_assets.py`
- Isaac Sim 验证：`scripts/run_isaac_insertion.py`
- Isaac 完整视频：`scripts/run_isaac_video.py`

状态诊断、depth 稳定性审计和多视角评审位于 `tools/diagnostics/`；high-FPS
重采样与局部轨迹替换位于 `tools/highfps/`；runner 调用的模型/solver 内部阶段
位于 `tools/stages/`。它们不是新的并行 pipeline。

当前 rice-cooker 数据使用 `configs/*_data_1_8view.json`。这些文件包含当前数据的
路径和先验，不是通用算法的一部分；处理新对象时应生成新配置。

视频渲染默认覆盖 `trajectory.json` 的每一帧。`--timestamps` 只用于快速关键帧
QA，产物不应作为完整视频交付；如需同时保留轨迹开始前的原始画面，可使用
`--include-source-prelude`，该段只显示原图、不伪造 mesh pose。

### `data/1` 八视角数据

`/data_ft_9_10/wentai/projects/data/1` 使用以下配置：

- `configs/data_1_preprocess_8view.json`：八路视频同步抽帧
- `configs/mask_pipeline_data_1_reusable.json`：可配置部件、出现帧、Qwen 种子与 SAM3 分段传播
- `configs/pipeline_data_1_8view.json`：depth gauge 与分部件点云反投影
- `configs/pose_data_1_8view.json`：6D pose 求解、mesh 回投影和评审

物体出现帧是硬约束：`body=40`、`inner_pot=65`、`lid=89`。对应帧之前的 mask、
点云和 mesh 投影均为空。`inner_pot` 的 65–123 帧使用 Qwen 关键帧检测和 SAM3
传播后的修复 mask；`GX013140` 的 `lid` 在 214–245 帧也使用 230 帧
Qwen→SAM3 关键帧做了局部时序修复。

三个 mesh 不从 `data/1` 的场景帧重建，而是从
`/data_ft_9_10/wentai/projects/data/obiect/{body,inner_pot,lid}.mp4` 抽帧，
同样经过 Qwen→SAM3 得到透明背景 RGBA，再输入 ReconViaGen。正式 mesh 位于
`experiments/reconviagen_objects/reconviagen_meshes/`，整条 mesh 流程只使用
`configs/reconviagen_objects.json`：

```bash
.venv/bin/python scripts/run_reconviagen_pipeline.py \
  --config configs/reconviagen_objects.json --stage all
```

runner 会按配置切换 Qwen、SAM 和 ReconViaGen Python 环境；单阶段恢复使用
`--stage frames|masks|rgba|mesh`。

八视角正式批处理调用 `run_pose_pipeline.py`，它依次执行状态诊断、求解、多视角
评审和主视角渲染，并默认复用已验证的阶段产物。

## 验证

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m compileall -q common scripts tests
```

Isaac Sim 脚本必须使用 Isaac Sim 自带的 `python.sh`，不能用项目 `.venv` 直接运行。
具体命令和结果解释见
[docs/SIMULATION_ASSET_PIPELINE.md](docs/SIMULATION_ASSET_PIPELINE.md)。

## 专项文档

- [MASK_PIPELINE.md](MASK_PIPELINE.md)：Qwen + SAM3 多视角 mask 流程
- [CURRENT_BOTTLENECKS_AND_ROADMAP.md](CURRENT_BOTTLENECKS_AND_ROADMAP.md)：
  当前精度瓶颈与路线
- [docs/SIMULATION_ASSET_PIPELINE.md](docs/SIMULATION_ASSET_PIPELINE.md)：
  URDF 与 Isaac Sim 插入验证
