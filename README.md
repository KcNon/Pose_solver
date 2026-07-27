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
   ↓ 多视角 pose solve + 对称感知 render-loss refinement
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

- 数据集端到端自动流程：`scripts/run_automated_workflow.py`
- Mask 全流程：`scripts/run_mask_pipeline.py`
- ReconViaGen 全流程：`scripts/run_reconviagen_pipeline.py`
- 深度与点云：`scripts/run_depth_pipeline.py`
- Pose 全流程：`scripts/run_pose_pipeline.py`
- 视频渲染：`scripts/render_multiview_pose.py`
- URDF 导出：`scripts/export_simulation_assets.py`
- Isaac Sim 验证：`scripts/run_isaac_insertion.py`
- Isaac 完整视频：`scripts/run_isaac_video.py`
- Isaac 完整物理驱动视频：`scripts/run_isaac_physics_video.py`

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
渲染损失修正、连续几何约束、评审和主视角渲染，并默认复用已验证的阶段产物。渲染损失阶段以
固定尺度的 canonical mesh 回投影为预测，以 Qwen→SAM3 mask、轮廓和 DA3 depth
为间接监督；只优化有观测的运动区间，并保留独立相机作为 holdout gate。连续轴
对称物体不优化不可观测的轴向自旋。该阶段写出新的
`trajectory_render_refined.json`，不会覆盖基础轨迹。

`trajectory_constraints` 随后在 pose solver 内检查原始帧及相邻帧插值，不调用
Isaac 或其他物理引擎。通用 `pairwise_contact` 后端只接收任意 reference/moving
刚体、几何代理和可组合的非穿透、接触、轴对齐及轴线偏移因子；`insert_into`
保留为理解空腔语义的兼容后端。几何候选和静态传播都需通过多视角
render-loss gate，最终写入 `trajectory_collision_refined.json`。物体名称、
关系、代理形状、检测子帧数和修正上限都来自配置，不在 solver 中硬编码。
pose 侧代理位于独立的 `geometry_proxy_config`，与质量、摩擦或 PhysX 配置解耦。

端到端执行使用一份只引用各阶段配置的 workflow 文件。runner 会生成 runtime
配置，将实际 mask、ReconViaGen mesh、depth gauge 和点云路径显式串起来，并拒绝
视角、部件或 palette ID 不一致。depth、点云和 pose 阶段均使用输入内容指纹；
mask、相机深度、mesh 或配置变化时不会仅因旧输出仍存在而错误恢复：

```bash
.venv/bin/python scripts/run_automated_workflow.py \
  --config configs/workflow_data_1.json --stage preflight

.venv/bin/python scripts/run_automated_workflow.py \
  --config configs/workflow_data_1.json --stage all
```

只更新 mask、且 RGB 与固定相机 DA3 结果未变化时，可使用
`--stage depth-postprocess` 仅重算 depth gauge 和分部件点云。

Pose 配置中的 `automation.enabled` 会把状态诊断解析成一份普通
`resolved_pose_config.json`。运动区间、静止补集、标定窗口、anchor 窗口及纹理
证据帧都在求解前确定并记录来源；solver 本身不暗中修改配置。对称性分析只在没有
显式 override 时生效，带纹理 mesh 会自动启用 appearance 证据。

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
