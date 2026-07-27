# pose_solver 代码架构

## 1. 架构目标

项目同时包含 mask、深度、点云、6D pose、渲染和仿真。代码边界遵循三个原则：

1. `common/` 只放可复用算法和稳定数据约定，不读取命令行参数；
2. `scripts/` 每个文件只完成一个可独立恢复的阶段；
3. 数据路径、部件名称、状态区间和实验阈值全部放在 JSON 配置中。

因此，换数据集或换对象时优先替换配置和输入资产，不应复制一份 solver 再硬编码。

## 2. 分层与依赖方向

```text
configs + input data
        ↓
scripts/（CLI / orchestration）
        ↓
common/（data contract / geometry / algorithms）
        ↓
NumPy / OpenCV / SciPy / trimesh / pyrender / small_gicp
```

允许 `scripts → common`，不允许 `common → scripts`。不同脚本之间通过 JSON、PLY、
视频和 mesh 文件交接，避免隐式共享进程状态。

## 3. `common/` 模块职责

| 模块 | 单一职责 |
|---|---|
| `io_utils.py` | UTF-8 JSON 读写 |
| `pose_transforms.py` | rigid/similarity 转换、轴旋转、点变换 |
| `mask_io.py` | normalized mask、视角和时间戳约定 |
| `masking/` | 任意部件配置、独立二值 track、遮挡合成、质量报告与多视角几何先验 |
| `masking/planning.py` | Qwen 多视角出现帧/种子推导与异常段修复计划 |
| `qwen_bbox.py` | Qwen bbox prompt、解析和可视化 |
| `normalized_recon.py` | 当前 normalized DA3/VGGT 数据布局加载 |
| `backproject_utils.py` / `geom.py` | mask + depth 反投影 |
| `depth_gauge.py` | 静态参考部件的逐帧/跨视角深度校正 |
| `cloud_io.py` | 无 ICP 后端依赖的 PLY 读写与最近邻诊断 |
| `icp.py` / `mesh_align.py` | 基础 ICP 兼容接口与 mesh similarity 标定 |
| `gicp.py` / `pose_tracking.py` | 正式多尺度 GICP、对称性连续化和三种跟踪策略 |
| `pose_refinement.py` | silhouette、点云评分、装配/速度约束 |
| `render_loss_refinement.py` | 多视角 sampled-mesh silhouette/depth 损失、有界 SE(3) 搜索和 holdout gate |
| `trajectory_constraints.py` | 通用刚体对接触因子、任意采样表面连续检测、空腔兼容后端与有界轨迹修正 |
| `pose_config.py` / `pose_validation.py` | pose 配置 preflight 与轨迹约束验证 |
| `pose_autoconfig.py` / `mesh_observability.py` | 状态、锚点、标定窗口和 mesh 对称/纹理元数据推导 |
| `stage_cache.py` | 基于命令、内容和产物树的阶段输入指纹 |
| `appearance_pose.py` | 纹理/轮廓候选、对称轴翻面假设、时序路径选择 |
| `symmetry.py` | 连续轴对称、有限阶对称、观测翻转歧义及统一姿态消歧 |
| `calibration_cache.py` | mesh、锚点点云、mask/RGB/相机输入指纹 |
| `trajectory_io.py` | 派生字段归一化及 trajectory JSON/CSV 统一写出 |
| `mesh_render.py` | DA3 相机约定下的离屏 mesh 渲染 |
| `simulation_assets.py` / `simulation_export.py` | canonical mesh、装配位姿、URDF 和 QA |
| `isaac_runtime.py` | USD 导入、碰撞设置、轨迹回放和落座试验 |
| `isaac_video.py` | 复用 USD cache 的完整时间线多视角视频 |
| `isaac_physics_video.py` | 有限力控制、接触统计和完整 PhysX 轨迹视频 |

## 4. 正式流程阶段

| 阶段 | 入口 | 主要输出 |
|---|---|---|
| dataset workflow | `run_automated_workflow.py` | 跨阶段 resolved configs、contract、完成标记 |
| mask | `run_mask_pipeline.py` | 独立部件 track、palette PNG、mask QA |
| depth + 点云 | `run_depth_pipeline.py` | `depth_gauge.json`、分部件 PLY |
| pose 调度 | `run_pose_pipeline.py` | 状态、求解、render-loss、连续几何约束、评审和渲染的断点执行 |
| 渲染 | `render_multiview_pose.py` | overlay、纯 mesh、纹理、坐标轴视频 |
| 仿真资产 | `export_simulation_assets.py` | canonical OBJ、URDF、manifest |
| Isaac 验证 | `run_isaac_insertion.py` | USD、插入报告、最终截图 |
| Isaac 视频 | `run_isaac_video.py` | 完整时间线三视角 MP4 |
| Isaac 物理视频 | `run_isaac_physics_video.py` | 实际刚体与目标 ghost 的完整 PhysX MP4 |

诊断脚本不应改写源 trajectory。refinement 也写入新的 output root，保证每次结果都能
回溯到输入轨迹。

render-loss refinement 不是 6D pose 真值监督，而是利用已标定相机建立的可微性
无关、可量化间接监督：将固定尺度 mesh 的表面采样点投影到多个视角，联合计算
silhouette IoU、轮廓 Chamfer、mask coverage 和截断 depth residual。优化相机与
holdout 相机在配置中分开；候选只有在优化集改善且 holdout 不明显退化时才写入新
trajectory。搜索始终受相对基础 pose 的平移/旋转上限和相邻帧 correction prior
约束，并通过统一 symmetry contract 屏蔽连续轴对称的不可观测自旋。

连续几何约束不是把容器当成封闭实体做 SDF 排斥。`insert_into` 明确区分容器壁、
底部和合法空腔，并对相邻视频帧做 SE(3) 插值采样以检测 tunnelling。接近底部时的
轴线对齐是关系参数，不是对象名称分支。几何候选通过独立相机 render-loss gate 后
才写入轨迹；报告同时保存优化前、纯几何候选和视觉门控后的穿透深度。

可选工具不放在正式入口目录：

| 类型 | 位置 | 用途 |
|---|---|---|
| depth/状态/多视角评审 | `tools/diagnostics/` | QA 与人工验收 |
| high-FPS | `tools/highfps/` | gauge 重采样与局部轨迹替换 |
| runner 内部阶段 | `tools/stages/` | Qwen、SAM 和 pose solver 的跨环境/断点入口 |

## 5. 核心数据约定

### 5.1 相机

DA3 extrinsic 为 world-to-camera：

```text
X_camera = R * X_world + t
```

### 5.2 位姿

`trajectory.json` 中：

- `T_world_from_part`：canonical part frame 到 world 的刚体变换；
- `T_body_from_part`：`inv(T_world_from_body) @ T_world_from_part`；
- `S_world_from_raw_mesh`：raw GLB 到 world 的 similarity，包含固定 mesh scale；
- quaternion 顺序为 `xyzw`。

raw mesh 顶点到 canonical part frame：

```text
X_part = scale * (X_raw - raw_mesh_origin)
```

这套约定由 `common/pose_transforms.py` 和 `common/simulation_assets.py` 共同维护，
solver、renderer 与 URDF 导出不得各自重新实现。

### 5.3 状态

主 trajectory 的求解状态和自动诊断状态是两个层次：

- `state`：solver 实际采用的 `static`、`moving`、`inferred_unobservable`；
- `detected_state`：状态诊断器输出，可包含 occluded/assembled 等语义。

在自动 FSM 完全验证前，状态诊断只作为附加证据，不暗中改变 pose。

## 6. 正式代码

本文件第 4 节列出的入口和它们直接依赖的 `common/` 模块。`data/1` 的正式批处理
入口是 `run_pose_pipeline.py`；各阶段通过 `--stages` 独立恢复。

## 7. 本轮清理

本轮把 `data/1` 的正式流程收敛到 `run_pose_pipeline.py`，并将点云 I/O、三种 tracking
strategy、轨迹派生字段和验证逻辑抽到 `common/`。旧六视角 V6、FoundationPose
候选实验和旧 mask 兼容入口已移除，避免它们继续成为并行但不一致的 pipeline。

## 8. 批量复用方式

批量处理时建议由上层任务系统为每个序列生成一份配置，并按阶段检查输出标记：

```text
mask.complete
depth_gauge.json
parts_ply complete
trajectory.json
render metrics
manifest.json
isaac.complete.json
```

对象差异应通过以下配置表达：

- parts 和 palette IDs；
- reference part；
- mesh 路径；
- static/dynamic/assembly 先验；
- `symmetry.equivalence`（`none`、`continuous_axial`、`cyclic`）及
  `axis_raw`/`discrete_order`；
- `symmetry.observation_ambiguities`（例如 `axis_flip`）及 appearance 证据帧；
- 通用单帧平移、SO(3) 旋转和对称轴方向变化门限；
- 质量、摩擦、碰撞角色和成功阈值。

后续 tracking/refinement 代码统一调用
`symmetry_spec_from_state(state)` 和
`resolve_symmetric_pose(measured_pose, reference_pose, symmetry)`；不要再各自
枚举绕轴角度。连续轴对称使用解析解，`axis_flip` 只在显式启用观测歧义候选时参与选择。

锚点先由点云得到尺度、平移和几何旋转候选，再由多视角 silhouette 与纹理边缘解除
对称歧义；多个锚点通过有界运动先验联合选路。该流程不检查物体名称。输入指纹不一致
时，`--reuse-calibration` 会拒绝旧缓存，必须重标定或显式强制复用。

如果新对象需要修改 `solve_multiview_pose.py` 中的分支才能运行，应先判断该差异能否
抽象成新的 tracking strategy 或约束配置，避免形成按对象复制的 solver。

## 9. 后续架构改进顺序

1. 增加跨视角 silhouette、depth 和时序 pose 的联合回归测试；
2. 将 Qwen/SAM/ReconViaGen 的模型版本和权重摘要纳入 workflow provenance；
3. 将 Isaac 场景 authoring 与物理 trial 进一步拆开，便于批量无渲染验证。
