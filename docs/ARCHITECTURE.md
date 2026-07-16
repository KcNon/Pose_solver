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
| `qwen_bbox.py` | Qwen bbox prompt、解析和可视化 |
| `normalized_recon.py` | 当前 normalized DA3/VGGT 数据布局加载 |
| `backproject_utils.py` / `geom.py` | mask + depth 反投影 |
| `depth_gauge.py` | 静态参考部件的逐帧/跨视角深度校正 |
| `icp.py` / `mesh_align.py` | 刚体点云配准与 mesh similarity 标定 |
| `mesh_render.py` | DA3 相机约定下的离屏 mesh 渲染 |
| `simulation_assets.py` | canonical mesh、装配位姿、URDF 和 QA |
| `recon_loader.py` | 旧目录布局兼容层 |

`recon_loader.py` 暂时保留，因为 `align_meshes.py`、`render_meshes.py` 和部分
`scripts/legacy/` 仍引用它。新代码必须使用 `normalized_recon.py`，待历史结果不再
需要复现时再整体删除兼容层。

## 4. 正式流程阶段

| 阶段 | 入口 | 主要输出 |
|---|---|---|
| mask | `run_temporal_mask_pipeline.py` | palette PNG、mask QA |
| depth 诊断 | `diagnose_depth_stability.py` | depth 时序报告 |
| depth gauge | `calibrate_depth_gauge.py` | `depth_gauge.json` |
| 点云 | `backproject_normalized.py` | `parts_ply/<variant>/<frame>/<part>.ply` |
| 状态诊断 | `detect_part_states.py` | `part_states.json` |
| 主求解 | `solve_multiview_pose.py` | `trajectory.json/csv`、配准诊断 |
| body refinement | `refine_body_multiview_yaw.py` | 派生 trajectory、A/B 指标 |
| lid refinement | `refine_lid_multiview_se3.py` | 派生 trajectory、可观测性指标 |
| 渲染 | `render_multiview_pose.py` | overlay、纯 mesh、纹理、坐标轴视频 |
| 六视角评审 | `export_multiview_pose_review.py` | silhouette 指标、关键帧评审图 |
| 仿真资产 | `export_simulation_assets.py` | canonical OBJ、URDF、manifest |
| Isaac 验证 | `run_isaac_insertion.py` | USD、插入报告、最终截图 |

诊断脚本不应改写源 trajectory。refinement 也写入新的 output root，保证每次结果都能
回溯到输入轨迹。

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

## 6. 正式代码、兼容代码与实验代码

### 正式代码

本文件第 4 节列出的入口和它们直接依赖的 `common/` 模块。

### 兼容代码

- `common/recon_loader.py`
- `scripts/align_meshes.py`
- `scripts/render_meshes.py`
- `scripts/legacy/`

它们服务旧的 `configs/pipeline.json` 和早期结果复现。不要在新批处理流程中继续扩展。

### 数据准备/研究工具

`run_da3_joint.py`、`compare_backproject_view.py`、`compare_depth_gauge.py`、
`plot_diagnostics.py` 等用于重建或诊断，保留独立 CLI，但不进入线上最短路径。

## 7. 本轮清理

本轮删除或收敛了以下内容：

- 删除无任何调用方的旧 `common/gauge.py`；正式实现是 `depth_gauge.py`；
- 删除已转向 Isaac Sim 后遗留的 `check_isaac_gym_runtime.py`；
- 删除引用不存在脚本的 `run_full_pipeline.sh`；
- 删除 `pyproject.toml` 中指向不存在模块的 console entry point；
- 抽取重复的 JSON 和 rigid/similarity 变换实现；
- 删除 Isaac demo 中未使用的 contact query、旧相机和临时调试分支。

没有删除仍有调用方的旧 loader/renderer。删除代码的标准是“无调用方且无文档承诺”
或“入口已确认损坏”，而不是仅凭文件较旧。

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
- symmetry/semantic axis；
- 质量、摩擦、碰撞角色和成功阈值。

如果新对象需要修改 `solve_multiview_pose.py` 中的分支才能运行，应先判断该差异能否
抽象成新的 tracking strategy 或约束配置，避免形成按对象复制的 solver。

## 9. 后续架构改进顺序

1. 增加配置 schema 和输入 preflight，尽早报告缺帧、缺 mask、路径和单位错误；
2. 将 solver 中的 tracking strategy 拆成独立模块并建立小型合成数据测试；
3. 用一个轻量 pipeline runner 串联正式阶段，基于输出哈希恢复，而不是 shell 脚本；
4. 当历史流程归档后，删除 `recon_loader.py` 和对应旧脚本；
5. 将 Isaac 场景 authoring 与物理 trial 进一步拆开，便于批量无渲染验证。
