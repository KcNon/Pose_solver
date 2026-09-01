# Pose Solver

Pose Solver 将同步多视角视频或图像转换为刚性部件的逐帧 6D pose，并通过
多视角 mesh 回渲染检查结果。默认流程覆盖：

```text
frames → mask → depth / camera rig / point cloud → pose → review / render
```

项目对外只有一个生产入口：

```bash
.venv/bin/python -u -m pose_solver run --config configs/my_object.json
```

`scripts/` 和 `tools/stages/` 是内部阶段适配器。生产 mask、depth、pose、mesh
和视频任务不要直接调用它们，也不要用临时 `python -` 脚本绕过资源保护。

## 快速开始

先检查数据契约和 GPU 配置：

```bash
.venv/bin/python -u -m pose_solver run \
  --config configs/my_object.json \
  --stage preflight
```

查看将执行的阶段和 resolved config，不运行模型：

```bash
.venv/bin/python -u -m pose_solver run \
  --config configs/my_object.json \
  --dry-run
```

运行或继续完整流水线：

```bash
.venv/bin/python -u -m pose_solver run \
  --config configs/my_object.json
```

只运行一个阶段：

```bash
.venv/bin/python -u -m pose_solver run --config configs/my_object.json --stage mask
.venv/bin/python -u -m pose_solver run --config configs/my_object.json --stage depth
.venv/bin/python -u -m pose_solver run --config configs/my_object.json --stage pose
.venv/bin/python -u -m pose_solver run --config configs/my_object.json --stage render
```

`--force` 会使选中阶段重新计算。使用前确认输出目录和正在运行的进程，避免
两个重任务同时写同一实验。

只读检查已有结果：

```bash
.venv/bin/python -u -m pose_solver inspect --config configs/my_object.json
```

## 数据流

```text
同步视频 / 已抽帧图像
          │
          ▼
frames：统一视角、时间轴和帧号
          │
          ▼
mask：检测、SAM 时序跟踪、多视角验证和标签合成
          │
          ▼
depth：DA3 深度与 camera rig、尺度校准、分部件质量点云
          │
          ▼
pose：状态检测、绝对 anchor、相邻帧跟踪、渲染与多帧优化
          │
          ├── trajectory_final.json
          ├── multiview_metrics.json
          ├── keyframe review
          └── 单视角 / 多视角渲染视频
```

世界坐标系通常由稳定参考部件和固定 camera rig 共同确定。参考部件不能简单
设为单位 pose；mesh 相对相机的尺度、旋转和平移仍必须由多视角观测标定。

### Pose 阶段内部流程

当前 pose adapter 按以下顺序执行：

1. 从 mask 和点云自动识别静止、运动和装配区间；
2. 选择稳定参考窗口，验证桌面支撑和观测质量；
3. 对每个刚性 part 求尺度、绝对 anchor 和外观分支；
4. 在运动区间进行多尺度点云配准；
5. 检查 endpoint 旋转分支，拒绝缺少几何证据的大角度修正；
6. 用多视角 silhouette、轮廓和深度做 render-loss refinement；
7. 进行多帧时序优化和可选装配约束；
8. 对静止区间做 robust consensus 锁定；
9. 输出轨迹、质量指标、关键帧和渲染视频。

ICP 在这里是粗 pose 之后的局部跟踪器，不是独立的全局位姿估计器。

没有 pose GT 时，render-loss refinement 可以用未参与优化的同步相机反向修正
粗 pose。四个及以上相机的自动配置会保留独立 holdout，并按帧轮换缺省留出
视角；holdout 退化、最差视角失败或手部遮挡证据不足时不会写回候选。原理、
配置和报告字段见[无 GT 的多视角 Pose 修正](docs/multiview_pose_correction.md)。

## 输入契约

### 多视角帧

```text
frames/
├── GX000001/
│   ├── 000300.jpg
│   ├── 000301.jpg
│   └── ...
├── GX000002/
│   └── ...
└── ...
```

- 视角目录必须与 `input.views` 完全一致；
- 所有视角使用同一个六位帧号时间轴；
- `input.frame_range` 是闭区间 `[start, end]`；
- 跨相机不同步会同时破坏点云融合和 render loss。

若输入是视频，在 `input.videos` 中为每个视角提供路径。frames 阶段会根据
`sample_fps` 和 `sync_offsets_s` 生成统一帧目录。

### Mesh

每个 part 对应一个 `<part>.glb`。所有 mesh 必须位于同一个逻辑目录；可以
使用同目录 symlink 指向各自重建结果。

```text
meshes/
├── nozzle.glb
└── body.glb
```

Pose Solver 假设一个 part 内部是刚性的。软管、电线、布料等变形结构应：

- 从 pose loss、点云和评价 mask 中排除；或
- 拆成独立模型，不与刚性主体共享一个 SE(3)。

### Mask

内部标准布局为 frame-first：

```text
masks/
├── 000300/
│   ├── GX000001.png
│   └── GX000002.png
└── ...
```

PNG 是单通道标签图，背景为 0，part 标签与配置中的 `id` 一致。已有 mask
若采用 `masks/<view>/<frame>.png`，设置：

```json
"mask": {
  "mode": "reuse",
  "artifact": "/data/object/masks",
  "overrides": {"source_layout": "view_first"}
}
```

统一入口会在实验输出内生成标准化链接，不修改源 mask。

### 深度和点云

质量点云标准布局为：

```text
parts_ply/da3_self_cond_quality/
├── 000300/
│   ├── nozzle.ply
│   └── body.ply
└── quality_cloud_summary.json
```

如果只复用分部件点云，将 `depth.mode` 设为 `reuse`。如果 pose 中启用了
桌面验证、深度项或 render refinement，还应通过 `input.depth_dir` 指向对应
DA3 reconstruction：

```json
"input": {
  "depth_dir": "/data/object/da3-self-cond"
},
"depth": {
  "mode": "reuse",
  "artifact": "/data/object/parts_ply/da3_self_cond_quality"
}
```

不要把来自不同 camera rig、帧范围或尺度规范的点云和 DA3 目录混用。

## 单一源配置

从 [`configs/pipeline.example.json`](configs/pipeline.example.json) 开始。常规
数据集只需要维护这一份配置：

```json
{
  "schema_version": 1,
  "dataset": "my-object",
  "input": {
    "frames_dir": "/data/my-object/frames",
    "views": ["GX000001", "GX000002"],
    "frame_range": [300, 500]
  },
  "parts": {
    "nozzle": {
      "id": 1,
      "mesh": "/data/my-object/meshes/nozzle.glb",
      "prompts": ["rigid pump head and threaded collar"],
      "appearance_hint": 300
    },
    "body": {
      "id": 2,
      "mesh": "/data/my-object/meshes/body.glb",
      "prompts": ["bottle body and neck"],
      "appearance_hint": 300,
      "reference": true
    }
  },
  "output": {"root": "/experiments/my-object/baseline"},
  "runtime": {
    "devices": [6],
    "egl_device": 6,
    "memory_guard": {
      "enabled": true,
      "minimum_available_gib": 128,
      "maximum_process_rss_gib": 32,
      "poll_seconds": 1,
      "report_seconds": 10,
      "stop_grace_seconds": 2
    }
  },
  "models": {
    "qwen_python": "/path/to/qwen/.venv/bin/python",
    "qwen_model": "/path/to/qwen/model",
    "sam_python": "/path/to/sam/.venv/bin/python",
    "sam_checkpoint": "/path/to/sam/checkpoint.pt",
    "da3_python": "/path/to/da3/.venv/bin/python"
  },
  "mask": {"mode": "run"},
  "depth": {"mode": "run"},
  "pose": {"mode": "run"}
}
```

关键规则：

- part 名、mesh 文件名和 mask 标签语义必须一致；
- `id` 唯一且位于 1～255；
- 最多配置两张 GPU；
- `appearance_hint` 只提供 mask 搜索起点，不是 pose anchor；
- `reference: true` 表示物理参考部件，不代表它的 pose 是单位矩阵；
- 算法微调写在阶段的 `overrides` 中，不复制内部 resolved config。

### Run 与 reuse

每个生产阶段只有两种模式：

```json
"mask": {"mode": "run"},
"depth": {"mode": "reuse", "artifact": "/data/object/parts_ply"},
"pose": {"mode": "reuse", "artifact": "/exp/pose/trajectory_final.json"}
```

`reuse` 只表示跳过计算，不会自动证明 artifact 与当前视角、帧范围、mesh、
mask 或 camera rig 相容。新实验应使用独立 `output.root`。

### 全视角渲染

在 `pose.overrides` 中配置：

```json
"render": {
  "views": "all",
  "primary_view": "GX000001",
  "width": 1280,
  "height": 720,
  "basic_only": true,
  "grid": {
    "enabled": true,
    "video_names": ["overlay.mp4", "mesh_only.mp4"]
  }
}
```

`--stage render` 可以对已有 `trajectory_final.json` 重新生成视频，不重新求解
pose。多视角 grid 会检查输入视频和期望帧数后再编码。

### 已有 frames、mask 和 DA3 的完整运行示例（Object-1）

Object-1 展示了最常见的新数据接入方式：同步帧、view-first 标签 mask、重建
mesh 和逐帧 DA3 已经存在，但还没有质量点云、pose 和完整渲染视频。对应单一
源配置是
[`configs/objects_0827/object1_current_pipeline_full.json`](configs/objects_0827/object1_current_pipeline_full.json)。

这个配置中的阶段语义是：

- `mask.mode=reuse` 和 `source_layout=view_first`：标准化现有 mask，不重新运行
  检测或 SAM；
- `input.depth_dir` 加 `depth.mode=run`：验证现有 DA3 每一帧，然后运行 depth
  后处理并生成质量点云，**不会重新运行 DA3 模型**；
- `pose.mode=run`：执行当前完整 pose、render-loss、多帧优化和静态段锁定；
- `render.views=all`：输出每个视角的视频以及 4×2 多视角 grid；
- Object-1 的标签 3 是手，`mask_occluder_labels=[3]` 用于最终遮挡感知渲染。

先检查资源、输入和实际执行计划：

```bash
free -h
ps -eo user,pid,pgid,rss,etime,args | \
  rg 'pose_solver|solve_multiview_pose|render_multiview_pose'

.venv/bin/python -u -m pose_solver run \
  --config configs/objects_0827/object1_current_pipeline_full.json \
  --stage preflight

.venv/bin/python -u -m pose_solver run \
  --config configs/objects_0827/object1_current_pipeline_full.json \
  --dry-run
```

确认没有相同 config 或输出目录的任务后，运行或断点继续完整流程：

```bash
.venv/bin/python -u -m pose_solver run \
  --config configs/objects_0827/object1_current_pipeline_full.json
```

生产阶段会自动进入进程组内存守护器。命令暂时没有终端输出时，应检查
`runtime/memory_guard.jsonl` 和精确 PID，不能启动第二份相同任务。普通续跑
会根据 checkpoint 跳过有效产物；只有确认旧产物需要重算且没有活动进程时才
使用 `--force`。

可以单独续跑后续阶段：

```bash
# 已有 DA3，重新生成/继续质量点云
.venv/bin/python -u -m pose_solver run \
  --config configs/objects_0827/object1_current_pipeline_full.json \
  --stage depth

# 已有质量点云，继续 pose、review 和配置中的全部视角视频
.venv/bin/python -u -m pose_solver run \
  --config configs/objects_0827/object1_current_pipeline_full.json \
  --stage pose

# 已有 trajectory_final.json，只重新生成视频
.venv/bin/python -u -m pose_solver run \
  --config configs/objects_0827/object1_current_pipeline_full.json \
  --stage render
```

运行结束后先做只读汇总：

```bash
.venv/bin/python -u -m pose_solver inspect \
  --config configs/objects_0827/object1_current_pipeline_full.json
```

本配置的主要交付物位于：

```text
experiments/objects_0827/Object-1/current_pipeline_full/
├── depth/parts_ply/da3_self_cond_quality_hand_excluded/
├── pose/pose/trajectory_final.json
├── pose/diagnostics/multiview_metrics.json
└── pose/render/
    ├── multiview_overlay.mp4
    └── multiview_mesh_only.mp4
```

多视角 grid 生成时会核对输入帧数；交付前还应通过有界视频验证 CLI 完整
解码，不能用临时 Python 视频脚本：

```bash
.venv/bin/python -u tools/diagnostics/run_with_memory_guard.py \
  --log experiments/objects_0827/Object-1/current_pipeline_full/runtime/memory_guard_video_validation.jsonl \
  --cuda-visible-devices 1 \
  --minimum-available-gib 128 \
  --maximum-process-rss-gib 32 \
  --poll-seconds 1 --report-seconds 10 --stop-grace-seconds 2 \
  -- .venv/bin/python -u tools/diagnostics/validate_render_videos.py \
  --videos \
    experiments/objects_0827/Object-1/current_pipeline_full/pose/render/multiview_overlay.mp4 \
    experiments/objects_0827/Object-1/current_pipeline_full/pose/render/multiview_mesh_only.mp4 \
  --expected-frames 231 --expected-width 1920 --expected-height 540 \
  --expected-fps 29.97 --fps-tolerance 0.01 \
  --report experiments/objects_0827/Object-1/current_pipeline_full/pose/render/video_validation.json
```

### Object-1 喷头-only 刚性流程

Object-1 的 `sprayer_pump` 原始标签同时包含刚性喷头和会弯曲的吸管，不能把
两者放进同一个 SE(3) pose。喷头版本只保留粗触发头、喷嘴和瓶口连接环；吸管
不参与 mask、点云、尺度标定、pose loss 或质量评价。瓶身标签 2 和手部标签 3
保持不变，手部作为已知 modal occluder，而不是喷头背景。

先用有界预处理生成喷头可见 mask 和去吸管 mesh。该步骤不会修改原始数据，
也不会调用 `Trimesh.split()`：

```bash
.venv/bin/python -u tools/diagnostics/run_with_memory_guard.py \
  --log experiments/objects_0827/Object-1/rigid_nozzle_v1/runtime/memory_guard_inputs.jsonl \
  --minimum-available-gib 128 --maximum-process-rss-gib 32 \
  -- .venv/bin/python -u tools/stages/preprocess/prepare_rigid_pose_inputs.py \
  --config configs/objects_0827/object1_rigid_nozzle_inputs.json
```

然后所有生产阶段都从统一入口运行：

```bash
.venv/bin/python -u -m pose_solver run \
  --config configs/objects_0827/object1_rigid_nozzle_pose.json \
  --stage preflight

.venv/bin/python -u -m pose_solver run \
  --config configs/objects_0827/object1_rigid_nozzle_pose.json
```

该配置会用喷头-only mask 重新生成质量点云。尺度标定仅比较未被手或瓶身遮挡
的可见渲染区域，并且只允许一次尺度选择，避免把 modal mask 误解释为较小的
物理喷头。510–520 帧是完全手遮挡区，在瓶身坐标系内由 509/521 两个端点做
SE(3) 桥接；这是无观测插值，不声明该区间存在视觉测量。

521 帧重新可见后，局部 ICP/render loss 的初值仍可能落在错误盆地。当前配置先
用多视角喷头 mask 做有界全局重捕获；跨遮挡的首帧允许大范围平移，连续可见帧
则以最近可信 pose 为软平移/旋转先验。单个被手遮挡视角不会再一票否决一个在
其余视角上一致的候选。这样既能从视野外重新接回喷头，也不会让每帧独立选择
不同的弱纹理旋转分支。

Object-1 视频中喷头从拿起到放上瓶口不应自由自旋，因此配置使用显式的任务结构
约束：458–501 帧从首个可信朝向平滑过渡到 569 帧由多视角轮廓确定的最终朝向，
502 帧以后保持完整刚体朝向，只继续优化平移。实现上的过渡由同一对可信端点
生成，不能把每个含噪 ICP 朝向分别插值到终态。557–568 帧仍使用
`coaxial_snap_window` 提供接触初值；随后 569–630 帧的常量 pose 会在
569/583/601/616 上优化，并由 630 帧独立 holdout 验证。最终采用视觉 pose，
不会为了满足可能有偏的接口轴标定而强行改坏喷头轮廓；因此该结果用于视觉
轨迹/视频，不应在 `connector_readiness.json` 未通过时直接宣称仿真就绪。

主要结果位于：

```text
experiments/objects_0827/Object-1/rigid_nozzle_v1/pose_pipeline/
├── depth/parts_ply/da3_self_cond_quality_rigid_nozzle_hand_excluded/
├── pose/pose/trajectory_final.json
├── pose/diagnostics/multiframe_optimization.json
├── pose/diagnostics/render_loss_refinement.json
├── pose/diagnostics/point_cloud_jump_518_530/
│   ├── sprayer_pump_point_cloud_GX012854.mp4
│   └── point_cloud_metrics.json
└── pose/render/
    ├── multiview_overlay.mp4
    ├── multiview_mesh_only.mp4
    └── video_validation_multiview_final.json
```

若 mask/depth 已存在，只重跑 pose 和完整渲染可使用：

```bash
.venv/bin/python -u -m pose_solver run \
  --config configs/objects_0827/object1_rigid_nozzle_pose.json \
  --stage pose --skip-render

.venv/bin/python -u -m pose_solver run \
  --config configs/objects_0827/object1_rigid_nozzle_pose.json \
  --stage render
```

## 输出目录

```text
<output.root>/
├── runtime/
│   ├── pipeline_manifest.json
│   ├── mask.resolved.json
│   ├── depth.resolved.json
│   ├── pose.resolved.json
│   └── memory_guard.jsonl
├── mask/
├── depth/
│   ├── da3-self-cond/
│   └── parts_ply/
└── pose/
    ├── automation/
    │   ├── resolved_pose_config.json
    │   └── pose_autoconfig_report.json
    ├── pose/
    │   ├── calibration.json
    │   ├── trajectory.json
    │   ├── trajectory_render_refined.json
    │   ├── trajectory_multiframe.json
    │   ├── trajectory_final.json
    │   └── pair_registrations.json
    ├── diagnostics/
    │   ├── part_states.json
    │   ├── trajectory_validation.json
    │   ├── render_loss_refinement.json
    │   ├── multiframe_optimization.json
    │   └── multiview_metrics.json
    ├── review/keyframes/
    └── render/
```

`trajectory_final.json` 是下游默认输入。中间轨迹用于定位误差由哪个阶段引入，
不应在未审查的情况下替代最终轨迹。

## 质量检查

至少检查以下内容：

1. `part_states.json` 的运动区间是否与视频一致；
2. anchor 是否来自多视角可见、低遮挡的稳定帧；
3. `pair_registrations.json` 是否出现低 fitness、大平移或异常旋转分支；
4. `trajectory_validation.json` 是否存在单帧跳变；
5. `multiview_metrics.json` 的均值和最差视角是否同时可接受；
6. 关键帧和完整视频中是否存在漂移、穿手、尺度错误或错误对称分支。

IoU 高不等于绝对 6D pose 一定正确。以下情况可能产生外观正确但几何错误的
解：轴对称零件、单目深度尺度漂移、错误 camera rig、低重叠点云以及手部
遮挡。没有 GT 时，应同时报告多视角 IoU、轮廓误差、轨迹连续性和可观测性。

手部 mask 若存在，应作为 occluder 从 depth、点云和 IoU 中排除。仅在视频
overlay 中把 mesh 画在手上方或下方，不等于优化已经正确处理遮挡。

### 初始静态位姿与装配锁存

若零件在序列开头保持静止，但单帧 anchor 没有对齐，可让自动配置在初始静态
窗口上执行同一个常量 SE(3) 的多帧、多视角 refinement：

```json
{
  "pose": {
    "overrides": {
      "multiframe_optimization": {
        "auto": {
          "refine_initial_static_pose": true,
          "initial_refinement_parts": ["blade"],
          "initial_constant_optimize_rotation": true
        }
      }
    }
  }
}
```

对“子零件插入父零件后被遮挡”的情况，应显式声明真实装配父零件，而不是把
全局 pose reference 当成所有零件的父零件：

```json
{
  "pose": {
    "overrides": {
      "states": {
        "blade": {
          "assembly_parent": "bowl",
          "assembly_latch": {
            "minimum_stable_frames": 12,
            "minimum_approach_m": 0.05,
            "distance_tolerance_m": 0.025,
            "maximum_relative_anchor_frames": 5,
            "maximum_relative_translation_residual_m": 0.03,
            "maximum_relative_rotation_residual_deg": 15.0
          }
        }
      }
    }
  }
}
```

锁存只有在“零件先运动、明显接近父零件、随后连续稳定且父子距离稳定”全部成立
后才确认。确认后，后续 mask 抖动或盖子遮挡不会重新开启子零件跟踪；最终轨迹
使用确认窗口估计的 `T_parent_from_part`，将子零件以 `assembled_rigid_follow`
状态刚性附着到父零件。父零件不动时子零件保持不动，父零件移动时二者同步；即使
子零件完全不可见，`pose_valid` 仍保持为 true，renderer 也继续保留该 mesh。

应在 `part_states.json` 中检查 `detected_assembled_from`、
`detected_assembled_confirmed_at` 和 `assembly_parent`，并在
`static_pose_consensus.json` 中确认 `rigid_follow` 的父零件、证据帧、相对位姿
残差及 `applied_frame_count`。若没有有效接近或稳定窗口，系统不会自动锁存。

## Connector 与仿真

可选 `connectors` 描述插入、旋转或螺纹等装配关系。它用于质量门控和约束，
不能替代可靠的视觉 pose。进入 Isaac Sim 前必须确认：

- pose、尺度和 camera rig 已通过多视角检查；
- connector 轴、原点、间隙和碰撞 proxy 有明确几何证据；
- 物理碰撞使用低面数 proxy，不直接拆分高密度 reconstruction mesh。

面向组装的数据可进一步声明 `assembly_task`。它将完整轨迹划分为连续阶段，并为每个
阶段指定物理语义：`external_kinematic_constraint` 表示手或夹爪仍在持握，
`connector_constraint` 表示插入约束开始主导，`terminal_hold` 表示装配终态。
诊断结果刻意拆成两个互不替代的门：

- `pose_product_ready`：终态轨迹稳定，并且独立多视角视觉证据与当前轨迹哈希一致；
- `physics_replay_ready`：connector 几何、制造参数、碰撞验证以及手/夹爪约束均已准备。

因此没有吸管的仿真仍可使用观测到的刚性喷嘴 Pose，但抓持和接近段必须由外部运动学
约束驱动；缺少吸管会改变质量、惯量和碰撞，不能把这一简化仿真当成原视频的自由动力学
复现。

Isaac 导出是 pose 通过质量门后的独立步骤，不属于默认 pipeline。

对于上重下轻的喷嘴，直接自由落体不能代表手或机器人执行的插入；应使用有界力/力矩
跟踪装配轨迹，并在终点保持一段时间检查瓶口接触。只有终点位置、旋转和接触均通过
门限，才在 USD 中写入装配固定关节。Object-4 的无软管喷嘴示例配置为
`configs/objects_0827/object4_simulation_ready.json`，PhysX 参数保存在
`configs/objects_0827/object4_isaac_physics.json`。

Object-4 曾因 427--500 帧在状态检测后被提前固定，multiframe 阶段又以
`already_constant_pose` 跳过，导致错误终点传播到整段静态轨迹。当前配置会对“动态段
之后的恒定位姿段”进行有界的多帧精确渲染优化，并保留独立 holdout 帧；本次自动接受
的瓶体坐标系平移为 `[-4, +9, 0] mm`，旋转为 `0°`，优化帧平均 IoU 从
0.558 提升到 0.596。独立轴向扫描显示残余最优偏移为 +5 mm；继续施加旧的 Isaac
+15 mm 补偿会把平均 IoU 从 0.587 降到 0.544，因此该补偿已删除。

去掉补偿后，PhysX 的无螺纹碰撞代理仍停在离视觉目标 10.54 mm 的位置，未通过 5 mm
装配门；但末段接触占比为 49.2%，接触连续性通过。这一差异应归入 connector 原点、
碰撞 proxy、间隙和缺少螺纹约束，而不能再反向修改视觉 Pose。物理视频中，透明青色
mesh 是 trajectory target，带纹理 mesh 是实际 PhysX 位姿；只有配置显式声明
`assembly_target_corrections` 时，青色才表示接触修正后的目标。

Object-4 当前的装配契约将 300--349、350--387、388--416、417--426 和 427--500
依次定义为 pregrasp、grasped transport、approach、insertion 和 assembled。
终态视觉门通过，因此 `pose_product_ready=true`。瓶颈轴由 mesh 拟合的缩放后 RMS
残差约 0.37 mm；喷嘴 collar 重建拟合残差约 0.75 mm，且拟合轴与终态相差约
49.5°，不能作为独立机械证据。加上螺距、螺纹相位、径向间隙、碰撞验证和手/夹爪模型
均缺失，当前必须报告 `physics_replay_ready=false`，不能用终态 Pose 反推一个零误差轴。

装配终点通过力控接触门不等于喷嘴可以在无螺纹时被动站稳。可使用
`scripts/run_isaac_insertion.py --place-release-only` 从装配目标上方 3 mm
释放喷嘴；该模式明确关闭轨迹控制、轴向预载和固定关节，并通过末段持续接触、横向
漂移、轴向漂移、倾角和速度共同验收。Object-4 的中心、±1 mm 横向偏移和 ±1°
倾角共 5 个试验均失败：喷嘴先接触瓶口，随后因上重下轻且没有螺纹约束而倾倒或滑离。
因此当前 FixedJoint 只能表示已经拧紧的终态，不能代替螺纹装配过程。

### 通用的冻结 Pose 物理验证（当前推荐）

当前推荐入口是 `scripts/run_isaac_insertion.py --validate-only`，详细契约和命令见
[通用装配物理验证平台](docs/assembly_validation_platform.md)。它与
`--physics-refine-only` 严格分离：验证从未经接触补偿、未经轴线对齐的视觉装配终态
直接释放，禁止控制器、预载、FixedJoint 和轨迹改写。首项结果回答视觉 Pose 本身是否
物理可行；其余固定的 ±1/2/5 mm、±1/3/5° 试验只报告恢复域，不能当作 GT。

碰撞代理新增通用 `cylindrical_sleeve`。它把任意方向的空心套筒导出成多个低面数凸
楔块，避免动态凸分解把孔填死。接口尺寸来源必须声明为 measured、CAD、mesh fit 或
engineering estimate；只有测量/CAD 且高置信度时才允许声称 metric physical
accuracy。新增同类物体只改配置，不写对象专用代码。

Object-4 的新 preflight 通过，但 0.5 mm 径向间隙、质量和摩擦仍是低置信度工程假设，
因此只允许做可行性测试。冻结视觉 Pose 的 baseline 通过：无横向漂移、无可见倾斜，
在重力下沿接口轴下降 6.00 mm 后落到代理承托面并保持 PhysX 接触。完整固定试验中
baseline 为 1/1，24 个扰动为 6/24；这只说明当前假设下的被动可行性与较窄、明显
不对称的稳定域，不证明 metric Pose 准确。Object-9 使用同一资产
生成和 preflight 代码，但因缺少功能接口声明、已有 connector 轴线/径向证据失败而在
启动 Isaac 前被阻止。平台不会为了生成视频猜测其瓶口尺寸。

Object-4 报告位于
`experiments/objects_0827/Object-4/simulation_ready/simulation_validation/isaac_final_full/qa/assembly_validation_report.json`；Object-9
的失败关闭报告位于
`experiments/objects_0827/Object-9/simulation_validation/preflight_report.json`。
旧的吸管约束物理投影和完整视频保留为平台开发前的有界物理投影 Demo，不能作为 Pose
精度的独立验证。

### 吸管约束的物理 Pose 投影

Object-4 还提供 `--physics-refine-only`：Isaac 不再只是判定轨迹是否可重放，而是在
视觉 Pose 附近做一个严格有界的物理投影。该模式使用低面数、规则的喷嘴连接环和
瓶体/瓶颈碰撞 proxy，在若干毫米与若干度的候选邻域内释放刚体；重力、瓶口接触和
吸管的弹性横向力共同决定稳定状态。搜索只接受仍落在视觉门限内、持续接触、速度和
穿透均合格的结果，并将修正写入独立的
`trajectory_physics_refined.json`，不会覆盖视觉轨迹。

吸管始终不进入 ICP 或 render loss。未组装时，其根部与方向由 nozzle connector
变换给出；进入插入阶段后，其导向目标来自瓶体中轴线，仿真只计算横向挠曲和弯矩。
轴向不施加拉力，绕瓶轴旋转仍来自视觉，因此该模型不会伪造螺纹锁紧或用吸管修正
不可观测的 yaw。实现的是 stage-aware elastic rod proxy，不是可变形吸管网格。

Object-4 的最终 6 个有界候选全部通过。选中状态相对视觉 Pose 平移 0.919 mm
（轴向仅 -0.001 mm）、倾角 0.001°，最大穿透 0.017 mm，末态线速度
0.000614 m/s；轨迹从 417--427 帧平滑应用修正并延续到第 500 帧。整个过程没有
pose controller、轴向预载或 FixedJoint。可复现命令为：

```bash
/data_ft_9_10/ziang/code/IsaacSim/_build/linux-x86_64/release/python.sh \
  -u scripts/run_isaac_insertion.py \
  --asset-root experiments/objects_0827/Object-4/simulation_ready/simulation_assets \
  --runtime-output-dir experiments/objects_0827/Object-4/simulation_ready/isaac_runtime \
  --physics-refine-only --capture --device cpu
```

共享服务器上必须再由 `tools/diagnostics/run_with_memory_guard.py` 包裹该命令，并按
配置锁定 `runtime.devices`。本例配置见
`configs/objects_0827/object4_isaac_physics.json`，结果报告、独立轨迹和视频位于
`isaac_runtime/qa/isaac_physics_pose_refinement_report.json`、
`isaac_runtime/pose/trajectory_physics_refined.json` 和
`isaac_runtime/video/physics_pose_refinement.mp4`。

### Object-4 完整装配视频

完整视频使用物理精修轨迹覆盖 300--500 帧，并追加 3 秒终态保持。由于当前资产没有
手或夹爪，300--426 帧由外部运动学约束表示视频中的手，不能误解为喷嘴自由飞行；
427 帧开始释放，此后只使用重力、瓶口接触和吸管弹性。终态不使用 pose controller、
轴向预载或 FixedJoint。完整渲染命令必须由内存守护包裹，例如：

```bash
.venv/bin/python \
  tools/diagnostics/run_with_memory_guard.py \
  --log experiments/objects_0827/Object-4/simulation_ready/runtime/complete_assembly_video_final_guard.jsonl \
  --cuda-visible-devices 1 --maximum-process-rss-gib 32 \
  --minimum-available-gib 128 --minimum-gpu-free-mib 4096 -- \
  /data_ft_9_10/ziang/code/IsaacSim/_build/linux-x86_64/release/python.sh \
  -u scripts/run_isaac_physics_video.py \
  --asset-root experiments/objects_0827/Object-4/simulation_ready/simulation_assets \
  --runtime-root experiments/objects_0827/Object-4/simulation_ready/isaac_runtime \
  --output-dir experiments/objects_0827/Object-4/simulation_ready/isaac_complete_assembly \
  --trajectory experiments/objects_0827/Object-4/simulation_ready/isaac_runtime/pose/trajectory_physics_refined.json \
  --tube-constrained-assembly --start-frame 300 --end-frame 500 --fps 5 \
  --blocked-error-m 0.02 --width 1280 --height 720 --device cpu
```

最终视频为 1280×720、5 fps、216 帧、43.2 秒。被动终态保持通过：最后 0.5 秒接触
占比 100%，最终位置误差 0.000553 mm、旋转误差 0.000854°。产物位于
`isaac_complete_assembly/complete_physics_driven_trajectory.mp4`、
`isaac_complete_assembly/complete_physics_video_report.json` 和
`isaac_complete_assembly/complete_physics_driven_scene.usda`。Object-4 交付目录中的旧
FixedJoint、预载和接触补偿实验已经清理，只保留最终结果、质量证据与可复现输入。

## 代码结构

```text
pose_solver/
├── cli.py          # 唯一 CLI、内存守护入口
├── config.py       # 单一源配置解析和数据契约
├── resolved.py     # 内部阶段配置生成
├── pipeline.py     # frames/mask/depth/pose 编排
└── artifacts.py    # 输出布局和 manifest

scripts/
├── run_mask_pipeline.py
├── run_depth_pipeline.py
└── run_pose_pipeline.py   # 内部 adapter，不是生产入口

common/             # 配准、渲染、约束、质量和共享数据结构
tools/stages/       # 有界的阶段实现
tools/diagnostics/  # 有界、可审计的诊断工具
configs/            # 源配置与实验配置
tests/              # 单元和回归测试
```

新增数据集通常只增加一份 source config。只有现有配置无法表达新的观测模型或
约束时，才修改公共实现。

## 执行安全

运行重任务前阅读：

- [执行与资源安全规范](docs/execution_safety.md)
- [2026-08-20 mesh split OOM 事故报告](docs/incidents/2026-08-20-mesh-split-memory-exhaustion.md)

强制规则：

- 生产任务通过 `python -m pose_solver run` 启动；
- memory guard 不得关闭，默认上限为 32 GiB 进程组 RSS，并保留至少
  128 GiB host available memory；
- 禁止对 reconstruction mesh 调用 `trimesh.Trimesh.split()`；
- 不因任务暂时无输出而启动第二个相同重任务；
- guard 返回 125 或 137 时先保存证据、确认进程组退出，再调查算法；
- 不清 swap、不重启共享服务、不终止其他用户进程。

## 开发验证

编排或配置改动至少运行：

```bash
.venv/bin/python -m unittest tests.test_unified_pipeline
.venv/bin/python -m unittest tests.test_pose_pipeline_core
.venv/bin/python -m pose_solver run \
  --config configs/pipeline.example.json \
  --stage preflight
```

涉及 pose 算法时，还必须用真实多视角数据检查稳定 anchor、运动段、最差视角、
endpoint 旋转分支和完整渲染视频，不能只验证 JSON 是否生成。
