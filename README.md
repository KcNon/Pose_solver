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

## Connector 与仿真

可选 `connectors` 描述插入、旋转或螺纹等装配关系。它用于质量门控和约束，
不能替代可靠的视觉 pose。进入 Isaac Sim 前必须确认：

- pose、尺度和 camera rig 已通过多视角检查；
- connector 轴、原点、间隙和碰撞 proxy 有明确几何证据；
- 物理碰撞使用低面数 proxy，不直接拆分高密度 reconstruction mesh。

Isaac 导出是 pose 通过质量门后的独立步骤，不属于默认 pipeline。

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
