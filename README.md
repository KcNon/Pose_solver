# pose_solver

`pose_solver` 把同步多视角视频转换成每个刚性 part 的逐帧 6D pose，并用
mesh 回渲染做质量检查。Isaac Sim 消费最终 pose，但保持为独立模块；pose
没有通过多视角检查时，不应进入仿真。

```text
同步多视角帧
  -> Qwen + SAM3 mask
  -> 固定 rig 深度与分 part 点云
  -> 稳定 anchor、尺度、逐帧 pose
  -> 多视角 silhouette/depth render refinement
  -> trajectory_final.json + overlay/review
```

## 唯一用户入口

Mask、depth 和 pose 只通过一个 source config 和一个命令启动：

```bash
.venv/bin/python -m pose_solver run --config configs/<dataset>.json
```

常用操作：

```bash
# 只检查输入契约，不运行模型
.venv/bin/python -m pose_solver run \
  --config configs/<dataset>.json --stage preflight

# 查看将运行的命令和 resolved config
.venv/bin/python -m pose_solver run \
  --config configs/<dataset>.json --dry-run

# 单独重跑 pose；mask 和点云必须已存在或配置为 reuse
.venv/bin/python -m pose_solver run \
  --config configs/<dataset>.json --stage pose

# 只读查看已有轨迹或回归基线
.venv/bin/python -m pose_solver inspect --config configs/<dataset>.json
```

`scripts/run_mask_pipeline.py`、`run_depth_pipeline.py` 和
`run_pose_pipeline.py` 是内部 stage adapter，不再是用户接口。
旧的 `run_automated_workflow.py` 和历史多配置入口已经删除。

## 单一配置

[pipeline.example.json](configs/pipeline.example.json) 是配置模板。人工只维护：

- 数据集、视频或已同步帧目录、视角和帧范围；
- part 名称、mask ID、语义 prompt、mesh 和出现时间提示；
- 输出根目录与最多两张 GPU；
- 外部模型环境；
- 每个阶段是 `run` 还是复用已有 artifact。

`appearance_hint` 可以是整数或 `"auto"`。它只限制 mask 搜索起点，不能作为
pose anchor。参考帧、静止/运动区间、尺度和 pose 均由运行时证据决定。

示意结构：

```json
{
  "schema_version": 1,
  "dataset": "my_object",
  "input": {
    "frames_dir": "/data/my_object/frames",
    "views": ["cam0", "cam1", "cam2", "cam3"],
    "frame_range": [0, 300],
    "videos": {
      "cam0": "/data/my_object/videos/cam0.mp4",
      "cam1": "/data/my_object/videos/cam1.mp4",
      "cam2": "/data/my_object/videos/cam2.mp4",
      "cam3": "/data/my_object/videos/cam3.mp4"
    },
    "sample_fps": "source",
    "sync_offsets_s": {}
  },
  "parts": {
    "main": {
      "id": 1,
      "mesh": "/data/my_object/meshes/main.glb",
      "prompts": ["main object body"],
      "appearance_hint": "auto",
      "reference": true
    }
  },
  "output": {"root": "/experiments/my_object"},
  "runtime": {"devices": [6, 7]},
  "models": {
    "qwen_python": "/path/to/qwen/python",
    "qwen_model": "/path/to/qwen/model",
    "sam_python": "/path/to/sam/python",
    "sam_checkpoint": "/path/to/sam.pt",
    "da3_python": "/path/to/da3/python"
  },
  "mask": {"mode": "run"},
  "depth": {"mode": "run"},
  "pose": {"mode": "run"}
}
```

`overrides` 只用于算法实验。frames、views、parts、ID、输出路由等共享契约会被
source config 强制锁定，不能被 override 或旧配置悄悄改掉。
`compatibility_config` 只用于把已经验证过的旧实验迁移到新入口；新数据不应依赖它。

## 固定产物目录

每个 source config 只产生一棵稳定目录：

```text
<output.root>/
  runtime/
    frames.resolved.json
    mask.resolved.json
    depth.resolved.json
    pose.resolved.json
    pipeline_manifest.json
  mask_work/
  mask/masks/
  depth/parts_ply/
  pose/
    automation/
    diagnostics/
    pose/trajectory_final.json
    render/
    review/
```

resolved JSON 是可审计产物，不是第二套人工配置。manifest 记录 source hash、GPU、
阶段状态和实际 artifact 路径。失败会明确写为 `failed`，不会用旧文件冒充成功。
如果 `input.videos` 存在，`all` 会先生成同步帧；如果帧目录已完整，则自动复用。

## Pose 质量原则

- 世界坐标由固定相机 rig 和桌面平面定义，参考帧 pose 不能直接设单位矩阵。
- 每个 part 从首个可靠稳定窗口建立 anchor，不从刚出现或手持帧初始化。
- 每帧相对稳定 anchor 估计，避免连续 ICP 累计漂移。
- “可见但点云不可靠”和“真正出画”必须分开；空 mask 才能触发出画逻辑。
- 点云、mask silhouette、纹理和深度联合选方向与尺度。
- mesh scale 每个 part 固定一次，逐帧 SE(3) 不得吸收尺度误差。
- 接触约束只在接触区间启用；非接触 part 不能互相穿透。
- 最终必须查看同步多视角 overlay，单一主视角只能用于展示，不能替代八视角优化。

## Object-9 回归

[object9_pose.json](configs/object9_pose.json) 已把现有 Object-9 v7 pose 路径接入新接口。
Mask 和点云复用已验证 artifact，pose 保留旧算法参数作为迁移 preset，输出到新的隔离目录，
不会覆盖 v7 基线。只读检查：

```bash
.venv/bin/python -m pose_solver inspect --config configs/object9_pose.json
.venv/bin/python -m pose_solver run \
  --config configs/object9_pose.json --stage pose --dry-run
```

该配置固定使用 GPU 6、7，满足最多两张卡的资源约束。

## 代码职责

```text
pose_solver/   唯一公共 CLI、source config、artifact contract、编排
common/        数据集无关算法
scripts/       内部 stage adapter 与独立 Isaac 入口
tools/stages/  细粒度内部计算阶段
tools/diagnostics/  只读诊断与可视化
tests/         无 GPU 契约和算法测试
configs/       单一 source config
experiments/   生成产物，不反向成为代码默认值
```

算法代码不得引用 Object 编号或固定 part 名称。Isaac 相关入口和实现不在本次接口
收口范围内，仍独立运行。

### 当前保留的 mask/pose 调用链

```text
python -m pose_solver
  -> frames: extract_synchronized_video_frames
  -> mask: detect_mask_seeds -> track_part_masks
           -> validate_multiview_seeds -> compose/quality
  -> depth: fixed-rig DA3 -> depth gauge -> quality point cloud
  -> pose: detect_part_states -> solve_multiview_pose
           -> render refinement -> assembly/non-penetration constraints
           -> static stabilization -> multiview review/render
```

旧 workflow、多份 data_1 配置、旧 high-FPS 分支、palette mask compatibility、
trajectory branch fusion、未被调用的 pose bridge/prior 和一次性尺度/方向诊断已经
删除。点云可视化、anchor 验证和八视角 overlay 被保留，因为它们仍是定位 pose
误差所必需的正式质量工具。

## 开发验证

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
.venv/bin/python -m compileall -q common pose_solver scripts tools tests
git diff --check
```
