# Pose Solver → Isaac Sim

正式仿真流程包含资产导出、物理验证和可选的完整视频渲染：

```text
trajectory.json + ReconViaGen GLB
              │
              ▼
export_simulation_assets.py        pose_solver .venv
              │
      canonical OBJ + URDF
              │
              ▼
run_isaac_insertion.py             Isaac Sim python.sh
              │
       USD + replay/drop report
              │
              ▼
run_isaac_video.py                 Isaac Sim python.sh
              │
     complete multi-view MP4
              │
              ▼
run_isaac_physics_video.py         Isaac Sim python.sh
              │
 force-driven MP4 + contact report
```

`scripts/` 只保留轻量 CLI；资产算法和导出实现位于
`common/simulation_assets.py`、`common/simulation_export.py`，Isaac 实现位于
`common/isaac_runtime.py`，完整 pose 回放位于 `common/isaac_video.py`，完整
物理驱动视频位于 `common/isaac_physics_video.py`。

## 1. 配置

唯一配置是 `configs/simulation_assets.json`，包含：

- pose trajectory 和三个 ReconViaGen mesh；
- reference/container/inserted part；
- 装配静止帧区间；
- 质量、摩擦、重力落座扰动和成功阈值；
- `data/1` 默认输出目录 `experiments/data_1/simulation_assets`。

trajectory 的 canonical 约定为：

```text
X_part  = scale * (X_raw - raw_mesh_origin)
X_world = T_world_from_part * X_part
```

导出的 OBJ 已烘焙 scale 和 raw mesh origin，因此 URDF mesh origin 为零。

## 2. 导出资产

在 pose_solver 环境运行：

```bash
cd /data_ft_9_10/wentai/projects/pose_solver
.venv/bin/python scripts/export_simulation_assets.py \
  --config configs/simulation_assets.json
```

主要输出：

```text
experiments/data_1/simulation_assets/
├── manifest.json
├── meshes/
│   ├── visual/<part>/
│   └── collision/<part>.obj
├── urdf/
│   ├── body.urdf
│   ├── inner_pot.urdf
│   ├── lid.urdf
│   └── rice_cooker_display.urdf
└── qa/
    ├── geometry_report.json
    └── insertion_trajectory_body.json
```

`rice_cooker_display.urdf` 只用于固定装配展示；物理试验加载独立 body 和 inner pot。
body 使用静态 triangle mesh 以保留内腔，动态 inner pot 使用 convex decomposition。

## 3. Isaac 验证

必须使用 Isaac Sim 自带的 `python.sh`：

```bash
ISAAC_SIM_DIR=/data_ft_9_10/wentai/projects/isaacsim/_build/linux-x86_64/release

"$ISAAC_SIM_DIR/python.sh" scripts/run_isaac_insertion.py \
  --asset-root experiments/data_1/simulation_assets
```

如资产目录不可写，可分离 runtime 输出：

```bash
"$ISAAC_SIM_DIR/python.sh" scripts/run_isaac_insertion.py \
  --asset-root experiments/data_1/simulation_assets \
  --runtime-output-dir experiments/data_1/isaac_runtime
```

常用诊断参数：

```bash
# 仅验证 URDF→USD 导入
"$ISAAC_SIM_DIR/python.sh" scripts/run_isaac_insertion.py \
  --force-import --skip-replay --skip-drop --no-capture

# 只运行前一个落座扰动
"$ISAAC_SIM_DIR/python.sh" scripts/run_isaac_insertion.py \
  --skip-replay --trial-limit 1 --no-capture
```

运行会复用哈希一致的 USD cache，生成：

```text
<runtime-root>/
├── usd/import_cache.json
├── usd/insertion_scene.usda
├── qa/isaac_insertion_report.json
├── qa/isaac_final.png               # 启用 capture 时
└── stages/isaac.complete.json
```

报告同时检查最终平移、旋转、线速度和角速度。成功只说明“当前重建 mesh、尺度、
相对 pose 和碰撞近似彼此相容”，不等于真实制造公差或动力学参数已经验证。

## 4. 完整 Isaac 视频

先完成 Isaac 验证并生成 USD cache，再运行：

```bash
"$ISAAC_SIM_DIR/python.sh" scripts/run_isaac_video.py \
  --asset-root experiments/data_1/simulation_assets \
  --runtime-root experiments/data_1/isaac_runtime_proxy_v2 \
  --output-dir experiments/data_1/isaac_video_complete \
  --fps 5 --start-frame 0 --end-frame 245
```

该入口会渲染完整时间线和三个固定视角。轨迹开始前保持空场景；部件只在其状态变为
可观测后出现。视频是 pose 的 Isaac 回放，动态插入是否成功仍以 insertion report
为准。

如需原 `05_physics_driven_trajectory.mp4` 类型的完整视频，运行：

```bash
"$ISAAC_SIM_DIR/python.sh" scripts/run_isaac_physics_video.py \
  --asset-root experiments/data_1/simulation_assets \
  --runtime-root experiments/data_1/isaac_runtime_proxy_v2 \
  --output-dir experiments/data_1/isaac_video_physics_complete \
  --fps 5 --start-frame 0 --end-frame 245
```

其中带纹理 mesh 是实际 PhysX 刚体，青色和品红色半透明 mesh 是 pose solver
目标。部件在首次可观测时初始化一次；之后只施加有上限的力和力矩，不再逐帧覆盖
刚体 pose。碰撞始终参与实际轨迹，逐帧误差、接触数和穿透量写入
`complete_physics_video_report.json`。

## 5. 验证

普通 Python 环境：

```bash
.venv/bin/python -m unittest tests.test_simulation_assets -v
.venv/bin/python -m compileall -q common scripts tests
```

Isaac 环境的最小 smoke test 使用 `--skip-replay --skip-drop --no-capture`。这仍会真正
启动 Isaac、导入或复用 USD，并验证场景构建；普通 `.venv` 无法替代该检查。
