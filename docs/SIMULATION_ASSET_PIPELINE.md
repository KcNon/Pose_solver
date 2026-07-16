# Pose Solver 到 URDF / Isaac Sim 的资产与插入验证流程

## 目标

该流程使用 pose_solver 已有的三部件 mesh、真实尺度和相对位姿，生成：

- 可移植的独立 URDF：`body.urdf`、`inner_pot.urdf`、`lid.urdf`；
- 仅用于装配展示的 `rice_cooker_display.urdf`；
- 保留纹理的 visual mesh 和独立 collision mesh；
- Isaac Sim 导入后可缓存的 USD；
- observed trajectory 运动学回放报告；
- inner_pot 重力落座扰动实验报告。

展示 URDF 和插入实验必须分开。展示 URDF 使用 fixed joint 固定三个 part，只验证坐标和资产是否能正确加载；插入实验分别加载 body 和 inner_pot，body 为固定碰撞体，inner_pot 为动态刚体。

## 坐标约定

轨迹文件的约定为：

```text
canonical vertex = scale * (raw mesh vertex - raw mesh origin)
world vertex     = T_world_from_part * canonical vertex
```

导出的 OBJ 已经烘焙尺度和 raw mesh origin，因此 URDF 的 visual/collision origin 都是零，URDF link frame 就是 pose_solver 的 canonical part frame。所有距离单位为米。

`manifest.json` 保存了每个 part 的：

- 输入文件及 SHA-256；
- 固定尺度；
- `T_part_from_raw_mesh`；
- canonical bounds/extents；
- 质量及其来源；
- body 坐标系下的装配变换；
- 状态区间和碰撞策略。

当前质量、惯量和摩擦系数是配置假设。它们足以进行几何可装配性和初步落座实验，但不能作为真实动力学参数。

## 生成资产

在 pose_solver 环境中运行：

```bash
cd "$WORKSPACE_DIR"
.venv/bin/python scripts/export_simulation_assets.py \
  --config configs/simulation_assets.json
```

默认输出目录：

```text
experiments/rice_cooker_simulation_assets/
├── manifest.json
├── meshes/
│   ├── visual/<part>/<part>.obj + MTL + PNG
│   └── collision/<part>.obj
├── urdf/
│   ├── body.urdf
│   ├── inner_pot.urdf
│   ├── lid.urdf
│   └── rice_cooker_display.urdf
├── usd/
├── qa/
│   ├── geometry_report.json
│   └── insertion_trajectory_body.json
└── stages/
```

导出脚本是配置驱动的。批量处理其他物体时，替换配置中的 trajectory、meshes、reference part、assembly target 和 simulation role 即可，不需要修改脚本。

离线渲染装配预览和三个 solved part frame：

```bash
.venv/bin/python scripts/render_simulation_asset_preview.py \
  --asset-root experiments/rice_cooker_simulation_assets
```

坐标轴颜色为 X 红、Y 绿、Z 蓝；它们与 URDF link frame 完全相同。

## 碰撞策略

body 的碰撞不能使用整体 convex hull。整体凸包会封闭锅口并填满内腔，产生“inner_pot 永远进不去”的假失败。

- body：移除 RigidBodyAPI，作为静态 triangle-mesh collider，approximation=`none`；
- inner_pot：动态刚体，approximation=`convexDecomposition`；
- 导出 visual mesh：保留原始纹理，与碰撞计算完全分离；
- headless demo 截图：直接 author 纯色材质，避免 URDF importer 的实例化材质在
  RayTracedLighting 下不可见；
- lid：当前导出但不参与 inner_pot 插入实验。

三个原始 GLB 都不是水密网格，因此水密性只记录为 QA 信息，不作为导出失败条件。
落座过程由 PhysX 推进；当前报告按最终 pose 和静止条件验收，不直接输出接触力或穿透量。

## 运行 Isaac Sim

设置本机 Isaac 路径与工作区：

```bash
export ISAAC_SIM_DIR=/path/to/isaacsim/_build/linux-x86_64/release
export WORKSPACE_DIR=/path/to/pose_solver
cd "$WORKSPACE_DIR"

"$ISAAC_SIM_DIR/python.sh" scripts/run_isaac_insertion.py \
  --asset-root experiments/rice_cooker_simulation_assets
```

如果 Isaac 构建用户与资产导出用户不同，将运行时缓存和报告写到构建用户可写的新目录：

```bash
"$ISAAC_SIM_DIR/python.sh" scripts/run_isaac_insertion.py \
  --asset-root experiments/rice_cooker_simulation_assets \
  --runtime-output-dir experiments/rice_cooker_isaac_runtime
```

启动前可以先运行不加载 Kit 的 preflight：

```bash
.venv/bin/python scripts/check_isaac_runtime.py \
  --isaac-sim-dir "$ISAAC_SIM_DIR" \
  --output experiments/rice_cooker_simulation_assets/qa/isaac_runtime_preflight.json
```

首次运行会：

1. 将四个 URDF 导入 USD；
2. 写入 `usd/import_cache.json`，后续输入哈希不变时直接复用；
3. 生成 `usd/insertion_scene.usda`；
4. 将 body 的 +Y 轴对齐 Isaac 世界 +Z，使重力方向正确；
5. 回放 frame 0–41 的观测相对轨迹；
6. 运行 9 组重力落座扰动实验；
7. 输出 `qa/isaac_insertion_report.json` 和 `qa/isaac_final.png`。

常用选项：

```bash
# 只重新导入 USD
"$ISAAC_SIM_DIR/python.sh" scripts/run_isaac_insertion.py --force-import --skip-replay --skip-drop

# 不截图，适合纯物理服务器
"$ISAAC_SIM_DIR/python.sh" scripts/run_isaac_insertion.py --no-capture

# 使用 GPU physics
"$ISAAC_SIM_DIR/python.sh" scripts/run_isaac_insertion.py --device cuda:0
```

## 生成 Isaac 可视化与物理审计视频

在 `run_isaac_insertion.py` 已生成 `usd/import_cache.json` 后，可一次启动 Isaac
生成完整轨迹回放、逐帧碰撞审计和自由释放实验：

```bash
"$ISAAC_SIM_DIR/python.sh" scripts/run_isaac_visualization_suite.py \
  --asset-root experiments/rice_cooker_simulation_assets \
  --runtime-root experiments/rice_cooker_isaac_runtime \
  --output-dir experiments/rice_cooker_isaac_visualizations \
  --fps 15 \
  --physics-seconds 1.5 \
  --trial-limit 3 \
  --rt-subframes 1
```

输出包括：

```text
experiments/rice_cooker_isaac_visualizations/
├── 01_pose_replay_textured.mp4
├── 02_pose_replay_collision_debug.mp4
├── 03_inner_pot_physics_trials.mp4
├── 04_full_assembly_release.mp4
├── 05_physics_driven_trajectory.mp4
├── isaac_visualization_scene.usda
├── preview_contact_sheet.jpg
├── physics_driven_final.jpg
├── physics_driven_report.json
└── visualization_report.json
```

- `01` 是 pose_solver 输出的 111 帧三部件运动学回放，不代表物理预测；
- `02` 将重力关闭的动态碰撞代理重置到每帧 solved pose，并单步查询接触；
- `03` 仅把 solved pose 作为 inner_pot 的目标/初始条件，随后完全交给 PhysX；
- `04` 从最终装配 pose 同时释放 inner_pot 和 lid，观察是否保持装配。
- `05` 使用有限的 `PhysxForceAPI` 世界系力/力矩追踪 pose_solver 目标轨迹。实际
  inner_pot/lid 只在 frame 0 初始化一次，此后不再覆盖 pose；青色/洋红色半透明
  Mesh 是目标位姿，纹理 Mesh 与坐标轴是 PhysX 实际位姿。

轨迹审计阶段关闭地面碰撞，避免把物体与地面的接触误记为部件间穿透。
自由释放阶段重新启用地面，并限制最大去穿透速度，使失败过程保持可观察。

## 结果解释

每组 drop trial 的 success 同时要求：

- 最终平移误差小于配置阈值；
- 最终旋转误差小于配置阈值；
- 最终线速度和角速度足够小；
- 仿真持续时间至少 3 秒。

报告记录最终位姿、平移/旋转误差、速度峰值和最低高度。Isaac Sim 6 当前不通过
tensor contact query 采集接触力；几何落座结论由最终 pose 和静止条件判断。失败时应依次区分：

1. 坐标/尺度转换错误；
2. body collision 错误地封闭内腔；
3. convex decomposition 过度膨胀 inner_pot；
4. 当前求解的装配位姿或旋转有误；
5. reconstructed mesh 本身不满足真实装配间隙；
6. 摩擦、质量等假设导致物理落座失败。

仿真成功只能说明“当前重建 mesh、尺度、相对位姿和碰撞近似彼此相容”，不能替代真实 CAD 或制造公差验证。

## 当前验证结果

当前 Isaac Sim 6 流程已经能够：

- 成功导入四个 URDF 并缓存 USD；
- 正确显示 body 和位于其中的 inner_pot；
- 输出有效的 1280×720 headless 截图；
- 完成 9 组 PhysX 重力落座 trial。

结果位于：

```text
experiments/rice_cooker_isaac_runtime/
├── qa/isaac_final.png
├── qa/isaac_insertion_report.json
└── usd/insertion_scene.usda
```

当前 9 组 trial 的严格成功数为 `0/9`。这说明资产、坐标转换和仿真流程已跑通，
但当前 reconstructed mesh、碰撞近似或 solved assembly pose 尚未通过物理落座标准，
不能把“截图中看起来放入”解释成已经验证真实几何可装配。

最新可视化审计的 111 帧中，前 32 帧为 `clear`，从 frame 32 起的 79 帧为
`penetrating`。frame 32 首次检测到 inner_pot 最大约 10.0 mm 穿透；最终 frame 110
检测到最大约 26.5 mm 穿透。aligned 自由释放实验最终静止在目标位姿约 51.9 mm
之外；三部件整体释放后 inner_pot 和 lid 分别偏离约 41.7 mm 和 133.1 mm。
因此当前结论是“视觉装配成立，但现有碰撞代理和 solved assembly pose 不具备物理相容性”。

physics-driven 轨迹进一步确认了这一点：inner_pot 在 frame 33 首次被 body 阻挡。
控制器继续以不超过 12 N 的力跟踪目标，但实际 inner_pot 没有穿过 body。最终
inner_pot 平移/旋转误差约为 74.2 mm / 33.7°，lid 约为 88.9 mm / 36.4°。
这个实验与 `01/02` 的逐帧 pose 回放不同，碰撞约束对实际运动具有决定权。

## 验证

纯 Python 导出测试：

```bash
.venv/bin/python -m unittest tests.test_simulation_assets -v
```

测试覆盖 raw→canonical→world 变换一致性、鲁棒装配位姿、坐标轴对齐、URDF XML 和所有 mesh 引用路径。
