# 通用装配物理验证平台

## 定位

平台验证的是“冻结的视觉 Pose 在声明的碰撞几何和物理假设下是否可行”，不是 Pose
GT。验证模式不允许轴线自动对齐、目标平移补偿、控制器、预载、FixedJoint 或轨迹
改写。物理 Pose 精修是独立的可选实验，不能替代冻结 Pose 验证。

通用的含义是：同一种装配接口换物体时只增加配置，不增加对象专用 Python。平台不会
从任意重建 mesh 猜测螺纹、卡扣或功能接触面；这些信息必须来自测量、CAD、带置信度的
mesh 拟合或显式工程估计，并在报告中保留来源。

## 数据契约

仿真导出配置可声明 `assembly_interface`：

```json
{
  "type": "cylindrical_insertion",
  "reference_part": "bottle_body",
  "moving_part": "sprayer_pump",
  "reference_axis_part": [0, 1, 0],
  "moving_axis_part": [0, 0, 1],
  "reference_outer_radius_m": 0.016,
  "moving_inner_radius_m": 0.0165,
  "parameter_source": "measured",
  "confidence": "high"
}
```

`parameter_source` 只能是 `measured`、`cad`、`mesh_fit` 或
`engineering_estimate`。只有 measured/CAD 且 high confidence 才允许报告
`metric_physical_accuracy_claim_allowed=true`。

动态空心连接件使用 `cylindrical_sleeve`。它支持任意 part-frame 轴和原点，并导出为
多个低面数、闭合、凸的环形楔块；因此动态凸分解不会把中心孔填死。视觉 mesh 始终与
碰撞代理分开。

## 标准流程

1. 导出 visual/collision 资产和 manifest；
2. 运行 manifest preflight；
3. preflight 通过后，以 `--validate-only` 运行 Isaac；
4. 第一项试验从未经修改的视觉终态直接释放；
5. 后续以固定顺序测试 ±1/2/5 mm 横向偏移和 ±1/3/5° 倾斜；
6. 报告分别给出 baseline feasibility 与 perturbation recovery rate。

所有 mesh、Isaac 和视频命令必须由 `tools/diagnostics/run_with_memory_guard.py`
包裹；不得对 reconstruction mesh 调用 `Trimesh.split()`。

## 命令

资产导出：

```bash
.venv/bin/python -u tools/diagnostics/run_with_memory_guard.py \
  --log OUTPUT/runtime/export_guard.jsonl \
  --minimum-available-gib 128 --maximum-process-rss-gib 32 -- \
  .venv/bin/python -u scripts/export_simulation_assets.py \
  --config CONFIG.json --project-root . --output-dir OUTPUT/assets
```

预检：

```bash
.venv/bin/python -u tools/diagnostics/run_with_memory_guard.py \
  --log OUTPUT/runtime/preflight_guard.jsonl \
  --minimum-available-gib 128 --maximum-process-rss-gib 32 -- \
  .venv/bin/python -u tools/diagnostics/validate_simulation_manifest.py \
  --manifest OUTPUT/assets/manifest.json \
  --output OUTPUT/preflight_report.json
```

冻结 Pose 物理验证：

```bash
.venv/bin/python -u tools/diagnostics/run_with_memory_guard.py \
  --log OUTPUT/runtime/isaac_guard.jsonl --cuda-visible-devices 1 \
  --minimum-available-gib 128 --maximum-process-rss-gib 32 \
  --minimum-gpu-free-mib 8000 -- \
  /data_ft_9_10/ziang/code/IsaacSim/_build/linux-x86_64/release/python.sh \
  scripts/run_isaac_insertion.py --asset-root OUTPUT/assets \
  --runtime-output-dir OUTPUT/isaac --validate-only --no-capture --device cpu
```

## 当前结果

Object-4 preflight 允许启动，但接口尺寸和质量均不是测量值，因此不允许声称 metric
physical accuracy。使用空心套筒后，冻结视觉 Pose baseline 通过：横向误差约为零，
倾斜约为零，沿装配轴下降 6.00 mm 后落到代理承托面并保持 PhysX 接触。完整 25 项
试验中 baseline 为 1/1，24 个扰动为 6/24（25%）。各组通过率为：1/2/5 mm 平移
分别 25%/50%/25%，1/3/5° 倾斜分别 0%/25%/25%。恢复域不单调且明显不对称，
因此当前结果只支持“冻结 Pose 在低置信度代理下被动可行”，不支持鲁棒装配或 metric
精度声明。该结果没有使用吸管弹力、预载、控制器、轴向补偿或 FixedJoint。

运行时同时将 `contact_offset_m/rest_offset_m` 写入所有 PhysX collider，并在验证刚体
上设置 `contact threshold=0`、`sleepThreshold=0`。后者只关闭会隐藏静止接触流形的
休眠优化，不会施加力或修正 Pose。最终报告位于
`experiments/objects_0827/Object-4/simulation_ready/simulation_validation/isaac_final_full/qa/assembly_validation_report.json`。

Object-9 使用同一自动资产生成与 preflight 代码，无对象专用实现。preflight 阻止
Isaac 启动，因为没有 `assembly_interface`；已有 connector 证据同时报告轴线误差、
径向偏移和制造/碰撞元数据缺失。这是 fail-closed 行为，不能用猜测参数绕过。

当前结论是：平台能够区分 Object-4 的 baseline 可行性和较差的扰动鲁棒性，并阻止
Object-9 在接口证据不足时产生误导性 Demo；它尚未证明 Isaac 能提高真实 Pose 精度。
