# FoundationPose Hybrid 初始化 A/B 报告

## 结论

已实现并实跑“pose_solver 多视角 translation/scale + FoundationPose 全局 SO(3)
hypotheses/refiner + 六视角统一评分 + 有界 GICP + 全轨迹门控”。FoundationPose 不再
被当作最终单视角裁决器，而只负责提出旋转候选。

当前 v6 仍是最终接受结果。v7 的局部关键帧搜索找到一个 lid frame 100 候选，但将
修正插值到完整轨迹后，六视角总体指标和运动门没有同时改善，因此最终 selector 对
body、inner_pot、lid 均保留 v6 baseline。

## 实现

候选生成阶段固定现有 `T_world_from_part` translation 和 mesh scale。每个关键帧选择
两个 mask 面积大、未截断的来源视角；从 FoundationPose 的 icosphere/in-plane 旋转网格
中保留 48 个均匀 SO(3) hypotheses，并运行一次 PoseRefinePredictor。由于本地
ScorePredictor checkpoint 缺失，所有候选统一转换到 world frame 后交由 pose_solver
评分。

默认关键帧：

```text
body: 20
lid: 50, 84, 88, 100, 108
```

每帧共 97 个候选：baseline 1 个，加两个来源视角各 48 个 FoundationPose 候选。

评分与验收包括：

- 六视角 visible silhouette IoU；
- 六视角 silhouette contour distance；
- mesh texture 与观测 RGB edge；
- quality-cloud fitness@8mm 和 median NN；
- top-3 候选有界多尺度 GICP；
- translation、rotation branch 和完整轨迹速度门；
- 最终 111 帧多视角指标。

## 实跑结果

### Body frame 20

FoundationPose 最优候选约为 169.5° 对称翻转。二维目标有所改善，但点云几何门未
通过，因此拒绝。小规模 smoke test 也出现约 176.3° 的同类分支，说明该现象稳定，
不是候选数不足导致。

### Lid

- frame 50、84、88、108：baseline 最优；
- frame 100：找到来自 view `2-5` 的 22.24° 候选；translation 保持不变；关键帧联合
  分数提高 0.02153，并通过局部视觉、几何和 45° branch gate。

加入完整轨迹速度约束后，v7 candidate 相对 v6：

| Part | IoU 变化 | Chamfer 变化 |
|---|---:|---:|
| body | +0.00004 | -0.0020 px |
| inner_pot | -0.00116 | +0.0149 px |
| lid | -0.00096 | +0.0349 px |

lid 原始插值最大逐帧旋转约 4.02°，速度门将其限制为 3.00°/帧；最大平移步长
27.74 mm，低于 40 mm/帧门限。点云指标虽有改善（fitness@8mm 从 0.5354 升至
0.5681，median NN 从 7.28 mm 降至 4.29 mm），但 lid 全轨迹 IoU 和 contour
chamfer 同时退化，因此视觉硬门拒绝该分支，正式结果不晋升。

最终 `accepted_final` 的 111 帧 body、inner_pot、lid 位姿矩阵均已与 v6 baseline
逐项核对，完全一致。

## 输出

- FoundationPose 原始候选：`outputs_v7_foundationpose_hybrid/candidates.json`
- 关键帧与 GICP 诊断：`evaluated/diagnostics/foundationpose_hybrid.json`
- v7 候选轨迹：`evaluated/pose/trajectory.json`
- 111 帧六视角指标：`evaluated/diagnostics/multiview_metrics.json`
- 最终安全轨迹：`accepted_final/pose/trajectory.json`
- 最终决策：`accepted_final/diagnostics/multimetric_selection.json`

最终安全轨迹保留 v6 的三个 part。

## 复跑

```bash
.venv/bin/python scripts/run_foundationpose_hybrid.py
```

使用 `--force` 可重新计算全部阶段。GPU 候选生成会自动通过 `conda run -n spot`
调用已有 FoundationPose 环境。
