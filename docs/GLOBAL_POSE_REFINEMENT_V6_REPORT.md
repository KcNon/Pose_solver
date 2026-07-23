# V6 全局 6D Pose 精修报告

## 结论

本轮六项需求均已落地并完成 111 帧实跑。最终三部分候选均通过“普通位姿变化代理、ADD-S、六视角轮廓、轨迹分支”硬门控。DA3 点云指标保留为诊断项；按当前约定，在深度质量问题解决前不作为 inner_pot/lid 的一票否决条件。

最终轨迹：`outputs_v6_global_pose/accepted_final/pose/trajectory.json`。

## 流程

1. Body 全局 basin：围绕当前 local-Y 按 15° 枚举 24 个候选，完整覆盖 360°。每个候选使用六视角聚合 silhouette 与多帧质量点云打分。
2. Body 局部精修：从全局最优候选调用原有 12/6/3 mm 多尺度 GICP；若局部结果离开视觉 basin，则回退到离散候选。
3. Inner pot 装配先验：从已接受轨迹提取恒定 `T_body_from_inner_pot`，仅对 assembled 段优化；GICP 修正限制在 12 mm/12° 内。
4. Lid tilt：XYZ 局部旋转全部参与六视角 silhouette + RGB edge 搜索；不可观测轴归零，可观测 tilt 保留。关键帧修正经过平滑、插值以及 3°/帧、40 mm/帧速度门控。
5. 最终验收：body 与 assembled inner_pot 作为耦合分支验收，lid 独立验收。硬门控包括普通旋转/平移变化、ADD-S、六视角 IoU/轮廓 Chamfer 和逐帧运动上限。

## 实跑结果

Body 的 24 个 yaw proposal 中，当前 0° 邻域是全局最优 basin，不存在更优的 180° 翻转。GICP 在该 basin 内又修正了约 9.00° 和 16.13 mm：

| 指标 | baseline | refined |
|---|---:|---:|
| 聚合六视角 IoU | 0.7589 | 0.8386 |
| 聚合轮廓距离 / px | 6.44 | 3.88 |
| 融合点云 fitness@8mm | 0.785 | 0.908 |
| 融合点云 median NN / mm | 3.80 | 2.86 |

Inner pot 的无约束 GICP 倾向于移动 24.34 mm，装配约束将其截断至 12 mm；有界结果在融合 assembled 点云上的 fitness@8mm 从 0.375 提升至 0.801。

Lid 的 tilt 在运动中段可观测，第 88/92/96 帧保留的旋转修正分别约为 11.5°、10.0°、14.0°。最终最大逐帧旋转 2.33°、最大逐帧平移 27.74 mm，均未触发硬截断。

完整 111 帧、六视角平均结果：

| Part | IoU baseline | IoU V6 | Chamfer baseline / px | Chamfer V6 / px |
|---|---:|---:|---:|---:|
| body | 0.5197 | 0.6085 | 6.895 | 5.535 |
| inner_pot | 0.5511 | 0.5805 | 5.799 | 5.436 |
| lid | 0.6932 | 0.7365 | 4.038 | 3.552 |

## 输出与复跑

- 全局 yaw 诊断：`body_global/diagnostics/body_global_yaw.json`
- 装配先验诊断：`body_inner/diagnostics/inner_assembly_prior.json`
- Lid tilt 与速度门：`lid_tilt/diagnostics/lid_se3_refinement.json`
- 最终多指标决策：`accepted_final/diagnostics/multimetric_selection.json`
- 一键复跑：`python scripts/refine_multiview_pose.py`

最终视频位于 `accepted_final/render/2-3/`：`overlay.mp4`、`mesh_only.mp4`、`mesh_textured.mp4` 和 `mesh_axes.mp4`。坐标轴直接使用各 part 的 `T_world_from_part`，不是人为绘制的示意方向。
