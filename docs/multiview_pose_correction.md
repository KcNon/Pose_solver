# 无 GT 的多视角 Pose 修正

这套流程用于在没有 pose GT 时修正并审计已有粗 pose。它能拒绝与观测矛盾的
pose、比较候选轨迹并提高组装前的相对位姿质量，但不能把自一致性等同于绝对
GT。

## 数据流

```text
ICP / 运动模型粗 pose
          │
          ▼
同步多视角 render-and-compare
          │
          ├── optimize views：搜索局部 SE(3) 修正
          └── holdout views：只验收，绝不参与搜索
                         │
                         ▼
         时序步长、最差视角和独立视角门控
                         │
               通过 ────┴──── 拒绝
                │                 │
                ▼                 ▼
            写回候选 pose       保留原 pose
```

每个候选使用 silhouette IoU、轮廓 Chamfer 和可用时的可见深度残差。标签图中
其他刚性部件和手部是 `unknown/occluded`，不是目标部件的背景。可通过
`known_occluder_dilation_pixels` 扩张遮挡标签，覆盖手工 mask 的边缘误差。

## 严格验收配置

四个及以上同步相机的自动配置默认启用独立留出门控。典型设置为：

```json
{
  "render_loss_refinement": {
    "require_independent_holdout": true,
    "minimum_optimize_views": 3,
    "minimum_holdout_views": 1,
    "auto_holdout_policy": "rotating",
    "known_part_occlusion_aware": true,
    "known_occluder_labels": [1, 2, 3],
    "known_occluder_dilation_pixels": 1,
    "maximum_holdout_degradation": 0.015,
    "minimum_holdout_iou": 0.20,
    "minimum_per_view_iou": 0.05
  }
}
```

`rotating` 会按帧从 optimize 集移出一个相机作为 holdout；同一帧中两个集合
始终不相交。显式配置且可见的 `holdout_views` 优先使用。严格模式下，如果
可用相机不足以同时满足 optimize 和 holdout 数量，当前帧直接拒绝修正。

报告 `diagnostics/render_loss_refinement.json` 为每个已评估帧记录：

- 实际 optimize/holdout 相机及自动留出策略；
- 修正前后的 optimize 与 holdout IoU、轮廓和 loss；
- 最差视角门控、时序边界门控和拒绝原因；
- 最终写回的平移/旋转增量。

## 能力边界

- 对称物体的不可观测轴向旋转不能由 silhouette/depth 唯一恢复；
- 所有相机都被遮挡时应保持原 pose 或依靠时序桥接；
- 错误 mesh 尺度、camera rig 或同步偏差会形成跨视角系统误差；
- 吸管等可变形结构不应和喷嘴共享刚性 SE(3) 优化。

因此最终判定应分级为视觉一致、时序一致、装配就绪；Isaac 物理测试只在前三项
通过后作为冻结 pose 的下游验证，不能替代多视角证据。
