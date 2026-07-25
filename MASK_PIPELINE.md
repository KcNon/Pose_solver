# 可复用 Qwen + SAM3 多视角 Mask Pipeline

## 1. 数据流

```text
人工配置：part 列表、part 出现帧、种子帧、相机视角
                         │
同步多视角 RGB ── Qwen3-VL bbox（只跑种子帧）
                         │
                 SAM3 单 part 跟踪
                         │
        tracks/{part}/{frame}/{view}.png
                         │
       可选：depth + 标定相机的多视角几何先验
                         │
            前后遮挡顺序合成 + 质量检查
                         │
              masks/{frame}/{view}.png
```

Qwen、SAM 和最终合成相互解耦。每个部件先写独立的二值 track，修复一个
部件不会擦掉其他部件；最后才按 `occlusion_order` 合成 palette mask。

正式入口只有一个：

| 入口 | 职责 |
|---|---|
| `run_mask_pipeline.py` | 跨环境调度、可选多视角先验、最终合成 |

Qwen 与 SAM 的跨环境内部阶段位于 `tools/stages/masking/`，通常由 runner 调用。

## 2. 配置约定

部件数量、名字和标签 ID 都不再写死：

```json
{
  "frames_dir": "/path/to/frames",
  "work_root": "/path/to/mask_work",
  "output_root": "/path/to/final_masks",
  "views": ["cam0", "cam1", "cam2"],
  "parts": {
    "body": {
      "id": 2,
      "color": [52, 199, 89],
      "start_frame": 40,
      "prompts": ["rice cooker body"],
      "tracking": {
        "mode": "fixed-image",
        "seed_frames": {"default": 125, "cam2": 205}
      }
    },
    "inner_pot": {
      "id": 3,
      "color": [0, 122, 255],
      "start_frame": 65,
      "prompts": ["black removable inner pot"],
      "tracking": {
        "mode": "video",
        "seed_frame": 125,
        "segments": [
          {"range": [65, 123], "seed_frame": 75}
        ]
      }
    }
  },
  "occlusion_order": ["inner_pot", "body"]
}
```

- `start_frame` 是硬约束；此前的 track、最终 mask、点云都必须为空。
- 当前约定 part 出现后持续存在，但允许在某一视角因真实遮挡而得到空 mask。
- `segments` 用于快速运动、长期漂移或局部视角失败的重新播种。
- `id` 必须在 1–255 内且互不重复；最终 PNG 为单通道 palette 标签图。
- `occlusion_order` 按从前到后排列，必须恰好包含所有 part。

`data/1` 的正式配置是
[`configs/mask_pipeline_data_1_reusable.json`](configs/mask_pipeline_data_1_reusable.json)，
包含八视角、`body=40`、`inner_pot=65`、`lid=89`，以及 inner pot 和
`GX013140` lid 的局部重播种区间。

## 3. 运行

完整流程：

```bash
cd /data_ft_9_10/wentai/projects/pose_solver

.venv/bin/python -u scripts/run_mask_pipeline.py \
  --config configs/mask_pipeline_data_1_reusable.json \
  --stage all
```

也可按阶段恢复：

```bash
# Qwen 环境：只产生 bbox
/path/to/qwen/python -u tools/stages/masking/detect_mask_seeds.py \
  --config configs/my_mask.json --timestamps 000040 000065 --vis

# SAM 环境：只重跑一个 part/区间
/path/to/sam/python -u tools/stages/masking/track_part_masks.py \
  --config configs/my_mask.json --mode video --part inner_pot \
  --all --seed-frame 000075 --range-start 000065 --range-end 000123

# 只重新合成，不重复运行模型
.venv/bin/python -u scripts/run_mask_pipeline.py \
  --config configs/my_mask.json --stage compose
```

中间和最终输出：

```text
work_root/
├── bboxes/bbox.json
├── tracks/{part}/{timestamp}/{view}.png
├── manifests/
├── multiview_priors/                 # 启用时
└── multiview_completion.json         # 启用时

output_root/
├── masks/{timestamp}/{view}.png
├── mask_manifest.json
└── quality_report.json
```

合成阶段会拒绝缺失的“已出现部件”track，避免缺文件被静默解释为空 mask。
`quality_report.json` 会报告空帧区间、面积异常和建议重新播种帧。

## 4. 多视角能否弥补看不到视角的 mask

可以弥补“该视角分割失败”，但不能凭空恢复“所有相机都看不到”的表面。实现采用：

1. 用可靠源视角的 mask 和 depth 反投影出带 part 标签的 3D 点；
2. 用标定外参、内参投影到失败视角；
3. 用目标视角 depth 做遮挡检验；
4. 聚合一个或多个源视角，得到目标视角的几何 prior。

配置示例：

```json
{
  "multiview_completion": {
    "enabled": true,
    "apply_mode": "prior_only",
    "minimum_source_views": 2,
    "minimum_source_pixels": 100,
    "target_failure_pixels": 100,
    "depth_tolerance": 0.03
  }
}
```

默认推荐 `prior_only`：输出 prior 供 QA、SAM box/point prompt 或后续融合使用，
不直接覆盖原 mask。只有相机标定、同步和 depth 已验证，并且目标 mask 明确是算法
漏检时，才使用显式的 `replace_failed`。真实遮挡下目标视角本来就应为空，盲目复制
其他视角会制造穿透遮挡的假 mask。

## 5. 质量检查

至少检查：

1. 每个视角时间戳完全同步。
2. 已出现部件的二值 track 文件完整。
3. 最终 PNG 像素值只包含背景 0 和配置中的 part ID。
4. `quality_report.json` 的空区间与面积突变。
5. 多视角 prior 是否通过目标 depth 的遮挡检验。
6. contact sheet/视频中是否有跨物体漂移、手部粘连和镜像误跟踪。

多视角只能提高 mask 的可观测性，不能解决 mesh 对称性引起的 6D pose 多解；后者
仍需语义轴、时序连续性、装配约束或非对称特征共同消歧。
