# Qwen + SAM3 时序 Mask Pipeline

本文档说明 `pose_solver` 中正式使用的六视角部件分割流程。目标是对固定相机拍摄的电饭锅操作序列生成连续的 palette mask，同时避免逐帧 Qwen 检测带来的框抖动和普通 SAM 视频传播的长期漂移。

## 1. 方法概览

正式流程只在种子帧（默认 `000000`）调用一次 Qwen3-VL：

```text
六视角首帧
   │
   ├── Qwen3-VL：lid / body / inner_pot bbox
   │
   ├── SAM3 视频模型：lid + inner_pot 双向时序传播
   │
   ├── SAM3 图像模型：body 使用固定首帧框逐帧重锚
   │
   └── 可见性融合：lid > inner_pot > body
          ├── palette PNG
          ├── 每视角预览视频
          ├── 每视角 111 帧总览图
          └── 面积和空帧统计
```

采用两条 SAM 分支的原因：

- 锅盖和内锅会大幅运动，视频模型更容易保持物体身份和时序连续性。
- 锅体与相机基本固定，但会被手、内锅和锅盖遮挡。每帧使用相同 Qwen 框重新运行图像 SAM，比长期传播更稳定。
- 最后按物理前后关系消除重叠，保证一个像素只属于一个标签。

## 2. 输出格式

最终目录结构：

```text
output_root/
├── masks/
│   ├── bbox.json
│   ├── 000000/{2-1,2-2,...,2-6}.png
│   ├── 000001/{2-1,2-2,...,2-6}.png
│   └── ...
├── preview/
│   ├── 2-1_overlay.mp4
│   ├── 2-1_contact_sheet.jpg
│   └── ...
├── temporal_masks.json
└── pipeline.json
```

PNG 是 PIL palette 模式，固定标签如下：

| ID | 部件 | 显示颜色 |
|---:|---|---|
| 0 | background | black |
| 1 | lid | red |
| 2 | body | green |
| 3 | inner_pot | blue |

目标完全离开画面或被完全遮挡时允许出现空 mask。这类空帧不能简单当作漏检，应结合预览视频判断。

## 3. 环境和模型

```text
pose_solver/.venv          融合、统计、预览和调度
qwen3-vl/.venv             Qwen3-VL bbox
sam3/.venv                 SAM3 图像/视频模型
```

默认路径写在 [mask_pipeline_normalized.json](configs/mask_pipeline_normalized.json) 中：

- `frames_dir`：输入帧，布局必须是 `frames/{view}/{timestamp}.jpg`
- `work_root`：Qwen bbox 和两条 SAM 分支的可恢复中间结果
- `output_root`：最终 mask、预览和统计
- `qwen_python` / `sam_python`：两个模型环境的 Python
- `qwen_model` / `sam_ckpt`：本地模型和权重

六个视角名称固定为 `2-1` 至 `2-6`，同一批次必须拥有相同的时间戳序列。

## 4. 一条命令运行全部流程

```bash
cd /data_ft_9_10/wentai/projects/pose_solver

.venv/bin/python -u scripts/run_temporal_mask_pipeline.py \
  --config configs/mask_pipeline_normalized.json \
  --seed-timestamp 000000 \
  --qwen-gpu 5 \
  --video-gpu 6 \
  --body-gpus 4 5 6 7
```

默认行为：

1. 检查六个视角的时间戳是否一致。
2. 若 `work_root/temporal/masks/bbox.json` 不存在，运行一次六视角 Qwen。
3. 使用 SAM3 视频模型跟踪锅盖和内锅。
4. 将六个锅体任务调度到 `--body-gpus` 并行运行。
5. 融合到 `output_root`，生成六个 MP4 和六张 contact sheet。

已完整存在的中间分支会自动复用。全部强制重跑：

```bash
.venv/bin/python -u scripts/run_temporal_mask_pipeline.py \
  --config configs/mask_pipeline_normalized.json --force
```

常用断点选项：

```text
--skip-qwen       复用已有 bbox.json
--skip-temporal   复用锅盖/内锅时序结果
--skip-body       复用锅体逐帧结果
--skip-fusion     暂不生成最终目录
--skip-review     只生成 mask 和统计，不编码视频/总览图
```

如果指定跳过的分支并不完整，程序会直接报错，不会静默生成不完整结果。

## 5. 分步骤运行

通常建议使用统一入口。调试时可以分别运行以下阶段。

### 5.1 Qwen 首帧 bbox

Qwen 与 temporal SAM 共用一个运行时配置，其中 `masks_dir` 是 temporal 中间目录：

```bash
CUDA_VISIBLE_DEVICES=5 \
/data_ft_9_10/wentai/projects/qwen3-vl/.venv/bin/python -u \
  scripts/detect_bbox_batch.py \
  --pipeline WORK_ROOT/runtime_configs/temporal.json \
  --timestamp 000000 --vis
```

输出：

```text
WORK_ROOT/temporal/masks/bbox.json
WORK_ROOT/temporal/masks/_bbox_vis/000000/*.png
```

### 5.2 锅盖和内锅时序传播

```bash
CUDA_VISIBLE_DEVICES=6 \
/data_ft_9_10/wentai/projects/sam3/.venv/bin/python -u \
  scripts/seg_masks_temporal.py \
  --pipeline WORK_ROOT/runtime_configs/temporal.json \
  --all --init-timestamp 000000 \
  --views 2-1 2-2 2-3 2-4 2-5 2-6 \
  --parts lid inner_pot --gpu 6
```

`--range-start` 和 `--range-end` 可用于只写某一时间段，但模型仍会围绕种子帧做双向传播。

### 5.3 锅体逐帧重锚

单视角调试：

```bash
CUDA_VISIBLE_DEVICES=4 \
/data_ft_9_10/wentai/projects/sam3/.venv/bin/python -u \
  scripts/seg_masks_body_reanchor.py \
  --pipeline WORK_ROOT/runtime_configs/body.json \
  --view 2-1 --init-timestamp 000000 --gpu 4
```

六视角多 GPU：

```bash
.venv/bin/python -u scripts/run_body_multigpu.py \
  --pipeline WORK_ROOT/runtime_configs/body.json \
  --views 2-1 2-2 2-3 2-4 2-5 2-6 \
  --gpus 4 5 6 7 \
  --python /data_ft_9_10/wentai/projects/sam3/.venv/bin/python
```

### 5.4 融合与质检

```bash
.venv/bin/python -u scripts/fuse_all_views_temporal_masks.py \
  --temporal-dir WORK_ROOT/temporal/masks \
  --body-dir WORK_ROOT/body/masks \
  --frames-dir /path/to/frames \
  --output-dir /path/to/final_output \
  --views 2-1 2-2 2-3 2-4 2-5 2-6
```

融合规则是 `lid > inner_pot > body`。统计写入 `temporal_masks.json`，包括每个视角、每个部件的非空帧数和像素面积范围。

## 6. 新数据批处理

处理另一套同布局数据时，复制配置并修改以下字段即可：

```json
{
  "frames_dir": "/new/data/frames",
  "work_root": "/new/work/mask_pipeline",
  "output_root": "/new/data/temporal_masks",
  "qwen_python": "/path/to/qwen/.venv/bin/python",
  "sam_python": "/path/to/sam3/.venv/bin/python",
  "qwen_model": "/path/to/Qwen3-VL-8B-Instruct",
  "sam_ckpt": "/path/to/sam3.1_multiplex.pt"
}
```

种子帧应尽量同时看见锅盖、锅体和内锅。如果某个物体在 `000000` 完全不可见，应选择最早的完整可见帧作为 `--seed-timestamp`，或先为新的关键帧补充 Qwen bbox 再分段传播。

## 7. 质量检查

完成后至少检查：

1. `masks` 数量是否等于 `视角数 × 帧数`。
2. PNG 尺寸是否与输入一致，像素值是否只包含 `0,1,2,3`。
3. 六张 `*_contact_sheet.jpg` 是否存在跨物体漂移。
4. `temporal_masks.json` 中面积突变对应的是物体靠近相机、离开画面或真实遮挡，而不是背景误检。
5. 物体离开画面时应为空，不应为了“每帧非空”而复制旧 mask。

当前已验证数据的最终结果位于：

```text
/data_ft_9_10/wentai/projects/depth-anything-3/试标数据-6.30/2-normalized/
  temporal_sam3_all6_000000/
```

## 8. 正式代码索引

| 文件 | 作用 |
|---|---|
| `common/qwen_bbox.py` | Qwen prompt、bbox 解析和可视化 |
| `common/mask_io.py` | bbox.json、palette 和标签公共格式 |
| `scripts/detect_bbox_batch.py` | 六视角 Qwen bbox |
| `scripts/seg_masks_temporal.py` | 锅盖/内锅 SAM3 时序传播 |
| `scripts/seg_masks_body_reanchor.py` | 固定锅体逐帧 SAM3 重锚 |
| `scripts/run_body_multigpu.py` | 六视角锅体多 GPU 调度 |
| `scripts/fuse_all_views_temporal_masks.py` | 标签融合、统计和预览 |
| `scripts/run_temporal_mask_pipeline.py` | 完整流程统一入口 |
