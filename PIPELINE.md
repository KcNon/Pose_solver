# Normalized 数据流水线

输入数据在 `depth-anything-3/试标数据-6.30/2-normalized/`（frames + 深度 recon），
所有生成产物在 `pose_solver/outputs/normalized/`。

## 目录结构

```
2-normalized/                          # 输入（只读）
  frames/{view}/{timestamp}.jpg
  da3-self_cond/{ts}/predictions.npz
  da3-vggt_cond/{ts}/predictions.npz
  vggt-omega/{ts}/predictions.npz
  masks

pose_solver/
  configs/pipeline_normalized.json     # 统一配置
  outputs/normalized/
    masks/{timestamp}/{view}.png       # palette mask + bbox.json
    parts_ply/{backend}/{ts}/{part}.ply
    icp/{backend}/pose_{src}_to_{ref}.json
    icp/{backend}/chain_summary.json
    proj_vis/{backend}/raw/{ts}/       # 无 ICP 投影
    proj_vis/{backend}/{ref}_to_{src}/ # ICP 投影
```

## 环境

```bash
cd /data_ft_9_10/wentai/projects/pose_solver

# pose_solver venv: ICP / 反投影 / 可视化
# qwen3-vl/.venv:   Qwen bbox
# sam3/.venv:       SAM 分割
```

## 全流程（当前版本）

### Step 1 — Qwen bbox + SAM 掩码（111 帧）

```bash
# 单帧测试
./scripts/run_pipeline.sh masks 000000

# 全量（后台推荐）
nohup ./scripts/run_pipeline.sh masks --all > /tmp/run_masks.log 2>&1 &
```

环境变量：`QGPU=5 SGPU=4`，可选 `VIS=1` 生成 bbox 可视化，`FRESH=1` 重置 bbox.json。

### Step 2 — 反投影生成部件点云

```bash
# ICP 采样帧（17 帧，3 段链）
BACKEND=da3_self_cond ./scripts/run_pipeline.sh backproject --sample

# 指定 backend: da3_self_cond | da3_vggt_cond | vggt_omega
BACKEND=da3_vggt_cond ./scripts/run_pipeline.sh backproject --sample

# 指定帧
.venv/bin/python scripts/backproject_normalized.py \
  --pipeline configs/pipeline_normalized.json \
  --backend da3_self_cond --timestamps 000031 000034
```

### Step 3 — Raw 投影可视化（可选，检查点云质量）

```bash
BACKEND=da3_self_cond ./scripts/run_pipeline.sh viz-raw --sample
```

### Step 4 — 链式 ICP + 投影验证

```bash
BACKEND=da3_self_cond ./scripts/run_pipeline.sh icp
```

ICP 采样段（每 3 帧）：
- 段1: 000031 → 000034 → 000037 → 000040
- 段2: 000067 → … → 000088
- 段3: 000098 → … → 000110

链式 init：段内第一对用 camera+both，后续对用上一对变换。
`chain_summary.json` 含链式累积 vs 直接配准对比。

### 一键（Step 2–4，需已有 masks）

```bash
BACKEND=da3_self_cond ./scripts/run_pipeline.sh all --sample
```

## 脚本索引

| 脚本 | 作用 |
|------|------|
| `run_pipeline.sh` | 统一入口 |
| `run_timestamp.sh` | Qwen + SAM（被 run_pipeline 调用） |
| `detect_bbox_batch.py` | Qwen bbox → bbox.json |
| `seg_masks_only.py` | SAM → palette mask PNG |
| `backproject_normalized.py` | mask + depth → parts_ply |
| `visualize_raw_ply.py` | 无 ICP 点云投影 |
| `icp_chain.py` | 链式 per-part ICP |
| `visualize_icp_chain.py` | ICP 结果投影对比 |

## Legacy 脚本（旧 2/output_test 数据，normalized 不用）

- `seg_backproject_parts.py` + `run_pair.sh` + `icp_pose.py` + `visualize_projection.py`
- 配置：`configs/pipeline.json`
