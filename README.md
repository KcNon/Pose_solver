# pose_solver

`pose_solver` 将同步多视角 RGB、Qwen→SAM3 mask、DA3 深度/相机和
ReconViaGen mesh 转换成逐帧 6D pose，并导出可用于 Isaac Sim 的 URDF/USD
资产与物理验证视频。

当前 `data/1` 使用八视角，部件为 `body`、`inner_pot`、`lid`。

## Pipeline

```text
8 路视频
  → 同步抽帧
  → Qwen 检测关键帧 + SAM3 时序传播
  → DA3 固定相机深度 + depth gauge
  → mask 反投影得到分部件点云
  → 多视角 pose 初始化与追踪
  → render-loss / 连续几何约束
  → 冻结 pose 的 mesh 尺度诊断
  → canonical mesh / URDF / USD
  → Isaac PhysX 装配验证
```

mesh 不从操作场景重建。`body`、`inner_pot`、`lid` 分别来自
`/data_ft_9_10/wentai/projects/data/obiect` 下的视频，经过同样的
Qwen→SAM3 mask 流程生成 RGBA，再输入 ReconViaGen。

物体出现帧是硬约束：

| 部件 | 首次出现帧 |
|---|---:|
| `body` | 40 |
| `inner_pot` | 65 |
| `lid` | 89 |

此前的 mask、点云和 pose 输出必须为空。

## 目录约定

```text
common/       可复用算法；不包含数据集路径
configs/      当前正式 pipeline 的数据配置
scripts/      正式、稳定的阶段入口
tools/        诊断、high-FPS 和 runner 内部阶段
tests/        无 GPU 单元测试
experiments/  生成产物；不进入 git
```

`scripts/` 只保留八个正式入口：

| 入口 | 作用 |
|---|---|
| `run_automated_workflow.py` | 数据集端到端编排与契约检查 |
| `run_reconviagen_pipeline.py` | 物体视频到 ReconViaGen mesh |
| `run_mask_pipeline.py` | Qwen→SAM3 mask |
| `run_depth_pipeline.py` | depth gauge 与点云 |
| `run_pose_pipeline.py` | pose、约束、评审和渲染 |
| `export_simulation_assets.py` | canonical mesh 与 URDF |
| `run_isaac_insertion.py` | Isaac 导入和插入验证 |
| `run_isaac_physics_video.py` | 有碰撞的完整物理轨迹 |

不直接运行的模型阶段位于 `tools/stages/`；high-FPS 局部修复位于
`tools/highfps/`；尺度和状态诊断位于 `tools/diagnostics/`。

## 正式配置

`configs/` 只保留当前主流程配置：

| 配置 | 作用 |
|---|---|
| `workflow_data_1.json` | 端到端入口 |
| `data_1_preprocess_8view.json` | 八路同步抽帧 |
| `reconviagen_objects.json` | 独立物体 mesh 重建 |
| `mask_pipeline_data_1_reusable.json` | Qwen/SAM3 与出现帧 |
| `pipeline_data_1_8view.json` | depth 与点云 |
| `pose_data_1_8view.json` | 6D pose 与约束 |
| `geometry_proxies_data_1.json` | pose 侧几何代理 |
| `simulation_data_1.json` | 最终 Isaac 资产和接触控制 |

high-FPS 与尺度诊断配置跟随对应工具，分别位于
`tools/highfps/configs/` 和 `tools/diagnostics/configs/`。

## 推荐运行方式

先检查各阶段输入契约：

```bash
.venv/bin/python scripts/run_automated_workflow.py \
  --config configs/workflow_data_1.json \
  --stage preflight
```

运行 RGB→pose 主流程：

```bash
.venv/bin/python scripts/run_automated_workflow.py \
  --config configs/workflow_data_1.json \
  --stage all
```

也可以运行单一阶段：

```bash
.venv/bin/python scripts/run_mask_pipeline.py \
  --config configs/mask_pipeline_data_1_reusable.json \
  --stage all

.venv/bin/python scripts/run_depth_pipeline.py \
  --config configs/pipeline_data_1_8view.json \
  --stage all

.venv/bin/python scripts/run_pose_pipeline.py \
  --config configs/pose_data_1_8view.json \
  --stage all
```

只更新 mask 时，可运行 `run_automated_workflow.py --stage
depth-postprocess`，复用未变化的 RGB/DA3 固定相机结果。

### ReconViaGen

```bash
.venv/bin/python scripts/run_reconviagen_pipeline.py \
  --config configs/reconviagen_objects.json \
  --stage all
```

可恢复阶段为 `frames`、`masks`、`rgba`、`mesh`。

### Isaac

导出最终资产：

```bash
.venv/bin/python scripts/export_simulation_assets.py \
  --config configs/simulation_data_1.json
```

使用 Isaac Sim 自带的 `python.sh` 导入和验证：

```bash
"$ISAAC_SIM_DIR/python.sh" scripts/run_isaac_insertion.py
"$ISAAC_SIM_DIR/python.sh" scripts/run_isaac_physics_video.py \
  --fps 5 --start-frame 160 --end-frame 245 --keep-frames
```

无 RTX/Vulkan 设备时，第二条命令可增加 `--no-capture`，只运行 CPU
PhysX 并输出 JSON/USD，不生成 MP4。

## 算法与约束

### Mask

Qwen 只负责提供检测框/关键帧，SAM3 负责时序传播。配置中的
`start_frame`、分段种子、bbox override 和 palette ID 都是数据先验，
算法代码不包含部件名称特例。

### Pose

初始化使用多视角 mask 点云、mesh 几何和可用的纹理证据；连续轴对称物体不优化
不可观测的轴向自旋。render-loss 以固定尺度 mesh 的多视角投影、轮廓、mask
和深度为间接监督。轨迹约束只处理 SE(3) 与通用 pairwise/contact 几何，不调用
Isaac，也不把碰撞修复伪装成 pose 监督。

`inner_pot` 快速运动区间使用 29.97 FPS 局部重建，工具和配置位于
`tools/highfps/`。最终正式轨迹仍是一个自包含的 `trajectory.json`。

### 尺度

跨部件尺度误差不允许由 pose 吸收。冻结 pose 的视觉/CPU/PhysX 三重诊断工具位于
`tools/diagnostics/`。尺度应用只改变 part scale 和 render similarity，并断言
所有 pose 矩阵完全不变。

### Isaac 接触控制

最终配置显式使用 1 mm contact offset、0 rest offset 和 0 restitution。
物体进入 `static`、发生接触且位置误差小于 10 mm 后，控制器切换为
6 rad/s、阻尼比 2.5 的柔顺落座模式。误差较大的阻塞碰撞保持正常跟踪，
不会被误判成成功装配。该逻辑依赖状态、接触和误差，不依赖物体名称。

内胆落座阶段相对旧控制器：

- 高度峰峰值：2.68 mm → 1.55 mm；
- 竖直速度 RMS：0.0518 m/s → 0.0088 m/s；
- 平均 pose 误差：6.23 mm → 3.41 mm。

盖子当前仍被实心碰撞代理挡在目标上方约 62 mm。原因是内胆/盖子的空腔与嵌套
拓扑没有被现有实心代理表达；这不是通过关闭碰撞或增大控制力应当掩盖的问题。

## 当前正式产物

```text
experiments/reconviagen_objects/reconviagen_meshes/
experiments/data_1/normalized/
experiments/data_1/highfps_inner_170_195/
experiments/data_1/pose_outputs/solver_scale_calibrated/
experiments/data_1/simulation_assets_scale_calibrated/
experiments/data_1/isaac_runtime_scale_calibrated/
experiments/data_1/isaac_video_scale_calibrated_assembly_bounce_fixed_final/
```

其中最终视频为：

```text
experiments/data_1/isaac_video_scale_calibrated_assembly_bounce_fixed_final/
  complete_physics_driven_trajectory.mp4
```

## 验证

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
.venv/bin/python -m compileall -q common scripts tools tests
git diff --check
```

实验目录只保留最终产物和重建这些产物所需的输入缓存；历史调参 probe、旧 runtime、
旧 solver 版本和重复视频应删除，不作为 pipeline 接口。
