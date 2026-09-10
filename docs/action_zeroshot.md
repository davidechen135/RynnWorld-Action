# UMI 探针实验 — 双臂 20 维 action 作为 control（zero-shot，无微调）

把官方 control 从**人手 pose 视频**换成**双臂 20 维 action 向量流**，验证"低维 action 能否接进现有 control 管线并 zero-shot 出 rollout"。
脚本：[scripts/action_control_zeroshot.py](../scripts/action_control_zeroshot.py)。产物：`outputs/action_zeroshot/basic_pick_place_000/`。

> 定位：这是**接口打通级 smoke**（Execution-Timeline 第 4 步同类工作），**不做任何微调**，用现有 SFT checkpoint 直接推理。
> 目的是回答"接口能不能接上 + zero-shot 迁移性如何"，**不是**"action 已经能精确驱动画面"。

## 方法

### 1. 双臂 20 维 action 从哪来（诚实：合成，语义沿用原 control）
本机无简智/智元真实数据，故**从现有人手 pose control mp4 反推**一条 `[81,20]` 的双臂 action 序列：
- 蓝手→左臂（dim 0–9），红手→右臂（dim 10–19）；
- 每臂 10 维几何量：`[质心x, 质心y, 包围盒宽, 包围盒高, 主方向cos, 主方向sin, 面积, 展开度, dx, dy]`，归一化到 ~[-1,1]。
- 这模拟"双臂 20 维 robot action"，且**语义沿用原 control（手→臂）**，符合"原来的数据是什么类型就按原来来，只是变成双臂 20 维"的要求。

### 2. action_encoder（**确定性、零训练**）：`[81,20] → [48,21,30,52]`
- 时序 81→21（4× 压缩，对齐 latent 帧数）；
- 固定 seeded 投影 `20→48` 通道 + 平滑空间基（左臂偏左半、右臂偏右半的高斯先验）把向量"画"进空间图；
- **逐通道 z-score 对齐**到真实 control latent 的 per-channel mean/std —— 这是 zero-shot 唯一能让注入 latent 落在冻结 control 模块"认识"的分布里的手段。

> 关键诚实点：encoder 是**手工确定性映射**，不是学出来的。action↔画面是"几何注入"，不是"语义驱动"。

### 3. zero-shot 推理
组装 `input_latent.safetensors`（video/control/img latent）→ 官方 `WanImagePipeline`（`RynnWorld-Teleop` EMA，`control_scale=0.109`，50 步）→ rollout。**无微调**。

## 结果

### 接口打通 ✅
| 检查项 | 结果 |
|---|---|
| action 提取 | `[81,20]`，左右臂活动量 0.513/0.516（均衡），全帧非零 |
| encoder 输出 shape | `[48,21,30,52]`（与真实 control latent 完全一致） |
| 分布对齐 | mean/std = **-0.1577 / 1.6582**，与真实 control latent **完全一致**；逐通道均值 L1 差 = **0.0000** |
| pipeline | 冻结 SFT 权重加载正常（825 权重 missing=0），50 步去噪跑通，出 81 帧 832×480 mp4 |
| 显存/速度 | 与原 SFT 推理同量级（~1.09 s/it，50 步 ~55 s，单卡 disable_offload） |

### zero-shot 效果（诚实观测）
对比 `action_showcase.mp4`（三联屏：首帧 | 原 pose rollout | action zero-shot rollout）：
- **首/中帧**：与原 pose rollout **几乎一致**——手在拼图上操作，场景连贯、画质正常。→ 注入 latent 落在分布内，**画面不崩**。
- **末帧**：action 版出现**运动模糊/白色伪影**（手臂快速掠过），原 pose 版是干净抓取。→ 后段 action 时序信号与真实 pose 不完全一致，模型"编"出了动作。

## 结论
1. **接口层面：完全打通。** 双臂 20 维 action 可经 training-free encoder 接进现有 control 管线，shape/dtype/分布全对齐，冻结模型 zero-shot 能出完整 rollout，显存/速度无额外开销。
2. **有效性层面：zero-shot 部分听话、不精确。** 画面不崩、语义大体合理，但动作细节（尤其后段）不准——**符合预期**：control 模块是为真实手部 pose latent 训的，几何投影只"骗"进了分布区间，没有真正语义对齐。
3. **要 action 精确驱动画面，必须训 control 头**（需 `(action_20d, video)` 配对数据 + 短训 control_patch_embedding），这已越过"不做微调"边界，属改造阶段下一步。

## 局限
- action 是从人手 pose 合成的，**非真实机器人 action**（无简智/智元数据）。
- encoder 无学习，空间注入是固定先验，非语义解码。
- 仅 1 case（basic_pick_place_000），单 seed。
