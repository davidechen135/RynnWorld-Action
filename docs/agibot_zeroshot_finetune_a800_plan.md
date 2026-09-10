# AgiBot Zero-shot 与原生轨迹微调计划（2×A800 80GB）

日期：2026-08-12  
状态：评审稿，尚未开始实现或训练  
关联设计：`docs/agibot_native_action_plan.md`

## 1. 服务器能力与配置原则

本机实测：

| 资源 | 实际配置 |
|---|---|
| GPU | 2× NVIDIA A800-SXM4-80GB |
| GPU 互联 | NV8 NVLink |
| Compute Capability | 8.0（Ampere） |
| 主机内存 | 1.0 TiB，当前可用约 937 GiB |
| 数据盘 | 850 GB，当前可用约 614 GB |

结论：5B 模型可以双卡 bf16 训练。A800 不支持 Hopper FP8，因此不启用 FP8。历史同分辨率
双卡 ZeRO-2 + CPU offload 的实测峰值约 20.4 GB/GPU；本机 80 GB 显存充足，原生轨迹
adapter/LoRA 阶段优先使用 **ZeRO-2、无 CPU offload**，避免 PCIe/CPU offload 拖慢训练。

首轮保守配置：

```text
precision                    bf16
GPU                          2
micro batch / GPU            1
gradient accumulation        4
effective batch              1 × 2 × 4 = 8 clips
resolution                   81 × 480 × 832
latent                       48 × 21 × 30 × 52
gradient checkpointing       on
Flash Attention              on
DeepSpeed                    ZeRO-2, no optimizer offload
max grad norm                1.0
num workers                  8 / process
pin memory                   true
```

先跑 20-step memory smoke。若单卡峰值低于 55 GB，再试 `micro_batch=2,
grad_accum=2`，有效 batch 仍为 8；只有吞吐明显提升且显存低于 70 GB 才采用。正式计划不依赖
micro-batch 2。

## 2. 两条实验线必须分开

### 2.1 Zero-shot

Zero-shot 的定义是：**官方 RynnWorld 权重完全冻结，不训练任何新参数**。

原生 AgiBot 数值 `[T,D]` 不能直接 zero-shot 塞进官方骨架 CPE。官方 CPE 的输入是
`[48,21,30,52]` 的 VAE 骨架 latent，两者维度和语义都不同。随机 reshape 或固定随机投影
不是有效 zero-shot，只能算接口 smoke。

因此 zero-shot 线用于回答：官方模型面对真机外观和旧转换 control 时在哪里失败，而不是证明
模型天然理解机器人数值 action。

### 2.2 微调

微调线新增可学习的 `NativeTrajectoryEncoder`，直接学习 AgiBot 数值轨迹到 Wan 内部条件特征
的映射。先只训练 encoder，再加入低秩 LoRA；不再制造五指骨架，也不使用旧骨架 CPE。

## 3. Zero-shot 实验矩阵

所有实验固定官方 SFT checkpoint、seed 42、50 steps、guidance 1.0、相同输出编码参数。

| 编号 | 首帧 | Control | 目的 |
|---|---|---|---|
| Z0 | 官方 | 官方骨架 | 验证官方基线正常 |
| Z1 | AgiBot | 官方骨架 | 隔离真机外观域；必须真正混合 latent |
| Z2 | AgiBot | 当前人工五指骨架 | 复现 task-362 塌缩 |
| Z3 | AgiBot | null/zero control | 判断旧人工 control 是否比无 control 更坏 |
| Z4 | AgiBot | 同一旧骨架，时间修正为 5 秒 | 单独测 30→16 FPS 修复收益 |

注意：现有 `outputs/agibot_362_official_control/heldout_input.safetensors` 的 `img_latent`、
control 和 video latent 都与官方 fold 文件逐字节相同，并不是 Z1。Z1 需要新建：

```text
img_latent             = AgiBot held-out 首帧
control_video_latents  = 官方 fold control
video_latents          = AgiBot GT（仅用于评估）
```

Zero-shot 每个条件至少跑 3 个 held-out episode × 3 seeds（42/123/777）。报告 GT、生成视频、
五帧 strip、detail ratio、DINO/LPIPS drift 和时间一致性。motion energy 不能单独判定成功。

## 4. 微调架构：在哪里注入、训练什么

### 4.1 输入定义

第一阶段做**未来轨迹条件视频生成**：

```text
首帧 RGB + 文本 + 未来 5 秒机器人状态轨迹 -> 未来视频
```

每个时间点使用 24 维：

```text
state/end/position                2×3 = 6
state/end/orientation -> rot6d    2×6 = 12
state/effector/position           2×1 = 2
state/head/position               2
state/waist/position              2
-------------------------------------------
总计                                   24
```

加入 head/waist 是因为相机安装在头部，视角变化不能只由末端轨迹解释。所有维度使用训练集全局
统计归一化，禁止逐 episode min-max。

### 4.2 精确注入位置

Wan 输入 latent：

```text
[B,48,21,30,52]
    ↓ patch_embedding Conv3d(kernel=stride=(1,2,2))
[B,3072,21,15,26]
```

新 encoder：

```text
trajectory [B,81,24]
    ↓ timestamp-aware resample
[B,21,24]
    ↓ MLP(24→256→768)
    ↓ 2× bidirectional Temporal Transformer(dim=768, heads=12)
    ↓ zero-init Linear(768→3072)
[B,21,3072]
    ↓ reshape + broadcast over 15×26
[B,3072,21,15,26]
```

在 [rynnworld_teleop_trainer.py](/root/autodl-tmp/RynnWorld-Teleop/core/finetune/models/wan_i2v/rynnworld_teleop_trainer.py:181)
的 `patch_embedding` 后、`flatten(2).transpose(1,2)` 前相加：

```python
video_features = self.patch_embedding(hidden_states)
trajectory_features = self.native_trajectory_encoder(robot_trajectory)
hidden_states = video_features + trajectory_scale * trajectory_features
hidden_states = hidden_states.flatten(2).transpose(1, 2)
```

末层 projection 权重和 bias 全零初始化，`trajectory_scale=1.0`。因此初始化时
`trajectory_features=0`，原模型输出不变。

### 4.3 微调模块分层

| 阶段 | Native encoder | Wan LoRA | Wan 5B 主干 | 旧骨架 CPE | VAE/T5 |
|---|---:|---:|---:|---:|---:|
| F0 smoke | 训练 | 冻结 | 冻结 | 冻结且不用 | 冻结 |
| F1 单 clip | 训练 | 冻结 | 冻结 | 冻结且不用 | 冻结 |
| F2 多 episode | 训练 | 冻结 | 冻结 | 冻结且不用 | 冻结 |
| F3 LoRA | 训练 | 训练 | 冻结 | 冻结且不用 | 冻结 |

F3 LoRA 只挂在 30 个 Wan block 的：

```text
attn1.to_q
attn1.to_k
attn1.to_v
attn1.to_out.0
```

首轮不动 FFN、不全量 SFT，也不同时训练旧 `control_patch_embedding`。这样能明确区分“原生轨迹
adapter 是否有效”和“主干外观适配是否带来额外收益”。

## 5. 数据重建计划

旧 task-362 latent 不直接复用。重新生成：

```text
/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/agibot_362_native_v1/
/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/agibot_362_native_v1.json
/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/agibot_362_native_v1_dev.json
/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/agibot_362_native_v1_heldout.json
/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/agibot_362_native_v1_stats.safetensors
```

规则：

1. 目标为 81 点、16 FPS、首末跨度 5.0 秒。
2. 从 30 Hz 原始数据约 151 个帧间隔中采样。
3. RGB、state、head、waist、gripper 使用同一组 timestamp。
4. 窗口完整位于有效 `action/end/index` 区间。
5. 20 个原训练 episode 中取 18 train + 2 dev；原 3 个 held-out 只做最终报告。
6. 训练窗口 source stride=75 帧（约 2.5 秒），允许 50% 重叠；dev/held-out 按 episode 隔离。
7. 每个样本保存 episode、source indices、timestamp、schema 和归一化版本。

按现有 episode 长度估计，训练集约 300–380 个 5 秒窗口。实际数量以预处理审计输出为准，
训练步数根据样本数重新计算，不硬编码旧 395 clip 的 epoch 换算。

## 6. 分阶段训练参数

### F0：数据与显存 smoke

```text
GPU                         2×A800
steps                       20
micro batch/GPU             1
grad accumulation           1（有效 batch 2，仅 smoke）
bf16                        on
gradient checkpointing      on
DeepSpeed                   ZeRO-2 no-offload
EMA                         off
checkpoint                  不保存大 checkpoint
```

检查 shape、非零梯度、projection norm、两卡 loss 一致性、显存峰值和 step 时间。

### F1：单 clip overfit

推荐单卡 GPU0，GPU1 可并行跑固定验证推理；单样本使用双卡 DistributedSampler 没有收益。

```text
训练模块                    NativeTrajectoryEncoder only
steps                       500
micro batch                 1
grad accumulation           1
optimizer                   AdamW
encoder lr                  1e-4
weight decay                0.01
warmup                      20 steps
lr schedule                 cosine
EMA                         off
validation                  每 50 steps
checkpoint                  每 100 steps，仅保存 adapter
```

必须用 correct / shuffled / reversed / static trajectory 四组推理。若训练 clip 都不能稳定复现，
或四组输出没有差异，停止，不进入 F2。

### F2：18 train + 2 dev episode，encoder-only

```text
GPU                         2×A800
训练模块                    NativeTrajectoryEncoder only
micro batch/GPU             1
grad accumulation           4
effective batch             8
encoder lr                  5e-5
weight decay                0.01
epochs                      最多 10
warmup                      总 optimizer steps 的 5%
lr schedule                 cosine
EMA                         暂不启用，直接评估 raw checkpoint
validation                  每 100 optimizer steps
checkpoint                  每 100 steps，仅 adapter + stats + config
early stop                  dev 连续 3 次无改善
```

如果约 340 个训练窗口，每 epoch 约 `ceil(340/8)=43` optimizer steps，最多约 430 steps。

### F3：Native encoder + LoRA

从 F2 最佳 adapter 初始化：

```text
GPU                         2×A800
LoRA rank / alpha           16 / 16
LoRA targets                attn1 q/k/v/out only
micro batch/GPU             1
grad accumulation           4
effective batch             8
encoder lr                  2e-5
LoRA lr                     5e-6
weight decay                0.01
epochs                      3–5
warmup                      5%
EMA                         off（先保证 raw checkpoint 可解释）
checkpoint                  每 100 steps，adapter + LoRA
early stop                  dev 连续 3 次无改善
```

当前旧实验的 LoRA lr 是 `1e-4`。新计划降低到 `5e-6`，因为目标是轻微适配真机外观，避免
300 多个窗口覆盖 5B 基座能力。

## 7. 评估标准与验收条件

### 7.1 本阶段能证明什么

当前输入是未来机器人**状态轨迹**，不是底层控制命令；AgiBot 又是真机离线数据，不能像
仿真器一样执行任意反事实 action 并取得对应 GT。本阶段只证明三件事：

1. 模型不塌缩，能保持真机视频结构。
2. 输出确实响应轨迹，而不是只靠首帧和文本生成任务平均动作。
3. 对真实配对轨迹，机器人运动和画面变化比错误轨迹更接近 GT。

因此不能把结果称为严格的 action-conditioned simulator，也不能在缺少反事实 simulator GT
时报告成官方 WorldSimProbe 分数。

### 7.2 固定评估协议

| 项目 | 规则 |
|---|---|
| 模型选择 | 只使用 2 个 dev episode；不得根据 held-out 调参或选择 checkpoint |
| 最终测试 | 3 个 held-out episode × 3 seeds（42/123/777），共 9 个 rollout |
| 条件对照 | 每个首帧以相同 seed 生成 correct、shuffled、reversed、static |
| 生成配置 | 固定 50 steps、guidance 1.0、分辨率、编码器和 5.0 秒物理时长 |
| 时间对齐 | 所有指标按 timestamp 对齐，不只按帧号对齐 |
| 统计报告 | episode 明细、均值、标准差、bootstrap 95% CI 和逐样本 paired comparison |

同组条件对照必须复用扩散初始噪声，避免把 seed 差异误认为轨迹响应。

### 7.3 三级指标

#### L0：结构稳定性，硬门槛

- detail_ratio：最后 1/3 帧空间梯度能量 / 最前 1/3 帧；低于 0.80 初判塌缩。
- DINO 特征相对首帧的时间曲线：检查语义结构是否突然消失。
- 相邻帧 LPIPS 和逐帧像素差：检查闪烁、冻结和突跳。
- 解码失败、NaN、纯色帧、重复帧比例及五帧 strip。

motion energy 不能单独判断成功，画面融化同样会产生高运动。任何 dev rollout 出现融化、
纯色或结构消失均不进入下一阶段。0.80 是根据当前失败样本 0.62–0.74、正常官方样本约
0.92 设置的首轮阈值；先用 Z0 和 GT 核验并冻结，之后不得为迁就新结果修改。

#### L1：轨迹是否真正生效，核心门槛

对 correct / shuffled / reversed / static 测：

- 条件响应距离：相同 seed 下机器人区域的 DINO、LPIPS 和 optical-flow 差异。
- 机器人运动保真度：RobotSeg 或固定人工 mask 内，生成视频与配对 GT 的光流误差。
- 方向一致性：生成与 GT 机器人光流的 cosine similarity。
- 幅值误差：机器人区域 flow RMS 的相对误差。
- 时序误差：运动强度曲线互相关得到的最佳 lag（秒）。

主指标采用 WorldSimProbe T2/T3 的 masked-flow 思路：

    robot_flow_score =
    100 * max(0, 1 - flow_error / max(reference_flow_rms, motion_floor))

这里的 reference 是 AgiBot 配对 GT，不是反事实 simulator rollout。mask、flow estimator、
motion_floor 和窗口划分必须在查看 held-out 前固定。

通过条件：

1. correct 相对 shuffled 和 static 的平均 robot_flow_score 至少高 5 个点。
2. correct 至少在 2/3 dev episode 上优于 shuffled 和 static，不能由一个样本拉高。
3. correct 的方向一致性更高、绝对时序 lag 更小；reversed 应显著破坏时序匹配。
4. correct 与 shuffled 若近似，判定模型忽略条件，即使 reconstruction loss 很低也停止。

#### L2：视频重建与感知质量，次级指标

通过 L0/L1 后，再报告生成视频对 paired GT 的 DINO similarity、LPIPS、SSIM/PSNR，包括
全帧和机器人 mask 内的结果。held-out 太小时不把 FVD 用作主指标；它方差大，也检测不了
模型是否忽略 action。清晰但不跟随轨迹的模型，不能优于画质稍低但正确响应轨迹的模型。

### 7.4 WorldSimProbe 是否纳入

[WorldSimProbe](https://evophys.com/WorldSimProbe/) 沿 action → robot motion → contact →
object response 的因果链评估，这个方向应纳入；但要按现有离线真机数据的能力分级：

| WorldSimProbe 项目 | 当前处理 |
|---|---|
| T1 Local Action Calibration | 部分纳入：加小/大物理可行扰动，检查响应是否单调；无反事实 GT，只称 sensitivity |
| T2 Global Trajectory Coverage | 部分纳入：用跨 episode donor 轨迹检查是否退回任务平均动作；不报官方 flow score |
| T3 Action-Source Preservation | 暂不计分：没有 expert/policy/真人遥操作等可靠 source 标签 |
| T4 Interaction Grounding | 条件纳入：补齐物体 mask、接触/非接触标签后才能检查假接触和近距离幻觉 |
| T5 Interaction Dynamics | 后续阶段：需要 push/pull/rotate/shake/drop 等多原语标签；task-362 不足以支撑 |

T1 适配版使用同一首帧、同一 seed 的 original/small/large 轨迹。扰动必须保持关节和工作空间
有效，并分别作用于 position、rotation、gripper。大扰动的机器人区域响应应显著大于小扰动，
且与输入扰动幅值正相关。这只证明可控敏感性，不证明响应物理正确。

T4/T5 不参与 F2/F3 首轮 checkpoint 选择。若后续补齐物体 mask、接触时刻和原语标签，再
单独增加 interaction evaluation，不能用主观观看代替缺失的测量。

### 7.5 分阶段通过条件

#### Zero-shot

- Z0 必须通过 L0，否则先修官方基线或评估流水线。
- Z1 若通过 L0，说明真机首帧外观域不是唯一塌缩原因。
- Z2 塌而 Z3/Z4 改善，说明旧人工 control 或时间窗有直接责任。
- Z0–Z4 输出同格式 L0 指标和 strip，不能只挑成功案例。

#### F1：单 clip overfit

- flow loss 下降，projection 从全零学到非零且梯度稳定。
- correct rollout 通过 L0。
- correct 与三种错误条件产生可测的机器人运动差异，否则停止。

#### F2：encoder-only

- 2 个 dev episode、3 seeds 全部通过 L0。
- 满足 L1 的 5 点 margin、episode 一致性、方向和时序条件。
- 相对 zero/null control，correct 的 GT 运动保真度有净提升。

#### F3：encoder + LoRA

- 与 F2 使用相同 dev 样本、seed 和指标。
- L0 不退化，L1 必须有稳定净收益；只降低 loss 或 LPIPS 不算成功。
- F3 没超过 F2 就选择 F2，不因模型更复杂而保留 F3。

模型选定后才一次性运行最终 held-out。报告 L0/L1/L2 和四组条件视频，不能根据 held-out
结果继续选择 checkpoint。

### 7.6 评估产物

    reports/agibot_native/<run_id>/summary.json
    reports/agibot_native/<run_id>/per_episode.csv
    reports/agibot_native/<run_id>/condition_ablation.csv
    reports/agibot_native/<run_id>/metric_curves.png
    reports/agibot_native/<run_id>/video_strips/
    reports/agibot_native/<run_id>/videos/

报告不合成单一总分，固定按“塌缩率、轨迹响应、机器人运动保真度、感知质量、交互能力”
五栏展示；缺标签的交互项写 N/A，不按零分或满分处理。

## 8. 实施顺序与预计产物

```text
P0  完成 Z0–Z4 zero-shot 对照脚本和报告
P1  重建 native_v1 数据与审计报告
P2  实现 NativeTrajectoryEncoder、dataset、forward、轻量 checkpoint
P3  20-step 双卡 A800 smoke
P4  单 clip overfit
P5  encoder-only 多 episode 训练
P6  encoder + LoRA 训练
P7  固定 held-out 最终评估
```

预计新增：

```text
core/control/native_trajectory_encoder.py
scripts/agibot_native_prep.py
scripts/agibot_native_zeroshot.py
scripts/agibot_native_validate.py
scripts/agibot_native_overfit.sh
scripts/agibot_362_native_a800_2gpu.sh
configs_acc/2gpu_a800_native.yaml
configs_zero/zero2_a800_no_offload.json
```

在计划获批前不启动长训练。第一批应交付 Z0–Z4、数据审计和 F1 单 clip overfit；这三项足以
判断是否值得进行 F2/F3。
