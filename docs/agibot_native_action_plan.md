# AgiBot 原生轨迹条件世界模型：算法架构与训练计划

日期：2026-08-12  
状态：设计提案 v1，尚未实现  
基座：Wan2.2-TI2V-5B / RynnWorld-Teleop SFT

## 1. 目标与边界

本阶段不再把 AgiBot 机器人数据伪造成 21 点人手骨架，也不使用固定随机投影制造
`control_video_latents`。新增一个可学习的原生轨迹编码器，直接学习：

```text
首帧 RGB + 文本 + 未来机器人轨迹
                  -> 未来 81 帧真机第一视角视频
```

第一阶段的准确任务名称是**未来轨迹条件视频生成**，不是严格的 action-conditioned
动力学预测。原因是输入包含未来每帧的末端位姿；模型不需要自行预测机械臂轨迹，只需学习
轨迹与视觉变化的对应关系。这与官方 RynnWorld 的“未来手部姿态骨架 -> 视频”最接近，
适合作为原生 AgiBot 条件接口的第一步。

后续若要做严格世界模型，再将条件改为“当前状态 + 未来控制命令”，并禁止输入未来状态。

## 2. 数据定义

### 2.1 每个样本

统一为 81 个时间点、16 FPS、首末时间跨度 5.0 秒：

```text
img_latent       [48, 1, 30, 52]   真机 RGB 首帧
video_latents    [48,21, 30, 52]   真机 RGB 视频目标
robot_trajectory [81, 24]          原生数值轨迹
trajectory_mask  [81]              有效点标记
text_embedding   [L,4096]          任务文本
```

RGB 与 H5 必须按同一组 timestamp 最近邻/插值采样，不能再直接取连续 81 个 30 Hz 帧。
一个样本应覆盖约 151 个 AgiBot 原始帧。只保留完整落在 `action/end/index` 有效区间内的窗口。

### 2.2 v1 轨迹字段

优先使用实际姿态条件：

```text
左右末端 position                 2 x 3  = 6
左右末端 orientation quaternion
  -> rotation-6D                  2 x 6  = 12
左右夹爪 openness                 2 x 1  = 2
头部 position                              = 2
腰部 position                              = 2
------------------------------------------------
总计                                      24 维
```

来源：

```text
state/end/position
state/end/orientation
state/effector/position
state/head/position
state/waist/position
```

四元数先归一化，再转换为 rotation-6D，避免 `q` 与 `-q` 表示同一旋转造成的不连续。
夹爪实际反馈按机器人固定物理上下限归一化到 `[0,1]`，其中 `1=open`。上下限必须由训练集
统计或设备定义固定，禁止逐 episode min-max；逐 episode 归一化会消除跨 episode 的绝对含义。

数据文件同时保存以下元数据：

```json
{
  "episode_id": "650190",
  "source_indices": [729, 731, 733],
  "source_timestamps_ns": [],
  "target_fps": 16,
  "valid_range": [69, 1394],
  "trajectory_schema": "ee_pose6d_gripper_v1"
}
```

### 2.3 标准化

只用训练 episode 计算每一维的全局 `mean/std`，held-out 不参与。position 可先转为相对首帧：

```text
delta_position_t = position_t - position_0
```

头部和腰部状态保留绝对值，因为相机安装在头部，视角变化不能只由末端轨迹解释。末端位置
第一版也保留绝对值；若消融实验显示跨 episode 的基座偏置明显，再对比相对首帧表示。所有字段
必须使用同一套训练集全局统计。v1 输入为：

```text
[ee_position_t, ee_rotation6d_t, gripper_t, head_position_t, waist_position_t]
```

所有统计写入单独 checkpoint，并在训练、推理共用。禁止像旧版一样按每条 episode 单独缩放。

## 3. 模型架构

### 3.1 总体数据流

```text
robot trajectory [B,81,24]
        |
        | time-aware resample / Conv1d
        v
latent trajectory [B,21,24]
        |
        | input MLP + temporal Transformer
        v
temporal condition [B,21,768]
        |
        | projection 768 -> 3072 (zero-init)
        v
action condition [B,21,3072]
        |
        | broadcast over 15x26 patch grid
        v
[B,21*15*26,3072]
        |
        | x action_scale + patch_embedding(noisy video)
        v
Wan 30-block DiT -> velocity prediction -> flow-matching loss
```

### 3.2 NativeTrajectoryEncoder

第一版保持小型、可诊断：

```text
InputNorm(24)
Linear(24, 256)
SiLU
Linear(256, 768)
2 x TemporalTransformerBlock(
    dim=768,
    heads=12,
    mlp_ratio=4,
    causal=False
)
LayerNorm(768)
Linear(768, 3072, bias=True)  # weight/bias zero-init
```

使用双向 temporal transformer 是因为本阶段整条未来轨迹在生成前已知。严格在线 action world
model 阶段再改为 causal encoder。

参数量约为千万级，相对 5B backbone 很小。输出投影零初始化，确保 step 0 与不加轨迹条件的
基座模型一致。

### 3.3 空间注入 v1

每个 latent 时间步的 3072 维条件向量广播到 patch 后该时间步全部 `15 x 26` 个空间 token：

```python
video_tokens = patch_embedding(noisy_video)               # [B,N,3072]
trajectory_tokens = encoder(robot_trajectory)              # [B,21,3072]
trajectory_tokens = broadcast_spatial(trajectory_tokens)   # [B,N,3072]
hidden = video_tokens + trajectory_scale * trajectory_tokens
```

`trajectory_scale` 初始化为 1；真正的零扰动由末层 projection 的全零初始化保证。该方案不伪造
像素位置，要求模型通过真机配对数据学习“基座坐标轨迹 -> 画面变化”。

### 3.4 为什么不直接复用旧 control latent

旧 `control_patch_embedding` 是为 VAE 骨架 latent `[48,21,30,52]` 训练的。AgiBot 数值轨迹
没有相同的通道和空间语义。固定随机投影只能匹配 shape/均值/方差，不能建立可学习的条件映射。

新 encoder 直接输出 Wan 内部维度 3072，不经过 VAE，也不使用官方骨架 CPE。旧骨架入口保留，
通过 `condition_mode in {pose_video, native_trajectory}` 切换，方便严格回归测试。

### 3.5 v2 可选空间增强

若 v1 能过拟合但跨 episode 控制跟随弱，再增加可学习空间基：

```text
temporal feature [B,21,768]
  + learned spatial queries [15,26,Ds]
  -> factorized modulation / cross-attention
  -> [B,21,15,26,3072]
```

不建议第一版直接生成完整 3072 通道空间图，参数与显存浪费大，也容易记忆固定床面。v1 的
全局时间条件先验证原生轨迹是否有可学习信号。

## 4. 训练目标

### 4.1 主损失

沿用当前 Wan flow-matching 目标：

```text
z_t = (1-t) * video_latent + t * noise
target = noise - video_latent
L_flow = MSE(model(z_t, image, text, trajectory), target)
```

继续忽略首个 latent frame 的 loss，因为首帧由 `img_latent` 固定：

```python
L_flow = mse(pred[:, :, 1:], target[:, :, 1:])
```

第一版不增加复杂辅助 loss。先证明条件接口有效，避免 loss 改动掩盖数据问题。

### 4.2 条件 dropout

训练中以 10% 概率把整条 trajectory 替换为 learned/null trajectory，用于 classifier-free
condition dropout。另以 10% 概率 dropout 文本。禁止逐时间点随机丢弃，避免制造不连续轨迹。

### 4.3 条件有效性诊断

每次验证必须做 shuffled-condition 对照：同一首帧分别输入正确轨迹和另一 episode 的轨迹。
如果输出几乎相同，说明模型仍在忽略 trajectory，即便 reconstruction loss 很低也不能通过。

## 5. 分阶段训练计划

### Stage 0：数据审计与重建

产物：新版 train/held-out 数据，不覆盖旧目录。

建议路径：

```text
/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/agibot_362_native_v1/
/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/agibot_362_native_v1.json
/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/agibot_362_native_v1_heldout/
/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/agibot_362_native_v1_heldout.json
/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/agibot_362_native_v1_stats.safetensors
```

必须通过：

- 每个样本 81 点、16 FPS、跨度 5.0 秒。
- RGB 与 trajectory 使用同一 timestamp 索引。
- 所有窗口完整位于有效 action 区间。
- rotation-6D 无 NaN/突跳，夹爪方向人工抽查通过。
- train/held-out episode 完全不相交。
- 保存 source index、timestamp 和 schema，样本可追溯。

### Stage 1：单 clip overfit

目的：只验证实现和条件接口，不讨论泛化。

```text
数据：1 个动作明显的 5 秒 clip
冻结：Wan backbone、原 control CPE、文本编码器、VAE
训练：NativeTrajectoryEncoder
步数：500，验证每 50 步
优化器：AdamW
encoder lr：1e-4
weight decay：0.01
warmup：20 steps
batch：1，grad accumulation：1
EMA：关闭
```

通过标准：

- flow loss 明显下降并稳定。
- 训练 clip rollout 不塌，结构保持率接近 GT。
- 正确轨迹优于 shuffled/reversed/static 三种轨迹。
- projection 权重从 0 学到非零，gradient norm 非零且稳定。

若单 clip 无法过拟合，停止，不得扩大数据或解冻 LoRA。优先检查数据读取、shape、mask、
checkpoint 保存加载和模型是否真正消费 trajectory。

### Stage 2：单 episode 泛化

```text
数据：1 个 episode，按时间窗口划分 train/val
训练：NativeTrajectoryEncoder
backbone：冻结
步数：1000-2000，以 val 指标早停
encoder lr：5e-5
EMA：0.99，start step 100
```

通过标准：未见时间窗口不塌；正确轨迹与 shuffled 轨迹存在稳定、可视的动作差异。

### Stage 3：20 episode，encoder-only

```text
训练：20 个 task-362 train episodes
评估：3 个完整 held-out episodes
训练参数：仅 NativeTrajectoryEncoder
encoder lr：3e-5
epochs：10 上限，按 held-out proxy/episode-val 早停
有效 batch：16
EMA：0.99
```

每 200 optimizer steps 保存 checkpoint，并在固定 3 个验证 clip 上生成：

```text
correct trajectory
shuffled trajectory
reversed trajectory
static trajectory
```

注意：held-out episode 只用于最终选择前的固定评估，不应频繁据此调参。可从 20 个训练 episode
中再留 2 个作为开发集。

### Stage 4：解冻 LoRA

只有 Stage 3 证明 encoder-only 能消费轨迹后才进入：

```text
训练：NativeTrajectoryEncoder + Wan LoRA
初始化：Stage 3 最佳 encoder
LoRA：rank 16，alpha 16
target：先只选 attention q/k/v/out
encoder lr：1e-5
LoRA lr：5e-6
control/backbone weight decay：0.01
epochs：3-5，早停
```

不要第一轮就使用当前 `1e-4` LoRA 学习率。目标是轻微适配真实机器人外观，而不是让小数据覆盖
基座 5B 模型。

Stage 4 必须与 Stage 3 encoder-only 用同一数据、seed、验证 clip 比较，才能判断 LoRA 的净贡献。

### Stage 5：严格 action-conditioned 模型（后续）

将输入改为：

```text
当前 state_t + 未来 action_t:t+H
```

输出仍为未来视频。此阶段不得输入未来 `state/end/*`，否则存在 future-state leakage。需要先确认
AgiBot `action/*` 的命令语义、执行延迟和控制频率，并将 trajectory encoder 改为 causal/action encoder。

## 6. 实验矩阵

最少保留以下对照：

| 编号 | 条件 | 可训练模块 | 目的 |
|---|---|---|---|
| B0 | 无轨迹 | 无 | I2V 基座下限 |
| B1 | 旧人工手骨架 | 官方 CPE | 复现旧失败 |
| A1 | 原生轨迹 | encoder only | 验证新接口 |
| A2 | 原生轨迹 shuffled | encoder only | 验证是否真正听条件 |
| A3 | 原生轨迹 | encoder + LoRA | 测 LoRA 净收益 |

固定相同首帧、文本、seed、采样步数和验证 episode。禁止用不同输入比较模型优劣。

## 7. 评估指标

### 7.1 必须报告

- 帧网格和完整视频：GT / correct / shuffled / reversed / static。
- 结构保持：detail ratio、LPIPS/DINO feature drift。
- 视频质量：MUSIQ 或现有 imaging-quality 基线。
- 时间一致性：光流一致性、temporal feature distance。
- 条件敏感度：correct 与 shuffled 输出的 feature/video 差异。
- 训练指标：loss、grad norm、encoder projection norm、trajectory scale。

### 7.2 不得单独作为结论

`motion energy` 只能辅助观察。塌缩本身也会产生运动，必须与结构指标和帧可视化一起判断。

### 7.3 成功标准

第一阶段成功不要求精确像素复现，但必须同时满足：

1. held-out rollout 不出现当前的数帧后融化。
2. correct trajectory 的动作方向与时间明显优于 shuffled/static。
3. encoder-only 相对无条件基线有收益。
4. LoRA 若启用，必须相对 encoder-only 有稳定净收益，而非只降低训练 loss。

## 8. 工程改动范围

建议新增而非覆盖旧实现：

```text
core/control/native_trajectory_encoder.py
scripts/agibot_native_prep.py
scripts/agibot_native_validate.py
scripts/agibot_native_overfit.sh
scripts/agibot_362_native_2gpu.sh
docs/agibot_native_action_plan.md
```

需要修改：

```text
core/finetune/datasets/wan_dataset.py
  - 读取 robot_trajectory / trajectory_mask

core/finetune/models/wan_i2v/rynnworld_teleop_trainer.py
  - 初始化 encoder
  - collate trajectory
  - forward 传入 trajectory
  - optimizer/checkpoint 管理 encoder

core/inference/rynnworld_teleop.py 或独立 inference_native.py
  - 推理时加载 trajectory encoder 和统计量
```

旧 `pose_video` 路径必须保持 byte-identical，默认模式仍不变。新 checkpoint 至少保存：

```text
native_trajectory_encoder.safetensors
native_trajectory_stats.safetensors
native_trajectory_config.json
```

## 9. 推荐执行顺序

```text
1. 重建 timestamp 对齐的 native_v1 数据
2. 数据审计和可视化
3. 单 clip overfit
4. shuffled/reversed/static 条件测试
5. 单 episode 验证
6. 20 episode encoder-only
7. 3 held-out episode 固定评估
8. 最后才加入低学习率 LoRA
9. 通过后再研究严格 action-conditioned 版本
```

任何阶段若正确轨迹与 shuffled 轨迹没有明显差异，都应视为条件被忽略，停止扩大训练。
