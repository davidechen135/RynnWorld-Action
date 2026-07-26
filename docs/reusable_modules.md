# 可复用模块梳理（Stage 1 控制注入 / Stage 2 蒸馏）

以官方代码为准核对，标注后续三层视频模型改造中 **adopt / cherry-pick / reject** 的候选。
所有行号基于 pinned SHA `503333b`。

---

## 1. Stage 1 控制注入（核心复用点）

### 1.1 control_patch_embedding（零初始化 Conv3d）
- 定义：[core/finetune/models/wan_i2v/rynnworld_teleop_trainer.py:597-615](../core/finetune/models/wan_i2v/rynnworld_teleop_trainer.py#L597)
- `add` 模式：新建一个与原 `patch_embedding` 同形状的 `nn.Conv3d(in_channels, out_channels, kernel=patch_size, stride=patch_size)`，
  **weight/bias 全部 zero-init**，`control_scale = nn.Parameter(0.1)`。零初始化保证训练开始时控制分支不扰动 base 模型。
- `concat` 模式：Conv3d 输入通道翻倍（`2*in_channels`），前半拷贝原 patch_embedding 权重、后半 zero-init。

### 1.2 注入公式（forward）
- 位置：[rynnworld_teleop_trainer.py:179-191](../core/finetune/models/wan_i2v/rynnworld_teleop_trainer.py#L179)
- 三种 `control_type`：
  ```python
  # add / add-plus:
  h = patch_embedding(x)
  h = h + control_scale * control_patch_embedding(control_video_latent)
  # concat:
  h = control_patch_embedding( cat([x, control_video_latent], dim=1) )
  ```
- 实测加载值：SFT teacher `control_scale = 0.1091`（推理日志）。

### 1.3 checkpoint 拆分格式
- teacher checkpoint = `ema_weights.bin`（825 个 transformer 权重）+ `control_patch_embedding.bin` + `control_scale.bin`。
- 加载逻辑：[core/streaming/utils.py:24-160](../core/streaming/utils.py#L24)（`load_teacher_into_pipe`，含多 PATH 兼容）。
- 保存逻辑：[rynnworld_teleop_trainer.py:394-401](../core/finetune/models/wan_i2v/rynnworld_teleop_trainer.py#L394)。

> **改造结论（adopt）**：control_patch_embedding + control_scale 的"零初始化旁路 + 可学习标量"是干净的注入范式，
> 三层输出改造时每一层可各挂一套 control 分支（或共享 CPE、分层 control_scale），直接复用此保存/加载格式。

---

## 2. 单 RGB denoising head 的位置 = 三层输出挂载点

- 当前 forward 输出走 Wan DiT 原生 `patch_embedding → blocks → proj_out → unpatchify`，是**单 RGB latent** 输出。
- 三层改造挂载点：`proj_out` / `unpatchify` 之前的 `hidden_states`（token 空间）或之后的 latent 空间。
  - **方案 A（VAE latent 通道拼接）**：把单一 16/48-ch latent 头扩成 background/scene · object/contact · robot/actor 三组通道，decoder 分层解。
  - **方案 B（token fusion + layer embedding）**：给三层各一个 layer embedding，token 级融合后共享 proj_out。
- 两方案的最小接口设计另立文档（本阶段不实现）。

> **改造结论（cherry-pick）**：Wan DiT 主干 + RoPE + condition_embedder 全部复用；仅替换 `patch_embedding`/`proj_out` 头部为三层版本。

---

## 3. Stage 2 Streaming 蒸馏（causal 复用点）

### 3.1 causal 主干
- `WanCausalTransformer3DModel` / `DynamicCache` / `WanStreamingPipeline`：[core/streaming/model.py](../core/streaming/model.py)、[core/streaming/cache.py](../core/streaming/cache.py)。
- 帧块生成 + 滑动 KV cache（sink frame 保留），`num_frame_per_block=3`、`num_max_frames=21`、`sink_size=1`。
- **硬依赖 `flash_attn`**（[model.py:48](../core/streaming/model.py#L48)），非 Hopper 无 FP8。

### 3.2 control_running_stats（streaming 控制归一化）
- 机制：[inference_streaming.py:357-388](../inference_streaming.py#L357)
  ```
  ctrl_norm = (ctrl - c_run_mean) / c_run_std * v_std + v_mean
  ```
  `control_running_stats.bin` = 训练期累积的 control latent per-channel mean/std；单样本部署用经验 `v_mean=0, v_std=0.8`。
- 仅 Causal checkpoint 带此文件（sha256 见 [upstream_baseline.md](upstream_baseline.md)）。

### 3.3 MSE warm-up + DMD 蒸馏训练入口
- 训练脚本：[core/streaming/train_distill.py](../core/streaming/train_distill.py)；`--mode mse`（warm-up）→ `--mode dmd`（4-step 对抗蒸馏，CausVid recipe，critic 从 MSE student 初始化）。

> **改造结论（reject / 暂缓）**：streaming 蒸馏是"实时化"正交能力，与三层输出解耦。三层改造应先在 Stage 1 bidirectional teacher 上做；
> streaming 蒸馏留到三层 teacher 收敛后再接。physics forcing 如何接入每层同理——先不假定有效，作为独立实验。

---

## 4. dataloader（reconstruction / 数据格式）
- `core/finetune/datasets/wan_dataset.py`：产出 `*_rgb.safetensors` video latent（见 `data/video_latents/`）。
- reconstruction smoke（VAE encode→decode）结果见 [reconstruction_smoke.md](reconstruction_smoke.md)。
