# 真实机器人 ego 域 LoRA 微调 — AgiBot-357（越过 zero-shot 边界）

日期：2026-07-30 · 分支 `three-layer-transform` · 硬件：2× NVIDIA H20 96G
Base：Wan2.2-TI2V-5B-Diffusers · checkpoint 源：官方 SFT `RynnWorld-Teleop`

目标：把 zero-shot 跨域实验（[docs/agibot_crossdomain.md](agibot_crossdomain.md)、[docs/action_zeroshot.md](action_zeroshot.md)）
的结论——"要稳定跨域，需在目标域数据上继续训练"——真正落地：**在真实智元双臂机器人 ego 数据上做 LoRA 微调**，
让冻结世界模型适配新本体，而不再是分布对齐注入。

---

## 一句话总结

用**已有的 1 条真实洗碗长 episode**（AgiBot task 357 / episode 648751，4683 帧，0 新增下载）切成 **40 个 81 帧 clip**，
做成官方三件套，跑通 **rank-32 LoRA 微调**（2×H20，无 OOM，18 步 8min）。
共两轮：**v1** 用手工造的 action latent 当 control，**v2** 改成**真实末端轨迹渲染的骨架视频 → 官方 VAE 编码**，
并修正了 v1 暴露的 EMA / 超参缺陷。
**诚实结论**：v2 修掉了 v1 的发散退化（末帧浑浊、后段 motion 2.09→3.83），但**没有**做到贴近真实动作幅度
——它现在只有 GT 的 35%，即从"过量发散"变成"过于保守"。**单 episode / 单 task 的数据量仍是主要限制。**

---

## v1 暴露的两个实现缺陷（v2 已修）

**1. `ema_final` 是空适配器 —— v1 所有"微调后"推理其实等于原始 SFT。**
`EMA.__init__` 在训练开始时就 clone 参数进 shadow（此时 LoRA B 按惯例是零初始化），
而 `--ema_start_step 40` 大于总步数 24，`EMA.update()` **一次都没执行**，
结尾 `apply_shadow()` 又把这份初始 shadow 写回模型导出。

| checkpoint | LoRA A absmax | LoRA B absmax | B 非零张量 |
|---|---|---|---|
| v1 `ema_final` | 0.01807 | **0.000000** | **0/180** |
| v1 `checkpoint-24` | 0.01904 | 0.001060 | 180/180 |
| **v2 `ema_final`** | 0.01807 | 0.000083 | **180/180** ✅ |
| **v2 `checkpoint-18`** | 0.01868 | **0.000755** | 180/180 |

v2 的 EMA 确实跑了（B 全部非零）。但 `ema_decay 0.99` × 仅 12 次更新只走到训练量的 ~11%
（B absmax 0.000083 vs 0.000755），**评估仍应用 `checkpoint-18`**。

**2. v1 loss 是 V 形，不是单调下降。** v1：**0.389 → 谷底 0.117（~step21）→ 回升 0.282（step24）**，
40 clip 训 8 epoch，每条被看 8 遍，末段过拟合。此前文档只引 step 1→10 两点，掩盖了后段反弹。
另外 warmup 20/24 步、cosine 周期按更长训练设定（末尾 lr 反而抬头），超参与实际长度不匹配。

修正后的 v2（[scripts/agibot_lora_2gpu.sh](../scripts/agibot_lora_2gpu.sh)）：
epoch 8→6、`ema_start_step` 40→6、`ema_decay` 0.999→0.99、warmup 20→3、`checkpointing_steps` 40→6。
**v2 反弹消失**：0.389 → 谷底 0.139 → 收尾 0.155（平），见下。

---

## Control 信号修正（v2 的核心改动）

**问题**：v1 的 `control_video_latents` 是手工合成的（`ActionEncoderV2`：高斯团 + 正弦载波，
再按通道 z-score 对齐到真实 control latent 的 μ/σ）。一阶统计完全对得上，但
**逐帧时间 Δ 只有真实骨架 control 的 25%**（40 clip 均值 0.0398 vs 0.1587）。
Conv3d control head 靠时空梯度提取运动，喂进幅度偏小的 latent，信号就淹没在噪声里。

**修正**：[scripts/agibot_skeleton_render.py](../scripts/agibot_skeleton_render.py) —
把 h5 里真实的 `action/end/position`（末端位姿，机器人基座米制）+ `end/orientation`（四元数）+
`effector/position`（夹爪开合 0–1）**画成官方视觉语言的骨架视频**（白底 247/249/246，
左臂蓝右臂红，每手 21 关键点 = 腕 + 5 指 × 4 指节），再走**官方 VAE encode** —— 与真实 hand-pose control
完全同一条路径。开合夹爪 = 张开的手扇/握紧的拳，手尺寸按官方 control 的 bbox（83×75 / 95×111 px）标定。

**相机是拟合的，不是标定的（重要 caveat）**：AgiBot sample **不含任何内外参**。
做法：160×120 解码 → 时间中值背景 → |frame−bg| 92 分位阈值 → 左右半图各取质心 →
最小二乘拟合 **(x,y,z) 的二次多项式** 到 (u,v)。单条连续轨迹下 DLT 透视解退化（R² 跌到 −30/−216），
二次式则达到 arm0 R² +0.72/+0.77（中值误差 7.2px）、arm1 +0.57/+0.68（8.4px）。
**这是一个 fitted proxy，不是标定** —— 只保证 control 跟着手臂动，这正是 control head 消费的性质。

**效果**（[scripts/plot_control_fix.py](../scripts/plot_control_fix.py)）：

| | v1 action encoder | **v2 skeleton render** | 真实官方 control |
|---|---|---|---|
| 控制 latent 时间 Δ（40 clip 均值） | 0.0398（25%） | **0.0745（47%）** | 0.1587（100%） |
| 最慢 clip → 最快 clip | 0.026 → 0.055 | 0.064 → 0.093 | — |
| 与手臂像素速度相关性 | r = 0.70 | r = 0.70 | — |

- [outputs/agibot_control_fix/control_pixels.png](../outputs/agibot_control_fix/control_pixels.png) — 渲染出的骨架 control 长什么样
- [outputs/agibot_control_fix/control_delta.png](../outputs/agibot_control_fix/control_delta.png) — 逐 clip 时间 Δ，v1 vs v2 vs 真实参考线

注意：**两者与手臂速度的相关性一样（r≈0.70）**，v2 的收益在**幅度**（25%→47%），不在响应性。
离真实 control 仍差一半，多半是拟合相机把 3D 运动压扁 + 21 点手模型比真手简化所致。

---

## 数据准备（路径 A：单长 episode 切片）

脚本：[scripts/agibot_finetune_prep.py](../scripts/agibot_finetune_prep.py)（`--control-mode skeleton` 为默认，
`action_encoder` 保留供对比）。

- 源：`observations/357/648751/videos/head_color.mp4`（AV1，640×480，30fps，4683 帧）+ `proprio_stats.h5`。
- AV1 解码：decord/OpenCV 均无法读，用 **ffmpeg libdav1d rawvideo pipe** 按帧区间抽段 → resize 到官方 832×480。
- 切片：stride=81 不重叠 → 40 个 clip（`--max-clips 40`），覆盖帧 0–3239。
- **留出集**：`--skip-clips 40 --max-clips 17` 再切帧 3240–4616 的 17 个 clip。
  验证过训练/留出**帧级交集 = 0**，留出首帧与 40 个训练首帧**重复 0 对**。总覆盖 4617/4683 帧（99%）。
- 每 clip 编码成官方三件套（与 `/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/sample_data.json` 逐字段同构）：

| 键 | 来源 | shape | 说明 |
|---|---|---|---|
| `video_latents` | 真实 head_color 81 帧 → VAE | `[48,21,30,52]` | **去噪目标**（σ 0.84–1.04） |
| `control_video_latents` | **末端位姿 → 骨架视频 → 官方 VAE**（v2） | `[48,21,30,52]` | v1 为 ActionEncoderV2 手工造 |
| `img_latent` | clip 首帧 → VAE | `[48,1,30,52]` | I2V 条件 |
| text_embedding | task 文本 → UMT5 | `[226,4096]` | 共享，`/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/text_embeddings/agibot_task357.safetensors` |

产物：v1 `/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/agibot_357{,.json}`，v2 `/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/agibot_357_skel{,.json}` + `/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/agibot_357_skel_heldout/`
+ 拟合相机 `/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/agibot_357_camera.npy`。
**关键区别于 zero-shot**：zero-shot 的 `video_latents` 是用 control 顶替的（只 encode 首帧），这里是**真实 81 帧视频 latent**——
这才让 LoRA 有真实去噪目标可学，是"越过不微调边界"的实质。

## LoRA 微调（v2）

脚本：[scripts/agibot_lora_2gpu.sh](../scripts/agibot_lora_2gpu.sh)。官方 `finetune.py --training_type lora`。

- LoRA rank=32 / alpha=32，target = `attn1.{q,k,v,out.0}` + `ffn.net.{0.proj,2}`（30 blocks）+ 联合训练 `control_patch_embedding`。
- 6 epoch × 40 clip ÷ grad_accum 8 ÷ 2 GPU = **18 优化步**；lr 1e-4（control 5e-5），warmup 3 + cosine，EMA from step 6。

### 结果 ✅ 无 OOM 无报错（exit 0）

| 指标 | v2 | （v1 对照） |
|---|---|---|
| Loss（8-batch 滑动均值） | 0.389 → 谷底 **0.139** → 收尾 **0.155**（平，无反弹） | 0.389 → 0.117 → **回升 0.282** |
| Grad Norm | 0.87 → peak **1.13 @step3** → 0.098 | 0.86 → peak 3.09 @step4 → 0.019 |
| lr | peak 9.97e-05 @step3，其后 cosine 衰减到 1.2e-05 | 末尾仍在抬头（周期错配） |
| **VRAM Peak（per-GPU）** | **15.95 GB** allocated / 17.05 GB reserved | 15.73 / 16.56 |
| 速度 | 27.2 s/step，总 **8min08s**（18 步） | ~24–30 s/step，24 步 |
| checkpoint | `checkpoint-18/`（**用这个**）+ `ema_final/`（仅 11% 收敛，勿用） | `checkpoint-24/` |
| `control_scale` | 0.1091 → 0.1001（6/12/18 步均已收敛在此） | 0.1091 → 0.1001 |

曲线：[outputs/agibot_lora_357_v2/training_curves.png](../outputs/agibot_lora_357_v2/training_curves.png)
（脚本 [scripts/plot_lora_curves.py](../scripts/plot_lora_curves.py)）。

## 留出集评估（干净对照）

首帧取留出 clip（帧 3969–4049，训练从未见过），脚本
[scripts/agibot_heldout_eval.py](../scripts/agibot_heldout_eval.py)（内含 assert，clip 若在训练集直接报错）。
只换权重，首帧/control/text/seed 42/guidance 1.0 全同，50 步去噪。
**所有 motion 数值统一在 uint8 mp4 帧上重算**（不是 pipeline 内部 float），四个变体可比。

| 指标（seed42, 81 帧） | **真实 GT** | zero-shot（新 control） | v1 FT `ckpt-24`（旧 control） | **v2 FT `ckpt-18`（骨架 control）** |
|---|---|---|---|---|
| 逐帧 Δ 均值（motion energy） | **1.21** | 0.76 | 2.79 | **0.43** |
| 占 GT 比例 | 100% | 63% | 231% | **35%** |
| 逐帧 Δ 早 → 晚 | 1.13 → 1.47 | 1.12 → 0.57 | 2.17 → **3.86**（发散） | 0.62 → 0.27 |
| 末⅓/首⅓ 细节比（坍缩指标） | 0.987 | 0.986 | 0.992 | 1.016 |

可视化（脚本 [scripts/plot_heldout_compare.py](../scripts/plot_heldout_compare.py)）：
- [outputs/agibot_heldout_v2skel/heldout_grid.png](../outputs/agibot_heldout_v2skel/heldout_grid.png) — 四行帧网格，GT 在最上
- [outputs/agibot_heldout_v2skel/heldout_motion.png](../outputs/agibot_heldout_v2skel/heldout_motion.png) — 逐帧 motion energy 曲线

## 诚实结论（重要）

1. **v1 的"疯狂运动"很大一部分不是权重的锅，是 control 输入的锅。** 同样是 zero-shot（权重完全没动），
   换成修正后的 control 之后 motion 从 **208% → 63% of GT**。这意味着 v1 那张对比表里
   zero-shot 与 FT 的差距被坏 control 放大了，不能直接归因于微调。
2. **v2 修掉了退化，但没有修好动作幅度。** v1 `ckpt-24` 末帧（frame 80）画面明显浑浊、后段 Δ 发散到 3.86；
   v2 这两个现象都消失（坍缩比 1.016，画面干净）。但 v2 的 motion 只有 GT 的 **35%** ——
   **从过量发散变成了过于保守**，仍然不是"更接近真实"。
3. **control 幅度只到真实的 47% 是最可能的剩余瓶颈**：拟合相机（R²~0.7，7–8px）把 3D 运动压扁，
   21 点手模型也比真手简化。要继续推进，应先提高 control 保真度而不是加训练步数。
4. **坍缩指标在 357 上无区分度**：四者都在 0.986–1.016，包括真实 GT。再次印证 357（zero-shot 唯一时序稳住的
   纹理丰富 case，见 [agibot_crossdomain.md](agibot_crossdomain.md)）不是合适的评估 case。
5. **主要限制仍是数据太少**：单 episode / 单 task / 40 clip，同一厨房同一机位，共享 1 个 text embedding。
   18 个优化步也很难指望学到本体动力学。

## 下一步

**近端（0 下载）**：盘上另有 task 362（叠短裤）、392（洗瓶）各 1 条 episode，
三 task 混合可把场景数从 1 提到 3，且正好覆盖 zero-shot 会崩的 case
——那才是坍缩指标有区分度的地方。

**路径 B（需决策）**：补下 1 个会崩 task 的 observation tar（~48GB）**+** proprio tar（~48GB）≈ 97GB，
磁盘（剩 488G）需边解边删。**建议先做完近端 0 下载的部分再决定。**

## 复现命令

```bash
# 1. 数据准备（0 新增下载；默认 --control-mode skeleton）
python scripts/agibot_finetune_prep.py --max-clips 40 --output /mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/agibot_357_skel
# 1b. 留出集（与训练帧完全不重叠）
python scripts/agibot_finetune_prep.py --skip-clips 40 --max-clips 17 \
  --output /mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/agibot_357_skel_heldout

# 2. LoRA 微调 v2（2×H20，18 步 ~8min）
bash scripts/agibot_lora_2gpu.sh

# 3. 留出集评估：zero-shot vs 微调（用 checkpoint-18，不是 ema_final）
python scripts/agibot_heldout_eval.py \
  --clip /mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/agibot_357_skel_heldout/clip_003969.safetensors \
  --lora-checkpoint training/agibot_357_lora_v2_skel/checkpoint-18 \
  --output outputs/agibot_heldout_v2skel

# 4. 可视化
python scripts/plot_lora_curves.py        # 训练曲线
python scripts/plot_heldout_compare.py    # 留出集帧网格 + motion 曲线
python scripts/plot_control_fix.py        # control 修正前后对比
```
