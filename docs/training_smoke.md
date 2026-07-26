# 训练侧基线 — 最小 SFT Smoke Test（2×H20）

验证官方训练链路在双卡下**无 OOM、无报错**，并记录 Loss / 显存 / 速度 / Tensor Shape。
脚本：[scripts/smoke_sft_2gpu.sh](../scripts/smoke_sft_2gpu.sh)。日志：`training/smoke_sft_2gpu.log`。

## 配置
- 入口：`finetune.py --training_type sft --model_name rynnworld_teleop`（官方入口）
- 启动：`accelerate launch --config_file configs_acc/2gpu.yaml`（**DeepSpeed ZeRO-2 + CPU offload**，2 GPU）
- 数据：`data/sample_data.json`（**3 个共享 clip**：jigsaw_puzzle / basic_pick_place / blowdry_hair，预存 latent + text embedding）
- 分辨率 `81x480x832`，batch_size=1，grad_accum=1，bf16，`--train_steps 20`，seed=42，`control_type=add`
- EMA 关闭（`ema_start_step` 设很大，避免 smoke 干扰）

## 结果：✅ 双卡跑通，无 OOM 无报错（exit 0）

### Tensor Shape
| 张量 | shape |
|---|---|
| video_latent（输入） | `[48, 21, 30, 52]`（mean=-0.092, std=0.801） |
| img_latent | `[48, 1, 30, 52]` |
| control_video_latent | `[48, 21, 30, 52]` |
| text_embedding | `[512, 4096]` |

### Loss 曲线（20 步，下降 = 在学）
| step | loss | grad_norm | lr |
|---|---|---|---|
| 1 | 0.372 | 6.21 | 2.0e-7 |
| 10 | 0.159 | 0.47 | 2.0e-6 |
| 20 | 0.115 | 0.24 | 4.0e-6 |

逐步 loss：`0.372, 0.507, 0.204, 0.564, 0.208, 0.189, 0.337, 0.344, 0.166, 0.159, 0.171, 0.166, 0.286, ...`
（20 步内从 ~0.37 收敛区间下探到 ~0.12，grad_norm 由 6.2 稳定到 0.2，训练链路数值健康）

### 显存（per-GPU，ZeRO-2 offload）
| 指标 | 值 |
|---|---|
| max_memory_allocated | 17.77 → 18.98 GB |
| max_memory_reserved（**VRAM Peak**） | **18.45 → 20.42 GB** |
| memory_allocated（稳态） | 9.50 GB |

> ZeRO-2 + CPU offload 下单卡峰值 ~20 GB，远低于 H20 96G；A100 40G/80G 同样宽裕。

### 速度
- 稳态 **~10–13 s/step**（2 GPU，81×480×832，bf16）。
- 日志显示 `38.44s/it` 平均是把 step-20 的 DeepSpeed checkpoint 保存（~5 min，5B 优化器状态）摊进去了，非纯训练速度。

## 结论
- 官方 SFT 训练脚本在 **2×H20（等价 2×A100）** 双卡 ZeRO-2 offload 下可直接跑通，显存/数值均健康。
- checkpoint（`training/smoke_sft_2gpu/checkpoint-20/`）+ EMA + control_patch_embedding 正常保存。
- 为申请 32×A800 全量微调提供了显存/速度参考基线。
