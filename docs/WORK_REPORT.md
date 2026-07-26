# 工作结果报告 — RynnWorld-Teleop 纯复现阶段

日期：2026-07-26 · 基线 SHA `503333b`（= 官方 `alibaba-damo-academy/RynnWorld-Teleop` main HEAD）
硬件：2× NVIDIA H20 96G（等价 2×A100） · Base 模型：Wan2.2-TI2V-5B-Diffusers

目标：**锁定版本、跑通推理/训练 Smoke Test、记录基线性能与 VBench 跑分**，证明已完全掌握官方基线，
具备后续申请 32×A800 微调与三层架构改造的条件。

---

## 一句话总结
纯复现阶段 **4 项核心交付物全部完成**：① 复现文档齐全；② SFT 8/8 + Streaming 1/1 跑通；
③ 推理+训练性能/显存基线表已记录（含双卡 SFT 训练 smoke 无 OOM）；④ VBench 单帧质量基线已跑分。

---

## ① GitHub 交付文件与文档 ✅

| 内容 | 文件 |
|---|---|
| 复现步骤 / 命令 / 依赖版本 | [inference_reproduce.md](inference_reproduce.md), [env/](env/) |
| git SHA / 权重路径 / checkpoint sha256 / seeds | [upstream_baseline.md](upstream_baseline.md) |
| 可复用模块梳理（control injection / CPE / control_scale / causal cache / MSE-DMD 入口） | [reusable_modules.md](reusable_modules.md) |
| 模型 forward / pipeline / 数据流 | [model_forward.md](model_forward.md), [inference_pipeline.md](inference_pipeline.md) |
| 输入输出可视化 | `outputs/repro_sft/contact_sheet.png`（8-case 末帧）, `outputs/repro_streaming/`, `outputs/repro_recon/` |

- 环境：Python 3.12.3 · torch 2.8.0+cu128 · diffusers 0.39.0 · transformers 5.13.1 · flash_attn 2.8.3
- seeds：默认 42（同 seed 同 case 字节级可复现）

## ② 最小样例跑通证明 ✅（要求 ≥1 SFT + ≥1 Streaming）

- **SFT 8/8**（官方 `inference_user.py`，checkpoint `RynnWorld-Teleop`）：832×480/81帧 H.264，8 任务末帧均任务相符。
- **Streaming 1/1**（官方 `inference_streaming.py`，checkpoint `RynnWorld-Teleop-Causal`，DMD 4-step）：连贯 ego-view，`control_scale=0.1104`。
- flash_attn 依赖：本机无 nvcc，源码编译卡住 → 改用预编译 wheel（`cu12torch2.8cxx11abiTRUE-cp312`，经 `ghfast.top` 镜像）。

## ③ 运行性能与显存基线表 ✅

### 推理（SFT，H20，单卡，disable_cpu_offload）
| 指标 | 值 |
|---|---|
| Tensor: video/control latent | `[48,21,30,52]` · img `[48,1,30,52]` · text `[512,4096]` |
| 输出 | `[3,81,480,832]` RGB |
| 推理时间（50 步） | ~16.5 s · ~4.9 FPS · ~203 ms/frame |
| VRAM Peak | ~28.6 GB |

### 推理（Streaming causal DMD 4-step）
| 指标 | 值 |
|---|---|
| 推理时间 | **5.70 s**（vs SFT 16.5 s） |
| control 归一化 | `c_mean=-0.147 c_std=0.489 → N(0,0.8)` |

### 训练 Smoke（SFT，2×H20，DeepSpeed ZeRO-2 offload，3 clips，20 步）✅ 无 OOM 无报错
| 指标 | 值 |
|---|---|
| Loss（step 1→10→20） | 0.372 → 0.159 → 0.115（下降，健康） |
| Grad Norm | 6.21 → 0.24（稳定） |
| **VRAM Peak（per-GPU）** | **~18.4–20.4 GB**（稳态 alloc 9.5 GB） |
| 稳态速度 | ~10–13 s/step（81×480×832，bf16，bs=1） |
| checkpoint/EMA 保存 | ✅ `training/smoke_sft_2gpu/checkpoint-20/` |

详情：[training_smoke.md](training_smoke.md) · reconstruction smoke：[reconstruction_smoke.md](reconstruction_smoke.md)

## ④ 视频质量基线评估（VBench）✅

- 指标：**MUSIQ-KonIQ**（pyiqa，VBench `imaging_quality` 维度后端）= 单帧视频质量，越高越好。
- **SFT 8-case 均值 54.04**（区间 46.6–61.1）· **Streaming 45.16**（4-step 蒸馏，单帧质量略降符合预期）。
- 详情/范围说明：[vbench_baseline.md](vbench_baseline.md)（VBench 全维度依赖与复现环境冲突，采同源单维度离线评测）。

---

## 复现阶段结论
- ✅ 版本锁定、推理/训练 smoke 跑通、性能+显存+VBench 基线全部记录 → **基线已完全掌握**。
- ✅ 训练链路在 2×A100 等价硬件双卡 ZeRO-2 offload 下单卡峰值仅 ~20 GB，为 32×A800 全量微调提供显存/速度参考。
- 可复用模块与三层挂载点已梳理（[reusable_modules.md](reusable_modules.md)），具备进入改造阶段的条件。

## 下一步（改造阶段，非本次范围）
- [ ] 三层 decoder/latent 两种最小接口（A 通道拼接 / B token fusion + layer embedding）+ tensor/loss/显存/速度 ablation。
- [ ] 逐层指标（mask/recomposition/temporal/leakage/motion、layer IoU/Boundary-F/utility）+ physics forcing 逐层消融。
- [ ] 申请 32×A800 全量微调。
