# VBench 视频质量基线

对官方基线生成的 rollout 评单帧视频质量，作为后续改造对比的 baseline。
脚本：[scripts/vbench_imaging_quality.py](../scripts/vbench_imaging_quality.py)。原始结果：`outputs/vbench/imaging_quality.json`。

## 方法与范围（诚实记录）
- 指标：**MUSIQ-KonIQ**（via `pyiqa`），即 VBench `imaging_quality` 维度的评分后端——直接对应任务要求的"**单帧视频质量**"。
- 分数区间 ~0–100，**越高越好**。每个视频每 8 帧采一帧（81 帧 → 11 帧）取均值。
- 环境：VBench 0.1.5 装了但其**完整 pipeline 依赖冲突**（pin `transformers==4.33.2`/`numpy<2`，与复现环境的 5.13.1/2.x 冲突），
  故采用 VBench 同源的 pyiqa-MUSIQ 单维度评测（离线权重经 `hf-mirror.com` 下载），**未跑 VBench 全维度**（其余维度需额外权重与降级依赖）。
- 这满足验收 ④"单帧视频质量 baseline"；若需全 VBench 维度，建议在独立 conda 环境跑以免破坏复现环境。

## 结果

| mode | case | MUSIQ ↑ |
|---|---|---|
| SFT | color_004 | 60.42 |
| SFT | fold_unfold_paper_basic_008 | 61.07 |
| SFT | clip_unclip_papers_006 | 55.05 |
| SFT | assemble_jenga_001 | 54.59 |
| SFT | basic_fold_009 | 52.02 |
| SFT | flip_pages_008 | 51.81 |
| SFT | clean_surface_001 | 50.71 |
| SFT | basic_pick_place_000 | 46.65 |
| Streaming | basic_pick_place_000 | 45.16 |

**均值**：SFT **54.04** · Streaming **45.16**

## 观察
- SFT 8-case MUSIQ 均值 54.0，区间 46.6–61.1，属正常自然视频质量区间。
- Streaming（causal DMD 4-step）单帧质量低于 SFT（45.2 vs 46.6，同 case 对比）——符合预期：4-step 蒸馏换取实时性（5.70s vs 16.5s），单帧质量略降。
- 这组分数是后续三层改造的**整帧质量 baseline**：改造后重组整帧的 MUSIQ 应不显著低于此。
