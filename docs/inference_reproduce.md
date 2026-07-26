# Inference Reproduction Guide

以官方入口复现的记录。基线见 [upstream_baseline.md](upstream_baseline.md)。

## 官方入口（复现结论以此为准）

### SFT — `inference_user.py`（8-case 全跑通）
```bash
python inference_user.py \
  --image example/<case>/first_frame.png \
  --control_video example/<case>/control_video.mp4 \
  --text_embedding example/<case>/text_embedding.safetensors \
  --output outputs/repro_sft/<case> \
  --checkpoint pretrained/RynnWorld-Teleop \
  --mode sft --control_type add --seeds "42"
```
批量脚本：[scripts/repro_sft_all.sh](../scripts/repro_sft_all.sh)。加载值 `control_scale=0.1091`。

8-case 输出（全部 832×480 / 81 帧 / H.264，`outputs/repro_sft/<case>/generated_seed42.mp4`）：

| case | frames | size |
|---|---|---|
| assemble_jenga_001 | 81 | 175 KB |
| basic_fold_009 | 81 | 183 KB |
| basic_pick_place_000 | 81 | 192 KB |
| clean_surface_001 | 81 | 150 KB |
| clip_unclip_papers_006 | 81 | 156 KB |
| color_004 | 81 | 182 KB |
| flip_pages_008 | 81 | 129 KB |
| fold_unfold_paper_basic_008 | 81 | 140 KB |

### Streaming — `inference_streaming.py`（causal，✅ 已跑通）
```bash
python inference_streaming.py \
  --image example/basic_pick_place_000/first_frame.png \
  --control_video example/basic_pick_place_000/control_video.mp4 \
  --text_embedding example/basic_pick_place_000/text_embedding.safetensors \
  --checkpoint pretrained/RynnWorld-Teleop-Causal \
  --output outputs/repro_streaming/basic_pick_place_000 \
  --mode dmd --num_inference_steps 4 --seed 42
```
依赖 `flash_attn`（[core/streaming/model.py:48](../core/streaming/model.py#L48)），自动读取 `control_running_stats.bin`。

**结果**（`outputs/repro_streaming/basic_pick_place_000/generated_seed42.mp4`）：
- 832×480 / 81 帧 / H.264，328 KB，画面为连贯 ego-view（双手操作拼图）。
- causal DMD 4-step，`control_scale=0.1104`，control 归一化 `c_mean=-0.1465 c_std=0.4886 → N(0, 0.8)`。
- 推理耗时 **5.70s**（vs SFT bidirectional 50-step ~16.5s）。
- checkpoint overlay: `missing=0 unexpected=60`（60 个 unexpected 为 causal 专属 buffer，不影响加载）。

**环境补充**：`flash_attn==2.8.3`（预编译 wheel：`cu12torch2.8cxx11abiTRUE-cp312`，无 nvcc 环境下源码编译会卡住，改用 wheel）。

## 附加可视化工具（非官方入口，结论仅供参考）
`inference_benchmark.py`：自写封装，产出 rollout + trajectory + action curve + WandB。
注意 [inference_benchmark.py:174](../inference_benchmark.py#L174) 用 `control_video_latents.clone()` 初始化 latent，
与官方从纯噪声起步不同——数值上可能偏离官方 baseline，故复现结论一律以官方入口为准。

### Benchmark（H20 96G，SFT）
- 推理时间 ~16.5s（50 步）· ~4.91 FPS · ~203 ms/frame
- Peak GPU ~28.6 GB · CPU ~2.3 GB
