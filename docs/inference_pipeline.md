# Inference Pipeline Guide

This document describes the entry points, key components, and data flow of the `RynnWorld-Teleop` inference pipeline.

## 🚀 Entry Points
- **`inference_user.py`**: User-friendly SFT/LoRA inference script. Input: image + skeleton video -> Output: generated rollout video.
- **`inference_benchmark.py`**: Enhanced benchmarking and visual reporting tool.

## 📊 Pipeline Flow

```
[Inputs] image (RGB) + skeleton (MP4)
  │
  ├── VAE Encode: Image -> img_latent [16, 1, 30, 52]
  ├── VAE Encode: Skeleton -> control_video_latents [16, 21, 30, 52]
  └── Text Encode: T5 -> prompt_embeds [1, 512, 4096]
  │
[Pipeline Run] WanImagePipeline
  ├── Unet Denoising (UniPCMultistepScheduler)
  │    └── Inject control signal: noisy_latent + control_latent * control_scale
  └── Denoised Latent [16, 21, 30, 52]
  │
[Outputs]
  ├── rollout.mp4 & rollout.gif
  └── rgb/ (Keyframes)
```

## 📈 Experiment Tracking (Weights & Biases)
When running `inference_benchmark.py` with the `--wandb` flag, each run starts a new experiment tracking:
1. **Experiment Config**: Args parameters, model path, frame count, height, and width.
2. **Checkpoint Info**: Checkpoint folder, mode, control type, guidance scale, and loaded `control_scale` parameter.
3. **Rollout Videos**: `rollout.mp4` and `rollout.gif`.
4. **Input/Output Images**: Input first frame image and output rollout final frame.
5. **Action Trajectories**: Hand-pose action curves plot and 3D trajectory plot.
6. **Inference Performance**: FPS, average frame latency, and total inference time.
7. **System Resources**: Peak GPU memory usage and CPU memory footprint.
