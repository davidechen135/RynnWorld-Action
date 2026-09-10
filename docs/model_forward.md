# Model Forward Details

`RynnWorld-Teleop` Stage 1 forward 的 tensor 流。已与代码核对（[rynnworld_teleop_trainer.py:158-193](../core/finetune/models/wan_i2v/rynnworld_teleop_trainer.py#L158)），
非推测。VAE latent 通道数为 **48**（实测 `/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/video_latents/*.safetensors`），非早期文档写的 16。

## Tensor 流

```
first-frame RGB [3,480,832]
  └── VAE Encode ──> img_latent [48, 1, 30, 52]

control video (hand-pose) [81,3,480,832]
  └── VAE Encode ──> control_video_latents [48, 21, 30, 52]

noisy video latent [48, 21, 30, 52]
  │
  ├── patch_embedding(x)                         # Wan DiT 原生 patch embed
  ├── control_patch_embedding(control_latent)    # 零初始化 Conv3d 旁路
  │      × control_scale (=0.1091, learnable)
  └── ADD:  h = patch_embedding(x) + control_scale * control_patch_embedding(control_latent)
        (control_type='add'/'add-plus'; 'concat' 则通道拼接后过 2x-in Conv3d)
              │
              flatten+transpose ──> tokens
              + RoPE + condition_embedder(timestep, text_emb[, img_emb])
              │
              └── DiT blocks ──> proj_out ──> unpatchify
                    └── VAE Decode ──> rollout [81, 3, 480, 832]
```

## 关键点
- **单 RGB denoising head**：`proj_out`/`unpatchify` 输出单一 RGB latent。三层改造的挂载点见 [reusable_modules.md](reusable_modules.md#2)。
- **控制注入**是"零初始化 Conv3d 旁路 + 可学习 control_scale"，训练起点不扰动 base。
- 时间维：81 RGB 帧 → 21 latent 帧（VAE 时间压缩 4×，含首帧）。
