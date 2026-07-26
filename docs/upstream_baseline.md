# Upstream Baseline (frozen)

复现基线冻结记录。所有后续复现均以此为准。

## Git

| 项 | 值 |
|---|---|
| 官方 repo | `alibaba-damo-academy/RynnWorld-Teleop` |
| Pinned SHA | `503333bd4cd42e79d680f26298ea5e5625831dc6` |
| Commit | `Update links in README.md` (2026-07-08 11:43:03 +0800) |
| 本地 HEAD | `503333b`（与官方 main HEAD 一致，无 tracked 文件改动） |

> 注：本地 `origin` remote 走 `ghfast.top` 镜像，但 commit SHA 与官方 `github.com/alibaba-damo-academy/RynnWorld-Teleop` main HEAD 完全一致，基线干净。

## 环境

来源见 `docs/env/versions.txt` 与 `docs/env/requirements.lock.txt`。

| 组件 | 版本 |
|---|---|
| Python | 3.12.3 |
| torch | 2.8.0+cu128 |
| CUDA | 12.8 |
| diffusers | 0.39.0 |
| transformers | 5.13.1 |
| GPU | NVIDIA H20 (96G) |

## Checkpoints（sha256）

### pretrained/RynnWorld-Teleop（Stage 1 SFT teacher）
```
25601ec1...  control_patch_embedding.bin
b9ce95fb...  control_scale.bin
88ea4433...  ema_weights.bin
```

### pretrained/RynnWorld-Teleop-Causal（Stage 2 streaming student）
```
4254b817...  control_patch_embedding.bin
43cfc97c...  control_running_stats.bin   # streaming 控制归一化统计量
cb83d495...  control_scale.bin
605c9e8c...  ema_weights.bin
```

### pretrained/Wan2.2-TI2V-5B-Diffusers（base，diffusers 格式）
含 transformer / vae / text_encoder(T5) / tokenizer / scheduler。

## Seeds

默认 `42`（官方 `inference_user.py` / `inference_streaming.py` 默认值）。SFT 可复现性已验证：同 case 同 seed 字节级一致。
