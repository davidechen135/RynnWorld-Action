"""Dump the mid-frame of a decoded clip to PNG for visual reconstruction check."""
import torch, numpy as np, cv2
from safetensors.torch import load_file
from diffusers import AutoencoderKLWan

MODEL = "pretrained/Wan2.2-TI2V-5B-Diffusers"
device, dtype = "cuda", torch.bfloat16
vae = AutoencoderKLWan.from_pretrained(MODEL, subfolder="vae").to(device=device, dtype=dtype).eval()
lm = torch.tensor(vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
ls = torch.tensor(vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)

z = load_file("/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/video_latents/basic_pick_place_0_0_rgb.safetensors")["video_latents"].unsqueeze(0)
z = z.to(device).float() / ls + lm
with torch.no_grad():
    pix = vae.decode(z.to(dtype)).sample  # [1,3,T,H,W] in [-1,1]
pix = ((pix.float().clamp(-1, 1) + 1) / 2 * 255).byte().cpu().numpy()[0]  # [3,T,H,W]
for idx in (0, 40, 80):
    fr = pix[:, idx].transpose(1, 2, 0)  # HWC RGB
    cv2.imwrite(f"outputs/repro_recon/decoded_frame_{idx:03d}.png",
                cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
    print(f"wrote decoded_frame_{idx:03d}.png  shape={fr.shape}")
