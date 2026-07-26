"""Single-clip reconstruction smoke: decode a stored video latent to RGB, re-encode,
measure round-trip latent error. Verifies VAE + data format match the official pipeline.
"""
import os, sys, time, json
import torch
import numpy as np
from safetensors.torch import load_file
from diffusers import AutoencoderKLWan

MODEL = "pretrained/Wan2.2-TI2V-5B-Diffusers"
CLIPS = [
    "data/video_latents/basic_pick_place_0_0_rgb.safetensors",
    "data/video_latents/assemble_jenga_0_1_rgb.safetensors",
    "data/video_latents/clean_surface_0_1_rgb.safetensors",
]
device, dtype = "cuda", torch.bfloat16

vae = AutoencoderKLWan.from_pretrained(MODEL, subfolder="vae").to(device=device, dtype=dtype)
vae.eval()
sf = vae.config.scaling_factor if hasattr(vae.config, "scaling_factor") else None
lm = getattr(vae.config, "latents_mean", None)
ls = getattr(vae.config, "latents_std", None)
print(f"scaling_factor={sf}  has_mean={lm is not None} has_std={ls is not None}")

def denorm(z):
    z = z.to(device=device, dtype=torch.float32)
    if lm is not None and ls is not None:
        m = torch.tensor(lm, device=device).view(1, -1, 1, 1, 1)
        s = torch.tensor(ls, device=device).view(1, -1, 1, 1, 1)
        return z / s + m
    return z

def renorm(z):
    if lm is not None and ls is not None:
        m = torch.tensor(lm, device=device).view(1, -1, 1, 1, 1)
        s = torch.tensor(ls, device=device).view(1, -1, 1, 1, 1)
        return (z - m) * s
    return z

torch.cuda.reset_peak_memory_stats(device)
results = []
for path in CLIPS:
    name = os.path.basename(path).replace("_rgb.safetensors", "")
    z0 = load_file(path)["video_latents"].unsqueeze(0)  # [1,C,T,H,W]
    t0 = time.time()
    with torch.no_grad():
        pix = vae.decode(denorm(z0).to(dtype)).sample     # [1,3,Tp,Hp,Wp]
        z1 = vae.encode(pix).latent_dist.mode()
        z1 = renorm(z1)
    dt = time.time() - t0
    z0f, z1f = z0.to(device).float(), z1.float()
    # align shapes (encode may pad frames)
    T = min(z0f.shape[2], z1f.shape[2])
    z0c, z1c = z0f[:, :, :T], z1f[:, :, :T]
    mse = torch.mean((z0c - z1c) ** 2).item()
    cos = torch.nn.functional.cosine_similarity(
        z0c.flatten(), z1c.flatten(), dim=0).item()
    rel = (torch.norm(z0c - z1c) / torch.norm(z0c)).item()
    res = dict(clip=name, in_latent=list(z0.shape), decoded_pix=list(pix.shape),
               re_latent=list(z1.shape), roundtrip_mse=round(mse, 5),
               cosine=round(cos, 5), rel_l2=round(rel, 5), sec=round(dt, 2))
    print(json.dumps(res))
    results.append(res)

peak = torch.cuda.max_memory_allocated(device) / 1e9
summary = dict(model=MODEL, scaling_factor=sf, peak_gpu_gb=round(peak, 2),
               clips=results)
os.makedirs("outputs/repro_recon", exist_ok=True)
with open("outputs/repro_recon/reconstruction_smoke.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nPEAK_GPU_GB={peak:.2f}")
print("WROTE outputs/repro_recon/reconstruction_smoke.json")
