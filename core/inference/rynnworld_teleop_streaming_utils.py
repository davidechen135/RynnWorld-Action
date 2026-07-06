"""
rynnworld_teleop_streaming_utils.py — pure IO / VAE encoding / decoding helpers used by
inference_streaming.py. Extracted here to keep the top-level entry point
lean; no module-level state, no argparse, no side effects.
"""
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import imageio.v2 as imageio
import cv2
import decord
from safetensors.torch import load_file


def read_image(path, height, width):
    img = imageio.imread(path)
    if img.shape[-1] == 4:
        img = img[..., :3]
    if img.shape[0] != height or img.shape[1] != width:
        img = cv2.resize(img, (width, height))
    return img


def read_control_video(path, num_frames, height, width):
    """Read mp4 control video, sample/interpolate to exactly num_frames at target resolution."""
    reader = decord.VideoReader(uri=str(path), width=width, height=height)
    total = len(reader)
    if total < num_frames:
        frames = torch.from_numpy(reader.get_batch(range(total)).asnumpy()).float()
        frames = frames.permute(3, 0, 1, 2)
        frames = F.interpolate(
            frames.unsqueeze(0),
            size=(num_frames, height, width),
            mode='trilinear',
            align_corners=False,
        ).squeeze(0).permute(1, 2, 3, 0)
        return frames.byte().numpy()
    elif total <= int(1.5 * num_frames):
        indices = list(range(0, total, max(1, total // num_frames)))[:num_frames]
        return reader.get_batch(indices).asnumpy()
    else:
        return reader.get_batch(list(range(num_frames))).asnumpy()


def encode_image_to_latent(vae, image_np, device, dtype):
    """image_np: [H, W, C] uint8 -> img_latent [C_z, 1, H', W']"""
    tensor = torch.from_numpy(image_np).float().permute(2, 0, 1) / 255.0 * 2.0 - 1.0
    img_input = tensor.unsqueeze(0).unsqueeze(2).to(device=device, dtype=dtype)
    latents_mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(device=device, dtype=dtype)
    latents_std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(device=device, dtype=dtype)
    with torch.no_grad():
        latent = vae.encode(img_input).latent_dist.mode()
        latent = (latent - latents_mean) / latents_std
    return latent.squeeze(0).cpu().float()


def encode_video_to_latent(vae, video_np, device, dtype):
    """video_np: [F, H, W, C] uint8 -> latent [C_z, F', H', W']"""
    tensor = torch.from_numpy(video_np).float() / 255.0 * 2.0 - 1.0
    video_input = tensor.permute(3, 0, 1, 2).unsqueeze(0).to(device=device, dtype=dtype)
    latents_mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(device=device, dtype=dtype)
    latents_std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(device=device, dtype=dtype)
    with torch.no_grad():
        latent = vae.encode(video_input).latent_dist.mode()
        latent = (latent - latents_mean) / latents_std
    return latent.squeeze(0).cpu().float()


def _decode_latent_to_uint8(vae, lat_5d, device, dtype):
    """Decode latent to uint8 video frames."""
    z_dim = vae.config.z_dim
    v_dtype = next(vae.parameters()).dtype
    lat = lat_5d.to(device=device, dtype=v_dtype)
    lm = torch.tensor(vae.config.latents_mean, device=device, dtype=v_dtype).view(1, z_dim, 1, 1, 1)
    ls = torch.tensor(vae.config.latents_std, device=device, dtype=v_dtype).view(1, z_dim, 1, 1, 1)
    lat = lat * ls + lm
    with torch.no_grad():
        vid = vae.decode(lat, return_dict=False)[0]
    vid = ((vid.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    return vid[0].permute(1, 2, 3, 0).cpu().numpy()


def _write_mp4(frames_uint8, out_path, fps=16):
    """Write uint8 frames to mp4."""
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    tmp_path = f"/tmp/_infer_video_{os.getpid()}_{os.path.basename(out_path)}"
    with imageio.get_writer(tmp_path, fps=fps) as w:
        for f in frames_uint8:
            w.append_data(f)
    shutil.move(tmp_path, out_path)


def _make_time_strip(frames_uint8, n_show=None):
    """Create horizontal time-strip from video frames."""
    F_total = frames_uint8.shape[0]
    if n_show is None or n_show >= F_total:
        idx = list(range(F_total))
    else:
        idx = np.linspace(0, F_total - 1, n_show).astype(int).tolist()
    return np.concatenate([frames_uint8[i] for i in idx], axis=1)


def _dataset_group_of(p: str) -> str:
    """Infer dataset group from path."""
    if "/tianji_wuji/" in p:
        return "tianji_wuji"
    if "/video_latents/" not in p:
        return "other"
    seg = p.split("/video_latents/", 1)[1].split("/", 1)[0]
    return seg


def _case_name_of(item: dict, sample_idx: int) -> str:
    """Generate filesystem-safe case name."""
    p = item["video_latent_path"]
    group = _dataset_group_of(p)
    parts = Path(p).parts
    task = parts[-2] if len(parts) >= 2 else "unk"
    stem = Path(p).stem.replace("_rgb", "")
    name = f"{group}__{task}__{stem}"
    name = name.replace("/", "_").replace(" ", "_")
    return f"{sample_idx:08d}__{name}"


def _safetensors_load_robust(path, max_retries=3, backoff_initial=0.5):
    """Load safetensors with retry for FUSE/OSS tolerance."""
    from safetensors import SafetensorError
    last_exc = None
    for attempt in range(max_retries):
        try:
            return load_file(path)
        except (FileNotFoundError, OSError, RuntimeError, SafetensorError) as e:
            last_exc = e
            if attempt < max_retries - 1:
                sleep_s = backoff_initial * (2 ** attempt)
                time.sleep(sleep_s)
    raise last_exc
