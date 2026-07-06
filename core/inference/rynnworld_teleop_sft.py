"""
Inference for RynnWorld-Teleop SFT pretrained model (text + image -> video, no control video).
Loads EMA weights from checkpoint and generates video using the standard WanImageToVideoPipeline.
"""
import json
import os
import random
import shutil
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from PIL import Image
from pathlib import Path
from safetensors.torch import load_file

from diffusers import WanImageToVideoPipeline, WanTransformer3DModel
from diffusers.utils import export_to_video
from termcolor import cprint


def safe_export_to_video(frames, output_path, fps=16):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        export_to_video(frames, tmp_path, fps=fps)
        shutil.move(tmp_path, output_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def load_ema_transformer(model_path: str, ema_weights_path: str, dtype=torch.bfloat16):
    """Load transformer with EMA weights from SFT checkpoint."""
    transformer = WanTransformer3DModel.from_pretrained(model_path, subfolder="transformer")

    ema_state = torch.load(ema_weights_path, map_location="cpu", weights_only=False)
    cprint(f"Loaded EMA state dict with {len(ema_state)} keys", "green")

    missing, unexpected = transformer.load_state_dict(ema_state, strict=False)
    if missing:
        cprint(f"Warning: {len(missing)} missing keys when loading EMA weights", "yellow")
    if unexpected:
        cprint(f"Warning: {len(unexpected)} unexpected keys when loading EMA weights", "yellow")

    transformer = transformer.to(dtype=dtype)
    return transformer


def generate_video_sft(
    checkpoint_path: str = None,
    output_path: str = "results/sft_pretrain",
    model_path: str = None,
    data_json: str = None,
    num_samples: int = 20,
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 42,
    use_ema: bool = True,
):
    device = torch.device("cuda")

    if model_path is None:
        model_path = os.environ.get("MODEL_PATH", "pretrained/Wan2.2-TI2V-5B-Diffusers")
    if data_json is None:
        data_json = os.environ.get("DATA_JSON", "example/example_cases.json")

    if checkpoint_path is not None:
        if use_ema:
            ema_path = os.path.join(checkpoint_path, "ema_weights.pt")
            if os.path.exists(ema_path):
                transformer = load_ema_transformer(model_path, ema_path, dtype=dtype)
                cprint(f"Loaded transformer with EMA weights from {ema_path}", "green")
            else:
                cprint(f"EMA weights not found at {ema_path}, using base model", "yellow")
                transformer = WanTransformer3DModel.from_pretrained(model_path, subfolder="transformer").to(dtype=dtype)
        else:
            raw_path = os.path.join(checkpoint_path, "pytorch_model", "mp_rank_00_model_states.pt")
            if os.path.exists(raw_path):
                cprint(f"Loading raw model weights from {raw_path}", "cyan")
                state = torch.load(raw_path, map_location="cpu", weights_only=False)
                transformer = WanTransformer3DModel.from_pretrained(model_path, subfolder="transformer")
                missing, unexpected = transformer.load_state_dict(state["module"], strict=False)
                if missing:
                    cprint(f"Warning: {len(missing)} missing keys", "yellow")
                if unexpected:
                    cprint(f"Warning: {len(unexpected)} unexpected keys", "yellow")
                transformer = transformer.to(dtype=dtype)
                cprint("Loaded transformer with raw (non-EMA) weights", "green")
            else:
                cprint(f"Raw weights not found at {raw_path}, using base model", "yellow")
                transformer = WanTransformer3DModel.from_pretrained(model_path, subfolder="transformer").to(dtype=dtype)
    else:
        cprint("No checkpoint specified, using base Wan2.2 model", "yellow")
        transformer = WanTransformer3DModel.from_pretrained(model_path, subfolder="transformer").to(dtype=dtype)

    pipe = WanImageToVideoPipeline.from_pretrained(
        model_path,
        transformer=transformer,
        torch_dtype=dtype,
    )
    pipe.enable_model_cpu_offload()

    os.makedirs(output_path, exist_ok=True)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    cprint(f"Loading data from {data_json} ...", "cyan")
    with open(data_json, "r") as f:
        all_samples = json.load(f)
    rng = random.Random(seed)
    selected = rng.sample(all_samples, min(num_samples, len(all_samples)))
    cprint(f"Selected {len(selected)} samples for inference", "green")

    for idx, sample in enumerate(selected):
        video_latent_path = sample["video_latent_path"]
        text_embedding_path = sample["text_embedding_path"]

        # Derive case name from path: e.g. .../play_reversi/2576_1_rgb.safetensors -> play_reversi_2576_1
        parts = Path(video_latent_path).parts
        task_name = parts[-2]
        file_stem = Path(video_latent_path).stem.replace("_rgb", "")
        case_name = f"{task_name}_{file_stem}"

        if not os.path.exists(video_latent_path):
            cprint(f"Skipping {case_name}: file not found at {video_latent_path}", "yellow")
            continue

        cprint(f"Processing: {case_name}", "cyan")

        cache_data = load_file(video_latent_path)
        img_latent = cache_data["img_latent"]  # [48, 1, H, W]
        video_latents = cache_data["video_latents"]  # [48, T, H, W]

        # Decode first frame from img_latent as PIL image
        img_latent_5d = img_latent.unsqueeze(0).to(device=device, dtype=pipe.vae.dtype)  # [1, 48, 1, H, W]
        latents_mean = (
            torch.tensor(pipe.vae.config.latents_mean)
            .view(1, pipe.vae.config.z_dim, 1, 1, 1)
            .to(img_latent_5d.device, img_latent_5d.dtype)
        )
        latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(1, pipe.vae.config.z_dim, 1, 1, 1).to(
            img_latent_5d.device, img_latent_5d.dtype
        )
        img_latent_denorm = img_latent_5d / latents_std + latents_mean
        with torch.no_grad():
            decoded_img = pipe.vae.decode(img_latent_denorm, return_dict=False)[0]  # [1, 3, 1, H, W]
        decoded_img = decoded_img[:, :, 0]  # [1, 3, H, W]
        decoded_img = (decoded_img.clamp(-1, 1) + 1) / 2 * 255
        decoded_img = decoded_img[0].permute(1, 2, 0).float().cpu().numpy().astype("uint8")
        pil_image = Image.fromarray(decoded_img)

        # Decode GT video for comparison
        video_latents_5d = video_latents.unsqueeze(0).to(device=device, dtype=pipe.vae.dtype)
        video_latents_denorm = video_latents_5d / latents_std + latents_mean
        with torch.no_grad():
            gt_video = pipe.vae.decode(video_latents_denorm, return_dict=False)[0]
        gt_video = pipe.video_processor.postprocess_video(gt_video, output_type="np")

        # Derive text prompt from text embedding filename
        prompt_text = Path(text_embedding_path).stem.replace("_", " ")
        cprint(f"  Prompt: {prompt_text}", "cyan")

        output_vid = pipe(
            image=pil_image,
            prompt=prompt_text,
            negative_prompt="",
            height=480,
            width=832,
            num_frames=81,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

        gen_frames = output_vid.frames[0]
        gt_frames = gt_video[0]

        gen_vid_path = os.path.join(output_path, f"{case_name}.mp4")
        gt_vid_path = os.path.join(output_path, f"{case_name}_gt.mp4")
        img_path = os.path.join(output_path, f"{case_name}_input.png")

        safe_export_to_video(gen_frames, gen_vid_path, fps=16)
        safe_export_to_video(gt_frames, gt_vid_path, fps=16)
        pil_image.save(img_path)

        cprint(f"  Saved: {gen_vid_path}", "green")

    cprint("\nAll inference tasks completed!", "green")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inference for RynnWorld-Teleop SFT pretrained model")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to SFT checkpoint directory (containing ema_weights.pt)")
    parser.add_argument("--output", type=str, default="results/sft_pretrain",
                        help="Output directory for generated videos")
    parser.add_argument("--model_path", type=str,
                        default=os.environ.get("MODEL_PATH", "pretrained/Wan2.2-TI2V-5B-Diffusers"))
    parser.add_argument("--data_json", type=str,
                        default=os.environ.get("DATA_JSON", "example/example_cases.json"),
                        help="Path to JSON file with video_latent_path and text_embedding_path entries")
    parser.add_argument("--num_samples", type=int, default=20,
                        help="Number of samples to randomly select for inference")
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_video_sft(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        model_path=args.model_path,
        data_json=args.data_json,
        num_samples=args.num_samples,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
    )
