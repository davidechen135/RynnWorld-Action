"""
Streaming inference script for RynnWorld-Teleop.

Supports:
- Single sample inference (image + control video)
- Batch inference from dataset JSON with stratified sampling
- FP8 quantization (Hopper GPUs: H100/H800)
- torch.compile for faster inference
- Benchmarking mode (skip_save)

Usage (single sample):
    python inference_streaming.py \
        --image first_frame.png \
        --control_video control.mp4 \
        --checkpoint /path/to/checkpoint \
        --output results/

Usage (batch from dataset):
    python inference_streaming.py \
        --data_json /mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/sample_data.json \
        --checkpoint /path/to/checkpoint \
        --output_dir results/ \
        --num_samples_per_dataset 3 \
        --fp8 --compile
"""
import argparse
import json
import os
import random
import sys
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import imageio.v2 as imageio
import cv2
import decord
from safetensors.torch import load_file
from termcolor import cprint
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from transformers import T5TokenizerFast, UMT5EncoderModel

# Add core.streaming to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.streaming import WanCausalTransformer3DModel, DynamicCache, apply_monkey_patch
from core.streaming.utils import load_teacher_into_pipe
from core.streaming.train_distill import _streaming_generate
from core.streaming.streaming_generate_block_v2 import generate_streaming_block_v2


MODEL_PATH = os.environ.get('MODEL_PATH', 'pretrained/Wan2.2-TI2V-5B-Diffusers')
NUM_FRAMES = 81
HEIGHT = 480
WIDTH = 832

if not os.path.isdir(MODEL_PATH):
    print(f"ERROR: MODEL_PATH={MODEL_PATH} does not exist.\n"
          f"  Set MODEL_PATH env var to your local Wan2.2-TI2V-5B-Diffusers directory.",
          file=sys.stderr)
    sys.exit(1)


# ── Utility functions ─────────────────────────────────────────────────────────
# Pure IO / VAE encoding helpers live in core/inference/rynnworld_teleop_streaming_utils.py.
from core.inference.rynnworld_teleop_streaming_utils import (
    read_image,
    read_control_video,
    encode_image_to_latent,
    encode_video_to_latent,
    _decode_latent_to_uint8,
    _write_mp4,
    _make_time_strip,
    _dataset_group_of,
    _case_name_of,
    _safetensors_load_robust,
)


def encode_text(prompt, device, dtype):
    if not prompt:
        return None
    tokenizer = T5TokenizerFast.from_pretrained(MODEL_PATH, subfolder="tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(MODEL_PATH, subfolder="text_encoder").to(device=device, dtype=dtype)
    with torch.no_grad():
        ids = tokenizer(prompt, padding="max_length", max_length=226,
                        truncation=True, add_special_tokens=True, return_tensors="pt").input_ids
        emb = text_encoder(ids.to(device))[0]
    del text_encoder
    torch.cuda.empty_cache()
    return emb.squeeze(0).cpu()


# ── Model loading ─────────────────────────────────────────────────────────────

def load_streaming_pipeline(checkpoint_path, dtype, device, args):
    """Load streaming model from checkpoint with optional FP8/compile."""
    cprint(f"[infer] Loading base pipeline from {MODEL_PATH} ...", "cyan")
    pipe = WanImageToVideoPipeline.from_pretrained(MODEL_PATH, torch_dtype=dtype)
    
    # Drop unused submodules
    if getattr(pipe, "transformer_2", None) is not None:
        del pipe.transformer_2
        pipe.transformer_2 = None
    if getattr(pipe, "image_encoder", None) is not None:
        del pipe.image_encoder
        pipe.image_encoder = None
    
    # Build causal transformer
    cprint(f"[infer] Building WanCausalTransformer3DModel ...", "cyan")
    transformer = WanCausalTransformer3DModel.from_config(pipe.transformer.config).to(dtype)
    
    # Warm init from base transformer
    inc = transformer.load_state_dict(pipe.transformer.state_dict(), strict=False)
    cprint(f"[infer]  warm-init: missing={len(inc.missing_keys)} unexpected={len(inc.unexpected_keys)}", "yellow")
    
    # Initialize control patch embedding
    transformer.init_control_patch_embedding()
    
    # Drop pipe transformer
    pipe.transformer = None
    
    # Load checkpoint weights
    cprint(f"[infer] Loading checkpoint from {checkpoint_path} ...", "cyan")
    
    # Check if it's a .pt file or directory
    if os.path.isfile(checkpoint_path):
        sd = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    else:
        # Try to find generator.pt or generator_ema.pt
        for fname in ["generator_ema.pt", "generator.pt"]:
            fpath = os.path.join(checkpoint_path, fname)
            if os.path.exists(fpath):
                sd = torch.load(fpath, map_location="cpu", weights_only=False)
                cprint(f"[infer]  loaded {fname}", "green")
                break
        else:
            # Try HF split format: ema_weights.bin + control_patch_embedding.bin + control_scale.bin
            ema_path = os.path.join(checkpoint_path, "ema_weights.bin")
            if os.path.exists(ema_path):
                sd = torch.load(ema_path, map_location="cpu", weights_only=False)
                cprint(f"[infer]  loaded ema_weights.bin ({len(sd)} keys)", "green")
                cpe_path = os.path.join(checkpoint_path, "control_patch_embedding.bin")
                if os.path.exists(cpe_path):
                    cpe_sd = torch.load(cpe_path, map_location="cpu", weights_only=False)
                    for k, v in cpe_sd.items():
                        sd[f"control_patch_embedding.{k}"] = v
                    cprint(f"[infer]  loaded control_patch_embedding.bin", "green")
                cs_path = os.path.join(checkpoint_path, "control_scale.bin")
                if os.path.exists(cs_path):
                    cs_val = torch.load(cs_path, map_location="cpu", weights_only=False)
                    sd["control_scale"] = cs_val if isinstance(cs_val, torch.Tensor) else torch.tensor(cs_val)
                    cprint(f"[infer]  loaded control_scale.bin = {float(sd['control_scale']):.4f}", "green")
            else:
                # Try loading as teacher checkpoint format
                cpe_state, control_scale = load_teacher_into_pipe(
                    pipe=pipe,
                    teacher_ckpt_dir=checkpoint_path,
                    use_ema=True,
                    is_main=True,
                )
                if cpe_state is not None:
                    transformer.control_patch_embedding.load_state_dict(cpe_state)
                if control_scale is not None:
                    transformer.control_scale.data = torch.tensor(
                        control_scale, dtype=dtype, device=transformer.control_scale.device
                    )
                sd = {}
    
    if sd:
        inc = transformer.load_state_dict(sd, strict=False)
        cprint(f"[infer]  checkpoint overlay: missing={len(inc.missing_keys)} unexpected={len(inc.unexpected_keys)}", "green")
    
    # Print control_scale
    if transformer.control_scale is not None:
        cprint(f"[infer]  control_scale = {float(transformer.control_scale):.4f}", "green")
    
    transformer.set_attention_backend("flash")
    transformer.eval()
    transformer.to(device)
    for p in transformer.parameters():
        p.requires_grad = False
    
    # FP8 quantization (Hopper only)
    if args.fp8:
        cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
        if cap[0] < 9:
            cprint(f"[infer] WARN: --fp8 requested but GPU compute capability is {cap[0]}.{cap[1]} "
                   f"(< 9.0). FP8 requires Hopper (H100/H800). Skipping.", "red")
        else:
            try:
                from torchao.quantization import quantize_, Float8DynamicActivationFloat8WeightConfig
                cprint("[infer] applying torchao FP8 dynamic W8A8 quantization ...", "cyan")
                quantize_(transformer, Float8DynamicActivationFloat8WeightConfig())
                cprint("[infer]  FP8 quantization applied (Hopper Tensor Cores)", "green")
            except ImportError as e:
                cprint(f"[infer] WARN: torchao not installed ({e!r}); skipping FP8.", "red")
    
    # torch.compile
    if args.compile:
        cprint("[infer] torch.compile(transformer, mode='default', dynamic=True) ...", "cyan")
        transformer = torch.compile(transformer, mode="default", dynamic=True)
        cprint("[infer]  compiled — first sample will include compile overhead", "yellow")
    
    # VAE
    vae = pipe.vae
    vae.to(device).eval()
    
    # Scheduler
    scheduler = pipe.scheduler
    
    return transformer, vae, scheduler


# ── Streaming generation ──────────────────────────────────────────────────────

def streaming_generate(
    transformer,
    scheduler,
    img_latent,
    control_video_latents,
    prompt_embeds,
    num_inference_steps=1,
    guidance_scale=1.0,
    max_cache_frames=6,
    seed=42,
    device=None,
    dtype=torch.bfloat16,
):
    """Generate video frame-by-frame using streaming with KV cache.
    
    Delegates to _streaming_generate (same function used during training)
    to ensure train-test consistency.
    """
    if device is None:
        device = img_latent.device

    # img_latent: [B, C, H, W] → need [B, C, 1, H, W] for _streaming_generate
    if img_latent.ndim == 4:
        img_latent_5d = img_latent.unsqueeze(2)
    else:
        img_latent_5d = img_latent

    # control_video_latents: [C, F, H, W] → [1, C, F, H, W]
    if control_video_latents.ndim == 4:
        ctrl_5d = control_video_latents.unsqueeze(0).to(device=device, dtype=dtype)
    else:
        ctrl_5d = control_video_latents.to(device=device, dtype=dtype)

    num_latent_frames = ctrl_5d.shape[2]
    
    patch_size = transformer.config.patch_size if hasattr(transformer, 'config') else (1, 2, 2)
    rng = torch.Generator(device=device).manual_seed(seed)

    with torch.no_grad():
        _use_fixed_cache = os.environ.get("USE_FIXED_CACHE", "0") == "1"
        _nfpb = int(os.environ.get("NUM_FRAME_PER_BLOCK", "1"))
        if _use_fixed_cache and _nfpb > 1:
            output = generate_streaming_block_v2(
                generator=transformer,
                scheduler=scheduler,
                img_latent=img_latent_5d,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=None,
                num_latent_frames=num_latent_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                do_cfg=False,
                generator_rng=rng,
                device=device,
                dtype=dtype,
                patch_size=patch_size,
                num_max_frames=int(os.environ.get("NUM_MAX_FRAMES", str(num_latent_frames))),
                pe_mode="absolute",
                stochastic_grad_truncation=False,
                control_video_latent=ctrl_5d,
                num_frame_per_block=_nfpb,
                sink_size=int(os.environ.get("SINK_SIZE", "1")),
                local_attn_size=int(os.environ.get("LOCAL_ATTN_SIZE", "-1")),
            )
        else:
            output = _streaming_generate(
                generator=transformer,
                scheduler=scheduler,
                img_latent=img_latent_5d,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=None,
                num_latent_frames=num_latent_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                do_cfg=False,
                generator_rng=rng,
                device=device,
                dtype=dtype,
                patch_size=patch_size,
                with_grad=False,
                max_cache_frames=max_cache_frames,
                pe_mode="slot",
                stochastic_grad_truncation=False,
                control_video_latent=ctrl_5d,
            )

    return output


# ── Single sample inference ───────────────────────────────────────────────────

def run_single_sample(args, transformer, vae, scheduler, device, dtype):
    """Run inference on a single image + control video."""
    os.makedirs(args.output, exist_ok=True)
    
    # 1. Read inputs
    cprint(f"\n[1/4] Reading inputs", "cyan")
    cprint(f"  image: {args.image}", "cyan")
    cprint(f"  control_video: {args.control_video}", "cyan")
    image_np = read_image(args.image, HEIGHT, WIDTH)
    control_np = read_control_video(args.control_video, NUM_FRAMES, HEIGHT, WIDTH)
    cprint(f"  image shape: {image_np.shape}, control shape: {control_np.shape}", "green")
    
    # 2. Encode via VAE
    cprint(f"\n[2/4] Encoding image + control video", "cyan")
    img_latent = encode_image_to_latent(vae, image_np, device, dtype)
    control_video_latents = encode_video_to_latent(vae, control_np, device, dtype)
    cprint(f"  img_latent: {img_latent.shape}", "green")
    cprint(f"  control_video_latents: {control_video_latents.shape}", "green")
    
    # Text embedding
    prompt_embeds = None
    if args.prompt:
        cprint(f"  prompt: \"{args.prompt}\"", "cyan")
        text_emb = encode_text(args.prompt, device, dtype)
        prompt_embeds = text_emb.unsqueeze(0).to(device=device, dtype=dtype)
    elif args.text_embedding and os.path.exists(args.text_embedding):
        cprint(f"  text_embedding: {args.text_embedding}", "cyan")
        text_emb = load_file(args.text_embedding)["text_embedding"]
        prompt_embeds = text_emb.unsqueeze(0).to(device=device, dtype=dtype)
    else:
        cprint(f"  no prompt (using null embedding)", "yellow")
        null_prompt_path = os.environ.get(
            "NULL_PROMPT_PATH", "/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/null_prompt_embedding.safetensors"
        )
        if os.path.exists(null_prompt_path):
            prompt_embeds = load_file(null_prompt_path)["null_prompt_embedding"]
            prompt_embeds = prompt_embeds.unsqueeze(0).to(device=device, dtype=dtype)
        else:
            prompt_embeds = torch.zeros(1, 226, 4096, device=device, dtype=dtype)
    
    # 3. Generate
    cprint(f"\n[3/4] Generating video (steps={args.num_inference_steps})", "cyan")

    img_latent_5d = img_latent.unsqueeze(0).unsqueeze(2).to(device=device, dtype=dtype)
    control_video_latents_5d = control_video_latents.unsqueeze(0).to(device=device, dtype=dtype)

    # ── Add-plus control normalization (matches training) ──
    # Training-time formula:
    #     ctrl_norm = (ctrl - c_run_mean) / c_run_std * v_std + v_mean
    # v_mean/v_std are the video latent's per-channel stats. In batch mode
    # (--data_json) they come from GT. Single-sample deployment has no GT,
    # so we use the empirical training-time averages (0 / 0.8) — verified
    # from ctrl-stats log entries: v_mean ≈ 0 ± 0.1, v_std ≈ 0.7-1.0.
    if args.use_control_norm:
        if not args.control_running_stats:
            ckpt_dir = args.checkpoint if os.path.isdir(args.checkpoint) \
                else os.path.dirname(args.checkpoint)
            auto_path = os.path.join(ckpt_dir, "control_running_stats.bin")
            if os.path.exists(auto_path):
                args.control_running_stats = auto_path
                cprint(f"  auto-detected control_running_stats: {auto_path}", "cyan")

        if args.control_running_stats and os.path.exists(args.control_running_stats):
            crs = torch.load(args.control_running_stats, map_location="cpu", weights_only=False)
            c_mean = crs["mean"].to(device=device, dtype=torch.float32)
            c_std = crs["std"].to(device=device, dtype=torch.float32)
            cprint(f"  control norm: c_mean={c_mean.mean().item():+.4f} "
                   f"c_std={c_std.mean().item():.4f}  →  target N(v_mean={args.control_v_mean}, "
                   f"v_std={args.control_v_std})", "green")
            control_video_latents_5d = (
                (control_video_latents_5d.to(torch.float32) - c_mean) / c_std
                * args.control_v_std + args.control_v_mean
            ).to(dtype=dtype)
        else:
            cprint(f"  WARN: --use_control_norm set but no control_running_stats.bin "
                   f"found — skipping normalization", "yellow")
    else:
        cprint(f"  control norm: OFF (--use_control_norm 0)", "yellow")

    torch.cuda.synchronize(device)
    t0 = time.time()
    
    latents = streaming_generate(
        transformer=transformer,
        scheduler=scheduler,
        img_latent=img_latent_5d.squeeze(2),
        control_video_latents=control_video_latents_5d.squeeze(0),
        prompt_embeds=prompt_embeds,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        max_cache_frames=args.max_cache_frames,
        seed=args.seed,
        device=device,
        dtype=dtype,
    )
    
    torch.cuda.synchronize(device)
    elapsed = time.time() - t0
    cprint(f"  inference time: {elapsed:.2f}s", "green")
    
    # 4. Decode and save
    if not args.skip_save:
        cprint(f"\n[4/4] Decoding and saving...", "cyan")
        latents_5d = latents if latents.ndim == 5 else latents.unsqueeze(0)
        video = _decode_latent_to_uint8(vae, latents_5d, device, dtype)

        output_path = os.path.join(args.output, f"generated_seed{args.seed}.mp4")
        _write_mp4(video, output_path, fps=16)
        cprint(f"  Saved: {output_path}", "green")

        # Also copy the input control video for side-by-side comparison.
        control_path = os.path.join(args.output, "control.mp4")
        _write_mp4(control_np, control_path, fps=16)
        cprint(f"  Saved: {control_path}", "green")
    else:
        cprint(f"\n[4/4] skip_save enabled, skipping decode", "yellow")
    
    cprint(f"\nDone!", "green")


# ── Batch inference from dataset ──────────────────────────────────────────────

def run_batch_inference(args, transformer, vae, scheduler, device, dtype):
    """Run inference on samples from dataset JSON."""
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    with open(args.data_json, "r", encoding="utf-8") as f:
        all_items = json.load(f)
    
    cprint(f"[batch] Loaded {len(all_items)} samples from {args.data_json}", "cyan")
    
    # Build sample list
    if args.sample_indices:
        sample_idx_list = [int(s) for s in args.sample_indices.split(",") if s.strip()]
        cprint(f"[batch] explicit indices: {len(sample_idx_list)} samples", "cyan")
    elif args.num_samples_per_dataset > 0:
        # Stratified sampling
        bucket_to_idx = defaultdict(list)
        for i, it in enumerate(all_items):
            g = _dataset_group_of(it["video_latent_path"])
            bucket_to_idx[g].append(i)
        
        rng = random.Random(args.seed)
        sample_idx_list = []
        cprint(f"[batch] stratified sampling: {args.num_samples_per_dataset} per bucket", "cyan")
        for g in sorted(bucket_to_idx.keys()):
            pool = bucket_to_idx[g]
            n = min(args.num_samples_per_dataset, len(pool))
            picks = rng.sample(pool, n)
            sample_idx_list.extend(picks)
            cprint(f"  {g:<32s} pool={len(pool):>5d}  picked={n}", "yellow")
    else:
        raise ValueError("Must specify --sample_indices or --num_samples_per_dataset")
    
    # Save metadata
    meta = {
        "checkpoint": args.checkpoint,
        "sample_indices": sample_idx_list,
        "num_inference_steps": args.num_inference_steps,
        "max_cache_frames": args.max_cache_frames,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "fp8": args.fp8,
        "compile": args.compile,
        "skip_save": args.skip_save,
    }
    with open(os.path.join(args.output_dir, "_run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    
    # Load control running stats for normalization.
    # Auto-detect from checkpoint dir when not specified.
    if not args.control_running_stats:
        ckpt_dir = args.checkpoint if os.path.isdir(args.checkpoint) \
            else os.path.dirname(args.checkpoint)
        auto_path = os.path.join(ckpt_dir, "control_running_stats.bin")
        if os.path.exists(auto_path):
            args.control_running_stats = auto_path
            cprint(f"[batch] Auto-detected control_running_stats: {auto_path}", "cyan")

    c_mean_run, c_std_run = None, None
    if args.control_running_stats and os.path.exists(args.control_running_stats):
        cprint(f"[batch] Loading control_running_stats: {args.control_running_stats}", "cyan")
        crs = torch.load(args.control_running_stats, map_location="cpu", weights_only=False)
        c_mean_run = crs["mean"].to(device=device, dtype=torch.float32)
        c_std_run = crs["std"].to(device=device, dtype=torch.float32)
        cprint(f"[batch]  c_mean={c_mean_run.mean().item():+.4f} c_std={c_std_run.mean().item():.4f}", "green")
    
    # Per-sample inference
    timing_stats = []
    for sample_idx in sample_idx_list:
        if sample_idx >= len(all_items):
            cprint(f"[batch] WARN: idx {sample_idx} >= dataset size, skipping", "red")
            continue
        
        item = all_items[sample_idx]
        group = _dataset_group_of(item["video_latent_path"])
        cprint(f"\n[batch] === sample {sample_idx} [{group}] ===", "cyan")
        
        # Load latents
        data = _safetensors_load_robust(item["video_latent_path"])
        video_lat = data["video_latents"].to(dtype=dtype)
        ctrl_lat = data["control_video_latents"].to(dtype=dtype)
        
        video_lat_5d = video_lat.unsqueeze(0).to(device)
        ctrl_lat_5d = ctrl_lat.unsqueeze(0).to(device)
        img_latent_5d = video_lat_5d[:, :, :1]
        
        # Apply add-plus normalization to control (match training)
        if c_mean_run is not None:
            _v_fp32 = video_lat_5d.to(torch.float32)
            v_mean = _v_fp32.mean(dim=(0, 2, 3, 4), keepdim=True)
            v_std = _v_fp32.std(dim=(0, 2, 3, 4), keepdim=True) + 1e-8
            ctrl_lat_5d = (
                (ctrl_lat_5d.to(torch.float32) - c_mean_run) / c_std_run * v_std + v_mean
            ).to(dtype=dtype)
        
        # Text embedding
        text_embeds = None
        te_path = item.get("text_embedding_path")
        if te_path and os.path.exists(te_path):
            try:
                te_data = _safetensors_load_robust(te_path)
                if "text_embedding" in te_data:
                    text_embeds = te_data["text_embedding"].to(dtype=dtype).unsqueeze(0).to(device)
            except Exception:
                pass
        
        if text_embeds is None:
            null_paths = [
                os.environ.get("NULL_PROMPT_PATH", ""),
                "/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/null_prompt_embedding.safetensors",
            ]
            for npth in null_paths:
                if os.path.exists(npth):
                    text_embeds = load_file(npth)["null_prompt_embedding"].to(dtype=dtype).unsqueeze(0).to(device)
                    break
        
        if text_embeds is None:
            text_embeds = torch.zeros(1, 226, 4096, dtype=dtype, device=device)
        
        # Generate
        _seed = ((sample_idx * 2654435761) & 0x7FFFFFFF) ^ args.seed
        
        torch.cuda.synchronize(device)
        t0 = time.time()
        
        student_lat = streaming_generate(
            transformer=transformer,
            scheduler=scheduler,
            img_latent=img_latent_5d.squeeze(2),
            control_video_latents=ctrl_lat_5d.squeeze(0),
            prompt_embeds=text_embeds,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            max_cache_frames=args.max_cache_frames,
            seed=_seed,
            device=device,
            dtype=dtype,
        )
        
        torch.cuda.synchronize(device)
        elapsed = time.time() - t0
        
        num_latent_frames = video_lat.shape[1]
        cprint(f"  inference: {elapsed:.1f}s ({num_latent_frames} latent frames, "
               f"{num_latent_frames / elapsed:.1f} FPS)", "yellow")
        
        timing_stats.append({
            "sample_idx": sample_idx,
            "group": group,
            "latent_frames": int(num_latent_frames),
            "elapsed_s": float(elapsed),
        })
        
        if not args.skip_save:
            student_u8 = _decode_latent_to_uint8(vae, student_lat, device, dtype)
            gt_u8 = _decode_latent_to_uint8(vae, video_lat_5d, device, dtype)
            ctrl_u8 = _decode_latent_to_uint8(vae, ctrl_lat_5d, device, dtype)
            
            prefix = _case_name_of(item, sample_idx)
            _write_mp4(student_u8, os.path.join(args.output_dir, f"{prefix}_student.mp4"), fps=16)
            _write_mp4(gt_u8, os.path.join(args.output_dir, f"{prefix}_gt.mp4"), fps=16)
            _write_mp4(ctrl_u8, os.path.join(args.output_dir, f"{prefix}_control.mp4"), fps=16)
            
            cprint(f"  saved → {prefix}_{{student,control,gt}}.mp4", "green")
            
            del student_u8, gt_u8, ctrl_u8
        else:
            cprint("  [skip_save] decode skipped", "yellow")
        
        del student_lat, video_lat_5d, ctrl_lat_5d, video_lat, ctrl_lat, text_embeds, img_latent_5d
        torch.cuda.empty_cache()
    
    # Summary
    if timing_stats:
        total_samples = len(timing_stats)
        total_time = sum(t["elapsed_s"] for t in timing_stats)
        total_lf = sum(t["latent_frames"] for t in timing_stats)
        
        cprint("\n[batch] ========== FPS summary ==========", "cyan")
        cprint(f"  samples            : {total_samples}", "green")
        cprint(f"  total inference    : {total_time:.2f} s", "green")
        cprint(f"  total latent frames: {total_lf}", "green")
        cprint(f"  mean latent FPS    : {total_lf / total_time:.2f}", "green")
        
        with open(os.path.join(args.output_dir, "_timing_stats.json"), "w") as f:
            json.dump({"samples": timing_stats, "total_time_s": total_time}, f, indent=2)
    
    cprint(f"\n[batch] DONE. {args.output_dir}", "green")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    
    # Auto-set steps based on mode
    if args.mode == "dmd" and args.num_inference_steps == 1:
        args.num_inference_steps = 4

    # Propagate v3 block-streaming knobs into env for streaming_generate dispatch.
    os.environ["USE_FIXED_CACHE"] = "1" if args.use_fixed_cache else "0"
    os.environ["NUM_FRAME_PER_BLOCK"] = str(args.num_frame_per_block)
    os.environ["NUM_MAX_FRAMES"] = str(args.num_max_frames)
    os.environ["SINK_SIZE"] = str(args.sink_size)
    os.environ["LOCAL_ATTN_SIZE"] = str(args.local_attn_size)
    
    # Load model
    cprint(f"\n[infer] Loading streaming checkpoint (mode={args.mode})", "cyan")
    transformer, vae, scheduler = load_streaming_pipeline(
        args.checkpoint, dtype, device, args
    )
    
    # Run inference
    if args.data_json:
        run_batch_inference(args, transformer, vae, scheduler, device, dtype)
    else:
        run_single_sample(args, transformer, vae, scheduler, device, dtype)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Streaming inference for RynnWorld-Teleop")
    
    # Input options
    parser.add_argument("--image", type=str, default=None,
                        help="Path to first-frame image (for single sample)")
    parser.add_argument("--control_video", type=str, default=None,
                        help="Path to control video (for single sample)")
    parser.add_argument("--data_json", type=str, default=None,
                        help="Dataset JSON for batch inference")
    
    # Output options
    parser.add_argument("--output", type=str, default="results",
                        help="Output directory (single sample)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (batch, defaults to --output)")
    
    # Model options
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Streaming checkpoint path (.pt file or directory)")
    parser.add_argument("--mode", type=str, default="mse", choices=["mse", "dmd"],
                        help="Checkpoint type: 'mse' (1-step) or 'dmd' (4-step)")
    
    # Inference options
    parser.add_argument("--num_inference_steps", type=int, default=1,
                        help="Number of ODE steps (1 for MSE, 4 for DMD)")
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--max_cache_frames", type=int, default=6,
                        help="Max frames in KV cache (legacy per-frame path)")
    parser.add_argument("--seed", type=int, default=42)

    # ── v3 block streaming (Self-Forcing aligned) ──
    parser.add_argument("--use_fixed_cache", type=int, default=1,
                        help="1 = block streaming + FixedSizeCache (matches v3 training). "
                             "0 = legacy per-frame streaming with DynamicCache.")
    parser.add_argument("--num_frame_per_block", type=int, default=3,
                        help="Frames per block (v3 default 3). Set 1 for legacy per-frame.")
    parser.add_argument("--num_max_frames", type=int, default=21,
                        help="FixedSizeCache buffer size, in frames")
    parser.add_argument("--sink_size", type=int, default=1,
                        help="Frames kept forever at cache head (condition frame)")
    parser.add_argument("--local_attn_size", type=int, default=-1,
                        help="-1 = attend full cache (no eviction)")
    
    # Text options
    parser.add_argument("--prompt", type=str, default="",
                        help="Text prompt")
    parser.add_argument("--text_embedding", type=str, default=None,
                        help="Pre-encoded text embedding .safetensors")
    
    # Batch options
    parser.add_argument("--sample_indices", type=str, default="",
                        help="CSV of explicit sample indices")
    parser.add_argument("--num_samples_per_dataset", type=int, default=0,
                        help="Random samples per dataset bucket")
    
    # Performance options
    parser.add_argument("--fp8", action="store_true",
                        help="FP8 quantization (Hopper GPUs: H100/H800)")
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile for faster inference")
    parser.add_argument("--skip_save", action="store_true",
                        help="Skip saving videos (benchmarking mode)")
    parser.add_argument("--control_running_stats", type=str, default="",
                        help="Path to control_running_stats.bin for add-plus normalization. "
                             "Auto-detected from checkpoint directory when not specified.")
    parser.add_argument("--use_control_norm", type=int, default=1,
                        help="1 = apply add-plus control normalization at inference "
                             "(matches training). 0 = feed raw VAE-encoded control.")
    parser.add_argument("--control_v_mean", type=float, default=0.0,
                        help="Target video-latent mean for control normalization. "
                             "Empirical training-time value ≈ 0.0.")
    parser.add_argument("--control_v_std", type=float, default=0.8,
                        help="Target video-latent std for control normalization. "
                             "Empirical training-time value ≈ 0.8.")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.data_json and (not args.image or not args.control_video):
        parser.error("Must provide either --data_json or (--image and --control_video)")
    
    # Set output_dir default
    if args.output_dir is None:
        args.output_dir = args.output
    
    main(args)
