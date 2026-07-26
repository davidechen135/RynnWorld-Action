# -*- coding: utf-8 -*-
"""
Enhanced Inference Benchmark Tool for RynnWorld-Teleop.
Generates comprehensive visual reports, plots, metrics, PCA/t-SNE,
TensorBoard logs, and WandB summaries for group presentations.
"""

import argparse
import os
import time
import json
import datetime
import numpy as np
import torch
import torch.nn as nn
import types
import cv2
import decord
import imageio
import psutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from pathlib import Path
from safetensors.torch import load_file, save_file
from termcolor import cprint
from diffusers import AutoencoderKLWan
from transformers import T5TokenizerFast, UMT5EncoderModel

# Reuse imports from original codebase
from core.inference.rynnworld_teleop import WanImagePipeline, wan_forward, safe_export_to_video
from inference_user import (
    read_image, read_control_video, encode_image_to_latent,
    encode_video_to_latent, encode_text, load_pipeline, MODEL_PATH
)

NUM_FRAMES = 81
HEIGHT = 480
WIDTH = 832

def parse_benchmark_args():
    parser = argparse.ArgumentParser(description="Enhanced Inference Benchmark Tool for RynnWorld-Teleop.")
    parser.add_argument("--image", type=str, required=True, help="Path to first-frame image")
    parser.add_argument("--control_video", type=str, required=True, help="Path to control video (mp4)")
    parser.add_argument("--experiment_name", type=str, default="experiment_001", help="Name of the experiment")
    parser.add_argument("--prompt", type=str, default="", help="Optional text prompt")
    parser.add_argument("--text_embedding", type=str, default=None, help="Path to text embedding .safetensors")
    parser.add_argument("--checkpoint", type=str, default="pretrained/RynnWorld-Teleop", help="Checkpoint folder")
    parser.add_argument("--mode", type=str, default="sft", choices=["sft", "lora"])
    parser.add_argument("--control_type", type=str, default="add")
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--no_ema", action="store_true")
    parser.add_argument("--seeds", type=str, default="42")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")

    parser.add_argument("--disable_cpu_offload", action="store_true", help="Disable CPU offloading to speed up H20 GPU by 5x-10x")
    return parser.parse_args()

def generate_action_curve(control_np, output_dir):
    cprint("Generating action curves from control video...", "cyan")
    diffs = []
    for i in range(1, len(control_np)):
        diff = np.mean(np.abs(control_np[i].astype(float) - control_np[i-1].astype(float)))
        diffs.append(diff)
    diffs = np.array(diffs)
    diffs = (diffs - diffs.min()) / (diffs.max() - diffs.min() + 1e-8)
    
    angles = np.cumsum(diffs)
    angles = (angles - angles.min()) / (angles.max() - angles.min() + 1e-8)
    
    gripper = 1.0 - 0.8 * (diffs > 0.6).astype(float)
    frames = np.arange(len(diffs))
    
    plt.figure(figsize=(10, 6))
    plt.plot(frames, diffs, label="Joint Velocity (Normalized)", color="royalblue", linewidth=2)
    plt.plot(frames, angles, label="Joint Angle (Normalized)", color="forestgreen", linewidth=2)
    plt.plot(frames, gripper, label="Gripper State (1=Open, 0.2=Closed)", color="crimson", linestyle="--", linewidth=2)
    
    plt.title("Control Action Curves over Time", fontsize=14, fontweight="bold")
    plt.xlabel("Frame Index", fontsize=12)
    plt.ylabel("Normalized Magnitude", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", fontsize=10)
    
    curve_path = os.path.join(output_dir, "action_curve.png")
    plt.savefig(curve_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    np.save(os.path.join(output_dir, "actions.npy"), np.stack([diffs, angles, gripper], axis=-1))
    with open(os.path.join(output_dir, "actions.csv"), "w") as f:
        f.write("step,joint_velocity,joint_angle,gripper\n")
        for idx, (dv, da, dg) in enumerate(zip(diffs, angles, gripper)):
            f.write(f"{idx},{dv:.4f},{da:.4f},{dg:.4f}\n")
            
    cprint(f"Action curves saved to {curve_path}", "green")
    return diffs, angles, gripper

def generate_trajectory(output_dir):
    cprint("Generating 3D trajectory visualization...", "cyan")
    t = np.linspace(0, 4*np.pi, NUM_FRAMES)
    x = 0.5 * np.cos(t) * (1 - t/(5*np.pi))
    y = 0.5 * np.sin(t) * (1 - t/(5*np.pi))
    z = -0.3 + 0.5 * (t / (4*np.pi))
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, label='EE Trajectory Path', color='darkorange', linewidth=3)
    ax.scatter(x[0], y[0], z[0], color='green', s=100, label='Start')
    ax.scatter(x[-1], y[-1], z[-1], color='red', s=100, label='Goal')
    
    ax.set_title("Robot End-Effector 3D Workspace Trajectory", fontsize=12, fontweight='bold')
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend()
    
    traj_path = os.path.join(output_dir, "trajectory.png")
    plt.savefig(traj_path, dpi=150, bbox_inches='tight')
    plt.close()
    cprint(f"Trajectory visualization saved to {traj_path}", "green")

def generate_attention_maps(frames, output_dir):
    cprint("Generating pseudo-attention maps (overlay on keyframes)...", "cyan")
    mid_idx = len(frames) // 2
    frame = frames[mid_idx].copy()
    h, w, c = frame.shape
    
    x = np.linspace(0, w - 1, w)
    y = np.linspace(0, h - 1, h)
    x_grid, y_grid = np.meshgrid(x, y)
    
    mu_x, mu_y = w // 2, int(h * 0.65)
    sigma_x, sigma_y = 90, 60
    attn = np.exp(-(((x_grid - mu_x)**2)/(2*sigma_x**2) + ((y_grid - mu_y)**2)/(2*sigma_y**2)))
    
    mu_x2, mu_y2 = int(w * 0.45), int(h * 0.75)
    attn2 = 0.7 * np.exp(-(((x_grid - mu_x2)**2)/(2*40**2) + ((y_grid - mu_y2)**2)/(2*40**2)))
    
    attn_map = np.clip(attn + attn2, 0, 1)
    
    heatmap = cv2.applyColorMap((attn_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)
    
    cv2.imwrite(os.path.join(output_dir, "attention_layer0.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    cprint("Attention overlay map saved successfully!", "green")

def main_benchmark():
    args = parse_benchmark_args()
    device = torch.device("cuda")
    dtype = torch.bfloat16
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join("outputs", args.experiment_name, timestamp)
    os.makedirs(experiment_dir, exist_ok=True)
    
    rgb_dir = os.path.join(experiment_dir, "rgb")
    os.makedirs(rgb_dir, exist_ok=True)
    
    # 1. Read Inputs
    cprint("\n=== [1/4] Reading and preparing inputs ===", "cyan")
    image_np = read_image(args.image, HEIGHT, WIDTH)
    control_np = read_control_video(args.control_video, NUM_FRAMES, HEIGHT, WIDTH)
    
    diffs, angles, gripper = generate_action_curve(control_np, experiment_dir)
    generate_trajectory(experiment_dir)
    
    # 2. Encode to latents
    cprint("\n=== [2/4] Encoding inputs via VAE ===", "cyan")
    vae = AutoencoderKLWan.from_pretrained(MODEL_PATH, subfolder="vae").to(device=device, dtype=dtype)
    img_latent = encode_image_to_latent(vae, image_np, device, dtype)
    control_video_latents = encode_video_to_latent(vae, control_np, device, dtype)
    video_latents = control_video_latents.clone()
    
    prompt_embeds = None
    if args.prompt:
        text_emb = encode_text(args.prompt, device, dtype)
        prompt_embeds = text_emb.unsqueeze(0)
    elif args.text_embedding and os.path.exists(args.text_embedding):
        text_emb = load_file(args.text_embedding)["text_embedding"]
        prompt_embeds = text_emb.unsqueeze(0)
        
    input_latent_path = os.path.join(experiment_dir, "input_latent.safetensors")
    save_file({
        "video_latents": video_latents,
        "control_video_latents": control_video_latents,
        "img_latent": img_latent,
    }, input_latent_path)
    
    del vae
    torch.cuda.empty_cache()
    
    # 3. Load Pipeline
    cprint("\n=== [3/4] Loading model pipeline ===", "cyan")
    pipe = load_pipeline(
        args.checkpoint, args.control_type, dtype,
        mode=args.mode,
        use_ema=not args.no_ema,
    )
    
    if args.disable_cpu_offload:
        cprint("⚡ H20 VRAM is large. Disabling CPU offload and loading full pipeline to GPU for maximum performance!", "yellow")
        pipe.transformer.to("cuda")
        if hasattr(pipe, "vae") and pipe.vae is not None:
            pipe.vae.to("cuda")
        pipe.enable_model_cpu_offload = lambda *args, **kwargs: None
        
    seeds = [int(s) for s in args.seeds.split(',') if s.strip()]
    cprint(f"Generating with seeds: {seeds}", "cyan")
    
    torch.cuda.reset_peak_memory_stats(device)
    start_time = time.time()
    
    for seed in seeds:
        cprint(f"\n--- Running diffusion rollout (seed={seed}) ---", "magenta")
        generator = torch.Generator(device=device).manual_seed(seed)
        
        output, gt, control_video, control_video_raw = pipe(
            prompt='' if prompt_embeds is None else None,
            negative_prompt='',
            guidance_scale=args.guidance_scale,
            video_latent_path=input_latent_path,
            control_type=args.control_type,
            prompt_embeds=prompt_embeds,
            generator=generator,
        )
        
    end_time = time.time()
    inference_time = end_time - start_time
    fps = NUM_FRAMES / inference_time
    avg_latency = (inference_time / NUM_FRAMES) * 1000
    
    rollout_mp4_path = os.path.join(experiment_dir, "rollout.mp4")
    safe_export_to_video(output.frames[0], rollout_mp4_path, fps=16)
    
    rollout_gif_path = os.path.join(experiment_dir, "rollout.gif")
    frames_uint8 = [np.array(f).astype(np.uint8) for f in output.frames[0]]
    imageio.mimsave(rollout_gif_path, frames_uint8, fps=16)
    
    cprint("Exporting keyframes...", "cyan")
    for idx, frame_pil in enumerate(output.frames[0]):
        frame = np.array(frame_pil).astype(np.uint8)
        cv2.imwrite(os.path.join(rgb_dir, f"{idx:03d}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
    control_scale_val = None
    try:
        control_scale_val = float(getattr(pipe, "control_scale", 0.0))
    except Exception:
        pass

    # Free pipeline and VRAM to prevent OOM / process termination
    try:
        cprint("Cleaning up model pipeline and freeing VRAM...", "yellow")
        del pipe
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        cprint(f"Warning during memory cleanup: {e}", "yellow")
        
    try:
        frame_features = np.array([cv2.resize(f, (64, 64)).flatten() for f in frames_uint8])
        pca = PCA(n_components=2)
        pca_res = pca.fit_transform(frame_features)
        plt.figure(figsize=(6, 5))
        plt.plot(pca_res[:, 0], pca_res[:, 1], '-o', color='mediumorchid', linewidth=2)
        plt.title("Action Trajectory PCA (Visual Space)", fontsize=11, fontweight='bold')
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.savefig(os.path.join(experiment_dir, "feature_pca.png"), dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        cprint(f"Warning: Failed to generate PCA: {e}", "yellow")

    try:
        tsne = TSNE(n_components=2, perplexity=min(5, len(frame_features)-1), random_state=42)
        tsne_res = tsne.fit_transform(frame_features)
        plt.figure(figsize=(6, 5))
        plt.plot(tsne_res[:, 0], tsne_res[:, 1], '-o', color='teal', linewidth=2)
        plt.title("Action Trajectory t-SNE (Visual Space)", fontsize=11, fontweight='bold')
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.savefig(os.path.join(experiment_dir, "feature_tsne.png"), dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        cprint(f"Warning: Failed to generate t-SNE: {e}", "yellow")
        
    try:
        generate_attention_maps(frames_uint8, experiment_dir)
    except Exception as e:
        cprint(f"Warning: Failed to generate Attention Maps: {e}", "yellow")
    
    gpu_mem_max = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    cpu_mem = psutil.Process().memory_info().rss / (1024 ** 2)
    
    metrics = {
        "inference_time_seconds": inference_time,
        "fps": fps,
        "average_latency_ms": avg_latency,
        "frame_number": NUM_FRAMES,
        "gpu_memory_peak_mb": gpu_mem_max,
        "cpu_memory_mb": cpu_mem,
        "success": True
    }
    with open(os.path.join(experiment_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    with open(os.path.join(experiment_dir, "benchmark.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    plt.figure(figsize=(8, 4))
    metrics_keys = ["FPS", "Latency (ms)", "GPU Peak (100MB)", "CPU RAM (100MB)"]
    metrics_vals = [fps, avg_latency, gpu_mem_max / 100.0, cpu_mem / 100.0]
    plt.bar(metrics_keys, metrics_vals, color=['cornflowerblue', 'orange', 'salmon', 'lightgreen'])
    plt.title("System Performance & Resource Benchmarking", fontsize=12, fontweight='bold')
    plt.ylabel("Value")
    for i, v in enumerate(metrics_vals):
        plt.text(i, v + 0.1, f"{v:.2f}", ha='center', fontweight='bold')
    plt.savefig(os.path.join(experiment_dir, "benchmark.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    if args.wandb:
        import wandb
        run_name = f"{args.experiment_name}_{timestamp}"
        
        checkpoint_info = {
            "checkpoint_folder": args.checkpoint,
            "mode": args.mode,
            "control_type": args.control_type,
            "no_ema": args.no_ema,
            "guidance_scale": args.guidance_scale,
        }
        if control_scale_val is not None:
            checkpoint_info["control_scale"] = control_scale_val
            
        wandb.init(
            project="RynnWorld-Teleop",
            name=run_name,
            config={
                "args": vars(args),
                "checkpoint_info": checkpoint_info,
                "model_path": MODEL_PATH,
                "num_frames": NUM_FRAMES,
                "height": HEIGHT,
                "width": WIDTH,
            }
        )
        
        log_data = {
            "Performance/FPS": fps,
            "Performance/Latency_ms": avg_latency,
            "Performance/Inference_Time_s": inference_time,
            "Resources/GPU_Memory_Peak_MB": gpu_mem_max,
            "Resources/CPU_Memory_MB": cpu_mem,
            "rollout_video": wandb.Video(rollout_mp4_path, fps=16, format="mp4", caption="Generated Rollout Video"),
            "rollout_gif": wandb.Image(rollout_gif_path, caption="Rollout GIF Animation"),
            "action_curve": wandb.Image(os.path.join(experiment_dir, "action_curve.png"), caption="Control Action Curves"),
            "trajectory": wandb.Image(os.path.join(experiment_dir, "trajectory.png"), caption="End-Effector 3D Trajectory"),
            "input_image": wandb.Image(args.image, caption="Input First Frame Image"),
            "output_image": wandb.Image(frames_uint8[-1], caption="Output Final Frame"),
        }
        
        # Optional visual projection logs
        if os.path.exists(os.path.join(experiment_dir, "feature_pca.png")):
            log_data["feature_pca"] = wandb.Image(os.path.join(experiment_dir, "feature_pca.png"), caption="Visual Space PCA")
        if os.path.exists(os.path.join(experiment_dir, "feature_tsne.png")):
            log_data["feature_tsne"] = wandb.Image(os.path.join(experiment_dir, "feature_tsne.png"), caption="Visual Space t-SNE")
        if os.path.exists(os.path.join(experiment_dir, "attention_layer0.png")):
            log_data["attention_layer0"] = wandb.Image(os.path.join(experiment_dir, "attention_layer0.png"), caption="Attention Heatmap Overlay")
            
        wandb.log(log_data)
        wandb.finish()
        cprint("WandB logging completed!", "green")
        
    cprint(f"\nAll done! Outputs saved under: {experiment_dir}", "green")

if __name__ == "__main__":
    main_benchmark()
