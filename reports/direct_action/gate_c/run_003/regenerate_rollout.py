#!/usr/bin/env python
"""Regenerate rollouts using trained checkpoint with fixed F=9."""

import sys
import torch
import numpy as np
from pathlib import Path

# Add repo to path
repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

# Import helper functions from main script
import importlib.util
spec = importlib.util.spec_from_file_location("gate_c_visual_v2",
    repo_root / "reports/direct_action/gate_c/run_003/artifacts/gate_c_visual_v2.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# Extract needed functions
load_text_encoder = module.load_text_encoder
encode_text = module.encode_text
load_vae = module.load_vae
encode_video_frames = module.encode_video_frames
load_and_decode_video = module.load_and_decode_video
preprocess_video_frames = module.preprocess_video_frames
load_episode_data = module.load_episode_data
load_model = module.load_model
generate_rollout = module.generate_rollout
save_video_mp4 = module.save_video_mp4

def main():
    checkpoint_step = 1000
    checkpoint_path = repo_root / f"reports/direct_action/gate_c/run_003/checkpoints/adapter_step{checkpoint_step:04d}.pt"

    device = torch.device("cuda")
    dtype = torch.bfloat16

    print(f"[Regenerate] Loading checkpoint from {checkpoint_path}")

    # Load adapter
    from core.control.native_trajectory_encoder import NativeTrajectoryEncoder
    encoder = NativeTrajectoryEncoder(input_dim=33).to(device)
    encoder.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    encoder.eval()
    print(f"[Regenerate] Loaded adapter")

    # Load frozen models
    base_model_dir = repo_root / "pretrained/Wan2.2-TI2V-5B-Diffusers"
    checkpoint_dir = repo_root / "pretrained/RynnWorld-Teleop-Causal"

    tokenizer, text_encoder = load_text_encoder(base_model_dir, device, dtype)
    null_embedding = encode_text(tokenizer, text_encoder, "", device)
    vae, latents_mean, latents_std = load_vae(base_model_dir, device, dtype)

    # Load video and actions
    video_path = Path("/tmp/scratch/gate_c_video/data/videos/chunk-000/observation.images.top_head/episode_000008.mp4")
    parquet_path = Path("/tmp/scratch/gate_b_task3400/data/data/chunk-000/episode_000008.parquet")

    frames, video_metadata = load_and_decode_video(video_path, max_frames=33)
    video_preprocessed = preprocess_video_frames(frames, 480, 832).unsqueeze(0).to(device, dtype=dtype)
    video_latent = encode_video_frames(vae, video_preprocessed, latents_mean, latents_std)
    img_latent = video_latent[:, :, 0:1].clone()

    df, actions_33d = load_episode_data(parquet_path, video_metadata)
    robot_trajectory = torch.from_numpy(actions_33d).unsqueeze(0).to(device, dtype=torch.float32)

    # Load model
    model = load_model(checkpoint_dir, base_model_dir, sys.stdout)
    model.native_trajectory_encoder = encoder

    print(f"\n[Regenerate] Data shapes:")
    print(f"  img_latent: {img_latent.shape}")
    print(f"  video_latent: {video_latent.shape}")
    print(f"  robot_trajectory: {robot_trajectory.shape}")

    out_dir = repo_root / "reports/direct_action/gate_c/run_003/videos"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate C3 correct
    print(f"\n[Regenerate] Generating C3 correct action rollout (F will be {img_latent.shape[2]})...")
    video_correct = generate_rollout(
        model, encoder, vae, img_latent, robot_trajectory, null_embedding,
        latents_mean, latents_std, num_inference_steps=20, seed=42, null_condition=False
    )
    save_video_mp4(video_correct, out_dir / f"step{checkpoint_step:04d}_c3_correct_FIXED.mp4", fps=30)
    print(f"  Output shape: {video_correct.shape}")
    print(f"  Saved to step{checkpoint_step:04d}_c3_correct_FIXED.mp4")

    # Generate C0 no action
    print(f"\n[Regenerate] Generating C0 no action...")
    video_no_action = generate_rollout(
        model, encoder, vae, img_latent, robot_trajectory, null_embedding,
        latents_mean, latents_std, num_inference_steps=20, seed=42, null_condition=True
    )
    save_video_mp4(video_no_action, out_dir / f"step{checkpoint_step:04d}_c0_no_action_FIXED.mp4", fps=30)
    print(f"  Saved to step{checkpoint_step:04d}_c0_no_action_FIXED.mp4")

    # Generate C3 shuffled
    print(f"\n[Regenerate] Generating C3 shuffled action...")
    shuffled_traj = robot_trajectory.clone()
    perm = torch.randperm(shuffled_traj.shape[1])
    shuffled_traj = shuffled_traj[:, perm]
    video_shuffled = generate_rollout(
        model, encoder, vae, img_latent, shuffled_traj, null_embedding,
        latents_mean, latents_std, num_inference_steps=20, seed=42, null_condition=False
    )
    save_video_mp4(video_shuffled, out_dir / f"step{checkpoint_step:04d}_c3_shuffled_FIXED.mp4", fps=30)
    print(f"  Saved to step{checkpoint_step:04d}_c3_shuffled_FIXED.mp4")

    # Generate C3 wrong (reversed)
    print(f"\n[Regenerate] Generating C3 wrong action (reversed)...")
    reversed_traj = torch.flip(robot_trajectory, dims=[1])
    video_wrong = generate_rollout(
        model, encoder, vae, img_latent, reversed_traj, null_embedding,
        latents_mean, latents_std, num_inference_steps=20, seed=42, null_condition=False
    )
    save_video_mp4(video_wrong, out_dir / f"step{checkpoint_step:04d}_c3_wrong_FIXED.mp4", fps=30)
    print(f"  Saved to step{checkpoint_step:04d}_c3_wrong_FIXED.mp4")

    print(f"\n[Regenerate] Complete. All 4 rollouts saved to {out_dir}")

if __name__ == "__main__":
    main()
