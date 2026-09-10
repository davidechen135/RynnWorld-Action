#!/usr/bin/env python3
"""
Gate C run_002: Visual validation with real video.

Fine-tune the NativeTrajectoryEncoder adapter on one real action-aligned video clip
from episode 8 (top_head camera), then generate MP4 rollouts to verify no collapse.

Hard PASS criterion: the complete C3 correct-action rollout must not collapse.
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from termcolor import cprint

# Repo root computation
_SCRIPT_PATH = Path(__file__).resolve()
_REPO_ROOT = _SCRIPT_PATH.parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

def apply_monkey_patch(repo_root: Path):
    """Import rynnworld_teleop_trainer to apply wan_forward monkey-patch."""
    from diffusers import WanTransformer3DModel as _WTM
    forward_before = id(_WTM.forward)

    # Import trainer to trigger monkey-patch
    from core.finetune.models.wan_i2v import rynnworld_teleop_trainer

    forward_after = id(_WTM.forward)
    if forward_before == forward_after:
        raise RuntimeError("Monkey-patch FAILED: WanTransformer3DModel.forward unchanged after import")

    cprint("[Gate C run_002] wan_forward monkey-patch verified", "green")


def load_vae(base_model_dir: Path, device: torch.device, dtype: torch.dtype):
    """Load production VAE (AutoencoderKLWan) for encoding and decoding."""
    from diffusers.models.autoencoders import AutoencoderKLWan

    vae = AutoencoderKLWan.from_pretrained(base_model_dir, subfolder="vae")
    vae.to(device=device, dtype=dtype)
    vae.eval()
    vae.requires_grad_(False)

    cprint(f"[VAE] Loaded from {base_model_dir}/vae", "cyan")
    cprint(f"[VAE] Device: {device}, dtype: {dtype}", "cyan")
    cprint(f"[VAE] latents_mean: {vae.config.latents_mean[:5]}... shape {len(vae.config.latents_mean)}", "cyan")
    cprint(f"[VAE] latents_std: {vae.config.latents_std[:5]}... shape {len(vae.config.latents_std)}", "cyan")

    return vae


def encode_video_frames(vae, video_frames: torch.Tensor) -> torch.Tensor:
    """
    Encode video frames to latents using the production VAE.

    Args:
        vae: AutoencoderKLWan instance
        video_frames: [B, C, F, H, W] tensor, values in [-1, 1]

    Returns:
        latent: [B, z_dim, F_latent, H_latent, W_latent] normalized latent
    """
    video_frames = video_frames.to(vae.device, dtype=vae.dtype)

    with torch.no_grad():
        latent_dist = vae.encode(video_frames).latent_dist
        latent = latent_dist.sample()

        # Normalize using VAE config
        latents_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(latent.device, latent.dtype)
        latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(latent.device, latent.dtype)
        latent = (latent - latents_mean) * latents_std

    return latent


def decode_latents_to_video(vae, latents: torch.Tensor) -> torch.Tensor:
    """
    Decode latents back to video frames.

    Args:
        vae: AutoencoderKLWan instance
        latents: [B, z_dim, F_latent, H_latent, W_latent] normalized latent

    Returns:
        video_frames: [B, C, F, H, W] tensor, values in [-1, 1]
    """
    latents = latents.to(vae.device, dtype=vae.dtype)

    with torch.no_grad():
        # Denormalize
        latents_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
        latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
        latents_denorm = latents / latents_std + latents_mean

        # Decode
        video_frames = vae.decode(latents_denorm).sample

    return video_frames


def load_and_decode_video(video_path: Path, max_frames: int = None) -> Tuple[torch.Tensor, dict]:
    """
    Load and decode video using imageio (ffmpeg backend), return frames as torch tensor.

    Returns:
        frames: [F, H, W, C] uint8 tensor
        metadata: dict with fps, frame_count, duration, etc.
    """
    import imageio.v3 as iio
    import subprocess
    import json

    # Get video metadata using ffprobe
    result = subprocess.run([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=r_frame_rate,nb_frames,width,height',
        '-of', 'json', str(video_path)
    ], capture_output=True, text=True)

    probe_data = json.loads(result.stdout)
    stream = probe_data['streams'][0]

    fps_str = stream['r_frame_rate']
    fps_num, fps_den = map(int, fps_str.split('/'))
    fps = fps_num / fps_den

    total_frames = int(stream.get('nb_frames', 0))
    if total_frames == 0:
        # Fallback: count frames by reading
        total_frames = sum(1 for _ in iio.imiter(str(video_path)))

    width = int(stream['width'])
    height = int(stream['height'])

    # Read frames
    if max_frames is not None and total_frames > max_frames:
        # Sample evenly
        indices = np.linspace(0, total_frames - 1, max_frames, dtype=int).tolist()
        frames_list = []
        for i, frame in enumerate(iio.imiter(str(video_path))):
            if i in indices:
                frames_list.append(frame)
            if len(frames_list) >= max_frames:
                break
        frames = np.stack(frames_list)
    else:
        frames = iio.imread(str(video_path), plugin='pyav')

    # Convert to torch tensor
    frames = torch.from_numpy(frames)

    metadata = {
        "fps": fps,
        "frame_count": len(frames),
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "path": str(video_path),
    }

    return frames, metadata


def preprocess_video_frames(frames: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """
    Preprocess video frames: resize, normalize to [-1, 1], and permute to [C, F, H, W].

    Args:
        frames: [F, H, W, C] uint8 tensor
        height: target height
        width: target width

    Returns:
        video: [C, F, H, W] float tensor in [-1, 1]
    """
    # Permute to [F, C, H, W]
    frames = frames.permute(0, 3, 1, 2).float()

    # Resize each frame
    frames_resized = F.interpolate(frames, size=(height, width), mode='bilinear', align_corners=False)

    # Normalize to [-1, 1]
    frames_normalized = frames_resized / 255.0 * 2.0 - 1.0

    # Permute to [C, F, H, W]
    video = frames_normalized.permute(1, 0, 2, 3)

    return video


def load_model(checkpoint_dir: Path, base_model_dir: Path, out: Dict[str, Any]):
    """Load pretrained WanTransformer3DModel with EMA weights."""
    from diffusers import WanTransformer3DModel
    from safetensors.torch import load_file

    # Load base model from Wan2.2 directory
    model = WanTransformer3DModel.from_pretrained(
        base_model_dir,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    ).cuda()

    # Load EMA weights from checkpoint
    ema_path = checkpoint_dir / "ema_weights.bin"
    if ema_path.exists():
        ema_state = torch.load(ema_path, map_location="cuda")
        model.load_state_dict(ema_state, strict=False)
        cprint(f"[Model] Loaded EMA weights from {ema_path.name}", "cyan")
    else:
        cprint(f"[Model] Warning: EMA weights not found at {ema_path}", "yellow")

    model.eval()
    model.requires_grad_(False)

    out["model_dtype"] = model.patch_embedding.weight.dtype
    out["model_device"] = model.patch_embedding.weight.device
    out["model_params_B"] = sum(p.numel() for p in model.parameters()) / 1e9

    cprint(f"[Model] Loaded {out['model_params_B']:.2f}B params, dtype={out['model_dtype']}, device={out['model_device']}", "green")

    return model


def build_adapter_33d(device: torch.device):
    """Build and verify zero-init NativeTrajectoryEncoder(input_dim=33)."""
    from core.control import NativeTrajectoryEncoder

    enc = NativeTrajectoryEncoder(input_dim=33).to(device)
    enc.train()
    enc.requires_grad_(True)

    # Verify zero-init
    assert torch.allclose(enc.output_projection.weight, torch.zeros_like(enc.output_projection.weight))
    assert torch.allclose(enc.output_projection.bias, torch.zeros_like(enc.output_projection.bias))

    n_params = sum(p.numel() for p in enc.parameters())
    cprint(f"[Adapter] NativeTrajectoryEncoder(input_dim=33): {n_params:,} params, zero-init verified", "green")

    return enc


def extract_33d_action(action_40d: np.ndarray) -> np.ndarray:
    """
    Extract 33D action from 40D raw action using the audited contract.

    Contract (raw index order):
      end/position: [2:8] = 6D
      end/orientation: [8:16] = 8D
      joint/position: [16:30] = 14D
      waist/position: [33:38] = 5D
    Total: 33D
    """
    assert action_40d.shape[-1] == 40, f"Expected 40D action, got {action_40d.shape[-1]}D"

    end_pos = action_40d[..., 2:8]
    end_ori = action_40d[..., 8:16]
    joint_pos = action_40d[..., 16:30]
    waist_pos = action_40d[..., 33:38]

    action_33d = np.concatenate([end_pos, end_ori, joint_pos, waist_pos], axis=-1)
    assert action_33d.shape[-1] == 33

    return action_33d


def load_episode_data(parquet_path: Path, video_metadata: dict) -> Tuple[pd.DataFrame, np.ndarray]:
    """Load episode parquet and extract 33D actions aligned with video frames."""
    df = pd.read_parquet(parquet_path)

    # If video was clipped/sampled, clip parquet to match
    num_frames = video_metadata["frame_count"]
    if len(df) > num_frames:
        cprint(f"[Data] Clipping parquet from {len(df)} to {num_frames} rows to match video", "yellow")
        df = df.iloc[:num_frames]

    # Verify frame count matches video
    assert len(df) == num_frames, \
        f"Parquet has {len(df)} rows, video has {num_frames} frames"

    # Extract 33D actions
    actions_40d = np.stack(df["action"].values)
    actions_33d = extract_33d_action(actions_40d)

    # Verify finite
    assert np.all(np.isfinite(actions_33d)), "Non-finite values in 33D actions"

    cprint(f"[Data] Episode parquet: {len(df)} rows", "cyan")
    cprint(f"[Data] 33D actions: shape={actions_33d.shape}, min={actions_33d.min():.3f}, max={actions_33d.max():.3f}", "cyan")

    return df, actions_33d


def make_text_embedding(model, device: torch.device, dtype: torch.dtype, seed: int = 42) -> torch.Tensor:
    """Generate synthetic text embedding (real text encoder not needed for this validation)."""
    torch.manual_seed(seed)
    # Production shape: [512, 4096] for UMT5 embeddings
    text_emb = torch.randn(1, 512, 4096, device=device, dtype=dtype)
    return text_emb


def compute_flow_matching_loss(
    model,
    encoder,
    video_latent: torch.Tensor,
    img_latent: torch.Tensor,
    robot_trajectory: torch.Tensor,
    text_embedding: torch.Tensor,
    null_embedding: torch.Tensor,
    timestep_idx: int,
    seed: int,
    flow_shift: float = 5.0,
    num_train_timesteps: int = 1000,
) -> torch.Tensor:
    """
    Compute production flow-matching loss for native_trajectory conditioning.

    Exact reproduction of rynnworld_teleop_trainer.py compute_loss() lines 897-1042,
    restricted to native_trajectory branch only.
    """
    device = video_latent.device
    dtype = video_latent.dtype

    # Generate noise with fixed seed
    torch.manual_seed(seed)
    noise = torch.randn_like(video_latent)

    # Flow-matching schedule
    s = timestep_idx / num_train_timesteps
    sigma_t = torch.tensor(flow_shift * s / (1 + (flow_shift - 1) * s), device=device, dtype=dtype)

    # Noisy latents
    noisy_latents = (1 - sigma_t) * video_latent + sigma_t * noise

    # Replace frame 0 with image latent
    noisy_latents[:, :, 0:1] = img_latent

    # Target
    target = noise - video_latent

    # Forward pass
    timestep_tensor = torch.tensor([timestep_idx], device=device)

    pred = model(
        hidden_states=noisy_latents,
        timestep=timestep_tensor,
        encoder_hidden_states=text_embedding,
        encoder_hidden_states_image=None,
        robot_trajectory=robot_trajectory,
        null_condition=False,
    ).sample

    # Timestep weight
    timestep_weight = torch.clamp(1.0 / (sigma_t * (1 - sigma_t) + 1e-5), max=10.0)
    timestep_weight = timestep_weight / timestep_weight.mean()

    # Loss on frames [1:] only
    loss_per_sample = torch.mean((pred[:, :, 1:] - target[:, :, 1:]) ** 2, dim=(1, 2, 3, 4))
    loss = torch.mean(timestep_weight * loss_per_sample)

    return loss


def save_video_mp4(video_tensor: torch.Tensor, save_path: Path, fps: int = 30):
    """
    Save video tensor to MP4 file.

    Args:
        video_tensor: [C, F, H, W] or [F, H, W, C] tensor in [-1, 1]
        save_path: output path
        fps: frames per second
    """
    import imageio

    # Convert to [F, H, W, C] if needed
    if video_tensor.shape[0] == 3:
        video_tensor = video_tensor.permute(1, 2, 3, 0)

    # Denormalize to [0, 255] uint8
    video_np = ((video_tensor.cpu().float().numpy() + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)

    imageio.mimsave(str(save_path), video_np, fps=fps)
    cprint(f"[Video] Saved {video_np.shape[0]} frames to {save_path}", "green")


def run_single_clip_finetune(
    model,
    encoder,
    vae,
    video_latent: torch.Tensor,
    img_latent: torch.Tensor,
    robot_trajectory: torch.Tensor,
    text_embedding: torch.Tensor,
    null_embedding: torch.Tensor,
    lr: float,
    steps: int,
    seed: int,
    out_dir: Path,
) -> Dict[str, Any]:
    """Fine-tune adapter on one video clip for a fixed number of steps."""
    device = video_latent.device
    dtype = model.patch_embedding.weight.dtype

    optimizer = AdamW(encoder.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-2)

    losses = []
    grad_norms = []

    cprint(f"\n[Finetune] Starting: {steps} steps, LR={lr:.2e}, seed={seed}", "yellow")

    for step in range(steps):
        optimizer.zero_grad()

        # Sample random timestep
        timestep_idx = torch.randint(0, 1000, (1,)).item()

        loss = compute_flow_matching_loss(
            model, encoder, video_latent, img_latent,
            robot_trajectory, text_embedding, null_embedding,
            timestep_idx, seed + step,
        )

        loss.backward()

        # Clip gradients
        grad_norm = torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)

        optimizer.step()

        losses.append(loss.item())
        grad_norms.append(grad_norm.item())

        if step % 10 == 0 or step == steps - 1:
            cprint(f"  Step {step:04d}: loss={loss.item():.6f}, grad_norm={grad_norm.item():.4f}", "cyan")

    # Save final checkpoint
    ckpt_path = out_dir / "checkpoints" / f"adapter_step{steps:04d}.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(encoder.state_dict(), ckpt_path)
    cprint(f"[Finetune] Saved checkpoint: {ckpt_path}", "green")

    return {
        "losses": losses,
        "grad_norms": grad_norms,
        "final_loss": losses[-1],
        "mean_loss_last10": np.mean(losses[-10:]),
        "checkpoint": str(ckpt_path),
    }


def generate_rollout(
    model,
    encoder,
    vae,
    img_latent: torch.Tensor,
    robot_trajectory: torch.Tensor,
    text_embedding: torch.Tensor,
    null_embedding: torch.Tensor,
    num_inference_steps: int,
    seed: int,
    null_condition: bool = False,
) -> torch.Tensor:
    """
    Generate video rollout using flow-matching sampling.

    Returns:
        video_frames: [C, F, H, W] tensor in [-1, 1]
    """
    device = img_latent.device
    dtype = model.patch_embedding.weight.dtype
    B, C, _, H, W = img_latent.shape
    F = 5  # num_frames for latent (production uses 5 latent frames for video generation)

    # Initialize latents with noise
    torch.manual_seed(seed)
    latents = torch.randn(B, C, F, H, W, device=device, dtype=dtype)
    latents[:, :, 0:1] = img_latent

    # Flow-matching sampling (simplified Euler method)
    flow_shift = 5.0
    timesteps = torch.linspace(999, 0, num_inference_steps, device=device).long()

    with torch.no_grad():
        for i, t in enumerate(timesteps):
            s = t / 1000.0
            sigma_t = flow_shift * s / (1 + (flow_shift - 1) * s)

            pred = model(
                hidden_states=latents,
                timestep=t.unsqueeze(0),
                encoder_hidden_states=text_embedding,
                encoder_hidden_states_image=None,
                robot_trajectory=robot_trajectory if not null_condition else None,
                null_condition=null_condition,
            ).sample

            # Euler step
            dt = -1.0 / num_inference_steps
            latents = latents + pred * dt

            latents[:, :, 0:1] = img_latent

    # Decode to video
    video_frames = decode_latents_to_video(vae, latents)

    return video_frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True, help="Path to extracted episode video (MP4)")
    parser.add_argument("--parquet_path", type=str, required=True, help="Path to episode parquet")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Pretrained model checkpoint dir")
    parser.add_argument("--base_model_dir", type=str, required=True, help="Base model dir (for VAE/text_encoder)")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=100, help="Number of fine-tuning steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--height", type=int, default=480, help="Video height")
    parser.add_argument("--width", type=int, default=832, help="Video width")
    parser.add_argument("--max_frames", type=int, default=21, help="Max frames to load from video")
    args = parser.parse_args()

    video_path = Path(args.video_path)
    parquet_path = Path(args.parquet_path)
    checkpoint_dir = Path(args.checkpoint_dir)
    base_model_dir = Path(args.base_model_dir)
    out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "videos").mkdir(exist_ok=True)
    (out_dir / "checkpoints").mkdir(exist_ok=True)

    # Apply monkey-patch
    apply_monkey_patch(_REPO_ROOT)

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Load VAE
    vae = load_vae(base_model_dir, device, dtype)

    # Load and preprocess video
    cprint(f"\n[Video] Loading from {video_path}", "yellow")
    raw_frames, video_metadata = load_and_decode_video(video_path, max_frames=args.max_frames)
    video_frames = preprocess_video_frames(raw_frames, args.height, args.width)
    video_frames = video_frames.unsqueeze(0)  # Add batch dim: [1, C, F, H, W]

    cprint(f"[Video] Preprocessed: shape={video_frames.shape}, range=[{video_frames.min():.3f}, {video_frames.max():.3f}]", "cyan")

    # Encode video to latent
    cprint(f"\n[VAE] Encoding video to latent...", "yellow")
    video_latent = encode_video_frames(vae, video_frames)
    cprint(f"[VAE] Encoded latent: shape={video_latent.shape}", "green")

    # Extract first frame as image latent
    img_latent = video_latent[:, :, 0:1]

    # Load episode parquet and extract 33D actions
    cprint(f"\n[Data] Loading episode data from {parquet_path}", "yellow")
    df, actions_33d = load_episode_data(parquet_path, video_metadata)

    # Create trajectory window (use all frames as one window for this single-clip validation)
    robot_trajectory = torch.from_numpy(actions_33d).float().to(device).unsqueeze(0)  # [1, T, 33]
    cprint(f"[Data] Robot trajectory: shape={robot_trajectory.shape}", "green")

    # Load model
    cprint(f"\n[Model] Loading from {checkpoint_dir}", "yellow")
    out_dict = {}
    model = load_model(checkpoint_dir, base_model_dir, out_dict)

    # Build adapter
    encoder = build_adapter_33d(device)
    model.native_trajectory_encoder = encoder

    # Generate synthetic text embeddings
    text_embedding = make_text_embedding(model, device, dtype, args.seed)
    null_embedding = make_text_embedding(model, device, dtype, args.seed + 1)

    # Fine-tune
    cprint(f"\n{'='*80}", "yellow")
    cprint(f"STARTING SINGLE-CLIP FINE-TUNE", "yellow")
    cprint(f"{'='*80}", "yellow")

    finetune_result = run_single_clip_finetune(
        model, encoder, vae,
        video_latent, img_latent, robot_trajectory,
        text_embedding, null_embedding,
        lr=args.lr, steps=args.steps, seed=args.seed,
        out_dir=out_dir,
    )

    # Generate rollouts
    cprint(f"\n{'='*80}", "yellow")
    cprint(f"GENERATING ROLLOUTS", "yellow")
    cprint(f"{'='*80}", "yellow")

    # Save target (real video)
    target_video_path = out_dir / "videos" / "target_real.mp4"
    save_video_mp4(video_frames[0], target_video_path, fps=int(video_metadata["fps"]))

    # C3 correct action
    cprint(f"\n[Rollout] C3 correct action", "yellow")
    c3_correct = generate_rollout(
        model, encoder, vae, img_latent, robot_trajectory,
        text_embedding, null_embedding,
        num_inference_steps=50, seed=args.seed + 100,
    )
    save_video_mp4(c3_correct[0], out_dir / "videos" / "c3_correct_action.mp4", fps=int(video_metadata["fps"]))

    # C0 no action
    cprint(f"\n[Rollout] C0 no action", "yellow")
    c0_no_action = generate_rollout(
        model, encoder, vae, img_latent, robot_trajectory,
        text_embedding, null_embedding,
        num_inference_steps=50, seed=args.seed + 200,
        null_condition=True,
    )
    save_video_mp4(c0_no_action[0], out_dir / "videos" / "c0_no_action.mp4", fps=int(video_metadata["fps"]))

    # C3 shuffled action (random permutation of timesteps)
    cprint(f"\n[Rollout] C3 shuffled action", "yellow")
    shuffled_trajectory = robot_trajectory.clone()
    perm = torch.randperm(shuffled_trajectory.shape[1])
    shuffled_trajectory = shuffled_trajectory[:, perm]
    c3_shuffled = generate_rollout(
        model, encoder, vae, img_latent, shuffled_trajectory,
        text_embedding, null_embedding,
        num_inference_steps=50, seed=args.seed + 300,
    )
    save_video_mp4(c3_shuffled[0], out_dir / "videos" / "c3_shuffled_action.mp4", fps=int(video_metadata["fps"]))

    # C3 wrong action (episode 3 actions if available, else random noise)
    cprint(f"\n[Rollout] C3 wrong action", "yellow")
    wrong_trajectory = torch.randn_like(robot_trajectory) * 0.5  # Gaussian noise scaled to roughly match action magnitudes
    c3_wrong = generate_rollout(
        model, encoder, vae, img_latent, wrong_trajectory,
        text_embedding, null_embedding,
        num_inference_steps=50, seed=args.seed + 400,
    )
    save_video_mp4(c3_wrong[0], out_dir / "videos" / "c3_wrong_action.mp4", fps=int(video_metadata["fps"]))

    # Save metrics
    metrics = {
        "run_id": "direct_action/gate_c/run_002",
        "video_metadata": video_metadata,
        "finetune": finetune_result,
        "video_latent_shape": list(video_latent.shape),
        "robot_trajectory_shape": list(robot_trajectory.shape),
        "num_frames": video_metadata["frame_count"],
        "steps": args.steps,
        "lr": args.lr,
        "seed": args.seed,
    }

    metrics_path = out_dir / "artifacts" / "gate_c_visual_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    cprint(f"\n{'='*80}", "green")
    cprint(f"GATE C RUN_002 COMPLETE", "green")
    cprint(f"{'='*80}", "green")
    cprint(f"Final loss: {finetune_result['final_loss']:.6f}", "green")
    cprint(f"Videos saved to: {out_dir / 'videos'}", "green")
    cprint(f"Metrics saved to: {metrics_path}", "green")


if __name__ == "__main__":
    main()
