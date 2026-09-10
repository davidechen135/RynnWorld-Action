#!/usr/bin/env python3
"""
Gate C run_003: Visual validation with REAL UMT5 text encoding and baseline inference test.

Key fixes from run_002:
- Use real UMT5 T5TokenizerFast + UMT5EncoderModel for text encoding (not torch.randn)
- Baseline inference test BEFORE training (verify pretrained model outputs coherent frames)
- Extended training: 50/200/500/1000 steps with checkpoint + rollout at each milestone
- Pass criteria: C3 correct-action video shows stable subject/viewpoint, no collapse, better than C0

Standing guardrails:
- Never use data/agibot_362_native_smoke or episodes 649616/650872/650989
- 33D channel order: raw index (end/pos, end/ori, joint/pos, waist/pos)
- No parquet duplication into repo tree
- Torch 2.7.0a0+7c8ec84dab.nv25.03 (no replacement)
- Preserve pre-existing dirty changes
- No commits/pushes
- Gate D not started until Gate C high-quality result + independent review
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from termcolor import cprint
from transformers import T5TokenizerFast, UMT5EncoderModel

# Add repo root to path
repo_root = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(repo_root))


def apply_monkey_patch(repo_root: Path):
    """
    Import rynnworld_teleop_trainer to apply wan_forward monkey-patch.
    Verify that the patch was actually applied by checking function identity.
    """
    from diffusers import WanTransformer3DModel as _WTM

    # Capture original forward id
    original_forward_id = id(_WTM.forward)

    # Import trainer to apply monkey-patch
    sys.path.insert(0, str(repo_root / "core" / "finetune" / "models" / "wan_i2v"))
    from core.finetune.models.wan_i2v import rynnworld_teleop_trainer

    # Verify patch was applied
    patched_forward_id = id(_WTM.forward)
    if original_forward_id == patched_forward_id:
        raise RuntimeError(
            "wan_forward monkey-patch verification FAILED: "
            "WanTransformer3DModel.forward identity unchanged after importing rynnworld_teleop_trainer. "
            "The robot_trajectory conditioning path will not work."
        )

    cprint("[Gate C run_003] wan_forward monkey-patch verified", "green")


def load_text_encoder(base_model_dir: Path, device: torch.device, dtype: torch.dtype):
    """
    Load UMT5 text encoder and tokenizer for real text conditioning.

    Returns:
        (tokenizer, text_encoder)
    """
    cprint(f"[Text Encoder] Loading UMT5 from {base_model_dir}/text_encoder", "cyan")
    tokenizer = T5TokenizerFast.from_pretrained(base_model_dir, subfolder="tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(base_model_dir, subfolder="text_encoder")
    text_encoder = text_encoder.to(device=device, dtype=dtype)
    text_encoder.eval()
    text_encoder.requires_grad_(False)
    cprint(f"[Text Encoder] Loaded, device={device}, dtype={dtype}", "cyan")
    return tokenizer, text_encoder


def encode_text(tokenizer, text_encoder, prompt: str, device: torch.device) -> torch.Tensor:
    """
    Encode text prompt using UMT5.

    Args:
        tokenizer: T5TokenizerFast
        text_encoder: UMT5EncoderModel
        prompt: text string to encode
        device: torch device

    Returns:
        text_embedding: [1, 512, 4096] tensor
    """
    with torch.no_grad():
        prompt_token_ids = tokenizer(
            prompt,
            padding="max_length",
            max_length=512,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        prompt_token_ids = prompt_token_ids.input_ids.to(device)
        prompt_embedding = text_encoder(prompt_token_ids)[0]  # [1, 512, 4096]
    return prompt_embedding


def load_vae(base_model_dir: Path, device: torch.device, dtype: torch.dtype):
    """Load VAE for video encoding/decoding."""
    from diffusers import AutoencoderKLWan

    cprint(f"[VAE] Loading from {base_model_dir}/vae", "cyan")
    vae = AutoencoderKLWan.from_pretrained(base_model_dir, subfolder="vae")
    vae = vae.to(device=device, dtype=dtype)
    vae.eval()
    vae.requires_grad_(False)

    # Extract latents_mean and latents_std
    latents_mean = torch.tensor(vae.config.latents_mean, device=device, dtype=dtype).view(1, -1, 1, 1, 1)
    latents_std = torch.tensor(vae.config.latents_std, device=device, dtype=dtype).view(1, -1, 1, 1, 1)

    cprint(f"[VAE] Device: {device}, dtype: {dtype}", "cyan")
    cprint(f"[VAE] latents_mean: {vae.config.latents_mean[:5]}... shape {len(vae.config.latents_mean)}", "cyan")
    cprint(f"[VAE] latents_std: {vae.config.latents_std[:5]}... shape {len(vae.config.latents_std)}", "cyan")

    return vae, latents_mean, latents_std


def encode_video_frames(vae, video_frames: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor) -> torch.Tensor:
    """
    Encode video frames to latent using VAE.

    Args:
        vae: AutoencoderKLWan
        video_frames: [B, C, F, H, W] in [-1, 1]
        latents_mean: [1, 48, 1, 1, 1]
        latents_std: [1, 48, 1, 1, 1]

    Returns:
        latent: [B, 48, F//4, H//16, W//16] normalized
    """
    with torch.no_grad():
        latent_dist = vae.encode(video_frames).latent_dist
        latent = latent_dist.sample()
        # Normalize
        latent_normalized = (latent - latents_mean) / latents_std
    return latent_normalized


def decode_latents_to_video(vae, latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor) -> torch.Tensor:
    """
    Decode latents back to video frames.

    Args:
        vae: AutoencoderKLWan
        latents: [B, 48, F, H, W] normalized
        latents_mean, latents_std: normalization params

    Returns:
        video_frames: [B, C, F, H, W] in [-1, 1]
    """
    with torch.no_grad():
        # Denormalize
        latents_denorm = latents * latents_std + latents_mean
        video_frames = vae.decode(latents_denorm).sample
    return video_frames


def load_and_decode_video(video_path: Path, max_frames: int = None, start_frame: int = 0) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Load and decode video using imageio with ffmpeg backend.

    IMPORTANT: Loads CONSECUTIVE frames starting from start_frame, not uniformly sampled.

    Args:
        video_path: path to MP4 file
        max_frames: if specified, load this many CONSECUTIVE frames starting from start_frame
        start_frame: frame index to start from (default 0 = beginning of video)

    Returns:
        frames: [F, H, W, C] uint8 tensor
        metadata: dict with fps, frame_count, etc.
    """
    import imageio.v3 as iio

    # Read video properties
    props = iio.improps(video_path, plugin="pyav")
    total_frames = props.n_images

    # Get FPS from metadata
    video_metadata = iio.immeta(video_path, plugin="pyav")
    fps = video_metadata.get("fps", 30)  # Default to 30 if not found

    # Read CONSECUTIVE frames from start_frame
    if max_frames is not None:
        end_frame = min(start_frame + max_frames, total_frames)
        frames_list = []
        for idx in range(start_frame, end_frame):
            frame = iio.imread(video_path, index=idx, plugin="pyav")
            frames_list.append(frame)
        frames = np.stack(frames_list, axis=0)
        cprint(f"[Video] Loaded {len(frames)} CONSECUTIVE frames from index {start_frame} to {end_frame-1}", "cyan")
    else:
        # Read all frames
        frames = iio.imread(video_path, plugin="pyav")

    frames_tensor = torch.from_numpy(frames)

    metadata = {
        "fps": fps,
        "frame_count": len(frames_tensor),
        "total_frames": total_frames,
        "height": frames_tensor.shape[1],
        "width": frames_tensor.shape[2],
        "start_frame": start_frame,
    }

    return frames_tensor, metadata


def preprocess_video_frames(frames: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """
    Preprocess video frames: resize, normalize to [-1, 1], permute to [C, F, H, W].

    Args:
        frames: [F, H, W, C] uint8 tensor
        height, width: target resolution

    Returns:
        video: [C, F, H, W] float tensor in [-1, 1]
    """
    # Permute to [F, C, H, W]
    video = frames.permute(0, 3, 1, 2).float() / 255.0  # [F, C, H, W] in [0, 1]

    # Resize if needed
    if video.shape[2] != height or video.shape[3] != width:
        video = F.interpolate(video, size=(height, width), mode="bilinear", align_corners=False)

    # Normalize to [-1, 1]
    video = video * 2.0 - 1.0

    # Permute to [C, F, H, W]
    video = video.permute(1, 0, 2, 3)

    return video


def load_model(checkpoint_dir: Path, base_model_dir: Path, out) -> torch.nn.Module:
    """
    Load WanTransformer3DModel with EMA weights.

    Args:
        checkpoint_dir: RynnWorld-Teleop-Causal directory with ema_weights.bin
        base_model_dir: Wan2.2-TI2V-5B-Diffusers directory with transformer config
        out: output stream for logging

    Returns:
        model: frozen WanTransformer3DModel
    """
    from diffusers import WanTransformer3DModel

    print(f"[Model] Loading from {checkpoint_dir}", file=out)

    # Load base transformer
    model = WanTransformer3DModel.from_pretrained(base_model_dir, subfolder="transformer")

    # Load EMA weights
    ema_path = checkpoint_dir / "ema_weights.bin"
    ema_state_dict = torch.load(ema_path, map_location="cpu")
    model.load_state_dict(ema_state_dict, strict=False)
    print(f"[Model] Loaded EMA weights from {ema_path.name}", file=out)

    # Move to device and freeze
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    model = model.to(device=device, dtype=dtype)
    model.eval()
    model.requires_grad_(False)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] Loaded {total_params/1e9:.2f}B params, dtype={dtype}, device={device}", file=out)

    return model


def build_adapter_33d(device: torch.device) -> torch.nn.Module:
    """
    Build NativeTrajectoryEncoder adapter for 33D actions.

    Returns:
        encoder: NativeTrajectoryEncoder with zero-init output_projection
    """
    from core.control.native_trajectory_encoder import NativeTrajectoryEncoder

    encoder = NativeTrajectoryEncoder(input_dim=33).to(device)

    # Verify zero-init on output_projection
    output_weight = encoder.output_projection.weight.data
    output_bias = encoder.output_projection.bias.data if encoder.output_projection.bias is not None else None

    assert torch.allclose(output_weight, torch.zeros_like(output_weight), atol=1e-6), \
        "output_projection.weight is not zero-initialized"
    if output_bias is not None:
        assert torch.allclose(output_bias, torch.zeros_like(output_bias), atol=1e-6), \
            "output_projection.bias is not zero-initialized"

    total_params = sum(p.numel() for p in encoder.parameters())
    cprint(f"[Adapter] NativeTrajectoryEncoder(input_dim=33): {total_params:,} params, zero-init verified", "cyan")

    return encoder


def extract_33d_action(action_40d: np.ndarray) -> np.ndarray:
    """
    Extract 33D action from 40D: end/position[2:8], end/orientation[8:16], joint/position[16:30], waist/position[33:38].

    Args:
        action_40d: [T, 40] array

    Returns:
        action_33d: [T, 33] array
    """
    end_pos = action_40d[:, 2:8]      # 6
    end_ori = action_40d[:, 8:16]     # 8
    joint_pos = action_40d[:, 16:30]  # 14
    waist_pos = action_40d[:, 33:38]  # 5
    action_33d = np.concatenate([end_pos, end_ori, joint_pos, waist_pos], axis=1)
    return action_33d


def load_episode_data(parquet_path: Path, video_metadata: Dict[str, Any]):
    """
    Load episode parquet and extract 33D actions, clipping to match video frame count.

    Returns:
        (df, actions_33d)
    """
    import pandas as pd

    df = pd.read_parquet(parquet_path)

    # Clip parquet to match video frame count if needed
    num_frames = video_metadata["frame_count"]
    if len(df) > num_frames:
        cprint(f"[Data] Clipping parquet from {len(df)} to {num_frames} rows to match video", "yellow")
        df = df.iloc[:num_frames]

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


def compute_flow_matching_loss(
    model,
    encoder,
    video_latent: torch.Tensor,
    img_latent: torch.Tensor,
    robot_trajectory: torch.Tensor,
    text_embedding: torch.Tensor,
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
    video_latent: torch.Tensor,
    img_latent: torch.Tensor,
    robot_trajectory: torch.Tensor,
    text_embedding: torch.Tensor,
    steps: int,
    lr: float,
    seed: int,
    out_dir: Path,
    checkpoint_interval: int = 10,
    global_step_start: int = 0,
):
    """
    Fine-tune the adapter on a single clip.

    Returns:
        dict with losses, grad_norms, checkpoints
    """
    device = video_latent.device

    # Ensure encoder parameters require grad
    encoder.train()
    for param in encoder.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(encoder.parameters(), lr=lr)

    losses = []
    grad_norms = []
    checkpoints = []

    cprint(f"\n{'='*80}", "cyan")
    cprint("STARTING SINGLE-CLIP FINE-TUNE", "cyan")
    cprint(f"{'='*80}\n", "cyan")
    cprint(
        f"[Finetune] Starting: {steps} steps, LR={lr:.2e}, seed={seed}, "
        f"global_step_start={global_step_start}",
        "cyan",
    )

    encoder.train()

    for step in range(steps):
        # Random timestep for this step
        timestep_idx = torch.randint(0, 1000, (1,)).item()

        # Compute loss
        loss = compute_flow_matching_loss(
            model, encoder, video_latent, img_latent, robot_trajectory,
            text_embedding, timestep_idx, seed
        )

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Grad norm
        grad_norm = torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)

        optimizer.step()

        losses.append(loss.item())
        grad_norms.append(grad_norm.item())

        if step % 10 == 0 or step == steps - 1:
            print(f"  Step {step:04d}: loss={loss.item():.6f}, grad_norm={grad_norm.item():.4f}")

        # Save checkpoint at intervals
        if (step + 1) % checkpoint_interval == 0 or step == steps - 1:
            global_step = global_step_start + step + 1
            ckpt_path = out_dir / "checkpoints" / f"adapter_step{global_step:04d}.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(encoder.state_dict(), ckpt_path)
            checkpoints.append(str(ckpt_path))
            if (step + 1) % 50 == 0 or step == steps - 1:
                cprint(f"[Finetune] Saved checkpoint: {ckpt_path}", "green")

    encoder.eval()

    return {
        "losses": losses,
        "grad_norms": grad_norms,
        "final_loss": losses[-1],
        "mean_loss_last10": np.mean(losses[-10:]),
        "checkpoints": checkpoints,
    }


def generate_rollout(
    model,
    encoder,
    vae,
    img_latent: torch.Tensor,
    robot_trajectory: torch.Tensor,
    text_embedding: torch.Tensor,
    latents_mean: torch.Tensor,
    latents_std: torch.Tensor,
    num_inference_steps: int,
    seed: int,
    num_latent_frames: int,
    null_condition: bool = False,
) -> torch.Tensor:
    """
    Generate video rollout using flow-matching sampling.

    Args:
        num_latent_frames: total latent frame count, MUST match the training
            video_latent temporal dim (not img_latent's, which is always 1).

    Returns:
        video_frames: [C, F, H, W] tensor in [-1, 1]
    """
    device = img_latent.device
    dtype = model.patch_embedding.weight.dtype
    B, C, _, H, W = img_latent.shape
    F = num_latent_frames

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
    video_frames = decode_latents_to_video(vae, latents, latents_mean, latents_std)

    return video_frames[0]  # Return [C, F, H, W]


def baseline_inference_test(
    model,
    encoder,
    vae,
    img_latent: torch.Tensor,
    robot_trajectory: torch.Tensor,
    text_embedding: torch.Tensor,
    latents_mean: torch.Tensor,
    latents_std: torch.Tensor,
    out_dir: Path,
    num_latent_frames: int,
    seed: int = 42,
):
    """
    Baseline inference test: generate rollout with ZERO training steps.
    This verifies that the pretrained model + real conditioning produces coherent frames.

    If output is pure noise, the pipeline is broken and training should not proceed.

    Returns:
        baseline_video: [C, F, H, W] tensor
    """
    cprint(f"\n{'='*80}", "yellow")
    cprint("BASELINE INFERENCE TEST (ZERO TRAINING)", "yellow")
    cprint("Verifying pretrained model + real conditioning produces coherent output", "yellow")
    cprint(f"{'='*80}\n", "yellow")

    # Generate with adapter at zero-init (contributes nothing)
    baseline_video = generate_rollout(
        model, encoder, vae, img_latent, robot_trajectory, text_embedding,
        latents_mean, latents_std, num_inference_steps=20, seed=seed,
        num_latent_frames=num_latent_frames, null_condition=False
    )

    # Save baseline video
    baseline_path = out_dir / "videos" / "baseline_pretrained.mp4"
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    save_video_mp4(baseline_video, baseline_path, fps=30)

    cprint(f"[Baseline] Saved to {baseline_path}", "green")
    cprint("[Baseline] INSPECT THIS VIDEO MANUALLY:", "yellow")
    cprint("  - If coherent frames with stable subject/viewpoint: PASS, proceed to training", "yellow")
    cprint("  - If pure noise/collapsed: FAIL, pipeline broken, do not train", "yellow")

    return baseline_video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True, help="Path to extracted episode video (MP4)")
    parser.add_argument("--parquet_path", type=str, required=True, help="Path to episode parquet")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Pretrained model checkpoint dir")
    parser.add_argument("--base_model_dir", type=str, required=True, help="Base model dir (Wan2.2-TI2V-5B-Diffusers)")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--height", type=int, default=480, help="Video height")
    parser.add_argument("--width", type=int, default=832, help="Video width")
    parser.add_argument("--max_frames", type=int, default=21, help="Max frames to sample from video")
    parser.add_argument("--training_milestones", type=str, default="500,1000,2000,3000", help="Training step milestones")
    parser.add_argument("--skip_baseline", action="store_true", help="Skip baseline inference test")

    args = parser.parse_args()

    video_path = Path(args.video_path)
    parquet_path = Path(args.parquet_path)
    checkpoint_dir = Path(args.checkpoint_dir)
    base_model_dir = Path(args.base_model_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    training_milestones = [int(x) for x in args.training_milestones.split(",")]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    out = sys.stdout

    # Apply monkey-patch
    apply_monkey_patch(repo_root)

    # Load text encoder
    tokenizer, text_encoder = load_text_encoder(base_model_dir, device, dtype)

    # Encode empty string to get null_embedding
    cprint("[Text] Encoding empty string for null_embedding", "cyan")
    null_embedding = encode_text(tokenizer, text_encoder, "", device)
    cprint(f"[Text] null_embedding shape: {null_embedding.shape}", "cyan")

    # Load VAE
    vae, latents_mean, latents_std = load_vae(base_model_dir, device, dtype)

    # Load and preprocess video
    cprint(f"\n[Video] Loading from {video_path}", "cyan")
    frames, video_metadata = load_and_decode_video(video_path, max_frames=args.max_frames)
    video_preprocessed = preprocess_video_frames(frames, args.height, args.width).unsqueeze(0).to(device, dtype=dtype)
    cprint(f"[Video] Preprocessed: shape={video_preprocessed.shape}, range=[{video_preprocessed.min():.3f}, {video_preprocessed.max():.3f}]", "cyan")

    # Encode video to latent
    cprint("\n[VAE] Encoding video to latent...", "cyan")
    video_latent = encode_video_frames(vae, video_preprocessed, latents_mean, latents_std)
    cprint(f"[VAE] Encoded latent: shape={video_latent.shape}", "cyan")

    # Extract first frame as img_latent
    img_latent = video_latent[:, :, 0:1].clone()

    # Load episode data
    cprint(f"\n[Data] Loading episode data from {parquet_path}", "cyan")
    df, actions_33d = load_episode_data(parquet_path, video_metadata)

    # Convert to tensor
    robot_trajectory = torch.from_numpy(actions_33d).unsqueeze(0).to(device, dtype=torch.float32)
    cprint(f"[Data] Robot trajectory: shape={robot_trajectory.shape}", "cyan")

    # Load model
    model = load_model(checkpoint_dir, base_model_dir, out)

    # Build adapter
    encoder = build_adapter_33d(device)

    # Attach encoder to model for wan_forward to recognize it
    model.native_trajectory_encoder = encoder
    cprint("[Adapter] Attached to model.native_trajectory_encoder", "cyan")

    # Save real target video
    target_path = out_dir / "videos" / "target_real.mp4"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    save_video_mp4(video_preprocessed[0], target_path, fps=int(video_metadata["fps"]))

    # Baseline inference test (unless skipped)
    if not args.skip_baseline:
        baseline_video = baseline_inference_test(
            model, encoder, vae, img_latent, robot_trajectory, null_embedding,
            latents_mean, latents_std, out_dir, num_latent_frames=video_latent.shape[2], seed=args.seed
        )
        cprint("\n[Baseline] Baseline test complete. Review baseline_pretrained.mp4 before proceeding.", "yellow")
        cprint("[Baseline] If baseline is noise, STOP HERE and debug pipeline.", "red")
        input("\nPress Enter to proceed with training, or Ctrl+C to abort...")

    # Training loop: cumulative steps with checkpoint + rollout at each milestone
    cumulative_steps = 0
    all_results = []

    for milestone in training_milestones:
        steps_this_phase = milestone - cumulative_steps

        cprint(f"\n{'='*80}", "magenta")
        cprint(f"TRAINING MILESTONE: {milestone} steps (running {steps_this_phase} more steps)", "magenta")
        cprint(f"{'='*80}\n", "magenta")

        # Fine-tune
        finetune_result = run_single_clip_finetune(
            model, encoder, video_latent, img_latent, robot_trajectory,
            null_embedding, steps=steps_this_phase, lr=args.lr, seed=args.seed,
            out_dir=out_dir,
            checkpoint_interval=10,
            global_step_start=cumulative_steps,
        )

        cumulative_steps = milestone

        # Generate rollouts with trained adapter
        cprint(f"\n{'='*80}", "green")
        cprint(f"GENERATING ROLLOUTS AT {milestone} STEPS", "green")
        cprint(f"{'='*80}\n", "green")

        # C3 correct action
        cprint("[Rollout] C3 correct action", "green")
        c3_correct = generate_rollout(
            model, encoder, vae, img_latent, robot_trajectory, null_embedding,
            latents_mean, latents_std, num_inference_steps=20, seed=args.seed,
            num_latent_frames=video_latent.shape[2], null_condition=False
        )
        save_video_mp4(c3_correct, out_dir / "videos" / f"step{milestone:04d}_c3_correct.mp4", fps=int(video_metadata["fps"]))

        # C0 no action
        cprint("[Rollout] C0 no action", "green")
        c0_no_action = generate_rollout(
            model, encoder, vae, img_latent, robot_trajectory, null_embedding,
            latents_mean, latents_std, num_inference_steps=20, seed=args.seed,
            num_latent_frames=video_latent.shape[2], null_condition=True
        )
        save_video_mp4(c0_no_action, out_dir / "videos" / f"step{milestone:04d}_c0_no_action.mp4", fps=int(video_metadata["fps"]))

        # C3 shuffled action
        cprint("[Rollout] C3 shuffled action", "green")
        shuffled_traj = robot_trajectory.clone()
        shuffled_traj[0] = shuffled_traj[0][torch.randperm(shuffled_traj.shape[1])]
        c3_shuffled = generate_rollout(
            model, encoder, vae, img_latent, shuffled_traj, null_embedding,
            latents_mean, latents_std, num_inference_steps=20, seed=args.seed,
            num_latent_frames=video_latent.shape[2], null_condition=False
        )
        save_video_mp4(c3_shuffled, out_dir / "videos" / f"step{milestone:04d}_c3_shuffled.mp4", fps=int(video_metadata["fps"]))

        # C3 wrong action (use reversed trajectory as proxy)
        cprint("[Rollout] C3 wrong action (reversed)", "green")
        reversed_traj = robot_trajectory.clone()
        reversed_traj[0] = torch.flip(reversed_traj[0], dims=[0])
        c3_wrong = generate_rollout(
            model, encoder, vae, img_latent, reversed_traj, null_embedding,
            latents_mean, latents_std, num_inference_steps=20, seed=args.seed,
            num_latent_frames=video_latent.shape[2], null_condition=False
        )
        save_video_mp4(c3_wrong, out_dir / "videos" / f"step{milestone:04d}_c3_wrong.mp4", fps=int(video_metadata["fps"]))

        all_results.append({
            "milestone": milestone,
            "final_loss": finetune_result["final_loss"],
            "mean_loss_last10": finetune_result["mean_loss_last10"],
            "losses": finetune_result["losses"],
            "grad_norms": finetune_result["grad_norms"],
        })

    # Save metrics
    metrics_path = out_dir / "artifacts" / "gate_c_visual_v2_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump({
            "training_milestones": training_milestones,
            "results": all_results,
            "config": {
                "lr": args.lr,
                "seed": args.seed,
                "height": args.height,
                "width": args.width,
                "max_frames": args.max_frames,
            }
        }, f, indent=2)

    cprint(f"\n{'='*80}", "cyan")
    cprint("GATE C RUN_003 COMPLETE", "cyan")
    cprint(f"{'='*80}", "cyan")
    cprint(f"Videos saved to: {out_dir / 'videos'}", "green")
    cprint(f"Metrics saved to: {metrics_path}", "green")
    cprint("\nNext: Visually inspect rollout videos at each milestone.", "yellow")
    cprint("Pass criteria: C3 correct-action shows stable subject/viewpoint, no collapse, better than C0/shuffled/wrong.", "yellow")


if __name__ == "__main__":
    main()