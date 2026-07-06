"""
streaming_v_flow_block.py — Block-level MSE warmup forward.

Runs a v-flow matching MSE forward pass block by block. Each block covers
num_frame_per_block frames, uses FixedSizeCache for the KV state, and applies
RoPE to K in absolute-position space before the cache write. Compatible with
gradient_checkpointing.

Training pattern (per block):
  (a) noisy block forward (cache frozen, with grad) → v_pred for this block
  (b) clean block forward (cache unfrozen, no grad) → write clean GT K/V

v_target is (noise - GT).
"""

import os
from typing import Optional

import torch
from diffusers.utils.torch_utils import randn_tensor
from core.streaming.cache import FixedSizeCache


def streaming_v_flow_matching_forward_block(
    generator,
    scheduler,
    gt_latents,                    # [1, C, F, H, W] — full GT video latent
    prompt_embeds,                 # [1, seq, dim]
    generator_rng: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
    patch_size,
    control_video_latent: Optional[torch.Tensor] = None,
    num_frame_per_block: int = 3,
    num_max_frames: int = 21,
    sink_size: int = 1,
    local_attn_size: int = -1,
    trace_dump: bool = False,
    trace_tag: str = "v_flow_block",
):
    """Block-level v_flow MSE forward with FixedSizeCache + absolute PE.

    Returns:
        v_pred_all: [1, C, F-1, H, W] — predicted v for frames 1..F-1
        v_target:   [1, C, F-1, H, W] — target v (noise - GT)
        x0_for_decode: [1, C, F-1, H, W] — debug only, x0 reconstruction
    """
    batch_size = 1
    _, num_channels, F, latent_h, latent_w = gt_latents.shape
    p_t, p_h, p_w = patch_size
    frame_seq_len = (latent_h // p_h) * (latent_w // p_w)
    _frame_seq_h = latent_h // p_h
    _frame_seq_w = latent_w // p_w

    # Resolve module config (handle DeepSpeed engine wrap)
    if hasattr(generator, 'module') and hasattr(generator.module, 'config'):
        cfg = generator.module.config
        gen_mod = generator.module
    elif hasattr(generator, 'config') and hasattr(generator.config, 'num_attention_heads'):
        cfg = generator.config
        gen_mod = generator
    else:
        raise RuntimeError("Cannot find generator.config")

    num_heads = cfg.num_attention_heads
    head_dim = cfg.attention_head_dim
    num_layers = cfg.num_layers

    # Enable gradient_checkpointing for memory savings.
    # Critical: without this, block=3 + 5B + 21F OOMs on 95GB H20.
    # Leave it enabled across calls (cheap idempotent op).
    if not getattr(gen_mod, 'gradient_checkpointing', False):
        if hasattr(gen_mod, 'enable_gradient_checkpointing'):
            gen_mod.enable_gradient_checkpointing()

    max_tokens = num_max_frames * frame_seq_len

    # Single cache (no CFG in MSE warmup)
    cache = FixedSizeCache(
        batch_size=batch_size, num_layers=num_layers,
        num_heads=num_heads, head_dim=head_dim,
        max_tokens=max_tokens, dtype=dtype, device=device,
        sink_size=sink_size, local_attn_size=local_attn_size,
        tokens_per_frame=frame_seq_len,
    )

    # ── Control slicing helper ──
    def _ctrl_block(block_start, block_size):
        if control_video_latent is None:
            return None
        end = min(block_start + block_size, control_video_latent.shape[2])
        ctrl = control_video_latent[:, :, block_start:end].to(device=device, dtype=dtype)
        if ctrl.shape[2] < block_size:
            pad = ctrl[:, :, -1:].expand(-1, -1, block_size - ctrl.shape[2], -1, -1)
            ctrl = torch.cat([ctrl, pad], dim=2)
        return ctrl

    # ── Build position_ids (absolute frame index) ──
    def _build_pos_ids(frame_start, num_frames):
        pos_ids = []
        for f_offset in range(num_frames):
            frame_pos = torch.cartesian_prod(
                torch.arange(frame_start + f_offset, frame_start + f_offset + 1,
                             dtype=torch.long, device=device),
                torch.arange(_frame_seq_h, dtype=torch.long, device=device),
                torch.arange(_frame_seq_w, dtype=torch.long, device=device),
            )
            pos_ids.append(frame_pos)
        return torch.cat(pos_ids, dim=0).unsqueeze(0).repeat(batch_size, 1, 1)

    # ── Generator forward helper ──
    def _forward(hidden_states, timestep, position_ids, ctrl,
                 current_start, current_end):
        akw = {
            "fixed_cache": cache,
            "current_start": current_start,
            "current_end": current_end,
            "tokens_per_frame": frame_seq_len,
        }
        v_pred = generator(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            encoder_hidden_states_image=None,
            attention_kwargs=akw,
            position_ids=position_ids,
            return_dict=False,
            control_video_latent=ctrl,
        )[0]
        return v_pred

    # ── σ sampling: one σ per frame ──
    num_train_timesteps = scheduler.config.num_train_timesteps

    # Sample per-frame timesteps: shape [B, F]
    timesteps_per_frame = torch.randint(
        0, num_train_timesteps, (batch_size, F),
        device=device, generator=generator_rng if generator_rng.device.type == 'cuda' else None,
    )
    if generator_rng.device.type != 'cuda':
        timesteps_per_frame = torch.randint(
            0, num_train_timesteps, (batch_size, F),
            generator=generator_rng,
        ).to(device)

    # Force frame 0 to t=0 (condition frame, no noise)
    timesteps_per_frame[:, 0] = 0

    # Optional timestep_shift mapping (matches original FLOW_SHIFT env var)
    _flow_shift = float(os.environ.get("FLOW_SHIFT", "1.0"))
    if _flow_shift != 1.0:
        _t_norm = timesteps_per_frame.float() / num_train_timesteps
        _t_norm = _flow_shift * _t_norm / (1 + (_flow_shift - 1) * _t_norm)
        timesteps_per_frame = (_t_norm * num_train_timesteps).long().clamp(0, num_train_timesteps - 1)
        timesteps_per_frame[:, 0] = 0  # ensure frame 0 stays clean

    sigma_per_frame = (timesteps_per_frame.float() / num_train_timesteps).to(dtype)  # [B, F]

    # Add noise to gt_latents: noisy_latents[:, :, f] = (1-σ_f) * GT[f] + σ_f * noise[f]
    noise = randn_tensor(gt_latents.shape, generator=generator_rng,
                         device=device, dtype=dtype)
    sigma_5d = sigma_per_frame.view(batch_size, 1, F, 1, 1).to(dtype)
    noisy_latents = (1 - sigma_5d) * gt_latents.to(dtype) + sigma_5d * noise

    # Ensure frame 0 stays clean
    noisy_latents[:, :, 0:1] = gt_latents[:, :, 0:1].to(dtype)

    # ── Seed cache with condition (frame 0) ──
    cond_pos = _build_pos_ids(0, 1)
    cond_t = torch.zeros(batch_size, frame_seq_len, dtype=torch.long, device=device)
    with torch.no_grad():
        _forward(
            gt_latents[:, :, 0:1].to(dtype), cond_t, cond_pos, _ctrl_block(0, 1),
            current_start=0, current_end=frame_seq_len,
        )

    # ── Block-level temporal loop ──
    v_preds = []  # per-block v_pred tensors

    for block_start in range(1, F, num_frame_per_block):
        block_size = min(num_frame_per_block, F - block_start)
        block_tokens = block_size * frame_seq_len
        current_start = block_start * frame_seq_len
        current_end = current_start + block_tokens

        position_ids = _build_pos_ids(block_start, block_size)
        ctrl = _ctrl_block(block_start, block_size)

        # Build per-token timestep [B, block_tokens]
        # Each token of frame f gets timesteps_per_frame[B, f]
        block_t = torch.zeros(batch_size, block_tokens, dtype=torch.long, device=device)
        for f_offset in range(block_size):
            t_val = timesteps_per_frame[:, block_start + f_offset].unsqueeze(1)  # [B, 1]
            block_t[:, f_offset * frame_seq_len:(f_offset + 1) * frame_seq_len] = t_val

        # ── (a) Noisy block forward (cache frozen, with grad) ──
        cache.freeze()
        with torch.enable_grad():
            v_pred = _forward(
                noisy_latents[:, :, block_start:block_start + block_size].to(dtype),
                block_t, position_ids, ctrl,
                current_start=current_start, current_end=current_end,
            )
        # v_pred shape: [B, C, block_size, H, W] — same as input
        v_preds.append(v_pred)

        # ── (b) Clean GT block forward (cache unfrozen, no grad) ──
        # Writes clean GT K/V into cache for next block's attention.
        clean_t = torch.zeros(batch_size, block_tokens, dtype=torch.long, device=device)
        with torch.no_grad():
            cache.unfreeze()
            _forward(
                gt_latents[:, :, block_start:block_start + block_size].to(dtype),
                clean_t, position_ids, ctrl,
                current_start=current_start, current_end=current_end,
            )

    # ── Assemble v_pred_all and v_target ──
    v_pred_all = torch.cat(v_preds, dim=2)  # [1, C, F-1, H, W]
    v_target = (noise - gt_latents.to(dtype))[:, :, 1:]  # [1, C, F-1, H, W]

    # x_0 reconstruction for debug decode (no grad)
    with torch.no_grad():
        noisy_for_recon = noisy_latents[:, :, 1:].to(v_pred_all.dtype)
        sigma_recon = sigma_5d[:, :, 1:].to(v_pred_all.dtype)
        x0_for_decode = (noisy_for_recon - sigma_recon * v_pred_all).detach()

    return v_pred_all, v_target, x0_for_decode
