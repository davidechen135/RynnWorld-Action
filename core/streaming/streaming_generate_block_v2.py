"""
streaming_generate_block_v2.py — Block streaming with FixedSizeCache.

Generates video latents block by block. Within each block frames attend
bidirectionally; across blocks a causal KV cache is maintained.
FixedSizeCache pre-allocates the KV buffer and writes in place, keeping
tensor shapes invariant across recompute for gradient_checkpointing
compatibility. RoPE is baked into K before the cache write using absolute
frame positions.

Aligned with the Self-Forcing training pipeline.
"""

import os
from typing import Optional

import torch
from diffusers.utils.torch_utils import randn_tensor
from core.streaming.cache import FixedSizeCache


def _streaming_generate_block_v2(
    generator,
    scheduler,
    img_latent,
    prompt_embeds,
    negative_prompt_embeds,
    num_latent_frames: int,
    num_inference_steps: int,
    guidance_scale: float,
    do_cfg: bool,
    generator_rng: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
    patch_size,
    with_grad: bool,
    num_max_frames: int = 21,         # FixedSizeCache buffer size (frames)
    pe_mode: str = "absolute",          # absolute positions (matches Self-Forcing)
    stochastic_grad_truncation: bool = False,
    control_video_latent: Optional[torch.Tensor] = None,
    num_frame_per_block: int = 3,
    context_noise: int = 0,
    sink_size: int = 1,
    local_attn_size: int = -1,         # -1 = no eviction (full cache)
):
    """Block-level streaming generation with FixedSizeCache.

    KV cache is pre-allocated with shape [B, num_max_frames * frame_seq_len,
    H, D] and updated by in-place index assignments. RoPE is applied to K
    before the cache write, so cached K already carries absolute positional
    encoding.

    Returns: [B, C, num_latent_frames, H, W] generated latents (frame 0 = condition)
    """
    batch_size = 1
    _, num_channels, _, latent_h, latent_w = img_latent.shape
    p_t, p_h, p_w = patch_size
    frame_seq_len = (latent_h // p_h) * (latent_w // p_w)
    _frame_seq_h = latent_h // p_h
    _frame_seq_w = latent_w // p_w

    condition = img_latent.to(device=device, dtype=dtype)  # [B, C, 1, H, W]

    # ── Initialize FixedSizeCache ──
    # Resolve generator.config: prefer .module.config (DeepSpeed engine has its own
    # .config dict that would shadow the inner model's config).
    if hasattr(generator, 'module') and hasattr(generator.module, 'config'):
        cfg = generator.module.config
    elif hasattr(generator, 'config') and hasattr(generator.config, 'num_attention_heads'):
        cfg = generator.config
    else:
        raise RuntimeError("Cannot find generator.config for cache setup")

    num_heads = cfg.num_attention_heads
    head_dim = cfg.attention_head_dim
    num_layers = cfg.num_layers

    max_tokens = num_max_frames * frame_seq_len

    cache_cond = FixedSizeCache(
        batch_size=batch_size, num_layers=num_layers,
        num_heads=num_heads, head_dim=head_dim,
        max_tokens=max_tokens, dtype=dtype, device=device,
        sink_size=sink_size, local_attn_size=local_attn_size,
        tokens_per_frame=frame_seq_len,
    )
    cache_uncond = FixedSizeCache(
        batch_size=batch_size, num_layers=num_layers,
        num_heads=num_heads, head_dim=head_dim,
        max_tokens=max_tokens, dtype=dtype, device=device,
        sink_size=sink_size, local_attn_size=local_attn_size,
        tokens_per_frame=frame_seq_len,
    ) if do_cfg else None

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

    # ── Build position_ids for a block (absolute frame index, baked into RoPE) ──
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

    # ── Model forward (with CFG) ──
    def _forward(hidden_states, timestep, position_ids, ctrl,
                 current_start, current_end):
        akw = {
            "fixed_cache": cache_cond,
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
        if do_cfg and cache_uncond is not None:
            akw_uncond = {
                "fixed_cache": cache_uncond,
                "current_start": current_start,
                "current_end": current_end,
                "tokens_per_frame": frame_seq_len,
            }
            v_uncond = generator(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=negative_prompt_embeds,
                encoder_hidden_states_image=None,
                attention_kwargs=akw_uncond,
                position_ids=position_ids,
                return_dict=False,
                control_video_latent=ctrl,
            )[0]
            v_pred = v_uncond + guidance_scale * (v_pred - v_uncond)
        return v_pred

    # ── SGT: select one spatial denoising step ──
    if with_grad and stochastic_grad_truncation and num_inference_steps > 1:
        try:
            import torch.distributed as _dist
            if _dist.is_available() and _dist.is_initialized():
                _gsi = torch.empty(1, dtype=torch.long, device=device)
                if _dist.get_rank() == 0:
                    _gsi.fill_(int(torch.randint(0, num_inference_steps, (1,)).item()))
                _dist.broadcast(_gsi, src=0)
                grad_step_idx = int(_gsi.item())
            else:
                grad_step_idx = int(torch.randint(0, num_inference_steps, (1,)).item())
        except Exception:
            grad_step_idx = int(torch.randint(0, num_inference_steps, (1,)).item())
    else:
        grad_step_idx = num_inference_steps - 1 if with_grad else -1

    num_train_timesteps = scheduler.config.num_train_timesteps

    # ── Seed cache with condition (frame 0) ──
    cond_pos = _build_pos_ids(0, 1)
    cond_t = torch.zeros(batch_size, frame_seq_len, dtype=torch.long, device=device)
    with torch.no_grad():
        _forward(
            condition.to(dtype), cond_t, cond_pos, _ctrl_block(0, 1),
            current_start=0, current_end=frame_seq_len,
        )

    # ── Block-level temporal loop ──
    collected = [condition]
    _sched_cls = type(scheduler)
    _sched_config = scheduler.config

    for block_start in range(1, num_latent_frames, num_frame_per_block):
        block_size = min(num_frame_per_block, num_latent_frames - block_start)

        # Fresh scheduler per block (UniPC has internal state)
        frame_scheduler = _sched_cls(**_sched_config)
        frame_scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = frame_scheduler.timesteps

        # Initial noise [B, C, block_size, H, W]
        block_shape = (batch_size, num_channels, block_size, latent_h, latent_w)
        noisy_input = randn_tensor(block_shape, generator=generator_rng,
                                   device=device, dtype=dtype)

        # Position IDs for this block (absolute frame index)
        position_ids = _build_pos_ids(block_start, block_size)

        # Control for this block
        ctrl = _ctrl_block(block_start, block_size)

        # current_start/end (in tokens) for this block's K/V write
        block_tokens = block_size * frame_seq_len
        current_start = block_start * frame_seq_len
        current_end = current_start + block_tokens

        # ── Spatial denoising loop ──
        denoised_pred = None
        for step_idx, t in enumerate(timesteps):
            block_t = torch.full(
                (batch_size, block_tokens),
                t.item(), dtype=torch.long, device=device,
            )

            is_exit = (step_idx == grad_step_idx) if (with_grad and stochastic_grad_truncation) \
                      else (step_idx == num_inference_steps - 1)

            if not is_exit:
                # Non-exit: no_grad ODE step
                with torch.no_grad():
                    cache_cond.freeze()
                    if cache_uncond is not None:
                        cache_uncond.freeze()
                    v_pred = _forward(
                        noisy_input.to(dtype), block_t, position_ids, ctrl,
                        current_start=current_start, current_end=current_end,
                    )
                    noisy_input = frame_scheduler.step(v_pred, t, noisy_input, return_dict=False)[0]
            else:
                # Exit step (SGT)
                cache_cond.freeze()
                if cache_uncond is not None:
                    cache_uncond.freeze()
                if with_grad:
                    with torch.enable_grad():
                        v_pred = _forward(
                            noisy_input.to(dtype), block_t, position_ids, ctrl,
                            current_start=current_start, current_end=current_end,
                        )
                        sigma_t = t.float() / num_train_timesteps
                        denoised_pred = noisy_input - sigma_t * v_pred
                else:
                    with torch.no_grad():
                        v_pred = _forward(
                            noisy_input.to(dtype), block_t, position_ids, ctrl,
                            current_start=current_start, current_end=current_end,
                        )
                        sigma_t = t.float() / num_train_timesteps
                        denoised_pred = noisy_input - sigma_t * v_pred
                break

        collected.append(denoised_pred)

        # ── Persist clean K/V (unfreeze + final forward at t=context_noise) ──
        clean_input = denoised_pred.detach().to(dtype)
        if context_noise > 0:
            ctx_sigma = context_noise / num_train_timesteps
            clean_input = (1 - ctx_sigma) * clean_input + ctx_sigma * torch.randn_like(clean_input)

        clean_t = torch.full(
            (batch_size, block_tokens),
            context_noise, dtype=torch.long, device=device,
        )
        with torch.no_grad():
            cache_cond.unfreeze()
            if cache_uncond is not None:
                cache_uncond.unfreeze()
            _forward(
                clean_input, clean_t, position_ids, ctrl,
                current_start=current_start, current_end=current_end,
            )

    output = torch.cat(collected, dim=2)
    return output


def generate_streaming_block_v2(*args, **kwargs):
    """No-grad wrapper (for critic data generation)."""
    kwargs.setdefault("with_grad", False)
    with torch.no_grad():
        return _streaming_generate_block_v2(*args, **kwargs)


def generate_streaming_block_with_grad_v2(*args, **kwargs):
    """Grad-enabled wrapper for DMD generator step.
    
    Also enables gradient_checkpointing on the generator (safe with FixedSizeCache
    because cache tensor shape is invariant across recompute calls).
    """
    kwargs["with_grad"] = True
    generator = kwargs.get("generator")
    # Resolve module (handle DeepSpeed engine / FSDP wrap)
    gen_mod = generator
    if hasattr(generator, 'module'):
        gen_mod = generator.module
    
    was_enabled = getattr(gen_mod, 'gradient_checkpointing', False)
    if not was_enabled and hasattr(gen_mod, 'enable_gradient_checkpointing'):
        gen_mod.enable_gradient_checkpointing()
    try:
        return _streaming_generate_block_v2(*args, **kwargs)
    finally:
        if not was_enabled and hasattr(gen_mod, 'disable_gradient_checkpointing'):
            gen_mod.disable_gradient_checkpointing()
