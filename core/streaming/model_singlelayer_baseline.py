"""
Streaming causal transformer for Wan2.2-TI2V-5B.

Extends `diffusers.models.transformers.transformer_wan.WanTransformer3DModel`
with a causal sliding-window KV cache in the style of StreamingLLM
(Xiao et al. 2024) and LongLive (Yang et al. 2025). Baseline transformer,
attention, and pipeline classes are imported from HuggingFace diffusers.

Attribution:
- Base transformer / attention: diffusers `transformer_wan.py`
  (https://github.com/huggingface/diffusers).
- Sliding-window KV cache & absolute-slot position assignment: LongLive
  (https://github.com/NVlabs/LongLive) and StreamingLLM
  (https://github.com/mit-han-lab/streaming-llm).
- Individual borrowed code paths are annotated inline.
"""
import inspect
import itertools
import math
import os
from copy import deepcopy
from typing import Any, Dict, Optional, Union, List, Tuple

import diffusers
import torch
import torch.nn as nn
import torch.nn.functional as F
from termcolor import cprint
from diffusers.pipelines.pipeline_loading_utils import _fetch_class_library_tuple
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.transformers.transformer_wan import (
    WanTimeTextImageEmbedding,
    WanTransformer3DModel,
    FP32LayerNorm,
    WanAttnProcessor,
    WanAttention,
    FeedForward,
    _get_qkv_projections,
)
from diffusers.models.embeddings import get_1d_rotary_pos_embed
from diffusers.video_processor import VideoProcessor
from diffusers.models.autoencoders.autoencoder_kl_wan import (
    AutoencoderKLWan,
)
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from flash_attn import flash_attn_func
try:
    from magi_attention.api import flex_flash_attn_func
except ImportError:
    flex_flash_attn_func = None  # only needed for non-cached attention paths
from transformers import AutoTokenizer, UMT5EncoderModel

from .cache import DynamicCache


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def _apply_rotary_emb(
    hidden_states: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> torch.Tensor:
    """RoPE application for [B, seq, heads, head_dim] tensors.

    Lifted out of WanCausalAttnProcessor.__call__ as a module-level function so
    it can be reused by the sliding-window branch (which applies different
    rotaries to Q vs the combined K).
    """
    x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]
    out = torch.empty_like(hidden_states)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out.type_as(hidden_states)


class WanCausalAttnProcessor(WanAttnProcessor):
    def __init__(self, *args, layer_idx: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer_idx = layer_idx

    def __call__(
        self,
        attn: "WanAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Dict[str, Any] = {},
    ) -> torch.Tensor:
        assert encoder_hidden_states is None
        assert attention_mask is None

        query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        # ── Sliding-window KV cache (LongLive / StreamingLLM-style) ──
        # Caller signals via attention_kwargs:
        #   "sliding_mode": True
        #   "sliding_k_rotary": (cos, sin) for the combined K (cache + new), with
        #                       positions assigned by SLOT in the rolling window
        #                       (not original frame index). Computed in
        #                       WanCausalTransformer3DModel.forward from
        #                       attention_kwargs["sliding_k_position_ids"].
        # In sliding mode we store RAW K (no RoPE baked in) so that after a
        # cache trim the kept K's can be re-positioned in the next forward
        # without numerical loss. This eliminates the train-test position drift
        # that legacy mode suffers when the cache trims frames whose RoPE was
        # baked in at original (now-stale) positions.
        sliding_mode = (
            attention_kwargs is not None
            and attention_kwargs.get("sliding_mode", False)
        )

        # FixedSizeCache path: pre-allocated buffer with in-place writes.
        # Triggered by attention_kwargs["fixed_cache"] = FixedSizeCache(...).
        fixed_cache_mode = (
            attention_kwargs is not None
            and attention_kwargs.get("fixed_cache", None) is not None
        )

        if fixed_cache_mode:
            # Apply RoPE to Q and K before cache write so cache shape stays
            # invariant across recompute (gradient_checkpointing safe).
            if rotary_emb is not None:
                query = _apply_rotary_emb(query, *rotary_emb)
                key = _apply_rotary_emb(key, *rotary_emb)

            cache = attention_kwargs["fixed_cache"]
            current_start = attention_kwargs["current_start"]
            current_end = attention_kwargs["current_end"]
            tokens_per_frame = attention_kwargs.get("tokens_per_frame", None)

            key_full, value_full = cache.update(
                key, value, self.layer_idx,
                current_start=current_start,
                current_end=current_end,
                tokens_per_frame=tokens_per_frame,
            )

            q_dtype = query.dtype
            query = query.to(torch.bfloat16)
            key_full = key_full.to(torch.bfloat16)
            value_full = value_full.to(torch.bfloat16)
            hidden_states = flash_attn_func(
                query, key_full, value_full,
                causal=False, softmax_scale=None,
            )
            hidden_states = hidden_states.to(q_dtype)

        elif sliding_mode:
            # ── Sliding path ──
            # 1. Apply rotary to Q using its (slot-based) position
            if rotary_emb is not None:
                query = _apply_rotary_emb(query, *rotary_emb)
            # 2. Update cache with RAW K (no rotary). Returns the full
            #    cache + new K concatenation, also raw.
            cache = attention_kwargs["past_key_values"]
            key_full_raw, value_full = cache.update(key, value, self.layer_idx)
            # 3. Apply slot-position rotary to the entire K stack. Cache K's
            #    that were trimmed/re-numbered get fresh rotary every forward,
            #    so there's no positional drift across trims.
            sliding_k_rotary = attention_kwargs["sliding_k_rotary"]
            key_full = _apply_rotary_emb(key_full_raw, *sliding_k_rotary)
            # 4. Flash attention (bf16, non-causal — bidir within the window)
            q_dtype = query.dtype
            query = query.to(torch.bfloat16)
            key_full = key_full.to(torch.bfloat16)
            value_full = value_full.to(torch.bfloat16)
            hidden_states = flash_attn_func(
                query, key_full, value_full,
                causal=False, softmax_scale=None,
            )
            hidden_states = hidden_states.to(q_dtype)

        elif rotary_emb is not None:
            # ── Legacy path: rotary applied to both Q and K, cache stores RoPE'd K ──
            query = _apply_rotary_emb(query, *rotary_emb)
            key = _apply_rotary_emb(key, *rotary_emb)

        if not sliding_mode and not fixed_cache_mode and attention_kwargs is not None and "past_key_values" in attention_kwargs:
            # Diagnostic: when caller stamps trace_pe=True (set by train_distill
            # only when pe_mode=absolute AND first-iter trace), print per-frame
            # K-after-RoPE absmax + cache K seqlen so we can confirm
            # (a) K's RoPE is baked PRE-update (next forward sees baked K
            #     even after trim, matching LongLive/SF/Distillation), and
            # (b) cache K seqlen grows then plateaus at cap*tokens_per_frame.
            # Gated layer-0 only to keep the log readable.
            _cache = attention_kwargs["past_key_values"]
            _trace_pe = (
                attention_kwargs.get("trace_pe", False)
                and self.layer_idx == 0
            )
            if _trace_pe:
                _q_absmax = float(query.abs().max().item())
                _k_absmax = float(key.abs().max().item())
                _cache_k_pre = (
                    _cache.keys[self.layer_idx].shape[1]
                    if self.layer_idx in _cache.keys else 0
                )
                _fc_pre = _cache._frame_count.get(self.layer_idx, 0)
                _frozen = _cache.is_frozen
                print(
                    f"  [trace:abs_pe:L0] pre-update | "
                    f"new_K seqlen={key.shape[1]} | cache_K seqlen={_cache_k_pre} | "
                    f"fc_pre={_fc_pre} | frozen={_frozen} | "
                    f"Q absmax={_q_absmax:.3f} | K-after-RoPE absmax={_k_absmax:.3f}",
                    flush=True,
                )
            key, value = _cache.update(key, value, self.layer_idx)
            if _trace_pe:
                _cache_k_post = (
                    _cache.keys[self.layer_idx].shape[1]
                    if self.layer_idx in _cache.keys else 0
                )
                _fc_post = _cache._frame_count.get(self.layer_idx, 0)
                # When unfrozen and frame_count is capped at max_frames, that's
                # the trim signal (frame_count was supposed to grow but didn't).
                _capped = (_cache.max_frames is not None and _fc_post == _cache.max_frames and _fc_pre >= _cache.max_frames)
                _trim_marker = " [TRIM applied]" if _capped else ""
                print(
                    f"  [trace:abs_pe:L0] post-update | "
                    f"cache_K seqlen={_cache_k_post} | fc_post={_fc_post}{_trim_marker}",
                    flush=True,
                )
            # Ensure proper dtype for flash attention
            q_dtype = query.dtype
            query = query.to(torch.bfloat16)
            key = key.to(torch.bfloat16)
            value = value.to(torch.bfloat16)
            hidden_states = flash_attn_func(
                query,
                key,
                value,
                causal=False,
                softmax_scale=None,
            )
            hidden_states = hidden_states.to(q_dtype)
        elif sliding_mode or fixed_cache_mode:
            # already handled above, fall through to projection
            pass
        elif attention_kwargs is not None and "q_ranges" in attention_kwargs:
            # magi_attention path — used by the action-conditioned training pipeline
            # to express custom causal mask ranges across frames.
            assert flex_flash_attn_func is not None, "magi_attention not installed but q_ranges path requested"
            hidden_states, _ = flex_flash_attn_func(
                query[0],
                key[0],
                value[0],
                q_ranges=attention_kwargs["q_ranges"],
                k_ranges=attention_kwargs["k_ranges"],
                max_seqlen_q=attention_kwargs["max_seqlen_q"],
                attn_type_map=attention_kwargs["attn_type_map"],
                softmax_scale=None, # defaults to 1/sqrt(head_dim)
            )
            hidden_states = hidden_states[None]
        else:
            # Plain non-causal flash attention path — used by the DMD critic /
            # real_score forwards (no KV cache, no ranges; the whole video is one
            # bidirectional chunk). This is what attention_kwargs=None means.
            q_dtype = query.dtype
            query = query.to(torch.bfloat16)
            key = key.to(torch.bfloat16)
            value = value.to(torch.bfloat16)
            hidden_states = flash_attn_func(
                query, key, value, causal=False, softmax_scale=None,
            )
            hidden_states = hidden_states.to(q_dtype)

        hidden_states = hidden_states.flatten(2)
        # Convert to match the output projection layer weight dtype
        target_dtype = attn.to_out[0].weight.dtype
        hidden_states = hidden_states.to(target_dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class WanCausalTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
        layer_idx: int = 0,
    ):
        super().__init__()

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim_head=None,
            processor=WanCausalAttnProcessor(layer_idx=layer_idx),
        )

        # 2. Cross-attention
        self.attn2 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            added_kv_proj_dim=added_kv_proj_dim,
            cross_attention_dim_head=dim // num_heads,
            processor=WanAttnProcessor(),
        )
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 3. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
        attention_kwargs: Dict[str, Any],
    ) -> torch.Tensor:
        if temb.ndim == 4:
            # temb: batch_size, seq_len, 6, inner_dim (wan2.2 ti2v)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table.unsqueeze(0) + temb.float()
            ).chunk(6, dim=2)
            # batch_size, seq_len, 1, inner_dim
            shift_msa = shift_msa.squeeze(2)
            scale_msa = scale_msa.squeeze(2)
            gate_msa = gate_msa.squeeze(2)
            c_shift_msa = c_shift_msa.squeeze(2)
            c_scale_msa = c_scale_msa.squeeze(2)
            c_gate_msa = c_gate_msa.squeeze(2)
        else:
            # temb: batch_size, 6, inner_dim (wan2.1/wan2.2 14B)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table + temb.float()
            ).chunk(6, dim=1)

        # 1. Self-attention
        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_output = self.attn1(norm_hidden_states, None, None, rotary_emb, attention_kwargs=attention_kwargs)
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

        # 2. Cross-attention
        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(norm_hidden_states, encoder_hidden_states, None, None)
        hidden_states = hidden_states + attn_output

        # 3. Feed-forward
        norm_hidden_states = (self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(
            hidden_states
        )
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)

        return hidden_states


class WanRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        attention_head_dim: int,
        patch_size: Tuple[int, int, int],
        max_seq_len: int,
        theta: float = 10000.0,
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len

        h_dim = w_dim = 2 * (attention_head_dim // 6)
        t_dim = attention_head_dim - h_dim - w_dim

        self.t_dim = t_dim
        self.h_dim = h_dim
        self.w_dim = w_dim

        freqs_dtype = torch.float32 if torch.backends.mps.is_available() else torch.float64

        freqs_cos = []
        freqs_sin = []

        for dim in [t_dim, h_dim, w_dim]:
            freq_cos, freq_sin = get_1d_rotary_pos_embed(
                dim,
                max_seq_len,
                theta,
                use_real=True,
                repeat_interleave_real=True,
                freqs_dtype=freqs_dtype,
            )
            freqs_cos.append(freq_cos)
            freqs_sin.append(freq_sin)

        self.register_buffer("freqs_cos", torch.cat(freqs_cos, dim=1), persistent=False)
        self.register_buffer("freqs_sin", torch.cat(freqs_sin, dim=1), persistent=False)

    def forward(self, position_ids: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        split_sizes = [self.t_dim, self.h_dim, self.w_dim]
        freqs_cos = self.freqs_cos.split(split_sizes, dim=1)
        freqs_sin = self.freqs_sin.split(split_sizes, dim=1)

        freqs_cos = torch.cat(
            [freqs_cos[i][position_ids[..., i]] for i in range(3)],
            dim=-1,
        ).unsqueeze(-2)
        freqs_sin = torch.cat(
            [freqs_sin[i][position_ids[..., i]] for i in range(3)],
            dim=-1,
        ).unsqueeze(-2)

        return freqs_cos, freqs_sin


class WanCausalTransformer3DModel(WanTransformer3DModel):
    @register_to_config
    def __init__(
        self,
        patch_size: Tuple[int, ...] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        qk_norm: Optional[str] = "rms_norm_across_heads",
        eps: float = 1e-6,
        image_dim: Optional[int] = None,
        added_kv_proj_dim: Optional[int] = None,
        rope_max_seq_len: int = 1024,
        pos_embed_seq_len: Optional[int] = None,
    ) -> None:
        super(WanTransformer3DModel, self).__init__()

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        # 1. Patch & position embedding
        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.patch_embedding = nn.Conv3d(in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)

        # Control patch embedding (for control-guided generation, e.g. skeleton-guided).
        # Uses add-plus mode: hidden = patch_embedding(video) + control_scale * control_patch_embedding(skeleton).
        # CPE: Conv3d(in_ch=48, out_ch=3072) — SAME in_ch as patch_embedding (not 2×).
        # init: weight.zero_(), bias.zero_() — start contributing nothing, learn from 0.
        # control_scale: nn.Parameter(0.1, learnable) — gating to prevent control
        # from overwhelming base path early in training.
        # Created lazily by init_control_patch_embedding(). Both remain None when
        # not initialized (no impact on non-control runs like Run C/E3/richneg/Run E).
        self.control_patch_embedding: Optional[nn.Conv3d] = None
        self.control_scale: Optional[nn.Parameter] = None

        # 2. Condition embeddings
        # image_embedding_dim=1280 for I2V model
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
            image_embed_dim=image_dim,
            pos_embed_seq_len=pos_embed_seq_len,
        )

        # 3. Transformer blocks
        self.blocks = nn.ModuleList(
            [
                WanCausalTransformerBlock(
                    inner_dim, ffn_dim, num_attention_heads, qk_norm, cross_attn_norm, eps, added_kv_proj_dim, i
                )
                for i in range(num_layers)
            ]
        )

        # 4. Output norm & projection
        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)

        self.gradient_checkpointing = False

    def init_control_patch_embedding(self):
        """Initialize control_patch_embedding for ADD-PLUS control fusion.

        Semantics (control_type=add-plus):
          - Conv3d with SAME in_channels as patch_embedding (NOT 2× like concat).
          - weight.zero_(), bias.zero_() — CPE contributes nothing initially,
            learns control mapping from 0.
          - Adds learnable control_scale = nn.Parameter(0.1) — gating to prevent
            control signal from overwhelming base path early in training.

        Forward (see model.py forward() below):
            hidden = patch_embedding(noisy_video)
                   + control_scale * control_patch_embedding(skeleton_latent)

        Note: skeleton latent must be DISTRIBUTION-NORMALIZED to match video
        latent stats BEFORE being passed in (see train_control_distill_v2.py
        running-stats EMA: skel = (skel - c_run_mean) / c_run_std * v_std + v_mean).
        Without normalize, skeleton's mostly-white-background latent has a
        very different distribution → CPE output dominates wrongly.
        """
        patch_embedding = self.patch_embedding
        in_channels = patch_embedding.in_channels
        out_channels = patch_embedding.out_channels
        kernel_size = patch_embedding.kernel_size
        device = patch_embedding.weight.device
        dtype = patch_embedding.weight.dtype

        control_patch_embedding = nn.Conv3d(
            in_channels,  # ADD-PLUS: same in_ch as patch_embedding (NOT 2×)
            out_channels,
            kernel_size=kernel_size,
            stride=kernel_size,
        ).to(device=device, dtype=dtype)

        with torch.no_grad():
            control_patch_embedding.weight.zero_()
            if control_patch_embedding.bias is not None:
                control_patch_embedding.bias.zero_()

        self.control_patch_embedding = control_patch_embedding
        self.control_patch_embedding.requires_grad_(True)
        # control_scale: learnable scalar, init 0.1 (matches reference)
        self.control_scale = nn.Parameter(
            torch.tensor(0.1, dtype=dtype, device=device)
        )
        self.control_scale.requires_grad_(True)
        cprint(
            f"[model] add-plus control_patch_embedding init: "
            f"in_ch={in_channels}, out_ch={out_channels}, kernel={kernel_size}, "
            f"weight=zero, bias=zero, control_scale=0.1 (learnable)",
            "green",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        attention_kwargs: Dict[str, Any],
        position_ids: Optional[torch.LongTensor] = None,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        control_video_latent: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        if position_ids is None:
            position_ids = torch.cartesian_prod(
                torch.arange(post_patch_num_frames, dtype=torch.long, device=hidden_states.device),
                torch.arange(post_patch_height, dtype=torch.long, device=hidden_states.device),
                torch.arange(post_patch_width, dtype=torch.long, device=hidden_states.device),
            ).unsqueeze(0).repeat(batch_size, 1, 1)

        rotary_emb = self.rope(position_ids)

        # ── Sliding-window mode: compute K-side rotary ──
        # In sliding mode the K stack inside the attention processor is
        # (cache K + new K), with positions assigned by slot index in the
        # rolling window (not the absolute frame index). We compute that K
        # rotary here ONCE per forward (shared across all 30 transformer
        # layers via attention_kwargs) so each layer's processor can grab it
        # without re-doing the slot bookkeeping or rope buffer indexing.
        #
        # Caller (e.g. _streaming_generate) is responsible for setting:
        #   attention_kwargs["sliding_mode"] = True
        #   attention_kwargs["sliding_k_position_ids"] = [B, K_seq, 3]
        # where K_seq spans the combined cache+new K tokens in slot-order.
        #
        # We ALWAYS recompute (not conditional on absence) — the cost is
        # ~one indexing op + one cat, dwarfed by the actual block compute.
        # Always-recompute eliminates the stale-rotary class of bugs (e.g.
        # caller forgetting to pop the cached rotary between frames).
        if (
            attention_kwargs is not None
            and attention_kwargs.get("sliding_mode", False)
        ):
            k_pos_ids = attention_kwargs["sliding_k_position_ids"]
            attention_kwargs["sliding_k_rotary"] = self.rope(k_pos_ids)

        # Control-aware patch embedding (ADD-PLUS mode).
        # Semantics (control_type=add-plus):
        #   hidden = patch_embedding(video) + control_scale * control_patch_embedding(skeleton)
        # When control_video_latent is None or CPE not initialized → use regular patch_embedding
        # (no impact on non-control runs).
        if control_video_latent is not None and self.control_patch_embedding is not None:
            h_video = self.patch_embedding(hidden_states)
            h_control = self.control_patch_embedding(control_video_latent)
            # control_scale is a learnable Parameter created alongside CPE in
            # init_control_patch_embedding(); fallback to 1.0 if missing (defensive).
            _scale = self.control_scale if self.control_scale is not None else 1.0
            hidden_states = h_video + _scale * h_control
        else:
            hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        # timestep shape: batch_size, or batch_size, seq_len (wan 2.2 ti2v)
        if timestep.ndim == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()  # batch_size * seq_len
        else:
            ts_seq_len = None

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=ts_seq_len
        )
        if ts_seq_len is not None:
            # batch_size, seq_len, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            # batch_size, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        # 4. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    rotary_emb,
                    attention_kwargs,
                )
        else:
            for block in self.blocks:
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    rotary_emb,
                    attention_kwargs,
                )

        # 5. Output norm, projection & unpatchify
        if temb.ndim == 3:
            # batch_size, seq_len, inner_dim (wan 2.2 ti2v)
            shift, scale = (self.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            # batch_size, inner_dim
            shift, scale = (self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)

        # Move the shift and scale tensors to the same device as hidden_states.
        # When using multi-GPU inference via accelerate these will be on the
        # first device rather than the last device, which hidden_states ends up
        # on.
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)


def apply_monkey_patch():
    """Training pipeline for text+image to video, without action conditioning.

    Works with pre-encoded latents loaded from .safetensors files.
    Uses flow matching loss like the original Wan2.2 training.
    """

    config_name = "model_index.json"

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: UMT5EncoderModel,
        vae: AutoencoderKLWan,
        scheduler: FlowMatchEulerDiscreteScheduler,
        transformer: Optional[WanCausalTransformer3DModel] = None,
        transformer_2: Optional[WanCausalTransformer3DModel] = None,
        boundary_ratio: Optional[float] = None,
        expand_timesteps: bool = False,
    ):
        super().__init__()
        assert transformer_2 is None
        assert expand_timesteps

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
            transformer_2=transformer_2,
        )
        self.register_to_config(boundary_ratio=boundary_ratio)
        self.register_to_config(expand_timesteps=expand_timesteps)
        self.vae_scale_factor_temporal = self.vae.config.scale_factor_temporal if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = self.vae.config.scale_factor_spatial if getattr(self, "vae", None) else 8
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

        latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, self.vae.config.z_dim, 1, 1, 1)
        latents_std = torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1)
        self.register_buffer("latents_mean", latents_mean)
        self.register_buffer("latents_std", latents_std)

        for param in itertools.chain(self.vae.parameters(), self.text_encoder.parameters()):
            param.requires_grad_(False)

        self.prompt_embeds = None

    @property
    def config(self):
        return None

    def __call__(
        self,
        video_latents: List[torch.Tensor],
        text_embeds: List[torch.Tensor],
    ):
        assert len(video_latents) == 1

        transformer_device = next(self.transformer.parameters()).device
        transformer_dtype = self.transformer.dtype

        if text_embeds[0] is None:
            if self.prompt_embeds is None:
                prompt = ""
                text_inputs = self.tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=512,
                    truncation=True,
                    add_special_tokens=True,
                    return_attention_mask=True,
                    return_tensors="pt",
                )
                text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
                device = self.text_encoder.device
                self.prompt_embeds = self.text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
            encoder_hidden_states = self.prompt_embeds.to(device=transformer_device, dtype=transformer_dtype)
        else:
            encoder_hidden_states = text_embeds[0].to(device=transformer_device, dtype=transformer_dtype)

        latents = video_latents[0].to(device=transformer_device, dtype=transformer_dtype)
        device = latents.device
        batch_size = latents.shape[0]

        num_latent_frames, latent_height, latent_width = latents.shape[-3:]
        p_t, p_h, p_w = self.transformer.config.patch_size
        frame_seq_len = latent_height * latent_width // p_h // p_w

        # Create condition latent (first frame) — matches inference_sync.py
        condition = latents[:, :, 0:1, :, :]

        first_frame_mask = torch.ones(
            batch_size, 1, num_latent_frames, latent_height, latent_width,
            dtype=transformer_dtype, device=device,
        )
        first_frame_mask[:, :, 0] = 0

        # Single-step flow matching with shared timestep
        num_train_timesteps = self.scheduler.config.num_train_timesteps
        t = torch.randint(0, num_train_timesteps, (), device=device)  # single scalar timestep

        noise = torch.randn_like(latents)
        sigmas = (t / num_train_timesteps).to(transformer_dtype)
        noisy_latents = (1 - sigmas) * latents + sigmas * noise

        # Apply condition mask: condition frame is kept clean, other frames are noisy
        noisy_latents = (1 - first_frame_mask) * condition + first_frame_mask * noisy_latents

        # ── Teacher-forcing with clean KV cache ──
        # Matches inference_streaming.py: KV cache stores clean (denoised) representations.
        #
        # In inference_streaming.py:
        #   - Each frame is denoised over many steps with cache frozen
        #   - On the LAST step, cache unfreezes to store that frame's clean KV
        #   - Next frame then attends to accumulated clean KV
        #
        # Training equivalent:
        #   1. Seed cache with condition frame (clean, t=0)
        #   2. For each generated frame:
        #      a. Freeze cache → noisy forward → prediction (attends to clean context)
        #      b. Unfreeze cache → clean forward (no_grad) → persist clean KV
        attention_kwargs = {"past_key_values": DynamicCache()}

        # ── Step 0: Seed cache with condition frame ──
        cond_position_ids = torch.cartesian_prod(
            torch.arange(0, 1, dtype=torch.long, device=device),  # temporal pos 0
            torch.arange(latent_height // p_h, dtype=torch.long, device=device),
            torch.arange(latent_width // p_w, dtype=torch.long, device=device),
        ).unsqueeze(0).repeat(batch_size, 1, 1)
        cond_t = torch.zeros(batch_size, frame_seq_len, dtype=torch.long, device=device)

        with torch.no_grad():
            with self.transformer.cache_context("cond"):
                self.transformer(
                    hidden_states=condition.to(transformer_dtype),
                    timestep=cond_t,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_kwargs=attention_kwargs,
                    position_ids=cond_position_ids,
                    return_dict=False,
                )

        noise_preds = []
        for frame_idx in range(1, num_latent_frames):
            frame_slice = slice(frame_idx, frame_idx + 1)

            position_ids = torch.cartesian_prod(
                torch.arange(frame_idx, frame_idx + 1, dtype=torch.long, device=device),
                torch.arange(latent_height // p_h, dtype=torch.long, device=device),
                torch.arange(latent_width // p_w, dtype=torch.long, device=device),
            ).unsqueeze(0).repeat(batch_size, 1, 1)

            # ── Step 2a: Noisy forward with frozen cache → prediction ──
            attention_kwargs["past_key_values"].freeze()
            noisy_input = noisy_latents[:, :, frame_slice].to(transformer_dtype)
            t_chunk = t.unsqueeze(0).expand(batch_size, -1)

            with self.transformer.cache_context("cond"):
                pred = self.transformer(
                    hidden_states=noisy_input,
                    timestep=t_chunk,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_kwargs=attention_kwargs,
                    position_ids=position_ids,
                    return_dict=False,
                )[0]
            noise_preds.append(pred)

            # ── Step 2b: Clean forward to persist clean KV (no grad) ──
            attention_kwargs["past_key_values"].unfreeze()
            clean_input = latents[:, :, frame_slice].to(transformer_dtype)
            clean_t = torch.zeros(batch_size, frame_seq_len, dtype=torch.long, device=device)

            with torch.no_grad():
                with self.transformer.cache_context("cond"):
                    self.transformer(
                        hidden_states=clean_input,
                        timestep=clean_t,
                        encoder_hidden_states=encoder_hidden_states,
                        attention_kwargs=attention_kwargs,
                        position_ids=position_ids,
                        return_dict=False,
                    )

        noise_pred = torch.cat(noise_preds, dim=2)

        targets = noise - latents

        # Loss: only on generated frames (skip condition frame)
        loss = F.mse_loss(noise_pred, targets[:, :, 1:])
        return {"loss": loss}

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.transformer.enable_gradient_checkpointing()

    def register_modules(self, **kwargs):
        for name, module in kwargs.items():
            if module is None or isinstance(module, (tuple, list)) and module[0] is None:
                register_dict = {name: (None, None)}
            elif isinstance(module, WanCausalTransformer3DModel):
                register_dict = {name: ("diffusers", "WanCausalTransformer3DModel")}
            else:
                library, class_name = _fetch_class_library_tuple(module)
                register_dict = {name: (library, class_name)}

            self.register_to_config(**register_dict)
            setattr(self, name, module)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        **kwargs,
    ):
        from diffusers import WanImageToVideoPipeline as WanI2VPipeline

        pipeline = WanI2VPipeline.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )

        init_kwargs = {}
        for param in inspect.signature(cls).parameters:
            if hasattr(pipeline, param):
                init_kwargs[param] = getattr(pipeline, param)

        transformer = WanCausalTransformer3DModel.from_config(
            deepcopy(pipeline.transformer.config),
        )
        dtype = kwargs.get("torch_dtype", pipeline.transformer.dtype)
        transformer.to(dtype)
        init_kwargs["transformer"] = transformer

        incompatible_keys = transformer.load_state_dict(pipeline.transformer.state_dict(), strict=False)
        print(f"Transformer load: {incompatible_keys}")

        return cls(**init_kwargs)

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        **kwargs,
    ):
        self.tokenizer.save_pretrained(os.path.join(save_directory, "tokenizer"))
        self.text_encoder.save_pretrained(os.path.join(save_directory, "text_encoder"))
        self.vae.save_pretrained(os.path.join(save_directory, "vae"))
        self.scheduler.save_pretrained(os.path.join(save_directory, "scheduler"))
        self.transformer.save_pretrained(os.path.join(save_directory, "transformer"))
        self.save_config(save_directory)


def apply_monkey_patch():
    diffusers.WanCausalTransformer3DModel = WanCausalTransformer3DModel
