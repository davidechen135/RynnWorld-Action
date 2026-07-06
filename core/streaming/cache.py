"""
Streaming KV cache for causal video generation.

Implements a sliding-window KV cache with sink frame preservation,
used for frame-by-frame streaming video generation.
"""
import os
from typing import Dict

import torch


class DynamicCache(object):
    """Sliding-window KV cache for streaming video generation.
    
    Supports:
    - Freeze/unfreeze for denoising loops (cache only written on clean forward)
    - Sliding window with configurable max_frames and sink_size
    - Snapshot/restore for gradient checkpointing
    - Debug invariants (env-gated) for catching common bugs
    
    Args:
        max_frames: Maximum frames to keep in cache. If None, grows unbounded.
        sink_size: Number of leading frames to pin (never evicted). Default 1.
    """
    
    # Debug invariant gate. Setting STREAMING_DEBUG_INVARIANTS=1 enables
    # asserts inside update()/_trim_cache()/restore() that catch:
    #   (A) frame_count drifting from K seqlen / tokens_per_frame
    #   (B) trim not capping frame_count to max_frames
    #   (C) layers desyncing (one layer skipped a forward)
    #   (D) snapshot <-> restore round-trip preserves invariants
    _INVARIANT_GATE_ENV = "STREAMING_DEBUG_INVARIANTS"

    def __init__(self, max_frames=None, sink_size=1):
        self.keys: Dict[int, torch.Tensor] = {}
        self.values: Dict[int, torch.Tensor] = {}
        self.is_frozen = False
        self.max_frames = max_frames
        self.sink_size = sink_size
        if max_frames is not None and sink_size > max_frames:
            raise ValueError(
                f"DynamicCache: sink_size={sink_size} > max_frames={max_frames} "
                f"-- sink would consume the entire cache."
            )
        # Track how many frames have been stored per layer
        self._frame_count: Dict[int, int] = {}
        # Layer-0's frame_count after the most recent unfrozen update
        self._fc_layer0_after_update = None

    def _invariants_on(self) -> bool:
        return os.environ.get(self._INVARIANT_GATE_ENV, "0") == "1"

    def _check_post_update_invariants(self, layer_idx: int):
        """Run cheap asserts on a single layer's state after update()."""
        seq_len = self.keys[layer_idx].shape[1]
        fc = self._frame_count[layer_idx]
        # (A) K seqlen must be a multiple of frame_count
        assert fc > 0, f"L{layer_idx}: frame_count=0 with non-empty K (seq={seq_len})"
        assert seq_len % fc == 0, (
            f"L{layer_idx} INVARIANT (A): K seq {seq_len} not divisible by "
            f"frame_count {fc} (tokens_per_frame would be non-integer)"
        )
        # (B) frame_count must never exceed max_frames after trim
        if self.max_frames is not None:
            assert fc <= self.max_frames, (
                f"L{layer_idx} INVARIANT (B): fc={fc} > max_frames={self.max_frames} "
                f"(trim failed to cap)"
            )
        # (C) all layers must move in lockstep
        if layer_idx == 0:
            self._fc_layer0_after_update = fc
        else:
            assert self._fc_layer0_after_update is not None, (
                f"L{layer_idx} INVARIANT (C): updated before L0 -- layer order broke"
            )
            assert fc == self._fc_layer0_after_update, (
                f"L{layer_idx} INVARIANT (C): fc={fc} drifted from "
                f"L0 fc={self._fc_layer0_after_update} (layer desync)"
            )

    def freeze(self):
        """Freeze cache -- updates return concatenated K/V but don't persist."""
        self.is_frozen = True

    def unfreeze(self):
        """Unfreeze cache -- updates persist to cache."""
        self.is_frozen = False

    def snapshot(self):
        """Capture cache state for gradient checkpointing recompute."""
        return {
            "keys": self.keys.copy(),
            "values": self.values.copy(),
            "frame_count": self._frame_count.copy(),
            "is_frozen": self.is_frozen,
        }

    def restore(self, snap):
        """Restore cache state from snapshot."""
        self.keys = snap["keys"].copy()
        self.values = snap["values"].copy()
        self._frame_count = snap["frame_count"].copy()
        self.is_frozen = snap["is_frozen"]
        if self._invariants_on():
            # (D) round-trip check
            assert self.keys.keys() == snap["keys"].keys(), \
                f"INVARIANT (D): restored layer set {set(self.keys)} != snap {set(snap['keys'])}"
            for L in self.keys:
                assert self.keys[L] is snap["keys"][L], \
                    f"L{L} INVARIANT (D): K tensor identity drifted (snap.copy() corrupted)"
                assert self._frame_count[L] == snap["frame_count"][L], \
                    f"L{L} INVARIANT (D): frame_count drifted across restore"
            assert self.is_frozen == snap["is_frozen"], \
                f"INVARIANT (D): is_frozen flag drifted across restore"

    def _trim_cache(self, layer_idx):
        """Trim cache to keep only the most recent max_frames frames.
        
        Always preserves the first `sink_size` frames. The cache structure is:
          [sink_frames (size=sink_size)][recent_frames...]
        When trimming, we drop the oldest frames between sink and recent.
        """
        if self.max_frames is None:
            return
        if layer_idx not in self._frame_count:
            return

        total_frames = self._frame_count[layer_idx]
        if total_frames <= self.max_frames:
            return

        key = self.keys[layer_idx]
        seq_len = key.shape[1]
        frames_stored = self._frame_count[layer_idx]
        tokens_per_frame = seq_len // frames_stored

        frames_to_drop = total_frames - self.max_frames
        cond_tokens = self.sink_size * tokens_per_frame
        drop_count = frames_to_drop * tokens_per_frame

        kept_after_cond = seq_len - cond_tokens - drop_count
        
        # Diagnostic: layer-0 only, gated on env STREAMING_TRIM_TRACE=1
        _trim_trace = (
            os.environ.get("STREAMING_TRIM_TRACE", "0") == "1" and layer_idx == 0
        )
        if _trim_trace:
            _k_before_absmax = float(key.abs().max().item())
            
        if kept_after_cond > 0:
            self.keys[layer_idx] = torch.cat([
                key[:, :cond_tokens, :],
                key[:, cond_tokens + drop_count:, :],
            ], dim=1)
            self.values[layer_idx] = torch.cat([
                self.values[layer_idx][:, :cond_tokens, :],
                self.values[layer_idx][:, cond_tokens + drop_count:, :],
            ], dim=1)
        else:
            # Only keep condition frame
            self.keys[layer_idx] = key[:, :cond_tokens, :]
            self.values[layer_idx] = self.values[layer_idx][:, :cond_tokens, :]

        self._frame_count[layer_idx] = self.max_frames

        if _trim_trace:
            _k_after_absmax = float(self.keys[layer_idx].abs().max().item())
            print(
                f"  [trace:trim:L{layer_idx}] frames_dropped={frames_to_drop} | "
                f"K_seq {seq_len} -> {self.keys[layer_idx].shape[1]} | "
                f"K_absmax {_k_before_absmax:.3f} -> {_k_after_absmax:.3f} "
                f"(should be ~unchanged: physical memcpy preserves baked RoPE)",
                flush=True,
            )

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ):
        """Update cache with new K/V states.
        
        Returns the full (cached + new) K/V for attention computation.
        Only persists to cache when unfrozen.
        """
        new_seq_len = key_states.shape[1]
        if layer_idx in self.keys:
            existing_seq_len = self.keys[layer_idx].shape[1]
            frames_stored = self._frame_count.get(layer_idx, 0)
            if frames_stored > 0:
                tokens_per_frame = existing_seq_len // frames_stored
            else:
                tokens_per_frame = new_seq_len
            new_frames = new_seq_len // tokens_per_frame if tokens_per_frame > 0 else 1

            full_key = torch.cat([self.keys[layer_idx], key_states], dim=1)
            full_value = torch.cat([self.values[layer_idx], value_states], dim=1)
        else:
            new_frames = 1
            full_key = key_states
            full_value = value_states

        # Only mutate persistent cache state when unfrozen
        if not self.is_frozen:
            self.keys[layer_idx] = full_key
            self.values[layer_idx] = full_value
            self._frame_count[layer_idx] = self._frame_count.get(layer_idx, 0) + new_frames
            self._trim_cache(layer_idx)
            if self._invariants_on():
                self._check_post_update_invariants(layer_idx)

        return full_key, full_value
class FixedSizeCache(object):
    """Pre-allocated KV cache aligned with Self-Forcing design.

    Reference:
    - Self-Forcing pipeline/self_forcing_training.py:_initialize_kv_cache (lines 239-253)
    - Self-Forcing wan/modules/causal_model.py:CausalWanSelfAttention.forward (lines 193-235)

    Unlike DynamicCache (which uses torch.cat to dynamically grow), this cache
    pre-allocates a fixed-size tensor and writes in-place. This makes the cache
    tensor shape INVARIANT across forward/recompute calls, which is required for
    gradient_checkpointing compatibility (the autograd graph sees the same shape).

    Cache layout per layer:
        k: [batch_size, max_tokens, num_heads, head_dim]
        v: [batch_size, max_tokens, num_heads, head_dim]
        global_end_index: scalar tensor, total tokens written across rollout
        local_end_index: scalar tensor, current position in the cache buffer

    The caller provides `current_start` and `current_end` (global token offsets)
    on each `update()` call. The cache figures out where to write in its
    internal buffer (with optional eviction if local_attn_size limits reached).

    Usage:
        cache = FixedSizeCache(
            batch_size=1, num_layers=30, num_heads=24, head_dim=128,
            max_tokens=21*1560,  # num_max_frames * frame_seq_len
            dtype=torch.bfloat16, device='cuda',
            sink_size=1,           # keep frame 0 always
            local_attn_size=-1,    # no limit (attend full cache)
            tokens_per_frame=1560,
        )
        full_k, full_v = cache.update(
            roped_key, value, layer_idx=0,
            current_start=block_idx * tokens_per_block,
            current_end=current_start + new_tokens,
        )
    """

    def __init__(self, batch_size, num_layers, num_heads, head_dim,
                 max_tokens, dtype, device,
                 sink_size=1, local_attn_size=-1, tokens_per_frame=None):
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_tokens = max_tokens
        self.dtype = dtype
        self.device = device
        self.sink_size = sink_size      # number of FRAMES pinned at start
        self.local_attn_size = local_attn_size  # max frames in local window (-1 = no limit)
        self.tokens_per_frame = tokens_per_frame  # set lazily on first update if None

        # Pre-allocate per-layer K/V buffers
        self.cache = []
        for _ in range(num_layers):
            self.cache.append({
                "k": torch.zeros(
                    [batch_size, max_tokens, num_heads, head_dim],
                    dtype=dtype, device=device,
                ),
                "v": torch.zeros(
                    [batch_size, max_tokens, num_heads, head_dim],
                    dtype=dtype, device=device,
                ),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
            })

        # Max attention size derived from local_attn_size
        if local_attn_size == -1:
            self.max_attention_size = max_tokens
        else:
            # Will be set on first update once tokens_per_frame is known
            self.max_attention_size = max_tokens if tokens_per_frame is None \
                else local_attn_size * tokens_per_frame

        # Freeze flag: when frozen, writes are temporary (used by denoising sub-steps)
        self.is_frozen = False

    def freeze(self):
        """Freeze cache: writes during update() don't advance indices.
        Used during spatial denoising loop where each step overwrites the
        same cache position (matches Self-Forcing pattern)."""
        self.is_frozen = True

    def unfreeze(self):
        """Unfreeze cache: subsequent writes advance indices.
        Used for the final 'persist clean K/V' write per block."""
        self.is_frozen = False

    def reset(self):
        """Reset cache to empty state (zeros + indices=0). Used between
        different inference calls to avoid carrying over stale state."""
        for kv in self.cache:
            kv["k"].zero_()
            kv["v"].zero_()
            kv["global_end_index"].fill_(0)
            kv["local_end_index"].fill_(0)
        self.is_frozen = False

    def update(self, key, value, layer_idx, current_start, current_end,
               tokens_per_frame=None):
        """Write new K/V at the position implied by (current_start, current_end).

        Self-Forcing eviction rule (causal_model.py:206-219):
            if local_attn_size != -1 AND current_end > global_end_index AND
               num_new_tokens + local_end_index > kv_cache_size:
                evict oldest non-sink tokens, then write new at the end
            else:
                write new directly at local_end_index + delta

        Args:
            key:   [B, num_new_tokens, num_heads, head_dim] — caller already applied RoPE
            value: [B, num_new_tokens, num_heads, head_dim]
            layer_idx: which transformer layer's cache to update
            current_start: GLOBAL token offset where this block starts
            current_end: GLOBAL token offset where this block ends
            tokens_per_frame: optional, set on first call

        Returns:
            full_k: [B, attn_len, num_heads, head_dim] — cache content up to local_end
            full_v: same shape
        """
        if tokens_per_frame is not None and self.tokens_per_frame is None:
            self.tokens_per_frame = tokens_per_frame
            if self.local_attn_size != -1:
                self.max_attention_size = self.local_attn_size * tokens_per_frame

        kv = self.cache[layer_idx]
        num_new_tokens = key.shape[1]
        kv_size = self.max_tokens

        global_end = kv["global_end_index"].item()
        local_end = kv["local_end_index"].item()

        sink_tokens = self.sink_size * (self.tokens_per_frame or 0)

        # Self-Forcing eviction logic
        need_evict = (
            self.local_attn_size != -1
            and current_end > global_end
            and num_new_tokens + local_end > kv_size
        )

        if need_evict and not self.is_frozen:
            num_evicted_tokens = num_new_tokens + local_end - kv_size
            num_rolled_tokens = local_end - num_evicted_tokens - sink_tokens

            # Shift kept tokens left after the sink (clone to avoid memory overlap)
            if num_rolled_tokens > 0:
                kv["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    kv["k"][:, sink_tokens + num_evicted_tokens:
                                sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                kv["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    kv["v"][:, sink_tokens + num_evicted_tokens:
                                sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()

            new_local_end = local_end + current_end - global_end - num_evicted_tokens
            new_local_start = new_local_end - num_new_tokens
        else:
            new_local_end = local_end + current_end - global_end
            new_local_start = new_local_end - num_new_tokens

        # Read cached history (BEFORE writing new K/V).
        # Cache history at [attn_start:cached_end] is detached, and we use
        # torch.no_grad() around the in-place writes so the buffer's version
        # counter doesn't interfere with autograd's saved view tracking.
        attn_start = max(0, new_local_end - self.max_attention_size)
        cached_end = new_local_start  # everything written BEFORE this forward
        if cached_end > attn_start:
            cached_k = kv["k"][:, attn_start:cached_end]
            cached_v = kv["v"][:, attn_start:cached_end]
            full_k = torch.cat([cached_k, key], dim=1)
            full_v = torch.cat([cached_v, value], dim=1)
        else:
            full_k = key
            full_v = value

        # In-place write to cache for NEXT forward to read.
        # torch.no_grad() prevents autograd from tracking the buffer's version,
        # so subsequent writes don't invalidate views we just returned.
        with torch.no_grad():
            kv["k"][:, new_local_start:new_local_end] = key.detach()
            kv["v"][:, new_local_start:new_local_end] = value.detach()

            if not self.is_frozen:
                kv["global_end_index"].fill_(current_end)
                kv["local_end_index"].fill_(new_local_end)

        return full_k, full_v

    def get_local_end(self, layer_idx=0):
        return self.cache[layer_idx]["local_end_index"].item()

    def get_global_end(self, layer_idx=0):
        return self.cache[layer_idx]["global_end_index"].item()

