"""
train_distill.py  —  Streaming DMD Distillation Training

Trains a streaming (frame-by-frame) video generator using DMD
(Distribution Matching Distillation).

- Generator: WanCausalTransformer3DModel (trainable)
- Critic (fake_score): WanCausalTransformer3DModel (trainable)
- Real score: Frozen pre-trained Wan2.2-TI2V-5B transformer (teacher)

Uses pre-encoded latents from .safetensors files (same format as train_latent.py).
Streaming generation aligns with inference_streaming.py — frame-by-frame with KV cache.

Attribution:
- DMD generator/critic loss and ODE warmup follow the Self-Forcing recipe
  (Huang et al. 2025, https://github.com/guandeh17/Self-Forcing).
- ODE-regression trajectory forward and per-frame timestep sampling follow
  CausVid (Yin et al. 2024, https://github.com/tianweiy/CausVid).
- Three-segment streaming window, sliding-window position assignment, and
  EMA start-step conventions follow LongLive (Yang et al. 2025,
  https://github.com/NVlabs/LongLive) and StreamingLLM.
- Individual borrowed code paths are annotated inline as
  "Mirrors <file>:<lines>" or "<Project> §<section>".

Supports single-GPU and multi-GPU (via accelerate launch).

Usage (single GPU):
    python train_distill.py \\
        --model_path Wan-AI/Wan2.2-TI2V-5B-Diffusers \\
        --data_path /path/to/data.json \\
        --output_dir outputs/distill \\
        --num_train_steps 10000 \\
        --batch_size 1 \\
        --gradient_accumulation_steps 4 \\
        --learning_rate_gen 1e-5 \\
        --learning_rate_critic 1e-5 \\
        --num_latent_frames 6 \\
        --dfake_gen_update_ratio 1 \\
        --bf16

Usage (multi GPU):
    accelerate launch --num_processes 8 train_distill.py ...
"""

import os
import sys
import json
import math
import time
import argparse
import shutil
from dataclasses import dataclass
from typing import Optional, List

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from safetensors import SafetensorError   # needed in dataset except clauses
from torch.utils.data import Dataset
from termcolor import cprint

import deepspeed
import torch.distributed as dist

from diffusers import WanImageToVideoPipeline, FlowMatchEulerDiscreteScheduler
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
from diffusers.utils.torch_utils import randn_tensor

from core.streaming.model import WanCausalTransformer3DModel, apply_monkey_patch
from core.streaming.cache import DynamicCache


# ──────────────────────────────────────────────────────────────
# Robust safetensors loader (FUSE/OSS-tolerant)
# ──────────────────────────────────────────────────────────────
#
# WHY: cluster storage (FUSE/OSS-backed) occasionally returns:
#   * FileNotFoundError — file briefly invisible during OSS sync
#   * OSError(107, 'Transport endpoint is not connected') — FUSE daemon hiccup
#   * RuntimeError on safetensors header — file partially-written/corrupted
# A single-rank load failure inside DataLoader.__getitem__ propagates up,
# kills that rank, and NCCL terminates the entire 8-GPU group.
#
# Two-level defense:
#   1. _safetensors_load_robust(path, ...): retry SAME path with exponential
#      backoff. Most FUSE hiccups resolve in <1 s.
#   2. Each Dataset.__getitem__ wraps that in a fallback loop: if a path
#      persistently fails, swap to a random different sample. The training
#      step gets a different (still valid) sample instead of crashing.
#
# Both levels emit warning logs so we can spot if a path is genuinely
# broken (vs. transient).

def _safetensors_load_robust(path, max_retries=3, backoff_initial=0.5):
    """load_file() with retry-on-same-path. Raises on persistent failure.

    Catches transient FUSE/OSS errors (FileNotFoundError, OSError 107,
    safetensors header errors). After max_retries attempts, re-raises the
    last exception so the caller (Dataset.__getitem__) can decide whether
    to fall back to a different sample.

    SafetensorError is also caught: networked filesystems can return a
    truncated header on first read ("Error while deserializing header:
    header too small"); a same-path retry after backoff usually succeeds.
    SafetensorError is a top-level Exception subclass (NOT OSError /
    RuntimeError), so it must be listed explicitly to avoid escaping the
    except clause.

    Backoff schedule (default): 0.5s, 1.0s, 2.0s.
    """
    import time
    from safetensors import SafetensorError
    last_exc = None
    for attempt in range(max_retries):
        try:
            return load_file(path)
        except (FileNotFoundError, OSError, RuntimeError, SafetensorError) as e:
            last_exc = e
            if attempt < max_retries - 1:
                sleep_s = backoff_initial * (2 ** attempt)
                print(
                    f"[dataset:retry] attempt {attempt+1}/{max_retries} failed "
                    f"({type(e).__name__}: {e!s:.150}); sleep {sleep_s:.1f}s "
                    f"path={path}",
                    flush=True,
                )
                time.sleep(sleep_s)
    # All retries exhausted
    raise last_exc


# ──────────────────────────────────────────────────────────────
# Dataset (reused from train_latent.py)
# ──────────────────────────────────────────────────────────────

class LatentDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer=None,
        max_samples: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.samples = []

        if data_path.endswith(".json"):
            with open(data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                self.samples.append({
                    "latent_path": item["rgb_latents"],
                    "text_path": item.get("rgb_latents"),
                    "prompt": item.get("prompt", ""),
                })
        else:
            for fname in sorted(os.listdir(data_path)):
                if fname.endswith(".safetensors"):
                    self.samples.append({
                        "latent_path": os.path.join(data_path, fname),
                        "text_path": os.path.join(data_path, fname),
                        "prompt": "",
                    })

        # ── Optional subset filter for debug/diagnostic runs ──
        # Set env var DATA_SUBSET_IDX_FILE to a JSON file containing an
        # "indices" list (e.g. outputs/debug_subsets/high_motion_top20.json
        # produced by scripts/compute_high_motion_subset.py). When set, only
        # samples at those source-JSON indices are kept. Use this to pin
        # training + decode to a small high-motion subset for fast
        # iteration when chasing failure modes (e.g. "first 14 frames frozen + later frames collapsed").
        # Applied BEFORE max_samples so subset+max_samples compose.
        _subset_path = os.environ.get("DATA_SUBSET_IDX_FILE", "").strip()
        if _subset_path:
            with open(_subset_path, "r", encoding="utf-8") as f:
                _subset = json.load(f)
            _idx_list = _subset.get("indices", [])
            if not isinstance(_idx_list, list) or not _idx_list:
                raise RuntimeError(
                    f"DATA_SUBSET_IDX_FILE={_subset_path} has no usable "
                    f"'indices' list."
                )
            _orig_n = len(self.samples)
            _bad = [i for i in _idx_list if i < 0 or i >= _orig_n]
            if _bad:
                raise RuntimeError(
                    f"DATA_SUBSET_IDX_FILE has {len(_bad)} out-of-range "
                    f"indices (dataset size = {_orig_n}); first few: {_bad[:5]}"
                )
            self.samples = [self.samples[i] for i in _idx_list]
            try:
                import torch.distributed as _dist
                _is_rank0 = (not _dist.is_initialized()) or _dist.get_rank() == 0
            except Exception:
                _is_rank0 = True
            if _is_rank0:
                print(
                    f"[LatentDataset] DATA_SUBSET_IDX_FILE={_subset_path} → "
                    f"keeping {len(self.samples)}/{_orig_n} samples "
                    f"(schema={_subset.get('schema', 'n/a')})",
                    flush=True,
                )

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # ── FUSE/OSS-tolerant load with retry + random-fallback ──
        # Inner loop: 3 retries on the SAME path with backoff (handles
        # transient FUSE 107). Outer loop: 5 fallbacks to a NEW random idx
        # (handles a genuinely-deleted/corrupted file). Total worst case
        # = 15 attempts before raising. Logging at each fallback so
        # downstream we can grep [dataset:fallback] to spot bad paths.
        import random as _random
        _MAX_FALLBACKS = 5
        _paths_tried = []
        _orig_idx = idx
        for _fb_attempt in range(_MAX_FALLBACKS):
            sample = self.samples[idx]
            path = sample["latent_path"]
            _paths_tried.append(path)
            try:
                data = _safetensors_load_robust(path)
                video_latent = data["video_latents"]  # [C, F_l, H_l, W_l]
                text_embeds = data["text_embeds"] if "text_embeds" in data else None
                if _fb_attempt > 0:
                    print(
                        f"[dataset:fallback] recovered from idx={_orig_idx} → "
                        f"idx={idx} after {_fb_attempt} swap(s); path={path}",
                        flush=True,
                    )
                return {
                    "video_latent": video_latent,
                    "text_embeds": text_embeds,
                    "prompt": sample["prompt"],
                }
            except (FileNotFoundError, OSError, RuntimeError) as e:
                print(
                    f"[dataset:fallback] persistent failure idx={idx} "
                    f"({type(e).__name__}: {e!s:.150}); swap to random idx",
                    flush=True,
                )
                idx = _random.randint(0, len(self.samples) - 1)
        # All fallbacks exhausted — escalate.
        raise RuntimeError(
            f"LatentDataset failed after {_MAX_FALLBACKS} fallback attempts "
            f"starting from idx={_orig_idx}. Tried paths: {_paths_tried}"
        )


# ──────────────────────────────────────────────────────────────
# Egocentric control dataset (skeleton→hand video distillation)
# ──────────────────────────────────────────────────────────────

def _is_21F_path(path: str) -> bool:
    """Path-based 21F classifier.

       - 21F (teleop, lab data):
           * /video_latents/part1..5/, /video_latents/extra/  (paired teleop)
           * /tianji_wuji/video_latents/                       (bimanual teleop)
       - 7F  (internet egocentric): everything else (ego4d-*, ego-exo4d, epic, ssv).
    """
    return any(seg in path for seg in (
        "/video_latents/part1/", "/video_latents/part2/",
        "/video_latents/part3/", "/video_latents/part4/",
        "/video_latents/part5/", "/video_latents/extra/",
        "/tianji_wuji/video_latents/",
    ))


class EgoVerseControlDataset(Dataset):
    """Drop-in replacement for LatentDataset with control_video_latents support.

    Returns dict shape compatible with LatentDataset's downstream consumers
    (`video_latent`, `text_embeds`, `prompt`) PLUS adds `control_video_latent`.

    When ``filter_21f_only=True`` (default, backward-compat), only 21F teleop
    samples pass the path-based whitelist.  Set ``filter_21f_only=False`` to
    include ALL samples (mixed 7F + 21F).  ZeRO-2 handles mixed frame counts
    because all collective comms live outside the per-frame loop; ZeRO-3 will
    hang on mixed frames (per-forward all-gather count mismatch).
    """

    # Path to precomputed null prompt embedding (sha256 of "" → file).
    # Teacher trained with --prompt '' so every sample uses this null embedding.
    # Override via NULL_PROMPT_PATH env var.
    NULL_PROMPT_PATH = os.environ.get(
        "NULL_PROMPT_PATH",
        "/path/to/null_prompt_embedding.safetensors",
    )

    def __init__(
        self,
        data_path: str,
        tokenizer=None,
        max_samples: Optional[int] = None,
        filter_21f_only: bool = True,
    ):
        self.tokenizer = tokenizer
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # filter_21f_only toggle — when False, ALL samples pass
        # through regardless of path (supports mixed 7F+21F with ZeRO-2).
        # Per-sample text_embedding_path support: new JSON has
        # text_embedding_path → category-specific T5; legacy JSON does not
        # → fall back to the null embed.
        self.samples = []
        _n_skipped = 0
        for item in data:
            p = item["video_latent_path"]
            if filter_21f_only and not _is_21F_path(p):
                _n_skipped += 1
                continue
            self.samples.append({
                "latent_path": p,
                "text_path": p,
                "prompt": "",
                "text_embedding_path": item.get("text_embedding_path"),
            })
        self.has_text_embeddings = (
            len(self.samples) > 0
            and self.samples[0].get("text_embedding_path") is not None
        )
        try:
            import torch.distributed as _dist
            _is_rank0 = (not _dist.is_initialized()) or _dist.get_rank() == 0
        except Exception:
            _is_rank0 = True
        if _is_rank0:
            from termcolor import cprint as _cp
            _filter_tag = ("21F-ONLY (filtered)" if filter_21f_only
                           else "ALL FRAMES (mixed 7F+21F, no filter)")
            _cp(f"[EgoVerseControlDataset] {len(self.samples)} samples "
                f"(skipped {_n_skipped}) | filter={_filter_tag} | "
                f"text_embeddings={'PER-SAMPLE (new JSON)' if self.has_text_embeddings else 'NULL (legacy JSON)'}",
                'cyan')
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        # Pre-load the null prompt embedding (~32 KB) so we don't trigger
        # encode_prompt() at sample fetch time — train_distill.py frees the
        # T5 text_encoder after computing negative_embeds and raises if anyone
        # tries to encode on the fly.
        from safetensors.torch import load_file as _lf
        if os.path.exists(self.NULL_PROMPT_PATH):
            self.null_prompt_embed = _lf(self.NULL_PROMPT_PATH)["null_prompt_embedding"]
        else:
            raise FileNotFoundError(
                f"Null prompt embedding not found at {self.NULL_PROMPT_PATH}. "
                f"This is required because the T5 text_encoder is freed after "
                f"negative_embeds computation. Pre-encode '' via teacher's "
                f"prompt-embedding pipeline."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Same FUSE/OSS-tolerant retry+fallback as LatentDataset.
        import random as _random
        _MAX_FALLBACKS = 5
        _paths_tried = []
        _orig_idx = idx
        for _fb_attempt in range(_MAX_FALLBACKS):
            sample = self.samples[idx]
            path = sample["latent_path"]
            _paths_tried.append(path)
            try:
                data = _safetensors_load_robust(path)
                video_latent = data["video_latents"]                    # [C, T, H, W]
                control_video_latent = data["control_video_latents"]    # [C, T, H, W]
                # img_latent stored separately in some files; not strictly needed
                # since downstream uses video_latent[:, :, :1] as the cond.
                if _fb_attempt > 0:
                    print(
                        f"[ctrl_dataset:fallback] recovered idx={_orig_idx} → "
                        f"idx={idx} after {_fb_attempt} swap(s); path={path}",
                        flush=True,
                    )
                # Per-sample text embedding: if the JSON entry has
                # text_embedding_path, load that file's "text_embedding"
                # field; on failure, fall back to the null embed.
                text_embeds = self.null_prompt_embed
                if self.has_text_embeddings:
                    te_path = sample.get("text_embedding_path")
                    if te_path:
                        try:
                            te_data = _safetensors_load_robust(te_path)
                            if "text_embedding" in te_data:
                                text_embeds = te_data["text_embedding"]
                        except Exception as _te_err:
                            # Silent fallback: per-sample text load failure
                            # uses null embed for this sample. Log once per
                            # path-class to avoid spamming.
                            pass
                return {
                    "video_latent": video_latent,
                    "control_video_latent": control_video_latent,
                    "text_embeds": text_embeds,
                    "prompt": sample["prompt"],
                }
            except (FileNotFoundError, OSError, RuntimeError, SafetensorError) as e:
                # See _safetensors_load_robust for the rationale on catching
                # SafetensorError alongside the OS-level errors.
                print(
                    f"[ctrl_dataset:fallback] persistent failure idx={idx} "
                    f"({type(e).__name__}: {e!s:.150}); swap to random idx",
                    flush=True,
                )
                idx = _random.randint(0, len(self.samples) - 1)
        raise RuntimeError(
            f"EgoVerseControlDataset failed after {_MAX_FALLBACKS} fallback "
            f"attempts from idx={_orig_idx}. Tried: {_paths_tried}"
        )


# ──────────────────────────────────────────────────────────────
# Teacher LoRA + CPE loading (for control distill student init)
# ──────────────────────────────────────────────────────────────

def load_teacher_into_pipe(pipe, teacher_ckpt_dir, use_ema, is_main,
                           teacher_init_from_checkpoint=None):
    """Mutates pipe.transformer in-place: loads/fuses teacher weights into the
    base. Returns a dict with the teacher's control_patch_embedding state_dict
    and control_scale value (or None) so the caller can load them into the
    student's CPE/scale.

    Return type:
        dict | None — keys:
            "cpe":           dict (state_dict for nn.Conv3d), or None
            "control_scale": float or torch.Tensor (scalar), or None
        OR None if no control conditioning info available.

    Five formats supported (auto-detected from ckpt directory contents):

    1. **Legacy LoRA EMA** (ema_weights.bin with lora_A/lora_B/CPE keys):
       Older training format. Manually compute
       W_merged = W_base + B @ A (rank=alpha=64, scale=1.0). CPE returned.

    2. **Legacy LoRA live** (high_noise_lora/pytorch_lora_weights.safetensors):
       Live LoRA + control_patch_embedding.bin alongside. Same fuse formula.

    3. **Full SFT pretrain**:
       Detect when ema_weights.pt exists with NO lora_A/B keys (= 825 plain base
       transformer keys). Two sub-paths:
         * use_ema=True : load ema_weights.pt directly into pipe.transformer
                          (EMA shadow of full SFT training)
         * use_ema=False: consolidate pytorch_model/ ZeRO-3 shards via
                          deepspeed.utils.zero_to_fp32, load consolidated
                          fp32 state_dict into pipe.transformer (raw training
                          weights, NOT EMA shadow)
       No LoRA fuse, no CPE (full SFT didn't use control). Student's
       control_patch_embedding will stay at zero-init (built fresh by
       generator.init_control_patch_embedding() later).

    4. **SFT + LoRA + CPE** (used in the DMD phase):
       Detect when ckpt has `high_noise_lora/pytorch_lora_weights.safetensors`
       AND `control_patch_embedding.bin` AND `control_scale.bin`. This format
       layers LoRA + control_patch_embedding ON TOP OF a separate SFT pretrain
       checkpoint (which must be passed as `teacher_init_from_checkpoint`).
       Process:
         a. Load SFT base from `teacher_init_from_checkpoint` (PATH 3 raw mode)
            → pipe.transformer is now the full SFT model.
         b. Load + fuse live LoRA from `high_noise_lora/pytorch_lora_weights.
            safetensors` (HF diffusers format with `transformer.` prefix).
         c. Load CPE state from `control_patch_embedding.bin` → return for
            caller to load into student/critic/real_score CPE.
         d. Load control_scale scalar from `control_scale.bin` → return for
            caller to load into student/critic/real_score control_scale.
       Note: `use_ema` is IGNORED for PATH 4 (inference always uses live LoRA;
       ema_weights.bin in this format is for diagnostics only).

    5. **Full SFT + CPE (no LoRA)**:
       Detect when ckpt has `pytorch_model/` AND `control_patch_embedding.bin`
       AND `control_scale.bin` but NO `high_noise_lora/`. This is a full-param
       training with CPE baked in (828 keys). Load base weights from ZeRO
       shards (raw) or ema_weights.bin/.pt (EMA), skip CPE keys when loading
       into pipe.transformer, load CPE + control_scale from standalone files.
    """
    import torch
    from safetensors.torch import load_file as _lf
    from termcolor import cprint

    if teacher_ckpt_dir is None:
        if is_main:
            cprint("[teacher] No teacher_ckpt provided — student initializes "
                   "from base Wan2.2 (CPE zero-init).", "yellow")
        return (None, None)

    teacher_ckpt_dir = os.path.abspath(teacher_ckpt_dir)
    if not os.path.isdir(teacher_ckpt_dir):
        raise FileNotFoundError(
            f"teacher_ckpt_dir does not exist: {teacher_ckpt_dir}")

    # ── PATH 4 detection (SFT + LoRA + CPE; for DMD phase) ──
    # Detection signature: presence of control_scale.bin (= learnable control_scale,
    # absent in legacy LoRA ckpts). Also requires high_noise_lora/ + control_patch_
    # embedding.bin + teacher_init_from_checkpoint (the SFT base).
    cpe_bin_path = os.path.join(teacher_ckpt_dir, "control_patch_embedding.bin")
    control_scale_bin_path = os.path.join(teacher_ckpt_dir, "control_scale.bin")
    lora_safetensors_path = os.path.join(
        teacher_ckpt_dir, "high_noise_lora", "pytorch_lora_weights.safetensors")
    is_sft_plus_lora = (
        os.path.exists(cpe_bin_path)
        and os.path.exists(control_scale_bin_path)
        and os.path.exists(lora_safetensors_path)
    )

    if is_sft_plus_lora:
        # ───────────────────────── PATH 4: SFT + LoRA + CPE ─────────────────
        if teacher_init_from_checkpoint is None:
            raise ValueError(
                "[teacher] [PATH 4] Detected SFT+LoRA+CPE format "
                f"({teacher_ckpt_dir} has control_scale.bin + high_noise_lora/) "
                "but --teacher_init_from_checkpoint not provided. PATH 4 requires "
                "the SFT base checkpoint (a directory with pytorch_model/ ZeRO "
                "shards) to load BEFORE fusing LoRA. "
                "Pass --teacher_init_from_checkpoint <SFT_dir> via CLI."
            )
        sft_init_dir = os.path.abspath(teacher_init_from_checkpoint)
        sft_pytorch_model_dir = os.path.join(sft_init_dir, "pytorch_model")
        if not os.path.isdir(sft_pytorch_model_dir):
            raise FileNotFoundError(
                f"[teacher] [PATH 4] SFT base pytorch_model/ not found: "
                f"{sft_pytorch_model_dir}"
            )
        if is_main:
            cprint("─" * 76, "cyan")
            cprint(f"[teacher] [PATH 4: SFT+LoRA+CPE] {teacher_ckpt_dir}", "cyan")
            cprint(f"[teacher]   SFT base init_from: {sft_init_dir}", "cyan")
            cprint(f"[teacher]   LoRA:               {lora_safetensors_path}", "cyan")
            cprint(f"[teacher]   CPE:                {cpe_bin_path}", "cyan")
            cprint(f"[teacher]   control_scale:      {control_scale_bin_path}", "cyan")
            cprint(f"[teacher]   use_ema (IGNORED for PATH 4) = {use_ema}", "yellow")
            cprint("─" * 76, "cyan")

        # ── Step a: Load SFT base (PATH 3 raw-mode logic) ──
        # We always use raw (NOT EMA) for the SFT base in PATH 4 — the LoRA was
        # trained on top of raw SFT (init_from_checkpoint loads pytorch_model,
        # not ema_weights.pt).
        if is_main:
            cprint(f"[teacher] [PATH 4-a] Consolidating SFT base DeepSpeed ZeRO "
                   f"shards from {sft_pytorch_model_dir} (~30s for 5B params)...",
                   "cyan")
        from deepspeed.utils.zero_to_fp32 import (
            get_fp32_state_dict_from_zero_checkpoint,
        )
        sft_state = get_fp32_state_dict_from_zero_checkpoint(
            sft_init_dir, tag="pytorch_model",
        )
        n_sft_lora = sum(1 for k in sft_state if ".lora_A." in k or ".lora_B." in k)
        if n_sft_lora > 0:
            raise RuntimeError(
                f"[teacher] [PATH 4-a] SFT base has {n_sft_lora} LoRA keys — "
                f"expected plain SFT format. teacher_init_from_checkpoint "
                f"{sft_init_dir} looks like a LoRA ckpt, not SFT pretrain."
            )
        sd = pipe.transformer.state_dict()
        n_loaded = 0
        for k, v in sft_state.items():
            if k in sd:
                sd[k] = v.to(sd[k].dtype)
                n_loaded += 1
        pipe.transformer.load_state_dict(sd, strict=True)
        if is_main:
            cprint(f"[teacher] [PATH 4-a] SFT base loaded: {n_loaded}/{len(sft_state)} "
                   f"keys into pipe.transformer", "green")
        del sft_state, sd
        import gc; gc.collect()

        # ── Step b: Fuse live LoRA on top of SFT base ──
        # HF diffusers safetensors format: keys have "transformer." prefix.
        # We strip that to match pipe.transformer's state_dict keys.
        if is_main:
            cprint(f"[teacher] [PATH 4-b] Loading + fusing live LoRA from "
                   f"{lora_safetensors_path} ...", "cyan")
        lora_sd = _lf(lora_safetensors_path)
        a_map, b_map = {}, {}
        for k, v in lora_sd.items():
            # Strip HF "transformer." prefix
            k_stripped = k[len("transformer."):] if k.startswith("transformer.") else k
            if ".lora_A.weight" in k_stripped:
                base_key = k_stripped.replace(".lora_A.weight", "")
                a_map[base_key] = v
            elif ".lora_B.weight" in k_stripped:
                base_key = k_stripped.replace(".lora_B.weight", "")
                b_map[base_key] = v
        sd = pipe.transformer.state_dict()
        n_fused, n_missing_wkey = 0, []
        for base_key in a_map:
            if base_key not in b_map:
                continue
            wkey = base_key + ".weight"
            if wkey not in sd:
                n_missing_wkey.append(wkey)
                continue
            # B @ A in fp32 to avoid bf16 precision loss in fused delta
            A = a_map[base_key].to(torch.float32)
            B = b_map[base_key].to(torch.float32)
            sd[wkey] = sd[wkey] + (B @ A).to(sd[wkey].dtype)
            n_fused += 1
        pipe.transformer.load_state_dict(sd, strict=True)
        if is_main:
            cprint(f"[teacher] [PATH 4-b] LoRA fused: {n_fused}/{len(a_map)} layers "
                   f"(lora_scale=1.0)", "green")
            if n_missing_wkey:
                cprint(f"[teacher] [PATH 4-b] WARN: {len(n_missing_wkey)} LoRA target "
                       f"weights NOT in pipe.transformer (skipped). Sample: "
                       f"{n_missing_wkey[:3]}", "yellow")
        del lora_sd, a_map, b_map, sd
        gc.collect()

        # ── Step c: Load CPE state ──
        ctrl_pe_state = torch.load(cpe_bin_path, map_location="cpu", weights_only=False)
        if is_main:
            cprint(f"[teacher] [PATH 4-c] CPE loaded: keys={list(ctrl_pe_state.keys())}, "
                   f"weight={tuple(ctrl_pe_state['weight'].shape)} "
                   f"absmax={ctrl_pe_state['weight'].abs().max().item():.4f}",
                   "green")

        # ── Step d: Load control_scale scalar ──
        cs_loaded = torch.load(control_scale_bin_path, map_location="cpu", weights_only=False)
        # control_scale.bin is a 0-dim tensor (scalar)
        if hasattr(cs_loaded, "numel") and cs_loaded.numel() == 1:
            cs_val = float(cs_loaded.item())
        else:
            cs_val = float(cs_loaded)
        if is_main:
            cprint(f"[teacher] [PATH 4-d] control_scale loaded: {cs_val:.6f}", "green")

        return (ctrl_pe_state, cs_val)

    # ── PATH 5 detection: Full SFT with CPE baked in (no LoRA) ──
    # Layout:
    #   - pytorch_model/ (ZeRO shards, 828 keys = 825 base + CPE + control_scale)
    #   - control_patch_embedding.bin + control_scale.bin (standalone)
    #   - ema_weights.bin (full, NOT LoRA)
    #   - NO high_noise_lora/
    pytorch_model_dir_p5 = os.path.join(teacher_ckpt_dir, "pytorch_model")
    is_full_sft_with_cpe = (
        os.path.isdir(pytorch_model_dir_p5)
        and os.path.exists(cpe_bin_path)
        and os.path.exists(control_scale_bin_path)
        and not os.path.exists(lora_safetensors_path)
    )

    if is_full_sft_with_cpe:
        # ───────────────────────── PATH 5: Full SFT + CPE (no LoRA) ────────────
        ema_pt = os.path.join(teacher_ckpt_dir, "ema_weights.pt")
        ema_bin = os.path.join(teacher_ckpt_dir, "ema_weights.bin")
        if use_ema and os.path.exists(ema_pt):
            if is_main:
                cprint(f"[teacher] [PATH 5: SFT+CPE, EMA .pt] Loading {ema_pt}", "cyan")
            full_state = torch.load(ema_pt, map_location="cpu", weights_only=False)
        elif use_ema and os.path.exists(ema_bin):
            if is_main:
                cprint(f"[teacher] [PATH 5: SFT+CPE, EMA .bin] Loading {ema_bin}", "cyan")
            full_state = torch.load(ema_bin, map_location="cpu", weights_only=False)
        else:
            if is_main:
                cprint(f"[teacher] [PATH 5: SFT+CPE, RAW] Consolidating ZeRO shards "
                       f"from {pytorch_model_dir_p5} (~30s for 5B params)...", "cyan")
            from deepspeed.utils.zero_to_fp32 import (
                get_fp32_state_dict_from_zero_checkpoint,
            )
            full_state = get_fp32_state_dict_from_zero_checkpoint(
                teacher_ckpt_dir, tag="pytorch_model",
            )

        n_lora = sum(1 for k in full_state if ".lora_A." in k or ".lora_B." in k)
        if n_lora > 0:
            raise RuntimeError(
                f"[teacher] [PATH 5] Expected full SFT+CPE but found {n_lora} LoRA "
                f"keys. This checkpoint looks like a LoRA ckpt, not full SFT.")

        # Load base keys into pipe.transformer (skip CPE/control_scale keys)
        cpe_skip = {"control_patch_embedding.weight", "control_patch_embedding.bias",
                     "control_scale"}
        sd = pipe.transformer.state_dict()
        n_loaded = 0
        for k, v in full_state.items():
            if k in cpe_skip:
                continue
            if k in sd:
                sd[k] = v.to(sd[k].dtype)
                n_loaded += 1
        pipe.transformer.load_state_dict(sd, strict=True)
        if is_main:
            cprint(f"[teacher] [PATH 5] Base weights loaded: {n_loaded}/{len(sd)} keys "
                   f"(skipped {len(cpe_skip)} CPE/scale keys)", "green")
        del full_state, sd
        import gc; gc.collect()

        # Load CPE from standalone files (same as PATH 4 step c/d)
        ctrl_pe_state = torch.load(cpe_bin_path, map_location="cpu", weights_only=False)
        if is_main:
            cprint(f"[teacher] [PATH 5] CPE loaded: keys={list(ctrl_pe_state.keys())}, "
                   f"weight={tuple(ctrl_pe_state['weight'].shape)} "
                   f"absmax={ctrl_pe_state['weight'].abs().max().item():.4f}", "green")

        cs_loaded = torch.load(control_scale_bin_path, map_location="cpu", weights_only=False)
        if hasattr(cs_loaded, "numel") and cs_loaded.numel() == 1:
            cs_val = float(cs_loaded.item())
        else:
            cs_val = float(cs_loaded)
        if is_main:
            cprint(f"[teacher] [PATH 5] control_scale loaded: {cs_val:.6f}", "green")

        return (ctrl_pe_state, cs_val)

    # ── Format auto-detection (legacy paths 1/2/3) ──
    # Full SFT pretrain ckpt has:
    #   - ema_weights.pt (PyTorch .pt, NOT .bin like LoRA ckpt)
    #   - pytorch_model/ subdir with bf16_zero_pp_rank_*_*_optim_states.pt
    #   - latest file pointing to "pytorch_model"
    #   - NO high_noise_lora/, NO control_patch_embedding.bin
    # Legacy LoRA ckpt has:
    #   - ema_weights.bin (lora_A/B/CPE keys)
    #   - high_noise_lora/pytorch_lora_weights.safetensors
    #   - control_patch_embedding.bin
    ema_pt_path = os.path.join(teacher_ckpt_dir, "ema_weights.pt")
    pytorch_model_dir = os.path.join(teacher_ckpt_dir, "pytorch_model")
    is_full_sft = os.path.isdir(pytorch_model_dir) and os.path.exists(ema_pt_path)

    if is_full_sft:
        # ───────────────────────── PATH 3: Full SFT pretrain ────────────────
        if use_ema:
            if is_main:
                cprint(f"[teacher] [SFT-EMA] Loading EMA weights from {ema_pt_path}",
                       "cyan")
            ema_state = torch.load(ema_pt_path, map_location="cpu", weights_only=False)
        else:
            # Raw mode: consolidate DeepSpeed ZeRO shards. Standard public API
            # from deepspeed.utils.zero_to_fp32 (no need to invoke the bundled
            # zero_to_fp32.py script, which would require a disk write).
            if is_main:
                cprint(f"[teacher] [SFT-RAW] Consolidating DeepSpeed ZeRO shards "
                       f"from {pytorch_model_dir} (this takes ~30s for 5B params)...",
                       "cyan")
            from deepspeed.utils.zero_to_fp32 import (
                get_fp32_state_dict_from_zero_checkpoint,
            )
            # tag = "pytorch_model" matches the contents of `latest` file
            ema_state = get_fp32_state_dict_from_zero_checkpoint(
                teacher_ckpt_dir, tag="pytorch_model",
            )

        # Sanity check: ensure no LoRA keys (= confirms full SFT format)
        n_lora = sum(1 for k in ema_state if ".lora_A." in k or ".lora_B." in k)
        n_cpe  = sum(1 for k in ema_state if "control_patch_embedding" in k)
        if is_main:
            cprint(f"[teacher] [SFT-{'EMA' if use_ema else 'RAW'}] state_dict has "
                   f"{len(ema_state)} keys (lora_A/B={n_lora}, CPE={n_cpe})",
                   "cyan")
        if n_lora > 0:
            # Surprise — this is NOT a full SFT ckpt, it's a LoRA ckpt
            # mis-stored under the SFT layout. Bail out loud rather than
            # silently doing wrong fuse.
            raise RuntimeError(
                f"[teacher] [SFT-{'EMA' if use_ema else 'RAW'}] expected full SFT "
                f"format but found {n_lora} LoRA keys. Use the legacy LoRA path "
                f"(set USE_EMA_TEACHER=1 with .bin ema_weights, or rename .pt → "
                f".bin to trigger the old EMA-LoRA branch)."
            )

        # Direct overwrite into pipe.transformer (keys should match base format)
        sd = pipe.transformer.state_dict()
        n_loaded, n_missing_in_pipe = 0, []
        for k, v in ema_state.items():
            if k in sd:
                sd[k] = v.to(sd[k].dtype)
                n_loaded += 1
            else:
                n_missing_in_pipe.append(k)
        n_missing_in_state = [k for k in sd if k not in ema_state]
        pipe.transformer.load_state_dict(sd, strict=True)
        if is_main:
            cprint(f"[teacher] [SFT-{'EMA' if use_ema else 'RAW'}] loaded {n_loaded}"
                   f"/{len(ema_state)} weights into pipe.transformer "
                   f"(state-only-keys={len(n_missing_in_pipe)}, "
                   f"pipe-only-keys={len(n_missing_in_state)})", "green")
            if n_missing_in_pipe:
                cprint(f"[teacher]   state-only sample: {n_missing_in_pipe[:3]}",
                       "yellow")
            if n_missing_in_state:
                cprint(f"[teacher]   pipe-only sample: {n_missing_in_state[:3]}",
                       "yellow")
        # Full SFT has no CPE → student CPE will be zero-init by
        # generator.init_control_patch_embedding() later.
        return (None, None)

    # ───────────────────────── Legacy LoRA paths (unchanged) ─────────────────
    ema_path = os.path.join(teacher_ckpt_dir, "ema_weights.bin")
    if use_ema and os.path.exists(ema_path):
        if is_main:
            cprint(f"[teacher] Loading EMA weights from {ema_path} ...", "cyan")
        ema_state = torch.load(ema_path, map_location="cpu", weights_only=False)
        ema_lora_a, ema_lora_b, ctrl_pe_state = {}, {}, {}
        for k, v in ema_state.items():
            if "control_patch_embedding" in k:
                ctrl_pe_state[k.replace("control_patch_embedding.", "")] = v
            elif ".lora_A." in k:
                ema_lora_a[k.split(".lora_A.")[0]] = v
            elif ".lora_B." in k:
                ema_lora_b[k.split(".lora_B.")[0]] = v
        if is_main:
            cprint(f"[teacher] EMA: {len(ema_lora_a)} LoRA pairs, "
                   f"{len(ctrl_pe_state)} CPE keys", "cyan")
        sd = pipe.transformer.state_dict()
        n_fused = 0
        for base_key in ema_lora_a:
            if base_key not in ema_lora_b:
                continue
            wkey = base_key + ".weight"
            if wkey not in sd:
                continue
            A = ema_lora_a[base_key].to(torch.float32)
            B = ema_lora_b[base_key].to(torch.float32)
            sd[wkey] = sd[wkey] + (B @ A).to(sd[wkey].dtype)
            n_fused += 1
        pipe.transformer.load_state_dict(sd, strict=True)
        if is_main:
            cprint(f"[teacher] EMA LoRA manually fused: {n_fused} layers.", "green")
        return (ctrl_pe_state if ctrl_pe_state else None, None)

    if use_ema and not os.path.exists(ema_path):
        if is_main:
            cprint(f"[teacher] WARNING: use_ema=True but {ema_path} missing; "
                   f"falling back to live LoRA.", "yellow")

    lora_safetensors = os.path.join(
        teacher_ckpt_dir, "high_noise_lora", "pytorch_lora_weights.safetensors")
    if os.path.exists(lora_safetensors):
        if is_main:
            cprint(f"[teacher] Loading live LoRA from {lora_safetensors}", "cyan")
        lora_sd = _lf(lora_safetensors)
        a_map, b_map = {}, {}
        for k, v in lora_sd.items():
            if ".lora_A." in k:
                a_map[k.replace("transformer.", "").split(".lora_A.")[0]] = v
            elif ".lora_B." in k:
                b_map[k.replace("transformer.", "").split(".lora_B.")[0]] = v
        sd = pipe.transformer.state_dict()
        n_fused = 0
        for base_key in a_map:
            if base_key not in b_map:
                continue
            wkey = base_key + ".weight"
            if wkey not in sd:
                continue
            A = a_map[base_key].to(torch.float32)
            B = b_map[base_key].to(torch.float32)
            sd[wkey] = sd[wkey] + (B @ A).to(sd[wkey].dtype)
            n_fused += 1
        pipe.transformer.load_state_dict(sd, strict=True)
        if is_main:
            cprint(f"[teacher] Live LoRA fused: {n_fused} layers.", "green")
    else:
        if is_main:
            cprint(f"[teacher] WARNING: no LoRA at {lora_safetensors}; "
                   f"using base weights.", "yellow")

    cpe_path = os.path.join(teacher_ckpt_dir, "control_patch_embedding.bin")
    if os.path.exists(cpe_path):
        ctrl_pe_state = torch.load(cpe_path, map_location="cpu", weights_only=False)
        if is_main:
            cprint(f"[teacher] Loaded control_patch_embedding: "
                   f"weight={tuple(ctrl_pe_state['weight'].shape)}", "green")
        return (ctrl_pe_state, None)
    if is_main:
        cprint(f"[teacher] WARNING: no control_patch_embedding.bin at {cpe_path}; "
               f"student CPE will be zero-init.", "yellow")
    return (None, None)


# ──────────────────────────────────────────────────────────────
# Streaming generation (training version, aligned with inference_streaming.py)
# ──────────────────────────────────────────────────────────────

def _mem_probe(tag: str, trace_tag: str = "", enabled: bool = False, do_sync: bool = True):
    """Lightweight CUDA memory probe — rank-0 only, opt-in.

    Why this exists:
    --------------------------------------------------------
    Investigating an 8×L20Z (~79 GB / GPU) OOM at ODE warmup step 1, frame 1
    of generate_streaming_with_grad, inside the very first transformer
    block's ZeRO-3 all_gather: the OOM message reported 71+ GB already
    allocated when only 90 MB more was being requested, but the
    [mem pre-train] log just before training reported only 11.6 GB / rank,
    leaving roughly 60 GB unaccounted for.

    Activations from the 21-frame autograd graph were ruled out because the
    OOM strikes BEFORE any frame-1 forward completes. The leading hypothesis
    is that ZeRO-3's `stage3_max_live_parameters` and prefetch buckets (set
    to 50 MB each in ds_config_zero3.json) are smaller than a single Wan2.2-5B
    transformer block (~340 MB bf16), so DeepSpeed ends up holding the entire
    30-block model live on every GPU instead of partitioning + offloading
    as designed.

    This probe lets us pin down WHERE the 60 GB enters the picture by
    reporting allocated/reserved/peak deltas around each major substep:
      * before / after _seed_cache_with_condition (frame 0 no-grad fwd)
      * before / after frame 1 forward (the OOM site)
      * before / after MSE backward (where ZeRO-3 reduce-scatter happens)

    Args:
      tag:        short label for this measurement point (e.g. "pre-frame1").
      trace_tag:  upstream trace tag (e.g. "gen") — gets prefixed for grep-ability.
      enabled:    if False, this is a no-op (cheap) — the caller controls
                  cadence so production runs aren't spammed.
      do_sync:    cuda.synchronize() before reading. Adds ~1 ms but ensures
                  the printed numbers reflect outstanding async kernels too;
                  important because async memory frees from prior ZeRO-3
                  releases can otherwise leak into the next probe's reading.
    """
    if not enabled:
        return
    try:
        import torch.distributed as _dist
        if _dist.is_initialized() and _dist.get_rank() != 0:
            return
    except Exception:
        pass
    if not torch.cuda.is_available():
        return
    if do_sync:
        torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated() / 1e9
    reserv = torch.cuda.memory_reserved() / 1e9
    peak = torch.cuda.max_memory_allocated() / 1e9
    cprint(
        f"  [memprobe:{trace_tag}:{tag}] alloc={alloc:6.2f} GB  "
        f"reserved={reserv:6.2f} GB  peak={peak:6.2f} GB",
        "yellow",
    )


# ── Per-step SGT (stochastic gradient truncation) diagnostics ──────────────
# Why this exists:
# Self Forcing's SGT picks one denoising sub-step s ∈ [1, NIS] per training
# iter and uses that step's 1-step x_0 estimate as the final output (see
# Algorithm 1 lines 4-11 in 2506.08009v2.pdf). When the model is still
# under-trained, the highest-noise s (= NIS-1 in our 0-based indexing,
# i.e. sigma_t ≈ 1.0) produces severely OOD latents that decode to nearly
# pure-red saturation. The DMD critic eventually cleans this up, but during
# the ODE-MSE warmup phase there's no per-noise-level signal — every s
# learns at the same rate, just less of it at the high-noise tail.
#
# To diagnose how fast each s heals, we need:
#   (a) per-step record of WHICH s was sampled (so we can correlate with
#       the debug grid: red-blob steps should correspond to s=NIS-1)
#   (b) a running histogram (so we can verify the uniform-sampling claim:
#       each s should be ~25% of iters under NIS=4)
#   (c) per-bucket loss (so we can SEE whether s=NIS-1 loss decays slower
#       than s=0 — confirming the "high-noise tail learns slower" theory)
#
# The state below is mutated by `_streaming_generate` (when an SGT-eligible
# call returns) and read by the per-step diag log line. Read/write is rank-0
# only: under ZeRO-3 all ranks get the SAME grad_step_idx via the broadcast
# at L520-527, so rank 0's record is authoritative for the whole node.
#
# Memory: at most NIS entries (≤4 in production), each a (count, loss_sum)
# tuple. Histogram + means computable in O(NIS) at log time.
_sgt_diag = {
    "last_grad_step_idx": -1,    # most recent SGT pick (rank 0); -1 = SGT not used this step
    "hist_count": {},            # idx -> # times it was picked since training start
    "hist_loss_sum": {},         # idx -> sum of gen_loss values observed when picked
}


def _record_sgt_pick(grad_step_idx: int):
    """Stash the SGT pick on rank 0. Called from `_streaming_generate` right
    after the broadcast resolves. Loss accumulation is deferred to the
    train-loop logging block because the loss isn't known until after the
    forward+backward returns.
    """
    if grad_step_idx < 0:
        return  # not an SGT call
    try:
        import torch.distributed as _dist
        if _dist.is_initialized() and _dist.get_rank() != 0:
            return
    except Exception:
        pass
    _sgt_diag["last_grad_step_idx"] = int(grad_step_idx)


def _attribute_sgt_loss(loss_value: float):
    """Attribute `loss_value` (typically gen_loss for this iter) to the s
    bucket that was picked on this iter's `_streaming_generate` call.
    Caller (train-loop diag block) invokes this after the forward.

    Why we route loss attribution through this helper rather than reading
    `_sgt_diag["last_grad_step_idx"]` in the diag log directly: under DMD
    the same training iter calls `_streaming_generate` TWICE (once for the
    generator update, once with critic inputs / no_grad), and only the
    grad-bearing call's pick is meaningful. By having the train loop
    explicitly call this AFTER the grad-bearing forward + backward returns
    (and before the next iter's call), we ensure the bucketed average
    matches what the gen_loss actually saw.
    """
    idx = _sgt_diag["last_grad_step_idx"]
    if idx < 0:
        return
    _sgt_diag["hist_count"][idx] = _sgt_diag["hist_count"].get(idx, 0) + 1
    _sgt_diag["hist_loss_sum"][idx] = _sgt_diag["hist_loss_sum"].get(idx, 0.0) + float(loss_value)


def _sgt_hist_summary() -> str:
    """Render the running histogram as a compact one-line string for the
    [diag] log. Format:
       sgt_hist[ s=0:n=12 µL=4.81 | s=1:n=10 µL=4.92 | s=2:n=14 µL=5.30 | s=3:n=11 µL=6.74 ]
    where µL is mean gen_loss conditioned on that bucket. Sorted by s asc.
    Empty buckets are omitted (e.g. plain mode → empty string ""). At first
    log call before any SGT pick, returns "(no SGT picks yet)".
    """
    if not _sgt_diag["hist_count"]:
        return "(no SGT picks yet)"
    parts = []
    for idx in sorted(_sgt_diag["hist_count"].keys()):
        n = _sgt_diag["hist_count"][idx]
        ls = _sgt_diag["hist_loss_sum"].get(idx, 0.0)
        mean_l = (ls / n) if n > 0 else float("nan")
        # Scientific notation: after v_flow warmup pushes
        # the student onto the GT manifold, dmd_loss settles at ~1e-5 — fp
        # `.3f` prints "0.000" and we lose the actual signal magnitude.
        # `.3e` shows e.g. `1.245e-05`, which is what we need to verify DMD
        # is doing real work vs collapsed-to-zero.
        parts.append(f"s={idx}:n={n} µL={mean_l:.3e}")
    return "[ " + " | ".join(parts) + " ]"


def _streaming_generate(
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
    max_cache_frames: Optional[int] = None,
    pe_mode: str = "slot",
    stochastic_grad_truncation: bool = False,
    # ── Three-segment streaming window (LongLive-inspired) ──────────────────
    # All None / 0 → 100% backward-compat: legacy "all frames with_grad" path.
    train_window_size: Optional[int] = None,
    train_window_start: Optional[int] = None,
    gt_far_latents: Optional[torch.Tensor] = None,  # [1, C, F, H, W] full GT
    # ── Debug instrumentation (off by default) ─────────────────────────────
    # When True, rank-0 dumps a per-frame trace of cache state + position-id
    # geometry. Wire up from the outer training loop only at step==1 / step==2
    # of the DMD phase so we don't blow up logs for production runs.
    trace_dump: bool = False,
    trace_tag: str = "",  # short label for the log line, e.g. "gen" / "critic"
    # ── Control distill (used in the DMD phase) ──
    # [B, C, F, H, W] full skel latent; sliced per-frame via _ctrl_f(f).
    # None = no control (pre-control-distill behavior, kept for backward compat).
    control_video_latent: Optional[torch.Tensor] = None,
):
    """
    Unified streaming generator used by both the no-grad (critic data) and
    with-grad (DMD generator loss) paths.

    Cache protocol — matches WanTI2VTrainingPipeline.__call__ and
    inference_streaming.py:
      For each chunk:
        1) Run all denoise steps with the KV cache FROZEN (no writes).
        2) After denoising is done, run an extra forward of the *clean*
           (just-denoised) latent at t=0 with the cache UNFROZEN. This
           persists CLEAN K/V — not the noisy K/V the original code wrote
           by unfreezing on the last denoise step. With num_inference_steps=1
           the original protocol wrote pure-noise K/V into the cache, which
           does not match what the model sees at inference.

    Gradient handling (`with_grad=True`):
      - Default (stochastic_grad_truncation=False): all denoise forwards
        keep grad so the DMD loss can backprop through the entire
        T-step denoising chain of every frame. Memory cost scales with
        T × num_latent_frames.
      - With stochastic_grad_truncation=True (Self-Forcing §3.2 +
        Algorithm 1 lines 4 + 8-12): we sample a SINGLE step
        s ∈ [1, T] once per call (shared across all frames of this
        video, per the paper). For every frame:
          * Denoising steps j=T..s+1 run under torch.no_grad() and
            advance via scheduler.step (sample new noise, get noisy
            input for next step).
          * At step j=s, we enable grad ONCE, predict noise → x0,
            and use that x0 as the frame's final output. We do NOT
            continue the denoise loop past s (the autograd graph is
            cut here).
        This brings the memory cost back to ~1 forward per frame
        (matching the 1-step regime) regardless of num_inference_steps,
        and matches what the paper actually trains. Without this, with
        num_inference_steps=4 and num_latent_frames=5, the autograd graph
        holds 20 forwards instead of 5.

      - The clean-K/V forward is always wrapped in torch.no_grad() and
        receives `latents.detach()`, so the KV cache holds tensors that
        are detached from the autograd graph. Without this, the cache
        would chain gradients across every generated frame and the graph
        would balloon (and in-place writes into `latents` would break
        autograd's version counters).

    ── Three-segment streaming window (LongLive-inspired) ──
    When `train_window_size` is set (e.g. K=7), the rollout is split into
    three segments to reduce the autograd graph from `num_latent_frames`
    down to `train_window_size` forwards:

       [cond=frame 0]                            ← always (1 forward, no_grad)
       [GT teacher-forcing: frames 1..T_start-K-1]   ← far history,
                              source = gt_far_latents (e.g. x0_teacher)
                              1 forward/frame at t=0, no_grad, writes clean KV
       [self-forcing buffer: frames T_start-K..T_start-1]  ← K-frame near history,
                              source = student no_grad rollout (NIS=4 denoise)
                              models inference-time "dirty" cache
       [training window: frames T_start..T_start+K-1]   ← K frames, with_grad,
                              full denoise, returned for DMD loss

    Why this matches inference:
      - At inference, far history (frames 1..T_start-K-1) doesn't exist
        when generating from cond + the model just rolls forward chunk by
        chunk. The closest training-time analogue is "give the model a
        clean global anchor" — GT serves this role. Trade-off: at T_start
        positions where far-history exists in training, there is a small
        train-test gap (training sees GT, inference will see student-self).
        Mitigation: attention distance decay makes the gap minor, AND
        for small K (e.g. T_start ∈ {1,8,14} for total=21), 2/3 of training
        positions don't touch GT at all.
      - Near history (the buffer) IS what the model sees at inference:
        its own just-generated previous frames. By rolling out with no_grad
        we provide the same "dirty self-output cache" that inference will,
        so the training window learns to correct errors rather than assume
        a clean past.

    `train_window_start` (T_start) must be in [1, num_latent_frames - K].
    For ZeRO-3 safety, pick T_start on rank 0 and broadcast (callers'
    responsibility — same pattern as grad_step_idx below).

    Returns: [1, C, K, H_l, W_l] when train_window_size is set
             (only the K training-window frames; far/buffer frames are
             discarded since DMD loss only sees the window).
             [1, C, num_latent_frames, H_l, W_l] in legacy mode.

    Returns generated latents: [1, C, num_latent_frames, H_l, W_l].
    """
    # ── Argument validation for H2 mode ──
    use_window_mode = train_window_size is not None and train_window_size > 0
    if use_window_mode:
        assert 1 <= train_window_size <= num_latent_frames - 1, (
            f"train_window_size={train_window_size} must be in "
            f"[1, num_latent_frames-1={num_latent_frames-1}]"
        )
        if train_window_start is None:
            train_window_start = 1  # default: window at the very front
        assert 1 <= train_window_start <= num_latent_frames - train_window_size, (
            f"train_window_start={train_window_start} must be in "
            f"[1, {num_latent_frames - train_window_size}] "
            f"(num_latent_frames={num_latent_frames}, K={train_window_size})"
        )
        # Buffer = K frames immediately before the training window (same size
        # as the window per LongLive Figure 4(c) intuition: the model needs
        # to see a near-history of ITS OWN scale to learn error correction).
        # If the window starts close to frame 1, the buffer auto-shrinks
        # (and fills with whatever GT/cond is there). If T_start <= K, the
        # buffer is empty and we go straight from GT (or cond) into the window.
        buf_size = train_window_size
        buf_start = max(1, train_window_start - buf_size)  # frame idx
        buf_end = train_window_start  # exclusive
        # GT range is [1, buf_start) — everything before the buffer
        gt_start = 1
        gt_end = buf_start
        # Sanity check: GT segment requested but no GT provided
        if gt_end > gt_start and gt_far_latents is None:
            raise ValueError(
                f"H2 window mode with T_start={train_window_start}, K={train_window_size} "
                f"requires GT for far history frames [{gt_start}, {gt_end}), "
                f"but gt_far_latents is None. Pass x0_teacher or set T_start <= K."
            )
        if gt_far_latents is not None:
            assert gt_far_latents.shape[2] >= gt_end, (
                f"gt_far_latents has {gt_far_latents.shape[2]} frames but "
                f"H2 needs at least {gt_end}"
            )
    else:
        # Legacy path — initialize sentinels so the rest of the code can
        # still inspect them without branching everywhere.
        buf_size = 0
        buf_start = 0
        buf_end = 1  # so frame_idx=1 is treated as "training" (legacy with_grad)
        gt_start = 0
        gt_end = 0

    batch_size = 1
    _, num_channels, _, latent_h, latent_w = img_latent.shape
    p_t, p_h, p_w = patch_size

    # Separate KV caches for cond / uncond branches so they don't overwrite each other.
    attention_kwargs = {"past_key_values": DynamicCache(max_frames=max_cache_frames)}
    attention_kwargs_uncond = (
        {"past_key_values": DynamicCache(max_frames=max_cache_frames)} if do_cfg else None
    )

    # ── Per-frame control slicing helper (for DMD generator skel conditioning) ──
    # Same pattern as streaming_v_flow_matching_forward._ctrl_f at L2121.
    # Returns the f-th frame slice [1, C, 1, H, W] cast to dtype, or None if
    # control_video_latent was not provided (= pre-control-distill behavior).
    def _ctrl_f(f):
        if control_video_latent is None:
            return None
        # Defensive bounds check: if caller passes more frames than control has,
        # clamp to last frame (skel was 21F-only filtered upstream so this should
        # not normally trigger).
        f_clamped = min(f, control_video_latent.shape[2] - 1)
        return control_video_latent[:, :, f_clamped:f_clamped + 1].to(dtype)

    # ── Trace dump: one-time geometry summary (rank 0 only, opt-in) ──
    # Lets us confirm that the patch_size / latent shape / per-frame token
    # count we're computing match what the model actually consumes. Cheap
    # (one print) and only runs when caller opts in.
    if trace_dump:
        try:
            import torch.distributed as _dist
            _is_rank0 = (not _dist.is_initialized()) or _dist.get_rank() == 0
        except Exception:
            _is_rank0 = True
        if _is_rank0:
            cap_str = (str(max_cache_frames) if max_cache_frames is not None else "unbounded")
            cprint(
                f"  [trace:{trace_tag}] geometry | latent=[B={batch_size}, C={num_channels}, "
                f"F={num_latent_frames}, H={latent_h}, W={latent_w}] | patch=({p_t},{p_h},{p_w}) "
                f"| frame_seq=({latent_h//p_h}x{latent_w//p_w})={(latent_h//p_h)*(latent_w//p_w)} tokens "
                f"| sliding_mode={max_cache_frames is not None} | cap={cap_str} "
                f"| do_cfg={do_cfg} | NIS={num_inference_steps} | with_grad={with_grad} "
                f"| SGT={stochastic_grad_truncation} | window_K={train_window_size} | T_start={train_window_start}",
                "blue",
            )

    # ── LongLive / StreamingLLM sliding-window position assignment ──
    # When max_cache_frames is set, the cache is a rolling window: as frames
    # are appended, the OLDEST non-sink frames get evicted. The naive
    # alternative (kept for max_cache_frames=None) bakes RoPE into K at
    # absolute frame index — but after eviction, the kept K's would still
    # carry rotary phases for stale positions, creating gaps the model
    # never trained on.
    #
    # Sliding mode (this branch) instead:
    #   * stores RAW K in the cache (no rotary baked in)
    #   * recomputes rotary on the combined (cache K + new K) every forward,
    #     using SLOT positions: [0..frames_in_cache] (always contiguous!)
    #   * Q gets rotary at slot position = frames_in_cache (next slot)
    # The model only ever sees positions in [0, max_cache_frames], regardless
    # of how long the actual video is. This is exactly what StreamingLLM
    # (Xiao et al. ICLR 2024) §3.2 calls "positions within the cache" and
    # what LongLive (Yang et al. 2025) §3.3 inherits.
    #
    # Disable sliding mode (i.e. fall back to the legacy absolute-position
    # behaviour) when max_cache_frames is None — this preserves backward
    # compat for the H1 / H2 three-segment recipes that don't rely on cache
    # eviction.
    sliding_mode = (max_cache_frames is not None)

    # Pre-compute the per-frame token grid used both for Q and for slot rotaries.
    _frame_seq_h = latent_h // p_h
    _frame_seq_w = latent_w // p_w
    _frame_seq_len = _frame_seq_h * _frame_seq_w

    def _build_sliding_position_ids(prev_count: int, frame_idx: int = 0):
        """Build (q_position_ids, k_position_ids) for the next forward.

        prev_count: how many frames are CURRENTLY in the cache (BEFORE this
                    forward appends one). Equal for cond / uncond caches when
                    they're advanced in lockstep (we always do that).
        frame_idx:  absolute index of the frame being processed in the source
                    video (only consulted when pe_mode == "absolute").

        Q is one frame:
          - slot mode: at slot `min(prev_count, max_cache_frames)`
          - absolute mode: at absolute frame index `frame_idx`
        K spans the post-update cache:
          - slot mode: positions [0..q_slot] inclusive
          - absolute mode: when cache is past trim threshold,
              [0] + [frame_idx-cap+1, frame_idx-cap+2, ..., frame_idx]
            otherwise [0..prev_count] inclusive (matches actual frame indices
            since no frame has been dropped yet).

        Returns two tensors of shape [B, seq, 3] for (t, h, w) coords.
        """
        cap = max_cache_frames
        if pe_mode == "absolute":
            q_t_val = frame_idx
            # cache.update returns full = cat(cache_K, new_K). Cache_K layout
            # at this moment (before update): trim happens AFTER append, so
            # what's currently in cache reflects the previous frame's trim.
            # Cases:
            #   - frame_idx <= cap-1: no trim yet, cache_K = [0..frame_idx-1],
            #       full K = [0..frame_idx]
            #   - frame_idx >= cap: trim already ran, cache_K = [0]+[frame_idx-cap+1..frame_idx-1],
            #       full K = [0]+[frame_idx-cap+1..frame_idx]
            if frame_idx >= cap and prev_count >= cap:
                k_t_list = [0] + list(range(frame_idx - cap + 1, frame_idx + 1))
            else:
                # No trim has happened yet OR cache underfilled (frame_idx < prev_count
                # is impossible by construction: caller advances frame_idx in lockstep
                # with cache writes). K positions = [0..frame_idx] inclusive.
                k_t_list = list(range(frame_idx + 1))
            q_pos = torch.cartesian_prod(
                torch.tensor([q_t_val], dtype=torch.long, device=device),
                torch.arange(_frame_seq_h, dtype=torch.long, device=device),
                torch.arange(_frame_seq_w, dtype=torch.long, device=device),
            ).unsqueeze(0).repeat(batch_size, 1, 1)
            k_pos = torch.cartesian_prod(
                torch.tensor(k_t_list, dtype=torch.long, device=device),
                torch.arange(_frame_seq_h, dtype=torch.long, device=device),
                torch.arange(_frame_seq_w, dtype=torch.long, device=device),
            ).unsqueeze(0).repeat(batch_size, 1, 1)
            return q_pos, k_pos
        # ── slot mode (legacy default) ──
        # Q slot — clamped at cap (after cache fills, Q always lives at slot=cap)
        q_slot = min(prev_count, cap)
        q_pos = torch.cartesian_prod(
            torch.tensor([q_slot], dtype=torch.long, device=device),
            torch.arange(_frame_seq_h, dtype=torch.long, device=device),
            torch.arange(_frame_seq_w, dtype=torch.long, device=device),
        ).unsqueeze(0).repeat(batch_size, 1, 1)
        # K slots — [0..q_slot] inclusive  (cache K = q_slot frames @ slots
        # 0..q_slot-1, plus the new K being appended this forward @ slot q_slot).
        k_pos = torch.cartesian_prod(
            torch.arange(q_slot + 1, dtype=torch.long, device=device),
            torch.arange(_frame_seq_h, dtype=torch.long, device=device),
            torch.arange(_frame_seq_w, dtype=torch.long, device=device),
        ).unsqueeze(0).repeat(batch_size, 1, 1)
        return q_pos, k_pos

    def _set_sliding_kwargs(akw, akw_uncond, frame_idx: int = 0):
        """Stamp sliding_mode + sliding_k_position_ids into the attention_kwargs
        dicts so the model.forward picks the sliding code path. Returns the
        Q position_ids the caller should pass into model.forward (the K
        side is encoded in akw[\"sliding_k_position_ids\"]).

        In `pe_mode=="absolute"` we DO NOT engage the sliding-mode K rotary
        bookkeeping at all — K is written into the cache with RoPE already
        baked at its absolute frame index (matching LongLive/SF/Distillation
        reference implementations). The caller passes Q position_ids built
        from the absolute frame_idx, and the legacy attention path applies
        RoPE to (Q, K) before write — DynamicCache trims keep the physically-
        displaced-but-RoPE-baked K untouched, identical to the reference.
        """
        # Both caches advance in lockstep so prev_count is the same for both.
        # Look up via cache._frame_count[0] (layer 0 — all layers stay synced
        # since update() is called per-layer on the same forward).
        cache = akw["past_key_values"]
        prev_count = cache._frame_count.get(0, 0)
        q_pos, k_pos = _build_sliding_position_ids(prev_count, frame_idx=frame_idx)
        if pe_mode == "absolute":
            # Bypass the sliding-mode K-rotary path. Tell the cache to NOT
            # try to renumber/trim K positions — its existing trim policy
            # (drop oldest middle, keep sink + recent) is already what we
            # want for abs PE.
            akw.pop("sliding_mode", None)
            akw.pop("sliding_k_position_ids", None)
            if akw_uncond is not None:
                akw_uncond.pop("sliding_mode", None)
                akw_uncond.pop("sliding_k_position_ids", None)
        else:
            akw["sliding_mode"] = True
            akw["sliding_k_position_ids"] = k_pos
            if akw_uncond is not None:
                akw_uncond["sliding_mode"] = True
                akw_uncond["sliding_k_position_ids"] = k_pos  # cond/uncond same geometry
        # Trace: per-frame slot geometry — confirms (a) slot evolution
        # 0,1,...,cap,cap,cap,... and (b) trim event happens exactly when
        # prev_count first exceeds cap. Rank-0 only.
        if trace_dump:
            try:
                import torch.distributed as _dist
                _is_rank0 = (not _dist.is_initialized()) or _dist.get_rank() == 0
            except Exception:
                _is_rank0 = True
            if _is_rank0:
                cap = max_cache_frames
                q_slot = min(prev_count, cap)
                trim_marker = " ← TRIM" if (prev_count >= cap) else ""
                # Cache K-tensor seq-len BEFORE this forward, by layer 0
                cur_k_len = (
                    cache.keys[0].shape[1]
                    if 0 in cache.keys else 0
                )
                cprint(
                    f"  [trace:{trace_tag}] frame_set pe={pe_mode} | prev_count={prev_count} "
                    f"| frame_idx={frame_idx} | q_slot={q_slot} | k_slots=[0..{q_slot}] "
                    f"| q_pos.shape={tuple(q_pos.shape)} | k_pos.shape={tuple(k_pos.shape)} "
                    f"| cur_cache_K_seq_len={cur_k_len}{trim_marker}",
                    "blue",
                )
        return q_pos

    condition = img_latent.to(device=device, dtype=dtype)
    prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
    if do_cfg and negative_prompt_embeds is not None:
        negative_prompt_embeds = negative_prompt_embeds.to(device=device, dtype=dtype)

    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps

    # ── Stochastic gradient truncation (Self-Forcing Algorithm 1 line 4) ──
    # Sample ONE s ∈ [1, T] for this video; same s applies to all frames.
    # When with_grad=True AND num_inference_steps>1, only the j==(s-1) step
    # carries grad. For num_inference_steps==1 this degenerates to the
    # original behaviour (the only step IS the gradient step).
    #
    # Indexing is 0-based here; the paper's 1-based "s ∈ [1, T]" maps to
    # our grad_step_idx ∈ [0, T-1].
    #
    # ── ZeRO-3 SAFETY: must be IDENTICAL across all ranks ────────────────────
    # Under ZeRO-3 the grad-bearing forward unshards a much larger param set
    # than the no_grad forwards. If two ranks pick different `grad_step_idx`,
    # they issue different sequences of all-gather collectives on the param
    # shards → NCCL desync → 30-min watchdog timeout → SIGABRT.
    # Concrete repro on 2-GPU bf16 H20 + num_inference_steps=4 (variant H):
    #   rank 0 SeqNum=952 _ALLGATHER_BASE NumelIn=294912    (no-grad path)
    #   rank 1 SeqNum=952 _ALLGATHER_BASE NumelIn=33718272  (grad path)
    # We cannot reuse `generator_rng` because it's seeded with args.seed+rank
    # (see main()), so it produces different draws per rank by design — that
    # rank-dep seeding is REQUIRED for the per-rank initial noise draws to
    # be different (otherwise every rank would generate the same scene),
    # so we must NOT change it. Instead we sample on rank 0 with torch's
    # default global RNG (also rank-dep, but we only read rank 0) and
    # broadcast the int to every rank. Cost: one tiny long-tensor broadcast
    # per generate_streaming call. Single-process / non-init'd dist gracefully
    # falls back to a plain torch.randint draw.
    if with_grad and stochastic_grad_truncation and num_inference_steps > 1:
        # Debug knob: force a fixed grad_step_idx for bit-identical
        # reproducibility tests (e.g. validating gradient checkpointing).
        # NEVER set this in production — it removes the stochastic averaging
        # over denoising timesteps that is the whole point of SGT.
        _force_sgt = os.environ.get("_DEBUG_FORCE_SGT_IDX", "")
        if _force_sgt != "":
            grad_step_idx = int(_force_sgt) % num_inference_steps
        else:
            try:
                import torch.distributed as _dist
                if _dist.is_available() and _dist.is_initialized():
                    _gsi = torch.empty(1, dtype=torch.long, device=device)
                    if _dist.get_rank() == 0:
                        _gsi.fill_(int(torch.randint(
                            low=0, high=num_inference_steps, size=(1,),
                        ).item()))
                    _dist.broadcast(_gsi, src=0)
                    grad_step_idx = int(_gsi.item())
                else:
                    grad_step_idx = int(torch.randint(
                        low=0, high=num_inference_steps, size=(1,),
                    ).item())
            except Exception:
                grad_step_idx = int(torch.randint(
                    low=0, high=num_inference_steps, size=(1,),
                ).item())
    else:
        # Plain mode: every step has grad if with_grad, else none. Setting
        # grad_step_idx = -1 means "no special truncation step" — the
        # legacy code path is selected by checking `stochastic_grad_truncation`
        # below and ignoring this value.
        grad_step_idx = -1

    # Trace: log SGT step pick (rank 0 only) so a `grep '\[trace.*sgt'` in
    # logs shows the per-call grad_step_idx and confirms broadcast worked
    # (would be obvious from per-rank prints if it ever desyncs again).
    if trace_dump and grad_step_idx >= 0:
        try:
            import torch.distributed as _dist
            if (not _dist.is_initialized()) or _dist.get_rank() == 0:
                cprint(
                    f"  [trace:{trace_tag}] sgt | grad_step_idx={grad_step_idx} / NIS={num_inference_steps} "
                    f"(broadcast to all ranks)",
                    "blue",
                )
        except Exception:
            pass

    # ── Record this call's SGT pick for the per-step diag log (rank 0 only) ──
    # This runs unconditionally (not gated on trace_dump) because we want the
    # histogram to accumulate across ALL training steps, not just the rare
    # debug-trace step. The function is a no-op when grad_step_idx < 0
    # (plain mode / no SGT) so legacy non-SGT callers pay nothing.
    #
    # Subtlety: in DMD steps `_streaming_generate` may be called twice — once
    # for the generator update (grad-bearing → meaningful pick) and once for
    # critic data prep (with_grad=False → grad_step_idx == -1 → no-op here).
    # So `last_grad_step_idx` ends up reflecting the grad-bearing call as
    # intended. The ODE warmup branch only ever calls with_grad=True, so
    # there's no ambiguity there.
    _record_sgt_pick(grad_step_idx)

    # Final container for all generated latents (kept on the autograd graph if with_grad).
    # We collect per-frame results in a list and torch.cat them at the end so we don't
    # have to do in-place slice writes into a leaf tensor (which breaks autograd).
    # Legacy mode: prepend cond so output shape = [B, C, num_latent_frames, H, W].
    # H2 mode: cond is NOT in the output — it's only in the KV cache; DMD operates
    # only on the K training-window frames.
    if use_window_mode:
        collected = []
    else:
        collected = [condition]  # [B, C, 1, H, W] for frame 0

    def _seed_cache_with_condition():
        """One-shot: write the clean condition frame's K/V into both caches (no grad)."""
        # In sliding mode the cond frame goes into slot 0 (cache empty);
        # in legacy mode it uses absolute frame index 0. Either way, cond is
        # frame 0 → slot 0 → identical position_ids, but we route through
        # _set_sliding_kwargs to install sliding_mode flag in attention_kwargs.
        if sliding_mode:
            cond_position_ids = _set_sliding_kwargs(
                attention_kwargs, attention_kwargs_uncond, frame_idx=0
            )
        else:
            cond_position_ids = torch.cartesian_prod(
                torch.arange(0, 1, dtype=torch.long, device=device),
                torch.arange(_frame_seq_h, dtype=torch.long, device=device),
                torch.arange(_frame_seq_w, dtype=torch.long, device=device),
            ).unsqueeze(0).repeat(batch_size, 1, 1)
        cond_t = torch.zeros(batch_size, _frame_seq_len, dtype=torch.long, device=device)

        with torch.no_grad():
            attention_kwargs["past_key_values"].unfreeze()
            generator(
                hidden_states=condition.to(dtype),
                timestep=cond_t,
                encoder_hidden_states=prompt_embeds,
                encoder_hidden_states_image=None,
                attention_kwargs=attention_kwargs,
                position_ids=cond_position_ids,
                return_dict=False,
                control_video_latent=_ctrl_f(0),
            )
            if do_cfg and attention_kwargs_uncond is not None:
                attention_kwargs_uncond["past_key_values"].unfreeze()
                generator(
                    hidden_states=condition.to(dtype),
                    timestep=cond_t,
                    encoder_hidden_states=negative_prompt_embeds,
                    encoder_hidden_states_image=None,
                    attention_kwargs=attention_kwargs_uncond,
                    position_ids=cond_position_ids,
                    return_dict=False,
                    control_video_latent=_ctrl_f(0),
                )

    # MEM PROBE 1: before seeding the cache with cond frame.
    # Compare against MEM PROBE 0 (caller-side, "pre-streaming") to see how
    # much the function-entry overhead (cache alloc, sliding-mode setup) adds.
    _mem_probe("pre-seed-cache", trace_tag, enabled=trace_dump, do_sync=False)

    _seed_cache_with_condition()

    # MEM PROBE 2: after seeding the cache with cond frame.
    # The seed forward is no_grad but still triggers ZeRO-3 all_gather across
    # all 30 transformer blocks. If `alloc` jumps significantly here, it
    # confirms that DeepSpeed isn't releasing the all_gather'd params back
    # to CPU after each block (the leading hypothesis for the H3 OOM).
    _mem_probe("post-seed-cache", trace_tag, enabled=trace_dump)

    # In H2 window mode, the rollout stops AFTER the training window — we
    # don't need to generate frames past T_start+K (they wouldn't contribute
    # to DMD loss anyway). This bounds the wall-time as T_start grows.
    end_frame = (
        train_window_start + train_window_size
        if use_window_mode
        else num_latent_frames
    )

    # Each subsequent frame (idx 1 .. end_frame - 1) is a length-1 chunk that
    # attends to the accumulated clean K/V cache.
    for frame_idx in range(1, end_frame):
        # ── H2 segment dispatch ──
        # Decide what this frame is: GT far history / SF buffer / training window.
        # Outside H2 mode, every frame is treated as "training" (legacy behaviour).
        if use_window_mode:
            in_gt = (gt_start <= frame_idx < gt_end)
            in_buf = (buf_start <= frame_idx < buf_end)
            in_window = (train_window_start <= frame_idx < end_frame)
            assert in_gt + in_buf + in_window == 1, (
                f"frame {frame_idx} doesn't fall in exactly one segment "
                f"(gt=[{gt_start},{gt_end}), buf=[{buf_start},{buf_end}), "
                f"window=[{train_window_start},{end_frame}))"
            )
        else:
            in_gt = False
            in_buf = False
            in_window = True  # legacy: treat every frame as training (with_grad ctx)

        # ── Position ids — depends on mode ──
        # Sliding mode (LongLive / StreamingLLM): Q lives at the next slot in
        # the rolling KV window, NOT at absolute frame_idx. K positions are
        # also slot-based (set in attention_kwargs by _set_sliding_kwargs).
        # See _build_sliding_position_ids docstring for the full mapping.
        # `prev_count` (cache size BEFORE this frame's writes) is identical
        # across all forwards within this frame iteration: the denoise loop
        # runs cache-FROZEN (no increment), and the clean-K/V write reads
        # _frame_count BEFORE its update() call increments it.
        if sliding_mode:
            position_ids = _set_sliding_kwargs(
                attention_kwargs, attention_kwargs_uncond, frame_idx=frame_idx
            )
        else:
            # Legacy (no max_cache_frames): RoPE encodes absolute frame index.
            position_ids = torch.cartesian_prod(
                torch.arange(1 // p_t, dtype=torch.long, device=device) + frame_idx,
                torch.arange(_frame_seq_h, dtype=torch.long, device=device),
                torch.arange(_frame_seq_w, dtype=torch.long, device=device),
            ).unsqueeze(0).repeat(batch_size, 1, 1)

        # ── GT TEACHER-FORCING SEGMENT ──
        # Just write clean KV from the GT latent at this frame. No denoising
        # needed — we already have the "answer" (teacher's x0 at this frame).
        # This mirrors what _seed_cache_with_condition does for frame 0.
        # Cost: 1 forward at t=0 (vs NIS forwards for buffer/window).
        if in_gt:
            gt_frame = gt_far_latents[:, :, frame_idx:frame_idx+1].to(
                device=device, dtype=dtype
            )
            frame_seq_len = (latent_h // p_h) * (latent_w // p_w)
            clean_t = torch.zeros(batch_size, frame_seq_len, dtype=torch.long, device=device)
            with torch.no_grad():
                attention_kwargs["past_key_values"].unfreeze()
                generator(
                    hidden_states=gt_frame,
                    timestep=clean_t,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_image=None,
                    attention_kwargs=attention_kwargs,
                    position_ids=position_ids,
                    return_dict=False,
                    control_video_latent=_ctrl_f(frame_idx),
                )
                if do_cfg and attention_kwargs_uncond is not None:
                    attention_kwargs_uncond["past_key_values"].unfreeze()
                    generator(
                        hidden_states=gt_frame,
                        timestep=clean_t,
                        encoder_hidden_states=negative_prompt_embeds,
                        encoder_hidden_states_image=None,
                        attention_kwargs=attention_kwargs_uncond,
                        position_ids=position_ids,
                        return_dict=False,
                        control_video_latent=_ctrl_f(frame_idx),
                    )
            # Don't append to `collected` — GT frames are not part of the
            # output (they're history-only).
            continue

        # ── BUFFER + TRAINING WINDOW: full denoise + clean KV write ──
        # The only difference between BUF and WIN is whether the denoise
        # loop runs with grad. We compute the effective `with_grad` flag
        # for this frame here.
        frame_with_grad = with_grad and in_window  # buffer is no_grad

        num_chunk_frames = 1
        shape_chunk = (batch_size, num_channels, num_chunk_frames, latent_h, latent_w)
        # Sample the initial noise on CPU for reproducibility, then move to device.
        chunk_latents = randn_tensor(shape_chunk, generator=generator_rng, device=device, dtype=dtype)

        # Per-chunk scheduler so step_index resets cleanly.
        frame_scheduler = type(scheduler)(**scheduler.config)
        frame_scheduler.set_timesteps(num_inference_steps, device=device)

        # ── Denoise loop: cache stays FROZEN for the whole loop ──
        attention_kwargs["past_key_values"].freeze()
        if do_cfg and attention_kwargs_uncond is not None:
            attention_kwargs_uncond["past_key_values"].freeze()

        # MEM PROBE 3: about to enter the denoise loop for THIS frame.
        # We probe at frame_idx=1 (the OOM site for H3 ODE warmup; first
        # gradient-bearing forward of the run) and the LAST frame (peak
        # of the autograd graph). Skipping intermediate frames keeps logs
        # readable while still catching both extremes.
        # No do_sync to avoid serializing the denoise loop on every iter.
        if trace_dump and (frame_idx == 1 or frame_idx == end_frame - 1):
            _mem_probe(f"pre-frame{frame_idx}-denoise", trace_tag,
                       enabled=True, do_sync=True)

        x = chunk_latents
        # ── Denoise loop ──
        # Two modes:
        #   (A) Plain (stochastic_grad_truncation=False or num_inference_steps==1):
        #       wrap the whole loop in enable_grad()/no_grad() per `frame_with_grad`.
        #       Every step does a normal scheduler.step update.
        #   (B) Stochastic gradient truncation (frame_with_grad + truncation flag,
        #       num_inference_steps>1): steps j != grad_step_idx run no_grad
        #       and advance via scheduler.step(); step j == grad_step_idx runs
        #       with grad, predicts x_0 from v_pred via flow-matching identity
        #         x_0 = x_t - sigma_t * v_pred
        #       and uses that x_0 as the FINAL output (loop break — we do
        #       NOT continue denoising). This matches Algorithm 1 lines 8-12
        #       and reduces the autograd graph from T forwards/frame to 1.
        truncation_active = (
            frame_with_grad and stochastic_grad_truncation and num_inference_steps > 1
        )

        # Capture per-frame skel slice so closure reuses it across denoise steps
        # (frame_idx is stable within this iteration of the outer loop).
        _ctrl_this_frame = _ctrl_f(frame_idx)

        def _model_forward(latent_model_input, timestep):
            """One generator forward (with optional CFG combine). Caller decides
            grad-vs-no-grad context."""
            noise_pred = generator(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                encoder_hidden_states_image=None,
                attention_kwargs=attention_kwargs,
                position_ids=position_ids,
                return_dict=False,
                control_video_latent=_ctrl_this_frame,
            )[0]
            if do_cfg and attention_kwargs_uncond is not None:
                noise_uncond = generator(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=negative_prompt_embeds,
                    encoder_hidden_states_image=None,
                    attention_kwargs=attention_kwargs_uncond,
                    position_ids=position_ids,
                    return_dict=False,
                    control_video_latent=_ctrl_this_frame,
                )[0]
                noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)
            return noise_pred

        # ── Cache-aware activation checkpointing for the grad-bearing fwd ──
        # Why a custom wrapper (not generator.enable_gradient_checkpointing()):
        # the streaming forward mutates DynamicCache between frames (each
        # frame's clean-K/V write appends to cache). torch.utils.checkpoint
        # replays the forward at backward time, but by then `cache.keys[...]`
        # has grown — recomputed Q,K shapes won't match the saved metadata
        # and we get `CheckpointError: Recomputed values ... different metadata`.
        #
        # Fix: snapshot the cache AND every mutable kwarg (position_ids,
        # sliding_k_position_ids) BEFORE the grad-bearing forward, then
        # restore inside the recompute. snapshot()/restore() are O(1)
        # (shallow dict copies of tensor refs — update() always replaces
        # dict entries with fresh tensors, never mutates in place) so the
        # math is bit-identical to the no-checkpoint path.
        #
        # Closure trap to avoid: _model_forward reads `position_ids`,
        # `attention_kwargs` etc. from enclosing scope. By backward time,
        # the outer loop has advanced — those refs now point at frame N's
        # values, not the original frame's. Also `attention_kwargs.clear()`
        # runs at function exit before backward. Hence we bind a frozen
        # snapshot of all the per-frame state.
        # Gated by env STREAMING_GRAD_CHECKPOINT=1 (default off).
        _use_grad_ckpt = (
            os.environ.get("STREAMING_GRAD_CHECKPOINT", "0") == "1"
            and frame_with_grad
        )

        def _make_ckpt_fwd():
            """Capture this-frame state into a closure that survives until backward.

            Returns a fn (latent_model_input, timestep) → noise_pred that:
              1) On forward pass: snapshots cache + freezes a copy of
                 attention_kwargs / position_ids, runs the generator forward.
              2) On backward recompute: restores cache to the snapshotted
                 state, rebuilds attention_kwargs from the frozen copy, and
                 reruns the same forward → same outputs (bit-identical).
            """
            cache = attention_kwargs["past_key_values"]
            cache_uncond = (
                attention_kwargs_uncond["past_key_values"]
                if (do_cfg and attention_kwargs_uncond is not None) else None
            )
            cache_snap = cache.snapshot()
            cache_uncond_snap = cache_uncond.snapshot() if cache_uncond is not None else None

            # Freeze a copy of attention_kwargs WITHOUT the cache (cache is
            # restored separately). Per-frame keys like "sliding_mode" and
            # "sliding_k_position_ids" stay frozen here.
            akw_static = {k: v for k, v in attention_kwargs.items() if k != "past_key_values"}
            akw_uncond_static = (
                {k: v for k, v in attention_kwargs_uncond.items() if k != "past_key_values"}
                if (do_cfg and attention_kwargs_uncond is not None) else None
            )
            pos_ids_static = position_ids  # tensor — held by reference, immutable
            prompt_embeds_static = prompt_embeds
            negative_prompt_embeds_static = negative_prompt_embeds

            def _fn(lmi, ts):
                # Restore cache state so recompute sees the EXACT state that
                # forward saw. No-op on forward (snapshot is the current
                # state); meaningful on backward recompute.
                cache.restore(cache_snap)
                if cache_uncond is not None and cache_uncond_snap is not None:
                    cache_uncond.restore(cache_uncond_snap)

                # Rebuild attention_kwargs from frozen copy + restored cache.
                # Use a fresh dict so any later .clear() on the outer
                # attention_kwargs can't disturb this call's inputs.
                akw_local = dict(akw_static)
                akw_local["past_key_values"] = cache
                akw_uncond_local = None
                if akw_uncond_static is not None:
                    akw_uncond_local = dict(akw_uncond_static)
                    akw_uncond_local["past_key_values"] = cache_uncond

                noise_pred = generator(
                    hidden_states=lmi,
                    timestep=ts,
                    encoder_hidden_states=prompt_embeds_static,
                    encoder_hidden_states_image=None,
                    attention_kwargs=akw_local,
                    position_ids=pos_ids_static,
                    return_dict=False,
                    control_video_latent=_ctrl_this_frame,
                )[0]
                if do_cfg and akw_uncond_local is not None:
                    noise_uncond = generator(
                        hidden_states=lmi,
                        timestep=ts,
                        encoder_hidden_states=negative_prompt_embeds_static,
                        encoder_hidden_states_image=None,
                        attention_kwargs=akw_uncond_local,
                        position_ids=pos_ids_static,
                        return_dict=False,
                        control_video_latent=_ctrl_this_frame,
                    )[0]
                    noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)
                return noise_pred

            from torch.utils.checkpoint import checkpoint as _ckpt
            def _ckpt_call(lmi, ts):
                return _ckpt(_fn, lmi, ts, use_reentrant=False)
            return _ckpt_call

        _grad_fwd = _make_ckpt_fwd() if _use_grad_ckpt else _model_forward

        if not truncation_active:
            # Path (A): legacy behaviour. enable/disable grad for the whole loop.
            # Use `frame_with_grad` (NOT outer `with_grad`) so buffer frames in
            # H2 mode run no_grad even when the call has with_grad=True.
            grad_ctx = torch.enable_grad() if frame_with_grad else torch.no_grad()
            with grad_ctx:
                for i, t in enumerate(timesteps):
                    latent_model_input = x.to(dtype)
                    timestep = t.unsqueeze(0).expand(batch_size, -1)
                    # Grad-bearing path uses checkpoint wrapper (when enabled);
                    # no-grad path skips it (no activations to save anyway).
                    if frame_with_grad:
                        noise_pred = _grad_fwd(latent_model_input, timestep)
                    else:
                        noise_pred = _model_forward(latent_model_input, timestep)
                    # Functional update — no in-place writes that would conflict with autograd.
                    x = frame_scheduler.step(noise_pred, t, x, return_dict=False)[0]
        else:
            # Path (B): stochastic gradient truncation.
            # frame_scheduler.sigmas has length T+1; sigmas[i] corresponds to
            # timesteps[i]. We use it to compute x_0 from v_pred at the grad step.
            for i, t in enumerate(timesteps):
                latent_model_input = x.to(dtype)
                timestep = t.unsqueeze(0).expand(batch_size, -1)
                if i == grad_step_idx:
                    # Final, gradient-bearing step: predict noise/v under
                    # enable_grad, recover x_0, USE THAT AS THE OUTPUT,
                    # and break out of the denoise loop.
                    with torch.enable_grad():
                        noise_pred = _grad_fwd(latent_model_input, timestep)
                        # Flow-matching identity (works for UniPC w/ flow_prediction
                        # and FlowMatchEulerDiscreteScheduler):
                        #   x_t = (1 - sigma_t) * x_0 + sigma_t * noise
                        #   model target v = noise - x_0
                        #   ⇒ x_0 = x_t - sigma_t * v_pred
                        sigma_t = frame_scheduler.sigmas[i].to(
                            device=x.device, dtype=x.dtype
                        )
                        x = x - sigma_t * noise_pred
                    break
                else:
                    # No-grad denoising step.
                    with torch.no_grad():
                        noise_pred = _model_forward(latent_model_input, timestep)
                        x = frame_scheduler.step(noise_pred, t, x, return_dict=False)[0]

        # ── Persist CLEAN K/V into the cache ──
        # This is a separate forward at t=0 of the just-denoised latent. It runs under
        # no_grad with .detach() so the cache holds tensors disconnected from the graph;
        # otherwise the next frame's attention would chain gradients through every prior
        # frame and the autograd graph would explode.
        clean_input = x.detach().to(dtype)
        frame_seq_len = (latent_h // p_h) * (latent_w // p_w)
        clean_t = torch.zeros(batch_size, frame_seq_len, dtype=torch.long, device=device)
        with torch.no_grad():
            attention_kwargs["past_key_values"].unfreeze()
            generator(
                hidden_states=clean_input,
                timestep=clean_t,
                encoder_hidden_states=prompt_embeds,
                encoder_hidden_states_image=None,
                attention_kwargs=attention_kwargs,
                position_ids=position_ids,
                return_dict=False,
                control_video_latent=_ctrl_f(frame_idx),
            )
            if do_cfg and attention_kwargs_uncond is not None:
                attention_kwargs_uncond["past_key_values"].unfreeze()
                generator(
                    hidden_states=clean_input,
                    timestep=clean_t,
                    encoder_hidden_states=negative_prompt_embeds,
                    encoder_hidden_states_image=None,
                    attention_kwargs=attention_kwargs_uncond,
                    position_ids=position_ids,
                    return_dict=False,
                    control_video_latent=_ctrl_f(frame_idx),
                )

        # Append to output ONLY for training-window frames. In legacy mode
        # `in_window=True` for every frame so this matches old behaviour.
        # In H2 mode buffer frames are skipped (they ran no_grad just to
        # populate KV; they're not part of the DMD loss target).
        if in_window:
            collected.append(x)

        # MEM PROBE 4: after this frame's denoise + clean K/V write.
        # Same cadence as PROBE 3 (frame 1 and last frame). At frame_idx=1
        # this is the "after first frame fully done" snapshot — compare
        # against pre-frame1 to see the per-frame autograd-graph delta.
        # At end_frame-1 we get the cumulative peak.
        if trace_dump and (frame_idx == 1 or frame_idx == end_frame - 1):
            _mem_probe(f"post-frame{frame_idx}-done", trace_tag,
                       enabled=True, do_sync=True)

    # ── Assemble output ──
    if use_window_mode:
        # In H2 mode, `collected` contains only the K window frames (no cond,
        # no GT, no buffer). DMD loss expects [B, C, F, H, W] with F=K. We do
        # NOT prepend the cond frame here — DMD operates on the actual generated
        # window, and including cond would dilute the per-frame gradient signal
        # for the K frames we actually trained (cond's "loss" is a no-op since
        # it equals the input image latent and DMD would just denoise it).
        all_latents = torch.cat(collected, dim=2)  # [B, C, K, H, W]
        assert all_latents.shape[2] == train_window_size, (
            f"H2 collected {all_latents.shape[2]} window frames, "
            f"expected K={train_window_size}"
        )
    else:
        all_latents = torch.cat(collected, dim=2)  # [B, C, num_latent_frames, H, W]

    # Drop cache references before returning.
    attention_kwargs.clear()
    if attention_kwargs_uncond is not None:
        attention_kwargs_uncond.clear()
    if not with_grad:
        torch.cuda.empty_cache()

    # MEM PROBE 5: function exit. Compare against post-frame{N-1} to see how
    # much was freed by cache.clear() + (no_grad path only) empty_cache().
    # In the with_grad path the autograd graph is still alive in `all_latents`
    # so this should be ~equal to post-frame{N-1} (and the caller will see
    # the bulk of the free only after backward + step).
    _mem_probe("exit-streaming", trace_tag, enabled=trace_dump)

    return all_latents


def generate_streaming(*args, **kwargs):
    """No-grad wrapper for use in critic data generation."""
    kwargs.setdefault("with_grad", False)
    with torch.no_grad():
        return _streaming_generate(*args, **kwargs)


def generate_streaming_with_grad(*args, **kwargs):
    """Grad-enabled wrapper for use in DMD generator loss."""
    kwargs["with_grad"] = True
    return _streaming_generate(*args, **kwargs)


# ──────────────────────────────────────────────────────────────
# Single-step v-space flow-matching forward (warmup primitive)
# ──────────────────────────────────────────────────────────────
def streaming_v_flow_matching_forward(
    generator,
    scheduler,
    gt_latents,                    # [1, C, F, H, W] — full GT video latent
    prompt_embeds,                 # [1, seq, dim]
    generator_rng: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
    patch_size,
    max_cache_frames: Optional[int] = None,
    trace_dump: bool = False,
    trace_tag: str = "v_flow",
    pe_mode: str = "slot",
    control_video_latent: Optional[torch.Tensor] = None,  # [1, C, F, H, W] for control distill
):
    """Single-step flow-matching forward with streaming GT teacher-forcing KV cache.

    This is the warmup primitive used by --ode_target_kind=v_flow.
    It REPLACES `_streaming_generate` for the warmup phase only; the DMD phase
    still uses `_streaming_generate` (4-step ODE rollout) unchanged.

    Why this exists
    ───────────────
    The --ode_target_kind=gt path runs a 4-step ODE student with
    Stochastic Gradient Truncation, then computes MSE in x_0 space:
        x_0_student = student.run_4_step_ode(noise, ...)
        loss = MSE(x_0_student, gt_latent)
    Cold-start attention overflow in the very first step → v_pred outliers in
    a few tokens → x_0 = x_t - sigma_t * v_pred reconstruction is dominated by
    those outliers → fp32 MSE overflows to inf. We catch this with a 3-layer
    defense (clamp / skip / fail-fast), but defense is a band-aid: the root
    cause is "4-step x_0 reconstruction with no GT signal in the input".

    This function fixes the root cause:
      * input is (1-σ)·GT + σ·noise (NOT pure noise) → cold attention always
        sees a GT-anchored input, so v_pred can't drift to outlier values
      * loss target is `noise - GT` (v-space, raw model output) → bounded O(1)
        regardless of σ; no σ amplification through reconstruction
      * 1 forward per frame (no 4-step feedback loop) → an outlier in one frame
        cannot self-amplify across denoise steps

    What's preserved from `_streaming_generate`
    ───────────────────────────────────────────
    * DynamicCache(max_frames=...) sliding KV cache (LongLive H3 axis)
    * Frame-0 sink + slot-based RoPE for sliding mode
    * Clean K/V write protocol after each frame's grad-bearing forward
      (NOT noisy K/V — same as the production streaming-generate path)
    * Per-frame teacher-forcing using GT latents

    What's different
    ────────────────
    * No denoise loop: just one grad-bearing forward at a single shared
      timestep `t` per call (sampled from generator_rng).
    * No SGT (no gradient truncation needed; only 1 grad-bearing forward
      per frame already).
    * No CFG branch (warmup doesn't need uncond classifier-free guidance).
    * Returns (v_pred, v_target) for caller to compute MSE — does NOT
      compute loss internally so caller can wrap it in clamp/skip diagnostics.

    Returns
    ───────
    v_pred:        [1, C, F-1, H, W] grad-bearing v predictions for frames 1..F-1
    v_target:      [1, C, F-1, H, W] = (noise - gt_latents)[:, :, 1:] (no grad)
    x0_for_decode: [1, C, F-1, H, W] x_0 reconstruction = noisy - σ·v_pred,
                   detached. For DEBUG IMAGE DECODE ONLY — loss is computed
                   in v-space on (v_pred, v_target). The flow-matching identity
                       noisy = (1-σ)·x_0 + σ·noise   AND
                       v     = noise - x_0
                   ⟹ noisy = x_0 + σ·v   ⟹   x_0 = noisy - σ·v
                   At step 0 (random init) v_pred ≈ 0 so x_0_recon ≈ noisy
                   (a noisy GT). As training converges v_pred → v_target so
                   x_0_recon → x_0 (clean GT). VAE-decoding this gives a
                   physically meaningful "predicted clean GT" image, unlike
                   decoding raw v_pred which the VAE never trained on.

    Frame 0 is the cond/sink frame; we don't predict v for it (it's clean by
    construction, exactly matches `model.py` L686 `targets[:, :, 1:]` and
    L963 `noise_pred[:, :, 1:]` — the sister codebase does the same skip).
    """
    batch_size, num_channels, F, latent_h, latent_w = gt_latents.shape
    p_t, p_h, p_w = patch_size
    _frame_seq_h = latent_h // p_h
    _frame_seq_w = latent_w // p_w
    _frame_seq_len = _frame_seq_h * _frame_seq_w
    sliding_mode = (max_cache_frames is not None)

    # ── Cache (single cache; no uncond/CFG branch in warmup) ──
    attention_kwargs = {"past_key_values": DynamicCache(max_frames=max_cache_frames)}
    # Diagnostic: when trace_dump is on AND pe_mode=absolute, propagate a
    # trace_pe flag into attention_kwargs so the abs PE legacy path inside
    # WanCausalAttnProcessor prints per-frame K-after-RoPE absmax + cache K
    # seqlen (layer-0 only, see model.py L141-167). Confirms (a) RoPE is
    # baked at write time, (b) cache K seqlen grows then plateaus at
    # cap*tokens_per_frame, (c) trims happen at the right step.
    if trace_dump and pe_mode == "absolute":
        attention_kwargs["trace_pe"] = True

    # ── Trace banner (rank 0) ──
    if trace_dump:
        try:
            import torch.distributed as _dist
            _is_rank0 = (not _dist.is_initialized()) or _dist.get_rank() == 0
        except Exception:
            _is_rank0 = True
        if _is_rank0:
            cap_str = (str(max_cache_frames) if max_cache_frames is not None else "unbounded")
            cprint(
                f"  [trace:{trace_tag}] geometry | latent=[B={batch_size}, C={num_channels}, "
                f"F={F}, H={latent_h}, W={latent_w}] | patch=({p_t},{p_h},{p_w}) "
                f"| frame_seq=({_frame_seq_h}x{_frame_seq_w})={_frame_seq_len} tokens "
                f"| sliding_mode={sliding_mode} | cap={cap_str} "
                f"| mode=single-step-flow-matching",
                "blue",
            )

    # ── Slot-aware position-id helpers (mirror _streaming_generate L489-563) ──
    def _build_sliding_position_ids(prev_count: int, frame_idx: int = 0):
        cap = max_cache_frames
        if pe_mode == "absolute":
            # Q is at absolute frame index; K positions are this frame's true
            # video indices.  Cache layout after trim is [cond=frame0]+[recent
            # (cap-1)]. For the cond seed (frame_idx==0, prev_count==0) → K=[0].
            # Total K span per forward: 1 (sink) + cap (cache after trim ⊕ new).
            q_t_val = frame_idx
            if frame_idx >= cap and prev_count >= cap:
                # cache trimmed → cache_K = [0] + [frame_idx-cap+1..frame_idx-1],
                # new K appended at frame_idx → full K = [0]+[frame_idx-cap+1..frame_idx]
                k_t_list = [0] + list(range(frame_idx - cap + 1, frame_idx + 1))
            else:
                # cache not yet trimmed → frames [0..frame_idx]
                k_t_list = list(range(frame_idx + 1))
            q_pos = torch.cartesian_prod(
                torch.tensor([q_t_val], dtype=torch.long, device=device),
                torch.arange(_frame_seq_h, dtype=torch.long, device=device),
                torch.arange(_frame_seq_w, dtype=torch.long, device=device),
            ).unsqueeze(0).repeat(batch_size, 1, 1)
            k_pos = torch.cartesian_prod(
                torch.tensor(k_t_list, dtype=torch.long, device=device),
                torch.arange(_frame_seq_h, dtype=torch.long, device=device),
                torch.arange(_frame_seq_w, dtype=torch.long, device=device),
            ).unsqueeze(0).repeat(batch_size, 1, 1)
            return q_pos, k_pos
        # ── slot mode (legacy default) ──
        q_slot = min(prev_count, cap)
        q_pos = torch.cartesian_prod(
            torch.tensor([q_slot], dtype=torch.long, device=device),
            torch.arange(_frame_seq_h, dtype=torch.long, device=device),
            torch.arange(_frame_seq_w, dtype=torch.long, device=device),
        ).unsqueeze(0).repeat(batch_size, 1, 1)
        k_pos = torch.cartesian_prod(
            torch.arange(q_slot + 1, dtype=torch.long, device=device),
            torch.arange(_frame_seq_h, dtype=torch.long, device=device),
            torch.arange(_frame_seq_w, dtype=torch.long, device=device),
        ).unsqueeze(0).repeat(batch_size, 1, 1)
        return q_pos, k_pos

    def _set_sliding_kwargs(akw, frame_idx: int = 0):
        cache = akw["past_key_values"]
        prev_count = cache._frame_count.get(0, 0)
        q_pos, k_pos = _build_sliding_position_ids(prev_count, frame_idx=frame_idx)
        if pe_mode == "absolute":
            # See _streaming_generate's _set_sliding_kwargs for the rationale.
            # In abs PE we use the legacy attention path (K's RoPE is baked at
            # write time, cache trims keep the baked K untouched).
            akw.pop("sliding_mode", None)
            akw.pop("sliding_k_position_ids", None)
        else:
            akw["sliding_mode"] = True
            akw["sliding_k_position_ids"] = k_pos
        return q_pos

    def _absolute_position_ids(frame_idx: int):
        return torch.cartesian_prod(
            torch.arange(1 // p_t, dtype=torch.long, device=device) + frame_idx,
            torch.arange(_frame_seq_h, dtype=torch.long, device=device),
            torch.arange(_frame_seq_w, dtype=torch.long, device=device),
        ).unsqueeze(0).repeat(batch_size, 1, 1)

    # ── Sample shared scalar timestep + per-element noise ──
    # Why shared scalar t (not per-frame): matches model.py L870 streaming
    # variant. Per-frame independent t would force a different sigma per
    # frame in the same forward, breaking the "one cache state ↔ one denoise
    # snapshot" mental model that streaming inference relies on.
    #
    # ZeRO-3 SAFETY: t MUST be identical across all ranks, otherwise different
    # ranks compute different sigmas → different noisy_latents shapes/values
    # would not directly NCCL-desync (we don't all_gather on t), but the
    # learning signal would be incoherent across ranks. Easy fix: rank-0 sample
    # + broadcast, same pattern as grad_step_idx in _streaming_generate L599-625.
    #
    # Optional σ-range cropping (env-controlled, default = [0, T) full range):
    #   V_FLOW_MIN_STEP_FRAC (default 0.0) -> low_t  = floor(frac * T)
    #   V_FLOW_MAX_STEP_FRAC (default 1.0) -> high_t = ceil(frac * T)
    # Set 0.02/0.98 to crop σ ≈ 0 (trivial) and σ ≈ 1 (noise-dominated where
    # v_target = noise - GT carries no learnable signal). LongLive and the
    # project's own DMD already use 0.02/0.98; v_flow MSE warmup was the
    # only path left uncropped. Defaults preserve baseline behavior bit-
    # identically; the wrapper opts in via `V_FLOW_MIN_STEP_FRAC=... bash ...`.
    # σ comes from `scheduler.timesteps[index]` lookup (Self-Forcing
    # `model/diffusion.py:78` style). The caller passes `v_flow_scheduler`
    # whose `timesteps` array was populated via one-time
    # `set_timesteps(num_train_timesteps)` under the FLOW_SHIFT-controlled
    # config, so `scheduler.timesteps[idx]` returns the shift-mapped
    # timestep. We sample `index` uniformly over `[V_FLOW_MIN_STEP_FRAC * T,
    # V_FLOW_MAX_STEP_FRAC * T)`. With FLOW_SHIFT=1.0 (default) this is
    # statistically equivalent to baseline `randint(0, T)` direct-divide
    # (just iterated in reverse t order — set_timesteps under flow_shift=1
    # produces timesteps = [T-1, T-2, ..., 0]).
    num_train_timesteps = scheduler.config.num_train_timesteps
    _vmin_frac = float(os.environ.get("V_FLOW_MIN_STEP_FRAC", "0.0"))
    _vmax_frac = float(os.environ.get("V_FLOW_MAX_STEP_FRAC", "1.0"))
    _vmin_idx = int(_vmin_frac * num_train_timesteps)
    _vmax_idx = max(_vmin_idx + 1, int(_vmax_frac * num_train_timesteps))
    try:
        import torch.distributed as _dist
        if _dist.is_available() and _dist.is_initialized():
            _idx_buf = torch.empty(1, dtype=torch.long, device=device)
            if _dist.get_rank() == 0:
                _idx_buf.fill_(int(torch.randint(
                    low=_vmin_idx, high=_vmax_idx, size=(1,),
                    generator=generator_rng,
                ).item()))
            _dist.broadcast(_idx_buf, src=0)
            index = _idx_buf
        else:
            index = torch.randint(
                low=_vmin_idx, high=_vmax_idx, size=(1,),
                device=device, generator=generator_rng,
            )
    except Exception:
        index = torch.randint(
            low=_vmin_idx, high=_vmax_idx, size=(1,),
            device=device, generator=generator_rng,
        )
    # SF-style: lookup shift-mapped timestep from scheduler.timesteps array.
    # scheduler.timesteps lives on CPU after set_timesteps(..., device="cpu");
    # gather then move the scalar onto our device.
    t_scalar = scheduler.timesteps.to(device)[index].long()
    sigma = (t_scalar.float() / num_train_timesteps).to(dtype)  # [1]

    # noise: per-rank fresh, NOT broadcast — each rank trains on a different
    # noisy realization of the (broadcast) GT clip. This is the same diversity
    # pattern as the DMD path (each rank sees independent noise).
    noise = randn_tensor(
        gt_latents.shape, generator=generator_rng, device=device, dtype=dtype,
    )
    # noisy_latents[f] = (1 - σ) * GT[f] + σ * noise[f]
    # sigma is a scalar broadcast over (B, C, F, H, W).
    noisy_latents = (1 - sigma) * gt_latents.to(dtype) + sigma * noise

    # ── Frame 0: seed cache with clean GT cond (always at slot 0, t=0) ──
    if sliding_mode:
        cond_position_ids = _set_sliding_kwargs(attention_kwargs, frame_idx=0)
    else:
        cond_position_ids = _absolute_position_ids(0)
    cond_t = torch.zeros(batch_size, _frame_seq_len, dtype=torch.long, device=device)
    # Per-frame control slicer (None when not in control distill mode).
    def _ctrl_f(f):
        if control_video_latent is None:
            return None
        return control_video_latent[:, :, f:f+1].to(dtype)
    with torch.no_grad():
        attention_kwargs["past_key_values"].unfreeze()
        generator(
            hidden_states=gt_latents[:, :, 0:1].to(dtype),
            control_video_latent=_ctrl_f(0),
            timestep=cond_t,
            encoder_hidden_states=prompt_embeds,
            encoder_hidden_states_image=None,
            attention_kwargs=attention_kwargs,
            position_ids=cond_position_ids,
            return_dict=False,
        )

    if trace_dump:
        _mem_probe("post-seed-cache(v_flow)", trace_tag, enabled=True, do_sync=False)

    # ── Frames 1..F-1: noisy forward (grad) → freeze cache;
    #                   clean GT forward (no_grad) → write clean KV ──
    # The "freeze → predict → unfreeze → write" sequence exactly mirrors
    # _streaming_generate L831-1067 and the sister model.py L915-963.
    v_preds = []
    # Build per-frame timestep tensor once and reuse the cell value (it's the
    # same scalar t broadcast across the frame's spatial tokens).
    t_chunk = t_scalar.expand(batch_size, _frame_seq_len).to(dtype=torch.long)

    # ── Cache-aware activation checkpointing (gates on STREAMING_GRAD_CHECKPOINT=1) ──
    # Without this, the grad-bearing forward at each frame holds ~13 GB of
    # activations on 5-B Wan2.2; 20 grad-bearing frames = ~260 GB / rank → OOM
    # on H20 (~95 GB). With it on, per-frame activation drops to ~0.3 GB
    # (forward saves only inputs; activations are recomputed during backward).
    # The recompute is bit-identical because we snapshot/restore the cache
    # state at this frame (same mechanism _streaming_generate uses at L915-988).
    # When OFF, the legacy direct-call path runs (useful when memory headroom
    # exists OR to ablate checkpointing cost).
    _use_grad_ckpt = (os.environ.get("STREAMING_GRAD_CHECKPOINT", "0") == "1")

    def _make_v_flow_ckpt_fwd(frame_position_ids):
        """Closure-capture this frame's state (cache snapshot, position_ids,
        sliding kwargs) so backward recompute sees the EXACT inputs forward
        saw, despite the outer for-loop having moved on. v_flow has no CFG
        and no denoise loop, so the closure is simpler than
        _streaming_generate's analogous helper at L915-988.
        """
        cache = attention_kwargs["past_key_values"]
        cache_snap = cache.snapshot()
        # Freeze attention_kwargs without the cache (cache is restored
        # separately via the snapshot mechanism).
        akw_static = {k: v for k, v in attention_kwargs.items()
                      if k != "past_key_values"}
        pos_ids_static = frame_position_ids
        prompt_embeds_static = prompt_embeds

        def _fn(lmi, ts, ctrl):
            # On forward: snapshot is a no-op restore (cache is already in
            # this state). On backward recompute: restore brings the cache
            # back to the snapshotted frozen state so the recomputed forward
            # is bit-identical.
            cache.restore(cache_snap)
            akw_local = dict(akw_static)
            akw_local["past_key_values"] = cache
            return generator(
                hidden_states=lmi,
                control_video_latent=ctrl,
                timestep=ts,
                encoder_hidden_states=prompt_embeds_static,
                encoder_hidden_states_image=None,
                attention_kwargs=akw_local,
                position_ids=pos_ids_static,
                return_dict=False,
            )[0]

        from torch.utils.checkpoint import checkpoint as _ckpt
        def _ckpt_call(lmi, ts, ctrl):
            return _ckpt(_fn, lmi, ts, ctrl, use_reentrant=False)
        return _ckpt_call

    for f in range(1, F):
        # Position ids for this frame's Q (and stamp K-side slot positions
        # into attention_kwargs if sliding).
        if sliding_mode:
            position_ids = _set_sliding_kwargs(attention_kwargs, frame_idx=f)
        else:
            position_ids = _absolute_position_ids(f)

        # ── (a) grad-bearing forward on the noisy frame, cache FROZEN ──
        attention_kwargs["past_key_values"].freeze()
        with torch.enable_grad():
            if _use_grad_ckpt:
                _ckpt_fwd = _make_v_flow_ckpt_fwd(position_ids)
                v_pred = _ckpt_fwd(
                    noisy_latents[:, :, f:f+1].to(dtype),
                    t_chunk,
                    _ctrl_f(f),
                )
            else:
                v_pred = generator(
                    hidden_states=noisy_latents[:, :, f:f+1].to(dtype),
                    control_video_latent=_ctrl_f(f),
                    timestep=t_chunk,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_image=None,
                    attention_kwargs=attention_kwargs,
                    position_ids=position_ids,
                    return_dict=False,
                )[0]
        v_preds.append(v_pred)

        # ── (b) clean forward on the GT frame, cache UNFROZEN, no grad ──
        # Writes clean GT K/V into the cache so the next frame attends to
        # the "right answer" history. This is teacher forcing — matches the
        # `latents[:, :, frame_slice]` + `clean_t = zeros` pattern in
        # model.py L943-944 and the per-frame clean-KV-write in
        # _streaming_generate L1043-1056.
        attention_kwargs["past_key_values"].unfreeze()
        # Re-stamp sliding kwargs for the clean write. prev_count grew by 1
        # after the freeze-noisy forward (no — freeze prevents writes), so
        # the Q slot for the clean write is the SAME as for the noisy fwd.
        # However we still re-stamp because _set_sliding_kwargs reads
        # _frame_count fresh; this is cheap (one tensor build) and explicit.
        if sliding_mode:
            clean_position_ids = _set_sliding_kwargs(attention_kwargs, frame_idx=f)
        else:
            clean_position_ids = _absolute_position_ids(f)
        with torch.no_grad():
            generator(
                hidden_states=gt_latents[:, :, f:f+1].to(dtype),
                control_video_latent=_ctrl_f(f),
                timestep=cond_t,
                encoder_hidden_states=prompt_embeds,
                encoder_hidden_states_image=None,
                attention_kwargs=attention_kwargs,
                position_ids=clean_position_ids,
                return_dict=False,
            )

        if trace_dump and (f == 1 or f == F - 1):
            _mem_probe(f"post-frame{f}(v_flow)", trace_tag,
                       enabled=True, do_sync=True)

    v_pred_all = torch.cat(v_preds, dim=2)  # [1, C, F-1, H, W]
    v_target = (noise - gt_latents.to(dtype))[:, :, 1:]  # [1, C, F-1, H, W]

    # x_0 reconstruction for DEBUG DECODE ONLY (loss is on v_pred / v_target).
    # See docstring "Returns" section for the math; .detach() so this branch
    # adds zero to the autograd graph.
    with torch.no_grad():
        noisy_for_recon = noisy_latents[:, :, 1:].to(v_pred_all.dtype)
        x0_for_decode = (noisy_for_recon - sigma.to(v_pred_all.dtype) * v_pred_all).detach()

    # Drop cache references; backward keeps the v_pred_all autograd graph alive.
    attention_kwargs.clear()

    if trace_dump:
        _mem_probe("exit-v_flow", trace_tag, enabled=True, do_sync=False)

    return v_pred_all, v_target, x0_for_decode


def streaming_v_flow_matching_forward_per_frame_t(
    generator,
    scheduler,
    gt_latents,                    # [1, C, F, H, W] — full GT video latent
    prompt_embeds,                 # [1, seq, dim]
    generator_rng: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
    patch_size,
    max_cache_frames: Optional[int] = None,
    trace_dump: bool = False,
    trace_tag: str = "v_flow_pft",
    pe_mode: str = "slot",
):
    """Per-frame-independent-t variant of streaming_v_flow_matching_forward.

    Only difference vs the shared-scalar-t baseline (V_FLOW_T_SAMPLING=shared,
    the default): every frame f∈[1, F-1] gets its OWN independently sampled
    t_f ∈ [0, num_train_timesteps), so each frame trains at a different noise
    level σ_f within the same forward pass.

    Why this exists
    ───────────────
    The shared-scalar-t baseline (mirrored from model/model.py's
    WanTI2VTrainingPipeline at L1004-1006) collapses to "static video"
    predictions over training because all 21 frames see the SAME σ in every
    optimizer step. At high σ the v-target = noise - GT is dominated by noise
    (per-frame independent but same magnitude), so MSE prefers a smooth /
    time-averaged v_pred that ignores motion. Verified empirically on
    run_20260516_102748_normandy: student frame-diff / GT frame-diff drops
    from 2.5 (step 50, over-active) to 0.68 (step 450, under-active).

    Per-frame t breaks this symmetry: in any given step some frames are near
    clean (small σ_f, target ≈ -GT[f], strong motion signal) while others
    are near noisy (large σ_f). Model is FORCED to learn the time dynamics
    because the gradient at low-σ frames directly carries GT motion info.
    Matches CausVid causal_video mode at num_frame_per_block=1
    (causvid/ode_regression.py L84-90).

    What's preserved (bit-identical vs baseline)
    ────────────────────────────────────────────
    * KV cache protocol: freeze-noisy-fwd → unfreeze-clean-GT-fwd per frame
    * Sliding cache + frame-0 sink + slot/absolute PE handling
    * Cache-aware activation checkpointing (STREAMING_GRAD_CHECKPOINT)
    * Frame 0 is clean cond/sink (σ_0 = 0 implicit, cond_t = zeros)
    * v_target formula: noise - GT (per-element, unchanged)
    * x_0 reconstruction for decode: noisy - σ · v_pred (now per-frame σ)
    * Rank-0-broadcast t for ZeRO-3 sync (now broadcasts a [F] vector
      instead of a scalar; same broadcast pattern just larger buf)

    What's different
    ────────────────
    * t is sampled as a [F] vector (frame 0 entry unused — kept for clean
      indexing). σ becomes a [F]-shaped tensor; noisy_latents and
      x0_for_decode broadcast σ frame-by-frame.
    * Per-frame t_chunk is constructed inside the for-loop using t_per_frame[f]
      (not a fixed t_scalar pre-built before the loop).

    Returns
    ───────
    v_pred:        [1, C, F-1, H, W] grad-bearing v predictions for frames 1..F-1
    v_target:      [1, C, F-1, H, W] = (noise - gt_latents)[:, :, 1:] (no grad)
    x0_for_decode: [1, C, F-1, H, W] = noisy - σ_f · v_pred per frame (detached)
    """
    batch_size, num_channels, F, latent_h, latent_w = gt_latents.shape
    p_t, p_h, p_w = patch_size
    _frame_seq_h = latent_h // p_h
    _frame_seq_w = latent_w // p_w
    _frame_seq_len = _frame_seq_h * _frame_seq_w
    sliding_mode = (max_cache_frames is not None)

    attention_kwargs = {"past_key_values": DynamicCache(max_frames=max_cache_frames)}
    if trace_dump and pe_mode == "absolute":
        attention_kwargs["trace_pe"] = True

    if trace_dump:
        try:
            import torch.distributed as _dist
            _is_rank0 = (not _dist.is_initialized()) or _dist.get_rank() == 0
        except Exception:
            _is_rank0 = True
        if _is_rank0:
            cap_str = (str(max_cache_frames) if max_cache_frames is not None else "unbounded")
            cprint(
                f"  [trace:{trace_tag}] geometry | latent=[B={batch_size}, C={num_channels}, "
                f"F={F}, H={latent_h}, W={latent_w}] | patch=({p_t},{p_h},{p_w}) "
                f"| frame_seq=({_frame_seq_h}x{_frame_seq_w})={_frame_seq_len} tokens "
                f"| sliding_mode={sliding_mode} | cap={cap_str} "
                f"| mode=single-step-flow-matching-per-frame-t",
                "blue",
            )

    def _build_sliding_position_ids(prev_count: int, frame_idx: int = 0):
        cap = max_cache_frames
        if pe_mode == "absolute":
            q_t_val = frame_idx
            if frame_idx >= cap and prev_count >= cap:
                k_t_list = [0] + list(range(frame_idx - cap + 1, frame_idx + 1))
            else:
                k_t_list = list(range(frame_idx + 1))
            q_pos = torch.cartesian_prod(
                torch.tensor([q_t_val], dtype=torch.long, device=device),
                torch.arange(_frame_seq_h, dtype=torch.long, device=device),
                torch.arange(_frame_seq_w, dtype=torch.long, device=device),
            ).unsqueeze(0).repeat(batch_size, 1, 1)
            k_pos = torch.cartesian_prod(
                torch.tensor(k_t_list, dtype=torch.long, device=device),
                torch.arange(_frame_seq_h, dtype=torch.long, device=device),
                torch.arange(_frame_seq_w, dtype=torch.long, device=device),
            ).unsqueeze(0).repeat(batch_size, 1, 1)
            return q_pos, k_pos
        q_slot = min(prev_count, cap)
        q_pos = torch.cartesian_prod(
            torch.tensor([q_slot], dtype=torch.long, device=device),
            torch.arange(_frame_seq_h, dtype=torch.long, device=device),
            torch.arange(_frame_seq_w, dtype=torch.long, device=device),
        ).unsqueeze(0).repeat(batch_size, 1, 1)
        k_pos = torch.cartesian_prod(
            torch.arange(q_slot + 1, dtype=torch.long, device=device),
            torch.arange(_frame_seq_h, dtype=torch.long, device=device),
            torch.arange(_frame_seq_w, dtype=torch.long, device=device),
        ).unsqueeze(0).repeat(batch_size, 1, 1)
        return q_pos, k_pos

    def _set_sliding_kwargs(akw, frame_idx: int = 0):
        cache = akw["past_key_values"]
        prev_count = cache._frame_count.get(0, 0)
        q_pos, k_pos = _build_sliding_position_ids(prev_count, frame_idx=frame_idx)
        if pe_mode == "absolute":
            akw.pop("sliding_mode", None)
            akw.pop("sliding_k_position_ids", None)
        else:
            akw["sliding_mode"] = True
            akw["sliding_k_position_ids"] = k_pos
        return q_pos

    def _absolute_position_ids(frame_idx: int):
        return torch.cartesian_prod(
            torch.arange(1 // p_t, dtype=torch.long, device=device) + frame_idx,
            torch.arange(_frame_seq_h, dtype=torch.long, device=device),
            torch.arange(_frame_seq_w, dtype=torch.long, device=device),
        ).unsqueeze(0).repeat(batch_size, 1, 1)

    # ── Sample per-frame t ── [F] long tensor, rank-0-broadcast for ZeRO-3 sync.
    # Frame 0's entry is unused (cond seed uses cond_t = zeros), but kept in the
    # tensor for clean per-frame indexing below.
    #
    # σ-range cropping + SF-style lookup (same env vars as the shared-t
    # variant above; defaults preserve baseline). Each frame's index is
    # drawn independently; lookup `scheduler.timesteps[index]` returns the
    # shift-mapped timestep under the caller-provided `v_flow_scheduler`
    # (set_timesteps(num_train_timesteps) was called once at startup).
    num_train_timesteps = scheduler.config.num_train_timesteps
    _vmin_frac = float(os.environ.get("V_FLOW_MIN_STEP_FRAC", "0.0"))
    _vmax_frac = float(os.environ.get("V_FLOW_MAX_STEP_FRAC", "1.0"))
    _vmin_idx = int(_vmin_frac * num_train_timesteps)
    _vmax_idx = max(_vmin_idx + 1, int(_vmax_frac * num_train_timesteps))
    try:
        import torch.distributed as _dist
        if _dist.is_available() and _dist.is_initialized():
            _idx_buf = torch.empty(F, dtype=torch.long, device=device)
            if _dist.get_rank() == 0:
                _idx_buf.copy_(torch.randint(
                    low=_vmin_idx, high=_vmax_idx, size=(F,),
                    generator=generator_rng,
                ).to(device))
            _dist.broadcast(_idx_buf, src=0)
            index_per_frame = _idx_buf
        else:
            index_per_frame = torch.randint(
                low=_vmin_idx, high=_vmax_idx, size=(F,),
                device=device, generator=generator_rng,
            )
    except Exception:
        index_per_frame = torch.randint(
            low=_vmin_idx, high=_vmax_idx, size=(F,),
            device=device, generator=generator_rng,
        )
    # SF-style lookup over scheduler.timesteps (CPU array → device).
    t_per_frame = scheduler.timesteps.to(device)[index_per_frame].long()
    # σ_f for f=0..F-1. Shape [F]; broadcast to [1, 1, F, 1, 1] for noisy_latents.
    sigmas = (t_per_frame.float() / num_train_timesteps).to(dtype)
    sigmas_5d = sigmas.view(1, 1, F, 1, 1)

    noise = randn_tensor(
        gt_latents.shape, generator=generator_rng, device=device, dtype=dtype,
    )
    # noisy_latents[f] = (1 - σ_f) * GT[f] + σ_f * noise[f]
    noisy_latents = (1 - sigmas_5d) * gt_latents.to(dtype) + sigmas_5d * noise

    # ── Frame 0: seed cache with clean GT cond (always at slot 0, t=0) ──
    # (frame-0 entry of sigmas/t_per_frame is unused by design; cond seed uses
    # cond_t = zeros to signal "clean" to the AdaLN modulation.)
    if sliding_mode:
        cond_position_ids = _set_sliding_kwargs(attention_kwargs, frame_idx=0)
    else:
        cond_position_ids = _absolute_position_ids(0)
    cond_t = torch.zeros(batch_size, _frame_seq_len, dtype=torch.long, device=device)
    # Slice helper for per-frame control (None when not in control distill mode)
    def _ctrl_f(f):
        if control_video_latent is None:
            return None
        return control_video_latent[:, :, f:f+1].to(dtype)
    with torch.no_grad():
        attention_kwargs["past_key_values"].unfreeze()
        generator(
            hidden_states=gt_latents[:, :, 0:1].to(dtype),
            control_video_latent=_ctrl_f(0),
            timestep=cond_t,
            encoder_hidden_states=prompt_embeds,
            encoder_hidden_states_image=None,
            attention_kwargs=attention_kwargs,
            position_ids=cond_position_ids,
            return_dict=False,
        )

    if trace_dump:
        _mem_probe("post-seed-cache(v_flow_pft)", trace_tag, enabled=True, do_sync=False)

    v_preds = []

    _use_grad_ckpt = (os.environ.get("STREAMING_GRAD_CHECKPOINT", "0") == "1")

    def _make_v_flow_ckpt_fwd(frame_position_ids):
        cache = attention_kwargs["past_key_values"]
        cache_snap = cache.snapshot()
        akw_static = {k: v for k, v in attention_kwargs.items()
                      if k != "past_key_values"}
        pos_ids_static = frame_position_ids
        prompt_embeds_static = prompt_embeds

        def _fn(lmi, ts, ctrl):
            cache.restore(cache_snap)
            akw_local = dict(akw_static)
            akw_local["past_key_values"] = cache
            return generator(
                hidden_states=lmi,
                control_video_latent=ctrl,
                timestep=ts,
                encoder_hidden_states=prompt_embeds_static,
                encoder_hidden_states_image=None,
                attention_kwargs=akw_local,
                position_ids=pos_ids_static,
                return_dict=False,
            )[0]

        from torch.utils.checkpoint import checkpoint as _ckpt
        def _ckpt_call(lmi, ts, ctrl):
            # When ctrl is None, checkpoint handles None as a non-tensor arg.
            return _ckpt(_fn, lmi, ts, ctrl, use_reentrant=False)
        return _ckpt_call

    for f in range(1, F):
        if sliding_mode:
            position_ids = _set_sliding_kwargs(attention_kwargs, frame_idx=f)
        else:
            position_ids = _absolute_position_ids(f)

        # Per-frame timestep tensor: frame f's scalar t_per_frame[f] broadcast
        # across this frame's spatial tokens.
        t_chunk = t_per_frame[f].view(1).expand(batch_size, _frame_seq_len).to(dtype=torch.long)

        attention_kwargs["past_key_values"].freeze()
        with torch.enable_grad():
            if _use_grad_ckpt:
                _ckpt_fwd = _make_v_flow_ckpt_fwd(position_ids)
                v_pred = _ckpt_fwd(
                    noisy_latents[:, :, f:f+1].to(dtype),
                    t_chunk,
                    _ctrl_f(f),
                )
            else:
                v_pred = generator(
                    hidden_states=noisy_latents[:, :, f:f+1].to(dtype),
                    control_video_latent=_ctrl_f(f),
                    timestep=t_chunk,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_image=None,
                    attention_kwargs=attention_kwargs,
                    position_ids=position_ids,
                    return_dict=False,
                )[0]
        v_preds.append(v_pred)

        attention_kwargs["past_key_values"].unfreeze()
        if sliding_mode:
            clean_position_ids = _set_sliding_kwargs(attention_kwargs, frame_idx=f)
        else:
            clean_position_ids = _absolute_position_ids(f)
        with torch.no_grad():
            generator(
                hidden_states=gt_latents[:, :, f:f+1].to(dtype),
                control_video_latent=_ctrl_f(f),
                timestep=cond_t,
                encoder_hidden_states=prompt_embeds,
                encoder_hidden_states_image=None,
                attention_kwargs=attention_kwargs,
                position_ids=clean_position_ids,
                return_dict=False,
            )

        if trace_dump and (f == 1 or f == F - 1):
            _mem_probe(f"post-frame{f}(v_flow_pft)", trace_tag,
                       enabled=True, do_sync=True)

    v_pred_all = torch.cat(v_preds, dim=2)  # [1, C, F-1, H, W]
    v_target = (noise - gt_latents.to(dtype))[:, :, 1:]  # [1, C, F-1, H, W]

    # Per-frame x_0 reconstruction for decode: σ now varies along the F axis.
    with torch.no_grad():
        noisy_for_recon = noisy_latents[:, :, 1:].to(v_pred_all.dtype)
        sigmas_recon = sigmas_5d[:, :, 1:].to(v_pred_all.dtype)
        x0_for_decode = (noisy_for_recon - sigmas_recon * v_pred_all).detach()

    attention_kwargs.clear()

    if trace_dump:
        _mem_probe("exit-v_flow_pft", trace_tag, enabled=True, do_sync=False)

    return v_pred_all, v_target, x0_for_decode


def streaming_causvid_traj_forward(
    generator,
    scheduler,
    gt_latents,          # [1, C, F, H, W] real GT (used for clean teacher-forcing prefix)
    trajectory,          # [N+1, C, F, H, W] teacher ODE snapshots (last entry σ=0 = clean)
    traj_sigmas,         # [N+1] σ values; last entry = 0.0
    prompt_embeds,
    img_latent,
    num_latent_frames,
    max_cache_frames,
    pe_mode,
    patch_size,
    generator_rng,
    device,
    dtype,
    trace_dump=False,
    trace_tag="causvid_traj",
):
    """CausVid §4.3 ODE-regression forward, adapted to streaming + cache.

    Maps `causvid/ode_regression.py:108-160` (random σ-snapshot gather + 1
    forward + x_0 MSE) to our streaming + KV cache architecture, with a
    twist: target = REAL GT video (passed via `gt_latents`), not
    teacher self-distill `trajectory[-1]`. Frame 0 of both equals cond image
    by design; frames 1..F-1 may differ if teacher generated different
    motion than the source video. We supervise on REAL GT to avoid hallucinated-
    motion drift.

    Per-frame:
      - sample index_f ∈ [0, N), broadcast across ranks (ZeRO-3 sync)
      - σ_f = traj_sigmas[index_f]
      - noisy_f = trajectory[index_f, :, f, :, :]  (snapshot at σ_f for frame f)
      - generator forward at frame f (cache frozen, GT-teacher-forced prefix)
      - reconstruct x_0_pred = noisy_f - σ_f * v_pred_f
      - target = gt_latents[:, :, f, :, :]  (REAL GT — not trajectory[-1])
      - mask = (σ_f > 0)  — t=0 entries are already clean, no learning signal

    Returns:
      x0_pred:    [1, C, F-1, H, W] grad-bearing x_0 reconstruction
      x0_target:  [1, C, F-1, H, W] = trajectory[-1, :, 1:]  (no grad)
      mask:       [1, 1, F-1, 1, 1] float — caller multiplies before MSE

    Caller loss:
      loss = ((x0_pred - x0_target) * mask).pow(2).sum() / mask.sum() / C / H / W
      (mean over masked elements, broadcasting σ-mask across spatial dims)

    Why this shape vs v_flow MSE:
      - v_flow trains on linear-interp noisy = (1-σ)·GT + σ·noise
      - causvid_traj trains on ODE-solver-path noisy = trajectory[idx]
      - Same downstream architecture (streaming + cache + GT prefix); only
        the (σ, noisy) input differs, which makes for a clean ablation.
    """
    batch_size, num_channels, F, latent_h, latent_w = gt_latents.shape
    p_t, p_h, p_w = patch_size
    _frame_seq_h = latent_h // p_h
    _frame_seq_w = latent_w // p_w
    _frame_seq_len = _frame_seq_h * _frame_seq_w
    sliding_mode = (max_cache_frames is not None)

    # Trajectory shape: [N+1, C, F, H, W]. Last entry σ=0 = clean target.
    # We sample only from the first N entries (the "noisy" snapshots).
    N_target = traj_sigmas.shape[0] - 1  # = NIS (number of training σ levels)
    assert trajectory.shape[0] == traj_sigmas.shape[0], (
        f"trajectory.shape[0]={trajectory.shape[0]} vs traj_sigmas.shape[0]="
        f"{traj_sigmas.shape[0]} mismatch")
    # Cast trajectory to compute dtype (it lives as fp32 in the cache for
    # checkpoint robustness).
    trajectory = trajectory.to(device=device, dtype=dtype)
    traj_sigmas = traj_sigmas.to(device=device)
    # Clean target = REAL GT video (= `gt_latents` parameter). Caller passes
    # pair["gt_video_latent"]. Differs from CausVid §4.3 `target = trajectory[-1]`
    # — we supervise on REAL EPIC video motion, not teacher's
    # generated motion. Frame 0 of both equals cond image; frames 1..F-1
    # differ when teacher's prompt-conditioned generation diverged from
    # source video.

    attention_kwargs = {"past_key_values": DynamicCache(max_frames=max_cache_frames)}
    if trace_dump and pe_mode == "absolute":
        attention_kwargs["trace_pe"] = True

    # Local copy of v_flow's _build_sliding_position_ids (nested helpers from
    # streaming_v_flow_matching_forward are not in scope here).
    def _build_sliding_position_ids(prev_count: int, frame_idx: int = 0):
        cap = max_cache_frames
        if pe_mode == "absolute":
            q_t_val = frame_idx
            if frame_idx >= cap and prev_count >= cap:
                k_t_list = [0] + list(range(frame_idx - cap + 1, frame_idx + 1))
            else:
                k_t_list = list(range(frame_idx + 1))
            q_pos = torch.cartesian_prod(
                torch.tensor([q_t_val], dtype=torch.long, device=device),
                torch.arange(_frame_seq_h, dtype=torch.long, device=device),
                torch.arange(_frame_seq_w, dtype=torch.long, device=device),
            ).unsqueeze(0).repeat(batch_size, 1, 1)
            k_pos = torch.cartesian_prod(
                torch.tensor(k_t_list, dtype=torch.long, device=device),
                torch.arange(_frame_seq_h, dtype=torch.long, device=device),
                torch.arange(_frame_seq_w, dtype=torch.long, device=device),
            ).unsqueeze(0).repeat(batch_size, 1, 1)
            return q_pos, k_pos
        q_slot = min(prev_count, cap)
        q_pos = torch.cartesian_prod(
            torch.tensor([q_slot], dtype=torch.long, device=device),
            torch.arange(_frame_seq_h, dtype=torch.long, device=device),
            torch.arange(_frame_seq_w, dtype=torch.long, device=device),
        ).unsqueeze(0).repeat(batch_size, 1, 1)
        k_pos = torch.cartesian_prod(
            torch.arange(q_slot + 1, dtype=torch.long, device=device),
            torch.arange(_frame_seq_h, dtype=torch.long, device=device),
            torch.arange(_frame_seq_w, dtype=torch.long, device=device),
        ).unsqueeze(0).repeat(batch_size, 1, 1)
        return q_pos, k_pos

    def _set_sliding_kwargs(akw, frame_idx: int = 0):
        cache = akw["past_key_values"]
        prev_count = cache._frame_count.get(0, 0)
        q_pos, k_pos = _build_sliding_position_ids(prev_count, frame_idx=frame_idx)
        if pe_mode == "absolute":
            akw.pop("sliding_mode", None)
            akw.pop("sliding_k_position_ids", None)
        else:
            akw["sliding_mode"] = True
            akw["sliding_k_position_ids"] = k_pos
        return q_pos

    def _absolute_position_ids(frame_idx: int):
        return torch.cartesian_prod(
            torch.arange(1 // p_t, dtype=torch.long, device=device) + frame_idx,
            torch.arange(_frame_seq_h, dtype=torch.long, device=device),
            torch.arange(_frame_seq_w, dtype=torch.long, device=device),
        ).unsqueeze(0).repeat(batch_size, 1, 1)

    # ── Sample per-frame index_f ∈ [0, N), broadcast across ranks ──
    # Per-(batch, frame) random index. CausVid `_prepare_generator_input`
    # samples `[B, F]` with replacement; we use `[F]` (B=1 in our setup) and
    # broadcast from rank 0 for ZeRO-3 sync. Frame 0 is unused (cond seed at
    # t=0); we keep [F] indexing for clean tensor ops.
    num_train_timesteps = scheduler.config.num_train_timesteps
    try:
        import torch.distributed as _dist
        if _dist.is_available() and _dist.is_initialized():
            idx_buf = torch.empty(F, dtype=torch.long, device=device)
            if _dist.get_rank() == 0:
                idx_buf.copy_(torch.randint(
                    low=0, high=N_target, size=(F,),
                    generator=generator_rng,
                ).to(device))
            _dist.broadcast(idx_buf, src=0)
            index_per_frame = idx_buf
        else:
            index_per_frame = torch.randint(
                low=0, high=N_target, size=(F,), device=device,
                generator=generator_rng,
            )
    except Exception:
        index_per_frame = torch.randint(
            low=0, high=N_target, size=(F,), device=device,
            generator=generator_rng,
        )

    # σ per frame: [F] gathered from traj_sigmas
    sigmas_per_frame = traj_sigmas[index_per_frame]  # [F] in [0, 1]
    # timestep per frame: σ * num_train_timesteps as long
    t_per_frame = (sigmas_per_frame.float() * num_train_timesteps).long().clamp(
        min=0, max=num_train_timesteps - 1
    )  # [F]

    # noisy per frame: gather trajectory[index_per_frame[f], :, f, :, :]
    # → shape [C, F, H, W] then unsqueeze for batch
    # trajectory: [N+1, C, F, H, W] — gather along dim 0 for index_per_frame[f] at frame f
    # This is per-frame gather — easiest is a python loop (F=21 small).
    noisy_per_frame_list = []
    for f in range(F):
        noisy_per_frame_list.append(trajectory[index_per_frame[f], :, f, :, :])
    noisy_latents = torch.stack(noisy_per_frame_list, dim=1).unsqueeze(0).contiguous()
    # noisy_latents: [1, C, F, H, W]

    # Clean target = REAL GT video (= caller-passed `gt_latents`).
    # We supervise on real motion instead of the teacher's
    # self-distill final. See docstring above for the rationale.
    x0_target_full = gt_latents.to(device=device, dtype=dtype)  # [1, C, F, H, W]

    # ── Frame 0: seed cache with clean GT (= trajectory[-1, :, 0:1]) ──
    if sliding_mode:
        cond_position_ids = _set_sliding_kwargs(attention_kwargs, frame_idx=0)
    else:
        cond_position_ids = _absolute_position_ids(0)
    cond_t = torch.zeros(batch_size, _frame_seq_len, dtype=torch.long, device=device)
    with torch.no_grad():
        attention_kwargs["past_key_values"].unfreeze()
        generator(
            hidden_states=gt_latents[:, :, 0:1].to(dtype),
            timestep=cond_t,
            encoder_hidden_states=prompt_embeds,
            encoder_hidden_states_image=None,
            attention_kwargs=attention_kwargs,
            position_ids=cond_position_ids,
            return_dict=False,
        )

    if trace_dump:
        _mem_probe("post-seed-cache(causvid_traj)", trace_tag, enabled=True, do_sync=False)

    # ── Frames 1..F-1: noisy forward (grad) → freeze cache;
    #                   clean GT forward (no_grad) → write clean KV ──
    v_preds = []
    _use_grad_ckpt = (os.environ.get("STREAMING_GRAD_CHECKPOINT", "0") == "1")

    def _make_traj_ckpt_fwd(frame_position_ids):
        cache = attention_kwargs["past_key_values"]
        cache_snap = cache.snapshot()
        akw_static = {k: v for k, v in attention_kwargs.items()
                      if k != "past_key_values"}
        pos_ids_static = frame_position_ids
        prompt_embeds_static = prompt_embeds

        def _fn(latents_in, ts_in):
            cache.restore(cache_snap)
            cache.freeze()
            return generator(
                hidden_states=latents_in,
                timestep=ts_in,
                encoder_hidden_states=prompt_embeds_static,
                encoder_hidden_states_image=None,
                attention_kwargs={**akw_static, "past_key_values": cache},
                position_ids=pos_ids_static,
                return_dict=False,
            )[0]

        from torch.utils.checkpoint import checkpoint as _ckpt
        def _ckpt_call(lmi, ts):
            return _ckpt(_fn, lmi, ts, use_reentrant=False)
        return _ckpt_call

    for f in range(1, F):
        if sliding_mode:
            position_ids = _set_sliding_kwargs(attention_kwargs, frame_idx=f)
        else:
            position_ids = _absolute_position_ids(f)

        # Per-frame timestep tensor (sigma_f * num_train broadcast across spatial tokens)
        t_chunk = t_per_frame[f].expand(batch_size, _frame_seq_len).to(dtype=torch.long)

        # ── (a) grad-bearing forward at noisy=trajectory[idx_f, :, f, :, :], cache FROZEN ──
        attention_kwargs["past_key_values"].freeze()
        with torch.enable_grad():
            if _use_grad_ckpt:
                _ckpt_fwd = _make_traj_ckpt_fwd(position_ids)
                v_pred = _ckpt_fwd(
                    noisy_latents[:, :, f:f+1].to(dtype),
                    t_chunk,
                )
            else:
                v_pred = generator(
                    hidden_states=noisy_latents[:, :, f:f+1].to(dtype),
                    timestep=t_chunk,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_image=None,
                    attention_kwargs=attention_kwargs,
                    position_ids=position_ids,
                    return_dict=False,
                )[0]
        v_preds.append(v_pred)

        # ── (b) clean forward on real GT[f], cache UNFROZEN, no grad ──
        # (same as v_flow MSE warmup — write clean GT prefix into cache for next frame.
        #  trajectory[-1, :, f] should equal gt_latents[:, :, f] since frame 0 is pinned
        #  to condition; we use gt_latents to keep the prefix path identical to
        #  the streaming v_flow MSE warmup.)
        attention_kwargs["past_key_values"].unfreeze()
        if sliding_mode:
            clean_position_ids = _set_sliding_kwargs(attention_kwargs, frame_idx=f)
        else:
            clean_position_ids = _absolute_position_ids(f)
        with torch.no_grad():
            generator(
                hidden_states=gt_latents[:, :, f:f+1].to(dtype),
                timestep=cond_t,
                encoder_hidden_states=prompt_embeds,
                encoder_hidden_states_image=None,
                attention_kwargs=attention_kwargs,
                position_ids=clean_position_ids,
                return_dict=False,
            )

        if trace_dump and (f == 1 or f == F - 1):
            _mem_probe(f"post-frame{f}(causvid_traj)", trace_tag,
                       enabled=True, do_sync=True)

    v_pred_all = torch.cat(v_preds, dim=2)  # [1, C, F-1, H, W]

    # x_0 reconstruction: x_0_pred = noisy - σ * v_pred (per-frame σ broadcast)
    sigmas_for_recon = sigmas_per_frame[1:].view(1, 1, F - 1, 1, 1).to(v_pred_all.dtype)
    noisy_for_recon = noisy_latents[:, :, 1:].to(v_pred_all.dtype)
    x0_pred = noisy_for_recon - sigmas_for_recon * v_pred_all

    # x_0 target: trajectory[-1, :, 1:]
    x0_target = x0_target_full[:, :, 1:].to(v_pred_all.dtype)

    # mask: σ > 0  — t=0 frames have no learning signal (already clean)
    mask = (sigmas_per_frame[1:] > 1e-6).view(1, 1, F - 1, 1, 1).to(v_pred_all.dtype)

    # Drop cache references; backward keeps the v_pred_all autograd graph alive.
    attention_kwargs.clear()

    if trace_dump:
        _mem_probe("exit-causvid_traj", trace_tag, enabled=True, do_sync=False)

    return x0_pred, x0_target, mask


# ──────────────────────────────────────────────────────────────
# DMD Loss Functions
# ──────────────────────────────────────────────────────────────

# ── DMD debug snapshot buffer (module-level, rank-0-only readable) ────
# A small RING buffer (default size 2) of "the exact 4 tensors
# `compute_dmd_generator_loss` differentiated, plus σ + cos + flags"
# from the most recent N DMD steps. The debug-grid consumer
# (`_decode_and_save_dmd_grid`) dumps each occupied slot to a separate
# JPEG so a single decode trigger captures multiple DMD-step snapshots
# instead of N copies of the same one.
#
# Strategy:
#   - Ring size set at startup via env DMD_DEBUG_RING (default 2).
#   - `writes` is the monotonic write counter; (writes % ring_size) is
#     the slot just written. Consumer iterates slots in chronological
#     order (oldest first).
#   - Tensors detached + .cpu() before storing so subsequent backward()
#     can free the GPU graph immediately.
#   - Each slot ~5 MB × 4 tensors = ~20 MB CPU; 2-deep ring = ~40 MB,
#     trivial vs 600 GB host RAM. Set DMD_DEBUG_RING=0 to disable
#     entirely (frees the ring & skips the snapshot copy each step).
_DMD_DEBUG_RING_SIZE = int(os.environ.get("DMD_DEBUG_RING", "2") or 2)
_DMD_DEBUG_RING: list = [None] * max(0, _DMD_DEBUG_RING_SIZE)
_DMD_DEBUG_WRITES = [0]   # 1-element list so the inner closure can mutate

# ── σ-bucketed cos(g, p_real) rolling histogram ──────────────────────
# Catches the "DMD only works in some σ band" silent-failure mode that a
# single global cos value averages away. Bucket by σ ∈ [0, 1] into 5
# equal-width bins, keep last N records per bin via deque, emit a
# one-line summary every 50 DMD steps.
import collections as _collections
_DMD_SIGMA_COS_BINS = [
    (0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.001),
]
_DMD_SIGMA_COS_DEQUES = [
    _collections.deque(maxlen=200) for _ in _DMD_SIGMA_COS_BINS
]

def _record_dmd_sigma_cos(sigma_mean: float, cos: float):
    """Append (cos) to the bucket containing sigma_mean."""
    for i, (lo, hi) in enumerate(_DMD_SIGMA_COS_BINS):
        if lo <= sigma_mean < hi:
            _DMD_SIGMA_COS_DEQUES[i].append(cos)
            return

def _dmd_sigma_cos_summary() -> str:
    """One-line summary across the 5 σ buckets: mean cos + sample count.
    Empty buckets show 'nan(0)'. ⚠ marker when bucket mean < 0 (DMD
    teaching wrong direction in that σ band).
    """
    parts = []
    for (lo, hi), dq in zip(_DMD_SIGMA_COS_BINS, _DMD_SIGMA_COS_DEQUES):
        if len(dq) == 0:
            parts.append(f"σ[{lo:.1f}-{hi:.1f}]: nan(0)")
        else:
            m = sum(dq) / len(dq)
            mark = " ⚠" if m < 0 else ""
            parts.append(f"σ[{lo:.1f}-{hi:.1f}]: {m:+.2f}({len(dq)}){mark}")
    return f"  [dmd-sigma] cos by σ bucket | " + " | ".join(parts)


# ── JSONL writer for per-step DMD metrics ────────────────────────────
# Goal: machine-readable per-step record, one line per DMD step. Far
# easier to post-process than scraping multi-line text logs.
# - File: <output_dir>/dmd_metrics.jsonl
# - One line per step, append-only, flushed each write so a crash
#   doesn't lose the last batch.
# - Only the fields needed for diagnosis (drops verbose noisy stats).
# - Skipped on warmup steps (no DMD compute that step → no fields).
_DMD_JSONL_FH = {"path": None, "fh": None}

def _dmd_jsonl_write(output_dir: str, step: int, log_dict: dict):
    """Append one JSON record per DMD step.
    Whitelisted fields only; missing keys default to None (json null).
    """
    import json
    path = os.path.join(output_dir, "dmd_metrics.jsonl")
    if _DMD_JSONL_FH["path"] != path:
        try:
            if _DMD_JSONL_FH["fh"] is not None:
                _DMD_JSONL_FH["fh"].close()
        except Exception:
            pass
        os.makedirs(output_dir, exist_ok=True)
        _DMD_JSONL_FH["fh"] = open(path, "a", buffering=1)  # line-buffered
        _DMD_JSONL_FH["path"] = path
    fh = _DMD_JSONL_FH["fh"]
    # Whitelist — keep this list narrow; if you need more, add explicitly.
    keys = [
        # General loop signals
        "gen_loss", "crit_loss", "gen_grad_norm", "crit_grad_norm",
        "dmd_grad_norm",
        # Latent stats (already in log_dict from main loop)
        "gen_latent_mean", "gen_latent_std", "gen_latent_absmax",
        # σ + direction
        "dmd_sigma_min", "dmd_sigma_mean", "dmd_sigma_max",
        "dmd_grad_p_real_cos",
        # Branch magnitudes
        "dmd_fake_x0_absmean", "dmd_real_x0_absmean", "dmd_gen_x0_absmean",
        # Normalizer health
        "dmd_normalizer_min", "dmd_normalizer_mean", "dmd_normalizer_max",
        "dmd_normalizer_safemask_zero_frac",
        # Raw vs norm
        "dmd_raw_grad_absmean", "dmd_norm_grad_absmean",
        # Additional diagnostics
        "dmd_f0_dev_fake", "dmd_f0_dev_real",
        "dmd_grad_f1", "dmd_grad_fN", "dmd_grad_f1_fN_ratio",
        "dmd_real_v_absmean", "dmd_fake_v_absmean",
        "dmd_v_diff_absmean", "dmd_v_diff_to_real_ratio",
        "dmd_real_cfg_strength",
    ]
    rec = {"step": int(step)}
    for k in keys:
        v = log_dict.get(k, None)
        if v is None:
            rec[k] = None
        else:
            try:
                rec[k] = float(v)
            except Exception:
                rec[k] = None
    fh.write(json.dumps(rec, separators=(",", ":")) + "\n")


def _dmd_debug_slot_template():
    """Return an empty slot dict (one ring entry)."""
    return {
        "generated_for_loss": None,
        "fake_x0":            None,
        "real_x0":            None,
        "noisy_for_score":    None,
        "sigmas":             None,
        "cos":                None,
        "safemask_zero":      None,
        "sigma_mean":         None,
        "sigma_min":          None,
        "sigma_max":          None,
        "gen_x0_absmean":     None,
        "fake_x0_absmean":    None,
        "real_x0_absmean":    None,
        "raw_grad_absmean":   None,
        "normer_min":         None,
        # ★ Per-pixel normalized DMD gradient = (fake_x0 - real_x0)/normer * mask,
        # frame-0-zero-padded. Visualized as the Row-4 heatmap; this is the
        # actual signal `loss.backward()` propagates back to the student.
        "dmd_grad":           None,
        "step_id":            -1,
    }


def compute_dmd_generator_loss(
    generated_latents,       # [B, C, F, H, W] — output of generator
    critic,                  # fake_score
    real_score,              # frozen teacher
    conditional_dict,
    unconditional_dict,
    num_train_timesteps,
    real_guidance_scale,
    fake_guidance_scale,
    min_step_frac=0.02,
    max_step_frac=0.98,
    patch_size=(1, 2, 2),    # (p_t, p_h, p_w)
    # ── Control distill (used in the DMD phase) ──
    student_control_latent=None,    # [B, C, F, H, W] — NORMALIZED skel for critic
                                    # (critic = fake_score is trained with student's
                                    # add-plus mode → uses normalized skel).
    teacher_control_latent=None,    # [B, C, F, H, W] — RAW skel for real_score when
                                    # teacher_control_type='add', or NORMALIZED when
                                    # 'add-plus'. Caller decides which based on
                                    # --teacher_control_type. None = no control.
):
    """
    DMD generator loss (eq. 7 from DMD paper). Aligned with Self-Forcing
    reference (model/dmd.py).

    KEY INVARIANT — grad is computed in x_0 space, NOT v space.
    The Wan2.2 transformer outputs flow velocity v = noise - x_0. We must
    convert that to an x_0 prediction via x_0 = noisy - σ * v before
    computing the DMD gradient. Computing grad on raw v_pred would flip
    the sign (`v_fake - v_real == -(x0_fake - x0_real)/σ`), pushing the
    student AWAY from the real distribution. See SF dmd.py:75-113 + the
    `(flow_pred, pred_x0)` return tuple from WanDiffusionWrapper.forward.

    grad   = fake_x0 - real_x0                      (sign correct)
    p_real = generated - real_x0                    (physically coherent: x0 - x0)
    normalizer per-sample (dim=[1,2,3,4]) — matches SF, scales each video
    independently rather than mixing magnitudes across batch.
    loss   = 0.5 * MSE(generated, (generated - grad).detach())

    Control distill:
        critic gets student_control_latent (normalized) — must match how generator
            was trained (add-plus with running_stats normalization).
        real_score gets teacher_control_latent — caller-controlled raw vs normalized
            depending on which control_type the teacher LoRA was trained with.
        Typical DMD configuration for this codebase:
          student_control_latent = NORMALIZED skel  (student trained add-plus)
          teacher_control_latent = RAW skel         (LoRA teacher trained add)
    """
    batch_size, num_channels, num_frames, latent_h, latent_w = generated_latents.shape
    p_t, p_h, p_w = patch_size
    seq_len = (num_frames // p_t) * (latent_h // p_h) * (latent_w // p_w)

    with torch.no_grad():
        # Sample timesteps: [B, F] — one per latent frame
        min_step = int(min_step_frac * num_train_timesteps)
        max_step = int(max_step_frac * num_train_timesteps)

        # ── DMD timestep sampling: per-frame OR uniform-over-video (LongLive style) ──
        # Default: per-frame (CausVid causal_video block_size=1 style; each frame
        # gets its own random t). This is the project's fork-local default.
        # Set DMD_UNIFORM_TIMESTEP=1 to draw ONE scalar t per video and broadcast
        # to all F frames (LongLive `uniform_timestep=True` / Self-Forcing same).
        # Statistically equivalent expected gradient, but lower per-step variance
        # and matches reference implementations.
        _dmd_uniform_t = bool(int(os.environ.get("DMD_UNIFORM_TIMESTEP", "0")))
        if _dmd_uniform_t:
            # [B, 1] then repeat to [B, F]
            _t_shared = torch.randint(min_step, max_step, (batch_size, 1), device=generated_latents.device)
            timesteps_per_frame = _t_shared.repeat(1, num_frames)
        else:
            timesteps_per_frame = torch.randint(min_step, max_step, (batch_size, num_frames), device=generated_latents.device)

        # ── Optional DMD_FLOW_SHIFT mapping (CausVid dmd.py:237-241 / LongLive style) ──
        # Without shift the σ-t relation is linear; uniform [min_step, max_step)
        # randint yields uniform σ ∈ [0.02, 0.98]. Applying shift>1 pushes σ
        # toward the high band (e.g. shift=5 → ~75% of σ in [0.5, 1.0]).
        # SF / LongLive / CausVid all use shift=5 or 8 in DMD.
        #
        # NOTE: this is INDEPENDENT of `FLOW_SHIFT` (which controls v_flow MSE
        # warmup σ distribution via v_flow_scheduler). Splitting them keeps
        # ablation isolation: e.g. set `DMD_FLOW_SHIFT=5.0` alone to leave
        # MSE warmup linear (matching baseline) while pushing only DMD σ.
        _dmd_shift = float(os.environ.get("DMD_FLOW_SHIFT", "1.0"))
        if _dmd_shift != 1.0:
            _t_norm = timesteps_per_frame.float() / num_train_timesteps
            _t_norm = _dmd_shift * _t_norm / (1 + (_dmd_shift - 1) * _t_norm)
            timesteps_per_frame = (_t_norm * num_train_timesteps).long().clamp(
                min=min_step, max=max_step - 1
            )

        # Add noise (sigmas need per-frame shape for broadcasting)
        noise = torch.randn_like(generated_latents)
        sigmas = timesteps_per_frame.float() / num_train_timesteps
        # Latents are [B, C, F, H, W]; per-frame sigma needs to broadcast across C/H/W,
        # so its shape must be [B, 1, F, 1, 1] (not [B, F, 1, 1, 1] which would line up
        # against the channel dim).
        sigmas = sigmas.view(batch_size, 1, num_frames, 1, 1).to(generated_latents.dtype)
        noisy_latents = (1 - sigmas) * generated_latents + sigmas * noise

        # ── i2v fix: frame 0 must stay clean cond w/ timestep=0 ──
        # Both teacher (base Wan2.2) and critic were trained with the i2v
        # convention: frame 0 = clean cond image, timestep[frame 0] = 0.
        # If we feed them a NOISED frame 0 with timestep=t, they go OOD and
        # output garbage v_pred at frame 0 → DMD gradient there is noise →
        # contaminates generator weights. The streaming forward already keeps
        # frame 0 = cond at the cache-seed step, so we just need to (1) restore
        # the unnoised cond at frame 0 here and (2) zero its timestep, mirroring
        # `inference-sft.py` L173 (`latents_video[:, :, 0:1] = img_latent`) +
        # L181 (`first_frame_mask` zeros frame 0's timestep).
        # Also masks frame 0 out of the gradient/loss further below.
        noisy_latents[:, :, 0:1, :, :] = generated_latents[:, :, 0:1, :, :].detach()
        timesteps_per_frame = timesteps_per_frame.clone()
        timesteps_per_frame[:, 0] = 0
        sigmas = sigmas.clone()
        sigmas[:, :, 0:1, :, :] = 0

        # Expand timesteps to match spatial sequence length: [B, F] -> [B, seq_len]
        # Each frame's timestep is repeated for all spatial tokens in that frame.
        # MUST come AFTER the frame-0 mask so the spatial-token timesteps for
        # frame 0 are also zeroed.
        spatial_tokens_per_frame = (latent_h // p_h) * (latent_w // p_w)
        timesteps = timesteps_per_frame.repeat_interleave(spatial_tokens_per_frame, dim=1)
        assert timesteps.shape == (batch_size, seq_len), f"Expected timesteps shape {(batch_size, seq_len)}, got {timesteps.shape}"

        # Keep original shape [B, C, F, H, W] for model input
        noisy_for_score = noisy_latents
        generated_for_loss = generated_latents

        # ── Step 1: fake_score forward (raw v_pred) ──
        # critic uses student-style control: normalized skel (add-plus)
        fake_v = critic(
            hidden_states=noisy_for_score,
            timestep=timesteps,
            encoder_hidden_states=conditional_dict["prompt_embeds"],
            encoder_hidden_states_image=None,
            attention_kwargs=None,
            return_dict=False,
            control_video_latent=student_control_latent,
        )[0]

        if torch.isnan(fake_v).any() or torch.isinf(fake_v).any():
            print(
                f"[FAKE_NAN] fake_v has NaN/Inf | "
                f"noisy stats: mean={noisy_for_score.float().mean().item():.4f} "
                f"std={noisy_for_score.float().std().item():.4f} "
                f"absmax={noisy_for_score.float().abs().max().item():.4f} | "
                f"sigmas: min={sigmas.float().min().item():.4f} max={sigmas.float().max().item():.4f} | "
                f"fake_v: nan_count={torch.isnan(fake_v).sum().item()}/{fake_v.numel()}",
                flush=True,
            )

        if fake_guidance_scale != 0.0 and unconditional_dict is not None:
            fake_v_uncond = critic(
                hidden_states=noisy_for_score,
                timestep=timesteps,
                encoder_hidden_states=unconditional_dict["prompt_embeds"],
                encoder_hidden_states_image=None,
                attention_kwargs=None,
                return_dict=False,
                control_video_latent=student_control_latent,
            )[0]
            fake_v = fake_v + fake_guidance_scale * (fake_v - fake_v_uncond)

        # ── Step 2: real_score forward (raw v_pred, with CFG) ──
        # real_score uses teacher-style control: caller decides RAW vs NORMALIZED
        # via the teacher_control_latent argument (depends on --teacher_control_type)
        real_v = real_score(
            hidden_states=noisy_for_score,
            timestep=timesteps,
            encoder_hidden_states=conditional_dict["prompt_embeds"],
            encoder_hidden_states_image=None,
            attention_kwargs=None,
            return_dict=False,
            control_video_latent=teacher_control_latent,
        )[0]

        # NaN diagnostic: real_score is frozen, so any NaN here points to
        # input/weight/numerical issues. Dump first-fault stats for debugging.
        if torch.isnan(real_v).any() or torch.isinf(real_v).any():
            _ni = torch.isnan(noisy_for_score).any().item() or torch.isinf(noisy_for_score).any().item()
            _wi_first = next(real_score.parameters())
            _wi = torch.isnan(_wi_first).any().item() or torch.isinf(_wi_first).any().item()
            _ei = torch.isnan(conditional_dict["prompt_embeds"]).any().item() or torch.isinf(conditional_dict["prompt_embeds"]).any().item()
            _ti = torch.isnan(timesteps.float()).any().item() or torch.isinf(timesteps.float()).any().item()
            print(
                f"[REAL_NAN] real_v has NaN/Inf | "
                f"input nan/inf: {_ni} | weight nan/inf: {_wi} | embeds nan/inf: {_ei} | timesteps nan/inf: {_ti} | "
                f"noisy stats: mean={noisy_for_score.float().mean().item():.4f} "
                f"std={noisy_for_score.float().std().item():.4f} "
                f"absmax={noisy_for_score.float().abs().max().item():.4f} | "
                f"weight stats: absmax={_wi_first.float().abs().max().item():.4f} | "
                f"sigmas: min={sigmas.float().min().item():.4f} max={sigmas.float().max().item():.4f} | "
                f"timesteps: min={timesteps.min().item()} max={timesteps.max().item()} | "
                f"real_v: nan_count={torch.isnan(real_v).sum().item()}/{real_v.numel()} inf_count={torch.isinf(real_v).sum().item()}",
                flush=True,
            )

        # Diagnostic 7: CFG strength on the teacher (real_score).
        # When real_guidance_scale > 0 we run real_score twice (cond +
        # uncond) and fuse. If `|real_v_cfg - real_v_cond| / |real_v_cond|`
        # is ~0 then the prompt has no effect on the teacher → silent
        # failure (CFG path is doing nothing useful). Whenever
        # real_guidance_scale != 0 (the default), this metric is populated.
        _diag_real_cfg_strength = 0.0  # populated only if CFG path taken
        if real_guidance_scale != 0.0 and unconditional_dict is not None:
            real_v_uncond = real_score(
                hidden_states=noisy_for_score,
                timestep=timesteps,
                encoder_hidden_states=unconditional_dict["prompt_embeds"],
                encoder_hidden_states_image=None,
                attention_kwargs=None,
                return_dict=False,
                control_video_latent=teacher_control_latent,
            )[0]
            if torch.isnan(real_v_uncond).any() or torch.isinf(real_v_uncond).any():
                print(
                    f"[REAL_NAN] real_v_uncond has NaN/Inf | "
                    f"uncond_embeds nan/inf: {torch.isnan(unconditional_dict['prompt_embeds']).any().item() or torch.isinf(unconditional_dict['prompt_embeds']).any().item()} | "
                    f"real_v_uncond: nan_count={torch.isnan(real_v_uncond).sum().item()}/{real_v_uncond.numel()}",
                    flush=True,
                )
            _real_v_pre_cfg_absmean = float(real_v.abs().mean().item())
            real_v = real_v + real_guidance_scale * (real_v - real_v_uncond)
            _real_v_post_cfg_absmean = float(real_v.abs().mean().item())
            # Relative shift in real_v magnitude after CFG fusion. ~0 = no
            # effect (silent failure), large = healthy CFG impact.
            _diag_real_cfg_strength = abs(
                _real_v_post_cfg_absmean - _real_v_pre_cfg_absmean
            ) / max(_real_v_pre_cfg_absmean, 1e-12)

        # ── Step 3: convert v_pred → x_0 (CRITICAL — see docstring) ──
        # x_0 = noisy - σ * v   (flow matching identity)
        #
        # fp32 cast: bf16 has only 7-bit mantissa. At
        # high σ (e.g. with DMD_FLOW_SHIFT=5.0 → σ median ≈ 0.84, σ>0.9 ≈ 35%),
        # `σ·v_pred` and `noisy` are similar magnitude → bf16 subtraction
        # loses 1-2% of the result via catastrophic cancellation. Compounded
        # across the two independent x_0 reconstructions, the differential
        # `grad = fake_x0 - real_x0` picks up ~2% relative noise — exactly
        # the quantity we differentiate. Casting all three operands to fp32
        # for the FMA eliminates this noise floor. Transient VRAM ~3GB/rank
        # @ B=1,C=16,F=21,H=60,W=104 (well within headroom).
        # Override via DMD_X0_FP32=0 to revert to bf16 (legacy behavior).
        if os.environ.get("DMD_X0_FP32", "1") == "1":
            _noisy_f32 = noisy_for_score.float()
            _sigmas_f32 = sigmas.float()
            fake_x0 = _noisy_f32 - _sigmas_f32 * fake_v.float()
            real_x0 = _noisy_f32 - _sigmas_f32 * real_v.float()
            del _noisy_f32, _sigmas_f32
        else:
            fake_x0 = noisy_for_score - sigmas * fake_v
            real_x0 = noisy_for_score - sigmas * real_v

        # ── Step 4: DMD gradient in x_0 space (FRAME 1..F-1 ONLY) ──
        # Slice to skip frame 0 right here — after the i2v fix above,
        # grad[:, :, 0] = (fake_x0 - real_x0)[:, :, 0] is structurally 0
        # (both sides equal noisy[0] when σ[0]=0). Pre-slicing makes the
        # normalizer math (B1 fix) and diagnostic absmean stats consistent
        # with what the loss actually sees.
        grad = (fake_x0 - real_x0)[:, :, 1:, :, :]

        # ── Step 5: per-sample normalization (SF dmd.py:118-120) ──
        # `p_real` measures the per-sample distance from generator output to
        # the real-score's denoising target. Both terms are in x_0 space.
        # `dim=[1,2,3,4]` keeps batch dim → each video gets its own
        # normalizer, so loss magnitude doesn't mix across the batch.
        #
        # Skip frame 0 from p_real: the i2v fix above
        # forces sigmas[0]=0 → real_x0[0] = noisy[0] = generated[0] →
        # p_real[0] is structurally zero. Including it shrinks the
        # normalizer by (F-1)/F and scales `grad` UP by F/(F-1) ≈ 5%
        # at F=21 — a constant systematic bias on every DMD step. Slice
        # to the actual training frames so normalizer reflects real signal.
        #
        # Replace `clamp(min=1e-6)` with explicit
        # safe-mask. If a sample's p_real collapses (e.g. post-warmup the
        # generator briefly matches real_score x0 reconstruction at high σ),
        # the old code amplified that sample's `grad / 1e-6 = grad * 1e6`
        # — finite but enormous, evading the `nan_to_num` net, and
        # dominating the per-step gradient direction (magnitude clipped to
        # 1.0 by DeepSpeed grad_clip, but DIRECTION corrupted by the
        # one collapsed sample). New behavior: degenerate samples get
        # `grad := 0` (no contribution that step), healthy samples
        # normalize as before. LongLive `dmd.py:125-127` does not clamp
        # — relies on nan_to_num → finfo.max which is even worse on bf16.
        # Threshold tunable via DMD_NORMALIZER_THRESH (default 1e-6 = old
        # clamp value, just used as on/off boundary now).
        p_real_sliced = (generated_for_loss.float() - real_x0)[:, :, 1:, :, :]
        normalizer = p_real_sliced.abs().mean(dim=[1, 2, 3, 4], keepdim=True)
        _norm_thresh = float(os.environ.get("DMD_NORMALIZER_THRESH", "1e-6"))
        _safe_mask = (normalizer > _norm_thresh).float()
        # Set degenerate normalizers to 1 to avoid 0/0 → nan; the
        # multiplicative mask below zeros those samples' grad anyway.
        _norm_safe = torch.where(
            _safe_mask > 0, normalizer, torch.ones_like(normalizer))
        grad = grad / _norm_safe
        grad = grad * _safe_mask
        grad = torch.nan_to_num(grad)

        # ── Diagnostics (in no_grad block — these are scalars only) ──
        # All means/abs/std before the loss is computed, so we can read off
        # the actual driving signal each step in the train log.
        # NB: fake_x0/real_x0/generated absmean kept over ALL frames so the
        # number is a stable "snapshot of model output magnitude". Other
        # diagnostics use the sliced (frame 1..F-1) view so they describe
        # the actual training signal.
        _diag_fake_x0_absmean = float(fake_x0.abs().mean().item())
        _diag_real_x0_absmean = float(real_x0.abs().mean().item())
        _diag_gen_x0_absmean = float(generated_for_loss.abs().mean().item())
        _raw_grad_sliced = (fake_x0[:, :, 1:, :, :] - real_x0[:, :, 1:, :, :])
        _diag_raw_grad_absmean = float(_raw_grad_sliced.abs().mean().item())
        _diag_normalized_grad_absmean = float(grad.abs().mean().item())  # already sliced + masked
        _diag_normalizer_mean = float(normalizer.mean().item())
        _diag_normalizer_min = float(normalizer.min().item())
        _diag_normalizer_max = float(normalizer.max().item())
        # B3 fix telemetry: fraction of batch samples that got safe-masked
        # to zero. 0.0 = all healthy; 1.0 = whole batch collapsed (would
        # have made the old clamp blow up). Log to spot the bug recurring.
        _diag_normalizer_safemask_frac_zero = float((1.0 - _safe_mask).mean().item())
        # cosine(fake_x0 - real_x0, generated - real_x0): if positive, the DMD
        # gradient direction is "moving generator output toward real_x0",
        # i.e. the loss is teaching the right thing this step. Negative = the
        # student is closer to real than to fake along this axis. Shape-
        # consistent: both terms are the sliced fp32 versions.
        _flat_grad = _raw_grad_sliced.flatten()
        _flat_p_real = p_real_sliced.flatten()
        _denom = (_flat_grad.norm() * _flat_p_real.norm()).clamp(min=1e-12)
        _diag_grad_p_real_cosine = float((_flat_grad * _flat_p_real).sum().item() / _denom.item())
        # Per-sample sigma stats (training timestep distribution this step)
        _diag_sigma_mean = float(sigmas.mean().item())
        _diag_sigma_min = float(sigmas.min().item())
        _diag_sigma_max = float(sigmas.max().item())

        # ────────────────────────────────────────────────────────────────
        # DMD diagnostics (systematic per-step debug signals)
        # ────────────────────────────────────────────────────────────────

        # Diag 1: i2v frame-0 sanity. After i2v fix sigmas[:,:,0]=0 →
        # fake_x0[:,:,0] == real_x0[:,:,0] == noisy[:,:,0] == generated[:,:,0]
        # (all four are identically the cond image). Any deviation > 0
        # signals the i2v invariant got broken somewhere (e.g. a code path
        # that doesn't honor the timesteps_per_frame[:,0]=0 clamp). Should
        # be EXACTLY 0.0 every step; >1e-4 = bug, >1e-2 = severe.
        _diag_f0_dev_fake = float(
            (fake_x0[:, :, 0].float() - generated_for_loss[:, :, 0].float()).abs().max().item()
        )
        _diag_f0_dev_real = float(
            (real_x0[:, :, 0].float() - generated_for_loss[:, :, 0].float()).abs().max().item()
        )

        # Diag 2: per-frame grad distribution. `grad` is already sliced to
        # [B, C, F-1, H, W] (the actual training signal). Aggregate to [F-1]
        # by averaging over batch/channel/spatial dims. Then report:
        #   - first frame (frame 1, immediately after cond)
        #   - last frame (frame F-1, far end of rollout)
        #   - ratio (last/first)
        # Diagnostic value:
        #   - ratio > 5 sustained → SGT picks late-rollout substeps too often
        #     (model overtrained on rollout tail, undertrained on early frames)
        #   - ratio < 0.2 sustained → SGT biased toward early substeps
        #   - healthy: ratio ∈ [0.5, 2.0]
        # Slight degenerate case: F=1 → grad is empty after slice; guard.
        if grad.shape[2] >= 2:
            _grad_per_frame = grad.abs().float().mean(dim=[0, 1, 3, 4])  # [F-1]
            _diag_grad_f1 = float(_grad_per_frame[0].item())
            _diag_grad_fN = float(_grad_per_frame[-1].item())
            _diag_grad_f1_fN_ratio = float(
                _diag_grad_fN / max(_diag_grad_f1, 1e-12)
            )
            # Diag 2b: per-frame cos(g, p_real) — direction
            # quality varies by frame? Late frames in cap=6 cache have less
            # GT-anchored context, so we expect their DMD signal to be
            # noisier. If `cos_fN ≪ cos_f1` sustained → late-frame cache
            # corruption is hurting the DMD gradient direction → no amount
            # of more critic updates will fix it (need structural change to
            # cache or rollout).
            #   _raw_grad_sliced  shape: [B, C, F-1, H, W]   (fake_x0 - real_x0, frame 1..F-1)
            #   p_real_sliced     shape: [B, C, F-1, H, W]   (gen     - real_x0, frame 1..F-1)
            # For each frame f ∈ [0, F-2], flatten over [B, C, H, W], compute cos.
            # Use fp32 for numerical sanity at small magnitudes.
            _rg32 = _raw_grad_sliced.float()      # [B, C, F-1, H, W]
            _pr32 = p_real_sliced.float()
            _F_grad = _rg32.shape[2]
            # Sum over [B, C, H, W] but keep F-1 axis.
            _dot_f = (_rg32 * _pr32).sum(dim=[0, 1, 3, 4])    # [F-1]
            _norm_g_f = _rg32.pow(2).sum(dim=[0, 1, 3, 4]).sqrt()
            _norm_p_f = _pr32.pow(2).sum(dim=[0, 1, 3, 4]).sqrt()
            _cos_per_f = (_dot_f / (_norm_g_f * _norm_p_f).clamp(min=1e-12)).cpu().tolist()
            # Frame indices in the SLICED grad map to global frame f+1
            # (since we sliced off frame 0). Pick f1, f_mid, f_last for log.
            _idx_mid = _F_grad // 2
            _diag_cos_f1   = float(_cos_per_f[0])
            _diag_cos_fMid = float(_cos_per_f[_idx_mid])
            _diag_cos_fN   = float(_cos_per_f[-1])
            # Also pull raw_g for the same 3 frames (raw, not normalized).
            _raw_g_per_f = _rg32.abs().mean(dim=[0, 1, 3, 4]).cpu().tolist()  # [F-1]
            _diag_rawg_f1   = float(_raw_g_per_f[0])
            _diag_rawg_fMid = float(_raw_g_per_f[_idx_mid])
            _diag_rawg_fN   = float(_raw_g_per_f[-1])
            del _rg32, _pr32, _dot_f, _norm_g_f, _norm_p_f
        else:
            _diag_grad_f1 = float("nan")
            _diag_grad_fN = float("nan")
            _diag_grad_f1_fN_ratio = float("nan")
            _diag_cos_f1 = float("nan")
            _diag_cos_fMid = float("nan")
            _diag_cos_fN = float("nan")
            _diag_rawg_f1 = float("nan")
            _diag_rawg_fMid = float("nan")
            _diag_rawg_fN = float("nan")

        # Diag 3: v-space critic vs real_score divergence. DMD core
        # hypothesis = critic has LEARNED the student-vs-real gap. If
        # |fake_v - real_v| ≈ |real_v| × small → critic just mirrors
        # real_score → no DMD signal (cos will still be ~0, but with
        # potentially nonzero raw_g due to bf16 noise). If
        # |fake_v - real_v| ≫ |real_v| → critic blew up.
        # Healthy: 0.05 < ratio < 0.3.
        _diag_real_v_absmean = float(real_v.abs().mean().item())
        _diag_fake_v_absmean = float(fake_v.abs().mean().item())
        _diag_v_diff_absmean = float((fake_v - real_v).abs().mean().item())
        _diag_v_diff_to_real_ratio = (
            _diag_v_diff_absmean / max(_diag_real_v_absmean, 1e-12)
        )

        # Diag 7 (continued): bring the CFG-strength scalar computed up in
        # Step 2 (uses outer scope variable) into the local namespace for
        # the return dict. Default 0.0 if real_guidance_scale==0 path.
        _diag_real_cfg_strength_out = _diag_real_cfg_strength

    # Loss — gradient flows back through generated_for_loss only.
    # i2v fix: drop frame 0 from the loss (it's the cond image, not a target the
    # DMD push is supposed to move). Note: `grad` is now already sliced to
    # [B, C, F-1, H, W] above, so we no longer slice it here.
    loss = 0.5 * F.mse_loss(
        generated_for_loss[:, :, 1:, :, :].double(),
        (generated_for_loss[:, :, 1:, :, :].double()
         - grad.double()).detach(),
        reduction="mean",
    )

    # ── DMD debug snapshot ──────────────────────────────────────────────
    # Populate one slot of the module-level ring buffer with the EXACT 4
    # tensors the loss just differentiated:
    #   generated_for_loss   = student rollout output (the DMD "subject")
    #   fake_x0              = critic's reconstruction of x_0 from noised version
    #   real_x0              = teacher's reconstruction of x_0 from noised version
    #   noisy_for_score      = (1-σ)·student + σ·noise, fed to both networks
    # Plus σ context + cos(g, p_real) + safemask telemetry for the title.
    #
    # All-rank populate so any rank can serve the consumer (though
    # downstream decode is rank-0-only). Cost: 4 × ~5 MB GPU→CPU copy
    # (~10 ms total at B=1). DMD_DEBUG_RING=0 / DECODE_DMD_DEBUG=0
    # both disable.
    if (_DMD_DEBUG_RING_SIZE > 0
            and os.environ.get("DECODE_DMD_DEBUG", "1") != "0"):
        with torch.no_grad():
            try:
                slot = _dmd_debug_slot_template()
                slot["generated_for_loss"] = generated_for_loss.detach().float().cpu()
                slot["fake_x0"]            = fake_x0.detach().float().cpu()
                slot["real_x0"]            = real_x0.detach().float().cpu()
                slot["noisy_for_score"]    = noisy_for_score.detach().float().cpu()
                slot["sigmas"]             = sigmas.detach().float().cpu()
                # ★ The actual normalized DMD gradient at the student's pixels —
                # this is what `loss.backward()` receives (up to constant factor)
                # and what the heatmap row visualizes. `grad` is sliced to frames
                # 1..F-1 (frame 0 = cond, no DMD push); we pad with zero at frame
                # 0 so shape matches generated/fake/real and `_make_heatmap_row`
                # can index uniformly.
                _grad_full_shape = torch.zeros_like(fake_x0)
                _grad_full_shape[:, :, 1:, :, :] = grad.to(fake_x0.dtype)
                slot["dmd_grad"] = _grad_full_shape.detach().float().cpu()
                slot["cos"]                = float(_diag_grad_p_real_cosine)
                slot["safemask_zero"]      = float(_diag_normalizer_safemask_frac_zero)
                slot["sigma_mean"]         = float(_diag_sigma_mean)
                slot["sigma_min"]          = float(_diag_sigma_min)
                slot["sigma_max"]          = float(_diag_sigma_max)
                slot["gen_x0_absmean"]     = float(_diag_gen_x0_absmean)
                slot["fake_x0_absmean"]    = float(_diag_fake_x0_absmean)
                slot["real_x0_absmean"]    = float(_diag_real_x0_absmean)
                slot["raw_grad_absmean"]   = float(_diag_raw_grad_absmean)
                slot["normer_min"]         = float(_diag_normalizer_min)
                # `step_id` filled by the train loop right after this fn
                # returns (caller has access to the step counter).
                idx = _DMD_DEBUG_WRITES[0] % _DMD_DEBUG_RING_SIZE
                _DMD_DEBUG_RING[idx] = slot
                _DMD_DEBUG_WRITES[0] += 1
            except Exception:
                # Snapshot is non-critical — never crash DMD over it.
                pass

    return loss, {
        # Legacy field — kept for compatibility with existing log parsers
        "dmdtrain_gradient_norm": _diag_normalized_grad_absmean,
        # x_0 magnitudes from each branch — useful for spotting collapse:
        #   fake_x0_absmean → 0  : critic learned "everything is zero" (mode collapse)
        #   real_x0_absmean huge : real_score is OOD on this batch (rare for frozen teacher)
        #   gen_x0_absmean huge  : generator output exploding (will cascade to NaN)
        "dmd_fake_x0_absmean": _diag_fake_x0_absmean,
        "dmd_real_x0_absmean": _diag_real_x0_absmean,
        "dmd_gen_x0_absmean": _diag_gen_x0_absmean,
        # Raw vs normalized DMD signal — ratio = 1/normalizer.mean() roughly
        "dmd_raw_grad_absmean": _diag_raw_grad_absmean,
        "dmd_norm_grad_absmean": _diag_normalized_grad_absmean,
        # Normalizer health — if min → 0 the safe-mask kicked in, signals trouble.
        # With B3 fix the old `clamp(1e-6)` is replaced by a 0/1 safe-mask:
        # samples whose normalizer < DMD_NORMALIZER_THRESH (default 1e-6) get
        # `grad := 0` (no contribution this step) instead of `grad *= 1e6`.
        "dmd_normalizer_mean": _diag_normalizer_mean,
        "dmd_normalizer_min": _diag_normalizer_min,
        "dmd_normalizer_max": _diag_normalizer_max,
        # B3 telemetry: fraction of batch with normalizer below threshold
        # (these samples contribute zero grad this step). 0 = healthy,
        # >0 = some video collapsed at this σ-sample.
        "dmd_normalizer_safemask_zero_frac": _diag_normalizer_safemask_frac_zero,
        # Direction sanity: cos((fake_x0-real_x0), (gen-real_x0)). Positive
        # = the gradient is pulling the generator toward real_x0 (correct);
        # negative = pulling it AWAY (would have indicated a sign flip bug
        # before this fix; should now be consistently positive).
        "dmd_grad_p_real_cos": _diag_grad_p_real_cosine,
        # Sigma (timestep) distribution this step
        "dmd_sigma_mean": _diag_sigma_mean,
        "dmd_sigma_min": _diag_sigma_min,
        "dmd_sigma_max": _diag_sigma_max,
        # ── Additional diagnostics ───────────────────────────────────
        # Diag 1: frame-0 i2v sanity. Both should be EXACTLY 0.0
        # under correct i2v invariant. >1e-4 = bug.
        "dmd_f0_dev_fake": _diag_f0_dev_fake,
        "dmd_f0_dev_real": _diag_f0_dev_real,
        # Diag 2: per-frame grad spread (catches SGT bucket imbalance)
        "dmd_grad_f1": _diag_grad_f1,
        "dmd_grad_fN": _diag_grad_fN,
        "dmd_grad_f1_fN_ratio": _diag_grad_f1_fN_ratio,
        # Diag 2b: per-frame raw_g + cos for f1/mid/fN.
        # Decides whether DMD gradient quality degrades along temporal axis.
        # If cos_fN ≪ cos_f1 sustained → late frames in cap=6 cache get noisy
        # DMD direction → structural issue, hparam changes won't help.
        "dmd_rawg_f1": _diag_rawg_f1,
        "dmd_rawg_fMid": _diag_rawg_fMid,
        "dmd_rawg_fN": _diag_rawg_fN,
        "dmd_cos_f1": _diag_cos_f1,
        "dmd_cos_fMid": _diag_cos_fMid,
        "dmd_cos_fN": _diag_cos_fN,
        # Diag 3: v-space critic vs real_score divergence
        "dmd_real_v_absmean": _diag_real_v_absmean,
        "dmd_fake_v_absmean": _diag_fake_v_absmean,
        "dmd_v_diff_absmean": _diag_v_diff_absmean,
        "dmd_v_diff_to_real_ratio": _diag_v_diff_to_real_ratio,
        # Diag 7: real_score CFG strength (silent failure detector)
        "dmd_real_cfg_strength": _diag_real_cfg_strength_out,
    }


def compute_critic_loss(
    generated_latents,
    critic,
    conditional_dict,
    num_train_timesteps,
    min_step_frac=0.02,
    max_step_frac=0.98,
    patch_size=(1, 2, 2),
    student_control_latent=None,    # [B, C, F, H, W] — NORMALIZED skel for critic
                                    # (must match generator's add-plus mode)
):
    """
    Critic loss: denoising loss on generated samples.

    Control distill: critic sees same NORMALIZED skel as generator (add-plus)
    so its denoising target distribution matches what generator produces.
    """
    batch_size, num_channels, num_frames, latent_h, latent_w = generated_latents.shape
    p_t, p_h, p_w = patch_size
    seq_len = (num_frames // p_t) * (latent_h // p_h) * (latent_w // p_w)

    min_step = int(min_step_frac * num_train_timesteps)
    max_step = int(max_step_frac * num_train_timesteps)

    # ── Match compute_dmd_generator_loss timestep-sampling logic ──
    # Same DMD_UNIFORM_TIMESTEP / FLOW_SHIFT env vars so generator and
    # critic see consistent σ distribution. See compute_dmd_generator_loss
    # docstring for rationale.
    _dmd_uniform_t = bool(int(os.environ.get("DMD_UNIFORM_TIMESTEP", "0")))
    if _dmd_uniform_t:
        _t_shared = torch.randint(min_step, max_step, (batch_size, 1), device=generated_latents.device)
        timesteps_per_frame = _t_shared.repeat(1, num_frames)
    else:
        timesteps_per_frame = torch.randint(min_step, max_step, (batch_size, num_frames), device=generated_latents.device)

    _dmd_shift = float(os.environ.get("DMD_FLOW_SHIFT", "1.0"))
    if _dmd_shift != 1.0:
        _t_norm = timesteps_per_frame.float() / num_train_timesteps
        _t_norm = _dmd_shift * _t_norm / (1 + (_dmd_shift - 1) * _t_norm)
        timesteps_per_frame = (_t_norm * num_train_timesteps).long().clamp(
            min=min_step, max=max_step - 1
        )

    noise = torch.randn_like(generated_latents)
    sigmas = timesteps_per_frame.float() / num_train_timesteps
    # See compute_dmd_generator_loss — sigmas must be [B, 1, F, 1, 1] to broadcast
    # over channels and spatial dims of the [B, C, F, H, W] latent.
    sigmas = sigmas.view(batch_size, 1, num_frames, 1, 1).to(generated_latents.dtype)
    noisy_latents = (1 - sigmas) * generated_latents + sigmas * noise

    # ── i2v fix: same as compute_dmd_generator_loss ──
    # Frame 0 must stay clean cond w/ timestep=0 so critic sees in-distribution
    # input there (mirrors `inference-sft.py`'s first_frame_mask convention).
    # Without this, critic learns to "predict noise residual" for cond image
    # at high σ — out-of-distribution and produces noisy fake_score in DMD.
    noisy_latents[:, :, 0:1, :, :] = generated_latents[:, :, 0:1, :, :].detach()
    timesteps_per_frame = timesteps_per_frame.clone()
    timesteps_per_frame[:, 0] = 0
    sigmas = sigmas.clone()
    sigmas[:, :, 0:1, :, :] = 0

    # Expand timesteps to match spatial sequence length (AFTER frame-0 zeroing)
    spatial_tokens_per_frame = (latent_h // p_h) * (latent_w // p_w)
    timesteps = timesteps_per_frame.repeat_interleave(spatial_tokens_per_frame, dim=1)

    # Keep original shape [B, C, F, H, W] for model input
    pred = critic(
        hidden_states=noisy_latents,
        timestep=timesteps,
        encoder_hidden_states=conditional_dict["prompt_embeds"],
        encoder_hidden_states_image=None,
        attention_kwargs=None,
        return_dict=False,
        control_video_latent=student_control_latent,
    )[0]

    # Target is the noise residual (flow matching target).
    # i2v fix: skip frame 0 — it's the cond image, σ=0, target=0 trivially,
    # but more importantly we don't want critic to "learn" anything from it.
    targets = noise - generated_latents

    loss = F.mse_loss(
        pred[:, :, 1:, :, :].double(),
        targets[:, :, 1:, :, :].double(),
        reduction="mean",
    )

    return loss


# ──────────────────────────────────────────────────────────────
# EMA helpers (for --ema_decay > 0)
# ──────────────────────────────────────────────────────────────
#
# Why this design:
# ----------------
# Under ZeRO-3 with offload_optimizer.device=cpu the *fp32 master* weights
# (the ones the optimizer.step() actually mutates) live as flat CPU tensors
# at  `engine.optimizer.fp32_partitioned_groups_flat[i]`. Each rank holds
# exactly its own 1/world_size slice — no cross-rank duplication. We exploit
# this:
#   - EMA shadow = list of CPU fp32 tensors with identical shape to those
#     master partitions (one per ZeRO sub-group, one shard per rank).
#   - Per-step update: `shadow.lerp_(master, 1-decay)`. Pure CPU op, no
#     collectives, no GPU memory hit. Cost = O(local-shard-bytes) per step.
#   - Save: all-gather each shard (mimic `_fp32_state_allgather`) and write
#     a flat dict matching the model's state_dict keys.
#
# This is the SAME approach DeepSpeed itself uses for its own EMA (see
# `deepspeed.runtime.bf16_optimizer.BF16_Optimizer.update_lp_params`-adjacent
# code), just spelled out so we don't pull in a deepspeed-version-specific
# private API that may break across point releases.

# ──────────────────────────────────────────────────────────────────────────
# ZeRO-2 EMA support
# ──────────────────────────────────────────────────────────────────────────
# Different from ZeRO-3 path:
# - ZeRO-3: params are sharded across ranks; we EMA the sharded fp32 master
#   (engine.optimizer.fp32_partitioned_groups_flat). 1/world_size memory/rank.
# - ZeRO-2: params are NOT sharded (each rank holds full fp32 master). We
#   EMA the named_parameters() of engine.module directly. ~10 GB CPU/rank for
#   5B model in fp32 (cluster has 1 TB so plenty).
#
# Each rank does the same EMA update independently (because params are
# identical across ranks after backward+step). At save time only rank 0 dumps
# its shadow to disk. Matches the pattern used by `diffusers.training_utils.
# EMAModel`.
# ──────────────────────────────────────────────────────────────────────────

def _build_ema_shadow_zero2(engine):
    """Allocate a CPU fp32 EMA shadow that mirrors `engine.module.named_parameters()`.

    Returns a dict {name: cpu_fp32_tensor} initialized from current params.
    Memory: ~10 GB CPU per rank for 5B fp32 model. Each rank holds an
    identical copy (since params are not sharded under ZeRO-2).
    """
    shadow = {}
    for name, p in engine.module.named_parameters():
        shadow[name] = p.detach().to(device="cpu", dtype=torch.float32).clone()
    return shadow


@torch.no_grad()
def _ema_update_zero2(shadow, engine, decay: float):
    """In-place EMA: shadow[name] ← decay*shadow[name] + (1-decay)*param[name]
    (CPU fp32). Pure CPU op, no NCCL.
    """
    one_minus_decay = 1.0 - decay
    for name, p in engine.module.named_parameters():
        if name not in shadow:
            # Param added after shadow was built (e.g. lazy CPE init).
            shadow[name] = p.detach().to(device="cpu", dtype=torch.float32).clone()
            continue
        # detach + cast to fp32 CPU; lerp_ then is a contiguous fp32 vector op
        p_cpu_fp32 = p.detach().to(device="cpu", dtype=torch.float32)
        shadow[name].lerp_(p_cpu_fp32, one_minus_decay)


@torch.no_grad()
def _ema_consolidated_state_dict_zero2(shadow):
    """ZeRO-2 EMA shadow is already a full state_dict on each rank
    (no sharding). Just cast to bf16 for save (matches ZeRO-3's behavior so
    downstream load_teacher_into_pipe can read either one identically).

    Only the calling rank's shadow is read — caller should ensure only rank 0
    runs this to avoid wasted work.
    """
    return {name: t.to(dtype=torch.bfloat16) for name, t in shadow.items()}


@torch.no_grad()
def _ema_load_into_shadow_zero2(shadow, engine, full_state_dict_cpu):
    """Inverse of _ema_consolidated_state_dict_zero2: copy a full bf16
    state_dict (loaded from generator_ema.pt) into this rank's ZeRO-2 shadow
    dict (cast to fp32 CPU). Used on resume.

    No collectives needed: every rank loaded the same dict and ZeRO-2 keeps
    a full per-rank copy of the shadow.
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    n_loaded = 0
    n_skipped = 0
    for name, _p in engine.module.named_parameters():
        if name not in full_state_dict_cpu:
            n_skipped += 1
            continue
        # Force fp32 CPU contiguous so subsequent lerp_ is a fast vector op.
        shadow[name] = (
            full_state_dict_cpu[name]
            .detach()
            .to(dtype=torch.float32, device="cpu")
            .clone()
        )
        n_loaded += 1
    if rank == 0:
        cprint(f"[ema] resume (zero2): loaded {n_loaded} params into shadow "
               f"(skipped {n_skipped} not in checkpoint)", "green")


# ──────────────────────────────────────────────────────────────────────────
# Stage-aware EMA dispatcher
# ──────────────────────────────────────────────────────────────────────────
# Auto-detects ZeRO stage from engine.config and routes to the right impl.
# ZeRO-2 shadow is a dict {name: cpu_fp32}; ZeRO-3 shadow is list[CPU fp32].
# Both are CPU-only, no NCCL collectives during update.
# ──────────────────────────────────────────────────────────────────────────

def _detect_zero_stage(engine):
    """Probe engine to determine ZeRO stage. Falls back to 3 if unknown."""
    try:
        # DeepSpeed engine exposes config via engine.config (DeepSpeedConfig)
        # or via engine._config.zero_config.stage (private but stable).
        if hasattr(engine, "zero_optimization_stage"):
            return int(engine.zero_optimization_stage())
        if hasattr(engine, "_config") and hasattr(engine._config, "zero_config"):
            return int(engine._config.zero_config.stage)
    except Exception:
        pass
    # Heuristic: ZeRO-3 has fp32_partitioned_groups_flat, ZeRO-2 doesn't
    return 3 if hasattr(getattr(engine, "optimizer", None),
                        "fp32_partitioned_groups_flat") else 2


def _build_ema_shadow(engine):
    """Stage-aware: returns a shadow object usable with _ema_update / _ema_consolidated."""
    stage = _detect_zero_stage(engine)
    if stage == 2:
        return _build_ema_shadow_zero2(engine)
    return _build_ema_shadow_zero3(engine)


def _ema_update(shadow, engine, decay: float):
    stage = _detect_zero_stage(engine)
    if stage == 2:
        return _ema_update_zero2(shadow, engine, decay)
    return _ema_update_zero3(shadow, engine, decay)


def _ema_consolidated_state_dict(shadow, engine):
    """Returns a full bf16 state_dict (or {} on non-rank-0 for ZeRO-3 path)."""
    stage = _detect_zero_stage(engine)
    if stage == 2:
        # ZeRO-2: shadow IS already a full dict. Caller decides whether to
        # gate by rank 0 (we keep the same interface — caller should check rank).
        return _ema_consolidated_state_dict_zero2(shadow)
    return _ema_consolidated_state_dict_zero3(shadow, engine)


def _ema_load_into_shadow(shadow, engine, full_state_dict_cpu):
    """Stage-aware loader for resume: re-shards the saved generator_ema.pt
    into this rank's shadow regardless of ZeRO stage."""
    stage = _detect_zero_stage(engine)
    if stage == 2:
        return _ema_load_into_shadow_zero2(shadow, engine, full_state_dict_cpu)
    return _ema_load_into_shadow_zero3(shadow, engine, full_state_dict_cpu)


def _build_ema_shadow_zero3(engine):
    """Allocate a CPU fp32 EMA shadow that mirrors `engine.optimizer.fp32_partitioned_groups_flat`.

    Returns a list of CPU fp32 tensors (one per sub-group), each having the
    same numel as this rank's slice of that sub-group. Initialized from the
    current master weights so that EMA at step 0 == current weights.

    Memory: roughly (model_params_fp32 / world_size) bytes per rank. For our
    5B model on 8 ranks that's ~2.5 GB per rank, comfortably small.
    """
    masters = engine.optimizer.fp32_partitioned_groups_flat
    shadow = []
    for m in masters:
        # m is already CPU fp32 under offload_optimizer.device=cpu.
        # `.detach().clone()` so the shadow's storage is independent of the
        # master (otherwise lerp_ would no-op when src and dst alias).
        shadow.append(m.detach().to(device="cpu", dtype=torch.float32).clone())
    return shadow


@torch.no_grad()
def _ema_update_zero3(shadow, engine, decay: float):
    """In-place EMA: shadow = decay*shadow + (1-decay)*master.

    Implemented via tensor.lerp_(src, weight=1-decay): mathematically
        shadow ← shadow + (1-decay) * (master - shadow)
                = decay*shadow + (1-decay)*master.
    Pure CPU op; no GPU touch, no NCCL calls. Each rank updates its own slice.
    """
    masters = engine.optimizer.fp32_partitioned_groups_flat
    if len(masters) != len(shadow):
        raise RuntimeError(
            f"EMA shadow has {len(shadow)} groups but optimizer has "
            f"{len(masters)}. Did the model topology change mid-training?"
        )
    one_minus_decay = 1.0 - decay
    for sh, m in zip(shadow, masters):
        # m is fp32 CPU; sh is fp32 CPU. Same dtype → lerp_ is fast vector op.
        # If for some reason m moved to GPU (e.g. swap-in for an optimizer
        # state op), bring it back as fp32 CPU for this update.
        if m.device.type != "cpu":
            m_cpu = m.detach().to(device="cpu", dtype=torch.float32)
        else:
            m_cpu = m if m.dtype == torch.float32 else m.detach().float()
        sh.lerp_(m_cpu, one_minus_decay)


@torch.no_grad()
def _ema_consolidated_state_dict_zero3(shadow, engine):
    """All-gather EMA shadow shards from all ranks into a full state_dict
    (bf16) keyed by the same names as `engine.module.state_dict()`.

    Returns: a dict {param_name: full_bf16_tensor_on_cpu} on rank 0,
             {} on non-zero ranks (they participate in the all-gather but
             don't keep the result).

    Layout reminder (from deepspeed/runtime/zero/stage3.py):
    -------------------------------------------------------
    `fp32_partitioned_groups_flat[g]` is THIS RANK's flat shard for sub-group
    `g`. Within that shard, parameters are concatenated end-to-end at offsets
    given by `optimizer.grad_position[param_id] = (g, dest_offset, num_elements)`,
    where `num_elements == param.partition_numel()` (i.e. this rank's slice
    of that one param). The cross-rank layout for a single param is
    `[rank0_shard, rank1_shard, …, rankN_shard]` — gathered exactly that way
    by `_fp32_state_allgather`, then narrowed to `param.ds_numel` and
    reshaped to `param.ds_shape`.

    The bug-prone alternative — gathering the whole sub-group flat in one
    all-gather and then carving by `ds_numel` — does NOT work because the
    gathered buffer interleaves per-rank concatenations of the param shards,
    not full params back-to-back.

    Implementation: walk every model param, do per-param all-gather of this
    rank's EMA shadow slice (mirroring `_fp32_state_allgather`), and on
    rank 0 stash the bf16 result. Then add non-ZeRO buffers from the regular
    state_dict so the resulting checkpoint is a drop-in for `load_state_dict`.
    """
    out = {}
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    is_main = (rank == 0)

    optimizer = engine.optimizer
    masters = optimizer.fp32_partitioned_groups_flat

    # Sanity: same-sized shards across ranks (ZeRO-3 pads sub-groups so
    # they divide evenly by world_size).
    for i, (sh, m) in enumerate(zip(shadow, masters)):
        if sh.numel() != m.numel():
            raise RuntimeError(
                f"sub-group {i}: shadow numel {sh.numel()} != master numel "
                f"{m.numel()} — EMA shadow got out of sync. "
                "This usually means the model was re-prepared after EMA init."
            )

    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    dp_group = optimizer.dp_process_group

    # Walk every named parameter. Skip non-ZeRO ones (no .ds_id) — they get
    # picked up below from the regular state_dict.
    for name, param in engine.module.named_parameters():
        if not hasattr(param, "ds_id"):
            continue
        # Locate this param's slice inside the flat shadow buffer.
        # grad_position is keyed by the optimizer's internal param id.
        try:
            param_id = optimizer.get_param_id(param)
            group_idx, dest_offset, num_elements = optimizer.grad_position[param_id]
        except (AttributeError, KeyError):
            # Frozen / no-grad param — skip (would be all zeros in EMA anyway).
            continue

        # My rank's shard of this one param, fp32 on CPU → move to GPU for
        # all-gather (NCCL requires accelerator-resident buffers).
        local_shard = shadow[group_idx].narrow(0, dest_offset, num_elements).contiguous()
        local_shard_dev = local_shard.to(device=device, non_blocking=False)

        # Mirror _fp32_state_allgather: gather all rank shards into one buffer.
        full_buf = torch.empty(
            world_size * num_elements, dtype=torch.float32, device=device,
        )
        if dist.is_initialized() and world_size > 1:
            dist.all_gather_into_tensor(full_buf, local_shard_dev, group=dp_group)
        else:
            full_buf.copy_(local_shard_dev)

        if is_main:
            # Narrow to the unpadded numel and reshape to the param's logical shape.
            # Handle 0-dim scalar Parameters (e.g. control_scale).
            # `view(*())` raises TypeError because view() needs at least one arg.
            # For scalar params (ds_shape=(), ds_numel=1) use reshape(()) instead.
            narrowed = full_buf.narrow(0, 0, param.ds_numel)
            if len(param.ds_shape) == 0:
                # 0-dim scalar Parameter
                full = narrowed.reshape(())
            else:
                full = narrowed.view(*param.ds_shape)
            out[name] = full.to(dtype=torch.bfloat16, device="cpu").clone()

        del full_buf, local_shard_dev, local_shard

    # Add non-ZeRO buffers / non-param state from the underlying module so
    # the resulting checkpoint is a drop-in for `model.load_state_dict()`.
    # Only rank 0 keeps the dict; non-zero ranks return {}.
    if is_main:
        # `state_dict()` under ZeRO-3 gives partitioned (.ds_tensor) for ZeRO
        # params, which is wrong for our purposes — we already filled those in.
        # We only want buffers + any params we missed. Skip anything already in `out`.
        full_sd = engine.module.state_dict()
        for k, v in full_sd.items():
            if k in out:
                continue
            # Only safe to take if it isn't a ZeRO-partitioned tensor (those
            # would have wrong size). Buffers don't have .ds_id, so check
            # via the named_buffers route.
            out[k] = v.detach().to(dtype=torch.bfloat16, device="cpu").clone()
    return out


@torch.no_grad()
def _ema_load_into_shadow_zero3(shadow, engine, full_state_dict_cpu):
    """Inverse of _ema_consolidated_state_dict_zero3: take a full bf16
    state_dict (loaded from generator_ema.pt) and re-shard it into this
    rank's portion of `shadow`. Used on resume.

    Mirrors `set_full_hp_param` (stage3.py:2460): for each ZeRO param, take
    the full tensor, flatten it, narrow to `[my_rank * partition_numel :
    (my_rank+1) * partition_numel]` (zero-padding the global flat if it
    isn't already a multiple of world_size — same convention as DeepSpeed),
    then copy into the matching slice of `shadow[group_idx]`. No collectives
    needed: every rank loaded the same dict.
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    optimizer = engine.optimizer

    n_loaded = 0
    n_skipped = 0
    for name, param in engine.module.named_parameters():
        if not hasattr(param, "ds_id"):
            continue
        try:
            param_id = optimizer.get_param_id(param)
            group_idx, dest_offset, num_elements = optimizer.grad_position[param_id]
        except (AttributeError, KeyError):
            n_skipped += 1
            continue
        if name not in full_state_dict_cpu:
            raise KeyError(
                f"EMA resume: param '{name}' (ds_id={param.ds_id}) not found "
                "in checkpoint state_dict. Did the model architecture change?"
            )
        full_flat = full_state_dict_cpu[name].to(dtype=torch.float32, device="cpu").reshape(-1)
        # Pad up to world_size * partition_numel if needed (DeepSpeed pads
        # at allocation time; the unpadded numel == ds_numel).
        target_total = world_size * num_elements
        if full_flat.numel() < target_total:
            full_flat = torch.cat(
                [full_flat, torch.zeros(target_total - full_flat.numel(), dtype=torch.float32)],
                dim=0,
            )
        elif full_flat.numel() > target_total:
            raise RuntimeError(
                f"EMA resume '{name}': flat size {full_flat.numel()} "
                f"exceeds expected {target_total} (partition_numel={num_elements} × "
                f"world_size={world_size}). Padding logic is wrong."
            )
        my_slice = full_flat.narrow(0, rank * num_elements, num_elements).contiguous()
        shadow[group_idx].narrow(0, dest_offset, num_elements).copy_(my_slice)
        n_loaded += 1

    if rank == 0:
        cprint(f"[ema] resume: loaded {n_loaded} params into shadow "
               f"(skipped {n_skipped} non-trainable)", "green")


# ──────────────────────────────────────────────────────────────
# ODE warmup helpers (for --ode_warmup_steps > 0)
# ──────────────────────────────────────────────────────────────
#
# What gets cached on disk (one safetensors file per pair):
#   {
#     "noise":       [C=48, F=6, H=30, W=40] fp32 — random init (the same
#                                                   noise the student will
#                                                   receive at training time)
#     "x0_teacher":  [C=48, F=6, H=30, W=40] fp32 — teacher's 4-step output
#                                                   on this noise
#     "img_latent":  [C=48, 1, H=30, W=40]   fp32 — first frame condition
#                                                   (taken from a real video)
#     "text_embeds": [seq_len, 4096]         fp32 — T5-XXL embedding for the
#                                                   text prompt used by the teacher
#   }
# Generated by scripts/generate_ode_pairs.py — see that file for details.

class ODEPairDataset(Dataset):
    """Loads cached (noise, x0_teacher) pairs for ODE warmup.

    Independent of LatentDataset because the schema is different: ODE pairs
    don't carry full video latents, only the matched (noise, target) tuples
    needed for MSE warmup. Sharded by rank just like LatentDataset.

    Two schemas supported:
      - **Legacy** (`teacher` mode): keys = noise, x0_teacher, img_latent, text_embeds
      - **CausVid traj** (`causvid_traj` mode):
        all legacy keys PLUS:
          - x0_trajectory: [N+1, C, F, H, W]  teacher snapshots at N σ levels + clean final
          - traj_sigmas:   [N+1]              σ values; last entry = 0
        Generated by `scripts/generate_ode_pairs_v2.py --save_trajectory`.
    """

    def __init__(self, pairs_dir: str, max_samples: Optional[int] = None,
                 require_trajectory: bool = False):
        if not os.path.isdir(pairs_dir):
            raise FileNotFoundError(f"ODE pairs dir not found: {pairs_dir}")
        self.pairs = sorted(
            os.path.join(pairs_dir, f)
            for f in os.listdir(pairs_dir)
            if f.endswith(".safetensors")
        )
        if max_samples is not None:
            self.pairs = self.pairs[:max_samples]
        if not self.pairs:
            raise RuntimeError(f"No .safetensors files in {pairs_dir}")
        self.require_trajectory = require_trajectory

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        # ── FUSE/OSS-tolerant load (see LatentDataset for full rationale) ──
        import random as _random
        _MAX_FALLBACKS = 5
        _paths_tried = []
        _orig_idx = idx
        for _fb_attempt in range(_MAX_FALLBACKS):
            path = self.pairs[idx]
            _paths_tried.append(path)
            try:
                d = _safetensors_load_robust(path)
                # Validate schema once per file so a corrupt cache fails loudly,
                # not silently with a shape-mismatch exception 100 steps later.
                legacy_keys = ("noise", "x0_teacher", "img_latent", "text_embeds")
                for k in legacy_keys:
                    if k not in d:
                        raise KeyError(
                            f"ODE pair file missing key '{k}': {path}. "
                            "Re-generate via scripts/generate_ode_pairs.py."
                        )
                if self.require_trajectory:
                    for k in ("x0_trajectory", "traj_sigmas", "gt_video_latent"):
                        if k not in d:
                            raise KeyError(
                                f"ODE pair file missing key '{k}': {path}. "
                                "ODE_TARGET_KIND=causvid_traj requires --save_trajectory "
                                "pair-gen mode (scripts/generate_ode_pairs_v2.py). "
                                "gt_video_latent is the REAL GT video MSE target "
                                "(distinct from x0_teacher = teacher self-distill final)."
                            )
                if _fb_attempt > 0:
                    print(
                        f"[ode_pairs:fallback] recovered from idx={_orig_idx} → "
                        f"idx={idx} after {_fb_attempt} swap(s); path={path}",
                        flush=True,
                    )
                return d
            except (FileNotFoundError, OSError, RuntimeError) as e:
                # KeyError above re-raises as-is (schema is a hard error).
                print(
                    f"[ode_pairs:fallback] persistent failure idx={idx} "
                    f"({type(e).__name__}: {e!s:.150}); swap to random idx",
                    flush=True,
                )
                idx = _random.randint(0, len(self.pairs) - 1)
        raise RuntimeError(
            f"ODEPairDataset failed after {_MAX_FALLBACKS} fallback attempts "
            f"starting from idx={_orig_idx}. Tried paths: {_paths_tried}"
        )


# ──────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Streaming DMD Distillation Training")
    parser.add_argument("--model_path", type=str, default="Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    parser.add_argument("--data_path", type=str, required=True, help="JSON or directory of .safetensors")
    parser.add_argument("--output_dir", type=str, default="outputs/distill")
    parser.add_argument("--num_train_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate_gen", type=float, default=1e-5)
    parser.add_argument("--learning_rate_critic", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    # Adam/AdamW hyperparameters — default to the Self-Forcing/CausVid DMD
    # recipe (beta1=0, beta2=0.999, eps=1e-8). Unlike SGD-flavored training,
    # DMD critics benefit from zeroed momentum (beta1=0) because the gradient
    # direction shifts as the generator distribution moves — persistent
    # first-moment estimates would track a stale distribution. See
    # "Self Forcing" (Huang et al. 2025) Table 3.
    parser.add_argument("--adam_beta1", type=float, default=0.0,
                        help="Adam β1 (first moment). Self Forcing uses 0.0 for DMD.")
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_eps", type=float, default=1e-8)
    parser.add_argument("--num_latent_frames", type=int, default=6,
                        help=("Per-iter latent-frame CAP. >0 truncates the "
                              "dataset's video_latent to this many frames; "
                              "0 = use the dataset's native length per sample "
                              "(supports mixed 7F/21F clips in one run). "
                              "ODE warmup ignores this and uses the pair "
                              "x0_teacher's native frame count."))
    parser.add_argument("--filter_21f_only", type=int, default=1,
                        help=("1 (default) = keep only 21F teleop samples in "
                              "EgoVerseControlDataset (backward-compat). "
                              "0 = load ALL samples including 7F internet ego "
                              "data (mixed 7F+21F training). Requires ZeRO-2 "
                              "and NUM_LATENT_FRAMES=0."))
    parser.add_argument("--num_inference_steps", type=int, default=1, help="Denoising steps per frame (1 for distillation)")
    parser.add_argument("--max_cache_frames", type=int, default=None,
                        help=("Sliding-window KV cache size. None = legacy "
                              "behavior (cache grows unbounded, RoPE baked at "
                              "absolute frame index). When set, enables "
                              "LongLive/StreamingLLM-style sliding mode: "
                              "RAW K stored, slot-based positions, frame-0 "
                              "sink always preserved. Recommended: 6 (= 1 "
                              "sink + 5 recent) for 21F training."))
    parser.add_argument("--pe_mode", type=str, default="absolute",
                        choices=["slot", "absolute"],
                        help=("Position-encoding scheme inside sliding mode. "
                              "Only used when --max_cache_frames is set. "
                              "'absolute' (default, matches LongLive / "
                              "Self-Forcing / Streaming_model_fixed reference "
                              "implementations) = K positions follow original "
                              "frame index in the source video; sink stays at "
                              "0, recent frames get their absolute idx (e.g. "
                              "for cap=6 at frame_idx=8, K positions = "
                              "[0,3,4,5,6,7,8]). Train-test PE distribution "
                              "becomes non-uniform but matches typical "
                              "inference where positions extend far past cap. "
                              "'slot' = StreamingLLM-style: cache K positions "
                              "are renumbered every forward to fit [0..cap]; "
                              "positions never exceed cap regardless of video "
                              "length (legacy path; kept for ablations)."))
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--real_guidance_scale", type=float, default=1.0,
                        help=("CFG scale for real_score (frozen teacher) in DMD. "
                              "Default 1.0 (effectively CFG OFF). "
                              "Wan2.2-TI2V-5B-Diffusers is CFG-distilled, so "
                              "feeding it any guidance_scale > 1 amplifies "
                              "real_pred extremes — measured |max| of "
                              "real_pred = uncond + scale*(cond-uncond) on a "
                              "clean teacher latent: scale=1→5.3, scale=3→9.3, "
                              "scale=5→14.4. Since DMD gradient is "
                              "(fake_pred - real_pred), large real_pred "
                              "extremes dominate the gradient signal and "
                              "cause the gen latents to collapse over time. "
                              "CausVid/Self-Forcing "
                              "use scale=5 because their teacher is NOT "
                              "CFG-distilled — copying that hyperparameter "
                              "for a CFG-distilled teacher is the bug."))
    parser.add_argument("--fake_guidance_scale", type=float, default=0.0)
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="",
        help=(
            "Negative prompt used for the CFG uncond direction when "
            "real_guidance_scale > 0. Default = '' preserves legacy behaviour "
            "(empty-string T5 embed). For Wan2.2 distillation the recommended "
            "value is the standard Wan rich neg prompt — exposed via the "
            "NEGATIVE_PROMPT env var. See "
            "Self-Forcing/configs/self_forcing_dmd.yaml and "
            "LongLive/configs/longlive_train_init.yaml for the reference text. "
            "Why this matters: with empty uncond, CFG `cond + g*(cond-empty)` "
            "amplifies whatever bias the cond embed already has (incl. base-"
            "model garish-color prior). With the rich Wan neg, CFG actively "
            "cancels those failure modes."
        ),
    )
    parser.add_argument("--dfake_gen_update_ratio", type=int, default=1, help="Train generator every N critic steps")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=20)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap on the DMD LatentDataset (controls how many video clips "
                             "are visible to the DMD post-warmup phase). Default None = use all. "
                             "NOTE: this does NOT affect the ODE pair dataset — see "
                             "--ode_max_samples for that. Earlier versions accidentally "
                             "applied this to ODE pairs too, which silently truncated "
                             "4000 cached pairs down to 100 (then sharded across 8 ranks "
                             "= 13 pairs/rank), causing severe overfit during warmup.")
    parser.add_argument("--ode_max_samples", type=int, default=None,
                        help="Cap on the ODE pair dataset, separate from --max_samples. "
                             "Default None = use ALL cached pairs. Only set this if you "
                             "explicitly want to test with a small subset of pairs (e.g. "
                             "for debugging). For real training, leave at None so the full "
                             "ode_pairs_dir is visible to ODE warmup.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--min_step_frac", type=float, default=0.02)
    parser.add_argument("--max_step_frac", type=float, default=0.98)
    # Override DeepSpeed's gradient_clipping (default 0.5 in ds_config_zero3.json).
    # Set to a smaller value (e.g. 0.1) to harden against the crash pattern seen
    # in the previous "fixed" branch where gen_gn explodes ~step 4000 then drags
    # loss with it. Paper's recipe uses 1.0; we default to None (= keep whatever
    # ds_config_zero3.json says) so behaviour is unchanged unless the launcher
    # asks for it explicitly. train_distill_8.sh now sets this to 0.1 by default.
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=None,
        help=("Override ds_config_zero3.json's gradient_clipping setting. "
              "Lower = safer against gradient explosion. None = keep ds_config "
              "default (0.5)."),
    )
    parser.add_argument("--smoke_test_layers", type=int, default=0,
                        help="If >0, shrink transformer to this many layers (random init) so the full "
                             "training pipeline can be exercised on tiny hardware. 0 = use the real 5B model.")
    parser.add_argument("--decode_every", type=int, default=0,
                        help="If >0, every N steps rank0 reloads the VAE, decodes one frame from the "
                             "no-grad generator output, and saves it under output_dir/debug_frames/. "
                             "0 disables (default). Useful for visually checking that training is converging.")
    parser.add_argument("--decode_model_path", type=str, default=None,
                        help="If --decode_every>0, where to load the VAE from. Defaults to --model_path.")
    parser.add_argument("--decode_dtype", type=str, default="fp32",
                        choices=["fp32", "bf16", "fp16"],
                        help=("Precision for the debug VAE decode. Default fp32 — Wan VAE's "
                              "3D causal decoder accumulates noticeable quantization noise "
                              "in bf16 (the GroupNorm + temporal upsample chain is sensitive "
                              "to ~3-bit mantissa rounding), so debug images can look 'noisier' "
                              "than the underlying latent really is. Use bf16/fp16 only if "
                              "decode VRAM is tight."))
    parser.add_argument("--decode_num_scenes", type=int, default=1,
                        help=("How many distinct ODE pairs (or DMD samples) to decode per "
                              "debug step. >1 produces a vertical strip per saved JPG, with "
                              "scenes stacked top-to-bottom. Default 1 = old behaviour. "
                              "Use 4 for a quick visual diversity / mode-collapse check."))
    parser.add_argument("--decode_with_teacher", action="store_true",
                        help=("During ODE warmup, also decode each scene's teacher x0 from "
                              "the cached pair file and save it on the row directly BELOW "
                              "the matching student row, so you can eyeball student-vs-teacher "
                              "frame-by-frame across the time axis. Cheap (~1s extra per "
                              "decode call). No-op when step >= ode_warmup_steps (DMD has no "
                              "per-step teacher target)."))
    parser.add_argument("--decode_max_frames", type=int, default=21,
                        help=("How many time-frames to show per scene-row in the debug grid. "
                              "Wan VAE produces 21 RGB frames per 6-latent-frame chunk. "
                              "Default 21 = show ALL frames (no temporal subsampling) so we "
                              "never confuse a frame-skip artifact with real model behaviour. "
                              "Row width = 21 × 640 + 20 × 2 = 13480 px — wide, but with the "
                              "downsample removed in the grid-write path the file is still <5 MB "
                              "@ JPEG q=95. Set lower (e.g. 8) only if you need a narrower "
                              "image at the cost of skipping intermediate frames."))

    # ── EMA (#5 from Self-Forcing recipe) ────────────────────────────────────
    # Self-Forcing maintains an EMA shadow of the generator with decay 0.99.
    # Pure DMD generators are noisy from step to step; the EMA gives a smoother
    # checkpoint for downstream inference. Implementation:
    #   - We mirror the fp32 *master* params (the same buffers DeepSpeedCPUAdam
    #     updates, on CPU under offload_optimizer.device=cpu). Each rank holds
    #     ONLY its own shard — no cross-rank communication for the per-step
    #     lerp, just CPU arithmetic ~tens of MB on each rank.
    #   - At save time we all-gather the shards (mimicking the path that
    #     gen_engine.save_16bit_model uses) and write `generator_ema.pt`
    #     alongside `generator.pt`.
    #   - 0.0 = disabled (skip the bookkeeping entirely so this is a true
    #     no-op for users who don't ask for it).
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.0,
        help=("Generator EMA decay (e.g. 0.99 from Self-Forcing). "
              "0.0 disables EMA. Only the GENERATOR gets an EMA — the critic "
              "and real_score don't need one (critic is consumed by gen step "
              "in real time; real_score is frozen)."),
    )
    parser.add_argument(
        "--ema_start_step",
        type=int,
        default=0,
        help=("Step at which EMA tracking begins (LongLive `ema_start_step` "
              "convention). Below this step the EMA shadow is None and "
              "_ema_update_zero3 is skipped; at step >= ema_start_step the "
              "shadow is lazily allocated from the current master weights "
              "(see EMA_FSDP._init_shadow + lazy create in LongLive "
              "trainer/distillation.py:1303-1309). Default 0 = track from "
              "iter 1 (legacy behaviour). For ode_warmup + DMD pipelines, set "
              "to ode_warmup_steps + 1 so the EMA shadow only averages the "
              "DMD-phase generator (the v_flow MSE warmup phase trains a "
              "different objective and shouldn't pollute the production EMA)."),
    )

    # ── ODE warmup (#6 from CausVid / Self-Forcing recipe) ───────────────────
    # The CausVid paper bootstraps DMD with a Stage-0 supervised step:
    #   1) Run the teacher (Wan2.2 base, 4-step) on N noise prompts → cache (noise, x0).
    #   2) Train the student via MSE(student_1step(noise), x0) for ~16k steps.
    #   3) Then switch to DMD.
    # Without it, DMD has to learn 1-step decoding from cold start, which
    # causes the first ~5k steps to look like noise and slows convergence
    # significantly (sometimes diverges entirely on small batch sizes).
    #
    # Pair generation lives in scripts/generate_ode_pairs.py (separate offline
    # job — does NOT need DMD setup, just the teacher pipeline). The generated
    # safetensors directory has shape:
    #     <ode_pairs_dir>/pair_000000.safetensors → {noise, x0_teacher,
    #         text_embeds, img_latent}
    # See that script for the exact schema.
    parser.add_argument(
        "--ode_warmup_steps",
        type=int,
        default=0,
        help=("If >0, train the generator for N steps on cached (noise, x0_teacher) "
              "MSE pairs BEFORE switching to DMD. The critic is frozen during "
              "warmup (no critic step taken). Self-Forcing uses 16k. 0 = skip "
              "(start DMD immediately, like train_distill_8.sh originally did)."),
    )
    parser.add_argument(
        "--ode_pairs_dir",
        type=str,
        default=None,
        help=("Directory of cached ODE pairs (safetensors). Required IF "
              "--ode_warmup_steps > 0 AND --ode_target_kind=teacher. "
              "Generate via scripts/generate_ode_pairs.py. Not used when "
              "--ode_target_kind=gt."),
    )
    # ODE warmup target source switch.
    # 'teacher' = original behavior (Self-Forcing/CausVid recipe):
    #             student MSE-trains against pre-cached x0_teacher pairs
    #             generated by scripts/generate_ode_pairs.py. Requires
    #             --ode_pairs_dir and ~23h of pair-gen wall-clock for 16k
    #             pairs.
    # 'gt'      = ablation: student MSE-trains directly against the
    #             GT video latent from --data_path (LatentDataset, the same
    #             data DMD itself uses). No pair gen needed; warmup data
    #             pool grows from 16k pairs → 64778 GT clips. Trade-off:
    #             critic still initializes from base teacher weights, so
    #             at the ODE→DMD switchover (step args.ode_warmup_steps+1)
    #             the critic's "real distribution" assumption (= teacher
    #             manifold) doesn't match the student's freshly-warmed
    #             state (= GT manifold) — DMD's first ~50 steps may show
    #             higher crit_loss than the teacher-warmup baseline.
    # Default 'teacher' so adding this flag does NOT change legacy behavior;
    # all existing launchers keep working with no edits.
    parser.add_argument(
        "--ode_target_kind",
        type=str,
        choices=["teacher", "gt", "v_flow", "causvid_traj"],
        default="teacher",
        help=("Source/form of the warmup loss. "
              "'teacher' (default) = MSE(x0_student_4step, x0_teacher_cached), "
              "needs --ode_pairs_dir (Self-Forcing/CausVid recipe). "
              "'gt' = MSE(x0_student_4step, gt_latent) from --data_path; "
              "no pair cache, but 4-step x0 reconstruction can overflow under "
              "cold-start attention (mitigated by 3-layer defense). "
              "'v_flow' = single-step v-space flow "
              "matching with streaming GT teacher-forcing KV cache. Loss = "
              "MSE(v_pred, noise-gt_latent). v-target is always O(1) so cold-"
              "start NaN risk is eliminated by construction. Cache write uses "
              "clean GT latents (teacher forcing); position protocol matches "
              "DMD streaming inference. Trade-off: warmup endpoint is 'epic_rdt-"
              "domain-adapted base Wan2.2', NOT a 4-step ODE solver, so DMD "
              "may need more steps to converge to streaming inference quality. "
              "'causvid_traj' = CausVid §4.3 "
              "ODE-regression adapted to streaming. Per-frame: sample idx ∈ [0,N), "
              "noisy=trajectory[idx], σ=traj_sigmas[idx], target=trajectory[-1] "
              "(=clean x_0). Streaming + cache + GT teacher-forcing prefix (same "
              "as v_flow), only (σ, noisy) input differs from v_flow's linear "
              "interp → clean ablation isolating noisy-form effect. Requires "
              "--ode_pairs_dir generated with --save_trajectory."),
    )
    # When set, reset the gen+critic LR schedulers at the
    # ODE→DMD switchover so DMD starts from lr=0 and re-runs its own warmup.
    # Reason: with ODE_WARMUP_STEPS=2000 + warmup_ratio=0.1 (= 1000 scheduler
    # warmup steps), the generator's lr already climbs to 50% of peak
    # during ODE training, then continues climbing into DMD. This means at
    # DMD's first step the generator gets a noisy gradient from a completely
    # cold critic AT a 50% lr — pushing it off the manifold ODE built.
    # Resetting gives DMD its own clean warmup starting from lr=0, during
    # which (a) lr stays small while critic is still cold, and (b) critic's
    # own scheduler also starts over so it ramps in lockstep with the
    # generator instead of ramping concurrently but offset.
    parser.add_argument(
        "--reset_lr_at_dmd",
        action="store_true",
        help=("If --ode_warmup_steps > 0, reset both LR schedulers' "
              "internal step counters at the ODE→DMD switchover so DMD "
              "starts from lr=0 and re-runs its own warmup_ratio fraction. "
              "Without this, scheduler steps taken during ODE 'count' "
              "toward warmup_ratio and DMD inherits a half-warmed lr."),
    )

    # ── Self-Forcing stochastic gradient truncation (paper §3.2 + Algo 1) ────
    # When --num_inference_steps>1, the default behaviour is to keep grad on
    # the entire T-step denoise chain of every frame — autograd graph holds
    # T × num_latent_frames forwards (e.g. 4 × 5 = 20 for H's recipe). This
    # is what the paper EXPLICITLY warns against: "naively backpropagating
    # through the entire autoregressive diffusion process would still lead
    # to excessive memory consumption" (§3.2). Their fix:
    #   1. Sample s ∈ [1, T] uniformly per video.
    #   2. Steps j != s run no_grad and advance via scheduler.step.
    #   3. Step j == s runs with grad, predicts x_0 from v_pred, and is
    #      used as the FINAL frame output (denoise loop breaks here).
    # This pulls the autograd graph back to ~1 forward/frame regardless of
    # T, AND ensures all T steps receive supervision signal across batches
    # (each step gets picked as `s` with probability 1/T).
    #
    # Default OFF for backward-compat. Enable with --stochastic_grad_truncation
    # (paired with --num_inference_steps=4) to actually run the paper's recipe.
    parser.add_argument(
        "--stochastic_grad_truncation",
        action="store_true",
        help=("Enable Self-Forcing stochastic gradient truncation in "
              "_streaming_generate. Sample s ∈ [1, num_inference_steps] "
              "per video; only the s-th denoise step carries grad; the "
              "frame's final output is the x_0 prediction at step s. "
              "Required for memory-tractable training when "
              "num_inference_steps>1 (e.g. paper-faithful 4-step student). "
              "No-op when num_inference_steps==1."),
    )

    # ── Three-segment streaming window (LongLive-inspired) ─────────────────
    # When --train_window_size K is set (with K < num_latent_frames), each
    # rollout is split into three segments to bound the autograd graph:
    #   [cond] [GT teacher-forcing far history] [SF buffer (K frames)] [training window (K)]
    # Only the K training-window frames carry gradient and feed into DMD loss.
    # Far history reuses cached x0_teacher from the ODE pair store (free).
    # Buffer is K frames of student no_grad rollout providing realistic
    # "dirty" near-history KV — required so the student learns error
    # correction rather than assuming a clean self-history at inference time.
    #
    # Memory: autograd graph drops from O(num_latent_frames × NIS) to
    # O(K × NIS). For our setup (21F × NIS=4 = 84 fwd → OOM on 80GB), this
    # gives K=7 × NIS=4 = 28 fwd ≈ 30 GB peak (fits comfortably).
    #
    # Wall time per iter: roughly proportional to (T_start + K + buf_size).
    # T_start is sampled per iter from {1, 1+K, 1+2K, ...} via rank-0 broadcast
    # (selection logic lives in the main DMD loop, not _streaming_generate).
    #
    # Default 0 (= disabled) preserves 100% backward-compat with the legacy path.
    parser.add_argument(
        "--train_window_size", type=int, default=0,
        help=("Three-segment streaming window. Size K of the gradient-bearing "
              "training window. "
              "0 (default) = disabled, all `num_latent_frames` get grad "
              "(legacy behaviour). When set (K>0), the rollout "
              "uses GT teacher-forcing for far history + K-frame student "
              "no_grad buffer + K-frame with_grad training window. "
              "Requires --ode_pairs_dir set (for GT source). Example: "
              "K=7 with num_latent_frames=21 gives T_start ∈ {1, 8, 14}."),
    )

    # ── Resume from checkpoint (explicit override of auto-scan) ──────────────
    # Default behaviour is to auto-scan --output_dir for the highest-numbered
    # valid `checkpoint-N/` subdir and resume from it (handles "job crashed,
    # re-launch with the same OUTPUT_DIR/RUN_TAG" automatically). This is
    # safe-by-design because every fresh launch normally gets its own
    # timestamped OUTPUT_DIR (see RUN_TAG logic in train_distill_8.sh) so a
    # new experiment never collides with an older run's checkpoints.
    #
    # --resume_from is an EXPLICIT override for cases where you want to
    # resume from a checkpoint that ISN'T inside the current --output_dir
    # (e.g. branching a new experiment from an older snapshot, or pinning
    # to a specific step rather than 'whatever's latest'). When set:
    #   - Path may be absolute or relative-to-output_dir.
    #   - Errors hard if the dir is missing or its weights look truncated.
    #   - Takes priority over the auto-scan.
    parser.add_argument(
        "--resume_from",
        type=str,
        default="",
        help=("Explicit path to a checkpoint dir to resume from "
              "(e.g. 'outputs/distill_C/checkpoint-2000' or an absolute path). "
              "Empty (default) = auto-scan --output_dir for the latest valid "
              "`checkpoint-N/` and resume from it (or start fresh if none). "
              "When set, this OVERRIDES the auto-scan."),
    )

    # ── Control-guided streaming (added for Stage 1 control distill) ──
    # When --use_control_dataset is set, dataset becomes EgoVerseControlDataset
    # which loads (video, control_video, img_latent) tuples from EgoVerse_cleaned.json.
    # Filter to 21F samples only (path-based) so all ranks see same frame count.
    parser.add_argument(
        "--use_control_dataset", action="store_true",
        help=("If set, replace LatentDataset with EgoVerseControlDataset "
              "(skeleton→hand format). Filters to 21F samples only. "
              "Required for control-distill training."),
    )
    parser.add_argument(
        "--teacher_ckpt", type=str, default=None,
        help=("Path to teacher checkpoint dir containing high_noise_lora/ + "
              "control_patch_embedding.bin (and optionally ema_weights.bin). "
              "Loaded into pipe.transformer via fuse-and-extract before generator init."),
    )
    parser.add_argument(
        "--use_ema_teacher", action="store_true",
        help=("Use teacher's ema_weights.bin instead of live LoRA weights."),
    )
    parser.add_argument(
        "--teacher_init_from_checkpoint", type=str, default=None,
        help=("PATH 4 (SFT + LoRA + CPE format) ONLY. Path to the SFT pretrain "
              "checkpoint dir that the LoRA was trained ON TOP OF (a directory "
              "with pytorch_model/ ZeRO shards). PATH 4 first "
              "loads SFT base from this dir (raw mode via deepspeed.utils."
              "zero_to_fp32), then fuses LoRA from --teacher_ckpt on top, then "
              "loads CPE + control_scale. Required when --teacher_ckpt points "
              "at a checkpoint that has control_scale.bin."),
    )
    parser.add_argument(
        "--teacher_control_type", type=str, default="add-plus",
        choices=["add", "add-plus"],
        help=("Skeleton conditioning mode for teacher (real_score) forward in "
              "DMD. 'add-plus' (default, MSE phase) normalizes skeleton latent "
              "to match video latent distribution via running_stats EMA. "
              "'add' (DMD phase with new LoRA teacher) uses RAW skeleton latent. "
              "Student always uses 'add-plus' regardless of this flag — only "
              "affects what skeleton input the teacher sees."),
    )
    parser.add_argument(
        "--critic_ckpt", type=str, default="",
        help=("Path to a critic.pt file to initialize the critic from. "
              "When set, overrides the critic.pt in --resume_from directory. "
              "Use to initialize critic from teacher/pretrained weights."),
    )

    args = parser.parse_args()

    # Validate mutually-required args. Done early so users don't waste the
    # 2-3 minute model-load before hitting a trivial config mistake.
    if args.ode_warmup_steps > 0:
        if (args.ode_target_kind in ("teacher", "causvid_traj")
                and not args.ode_pairs_dir):
            parser.error(
                f"--ode_warmup_steps > 0 with --ode_target_kind={args.ode_target_kind} "
                "requires --ode_pairs_dir (generate via "
                "scripts/generate_ode_pairs.py for teacher / "
                "scripts/generate_ode_pairs_traj.sh for causvid_traj), "
                "OR pass --ode_target_kind=gt (4-step x0 MSE on GT) or "
                "--ode_target_kind=v_flow (single-step v-MSE on GT, NaN-safe) "
                "to skip pair gen and warmup against GT video latents from "
                "--data_path instead."
            )
        # The 'gt' and 'v_flow' modes both pull targets from the
        # LatentDataset at --data_path; neither needs --ode_pairs_dir.
        # We still validate ode_pairs_dir if it was passed (in case a launcher
        # accidentally exports both env vars).
    if args.ode_pairs_dir and not os.path.isdir(args.ode_pairs_dir):
        parser.error(f"--ode_pairs_dir not found: {args.ode_pairs_dir}")
    if not (0.0 <= args.ema_decay < 1.0):
        parser.error(f"--ema_decay must be in [0, 1); got {args.ema_decay}")
    # train_window_size validation
    if args.train_window_size > 0:
        if args.train_window_size >= args.num_latent_frames:
            parser.error(
                f"--train_window_size={args.train_window_size} must be < "
                f"--num_latent_frames={args.num_latent_frames}"
            )
        if not args.ode_pairs_dir:
            parser.error(
                "--train_window_size > 0 requires --ode_pairs_dir "
                "to provide x0_teacher for GT far history. "
                "Generate via scripts/generate_ode_pairs.sh first."
            )

    # ── Initialize distributed + DeepSpeed manually ──
    # We can't use HuggingFace Accelerator with DeepSpeed for multiple models
    # (it asserts a single-model-per-Accelerator constraint). Instead we call
    # `deepspeed.initialize()` per model, giving each one its own ZeRO-3 engine.
    if not dist.is_initialized():
        deepspeed.init_distributed(dist_backend="nccl")

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    is_main = (rank == 0)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # ── Verbose environment dump (rank 0 only) ────────────────────────────────
    # 8-card cluster runs: when something goes wrong in the first 30 minutes the
    # only thing you have is the log. Dump everything that affects reproducibility
    # so you don't need to ssh back in and grep.
    if is_main:
        import platform, socket, subprocess, datetime
        cprint("=" * 78, "cyan")
        cprint(f"[env] start_time:     {datetime.datetime.now().isoformat()}", "cyan")
        cprint(f"[env] hostname:       {socket.gethostname()}", "cyan")
        cprint(f"[env] cwd:            {os.getcwd()}", "cyan")
        cprint(f"[env] python:         {platform.python_version()} ({sys.executable})", "cyan")
        cprint(f"[env] torch:          {torch.__version__} (cuda {torch.version.cuda})", "cyan")
        try:
            import deepspeed as _ds
            cprint(f"[env] deepspeed:      {_ds.__version__}", "cyan")
        except Exception as e:
            cprint(f"[env] deepspeed:      <import failed: {e}>", "red")
        try:
            import diffusers as _df
            cprint(f"[env] diffusers:      {_df.__version__}", "cyan")
        except Exception:
            pass
        try:
            import accelerate as _ac
            cprint(f"[env] accelerate:     {_ac.__version__}", "cyan")
        except Exception:
            pass
        try:
            import flash_attn as _fa
            cprint(f"[env] flash_attn:     {_fa.__version__}", "cyan")
        except Exception as e:
            cprint(f"[env] flash_attn:     <not importable: {e}>", "yellow")

        cprint(f"[env] world_size:     {world_size}", "cyan")
        cprint(f"[env] cuda devs:      {torch.cuda.device_count()} visible "
               f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')})", "cyan")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            cprint(f"[env]   gpu{i}: {props.name} ({props.total_memory / 1024**3:.1f} GB, "
                   f"sm_{props.major}{props.minor})", "cyan")

        # Host RAM
        try:
            with open("/proc/meminfo") as f:
                meminfo = {l.split(":")[0]: l.split(":")[1].strip() for l in f if ":" in l}
            cprint(f"[env] host RAM:       MemTotal={meminfo.get('MemTotal','?')}  "
                   f"MemAvailable={meminfo.get('MemAvailable','?')}  "
                   f"SwapTotal={meminfo.get('SwapTotal','?')}", "cyan")
        except Exception:
            pass

        # NCCL / distributed env
        for k in ("MASTER_ADDR", "MASTER_PORT", "NCCL_DEBUG", "NCCL_SOCKET_IFNAME",
                  "NCCL_IB_DISABLE", "NCCL_P2P_DISABLE", "TORCH_DISTRIBUTED_DEBUG"):
            v = os.environ.get(k)
            if v is not None:
                cprint(f"[env] {k}={v}", "cyan")

        cprint(f"[env] argv:           {' '.join(sys.argv)}", "cyan")

        # ── Hyperparameters dump (one-per-line, easy to grep / diff between runs) ──
        # Why split out from the old `[args] {...dict...}` one-liner:
        #   - 30+ args mashed into one line → hard to read in tail -f
        #   - `grep '^\[hparams\]' train.log` now gives a clean recipe of the run
        #   - `diff <(grep ^\[hparams\] runA.log) <(grep ^\[hparams\] runB.log)`
        #     is the single most useful command when reproducing a regression
        # Also persisted to <output_dir>/hparams.json for programmatic comparison
        # (e.g. wandb.config equivalent without needing wandb).
        cprint("=" * 78, "cyan")
        cprint("[hparams] === training configuration ===", "yellow")
        # Group order: training schedule first (most likely tuned), then DMD knobs,
        # then data/io (rarely changed). Within each group: alphabetical.
        _arg_groups = [
            ("schedule", ["num_train_steps", "batch_size", "gradient_accumulation_steps",
                          "learning_rate_gen", "learning_rate_critic", "weight_decay",
                          "adam_beta1", "adam_beta2", "adam_eps",
                          "warmup_ratio", "min_step_frac", "max_step_frac"]),
            ("dmd",      ["num_latent_frames", "num_inference_steps", "max_cache_frames",
                          "guidance_scale", "real_guidance_scale", "fake_guidance_scale",
                          "dfake_gen_update_ratio"]),
            ("ema_ode",  ["ema_decay", "ode_warmup_steps", "ode_pairs_dir",
                          "ode_max_samples", "reset_lr_at_dmd",
                          "stochastic_grad_truncation"]),
            ("io",       ["model_path", "data_path", "output_dir", "max_samples",
                          "logging_steps", "save_steps", "save_total_limit",
                          "decode_every", "decode_model_path", "decode_dtype",
                          "decode_num_scenes", "decode_with_teacher",
                          "decode_max_frames"]),
            ("misc",     ["seed", "bf16", "smoke_test_layers"]),
        ]
        _argd = vars(args)
        _printed = set()
        for group_name, keys in _arg_groups:
            cprint(f"[hparams] --- {group_name} ---", "yellow")
            for k in keys:
                if k in _argd:
                    cprint(f"[hparams]   {k} = {_argd[k]!r}", "yellow")
                    _printed.add(k)
        # Catch anything we forgot to assign to a group
        leftover = sorted(set(_argd) - _printed)
        if leftover:
            cprint("[hparams] --- other ---", "yellow")
            for k in leftover:
                cprint(f"[hparams]   {k} = {_argd[k]!r}", "yellow")
        cprint("=" * 78, "cyan")

        # Persist as JSON so a downstream script can read it without re-parsing logs.
        # Wrapped in try/except: if output_dir doesn't exist yet we don't want to
        # crash 30s into a 70-h training run.
        try:
            os.makedirs(args.output_dir, exist_ok=True)
            hparams_path = os.path.join(args.output_dir, "hparams.json")
            with open(hparams_path, "w") as _hf:
                json.dump(_argd, _hf, indent=2, default=str, sort_keys=True)
            cprint(f"[hparams] saved → {hparams_path}", "yellow")
        except Exception as _e:
            cprint(f"[hparams] WARNING: could not save hparams.json ({_e})", "red")

        # Sanity-check input paths up front so failures surface in the first
        # second of the log instead of 5 minutes in when we try to load.
        for p in (args.model_path, args.data_path):
            ok = os.path.exists(p)
            color = "green" if ok else "red"
            cprint(f"[paths] exists={ok}  {p}", color)
        cprint("=" * 78, "cyan")

    # All ranks: print a one-line "I'm alive" so the log shows every rank
    # actually entered main(). If one is missing, that rank failed to init NCCL.
    import socket as _sock
    _hostname = _sock.gethostname()
    print(f"[boot] rank={rank} local_rank={local_rank} world_size={world_size} "
          f"device={device} pid={os.getpid()} host={_hostname}", flush=True)

    # ── MULTI-NODE SANITY CHECK ─────────────────────────────────────────────
    # 2026-05-25: history bug — accelerate_config.yaml hardcoded num_machines=1
    # caused 2-node launches to silently degenerate into 2 independent 8-GPU jobs.
    # This block makes the failure mode IMPOSSIBLE TO MISS in the first 10 lines
    # of train.log.
    #
    # Cluster env vars (set by platform launcher, then re-exported by wrapper
    # under CLUSTER_* prefix BEFORE accelerate launch):
    #   CLUSTER_WORLD_SIZE      = number of MACHINES (not processes!)
    #   CLUSTER_NPROC_PER_NODE  = GPUs per machine
    #   CLUSTER_RANK            = current machine's rank ∈ [0, CLUSTER_WORLD_SIZE)
    # Why CLUSTER_* prefix: accelerate launch (via torchrun) resets WORLD_SIZE/
    # RANK/LOCAL_RANK in the child env to the GLOBAL process counts, shadowing
    # the cluster's "number of machines" semantics. Reading WORLD_SIZE here
    # would give us the global rank count (e.g. 32 for 4-node × 8GPU), making
    # `expected_total = WORLD_SIZE × NPROC_PER_NODE` produce nonsense like 256.
    # The wrapper saves cluster values under CLUSTER_* names that torchrun
    # leaves untouched, so this block reads the correct values.
    # Expected total ranks: CLUSTER_WORLD_SIZE × CLUSTER_NPROC_PER_NODE
    _env_ws = os.environ.get("CLUSTER_WORLD_SIZE")
    _env_np = os.environ.get("CLUSTER_NPROC_PER_NODE")
    _env_rk = os.environ.get("CLUSTER_RANK")
    _env_ma = os.environ.get("MASTER_ADDR")
    _env_mp = os.environ.get("MASTER_PORT")
    # Also peek at torchrun-set vars for diagnostics
    _torchrun_ws = os.environ.get("WORLD_SIZE")  # global rank count after launch
    _torchrun_rk = os.environ.get("RANK")        # global rank
    _torchrun_lrk = os.environ.get("LOCAL_RANK") # local rank within node
    _expected_total = None
    try:
        if _env_ws and _env_np:
            _expected_total = int(_env_ws) * int(_env_np)
    except Exception:
        pass

    # Per-rank info line — every rank prints (so master log + slave log
    # both show all their local ranks with their respective hostnames).
    print(
        f"[MULTINODE-CHECK] rank={rank}/{world_size} local={local_rank} "
        f"host={_hostname} | "
        f"cluster: WS(#nodes)={_env_ws} NPROC={_env_np} RANK(node_rank)={_env_rk} "
        f"MASTER={_env_ma}:{_env_mp} | "
        f"torchrun: WS={_torchrun_ws} RANK={_torchrun_rk} LOCAL={_torchrun_lrk} | "
        f"expected_total={_expected_total} actual_world_size={world_size}",
        flush=True,
    )

    # Per-node verdict (only local_rank 0 of each node prints, so we get
    # exactly N verdict lines for N nodes — easy to grep).
    if local_rank == 0:
        if _expected_total is None:
            print(
                f"[MULTINODE-CHECK] host={_hostname} VERDICT: ⚠️  cannot verify "
                f"(CLUSTER_WORLD_SIZE={_env_ws} CLUSTER_NPROC_PER_NODE={_env_np} — "
                f"wrapper should export CLUSTER_* vars before accelerate launch)",
                flush=True,
            )
        elif _expected_total == world_size:
            print(
                f"[MULTINODE-CHECK] host={_hostname} VERDICT: ✅ OK — "
                f"{_env_ws} nodes × {_env_np} GPUs/node = {_expected_total} "
                f"== actual world_size {world_size}",
                flush=True,
            )
        else:
            print(
                "\n" + "!" * 80 + "\n"
                f"[MULTINODE-CHECK] host={_hostname} VERDICT: ❌ BROKEN — "
                f"multi-node DID NOT initialize correctly!\n"
                f"[MULTINODE-CHECK]   expected total ranks: "
                f"CLUSTER_WORLD_SIZE({_env_ws}) × CLUSTER_NPROC_PER_NODE({_env_np}) "
                f"= {_expected_total}\n"
                f"[MULTINODE-CHECK]   actual world_size:    {world_size}\n"
                f"[MULTINODE-CHECK]   → this node is running INDEPENDENT job, not part of cluster.\n"
                f"[MULTINODE-CHECK]   → check accelerate yaml: num_machines must NOT be hardcoded.\n"
                f"[MULTINODE-CHECK]   → check wrapper: must pass --num_machines / --machine_rank / "
                f"--main_process_ip / --main_process_port via CLI.\n"
                + "!" * 80 + "\n",
                flush=True,
            )

    # Load DeepSpeed config and resolve "auto" placeholders we know.
    # 2026-05-25: DS_CONFIG_FILE env var overrides default ds_config_zero3.json.
    # Use case: training cluster has no libibverbs.so → NCCL forced to TCP →
    # ZeRO-3 cross-node all-gather of params dominates step time (60x more
    # cross-node traffic than ZeRO-2). On TCP-only clusters, set:
    #   DS_CONFIG_FILE=ds_config_zero2.json
    # → expected ~3x speedup vs ZeRO-3+TCP, at the cost of each rank holding
    # full 5B params (10 GB bf16) in memory (vs 313 MB sharded for ZeRO-3).
    _ds_config_filename = os.environ.get("DS_CONFIG_FILE", "ds_config_zero3.json")
    ds_config_path = os.path.join(os.path.dirname(__file__), _ds_config_filename)
    if not os.path.isfile(ds_config_path):
        raise FileNotFoundError(
            f"DeepSpeed config not found: {ds_config_path}. "
            f"Available: ds_config_zero3.json, ds_config_zero2.json. "
            f"Override via DS_CONFIG_FILE=<filename> env."
        )
    with open(ds_config_path, "r") as f:
        ds_config = json.load(f)
    ds_config["train_micro_batch_size_per_gpu"] = args.batch_size
    ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    ds_config["train_batch_size"] = args.batch_size * args.gradient_accumulation_steps * world_size
    ds_config["bf16"] = {"enabled": bool(args.bf16)}
    ds_config["fp16"] = {"enabled": False}
    # CLI override for gradient_clipping (so 8-GPU script can tighten to 0.1
    # without editing ds_config_zero3.json). When None, keep whatever the
    # config file already specifies.
    if args.max_grad_norm is not None:
        ds_config["gradient_clipping"] = float(args.max_grad_norm)
    if is_main:
        cprint(f"[ds] config path:        {ds_config_path}", "cyan")
        cprint(f"[ds] zero stage:         {ds_config['zero_optimization']['stage']}", "cyan")
        cprint(f"[ds] offload optimizer:  {ds_config['zero_optimization'].get('offload_optimizer')}", "cyan")
        cprint(f"[ds] offload param:      {ds_config['zero_optimization'].get('offload_param', 'NOT OFFLOADED')}", "cyan")
        cprint(f"[ds] train_batch_size:   {ds_config['train_batch_size']} "
               f"(= micro {args.batch_size} × accum {args.gradient_accumulation_steps} × world {world_size})", "cyan")
        cprint(f"[ds] precision:          bf16={args.bf16}", "cyan")
        cprint(f"[ds] gradient_clipping:  {ds_config.get('gradient_clipping')}"
               + ("  (CLI override)" if args.max_grad_norm is not None else "  (from ds_config json)"), "cyan")

    os.makedirs(args.output_dir, exist_ok=True)

    dtype = torch.bfloat16 if args.bf16 else torch.float32

    # Seed each process differently
    generator_rng = torch.Generator(device="cpu").manual_seed(args.seed + rank)

    if is_main:
        cprint(f"Loading base pipeline from {args.model_path} ...", "cyan")
    pipe = WanImageToVideoPipeline.from_pretrained(args.model_path, torch_dtype=dtype)

    # ── Teacher LoRA + CPE (control distill only) ──
    # Fuse teacher's LoRA delta into pipe.transformer in-place, BEFORE we
    # copy pipe.transformer.state_dict() into the freshly-built generator/
    # critic. This way the student starts from the teacher's domain-adapted
    # weights instead of bare Wan2.2 base. The returned tuple is
    #   (ctrl_pe_state, ctrl_scale_value)
    # ctrl_pe_state: dict with "weight" and "bias" keys (or None) — Conv3d
    #                state_dict to load into student CPE.
    # ctrl_scale_value: float (or None) — teacher's learned control_scale
    #                   (only present in PATH 4: SFT+LoRA+CPE format).
    ctrl_pe_state, ctrl_scale_value = (None, None)
    if args.teacher_ckpt:
        ctrl_pe_state, ctrl_scale_value = load_teacher_into_pipe(
            pipe, args.teacher_ckpt, args.use_ema_teacher, is_main,
            teacher_init_from_checkpoint=args.teacher_init_from_checkpoint,
        )

    # ── CPU-memory optimization: drop pipe components we never use ──
    # We keep three full 5B transformers (generator, critic, real_score) in CPU
    # memory before they get sharded by ZeRO-3. With N ranks per node each holding
    # a full copy, the working set quickly exceeds host RAM. Dropping unused
    # components (`transformer_2` ≈10 GB, `image_encoder`, and `vae` ≈1 GB which
    # is *never* referenced in this latent-space training loop) saves ~11–12 GB
    # per rank before we even build gen/critic. text_encoder is freed later,
    # after we've used it once to compute the negative-prompt embeds.
    if hasattr(pipe, "transformer_2") and pipe.transformer_2 is not None:
        del pipe.transformer_2
        pipe.transformer_2 = None
    if hasattr(pipe, "image_encoder") and pipe.image_encoder is not None:
        del pipe.image_encoder
        pipe.image_encoder = None
    if hasattr(pipe, "vae") and pipe.vae is not None:
        # VAE is only used at inference (decoding latents → RGB). This script
        # trains entirely in latent space — grep the file for "vae" / "VAE" and
        # you'll find zero references in the train loop. Free it.
        del pipe.vae
        pipe.vae = None
    import gc
    gc.collect()

    # In smoke-test mode, shrink the transformer config to a tiny number of layers
    # so the full ZeRO-3 + DMD + streaming pipeline can run on a single small GPU.
    # All three transformers (gen / critic / real_score) get the same shrunk config
    # with random init — this is for code-path validation only, not for quality.
    base_config = dict(pipe.transformer.config)
    if args.smoke_test_layers > 0:
        if is_main:
            cprint(f"[SMOKE TEST] Shrinking transformer to {args.smoke_test_layers} layers (random init).", "magenta")
        base_config["num_layers"] = args.smoke_test_layers

    # ── Generator ──
    # Do NOT use `assign=True`: we want generator to own its own parameter storage
    # so that when ZeRO-3 shards it, real_score (which aliases pipe.transformer)
    # is not corrupted. We immediately cast to bf16 BEFORE load_state_dict so the
    # transient memory footprint is 10 GB (bf16) per copy instead of 20 GB (fp32).
    # With transformer_2/image_encoder already dropped, peak per-rank CPU memory
    # is roughly: pipe.transformer (10) + T5 (9) + VAE (1) + generator (10) +
    # state_dict copy (10) ≈ 40 GB → 80 GB for 2 ranks, well within 200 GB.
    if is_main:
        cprint("Creating generator (WanCausalTransformer3DModel) ...", "cyan")
    generator = WanCausalTransformer3DModel.from_config(base_config).to(dtype)
    if args.smoke_test_layers == 0:
        incompatible = generator.load_state_dict(pipe.transformer.state_dict(), strict=False)
        if is_main:
            cprint(f"Generator load: {incompatible}", "yellow")
        gc.collect()
    # ── Control patch embedding (control distill only) ──
    # init_control_patch_embedding() doubles the in-channels of the student's
    # patch_embedding (96 = 48 video + 48 control) and zero-inits the control
    # half. If teacher's ctrl_pe_state is available, load it into the student's
    # CPE so the student starts from the teacher's control-fusion weights.
    if args.use_control_dataset:
        generator.init_control_patch_embedding()
        if ctrl_pe_state is not None:
            cpe_dtype = generator.control_patch_embedding.weight.dtype
            cpe_cast = {k: v.to(cpe_dtype) for k, v in ctrl_pe_state.items()}
            cinc = generator.control_patch_embedding.load_state_dict(cpe_cast, strict=True)
            if is_main:
                cprint(f"[gen] CPE loaded from teacher: missing={cinc.missing_keys}, "
                       f"unexpected={cinc.unexpected_keys}", "green")
        elif is_main:
            cprint("[gen] CPE kept at zero-init (no teacher CPE provided)", "yellow")
        # PATH 4: load teacher's learned control_scale (mirrors teacher inference).
        # NOTE: if --resume_from is provided, the resume step LATER overwrites
        # generator's full state_dict (including control_scale) — so this only
        # affects FRESH starts, not DMD resume from MSE ckpt. Resume keeps the
        # student's MSE-trained control_scale, which is the desired behavior.
        if ctrl_scale_value is not None and generator.control_scale is not None:
            with torch.no_grad():
                generator.control_scale.fill_(float(ctrl_scale_value))
            if is_main:
                cprint(f"[gen] control_scale init from teacher: "
                       f"{float(ctrl_scale_value):.6f} (will be overwritten if "
                       f"--resume_from sets generator state_dict)", "green")
    generator.train()
    generator.set_attention_backend("flash")
    # NOTE: Do NOT call `generator.enable_gradient_checkpointing()` here.
    # diffusers' built-in gradient_checkpointing wraps every block in
    # torch.utils.checkpoint, but our streaming forward mutates a
    # DynamicCache between frames (each frame's clean-K/V write appends to
    # the cache). When the checkpoint recompute runs at backward time, the
    # cache has grown beyond what the original forward saw, and recomputed
    # Q,K shapes don't match — `CheckpointError: Recomputed values ...
    # different metadata`.
    #
    # Activation checkpointing in this repo is implemented in
    # `_streaming_generate` instead, around the grad-bearing _model_forward
    # only, with cache snapshot/restore so the recompute sees identical
    # state. Enable via env STREAMING_GRAD_CHECKPOINT=1 (used by H3 to fit
    # 21F + NIS=4 + SGT=1 into 80 GB; ~40x activation savings, bit-identical).

    # ── Critic (fake_score) ──
    if is_main:
        cprint("Creating critic (WanCausalTransformer3DModel) ...", "cyan")
    critic = WanCausalTransformer3DModel.from_config(base_config).to(dtype)
    if args.smoke_test_layers == 0:
        incompatible = critic.load_state_dict(pipe.transformer.state_dict(), strict=False)
        if is_main:
            cprint(f"Critic load: {incompatible}", "yellow")
        gc.collect()
    # Mirror the generator's CPE setup for the critic. Stage 1 MSE warmup never
    # runs the critic forward, but we add CPE here so DMD phase (post-warmup)
    # works without re-running this script.
    if args.use_control_dataset:
        critic.init_control_patch_embedding()
        if ctrl_pe_state is not None:
            cpe_dtype = critic.control_patch_embedding.weight.dtype
            cpe_cast = {k: v.to(cpe_dtype) for k, v in ctrl_pe_state.items()}
            cinc = critic.control_patch_embedding.load_state_dict(cpe_cast, strict=True)
            if is_main:
                cprint(f"[critic] CPE loaded from teacher: missing={cinc.missing_keys}, "
                       f"unexpected={cinc.unexpected_keys}", "green")
        # Same control_scale init as generator (overwritten by --resume_from)
        if ctrl_scale_value is not None and critic.control_scale is not None:
            with torch.no_grad():
                critic.control_scale.fill_(float(ctrl_scale_value))
            if is_main:
                cprint(f"[critic] control_scale init from teacher: "
                       f"{float(ctrl_scale_value):.6f}", "green")
    critic.train()
    critic.set_attention_backend("flash")
    # Critic does NOT run through the streaming-cache forward path. It
    # consumes already-generated latents in a single non-streaming forward
    # (compute_critic_loss → critic(noisy, ...)). diffusers' built-in
    # gradient_checkpointing wraps each block with torch.utils.checkpoint
    # which is SAFE for critic (no streaming cache mutations) and saves
    # ~10x activation memory on the 21F single-shot forward.
    #
    # 2026-05-25 fix: H3 baseline OOMed at critic 21F forward (~93 GB / 95 GB
    # H20) because per-sample activation graph through 30 layers × 8190 token
    # seq is huge. With grad checkpointing it drops to ~5 GB → fits easily.
    # Enable via CRITIC_GRAD_CHECKPOINT env (default ON since DMD always
    # benefits; user can disable for backward-compat with H3 ablations).
    if int(os.environ.get("CRITIC_GRAD_CHECKPOINT", "1")):
        try:
            critic.enable_gradient_checkpointing()
            if is_main:
                cprint("[critic] enabled gradient_checkpointing (saves "
                       "~10x activation memory on 21F single-shot forward)",
                       "cyan")
        except Exception as _e:
            if is_main:
                cprint(f"[critic] WARN: enable_gradient_checkpointing failed: "
                       f"{_e} — proceeding without it. May OOM at 21F DMD.",
                       "yellow")

    # ── Real score (frozen teacher) ──
    # Wrap as WanCausalTransformer3DModel (NOT the diffusers base
    # WanTransformer3DModel) so the debug decode hook can run
    # `generate_streaming(real_score, ...)` — that path passes
    # `position_ids` / `attention_kwargs` keywords the base class rejects.
    # Weights are copied from pipe.transformer (strict=False is safe: the
    # causal subclass only adds an attention-processor wrapper, no new
    # learnable params). DMD uses real_score in a SINGLE non-streaming
    # forward (compute_dmd_generator_loss → real_score(noisy, ...))
    # without position_ids, which works on both classes — so this wrap
    # changes nothing for the DMD math but makes the streaming-debug
    # path callable.
    if is_main:
        cprint("Loading real score (frozen teacher) ...", "cyan")
    if args.smoke_test_layers > 0:
        # Smoke: same shrunk config as gen/critic, random init (no weights
        # copied because pipe.transformer is full-size and shape-mismatches
        # the shrunk config).
        real_score = WanCausalTransformer3DModel.from_config(base_config).to(dtype)
    else:
        real_score = WanCausalTransformer3DModel.from_config(base_config).to(dtype)
        incompatible = real_score.load_state_dict(
            pipe.transformer.state_dict(), strict=False)
        if is_main:
            cprint(f"Real score load: {incompatible}", "yellow")
        gc.collect()

    # ── Init CPE on real_score (DMD requires teacher with control conditioning) ──
    # 2026-05-25: previously only generator/critic got init_control_patch_embedding,
    # so real_score had no control_patch_embedding/control_scale → DMD teacher
    # forward couldn't use skeleton conditioning at all → real_score.noise_pred
    # was just generic Wan2.2 prediction, useless as DMD distillation target.
    # Fix: also init CPE on real_score and load teacher's learned weights.
    # Real_score is frozen below (requires_grad=False), so CPE+control_scale
    # are also frozen — they remain at teacher's loaded values throughout DMD.
    if args.use_control_dataset:
        real_score.init_control_patch_embedding()
        if ctrl_pe_state is not None:
            cpe_dtype = real_score.control_patch_embedding.weight.dtype
            cpe_cast = {k: v.to(cpe_dtype) for k, v in ctrl_pe_state.items()}
            cinc = real_score.control_patch_embedding.load_state_dict(cpe_cast, strict=True)
            if is_main:
                cprint(f"[real_score] CPE loaded from teacher: "
                       f"missing={cinc.missing_keys}, unexpected={cinc.unexpected_keys}",
                       "green")
        elif is_main:
            cprint("[real_score] WARN: CPE kept at zero-init (no teacher CPE provided) "
                   "— DMD distillation will be unconditional!", "red")
        if ctrl_scale_value is not None and real_score.control_scale is not None:
            with torch.no_grad():
                real_score.control_scale.fill_(float(ctrl_scale_value))
            if is_main:
                cprint(f"[real_score] control_scale loaded from teacher: "
                       f"{float(ctrl_scale_value):.6f}", "green")

    real_score.set_attention_backend("flash")
    real_score.eval()
    for p in real_score.parameters():
        p.requires_grad = False

    # Save references before deleting pipe
    scheduler = pipe.scheduler
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer

    # ── Override scheduler flow_shift ─────────────────────────────────────────
    # Wan2.2 ships scheduler with flow_shift=5.0 (heavy temporal shift). This
    # primary `scheduler` is for INFERENCE paths (4-step student rollout,
    # 50-step UniPC teacher decode) AND DMD generator unroll (student WITH
    # grad). Its `timesteps` array gets rewritten by `set_timesteps(N)` at
    # every inference call.
    #
    # The flow_shift VALUE matters because it controls the 4-step
    # `denoising_step_list` schedule for student rollout:
    #   - flow_shift=1.0 → [999, 749, 499, 249]  (σ ends at 0.25)
    #   - flow_shift=5.0 → [1000, 937, 833, 625] (σ ends at 0.625, LongLive)
    #
    # LongLive's recipe uses flow_shift=5.0 (`utils/wan_wrapper.py:175,197-200`
    # and config L9-13 with `warp_denoising_step:true` mapping the raw
    # [1000,750,500,250] schedule through shift=5). The terminal σ at step 4
    # is 0.625 — much closer to the noise manifold the critic scores at.
    # Earlier fork choice was 1.0 (terminal σ=0.25, more aggressive denoise);
    # this is now opt-in via INFERENCE_FLOW_SHIFT env var for backward compat
    # with the Run A/B/C ablations.
    #
    # For v_flow MSE warmup TRAINING σ, we keep an INDEPENDENT scheduler copy
    # below (`v_flow_scheduler`) whose `timesteps` array stays stable at
    # set_timesteps(num_train_timesteps). MSE phase σ controlled by FLOW_SHIFT
    # env var (independent from this inference shift).
    #
    # INFERENCE_FLOW_SHIFT default = 1.0 keeps backward-compat with existing
    # ablations (Run A/B/C). New LongLive-aligned wrappers should explicitly
    # export INFERENCE_FLOW_SHIFT=5.0.
    _inference_flow_shift = float(os.environ.get("INFERENCE_FLOW_SHIFT", "1.0"))
    _sched_cfg = dict(scheduler.config)
    _sched_cfg["flow_shift"] = _inference_flow_shift
    scheduler = type(scheduler).from_config(_sched_cfg)
    print(f"[scheduler] inference scheduler flow_shift -> {scheduler.config.flow_shift}"
          + (" (LongLive-aligned)" if _inference_flow_shift == 5.0 else ""))

    # ── v_flow training scheduler (independent copy, decoupled from inference) ─
    # SF-style σ-shift mechanism: σ is sampled by `index = randint([0, T))`
    # then `t = v_flow_scheduler.timesteps[index]`, where `timesteps` was
    # populated by a one-time `set_timesteps(num_train_timesteps)` under
    # this scheduler's flow_shift. Equivalent to Self-Forcing
    # `model/diffusion.py:78` (`timestep = self.scheduler.timesteps[index]`).
    #
    # FLOW_SHIFT env var (default 1.0) controls this copy's flow_shift:
    #   - FLOW_SHIFT=1.0 (default) → linear σ-t (statistically equivalent to
    #     baseline `randint(0,T)` direct-divide, just iterated in reverse t
    #     order). Backward-compat for runs that don't opt in.
    #   - FLOW_SHIFT=5.0/8.0 → σ pushed toward mid-high band (LongLive / SF /
    #     CausVid recipe).
    # The inference `scheduler` above is unaffected; only v_flow MSE warmup
    # forward consumes this copy.
    _v_flow_sched_cfg = dict(_sched_cfg)
    _v_flow_sched_cfg["flow_shift"] = float(os.environ.get("FLOW_SHIFT", "1.0"))
    v_flow_scheduler = type(scheduler).from_config(_v_flow_sched_cfg)
    _n_train_t = int(_v_flow_sched_cfg.get("num_train_timesteps", 1000))
    v_flow_scheduler.set_timesteps(_n_train_t, device="cpu")
    print(
        f"[v_flow_scheduler] flow_shift={v_flow_scheduler.config.flow_shift} "
        f"num_train_timesteps={_n_train_t} | timesteps[100]={v_flow_scheduler.timesteps[100].item():.0f} "
        f"timesteps[500]={v_flow_scheduler.timesteps[500].item():.0f} "
        f"timesteps[900]={v_flow_scheduler.timesteps[900].item():.0f}"
    )

    # Free pipeline memory to reduce VRAM. real_score now holds its own
    # WanCausalTransformer3DModel copy of the weights (state_dict was
    # copied into the wrapped subclass above), so we can drop pipe.transformer.
    pipe.transformer = None
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    # Real_score (frozen teacher) is NOT in ZeRO-3 (it aliases pipe.transformer
    # and we just set requires_grad=False; DeepSpeed wouldn't shard it anyway
    # without an engine wrapper). Keeping it permanently on GPU costs ~10 GB/
    # rank of VRAM that competes with generator/critic activations.
    #
    # We instead keep it on CPU long-term and `to(device)` only inside the DMD
    # loss block (compute_dmd_generator_loss does `with torch.no_grad()` for
    # the real_score forward, so the swap doesn't allocate any activation
    # tape). Math is identical — moving a frozen no_grad model between devices
    # is bit-exact. Cost: ~0.5–1 s of PCIe gen5 transfer per gen step
    # (acceptable; gen step is on the order of seconds anyway).
    real_score = real_score.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()

    # Move T5 text_encoder to GPU, precompute the (single, fixed) negative-prompt
    # embeds, then FREE T5 entirely. The dataset's safetensors samples already
    # carry precomputed `text_embeds` (verified: 64778/64778 samples in epic_rdt
    # have it), so T5 is only ever used here for the empty-string negative
    # prompt. Freeing it saves ~9 GB / rank of VRAM that would otherwise sit
    # idle for the entire training run.
    text_encoder = text_encoder.to(device)
    # CRITICAL: max_length MUST match the preprocess script (utils/pre-process-egovid.py
    # uses max_sequence_length=226). Wan transformer cross-attention does NOT take an
    # attention_mask, so the padding tokens at positions [actual_len ... max_length]
    # all participate in attention. If conditional embeds are length 226 (from cache)
    # but unconditional embeds are length 512, the CFG subtraction (cond - uncond)
    # mixes a 226-token attention pool with a 512-token attention pool — these are
    # not comparable, so the resulting "guidance direction" is partly noise. Using
    # 226 here keeps both branches mathematically aligned. Diffusers' official
    # WanImageToVideoPipeline.encode_prompt also defaults to 226.
    #
    # 2026-05-26 fix (parity with train_distill_e3_fork.py): honor
    # --negative_prompt CLI arg. Default "" matches legacy/buggy behavior
    # (empty uncond → CFG amplifies cond bias instead of cancelling failure
    # modes). For Wan2.2 DMD, pass the SF/LongLive Wan rich neg via env
    # NEGATIVE_PROMPT (wrapper plumbs through --negative_prompt).
    _neg_prompt_text = getattr(args, "negative_prompt", "") or ""
    if is_main:
        if _neg_prompt_text:
            _preview = _neg_prompt_text[:40].replace("\n", " ")
            cprint(
                f"[neg_prompt] using NON-EMPTY negative_prompt "
                f"(len={len(_neg_prompt_text)} chars, preview='{_preview}...')",
                "green",
            )
        else:
            cprint(
                "[neg_prompt] WARNING: negative_prompt is empty — CFG uncond "
                "branch will use empty T5 embed. If REAL_GUIDANCE_SCALE>0 this "
                "AMPLIFIES cond bias (色调艳丽 etc.) instead of cancelling it. "
                "For Wan2.2 DMD set NEGATIVE_PROMPT='色调艳丽,过曝,...' (Wan rich neg).",
                "yellow" if args.real_guidance_scale > 0 else "cyan",
            )
    _neg_text_inputs = tokenizer(
        _neg_prompt_text,
        padding="max_length",
        max_length=226,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        negative_embeds = text_encoder(
            _neg_text_inputs.input_ids.to(device),
            _neg_text_inputs.attention_mask.to(device),
        ).last_hidden_state.to(dtype=dtype)
    # Free the encoder. After this point any code path that calls
    # `encode_prompt(...)` will hit the guard inside that helper (defined below).
    del text_encoder
    text_encoder = None
    gc.collect()
    torch.cuda.empty_cache()

    # ── Optimizers ──
    # ZeRO-3 with CPU offload requires DeepSpeedCPUAdam (otherwise DeepSpeed
    # raises ZeRORuntimeException). It runs the optimizer math on CPU and is
    # the intended pairing with offload_optimizer.device=cpu.
    #
    # Beta/eps come from argparse (default: Self-Forcing DMD recipe
    # β=(0, 0.999), eps=1e-8). AdamW weight_decay is common to both networks.
    # See Self Forcing (Huang et al. 2025) Table 3.
    from deepspeed.ops.adam import DeepSpeedCPUAdam

    _adam_betas = (args.adam_beta1, args.adam_beta2)
    gen_optimizer = DeepSpeedCPUAdam(
        [p for p in generator.parameters() if p.requires_grad],
        lr=args.learning_rate_gen,
        betas=_adam_betas,
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )
    critic_optimizer = DeepSpeedCPUAdam(
        [p for p in critic.parameters() if p.requires_grad],
        lr=args.learning_rate_critic,
        betas=_adam_betas,
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )

    # ── LR schedulers ──
    # Original code parsed --warmup_ratio but never built a scheduler, so the LR
    # was constant throughout training. Use a linear-warmup-then-constant schedule
    # (common for DMD distillation) — switch to cosine if you want decay.
    #
    # 2026-06-22 fix: scheduler.step() is called once per engine.step() (GAS
    # boundary), NOT once per inner step. So gen_scheduler advances every
    # ratio*GAS inner steps and critic_scheduler advances every GAS inner steps.
    # Previously `num_warmup_steps = warmup_ratio * num_train_steps` was the
    # same constant for both, but interpreted as scheduler-steps on each — which
    # silently inflated gen warmup by ratio*GAS (e.g. 0.05 * 14000 = 700 →
    # 14000 inner steps for gen with ratio=5, GAS=4, i.e. the entire run).
    # Fix: compute per-scheduler warmup so warmup_ratio means "fraction of inner
    # training steps" (the intuitive meaning).
    _gas = int(args.gradient_accumulation_steps)
    _ratio = int(getattr(args, "dfake_gen_update_ratio", 1) or 1)
    _total_warmup_inner = int(args.warmup_ratio * args.num_train_steps)
    num_warmup_steps_gen = max(0, _total_warmup_inner // (_gas * _ratio))
    num_warmup_steps_critic = max(0, _total_warmup_inner // _gas)

    def _make_warmup_lambda(num_warmup):
        def _warmup_lr_lambda(current_step: int):
            if current_step < num_warmup and num_warmup > 0:
                return float(current_step) / float(max(1, num_warmup))
            return 1.0
        return _warmup_lr_lambda

    gen_scheduler = torch.optim.lr_scheduler.LambdaLR(gen_optimizer, _make_warmup_lambda(num_warmup_steps_gen))
    critic_scheduler = torch.optim.lr_scheduler.LambdaLR(critic_optimizer, _make_warmup_lambda(num_warmup_steps_critic))
    if is_main:
        cprint(f"[lr-sched] warmup: total_inner={_total_warmup_inner} | "
               f"gen_scheduler_steps={num_warmup_steps_gen} (advances every {_gas*_ratio} inner) | "
               f"critic_scheduler_steps={num_warmup_steps_critic} (advances every {_gas} inner)", "cyan")

    # ── Dataset (shard across processes) ──
    if is_main:
        cprint(f"Loading dataset from {args.data_path} ...", "cyan")
    # Treat 0 as "no cap" so the launcher's MAX_SAMPLES=0 default uses the
    # whole dataset. Anything > 0 caps as before. Catches the foot-gun where
    # the launcher's default was 100, which sharded across 8 ranks left only
    # 12-13 clips per rank and mode-collapsed B+ 方案's warmup.
    _cap = args.max_samples if (args.max_samples and args.max_samples > 0) else None
    if args.use_control_dataset:
        dataset = EgoVerseControlDataset(
            data_path=args.data_path,
            tokenizer=tokenizer,
            max_samples=_cap,
            filter_21f_only=args.filter_21f_only,
        )
        if is_main:
            cprint(f"[ctrl_dataset] Using EgoVerseControlDataset, "
                   f"filter_21f_only={args.filter_21f_only}: "
                   f"{len(dataset)} samples", "cyan")
    else:
        dataset = LatentDataset(
            data_path=args.data_path,
            tokenizer=tokenizer,
            max_samples=_cap,
        )

    # Shard dataset across processes
    dataset.samples = dataset.samples[rank::world_size]
    if is_main:
        cprint(f"Dataset size (this rank): {len(dataset)}", "green")
        # Fail-loud guard for the B/B+ 方案 mode-collapse trap: when warmup
        # uses LatentDataset directly (--ode_target_kind gt|v_flow), too few
        # samples per rank overfits within a few hundred steps and decodes
        # become solid-colour noise. 1000 samples gives ~125/rank on 8 ranks,
        # below that is almost certainly an unintended MAX_SAMPLES override.
        _warmup_kind = getattr(args, "ode_target_kind", "teacher")
        _cap_was_set = args.max_samples and args.max_samples > 0
        if (_cap_was_set and args.max_samples < 1000
                and _warmup_kind in ("gt", "v_flow")
                and args.ode_warmup_steps > 0):
            cprint(
                f"[FATAL] --max_samples={args.max_samples} with "
                f"--ode_target_kind={_warmup_kind} will overfit the warmup. "
                f"On 8 ranks each gets {args.max_samples // world_size} clips; "
                f"warmup mode-collapses to noise within ~500 steps. Set "
                f"MAX_SAMPLES=0 (full dataset) or >= 1000 for real runs.",
                "red")
            raise SystemExit(
                f"Refusing to launch: max_samples={args.max_samples} too small "
                f"for {_warmup_kind} warmup. Pass MAX_SAMPLES=0 (= use all) "
                f"or MAX_SAMPLES>=1000 explicitly to override.")

    # ── Check for resume checkpoint ──
    # Two ways to resume, in priority order:
    #   1) `--resume_from <path>` explicit override.
    #      Path may be absolute or relative-to-output_dir. Errors hard if
    #      it doesn't exist or its generator.pt/critic.pt look truncated.
    #      Use this when you want to resume a specific old ckpt that ISN'T
    #      in args.output_dir (e.g. branching a new experiment from an
    #      older snapshot).
    #   2) Otherwise auto-scan args.output_dir for `checkpoint-N` subdirs
    #      and pick the highest valid one. This is the OG behaviour and
    #      handles the common case of "job crashed, re-launch with the
    #      same OUTPUT_DIR/RUN_TAG to pick up where we left off". Safe by
    #      design because every fresh launch gets a NEW timestamped
    #      OUTPUT_DIR (see RUN_TAG in train_distill_8.sh) so a fresh
    #      experiment never collides with an older run's checkpoints.
    #
    # Sort NUMERICALLY (default `sorted()` would put 'checkpoint-500' AFTER
    # 'checkpoint-2500' because of lexicographic compare on '5' vs '2').
    # Also skip checkpoints whose generator.pt / critic.pt are missing or
    # truncated — happened on 2026-05-06 when FUSE rejected a 73 GB concurrent
    # critic-shard write at step 2500 and left an incomplete dir behind.
    def _ckpt_step(d: str) -> int:
        # 2026-05-27 fix: regex-extract step number after "checkpoint-".
        # Old logic `int(d.split("-")[-1])` broke on suffixed names like
        # `checkpoint-2500_scale05_boost` (manually-edited ckpt for ablation)
        # → returned -1 → resume_step=0 → re-ran MSE warmup wasting cycles.
        import re as _re
        m = _re.search(r"checkpoint-(\d+)", d)
        if m:
            return int(m.group(1))
        # Fallback: try old behavior (last segment after "-" is integer)
        try:
            return int(d.split("-")[-1])
        except ValueError:
            return -1

    def _ckpt_complete_abs(d_abs: str) -> bool:
        # d_abs is an ABSOLUTE path to a checkpoint dir.
        # Both per-engine merged bf16 files must be present and big (>1 GB).
        # We use these (not the ZeRO sharded dirs) for resume below.
        for fname in ("generator.pt", "critic.pt"):
            fp = os.path.join(d_abs, fname)
            if not os.path.isfile(fp):
                return False
            if os.path.getsize(fp) < (1 << 30):  # 1 GB sanity floor
                return False
        return True

    def _ckpt_complete(d: str) -> bool:
        # d is a basename inside args.output_dir (back-compat with original
        # signature so the prune-loop code path further down still works).
        return _ckpt_complete_abs(os.path.join(args.output_dir, d))

    resume_from_ckpt = None  # the basename, e.g. 'checkpoint-2000'
    ckpt_path = None         # the absolute path

    if args.resume_from:
        # ── Path 1: explicit override ──
        if os.path.isabs(args.resume_from):
            ckpt_path = args.resume_from
        else:
            ckpt_path = os.path.join(args.output_dir, args.resume_from)
        ckpt_path = os.path.normpath(ckpt_path)
        resume_from_ckpt = os.path.basename(ckpt_path.rstrip("/"))

        if not os.path.isdir(ckpt_path):
            raise FileNotFoundError(
                f"--resume_from points at a non-existent dir: {ckpt_path}\n"
                f"  (resolved from --resume_from='{args.resume_from}', "
                f"--output_dir='{args.output_dir}')"
            )
        if not _ckpt_complete_abs(ckpt_path):
            raise RuntimeError(
                f"--resume_from={ckpt_path} is missing generator.pt and/or "
                f"critic.pt (or they are <1 GB → likely truncated). Refusing "
                f"to silently start fresh; either point at a valid checkpoint "
                f"or omit --resume_from."
            )
        if _ckpt_step(resume_from_ckpt) < 0 and is_main:
            cprint(f"[resume] WARN: --resume_from basename '{resume_from_ckpt}' "
                   f"doesn't match 'checkpoint-N' pattern; resume_step will be 0 "
                   f"and global step counter restarts from 0. If you want the "
                   f"step counter to continue, rename the dir to 'checkpoint-N'.",
                   "yellow")
        if is_main:
            cprint(f"[resume] EXPLICIT --resume_from: {ckpt_path}", "cyan")
    else:
        # ── Path 2: auto-scan args.output_dir (OG behaviour) ──
        all_ckpts = sorted(
            [d for d in os.listdir(args.output_dir)
             if d.startswith("checkpoint-") and _ckpt_step(d) >= 0],
            key=_ckpt_step,
        )
        valid_ckpts = [d for d in all_ckpts if _ckpt_complete(d)]
        skipped = [d for d in all_ckpts if d not in valid_ckpts]
        if skipped and is_main:
            cprint(f"[resume] Skipping incomplete checkpoint(s): {skipped}", "yellow")

        resume_from_ckpt = valid_ckpts[-1] if valid_ckpts else None
        if resume_from_ckpt:
            ckpt_path = os.path.join(args.output_dir, resume_from_ckpt)
            if is_main:
                cprint(f"[resume] Auto-detected ckpt in {args.output_dir}: "
                       f"{resume_from_ckpt}", "cyan")
        else:
            if is_main:
                cprint(f"[resume] No checkpoint found in {args.output_dir}; "
                       f"starting fresh from step 0.", "cyan")

    resume_step = 0

    if resume_from_ckpt:
        if is_main:
            cprint(f"Will resume from {ckpt_path} ...", "cyan")

        # ── Stage ckpt to local NVMe before reading ──
        # Why: ckpt_path may live on a network filesystem (e.g. OSS-FUSE). If
        # all 8 ranks each call torch.load() on the same 12 GB file, we get 8
        # concurrent streams competing for bandwidth on a single object,
        # which throttles per rank → resume can take 30+ min.
        #
        # Fix: rank 0 copies the two .pt files ONCE to a local NVMe staging
        # dir (~100 MB/s sequential OSS read, single stream, ~4 min for 24GB),
        # then all ranks read from local NVMe in parallel (~1 GB/s each, done
        # in seconds). Net: ~30 min → ~5 min, and ranks don't fight on OSS.
        #
        # Idempotent: skip the copy if local file already matches src size,
        # so re-launching after a crash that already staged is instant.
        stage_root = os.environ.get("CKPT_STAGE_DIR", "/tmp/_ckpt_stage")
        stage_ckpt_dir = os.path.join(stage_root, resume_from_ckpt)
        # ──────────────────────────────────────────────────────────────────
        # Multi-node staging: each node's LOCAL rank-0 (local_rank==0) stages
        # to that node's local NVMe. All nodes stage in parallel — Nx more concurrent
        # reads BUT each reads to DIFFERENT physical disks, so OSS is the only
        # bottleneck and total wall time is similar to single-node staging
        # (still ~5 min for 36 GB on each node).
        #
        # Failure handling: if ANY node's local-rank-0 stage fails, ALL ranks
        # fall back to direct OSS read (via all_reduce of failure bool). This
        # keeps `read_dir` consistent across ranks (critical — different paths
        # would cause ckpt load skew → ZeRO desync → silent corruption).
        # ──────────────────────────────────────────────────────────────────
        _stage_failed_local = False
        if local_rank == 0:
            try:
                os.makedirs(stage_ckpt_dir, exist_ok=True)
                # 2026-05-26 fix: also stage control_running_stats.bin (tiny —
                # ~1.7 KB) and generator_ema.pt (12.3 GB). Previously only
                # generator.pt + critic.pt were copied, which (a) silently lost
                # MSE-trained skel-norm EMA stats at DMD start (CPE saw OOD
                # input for 50-200 steps), and (b) made EMA-resume hit OSS
                # directly for the 12 GB shadow.
                _stage_files = ("generator.pt", "critic.pt",
                                "generator_ema.pt", "control_running_stats.bin")
                for fname in _stage_files:
                    src = os.path.join(ckpt_path, fname)
                    if not os.path.exists(src):
                        # Optional files (generator_ema.pt missing on EMA-off
                        # ckpts; control_running_stats.bin missing on pre-2026-05
                        # ckpts). Skip silently — they're not always present.
                        continue
                    dst = os.path.join(stage_ckpt_dir, fname)
                    src_sz = os.path.getsize(src)
                    if os.path.isfile(dst) and os.path.getsize(dst) == src_sz:
                        if is_main:  # only global rank 0 logs to avoid 4x spam
                            cprint(f"[resume] stage cache hit: {dst} ({src_sz/1e9:.2f} GB)", "green")
                        continue
                    if is_main:
                        cprint(f"[resume] staging {fname} OSS→local ({src_sz/1e9:.2f} GB) "
                               f"(each node's local_rank=0 stages in parallel) ...", "cyan")
                        sys.stdout.flush()
                    t0 = time.time()
                    shutil.copyfile(src, dst)
                    dt = time.time() - t0
                    if is_main:
                        cprint(f"[resume] staged in {dt:.1f}s ({src_sz/1e6/max(dt,1e-3):.0f} MB/s) → {dst}", "green")
                        sys.stdout.flush()
            except Exception as e:
                # Don't print rank prefix on is_main only — every failing node
                # should report so we can see which node had the issue.
                cprint(f"[resume] WARN: rank {rank} (node-local stager) failed "
                       f"({e}); will fall back to direct OSS read", "yellow")
                _stage_failed_local = True

        # Optimistic read_dir; will be overridden below if any node failed.
        read_dir = stage_ckpt_dir

        # All-reduce SUM of failure flag across ALL ranks. Non-staging ranks
        # contribute 0 (their _stage_failed_local stays False). If sum > 0,
        # at least one local-rank-0 failed — fall back to OSS path globally.
        if dist.is_initialized():
            _flag = torch.tensor(
                [1 if _stage_failed_local else 0],
                dtype=torch.long,
                device=device,  # NCCL needs GPU tensor
            )
            dist.all_reduce(_flag, op=dist.ReduceOp.SUM)
            _num_failed = int(_flag.item())
            if _num_failed > 0:
                if is_main:
                    cprint(
                        f"[resume] WARN: {_num_failed} rank(s) failed to stage; "
                        f"ALL ranks falling back to direct OSS read at {ckpt_path}",
                        "yellow",
                    )
                read_dir = ckpt_path
            # Barrier ensures all local stagers finished before anyone reads.
            dist.barrier()
        elif _stage_failed_local:
            # Single-process fallback (no dist) — just use OSS.
            read_dir = ckpt_path

        if is_main:
            cprint(f"[resume] All ranks loading state_dict from {read_dir}", "cyan")
            sys.stdout.flush()

        gen_state = torch.load(os.path.join(read_dir, "generator.pt"), map_location="cpu", weights_only=True)
        if args.critic_ckpt:
            _critic_path = args.critic_ckpt
            if is_main:
                cprint(f"[resume] Loading critic from --critic_ckpt: {_critic_path}", "cyan")
        else:
            _critic_path = os.path.join(read_dir, "critic.pt")
        crit_state = torch.load(_critic_path, map_location="cpu", weights_only=True)

        # Load model states BEFORE prepare (DeepSpeed ZeRO-3 needs this)
        gen_inc = generator.load_state_dict(gen_state, strict=False)
        if is_main and (gen_inc.missing_keys or gen_inc.unexpected_keys):
            cprint(f"[resume] generator load: missing={len(gen_inc.missing_keys)} "
                   f"unexpected={len(gen_inc.unexpected_keys)}", "yellow")
        crit_inc = critic.load_state_dict(crit_state, strict=False)
        # Validate: only allow known-safe missing keys (action_proj, CPE, control_scale).
        # These are modules that the teacher may not have but our model structure adds;
        # they get properly initialized later (CPE from generator, action_proj zero-init).
        _allowed_missing = {"action_proj", "control_patch_embedding", "control_scale"}
        _unexpected_missing = [
            k for k in crit_inc.missing_keys
            if not any(pat in k for pat in _allowed_missing)
        ]
        if _unexpected_missing:
            raise RuntimeError(
                f"[resume] critic.pt is missing {len(_unexpected_missing)} UNEXPECTED keys "
                f"(not in allowlist {_allowed_missing}). First 5: {_unexpected_missing[:5]}. "
                f"This likely means the wrong file was used as critic.pt."
            )
        if crit_inc.unexpected_keys:
            _allowed_unexpected = {"action_proj"}
            _real_unexpected = [
                k for k in crit_inc.unexpected_keys
                if not any(pat in k for pat in _allowed_unexpected)
            ]
            if _real_unexpected:
                raise RuntimeError(
                    f"[resume] critic.pt has {len(_real_unexpected)} UNEXPECTED extra keys. "
                    f"First 5: {_real_unexpected[:5]}. "
                    f"This likely means the wrong file was used as critic.pt."
                )
            elif is_main:
                cprint(f"[resume] critic load: {len(crit_inc.unexpected_keys)} ignored "
                       f"unexpected keys (action_proj — not in new model)", "yellow")
        if is_main and crit_inc.missing_keys:
            cprint(f"[resume] critic load: {len(crit_inc.missing_keys)} expected-missing keys "
                   f"(action_proj/CPE/control_scale — will be initialized later)", "yellow")
        # Free the 24 GB CPU buffers immediately. Without this, both copies
        # stay alive in scope through deepspeed.initialize() (which itself
        # needs lots of CPU working memory for ZeRO-3 shard gather), and on
        # tight nodes that drives the system into swap thrashing.
        del gen_state, crit_state
        gc.collect()
        if is_main:
            cprint(f"Model states loaded from checkpoint", "green")
            sys.stdout.flush()

        # ── Critic CPE/control_scale alignment (added 2026-05-25 for DMD phase) ──
        # In MSE-only ckpts, critic is never trained → critic.pt has its
        # original-init CPE (from old SFT teacher's CPE = zero in our case)
        # and scale=0.1. After resume, critic's CPE is misaligned with
        # generator's MSE-trained CPE: critic effectively "can't see" skel
        # for the first ~few hundred DMD steps until backward grows its CPE
        # from zero. Generator's CPE on the other hand was trained for
        # NORMALIZED skel input — exactly the input we feed to critic in DMD.
        # So mirror generator's CPE/scale into critic so critic starts from
        # an "already-skel-aware" state, matching student's input distribution.
        # Only triggers when:
        #   * use_control_dataset (we have CPE)
        #   * resume_from given (= we just loaded a possibly-stale critic CPE)
        #   * generator HAS a non-zero CPE (= MSE produced one; otherwise this
        #     would copy zero → zero, which is harmless but pointless)
        # If gen.CPE is also zero (rare), we noop. The runtime cost is one
        # state_dict copy of a tiny Conv3d (~5.5 MB params) — negligible.
        if args.use_control_dataset and getattr(generator, "control_patch_embedding", None) is not None \
                and getattr(critic, "control_patch_embedding", None) is not None:
            with torch.no_grad():
                gen_cpe_w = generator.control_patch_embedding.weight
                gen_cpe_absmax = float(gen_cpe_w.detach().float().abs().max().item())
                if gen_cpe_absmax > 0.0:
                    # Copy generator's CPE → critic's CPE. Both are nn.Conv3d
                    # with identical Conv3d(48 → 3072, kernel=(1,2,2)) shape.
                    critic.control_patch_embedding.load_state_dict(
                        generator.control_patch_embedding.state_dict()
                    )
                    if (getattr(generator, "control_scale", None) is not None
                            and getattr(critic, "control_scale", None) is not None):
                        critic.control_scale.copy_(generator.control_scale.detach())
                        gen_scale_val = float(generator.control_scale.detach().item())
                        if is_main:
                            cprint(
                                f"[critic-align] Mirrored generator's CPE → critic "
                                f"(gen.CPE.absmax={gen_cpe_absmax:.4f}, "
                                f"control_scale={gen_scale_val:.4f}) "
                                f"so critic sees skel from DMD step 1.",
                                "green",
                            )
                    else:
                        if is_main:
                            cprint(
                                f"[critic-align] Mirrored generator's CPE → critic "
                                f"(gen.CPE.absmax={gen_cpe_absmax:.4f}); "
                                f"control_scale not present on critic — skipped.",
                                "yellow",
                            )
                else:
                    if is_main:
                        cprint(
                            f"[critic-align] generator's CPE is still zero "
                            f"(absmax={gen_cpe_absmax:.4f}) — nothing to mirror. "
                            f"This is expected if MSE was very short (<100 steps); "
                            f"otherwise check that MSE actually trained CPE.",
                            "yellow",
                        )

        # Clamp to >=0 — a basename that doesn't match 'checkpoint-N' returns
        # -1 (we already warned about this above when --resume_from was
        # parsed), which would make the global step counter go negative and
        # trip every "step % save_steps == 0" check.
        resume_step = max(0, _ckpt_step(resume_from_ckpt))

        # ── CausVid-recipe critic init at MSE→DMD transition (added 2026-05-26) ──
        # If we're resuming an MSE-only ckpt (resume_step <= ode_warmup_steps),
        # `critic.pt` is just an untrained snapshot — MSE never updates the critic
        # so it holds whatever pipe.transformer was when MSE first saved it. In
        # our setup that's pure SFT base (no LoRA, no CPE drift), which is FAR
        # from student's actual MSE-trained distribution.
        #
        # CausVid (Yin et al. 2024) and Self-Forcing both initialize critic =
        # student at DMD start so DMD pseudo-gradient `fake_score - real_score`
        # is meaningful from step 1 (= "student vs teacher" direction). Without
        # this, the first ~100-300 DMD steps run with a bad critic that reports
        # SFT-base score on student-distribution samples → noisy gradient,
        # critic_loss spikes while it catches up.
        #
        # We trigger ONLY when resume_step <= ode_warmup_steps (= MSE just
        # finished, DMD just starting). For mid-DMD resumes (resume_step >
        # ode_warmup_steps) the loaded critic.pt has real DMD-trained weights
        # → keep it, don't overwrite.
        if (args.resume_from and args.ode_warmup_steps > 0
                and resume_step <= args.ode_warmup_steps):
            with torch.no_grad():
                critic.load_state_dict(generator.state_dict())
            if is_main:
                cprint(
                    f"[critic-init] CausVid recipe: critic ← student.state_dict() "
                    f"(resume_step={resume_step} <= ode_warmup_steps="
                    f"{args.ode_warmup_steps} → MSE→DMD transition). "
                    f"DMD pseudo-gradient meaningful from step 1.",
                    "cyan",
                )

    # ── Wrap generator & critic in separate DeepSpeed engines ──
    # Two engines, two ZeRO-3 param groups — each model's forward/backward is
    # bookkept independently and backward() on one loss does not touch the other.
    gen_engine, gen_optimizer, _, gen_scheduler = deepspeed.initialize(
        model=generator,
        optimizer=gen_optimizer,
        lr_scheduler=gen_scheduler,
        config=ds_config,
    )
    critic_engine, critic_optimizer, _, critic_scheduler = deepspeed.initialize(
        model=critic,
        optimizer=critic_optimizer,
        lr_scheduler=critic_scheduler,
        config=ds_config,
    )

    # ── EMA shadow (opt-in) ──────────────────────────────────────────────
    # Built AFTER deepspeed.initialize() because that call is what creates
    # `gen_engine.optimizer.fp32_partitioned_groups_flat`. We only allocate
    # the shadow when EMA is enabled (keeps the disabled path allocation-free).
    #
    # Resume semantics:
    #   - If `<resume_ckpt>/generator_ema.pt` exists, load it into the
    #     shadow (shard back across ranks). EMA continues uninterrupted.
    #   - If it doesn't exist (e.g. resuming a pre-EMA checkpoint), the
    #     shadow is initialized from the just-loaded master weights — i.e.
    #     EMA restarts from the current weights. Loud-warn so we notice.
    gen_ema_shadow = None
    if args.ema_decay > 0.0:
        # Defer shadow creation if resume_step < ema_start_step (LongLive style:
        # see distillation.py:561-562, 1303-1309). The lazy-create site lives
        # next to the EMA update calls below.
        if resume_step < args.ema_start_step:
            if is_main:
                cprint(
                    f"[ema] Deferring EMA shadow init until step "
                    f"{args.ema_start_step} (current resume_step={resume_step}, "
                    f"decay={args.ema_decay}). Will lazy-create when train loop "
                    f"reaches that step.",
                    "cyan",
                )
        else:
            if is_main:
                cprint(f"[ema] Initializing generator EMA shadow (decay={args.ema_decay})", "cyan")
            gen_ema_shadow = _build_ema_shadow(gen_engine)

            if resume_from_ckpt:
                ema_src = os.path.join(read_dir, "generator_ema.pt")
                if os.path.isfile(ema_src):
                    if is_main:
                        cprint(f"[ema] Resuming EMA shadow from {ema_src}", "cyan")
                    ema_state = torch.load(ema_src, map_location="cpu", weights_only=True)
                    _ema_load_into_shadow(gen_ema_shadow, gen_engine, ema_state)
                    del ema_state
                    gc.collect()
                    if is_main:
                        cprint("[ema] EMA shadow restored from checkpoint.", "green")
                else:
                    if is_main:
                        cprint(
                            f"[ema] WARNING: no generator_ema.pt found in {read_dir}. "
                            "Initializing EMA from current (just-loaded) weights. "
                            "The EMA shadow will effectively restart tracking from here.",
                            "yellow",
                        )

    # real_score, text_encoder, and negative_embeds were already moved/computed
    # before the deepspeed.initialize() calls above (to free CPU headroom for
    # the engines' Adam state). Just need the scheduler on device.
    scheduler.set_timesteps(1, device=device)

    # ── Text encoding helper (DISABLED — see above) ──
    # The T5 text_encoder is freed after we computed `negative_embeds`. Every
    # dataset sample is required to carry precomputed `text_embeds`; the loop's
    # `if text_embeds is None: encode_prompt(...)` fallback is a deliberate
    # tripwire pointing here. If you actually need on-the-fly encoding, undo
    # the `del text_encoder` block above and remove this guard.
    def encode_prompt(prompt):
        raise RuntimeError(
            "encode_prompt() was called, but the T5 text_encoder was freed "
            "after computing negative_embeds (saves ~9 GB VRAM/rank). All "
            "training samples must provide precomputed `text_embeds` in their "
            "safetensors file. To re-enable on-the-fly encoding, remove the "
            "`del text_encoder` block in main()."
        )

    # ── Training loop ──
    dist.barrier() if dist.is_initialized() else None
    if is_main:
        # Final pre-training memory snapshot — useful baseline for spotting
        # leaks ("step 1 used 30 GB, step 1000 uses 60 GB → leak").
        gpu_summary = []
        for i in range(torch.cuda.device_count()):
            a = torch.cuda.memory_allocated(i) / 1024**3
            r = torch.cuda.memory_reserved(i) / 1024**3
            gpu_summary.append(f"gpu{i}: alloc {a:.1f}/reserv {r:.1f} GB")
        try:
            with open("/proc/meminfo") as f:
                mi = {l.split(":")[0]: l.split(":")[1].strip() for l in f if ":" in l}
            cpu_str = f"CPU avail {mi.get('MemAvailable','?')}"
        except Exception:
            cpu_str = "CPU mem ?"
        cprint(f"[mem pre-train] {cpu_str} | " + " | ".join(gpu_summary), "magenta")

        # ── Variable-length dataset survey (rank 0, first 32 samples) ──
        # epic_rdt.json mixes 7-frame and 21-frame clips. We never want to
        # silently train on only one bucket because the dataset shuffler
        # happened to be unlucky. Sample 32 entries and dump the F-distribution
        # so it's grep-able as `[dataset]`.
        try:
            from collections import Counter
            _f_counts = Counter()
            _shape_examples = []
            _n_probe = min(32, len(dataset))
            for _i in range(_n_probe):
                _s = dataset[_i]
                _vl = _s["video_latent"]
                _f_counts[int(_vl.shape[1])] += 1   # video_latent shape: [C, F, H, W]
                if _i < 3:
                    _shape_examples.append(tuple(_vl.shape))
            cprint(
                f"[dataset] (rank-0, probed {_n_probe}/{len(dataset)} samples) "
                f"frame-count histogram: {dict(_f_counts)} | first 3 shapes: {_shape_examples}",
                "magenta",
            )
            if args.max_cache_frames is not None:
                cprint(
                    f"[dataset] sliding cache cap={args.max_cache_frames} → all probed F values "
                    f"<=cap will SKIP trim, F>cap WILL trim at frame {args.max_cache_frames}+1.",
                    "magenta",
                )
        except Exception as _e:
            cprint(f"[dataset] WARN: variable-length probe failed: {_e!r}", "yellow")

        cprint(f"\nStarting training ... ({args.num_train_steps} steps)\n", "green")
        sys.stdout.flush()

    step = resume_step
    start_time = time.time()
    log_dict = {
        "gen_loss": 0.0, "crit_loss": 0.0, "grad_norm": 0.0,
        # DMD-specific diagnostics — see logging block ~line 1100 for use.
        "dmd_grad_norm": float("nan"),     # mean(|fake_score - real_score|) — DMD pseudo-gradient magnitude
        "gen_latent_std": float("nan"),    # std of `generated` — should be O(1), not collapsing or exploding
        "gen_latent_mean": float("nan"),   # mean of `generated` — should be near 0 for healthy bf16 dynamics
        "gen_latent_absmax": float("nan"), # max abs — early warning for NaN-edge / saturation
        "gen_grad_norm": float("nan"),     # DeepSpeed-reported global grad norm of generator (post-clip view)
        "crit_grad_norm": float("nan"),    # same for critic
    }
    step_times_window: list = []   # rolling window for avg step time / ETA

    # ── Lazy VAE decoder for periodic visual debug (rank 0 only) ─────────────
    # When --decode_every > 0 we save decoded frames as JPEGs under
    # output_dir/debug_frames/step{N:07d}__{LAUNCH_TAG}_{last,all}.jpg.
    # See LAUNCH_TAG block below for why filenames are launch-tagged.
    # The VAE itself was deleted
    # earlier with `del pipe` to save VRAM; we lazily reload it (CPU) only on
    # the first decode call, do the decode under no_grad, and immediately
    # `to('cpu')` after — so peak GPU impact during the brief decode window
    # is ~1 GB (Wan VAE is small). The decode is rank-0-only so it doesn't
    # block the other 7 ranks waiting at the next `gen_engine.backward()`.
    _vae_holder = {"vae": None}
    debug_frames_dir = os.path.join(args.output_dir, "debug_frames")

    def _step_debug_dir(step_id: int) -> str:
        """Per-step subdir to keep `debug_frames/` navigable when 250+ decode
        triggers accumulate over a 5-day run. Each `_decode_and_save_*`
        writes into this dir.

        Layout:
            debug_frames/
              step_0002020/
                grid_a.jpg, grid_b.jpg, grid_c.jpg
                grid_a.jpg.motion.json, grid_b.jpg.motion.json, ...
                dmd_a.jpg, dmd_b.jpg
                metrics.png
              step_0002040/
                ...
              metrics_latest.png       ← root, always-updated symlink-of-latest
              {step_NNN}__{LAUNCH_TAG}_metrics.png  ← legacy, written by plot script

        Filename within subdir drops the `step{N}__` prefix (redundant) but
        keeps LAUNCH_TAG as suffix so a v1-style resume that re-decodes the
        same step adds files instead of overwriting (e.g. `grid_a__launch_TAG.jpg`).
        """
        d = os.path.join(debug_frames_dir, f"step_{step_id:07d}")
        os.makedirs(d, exist_ok=True)
        return d
    # LAUNCH_TAG (set by the launch script) tags every debug frame written by
    # THIS launch. Without it, resume@step3500 overwrites the original
    # step3550 image with new content from a different hyperparameter regime,
    # making post-mortem comparison impossible. Fall back to a startup-time
    # string when launched outside the shell wrapper.
    _launch_tag = os.environ.get("LAUNCH_TAG") or time.strftime("%Y%m%d_%H%M%S")
    if is_main:
        cprint(f"[launch] LAUNCH_TAG={_launch_tag}  (debug frames will be suffixed with this)", "cyan")

    # Decode dtype: parse once, reuse. Wan VAE in bf16 is a measurable noise
    # source; fp32 is the safe default. Cost: ~1 GB extra VRAM during the
    # brief decode window (rank 0 only, function runs every decode_every
    # steps), which is well within budget since gen activations are freed
    # by then.
    _decode_dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[getattr(args, "decode_dtype", "fp32")]

    # Lazy holder for diffusers' canonical VideoProcessor. Same object used by
    # the official Wan I2V pipeline (see pipeline_wan_i2v.py:196 and
    # inference-sft.py:208). Constructing it is cheap (~µs, stateless) but
    # we still cache because _decode_latent is called multiple times per
    # debug step (one per scene × maybe ×2 for teacher). vae_scale_factor=8
    # matches `self.vae.config.scale_factor_spatial` for Wan2.2.
    _video_processor_holder = {"vp": None}

    def _decode_latent(vae, latent_5d: torch.Tensor) -> "_np.ndarray":
        """Lower-level decode: latent [B,C,F,H,W] → uint8 numpy [B,3,Fout,H,W].

        Pixel-conversion strategy MIRRORS the official Wan I2V pipeline and
        FlowWorld/inference-sft.py exactly:
          1.  Un-normalize latents:  lat = lat * latents_std + latents_mean
              (mathematically identical to inference-sft.py's
              `lat / (1/std) + mean`.)
          2.  vae.decode(...)        → [B,3,F_out,H_pix,W_pix] in [-1, 1]
              (Wan VAE temporally upsamples ×4: 6 latent F → 21 RGB F.)
          3.  video_processor.postprocess_video(vid, output_type='np')
              → [B, F_out, H_pix, W_pix, 3] float32 in [0, 1]
              Internally that's `(vid*0.5 + 0.5).clamp(0,1)` then channels-last.
          4.  Multiply by 255 → uint8.
          5.  Re-permute to [B, 3, F_out, H_pix, W_pix] which is what
              s5_to_time_strip downstream expects (it does `[0, :3, i]`).

        Why we don't keep the old manual `clamp(-1,1) + (vid+1)*127.5`:
        mathematically the same answer, but going through diffusers'
        VideoProcessor means any future Wan-VAE output-range fixes upstream
        are picked up automatically without us drifting.

        Caller owns VAE placement (we just call .decode); we cast the latent
        to whatever dtype the VAE itself is in (set by --decode_dtype).
        Returns CPU numpy so callers can compose panels without holding GPU.
        """
        # Local imports keep _decode_latent independent of where it gets
        # called from (the outer `import numpy as _np` lives inside the
        # train loop body, but a future refactor could move that around).
        from diffusers.video_processor import VideoProcessor
        import numpy as _np
        if _video_processor_holder["vp"] is None:
            scale_factor = getattr(vae.config, "scale_factor_spatial", 8)
            _video_processor_holder["vp"] = VideoProcessor(vae_scale_factor=scale_factor)
        vp = _video_processor_holder["vp"]

        z_dim = vae.config.z_dim
        v_dtype = next(vae.parameters()).dtype
        lat = latent_5d.to(device=device, dtype=v_dtype)
        lm = torch.tensor(vae.config.latents_mean, device=device, dtype=v_dtype).view(1, z_dim, 1, 1, 1)
        ls = torch.tensor(vae.config.latents_std,  device=device, dtype=v_dtype).view(1, z_dim, 1, 1, 1)
        lat = lat * ls + lm
        vid = vae.decode(lat, return_dict=False)[0]              # [B,3,F,H,W] in [-1,1]
        # postprocess_video returns numpy [B, F, H, W, 3] in [0, 1] when
        # output_type='np' (see VideoProcessor.postprocess_video → VaeImageProcessor.postprocess).
        vid_np = vp.postprocess_video(vid, output_type="np")
        # Convert to uint8 and re-permute to [B, 3, F, H, W] for s5_to_time_strip.
        vid_u8 = (vid_np * 255.0).clip(0, 255).astype(_np.uint8)
        # _np.transpose: (B, F, H, W, 3) → (B, 3, F, H, W)
        return vid_u8.transpose(0, 4, 1, 2, 3)

    def _decode_and_save_frame(latent_5d: torch.Tensor, step_id: int,
                               extra_student_latents: list = None,
                               extra_pair_indices: list = None,
                               teacher_streaming_latents: list = None,
                               teacher_dmd_latents: dict = None,
                               grid_suffix: str = "") -> None:
        """latent_5d: [B, C=z_dim=48, F, H, W]. The "current" student output
        for ONE scene. Multi-scene comparison + teacher reference are layered
        on top via the closures (current_pair_idx_holder, ode_dataset, etc.).

        IMPORTANT: extra_student_latents must have been gathered by ALL ranks
        BEFORE this function is called (because the function body runs only on
        rank 0). Under ZeRO-3 + param offload, doing extra `generate_streaming`
        forwards inside this rank-0-only block would deadlock — the gather of
        sharded params is a collective, all 8 ranks must participate. The
        outer caller (the train loop) is responsible for running the extra
        forwards in lockstep on all ranks and passing rank-0-relevant tensors
        in here. Ranks 1-7 just call this function and immediately return.

        Output layout
        -------------
        Each call writes ONE jpg per saved file:
            step{N:07d}__{LAUNCH_TAG}_grid.jpg

        Each ROW is a TIME STRIP: --decode_max_frames frames sampled
        uniformly from the scene's 21-RGB-frame video, concatenated
        horizontally with thin gray separators. So you can see WITHIN
        a single scene how the model's output evolves through the clip,
        not just the last frame.

        Per-scene rows:
            • without teacher (DMD or --decode_with_teacher off):
                1 row  →  [student frames f0..fN]
            • with teacher (ODE warmup + --decode_with_teacher):
                2 rows →  row k:   [student frames f0..fN]
                          row k+1: [teacher frames f0..fN]
                The student row is directly above the matching teacher
                row, so eyes can scan column-by-column to spot temporal
                drift / per-frame misalignment.

        N = decode_num_scenes scenes are stacked top-to-bottom.
        WITHIN-scene gap (student↔teacher): thin 2-px gray strip.
        BETWEEN-scenes gap: thick 12-px white strip.
        The asymmetric gaps make 4-scene panels read as "4 paired
        comparisons" not "8 unrelated images".

        Why fp32 VAE: bf16 decode introduces visible noise from the GroupNorm
        + temporal-upsample chain (~3-bit mantissa rounding accumulates).
        For debug-only decoding (every N steps on rank 0) the ~1 GB extra
        VRAM and ~2× slower decode are negligible.
        """
        if not is_main:
            return
        try:
            from diffusers import AutoencoderKLWan
            from PIL import Image
            import numpy as _np
            os.makedirs(debug_frames_dir, exist_ok=True)
            if _vae_holder["vae"] is None:
                vae_src = args.decode_model_path or args.model_path
                cprint(f"[decode] First call — loading VAE from {vae_src} in "
                       f"{args.decode_dtype} (rank0 stalls ~5-30s) ...", "yellow")
                sys.stdout.flush()
                _vae_holder["vae"] = AutoencoderKLWan.from_pretrained(
                    vae_src, subfolder="vae", torch_dtype=_decode_dtype
                ).eval()
            # Free reserved-but-unallocated GPU cache before swapping VAE in.
            torch.cuda.empty_cache()
            vae = _vae_holder["vae"].to(device)

            # ── Compose scenes (no DeepSpeed-collective ops happen here) ──
            # Scene 0 is the live training output. Extra scenes were generated
            # in lockstep on all ranks by the caller; we only got rank 0's
            # tensors via extra_student_latents. Teacher latents come from disk
            # cache (no DeepSpeed involvement → safe to load rank-0-only).
            extra_student_latents = extra_student_latents or []
            extra_pair_indices = extra_pair_indices or []
            assert len(extra_student_latents) == len(extra_pair_indices), \
                "internal: extra_student_latents / extra_pair_indices mismatch"

            # All scenes in a single decode call must share the SAME column
            # count or np.concatenate(rows, axis=0) will throw a "dimension
            # mismatch" ValueError. Pick the reference-row decision ONCE per
            # call and apply it uniformly to live + extras. Without this,
            # at exactly step_id == ode_warmup_steps the live scene drops
            # the reference (step_id < warmup → False) but extras keep it
            # (no step_id check), producing rows of width 640 vs 1284 →
            # decode crashes, leaving the VAE on GPU → next train step OOM.
            # Same step-boundary fix applies to extras' reference condition.
            #
            # Reference-row source picked from `args.ode_target_kind`:
            #   teacher (A 方案): pull x0_teacher from ode_dataset (disk pair cache)
            #   gt      (B 方案): pull video_latent from `dataset` (DMD GT clips)
            # In B mode `ode_dataset` stays None (we never load it), so the
            # teacher path's `ode_dataset is not None` guard naturally bypasses
            # — we route to the GT branch via ode_target_kind instead.
            in_warmup_window = step_id < args.ode_warmup_steps
            with_teacher_ref = (args.decode_with_teacher
                                and ode_dataset is not None
                                and in_warmup_window
                                and args.ode_target_kind in ("teacher", "causvid_traj"))
            # NOTE (2026-05-14): widened from `== "gt"` to also accept v_flow.
            # Both warmup modes pull their MSE target from the DMD `dataset`
            # (LatentDataset → GT video latents) and both populate
            # `_current_gt_idx_holder["idx"]` in their training-loop branches
            # (gt: L3832, v_flow: L3888). The GT helpers (_ode_gt_latent_for,
            # _pick_extra_scene_gt_indices, _student_inference_for_gt_idx)
            # work bit-identically for both modes — they don't care which
            # warmup loss formulation produced the live student latent. Without
            # this widening v_flow's debug grid was silently 1 row instead of
            # the expected 4 scenes × 2 rows (student+GT).
            #
            # NOTE (2026-05-14, 2nd pass): also dropped the `in_warmup_window`
            # constraint. After the warmup→DMD switch the student keeps
            # consuming samples from the SAME `dataset` (now via the DMD
            # branch at L4400, `sample = dataset[idx]`), so a "GT row" comparison
            # is just as meaningful in DMD as in warmup — it shows whether the
            # student is staying on the GT manifold under the adversarial
            # objective. This requires the DMD branch to populate
            # `_current_gt_idx_holder["idx"]` with the actual sample idx (done
            # at L4396-4404 after this edit) instead of leaving it -1.
            with_gt_ref = (args.decode_with_teacher
                           and args.ode_target_kind in ("gt", "v_flow")
                           and dataset is not None)
            # 2026-05-14 (3rd pass): teacher/GT rows are NO LONGER mutually
            # exclusive. In gt/v_flow modes we now want BOTH:
            #   • GT row    — VAE-roundtripped GT video latent (target manifold)
            #   • teacher row — frozen base Wan2.2-5B running streaming on the
            #                   same cond image (independent baseline that
            #                   tells us whether sliding-cache + slot-RoPE
            #                   itself can produce motion at this F+cache cap)
            # Caller passes pre-computed teacher latents via
            # `extra_teacher_streaming_latents` for the live scene + each extra.
            # Empty list = teacher row disabled.
            extra_teacher_streaming_latents = list(teacher_streaming_latents or [])
            # teacher_dmd_latents: {t_value: [latent_for_live, latent_for_extra_0, ...]}
            # One row per t_value, in ascending t order. Caller passes pre-computed
            # tensors (rank-0-only, no collective in real_score forward).
            teacher_dmd_by_t = dict(teacher_dmd_latents or {})
            ref_label = ("teacher_pair" if with_teacher_ref
                         else ("GT" if with_gt_ref else None))

            def _resolve_ref_for_live():
                if with_teacher_ref:
                    return _ode_teacher_latent_for(_current_pair_idx_holder["idx"])
                if with_gt_ref:
                    return _ode_gt_latent_for(_current_gt_idx_holder["idx"])
                return None

            def _resolve_ref_for_extra(extra_idx):
                if with_teacher_ref:
                    return _ode_teacher_latent_for(extra_idx)
                if with_gt_ref:
                    return _ode_gt_latent_for(extra_idx)
                return None

            # Each scene = (student_5d, [(row_label, ref_5d), ...], scene_label)
            scenes: list = []
            if with_gt_ref:
                live_idx = _current_gt_idx_holder["idx"]
                live_kind = "gt"
            else:
                live_idx = _current_pair_idx_holder["idx"]
                live_kind = "pair"
            live_refs = []
            _live_ref = _resolve_ref_for_live()
            if _live_ref is not None:
                live_refs.append((ref_label or "ref", _live_ref))
            if with_gt_ref and args.use_control_dataset:
                _live_ctrl = _ode_control_latent_for(live_idx)
                if _live_ctrl is not None:
                    live_refs.append(("control", _live_ctrl))
            if extra_teacher_streaming_latents:
                live_refs.append(("teacher_stream", extra_teacher_streaming_latents[0]))
            # DMD-signal teacher rows (one per t_value, ascending t).
            for _t in sorted(teacher_dmd_by_t.keys()):
                _arr = teacher_dmd_by_t[_t]
                if _arr and _arr[0] is not None:
                    live_refs.append((f"teacher@t={_t}", _arr[0]))
            scenes.append((latent_5d.detach(), live_refs,
                           f"scene0 (live, {live_kind}={live_idx})"))

            # Append extras. We trust the caller did the right thing (all-rank
            # forwards) — they're already concrete tensors at this point.
            extra_kind = "gt" if with_gt_ref else "pair"
            for i, (student_5d, px) in enumerate(zip(extra_student_latents, extra_pair_indices)):
                refs = []
                _ref = _resolve_ref_for_extra(px)
                if _ref is not None:
                    refs.append((ref_label or "ref", _ref))
                if with_gt_ref and args.use_control_dataset:
                    _ex_ctrl = _ode_control_latent_for(px)
                    if _ex_ctrl is not None:
                        refs.append(("control", _ex_ctrl))
                # extra_teacher_streaming_latents[0] is for live; +1 offset.
                if (extra_teacher_streaming_latents
                        and (i + 1) < len(extra_teacher_streaming_latents)):
                    refs.append(("teacher_stream",
                                 extra_teacher_streaming_latents[i + 1]))
                # DMD-signal teacher rows for this extra (same +1 offset).
                for _t in sorted(teacher_dmd_by_t.keys()):
                    _arr = teacher_dmd_by_t[_t]
                    if _arr and (i + 1) < len(_arr) and _arr[i + 1] is not None:
                        refs.append((f"teacher@t={_t}", _arr[i + 1]))
                scenes.append((student_5d, refs, f"{extra_kind}={px}"))

            with torch.no_grad():
                # Decode each (student, teacher) into uint8 numpy. Produce TWO
                # rows per scene when teacher is present (student row, teacher
                # row directly below — frame-by-frame visual alignment), one
                # row per scene otherwise.
                #
                # Row width = decode_max_frames * W + (decode_max_frames-1) * 2px
                # For W=640, max_frames=8 → row width 5106 px. Without the
                # final 2× downsample below: too wide for most viewers.
                # With downsample: 2553 px wide × ~8 rows tall ≈ 1080 tall →
                # readable on a 4K monitor and ~120 KB JPEG.
                n_frames = max(1, args.decode_max_frames)
                # rows = list of (np_strip, is_new_scene_marker)
                # is_new_scene_marker is True for the very first row of a scene
                # (used to pick thick vs thin separator above this row).
                # ── DECODE_SAVE_VIDEOS env (added 2026-05-26): also dump each
                #    scene's student / GT / control rows as separate .mp4 files
                #    next to the grid jpg. Skips teacher@t=* rows (those are
                #    not coherent video, just DMD diagnostics).
                #    Default off (don't slow down legacy MSE runs); enable per
                #    wrapper. ~9 mp4/scene/decode trigger (3 rows × 3 scenes
                #    default) × ~1 MB each ≈ 9 MB per decode, 250 decode
                #    triggers × 3 grid_repeats ≈ 7 GB total over 10k-step run. ──
                _save_videos = int(os.environ.get("DECODE_SAVE_VIDEOS", "0"))
                # DECODE_VIDEO_ROW_WHITELIST: CSV of row labels to save as mp4.
                # Default = "student" only (DMD use case: only need student
                # outputs for cross-decode comparison; GT/control are static
                # and visible in the grid jpg). Set to "student,GT,control"
                # to save all 3 (legacy behavior + cluster 16-card pre-fix).
                _video_whitelist_env = os.environ.get(
                    "DECODE_VIDEO_ROW_WHITELIST", "student"
                )
                _video_row_whitelist = {
                    s.strip() for s in _video_whitelist_env.split(",") if s.strip()
                }  # skip teacher@t=* unconditionally (those are diagnostics not video)
                # Lazy import imageio (only when this branch fires + flag on).
                if _save_videos:
                    try:
                        import imageio.v2 as _imageio
                    except Exception:
                        try:
                            import imageio as _imageio  # legacy fallback
                        except Exception as _e:
                            if rank == 0:
                                cprint(f"[decode-video] WARN: imageio unavailable "
                                       f"({_e!r}) — skipping mp4 save.", "yellow")
                            _save_videos = 0
                    # Backend probe: PyAV (auto-selected when imageio_ffmpeg
                    # missing) has incompatible API. Probe with a 1×8×8×3 dummy
                    # to fail fast at startup, not after 40 step-1080 attempts.
                    # 2026-05-26: cluster containers ship `av` but NOT
                    # `imageio_ffmpeg` → all 21F writes fail with TypeError
                    # ("PyAVPlugin.write() got unexpected kwarg") or codec
                    # broadcasting bug. Fix at cluster level: pip install
                    # imageio-ffmpeg (4 MB, no compile, just bundles ffmpeg
                    # binary).
                    if _save_videos:
                        try:
                            _probe_path = os.path.join(
                                os.environ.get("DECODE_STAGE_DIR", "/tmp"),
                                f"_mp4_probe_rank{rank}.mp4",
                            )
                            _dummy = _np.zeros((8, 8, 3), dtype=_np.uint8)
                            with _imageio.get_writer(_probe_path, fps=8) as _pw:
                                _pw.append_data(_dummy)
                            try:
                                os.remove(_probe_path)
                            except Exception:
                                pass
                        except Exception as _pe:
                            if rank == 0:
                                cprint(
                                    f"[decode-video] WARN: mp4 backend probe "
                                    f"failed ({_pe!r}). Disabling video save "
                                    f"for this run. Fix: `pip install "
                                    f"imageio-ffmpeg` in the env (4 MB, no "
                                    f"compile). PyAV alone doesn't work with "
                                    f"imageio default kwargs.",
                                    "yellow",
                                )
                            _save_videos = 0

                def _decoded_to_video_uint8(u8_5d):
                    """[1, 3, F, H, W] uint8 → [F, H, W, 3] uint8 for imageio."""
                    return u8_5d[0].transpose(1, 2, 3, 0)  # (3,F,H,W) → (F,H,W,3)

                def _save_video_mp4(u8_5d, out_path, fps=8):
                    """Write a single-row mp4 via imageio. fps=8 matches Wan2.2
                    VAE temporal stride (21 latent frames ≈ 2.6 s @ 8 fps visual;
                    24 fps native, but VAE compresses 4× temporally).

                    2026-05-26 fix: use plugin-agnostic `get_writer` context
                    manager. Earlier `mimwrite(..., macro_block_size=1, quality=8)`
                    failed with "PyAVPlugin.write() got an unexpected keyword
                    argument 'macro_block_size'" — those kwargs are
                    imageio-ffmpeg specific; PyAV (auto-selected when installed)
                    doesn't accept them. `get_writer(path, fps=N).append_data(f)`
                    works on both backends with sensible defaults.
                    """
                    frames = _decoded_to_video_uint8(u8_5d)
                    tmp_path = os.path.join(
                        os.environ.get("DECODE_STAGE_DIR", "/tmp"),
                        f"_video_{os.path.basename(out_path)}",
                    )
                    try:
                        with _imageio.get_writer(tmp_path, fps=fps) as _w:
                            for _frame in frames:
                                _w.append_data(_frame)
                        shutil.move(tmp_path, out_path)
                    except Exception as _ve:
                        if rank == 0:
                            cprint(f"[decode-video] WARN: mp4 write failed for "
                                   f"{out_path}: {_ve!r}", "yellow")

                rows = []
                _video_save_count = 0
                for scene_i, (s5, refs, label) in enumerate(scenes):
                    s_u8 = _decode_latent(vae, s5)
                    rows.append((s5_to_time_strip(s_u8, n_frames), scene_i > 0))
                    # NEW: per-scene student mp4 (gated by whitelist)
                    if _save_videos and "student" in _video_row_whitelist:
                        _vfile = os.path.join(
                            _step_debug_dir(step_id),
                            f"scene{scene_i}_student"
                            f"{'_' + grid_suffix if grid_suffix else ''}"
                            f"__{_launch_tag}.mp4",
                        )
                        _save_video_mp4(s_u8, _vfile)
                        _video_save_count += 1
                    for _row_lbl, ref_5d in refs:
                        if ref_5d is None:
                            continue
                        ref_u8 = _decode_latent(vae, ref_5d)
                        rows.append((s5_to_time_strip(ref_u8, n_frames), False))
                        # NEW: per-scene ref mp4 (only for whitelisted labels)
                        if _save_videos and _row_lbl in _video_row_whitelist:
                            _vfile = os.path.join(
                                _step_debug_dir(step_id),
                                f"scene{scene_i}_{_row_lbl}"
                                f"{'_' + grid_suffix if grid_suffix else ''}"
                                f"__{_launch_tag}.mp4",
                            )
                            _save_video_mp4(ref_u8, _vfile)
                            _video_save_count += 1

                # Stack rows top-to-bottom with separators that distinguish
                # WITHIN-scene gap (student↔teacher/GT: thin 2-px gray) from
                # BETWEEN-scene gap (thick 12-px white). Asymmetric gaps make
                # an N-row panel read as paired comparisons.
                # Mixed-frame batches (e.g. 7F + 21F) produce time-strips of
                # different widths since s5_to_time_strip lays frames out
                # horizontally. Right-pad shorter rows with white so vertical
                # concat works; pad (not resize) keeps per-frame scale honest.
                row_w = max(r.shape[1] for r, _ in rows)
                rows = [
                    (r if r.shape[1] == row_w else _np.pad(
                        r, ((0, 0), (0, row_w - r.shape[1]), (0, 0)),
                        constant_values=255), flag)
                    for r, flag in rows
                ]
                thin = _np.full((2, row_w, 3), 200, dtype=_np.uint8)
                thick = _np.full((12, row_w, 3), 255, dtype=_np.uint8)
                panel = []
                for i, (r, is_new_scene) in enumerate(rows):
                    if i > 0:
                        panel.append(thick if is_new_scene else thin)
                    panel.append(r)
                grid = _np.concatenate(panel, axis=0)

                # NO downsample — we MUST show truthful pixels for diagnosis.
                # (Used to do `grid = grid[::2, ::2]` here to halve filesize,
                # but that visibly softens the image and makes it impossible
                # to distinguish a real model-side blur from the resampling
                # artifact. With quality=95 JPEG below the panel comes out at
                # ~3-5 MB for a 4-scene × 21-frame × 2-row layout, which is
                # acceptable on FUSE since we only write every decode_every
                # steps.)

            # ── Write to disk via /tmp staging (FUSE doesn't support PIL's
            # mmap / fsync / ftruncate syscalls). ──
            # Per-step subdir layout (added 2026-05-19, see _step_debug_dir
            # docstring): debug_frames/step_{N:07d}/grid_{a,b,c}__{TAG}.jpg
            _step_dir = _step_debug_dir(step_id)
            _suffix = f"_{grid_suffix}" if grid_suffix else ""
            _fname = f"grid{_suffix}__{_launch_tag}.jpg"
            out_grid = os.path.join(_step_dir, _fname)
            tmp_dir = os.environ.get("DECODE_STAGE_DIR", "/tmp")
            tmp_grid = os.path.join(tmp_dir, f"step{step_id:07d}_{_fname}")
            # quality=95 (vs 85): visually near-lossless. At 85 the JPEG
            # quantizer wipes out fine high-freq content that we'd otherwise
            # mistake for "model said it's blurry". 95 keeps everything we
            # actually care about for diagnosis.
            Image.fromarray(grid).save(tmp_grid, quality=95, subsampling=0)
            shutil.move(tmp_grid, out_grid)

            n_scenes = len(scenes)
            _row_kinds = ["student"] + [lbl for lbl, _ in scenes[0][1]]
            rows_per_scene_lbl = "+".join(_row_kinds)
            _vid_msg = f", videos={_video_save_count}" if _video_save_count else ""
            cprint(f"[decode] step {step_id} → {out_grid} "
                   f"({n_scenes} scene{'s' if n_scenes > 1 else ''} × "
                   f"{n_frames} frame{'s' if n_frames > 1 else ''}, "
                   f"per-scene rows: {rows_per_scene_lbl}, "
                   f"vae={args.decode_dtype}, shape={grid.shape}{_vid_msg})", "yellow")

            # ── Plot training metrics (loss curves, grad norms, etc.) ──
            # Subprocess so a matplotlib bug never crashes the train loop.
            # Async (Popen, no wait) so we don't pay the ~1-2s plot cost on
            # the train thread. Plot reads train.log so it sees ALL history
            # up to this point, regardless of how often we call it.
            #
            # IMPORTANT (2026-05-19 fix): only spawn plot for the FIRST grid
            # of each decode trigger (= grid_suffix == "" or "a"). With
            # DECODE_GRID_REPEATS=3 we used to spawn 3 plot subprocesses per
            # decode, all reading the same train.log + writing to the same
            # `_metrics_latest.png` → FUSE write contention + matplotlib font
            # cache races caused subprocesses to hang for 30+ min, blocking
            # rank-0's main thread (in `pselect6` waiting on something) → 8x
            # zombie children → whole 8-GPU training stuck. Fix: spawn just
            # ONE plot subprocess per decode trigger (grid_suffix=="a" with
            # n_grids>1, or grid_suffix=="" otherwise).
            _is_first_grid_of_decode = (grid_suffix == "" or grid_suffix == "a")
            if _is_first_grid_of_decode:
                try:
                    import subprocess as _subprocess
                    _plot_script = os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        "scripts", "plot_train_metrics.py",
                    )
                    if os.path.isfile(_plot_script):
                        _subprocess.Popen(
                            ["python", _plot_script, args.output_dir,
                             "--step", str(step_id)],
                            stdout=_subprocess.DEVNULL,
                            stderr=_subprocess.DEVNULL,
                            stdin=_subprocess.DEVNULL,
                            # `start_new_session=True` puts the child in its
                            # own session leader → if the parent dies/hangs,
                            # the child's stdin/stdout don't get into a
                            # weird state where parent's pselect on inherited
                            # fds blocks forever.
                            start_new_session=True,
                        )
                except Exception as _plot_e:
                    cprint(f"[plot] WARN: launch failed: {_plot_e!r}", "yellow")
        except Exception as e:
            # Decode is debug-only — never crash training over it. Log the
            # full traceback the FIRST time it fails so we can fix the bug.
            import traceback
            cprint(f"[decode] WARNING: failed at step {step_id}: {e!r}", "red")
            if step_id <= args.decode_every * 2:  # only spam on first 1-2 failures
                traceback.print_exc()
        finally:
            # Always offload VAE back to CPU, even on exception. Without this,
            # a mid-decode crash leaves the ~3.6 GB fp32 VAE on GPU; the very
            # next training step's critic forward then OOMs at 79 GB capacity
            # because gen+critic+real+VAE > 79. Observed in D run 190650 at
            # step 2000 (decode raised ValueError → OOM next step). The CPU
            # move + empty_cache must run unconditionally; both are no-ops if
            # the VAE was never loaded onto GPU this call.
            try:
                if _vae_holder["vae"] is not None:
                    _vae_holder["vae"].to("cpu")
            except Exception:
                pass
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            sys.stdout.flush()

    def _decode_and_save_dmd_grid(step_id: int, slot: dict,
                                  grid_suffix: str = "") -> None:
        """NEW debug grid (added 2026-05-18) — visualizes the EXACT four
        tensors `compute_dmd_generator_loss` differentiated this step.

        Source of truth: `slot` argument, one entry from
        `_DMD_DEBUG_RING` (module-level), populated inside
        `compute_dmd_generator_loss` itself so we GUARANTEE the decoded
        rows are the tensors the DMD gradient was computed on, not a
        separate forward pass after `gen_engine.step()`.

        Layout (per scene b ∈ [0, B-1] in the batch dim of the slot):
            Row 1: STUDENT       = generated_for_loss[b]    (clean x_0 the loss target)
            Row 2: CRITIC x_0    = fake_x0[b]               (critic's denoising of row 4)
            Row 3: TEACHER x_0   = real_x0[b]               (real_score's denoising of row 4)
            Row 4: NOISED INPUT  = noisy_for_score[b]       (the (1-σ)·row1 + σ·noise fed to row2/3)
        Each row decoded via the SAME `_decode_latent` helper used by
        `_decode_and_save_frame` (Wan VAE, fp32 by default), then
        composed via `s5_to_time_strip` into a 21-frame horizontal strip.

        File naming (deliberately distinct from `_decode_and_save_frame`):
            step{N:07d}__{LAUNCH_TAG}_dmd{_suffix}.jpg
        Suffix `_a`, `_b`, ... when caller dumps multiple ring slots.

        Trigger: DMD-phase only (caller gates on `is_warmup_step==False`).
        Per-call cost: rank-0-only, 4 rows × B scenes × VAE decode (~6s
        total at B=1,F=21,H=480,W=832 in fp32). Negligible inside
        decode_every=40 cadence.

        Skips silently when the slot is empty / partially filled.
        """
        if not is_main:
            return
        # Slot empty / never filled → silent skip.
        if slot is None or slot.get("generated_for_loss") is None:
            return
        try:
            from diffusers import AutoencoderKLWan
            from PIL import Image
            import numpy as _np
            os.makedirs(debug_frames_dir, exist_ok=True)
            if _vae_holder["vae"] is None:
                vae_src = args.decode_model_path or args.model_path
                cprint(f"[decode-dmd] First call — loading VAE from {vae_src} in "
                       f"{args.decode_dtype} (rank0 stalls ~5-30s) ...", "yellow")
                sys.stdout.flush()
                _vae_holder["vae"] = AutoencoderKLWan.from_pretrained(
                    vae_src, subfolder="vae", torch_dtype=_decode_dtype
                ).eval()
            torch.cuda.empty_cache()
            vae = _vae_holder["vae"].to(device)

            # Pull from the slot — these are CPU tensors already, no GPU
            # state held across iterations.
            stu_t     = slot["generated_for_loss"]
            fake_t    = slot["fake_x0"]
            real_t    = slot["real_x0"]
            noisy_t   = slot["noisy_for_score"]
            sigmas_t  = slot["sigmas"]
            # ★ heatmap source: actual normalized DMD gradient. Old slots
            # (pre-2026-05-19 v3) lack this key — fall back to |gen − real|
            # so we don't crash on resumed-from-checkpoint runs.
            grad_t    = slot.get("dmd_grad", None)
            snap_step = int(slot.get("step_id", -1))
            cos_val   = slot.get("cos", float("nan"))
            sm_zero   = slot.get("safemask_zero", float("nan"))
            s_mean    = slot.get("sigma_mean", float("nan"))
            s_min     = slot.get("sigma_min",  float("nan"))
            s_max     = slot.get("sigma_max",  float("nan"))
            gen_amean = slot.get("gen_x0_absmean",  float("nan"))
            fak_amean = slot.get("fake_x0_absmean", float("nan"))
            rea_amean = slot.get("real_x0_absmean", float("nan"))
            rg_amean  = slot.get("raw_grad_absmean", float("nan"))
            nmin      = slot.get("normer_min", float("nan"))

            # Each scene = one sample from the batch dim. With Run C B=1
            # this is a single scene; B=2/4 datasets multi-row out naturally.
            # Cap at DECODE_DMD_MAX_SCENES (default 4) to bound IO size.
            _max_scenes = int(os.environ.get("DECODE_DMD_MAX_SCENES", "4") or 4)
            B = int(stu_t.shape[0])
            n_scenes = max(1, min(B, _max_scenes))
            n_frames = max(1, args.decode_max_frames)

            # ── Decode each row (4 rows per scene) ──────────────────────
            # CHANGED 2026-05-19 (v3): Row 4 visualizes |dmd_grad| =
            # |(fake_x0 − real_x0) / normalizer * safe_mask| — the ACTUAL
            # per-pixel DMD gradient signal the student receives during
            # `loss.backward()`. Directly answers "DMD 优化哪里" because the
            # student moves in proportion to this magnitude at each pixel.
            #
            # History
            # ────────
            #   v1 (|fake − real|):   useless at step ~2005 (critic ≡ teacher
            #                         bit-equally → all zeros), and bf16
            #                         quantization later masks most pixels.
            #   v2 (|gen − real|):    non-zero from step 1, but it's "reference
            #                         direction" not "actual gradient" — when
            #                         critic learns well, fake → real and the
            #                         REAL gradient |fake−real|/normer shrinks
            #                         even when |gen−real| stays large. So v2
            #                         doesn't reflect what DMD actually pushes.
            #   v3 (|dmd_grad|):      = exactly the normalized DMD gradient
            #                         that backprop receives, scaled per
            #                         pixel. p99-normalized to avoid sparse
            #                         outlier compression.
            with torch.no_grad():
                rows = []  # list of (np_strip, is_new_scene)

                # Precompute diff heatmap latents (one per scene).
                import matplotlib  # type: ignore
                matplotlib.use("Agg")
                _cmap = matplotlib.colormaps["magma"]

                _vae_spatial = int(getattr(vae.config, "scale_factor_spatial", 8))

                def _make_heatmap_row(grad_5d_local):
                    """Return a [1, 3, F, H_dec, W_dec] uint8 tensor of the
                    |dmd_grad| heatmap (channels-mean, magma-colormapped,
                    p99-per-frame-normalized, upscaled to decoded resolution).

                    Input
                    ─────
                    grad_5d_local: [1, C, F, Hlat, Wlat] = the SAVED `dmd_grad`
                        slot tensor = `(fake_x0 - real_x0) / normalizer * safe_mask`
                        with frame 0 zero-padded. This is the literal gradient
                        backprop pushes through generated_for_loss in
                        `compute_dmd_generator_loss`.

                    Why p99 not max: |dmd_grad| typically has 1-5% sparse outlier
                    pixels (~ a few channels' worst case at noisy interfaces) and
                    95%+ pixels with much smaller magnitude. max-normalize would
                    compress everyone except the outliers to ~0 (black).
                    p99-normalize makes the bulk distribution span [0, 1] cleanly;
                    outliers saturate at 1.0 (still visible as bright).
                    """
                    diff = grad_5d_local.float().abs()  # [1, C, F, Hlat, Wlat]
                    diff_mean = diff.mean(dim=1)  # [1, F, Hlat, Wlat] — collapse channels
                    abs_max = float(diff_mean.max().item())
                    # 99th percentile per frame (NOT max — see docstring why)
                    F_local = diff_mean.shape[1]
                    flat_per_frame = diff_mean.flatten(start_dim=2)  # [1, F, H*W]
                    # torch.quantile is exact but slow; use kthvalue for speed
                    n_pixels = flat_per_frame.shape[-1]
                    k99 = max(1, int(n_pixels * 0.99))
                    p99, _ = flat_per_frame.kthvalue(k99, dim=-1, keepdim=True)  # [1, F, 1]
                    p99 = p99.unsqueeze(-1).clamp(min=1e-12)  # [1, F, 1, 1]
                    diff_norm = (diff_mean / p99).clamp(0, 1).cpu().numpy()  # [1, F, Hlat, Wlat]
                    # magma colormap → [1, F, Hlat, Wlat, 3]
                    colored = _cmap(diff_norm)[..., :3]
                    # Upscale spatially (Wan2.2 VAE = 16x) — nearest to keep
                    # the per-pixel heatmap structure crisp (bilinear blurs).
                    F_, Hlat, Wlat = F_local, colored.shape[2], colored.shape[3]
                    Hdec, Wdec = Hlat * _vae_spatial, Wlat * _vae_spatial
                    upsampled = _np.zeros((1, F_, Hdec, Wdec, 3), dtype=_np.uint8)
                    for f_idx in range(F_):
                        rgb_f = (colored[0, f_idx] * 255).astype(_np.uint8)
                        im = Image.fromarray(rgb_f, mode="RGB").resize(
                            (Wdec, Hdec), Image.NEAREST
                        )
                        upsampled[0, f_idx] = _np.asarray(im)
                    return upsampled.transpose(0, 4, 1, 2, 3), abs_max

                _diff_abs_max_global = 0.0  # for title bar
                for b in range(n_scenes):
                    # Per-scene 4-row block. Each is a [1, C, F, H, W] latent.
                    stu_5d   = stu_t[b:b+1]
                    fake_5d  = fake_t[b:b+1]
                    real_5d  = real_t[b:b+1]
                    # σ for this sample: take per-frame, format as min..max
                    sb = sigmas_t[b].flatten()  # [F]
                    sb_min = float(sb.min().item())
                    sb_max = float(sb.max().item())
                    sb_mean = float(sb.mean().item())
                    is_new_scene = (b > 0)

                    # Rows 1-3 (student, critic_x0, teacher_x0): VAE decode.
                    # Row 4 (HEATMAP): |dmd_grad| colormapped — the literal
                    # per-pixel push the student receives this DMD step.
                    for k, lat5 in enumerate([stu_5d, fake_5d, real_5d]):
                        u8 = _decode_latent(vae, lat5)
                        rows.append((s5_to_time_strip(u8, n_frames),
                                     is_new_scene and k == 0))
                    # Row 4: |dmd_grad| heatmap (or fallback to |gen−real|).
                    if grad_t is not None:
                        grad_5d = grad_t[b:b+1]
                    else:
                        # Backward-compat: pre-v3 slots lack dmd_grad. Fall
                        # back to |gen − real| so old runs still render.
                        grad_5d = (stu_5d.float() - real_5d.float())
                    heat_u8, heat_max = _make_heatmap_row(grad_5d)
                    _diff_abs_max_global = max(_diff_abs_max_global, heat_max)
                    rows.append((s5_to_time_strip(heat_u8, n_frames), False))

                # Mixed-frame batches → variable strip widths; pad to max.
                row_w = max(r.shape[1] for r, _ in rows)
                rows = [
                    (r if r.shape[1] == row_w else _np.pad(
                        r, ((0, 0), (0, row_w - r.shape[1]), (0, 0)),
                        constant_values=255), flag)
                    for r, flag in rows
                ]
                thin  = _np.full((2,  row_w, 3), 200, dtype=_np.uint8)
                thick = _np.full((12, row_w, 3), 255, dtype=_np.uint8)
                panel = []
                for i, (r, is_new_scene) in enumerate(rows):
                    if i > 0:
                        panel.append(thick if is_new_scene else thin)
                    panel.append(r)
                grid = _np.concatenate(panel, axis=0)

                # Title band on top: sigma + cos + safemask + step + heatmap-max
                from PIL import ImageDraw
                # ASCII-only title — PIL default font is latin-1 and raises
                # UnicodeEncodeError on Greek sigma. (Was σ; now 'sigma'.)
                title = (
                    f"DMD snapshot @ snap_step={snap_step}  (decode @ step={step_id})  |  "
                    f"sigma[min/mean/max]={sb_min:.3f}/{sb_mean:.3f}/{sb_max:.3f}  |  "
                    f"cos(g,p_real)={cos_val:+.3f}  |  safemask_zero={sm_zero:.3f}  |  "
                    f"|gen_x0|/|fake_x0|/|real_x0|={gen_amean:.2e}/{fak_amean:.2e}/{rea_amean:.2e}  |  "
                    f"raw_g_absmean={rg_amean:.2e}  normer_min={nmin:.2e}  |  "
                    f"heat_p99=|dmd_grad|_max={_diff_abs_max_global:.3e}"
                    + ("" if grad_t is not None else "  (FALLBACK: |gen-real|)")
                )
                title_h = 32
                title_band = _np.full((title_h, row_w, 3), 255, dtype=_np.uint8)
                title_img = Image.fromarray(title_band)
                draw = ImageDraw.Draw(title_img)
                draw.text((6, 8), title, fill=(0, 0, 0))
                grid = _np.concatenate([_np.asarray(title_img), grid], axis=0)

            # Step-subdir layout (added 2026-05-19):
            # debug_frames/step_{N:07d}/dmd_{a,b}__{TAG}.jpg
            _step_dir = _step_debug_dir(step_id)
            _suffix = f"_{grid_suffix}" if grid_suffix else ""
            _fname = f"dmd{_suffix}__{_launch_tag}.jpg"
            out_grid = os.path.join(_step_dir, _fname)
            tmp_dir = os.environ.get("DECODE_STAGE_DIR", "/tmp")
            tmp_grid = os.path.join(tmp_dir, f"step{step_id:07d}_{_fname}")
            Image.fromarray(grid).save(tmp_grid, quality=95, subsampling=0)
            shutil.move(tmp_grid, out_grid)

            _heat_label = "|dmd_grad|" if grad_t is not None else "|gen-real|(fallback)"
            cprint(f"[decode-dmd] step {step_id} (snap@{snap_step}) → {out_grid} "
                   f"({n_scenes} scenes x 4 rows [stu/fake_x0/real_x0/heatmap{_heat_label}p99] x "
                   f"{n_frames} frames | sigma_mean={s_mean:.2f} cos={cos_val:+.2f} "
                   f"sm0={sm_zero:.2f} diff_max={_diff_abs_max_global:.2e})",
                   "magenta")
            sys.stdout.flush()
        except Exception as e:
            import traceback
            cprint(f"[decode-dmd] WARNING: failed at step {step_id}: {e!r}", "red")
            if step_id <= args.decode_every * 2:
                traceback.print_exc()
        finally:
            # Mirror _decode_and_save_frame's VAE offload discipline so we
            # don't leave a ~3.6 GB fp32 VAE on GPU if a future train step
            # is tight on VRAM.
            try:
                if _vae_holder["vae"] is not None:
                    _vae_holder["vae"].to("cpu")
            except Exception:
                pass
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            sys.stdout.flush()

    def s5_last_to_hwc(u8_bcfhw):
        """Helper: pick the LAST RGB frame from the VAE's [B,3,Fout,H,W] uint8
        output and reshape to PIL-friendly [H, W, 3]. Used by the legacy
        decode_max_frames=1 path (single column per scene)."""
        return u8_bcfhw[0, :3, -1].transpose(1, 2, 0)   # [H, W, 3]

    def s5_to_time_strip(u8_bcfhw, n_frames: int, sep_px: int = 2):
        """Convert VAE uint8 output [B, 3, Fout, H, W] into a horizontal time
        strip [H, n_frames * W + sep_px * (n_frames-1), 3] by uniformly
        subsampling the time axis to `n_frames` and concatenating along width.

        Subsample policy: include first AND last frame, then evenly-spaced
        in between (`np.linspace`). For n_frames=1 returns just the last
        frame to match s5_last_to_hwc's behaviour. For n_frames >= Fout
        returns all frames. Thin (2-px) gray separators between frames make
        per-frame boundaries visually obvious without taking much space.
        """
        import numpy as _np
        Fout = u8_bcfhw.shape[2]
        if n_frames <= 1:
            return s5_last_to_hwc(u8_bcfhw)
        if n_frames >= Fout:
            idx = list(range(Fout))
        else:
            # Evenly-spaced including endpoints. linspace gives floats; round
            # to int and dedup-while-preserving-order to avoid sampling the
            # same frame twice when Fout / n_frames is small.
            raw = _np.linspace(0, Fout - 1, n_frames)
            seen = []
            for v in raw:
                i = int(round(float(v)))
                if i not in seen:
                    seen.append(i)
            idx = seen
        # Each frame: [H, W, 3]; insert a 2-px gray separator between them.
        H = u8_bcfhw.shape[3]
        sep = _np.full((H, sep_px, 3), 200, dtype=_np.uint8)
        tiles = []
        for k, i in enumerate(idx):
            if k > 0:
                tiles.append(sep)
            tiles.append(u8_bcfhw[0, :3, i].transpose(1, 2, 0))   # [H, W, 3]
        return _np.concatenate(tiles, axis=1)

    def _ode_teacher_latent_for(pair_idx):
        """Load just the x0_teacher tensor for pair `pair_idx` from disk and
        return it as a [1, C, F, H, W] tensor in fp32. Returns None on any
        failure (caller treats absence as 'just show student').

        DECODE-ONLY path: we keep teacher in fp32 (the dtype it was saved in
        by generate_ode_pairs.py line 247: `latents.float()`). Previously we
        cast to `dtype` (= bf16 when training in bf16), which round-tripped
        fp32→bf16→fp32 before the VAE forward and introduced visible
        per-channel quantization that the GroupNorm+temporal-upsample chain
        amplified into red/green blotches in the decoded teacher panel.
        `_decode_latent` casts to the VAE dtype at decode time anyway.
        The training loss (MSE) casts x0_teacher explicitly to fp32 at
        line 2041 so this change does not affect training semantics."""
        if ode_dataset is None or pair_idx is None or pair_idx < 0:
            return None
        try:
            pair = ode_dataset[pair_idx]
            return pair["x0_teacher"].unsqueeze(0).to(device=device, dtype=torch.float32)
        except Exception as e:
            cprint(f"[decode] could not load teacher for pair {pair_idx}: {e!r}", "red")
            return None

    def _ode_gt_latent_for(gt_idx):
        """B 方案 (added 2026-05-13): dual of `_ode_teacher_latent_for` for GT
        warmup mode. Returns the GT video_latent at `gt_idx` in the DMD
        `dataset` as a [1, C, F, H, W] fp32 tensor truncated to
        args.num_latent_frames. The decoded result is what MSE *should*
        push the student toward — visualizing it lets us eyeball that the
        GT VAE-encode/decode roundtrip itself is artifact-free (so any
        weirdness in the student row is a model problem, not a target
        problem).

        Returns None on failure (caller treats absence as 'student only')."""
        if gt_idx is None or gt_idx < 0:
            return None
        try:
            sample = dataset[gt_idx]
            gt = sample["video_latent"].unsqueeze(0).to(
                device=device, dtype=torch.float32
            )
            if (args.num_latent_frames > 0
                    and gt.shape[2] > args.num_latent_frames):
                gt = gt[:, :, :args.num_latent_frames]
            return gt
        except Exception as e:
            cprint(f"[decode] could not load GT latent for idx {gt_idx}: {e!r}", "red")
            return None

    def _ode_control_latent_for(gt_idx):
        """Return control_video_latent at gt_idx as [1, C, F, H, W] fp32,
        or None if dataset has no control data."""
        if gt_idx is None or gt_idx < 0:
            return None
        try:
            sample = dataset[gt_idx]
            ctrl = sample.get("control_video_latent", None)
            if ctrl is None:
                return None
            ctrl = ctrl.unsqueeze(0).to(device=device, dtype=torch.float32)
            if (args.num_latent_frames > 0
                    and ctrl.shape[2] > args.num_latent_frames):
                ctrl = ctrl[:, :, :args.num_latent_frames]
            return ctrl
        except Exception as e:
            cprint(f"[decode] could not load control latent for idx {gt_idx}: {e!r}", "red")
            return None

    def _pick_extra_scene_pair_indices(n: int, avoid: int, step_id: int):
        """Pick `n` distinct pair indices from this rank's ODE shard for extra
        debug scenes. Deterministic per step (so repeat decodes at the same
        step are reproducible) and excludes the live `avoid` index. Uses the
        SAME shard the main loop reads from, so we never accidentally decode
        a pair that this rank can't access via ode_dataset."""
        if ode_dataset is None or len(ode_dataset) <= 1:
            return []
        # Reproducible RNG keyed on step so two calls at the same step give
        # identical extras; different steps shuffle.
        import random as _random
        rng = _random.Random(step_id ^ 0x1234567)
        candidates = [i for i in range(len(ode_dataset)) if i != avoid]
        rng.shuffle(candidates)
        return candidates[:n]

    def _pick_extra_scene_gt_indices(n: int, avoid: int, step_id: int):
        """B 方案 dual of `_pick_extra_scene_pair_indices`: pick `n` distinct
        DMD-`dataset` indices for extra debug scenes during GT warmup. Same
        reproducible-shuffle scheme so repeat decodes at the same step give
        identical extras."""
        if dataset is None or len(dataset) <= 1:
            return []
        import random as _random
        rng = _random.Random(step_id ^ 0x1234567)
        candidates = [i for i in range(len(dataset)) if i != avoid]
        rng.shuffle(candidates)
        return candidates[:n]

    def _resolve_extras_with_fixed(n_extra: int, avoid: int, nonce: int, mode: str):
        """Mix fixed + random extra scene indices for decode.

        2026-05-26: support user-defined "fixed sample" set for cross-decode
        comparison. Fixed indices come from env DECODE_FIXED_SAMPLE_IDS
        (CSV, e.g. "0,1000,5000,20000,50000"). These same indices appear in
        EVERY decode trigger (deterministic) → directly compare student
        output evolution across step_0001040 / step_0001200 / ... mp4 files.
        Remaining slots filled by random picker (existing behavior).

        Layout in returned list:
            [random_extras..., fixed_extras...]
        Scene positions in the grid:
            scene_0           = live (always)
            scene_1..n_rand   = random extras
            scene_n_rand+1..  = fixed extras  ← compare these across decodes

        Args:
            n_extra: total extra scenes wanted (= num_scenes - 1, excluding live)
            avoid: live sample index to skip
            nonce: random seed for the random portion
            mode: 'pair' (ODE pair cache) or 'gt' (B+ v_flow path)
        """
        if mode == 'pair':
            picker = _pick_extra_scene_pair_indices
            ds_len = len(ode_dataset) if ode_dataset is not None else 0
        else:  # 'gt'
            picker = _pick_extra_scene_gt_indices
            ds_len = len(dataset) if dataset is not None else 0
        fixed_csv = os.environ.get("DECODE_FIXED_SAMPLE_IDS", "")
        fixed = []
        if fixed_csv and ds_len > 0:
            # 2026-05-27 fix: dataset is sharded per-rank at line 5428
            # (`dataset.samples[rank::world_size]`), so `ds_len` is the LOCAL
            # shard size (e.g. 116099 on 1n8g, 58050 on 16-card). User
            # passes "global" indices (e.g. 0,180000,360000,...) intuitively;
            # we modulo to local range so all indices fit. Trade-off: same
            # global idx maps to DIFFERENT physical samples across HW
            # configurations (8 vs 16 vs 32 cards), but within one run the
            # mapping is deterministic → cross-decode fixed comparison works.
            for tok in fixed_csv.split(","):
                tok = tok.strip()
                if not tok:
                    continue
                try:
                    i = int(tok)
                    if i < 0:
                        continue
                    local_idx = i % ds_len   # wrap to local shard range
                    if local_idx != avoid and local_idx not in fixed:
                        fixed.append(local_idx)
                except ValueError:
                    pass
        fixed = fixed[:n_extra]  # don't overflow
        n_random = max(0, n_extra - len(fixed))
        if n_random > 0:
            random_extras = picker(n_random, avoid, nonce)
            # Filter out any random extras that collide with fixed set
            random_extras = [r for r in random_extras if r not in fixed]
            random_extras = random_extras[:n_random]
        else:
            random_extras = []
        # 2026-05-27 diagnostic: log fixed/random split per call (rank 0 only,
        # to verify env propagation + scene assignment). Helps debug "why is
        # scene_5 not actually fixed" type issues. Cheap (1 print per decode).
        try:
            import torch.distributed as _dist
            _is_rank0 = (not _dist.is_initialized()) or _dist.get_rank() == 0
        except Exception:
            _is_rank0 = True
        if _is_rank0:
            cprint(
                f"[fixed-extras] mode={mode} avoid={avoid} ds_len={ds_len} "
                f"env='{fixed_csv[:60]}...' fixed={fixed} "
                f"random={random_extras}",
                "magenta",
            )
        return random_extras + fixed

    def _student_inference_for_pair(pair_idx: int):
        """Run a no-grad student streaming generation for the given ODE pair,
        using the SAME deterministic noise seed as the live training step
        would (so the extra scenes look exactly like what the student would
        produce in the next epoch when this pair comes up). Returns a
        [1, C, F, H, W] latent on the live dtype.

        Cost: one full streaming forward (~1-3s on H20). Times decode_num_scenes-1
        per decode call. With decode_every=50 and decode_num_scenes=4, this is
        +9s per 50 steps (~0.18s/step amortized) — well below the 4.5-8s/step
        of training itself, so a real-time impact <5%.
        """
        pair = ode_dataset[pair_idx]
        img_latent_x = pair["img_latent"].unsqueeze(0).to(device=device, dtype=dtype)
        text_embeds_x = pair["text_embeds"].to(device=device, dtype=dtype)
        if text_embeds_x.ndim == 2:
            text_embeds_x = text_embeds_x.unsqueeze(0)
        # Same deterministic seed scheme as the main ODE loop (search for
        # _pair_seed in this file). Keeps "what we visualize" == "what will
        # actually train next" for that pair.
        _seed = (pair_idx * 2654435761 + rank * 1315423911) & 0x7FFFFFFF
        local_rng = torch.Generator(device=device).manual_seed(_seed)
        return generate_streaming(
            generator=gen_engine.module,
            scheduler=scheduler,
            img_latent=img_latent_x,
            prompt_embeds=text_embeds_x,
            negative_prompt_embeds=None,
            num_latent_frames=args.num_latent_frames,
            num_inference_steps=int(os.environ.get("DECODE_NIS", str(args.num_inference_steps))),
            guidance_scale=1.0,
            do_cfg=False,
            generator_rng=local_rng,
            device=device,
            dtype=dtype,
            patch_size=gen_engine.module.config.patch_size,
            max_cache_frames=args.max_cache_frames,
            pe_mode=args.pe_mode,
        )

    def _student_inference_for_gt_idx(gt_idx: int):
        """B 方案 dual of `_student_inference_for_pair`: extra-scene student
        rollout from a DMD `dataset` index. Uses sample's first-frame as the
        condition image (same convention as the GT-warmup train loop), and
        the sample's pre-encoded prompt embeds (or re-encodes from prompt if
        the dataset didn't cache them). Same deterministic per-(idx, rank)
        seed so repeat decodes are reproducible.
        """
        sample = dataset[gt_idx]
        # Condition frame = first frame of the GT clip (same as warmup loop).
        gt_lat = sample["video_latent"].unsqueeze(0).to(device=device, dtype=dtype)
        if (args.num_latent_frames > 0
                and gt_lat.shape[2] > args.num_latent_frames):
            gt_lat = gt_lat[:, :, :args.num_latent_frames]
        img_latent_x = gt_lat[:, :, :1]
        # When num_latent_frames=0 (mixed-frame mode), use the sample's
        # actual frame count so generate_streaming loops over real frames
        # instead of producing only the condition frame.
        _nlf = args.num_latent_frames if args.num_latent_frames > 0 else gt_lat.shape[2]
        # Prefer pre-cached text_embeds, fall back to live encode.
        text_embeds_x = sample["text_embeds"]
        if text_embeds_x is None:
            text_embeds_x = encode_prompt(sample["prompt"])
        text_embeds_x = text_embeds_x.to(device=device, dtype=dtype)
        if text_embeds_x.ndim == 2:
            text_embeds_x = text_embeds_x.unsqueeze(0)
        # Deterministic seed mirrored from _student_inference_for_pair — note
        # that GT-warmup training does NOT seed manual_seed per step
        # (intentional, see warmup branch comments at L3286+); this seed
        # only governs the EXTRA-scene visualization, not training.
        _seed = (gt_idx * 2654435761 + rank * 1315423911) & 0x7FFFFFFF
        local_rng = torch.Generator(device=device).manual_seed(_seed)
        return generate_streaming(
            generator=gen_engine.module,
            scheduler=scheduler,
            img_latent=img_latent_x,
            prompt_embeds=text_embeds_x,
            negative_prompt_embeds=None,
            num_latent_frames=_nlf,
            num_inference_steps=int(os.environ.get("DECODE_NIS", str(args.num_inference_steps))),
            guidance_scale=1.0,
            do_cfg=False,
            generator_rng=local_rng,
            device=device,
            dtype=dtype,
            patch_size=gen_engine.module.config.patch_size,
            max_cache_frames=args.max_cache_frames,
            pe_mode=args.pe_mode,
            control_video_latent=(
                sample["control_video_latent"].unsqueeze(0).to(device=device, dtype=dtype)
                if args.use_control_dataset and "control_video_latent" in sample
                else None
            ),
        )

    def _teacher_inference_for_gt_idx(gt_idx: int):
        """Bidirectional UniPC 50-step denoise with the FROZEN base teacher
        (`real_score` = pristine Wan2.2-5B). Mirrors
        `scripts/generate_ode_pairs.py::_teacher_one_pair` exactly: single
        bidirectional forward per step + UniPC scheduler with
        flow_shift=1.0 + guidance_scale=1.0 (no cfg, no neg embeds).

        Single-transformer note: `Wan2.2-TI2V-5B-Diffusers/model_index.json`
        has `boundary_ratio: null` and `transformer_2: [null, null]` — this
        checkpoint is NOT the dual-transformer Wan2.2 variant. There is no
        low-noise stage to swap to. `_teacher_one_pair` honours this via
        `if pipe.config.boundary_ratio is not None and transformer_2 is not
        None: ...` (L310) — when boundary_ratio is null the boundary
        switch never fires and `current_model = transformer` for every
        timestep. We mirror that: only `real_score` is used for all 50
        steps.

        Rank-0-only; `real_score` lives on CPU between calls.
        """
        from diffusers import UniPCMultistepScheduler

        sample = dataset[gt_idx]
        gt_lat = sample["video_latent"].unsqueeze(0).to(device=device, dtype=dtype)
        if (args.num_latent_frames > 0
                and gt_lat.shape[2] > args.num_latent_frames):
            gt_lat = gt_lat[:, :, :args.num_latent_frames]
        img_latent_x = gt_lat[:, :, :1]
        text_embeds_x = sample["text_embeds"]
        if text_embeds_x is None:
            text_embeds_x = encode_prompt(sample["prompt"])
        text_embeds_x = text_embeds_x.to(device=device, dtype=dtype)
        if text_embeds_x.ndim == 2:
            text_embeds_x = text_embeds_x.unsqueeze(0)
        _seed = (gt_idx * 2654435761 + rank * 1315423911) & 0x7FFFFFFF
        local_rng = torch.Generator(device=device).manual_seed(_seed)

        teacher = real_score.to(device)
        try:
            p_t, p_h, p_w = teacher.config.patch_size
            _, n_ch, _, h_lat, w_lat = img_latent_x.shape
            n_f = args.num_latent_frames if args.num_latent_frames > 0 else gt_lat.shape[2]

            shape = (1, n_ch, n_f, h_lat, w_lat)
            latents = randn_tensor(shape, generator=local_rng,
                                   device=device, dtype=dtype)
            first_frame_mask = torch.ones(
                1, 1, n_f, h_lat, w_lat, dtype=dtype, device=device)
            first_frame_mask[:, :, 0] = 0
            condition = img_latent_x

            # UniPC 50-step. flow_shift=1.0 matches what every production
            # pair-gen wrapper uses (generate_ode_pairs_v2.sh default,
            # run_H_pair_gen_only.sh, etc — all explicitly set FLOW_SHIFT=1.0
            # to align with FlowWorld's inference-sft.py pipeline). The
            # 5.0 value is the diffusers config default but is NEVER what
            # downstream training sees in cached pair targets.
            sched_cfg = dict(scheduler.config)
            sched_cfg["flow_shift"] = 1.0
            t_sched = UniPCMultistepScheduler.from_config(sched_cfg)
            t_sched.set_timesteps(50, device=device)

            for t in t_sched.timesteps:
                latent_model_input = (1 - first_frame_mask) * condition \
                                     + first_frame_mask * latents
                t_dev = t.to(device=device)
                temp_ts = (first_frame_mask[0][0][:, ::p_h, ::p_w]
                           * t_dev).flatten()
                timestep = temp_ts.unsqueeze(0)
                noise_pred = teacher(
                    hidden_states=latent_model_input.to(dtype),
                    timestep=timestep,
                    encoder_hidden_states=text_embeds_x,
                    encoder_hidden_states_image=None,
                    attention_kwargs=None,
                    position_ids=None,
                    return_dict=False,
                )[0]
                latents = t_sched.step(noise_pred, t, latents,
                                       return_dict=False)[0]
            latents = (1 - first_frame_mask) * condition \
                      + first_frame_mask * latents
            out = latents.float()
        finally:
            teacher.to("cpu")
            torch.cuda.empty_cache()
        return out

    def _dmd_teacher_signal_for_gt_idx(gt_idx: int, t_value: int):
        """SINGLE-STEP real_score forward — what `compute_dmd_generator_loss`
        actually feeds the student each train step. Mirrors that function's
        sigma scheme exactly:
            sigma = t_value / num_train_timesteps   (linear, NO flow_shift)
            noisy = (1-sigma)*gt + sigma*noise
            v_pred = real_score(noisy, timestep=t_value, ...)
            x0_recon = noisy - sigma * v_pred       (implicit teacher target)

        Difference vs `_teacher_inference_for_gt_idx` (50-step UniPC):
          • 1 step instead of 50 → no accumulated drift; faithful to DMD path
          • Uses SAME `t / 1000` linear sigma DMD uses (NOT flow_shift=1 UniPC's
            *nearly*-linear schedule — close but not identical)
          • Returns x0 RECONSTRUCTION, not iteratively denoised x0

        first_frame_mask: keep the cond image (frame 0) clean — it's the same
        sink frame the student sees in training, and decoding a noise-perturbed
        cond image would mislead visual judgement of "did teacher get the
        first-frame appearance right".

        Rank-0-only; real_score lives on CPU between calls.
        """
        sample = dataset[gt_idx]
        gt_lat = sample["video_latent"].unsqueeze(0).to(device=device, dtype=dtype)
        if (args.num_latent_frames > 0
                and gt_lat.shape[2] > args.num_latent_frames):
            gt_lat = gt_lat[:, :, :args.num_latent_frames]
        text_embeds_x = sample["text_embeds"]
        if text_embeds_x is None:
            text_embeds_x = encode_prompt(sample["prompt"])
        text_embeds_x = text_embeds_x.to(device=device, dtype=dtype)
        if text_embeds_x.ndim == 2:
            text_embeds_x = text_embeds_x.unsqueeze(0)

        # Deterministic noise per (gt_idx, t_value) so repeated decodes are
        # comparable across steps.
        _seed = (gt_idx * 2654435761 + t_value * 1315423911) & 0x7FFFFFFF
        local_rng = torch.Generator(device=device).manual_seed(_seed)

        teacher = real_score.to(device)
        try:
            num_train_ts = scheduler.config.num_train_timesteps
            sigma = float(t_value) / float(num_train_ts)
            _, n_ch, n_f, h_lat, w_lat = gt_lat.shape
            p_t, p_h, p_w = teacher.config.patch_size

            noise = randn_tensor(gt_lat.shape, generator=local_rng,
                                 device=device, dtype=dtype)
            sigmas_5d = torch.full((1, 1, n_f, 1, 1), sigma,
                                   device=device, dtype=dtype)
            # First frame stays clean (matches _teacher_inference_for_gt_idx).
            first_frame_mask = torch.ones(1, 1, n_f, h_lat, w_lat,
                                          dtype=dtype, device=device)
            first_frame_mask[:, :, 0] = 0
            noisy = (1 - sigmas_5d) * gt_lat + sigmas_5d * noise
            noisy = (1 - first_frame_mask) * gt_lat + first_frame_mask * noisy

            # Per-token timestep: t_value for non-first frames, 0 for first
            # frame (matches first_frame_mask gating in _teacher_inference).
            temp_ts = (first_frame_mask[0][0][:, ::p_h, ::p_w]
                       * float(t_value)).flatten()
            timestep = temp_ts.unsqueeze(0)

            v_pred = teacher(
                hidden_states=noisy.to(dtype),
                timestep=timestep,
                encoder_hidden_states=text_embeds_x,
                encoder_hidden_states_image=None,
                attention_kwargs=None,
                position_ids=None,
                return_dict=False,
            )[0]

            # x0_recon = noisy - sigma * v_pred.
            # First-frame branch: stays at GT (sigmas_5d * v contribution
            # is masked out via first_frame_mask just like the 50-step path).
            x0_recon = noisy - sigmas_5d * v_pred
            x0_recon = (1 - first_frame_mask) * gt_lat + first_frame_mask * x0_recon
            out = x0_recon.float()
        finally:
            teacher.to("cpu")
            torch.cuda.empty_cache()
        return out

    # Holder so the ODE loop body can record "the pair we just trained on this
    # step" for the decoder hook to read. Lives across loop iterations; idx=-1
    # means "no live ODE pair this step" (e.g. DMD step).
    _current_pair_idx_holder = {"idx": -1}
    # B 方案 dual (added 2026-05-13): records the DMD-`dataset` index used by
    # the live GT-warmup step. -1 means "no live GT step this iter" (e.g.
    # teacher-mode warmup, or DMD step). Used by the debug decoder to look
    # up the matching GT video latent for the reference row.
    _current_gt_idx_holder = {"idx": -1}

    # ── Control add-plus running stats ──
    # EMA-tracked mean/std of skeleton (control) latent across training steps.
    # Used to normalize skeleton latent to match video latent distribution before
    # the model forward in add-plus mode. Without this, skeleton's mostly-white
    # background latent has very different mean/std from video latent, so
    # control_patch_embedding output overwhelms patch_embedding output.
    # Momentum 0.01 matches reference (slow EMA → stable with batch_size=1).
    # `mean` / `std` shape will be [1, C, 1, 1, 1] once initialized.
    control_running_stats = {
        "mean": None,   # initialized lazily on first ctrl-distill batch
        "std":  None,
        "count": 0,
        "momentum": 0.01,
    }

    # Try to load persisted control_running_stats from resume ckpt.
    # Note: use `read_dir` (absolute path; either stage_ckpt_dir on
    # local NVMe or ckpt_path on OSS) NOT `resume_from_ckpt` (basename like
    # "checkpoint-1000"). The basename would make os.path.exists() always False
    # since cwd is repo root, not the run dir → silent miss → DMD step 1
    # would run with un-normalized skel input that the MSE-trained CPE wasn't
    # calibrated for, biasing the first ~50-200 steps.
    if resume_from_ckpt:
        try:
            _stats_path = os.path.join(read_dir, "control_running_stats.bin")
            if os.path.exists(_stats_path):
                _stats_loaded = torch.load(
                    _stats_path, map_location="cpu", weights_only=False,
                )
                control_running_stats["mean"] = _stats_loaded["mean"].to(device)
                control_running_stats["std"] = _stats_loaded["std"].to(device)
                control_running_stats["count"] = int(_stats_loaded.get("count", 0))
                control_running_stats["momentum"] = float(
                    _stats_loaded.get("momentum", 0.01)
                )
                if is_main:
                    cprint(
                        f"[ctrl-stats] resumed from {_stats_path} "
                        f"(count={control_running_stats['count']}, "
                        f"mean.abs.mean={control_running_stats['mean'].abs().mean().item():.4f}, "
                        f"std.mean={control_running_stats['std'].mean().item():.4f})",
                        "cyan",
                    )
            else:
                if is_main:
                    cprint(
                        f"[ctrl-stats] no {_stats_path} found; running stats "
                        f"will lazy-init from first batch.",
                        "yellow",
                    )
        except Exception as _e:
            if is_main:
                cprint(
                    f"[ctrl-stats] resume load FAILED ({type(_e).__name__}: "
                    f"{_e!s:.150}); running stats will lazy-init from first batch.",
                    "yellow",
                )

    if resume_from_ckpt:
        cprint(f"Resumed at step {step}", "green")

    # numpy is used by both the ODE warmup path (if enabled) and the normal
    # DMD path. Import once here so neither path has to.
    import numpy as _np

    # ── ODE warmup dataset (opt-in) ──────────────────────────────────────
    # Independent of `dataset` (the DMD training dataset) when in 'teacher'
    # mode. If warmup is disabled OR we're already past the warmup step,
    # ode_dataset stays None so the fast path doesn't waste memory loading
    # unused pairs.
    #
    # B 方案 ('gt' mode, added 2026-05-13): ode_dataset stays None even
    # during warmup — the warmup branch below pulls from the DMD `dataset`
    # variable directly. This avoids loading two copies of the GT data
    # and keeps the per-epoch shuffle order in sync with what DMD will
    # see (different seed offset though, see _ode_indices_gt below).
    ode_dataset = None
    if (args.ode_warmup_steps > 0 and step < args.ode_warmup_steps
            and args.ode_target_kind in ("teacher", "causvid_traj")):
        if is_main:
            cprint(
                f"[ode] Loading ODE warmup pairs from {args.ode_pairs_dir} "
                f"(will train MSE for step {step} → {args.ode_warmup_steps}, "
                f"target_kind={args.ode_target_kind})",
                "cyan",
            )
        ode_dataset = ODEPairDataset(
            pairs_dir=args.ode_pairs_dir,
            # BUG FIX (2026-05-08): previously passed args.max_samples, which
            # conflated "DMD clip cap" with "ODE pair cap". With the default
            # MAX_SAMPLES=100 in train_distill_8.sh, every ODE-using run was
            # silently truncating 4000 cached pairs → 100 (then 13/rank on 8
            # GPUs), causing the generator to memorize a tiny 13-pair subset
            # and fail to learn anything generalizable. Separate knob now.
            max_samples=args.ode_max_samples,
            # Run D / CausVid §4.3 mode requires trajectory snapshots.
            require_trajectory=(args.ode_target_kind == "causvid_traj"),
        )
        # Shard pairs across ranks, same convention as `dataset`.
        ode_dataset.pairs = ode_dataset.pairs[rank::world_size]
        if is_main:
            cprint(f"[ode] ODE pairs this rank: {len(ode_dataset)}", "green")
        if len(ode_dataset) == 0:
            raise RuntimeError(
                f"ODE dataset empty on rank {rank} (pairs_dir={args.ode_pairs_dir}). "
                f"Reduce --num_processes or generate more pairs."
            )
    elif (args.ode_warmup_steps > 0 and step < args.ode_warmup_steps
          and args.ode_target_kind == "gt"):
        if is_main:
            cprint(
                f"[ode] B方案 GT-warmup mode: reusing DMD dataset "
                f"({len(dataset)} samples this rank) as MSE target source. "
                f"No pair cache loaded; will warmup for step {step} → "
                f"{args.ode_warmup_steps}.",
                "cyan",
            )
    elif (args.ode_warmup_steps > 0 and step < args.ode_warmup_steps
          and args.ode_target_kind == "v_flow"):
        if is_main:
            cprint(
                f"[ode] B+方案 v-flow-warmup mode: single-step v-space flow matching on GT. "
                f"Reusing DMD dataset ({len(dataset)} samples this rank) as MSE target "
                f"source. No pair cache loaded; will warmup for step {step} → "
                f"{args.ode_warmup_steps}.",
                "cyan",
            )

    def _ode_indices(epoch_idx: int):
        """Same per-epoch shuffled indices as _epoch_indices, but over the
        ODE pair dataset. We deliberately use a DIFFERENT seed offset so
        pair order doesn't accidentally correlate with the DMD dataset order.
        """
        n = len(ode_dataset)
        rng = _np.random.RandomState(args.seed + 1000 * rank + epoch_idx + 7919)
        return rng.permutation(n).tolist()

    def _ode_indices_gt(epoch_idx: int):
        """B 方案 helper: per-epoch shuffled indices over the DMD `dataset`,
        but seeded differently from `_epoch_indices` so the ODE warmup
        traversal order is decorrelated from the DMD-phase traversal order.
        Same offset (+7919) as `_ode_indices` for symmetry — both
        warmup-phase shufflers diverge from DMD's by the same hash.
        """
        n = len(dataset)
        rng = _np.random.RandomState(args.seed + 1000 * rank + epoch_idx + 7919)
        return rng.permutation(n).tolist()

    # Per-epoch shuffled index order. Original code did `step % len(dataset)`
    # which traverses the shard in a fixed order forever — every epoch sees
    # the same sequence and the critic over-indexes the early samples.
    # Use a deterministic per-epoch RNG seeded by (seed, rank, epoch) so
    # each rank shuffles its own shard independently and a resumed run
    # follows the same trajectory.
    def _epoch_indices(epoch_idx: int):
        rng = _np.random.RandomState(args.seed + 1000 * rank + epoch_idx)
        return rng.permutation(len(dataset)).tolist()

    # ── ODE warmup non-finite-loss tracking (Layer 3 fail-fast) ───────
    # Layer 1 (clamp before MSE) below fixes ~all real cases. Layer 2
    # (skip backward if STILL non-finite) is a paranoia net. Layer 3
    # (this counter) catches the pathological case where Layer 1 + 2
    # don't help and we'd otherwise burn hours on a stuck run.
    #
    # Tunable via env STREAM_MAX_CONSEC_NONFINITE (default 50):
    # if 50 consecutive ODE steps produce non-finite loss EVEN AFTER
    # clamp, abort with a clear error. Realistic threshold: a healthy
    # cold-start hits ≤2-3 non-finite steps in the first ~20 iters,
    # never more after that. 50 in a row means model architecture or
    # data is broken, not just transient noise.
    _ode_nonfinite_consec = 0
    _ode_nonfinite_total = 0
    _ode_max_consec_nonfinite = int(
        os.environ.get("STREAM_MAX_CONSEC_NONFINITE", "50")
    )

    while step < args.num_train_steps:
        step_start = time.time()

        # ── ODE warmup branch (Stage-0) ─────────────────────────────────
        # For the first `args.ode_warmup_steps` steps, train the generator
        # via MSE against teacher-cached x0 targets instead of DMD. The
        # critic and real_score are completely untouched during warmup.
        # After warmup we fall through to the normal DMD path below.
        # Refactor note: this branch ONLY does (sample → forward → loss →
        # backward → step → EMA → log_dict). It does NOT touch step+=1,
        # logging, decoding, or saving — those happen at the bottom of
        # the loop and are shared between the ODE and DMD paths so a
        # warmup checkpoint reuses every battle-tested code path
        # (FUSE staging, atomic moves, EMA save, …).
        # is_warmup_step: True iff we're inside the warmup window AND the
        # configured target source is loaded/available.
        # - 'teacher' mode: depends on ode_dataset being non-None
        # - 'gt'      mode (B 方案):  depends only on the step counter,
        #                             since we pull from the DMD `dataset` directly
        # - 'v_flow'  mode (B+ 方案): same as gt — pulls from DMD `dataset`,
        #                             differs only in loss formulation (v-space)
        is_warmup_step = (
            step < args.ode_warmup_steps and (
                ode_dataset is not None
                or args.ode_target_kind == "gt"
                or args.ode_target_kind == "v_flow"
            )
        )

        if is_warmup_step:
            # Common var: patch_size for the streaming generator (same in
            # both branches). Pulled before the if/else so both branches
            # can use it without duplicating the line.
            gen_patch_size = gen_engine.module.config.patch_size

            # Mode flag set per-iter; consumed by the shared MSE / Layer 1
            # block below to short-circuit x_0-space-specific code paths
            # (clamp, x0_orig finite-element scan) that are inapplicable
            # when the loss lives in v-space OR when the student output is
            # clean by construction (causvid_traj's x_0 reconstruction from
            # ODE-snapshot input is already in clean-x_0 distribution).
            _is_v_flow_mode = (args.ode_target_kind == "v_flow")
            _is_causvid_traj_mode = (args.ode_target_kind == "causvid_traj")
            # Skip Layer-1 clamp for both modes — they don't go through the
            # σ-amplified x_0 reconstruction the legacy path needed (where
            # clamp protected against `noisy/σ → ∞` overflows at σ→0).
            _skip_x0_clamp = _is_v_flow_mode or _is_causvid_traj_mode

            if args.ode_target_kind == "teacher":
                # ===== A 方案 (Self-Forcing/CausVid recipe) =====
                # MSE target = pre-cached x0_teacher from pair file.
                epoch_o = step // max(1, len(ode_dataset))
                order_o = _ode_indices(epoch_o)
                idx_o = order_o[step % len(ode_dataset)]
                # Record for the debug decode hook (only meaningful in ODE
                # warmup; DMD path resets to -1 below). Used to label the
                # live scene and to look up the matching teacher x0 for
                # side-by-side comparison.
                _current_pair_idx_holder["idx"] = idx_o
                # Defensive: scrub the GT-mode holder so a teacher-mode run
                # never accidentally pulls a GT row (would happen if a prior
                # B-mode iter left a stale value, but we now run fully
                # mutually exclusive — this is just belt-and-braces).
                _current_gt_idx_holder["idx"] = -1
                pair = ode_dataset[idx_o]
                # Shapes from generate_ode_pairs.py:
                #   noise:       [C, F, H, W]       — per-pair unique random noise
                #   x0_teacher:  [C, F, H, W]       — teacher 4-step output
                #   img_latent:  [C, 1, H, W]       — condition frame
                #   text_embeds: [seq, 4096]        — prompt embedding used by teacher
                x0_teacher = pair["x0_teacher"].unsqueeze(0).to(device=device, dtype=torch.float32)
                img_latent_ode = pair["img_latent"].unsqueeze(0).to(device=device, dtype=dtype)
                # text_embeds saved as [seq, dim] in pair file; generator expects
                # [B, seq, dim] (matching the DMD path where the dataset already
                # returns batched text_embeds). Unsqueeze if it lost the batch dim.
                text_embeds_ode = pair["text_embeds"].to(device=device, dtype=dtype)
                if text_embeds_ode.ndim == 2:
                    text_embeds_ode = text_embeds_ode.unsqueeze(0)

                # Re-seed generator_rng deterministically from the pair index
                # so the student sees the SAME noise the teacher did at pair
                # generation time (pair-dependent integer × Knuth's golden-ratio
                # multiplicative-hash constant 2654435761, masked into int32 to
                # stay within manual_seed's accepted range).
                # Use pure Python arithmetic + bit-mask instead of np.uint32(...)
                # to dodge numpy's DeprecationWarning about out-of-bound casts.
                _pair_seed = (idx_o * 2654435761 + rank * 1315423911) & 0x7FFFFFFF
                generator_rng.manual_seed(_pair_seed)

                # ODE pair frame count is the source of truth — pairs were
                # generated at one fixed length (likely 21F), and student
                # output must match that for the MSE loss to make sense.
                # Falls back to args.num_latent_frames if the pair lacks an
                # x0_teacher tensor (shouldn't happen, but defensive).
                ode_F = (
                    pair["x0_teacher"].shape[1] if "x0_teacher" in pair
                    else args.num_latent_frames
                )
                # Common alias used by the shared MSE block below.
                target_for_mse = x0_teacher

            elif args.ode_target_kind == "causvid_traj":
                # ===== Run D: CausVid §4.3 ODE-regression =====
                # Same pair loading as teacher mode, but the forward path is
                # streaming_causvid_traj_forward (per-frame snapshot gather +
                # x_0 MSE vs trajectory[-1]). See pair-gen v2 with
                # --save_trajectory.
                epoch_o = step // max(1, len(ode_dataset))
                order_o = _ode_indices(epoch_o)
                idx_o = order_o[step % len(ode_dataset)]
                _current_pair_idx_holder["idx"] = idx_o
                _current_gt_idx_holder["idx"] = -1

                # ── Aggressive pair load with retry + FUSE cache drop ──
                # OSS-FUSE transient read bit-flip on H100 yields fp32
                # finite-but-absurd values (e.g. 9.66e+35, 1.84e+37). Disk
                # data IS healthy (verified locally, 3-trial reload all
                # consistent). The bit-flip happens during the OSS→FUSE
                # network/RAM transfer; FUSE may then cache the corrupt
                # bytes for some TTL window. Retry is only effective if
                # the kernel re-fetches fresh bytes, so we explicitly drop
                # the page cache between attempts via posix_fadvise().
                #
                # Strategy:
                #   1. Try load + verify (isfinite + absmax<=100)
                #   2. If bad: drop FUSE page cache, sleep, retry
                #   3. Up to 10 attempts (was 3 — bumped 2026-05-18 after
                #      observing 43% SKIP rate without aggressive retry)
                #   4. If all 10 fail: double guard below SKIPs world-wide
                #
                # Detailed logging on every retry so post-mortem can grep
                # frequency / success rate / ultimate failures.
                # See [[feedback-gt-guard-needs-absmax-threshold]].
                _max_pair_load_retries = 10
                _pair_load_attempt = 0
                _pair_local_ok = False
                _pair_retry_t0 = time.time()
                _pair_retry_log: list[tuple[int, float, str]] = []
                while _pair_load_attempt < _max_pair_load_retries and not _pair_local_ok:
                    _pair_load_attempt += 1
                    _attempt_t0 = time.time()
                    pair = ode_dataset[idx_o]
                    _attempt_load_ms = (time.time() - _attempt_t0) * 1000

                    # CPU-side sanity check (before .to(device) waste).
                    _local_traj = pair["x0_trajectory"]
                    _local_gt = pair["gt_video_latent"]
                    _local_sig = pair["traj_sigmas"]
                    _traj_abs_local = float(_local_traj.abs().max().item())
                    _gt_abs_local = float(_local_gt.abs().max().item())
                    _sig_abs_local = float(_local_sig.abs().max().item())
                    _traj_finite = bool(torch.isfinite(_local_traj).all().item())
                    _gt_finite = bool(torch.isfinite(_local_gt).all().item())
                    _sig_finite = bool(torch.isfinite(_local_sig).all().item())

                    if (_traj_finite and _gt_finite and _sig_finite
                            and _traj_abs_local <= 100.0
                            and _gt_abs_local <= 100.0
                            and _sig_abs_local <= 1.5):
                        _pair_local_ok = True
                        _pair_retry_log.append(
                            (_pair_load_attempt, _attempt_load_ms, "ok")
                        )
                    else:
                        # Build human-readable failure reason.
                        if not _traj_finite:
                            _reason = f"x0_trajectory non-finite"
                        elif not _gt_finite:
                            _reason = f"gt_video_latent non-finite"
                        elif not _sig_finite:
                            _reason = f"traj_sigmas non-finite"
                        elif _traj_abs_local > 100.0:
                            _reason = f"x0_trajectory absmax={_traj_abs_local:.3e}"
                        elif _gt_abs_local > 100.0:
                            _reason = f"gt_video_latent absmax={_gt_abs_local:.3e}"
                        else:
                            _reason = f"traj_sigmas absmax={_sig_abs_local:.3e}"
                        _pair_retry_log.append(
                            (_pair_load_attempt, _attempt_load_ms, _reason)
                        )

                        # Log every retry on rank 0 (was: only first).
                        if is_main:
                            cprint(
                                f"[ode-retry] step {step} | rank {rank} pair idx={idx_o} "
                                f"attempt {_pair_load_attempt}/{_max_pair_load_retries} "
                                f"BAD ({_reason}, "
                                f"traj_abs={_traj_abs_local:.3e}, "
                                f"gt_abs={_gt_abs_local:.3e}, "
                                f"sig_abs={_sig_abs_local:.3e}); "
                                f"dropping FUSE cache + sleep before retry",
                                "yellow",
                            )

                        # Drop references first (free memory before fadvise).
                        _pair_path_for_drop = ode_dataset.pairs[idx_o]
                        del _local_traj, _local_gt, _local_sig, pair

                        # Hint FUSE to drop page cache for this file → next
                        # load_file forces fresh fetch from OSS backend.
                        try:
                            _drop_fd = os.open(_pair_path_for_drop, os.O_RDONLY)
                            try:
                                os.posix_fadvise(
                                    _drop_fd, 0, 0, os.POSIX_FADV_DONTNEED
                                )
                            finally:
                                os.close(_drop_fd)
                        except OSError as _e:
                            if is_main:
                                cprint(
                                    f"[ode-retry] step {step} | fadvise(DONTNEED) "
                                    f"failed: {_e} (continuing without cache drop)",
                                    "yellow",
                                )

                        # Backoff: 0.2s × attempt → max 2s. Lets any
                        # in-flight FUSE prefetch finish before we re-read.
                        _backoff_s = min(0.2 * _pair_load_attempt, 2.0)
                        time.sleep(_backoff_s)

                _pair_retry_total_ms = (time.time() - _pair_retry_t0) * 1000

                # ── Per-step retry log + global aggregate counter ──
                # Init aggregate stats lazily on first step.
                if not hasattr(args, "_ode_retry_stats"):
                    args._ode_retry_stats = {
                        "steps_seen": 0,
                        "steps_with_retry": 0,
                        "steps_recovered_via_retry": 0,
                        "steps_ultimate_failure": 0,
                        "total_attempts": 0,
                        "max_attempts_seen": 0,
                        "histogram_attempts": [0] * (_max_pair_load_retries + 1),
                        "last_print_step": 0,
                    }
                _stats = args._ode_retry_stats
                _stats["steps_seen"] += 1
                _stats["total_attempts"] += _pair_load_attempt
                if _pair_load_attempt > _stats["max_attempts_seen"]:
                    _stats["max_attempts_seen"] = _pair_load_attempt
                _stats["histogram_attempts"][_pair_load_attempt] += 1
                if _pair_load_attempt > 1:
                    _stats["steps_with_retry"] += 1
                if _pair_local_ok and _pair_load_attempt > 1:
                    _stats["steps_recovered_via_retry"] += 1
                    if is_main:
                        cprint(
                            f"[ode-retry] step {step} | rank {rank} pair idx={idx_o} "
                            f"RECOVERED on attempt {_pair_load_attempt}/"
                            f"{_max_pair_load_retries} "
                            f"(total {_pair_retry_total_ms:.0f}ms; "
                            f"trail: {_pair_retry_log})",
                            "green",
                        )
                if not _pair_local_ok:
                    _stats["steps_ultimate_failure"] += 1
                    if is_main:
                        cprint(
                            f"[ode-retry] step {step} | rank {rank} pair idx={idx_o} "
                            f"ULTIMATE FAILURE after {_max_pair_load_retries} attempts "
                            f"(total {_pair_retry_total_ms:.0f}ms); "
                            f"falling through to allreduce(MAX) vote SKIP "
                            f"(trail: {_pair_retry_log})",
                            "red",
                        )

                # Aggregate summary every 50 steps (rank 0 only).
                if is_main and (step - _stats["last_print_step"]) >= 50:
                    _stats["last_print_step"] = step
                    _seen = _stats["steps_seen"]
                    _retry_pct = 100 * _stats["steps_with_retry"] / max(_seen, 1)
                    _recover_pct = 100 * _stats["steps_recovered_via_retry"] / max(_seen, 1)
                    _ultfail_pct = 100 * _stats["steps_ultimate_failure"] / max(_seen, 1)
                    _avg_attempts = _stats["total_attempts"] / max(_seen, 1)
                    _hist_str = ", ".join(
                        f"{i}x={_stats['histogram_attempts'][i]}"
                        for i in range(1, _max_pair_load_retries + 1)
                        if _stats["histogram_attempts"][i] > 0
                    )
                    cprint(
                        f"[ode-retry-stats] step {step} | over {_seen} steps: "
                        f"{_retry_pct:.1f}% had retries, "
                        f"{_recover_pct:.1f}% recovered via retry, "
                        f"{_ultfail_pct:.1f}% ultimate failure (SKIP), "
                        f"avg attempts={_avg_attempts:.2f}, "
                        f"max seen={_stats['max_attempts_seen']}, "
                        f"hist=[{_hist_str}]",
                        "cyan",
                    )

                # If still not OK after retries, _pair_local_ok=False → the
                # double guard below will detect it on this rank, allreduce
                # vote, and SKIP world-wide.

                # Trajectory + sigmas: shapes [N+1, C, F, H, W] and [N+1]
                x0_trajectory = pair["x0_trajectory"].to(
                    device=device, dtype=torch.float32
                )
                traj_sigmas = pair["traj_sigmas"].to(
                    device=device, dtype=torch.float32
                )
                img_latent_ode = pair["img_latent"].unsqueeze(0).to(
                    device=device, dtype=dtype
                )
                text_embeds_ode = pair["text_embeds"].to(device=device, dtype=dtype)
                if text_embeds_ode.ndim == 2:
                    text_embeds_ode = text_embeds_ode.unsqueeze(0)
                # Same deterministic seed as teacher mode for parity.
                _pair_seed = (idx_o * 2654435761 + rank * 1315423911) & 0x7FFFFFFF
                generator_rng.manual_seed(_pair_seed)
                # F from trajectory[-1] (= clean final, shape [C, F, H, W])
                ode_F = x0_trajectory.shape[2]
                # gt_latent — REAL GT video (用户选择 B 哲学:target = 真 GT
                # 而不是 teacher self-distill 的 trajectory[-1]). The pair file
                # carries `gt_video_latent` from the source EPIC video latent.
                # Used both as:
                #   - teacher-forcing prefix source (cache writes clean GT[0..f-1])
                #   - MSE target (passed to streaming_causvid_traj_forward and used
                #     as x0_target_full = trajectory[-1]'s replacement)
                gt_latent = pair["gt_video_latent"].unsqueeze(0).to(
                    device=device, dtype=torch.float32
                )  # [1, C, F, H, W]
                target_for_mse = gt_latent  # alias for shared logging

                # ── Pair-file corruption guard (added 2026-05-18, Run D fix) ──
                # OSS-FUSE bit-flip can corrupt pair tensors with values like
                # 9.66e+35 (fp32 finite-but-absurd). Run D failed at step 52
                # on H100 because Layer 2 SKIP came too late (cache already
                # poisoned by step 53 → NCCL 30min timeout → SIGABRT).
                # Same double-guard pattern as v_flow/gt branches: check
                # isfinite + absmax<=100 on the three pair tensors we'll
                # actually consume (trajectory snapshots, traj_sigmas, and the
                # REAL GT target), then allreduce(max) so all ranks vote
                # before the first collective in the forward.
                # See [[feedback-gt-guard-needs-absmax-threshold]] memory.
                _traj_abs = float(x0_trajectory.abs().max().item())
                _gt_abs = float(gt_latent.abs().max().item())
                _sigma_abs = float(traj_sigmas.abs().max().item())
                _pair_bad = torch.tensor(
                    [0 if (bool(torch.isfinite(x0_trajectory).all().item())
                           and bool(torch.isfinite(gt_latent).all().item())
                           and bool(torch.isfinite(traj_sigmas).all().item())
                           and _traj_abs <= 100.0
                           and _gt_abs <= 100.0
                           and _sigma_abs <= 1.5) else 1],
                    device=device, dtype=torch.int32,
                )
                if dist.is_initialized():
                    dist.all_reduce(_pair_bad, op=dist.ReduceOp.MAX)
                if int(_pair_bad.item()) > 0:
                    if is_main:
                        cprint(
                            f"[ode] step {step} | pair file corrupted on at "
                            f"least one rank (this rank: idx={idx_o}, "
                            f"trajectory absmax={_traj_abs:.2e}, "
                            f"gt_video_latent absmax={_gt_abs:.2e}, "
                            f"traj_sigmas absmax={_sigma_abs:.2e}); "
                            f"SKIPPING world-wide, no param update.",
                            "red",
                        )
                    del x0_trajectory, traj_sigmas, gt_latent, pair, _pair_bad
                    log_dict.update({
                        "gen_loss": float("nan"),
                        "crit_loss": 0.0,
                        "gen_grad_norm": float("nan"),
                        "gen_latent_std": float("nan"),
                        "gen_latent_mean": float("nan"),
                        "gen_latent_absmax": float("nan"),
                    })
                    step += 1
                    continue
                del _pair_bad

            elif args.ode_target_kind == "gt":
                # ===== B 方案 (added 2026-05-13): GT-warmup =====
                # MSE target = full GT video latent from the same
                # LatentDataset that the DMD phase consumes. No pair cache;
                # noise is sampled fresh per-step from generator_rng (which
                # already has a per-rank seed). Critic / real_score are
                # untouched, identical to the teacher branch.
                #
                # WHY no manual_seed re-seed: in 'teacher' mode the seed
                # ensures the student probes the SAME noise → x0 mapping
                # the teacher saw, so MSE is meaningful. In 'gt' mode the
                # teacher isn't in the loop, so locking to a specific
                # noise per (idx, rank) tuple buys us nothing — letting
                # generator_rng evolve naturally exposes the student to
                # noise diversity even within a single GT clip's training
                # iterations.
                epoch_o = step // max(1, len(dataset))
                order_o = _ode_indices_gt(epoch_o)
                idx_o = order_o[step % len(dataset)]
                # debug decode hook (B 方案):
                #   pair_idx = -1 → "no teacher pair" (correct: B-mode has none)
                #   gt_idx   = idx_o → records which GT clip the live training
                #              step is using, so the decoder can pull the
                #              matching GT video latent for the reference row.
                # The teacher-mode warmup branch above sets pair_idx≥0 and
                # leaves gt_idx at -1; the DMD branch below resets both to -1.
                _current_pair_idx_holder["idx"] = -1
                _current_gt_idx_holder["idx"] = idx_o
                sample = dataset[idx_o]
                # video_latent shape: [C, F_native, H, W]; promote to fp32
                # for the MSE math, same as x0_teacher in the teacher branch.
                gt_latent = sample["video_latent"].unsqueeze(0).to(
                    device=device, dtype=torch.float32
                )
                # GT-corruption guard. See L4393-4421 (v_flow branch) for the
                # full rationale; same FUSE/OSS bit-flip risk applies here.
                # Detect TWO failure modes:
                #   (a) inf/nan tokens (~isfinite)
                #   (b) "finite but absurdly large" — observed in production:
                #       FUSE bit-flip yields absmax=9.66e+35, which is fp32-
                #       representable so isfinite()==True but still poisons
                #       v_target = noise - GT. Healthy VAE-encoded GT latents
                #       have absmax <= ~20; threshold at 100 to give 5x
                #       headroom against any data-augmentation outlier.
                _gt_abs = float(gt_latent.abs().max().item())
                _gt_bad = torch.tensor(
                    [0 if (bool(torch.isfinite(gt_latent).all().item())
                           and _gt_abs <= 100.0) else 1],
                    device=device, dtype=torch.int32,
                )
                if dist.is_initialized():
                    dist.all_reduce(_gt_bad, op=dist.ReduceOp.MAX)
                if int(_gt_bad.item()) > 0:
                    if is_main:
                        _local_bad = (~torch.isfinite(gt_latent)).sum().item()
                        cprint(
                            f"[ode] step {step} | GT latent corrupted on at "
                            f"least one rank (this rank: idx={idx_o}, "
                            f"{int(_local_bad)} non-finite/{gt_latent.numel()}, "
                            f"absmax={_gt_abs:.2e}); "
                            f"SKIPPING world-wide, no param update.",
                            "red",
                        )
                    del gt_latent, sample, _gt_bad
                    log_dict.update({
                        "gen_loss": float("nan"),
                        "crit_loss": 0.0,
                        "gen_grad_norm": float("nan"),
                        "gen_latent_std": float("nan"),
                        "gen_latent_mean": float("nan"),
                        "gen_latent_absmax": float("nan"),
                    })
                    step += 1
                    continue
                del _gt_bad
                # Truncate to the args.num_latent_frames cap. This matches
                # what the DMD branch does (see L3334 region) so warmup and
                # DMD see clips of the same length distribution.
                if (args.num_latent_frames > 0
                        and gt_latent.shape[2] > args.num_latent_frames):
                    gt_latent = gt_latent[:, :, :args.num_latent_frames]
                # Condition frame = first frame of the GT clip (same
                # convention as the teacher branch and the DMD branch).
                # Cast back to the model dtype (bf16) for the forward pass.
                img_latent_ode = gt_latent[:, :, :1].to(dtype=dtype)
                # Text embeds: prefer the pre-encoded ones in the latent
                # file; fall back to encoding the prompt on-the-fly.
                text_embeds_ode = sample["text_embeds"]
                if text_embeds_ode is None:
                    text_embeds_ode = encode_prompt(sample["prompt"])
                text_embeds_ode = text_embeds_ode.to(device=device, dtype=dtype)
                if text_embeds_ode.ndim == 2:
                    text_embeds_ode = text_embeds_ode.unsqueeze(0)

                ode_F = gt_latent.shape[2]
                target_for_mse = gt_latent

            else:
                # ===== B+ 方案 (added 2026-05-13): v-space flow-matching =====
                # warmup. SAME data source as 'gt' (LatentDataset → GT video
                # latents from --data_path), but DIFFERENT loss formulation:
                #
                #   gt mode:     MSE(x0_student_4step, gt_latent)  (x0-space)
                #   v_flow mode: MSE(v_pred,    noise - gt_latent) (v-space)
                #
                # Why v-space fixes the cold-start NaN observed in 'gt' mode:
                #   1. Loss target is bounded O(1) regardless of σ (no
                #      reconstruction-via-σ amplification);
                #   2. Input is (1-σ)·GT + σ·noise (NOT pure noise) — cold
                #      attention always sees a GT-anchored input;
                #   3. 1 grad-bearing forward per frame (no 4-step feedback
                #      loop where one outlier self-amplifies across denoise
                #      steps).
                # See streaming_v_flow_matching_forward docstring for full
                # rationale. Critic/real_score untouched, same as gt branch.
                #
                # Data-loading is bit-identical to 'gt' branch — keep the
                # logic inline rather than DRY-ing because 'gt' may diverge
                # later (e.g. seed-locking, augmentations) and we want each
                # warmup recipe self-contained.
                epoch_o = step // max(1, len(dataset))
                order_o = _ode_indices_gt(epoch_o)
                idx_o = order_o[step % len(dataset)]
                _current_pair_idx_holder["idx"] = -1
                _current_gt_idx_holder["idx"] = idx_o
                sample = dataset[idx_o]
                gt_latent = sample["video_latent"].unsqueeze(0).to(
                    device=device, dtype=torch.float32
                )
                # Control distill: extract control_video_latent from sample
                # (None for vanilla LatentDataset, present for EgoVerseControlDataset).
                control_latent_ode = sample.get("control_video_latent", None)
                if control_latent_ode is not None:
                    control_latent_ode = control_latent_ode.unsqueeze(0).to(
                        device=device, dtype=dtype,
                    )
                # GT-corruption guard. FUSE/OSS occasionally yields a
                # .safetensors with bit-flipped pages → inf/nan in the latent.
                # That propagates into v_target = noise - GT, overflows MSE to
                # inf, then gets caught by Layer 2 SKIP — but Layer 2 still
                # does a 30s forward AND the inf loss tensor gets allreduced
                # via ZeRO-3, briefly poisoning every rank's view. Catch it
                # BEFORE forward + sync the decision across ranks so the
                # WHOLE world skips this sample together (otherwise NCCL
                # collectives mismatch: 7 ranks running forward, 1 rank
                # waiting at the next iter's barrier).
                #
                # Per-rank dataset shards differ (dataset.samples[rank::W])
                # AND per-rank shuffle is rank-seeded, so a corrupt file on
                # one rank may not show up on others — must allreduce.
                # See L4327 gt-mode guard for the absmax-threshold rationale —
                # 9.66e+35 is fp32-finite, so isfinite() alone is insufficient.
                _gt_abs = float(gt_latent.abs().max().item())
                _gt_bad = torch.tensor(
                    [0 if (bool(torch.isfinite(gt_latent).all().item())
                           and _gt_abs <= 100.0) else 1],
                    device=device, dtype=torch.int32,
                )
                if dist.is_initialized():
                    dist.all_reduce(_gt_bad, op=dist.ReduceOp.MAX)
                if int(_gt_bad.item()) > 0:
                    if is_main:
                        _local_bad = (~torch.isfinite(gt_latent)).sum().item()
                        cprint(
                            f"[ode] step {step} | GT latent corrupted on at "
                            f"least one rank (this rank: idx={idx_o}, "
                            f"{int(_local_bad)} non-finite/{gt_latent.numel()}, "
                            f"absmax={_gt_abs:.2e}); "
                            f"SKIPPING world-wide, no param update.",
                            "red",
                        )
                    del gt_latent, sample, _gt_bad
                    # Replicate the housekeeping the healthy iter would do.
                    log_dict.update({
                        "gen_loss": float("nan"),
                        "crit_loss": 0.0,
                        "gen_grad_norm": float("nan"),
                        "gen_latent_std": float("nan"),
                        "gen_latent_mean": float("nan"),
                        "gen_latent_absmax": float("nan"),
                    })
                    step += 1
                    continue
                del _gt_bad
                if (args.num_latent_frames > 0
                        and gt_latent.shape[2] > args.num_latent_frames):
                    gt_latent = gt_latent[:, :, :args.num_latent_frames]
                    if control_latent_ode is not None:
                        control_latent_ode = control_latent_ode[:, :, :args.num_latent_frames]
                img_latent_ode = gt_latent[:, :, :1].to(dtype=dtype)
                text_embeds_ode = sample["text_embeds"]
                if text_embeds_ode is None:
                    text_embeds_ode = encode_prompt(sample["prompt"])
                text_embeds_ode = text_embeds_ode.to(device=device, dtype=dtype)
                if text_embeds_ode.ndim == 2:
                    text_embeds_ode = text_embeds_ode.unsqueeze(0)
                ode_F = gt_latent.shape[2]
                # target_for_mse used for shape-mismatch check + decode hook;
                # set to gt_latent so the existing assertion against
                # x0_student.shape[2] still works (we'll override the shape
                # check below since v_flow returns F-1 frames not F).
                target_for_mse = gt_latent

            # ── ODE warmup: pin NIS=1 (independent of args.num_inference_steps) ──
            # Why not just use args.num_inference_steps here?
            #   ODE loss = MSE(x0_student, x0_teacher_precomputed). Only the
            #   FINAL x0 prediction matters; the # of denoise sub-steps the
            #   student takes to get there does NOT affect the loss target.
            #   At NIS=4 + SGT, all 21 frames keep their final-step activations
            #   in the autograd graph until the single MSE backward at the end
            #   of the ODE iter; combined with sliding cache (cap=6 → K_seq
            #   grows 300 → 2100 tokens, rotary tensors 7×) this OOMed at step
            #   1 on 8×L20Z (79 GB) — observed 71.77 GB used, 18 MB request
            #   denied. Pinning NIS=1 cuts the SGT fork entirely (the
            #   `num_inference_steps>1` guard short-circuits) and matches the
            #   memory profile of H1 baseline that's already validated on the
            #   same 8×L20Z box. The DMD branch below still uses
            #   args.num_inference_steps, so the paper-faithful 4-step
            #   student is preserved for the actual DMD training.
            #
            # 2026-05-12: H3 first 8-GPU launch on L20Z hit OOM at ODE step 1.
            # See run_20260512_085646_normandy/train.log for stack.

            # ── Memory probe gating (ODE branch) ──
            # The DMD branch defines _trace_this_step further below; the ODE
            # branch needs its own gate. Default: trace ONLY on the very first
            # ODE step of the run (step==0 when warmup is enabled and no
            # resume — see resume_step=0 at line ~2198 and step+=1 only happens
            # at the BOTTOM of the loop). On a resume, step==resume_step on
            # the first iter, so we explicitly include it.
            # Set STREAM_TRACE_EVERY=N to also probe every Nth ODE step.
            #
            # NOTE: step starts at 0 (or resume_step) and is incremented at the
            # END of the iter. The H3 OOM bites at the first forward pass, so
            # we MUST trace on the first iter regardless of step number. Hence
            # tracking via a one-shot flag (_ode_traced_once) is safer than
            # tying to a specific step value.
            if not hasattr(main, "_ode_traced_once"):
                main._ode_traced_once = False
            _ode_dump_every = int(os.environ.get("STREAM_TRACE_EVERY", "0") or 0)
            _ode_is_first = (not main._ode_traced_once) or (
                _ode_dump_every > 0 and step % _ode_dump_every == 0
            )
            _ode_trace = _ode_is_first and (rank == 0)
            if _ode_trace:
                main._ode_traced_once = True

            # MEM PROBE 0: caller-side baseline — what the GPU looks like
            # right before we hand off to streaming generation. Compare with
            # post-streaming (PROBE 5) and post-backward to attribute every
            # GB of growth to a specific stage.
            _mem_probe("pre-streaming(ode)", "ode-warmup",
                       enabled=_ode_trace, do_sync=True)

            if _is_causvid_traj_mode:
                # ===== Run D forward: CausVid §4.3 ODE-regression on streaming =====
                # Per-frame: noisy = trajectory[idx_f], σ = traj_sigmas[idx_f],
                # target = trajectory[-1] (clean x_0). Streaming + cache + GT
                # teacher-forcing prefix (same as v_flow); only (σ, noisy) diff.
                x0_pred, x0_target_traj, traj_mask = streaming_causvid_traj_forward(
                    generator=gen_engine,
                    scheduler=scheduler,
                    gt_latents=gt_latent,
                    trajectory=x0_trajectory,
                    traj_sigmas=traj_sigmas,
                    prompt_embeds=text_embeds_ode,
                    img_latent=img_latent_ode,
                    num_latent_frames=ode_F,
                    max_cache_frames=args.max_cache_frames,
                    pe_mode=args.pe_mode,
                    patch_size=gen_patch_size,
                    generator_rng=generator_rng,
                    device=device,
                    dtype=dtype,
                    trace_dump=_ode_trace,
                    trace_tag="causvid_traj",
                )
                # Masked x_0 MSE (mask = σ > 0, drops degenerate t=0 frames).
                # Same recipe as causvid/ode_regression.py:158-161 (their
                # mask = `timestep != 0`, ours = `sigmas > 1e-6`, equivalent).
                _mask_sum_elements = (
                    traj_mask.float().sum() * x0_pred.shape[1]
                    * x0_pred.shape[3] * x0_pred.shape[4]
                ).clamp(min=1.0)
                ode_loss = (
                    ((x0_pred.float() - x0_target_traj.float()) ** 2)
                    * traj_mask.float()
                ).sum() / _mask_sum_elements

                # Diagnostic alias vars for the shared logging block (same
                # pattern as v_flow). x_0 reconstruction is already in clean
                # range so no Layer-1 clamp.
                _x0_safe = x0_pred
                _x0_clamp = 0.0
                _x0_orig_finite_count = x0_pred.numel()
                _x0_raw_absmax = float(x0_pred.abs().max().item())
                _n_inf = 0
                _n_clipped = 0
                _n_total = x0_pred.numel()
                # decode hook expects this name (same as v_flow / teacher modes)
                x0_recon_for_decode = x0_pred
                # Healthy-step block (L5946) references `x0_student`. Same
                # alias as v_flow forward branch (L5615).
                x0_student = x0_pred

            elif _is_v_flow_mode:
                # ===== B+ 方案 forward: single-step v-space flow matching =====
                # Returns (v_pred, v_target) for frames 1..F-1 (frame 0 is
                # cond/sink, no prediction needed). The 4-step ODE rollout +
                # x_0 reconstruction path below is BYPASSED — v_flow's whole
                # point is to avoid the σ-amplified reconstruction that
                # caused the cold-start NaN in 'gt' mode.
                #
                # Dispatch on V_FLOW_T_SAMPLING env var:
                #   "shared"    (default) — one scalar t shared by all F frames.
                #                Baseline; bit-identical to historical behavior.
                #   "per_frame" — each frame draws its own t ~ U[0, T) i.i.d.
                #                CausVid causal_video mode with block_size=1;
                #                fixes the "越训练越静止" observed at shared-t.
                # ── Control add-plus skeleton normalization ──
                # (control_type='add-plus'). EMA-track skeleton latent stats
                # across steps, normalize per-batch skeleton to match current
                # video latent distribution before forward. Without this,
                # skeleton's mostly-white-background latent has very different
                # mean/std from video, so CPE output overwhelms patch_embed
                # output (== distribution mismatch failure mode).
                #
                # Sample level math (跟 reference 等价):
                #   v_mean, v_std         = video latent stats this batch
                #   c_mean_b, c_std_b     = control latent stats this batch
                #   running_stats EMA update with momentum=0.01
                #   control_normalized    = (control - c_run_mean) / c_run_std
                #                             * v_std + v_mean
                #
                # Streaming vs full-sequence note (Q6): normalize is per-sample
                # (whole 21-frame latent stats) BEFORE forward, then internal
                # streaming forward iterates frame-by-frame on normalized
                # tensor. Mathematically identical to reference (which does
                # one-shot forward on whole sample).
                # Save RAW skel for DMD teacher when teacher_control_type='add'
                # (teacher LoRA was trained without running_stats normalization).
                # control_latent_ode_raw is preserved AS-IS through the rest of
                # the loop. The post-normalize control_latent_ode below is what
                # student (generator) and critic see (add-plus / normalized).
                control_latent_ode_raw = control_latent_ode

                if control_latent_ode is not None:
                    with torch.no_grad():
                        # dim=(0,2,3,4) keepdim → shape [1, C, 1, 1, 1]
                        v_mean = gt_latent.mean(dim=(0, 2, 3, 4), keepdim=True)
                        v_std = gt_latent.std(dim=(0, 2, 3, 4), keepdim=True) + 1e-8
                        c_mean_batch = control_latent_ode.mean(
                            dim=(0, 2, 3, 4), keepdim=True
                        )
                        c_std_batch = control_latent_ode.std(
                            dim=(0, 2, 3, 4), keepdim=True
                        ) + 1e-8

                        momentum = control_running_stats["momentum"]
                        if control_running_stats["mean"] is None:
                            # Lazy-init from first batch (reference L873-877)
                            control_running_stats["mean"] = (
                                c_mean_batch.detach().clone()
                            )
                            control_running_stats["std"] = (
                                c_std_batch.detach().clone()
                            )
                            control_running_stats["count"] = 1
                        else:
                            # EMA update: running = running + momentum * (batch - running)
                            #           = (1-momentum)*running + momentum*batch
                            control_running_stats["mean"].lerp_(
                                c_mean_batch.detach(), momentum
                            )
                            control_running_stats["std"].lerp_(
                                c_std_batch.detach(), momentum
                            )
                            control_running_stats["count"] += 1

                        c_mean_run = control_running_stats["mean"]
                        c_std_run = control_running_stats["std"]
                        # NOTE: control_latent_ode is REASSIGNED (not in-place)
                        # so control_latent_ode_raw above stays at the original
                        # un-normalized tensor (Python aliasing).
                        # Add-plus normalize: rescale to match video latent dist.
                        control_latent_ode = (
                            (control_latent_ode - c_mean_run) / c_std_run
                            * v_std + v_mean
                        ).to(dtype=dtype)

                    # Periodic diag (every 50 steps, rank 0 only — mirrors ref L889-896)
                    if step % 50 == 0 and is_main:
                        cprint(
                            f"[ctrl-stats] step {step} | "
                            f"video: mean={gt_latent.mean().item():+.4f} "
                            f"std={gt_latent.std().item():.4f} | "
                            f"ctrl (after norm): mean={control_latent_ode.mean().item():+.4f} "
                            f"std={control_latent_ode.std().item():.4f} | "
                            f"running c_mean={c_mean_run.mean().item():+.4f} "
                            f"c_std={c_std_run.mean().item():.4f} "
                            f"(EMA count={control_running_stats['count']})",
                            "cyan",
                        )

                _v_flow_t_sampling = os.environ.get("V_FLOW_T_SAMPLING", "shared")
                _use_fixed_cache_mse = os.environ.get("USE_FIXED_CACHE", "0") == "1"
                _nfpb_mse = int(os.environ.get("NUM_FRAME_PER_BLOCK", "1"))
                if _use_fixed_cache_mse and _nfpb_mse > 1:
                    # Block-level MSE forward with FixedSizeCache (matches DMD block streaming)
                    from core.streaming.streaming_v_flow_block import streaming_v_flow_matching_forward_block
                    v_pred, v_target, x0_recon_for_decode = streaming_v_flow_matching_forward_block(
                        generator=gen_engine,
                        scheduler=v_flow_scheduler,
                        gt_latents=gt_latent,
                        prompt_embeds=text_embeds_ode,
                        generator_rng=generator_rng,
                        device=device,
                        dtype=dtype,
                        patch_size=gen_patch_size,
                        control_video_latent=control_latent_ode,
                        num_frame_per_block=_nfpb_mse,
                        num_max_frames=int(os.environ.get("NUM_MAX_FRAMES", "21")),
                        sink_size=int(os.environ.get("SINK_SIZE", "1")),
                        local_attn_size=int(os.environ.get("LOCAL_ATTN_SIZE", "-1")),
                        trace_dump=_ode_trace,
                        trace_tag="v_flow_block",
                    )
                elif _v_flow_t_sampling == "per_frame":
                    v_pred, v_target, x0_recon_for_decode = streaming_v_flow_matching_forward_per_frame_t(
                        generator=gen_engine,
                        scheduler=v_flow_scheduler,
                        gt_latents=gt_latent,
                        prompt_embeds=text_embeds_ode,
                        generator_rng=generator_rng,
                        device=device,
                        dtype=dtype,
                        patch_size=gen_patch_size,
                        max_cache_frames=args.max_cache_frames,
                        trace_dump=_ode_trace,
                        trace_tag="v_flow_pft",
                        pe_mode=args.pe_mode,
                    )
                else:
                    v_pred, v_target, x0_recon_for_decode = streaming_v_flow_matching_forward(
                        generator=gen_engine,
                        scheduler=v_flow_scheduler,
                        gt_latents=gt_latent,
                        prompt_embeds=text_embeds_ode,
                        generator_rng=generator_rng,
                        device=device,
                        dtype=dtype,
                        patch_size=gen_patch_size,
                        max_cache_frames=args.max_cache_frames,
                        trace_dump=_ode_trace,
                        trace_tag="v_flow",
                        pe_mode=args.pe_mode,
                        control_video_latent=control_latent_ode,
                    )
                # Sanity-check the returned shapes — v_pred/v_target should
                # be [1, C, F-1, H, W] (frame 0 is cond, not predicted). The
                # corresponding teacher/gt-mode check (`x0_student.shape[2]
                # != target_for_mse.shape[2]`) compares F-vs-F; here we
                # compare (F-1)-vs-(F-1) and additionally verify it equals
                # ode_F-1.
                if v_pred.shape != v_target.shape:
                    raise RuntimeError(
                        f"v_flow forward returned mismatched shapes: "
                        f"v_pred={tuple(v_pred.shape)} vs "
                        f"v_target={tuple(v_target.shape)}"
                    )
                if v_pred.shape[2] != ode_F - 1:
                    raise RuntimeError(
                        f"v_flow forward returned {v_pred.shape[2]} frames, "
                        f"expected ode_F - 1 = {ode_F - 1} "
                        f"(GT clip is {ode_F} frames; we skip frame 0)."
                    )

                # ── Compute v-space MSE in fp32 (BEFORE Layer 1 block below).
                # We compute loss here rather than in the shared MSE block
                # because the shared block computes
                # `F.mse_loss(_x0_safe, target_for_mse.float())` — that
                # signature assumes x_0-space tensors. v_flow's tensors are
                # already in v-space and need their own MSE.
                ode_loss = F.mse_loss(
                    v_pred.float(), v_target.float(), reduction="mean"
                )

                # ── Pre-fill the diagnostic vars the shared logging block reads ──
                # The healthy-step / Layer 2 / Layer 3 paths below all reach
                # for `_x0_safe`, `_n_inf`, `_n_clipped`, `_n_total`,
                # `_x0_raw_absmax`, `_x0_clamp`. Setting them here keeps that
                # shared logging code path one-branched (no new
                # `if _is_v_flow_mode` peppered through Layer 2/3).
                #
                # Naming: we keep the `_x0_*` prefix even though these are
                # v_pred stats — it's consistent with the var the existing
                # logger / SGT histogram / decode hook expect to find.
                _x0_safe = v_pred  # alias — used by log_dict["gen_latent_*"] below
                _x0_clamp = 0.0     # signals "clamp not applied" to log line
                with torch.no_grad():
                    _v_finite = torch.isfinite(v_pred)
                    _n_inf = int((~_v_finite).sum().item())
                    _n_clipped = 0
                    _n_total = v_pred.numel()
                    if _n_inf < _n_total:
                        _x0_raw_absmax = float(
                            v_pred[_v_finite].abs().max().item()
                        )
                    else:
                        _x0_raw_absmax = float("inf")
                    del _v_finite

                # `x0_student` is consumed by (a) Layer 2's `del x0_student`
                # on non-finite skip and (b) the decode hook that VAE-decodes
                # it for debug grids. Feed it the x_0 reconstruction
                # (noisy - σ·v_pred) so debug images are physically meaningful
                # — see streaming_v_flow_matching_forward docstring "Returns"
                # for the derivation. Loss math is UNCHANGED (still on
                # v_pred/v_target above). x0_recon_for_decode is already
                # .detach()-ed so it adds nothing to the autograd graph.
                x0_student = x0_recon_for_decode

                # Skip the x_0-space Layer 1 clamp + raw scan below by
                # short-circuiting; the rest of the loop (Layer 2 finite
                # check, backward, step, logging) runs unchanged.
                # Done via `if not _is_v_flow_mode` guard wrapping the
                # legacy block — see immediately below.

            if not _skip_x0_clamp:
                # ===== Legacy (teacher / gt) forward: 4-step ODE rollout =====
                x0_student = generate_streaming_with_grad(
                    generator=gen_engine,
                    scheduler=scheduler,
                    img_latent=img_latent_ode,
                    prompt_embeds=text_embeds_ode,
                    negative_prompt_embeds=None,
                    num_latent_frames=ode_F,
                    # Self-Forcing §4 / Appendix A: ODE init finetunes the
                    # base model's own NIS schedule (4-step). With SGT=1 only
                    # one of the 4 forwards bears grad, so wall-clock cost is
                    # similar to NIS=1 but the student learns the full sigma
                    # trajectory — no train-test gap when DMD switches on.
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=1.0,
                    do_cfg=False,
                    generator_rng=generator_rng,
                    device=device,
                    dtype=dtype,
                    patch_size=gen_patch_size,
                    max_cache_frames=args.max_cache_frames,
                    stochastic_grad_truncation=args.stochastic_grad_truncation,
                    trace_dump=_ode_trace,
                    trace_tag="ode",
                    pe_mode=args.pe_mode,
                )

                # Frame count must match — fail loudly rather than silently
                # broadcast, because mismatched F means the cached pairs (or
                # GT clips) were generated with a different --num_latent_frames
                # and silently slicing would give garbage targets.
                #
                # `target_for_mse` resolves to x0_teacher in 'teacher' mode and
                # to gt_latent in 'gt' mode (see if/else above).
                if x0_student.shape[2] != target_for_mse.shape[2]:
                    raise RuntimeError(
                        f"ODE target frame count mismatch: student "
                        f"{x0_student.shape[2]} vs target "
                        f"{target_for_mse.shape[2]} "
                        f"(ode_target_kind={args.ode_target_kind}). "
                        f"In 'teacher' mode: re-generate pairs with matching "
                        f"--num_latent_frames. In 'gt' mode: check "
                        f"--num_latent_frames vs the dataset clip length."
                    )

            # ── Layer 1: PROACTIVE clamp (defuse cold-start overflow) ────
            # Symptom: SGT can pick grad_step_idx=0 (highest-sigma step,
            # sigma_0 ≈ 1.0 for FlowMatch 4-step). On the cold-start
            # student, `x_0 = x_t - sigma_t * v_pred` is dominated by
            # v_pred — a 5-B bf16 model with no warmup can have local
            # attention-score overflow → v_pred absmax > 1e4 in a few
            # tokens → x_0 has localized inf/nan elements. Squaring those
            # in MSE → loss = inf even though 99.9% of the tensor is
            # well-behaved.
            #
            # B 方案 (GT-warmup) is more vulnerable than A 方案 because
            # the noise is RANDOM (not seeded to a teacher-cached pair),
            # so the student's first output is further from the MSE
            # target regardless.
            #
            # Why clamp instead of just skip:
            #   * skip-only wastes the entire forward — 21-frame streaming
            #     generation took ~30s, and we'd throw away the gradient
            #     signal from 99% of tokens because 0.1% overflowed.
            #   * clamp lets the model LEARN from the clean tokens, and
            #     at clipped positions sets gradient = 0 (clamp boundary
            #     derivative is 0 by convention), so weight updates aren't
            #     poisoned by the bad token.
            #
            # Choice of clamp value:
            #   GT video latents have absmax ≈ 6 (verified by scanning
            #   30 epic_rdt files: range 4.7-6.7). Teacher x0 cache also
            #   in same range. So legitimate student output should be
            #   |x| ≤ 10. Clamp at 100 leaves 10x slack for transient
            #   model "warmup overshoot" while preventing fp32 MSE
            #   overflow: (100-6)^2 = 8836 per element, sum over 1.2e9
            #   elements → mean ≤ 8836, well under fp32 max ≈ 3.4e38.
            #
            # Set STREAM_ODE_X0_CLAMP=0 to disable (returns to original
            # behaviour: any inf token → loss inf → Layer 2 skip).
            #
            # v_flow mode SKIPS this entire block: v-space outputs are
            # bounded O(1) by construction (target = noise - GT, both with
            # |x| ≤ ~7), so a 100-clamp on v_pred would be a no-op. The
            # diagnostic vars (_x0_safe, _n_inf, _n_clipped, _n_total,
            # _x0_raw_absmax, _x0_clamp) and ode_loss were already
            # populated in the v_flow forward branch above.
            if not _skip_x0_clamp:
                _x0_clamp = float(os.environ.get("STREAM_ODE_X0_CLAMP", "100.0"))
                if _x0_clamp > 0:
                    # nan_to_num replaces nan→0, +inf→clamp, -inf→-clamp;
                    # then clamp the rest into [-clamp, +clamp]. Doing this
                    # on the .float() copy preserves the autograd graph
                    # (clamp is differentiable a.e., zero-grad at boundary).
                    _x0_safe = torch.nan_to_num(
                        x0_student.float(),
                        nan=0.0, posinf=_x0_clamp, neginf=-_x0_clamp,
                    ).clamp(-_x0_clamp, _x0_clamp)
                else:
                    _x0_safe = x0_student.float()

                # Diagnostic: how much did we clip? On a healthy step this is
                # 0 elements clipped. On a cold-start "bad" step we expect
                # <0.01% of elements clipped (just a few overflow tokens).
                # If >1% clipped, the model is genuinely diverging.
                with torch.no_grad():
                    _x0_orig = x0_student.float()
                    _x0_orig_finite = torch.isfinite(_x0_orig)
                    _n_inf = int((~_x0_orig_finite).sum().item())
                    # Count clipped only among finite elements (inf are
                    # already handled by nan_to_num above). Use abs > clamp
                    # on the finite mask.
                    if _x0_clamp > 0:
                        _n_clipped = int(
                            ((_x0_orig.abs() > _x0_clamp) & _x0_orig_finite).sum().item()
                        )
                    else:
                        _n_clipped = 0
                    _n_total = _x0_orig.numel()
                    # raw absmax over FINITE elements only (so log doesn't
                    # show "inf" which is uninformative — we already log
                    # _n_inf separately).
                    if _n_inf < _n_total:
                        _x0_raw_absmax = float(
                            _x0_orig[_x0_orig_finite].abs().max().item()
                        )
                    else:
                        _x0_raw_absmax = float("inf")
                    del _x0_orig, _x0_orig_finite

                # fp32 MSE loss on the SAFE (clamped) student tensor. Casting
                # student up to fp32 for the loss math prevents bf16 rounding
                # from swallowing small residuals once the student has mostly
                # converged. target_for_mse is already fp32 (both branches above).
                ode_loss = F.mse_loss(
                    _x0_safe, target_for_mse.float(), reduction="mean"
                )

            # ── Layer 2: PARANOIA skip if loss STILL non-finite ──────────
            # Should be very rare after Layer 1: only triggers if
            # target_for_mse itself contains inf/nan, or if the clamped
            # student tensor's MSE somehow still overflows (mathematically
            # impossible with clamp=100 + a sane target, but kept as
            # defense-in-depth).
            _loss_is_finite = bool(torch.isfinite(ode_loss).item())
            if not _loss_is_finite:
                _ode_nonfinite_consec += 1
                _ode_nonfinite_total += 1
                if is_main:
                    cprint(
                        f"[ode] step {step} | loss STILL non-finite after "
                        f"clamp (value={ode_loss.item()}); SKIPPING. "
                        f"raw absmax={_x0_raw_absmax:.2e} "
                        f"(inf:{_n_inf}/{_n_total}), "
                        f"clamped absmax={_x0_safe.abs().max().item():.2e}, "
                        f"target absmax={target_for_mse.abs().max().item():.2e}, "
                        f"clipped {_n_clipped}/{_n_total} "
                        f"({100.0*_n_clipped/max(1,_n_total):.4f}%), "
                        f"sgt={_sgt_diag.get('last_grad_step_idx', '?')} "
                        f"| consec={_ode_nonfinite_consec} "
                        f"total={_ode_nonfinite_total}",
                        "red",
                    )
                # DO NOT call _attribute_sgt_loss(nan) — would poison the
                # bucket's running mean for the rest of training. The red
                # log line records the event with full context.
                # Stash detached clone for debug-decode hook (the JPEG
                # will look broken — useful signal, not a bug).
                _x0_for_decode = x0_student.detach().clone()
                del ode_loss, x0_student, _x0_safe
                log_dict["gen_loss"] = float("nan")
                log_dict["crit_loss"] = 0.0
                log_dict["dmd_grad_norm"] = float("nan")
                log_dict["crit_grad_norm"] = float("nan")
                log_dict["gen_grad_norm"] = float("nan")
                log_dict["gen_latent_std"] = float("nan")
                log_dict["gen_latent_mean"] = float("nan")
                log_dict["gen_latent_absmax"] = float("nan")
                _skip_warmup_update = True

                # ── Layer 3: fail-fast on consecutive non-finite ──
                # If Layer 1 (clamp) + Layer 2 (skip) BOTH fail
                # repeatedly, the model is genuinely broken. Aborting
                # here saves hours of stuck-run compute.
                if _ode_nonfinite_consec >= _ode_max_consec_nonfinite:
                    _msg = (
                        f"[ode] FATAL: {_ode_nonfinite_consec} consecutive "
                        f"non-finite ODE losses (total {_ode_nonfinite_total}). "
                        f"Layer 1 (x0 clamp at {_x0_clamp}) and Layer 2 "
                        f"(skip backward) both insufficient. Likely root "
                        f"causes: (a) target tensor has inf/nan (check "
                        f"data pipeline), (b) gen_engine has accumulated "
                        f"corrupted weights from a prior bad backward, "
                        f"(c) attention overflow in bf16 needs lower lr "
                        f"or more warmup ratio. Aborting before more compute "
                        f"is wasted. Override threshold via env "
                        f"STREAM_MAX_CONSEC_NONFINITE=N."
                    )
                    if is_main:
                        cprint(_msg, "red")
                    raise RuntimeError(_msg)
            else:
                # Healthy step: reset consec counter (so an isolated
                # bad step doesn't gradually trip the abort threshold).
                if _ode_nonfinite_consec > 0 and is_main:
                    cprint(
                        f"[ode] step {step} | recovered after "
                        f"{_ode_nonfinite_consec} non-finite step(s); "
                        f"resetting consec counter "
                        f"(running total: {_ode_nonfinite_total})",
                        "green",
                    )
                _ode_nonfinite_consec = 0
                _skip_warmup_update = False
                # Optional: warn if Layer 1 was actively clipping (more
                # than a token or two). Only fires when n_clipped > 0
                # OR we had any inf in the raw tensor.
                if is_main and (_n_clipped > 0 or _n_inf > 0):
                    _pct_clip = 100.0 * _n_clipped / max(1, _n_total)
                    _pct_inf = 100.0 * _n_inf / max(1, _n_total)
                    _color = ("yellow"
                              if (_pct_clip < 1.0 and _pct_inf < 0.1)
                              else "red")
                    cprint(
                        f"[ode] step {step} | Layer 1 clamp active: "
                        f"{_n_clipped} clipped ({_pct_clip:.4f}%), "
                        f"{_n_inf} inf→0 ({_pct_inf:.4f}%); "
                        f"raw absmax={_x0_raw_absmax:.2e} → "
                        f"clamped absmax={_x0_safe.abs().max().item():.2e}, "
                        f"sgt={_sgt_diag.get('last_grad_step_idx', '?')} "
                        f"| total non-finite skips so far: "
                        f"{_ode_nonfinite_total}",
                        _color,
                    )

            if not _skip_warmup_update:
                # MEM PROBE 6: pre-backward snapshot. Includes the autograd graph
                # built by all 21 streaming forwards + the .float()-cast student
                # tensor + the teacher reference. This is the right comparison
                # point for "how much does backward itself add" (next probe).
                _mem_probe("pre-backward(ode)", "ode-warmup",
                           enabled=_ode_trace, do_sync=True)

                gen_engine.backward(ode_loss)

                # MEM PROBE 7: post-backward, pre-step. backward() walks the
                # autograd graph in reverse, producing parameter gradients. With
                # ZeRO-3, those gradients are reduce-scattered across ranks
                # immediately, so the working set during backward briefly spikes
                # but should not persist. Compare against pre-backward to see
                # the residual cost (gradient bucket hold-overs, etc.).
                _mem_probe("post-backward(ode)", "ode-warmup",
                           enabled=_ode_trace, do_sync=True)

                gen_engine.step()

                # MEM PROBE 8: post-step. step() consumes the reduce-scattered
                # gradients to update CPU-offloaded optimizer state, then frees
                # the GPU-side gradient buffers. The drop from PROBE 7 to PROBE 8
                # is the "gradient memory" cost we'll re-incur next iter.
                _mem_probe("post-step(ode)", "ode-warmup",
                           enabled=_ode_trace, do_sync=True)
                # Lazy EMA shadow creation: only after step >= ema_start_step
                # (LongLive convention; default ema_start_step=0 → created at
                # init time so this branch is a no-op). Allocation is
                # rank-local — no collectives — so lazy creation here is safe.
                if (gen_ema_shadow is None
                        and args.ema_decay > 0.0
                        and step >= args.ema_start_step):
                    if is_main:
                        cprint(
                            f"[ema] Lazy-creating generator EMA shadow at step "
                            f"{step} (decay={args.ema_decay}, "
                            f"start_step={args.ema_start_step})",
                            "cyan",
                        )
                    gen_ema_shadow = _build_ema_shadow(gen_engine)
                if gen_ema_shadow is not None:
                    _ema_update(gen_ema_shadow, gen_engine, args.ema_decay)

                try:
                    _gn = gen_engine.get_global_grad_norm()
                    log_dict["gen_grad_norm"] = float(_gn) if _gn is not None else float("nan")
                except Exception:
                    pass
                with torch.no_grad():
                    # Use the SAFE (clamped) tensor for log diagnostics —
                    # raw x0_student may have a few inf elements (Layer 1
                    # didn't modify it), and inf in std/mean/absmax fields
                    # is uninformative. _x0_raw_absmax (computed above)
                    # already gives the unclamped raw absmax over finite
                    # elements; logging the safe version here keeps the
                    # standard fields well-bounded for downstream tooling
                    # (CSV → grafana, etc.).
                    gflat = _x0_safe.detach()
                    log_dict["gen_latent_std"] = float(gflat.std().item())
                    log_dict["gen_latent_mean"] = float(gflat.mean().item())
                    log_dict["gen_latent_absmax"] = float(gflat.abs().max().item())
                    # Per-channel std/absmax for color-shift diagnosis (2026-05-20).
                    # Latent shape [B, C, F, H, W]; std over [B, F, H, W] → [C].
                    # Log top-5 inflated channels (rank by std). When a few specific
                    # channels blow up while others stay normal → chroma channel
                    # collapse → visual color shift.
                    _per_ch_std = gflat.float().std(dim=[0, 2, 3, 4])  # [C]
                    _per_ch_abs = gflat.float().abs().amax(dim=[0, 2, 3, 4])  # [C]
                    _top5_idx = _per_ch_std.argsort(descending=True)[:5].tolist()
                    log_dict["gen_latent_per_ch_top5_std"] = [
                        (int(c), float(_per_ch_std[c].item())) for c in _top5_idx
                    ]
                    log_dict["gen_latent_per_ch_top5_absmax"] = [
                        (int(c), float(_per_ch_abs[c].item())) for c in _top5_idx
                    ]
                    del gflat, _per_ch_std, _per_ch_abs
                log_dict["gen_loss"] = ode_loss.item()
                # Attribute THIS iter's loss to the SGT bucket that was picked
                # inside the just-finished _streaming_generate call. With NIS=4 +
                # SGT the bucket is one of {0,1,2,3} (broadcast across ranks); in
                # plain mode (no SGT) it's -1 and the helper no-ops. Bucket 0 is
                # the lowest-noise step (cleanest 1-step x_0 estimate, lowest
                # expected loss); bucket NIS-1 is the highest-noise step (worst
                # expected loss). Watching the per-bucket mean loss decay tells
                # us how the high-noise tail is healing — which is exactly the
                # signal the static gen_loss line can't show.
                _attribute_sgt_loss(ode_loss.item())
                # Critic / DMD diagnostics are meaningless during warmup —
                # zero them so the shared log line doesn't show stale values
                # carried over from a prior resume.
                log_dict["crit_loss"] = 0.0
                log_dict["dmd_grad_norm"] = float("nan")
                log_dict["crit_grad_norm"] = float("nan")
                # `generated_no_grad` is consumed by the visual-decode hook at
                # the bottom of the loop; in the warmup branch we just hand it
                # the (detached) student output. Use raw x0_student (not the
                # clamped _x0_safe) so the debug JPEG shows what the model
                # actually produced — clipping artifacts visible in the
                # decode confirm Layer 1 fired.
                generated_no_grad = x0_student.detach()
                # Release the .float() copy from Layer 1 so it doesn't
                # carry across iterations (it's [1,48,21,30,40] fp32 ≈
                # 4.8 MB per iter — small but adds up over 2k warmup steps).
                del _x0_safe
            else:
                # Skip branch (non-finite loss): we already deleted ode_loss
                # and x0_student in the guard above, but stashed a detached
                # clone at `_x0_for_decode`. Hand it to the bottom-of-loop
                # debug-decode hook so the periodic visualization still
                # runs (it'll show whatever the cold-start student emitted,
                # garbage and all — that's actually useful debug signal:
                # rank-0 will see "all-black" or "saturated" JPEGs and we
                # know exactly which step blew up).
                generated_no_grad = _x0_for_decode
                del _x0_for_decode
            # Fall through to the shared logging/decode/save block.

        # ── DMD branch (normal training after warmup) ───────────────────
        # Guarded on `not is_warmup_step` so after refactoring to share the
        # bottom-of-loop book-keeping we don't re-indent this whole block.
        if not is_warmup_step:
            # Clear the ODE-pair idx (debug-decode hook uses -1 to mean
            # "this step isn't an ODE warmup step, so teacher-reference and
            # multi-pair-scene modes don't apply"). See _current_pair_idx_holder
            # definition near the decode function.
            #
            # NOTE (2026-05-14): the GT holder USED to also be cleared to -1
            # here, on the assumption that "DMD steps have no GT reference".
            # That was wrong for v_flow / gt warmup modes: the DMD branch
            # samples from the SAME `dataset` (L4400-4401), so a per-step GT
            # row is just as comparable as in warmup. We now defer the holder
            # write until after `idx` is defined (a few lines below) so the
            # widened `with_gt_ref` / `use_gt_extras` predicates at L3247 +
            # L4794 see a valid live-sample idx and the debug grid renders
            # 4 scenes × 2 rows in DMD just like in warmup.
            _current_pair_idx_holder["idx"] = -1
            # Sample data — shuffled per-epoch from this rank's shard
            epoch = step // max(1, len(dataset))
            order = _epoch_indices(epoch)
            idx = order[step % len(dataset)]
            # Wire the live idx through to the decode-hook holder. Only
            # meaningful for gt/v_flow modes; teacher mode reads from a
            # disjoint `ode_dataset`, so its decode hook continues to use
            # `_current_pair_idx_holder` (left at -1 above by design).
            _current_gt_idx_holder["idx"] = idx
            sample = dataset[idx]
            video_latent = sample["video_latent"].unsqueeze(0)  # [1, C, F, H, W]
            # GT-corruption guard. See L4393-4421 (v_flow branch) for full
            # rationale. DMD iter is heavier than ODE (4-step student +
            # critic forward), so a pre-forward skip is even more
            # important — and skipping here is *especially* clean because
            # `step += 1; continue` jumps over BOTH the gen sub-iter and
            # any critic sub-iter, keeping the NCCL collective count
            # bit-aligned across the world.
            # See L4327 gt-mode guard for the absmax-threshold rationale —
            # 9.66e+35 is fp32-finite, so isfinite() alone is insufficient.
            _gt_abs = float(video_latent.abs().max().item())
            _gt_bad = torch.tensor(
                [0 if (bool(torch.isfinite(video_latent).all().item())
                       and _gt_abs <= 100.0) else 1],
                device=device, dtype=torch.int32,
            )
            if dist.is_initialized():
                dist.all_reduce(_gt_bad, op=dist.ReduceOp.MAX)
            if int(_gt_bad.item()) > 0:
                if is_main:
                    _local_bad = (~torch.isfinite(video_latent)).sum().item()
                    cprint(
                        f"[dmd] step {step} | GT latent corrupted on at "
                        f"least one rank (this rank: idx={idx}, "
                        f"{int(_local_bad)} non-finite/{video_latent.numel()}, "
                        f"absmax={_gt_abs:.2e}); "
                        f"SKIPPING world-wide, no param update.",
                        "red",
                    )
                del video_latent, sample, _gt_bad
                log_dict.update({
                    "gen_loss": float("nan"),
                    "crit_loss": 0.0,
                    "dmd_grad_norm": float("nan"),
                    "crit_grad_norm": float("nan"),
                    "gen_grad_norm": float("nan"),
                    "gen_latent_std": float("nan"),
                    "gen_latent_mean": float("nan"),
                    "gen_latent_absmax": float("nan"),
                })
                step += 1
                continue
            del _gt_bad
            text_embeds = sample["text_embeds"]
            if text_embeds is None:
                text_embeds = encode_prompt(sample["prompt"])
            text_embeds = text_embeds.to(device=device, dtype=dtype)
            # 2026-05-25 fix: dataset returns 2D [seq, dim] for pre-encoded
            # text embeddings; ensure 3D [B, seq, dim] for cross-attention.
            # (MSE branch has the same fix at L7341-7342; this was missing in DMD
            # → cross-attention to_k(2D) → key.unflatten(2,...) IndexError.)
            if text_embeds.ndim == 2:
                text_embeds = text_embeds.unsqueeze(0)

            # ── Control distill: extract control_video_latent for DMD phase ──
            # Mirrors the MSE warmup path (L7381). Both `_dmd_raw` (un-normalized,
            # for teacher when teacher_control_type='add') and `_dmd_norm`
            # (running_stats-normalized, for student/critic add-plus mode).
            control_latent_dmd_raw = sample.get("control_video_latent", None)
            if control_latent_dmd_raw is not None:
                control_latent_dmd_raw = control_latent_dmd_raw.unsqueeze(0).to(
                    device=device, dtype=dtype,
                )

            # Variable-length support (LongLive-style):
            # If --num_latent_frames=0, use the dataset's native length per
            # sample (lets us mix 7F and 21F clips in one run). If >0, cap
            # to that length (legacy behavior — fixes total budget).
            #
            # In sliding-window cache mode the autograd cost scales with
            # actual frame count, so longer clips ARE pricier. We rely on
            # the per-sample length being small (<=21F here).
            if args.num_latent_frames > 0 and video_latent.shape[2] > args.num_latent_frames:
                video_latent = video_latent[:, :, :args.num_latent_frames]
                if control_latent_dmd_raw is not None:
                    control_latent_dmd_raw = control_latent_dmd_raw[:, :, :args.num_latent_frames]
            # `effective_F` = # frames we'll actually generate this iter; it
            # gets passed to _streaming_generate as num_latent_frames so the
            # loop knows when to stop. (args.num_latent_frames is the *cap*,
            # not the source of truth.)
            effective_F = video_latent.shape[2]

            # ── Skeleton normalization for student/critic (add-plus) ──
            # Update control_running_stats EMA same as MSE branch (so DMD-phase
            # stats keep tracking the live skel distribution; in practice
            # stats are nearly stable by end of MSE, so this is mostly a no-op).
            # control_latent_dmd_norm = (raw - c_run_mean)/c_run_std * v_std + v_mean
            # (matches add-plus inference semantics).
            control_latent_dmd_norm = None
            if control_latent_dmd_raw is not None:
                _gt_for_stats = video_latent.to(device=device, dtype=torch.float32)
                with torch.no_grad():
                    v_mean = _gt_for_stats.mean(dim=(0, 2, 3, 4), keepdim=True)
                    v_std = _gt_for_stats.std(dim=(0, 2, 3, 4), keepdim=True) + 1e-8
                    c_mean_batch = control_latent_dmd_raw.mean(
                        dim=(0, 2, 3, 4), keepdim=True
                    )
                    c_std_batch = control_latent_dmd_raw.std(
                        dim=(0, 2, 3, 4), keepdim=True
                    ) + 1e-8
                    momentum = control_running_stats["momentum"]
                    if control_running_stats["mean"] is None:
                        control_running_stats["mean"] = c_mean_batch.detach().clone()
                        control_running_stats["std"] = c_std_batch.detach().clone()
                        control_running_stats["count"] = 1
                    else:
                        control_running_stats["mean"].lerp_(
                            c_mean_batch.detach(), momentum
                        )
                        control_running_stats["std"].lerp_(
                            c_std_batch.detach(), momentum
                        )
                        control_running_stats["count"] += 1
                    c_mean_run = control_running_stats["mean"]
                    c_std_run = control_running_stats["std"]
                    # Build normalized version (without overwriting raw — we keep
                    # both, raw goes to teacher when teacher_control_type='add')
                    control_latent_dmd_norm = (
                        (control_latent_dmd_raw - c_mean_run) / c_std_run
                        * v_std + v_mean
                    ).to(dtype=dtype)
                if step % 50 == 0 and is_main:
                    cprint(
                        f"[ctrl-stats-dmd] step {step} | "
                        f"video: mean={_gt_for_stats.mean().item():+.4f} "
                        f"std={_gt_for_stats.std().item():.4f} | "
                        f"running c_mean={c_mean_run.mean().item():+.4f} "
                        f"c_std={c_std_run.mean().item():.4f} "
                        f"(EMA count={control_running_stats['count']})",
                        "cyan",
                    )

            # Decide which skel goes to teacher (real_score) based on CLI flag:
            #   teacher_control_type='add'      → RAW skel (new LoRA teacher)
            #   teacher_control_type='add-plus' → NORMALIZED skel (legacy teachers)
            if args.teacher_control_type == "add":
                _teacher_ctrl = control_latent_dmd_raw
            else:
                _teacher_ctrl = control_latent_dmd_norm

            # Condition frame
            img_latent = video_latent[:, :, :1, :, :]

            # Conditional / unconditional dicts
            conditional_dict = {"prompt_embeds": text_embeds}
            unconditional_dict = {"prompt_embeds": negative_embeds}

            do_cfg = args.guidance_scale != 1.0

            # patch_size is just a config value — safe to read off the underlying module.
            gen_patch_size = gen_engine.module.config.patch_size

            # ── H2 streaming-window setup (no-op if --train_window_size=0) ──
            # Sample T_start from {1, 1+K, 1+2K, ...} ∩ [1, num_latent_frames-K]
            # on rank 0 and broadcast. ZeRO-3 SAFETY: T_start affects which
            # frames carry grad → which params get unsharded → must match
            # across ranks (same lesson as grad_step_idx broadcast above).
            #
            # Why discrete positions {1, 1+K, 1+2K, ...} not random in [1, F-K]:
            #   - Cleaner coverage: with F=21, K=7 → positions {1, 8, 14},
            #     each iter trains a non-overlapping window. Over 3 iters
            #     every frame in [1, 21) gets gradient exactly once.
            #   - Reproducibility: easier to inspect "which window" in logs.
            #   - Same idea as LongLive Figure 4(c): tile the long sequence
            #     by clip length, train one clip per iter.
            h2_window = args.train_window_size > 0
            h2_t_start = None
            h2_gt_latents = None
            if h2_window:
                # Build legal start positions. NB: name the frame-count local
                # `_F_total` (not `F`) — the module-level `F` is
                # `torch.nn.functional` which is used 5 lines down for the
                # ODE warmup MSE loss. Shadowing it here would crash the
                # warmup branch later in the same iteration.
                K = args.train_window_size
                _F_total = effective_F  # use per-sample length (was args.num_latent_frames)
                legal_starts = list(range(1, _F_total - K + 1, K))
                # If _F_total-K-1 isn't a multiple of K, add _F_total-K so the tail of
                # the video is also covered (e.g. F=21, K=7 → [1,8] from
                # range, then explicitly add 14).
                if legal_starts[-1] != _F_total - K:
                    legal_starts.append(_F_total - K)
                # ZeRO-3-safe sampling: rank 0 picks, broadcast to all.
                _ts_buf = torch.empty(1, dtype=torch.long, device=device)
                if not dist.is_initialized() or dist.get_rank() == 0:
                    _ts_buf.fill_(legal_starts[
                        int(torch.randint(0, len(legal_starts), (1,)).item())
                    ])
                if dist.is_initialized():
                    dist.broadcast(_ts_buf, src=0)
                h2_t_start = int(_ts_buf.item())
                # GT far-history source = the GT video latent itself (we
                # already loaded it above as `video_latent`). Use only the
                # frames that fall in the GT range (frame 1 .. T_start-K-1)
                # to avoid copying extra data; _streaming_generate slices
                # internally via [:, :, frame_idx:frame_idx+1].
                # Cast to dtype here (dtype matches transformer activation).
                h2_gt_latents = video_latent.to(device=device, dtype=dtype)
                # Log the sampled position once per gen step (rank 0 only,
                # tiny overhead, big debug value).
                if rank == 0:
                    log_dict["h2_t_start"] = float(h2_t_start)

            # Trace dump: enable on the FIRST DMD step (and again right after
            # the cache trims, so both pre-trim and post-trim slot geometry
            # appear in the log). Cheap (rank 0 only, 1 print/frame), and
            # reading 21 lines once per debug session is way better than
            # re-instrumenting after the fact when something looks off.
            #
            # Heuristic: dump when this is the very first DMD step (right
            # after warmup), AND on the next dump_every_dmd_steps if the env
            # var is set (off by default = trace ONCE per run).
            _dump_every = int(os.environ.get("STREAM_TRACE_EVERY", "0") or 0)
            _is_first_dmd = (
                args.ode_warmup_steps > 0 and step == args.ode_warmup_steps + 1
            ) or (args.ode_warmup_steps == 0 and step == 1)
            _trace_this_step = _is_first_dmd or (
                _dump_every > 0 and step % _dump_every == 0
            )
            _trace_this_step = _trace_this_step and (rank == 0)

            if step % args.dfake_gen_update_ratio == 0:
                # ── Generator step ──
                # Pass the DeepSpeed engine `gen_engine` so backward goes through
                # ZeRO-3's grad bookkeeping. The engine is callable just like the module.
                _nfpb = int(os.environ.get("NUM_FRAME_PER_BLOCK", "1"))
                _use_fixed_cache = os.environ.get("USE_FIXED_CACHE", "0") == "1"
                if _nfpb > 1 and _use_fixed_cache:
                    # Block-level streaming with FixedSizeCache (Self-Forcing aligned).
                    # Enables gradient_checkpointing for memory savings.
                    from core.streaming.streaming_generate_block_v2 import generate_streaming_block_with_grad_v2
                    generated = generate_streaming_block_with_grad_v2(
                        generator=gen_engine,
                        scheduler=scheduler,
                        img_latent=img_latent,
                        prompt_embeds=text_embeds,
                        negative_prompt_embeds=negative_embeds if do_cfg else None,
                        num_latent_frames=effective_F,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale,
                        do_cfg=do_cfg,
                        generator_rng=generator_rng,
                        device=device,
                        dtype=dtype,
                        patch_size=gen_patch_size,
                        num_max_frames=int(os.environ.get("NUM_MAX_FRAMES", str(effective_F))),
                        pe_mode="absolute",  # Self-Forcing uses absolute position
                        stochastic_grad_truncation=args.stochastic_grad_truncation,
                        control_video_latent=control_latent_dmd_norm,
                        num_frame_per_block=_nfpb,
                        sink_size=int(os.environ.get("SINK_SIZE", "1")),
                        local_attn_size=int(os.environ.get("LOCAL_ATTN_SIZE", "-1")),
                    )
                else:
                    generated = generate_streaming_with_grad(
                        generator=gen_engine,
                        scheduler=scheduler,
                        img_latent=img_latent,
                        prompt_embeds=text_embeds,
                        negative_prompt_embeds=negative_embeds if do_cfg else None,
                        num_latent_frames=effective_F,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale,
                        do_cfg=do_cfg,
                        generator_rng=generator_rng,
                        device=device,
                        dtype=dtype,
                        patch_size=gen_patch_size,
                        max_cache_frames=args.max_cache_frames,
                        stochastic_grad_truncation=args.stochastic_grad_truncation,
                        train_window_size=args.train_window_size if h2_window else None,
                        train_window_start=h2_t_start,
                        gt_far_latents=h2_gt_latents,
                        trace_dump=_trace_this_step,
                        trace_tag="gen",
                        pe_mode=args.pe_mode,
                        # Student uses NORMALIZED skel (add-plus, matches MSE training)
                        control_video_latent=control_latent_dmd_norm,
                    )

                # DMD generator loss — critic and real_score forwards are no_grad
                # inside compute_dmd_generator_loss, so passing critic_engine.module
                # here is safe (no grad needs to flow through critic in the gen step).
                #
                # real_score lives on CPU between calls (saves ~10 GB/rank of VRAM
                # for activations); swap it onto GPU just for this forward, then
                # back to CPU before backward so its 10 GB doesn't compete with the
                # generator activation graph during backward.
                real_score = real_score.to(device)
                gen_loss, gen_log = compute_dmd_generator_loss(
                    generated_latents=generated,
                    critic=critic_engine.module,
                    real_score=real_score,
                    conditional_dict=conditional_dict,
                    unconditional_dict=unconditional_dict,
                    num_train_timesteps=scheduler.config.num_train_timesteps,
                    real_guidance_scale=args.real_guidance_scale,
                    fake_guidance_scale=args.fake_guidance_scale,
                    min_step_frac=args.min_step_frac,
                    max_step_frac=args.max_step_frac,
                    patch_size=gen_patch_size,
                    # Critic uses NORMALIZED skel (matches student add-plus).
                    # Real_score uses RAW or NORMALIZED based on
                    # --teacher_control_type (decided at sample-load time above).
                    student_control_latent=control_latent_dmd_norm,
                    teacher_control_latent=_teacher_ctrl,
                )
                real_score = real_score.to("cpu")
                torch.cuda.empty_cache()

                # Stamp the most-recently-written debug ring slot with
                # the current train step so the decode consumer can show
                # "snapshot @ step N" + tell whether a fresh DMD compute
                # has happened since last decode.
                try:
                    if _DMD_DEBUG_RING_SIZE > 0 and _DMD_DEBUG_WRITES[0] > 0:
                        _last_idx = (_DMD_DEBUG_WRITES[0] - 1) % _DMD_DEBUG_RING_SIZE
                        if _DMD_DEBUG_RING[_last_idx] is not None:
                            _DMD_DEBUG_RING[_last_idx]["step_id"] = int(step)
                except Exception:
                    pass

                # ── DMD diagnostics (cheap, do BEFORE backward frees the tensors) ──
                # `dmdtrain_gradient_norm` from compute_dmd_generator_loss is the
                # mean abs of (fake_score - real_score)/normalizer — the actual
                # signal driving the generator. If it collapses to ~0 the gen has
                # nothing to learn from; if it explodes (>>1) the critic is too far
                # off and updates will be unstable.
                log_dict["dmd_grad_norm"] = float(gen_log.get("dmdtrain_gradient_norm", float("nan")))
                # Forward all DMD diagnostic fields from compute_dmd_generator_loss
                # to the per-step log so we can grep them later. Naming kept
                # identical to gen_log keys for traceability.
                for _dk in (
                    "dmd_fake_x0_absmean", "dmd_real_x0_absmean", "dmd_gen_x0_absmean",
                    "dmd_raw_grad_absmean", "dmd_norm_grad_absmean",
                    "dmd_normalizer_mean", "dmd_normalizer_min", "dmd_normalizer_max",
                    "dmd_normalizer_safemask_zero_frac",
                    "dmd_grad_p_real_cos",
                    "dmd_sigma_mean", "dmd_sigma_min", "dmd_sigma_max",
                    # Additional diagnostics
                    "dmd_f0_dev_fake", "dmd_f0_dev_real",
                    "dmd_grad_f1", "dmd_grad_fN", "dmd_grad_f1_fN_ratio",
                    "dmd_rawg_f1", "dmd_rawg_fMid", "dmd_rawg_fN",
                    "dmd_cos_f1", "dmd_cos_fMid", "dmd_cos_fN",
                    "dmd_real_v_absmean", "dmd_fake_v_absmean",
                    "dmd_v_diff_absmean", "dmd_v_diff_to_real_ratio",
                    "dmd_real_cfg_strength",
                ):
                    if _dk in gen_log:
                        log_dict[_dk] = float(gen_log[_dk])
                with torch.no_grad():
                    gflat = generated.detach().float()
                    log_dict["gen_latent_std"] = float(gflat.std().item())
                    log_dict["gen_latent_mean"] = float(gflat.mean().item())
                    log_dict["gen_latent_absmax"] = float(gflat.abs().max().item())
                    # Per-channel std/absmax for color-shift diagnosis (2026-05-20).
                    # Latent shape [B, C, F, H, W]; std over [B, F, H, W] → [C].
                    # Top-5 inflated channels (rank by std). DMD divergence often
                    # shows up as 2-3 chroma channels inflating while others stay
                    # ~1, manifesting as color tone artifacts in decoded RGB.
                    _per_ch_std = gflat.std(dim=[0, 2, 3, 4])
                    _per_ch_abs = gflat.abs().amax(dim=[0, 2, 3, 4])
                    _top5_idx = _per_ch_std.argsort(descending=True)[:5].tolist()
                    log_dict["gen_latent_per_ch_top5_std"] = [
                        (int(c), float(_per_ch_std[c].item())) for c in _top5_idx
                    ]
                    log_dict["gen_latent_per_ch_top5_absmax"] = [
                        (int(c), float(_per_ch_abs[c].item())) for c in _top5_idx
                    ]
                    del gflat, _per_ch_std, _per_ch_abs

                # DeepSpeed engine.step() handles optimizer.step(), scheduler.step(),
                # and zero_grad together (and gradient clipping per ds_config).
                gen_engine.backward(gen_loss)
                gen_engine.step()
                # EMA update (if enabled). Runs on CPU, no collectives — pure local
                # lerp on this rank's master shard. Negligible cost (~10-30 ms/step
                # for a 5B model on 8 ranks ≈ 2.5 GB CPU data).
                # Lazy creation guard: shadow may be None until step reaches
                # args.ema_start_step (LongLive style). See ode-warmup branch
                # for the matching block.
                if (gen_ema_shadow is None
                        and args.ema_decay > 0.0
                        and step >= args.ema_start_step):
                    if is_main:
                        cprint(
                            f"[ema] Lazy-creating generator EMA shadow at step "
                            f"{step} (decay={args.ema_decay}, "
                            f"start_step={args.ema_start_step})",
                            "cyan",
                        )
                    gen_ema_shadow = _build_ema_shadow(gen_engine)
                if gen_ema_shadow is not None:
                    _ema_update(gen_ema_shadow, gen_engine, args.ema_decay)
                # Post-step: DeepSpeed exposes the (pre-clip) global grad norm. NaN
                # here = collapse (bf16 overflow), >>1 = on the edge of clipping.
                try:
                    _gn = gen_engine.get_global_grad_norm()
                    log_dict["gen_grad_norm"] = float(_gn) if _gn is not None else float("nan")
                except Exception:
                    pass

                # ── Critic step (regen samples, no grad) ──
                # In H2 mode, reuse the SAME T_start as the gen step so the
                # critic sees the same window distribution. This keeps the
                # critic well-calibrated to "the part of the video the
                # generator is actually producing right now".
                with torch.no_grad():
                    generated_no_grad = generate_streaming(
                        generator=gen_engine.module,
                        scheduler=scheduler,
                        img_latent=img_latent,
                        prompt_embeds=text_embeds,
                        negative_prompt_embeds=negative_embeds if do_cfg else None,
                        num_latent_frames=effective_F,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale,
                        do_cfg=do_cfg,
                        generator_rng=generator_rng,
                        device=device,
                        dtype=dtype,
                        patch_size=gen_patch_size,
                        max_cache_frames=args.max_cache_frames,
                        train_window_size=args.train_window_size if h2_window else None,
                        train_window_start=h2_t_start,
                        gt_far_latents=h2_gt_latents,
                        pe_mode=args.pe_mode,
                        # Critic-prep regen uses NORMALIZED skel (matches student)
                        control_video_latent=control_latent_dmd_norm,
                    )

                # Critic loss — pass critic_engine so backward goes through ZeRO-3.
                crit_loss = compute_critic_loss(
                    generated_latents=generated_no_grad,
                    critic=critic_engine,
                    conditional_dict=conditional_dict,
                    num_train_timesteps=scheduler.config.num_train_timesteps,
                    min_step_frac=args.min_step_frac,
                    max_step_frac=args.max_step_frac,
                    patch_size=gen_patch_size,
                    student_control_latent=control_latent_dmd_norm,
                )

                critic_engine.backward(crit_loss)
                critic_engine.step()
                try:
                    _gn = critic_engine.get_global_grad_norm()
                    log_dict["crit_grad_norm"] = float(_gn) if _gn is not None else float("nan")
                except Exception:
                    pass

                log_dict["gen_loss"] = gen_loss.item()
                log_dict["crit_loss"] = crit_loss.item()
                # Attribute DMD's gen_loss to the SGT bucket that drove the
                # grad-bearing generate_streaming call (the one in the
                # generator-update branch — critic prep uses with_grad=False
                # so its `_record_sgt_pick` call is a no-op, and the bucket
                # stays correctly pinned to the gen call). See ODE branch
                # above for the rationale.
                _attribute_sgt_loss(gen_loss.item())
                # Log per-step generation geometry so we can later grep for
                # "this iter generated a 7F clip vs that iter generated 21F"
                # — variable-length data path (epic_rdt.json has both).
                log_dict["effective_F"] = effective_F
            else:
                # Only critic step — same H2 args as gen step (same T_start,
                # same GT) so critic and generator stay aligned.
                with torch.no_grad():
                    generated_no_grad = generate_streaming(
                        generator=gen_engine.module,
                        scheduler=scheduler,
                        img_latent=img_latent,
                        prompt_embeds=text_embeds,
                        negative_prompt_embeds=negative_embeds if do_cfg else None,
                        num_latent_frames=effective_F,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale,
                        do_cfg=do_cfg,
                        generator_rng=generator_rng,
                        device=device,
                        dtype=dtype,
                        patch_size=gen_patch_size,
                        max_cache_frames=args.max_cache_frames,
                        train_window_size=args.train_window_size if h2_window else None,
                        train_window_start=h2_t_start,
                        gt_far_latents=h2_gt_latents,
                        pe_mode=args.pe_mode,
                        # Critic-only regen also uses NORMALIZED skel
                        control_video_latent=control_latent_dmd_norm,
                    )

                crit_loss = compute_critic_loss(
                    generated_latents=generated_no_grad,
                    critic=critic_engine,
                    conditional_dict=conditional_dict,
                    num_train_timesteps=scheduler.config.num_train_timesteps,
                    min_step_frac=args.min_step_frac,
                    max_step_frac=args.max_step_frac,
                    patch_size=gen_patch_size,
                    student_control_latent=control_latent_dmd_norm,
                )

                critic_engine.backward(crit_loss)
                critic_engine.step()
                try:
                    _gn = critic_engine.get_global_grad_norm()
                    log_dict["crit_grad_norm"] = float(_gn) if _gn is not None else float("nan")
                except Exception:
                    pass

                log_dict["crit_loss"] = crit_loss.item()

        step += 1
        step_time = time.time() - step_start
        step_times_window.append(step_time)
        if len(step_times_window) > 100:
            step_times_window.pop(0)

        # Announce warmup completion exactly once (rank 0). Triggered when
        # we just finished the LAST warmup step (step is now = warmup_steps).
        #
        # B 方案 fix (2026-05-13): the original guard was
        #   `ode_dataset is not None and is_warmup_step and step == warmup_steps`
        # which silently no-op'd in B 方案 (gt-warmup) because ode_dataset
        # stays None throughout — that branch never loads the pair cache
        # (see L3253-3258 ode_dataset init). Net effect on any B-mode run:
        # NO warmup-complete log line, NO LR reset at the warmup→DMD switch
        # even when --reset_lr_at_dmd was set. The generator therefore hit
        # DMD step 1 with lr already at the warmup-end value (~2e-6 for
        # lr_g) instead of warm-restarting from 0, which can slam an
        # under-trained critic with full-magnitude updates → DMD divergence.
        # New guard: trigger if EITHER teacher-mode pair cache is loaded
        # OR we're in gt/v_flow-mode warmup. All three modes are
        # is_warmup_step=True at this step, but the gt/v_flow branches don't
        # use ode_dataset (it stays None throughout). The 2026-05-14 widening
        # to include "v_flow" prevents the same silent-no-op bug the comment
        # above describes for the historical "gt" case — without it,
        # --reset_lr_at_dmd would silently fail in v_flow production runs,
        # leaving lr_g at ~2e-6 when DMD step 1 hits an under-trained critic.
        if ((ode_dataset is not None
             or args.ode_target_kind in ("gt", "v_flow"))
                and is_warmup_step
                and step == args.ode_warmup_steps):
            if is_main:
                cprint(
                    f"[ode] === Warmup complete at step {step}. "
                    f"Switching to DMD. ===",
                    "yellow",
                )
                sys.stdout.flush()

            # Variant G: reset both LR schedulers so DMD starts from lr=0 and
            # re-runs its own warmup. See parser docstring for rationale.
            # This must run on EVERY rank (not just is_main) — each rank holds
            # its own scheduler instance and they must stay in lock-step or the
            # next gen_engine.step() / critic_engine.step() will desync the
            # per-param-group lr across ranks (DeepSpeed's all-reduce of grads
            # is fine but the lr scaling happens locally).
            #
            # last_epoch=-1 + zeroing param-group lrs makes the next .step()
            # advance to last_epoch=0 with lr_lambda(0) ≈ 0, effectively
            # restarting the warmup curve for the remaining
            # (num_train_steps - ode_warmup_steps) steps.
            #
            # ────────── BUG FIX 2026-05-18 (Run C delay diagnosis) ──────────
            # The original reset block only touched the LambdaLR's `last_epoch`
            # and the param-group `lr`. It did NOT reset DeepSpeed's internal
            # `micro_steps` counter. DeepSpeedEngine.step() consults
            # `is_gradient_accumulation_boundary()` which checks
            # `(self.micro_steps + 1) % gas == 0`. If reset happens mid-GAS-cycle
            # (which it almost always does, because:
            #   - MSE phase only ever calls gen_engine.step() — critic_engine
            #     was NEVER stepped during warmup, so critic_engine.micro_steps
            #     is still 0 at the boundary,
            #   - gen_engine.micro_steps == (warmup_steps − ode_skips), which
            #     is rarely ≡ 0 mod GAS due to the FUSE-bit-flip SKIP path
            #     bypassing engine.step()),
            # then the FIRST several engine.step() calls in DMD just bump
            # micro_steps without triggering _take_model_step() → lr_lambda(0)
            # is never invoked → pg["lr"] stays at 0.0 we just zeroed.
            #
            # Concretely on Run C `_v_flow_dmd_full/run_20260517_150336_normandy`:
            #   GAS=8, dfake_gen_update_ratio=5,
            #   gen_engine.step() called every 5th DMD step,
            #   critic_engine.step() called every DMD step
            # → critic lr_c: 0 → 4e-7 took 8 DMD steps  (matches log: step 2008)
            # → gen    lr_g: 0 → 2e-6 took 26 DMD steps (matches log: step 2026)
            #
            # Result: 26-step "phantom warmup" where critic is updating against
            # a frozen student, causing the critic to over-fit a non-evolving
            # distribution. When gen finally starts updating, the critic's
            # gradient field is already badly miscalibrated → mode collapse /
            # divergence in the next 100-300 DMD steps (gen_lat std climbed
            # 1.1 → 3.6 over steps 2050-2400 in Run C, matching this story).
            #
            # Fix: zero out `engine.micro_steps` so the next engine.step()
            # is immediately a GAS boundary → _take_model_step() runs →
            # scheduler.step() invokes lr_lambda(0) → with warmup_ratio=0
            # the lambda returns 1.0 → pg["lr"] = base_lr immediately.
            # ─────────────────────────────────────────────────────────────────
            if args.reset_lr_at_dmd:
                for sched, opt, name in [
                    (gen_scheduler, gen_optimizer, "gen"),
                    (critic_scheduler, critic_optimizer, "critic"),
                ]:
                    sched.last_epoch = -1
                    for pg in opt.param_groups:
                        pg["lr"] = 0.0

                # Pre-position DS micro_steps so the NEXT engine.step() is a
                # GAS boundary on BOTH engines. Without this the LR reset
                # above has no effect for the first (GAS−1)*dfake_gen_update_ratio
                # steps on the generator and (GAS−1) steps on the critic.
                #
                # Why micro_steps = GAS - 1 (not 0):
                # DeepSpeedEngine.step() checks boundary with
                #   (self.micro_steps + 1) % gas == 0
                # then increments micro_steps += 1 at the end. So to make the
                # immediately following step() a boundary we need (X+1) % gas == 0
                # entering, i.e. X = gas - 1. (Setting micro_steps=0 would still
                # require GAS calls to reach the next boundary, which on this
                # run would mean lr_c stays 0 until step 2008 and lr_g until
                # step 2026 — i.e. exactly the bug we're fixing.)
                #
                # `micro_steps` is a public int attribute of DeepSpeedEngine
                # (see deepspeed/runtime/engine.py:117 `self.micro_steps += 1`),
                # initialized to 0 in __init__. Leaves `global_steps` untouched —
                # that counter is purely for logging/profiling and isn't used
                # in boundary checks.
                _gas = int(args.gradient_accumulation_steps)
                _pre_reset_micro = {
                    "gen": getattr(gen_engine, "micro_steps", None),
                    "critic": getattr(critic_engine, "micro_steps", None),
                }
                try:
                    gen_engine.micro_steps = _gas - 1
                    critic_engine.micro_steps = _gas - 1
                except Exception as _e:
                    # If the attribute name ever changes in a future DS
                    # release, surface it loudly rather than silently leaving
                    # the lr at 0 for many DMD steps.
                    if is_main:
                        cprint(
                            "[warmup] WARN: failed to reset DeepSpeed micro_steps "
                            f"({_e!r}). LR reset may not take effect immediately; "
                            "expect lr_g/lr_c to stay 0 for several DMD steps.",
                            "red",
                        )

                if is_main:
                    # Note: log prefix is `[warmup]` (generic) not `[ode]`
                    # because phase-1 is NOT always ODE regression — it can
                    # be teacher/gt/v_flow/causvid_traj depending on
                    # --ode_target_kind. The legacy `ode_warmup_steps` arg
                    # name is kept for backward CLI compatibility, but only
                    # `causvid_traj` (Run D) is actually ODE-trajectory
                    # regression; teacher/gt/v_flow modes are pure MSE.
                    cprint(
                        f"[warmup] reset_lr_at_dmd=True (target_kind="
                        f"{args.ode_target_kind}) → both LR schedulers reset "
                        f"to last_epoch=-1; DMD will run its own "
                        f"{int(args.warmup_ratio * args.num_train_steps)}-step "
                        "warmup from lr=0.",
                        "yellow",
                    )
                    cprint(
                        f"[warmup]   pre-reset micro_steps: "
                        f"gen={_pre_reset_micro['gen']}, "
                        f"critic={_pre_reset_micro['critic']}  →  both set to "
                        f"GAS-1={_gas - 1} (GAS={_gas}) so the next "
                        "engine.step() is a boundary and lr_lambda(0) "
                        "immediately restores base_lr.",
                        "yellow",
                    )
                    sys.stdout.flush()

        # Periodic visual decode for sanity-checking convergence.
        # Uses `generated_no_grad` which is always defined by this point
        # (both code paths above run a no-grad streaming generation for the
        # critic step). The decode itself is rank-0-only (~1-2 s on rank 0;
        # other ranks proceed past it almost immediately) BUT the multi-scene
        # extras need full streaming forwards, and under ZeRO-3 + param
        # offload those forwards are COLLECTIVE ops (param all-gather across
        # all 8 ranks). They MUST run on every rank in lock-step — calling
        # them from inside the rank-0-only decode block would deadlock
        # forever (rank 0 waits at the gather while ranks 1-7 are already
        # at the next training iteration's collective).
        #
        # Solution: run the extra forwards HERE, outside the rank-0 guard,
        # then hand the resulting tensors (rank 0's copies are the only
        # ones we need) into _decode_and_save_frame. To keep the collective
        # loop count identical on every rank — even if shards differ in
        # size by ±1 — we pad each rank's pair-index list to exactly
        # n_extra, repeating idx=0 (always exists).
        # Decode trigger: every `decode_every` steps, PLUS always trigger at
        # the first step where DMD-truth ring buffer has data populated. This
        # gates first-decode on `_DMD_DEBUG_WRITES[0] > 0` so that the dmd
        # grid (= which needs ring data) doesn't silently skip on the first
        # decode. Without this gate, first-decode fires at step 2001 but
        # `compute_dmd_generator_loss` only runs every `dfake_gen_update_ratio`
        # steps (default 5) — so step 2001 = critic-only step → ring empty
        # → dmd_a/b.jpg silent skip → user sees incomplete first decode.
        # By gating on ring-non-empty, first-decode fires at the first
        # gen-update step (= step 2005 with default 5x ratio) where all 5
        # files (grid_a/b/c + dmd_a/b) come out together.
        # In ODE warmup phase ring stays empty (warmup uses MSE not DMD), so
        # first-decode falls through to the regular `% decode_every == 0`
        # path. After warmup→DMD switch, the gate kicks in correctly.
        if not hasattr(main, "_decode_first_done"):
            main._decode_first_done = False
        _ring_has_data = (
            _DMD_DEBUG_RING_SIZE > 0 and _DMD_DEBUG_WRITES[0] > 0
        )
        # 2026-05-25: SKIP_FIRST_DECODE=1 跳过 step 1 的 decode (~15min on
        # multi-node with 3 grids). Regular cadence (decode_every) still fires.
        # Useful for production runs where startup time matters more than the
        # baseline-quality grid at step 1.
        _skip_first_decode = (os.environ.get("SKIP_FIRST_DECODE", "0") == "1")
        _is_first_step_decode = (
            args.decode_every > 0
            and not main._decode_first_done
            and (_ring_has_data or is_warmup_step)
            and not _skip_first_decode
            # warmup branch has no dmd grid by design, so trigger immediately
        )
        # If skip-first-decode is on, mark "first done" up front so the
        # first-step branch never fires later (regular cadence unaffected).
        if _skip_first_decode and not main._decode_first_done:
            main._decode_first_done = True
            if rank == 0 and step == 1:
                cprint(f"[decode] first-step decode SKIPPED (SKIP_FIRST_DECODE=1). "
                       f"Regular cadence still every {args.decode_every} steps.",
                       "yellow")
        if (args.decode_every > 0 and step % args.decode_every == 0) \
                or _is_first_step_decode:
            if _is_first_step_decode:
                main._decode_first_done = True
                if rank == 0:
                    cprint(f"[decode] first-step decode triggered at step {step} "
                           f"(ring_has_data={_ring_has_data}, is_warmup={is_warmup_step}; "
                           f"regular cadence still every {args.decode_every} steps)",
                           "magenta")
            # DECODE_GRID_REPEATS env (default 1): output N independent grid
            # files per decode call, each with a different extras seed so we
            # see N × decode_num_scenes distinct scenes per step. File names
            # are suffixed `_a`, `_b`, ... (only when N > 1).
            # 2026-05-25: at step 1 specifically, force n_grids=1 to save
            # ~10min — the 3-grid variety is only useful for later steps where
            # we want to see 12 scenes (3 grids × 4) of the trained student.
            # Step 1 student is barely trained, 1 grid is all we need.
            n_grids_env = max(1, int(os.environ.get("DECODE_GRID_REPEATS", "1") or 1))
            n_grids = 1 if (_is_first_step_decode and n_grids_env > 1) else n_grids_env
            if rank == 0 and _is_first_step_decode and n_grids_env > 1:
                cprint(f"[decode] first-step decode reduced to 1 grid (vs "
                       f"DECODE_GRID_REPEATS={n_grids_env}) to save startup time. "
                       f"Subsequent decodes will use full repeats.",
                       "yellow")
            in_warmup_window = step < args.ode_warmup_steps
            use_teacher_extras = (in_warmup_window
                                  and args.ode_target_kind in ("teacher", "causvid_traj")
                                  and ode_dataset is not None
                                  and len(ode_dataset) >= 1)
            use_gt_extras = (args.ode_target_kind in ("gt", "v_flow")
                             and dataset is not None
                             and len(dataset) >= 1)
            # Whether to also draw a "teacher streaming" row (frozen base
            # Wan2.2-5B running streaming under the same H3 settings as
            # student). Independent baseline: motion in this row proves the
            # streaming + sliding-cache + slot-RoPE path can produce motion
            # at all; absence of motion in student-only is then a training
            # issue, not an architecture issue.
            # DECODE_TEACHER_STREAM (env, default off): independent base-Wan2.2
            # streaming row. Useful when investigating sliding-cache / slot-RoPE
            # behaviour, but on EPIC OOD prompts the base teacher occasionally
            # drifts (verified 2026-05-14: not a bug, just OOD), so keep it off
            # by default. The GT row above is the canonical "target manifold"
            # reference; teacher_stream is opt-in.
            _stream_env = os.environ.get("DECODE_TEACHER_STREAM", "0")
            _stream_on = _stream_env in ("1", "true", "True", "on", "yes")
            with_teacher_stream = (_stream_on
                                   and args.decode_with_teacher
                                   and args.ode_target_kind in ("gt", "v_flow")
                                   and dataset is not None
                                   and len(dataset) >= 1)

            n_extra = max(0, args.decode_num_scenes - 1)
            for grid_i in range(n_grids):
                # Per-grid nonce → different extras choices each grid. Use
                # a 32-bit Knuth-multiplier mix to spread step ↔ grid_i.
                nonce = (step ^ (grid_i * 0x9E3779B1)) & 0x7FFFFFFF
                extra_student_latents = []
                extra_pair_indices = []
                local_extras = []
                if n_extra > 0 and (use_teacher_extras or use_gt_extras):
                    if use_teacher_extras:
                        live_idx = _current_pair_idx_holder["idx"]
                        local_extras = _resolve_extras_with_fixed(
                            n_extra, live_idx, nonce, mode='pair')
                        pool_pad = 0
                        infer_fn = _student_inference_for_pair
                    else:
                        live_idx = _current_gt_idx_holder["idx"]
                        local_extras = _resolve_extras_with_fixed(
                            n_extra, live_idx, nonce, mode='gt')
                        pool_pad = 0
                        infer_fn = _student_inference_for_gt_idx
                    real_count = len(local_extras)
                    while len(local_extras) < n_extra:
                        local_extras.append(pool_pad)
                    with torch.no_grad():
                        for i, px in enumerate(local_extras):
                            # Collective: every rank participates (ZeRO-3
                            # param all-gather happens inside).
                            latent = infer_fn(px)
                            if is_main and i < real_count:
                                extra_student_latents.append(latent)
                                extra_pair_indices.append(px)
                else:
                    real_count = 0

                # ── Teacher streaming row (independent baseline) ──
                # `real_score` aliases pipe.transformer with requires_grad=False
                # and is NOT wrapped in any DeepSpeed engine. Each rank has
                # its own full-precision copy and the forward contains no
                # collective ops, so rank-0-only is safe (no deadlock risk).
                # The next collective in the train loop is the BEGINNING of
                # the next iter — by then rank-0 is past this block.
                teacher_streaming_latents = []
                if with_teacher_stream and is_main:
                    # Build the GT-idx list this grid will visualize:
                    # [live_idx] + extras (real_count, not pad).
                    live_gt_idx = _current_gt_idx_holder["idx"]
                    gt_idx_list = []
                    if live_gt_idx is not None and live_gt_idx >= 0:
                        gt_idx_list.append(live_gt_idx)
                    gt_idx_list += local_extras[:real_count]
                    with torch.no_grad():
                        for gi in gt_idx_list:
                            try:
                                t_lat = _teacher_inference_for_gt_idx(gi)
                                teacher_streaming_latents.append(t_lat)
                            except Exception as _e:
                                cprint(f"[decode] teacher streaming for "
                                       f"idx={gi} failed: {_e!r}", "red")
                                teacher_streaming_latents.append(None)

                # ── DMD-signal teacher rows (single-step real_score) ──
                # Visualizes what `compute_dmd_generator_loss` actually
                # feeds the student each train step: at random t in
                # [20,980], the implicit target is
                #   x0_recon = noisy − σ * real_score(noisy, t).
                # We sample 3 t values that span the DMD training range:
                #   t=200 (low noise), 500 (mid), 800 (high noise).
                # All rank-0-only — real_score forward has no collective.
                # Env DECODE_DMD_TEACHER_T="" disables; default = three-tier.
                # Gated `not in_warmup_window` so MSE warmup grids stay
                # focused on (student vs GT) — DMD-time real_score signal
                # is not the MSE target and only adds visual noise to the
                # warmup-phase debug image.
                _t_csv = os.environ.get("DECODE_DMD_TEACHER_T", "200,500,800")
                _t_list = [int(x) for x in _t_csv.split(",") if x.strip()]
                teacher_dmd_latents: dict = {}
                if (_t_list and is_main
                        and not in_warmup_window
                        and args.ode_target_kind in ("gt", "v_flow")
                        and dataset is not None):
                    live_gt_idx2 = _current_gt_idx_holder["idx"]
                    gt_idx_list2 = []
                    if live_gt_idx2 is not None and live_gt_idx2 >= 0:
                        gt_idx_list2.append(live_gt_idx2)
                    gt_idx_list2 += local_extras[:real_count]
                    for _t in _t_list:
                        per_t_lats = []
                        with torch.no_grad():
                            for gi in gt_idx_list2:
                                try:
                                    per_t_lats.append(
                                        _dmd_teacher_signal_for_gt_idx(gi, _t))
                                except Exception as _e:
                                    cprint(f"[decode] DMD teacher@t={_t} for "
                                           f"idx={gi} failed: {_e!r}", "red")
                                    per_t_lats.append(None)
                        teacher_dmd_latents[_t] = per_t_lats

                # File suffix: a, b, c, ... when n_grids > 1.
                suffix = chr(ord('a') + grid_i) if n_grids > 1 else ""
                _decode_and_save_frame(generated_no_grad, step,
                                       extra_student_latents=extra_student_latents,
                                       extra_pair_indices=extra_pair_indices,
                                       teacher_streaming_latents=teacher_streaming_latents,
                                       teacher_dmd_latents=teacher_dmd_latents,
                                       grid_suffix=suffix)

            # ── DMD-loss truth grid (added 2026-05-18) ──────────────────
            # NEW debug file (distinct from above), independent layout.
            # Each file shows the EXACT 4 tensors
            # `compute_dmd_generator_loss` differentiated for ONE recent
            # DMD step (student / fake_x0 / real_x0 / noised_input),
            # pulled from `_DMD_DEBUG_RING`.
            #
            # The ring (default size 2) stores the last 2 DMD snapshots,
            # so each decode trigger emits up to 2 grids (_a = older,
            # _b = newer) — that's how we honor "每次保存两张":
            # genuinely 2 distinct DMD steps, not 2 copies of the same.
            #
            # - DMD-phase only (warmup snapshots never get populated).
            # - Rank-0-only (function self-guards).
            # - DMD_DEBUG_RING=0 / DECODE_DMD_DEBUG=0 disable.
            # - DECODE_DMD_MAX_SCENES caps batch rows rendered per grid.
            if (not is_warmup_step
                    and _DMD_DEBUG_RING_SIZE > 0
                    and os.environ.get("DECODE_DMD_DEBUG", "1") != "0"
                    and _DMD_DEBUG_WRITES[0] > 0):
                # Iterate ring entries in CHRONOLOGICAL order so file
                # suffix `_a` is the oldest snapshot, `_b` next, etc.
                # With writes counter W and ring size R, the oldest
                # occupied slot is (W − min(W,R)) % R; iterate forward
                # from there for min(W, R) entries.
                _w = _DMD_DEBUG_WRITES[0]
                _r = _DMD_DEBUG_RING_SIZE
                _n_filled = min(_w, _r)
                _start = (_w - _n_filled) % _r
                for _i in range(_n_filled):
                    _slot = _DMD_DEBUG_RING[(_start + _i) % _r]
                    _dmd_suffix = (chr(ord('a') + _i)
                                   if _n_filled > 1 else "")
                    _decode_and_save_dmd_grid(
                        step, _slot, grid_suffix=_dmd_suffix)

        # Logging (main process only)
        if step % args.logging_steps == 0 and is_main:
            elapsed = time.time() - start_time
            avg_step = sum(step_times_window) / len(step_times_window)
            remaining = max(0, args.num_train_steps - step)
            eta_sec = remaining * avg_step
            eta_str = f"{eta_sec / 3600:.1f}h" if eta_sec > 3600 else f"{eta_sec / 60:.1f}m"

            # GPU memory (from current rank's allocator)
            mem_alloc_gb = torch.cuda.memory_allocated() / 1024**3
            mem_reserved_gb = torch.cuda.memory_reserved() / 1024**3
            mem_peak_gb = torch.cuda.max_memory_allocated() / 1024**3

            # Current LR (after scheduler.step())
            try:
                cur_lr_gen = gen_optimizer.param_groups[0]["lr"]
                cur_lr_crit = critic_optimizer.param_groups[0]["lr"]
            except Exception:
                cur_lr_gen = cur_lr_crit = float("nan")

            # `effective_F` only set during DMD steps (not warmup). Show '-' otherwise.
            _eff_f_str = (
                f"F={int(log_dict['effective_F'])}"
                if "effective_F" in log_dict and log_dict.get("effective_F") not in (None, float("nan"))
                else "F=-"
            )
            # NOTE (2026-05-14, P0 fix): use `.4e` (scientific) for losses.
            # When v_flow warmup pushes the student close to the GT manifold,
            # the DMD loss can settle at ~1e-5 ~ 1e-4 — `.4f` rendered that
            # as a flat "0.0000" for 200 consecutive steps in the 1200-step
            # diagnostic run, hiding whether DMD was making any real
            # progress vs degenerating to zero. `.4e` keeps full magnitude
            # readable across the full ode→DMD dynamic range (~1e+0 in
            # warmup → ~1e-5 in DMD). Same applies to crit_loss.
            cprint(
                f"Step {step}/{args.num_train_steps}"
                + (" [ode-warmup]" if is_warmup_step else "")
                + f" | "
                f"gen_loss: {log_dict['gen_loss']:.4e} | "
                f"crit_loss: {log_dict['crit_loss']:.4e} | "
                f"lr_g: {cur_lr_gen:.2e} | lr_c: {cur_lr_crit:.2e} | "
                f"{_eff_f_str} | "
                f"t/step: {step_time:.2f}s (avg {avg_step:.2f}s) | "
                f"elapsed: {elapsed/60:.1f}m | ETA: {eta_str} | "
                f"GPU0: alloc {mem_alloc_gb:.1f}/peak {mem_peak_gb:.1f} GB",
                "green",
            )
            # Second line: DMD-specific health signals. Kept as a separate
            # cprint so a grep-friendly run can pull just `[diag]` lines.
            #   dmd_gn       — DMD pseudo-gradient mean-abs (~0.01-1; collapse=≪1e-3, blowup=≫10)
            #   gen_lat_*    — generator latent stats; std should stay O(1), absmax < 100ish
            #   gen_gn/crit_gn — DeepSpeed-reported pre-clip global grad norm
            #                    (clip threshold = 1.0 in ds_config; nan = bf16 overflow)
            #   sgt          — THIS iter's SGT bucket pick (s ∈ [0, NIS-1]; -1 = plain mode).
            #                  Per-bucket mean loss in the histogram below tells the FULL
            #                  picture (which buckets healed, which still spike).
            # NOTE (2026-05-14, P0+P1 fix):
            # • `dmd_gn` switched to `.4e` for the same reason as gen_loss
            #   above (it lives in the same ~1e-5 regime after v_flow
            #   warmup). `gen_gn` / `crit_gn` are DeepSpeed-reported pre-clip
            #   norms in O(1e-1 ~ 1e+0) territory — `.4e` here too for
            #   consistency and so a sudden bf16 overflow → 1e+30 reads
            #   distinctly from a healthy 5e-2.
            # • In WARMUP, the critic isn't being touched (no DMD step,
            #   `crit_loss = 0`, no critic backward), so `dmd_gn`,
            #   `crit_gn` are unconditionally NaN every line. Suppress them
            #   in warmup so the [diag] line shows only fields with
            #   meaningful values; resurfacing nan in DMD then becomes a
            #   real warning signal instead of background noise.
            if is_warmup_step:
                _diag_extra = ""
            else:
                _diag_extra = (
                    f"dmd_gn: {log_dict['dmd_grad_norm']:.4e} | "
                    f"crit_gn: {log_dict['crit_grad_norm']:.4e} | "
                )
            cprint(
                f"  [diag] step {step} | "
                f"{_diag_extra}"
                f"gen_lat: mean {log_dict['gen_latent_mean']:+.3f} std {log_dict['gen_latent_std']:.3f} absmax {log_dict['gen_latent_absmax']:.2f} | "
                f"gen_gn: {log_dict['gen_grad_norm']:.4e} | "
                f"sgt: s={_sgt_diag['last_grad_step_idx']}",
                "cyan",
            )
            # ── Per-channel diagnostic (2026-05-20): top-5 inflated channels ──
            # Compact format: c{ch}:s{std}|a{absmax}  (e.g. c34:s2.85|a18.2).
            # Whole-tensor std/absmax above ALWAYS hide the failure mode where
            # 2-3 specific channels blow up — the average looks fine because
            # the other 45 channels stay normal. The top-5 list surfaces
            # exactly which channels are drifting (=chroma-channel collapse
            # → visual color shift in decoded RGB).
            _top5_std = log_dict.get("gen_latent_per_ch_top5_std", [])
            _top5_abs = log_dict.get("gen_latent_per_ch_top5_absmax", [])
            if _top5_std and _top5_abs and isinstance(_top5_std, list):
                _per_ch_str = " ".join(
                    f"c{c}:s{s:.2f}|a{a:.1f}"
                    for (c, s), (_, a) in zip(_top5_std, _top5_abs)
                )
                cprint(
                    f"  [diag-ch] step {step} | top5_std_ch: {_per_ch_str}",
                    "cyan",
                )
            # ── DMD-specific diagnostics line — only meaningful on DMD steps ──
            # Six things to monitor:
            #   1. fake/real/gen_x0 absmean   — branch magnitudes; spotting
            #                                  collapse (fake→0) or runaway (gen→huge).
            #   2. raw vs norm grad           — normalizer effect; if `norm` ≪ `raw`
            #                                  the normalizer is doing its job;
            #                                  if `norm` ≈ `raw`, normalizer is ~1.
            #   3. normalizer min/mean/max    — if min hits clamp (1e-6), bug warning.
            #   4. grad↔p_real cosine         — direction sanity. POSITIVE = the
            #                                  DMD signal is teaching the right
            #                                  thing this step (after the v→x0
            #                                  fix this should consistently be
            #                                  in [0.3, 1.0]). NEGATIVE = bug.
            #   5. sigma min/mean/max         — current step's t distribution
            #                                  (each step samples a fresh batch;
            #                                  good to confirm we're hitting the
            #                                  full [0.02, 0.98] range over time).
            # Suppressed in warmup (no DMD computation runs that step).
            if not is_warmup_step and "dmd_norm_grad_absmean" in log_dict:
                cprint(
                    f"  [dmd]  step {step} | "
                    f"fake_x0: {log_dict['dmd_fake_x0_absmean']:.3e} | "
                    f"real_x0: {log_dict['dmd_real_x0_absmean']:.3e} | "
                    f"gen_x0:  {log_dict['dmd_gen_x0_absmean']:.3e} | "
                    f"raw_g:   {log_dict['dmd_raw_grad_absmean']:.3e} | "
                    f"norm_g:  {log_dict['dmd_norm_grad_absmean']:.3e} | "
                    f"normer[min/mean/max]: "
                    f"{log_dict['dmd_normalizer_min']:.3e}/"
                    f"{log_dict['dmd_normalizer_mean']:.3e}/"
                    f"{log_dict['dmd_normalizer_max']:.3e} | "
                    # B3 telemetry: fraction of batch samples whose
                    # normalizer fell below the threshold (these contribute
                    # zero grad this step). Non-zero = B3 fix actively
                    # protecting the gradient direction. Stays at 0.000
                    # under healthy DMD; spikes when generator briefly
                    # matches real_score x0 at the sampled σ.
                    f"safemask_zero: "
                    f"{log_dict.get('dmd_normalizer_safemask_zero_frac', 0.0):.3f} | "
                    f"cos(g, p_real): {log_dict['dmd_grad_p_real_cos']:+.3f} | "
                    f"σ[min/mean/max]: "
                    f"{log_dict['dmd_sigma_min']:.3f}/"
                    f"{log_dict['dmd_sigma_mean']:.3f}/"
                    f"{log_dict['dmd_sigma_max']:.3f}",
                    "magenta",
                )
                # ── New DMD diagnostics line (added 2026-05-18) ──
                # Read this AFTER the [dmd] line above for the per-step
                # health summary. Catches:
                #   - f0_dev > 0  → i2v invariant broken (bug)
                #   - grad[f1/fN] ratio > 5 or < 0.2 → SGT bucket imbalance
                #   - |v_diff|/|real_v| < 0.05 → critic learned nothing (B5/B6)
                #   - |v_diff|/|real_v| > 0.5  → critic blew up
                #   - cfg_str < 0.01 (only when real_guidance_scale > 0) →
                #                                CFG silent failure
                cprint(
                    f"  [dmd-x] step {step} | "
                    f"f0_dev[fake/real]: "
                    f"{log_dict.get('dmd_f0_dev_fake', 0.0):.1e}/"
                    f"{log_dict.get('dmd_f0_dev_real', 0.0):.1e} | "
                    f"grad[f1/fN/ratio]: "
                    f"{log_dict.get('dmd_grad_f1', float('nan')):.2e}/"
                    f"{log_dict.get('dmd_grad_fN', float('nan')):.2e}/"
                    f"{log_dict.get('dmd_grad_f1_fN_ratio', float('nan')):.2f} | "
                    f"|v|[fake/real/diff]: "
                    f"{log_dict.get('dmd_fake_v_absmean', float('nan')):.2e}/"
                    f"{log_dict.get('dmd_real_v_absmean', float('nan')):.2e}/"
                    f"{log_dict.get('dmd_v_diff_absmean', float('nan')):.2e} | "
                    f"v_diff/real: "
                    f"{log_dict.get('dmd_v_diff_to_real_ratio', float('nan')):.3f} | "
                    f"cfg_str: "
                    f"{log_dict.get('dmd_real_cfg_strength', 0.0):.3f}",
                    "magenta",
                )
                # ── Per-frame DMD diagnostics (2026-05-20) ──
                # raw_g and cos broken out for frame 1 (closest to cond,
                # full GT cache anchor), middle frame, and last frame
                # (cap=6 cache = no GT anchor, deepest rollout).
                # Healthy: rawg ratio ~1 (uniform across frames) AND cos
                # within ±20% across the three frames.
                # Pathological:
                #   - cos_fN ≪ cos_f1 → late-frame DMD gradient is noise;
                #     structural cache issue, lr/critic tuning won't fix
                #   - rawg_fN ≫ rawg_f1 → late frames dominate gradient;
                #     pushes student to over-correct late, drift early
                cprint(
                    f"  [dmd-f] step {step} | "
                    f"rawg[f1/mid/fN]: "
                    f"{log_dict.get('dmd_rawg_f1', float('nan')):.2e}/"
                    f"{log_dict.get('dmd_rawg_fMid', float('nan')):.2e}/"
                    f"{log_dict.get('dmd_rawg_fN', float('nan')):.2e} | "
                    f"cos[f1/mid/fN]: "
                    f"{log_dict.get('dmd_cos_f1', float('nan')):+.3f}/"
                    f"{log_dict.get('dmd_cos_fMid', float('nan')):+.3f}/"
                    f"{log_dict.get('dmd_cos_fN', float('nan')):+.3f}",
                    "magenta",
                )
                # ── σ-bucketed cos rolling histogram (every 50 steps) ──
                # Catches "DMD only works in some σ band" silent failures.
                # The deque + summary are module-level so they persist across
                # iterations; populated every DMD step, summarized every 50.
                try:
                    _record_dmd_sigma_cos(
                        log_dict['dmd_sigma_mean'],
                        log_dict['dmd_grad_p_real_cos'],
                    )
                    if step % 50 == 0:
                        cprint(_dmd_sigma_cos_summary(), "magenta")
                except Exception as _hist_e:
                    cprint(f"[dmd-sigma] WARN: histogram failed: {_hist_e!r}", "yellow")
                # ── JSONL dump (machine-readable, one line per step) ──
                # Goes to <output_dir>/dmd_metrics.jsonl. Easy post-hoc
                # analysis via: jq 'select(.cos < 0)' or pandas read_json.
                try:
                    _dmd_jsonl_write(args.output_dir, step, log_dict)
                except Exception as _jsonl_e:
                    cprint(f"[dmd-jsonl] WARN: write failed: {_jsonl_e!r}", "yellow")
            # Third line: SGT histogram (cumulative since training start).
            # Read this to diagnose the H3 red-blob phenomenon:
            #   - HEALTHY: µL roughly matches across s buckets, AND each
            #     bucket count is ~equal (uniform sampling working).
            #   - HIGH-NOISE TAIL STUCK: s=NIS-1 µL stays much larger than
            #     s=0 µL even after thousands of steps → the model isn't
            #     learning the high-noise 1-step x_0 prediction; expect
            #     persistent red-blob iters.
            #   - CRASHED: any µL → nan / inf, or bucket counts collapse to
            #     a single value → SGT broadcast is broken (would also
            #     manifest as NCCL desync hangs).
            # Cheap to print — a few dict lookups + one f-string format.
            cprint(
                f"  [diag] step {step} | sgt_hist {_sgt_hist_summary()}",
                "cyan",
            )
            sys.stdout.flush()

        # Periodic detailed memory dump for debugging slow OOMs / leaks.
        # Cheap (every N=100 steps), only on rank 0. Catches gradual creep that
        # the per-step line might hide because of fluctuation.
        if step % max(args.logging_steps * 10, 100) == 0 and is_main:
            gpu_summary = []
            for i in range(torch.cuda.device_count()):
                a = torch.cuda.memory_allocated(i) / 1024**3
                r = torch.cuda.memory_reserved(i) / 1024**3
                p = torch.cuda.max_memory_allocated(i) / 1024**3
                gpu_summary.append(f"gpu{i}: alloc {a:.1f}/reserv {r:.1f}/peak {p:.1f}")
            try:
                with open("/proc/meminfo") as f:
                    mi = {l.split(":")[0]: l.split(":")[1].strip() for l in f if ":" in l}
                cpu_str = f"CPU avail {mi.get('MemAvailable','?')}"
            except Exception:
                cpu_str = "CPU mem ?"
            cprint(f"[mem step={step}] {cpu_str} | " + " | ".join(gpu_summary), "magenta")
            sys.stdout.flush()

        # Saving — use DeepSpeed's per-engine checkpoint API. ZeRO-3 needs every
        # rank to participate so weights/optimizer shards can be gathered.
        if step % args.save_steps == 0:
            ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{step}")
            ckpt_t0 = time.time()
            if is_main:
                cprint(f"[ckpt] step {step}: saving to {ckpt_dir} ...", "cyan")
                sys.stdout.flush()
            # NOTE: We deliberately DO NOT call gen_engine.save_checkpoint() /
            # critic_engine.save_checkpoint() here. That API makes every rank
            # write its own ZeRO optim shard concurrently — for our 5B model
            # that's 8 ranks × ~9.2 GB = ~73 GB simultaneous writes. Some
            # network filesystems (e.g. FUSE-backed object storage) cannot
            # service that and can fail with errors like
            # "PytorchStreamWriter failed writing file data/2" part-way through
            # the second engine's save.
            #
            # Resume only needs the merged bf16 model state (see resume code
            # above — it loads generator.pt / critic.pt with weights_only=True
            # and never touches the ZeRO optim shards), so we keep ONLY the
            # rank-0 single-file save below. Cost: optimizer momentum is lost
            # on resume (acceptable; warmup re-establishes within ~50 steps).
            # Benefit: ~144 GB less per checkpoint, no concurrent FUSE writes.
            #
            # Stage rank-0's 12 GB write to local NVMe first, then rename onto
            # the (possibly FUSE-backed) output dir. Rename is atomic on POSIX
            # and on most FUSE backends, so a partial/interrupted write can
            # never leave a corrupt-looking checkpoint behind. If staging
            # fails (e.g. /root full), fall back to writing directly.
            STAGING_ROOT = os.environ.get("CKPT_STAGE_DIR", "/tmp/_ckpt_stage")
            if is_main:
                try:
                    os.makedirs(STAGING_ROOT, exist_ok=True)
                    stage_dir = os.path.join(STAGING_ROOT, f"step_{step}")
                    os.makedirs(stage_dir, exist_ok=True)
                except Exception as e:
                    cprint(f"[ckpt] WARN: cannot create staging dir {STAGING_ROOT}: {e}; "
                           f"falling back to direct write", "yellow")
                    stage_dir = None
            else:
                stage_dir = None
            # Broadcast stage_dir from rank 0 to all ranks so they all know
            # whether to use staging or direct write.
            if dist.is_initialized():
                obj_list = [stage_dir]
                dist.broadcast_object_list(obj_list, src=0)
                stage_dir = obj_list[0]

            write_dir = stage_dir if stage_dir is not None else ckpt_dir
            gen_engine.save_16bit_model(write_dir, save_filename="generator.pt")
            critic_engine.save_16bit_model(write_dir, save_filename="critic.pt")

            # Persist control_running_stats (add-plus normalize EMA). Without
            # this, resume re-inits EMA from first batch → ~few-hundred-step
            # distribution drift before stats stabilize. Rank 0 only.
            if is_main and control_running_stats["mean"] is not None:
                _stats_save_path = os.path.join(
                    write_dir, "control_running_stats.bin"
                )
                try:
                    torch.save({
                        "mean": control_running_stats["mean"].detach().cpu(),
                        "std": control_running_stats["std"].detach().cpu(),
                        "count": int(control_running_stats["count"]),
                        "momentum": float(control_running_stats["momentum"]),
                    }, _stats_save_path)
                    _mean_v = control_running_stats["mean"].mean().item()
                    _std_v = control_running_stats["std"].mean().item()
                    cprint(
                        f"[ckpt] step {step}: control_running_stats saved → "
                        f"{_stats_save_path} "
                        f"(count={control_running_stats['count']}, "
                        f"mean.mean={_mean_v:+.4f}, std.mean={_std_v:.4f})",
                        "cyan",
                    )
                except Exception as _e:
                    cprint(
                        f"[ckpt] WARN: control_running_stats save failed: "
                        f"{type(_e).__name__}: {_e!s:.150}",
                        "yellow",
                    )

            # ── Phase 2: Move staged main ckpt to real ckpt_dir BEFORE EMA save ──
            # 2026-05-25 reorder: previously EMA was saved to stage_dir BEFORE
            # the move loop. If EMA consolidate raised (e.g. control_scale 0-dim
            # bug observed at step 500 multi-node), the move never executed and
            # the entire main ckpt (generator.pt + critic.pt + ctrl_stats) was
            # lost with the staging dir. Now: commit main ckpt first via move,
            # then attempt EMA save directly to real ckpt_dir afterward — EMA
            # failure now only loses generator_ema.pt, not the main ckpt.
            #
            # rank 0 moves staged files into the real output dir.
            #
            # Why per-file try/except: OSS-FUSE has a known quirk where the
            # final fsync/close at the end of a multi-GB write returns EIO
            # *after* the data has actually made it to disk (observed
            # 2026-05-06 step 3000: generator.pt landed at the correct
            # 12,265,514,582 bytes but the move() call still raised
            # OSError(5)). Wrapping the whole loop in one try/except meant
            # the first file's spurious EIO killed the loop and critic.pt
            # never got moved at all — silently leaving an incomplete
            # checkpoint. Now each file moves independently AND we verify
            # by size; a size-match means we treat the spurious EIO as
            # success and clean up the staged source ourselves.
            if is_main and stage_dir is not None:
                try:
                    os.makedirs(ckpt_dir, exist_ok=True)
                except Exception as e:
                    cprint(f"[ckpt] ERROR: cannot mkdir {ckpt_dir}: {e}", "red")
                # Files to move out of staging. generator_ema.pt is included
                # only when EMA is enabled (otherwise it doesn't exist and the
                # `if not os.path.exists(src): continue` guard skips it harmlessly,
                # but listing it explicitly avoids spurious WARN spam).
                # 2026-05-25: removed generator_ema.pt from move list.
                # EMA is now saved AFTER move (directly to ckpt_dir). See
                # "Phase 2/3" reorder comments above.
                # Added control_running_stats.bin (add-plus skel normalize EMA)
                # so it's not silently lost when staging is on.
                _files_to_move = ["generator.pt", "critic.pt"]
                # control_running_stats.bin — only present if it was written.
                # Use os.path.exists check inside loop to skip silently.
                _files_to_move.append("control_running_stats.bin")
                for fname in _files_to_move:
                    src = os.path.join(stage_dir, fname)
                    dst = os.path.join(ckpt_dir, fname)
                    if not os.path.exists(src):
                        cprint(f"[ckpt] WARN: staged file missing: {src}", "yellow")
                        continue
                    # Capture intended size BEFORE attempting the move so we
                    # can verify dst even if shutil.move's internal unlink
                    # ran successfully (src gone) before the EIO triggered.
                    intended_sz = os.path.getsize(src)
                    try:
                        shutil.move(src, dst)
                    except Exception as e:
                        src_sz = os.path.getsize(src) if os.path.exists(src) else None
                        dst_sz = os.path.getsize(dst) if os.path.exists(dst) else None
                        # Two flavours of "spurious EIO" — both mean the data
                        # actually landed at dst with the right size:
                        #   1) src still there, dst has matching size (EIO during
                        #      copy_finalize or src-unlink after a successful copy).
                        #   2) src already gone, dst has the intended size (EIO
                        #      during a post-unlink fsync — even more common with
                        #      OSS-FUSE because unlink is metadata-fast).
                        if dst_sz == intended_sz:
                            cprint(
                                f"[ckpt] {fname}: move raised {type(e).__name__} but "
                                f"dst size matches intended ({dst_sz/1024**3:.1f} GB); "
                                f"treating as success (FUSE fsync quirk)",
                                "yellow",
                            )
                            if src_sz is not None:
                                try: os.unlink(src)
                                except Exception: pass
                        elif dst_sz is None and src_sz is not None:
                            # dst missing — real failure. Try one more time
                            # with explicit copy + verify before giving up.
                            try:
                                shutil.copy2(src, dst)
                                if os.path.getsize(dst) == src_sz:
                                    cprint(
                                        f"[ckpt] {fname}: retried via copy2 OK "
                                        f"({src_sz/1024**3:.1f} GB)",
                                        "yellow",
                                    )
                                    try: os.unlink(src)
                                    except Exception: pass
                                else:
                                    cprint(
                                        f"[ckpt] ERROR: {fname} retry size mismatch "
                                        f"(got {os.path.getsize(dst)}, want {src_sz})",
                                        "red",
                                    )
                            except Exception as e2:
                                cprint(
                                    f"[ckpt] ERROR: {fname} move failed "
                                    f"({type(e).__name__}: {e}); retry copy2 also failed: {e2}",
                                    "red",
                                )
                        else:
                            cprint(
                                f"[ckpt] ERROR: {fname} move failed "
                                f"(src_sz={src_sz} dst_sz={dst_sz}): {e}",
                                "red",
                            )
                # Best-effort stage cleanup; never raise (stage may be
                # partially-occupied if a file failed to move and we want
                # the next save to start clean).
                shutil.rmtree(stage_dir, ignore_errors=True)

            # Barrier so non-main ranks don't race past while rank0 finishes
            # the move (12 GB+ files take noticeable time).
            if dist.is_initialized():
                dist.barrier()

            # ── Phase 3: EMA save (after main ckpt committed) ──
            # 2026-05-25 reorder: now happens AFTER main ckpt is in real
            # ckpt_dir, so EMA failure doesn't lose the main ckpt.
            #
            # ALL ranks participate in _ema_consolidated_state_dict_zero3
            # (it does all_gather_into_tensor — collective op). Wrap the
            # call in try/except + sync the failure flag across ranks via
            # all_reduce, so any-rank failure → all-rank skip (no NCCL
            # desync from rank 0 raising while others wait at next collective).
            #
            # Defensive against future bugs:
            #   - 0-dim Parameter (control_scale) → already handled in fix above
            #   - Other view/reshape edge cases on scalar params → caught here
            #   - OSS write errors → caught here
            if gen_ema_shadow is not None:
                ema_consolidate_failed = False
                ema_sd = None
                try:
                    ema_sd = _ema_consolidated_state_dict(gen_ema_shadow, gen_engine)
                except Exception as _e:
                    ema_consolidate_failed = True
                    if is_main:
                        cprint(
                            f"[ckpt] step {step}: EMA consolidation FAILED "
                            f"({type(_e).__name__}: {_e!s:.200}). Main ckpt "
                            f"already committed — skipping generator_ema.pt only.",
                            "red",
                        )
                    import traceback as _tb
                    if is_main:
                        _tb.print_exc()

                # Sync failure flag — if ANY rank failed, all skip the save.
                # MAX op: any rank with flag=1 (failed) propagates to all.
                if dist.is_initialized():
                    _flag = torch.tensor(
                        [1 if ema_consolidate_failed else 0],
                        device=device, dtype=torch.int32,
                    )
                    dist.all_reduce(_flag, op=dist.ReduceOp.MAX)
                    ema_consolidate_failed = bool(_flag.item())

                if not ema_consolidate_failed and is_main and ema_sd is not None:
                    # EMA dest = real ckpt_dir directly (not stage). If staging
                    # was used, ckpt_dir is the final committed dir already.
                    ema_path = os.path.join(ckpt_dir, "generator_ema.pt")
                    try:
                        torch.save(ema_sd, ema_path)
                        cprint(f"[ckpt] step {step}: EMA saved → {ema_path}", "cyan")
                    except Exception as _e:
                        cprint(
                            f"[ckpt] step {step}: EMA save FAILED "
                            f"({type(_e).__name__}: {_e!s:.200}); "
                            f"main ckpt OK, only EMA missing for this step.",
                            "red",
                        )
                    del ema_sd

                # Barrier: ensure rank 0 finishes EMA save before next iter.
                if dist.is_initialized():
                    dist.barrier()

            if is_main:
                ckpt_dt = time.time() - ckpt_t0
                # Report disk usage of this checkpoint so log shows whether the
                # output dir is filling up faster than expected.
                try:
                    sz_bytes = sum(
                        os.path.getsize(os.path.join(dp, f))
                        for dp, _, files in os.walk(ckpt_dir) for f in files
                    )
                    sz_gb = sz_bytes / 1024**3
                except Exception:
                    sz_gb = float("nan")
                cprint(f"[ckpt] step {step}: saved in {ckpt_dt:.1f}s ({sz_gb:.1f} GB)", "cyan")

                # Sort numerically (see _ckpt_step above) so prune deletes the
                # genuinely-oldest dirs, not whatever lexicographic order says.
                all_ckpts = sorted(
                    [d for d in os.listdir(args.output_dir)
                     if d.startswith("checkpoint-") and _ckpt_step(d) >= 0],
                    key=_ckpt_step,
                )
                if args.save_total_limit > 0 and len(all_ckpts) > args.save_total_limit:
                    for old in all_ckpts[:len(all_ckpts) - args.save_total_limit]:
                        try:
                            shutil.rmtree(os.path.join(args.output_dir, old))
                            cprint(f"[ckpt] pruned old checkpoint: {old}", "yellow")
                        except Exception as e:
                            cprint(f"[ckpt] WARN: prune {old} failed: {e}", "yellow")
                sys.stdout.flush()

    # ── Save final model (post-loop) ────────────────────────────────
    # Reached when the while-loop exits cleanly (step == num_train_steps).
    # Every rank must enter save_16bit_model because it does an internal
    # all_gather to merge sharded ZeRO-3 weights into a single bf16 file
    # on rank 0; non-main ranks block inside the call until the gather
    # completes and then return.
    if dist.is_initialized():
        dist.barrier()
    final_dir = os.path.join(args.output_dir, "final")
    # Same rationale as the periodic save above: skip per-rank ZeRO sharded
    # save (crashes FUSE with 73 GB concurrent write); keep only rank-0
    # merged bf16 which is what downstream inference code reads.
    gen_engine.save_16bit_model(final_dir, save_filename="generator.pt")
    critic_engine.save_16bit_model(final_dir, save_filename="critic.pt")
    # 2026-05-25: persist control_running_stats to final ckpt too (mirrors
    # periodic save). Add-plus mode resume needs these stats.
    if is_main and control_running_stats["mean"] is not None:
        try:
            torch.save({
                "mean": control_running_stats["mean"].detach().cpu(),
                "std": control_running_stats["std"].detach().cpu(),
                "count": int(control_running_stats["count"]),
                "momentum": float(control_running_stats["momentum"]),
            }, os.path.join(final_dir, "control_running_stats.bin"))
            cprint(f"[ckpt] final: control_running_stats saved", "cyan")
        except Exception as _e:
            cprint(f"[ckpt] ERROR: final control_running_stats save failed: {_e}", "red")

    # EMA shadow (if active) — wrapped in try/except + sync flag (mirrors
    # periodic save above). EMA failure no longer crashes final save.
    if gen_ema_shadow is not None:
        ema_consolidate_failed = False
        ema_sd = None
        try:
            ema_sd = _ema_consolidated_state_dict(gen_ema_shadow, gen_engine)
        except Exception as _e:
            ema_consolidate_failed = True
            if is_main:
                cprint(
                    f"[ckpt] final: EMA consolidation FAILED "
                    f"({type(_e).__name__}: {_e!s:.200}); "
                    f"final main ckpt already committed.",
                    "red",
                )
                import traceback as _tb
                _tb.print_exc()
        if dist.is_initialized():
            _flag = torch.tensor(
                [1 if ema_consolidate_failed else 0],
                device=device, dtype=torch.int32,
            )
            dist.all_reduce(_flag, op=dist.ReduceOp.MAX)
            ema_consolidate_failed = bool(_flag.item())
        if not ema_consolidate_failed and is_main and ema_sd is not None:
            try:
                torch.save(ema_sd, os.path.join(final_dir, "generator_ema.pt"))
            except Exception as _e:
                cprint(f"[ckpt] ERROR: final EMA save failed: {_e}", "red")
            del ema_sd
    if is_main:
        cprint(f"Final model saved to {final_dir}", "green")


if __name__ == "__main__":
    main()