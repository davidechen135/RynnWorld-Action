"""
Utility functions for streaming distillation training.

Includes teacher checkpoint loading, EMA helpers, and memory diagnostics.
"""
import gc
import os
from typing import Optional, Tuple

import torch
from safetensors.torch import load_file as _lf
from termcolor import cprint


def load_teacher_into_pipe(
    pipe,
    teacher_ckpt_dir: Optional[str],
    use_ema: bool,
    is_main: bool,
    teacher_init_from_checkpoint: Optional[str] = None,
) -> Tuple[Optional[dict], Optional[float]]:
    """Load/fuse teacher weights into pipe.transformer.
    
    Returns (cpe_state_dict, control_scale_value) or (None, None).
    
    Supports 5 checkpoint formats:
    1. Legacy LoRA EMA (ema_weights.bin with lora_A/B/CPE keys)
    2. Legacy LoRA live (high_noise_lora/ + control_patch_embedding.bin)
    3. Full SFT pretrain (ema_weights.pt, no LoRA)
    4. SFT + LoRA + CPE (for DMD phase)
    5. Full SFT + CPE (no LoRA)
    """
    if teacher_ckpt_dir is None:
        if is_main:
            cprint("[teacher] No teacher_ckpt provided -- student initializes "
                   "from base Wan2.2 (CPE zero-init).", "yellow")
        return (None, None)

    teacher_ckpt_dir = os.path.abspath(teacher_ckpt_dir)
    if not os.path.isdir(teacher_ckpt_dir):
        raise FileNotFoundError(f"teacher_ckpt_dir does not exist: {teacher_ckpt_dir}")

    # Path detection
    cpe_bin_path = os.path.join(teacher_ckpt_dir, "control_patch_embedding.bin")
    control_scale_bin_path = os.path.join(teacher_ckpt_dir, "control_scale.bin")
    lora_safetensors_path = os.path.join(
        teacher_ckpt_dir, "high_noise_lora", "pytorch_lora_weights.safetensors")
    pytorch_model_dir = os.path.join(teacher_ckpt_dir, "pytorch_model")
    ema_pt_path = os.path.join(teacher_ckpt_dir, "ema_weights.pt")
    ema_bin_path = os.path.join(teacher_ckpt_dir, "ema_weights.bin")
    
    is_sft_plus_lora = (
        os.path.exists(cpe_bin_path)
        and os.path.exists(control_scale_bin_path)
        and os.path.exists(lora_safetensors_path)
    )
    is_full_sft_with_cpe = (
        os.path.isdir(pytorch_model_dir)
        and os.path.exists(cpe_bin_path)
        and os.path.exists(control_scale_bin_path)
        and not os.path.exists(lora_safetensors_path)
    )
    is_full_sft = os.path.isdir(pytorch_model_dir) and os.path.exists(ema_pt_path)

    # PATH 4: SFT + LoRA + CPE
    if is_sft_plus_lora:
        if teacher_init_from_checkpoint is None:
            raise ValueError(
                "[teacher] [PATH 4] Detected SFT+LoRA+CPE format but "
                "--teacher_init_from_checkpoint not provided."
            )
        if is_main:
            cprint(f"[teacher] [PATH 4: SFT+LoRA+CPE] {teacher_ckpt_dir}", "cyan")
        
        # Load SFT base
        sft_init_dir = os.path.abspath(teacher_init_from_checkpoint)
        from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
        sft_state = get_fp32_state_dict_from_zero_checkpoint(sft_init_dir, tag="pytorch_model")
        
        sd = pipe.transformer.state_dict()
        for k, v in sft_state.items():
            if k in sd:
                sd[k] = v.to(sd[k].dtype)
        pipe.transformer.load_state_dict(sd, strict=True)
        if is_main:
            cprint(f"[teacher] [PATH 4-a] SFT base loaded: {len(sft_state)} keys", "green")
        del sft_state, sd
        gc.collect()

        # Fuse LoRA
        lora_sd = _lf(lora_safetensors_path)
        a_map, b_map = {}, {}
        for k, v in lora_sd.items():
            k_stripped = k[len("transformer."):] if k.startswith("transformer.") else k
            if ".lora_A.weight" in k_stripped:
                a_map[k_stripped.replace(".lora_A.weight", "")] = v
            elif ".lora_B.weight" in k_stripped:
                b_map[k_stripped.replace(".lora_B.weight", "")] = v
        
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
            cprint(f"[teacher] [PATH 4-b] LoRA fused: {n_fused}/{len(a_map)} layers", "green")
        del lora_sd, a_map, b_map, sd
        gc.collect()

        # Load CPE
        ctrl_pe_state = torch.load(cpe_bin_path, map_location="cpu", weights_only=False)
        cs_loaded = torch.load(control_scale_bin_path, map_location="cpu", weights_only=False)
        cs_val = float(cs_loaded.item()) if hasattr(cs_loaded, "numel") and cs_loaded.numel() == 1 else float(cs_loaded)
        
        if is_main:
            cprint(f"[teacher] [PATH 4] CPE + control_scale={cs_val:.6f} loaded", "green")
        return (ctrl_pe_state, cs_val)

    # PATH 5: Full SFT + CPE (no LoRA)
    if is_full_sft_with_cpe:
        if is_main:
            cprint(f"[teacher] [PATH 5: SFT+CPE] {teacher_ckpt_dir}", "cyan")
        
        if use_ema and os.path.exists(ema_pt_path):
            full_state = torch.load(ema_pt_path, map_location="cpu", weights_only=False)
        elif use_ema and os.path.exists(ema_bin_path):
            full_state = torch.load(ema_bin_path, map_location="cpu", weights_only=False)
        else:
            from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
            full_state = get_fp32_state_dict_from_zero_checkpoint(teacher_ckpt_dir, tag="pytorch_model")
        
        cpe_skip = {"control_patch_embedding.weight", "control_patch_embedding.bias", "control_scale"}
        sd = pipe.transformer.state_dict()
        for k, v in full_state.items():
            if k in cpe_skip:
                continue
            if k in sd:
                sd[k] = v.to(sd[k].dtype)
        pipe.transformer.load_state_dict(sd, strict=True)
        if is_main:
            cprint(f"[teacher] [PATH 5] Base weights loaded", "green")
        del full_state, sd
        gc.collect()

        ctrl_pe_state = torch.load(cpe_bin_path, map_location="cpu", weights_only=False)
        cs_loaded = torch.load(control_scale_bin_path, map_location="cpu", weights_only=False)
        cs_val = float(cs_loaded.item()) if hasattr(cs_loaded, "numel") and cs_loaded.numel() == 1 else float(cs_loaded)
        
        if is_main:
            cprint(f"[teacher] [PATH 5] CPE + control_scale={cs_val:.6f} loaded", "green")
        return (ctrl_pe_state, cs_val)

    # PATH 3: Full SFT pretrain
    if is_full_sft:
        if use_ema:
            if is_main:
                cprint(f"[teacher] [SFT-EMA] Loading {ema_pt_path}", "cyan")
            ema_state = torch.load(ema_pt_path, map_location="cpu", weights_only=False)
        else:
            if is_main:
                cprint(f"[teacher] [SFT-RAW] Consolidating ZeRO shards...", "cyan")
            from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
            ema_state = get_fp32_state_dict_from_zero_checkpoint(teacher_ckpt_dir, tag="pytorch_model")
        
        sd = pipe.transformer.state_dict()
        for k, v in ema_state.items():
            if k in sd:
                sd[k] = v.to(sd[k].dtype)
        pipe.transformer.load_state_dict(sd, strict=True)
        if is_main:
            cprint(f"[teacher] [SFT] loaded {len(ema_state)} weights", "green")
        return (None, None)

    # Legacy LoRA paths
    if use_ema and os.path.exists(ema_bin_path):
        if is_main:
            cprint(f"[teacher] Loading EMA weights from {ema_bin_path}", "cyan")
        ema_state = torch.load(ema_bin_path, map_location="cpu", weights_only=False)
        ema_lora_a, ema_lora_b, ctrl_pe_state = {}, {}, {}
        for k, v in ema_state.items():
            if "control_patch_embedding" in k:
                ctrl_pe_state[k.replace("control_patch_embedding.", "")] = v
            elif ".lora_A." in k:
                ema_lora_a[k.split(".lora_A.")[0]] = v
            elif ".lora_B." in k:
                ema_lora_b[k.split(".lora_B.")[0]] = v
        
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
            cprint(f"[teacher] EMA LoRA fused: {n_fused} layers", "green")
        return (ctrl_pe_state if ctrl_pe_state else None, None)

    # Live LoRA
    if os.path.exists(lora_safetensors_path):
        if is_main:
            cprint(f"[teacher] Loading live LoRA from {lora_safetensors_path}", "cyan")
        lora_sd = _lf(lora_safetensors_path)
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
            cprint(f"[teacher] Live LoRA fused: {n_fused} layers", "green")

    # CPE
    cpe_path = os.path.join(teacher_ckpt_dir, "control_patch_embedding.bin")
    if os.path.exists(cpe_path):
        ctrl_pe_state = torch.load(cpe_path, map_location="cpu", weights_only=False)
        if is_main:
            cprint(f"[teacher] CPE loaded", "green")
        return (ctrl_pe_state, None)
    
    if is_main:
        cprint(f"[teacher] No CPE found; student CPE will be zero-init", "yellow")
    return (None, None)


def _mem_probe(tag: str, trace_tag: str = "", enabled: bool = False, do_sync: bool = True):
    """Lightweight CUDA memory probe for debugging."""
    if not enabled:
        return
    try:
        if do_sync:
            torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"[mem:{trace_tag}:{tag}] allocated={allocated:.2f}GB "
              f"reserved={reserved:.2f}GB peak={peak:.2f}GB", flush=True)
    except Exception as e:
        print(f"[mem:{trace_tag}:{tag}] probe failed: {e}", flush=True)
