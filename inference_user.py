"""
User-friendly inference: provide one first-frame image and one control video (mp4),
the model generates the corresponding video.

Optional: provide a text prompt (string), it will be encoded with the T5 text encoder.
If no prompt is given, an empty embedding is used.
"""
import argparse
import os
import torch
import torch.nn as nn
import types
import imageio.v2 as imageio
import cv2
import decord
import torch.nn.functional as F
from pathlib import Path
from safetensors.torch import load_file, save_file
from termcolor import cprint
from diffusers import AutoencoderKLWan
from transformers import T5TokenizerFast, UMT5EncoderModel

from core.inference.rynnworld_teleop import WanImagePipeline, wan_forward, safe_export_to_video


MODEL_PATH = os.environ.get('MODEL_PATH', 'pretrained/Wan2.2-TI2V-5B-Diffusers')
DEFAULT_CHECKPOINT = os.environ.get('CHECKPOINT_PATH')
NUM_FRAMES = 81
HEIGHT = 480
WIDTH = 832


def read_image(path, height, width):
    img = imageio.imread(path)
    if img.shape[-1] == 4:
        img = img[..., :3]
    if img.shape[0] != height or img.shape[1] != width:
        img = cv2.resize(img, (width, height))
    return img


def read_control_video(path, num_frames, height, width):
    """Read mp4 control video, sample/interpolate to exactly num_frames at target resolution."""
    reader = decord.VideoReader(uri=str(path), width=width, height=height)
    total = len(reader)
    if total < num_frames:
        # interpolate temporally
        frames = torch.from_numpy(reader.get_batch(range(total)).asnumpy()).float()
        frames = frames.permute(3, 0, 1, 2)  # [C, F, H, W]
        frames = F.interpolate(
            frames.unsqueeze(0),
            size=(num_frames, height, width),
            mode='trilinear',
            align_corners=False,
        ).squeeze(0).permute(1, 2, 3, 0)  # [F, H, W, C]
        return frames.byte().numpy()
    elif total <= int(1.5 * num_frames):
        indices = list(range(0, total, max(1, total // num_frames)))[:num_frames]
        return reader.get_batch(indices).asnumpy()
    else:
        return reader.get_batch(list(range(num_frames))).asnumpy()


def encode_image_to_latent(vae, image_np, device, dtype):
    """image_np: [H, W, C] uint8 -> img_latent [C_z, 1, H', W']"""
    tensor = torch.from_numpy(image_np).float().permute(2, 0, 1) / 255.0 * 2.0 - 1.0
    img_input = tensor.unsqueeze(0).unsqueeze(2).to(device=device, dtype=dtype)
    latents_mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(device=device, dtype=dtype)
    latents_std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(device=device, dtype=dtype)
    with torch.no_grad():
        latent = vae.encode(img_input).latent_dist.mode()
        latent = (latent - latents_mean) / latents_std
    return latent.squeeze(0).cpu().float()


def encode_video_to_latent(vae, video_np, device, dtype):
    """video_np: [F, H, W, C] uint8 -> latent [C_z, F', H', W']"""
    tensor = torch.from_numpy(video_np).float() / 255.0 * 2.0 - 1.0  # [F, H, W, C]
    video_input = tensor.permute(3, 0, 1, 2).unsqueeze(0).to(device=device, dtype=dtype)  # [1, C, F, H, W]
    latents_mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(device=device, dtype=dtype)
    latents_std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(device=device, dtype=dtype)
    with torch.no_grad():
        latent = vae.encode(video_input).latent_dist.mode()
        latent = (latent - latents_mean) / latents_std
    return latent.squeeze(0).cpu().float()


def encode_text(prompt, device, dtype):
    if not prompt:
        return None
    tokenizer = T5TokenizerFast.from_pretrained(MODEL_PATH, subfolder="tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(MODEL_PATH, subfolder="text_encoder").to(device=device, dtype=dtype)
    with torch.no_grad():
        ids = tokenizer(prompt, padding="max_length", max_length=226,
                        truncation=True, add_special_tokens=True, return_tensors="pt").input_ids
        emb = text_encoder(ids.to(device))[0]
    del text_encoder
    torch.cuda.empty_cache()
    return emb.squeeze(0).cpu()


def load_sft_weights(pipe, checkpoint_path, use_ema=True):
    """Load full transformer weights from SFT checkpoint (covers both transformer + control modules)."""
    if use_ema:
        ema_path = os.path.join(checkpoint_path, "ema_weights.bin")
        if not os.path.exists(ema_path):
            ema_path = os.path.join(checkpoint_path, "ema_weights.pt")
        cprint(f"Loading EMA weights from: {ema_path}", "green")
        state_dict = torch.load(ema_path, map_location="cpu", weights_only=False)
    else:
        raw_model_path = os.path.join(checkpoint_path, "pytorch_model", "mp_rank_00_model_states.pt")
        cprint(f"Loading raw weights from: {raw_model_path}", "green")
        raw_state = torch.load(raw_model_path, map_location="cpu", weights_only=False)
        state_dict = raw_state["module"]
        del raw_state

    control_patch_weights = {}
    control_scale_value = None
    transformer_weights = {}
    for k, v in state_dict.items():
        if k == 'control_scale':
            control_scale_value = v
        elif k.startswith('control_patch_embedding.'):
            control_patch_weights[k.replace('control_patch_embedding.', '')] = v
        else:
            transformer_weights[k] = v
    del state_dict

    missing, unexpected = pipe.transformer.load_state_dict(transformer_weights, strict=False)
    cprint(f"  Loaded {len(transformer_weights)} transformer weights | missing={len(missing)} unexpected={len(unexpected)}", "green")
    return control_patch_weights, control_scale_value


def load_lora_weights(pipe, checkpoint_path, rank, lora_alpha, target_modules, use_ema=True):
    """Attach a LoRA adapter to the base transformer and load weights from high_noise_lora/.
    Control modules are loaded separately from .bin files."""
    from peft import LoraConfig, set_peft_model_state_dict

    cprint(f"Attaching LoRA adapter (rank={rank}, alpha={lora_alpha})", "green")
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=lora_alpha,
        init_lora_weights=True,
        target_modules=target_modules,
    )
    pipe.transformer.add_adapter(lora_config, adapter_name="high_noise")

    high_noise_lora_path = os.path.join(checkpoint_path, "high_noise_lora")
    if not os.path.exists(high_noise_lora_path):
        raise FileNotFoundError(f"LoRA weights not found at {high_noise_lora_path}")
    cprint(f"Loading LoRA weights from: {high_noise_lora_path}", "green")
    lora_state_dict = pipe.lora_state_dict(high_noise_lora_path)
    set_peft_model_state_dict(pipe.transformer, lora_state_dict, adapter_name="high_noise")
    cprint(f"  Loaded {len(lora_state_dict)} LoRA weight tensors", "green")

    # Load control modules from sidecar .bin files
    control_patch_weights = {}
    control_scale_value = None
    cpe_path = os.path.join(checkpoint_path, "control_patch_embedding.bin")
    cs_path = os.path.join(checkpoint_path, "control_scale.bin")
    if os.path.exists(cpe_path):
        control_patch_weights = torch.load(cpe_path, map_location="cpu", weights_only=False)
        cprint(f"  Loaded control_patch_embedding.bin", "green")
    if os.path.exists(cs_path):
        control_scale_value = torch.load(cs_path, map_location="cpu", weights_only=False)
        cprint(f"  Loaded control_scale.bin", "green")
    return control_patch_weights, control_scale_value


def attach_control_modules(pipe, control_patch_weights, control_scale_value, control_type, dtype):
    """Attach control_patch_embedding (Conv3d) and control_scale (Parameter) onto the transformer."""
    pe = pipe.transformer.patch_embedding
    pe_device, pe_dtype = pe.weight.device, pe.weight.dtype

    if control_type in ('add', 'add-plus'):
        cpe = nn.Conv3d(pe.in_channels, pe.out_channels, kernel_size=pe.kernel_size, stride=pe.kernel_size).to(
            device=pe_device, dtype=pe_dtype)
        control_scale = nn.Parameter(torch.tensor(0.01, dtype=pe_dtype)).to(device=pe_device)
    elif control_type == 'concat':
        cpe = nn.Conv3d(pe.in_channels * 2, pe.out_channels, kernel_size=pe.kernel_size, stride=pe.kernel_size).to(
            device=pe_device, dtype=pe_dtype)
        control_scale = None

    if control_patch_weights:
        cpe.load_state_dict(control_patch_weights)
        cpe = cpe.to(device=pe_device, dtype=pe_dtype)
    pipe.transformer.control_patch_embedding = cpe

    if control_type in ('add', 'add-plus'):
        if control_scale_value is not None:
            control_scale.data = control_scale_value.squeeze().to(device=pe_device, dtype=pe_dtype)
            cprint(f"  control_scale: {control_scale.item():.4f}", "green")
        pipe.transformer.control_scale = control_scale


def load_pipeline(checkpoint_path, control_type, dtype, mode='sft',
                  lora_rank=64, lora_alpha=64, lora_target_modules=None, use_ema=True):
    pipe = WanImagePipeline.from_pretrained(MODEL_PATH, dtype=dtype)

    if mode == 'sft':
        control_patch_weights, control_scale_value = load_sft_weights(pipe, checkpoint_path, use_ema=use_ema)
    elif mode == 'lora':
        if lora_target_modules is None:
            lora_target_modules = ['attn1.to_q', 'attn1.to_k', 'attn1.to_v', 'attn1.to_out.0',
                                   'ffn.net.0.proj', 'ffn.net.2']
        control_patch_weights, control_scale_value = load_lora_weights(
            pipe, checkpoint_path, lora_rank, lora_alpha, lora_target_modules, use_ema=use_ema)
    else:
        raise ValueError(f"Unsupported mode: {mode!r} (expected 'sft' or 'lora')")

    attach_control_modules(pipe, control_patch_weights, control_scale_value, control_type, dtype)

    pipe.transformer.control_running_stats = None
    pipe.transformer.forward = types.MethodType(wan_forward, pipe.transformer)
    pipe.transformer.to(dtype=dtype)
    pipe.enable_model_cpu_offload()
    return pipe


def main(args):
    device = torch.device("cuda")
    dtype = torch.bfloat16

    os.makedirs(args.output, exist_ok=True)

    # 1. Read inputs
    cprint(f"\n[1/4] Reading inputs", "cyan")
    cprint(f"  image: {args.image}", "cyan")
    cprint(f"  control_video: {args.control_video}", "cyan")
    image_np = read_image(args.image, HEIGHT, WIDTH)
    control_np = read_control_video(args.control_video, NUM_FRAMES, HEIGHT, WIDTH)
    cprint(f"  image shape: {image_np.shape}, control shape: {control_np.shape}", "green")

    # 2. Encode via VAE & text
    cprint(f"\n[2/4] Encoding image + control video + text", "cyan")
    vae = AutoencoderKLWan.from_pretrained(MODEL_PATH, subfolder="vae").to(device=device, dtype=dtype)

    img_latent = encode_image_to_latent(vae, image_np, device, dtype)
    control_video_latents = encode_video_to_latent(vae, control_np, device, dtype)
    video_latents = control_video_latents.clone()  # placeholder, used only for GT decoding in pipeline
    cprint(f"  img_latent: {img_latent.shape}", "green")
    cprint(f"  control_video_latents: {control_video_latents.shape}", "green")

    prompt_embeds = None
    if args.prompt:
        cprint(f"  prompt: \"{args.prompt}\"", "cyan")
        text_emb = encode_text(args.prompt, device, dtype)
        prompt_embeds = text_emb.unsqueeze(0)
    elif args.text_embedding and os.path.exists(args.text_embedding):
        cprint(f"  text_embedding: {args.text_embedding}", "cyan")
        text_emb = load_file(args.text_embedding)["text_embedding"]
        prompt_embeds = text_emb.unsqueeze(0)
    else:
        cprint(f"  no prompt / text embedding (empty conditioning)", "yellow")

    # 3. Save assembled latent file
    cprint(f"\n[3/4] Saving assembled latent", "cyan")
    input_latent_path = os.path.join(args.output, "input_latent.safetensors")
    save_file({
        "video_latents": video_latents,
        "control_video_latents": control_video_latents,
        "img_latent": img_latent,
    }, input_latent_path)

    del vae
    torch.cuda.empty_cache()

    # 4. Load checkpoint & generate
    cprint(f"\n[4/4] Loading checkpoint and generating (mode={args.mode})", "cyan")
    lora_targets = [t.strip() for t in args.lora_target_modules.split(',') if t.strip()] if args.lora_target_modules else None
    pipe = load_pipeline(
        args.checkpoint, args.control_type, dtype,
        mode=args.mode,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_target_modules=lora_targets,
        use_ema=not args.no_ema,
    )

    seeds = [int(s) for s in args.seeds.split(',') if s.strip()]
    cprint(f"\nGenerating with seeds: {seeds}", "cyan")
    for seed in seeds:
        cprint(f"\n--- seed={seed} ---", "magenta")
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
        safe_export_to_video(output.frames[0], os.path.join(args.output, f"generated_seed{seed}.mp4"), fps=16)
        if seed == seeds[0]:
            safe_export_to_video(control_video.frames[0], os.path.join(args.output, "control.mp4"), fps=16)
            safe_export_to_video(control_video_raw.frames[0], os.path.join(args.output, "control_raw.mp4"), fps=16)

    cprint(f"\nDone! Outputs in {args.output}", "green")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="User-friendly inference: input a first-frame image and a control video mp4.")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to first-frame image (jpg/png)")
    parser.add_argument("--control_video", type=str, required=True,
                        help="Path to control video (mp4)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--prompt", type=str, default="",
                        help="Optional text prompt (will be encoded with T5)")
    parser.add_argument("--text_embedding", type=str, default=None,
                        help="Optional pre-encoded text embedding .safetensors (used if --prompt is empty)")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
                        help="Stage1/2 checkpoint directory (SFT or LoRA)")
    parser.add_argument("--mode", type=str, default="sft", choices=["sft", "lora"],
                        help="Checkpoint type: 'sft' (full weights) or 'lora' (adapter)")
    parser.add_argument("--lora_rank", type=int, default=64,
                        help="LoRA rank (must match training; only used when --mode lora)")
    parser.add_argument("--lora_alpha", type=int, default=64,
                        help="LoRA alpha (must match training; only used when --mode lora)")
    parser.add_argument("--lora_target_modules", type=str,
                        default="attn1.to_q,attn1.to_k,attn1.to_v,attn1.to_out.0,ffn.net.0.proj,ffn.net.2",
                        help="Comma-separated LoRA target module names (must match training)")
    parser.add_argument("--control_type", type=str, default="add")
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--no_ema", action="store_true")
    parser.add_argument("--seeds", type=str, default="42",
                        help="Comma-separated seeds, e.g. '42,123,7'")
    args = parser.parse_args()
    main(args)
