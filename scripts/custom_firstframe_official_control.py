"""Drive generation from YOUR OWN photo as the first frame, with the OFFICIAL
skeleton control latents.

Why this script exists instead of inference_user.py: that entry point requires
--control_video as an mp4, but the official demo control signal ships ONLY as
encoded latents (/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/video_latents/*.safetensors carries control_video_latents;
there is no mp4 anywhere in the repo). So we assemble the three-tuple directly in
latent space -- VAE-encode the user photo into img_latent, and carry the official
control_video_latents through untouched.

This is also the cleanest possible experiment for the collapse investigation:
the control signal is a KNOWN-GOOD one (official basic_fold dt=0.1523 vs my
AgiBot skeleton's 0.074), so the only variable is the first frame.

The pipeline hard-writes img_latent into latent frame 0 on every denoising step
(core/inference/rynnworld_teleop.py:452,496), which is what makes a swapped first
frame actually steer the rollout.

Usage:
  python scripts/custom_firstframe_official_control.py --image my_photo.jpg
  python scripts/custom_firstframe_official_control.py --image my_photo.jpg \
      --control basic_fold_1_6_rgb --prompt "Folding shorts"
"""
import argparse
import os
import sys

import torch
from safetensors.torch import load_file, save_file
from termcolor import cprint

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HEIGHT, WIDTH = 480, 832


def main(args):
    import inference_user as IU
    from diffusers import AutoencoderKLWan
    from core.inference.rynnworld_teleop import safe_export_to_video

    device, dtype = torch.device("cuda"), torch.bfloat16
    os.makedirs(args.output, exist_ok=True)

    ctl_path = args.control
    if not os.path.exists(ctl_path):
        ctl_path = f"/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/video_latents/{args.control}.safetensors"
    assert os.path.exists(ctl_path), f"control latent not found: {ctl_path}"

    # --- official control (the known-good signal; kept byte-identical) ---
    off = load_file(ctl_path)
    control = off["control_video_latents"]
    dt = (control[:, 1:] - control[:, :-1]).abs().mean().item()
    cprint(f"[control] {ctl_path}\n  shape={tuple(control.shape)}  temporal_dt={dt:.4f}", "green")

    # --- your photo -> img_latent (read_image resizes any resolution to 480x832) ---
    cprint(f"\n[first frame] {args.image}", "cyan")
    image_np = IU.read_image(args.image, HEIGHT, WIDTH)
    cprint(f"  read + resized -> {image_np.shape}", "green")

    vae = AutoencoderKLWan.from_pretrained(
        IU.MODEL_PATH, subfolder="vae").to(device=device, dtype=dtype)
    img_latent = IU.encode_image_to_latent(vae, image_np, device, dtype)
    cprint(f"  img_latent -> {tuple(img_latent.shape)}", "green")

    # video_latents is a placeholder: control_type='add' never feeds it into the
    # denoising loop, the pipeline only uses it to decode a GT reference strip.
    inp = os.path.join(args.output, "input_latent.safetensors")
    save_file({"video_latents": control.clone(),
               "control_video_latents": control,
               "img_latent": img_latent}, inp)
    cprint(f"  assembled -> {inp}", "green")

    del vae
    torch.cuda.empty_cache()

    # Text conditioning. Prefer an OFFICIAL pre-computed embedding when one exists
    # for this demo (/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/text_embeddings/*.safetensors) -- that reproduces official
    # SFT inference exactly, with no re-encoding drift. Fall back to encoding
    # --prompt ourselves for demos the official release shipped no embedding for.
    prompt_embeds = None
    if args.text_embedding:
        te = load_file(args.text_embedding)
        key = next(iter(te))
        prompt_embeds = te[key].to(device=device, dtype=dtype).unsqueeze(0)
        cprint(f"  text embedding: {args.text_embedding} [{key}] "
               f"{tuple(prompt_embeds.shape)} (official, pre-computed)", "green")
    elif args.prompt:
        emb = IU.encode_text(args.prompt, device, dtype)
        if emb is not None:
            prompt_embeds = emb.unsqueeze(0)
            cprint(f"  prompt: \"{args.prompt}\"", "green")

    cprint(f"\n[generate] {args.checkpoint} (mode={args.mode})", "cyan")
    pipe = IU.load_pipeline(args.checkpoint, args.control_type, dtype,
                            mode=args.mode, use_ema=not args.no_ema)

    for seed in [int(s) for s in args.seeds.split(",")]:
        gen = torch.Generator(device=device).manual_seed(seed)
        out, *_ = pipe(prompt='' if prompt_embeds is None else None,
                       negative_prompt='', guidance_scale=args.guidance_scale,
                       video_latent_path=inp, control_type=args.control_type,
                       prompt_embeds=prompt_embeds, generator=gen)
        frames = out.frames[0]
        mp4 = os.path.join(args.output, f"custom_firstframe_seed{seed}.mp4")
        safe_export_to_video(frames, mp4, fps=16)
        cprint(f"  -> {mp4}", "green")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="your own photo (any size/format)")
    p.add_argument("--control", default="basic_fold_1_6_rgb",
                   help="official demo name or path to a control latent")
    p.add_argument("--prompt", default="", help="task text, e.g. 'Folding shorts'")
    p.add_argument("--text_embedding", default="",
                   help="path to an official pre-computed text embedding; takes "
                        "precedence over --prompt (no re-encoding drift)")
    p.add_argument("--checkpoint", default="pretrained/RynnWorld-Teleop")
    p.add_argument("--mode", default="sft", choices=["sft", "lora"])
    p.add_argument("--control_type", default="add")
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--seeds", default="42")
    p.add_argument("--no_ema", action="store_true")
    p.add_argument("--output", default="outputs/custom_firstframe")
    main(p.parse_args())
