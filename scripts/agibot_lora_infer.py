"""Post-fine-tune inference: run the AgiBot-357 LoRA on the SAME zero-shot input
(same ego first frame + same 20-dim action control latent) and compare rollouts.

Isolates the fine-tuning effect: only the pipeline weights change (frozen SFT ->
SFT + AgiBot-357 LoRA + fine-tuned control_patch_embedding). Everything else --
first frame, action->control latent, text, seed -- is byte-identical to the
zero-shot run in outputs/agibot_zeroshot/357_648751/.

Usage:
  python scripts/agibot_lora_infer.py \
    --input outputs/agibot_zeroshot/357_648751/input_latent.safetensors \
    --lora-checkpoint training/agibot_357_lora/ema_final \
    --task-text "Washing dishes with a dishwasher" \
    --seeds 42 --output outputs/agibot_lora_357
"""
import argparse
import os
import sys

import torch
from safetensors.torch import load_file
from termcolor import cprint

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(args):
    import inference_user as IU

    device, dtype = torch.device("cuda"), torch.bfloat16
    os.makedirs(args.output, exist_ok=True)

    d = load_file(args.input)
    cprint(f"[input] {args.input}: " + ", ".join(f"{k}{tuple(v.shape)}" for k, v in d.items()), "green")

    prompt_embeds = None
    if args.task_text:
        emb = IU.encode_text(args.task_text, device, dtype)
        if emb is not None:
            prompt_embeds = emb.unsqueeze(0)
            cprint(f"[text] '{args.task_text}' -> {tuple(emb.shape)}", "green")

    cprint("\n[load] AgiBot-357 LoRA pipeline (rank 32) + fine-tuned control", "cyan")
    pipe = IU.load_pipeline(args.lora_checkpoint, args.control_type, dtype, mode="lora",
                            lora_rank=args.rank, lora_alpha=args.lora_alpha,
                            use_ema=not args.no_ema)

    cprint("\n[gen] fine-tuned rollout(s)", "cyan")
    from core.inference.rynnworld_teleop import safe_export_to_video
    for seed in [int(s) for s in args.seeds.split(",")]:
        gen = torch.Generator(device=device).manual_seed(seed)
        out, *_ = pipe(prompt='' if prompt_embeds is None else None, negative_prompt='',
                       guidance_scale=args.guidance_scale, video_latent_path=args.input,
                       control_type=args.control_type, prompt_embeds=prompt_embeds, generator=gen)
        outp = os.path.join(args.output, f"lora_rollout_seed{seed}.mp4")
        safe_export_to_video(out.frames[0], outp, fps=16)
        cprint(f"  seed {seed} -> {outp}", "green")
    cprint(f"\nDone. -> {args.output}", "green")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="outputs/agibot_zeroshot/357_648751/input_latent.safetensors")
    p.add_argument("--lora-checkpoint", default="training/agibot_357_lora/ema_final")
    p.add_argument("--task-text", default="Washing dishes with a dishwasher")
    p.add_argument("--control_type", default="add")
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--seeds", default="42")
    p.add_argument("--guidance_scale", type=float, default=5.0)
    p.add_argument("--no_ema", action="store_true")
    p.add_argument("--output", default="outputs/agibot_lora_357")
    main(p.parse_args())