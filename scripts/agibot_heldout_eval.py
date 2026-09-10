"""Held-out evaluation: zero-shot vs AgiBot-357 LoRA on clips the LoRA never saw.

The first comparison (outputs/agibot_lora_357/) evaluated on episode frame 0, which
is byte-identical to training clip_000000 -- an in-distribution reconstruction, not
a real test. This script instead drives both models from a clip in
/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/agibot_357_heldout/ (frames 3240-4616, disjoint from the 0-3239 training span,
verified zero frame overlap), so the first frame / action / control latent are all
genuinely unseen.

Everything except the weights is held fixed between the two runs: same held-out clip,
same control latent, same text embedding, same seed, same guidance. Real video latents
from the clip are carried through as `video_latents` so the pipeline can also decode
the ground-truth strip for reference (control_type='add' does not feed them into the
denoising loop -- see core/inference/rynnworld_teleop.py:510-517).

Usage:
  python scripts/agibot_heldout_eval.py --clip /mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/agibot_357_heldout/clip_003969.safetensors
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from safetensors.torch import load_file
from termcolor import cprint

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def collapse_metrics(frames):
    """frames: list of HxWx3 uint8. Returns temporal-collapse diagnostics.
    detail_ratio ~1.0 = no spatial-detail collapse; interframe delta = motion energy."""
    a = np.stack([np.asarray(f, np.float32) for f in frames])       # [T,H,W,3]
    d = np.abs(np.diff(a, axis=0)).mean(axis=(1, 2, 3))             # [T-1]
    std = a.std(axis=(1, 2, 3))                                     # [T]
    t = len(std) // 3
    return {
        "interframe_delta_mean": float(d.mean()),
        "interframe_delta_early": float(d[:t].mean()),
        "interframe_delta_late": float(d[-t:].mean()),
        "detail_std_first_third": float(std[:t].mean()),
        "detail_std_last_third": float(std[-t:].mean()),
        "detail_ratio_last_over_first": float(std[-t:].mean() / max(std[:t].mean(), 1e-6)),
    }


def main(args):
    import inference_user as IU
    from core.inference.rynnworld_teleop import safe_export_to_video

    device, dtype = torch.device("cuda"), torch.bfloat16
    os.makedirs(args.output, exist_ok=True)

    d = load_file(args.clip)
    cprint(f"[held-out clip] {args.clip}\n  " +
           ", ".join(f"{k}{tuple(v.shape)}" for k, v in d.items()), "green")

    # sanity: this clip must NOT be in the training index.
    # Compare NORMALIZED FULL paths, not basenames: the 362 layout preps every
    # episode independently from frame 0, so held-out episode 650872's
    # clip_000000.safetensors shares a BASENAME with training episode 650989's
    # clip_000000.safetensors — a basename check would falsely abort. The full
    # path is unique (it carries the per-episode subdir) and stays correct for
    # the 357 within-episode split too (that split used unique clip names anyway).
    train = {os.path.normpath(e["video_latent_path"])
             for e in json.load(open(args.train_json))}
    assert os.path.normpath(args.clip) not in train, "clip is in the training set!"
    cprint(f"  [check] not in {args.train_json} ({len(train)} training clips) -> held out", "green")

    inp = os.path.join(args.output, "heldout_input.safetensors")
    from safetensors.torch import save_file
    save_file({k: d[k].contiguous() for k in
               ("video_latents", "control_video_latents", "img_latent")}, inp)

    prompt_embeds = None
    if args.task_text:
        emb = IU.encode_text(args.task_text, device, dtype)
        if emb is not None:
            prompt_embeds = emb.unsqueeze(0)

    results = {"clip": args.clip, "seed": args.seed,
               "guidance_scale": args.guidance_scale, "runs": {}}

    for tag, ckpt, mode in (("zeroshot", args.sft_checkpoint, "sft"),
                            ("finetuned", args.lora_checkpoint, "lora")):
        cprint(f"\n[{tag}] loading {ckpt} (mode={mode})", "cyan")
        kw = dict(mode=mode)
        if mode == "lora":
            kw.update(lora_rank=args.rank, lora_alpha=args.lora_alpha)
        pipe = IU.load_pipeline(ckpt, args.control_type, dtype, use_ema=True, **kw)

        gen = torch.Generator(device=device).manual_seed(args.seed)
        out, gt_out, *_ = pipe(prompt='' if prompt_embeds is None else None, negative_prompt='',
                               guidance_scale=args.guidance_scale, video_latent_path=inp,
                               control_type=args.control_type, prompt_embeds=prompt_embeds,
                               generator=gen)
        frames = out.frames[0]
        mp4 = os.path.join(args.output, f"heldout_{tag}_seed{args.seed}.mp4")
        safe_export_to_video(frames, mp4, fps=16)
        results["runs"][tag] = {"video": mp4, **collapse_metrics(frames)}
        cprint(f"  -> {mp4}", "green")

        # GT anchor: the pipeline already decodes the clip's real video_latents and
        # runs them through the SAME video_processor.postprocess_video as the
        # generated frames. Export it once (identical both runs) so motion energy can
        # be read as a fraction of GT. Decoding GT by hand instead skips postprocess
        # and shifts the color/scale, which makes the %GT ratio meaningless.
        if "gt" not in results["runs"]:
            gt_frames = gt_out.frames[0]
            gt_mp4 = os.path.join(args.output, "heldout_gt.mp4")
            safe_export_to_video(gt_frames, gt_mp4, fps=16)
            results["runs"]["gt"] = {"video": gt_mp4, **collapse_metrics(gt_frames)}
            cprint(f"  -> {gt_mp4} (GT anchor)", "green")
        for k, v in results["runs"][tag].items():
            if k != "video":
                cprint(f"     {k:32s} {v:.4f}", "yellow")

        del pipe
        torch.cuda.empty_cache()

    jp = os.path.join(args.output, "heldout_metrics.json")
    with open(jp, "w") as f:
        json.dump(results, f, indent=2)
    cprint(f"\n[done] {jp}", "green")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--clip", default="/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/agibot_357_heldout/clip_003969.safetensors")
    p.add_argument("--train-json", default="/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/agibot_357.json")
    p.add_argument("--sft-checkpoint", default="pretrained/RynnWorld-Teleop")
    p.add_argument("--lora-checkpoint", default="training/agibot_357_lora/ema_final")
    p.add_argument("--task-text", default="Washing dishes with a dishwasher")
    p.add_argument("--control_type", default="add")
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--output", default="outputs/agibot_heldout")
    main(p.parse_args())
