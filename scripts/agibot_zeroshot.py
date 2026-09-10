"""
Cross-domain zero-shot test with REAL AgiBot World (智元) data.

Uses a real egocentric head-camera first frame + the real dual-arm 20-dim action
(joint14 + gripper2 + head2 + waist2) from proprio_stats.h5, feeds the action
through the SAME training-free action_encoder, and runs the frozen official SFT
model zero-shot. Measures cross-domain generalisation: the model was trained on
tabletop hand-pose ego data; AgiBot is a real dual-arm robot doing housework
(washing / folding) -- a genuinely different domain.

NO fine-tuning. Expectation: action stays imprecise (encoder untrained), but we
observe how the model renders an out-of-distribution real-robot ego first frame.

Usage:
  python scripts/agibot_zeroshot.py \
    --episode_dir /root/autodl-tmp/agibot_sample/extracted/sample_dataset \
    --task 362 --episode 649657 --task_text "Folding shorts" \
    --ref_control_latent outputs/repro_sft/basic_pick_place_000/input_latent.safetensors \
    --checkpoint pretrained/RynnWorld-Teleop \
    --output outputs/agibot_zeroshot/362_649657
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import cv2
import h5py
import imageio.v2 as imageio
from safetensors.torch import load_file, save_file
from termcolor import cprint

sys.path.insert(0, "scripts")
import action_control_zeroshot as AZ  # reuse ActionEncoder + constants

NUM_FRAMES, HEIGHT, WIDTH = AZ.NUM_FRAMES, AZ.HEIGHT, AZ.WIDTH


def load_agibot_action(h5_path):
    """Assemble the real dual-arm 20-dim action stream [T,20] from AgiBot proprio:
    joint(14) + gripper(2) + head(2) + waist(2). Per-dim normalized to ~[-1,1]."""
    with h5py.File(h5_path, "r") as f:
        joint = f["action/joint/position"][:]      # [T,14] rad
        grip = f["action/effector/position"][:]    # [T,2]  0..1
        head = f["action/head/position"][:]        # [T,2]  rad
        waist = f["action/waist/position"][:]      # [T,2]
    T = joint.shape[0]
    a = np.concatenate([joint, grip, head, waist], axis=1).astype(np.float32)  # [T,20]
    # per-dim min-max -> [-1,1] (robust to differing physical units/ranges)
    lo, hi = a.min(0, keepdims=True), a.max(0, keepdims=True)
    a = 2 * (a - lo) / np.clip(hi - lo, 1e-6, None) - 1
    return a  # [T,20]


def read_first_frame(video_path):
    """AgiBot head_color.mp4 is AV1-encoded; OpenCV can't decode it, so use
    ffmpeg (libdav1d) to extract frame 0 to a temp PNG, then read it."""
    import subprocess, tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
    r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-c:v", "libdav1d",
                        "-i", video_path, "-frames:v", "1", tmp],
                       capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(tmp):
        raise RuntimeError(f"ffmpeg decode failed for {video_path}: {r.stderr[:200]}")
    rgb = imageio.imread(tmp)
    os.unlink(tmp)
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    if rgb.shape[0] != HEIGHT or rgb.shape[1] != WIDTH:
        rgb = cv2.resize(rgb, (WIDTH, HEIGHT))
    return rgb


def main(args):
    from diffusers import AutoencoderKLWan
    import inference_user as IU

    device, dtype = torch.device("cuda"), torch.bfloat16
    os.makedirs(args.output, exist_ok=True)
    base = os.path.join(args.episode_dir)
    vid = os.path.join(base, "observations", args.task, args.episode, "videos", "head_color.mp4")
    h5 = os.path.join(base, "proprio_stats", args.task, args.episode, "proprio_stats.h5")

    cprint(f"\n[1/5] Real AgiBot action ({args.task}/{args.episode}) + ego first frame", "cyan")
    action = load_agibot_action(h5)
    cprint(f"  raw action: {action.shape} (joint14+grip2+head2+waist2=20)  "
           f"range[{action.min():.2f},{action.max():.2f}]", "green")
    # resample T -> 81
    a = torch.from_numpy(action).T.unsqueeze(0)
    action81 = torch.nn.functional.interpolate(a, size=NUM_FRAMES, mode="linear",
                                               align_corners=False).squeeze(0).T.numpy()
    np.save(os.path.join(args.output, "agibot_action_20d.npy"), action81)
    frame0 = read_first_frame(vid)
    imageio.imwrite(os.path.join(args.output, "agibot_first_frame.png"), frame0)
    cprint(f"  ego first frame: {frame0.shape}  task='{args.task_text}'", "green")

    cprint("\n[2/5] Action -> control latent (training-free, dist-aligned)", "cyan")
    ref = load_file(args.ref_control_latent)["control_video_latents"]
    Enc = AZ.ActionEncoderV2 if args.encoder == "v2" else AZ.ActionEncoder
    enc = Enc(ref, seed=42)
    control_video_latents = enc(action81)
    cprint(f"  control_video_latents: {tuple(control_video_latents.shape)} "
           f"mean={control_video_latents.mean():.4f} std={control_video_latents.std():.4f}", "green")

    cprint("\n[3/5] VAE-encode ego first frame", "cyan")
    vae = AutoencoderKLWan.from_pretrained(IU.MODEL_PATH, subfolder="vae").to(device=device, dtype=dtype)
    img_latent = IU.encode_image_to_latent(vae, frame0, device, dtype)
    del vae; torch.cuda.empty_cache()

    # encode task text with T5 (real AgiBot task description -> cross-domain prompt)
    prompt_embeds = None
    if args.task_text:
        emb = IU.encode_text(args.task_text, device, dtype)
        if emb is not None:
            prompt_embeds = emb.unsqueeze(0)
            cprint(f"  prompt encoded: '{args.task_text}'", "green")

    input_latent_path = os.path.join(args.output, "input_latent.safetensors")
    save_file({"video_latents": control_video_latents.clone(),
               "control_video_latents": control_video_latents,
               "img_latent": img_latent}, input_latent_path)

    cprint("\n[4/5] Load frozen SFT checkpoint (zero-shot)", "cyan")
    pipe = IU.load_pipeline(args.checkpoint, args.control_type, dtype, mode="sft", use_ema=not args.no_ema)

    cprint("\n[5/5] Generate cross-domain rollout", "cyan")
    for seed in [int(s) for s in args.seeds.split(",")]:
        gen = torch.Generator(device=device).manual_seed(seed)
        out, *_ = pipe(prompt='' if prompt_embeds is None else None, negative_prompt='',
                       guidance_scale=args.guidance_scale, video_latent_path=input_latent_path,
                       control_type=args.control_type, prompt_embeds=prompt_embeds, generator=gen)
        from core.inference.rynnworld_teleop import safe_export_to_video
        safe_export_to_video(out.frames[0], os.path.join(args.output, f"agibot_rollout_seed{seed}.mp4"), fps=16)
        cprint(f"  seed {seed} -> agibot_rollout_seed{seed}.mp4", "green")
    cprint(f"\nDone. Outputs in {args.output}", "green")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episode_dir", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--episode", required=True)
    p.add_argument("--task_text", default="")
    p.add_argument("--ref_control_latent", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--control_type", default="add")
    p.add_argument("--encoder", default="v2", choices=["v1", "v2"])
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--no_ema", action="store_true")
    p.add_argument("--seeds", default="42")
    main(p.parse_args())
