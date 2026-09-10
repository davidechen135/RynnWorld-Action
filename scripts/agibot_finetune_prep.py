"""AgiBot real-robot ego LoRA fine-tune data prep (path A: single long episode).

Takes ONE real AgiBot head_color episode (already on disk: task 357 "washing
dishes", episode 648751, 4683 frames @ 30fps, AV1) + its proprio 20-dim action,
and slices it into N non-overlapping 81-frame clips. Each clip is encoded into
the official training three-tuple safetensors that EgoVerseDataset22 reads:
    video_latents        [48,21,30,52]  real head_color -> VAE  (denoising TARGET)
    control_video_latents[48,21,30,52]  20-dim action  -> ActionEncoderV2 (dist-aligned)
    img_latent           [48, 1,30,52]  clip first frame -> VAE
plus a shared task text_embedding [226,4096] (UMT5).

Writes:
    /mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/agibot_357/clip_XXXXXX.safetensors
    /mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/text_embeddings/agibot_task357.safetensors
    /mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/agibot_357.json   (index consumed by finetune.py --validation_dir/--cache_dir)

This is the fine-tune counterpart of the zero-shot probes (docs/agibot_crossdomain.md,
docs/action_zeroshot.md): it CROSSES the "no fine-tune" boundary by producing real
target latents so LoRA can actually adapt the frozen world model to the robot domain.

Usage:
  python scripts/agibot_finetune_prep.py \
    --episode-dir /root/autodl-tmp/agibot_sample/extracted/sample_dataset \
    --task-id 357 --episode-id 648751 \
    --task-text "Washing dishes with a dishwasher" \
    --stride 81 --max-clips 40 --output /mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/agibot_357

Notes:
  * head_color.mp4 is AV1 -> decoded via ffmpeg libdav1d rawvideo pipe (decord/OpenCV
    cannot read AV1 here). Native 640x480 -> resized to official 832x480.
  * control latent reuses ActionEncoderV2 (training-free, dist-aligned) so the LoRA's
    control_patch_embedding sees the same distribution the frozen head expects; LoRA
    then learns to bind that control to the real robot dynamics in the target latents.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # scripts/ for AZ reuse
import h5py
from safetensors.torch import save_file
from termcolor import cprint

NUM_FRAMES = 81
HEIGHT, WIDTH = 480, 832           # official training resolution
NATIVE_W, NATIVE_H = 640, 480      # AgiBot head_color native


def decode_av1_segment(video_path, start, count):
    """Decode `count` frames starting at frame `start` from an AV1 mp4 via
    ffmpeg libdav1d -> [count,NATIVE_H,NATIVE_W,3] uint8 (RGB). decord/OpenCV
    cannot read these AV1 streams, so we pipe rawvideo."""
    end = start + count - 1
    cmd = ["ffmpeg", "-v", "error", "-c:v", "libdav1d", "-i", video_path,
           "-vf", f"select='between(n,{start},{end})'", "-vsync", "0",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed @ {start}: {p.stderr[:200]}")
    frame_bytes = NATIVE_W * NATIVE_H * 3
    n = len(p.stdout) // frame_bytes
    if n < count:
        raise RuntimeError(f"decoded {n} < {count} frames @ {start}")
    arr = np.frombuffer(p.stdout[:count * frame_bytes], np.uint8)
    return arr.reshape(count, NATIVE_H, NATIVE_W, 3).copy()


def resize_clip(frames):
    """[F,480,640,3] uint8 -> [F,480,832,3] uint8 (official W)."""
    import cv2
    out = np.empty((frames.shape[0], HEIGHT, WIDTH, 3), np.uint8)
    for i, f in enumerate(frames):
        out[i] = cv2.resize(f, (WIDTH, HEIGHT))
    return out


def load_action_20d(h5_path):
    """Real dual-arm 20-dim action [T,20] = joint14 + gripper2 + head2 + waist2,
    per-dim min-max normalized to [-1,1] (same assembly as agibot_zeroshot)."""
    with h5py.File(h5_path, "r") as f:
        joint = f["action/joint/position"][:]      # [T,14]
        grip = f["action/effector/position"][:]    # [T,2]
        head = f["action/head/position"][:]        # [T,2]
        waist = f["action/waist/position"][:]      # [T,2]
    a = np.concatenate([joint, grip, head, waist], axis=1).astype(np.float32)
    lo, hi = a.min(0, keepdims=True), a.max(0, keepdims=True)
    return 2 * (a - lo) / np.clip(hi - lo, 1e-6, None) - 1  # [T,20]


def resample_action(action_seg, n=NUM_FRAMES):
    """Resample an action segment to exactly n frames (linear) -> [n,20]."""
    import torch.nn.functional as F
    a = torch.from_numpy(action_seg).float().T.unsqueeze(0)   # [1,20,T]
    a = F.interpolate(a, size=n, mode="linear", align_corners=False)
    return a.squeeze(0).T.numpy()                             # [n,20]


def main(args):
    from diffusers import AutoencoderKLWan
    from safetensors.torch import load_file
    import inference_user as IU
    import action_control_zeroshot as AZ
    import agibot_skeleton_render as SK

    device = torch.device("cuda")
    dtype = torch.bfloat16
    obs = os.path.join(args.episode_dir, "observations", str(args.task_id),
                       str(args.episode_id), "videos", "head_color.mp4")
    h5 = os.path.join(args.episode_dir, "proprio_stats", str(args.task_id),
                      str(args.episode_id), "proprio_stats.h5")
    for p in (obs, h5):
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    # total frames
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                            "-count_frames", "-show_entries", "stream=nb_read_frames",
                            "-of", "default=nw=1:nk=1", obs], capture_output=True, text=True)
    total = int(probe.stdout.strip())
    action = load_action_20d(h5)                              # [T,20]
    cprint(f"[episode] {obs}\n  frames={total} action={action.shape}", "cyan")

    # --skip-clips lets a second call carve a DISJOINT held-out split from the tail
    # of the same episode (train = clips [0, max_clips), val = clips [skip, skip+n)),
    # so evaluation first frames are never ones the LoRA trained on.
    n_total = (total - NUM_FRAMES) // args.stride + 1
    n_clips = min(args.max_clips, max(0, n_total - args.skip_clips))
    cprint(f"[plan] stride={args.stride} skip={args.skip_clips} -> clips "
           f"{args.skip_clips}..{args.skip_clips + n_clips - 1} of {n_total} "
           f"({NUM_FRAMES} frames each)", "cyan")

    os.makedirs(args.output, exist_ok=True)
    os.makedirs("/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/text_embeddings", exist_ok=True)

    # models
    vae = AutoencoderKLWan.from_pretrained(IU.MODEL_PATH, subfolder="vae").to(device, dtype)

    if args.control_mode == "skeleton":
        # Render the REAL end-effector trajectory into a skeleton video and let the
        # official VAE make the latent -- the same route the real hand-pose control
        # took. The hand-built encoder below matched mu/sigma but carried only ~19%
        # of the real control's temporal delta, which is why v1 learned no motion.
        end_pos, end_ori, end_grip = SK.load_end_state(h5)
        cam, cam_report = SK.fit_camera(obs, end_pos)
        for r in cam_report:
            cprint(f"[camera] arm{r['arm']}: R2_u {r['r2_u']:+.3f} R2_v {r['r2_v']:+.3f} "
                   f"median err {r['median_err_px']}px", "yellow")
        np.save(args.camera_out, cam)
        skel_uv = SK.project(cam, end_pos)
    else:
        ref = load_file(args.ref_control_latent)["control_video_latents"]
        enc = AZ.ActionEncoderV2(ref)

    # shared task text embedding [226,4096]
    txt_path = os.path.join("/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/text_embeddings", f"agibot_task{args.task_id}.safetensors")
    if not os.path.exists(txt_path) or args.force:
        emb = IU.encode_text(args.task_text, device, dtype)   # [226,4096]
        save_file({"text_embedding": emb.contiguous()}, txt_path)
        cprint(f"[text] '{args.task_text}' -> {tuple(emb.shape)} -> {txt_path}", "green")

    index = []
    for i in range(n_clips):
        start = (i + args.skip_clips) * args.stride
        frames = decode_av1_segment(obs, start, NUM_FRAMES)   # [81,480,640,3]
        frames = resize_clip(frames)                          # [81,480,832,3]
        video_latents = IU.encode_video_to_latent(vae, frames, device, dtype)   # [48,21,30,52]
        img_latent = IU.encode_image_to_latent(vae, frames[0], device, dtype)   # [48,1,30,52]

        if args.control_mode == "skeleton":
            skel = SK.render(skel_uv, end_grip, end_ori, start, NUM_FRAMES)
            control_video_latents = IU.encode_video_to_latent(vae, skel, device, dtype)
        else:
            act_seg = resample_action(action[start:start + NUM_FRAMES])  # [81,20]
            control_video_latents = enc(act_seg)                  # [48,21,30,52]

        out = os.path.join(args.output, f"clip_{start:06d}.safetensors")
        save_file({"video_latents": video_latents.contiguous(),
                   "control_video_latents": control_video_latents.contiguous(),
                   "img_latent": img_latent.contiguous()}, out)
        index.append({"video_latent_path": out, "text_embedding_path": txt_path})
        cprint(f"  [{i+1}/{n_clips}] f{start}-{start+NUM_FRAMES} -> {out} "
               f"vid μ={video_latents.mean():.3f}σ={video_latents.std():.3f} "
               f"ctl μ={control_video_latents.mean():.3f}σ={control_video_latents.std():.3f} "
               f"Δt={control_video_latents.float().diff(dim=1).abs().mean():.4f}", "green")

    json_path = f"{args.output.rstrip('/')}.json"
    with open(json_path, "w") as f:
        json.dump(index, f, indent=2)
    cprint(f"\n[done] {len(index)} clips -> {json_path}", "green")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episode-dir", default="/root/autodl-tmp/agibot_sample/extracted/sample_dataset")
    p.add_argument("--task-id", type=int, default=357)
    p.add_argument("--episode-id", type=int, default=648751)
    p.add_argument("--task-text", default="Washing dishes with a dishwasher")
    p.add_argument("--stride", type=int, default=81)
    p.add_argument("--max-clips", type=int, default=40)
    p.add_argument("--skip-clips", type=int, default=0,
                   help="skip the first N clips; use to carve a held-out tail split "
                        "disjoint from the training clips (see --max-clips)")
    p.add_argument("--control-mode", choices=["skeleton", "action_encoder"],
                   default="skeleton",
                   help="skeleton: render real end-effector poses -> official VAE encode "
                        "(default, ~2x the temporal delta). action_encoder: the v1 "
                        "hand-built dist-aligned latent, kept for comparison.")
    p.add_argument("--camera-out", default="/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/agibot_357_camera.npy")
    p.add_argument("--ref-control-latent",
                   default="outputs/repro_sft/basic_pick_place_000/input_latent.safetensors")
    p.add_argument("--output", default="/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/agibot_357")
    p.add_argument("--force", action="store_true")
    main(p.parse_args())

