"""Render a REAL skeleton control video for an AgiBot episode, in the same visual
language as the official hand-pose control (white background, blue left arm, red
right arm, dots + bones), so it can go through the official VAE encode path.

Why this exists
---------------
The first fine-tune attempt synthesized `control_video_latents` by hand
(ActionEncoderV2): gaussian blobs + a sinusoidal carrier, then per-channel
z-scored onto the real control latent's statistics. That matched the first-order
stats (mu, sigma) but carried only **19% of the real control's temporal delta**
(0.0303 vs 0.1587). The Conv3d control head reads motion out of spatio-temporal
gradients, so a near-static latent carries no signal -- which is why the
fine-tuned model never learned to follow the action.

This script instead draws the actual end-effector trajectory into pixels and lets
the official VAE produce the latent, exactly as the real skeleton control was made.

The camera problem
------------------
The AgiBot sample ships no intrinsics/extrinsics -- `action/end/position` is in
robot-base metres and there is nothing to project it with. So we *fit* a camera
from the episode itself:

  1. decode head_color at 160x120, take a temporal median as the static kitchen
     background, and threshold |frame - background| -> the moving arms;
  2. the left image half is arm 0, the right half is arm 1 (verified by sign of
     correlation: base-frame +y runs image-left, +z runs image-up);
  3. least-squares fit a quadratic in (x,y,z) from base metres to (u,v).

A quadratic, not a pinhole DLT: with a single continuous trajectory the DLT is
degenerate (R2 went negative), while the quadratic reaches R2 ~ 0.7 / ~7px median
error on the 160x120 detection grid. It is a *fitted proxy*, not a calibration --
good enough to make the control move with the arms, which is the property the
control head actually consumes.

Usage:
  python scripts/agibot_skeleton_render.py --output outputs/agibot_skeleton/357.mp4
"""
import argparse
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HEIGHT, WIDTH = 480, 832           # official control resolution
NATIVE_W, NATIVE_H = 640, 480      # AgiBot head_color native
FIT_W, FIT_H = 160, 120            # camera-fit grid (cheap full-episode decode)

BG = (247, 249, 246)               # official control videos use this off-white
# blue = left arm, red = right arm, matching the official hand-pose control
ARM_RGB = ((30, 40, 200), (200, 30, 40))


def decode_small(video_path):
    """Whole episode at FIT_W x FIT_H -> [T,H,W,3] uint8. ~2s for 4683 frames."""
    cmd = ["ffmpeg", "-v", "error", "-c:v", "libdav1d", "-i", video_path,
           "-vf", f"scale={FIT_W}:{FIT_H}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed: {p.stderr[:200]}")
    return np.frombuffer(p.stdout, np.uint8).reshape(-1, FIT_H, FIT_W, 3)


def quad_features(X):
    """[N,3] base-frame metres -> [N,10] quadratic design matrix."""
    x, y, z = X.T
    o = np.ones_like(x)
    return np.c_[x, y, z, x * x, y * y, z * z, x * y, x * z, y * z, o]


def fit_camera(video_path, end_pos, percentile=92, min_pixels=25):
    """Fit base-metres -> pixel for each arm off the episode's own motion.

    Returns (coeffs [2,10,2], report dict). Coefficients map into the FIT grid;
    callers scale to their own resolution.
    """
    small = decode_small(video_path).astype(np.float32)
    T = min(len(small), len(end_pos))
    small, end_pos = small[:T], end_pos[:T]

    background = np.median(small[::7], axis=0)
    moving = np.abs(small - background).mean(-1)
    mask = moving > np.percentile(moving, percentile)

    ys, xs = np.mgrid[0:FIT_H, 0:FIT_W]
    coeffs, report = [], []
    for arm, half in enumerate((slice(0, FIT_W // 2), slice(FIT_W // 2, FIT_W))):
        m = mask[:, :, half]
        n = np.maximum(m.sum((1, 2)), 1)
        uv = np.stack([(m * xs[:, half]).sum((1, 2)) / n,
                       (m * ys[:, half]).sum((1, 2)) / n], axis=1)
        enough = m.sum((1, 2)) >= min_pixels
        F = quad_features(end_pos[:, arm])
        A, *_ = np.linalg.lstsq(F[enough], uv[enough], rcond=None)
        pred = F @ A
        r2 = [float(1 - ((pred[enough, j] - uv[enough, j]) ** 2).sum() /
                    ((uv[enough, j] - uv[enough, j].mean()) ** 2).sum()) for j in (0, 1)]
        err = float(np.median(np.sqrt(((pred[enough] - uv[enough]) ** 2).sum(1))))
        coeffs.append(A)
        report.append(dict(arm=arm, r2_u=round(r2[0], 3), r2_v=round(r2[1], 3),
                           median_err_px=round(err, 2), frames_used=int(enough.sum())))
    return np.stack(coeffs), report


def project(coeffs, end_pos, out_w=WIDTH, out_h=HEIGHT):
    """[T,2,3] metres -> [T,2,2] pixels at (out_w,out_h)."""
    sx, sy = out_w / FIT_W, out_h / FIT_H
    uv = np.stack([quad_features(end_pos[:, a]) @ coeffs[a] for a in range(2)], axis=1)
    uv[..., 0] *= sx
    uv[..., 1] *= sy
    return uv


N_FINGERS, N_JOINTS = 5, 4         # 5x4 + wrist = 21 keypoints, as the official control
PALM = 46.0                        # wrist -> knuckle, px
SEG = 21.0                         # per-phalanx length, px
# Sized so a rendered hand's bounding box (~90x80 px) matches the official hand-pose
# control's (83x75 blue / 95x111 red on the same 832x480 canvas).


def quat_to_frame(q):
    """Quaternion (x,y,z,w) -> the two in-plane axes of the hand.

    Only the in-plane projection of the rotation is recoverable without extrinsics.
    That is enough: the control head consumes a consistently-moving structure, not a
    metrically correct one. `roll` additionally foreshortens the palm, so out-of-plane
    rotation still shows up as visible articulation instead of being discarded.
    """
    qx, qy, qz, qw = q
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    roll = np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1, 1))
    return yaw, roll


def hand_keypoints(u, v, q, grip):
    """A 21-point hand at (u,v): wrist + 5 fingers x 4 phalanges.

    Fingers fan out around the palm axis and curl by (1 - gripper opening), so both
    the arm's translation AND its rotation/grasp move every distal point -- that is
    what puts temporal gradient into the control latent.
    """
    yaw, roll = quat_to_frame(q)
    open_amt = float(np.clip(grip, 0.0, 1.0))
    squash = 0.45 + 0.55 * abs(np.cos(roll))          # out-of-plane foreshortening
    pts, bones = [(u, v)], []
    for f in range(N_FINGERS):
        fan = np.deg2rad(-34 + 17 * f) * (0.55 + 0.75 * open_amt)
        a = yaw + fan
        x, y = u + np.cos(a) * PALM, v + np.sin(a) * PALM * squash
        bones.append((0, len(pts)))
        parent = len(pts)
        pts.append((x, y))
        curl = np.deg2rad(52) * (1.0 - open_amt)      # closed gripper = curled fingers
        for j in range(N_JOINTS - 1):
            a += curl
            x, y = x + np.cos(a) * SEG, y + np.sin(a) * SEG * squash
            bones.append((parent, len(pts)))
            parent = len(pts)
            pts.append((x, y))
    return np.array(pts), bones


def draw_frame(uv_t, grip_t, ori_t, canvas_w=WIDTH, canvas_h=HEIGHT):
    """One control frame in the official visual language: white background, a blue
    left hand and a red right hand, each a 21-keypoint skeleton of dots and bones."""
    import cv2
    img = np.full((canvas_h, canvas_w, 3), BG, np.uint8)
    for arm in range(2):
        u, v = uv_t[arm]
        if not np.isfinite(u) or not np.isfinite(v):
            continue
        col = ARM_RGB[arm]
        pts, bones = hand_keypoints(float(u), float(v), ori_t[arm], grip_t[arm])
        for i, j in bones:
            cv2.line(img, tuple(pts[i].astype(int)), tuple(pts[j].astype(int)),
                     col, 3, cv2.LINE_AA)
        for k, (px, py) in enumerate(pts):
            cv2.circle(img, (int(px), int(py)), 7 if k == 0 else 4, col, -1, cv2.LINE_AA)
    return img


def render(uv, grip, ori, start=0, count=None):
    """[T,2,2] px + [T,2] gripper + [T,2,4] quat -> [count,H,W,3] uint8."""
    count = count if count is not None else len(uv)
    return np.stack([draw_frame(uv[start + i], grip[start + i], ori[start + i])
                     for i in range(count)])


def load_end_state(h5_path):
    import h5py
    with h5py.File(h5_path, "r") as f:
        pos = f["action/end/position"][:]          # [T,2,3] base metres
        ori = f["action/end/orientation"][:]       # [T,2,4] quaternion
        grip = f["action/effector/position"][:]    # [T,2] gripper open/closed
    return pos, ori, grip


def main(args):
    import imageio.v2 as iio
    from termcolor import cprint

    obs = os.path.join(args.episode_dir, "observations", str(args.task_id),
                       str(args.episode_id), "videos", "head_color.mp4")
    h5 = os.path.join(args.episode_dir, "proprio_stats", str(args.task_id),
                      str(args.episode_id), "proprio_stats.h5")
    pos, ori, grip = load_end_state(h5)
    cprint(f"[episode] {args.task_id}/{args.episode_id}  end/position {pos.shape}", "cyan")

    coeffs, report = fit_camera(obs, pos)
    for r in report:
        cprint(f"  [camera] arm{r['arm']}: R2_u {r['r2_u']:+.3f} R2_v {r['r2_v']:+.3f} "
               f"median err {r['median_err_px']}px (on {FIT_W}x{FIT_H})", "yellow")
    np.save(args.coeffs_out, coeffs)
    cprint(f"  [camera] coeffs -> {args.coeffs_out}", "green")

    uv = project(coeffs, pos)
    n = args.count if args.count else len(uv) - args.start
    frames = render(uv, grip, ori, args.start, n)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    iio.mimwrite(args.output, frames, fps=30, quality=8)

    ink = (np.abs(frames.astype(int) - np.array(BG)).sum(-1) > 30).mean()
    delta = float(np.abs(np.diff(frames.astype(np.float32), axis=0)).mean())
    cprint(f"[done] {len(frames)} frames -> {args.output}\n"
           f"  ink coverage {ink:.4f} (official control ~0.019)  "
           f"pixel temporal delta {delta:.3f}", "green")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episode-dir", default="/root/autodl-tmp/agibot_sample/extracted/sample_dataset")
    p.add_argument("--task-id", type=int, default=357)
    p.add_argument("--episode-id", type=int, default=648751)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--count", type=int, default=0, help="0 = whole episode")
    p.add_argument("--coeffs-out", default="/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/agibot_357_camera.npy")
    p.add_argument("--output", default="outputs/agibot_skeleton/357_control.mp4")
    main(p.parse_args())
