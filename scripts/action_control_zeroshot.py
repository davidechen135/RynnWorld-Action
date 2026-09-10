"""
Zero-shot UMI-style probe: replace the hand-pose *video* control with a
dual-arm **20-dim action** vector stream, feed it through a training-free
action_encoder into the model's existing control latent slot, and run the
official SFT pipeline WITHOUT any fine-tuning.

Goal = INTERFACE smoke: prove that a low-dim [T,20] action can be mounted onto
the control path (shape/dtype/dist aligned) and the model still produces a
rollout. It does NOT claim the action semantically drives the video -- the
control modules were trained on hand-pose latents, so zero-shot "listening"
is exactly what we measure honestly.

Pipeline:
  pose control mp4 --extract--> action[81,20] --action_encoder(no-train)-->
  control_latent[48,21,30,52]  (per-channel z-score aligned to real control latents)
  --> assemble input_latent.safetensors --> official WanImagePipeline --> rollout

Usage:
  python scripts/action_control_zeroshot.py \
    --image <first_frame.png> --pose_control outputs/repro_sft/<case>/control.mp4 \
    --ref_control_latent outputs/repro_sft/<case>/input_latent.safetensors \
    --text_embedding /mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/text_embeddings/<task>.safetensors \
    --checkpoint <ckpt> --output outputs/action_zeroshot/<case>
"""
import argparse, os, types
import numpy as np
import torch
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch.nn as nn
import torch.nn.functional as F
import cv2
import imageio.v2 as imageio
from safetensors.torch import load_file, save_file
from termcolor import cprint

NUM_FRAMES = 81
HEIGHT, WIDTH = 480, 832
LAT_C, LAT_F, LAT_H, LAT_W = 48, 21, 30, 52  # control latent shape
ACTION_DIM = 20  # dual-arm: 10 per arm


# ---------- 1. extract dual-arm 20-dim action from the hand-pose control video ----------
def extract_dual_arm_action(pose_mp4):
    """From a hand-pose skeleton mp4 (blue=left hand, red=right hand on white bg),
    derive a [T,20] action stream: 10 dims per arm.
    Per arm: [cx, cy, bbox_w, bbox_h, orient_cos, orient_sin, area, spread, dx, dy]
    All normalized to ~[-1,1]. This mimics a dual-arm 20-dim robot action vector,
    following the ORIGINAL control's semantics (hand -> arm) as requested."""
    cap = cv2.VideoCapture(pose_mp4)
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)  # BGR
    cap.release()
    T = len(frames)
    action = np.zeros((T, ACTION_DIM), dtype=np.float32)
    prev = {}
    for t, bgr in enumerate(frames):
        b, g, r = bgr[..., 0].astype(np.int16), bgr[..., 1].astype(np.int16), bgr[..., 2].astype(np.int16)
        # blue-dominant = left hand ; red-dominant = right hand
        masks = {
            0: (b - r > 40) & (b - g > 20),   # left arm -> dims 0..9
            1: (r - b > 40) & (r - g > 20),   # right arm -> dims 10..19
        }
        for arm, m in masks.items():
            ys, xs = np.where(m)
            base = arm * 10
            if len(xs) < 5:
                # keep previous if the hand isn't visible this frame
                if arm in prev:
                    action[t, base:base + 10] = prev[arm]
                continue
            cx, cy = xs.mean() / WIDTH, ys.mean() / HEIGHT
            w = (xs.max() - xs.min()) / WIDTH
            h = (ys.max() - ys.min()) / HEIGHT
            # principal orientation via PCA on the point cloud
            pts = np.stack([xs - xs.mean(), ys - ys.mean()], 1).astype(np.float32)
            if len(pts) > 2:
                cov = pts.T @ pts / len(pts)
                evals, evecs = np.linalg.eigh(cov)
                v = evecs[:, -1]
                orient_cos, orient_sin = float(v[0]), float(v[1])
                spread = float(np.sqrt(max(evals[-1], 1e-6)) / (WIDTH * 0.5))
            else:
                orient_cos, orient_sin, spread = 1.0, 0.0, 0.0
            area = len(xs) / (WIDTH * HEIGHT)
            vec = np.array([cx * 2 - 1, cy * 2 - 1, w * 2 - 1, h * 2 - 1,
                            orient_cos, orient_sin, min(area * 40 - 1, 1.0),
                            min(spread * 4 - 1, 1.0), 0.0, 0.0], dtype=np.float32)
            if arm in prev:
                vec[8] = np.clip((vec[0] - prev[arm][0]) * 8, -1, 1)  # dx
                vec[9] = np.clip((vec[1] - prev[arm][1]) * 8, -1, 1)  # dy
            action[t, base:base + 10] = vec
            prev[arm] = vec
    # resample to exactly NUM_FRAMES
    if T != NUM_FRAMES:
        a = torch.from_numpy(action).T.unsqueeze(0)  # [1,20,T]
        a = F.interpolate(a, size=NUM_FRAMES, mode='linear', align_corners=False)
        action = a.squeeze(0).T.numpy()
    return action  # [81,20]


# ---------- 2. training-free action_encoder: [81,20] -> [48,21,30,52] ----------
class ActionEncoder:
    """Deterministic (NO learned weights) map from a dual-arm 20-dim action stream
    to the control latent tensor the model expects. It:
      (a) temporally pools 81 action frames -> 21 latent frames (4x compression),
      (b) spatially paints each 20-dim vector into a [48,30,52] latent map via a
          FIXED random-but-seeded projection + smooth spatial basis,
      (c) z-score aligns the result to the REAL per-channel control-latent stats
          so it lands in the distribution the frozen control modules understand.
    (a)+(c) are the parts that make zero-shot at all plausible; (b) is a fixed
    injection, not a learned decoder."""

    def __init__(self, ref_latent, seed=42):
        g = torch.Generator().manual_seed(seed)
        # fixed projection: 20 action dims -> 48 latent channels
        self.proj = torch.randn(ACTION_DIM, LAT_C, generator=g) / np.sqrt(ACTION_DIM)
        # smooth low-freq spatial basis so injection isn't white noise
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, LAT_H),
                                torch.linspace(-1, 1, LAT_W), indexing='ij')
        # left arm biased to left half, right arm to right half (spatial prior)
        self.basis_l = torch.exp(-((xx + 0.5) ** 2 + yy ** 2) / 0.6)
        self.basis_r = torch.exp(-((xx - 0.5) ** 2 + yy ** 2) / 0.6)
        # per-channel target stats from real control latents
        rl = ref_latent.float()  # [48,21,30,52]
        self.tgt_mean = rl.mean(dim=(1, 2, 3), keepdim=True)  # [48,1,1,1]
        self.tgt_std = rl.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-4)

    def __call__(self, action):
        a = torch.from_numpy(action).float()  # [81,20]
        # temporal 81 -> 21
        a = F.interpolate(a.T.unsqueeze(0), size=LAT_F, mode='linear',
                          align_corners=False).squeeze(0).T  # [21,20]
        left, right = a[:, :10], a[:, 10:]                    # [21,10] each
        # channel projection: [21,20]@[20,48] -> [21,48]
        chan = a @ self.proj                                   # [21,48]
        # spatial: combine arm channel activations with their spatial basis
        # scalar arm activity per frame drives left/right basis amplitude
        amp_l = left.mean(1)[:, None, None]                    # [21,1,1]
        amp_r = right.mean(1)[:, None, None]
        spatial = amp_l * self.basis_l + amp_r * self.basis_r  # [21,30,52]
        lat = chan[:, :, None, None] * (1.0 + 0.5 * spatial[:, None])  # [21,48,30,52]
        lat = lat.permute(1, 0, 2, 3).contiguous()             # [48,21,30,52]
        # per-channel z-score align to real control-latent distribution
        m = lat.mean(dim=(1, 2, 3), keepdim=True)
        s = lat.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-4)
        lat = (lat - m) / s * self.tgt_std + self.tgt_mean
        return lat.float()


class ActionEncoderV2:
    """v2: the v1 encoder produced a SMOOTH blob (spatial high-freq ~1.8% of real
    pose-control latents), so the Conv3d control head extracted no signal -> frozen
    frames / collapse. v2 instead PAINTS a skeleton-like high-frequency pattern:
    it renders per-frame dual-arm 'joints' as sharp gaussian dots + connecting
    segments on the latent grid (mimicking how a real hand-pose video encodes),
    so the injected control has the spatial structure the head was trained on.
    Still NO learned weights, still zero-shot: v2 makes the picture MOVE, it does
    NOT make the motion match the action's semantics."""

    def __init__(self, ref_latent, seed=42):
        g = torch.Generator().manual_seed(seed)
        self.proj = torch.randn(ACTION_DIM, LAT_C, generator=g) / np.sqrt(ACTION_DIM)
        yy, xx = torch.meshgrid(torch.linspace(0, LAT_H - 1, LAT_H),
                                torch.linspace(0, LAT_W - 1, LAT_W), indexing='ij')
        self.xx, self.yy = xx, yy  # pixel coords on latent grid
        rl = ref_latent.float()
        self.tgt_mean = rl.mean(dim=(1, 2, 3), keepdim=True)
        self.tgt_std = rl.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-4)
        # per-arm nominal keypoint layout (fractional grid positions), left/right
        # 4 joints per arm forming a short chain -> gives edges = high freq
        self.n_joint = 4
        # per-channel random high-freq carrier: sinusoidal grating whose frequency
        # & phase differ per latent channel. Multiplying the skeleton dots by this
        # gives each of the 48 channels its OWN sharp spatial texture (real pose
        # latents are NOT a single mask broadcast to all channels -> that was v1/v2a's
        # flaw: identical map on every channel -> low effective spatial HF).
        fx = torch.rand(LAT_C, generator=g) * 1.8 + 1.0   # cycles across width
        fy = torch.rand(LAT_C, generator=g) * 1.8 + 1.0
        ph = torch.rand(LAT_C, generator=g) * 6.283
        self.fx, self.fy, self.ph = fx, fy, ph
        yy2, xx2 = torch.meshgrid(torch.linspace(0, 6.283, LAT_H),
                                  torch.linspace(0, 6.283, LAT_W), indexing='ij')
        self.carrier = torch.sin(fx[:, None, None] * xx2[None] +
                                 fy[:, None, None] * yy2[None] + ph[:, None, None])  # [48,30,52]

    def _dots(self, cx, cy, sigma=0.8):
        """sharp gaussian dot at (cx,cy) on the latent grid -> high spatial freq."""
        return torch.exp(-(((self.xx - cx) ** 2 + (self.yy - cy) ** 2) / (2 * sigma ** 2)))

    def _arm_map(self, vec, xbias):
        """render one arm's 10-dim sub-vector into a [30,52] skeleton-ish map.
        Uses (cx,cy)=vec[0:2] as wrist anchor, orientation vec[4:6], spread vec[7]
        to lay out a 4-joint chain, painted as sharp dots (high freq)."""
        cx = (vec[0] * 0.5 + 0.5) * (LAT_W - 1) * 0.5 + xbias * (LAT_W - 1)
        cy = (vec[1] * 0.5 + 0.5) * (LAT_H - 1)
        ang = torch.atan2(vec[5], vec[4])
        length = (vec[7] * 0.5 + 0.5) * 8 + 3  # 3..11 px chain length
        m = torch.zeros(LAT_H, LAT_W)
        px, py = cx, cy
        for j in range(self.n_joint):
            m = torch.maximum(m, self._dots(px, py))
            px = px + torch.cos(ang) * (length / self.n_joint)
            py = py + torch.sin(ang) * (length / self.n_joint)
        return m  # [30,52], sharp

    def __call__(self, action):
        a = torch.from_numpy(action).float()  # [81,20]
        a = F.interpolate(a.T.unsqueeze(0), size=LAT_F, mode='linear',
                          align_corners=False).squeeze(0).T  # [21,20]
        chan = a @ self.proj  # [21,48] channel signature per frame
        maps = torch.zeros(LAT_F, LAT_H, LAT_W)
        for t in range(LAT_F):
            left = self._arm_map(a[t, :10], xbias=0.0)    # left half
            right = self._arm_map(a[t, 10:], xbias=0.5)   # right half
            maps[t] = torch.maximum(left, right)
        # Real pose latents carry VAE high-freq texture over the WHOLE frame (not just
        # at the joints) and that texture SHIFTS frame-to-frame as the hand moves.
        # Reproduce both: a full-frame per-channel carrier (spatial HF everywhere)
        # whose phase is shifted per-frame by the action's centroid motion (temporal
        # HF). The sparse skeleton `maps` then adds a localized bright footprint on top.
        cx_l = (a[:, 0] * 0.5 + 0.5)                       # [21] left-arm x in [0,1]
        cx_r = (a[:, 10] * 0.5 + 0.5)
        cy_l = (a[:, 1] * 0.5 + 0.5)
        shift_x = (cx_l + cx_r)[:, None, None] * 6.283     # [21,1,1] frame phase-x
        shift_y = cy_l[:, None, None] * 6.283
        # carrier: [48,30,52] gratings; shifted per frame -> [21,48,30,52]
        base = self.carrier[None]                          # [1,48,30,52]
        # approximate phase shift by rolling the sinusoid: recompute with added phase
        yy2, xx2 = torch.meshgrid(torch.linspace(0, 6.283, LAT_H),
                                  torch.linspace(0, 6.283, LAT_W), indexing='ij')
        # per-frame, per-channel shifted carrier
        car = torch.sin(self.fx[None, :, None, None] * xx2[None, None] + shift_x[:, None]
                        + self.fy[None, :, None, None] * yy2[None, None] + shift_y[:, None]
                        + self.ph[None, :, None, None])    # [21,48,30,52]
        foot = (0.4 + maps[:, None])                       # skeleton footprint gain
        lat = car * foot * chan[:, :, None, None]          # [21,48,30,52]
        lat = lat.permute(1, 0, 2, 3).contiguous()         # [48,21,30,52]
        m = lat.mean(dim=(1, 2, 3), keepdim=True)
        s = lat.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-4)
        lat = (lat - m) / s * self.tgt_std + self.tgt_mean
        return lat.float()


def read_image(path):
    img = imageio.imread(path)
    if img.shape[-1] == 4:
        img = img[..., :3]
    if img.shape[0] != HEIGHT or img.shape[1] != WIDTH:
        img = cv2.resize(img, (WIDTH, HEIGHT))
    return img


def main(args):
    from diffusers import AutoencoderKLWan
    from core.inference.rynnworld_teleop import WanImagePipeline, wan_forward, safe_export_to_video
    import inference_user as IU  # reuse official encode/attach helpers

    device = torch.device("cuda")
    dtype = torch.bfloat16
    os.makedirs(args.output, exist_ok=True)

    cprint("\n[1/5] Extracting dual-arm 20-dim action from pose control", "cyan")
    action = extract_dual_arm_action(args.pose_control)
    cprint(f"  action stream: {action.shape}  "
           f"range[{action.min():.2f},{action.max():.2f}] "
           f"mean={action.mean():.3f}", "green")
    np.save(os.path.join(args.output, "action_20d.npy"), action)

    cprint("\n[2/5] Encoding action -> control latent (training-free, dist-aligned)", "cyan")
    ref = load_file(args.ref_control_latent)["control_video_latents"]
    Enc = ActionEncoderV2 if args.encoder == "v2" else ActionEncoder
    enc = Enc(ref, seed=42)
    control_video_latents = enc(action)  # [48,21,30,52]
    cprint(f"  control_video_latents: {tuple(control_video_latents.shape)} "
           f"mean={control_video_latents.mean():.4f} std={control_video_latents.std():.4f} "
           f"(target mean={ref.float().mean():.4f} std={ref.float().std():.4f})", "green")

    cprint("\n[3/5] Encoding first-frame image via VAE", "cyan")
    vae = AutoencoderKLWan.from_pretrained(IU.MODEL_PATH, subfolder="vae").to(device=device, dtype=dtype)
    image_np = read_image(args.image)
    img_latent = IU.encode_image_to_latent(vae, image_np, device, dtype)
    video_latents = control_video_latents.clone()  # placeholder for GT decode slot
    del vae; torch.cuda.empty_cache()

    prompt_embeds = None
    if args.text_embedding and os.path.exists(args.text_embedding):
        prompt_embeds = load_file(args.text_embedding)["text_embedding"].unsqueeze(0)
        cprint(f"  text_embedding: {args.text_embedding}", "green")

    input_latent_path = os.path.join(args.output, "input_latent.safetensors")
    save_file({"video_latents": video_latents,
               "control_video_latents": control_video_latents,
               "img_latent": img_latent}, input_latent_path)

    cprint("\n[4/5] Loading SFT checkpoint (frozen, zero-shot)", "cyan")
    pipe = IU.load_pipeline(args.checkpoint, args.control_type, dtype,
                            mode="sft", use_ema=not args.no_ema)

    cprint("\n[5/5] Generating rollout", "cyan")
    for seed in [int(s) for s in args.seeds.split(",")]:
        gen = torch.Generator(device=device).manual_seed(seed)
        out, gt, ctrl, ctrl_raw = pipe(
            prompt='' if prompt_embeds is None else None, negative_prompt='',
            guidance_scale=args.guidance_scale, video_latent_path=input_latent_path,
            control_type=args.control_type, prompt_embeds=prompt_embeds, generator=gen)
        safe_export_to_video(out.frames[0], os.path.join(args.output, f"action_rollout_seed{seed}.mp4"), fps=16)
        cprint(f"  seed {seed} -> action_rollout_seed{seed}.mp4", "green")
    cprint(f"\nDone. Outputs in {args.output}", "green")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--pose_control", required=True, help="existing hand-pose control.mp4 to derive action from")
    p.add_argument("--ref_control_latent", required=True, help="a real input_latent.safetensors for per-channel dist target")
    p.add_argument("--text_embedding", default=None)
    p.add_argument("--output", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--control_type", default="add")
    p.add_argument("--encoder", default="v2", choices=["v1", "v2"])
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--no_ema", action="store_true")
    p.add_argument("--seeds", default="42")
    main(p.parse_args())
