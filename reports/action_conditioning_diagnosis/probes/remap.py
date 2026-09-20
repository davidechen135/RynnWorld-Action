"""Controlled test: does replacing the low-rank input_residual_projection with a
full-rank map restore action discriminability in the INJECTED signal?

Both the trained map and the replacement are evaluated on the identical
`output_norm(_encode(...))` tensor, so the only variable is the projection.
The replacement is scaled so its output RMS matches the trained map's output RMS
on the same input -- otherwise a bigger output would trivially look discriminable.
"""
import sys, glob, copy
import torch, torch.nn as nn, torch.nn.functional as F
from safetensors.torch import load_file
sys.path.insert(0, "/mnt/workspace/umi-world-model-lab/third_party/RynnWorld-Teleop-umi")
from core.control import NativeTrajectoryConditionerV11

REPO = "/mnt/workspace/umi-world-model-lab/third_party/RynnWorld-Teleop-umi"
DS   = "/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop"
D    = f"{DS}/agibot_action_v10_state_256_v1"

enc = NativeTrajectoryConditionerV11(input_dim=148).eval()
enc.load_state_dict(torch.load(
    f"{REPO}/training/native_action_v11_temporal_alignment_300/checkpoint-300/native_trajectory_encoder.bin",
    map_location="cpu", weights_only=False), strict=False)

wins = []
for p in sorted(glob.glob(f"{D}/task*.safetensors")):
    t = load_file(p)["robot_trajectory"].unsqueeze(0).float()
    if t.shape[-1] == 148 and t.abs().sum() > 0:
        wins.append((p.split("/")[-1], t))
    if len(wins) == 6: break

def tv_of(r5d):
    """r5d [N,3072,9,1,1] -> (time-varying rms, total rms). Time is dim 2."""
    x = r5d.float().squeeze(-1).squeeze(-1)              # [N,3072,9]
    v = x - x.mean(dim=2, keepdim=True)
    return v.square().mean().sqrt().item(), x.square().mean().sqrt().item()

# ---- build the full-rank replacement, output-RMS-matched on real data ----
W = enc.input_residual_projection.weight.detach()        # [3072, 768]
with torch.no_grad():
    feats = torch.cat([enc.output_norm(enc._encode(t, 9)) for _, t in wins], dim=0)
    tgt   = F.linear(feats, W).square().mean().sqrt().item()
    torch.manual_seed(0)
    Wr = torch.randn_like(W)
    Wr = Wr * (tgt / F.linear(feats, Wr).square().mean().sqrt().item())

enc_rand = copy.deepcopy(enc)
enc_rand.input_residual_projection = nn.Linear(768, 3072, bias=False).eval()
with torch.no_grad(): enc_rand.input_residual_projection.weight.copy_(Wr)

s = torch.linalg.svdvals(W.float()); e = s.square(); e = e/e.sum()
sr = torch.linalg.svdvals(Wr.float()); er = sr.square(); er = er/er.sum()
print("projection                rank@99%   top1     output_rms")
print(f"  trained                 {int((e.cumsum(0)<0.99).sum())+1:>6}   {e[0]:.4f}   {tgt:.6f}")
print(f"  random full-rank        {int((er.cumsum(0)<0.99).sum())+1:>6}   {er[0]:.4f}   "
      f"{F.linear(feats, Wr).square().mean().sqrt().item():.6f}")

def counterfactuals(enc, tag):
    rows = []
    for fname, t in wins:
        acts = {
          "reversed":  torch.flip(t, dims=[1]),
          "shift+8":   torch.cat((t[:, 8:], t[:, -1:].expand(-1, 8, -1)), dim=1),
          "zero":      torch.zeros_like(t),
          "other-clip": wins[(wins.index((fname,t))+1) % len(wins)][1],
        }
        with torch.no_grad():
            rc, _ = enc(t, 9)
            rows.append({k: (enc(v, 9)[0] - rc) for k, v in acts.items()} | {"__ref__": rc})
    print(f"\n  {tag}")
    print(f"    {'counterfactual':<14}{'|dr| rms':>11}{'|dr| tv rms':>13}"
          f"{'|dr|/|r|':>10}{'tv/|r|':>9}")
    for k in ("reversed", "shift+8", "zero", "other-clip"):
        d  = torch.cat([r[k] for r in rows], dim=0)
        ref= torch.cat([r["__ref__"] for r in rows], dim=0)
        _, d_rms = tv_of(d); dt, _ = tv_of(d)
        _, ref_rms = tv_of(ref)
        print(f"    {k:<14}{d_rms:>11.6f}{dt:>13.6f}"
              f"{d_rms/ref_rms*100:>9.3f}%{dt/ref_rms*100:>8.3f}%")
    # cross-window: how much does the SAME action-name differ between clips?
    print(f"    (ref |r| rms across {len(rows)} windows = {ref_rms:.6f})")

counterfactuals(enc,      "TRAINED low-rank projection")
counterfactuals(enc_rand, "RANDOM full-rank projection (RMS-matched)")
