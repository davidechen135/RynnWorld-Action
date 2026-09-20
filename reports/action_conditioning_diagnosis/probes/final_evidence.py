"""FINAL: counterfactual discriminability of the INJECTED residual, using the
eval harness's exact counterfactual definitions (scripts/eval_native_action_v2_rot6d37.py:251-257):
    correct  = raw
    held     = first frame repeated
    zero     = zeros
    reversed = flip(dim=1)
    shifted  = roll(T//2, 1)  i.e. roll(16,1) for T=33
Trained low-rank projection vs random full-rank (output-RMS-matched).
"""
import sys, glob, copy
import torch, torch.nn as nn, torch.nn.functional as F
from safetensors.torch import load_file
sys.path.insert(0,"/mnt/workspace/umi-world-model-lab/third_party/RynnWorld-Teleop-umi")
from core.control import NativeTrajectoryConditionerV11

REPO="/mnt/workspace/umi-world-model-lab/third_party/RynnWorld-Teleop-umi"
DS="/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop"
enc = NativeTrajectoryConditionerV11(input_dim=148).eval()
enc.load_state_dict(torch.load(
  f"{REPO}/training/native_action_v11_temporal_alignment_300/checkpoint-300/native_trajectory_encoder.bin",
  map_location="cpu", weights_only=False), strict=False)

wins=[]
for p in sorted(glob.glob(f"{DS}/agibot_action_v10_state_256_v1/task*.safetensors")):
    t=load_file(p)["robot_trajectory"].unsqueeze(0).float()
    if t.shape[-1]==148 and t.abs().sum()>0: wins.append(t)
    if len(wins)==8: break

# decode an action residual into the exact token tensor the DiT receives,
# then measure how distinguishable the counterfactual is from 'correct'.
def tokens(r5d):
    """[N,3072,9,1,1] -> [N,9,3072]  (trainer flattens (B,C,T,H,W) then
    transposes -> time is the outer token index)."""
    return r5d.float().squeeze(-1).squeeze(-1).permute(0,2,1).contiguous()

def dist(a,b):
    return (a-b).square().mean().sqrt().item()

W = enc.input_residual_projection.weight.detach()
with torch.no_grad():
    feats=torch.cat([enc.output_norm(enc._encode(t,9)) for t in wins],0)
    tgt=F.linear(feats,W).square().mean().sqrt().item()
    torch.manual_seed(0); Wr=torch.randn_like(W)
    Wr=Wr*(tgt/F.linear(feats,Wr).square().mean().sqrt().item())
enc_rand=copy.deepcopy(enc)
enc_rand.input_residual_projection=nn.Linear(768,3072,bias=False).eval()
with torch.no_grad(): enc_rand.input_residual_projection.weight.copy_(Wr)

def run(enc,tag):
    rows={}
    for t in wins:
        acts={
          "held":     t[:, :1].expand_as(t),
          "zero":     torch.zeros_like(t),
          "reversed": torch.flip(t,dims=[1]),
          "shifted":  torch.roll(t, t.shape[1]//2, dims=1),
        }
        with torch.no_grad():
            rc=tokens(enc(t,9)[0])
            for k,v in acts.items():
                rows.setdefault(k,[]).append((tokens(enc(v,9)[0]), rc))
    print(f"\n  {tag}")
    print(f"    {'counterfactual':<12}{'|Dtokens| rms':>15}{'/ |correct|':>13}{'time-varying part':>19}")
    # reference scale: rms of the correct tokens
    ref = torch.cat([rc for _,rc in rows['zero']],0).square().mean().sqrt().item()
    for k in ("held","zero","reversed","shifted"):
        d  = torch.cat([a-b for a,b in rows[k]],0)          # [N,9,3072] since correct minus counterfactual
        dt = d - d.mean(dim=1,keepdim=True)                 # variation ACROSS TIME = motion capacity
        print(f"    {k:<12}{d.square().mean().sqrt().item():>15.6f}"
              f"{d.square().mean().sqrt().item()/ref*100:>12.3f}%"
              f"{dt.square().mean().sqrt().item()/ref*100:>18.3f}%")
    print(f"    (|correct tokens| rms = {ref:.6f})")

print(f"projections (both output-RMS-matched to {tgt:.6f}):")
s=torch.linalg.svdvals(W.float()); e=s.square(); e=e/e.sum()
sr=torch.linalg.svdvals(Wr.float()); er=sr.square(); er=er/er.sum()
print(f"  trained        rank@99%={int((e.cumsum(0)<0.99).sum())+1:>3}  top1={e[0]:.4f}")
print(f"  full-rank      rank@99%={int((er.cumsum(0)<0.99).sum())+1:>3}  top1={er[0]:.4f}")
run(enc,      "TRAINED low-rank projection")
run(enc_rand, "RANDOM full-rank projection")
