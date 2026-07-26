"""VBench-style single-frame video quality baseline: MUSIQ imaging quality (via pyiqa,
the same metric backend VBench's imaging_quality dimension uses) over official-baseline
SFT + streaming rollouts. Score range ~0-100 (higher = better perceptual quality)."""
import os, glob, json
import torch, numpy as np, cv2
import pyiqa

device = "cuda"
metric = pyiqa.create_metric("musiq", device=device)  # MUSIQ-KonIQ, higher better

def score_video(path, stride=8):
    cap = cv2.VideoCapture(path)
    frames, i = [], 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % stride == 0:
            rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            frames.append(t)
        i += 1
    cap.release()
    if not frames:
        return None
    batch = torch.cat(frames).to(device)
    with torch.no_grad():
        s = metric(batch)
    return float(s.mean().item()), len(frames), i

targets = []
for f in sorted(glob.glob("outputs/repro_sft/*/generated_seed42.mp4")):
    targets.append(("SFT", os.path.basename(os.path.dirname(f)), f))
for f in sorted(glob.glob("outputs/repro_streaming/*/generated_seed42.mp4")):
    targets.append(("Streaming", os.path.basename(os.path.dirname(f)), f))

results = []
for mode, case, path in targets:
    r = score_video(path)
    if r is None:
        continue
    musiq, n_eval, n_total = r
    row = dict(mode=mode, case=case, musiq=round(musiq, 3),
               frames_scored=n_eval, total_frames=n_total)
    print(json.dumps(row))
    results.append(row)

sft = [r["musiq"] for r in results if r["mode"] == "SFT"]
stream = [r["musiq"] for r in results if r["mode"] == "Streaming"]
summary = dict(
    metric="MUSIQ-KonIQ (pyiqa, VBench imaging_quality backend)",
    higher_is_better=True,
    sft_mean=round(sum(sft) / len(sft), 3) if sft else None,
    streaming_mean=round(sum(stream) / len(stream), 3) if stream else None,
    per_case=results,
)
os.makedirs("outputs/vbench", exist_ok=True)
with open("outputs/vbench/imaging_quality.json", "w") as fp:
    json.dump(summary, fp, indent=2)
print("\nSFT mean MUSIQ:", summary["sft_mean"], "| Streaming mean:", summary["streaming_mean"])
print("WROTE outputs/vbench/imaging_quality.json")
