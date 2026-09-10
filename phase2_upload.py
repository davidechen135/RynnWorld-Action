#!/usr/bin/env python3
"""
Phase 2: 读取 manifest，批量上传所有推理结果到 WandB
包含骨骼控制视频 + 生成视频 + 所有图表
"""

import os, glob, json
import wandb

# 清除代理
for var in ["http_proxy","https_proxy","HTTP_PROXY","HTTPS_PROXY","ALL_PROXY","all_proxy"]:
    os.environ.pop(var, None)

API_KEY = os.environ["WANDB_API_KEY"]
ENTITY  = "lv-robotics-lab-nus"
PROJECT = "rynnworld"

MANIFEST = "/root/inference_manifest.json"

wandb.login(key=API_KEY)
print(f"Logged in. Uploading to {ENTITY}/{PROJECT}\n")

with open(MANIFEST) as f:
    results = json.load(f)

upload_summary = []

for r in results:
    name = r["name"]
    if not r.get("success"):
        print(f"[SKIP] {name} — {r.get('error','failed')}")
        continue

    out_dir      = r["out_dir"]
    control_path = r.get("control_path", "")
    image_path   = r.get("image_path", "")

    print(f"\n[UPLOAD] {name}  ({out_dir})")

    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        name=f"demo_{name}",
        tags=["demo", "inference", "teleop", "skeleton"],
        config={
            "task": name,
            "inference_time_s": r.get("inference_time", 0),
            "out_dir": out_dir,
        },
    )
    print(f"  Run URL: {run.url}")

    log_data = {}

    # 骨骼控制视频（输入）
    if os.path.isfile(control_path):
        log_data["control_skeleton_video"] = wandb.Video(
            control_path, fps=16, format="mp4",
            caption=f"[INPUT] Skeleton Control — {name}"
        )
        print(f"  + control_skeleton_video")

    # 生成 rollout 视频（输出）
    for vfile in ["rollout.mp4"]:
        p = os.path.join(out_dir, vfile)
        if os.path.isfile(p):
            log_data["rollout_video"] = wandb.Video(
                p, fps=16, format="mp4",
                caption=f"[OUTPUT] Generated Rollout — {name}"
            )
            print(f"  + rollout_video")

    # GIF
    gif = os.path.join(out_dir, "rollout.gif")
    if os.path.isfile(gif):
        log_data["rollout_gif"] = wandb.Image(gif, caption=f"Rollout GIF — {name}")
        print(f"  + rollout_gif")

    # 输入第一帧
    if os.path.isfile(image_path):
        log_data["input_first_frame"] = wandb.Image(
            image_path, caption=f"[INPUT] First Frame — {name}"
        )
        print(f"  + input_first_frame")

    # 图表
    for key, fname, cap in [
        ("action_curve",    "action_curve.png",    "Action Curves"),
        ("trajectory",      "trajectory.png",      "3D End-Effector Trajectory"),
        ("feature_pca",     "feature_pca.png",     "Visual Feature PCA"),
        ("feature_tsne",    "feature_tsne.png",    "Visual Feature t-SNE"),
        ("benchmark",       "benchmark.png",       "System Performance Benchmark"),
        ("attention_map",   "attention_layer0.png","Attention Heatmap"),
    ]:
        p = os.path.join(out_dir, fname)
        if os.path.isfile(p):
            log_data[key] = wandb.Image(p, caption=f"{cap} — {name}")
            print(f"  + {key}")

    # 指标
    for mfile in ["metrics.json", "benchmark.json"]:
        p = os.path.join(out_dir, mfile)
        if os.path.isfile(p):
            with open(p) as f:
                metrics = json.load(f)
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    log_data[f"perf/{k}"] = v
            break

    run.log(log_data)
    run.finish()

    print(f"  ✅ Done  ({len(log_data)} items)")
    upload_summary.append({"name": name, "url": run.url, "files": len(log_data)})

print(f"\n{'='*60}")
print(f"UPLOAD COMPLETE  — {len(upload_summary)}/{len(results)} runs")
for s in upload_summary:
    print(f"  {s['name']:30s}  {s['files']:2d} files  {s['url']}")
print(f"\nProject: https://wandb.ai/{ENTITY}/{PROJECT}")
