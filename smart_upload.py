#!/usr/bin/env python3
"""
扫描所有 outputs 目录，构建 manifest，直接上传到 WandB
不依赖 phase1 的成功结果
"""

import os, glob, json, time
import wandb

# 清除代理
for var in ["http_proxy","https_proxy","HTTP_PROXY","HTTPS_PROXY","ALL_PROXY","all_proxy"]:
    os.environ.pop(var, None)

API_KEY = os.environ["WANDB_API_KEY"]
ENTITY  = "lv-robotics-lab-nus"
PROJECT = "rynnworld"
BASE    = "/root/autodl-tmp/RynnWorld-Teleop"

# 8 个 demo case 及对应 outputs 子目录名
CASES = [
    {"name": "assemble_jenga",      "folder": "assemble_jenga_001",        "out_name": "assemble_jenga"},
    {"name": "basic_fold",          "folder": "basic_fold_009",            "out_name": "basic_fold"},
    {"name": "basic_pick_place",    "folder": "basic_pick_place_000",      "out_name": "basic_pick_place"},
    {"name": "clean_surface",       "folder": "clean_surface_001",         "out_name": "clean_surface"},
    {"name": "clip_unclip_papers",  "folder": "clip_unclip_papers_006",    "out_name": "clip_unclip_papers"},
    {"name": "color_task",          "folder": "color_004",                 "out_name": "color_task"},
    {"name": "flip_pages",          "folder": "flip_pages_008",            "out_name": "flip_pages"},
    {"name": "fold_unfold_paper",   "folder": "fold_unfold_paper_basic_008","out_name": "fold_unfold_paper"},
]

wandb.login(key=API_KEY)
print(f"Logged in. Target: {ENTITY}/{PROJECT}\n")

summary = []

for case in CASES:
    name       = case["name"]
    folder     = case["folder"]
    out_name   = case["out_name"]
    image_path = os.path.join(BASE, "example", folder, "first_frame.png")
    ctrl_path  = os.path.join(BASE, "example", folder, "control_video.mp4")

    # 找最新的输出目录
    pattern = os.path.join(BASE, "outputs", out_name, "*")
    dirs = sorted(glob.glob(pattern))
    if not dirs:
        print(f"[SKIP] {name} — no output dir found")
        continue

    out_dir = dirs[-1]

    # 检查是否有 rollout 视频
    rollout_mp4 = os.path.join(out_dir, "rollout.mp4")
    if not os.path.isfile(rollout_mp4):
        print(f"[SKIP] {name} — no rollout.mp4 in {out_dir}")
        continue

    print(f"\n[UPLOAD] {name}")
    print(f"  out_dir     : {out_dir}")
    print(f"  control_vid : {ctrl_path}")

    try:
        run = wandb.init(
            entity=ENTITY,
            project=PROJECT,
            name=f"demo_{name}",
            tags=["demo", "inference", "teleop", "skeleton"],
            config={
                "task": name,
                "example_folder": folder,
                "out_dir": out_dir,
            },
        )
        print(f"  Run: {run.url}")

        log_data = {}

        # ── 骨骼控制视频（输入）
        if os.path.isfile(ctrl_path):
            log_data["control_skeleton_video"] = wandb.Video(
                ctrl_path, fps=16, format="mp4",
                caption=f"[INPUT] Skeleton Control — {name}"
            )
            print(f"  + control_skeleton_video")

        # ── 生成的 rollout 视频（输出）
        log_data["rollout_video"] = wandb.Video(
            rollout_mp4, fps=16, format="mp4",
            caption=f"[OUTPUT] Generated Rollout — {name}"
        )
        print(f"  + rollout_video")

        # ── GIF
        gif = os.path.join(out_dir, "rollout.gif")
        if os.path.isfile(gif):
            log_data["rollout_gif"] = wandb.Image(gif, caption=f"Rollout GIF — {name}")
            print(f"  + rollout_gif")

        # ── 输入第一帧
        if os.path.isfile(image_path):
            log_data["input_first_frame"] = wandb.Image(
                image_path, caption=f"[INPUT] First Frame — {name}"
            )
            print(f"  + input_first_frame")

        # ── 图表
        for key, fname, cap in [
            ("action_curve",  "action_curve.png",  "Action Curves"),
            ("trajectory",    "trajectory.png",    "3D End-Effector Trajectory"),
            ("feature_pca",   "feature_pca.png",   "Visual Feature PCA"),
            ("feature_tsne",  "feature_tsne.png",  "Visual Feature t-SNE"),
            ("benchmark",     "benchmark.png",     "System Benchmark"),
        ]:
            p = os.path.join(out_dir, fname)
            if os.path.isfile(p):
                log_data[key] = wandb.Image(p, caption=f"{cap} — {name}")
                print(f"  + {key}")

        # ── 指标
        for mf in ["metrics.json", "benchmark.json"]:
            p = os.path.join(out_dir, mf)
            if os.path.isfile(p):
                with open(p) as f:
                    metrics = json.load(f)
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        log_data[f"perf/{k}"] = v
                break

        run.log(log_data)
        run.finish()
        print(f"  ✅  {len(log_data)} items uploaded → {run.url}")
        summary.append({"name": name, "url": run.url, "files": len(log_data)})

    except Exception as e:
        print(f"  ❌  {name}: {e}")
        summary.append({"name": name, "error": str(e)})

print(f"\n{'='*60}")
print(f"UPLOAD DONE — {len(summary)} runs")
for s in summary:
    url = s.get("url", s.get("error",""))
    n   = s.get("files", 0)
    print(f"  {s['name']:30s}  {n:2d} files  {url}")
print(f"\n➡  https://wandb.ai/{ENTITY}/{PROJECT}")
