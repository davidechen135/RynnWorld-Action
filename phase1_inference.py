#!/usr/bin/env python3
"""
Phase 1: 批量推理 (不调用任何 wandb)
对 8 个 example case 运行 inference_benchmark.py
"""

import os, sys, subprocess, glob, time, json

BASE_DIR = "/root/autodl-tmp/RynnWorld-Teleop"
CKPT     = "pretrained/RynnWorld-Teleop"

CASES = [
    {"name": "assemble_jenga",      "folder": "assemble_jenga_001"},
    {"name": "basic_fold",          "folder": "basic_fold_009"},
    {"name": "basic_pick_place",    "folder": "basic_pick_place_000"},
    {"name": "clean_surface",       "folder": "clean_surface_001"},
    {"name": "clip_unclip_papers",  "folder": "clip_unclip_papers_006"},
    {"name": "color_task",          "folder": "color_004"},
    {"name": "flip_pages",          "folder": "flip_pages_008"},
    {"name": "fold_unfold_paper",   "folder": "fold_unfold_paper_basic_008"},
]

os.chdir(BASE_DIR)
results = []

for idx, case in enumerate(CASES):
    name, folder = case["name"], case["folder"]
    print(f"\n{'='*60}")
    print(f"[{idx+1}/{len(CASES)}] {name}  ({folder})")
    print(f"{'='*60}")
    sys.stdout.flush()

    image_path   = f"example/{folder}/first_frame.png"
    control_path = f"example/{folder}/control_video.mp4"
    text_emb     = f"example/{folder}/text_embedding.safetensors"

    missing = [p for p in [image_path, control_path, text_emb] if not os.path.isfile(p)]
    if missing:
        print(f"  SKIP — missing: {missing}")
        results.append({"name": name, "success": False, "error": f"missing {missing}"})
        continue

    cmd = [
        sys.executable, "inference_benchmark.py",
        "--image",          image_path,
        "--control_video",  control_path,
        "--text_embedding", text_emb,
        "--experiment_name", name,
        "--checkpoint",     CKPT,
        "--disable_cpu_offload",
        "--seeds",          "42",
    ]
    print(f"  CMD: {' '.join(cmd)}")
    sys.stdout.flush()

    t0   = time.time()
    proc = subprocess.run(cmd, cwd=BASE_DIR)
    elapsed = time.time() - t0

    if proc.returncode != 0:
        print(f"  ERROR (code={proc.returncode}) after {elapsed:.0f}s")
        results.append({"name": name, "success": False, "error": f"exit {proc.returncode}"})
    else:
        # 找最新输出目录
        dirs = sorted(glob.glob(f"outputs/{name}/*"))
        out_dir = dirs[-1] if dirs else "NOT_FOUND"
        print(f"  OK  {elapsed:.0f}s  -> {out_dir}")
        results.append({"name": name, "success": True,
                        "out_dir": out_dir, "inference_time": elapsed,
                        "control_path": os.path.abspath(control_path),
                        "image_path": os.path.abspath(image_path)})

    sys.stdout.flush()

# 保存结果清单给 Phase 2 读取
manifest_path = "/root/inference_manifest.json"
with open(manifest_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*60}")
print("PHASE 1 COMPLETE")
for r in results:
    mark = "OK" if r.get("success") else "FAIL"
    print(f"  [{mark}] {r['name']}")
print(f"\nManifest saved: {manifest_path}")
