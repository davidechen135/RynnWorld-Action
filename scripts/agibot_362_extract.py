"""Extract only what task-362 fine-tune needs from the two 48GB AgiBot tars, then
delete the tars to respect the disk budget (/root/autodl-tmp ~438G free, 95% full).

We downloaded, per task 362 (Folding shorts):
  - observations tar  649552-654138.tar  (~48.6G, 8 camera streams; we keep only head_color.mp4)
  - proprio tar        648533-923022.tar  (~48G, ALL tasks mixed; we keep only task-362 h5 for chosen eps)

Selection (selection_362.json): 20 train + 3 held-out whole episodes, all in the tar range.

The tar internal layout is PROBED, not assumed: we list members once, learn the
prefix, then extract only the members whose episode id is in our set. Both tars are
deleted at the end (guarded: only if every expected file was extracted).

Usage: python scripts/agibot_362_extract.py [--keep-tars]
"""
import argparse
import json
import os
import subprocess
import sys
import tarfile
import time

DL = "/root/autodl-tmp"
OBS_TAR = f"{DL}/agibot_362_dl/observations/362/649552-654138.tar"
PROP_TAR = f"{DL}/agibot_362_proprio_dl/proprio_stats/648533-923022.tar"
OUT = f"{DL}/agibot_362_data"           # destination root for extracted files
SEL = f"{DL}/agibot_362_meta/selection_362.json"


def wait_for(tar_path, poll=60):
    """Block until <tar>.incomplete is gone and <tar> exists (download finished)."""
    inc = tar_path + ".incomplete"
    while True:
        if os.path.exists(tar_path) and not os.path.exists(inc):
            return
        sz = os.path.getsize(inc) / 1e9 if os.path.exists(inc) else 0
        print(f"  waiting for {os.path.basename(tar_path)} ... {sz:.1f}G downloaded", flush=True)
        time.sleep(poll)


def probe_members(tar_path, needles, limit=400):
    """Read the first `limit` member names, return those matching any needle.
    Used only to learn the path layout, printed for the operator."""
    names = []
    with tarfile.open(tar_path, "r|") as tf:      # streaming, no full scan
        for i, m in enumerate(tf):
            names.append(m.name)
            if i >= limit:
                break
    hits = [n for n in names if any(nd in n for nd in needles)]
    return names[:8], hits[:8]


def extract_selected(tar_path, episodes, want_suffix, out_root, sub_dir=""):
    """Stream the tar once; extract members whose path contains /<eid>/ and ends
    with want_suffix. Returns dict eid -> extracted path (or missing).

    sub_dir is inserted between the episode id and the file so the destination
    mirrors the canonical AgiBot layout that agibot_finetune_prep.py expects
    (observations/<task>/<eid>/videos/head_color.mp4)."""
    eids = {str(e) for e in episodes}
    found = {}
    with tarfile.open(tar_path, "r|") as tf:
        for m in tf:
            if not m.isfile() or not m.name.endswith(want_suffix):
                continue
            parts = m.name.split("/")
            hit = next((p for p in parts if p in eids), None)
            if hit is None:
                continue
            dest = os.path.join(out_root, hit, sub_dir, os.path.basename(m.name))
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with tf.extractfile(m) as src, open(dest, "wb") as dst:
                while True:
                    chunk = src.read(1 << 20)
                    if not chunk:
                        break
                    dst.write(chunk)
            found[hit] = dest
            print(f"  extracted ep {hit}: {os.path.getsize(dest)/1e6:.1f} MB", flush=True)
            if len(found) == len(eids):
                break
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-tars", action="store_true", help="do not delete tars after extract")
    ap.add_argument("--no-wait", action="store_true", help="fail instead of waiting if a tar is not ready")
    a = ap.parse_args()

    sel = json.load(open(SEL))
    all_eps = sel["train_episodes"] + sel["heldout_episodes"]
    print(f"task {sel['task_id']} — {len(all_eps)} episodes "
          f"({len(sel['train_episodes'])} train + {len(sel['heldout_episodes'])} held-out)")

    for t in (OBS_TAR, PROP_TAR):
        if not (os.path.exists(t) and not os.path.exists(t + ".incomplete")):
            if a.no_wait:
                sys.exit(f"not ready: {t}")
            print(f"waiting for {os.path.basename(t)} to finish downloading...")
            wait_for(t)
    print("both tars present. sizes:")
    for t in (OBS_TAR, PROP_TAR):
        print(f"  {os.path.basename(t)}: {os.path.getsize(t)/1e9:.1f} G")

    obs_out = os.path.join(OUT, "observations", "362")
    prop_out = os.path.join(OUT, "proprio_stats", "362")

    print("\n== probe observations tar layout ==")
    head, hits = probe_members(OBS_TAR, ["head_color.mp4"] + [str(e) for e in all_eps[:3]])
    print("first members:", *head, sep="\n  ")
    print("matches:", *hits, sep="\n  ")

    print("\n== extract head_color.mp4 for selected episodes ==")
    # prep expects observations/<task>/<eid>/videos/head_color.mp4
    obs_found = extract_selected(OBS_TAR, all_eps, "head_color.mp4", obs_out, sub_dir="videos")

    print("\n== extract proprio_stats.h5 for selected episodes (task-362 only) ==")
    # proprio tar mixes all tasks; the episode id in-path disambiguates ours.
    prop_found = extract_selected(PROP_TAR, all_eps, "proprio_stats.h5", prop_out)

    missing_obs = [e for e in all_eps if str(e) not in obs_found]
    missing_prop = [e for e in all_eps if str(e) not in prop_found]
    print(f"\nobs extracted {len(obs_found)}/{len(all_eps)}  missing={missing_obs}")
    print(f"prop extracted {len(prop_found)}/{len(all_eps)}  missing={missing_prop}")

    if missing_obs or missing_prop:
        print("\n!! not all files extracted — NOT deleting tars. Inspect above.")
        return
    if a.keep_tars:
        print("\n--keep-tars set; leaving tars in place.")
        return
    for t in (OBS_TAR, PROP_TAR):
        print(f"deleting {t} ({os.path.getsize(t)/1e9:.1f} G)")
        os.remove(t)
    print("done. freed ~97G.")


if __name__ == "__main__":
    main()
