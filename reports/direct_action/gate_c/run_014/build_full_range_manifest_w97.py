#!/usr/bin/env python3
"""Build run_014's WINDOW=97 (~3.2s @30fps) episode-stratified window manifest.

Verbatim reuse of run_005's build_full_range_manifest.py stratification method
(stratified_starts: STRIDE=16 eligible grid, up to MAX_WINDOWS_PER_EPISODE=16 picked evenly
by index across the full eligible range per episode) -- only WINDOW changed from 33 to 97.
Simulating this method beforehand (see conversation) showed: 1456 total train windows
(91 episodes x 16, unchanged from the WINDOW=33 manifest), median 18% worst-case adjacent
overlap, max 67% (only for the single shortest train episode, 699 frames), 35/91 episodes
with zero overlap. This was chosen over an ad hoc small fixed-stride resampling (e.g.
start=1, start=30) specifically because that pattern was already tried and rejected in
run_005 (it concentrated coverage near the start of each episode); the even-by-index
stratification here spreads any unavoidable overlap across each episode's full span instead.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd

REPO = Path('/mnt/workspace/RynnWorld-Teleop')
VIDEO_ROOT = Path('/tmp/scratch/gate_c_run004_data/data/videos/chunk-000/observation.images.top_head')
PARQUET_ROOT = Path('/tmp/scratch/gate_b_task3400/data/data/chunk-000')
SPLIT = REPO / 'reports/direct_action/gate_a/run_002/artifacts/episode_split.json'
OUT = REPO / 'reports/direct_action/gate_c/run_014'
WINDOW = 97
STRIDE = 16
MAX_WINDOWS_PER_EPISODE = 16
STRATA = 4


def extract(a):
    return np.concatenate([a[:, 2:8], a[:, 8:16], a[:, 16:30], a[:, 33:38]], axis=1)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def stratified_starts(n_frames: int):
    max_start = n_frames - WINDOW
    if max_start < 0:
        return []
    eligible = np.arange(0, max_start + 1, STRIDE, dtype=np.int64)
    if len(eligible) <= MAX_WINDOWS_PER_EPISODE:
        starts = eligible
    else:
        indexes = np.rint(np.linspace(0, len(eligible) - 1, MAX_WINDOWS_PER_EPISODE)).astype(int)
        starts = eligible[np.unique(indexes)]
    starts = starts.tolist()
    result = []
    for start in starts:
        fraction = start / max_start if max_start else 0.0
        stratum = min(STRATA - 1, int(fraction * STRATA))
        result.append({'start': int(start), 'stratum': stratum, 'fraction': float(fraction)})
    return result


def main():
    split = json.loads(SPLIT.read_text())
    train = set(split['train_episodes'])
    held = set(split['held_out_episodes'])
    records, errors = [], []
    for episode in range(110):
        video = VIDEO_ROOT / f'episode_{episode:06d}.mp4'
        parquet = PARQUET_ROOT / f'episode_{episode:06d}.parquet'
        try:
            props = iio.improps(video, plugin='pyav')
            meta = iio.immeta(video, plugin='pyav')
            n_frames = int(props.n_images)
            fps = float(meta.get('fps', 30))
            df = pd.read_parquet(parquet)
            if len(df) < n_frames:
                raise ValueError(f'parquet {len(df)} < video {n_frames}')
            actions = extract(np.stack(df.iloc[:n_frames]['action'].values))
            if actions.shape != (n_frames, 33):
                raise ValueError(f'action shape {actions.shape}')
            if not np.isfinite(actions).all():
                raise ValueError('nonfinite action')
            windows = stratified_starts(n_frames)
            records.append({
                'episode': episode,
                'split': 'train' if episode in train else 'held_out',
                'video': str(video),
                'parquet': str(parquet),
                'video_frames': n_frames,
                'parquet_rows': len(df),
                'fps': fps,
                'action_shape': list(actions.shape),
                'video_sha256': sha(video),
                'parquet_sha256': sha(parquet),
                'window_count': len(windows),
                'windows': windows,
                'window_starts': [item['start'] for item in windows],
                'window_size': WINDOW,
                'stride': STRIDE,
                'strata': sorted({item['stratum'] for item in windows}),
            })
        except Exception as exc:
            errors.append({'episode': episode, 'error': repr(exc)})
    if errors:
        raise RuntimeError(json.dumps(errors, indent=2))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'artifacts').mkdir(exist_ok=True)
    manifest = {
        'run_id': 'direct_action/gate_c/run_014',
        'sampling': 'full_range_even_stratified_window97',
        'window_size': WINDOW,
        'stride': STRIDE,
        'max_windows_per_episode': MAX_WINDOWS_PER_EPISODE,
        'strata': STRATA,
        'total_episodes': 110,
        'train_episodes': sorted(train),
        'held_out_episodes': sorted(held),
        'records': records,
        'train_window_count': sum(r['window_count'] for r in records if r['split'] == 'train'),
        'held_out_window_count': sum(r['window_count'] for r in records if r['split'] == 'held_out'),
        'train_start_min': min(item['start'] for r in records if r['split'] == 'train' for item in r['windows']),
        'train_start_max': max(item['start'] for r in records if r['split'] == 'train' for item in r['windows']),
    }
    (OUT / 'artifacts/window_manifest_full_range_w97.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({
        'episodes': len(records),
        'train_windows': manifest['train_window_count'],
        'held_out_windows': manifest['held_out_window_count'],
        'train_start_range': [manifest['train_start_min'], manifest['train_start_max']],
        'out': str(OUT / 'artifacts/window_manifest_full_range_w97.json'),
    }, indent=2))


if __name__ == '__main__':
    main()
