#!/usr/bin/env python3
"""Build the full 21-shard gate_d manifest (WINDOW=33, STRIDE=16, cap=42).

Covers all shards of task_3400 + task_3401. The reference shard
task_3400/313498_314085 uses gate_a's canonical 91/19 split; the other 20
shards use the hash-derived split from build_combined_split.json.

Global episode indices are assigned train-first (0..1161 for train), then
held-out (1162..1375), so cache_sample's `:06d` filename stays valid and
globally unique. Each record carries task_id/shard/real_episode/video_relpath/
archive_path so the trainer can locate the video (extracted dir or archive).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path('/mnt/workspace/RynnWorld-Teleop')
EXTRACT_ROOT = Path('/mnt/workspace/agibot_extracted')
COMBINED_SPLIT = REPO / 'reports/direct_action/gate_d/run_001/artifacts/build_combined_split.json'
GATE_A_SPLIT = REPO / 'reports/direct_action/gate_a/run_002/artifacts/episode_split.json'
SHARD_INVENTORY = REPO / 'reports/direct_action/gate_d/run_001/artifacts/shard_inventory.json'
OUT = REPO / 'reports/direct_action/gate_d/run_001'
REFERENCE_SHARD = '313498_314085'
CAMERA_KEY = 'observation.images.top_head'
WINDOW = 33
STRIDE = 16
CAP = 42
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
    if len(eligible) <= CAP:
        starts = eligible
    else:
        # Pick evenly across the complete eligible range, including both endpoints.
        indexes = np.rint(np.linspace(0, len(eligible) - 1, CAP)).astype(int)
        starts = eligible[np.unique(indexes)]
    result = []
    for start in starts.tolist():
        fraction = start / max_start if max_start else 0.0
        stratum = min(STRATA - 1, int(fraction * STRATA))
        result.append({'start': int(start), 'stratum': stratum, 'fraction': float(fraction)})
    return result


def build_shard_splits():
    combined = json.loads(COMBINED_SPLIT.read_text())
    gate_a = json.loads(GATE_A_SPLIT.read_text())
    inventory = json.loads(SHARD_INVENTORY.read_text())
    archive_by_shard = {}
    for task_id, tinfo in inventory['tasks'].items():
        for shard in tinfo['shards']:
            name = shard['name'].replace('.tar.gz', '')
            archive_by_shard[f'{task_id}/{name}'] = shard['path']

    splits = []
    for entry in combined['shards']:
        task_id = entry['task_id']
        shard_name = entry['shard_name']
        if shard_name == REFERENCE_SHARD:
            train_episodes = gate_a['train_episodes']
            held_episodes = gate_a['held_out_episodes']
        else:
            train_episodes = entry['train_episodes']
            held_episodes = entry['held_out_episodes']
        splits.append({
            'task_id': task_id,
            'shard_name': shard_name,
            'total_episodes': entry['total_episodes'],
            'archive_path': archive_by_shard.get(f'{task_id}/{shard_name}'),
            'train_episodes': sorted(train_episodes),
            'held_out_episodes': sorted(held_episodes),
        })
    return splits


def assign_global_episodes(splits):
    # train-first then held-out, deterministic across shard/episode order
    mapping = {}
    next_idx = 0
    for s in splits:
        for ep in s['train_episodes']:
            mapping[(s['task_id'], s['shard_name'], ep)] = next_idx
            next_idx += 1
    held_base = next_idx
    for s in splits:
        for ep in s['held_out_episodes']:
            mapping[(s['task_id'], s['shard_name'], ep)] = next_idx
            next_idx += 1
    return mapping, held_base, next_idx


def episode_parquet(task_id, shard_name, local_episode):
    return (EXTRACT_ROOT / f'task_{task_id}' / shard_name
            / 'data/data/chunk-000' / f'episode_{local_episode:06d}.parquet')


def record_for(shard_split, local_episode, global_episode, info, parquet_path, actions):
    task_id = shard_split['task_id']
    shard_name = shard_split['shard_name']
    train = set(shard_split['train_episodes'])
    n_frames = len(actions)
    windows = stratified_starts(n_frames)
    chunk = local_episode // info['chunks_size']
    # archive members are rooted at data/, so both parquet/video relpaths carry it
    rel_path = 'data/' + info['video_path'].format(
        episode_chunk=chunk, video_key=CAMERA_KEY, episode_index=local_episode)
    return {
        'episode': global_episode,
        'real_episode': local_episode,
        'task_id': task_id,
        'shard': shard_name,
        'archive_path': shard_split['archive_path'],
        'video_relpath': rel_path,
        'split': 'train' if local_episode in train else 'held_out',
        'parquet': str(parquet_path),
        'video_frames': n_frames,
        'fps': float(info.get('fps', 30)),
        'action_shape': list(actions.shape),
        'parquet_sha256': sha(parquet_path),
        'window_count': len(windows),
        'windows': windows,
        'window_starts': [item['start'] for item in windows],
        'window_size': WINDOW,
        'stride': STRIDE,
        'strata': sorted({item['stratum'] for item in windows}),
    }


def main():
    splits = build_shard_splits()
    mapping, held_base, total_global = assign_global_episodes(splits)
    total_train = sum(len(s['train_episodes']) for s in splits)
    total_held = sum(len(s['held_out_episodes']) for s in splits)
    assert total_global == total_train + total_held
    assert held_base == total_train

    records, errors = [], []
    for s in splits:
        task_id = s['task_id']
        shard_name = s['shard_name']
        shard_dir = EXTRACT_ROOT / f'task_{task_id}' / shard_name
        info = json.loads((shard_dir / 'data/meta/info.json').read_text())
        parquet_dir = shard_dir / 'data/data/chunk-000'
        for local_episode in range(s['total_episodes']):
            parquet_path = parquet_dir / f'episode_{local_episode:06d}.parquet'
            try:
                df = pd.read_parquet(parquet_path)
                if len(df) == 0:
                    raise ValueError('empty parquet')
                actions = extract(np.stack(df['action'].values))
                if actions.shape[1] != 33:
                    raise ValueError(f'action shape {actions.shape}')
                if not np.isfinite(actions).all():
                    raise ValueError('nonfinite action')
                global_episode = mapping[(task_id, shard_name, local_episode)]
                records.append(record_for(
                    s, local_episode, global_episode, info, parquet_path, actions))
            except Exception as exc:
                errors.append({
                    'task_id': task_id, 'shard': shard_name,
                    'episode': local_episode, 'error': repr(exc)})
    if errors:
        raise RuntimeError(json.dumps(errors, indent=2))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'artifacts').mkdir(exist_ok=True)
    by_global = {r['episode']: {
        'task_id': r['task_id'], 'shard': r['shard'], 'real_episode': r['real_episode'],
        'archive_path': r['archive_path'], 'video_relpath': r['video_relpath'],
    } for r in records}
    manifest = {
        'run_id': 'direct_action/gate_d/run_001',
        'sampling': 'full_range_even_stratified',
        'window_size': WINDOW,
        'stride': STRIDE,
        'max_windows_per_episode': CAP,
        'strata': STRATA,
        'total_episodes': len(records),
        'train_episodes': total_train,
        'held_out_episodes': total_held,
        'n_shards': len(splits),
        'shards': sorted({
            (r['task_id'], r['shard']) for r in records
        }),
        'global_episode_lookup': by_global,
        'records': records,
        'train_window_count': sum(r['window_count'] for r in records if r['split'] == 'train'),
        'held_out_window_count': sum(r['window_count'] for r in records if r['split'] == 'held_out'),
        'train_start_min': min(item['start'] for r in records if r['split'] == 'train' for item in r['windows']),
        'train_start_max': max(item['start'] for r in records if r['split'] == 'train' for item in r['windows']),
    }
    out_path = OUT / 'artifacts/window_manifest_full_dataset.json'
    out_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({
        'episodes': len(records),
        'train_episodes': total_train,
        'held_out_episodes': total_held,
        'train_windows': manifest['train_window_count'],
        'held_out_windows': manifest['held_out_window_count'],
        'train_start_range': [manifest['train_start_min'], manifest['train_start_max']],
        'shards': manifest['n_shards'],
        'out': str(out_path),
    }, indent=2))


if __name__ == '__main__':
    main()
