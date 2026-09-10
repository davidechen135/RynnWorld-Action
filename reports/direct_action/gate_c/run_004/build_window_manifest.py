#!/usr/bin/env python3
"""Build a validated run_004 contiguous-window manifest for task_3400."""
import json
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import imageio.v3 as iio

REPO = Path('/mnt/workspace/RynnWorld-Teleop')
VIDEO_ROOT = Path('/tmp/scratch/gate_c_run004_data/data/videos/chunk-000/observation.images.top_head')
PARQUET_ROOT = Path('/tmp/scratch/gate_b_task3400/data/data/chunk-000')
SPLIT = REPO / 'reports/direct_action/gate_a/run_002/artifacts/episode_split.json'
OUT = REPO / 'reports/direct_action/gate_c/run_004'
WINDOW = 33
STRIDE = 16
MAX_WINDOWS_PER_EPISODE = 16

def extract(a):
    return np.concatenate([a[:,2:8], a[:,8:16], a[:,16:30], a[:,33:38]], axis=1)

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def main():
    split=json.loads(SPLIT.read_text())
    train=set(split['train_episodes']); held=set(split['held_out_episodes'])
    records=[]; errors=[]
    for ep in range(110):
        vp=VIDEO_ROOT/f'episode_{ep:06d}.mp4'; pp=PARQUET_ROOT/f'episode_{ep:06d}.parquet'
        try:
            props=iio.improps(vp, plugin='pyav'); meta=iio.immeta(vp, plugin='pyav')
            n=int(props.n_images); fps=float(meta.get('fps',30))
            df=pd.read_parquet(pp)
            if len(df) < n: raise ValueError(f'parquet {len(df)} < video {n}')
            actions40=np.stack(df.iloc[:n]['action'].values)
            a33=extract(actions40)
            if a33.shape != (n,33): raise ValueError(f'action shape {a33.shape}')
            if not np.isfinite(a33).all(): raise ValueError('nonfinite action')
            starts=list(range(0,n-WINDOW+1,STRIDE))[:MAX_WINDOWS_PER_EPISODE]
            rec={'episode':ep,'split':'train' if ep in train else 'held_out','video':str(vp),'parquet':str(pp),'video_frames':n,'parquet_rows':len(df),'fps':fps,'action_shape':list(a33.shape),'video_sha256':sha(vp),'parquet_sha256':sha(pp),'window_count':len(starts),'window_starts':starts,'window_size':WINDOW,'stride':STRIDE}
            records.append(rec)
        except Exception as e: errors.append({'episode':ep,'error':repr(e)})
    if errors: raise RuntimeError(json.dumps(errors,indent=2))
    OUT.mkdir(parents=True, exist_ok=True); (OUT/'artifacts').mkdir(exist_ok=True)
    manifest={'run_id':'direct_action/gate_c/run_004','window_size':WINDOW,'stride':STRIDE,'max_windows_per_episode':MAX_WINDOWS_PER_EPISODE,'total_episodes':110,'train_episodes':sorted(train),'held_out_episodes':sorted(held),'records':records,'train_window_count':sum(r['window_count'] for r in records if r['split']=='train'),'held_out_window_count':sum(r['window_count'] for r in records if r['split']=='held_out')}
    (OUT/'artifacts/window_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({'episodes':len(records),'train_windows':manifest['train_window_count'],'held_out_windows':manifest['held_out_window_count'],'out':str(OUT)},indent=2))
if __name__=='__main__': main()
