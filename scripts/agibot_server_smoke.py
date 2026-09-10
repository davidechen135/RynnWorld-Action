"""Migration smoke tests for the AgiBot native-trajectory training server."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import time
from pathlib import Path

import h5py
import imageio_ffmpeg
import torch
import torch.distributed as dist


REQUIRED_STATE_FIELDS = (
    "state/end/position",
    "state/end/orientation",
    "state/effector/position",
    "state/head/position",
    "state/waist/position",
    "timestamp",
)


def check_data(h5_path: Path, video_path: Path) -> dict:
    with h5py.File(h5_path, "r") as handle:
        fields = {
            name: {
                "shape": list(handle[name].shape),
                "dtype": str(handle[name].dtype),
                "finite": bool(torch.from_numpy(handle[name][:]).isfinite().all()),
            }
            for name in REQUIRED_STATE_FIELDS
        }
        lengths = {value["shape"][0] for value in fields.values()}

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    decoded = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    return {
        "h5": str(h5_path),
        "video": str(video_path),
        "fields": fields,
        "consistent_lengths": len(lengths) == 1,
        "video_decode": decoded.returncode == 0,
    }


def check_cuda() -> dict:
    devices = []
    for index in range(torch.cuda.device_count()):
        with torch.cuda.device(index):
            torch.cuda.reset_peak_memory_stats()
            left = torch.randn((2048, 2048), device="cuda", dtype=torch.bfloat16)
            right = torch.randn((2048, 2048), device="cuda", dtype=torch.bfloat16)
            result = left @ right
            torch.cuda.synchronize()
            devices.append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "capability": list(torch.cuda.get_device_capability(index)),
                    "bf16_finite": bool(result.isfinite().all()),
                    "peak_memory_mib": round(
                        torch.cuda.max_memory_allocated(index) / 1024**2, 2
                    ),
                }
            )
    return {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
        "devices": devices,
        "sdpa_available": hasattr(
            torch.nn.functional, "scaled_dot_product_attention"
        ),
    }


def check_distributed() -> dict:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        return {"ran": False, "reason": "launch with torchrun for NCCL smoke"}

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    tensor = torch.tensor([float(dist.get_rank() + 1)], device="cuda")
    dist.all_reduce(tensor)
    expected = world_size * (world_size + 1) / 2
    result = {
        "ran": True,
        "rank": dist.get_rank(),
        "world_size": world_size,
        "all_reduce": tensor.item(),
        "expected": expected,
        "passed": tensor.item() == expected,
    }
    dist.destroy_process_group()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=Path)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    started = time.time()
    rank = int(os.environ.get("RANK", "0"))
    report = {
        "host": socket.gethostname(),
        "rank": rank,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cuda": check_cuda(),
        "distributed": check_distributed(),
    }
    if args.h5 and args.video:
        report["data"] = check_data(args.h5, args.video)
    report["elapsed_seconds"] = round(time.time() - started, 3)
    report["passed"] = (
        report["cuda"]["device_count"] >= 2
        and all(item["bf16_finite"] for item in report["cuda"]["devices"])
        and (
            not report["distributed"]["ran"]
            or report["distributed"]["passed"]
        )
        and (
            "data" not in report
            or (
                report["data"]["consistent_lengths"]
                and report["data"]["video_decode"]
                and all(
                    item["finite"] for item in report["data"]["fields"].values()
                )
            )
        )
    )

    if rank == 0:
        rendered = json.dumps(report, indent=2, ensure_ascii=False)
        print(rendered)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
