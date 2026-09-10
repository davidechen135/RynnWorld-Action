"""Gate D run_001, Phase 6 verification: byte-identical check of
`core.control.negative_controls.build_conditions` against run_011's inline
five-condition block (`train_5000_corrected.py:213-221`).

Usage:
    python3 verify_negative_controls.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_REPO_ROOT))

from core.control.negative_controls import build_conditions  # noqa: E402

SHUFFLE_SEED = 314159  # run_011's SHUFFLE_SEED
SEED = 42  # run_011's SEED, used only to generate deterministic test input


def normalize(action: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (action - mean) / std


def run_011_inline(correct_raw: torch.Tensor, mean_t: torch.Tensor, std_t: torch.Tensor, device) -> dict:
    """Verbatim reproduction of train_5000_corrected.py:213-221."""
    correct = normalize(correct_raw, mean_t, std_t)
    wrong_raw = torch.roll(correct_raw, shifts=correct_raw.shape[1] // 2, dims=1).clone()
    wrong = normalize(wrong_raw, mean_t, std_t)
    permutation = torch.randperm(
        correct_raw.shape[1], generator=torch.Generator(device=device).manual_seed(SHUFFLE_SEED), device=device
    )
    return {
        "correct": correct, "zero": torch.zeros_like(correct), "shuffled": correct[:, permutation],
        "reversed": torch.flip(correct, dims=[1]), "wrong": wrong,
    }


def main() -> None:
    device = torch.device("cpu")
    generator = torch.Generator(device=device).manual_seed(SEED)
    batch, window, action_dim = 1, 33, 37
    correct_raw = torch.randn(batch, window, action_dim, generator=generator, device=device)
    mean_t = torch.randn(action_dim, generator=generator, device=device)
    std_t = torch.rand(action_dim, generator=generator, device=device) + 0.5  # avoid near-zero std

    reference = run_011_inline(correct_raw, mean_t, std_t, device)
    correct_normalized = normalize(correct_raw, mean_t, std_t)
    candidate = build_conditions(correct_normalized, SHUFFLE_SEED, device)

    all_match = True
    for name in reference:
        match = torch.equal(reference[name], candidate[name])
        all_match &= match
        print(f"{name}: byte_identical={match}")
        if not match:
            print(f"  max_abs_diff={(reference[name] - candidate[name]).abs().max().item()}")

    print(f"\nALL_MATCH={all_match}")
    if not all_match:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
