"""Shared five-condition negative-control construction (Gate D Phase 6).

Extracted verbatim (see note below) from the inline block duplicated
byte-identically 3x across `reports/direct_action/gate_c/run_011/012/013`'s
`train_5000_corrected.py:213-221`, to avoid drift when a new Gate D training
script needs the same correct/zero/shuffled/reversed/wrong construction.

Note on the "wrong" condition: run_011 builds `wrong` by rolling the RAW
(un-normalized) trajectory and then normalizing, i.e.
    wrong = normalize(roll(correct_raw), mean, std)
rather than rolling the already-normalized tensor. Since normalize() is an
elementwise affine map applied identically at every window position (mean/std
broadcast over the time/window dimension, dim=1), it commutes with roll/flip/
permutation along dim=1:
    normalize(roll(x)) == roll(normalize(x))
so taking the already-normalized `correct` tensor as input (as the plan's
`build_conditions(correct, shuffle_seed, device)` signature requires) and
rolling/permuting/flipping it directly is numerically identical to run_011's
normalize-after-roll order. This is verified byte-for-byte in
`reports/direct_action/gate_d/run_001/artifacts/verify_negative_controls.py`.
"""

from __future__ import annotations

import torch


def build_conditions(correct: torch.Tensor, shuffle_seed: int, device) -> dict:
    """Build the five-condition dict {correct, zero, shuffled, reversed, wrong}.

    `correct` is the already-normalized robot_trajectory tensor, shape
    [..., window, action_dim] with the window dimension at index 1 (matching
    run_011's convention). `shuffle_seed` fixes the permutation used for the
    "shuffled" condition so results are reproducible across runs.
    """
    permutation = torch.randperm(
        correct.shape[1],
        generator=torch.Generator(device=device).manual_seed(shuffle_seed),
        device=device,
    )
    wrong = torch.roll(correct, shifts=correct.shape[1] // 2, dims=1).clone()
    return {
        "correct": correct,
        "zero": torch.zeros_like(correct),
        "shuffled": correct[:, permutation],
        "reversed": torch.flip(correct, dims=[1]),
        "wrong": wrong,
    }
