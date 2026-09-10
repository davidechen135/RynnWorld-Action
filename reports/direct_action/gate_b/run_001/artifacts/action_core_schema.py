"""
Gate B: D-action-core 33D channel-order lock.

This is the single source of truth for how the 33D D-action-core vector is
sliced out of the raw 40D `action` column and in what order. Every Gate B
loader/model consumer must import CORE_SLICES from here (or reproduce it
exactly and assert equality against this module) rather than re-deriving
indices ad hoc.

Source: reports/direct_action/gate_a/run_002/artifacts/schema_excerpt.json
        ("action_field_descriptions"), which gives the RAW index layout of
        the 40D action vector as stored in the task_3400 parquet:

    0       action/left_effector/position   (1D)  -- quarantined (D-action-effector)
    1       action/right_effector/position  (1D)  -- quarantined (D-action-effector)
    2:8     action/end/position             (6D)  -- D-action-core
    8:16    action/end/orientation          (8D)  -- D-action-core
    16:30   action/joint/position           (14D) -- D-action-core
    30:33   action/head/position            (3D)  -- excluded, state passthrough
    33:38   action/waist/position           (5D)  -- D-action-core
    38:40   action/robot/velocity           (2D)  -- not part of D-action-core

D-action-core is the concatenation of the four field slices IN RAW INDEX
ORDER as they appear in the source 40D vector -- i.e.
end/position, end/orientation, joint/position, waist/position -- NOT
joint-first. The Gate A contract (d_action_core_contract.json) lists the
four field names in a different prose order (joint, end/position,
end/orientation, waist), but that is a listing order, not a claim about
serialized tensor layout, and no prior run reordered the raw columns.
Do NOT describe or treat the Gate B 33D tensor as joint-first unless an
explicit reorder step is added AND tested (round-trip index assertion)
below. At present, no such reorder exists: raw slice order is preserved
as-is.
"""

CORE_SLICES = [
    ("action/end/position", slice(2, 8), 6),
    ("action/end/orientation", slice(8, 16), 8),
    ("action/joint/position", slice(16, 30), 14),
    ("action/waist/position", slice(33, 38), 5),
]

EXCLUDED_HEAD_PASSTHROUGH = ("action/head/position", slice(30, 33), 3)
QUARANTINED_EFFECTOR_SLICES = [
    ("action/left_effector/position", slice(0, 1), 1),
    ("action/right_effector/position", slice(1, 2), 1),
]
UNUSED_NON_CORE = ("action/robot/velocity", slice(38, 40), 2)

RAW_ACTION_DIM = 40
CORE_DIM = sum(dim for _, _, dim in CORE_SLICES)
assert CORE_DIM == 33, f"D-action-core must be 33D, got {CORE_DIM}"


def slice_core_33d(raw_action_40d):
    """Slice a [..., 40] raw action array/tensor down to [..., 33] D-action-core,
    concatenated in raw index order (end/position, end/orientation, joint/position,
    waist/position). Works for numpy arrays or torch tensors via duck typing on
    the last-axis concatenate.
    """
    assert raw_action_40d.shape[-1] == RAW_ACTION_DIM, (
        f"expected last dim {RAW_ACTION_DIM}, got {raw_action_40d.shape[-1]}"
    )
    parts = [raw_action_40d[..., sl] for _, sl, _ in CORE_SLICES]
    is_torch = hasattr(raw_action_40d, "cat") or type(raw_action_40d).__module__.startswith("torch")
    if is_torch:
        import torch
        out = torch.cat(parts, dim=-1)
    else:
        import numpy as np
        out = np.concatenate(parts, axis=-1)
    assert out.shape[-1] == CORE_DIM
    return out


def field_boundaries():
    """Return the [start, end) boundaries of each field within the sliced 33D
    output, in output order -- for downstream code that needs to address a
    specific field within the concatenated core tensor."""
    bounds = {}
    cursor = 0
    for name, _, dim in CORE_SLICES:
        bounds[name] = (cursor, cursor + dim)
        cursor += dim
    assert cursor == CORE_DIM
    return bounds


if __name__ == "__main__":
    import numpy as np

    boundaries = field_boundaries()
    print("D-action-core 33D output layout (raw index order, NOT joint-first):")
    for name, (start, end) in boundaries.items():
        print(f"  [{start:2d}:{end:2d}]  {name}")

    # Round-trip self-test: build a synthetic 40D vector with known per-field
    # marker values, slice it, and confirm each field lands where expected.
    raw = np.zeros((1, RAW_ACTION_DIM), dtype=np.float32)
    markers = {
        "action/end/position": 1.0,
        "action/end/orientation": 2.0,
        "action/joint/position": 3.0,
        "action/waist/position": 4.0,
        "action/head/position": 99.0,
        "action/left_effector/position": -1.0,
        "action/right_effector/position": -2.0,
        "action/robot/velocity": -3.0,
    }
    all_slices = CORE_SLICES + [EXCLUDED_HEAD_PASSTHROUGH] + QUARANTINED_EFFECTOR_SLICES + [UNUSED_NON_CORE]
    for name, sl, _ in all_slices:
        raw[:, sl] = markers[name]

    core = slice_core_33d(raw)
    assert core.shape == (1, 33)
    for name, (start, end) in boundaries.items():
        val = core[0, start:end]
        assert np.all(val == markers[name]), f"{name} landed wrong: {val}"
    assert 99.0 not in core, "head passthrough leaked into D-action-core"
    assert -1.0 not in core and -2.0 not in core, "quarantined effector leaked into D-action-core"
    assert -3.0 not in core, "unused robot/velocity leaked into D-action-core"
    print("SELF-TEST PASSED: 33D slice order verified, no head/effector/velocity leakage.")
