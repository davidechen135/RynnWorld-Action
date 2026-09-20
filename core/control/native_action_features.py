"""Feature construction for the 37D AgiBot rot6D action representation."""

from __future__ import annotations

import torch
import torch.nn.functional as F


ROTATION_SLICES = (slice(6, 12), slice(12, 18))


def rot6d_rows_to_matrix(value: torch.Tensor) -> torch.Tensor:
    """Convert the first two matrix rows to a valid rotation matrix."""
    first = F.normalize(value[..., :3], dim=-1, eps=1e-8)
    second = value[..., 3:6]
    second = F.normalize(
        second - (first * second).sum(dim=-1, keepdim=True) * first,
        dim=-1,
        eps=1e-8,
    )
    third = torch.linalg.cross(first, second, dim=-1)
    return torch.stack((first, second, third), dim=-2)


def matrix_to_rot6d_rows(value: torch.Tensor) -> torch.Tensor:
    return value[..., :2, :].flatten(-2)


def relative_and_velocity(
    raw_action: torch.Tensor,
    reference: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return zero-at-rest displacement and step motion in the original 37D layout.

    ``reference`` is the observed robot state at the first video frame.  When it
    is absent, the first command is retained as the backward-compatible V8
    reference.
    """
    if raw_action.shape[-1] != 37:
        raise ValueError(f"expected raw rot6D37 action, got {tuple(raw_action.shape)}")

    if reference is None:
        reference = raw_action[..., :1, :]
    else:
        if reference.shape[-1] != 37:
            raise ValueError(f"expected rot6D37 reference, got {tuple(reference.shape)}")
        if reference.ndim == raw_action.ndim - 1:
            reference = reference.unsqueeze(-2)
        if reference.shape[-2] != 1:
            raise ValueError(f"reference must have one time step, got {tuple(reference.shape)}")
    relative = raw_action - reference
    velocity = torch.zeros_like(raw_action)
    velocity[..., 1:, :] = raw_action[..., 1:, :] - raw_action[..., :-1, :]

    identity6 = raw_action.new_tensor((1.0, 0.0, 0.0, 0.0, 1.0, 0.0))
    for field in ROTATION_SLICES:
        rotation = rot6d_rows_to_matrix(raw_action[..., field])
        reference_rotation = rot6d_rows_to_matrix(reference[..., field])
        relative_rotation = rotation @ reference_rotation.transpose(-1, -2)
        relative[..., field] = matrix_to_rot6d_rows(relative_rotation) - identity6

        step_rotation = rotation[..., 1:, :, :] @ rotation[..., :-1, :, :].transpose(-1, -2)
        velocity[..., 0, field] = 0
        velocity[..., 1:, field] = matrix_to_rot6d_rows(step_rotation) - identity6
    return relative, velocity


def build_v8_features(
    raw_action: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    relative_scale: torch.Tensor,
    velocity_scale: torch.Tensor,
) -> torch.Tensor:
    """Build target/reference, relative-motion, and velocity features."""
    relative, velocity = relative_and_velocity(raw_action)
    target = (raw_action - target_mean) / target_std.clamp_min(1e-6)
    relative = relative / relative_scale.clamp_min(1e-6)
    velocity = velocity / velocity_scale.clamp_min(1e-6)
    return torch.cat((target, relative, velocity), dim=-1)


def build_v10_features(
    raw_action: torch.Tensor,
    observed_state: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    relative_scale: torch.Tensor,
    velocity_scale: torch.Tensor,
) -> torch.Tensor:
    """Build ``state_t + future action`` features without future-state leakage."""
    if observed_state.ndim == raw_action.ndim - 1:
        observed_state = observed_state.unsqueeze(-2)
    relative, velocity = relative_and_velocity(raw_action, observed_state)
    state = (observed_state - target_mean) / target_std.clamp_min(1e-6)
    state = state.expand(*raw_action.shape[:-1], 37)
    target = (raw_action - target_mean) / target_std.clamp_min(1e-6)
    relative = relative / relative_scale.clamp_min(1e-6)
    velocity = velocity / velocity_scale.clamp_min(1e-6)
    return torch.cat((state, target, relative, velocity), dim=-1)


def build_spatial_action_control(
    trajectory_uv: torch.Tensor,
    reference_uv: torch.Tensor,
    output_frames: int = 9,
    height: int = 15,
    width: int = 26,
    sigma: float = 1.25,
) -> torch.Tensor:
    """Rasterize bimanual target points and displacement vectors on the DiT grid.

    Returns six channels: left/right heatmaps followed by heatmap-weighted
    ``dx,dy`` for each hand. Pixel coordinates must already be expressed on the
    requested output grid.
    """
    squeeze = trajectory_uv.ndim == 3
    if squeeze:
        trajectory_uv = trajectory_uv.unsqueeze(0)
    if reference_uv.ndim == 2:
        reference_uv = reference_uv.unsqueeze(0)
    if trajectory_uv.ndim != 4 or trajectory_uv.shape[-2:] != (2, 2):
        raise ValueError(f"expected trajectory_uv [B,T,2,2], got {trajectory_uv.shape}")
    if reference_uv.shape[-2:] != (2, 2):
        raise ValueError(f"expected reference_uv [B,2,2], got {reference_uv.shape}")

    yy, xx = torch.meshgrid(
        torch.arange(height, device=trajectory_uv.device, dtype=trajectory_uv.dtype),
        torch.arange(width, device=trajectory_uv.device, dtype=trajectory_uv.dtype),
        indexing="ij",
    )
    center_x = trajectory_uv[..., 0, None, None]
    center_y = trajectory_uv[..., 1, None, None]
    heat = torch.exp(
        -((xx - center_x).square() + (yy - center_y).square()) / (2.0 * sigma**2)
    )
    valid = (
        torch.isfinite(center_x)
        & torch.isfinite(center_y)
        & (center_x >= 0)
        & (center_x <= width - 1)
        & (center_y >= 0)
        & (center_y <= height - 1)
    )
    heat = torch.where(valid, heat, torch.zeros_like(heat))
    displacement = trajectory_uv - reference_uv[:, None]
    dx = displacement[..., 0] / max(width - 1, 1)
    dy = displacement[..., 1] / max(height - 1, 1)
    channels = torch.stack(
        (
            heat[:, :, 0],
            heat[:, :, 1],
            heat[:, :, 0] * dx[:, :, 0, None, None],
            heat[:, :, 0] * dy[:, :, 0, None, None],
            heat[:, :, 1] * dx[:, :, 1, None, None],
            heat[:, :, 1] * dy[:, :, 1, None, None],
        ),
        dim=1,
    )
    if channels.shape[2] == 1 + 4 * (output_frames - 1):
        first = channels[:, :, :1]
        rest = channels[:, :, 1:].reshape(
            channels.shape[0], 6, output_frames - 1, 4, height, width
        ).mean(dim=3)
        channels = torch.cat((first, rest), dim=2)
    elif channels.shape[2] != output_frames:
        channels = F.adaptive_avg_pool3d(channels, (output_frames, height, width))
    return channels[0] if squeeze else channels
