"""Temporal encoder for normalized AgiBot native robot trajectories."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NativeTrajectoryEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 24,
        hidden_dim: int = 768,
        output_dim: int = 3072,
        num_layers: int = 2,
        num_heads: int = 12,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.SiLU(),
            nn.Linear(256, hidden_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, output_dim)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, trajectory: torch.Tensor, output_frames: int) -> torch.Tensor:
        if trajectory.ndim != 3 or trajectory.shape[-1] != self.input_dim:
            raise ValueError(
                f"expected trajectory [B,T,{self.input_dim}], got {trajectory.shape}"
            )
        if trajectory.shape[1] != output_frames:
            trajectory = F.interpolate(
                trajectory.transpose(1, 2),
                size=output_frames,
                mode="linear",
                align_corners=True,
            ).transpose(1, 2)
        trajectory = trajectory.to(dtype=self.input_norm.weight.dtype)
        hidden = self.input_projection(self.input_norm(trajectory))
        hidden = self.temporal_transformer(hidden)
        hidden = self.output_projection(self.output_norm(hidden))
        return hidden.transpose(1, 2).unsqueeze(-1).unsqueeze(-1)


class NativeTrajectoryConditionerV2(nn.Module):
    """Action conditioner that cannot solve the task with a constant bias.

    Unlike :class:`NativeTrajectoryEncoder`, this module encodes the complete
    action sequence before temporal compression.  It returns both an input
    residual and per-block AdaLN modulation for Wan.  Both outputs are
    explicitly centered against a null (all-zero, i.e. dataset-mean) action,
    so a null action produces exactly zero conditioning even though the
    temporal transformer itself contains biases.
    """

    def __init__(
        self,
        input_dim: int = 33,
        hidden_dim: int = 768,
        output_dim: int = 3072,
        num_layers: int = 2,
        num_heads: int = 12,
        max_frames: int = 256,
        use_input_residual: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_input_residual = use_input_residual

        # Dataset-level per-dimension z-scoring happens before this module.
        # Do not apply LayerNorm across physical action dimensions here: doing
        # so erases action magnitude independently at every time step.
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim * 2, 256, bias=False),
            nn.SiLU(),
            nn.Linear(256, hidden_dim, bias=False),
        )
        self.position_embedding = nn.Parameter(
            torch.zeros(1, max_frames, hidden_dim)
        )
        nn.init.normal_(self.position_embedding, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.input_residual_projection = nn.Linear(
            hidden_dim, output_dim, bias=False
        )
        self.adaln_projection = nn.Linear(
            hidden_dim, output_dim * 6, bias=False
        )
        nn.init.zeros_(self.input_residual_projection.weight)
        nn.init.zeros_(self.adaln_projection.weight)

    def _position(self, frames: int, dtype: torch.dtype) -> torch.Tensor:
        pos = self.position_embedding
        if frames != pos.shape[1]:
            pos = F.interpolate(
                pos.transpose(1, 2), size=frames, mode="linear", align_corners=True
            ).transpose(1, 2)
        return pos.to(dtype=dtype)

    def _encode(self, trajectory: torch.Tensor, output_frames: int) -> torch.Tensor:
        delta = torch.zeros_like(trajectory)
        delta[:, 1:] = trajectory[:, 1:] - trajectory[:, :-1]
        hidden = self.input_projection(torch.cat([trajectory, delta], dim=-1))
        hidden = hidden + self._position(hidden.shape[1], hidden.dtype)
        hidden = self.temporal_transformer(hidden)
        hidden = self.output_norm(hidden)
        # Pool encoded action intervals instead of subsampling/interpolating the
        # raw trajectory. Every source action therefore contributes to a latent
        # time token.
        if hidden.shape[1] != output_frames:
            hidden = F.adaptive_avg_pool1d(
                hidden.transpose(1, 2), output_frames
            ).transpose(1, 2)
        return hidden

    def forward(
        self, trajectory: torch.Tensor, output_frames: int
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        if trajectory.ndim != 3 or trajectory.shape[-1] != self.input_dim:
            raise ValueError(
                f"expected trajectory [B,T,{self.input_dim}], got {trajectory.shape}"
            )
        trajectory = trajectory.to(dtype=self.position_embedding.dtype)
        centered = self.encode_centered(trajectory, output_frames)

        residual = None
        if self.use_input_residual:
            residual = self.input_residual_projection(centered)
            residual = residual.transpose(1, 2).unsqueeze(-1).unsqueeze(-1)
        modulation = self.adaln_projection(centered)
        modulation = modulation.view(
            trajectory.shape[0], output_frames, 6, self.output_dim
        )
        return residual, modulation

    def encode_centered(
        self, trajectory: torch.Tensor, output_frames: int
    ) -> torch.Tensor:
        """Return zero-referenced per-frame action features for auxiliary losses."""
        trajectory = trajectory.to(dtype=self.position_embedding.dtype)
        encoded = self._encode(trajectory, output_frames)
        null_encoded = self._encode(torch.zeros_like(trajectory), output_frames)
        return encoded - null_encoded


class NativeTrajectoryConditionerV3(NativeTrajectoryConditionerV2):
    """Separate video-domain adaptation from zero-centered action control."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.base_residual = nn.Parameter(
            torch.zeros(1, self.output_dim, 1, 1, 1)
        )
        self.base_modulation = nn.Parameter(
            torch.zeros(1, 1, 6, self.output_dim)
        )

    def forward(
        self, trajectory: torch.Tensor, output_frames: int
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        action_residual, action_modulation = super().forward(
            trajectory, output_frames
        )
        residual = (
            None if action_residual is None
            else action_residual + self.base_residual
        )
        modulation = action_modulation + self.base_modulation.expand(
            trajectory.shape[0], output_frames, -1, -1
        )
        return residual, modulation


class NativeTrajectoryConditionerV4(NativeTrajectoryConditionerV3):
    """V3 plus an auxiliary action-to-video-motion prediction head.

    The head is used only while training.  It forces the shared temporal action
    representation to retain frame alignment without rewarding the denoiser
    for making counterfactual videos arbitrarily bad.
    """

    def __init__(self, *args, motion_dim: int = 48 * 4 * 4, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.motion_projection = nn.Linear(768, motion_dim, bias=False)
        nn.init.zeros_(self.motion_projection.weight)

    def predict_video_motion(
        self, trajectory: torch.Tensor, output_frames: int
    ) -> torch.Tensor:
        return self.motion_projection(
            self.encode_centered(trajectory, output_frames)
        )


class NativeTrajectoryConditionerV5(NativeTrajectoryConditionerV3):
    """Strictly local temporal conditioner that cannot pool away ordering.

    Each latent-frame token is computed only from the matching action window
    and its velocity.  The inherited transformer and position parameters stay
    in the state dict for V3 checkpoint compatibility but are intentionally
    bypassed.
    """

    # 0.1 preserves strong temporal separation but visibly distorts the held-out
    # correct-action rollout.  Start conservatively; training can adapt the
    # shared projections without sacrificing the frozen video baseline.
    local_action_scale: float = 0.03

    def _encode(self, trajectory: torch.Tensor, output_frames: int) -> torch.Tensor:
        delta = torch.zeros_like(trajectory)
        delta[:, 1:] = trajectory[:, 1:] - trajectory[:, :-1]
        local_input = torch.cat([trajectory, delta], dim=-1)
        if local_input.shape[1] != output_frames:
            local_input = F.adaptive_avg_pool1d(
                local_input.transpose(1, 2), output_frames
            ).transpose(1, 2)
        return self.output_norm(self.input_projection(local_input))

    def encode_centered(
        self, trajectory: torch.Tensor, output_frames: int
    ) -> torch.Tensor:
        # The inherited V3 output projections were trained on much smaller
        # global-transformer features.  Calibrate local features to the same
        # injection RMS before any fine-tuning.
        return self.local_action_scale * super().encode_centered(
            trajectory, output_frames
        )


class NativeTrajectoryConditionerV6(NativeTrajectoryConditionerV5):
    """V5 with causal 4x action-to-Wan-VAE temporal alignment."""

    def _encode(self, trajectory: torch.Tensor, output_frames: int) -> torch.Tensor:
        delta = torch.zeros_like(trajectory)
        delta[:, 1:] = trajectory[:, 1:] - trajectory[:, :-1]
        local_input = torch.cat([trajectory, delta], dim=-1)

        # Wan VAE maps T video frames to 1 + (T - 1) / 4 latent frames:
        # latent 0 is the fixed first frame, then each token covers the next
        # four causal frames.  Preserve that boundary instead of adaptive
        # pooling, which shifts action windows by up to half a frame group.
        if local_input.shape[1] == 1 + 4 * (output_frames - 1):
            first = local_input[:, :1]
            rest = local_input[:, 1:].reshape(
                local_input.shape[0], output_frames - 1, 4, local_input.shape[-1]
            ).mean(dim=2)
            local_input = torch.cat([first, rest], dim=1)
        elif local_input.shape[1] != output_frames:
            local_input = F.adaptive_avg_pool1d(
                local_input.transpose(1, 2), output_frames
            ).transpose(1, 2)
        return self.output_norm(self.input_projection(local_input))
