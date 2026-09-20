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


class NativeTrajectoryConditionerV7(NativeTrajectoryConditionerV6):
    """V6 alignment with magnitude-preserving normalized local features.

    V6 applies LayerNorm directly to the local MLP output.  LayerNorm makes
    0.5x, 1x, and 2x trajectories nearly equal in RMS, which removes a useful
    control variable before the residual and AdaLN projections.  V7 keeps the
    normalized feature direction but restores the per-token RMS of the
    z-scored action-and-velocity input.  The action-dependent component is
    exactly zero for a zero action; V3's frozen video-domain base residual is
    still added by ``forward``.

    This class adds no parameters, so V6 checkpoints load strictly into V7.
    """

    def _encode(self, trajectory: torch.Tensor, output_frames: int) -> torch.Tensor:
        delta = torch.zeros_like(trajectory)
        delta[:, 1:] = trajectory[:, 1:] - trajectory[:, :-1]
        local_input = torch.cat([trajectory, delta], dim=-1)

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

        direction = self.output_norm(self.input_projection(local_input))
        magnitude = local_input.float().square().mean(dim=-1, keepdim=True).sqrt()
        return direction * magnitude.to(dtype=direction.dtype)


class NativeTrajectoryConditionerV8(NativeTrajectoryConditionerV7):
    """Factorized target/reference and motion encoding with learned causal pooling.

    The input layout is ``[target37, relative37, velocity37]``.  Target values
    retain the task-wise z-score used by earlier versions. Relative and velocity
    features are divided by fixed training-set RMS values during preprocessing,
    so a held pose remains exactly zero in both motion branches.

    The inherited V7 modules remain in the state dict for checkpoint
    compatibility but its 74D input projection is bypassed.
    """

    component_dim: int = 37

    def __init__(self, input_dim: int = 111, *args, **kwargs) -> None:
        if input_dim != self.component_dim * 3:
            raise ValueError(
                f"V8 expects {self.component_dim * 3} features, got {input_dim}"
            )
        super().__init__(input_dim=self.component_dim, *args, **kwargs)
        self.input_dim = input_dim
        hidden_dim = self.output_norm.normalized_shape[0]

        def branch(source_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(source_dim, 256, bias=False),
                nn.SiLU(),
                nn.Linear(256, 256, bias=False),
            )

        self.reference_projection = branch(self.component_dim)
        self.target_projection = branch(self.component_dim)
        self.relative_projection = branch(self.component_dim)
        self.velocity_projection = branch(self.component_dim)
        self.feature_fusion = nn.Linear(256 * 4, hidden_dim, bias=False)
        self.temporal_compressor = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=4,
            stride=4,
            groups=hidden_dim,
            bias=False,
        )
        nn.init.constant_(self.temporal_compressor.weight, 0.25)

    def _encode(self, trajectory: torch.Tensor, output_frames: int) -> torch.Tensor:
        target, relative, velocity = trajectory.split(self.component_dim, dim=-1)
        reference = target[:, :1].expand_as(target)
        hidden = self.feature_fusion(
            torch.cat(
                (
                    self.reference_projection(reference),
                    self.target_projection(target),
                    self.relative_projection(relative),
                    self.velocity_projection(velocity),
                ),
                dim=-1,
            )
        )

        if hidden.shape[1] == 1 + 4 * (output_frames - 1):
            first = hidden[:, :1]
            rest = self.temporal_compressor(hidden[:, 1:].transpose(1, 2)).transpose(1, 2)
            hidden = torch.cat((first, rest), dim=1)
        elif hidden.shape[1] != output_frames:
            hidden = F.adaptive_avg_pool1d(
                hidden.transpose(1, 2), output_frames
            ).transpose(1, 2)

        direction = self.output_norm(hidden)
        motion_rms = torch.cat((relative, velocity), dim=-1)
        if motion_rms.shape[1] != output_frames:
            if motion_rms.shape[1] == 1 + 4 * (output_frames - 1):
                first = motion_rms[:, :1]
                rest = motion_rms[:, 1:].reshape(
                    motion_rms.shape[0], output_frames - 1, 4, motion_rms.shape[-1]
                ).square().mean(dim=2).sqrt()
                motion_rms = torch.cat((first, rest), dim=1)
            else:
                motion_rms = F.adaptive_avg_pool1d(
                    motion_rms.square().transpose(1, 2), output_frames).transpose(1, 2).sqrt()
        magnitude = (1.0 + motion_rms.float().square().mean(dim=-1, keepdim=True)).sqrt()
        return direction * magnitude.to(direction.dtype)


class NativeTrajectoryConditionerV9(NativeTrajectoryConditionerV8):
    """V8 with action-dependent spatial gating from the first-frame visual tokens."""

    spatial_gate_dim: int = 64

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        hidden_dim = self.output_norm.normalized_shape[0]
        self.visual_query_projection = nn.Conv3d(
            self.output_dim, self.spatial_gate_dim, kernel_size=1, bias=False
        )
        self.action_key_projection = nn.Linear(
            hidden_dim, self.spatial_gate_dim, bias=False
        )
        self.spatial_gate_bias = nn.Parameter(torch.zeros(()))
        # A zero action key makes 2*sigmoid(score) exactly one at initialization,
        # so loading a V8 checkpoint preserves its rollout before V9 training.
        nn.init.zeros_(self.action_key_projection.weight)
        self.last_spatial_gate: torch.Tensor | None = None

    def spatialize_residual(
        self,
        trajectory: torch.Tensor,
        output_frames: int,
        residual: torch.Tensor,
        visual_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if visual_tokens.ndim != 5:
            raise ValueError(f"expected visual tokens [B,C,T,H,W], got {visual_tokens.shape}")
        centered = self.encode_centered(trajectory, output_frames)
        action_key = self.action_key_projection(
            centered.to(dtype=self.action_key_projection.weight.dtype)
        ).to(visual_tokens.dtype)
        visual_query = self.visual_query_projection(visual_tokens[:, :, :1]).squeeze(2)
        score = torch.einsum("bchw,btc->bthw", visual_query, action_key)
        score = score / self.spatial_gate_dim**0.5 + self.spatial_gate_bias
        gate = 2.0 * torch.sigmoid(score)
        self.last_spatial_gate = gate.unsqueeze(1)
        return residual * self.last_spatial_gate.to(residual.dtype)


class NativeTrajectoryConditionerV10(NativeTrajectoryConditionerV8):
    """Observed-state action encoder plus an explicit spatial control branch.

    The trajectory layout is ``[state37, target37, relative37, velocity37]``.
    A separate six-channel bimanual point/vector map is projected into DiT
    tokens and injected at several depths.  The final spatial projection is
    zero-initialized, preserving the V8 rollout exactly at initialization.
    """

    spatial_input_channels: int = 6
    spatial_block_indices: tuple[int, ...] = (0, 10, 20, 30)

    def __init__(self, input_dim: int = 148, *args, **kwargs) -> None:
        if input_dim != self.component_dim * 4:
            raise ValueError(
                f"V10 expects {self.component_dim * 4} features, got {input_dim}"
            )
        super().__init__(input_dim=self.component_dim * 3, *args, **kwargs)
        self.input_dim = input_dim
        self.spatial_stem = nn.Sequential(
            nn.Conv3d(self.spatial_input_channels, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv3d(64, self.output_dim, kernel_size=1, bias=False),
        )
        nn.init.zeros_(self.spatial_stem[-1].weight)
        self.spatial_block_scales = nn.Parameter(
            torch.ones(len(self.spatial_block_indices))
        )

    def _encode(self, trajectory: torch.Tensor, output_frames: int) -> torch.Tensor:
        state, target, relative, velocity = trajectory.split(self.component_dim, dim=-1)
        hidden = self.feature_fusion(
            torch.cat(
                (
                    self.reference_projection(state),
                    self.target_projection(target),
                    self.relative_projection(relative),
                    self.velocity_projection(velocity),
                ),
                dim=-1,
            )
        )

        if hidden.shape[1] == 1 + 4 * (output_frames - 1):
            first = hidden[:, :1]
            rest = self.temporal_compressor(hidden[:, 1:].transpose(1, 2)).transpose(1, 2)
            hidden = torch.cat((first, rest), dim=1)
        elif hidden.shape[1] != output_frames:
            hidden = F.adaptive_avg_pool1d(
                hidden.transpose(1, 2), output_frames
            ).transpose(1, 2)

        direction = self.output_norm(hidden)
        motion_rms = torch.cat((relative, velocity), dim=-1)
        if motion_rms.shape[1] != output_frames:
            if motion_rms.shape[1] == 1 + 4 * (output_frames - 1):
                first = motion_rms[:, :1]
                rest = motion_rms[:, 1:].reshape(
                    motion_rms.shape[0], output_frames - 1, 4, motion_rms.shape[-1]
                ).square().mean(dim=2).sqrt()
                motion_rms = torch.cat((first, rest), dim=1)
            else:
                motion_rms = F.adaptive_avg_pool1d(
                    motion_rms.square().transpose(1, 2), output_frames
                ).transpose(1, 2).sqrt()
        magnitude = (1.0 + motion_rms.float().square().mean(dim=-1, keepdim=True)).sqrt()
        return direction * magnitude.to(direction.dtype)

    def encode_spatial_control(
        self,
        spatial_control: torch.Tensor,
        output_frames: int,
        output_height: int,
        output_width: int,
    ) -> torch.Tensor:
        if spatial_control.ndim != 5 or spatial_control.shape[1] != self.spatial_input_channels:
            raise ValueError(
                "expected spatial control [B,6,T,H,W], got "
                f"{tuple(spatial_control.shape)}"
            )
        spatial_control = spatial_control.to(dtype=self.spatial_stem[0].weight.dtype)
        if spatial_control.shape[2:] != (output_frames, output_height, output_width):
            spatial_control = F.interpolate(
                spatial_control,
                size=(output_frames, output_height, output_width),
                mode="trilinear",
                align_corners=False,
            )
        residual = self.spatial_stem(spatial_control)
        return residual.flatten(2).transpose(1, 2)


class NativeTrajectoryConditionerV11(NativeTrajectoryConditionerV10):
    """V10 plus explicit per-latent-frame action/video motion alignment.

    Diffusion reconstruction alone can learn pose and motion magnitude while
    ignoring temporal order.  This auxiliary head predicts a coarse 4x4 map
    of the next-frame latent change from every causal action token.  The head
    is used only by the training loss and adds no inference-time dependency.
    """

    def __init__(self, *args, motion_dim: int = 48 * 4 * 4, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        hidden_dim = self.output_norm.normalized_shape[0]
        self.motion_projection = nn.Linear(hidden_dim, motion_dim, bias=False)
        nn.init.zeros_(self.motion_projection.weight)

    def predict_video_motion(
        self, trajectory: torch.Tensor, output_frames: int
    ) -> torch.Tensor:
        return self.motion_projection(
            self.encode_centered(trajectory, output_frames)
        )
