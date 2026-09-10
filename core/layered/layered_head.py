"""Three-layer output heads for the RynnWorld-Teleop DiT (UMI transformation, stage A).

Extends the single RGB denoising head (WanTransformer3DModel.proj_out, see
core/streaming/model.py:698-705) into THREE layered latent outputs:
    layer 0 = background / scene   (z-index back)
    layer 1 = object  / contact    (z-index mid)
    layer 2 = robot   / actor      (z-index front)

Two minimal interfaces (ticket item A), both selectable via `layer_mode`:
  * "concat" -> LayeredConcatHead : one wide Linear head emits 3x the RGB latent
                channels, split into 3 layer latents.
  * "token"  -> LayeredTokenHead  : a learned layer_embedding[3, inner_dim] is
                added to the token stream, 3 layer-conditioned passes share the
                ORIGINAL proj_out. (token fusion + layer embedding)

Both produce, per layer: a [B,48,F,H,W] latent + a per-layer alpha, and a
recomposed RGB-latent via alpha over-compositing in z-order robot>object>background.

Design constraints:
  * NO change to the single-layer path. These modules are only invoked when
    layer_mode != "off"; the frozen model_singlelayer_baseline.py stays canonical.
  * Reuses the exact unpatchify reshape from model.py:701-705 so a single layer's
    latent is byte-identical in shape to the original proj_out output.
"""
import math
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_LAYERS = 3
LAYER_NAMES = ["background_scene", "object_contact", "robot_actor"]
# z-order for alpha over-compositing: robot(front) painted last over object over background.
# indices into the layer list above, back-to-front.
Z_ORDER_BACK_TO_FRONT = [0, 1, 2]


def unpatchify(tokens, batch_size, ppf, pph, ppw, patch_size, out_channels):
    """Token sequence -> [B, out_channels, F, H, W].

    Byte-identical to core/streaming/model.py:701-705 for a single head, so a
    layered head's per-layer output matches the original head's tensor layout.
    """
    p_t, p_h, p_w = patch_size
    x = tokens.reshape(batch_size, ppf, pph, ppw, p_t, p_h, p_w, out_channels)
    x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)
    return x.flatten(6, 7).flatten(4, 5).flatten(2, 3)


class AlphaCompositor(nn.Module):
    """Per-layer alpha head + over-compositing recomposition.

    Shared by BOTH interfaces so the recomposition path (and thus any downstream
    metric) is identical regardless of which head produced the layer latents.

    alpha_i = Conv3d(48 -> 1) per layer, then softmax across the 3 layers so the
    alphas form a per-voxel partition of unity (guarantees the recomposition is a
    convex blend and cannot blow up the latent range).
    """

    def __init__(self, latent_channels: int = 48):
        super().__init__()
        self.alpha_head = nn.Conv3d(latent_channels, NUM_LAYERS, kernel_size=1)
        # small init so alphas start near-uniform (log-softmax ~ equal weight)
        nn.init.zeros_(self.alpha_head.weight)
        nn.init.zeros_(self.alpha_head.bias)

    def forward(self, layer_latents: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        # layer_latents: list of 3 x [B,48,F,H,W]
        stack = torch.stack(layer_latents, dim=1)                 # [B,3,48,F,H,W]
        b, n, c, f, h, w = stack.shape
        # alpha logits from each layer's own latent, then softmax across layers
        logits = torch.stack(
            [self.alpha_head(layer_latents[i]) for i in range(n)], dim=1
        )                                                         # [B,3,3,F,H,W]
        # collapse the 3-channel conv output to a single scalar per layer via mean
        logits = logits.mean(dim=2)                               # [B,3,F,H,W]
        alpha = F.softmax(logits, dim=1)                          # [B,3,F,H,W] partition of unity
        # Recomposition = convex sum Sum_i alpha_i * latent_i. Because the softmax
        # makes the alphas a per-voxel partition of unity, this convex blend is the
        # correct recomposition AND is order-independent: at warm start (all 3 layers
        # equal the single-layer latent L, alpha uniform 1/3) it returns exactly L.
        # NB: iterative "over"-compositing (recomposed*(1-a)+L*a) does NOT reduce to
        # this and would leave a ~0.7*L residual at init. The z-order (robot>object>
        # background) is carried in the manifest for RGB-space compositing downstream;
        # in latent space the convex sum is the faithful, order-free recomposition.
        recomposed = (alpha.unsqueeze(2) * stack).sum(dim=1)      # [B,48,F,H,W]
        return {"alpha": alpha, "recomposed": recomposed}


class LayeredConcatHead(nn.Module):
    """Interface A: channel-concat. One wide Linear emits 3x the RGB-latent
    channels; unpatchify each third into a [B,48,F,H,W] layer latent.

    Weight init copies the original proj_out into ALL three layer slots, so at
    step 0 every layer equals the single-layer prediction (safe warm start; the
    recomposition with near-uniform alpha then also equals the original output).
    """

    def __init__(self, inner_dim: int, out_channels: int, patch_size):
        super().__init__()
        self.out_channels = out_channels
        self.patch_size = tuple(patch_size)
        self.per_layer_dim = out_channels * math.prod(self.patch_size)
        self.proj_out_layered = nn.Linear(inner_dim, NUM_LAYERS * self.per_layer_dim)
        self.compositor = AlphaCompositor(out_channels)

    @torch.no_grad()
    def init_from_proj_out(self, proj_out: nn.Linear):
        """Warm start: replicate the single-layer head into each layer slot."""
        for i in range(NUM_LAYERS):
            s = i * self.per_layer_dim
            self.proj_out_layered.weight[s:s + self.per_layer_dim].copy_(proj_out.weight)
            if proj_out.bias is not None:
                self.proj_out_layered.bias[s:s + self.per_layer_dim].copy_(proj_out.bias)

    def forward(self, tokens, batch_size, ppf, pph, ppw) -> Dict[str, torch.Tensor]:
        # tokens: [B, seq, inner_dim] (post norm_out, pre projection)
        proj = self.proj_out_layered(tokens)                      # [B,seq,3*per_layer_dim]
        chunks = proj.chunk(NUM_LAYERS, dim=-1)                   # 3 x [B,seq,per_layer_dim]
        layer_latents = [
            unpatchify(c, batch_size, ppf, pph, ppw, self.patch_size, self.out_channels)
            for c in chunks
        ]                                                         # 3 x [B,48,F,H,W]
        comp = self.compositor(layer_latents)
        return {"layers": layer_latents, **comp}


class LayeredTokenHead(nn.Module):
    """Interface B: token fusion + layer embedding. A learned embedding[3,inner_dim]
    is added to the (already block-processed) token stream; each layer's tokens go
    through the SHARED original proj_out. Cheaper params than A (no 3x head), at the
    cost of 3 proj_out passes.

    `proj_out` is passed in at call time (the model's own head) so this interface
    literally shares the single-layer projection weights.
    """

    def __init__(self, inner_dim: int, out_channels: int, patch_size):
        super().__init__()
        self.out_channels = out_channels
        self.patch_size = tuple(patch_size)
        self.layer_embedding = nn.Embedding(NUM_LAYERS, inner_dim)
        nn.init.zeros_(self.layer_embedding.weight)  # start == single-layer for all 3
        self.compositor = AlphaCompositor(out_channels)

    def forward(self, tokens, proj_out: nn.Linear, batch_size, ppf, pph, ppw) -> Dict[str, torch.Tensor]:
        layer_latents = []
        for i in range(NUM_LAYERS):
            emb = self.layer_embedding.weight[i].view(1, 1, -1)   # [1,1,inner_dim]
            proj = proj_out(tokens + emb)                         # [B,seq,per_layer_dim]
            layer_latents.append(
                unpatchify(proj, batch_size, ppf, pph, ppw, self.patch_size, self.out_channels)
            )
        comp = self.compositor(layer_latents)
        return {"layers": layer_latents, **comp}
