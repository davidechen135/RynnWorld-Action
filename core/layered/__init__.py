"""Three-layer output transformation (stage A: two minimal interfaces + ablation).

See docs/layered_interfaces.md. Original single-layer path is preserved in
core/streaming/model_singlelayer_baseline.py and remains the default (layer_mode="off").
"""
from .layered_head import (
    NUM_LAYERS,
    LAYER_NAMES,
    Z_ORDER_BACK_TO_FRONT,
    AlphaCompositor,
    LayeredConcatHead,
    LayeredTokenHead,
    unpatchify,
)
from .manifest import build_manifest, write_manifest, SCHEMA_VERSION

__all__ = [
    "NUM_LAYERS", "LAYER_NAMES", "Z_ORDER_BACK_TO_FRONT",
    "AlphaCompositor", "LayeredConcatHead", "LayeredTokenHead", "unpatchify",
    "build_manifest", "write_manifest", "SCHEMA_VERSION",
]
