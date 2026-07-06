"""Streaming distillation module for causal video generation.

Provides:
- WanCausalTransformer3DModel: Causal transformer with sliding KV cache
- DynamicCache: Sliding-window KV cache for streaming generation
"""
import diffusers

from .cache import DynamicCache
from .model import (
    WanCausalTransformer3DModel,
    WanCausalTransformerBlock,
    WanCausalAttnProcessor,
    WanRotaryPosEmbed,
    apply_monkey_patch,
)


apply_monkey_patch()


__all__ = [
    "DynamicCache",
    "WanCausalTransformer3DModel",
    "WanCausalTransformerBlock",
    "WanCausalAttnProcessor",
    "WanRotaryPosEmbed",
    "apply_monkey_patch",
]
