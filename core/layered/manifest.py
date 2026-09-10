"""layer_manifest — PLACEHOLDER (stage A).

The final schema is to be agreed with Lai / Ziyang / Jiahao (shared format).
Until then this writes a MINIMAL manifest recording only what stage-A ablation
needs: the 3 layer artifact paths, the interface used, and the z-order.

TODO(team-schema): align field names/versioning once the shared layer_manifest
spec lands. Do NOT depend on these field names downstream yet.
"""
import json
import os
from typing import Dict, List, Optional

from .layered_head import LAYER_NAMES, Z_ORDER_BACK_TO_FRONT

SCHEMA_VERSION = "0.0-placeholder"


def build_manifest(
    interface: str,
    layer_paths: List[Dict[str, Optional[str]]],
    recomposed_rgb_path: Optional[str] = None,
    source: Optional[Dict] = None,
) -> Dict:
    """Assemble the placeholder manifest dict.

    layer_paths: list of 3 dicts, each may carry {latent, rgb, alpha} paths.
    """
    assert len(layer_paths) == len(LAYER_NAMES), "expected 3 layers"
    layers = []
    for idx, name in enumerate(LAYER_NAMES):
        p = layer_paths[idx]
        layers.append({
            "name": name,
            "z_index": Z_ORDER_BACK_TO_FRONT.index(idx),  # 0=back .. 2=front
            "latent_path": p.get("latent"),
            "rgb_path": p.get("rgb"),
            "alpha_path": p.get("alpha"),
            "confidence": p.get("confidence"),
            "controllable": p.get("controllable"),
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "interface": interface,               # "concat" | "token"
        "layers": layers,
        "recomposition": {
            "method": "alpha_over",
            "order_back_to_front": Z_ORDER_BACK_TO_FRONT,
            "recomposed_rgb_path": recomposed_rgb_path,
        },
        "source": source or {},
    }


def write_manifest(path: str, **kwargs) -> Dict:
    manifest = build_manifest(**kwargs)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest
