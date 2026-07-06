#!/usr/bin/env python3
"""
load_config.py — Read a YAML config and emit shell `export VAR=value` lines.

Usage (from a shell script):
    eval "$(python scripts/load_config.py config.yaml)"

This sets env vars from the YAML, but ONLY when not already set in the
shell (i.e. command-line env vars take precedence). Precedence order:

    command-line env vars > config.yaml > script defaults

The script intentionally does nothing if the config file is missing
(allows running with no config when all env vars are set inline).
"""
import os
import sys


def emit(var: str, value) -> None:
    """Emit `export VAR=value` only if VAR is not already set."""
    if os.environ.get(var):
        return
    if value is None:
        return
    # Bash-safe single-quote escaping
    safe = str(value).replace("'", "'\\''")
    print(f"export {var}='{safe}'")


def emit_alias_pair(var_a: str, var_b: str, value) -> None:
    """Emit two env vars that should hold the same value (e.g. DATA_PATH/DATA_JSON).

    Resolution order:
      1. If either is already set in env, both get that value.
      2. Otherwise, both get `value` from the config.
    """
    a_val = os.environ.get(var_a)
    b_val = os.environ.get(var_b)
    if a_val and not b_val:
        # Bash-safe single-quote escaping
        safe = a_val.replace("'", "'\\''")
        print(f"export {var_b}='{safe}'")
    elif b_val and not a_val:
        safe = b_val.replace("'", "'\\''")
        print(f"export {var_a}='{safe}'")
    elif not a_val and not b_val and value is not None:
        safe = str(value).replace("'", "'\\''")
        print(f"export {var_a}='{safe}'")
        print(f"export {var_b}='{safe}'")
    # else: both already set — do nothing


def main() -> int:
    if len(sys.argv) < 2:
        print("# load_config.py: no config path", file=sys.stderr)
        return 0

    config_path = sys.argv[1]
    if not os.path.isfile(config_path):
        print(f"# load_config.py: config not found at {config_path}", file=sys.stderr)
        return 0

    try:
        import yaml
    except ImportError:
        print("# load_config.py: PyYAML not installed; skipping config", file=sys.stderr)
        return 0

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    paths = cfg.get("paths") or {}
    emit_alias_pair("DATA_PATH", "DATA_JSON", paths.get("data"))
    emit_alias_pair("SRC_MODEL_PATH", "MODEL_PATH", paths.get("model"))
    emit("TEACHER_CKPT", paths.get("teacher_ckpt"))
    emit("NULL_PROMPT_PATH", paths.get("null_prompt"))
    emit("LOCAL_MODEL_CACHE", paths.get("local_model_cache"))
    emit("TEACHER_PREFETCH_DST_BASE", paths.get("local_teacher_cache"))

    run = cfg.get("run") or {}
    emit("RUN_NAME", run.get("name"))
    emit("OUTPUT_DIR_BASE", run.get("output_base"))

    multinode = cfg.get("multinode") or {}
    emit("WORLD_SIZE", multinode.get("world_size"))
    emit("RANK", multinode.get("rank"))
    emit("MASTER_ADDR", multinode.get("master_addr"))
    emit("MASTER_PORT", multinode.get("master_port"))
    emit("NPROC_PER_NODE", multinode.get("nproc_per_node"))

    print(f"# load_config.py: loaded {config_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
