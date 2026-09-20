# Adopted from: https://github.com/3DTopia/4DNeX/blob/main/finetune_wan.py
from core.finetune.models.utils import get_model_cls
import argparse
import datetime
import logging
from pathlib import Path
from typing import Any, List, Literal, Tuple

from pydantic import BaseModel, ValidationInfo, field_validator


class Args(BaseModel):
    prompt: str = ""

    ########## Model ##########
    model_path: Path
    model_name: str
    model_type: Literal["i2v", "t2v", "i2pm", "i2dpm", "wan-i2v", "egoverse",'egoverse22','rynnworld_teleop','rynnworld_teleop_pretrain']
    training_type: Literal["lora", "sft"] = "lora"

    ########## Output ##########
    output_dir: Path = Path("train_results/{:%Y-%m-%d-%H-%M-%S}".format(datetime.datetime.now()))
    report_to: Literal["tensorboard", "wandb", "all"] | None = None
    tracker_name: str = "finetrainer-cogvideo"

    ########## Training #########
    resume_from_checkpoint: Path | None = None
    init_from_checkpoint: Path | None = None

    seed: int | None = None
    train_epochs: int
    train_steps: int | None = None
    checkpointing_steps: int = 200
    checkpointing_limit: int = 10

    batch_size: int
    gradient_accumulation_steps: int = 1

    train_resolution: Tuple[int, int, int]  # shape: (frames, height, width)

    mixed_precision: Literal["no", "fp16", "bf16"]

    learning_rate: float = 2e-5
    optimizer: str = "adamw"
    beta1: float = 0.9
    beta2: float = 0.95
    beta3: float = 0.98
    epsilon: float = 1e-8
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0

    ema_decay: float = 0.999
    ema_start_step: int = 100

    lr_scheduler: str = "cosine_with_warmup"
    lr_warmup_steps: int = 100
    lr_num_cycles: int = 1
    lr_power: float = 1.0

    num_workers: int = 8
    pin_memory: bool = True

    gradient_checkpointing: bool = True
    enable_slicing: bool = True
    enable_tiling: bool = True
    nccl_timeout: int = 1800

    ########## Lora ##########
    rank: int = 128
    lora_alpha: int = 64
    target_modules: List[str] = ["to_q", "to_k", "to_v", "to_out.0"]
    is_concat: bool = True
    freeze_lora: bool = False
    control_type: str = "add"
    control_lr: float | None = None
    native_projection_lr: float | None = None
    # Warm-start the control path from a trained SFT checkpoint dir instead of zero-init.
    # Must be declared here: Args is a pydantic BaseModel built via cls(**vars(args)),
    # so an argparse flag with no matching field is silently dropped.
    control_init_from: str | None = None
    condition_mode: Literal["pose_video", "native_trajectory"] = "pose_video"
    native_adapter_init: str | None = None
    native_baseline_init: str | None = None
    native_reset_action_outputs: bool = False
    # Gate D 37D rot6d core != task_362's original 24D trajectory (see
    # NativeTrajectoryEncoder's default). None preserves existing behavior.
    native_trajectory_dim: int | None = None
    native_conditioner_version: Literal["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11"] = "v1"
    action_dropout_prob: float = 0.0
    action_contrastive_weight: float = 0.0
    action_contrastive_margin: float = 0.02
    action_ranking_weight: float = 0.0
    action_ranking_margin: float = 0.02
    action_motion_weight: float = 0.0
    action_local_motion_weight: float = 0.0
    action_local_motion_focus: float = 4.0
    action_spatial_gate_weight: float = 0.0

    ########## Regularization ##########
    reg_weight_init: float = 0.01
    reg_weight_decay_steps: int = 0

    ########## Validation ##########
    do_validation: bool = False
    validation_steps: int | None  # if set, should be a multiple of checkpointing_steps
    validation_dir: Path | None  # if set do_validation, should not be None
    cache_dir: Path | None
    validation_prompts: str | None  # if set do_validation, should not be None
    validation_images: str | None  # if set do_validation and model_type == i2v, should not be None
    validation_videos: str | None  # if set do_validation and model_type == v2v, should not be None
    gen_fps: int = 15

    @field_validator("validation_dir", "validation_prompts")
    def validate_validation_required_fields(cls, v: Any, info: ValidationInfo) -> Any:
        values = info.data
        if values.get("do_validation") and not v:
            field_name = info.field_name
            raise ValueError(f"{field_name} must be specified when do_validation is True")
        return v

    @field_validator("validation_images")
    def validate_validation_images(cls, v: str | None, info: ValidationInfo) -> str | None:
        values = info.data
        if values.get("do_validation") and values.get("model_type") in ["i2v", "i2pm", "i2dpm"] and not v:
            raise ValueError("validation_images must be specified when do_validation is True and model_type is i2v")
        return v

    @field_validator("validation_videos")
    def validate_validation_videos(cls, v: str | None, info: ValidationInfo) -> str | None:
        values = info.data
        if values.get("do_validation") and values.get("model_type") == "v2v" and not v:
            raise ValueError("validation_videos must be specified when do_validation is True and model_type is v2v")
        return v

    @field_validator("validation_steps")
    def validate_validation_steps(cls, v: int | None, info: ValidationInfo) -> int | None:
        values = info.data
        if values.get("do_validation"):
            if v is None:
                raise ValueError("validation_steps must be specified when do_validation is True")
            if values.get("checkpointing_steps") and v % values["checkpointing_steps"] != 0:
                raise ValueError("validation_steps must be a multiple of checkpointing_steps")
        return v

    @field_validator("train_resolution")
    def validate_train_resolution(cls, v: Tuple[int, int, int], info: ValidationInfo) -> str:
        try:
            frames, height, width = v

            # Check if (frames - 1) is multiple of 8
            if (frames - 1) % 8 != 0:
                raise ValueError("Number of frames - 1 must be a multiple of 8")

            # Check resolution for cogvideox-5b models
            model_name = info.data.get("model_name", "")
            if model_name in ["cogvideox-5b-i2v", "cogvideox-5b-t2v"]:
                if (height, width) != (480, 720):
                    raise ValueError("For cogvideox-5b models, height must be 480 and width must be 720")

            return v

        except ValueError as e:
            if (
                str(e) == "not enough values to unpack (expected 3, got 0)"
                or str(e) == "invalid literal for int() with base 10"
            ):
                raise ValueError("train_resolution must be in format 'frames x height x width'")
            raise e

    @field_validator("mixed_precision")
    def validate_mixed_precision(cls, v: str, info: ValidationInfo) -> str:
        if v == "fp16" and "cogvideox-2b" not in str(info.data.get("model_path", "")).lower():
            logging.warning(
                "All CogVideoX models except cogvideox-2b were trained with bfloat16. "
                "Using fp16 precision may lead to training instability."
            )
        return v

    @classmethod
    def parse_args(cls):
        """Parse command line arguments and return Args instance"""
        parser = argparse.ArgumentParser()
        # Required arguments
        parser.add_argument("--prompt", type=str, default="", required=False)
        parser.add_argument("--model_path", type=str, required=True)
        parser.add_argument("--model_name", type=str, required=True)
        parser.add_argument("--model_type", type=str, required=True)
        parser.add_argument("--training_type", type=str, required=True)
        parser.add_argument("--output_dir", type=str, required=True)
        parser.add_argument("--train_resolution", type=str, required=True)
        parser.add_argument("--report_to", type=str, required=True)

        # Training hyperparameters
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--train_epochs", type=int, default=10)
        parser.add_argument("--train_steps", type=int, default=None)
        parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
        parser.add_argument("--batch_size", type=int, default=1)
        parser.add_argument("--learning_rate", type=float, default=2e-5)
        parser.add_argument("--optimizer", type=str, default="adamw")
        parser.add_argument("--beta1", type=float, default=0.9)
        parser.add_argument("--beta2", type=float, default=0.95)
        parser.add_argument("--beta3", type=float, default=0.98)
        parser.add_argument("--epsilon", type=float, default=1e-8)
        parser.add_argument("--weight_decay", type=float, default=1e-4)
        parser.add_argument("--max_grad_norm", type=float, default=1.0)
        parser.add_argument("--ema_decay", type=float, default=0.999)
        parser.add_argument("--ema_start_step", type=int, default=100)

        # Learning rate scheduler
        parser.add_argument("--lr_scheduler", type=str, default="cosine_with_warmup")
        parser.add_argument("--lr_warmup_steps", type=int, default=500)
        parser.add_argument("--lr_num_cycles", type=int, default=1)
        parser.add_argument("--lr_power", type=float, default=1.0)

        # Data loading
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--pin_memory", type=bool, default=True)
        parser.add_argument("--image_column", type=str, default=None)

        # Model configuration
        parser.add_argument("--mixed_precision", type=str, default="no")
        parser.add_argument("--gradient_checkpointing", type=bool, default=True)
        parser.add_argument("--enable_slicing", type=bool, default=True)
        parser.add_argument("--enable_tiling", type=bool, default=True)
        parser.add_argument("--nccl_timeout", type=int, default=1800)

        # LoRA parameters
        parser.add_argument("--rank", type=int, default=128)
        parser.add_argument("--lora_alpha", type=int, default=64)
        parser.add_argument("--target_modules", type=str, nargs="+", default=["attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0","attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0"])

        # Checkpointing
        parser.add_argument("--checkpointing_steps", type=int, default=200)
        parser.add_argument("--checkpointing_limit", type=int, default=10)
        parser.add_argument("--resume_from_checkpoint", type=str, default=None)
        parser.add_argument("--init_from_checkpoint", type=str, default=None)

        # Validation
        parser.add_argument("--do_validation", type=lambda x: x.lower() == 'true', default=False)
        parser.add_argument("--validation_steps", type=int, default=None)
        parser.add_argument("--validation_dir", type=str, default=None)
        parser.add_argument("--cache_dir", type=str, default=None)
        parser.add_argument("--validation_prompts", type=str, default=None)
        parser.add_argument("--validation_images", type=str, default=None)
        parser.add_argument("--validation_videos", type=str, default=None)
        parser.add_argument("--gen_fps", type=int, default=15)


        parser.add_argument("--is_concat", type=lambda x: (str(x).lower() == 'true'), default=True)
        parser.add_argument("--control_type", type=str, default='add', help="Control type: add, add-plus, or concat")
        parser.add_argument("--freeze_lora", type=lambda x: (str(x).lower() == 'true'), default=False, help="Freeze LoRA weights during training")
        parser.add_argument("--control_lr", type=float, default=None, help="Learning rate for control_patch_embedding (default: same as learning_rate)")
        parser.add_argument(
            "--native_projection_lr",
            type=float,
            default=None,
            help="Optional smaller LR for native residual/AdaLN output projections.",
        )
        parser.add_argument("--control_init_from", type=str, default=None,
                            help="Directory holding control_patch_embedding.bin / control_scale.bin "
                                 "(e.g. pretrained/RynnWorld-Teleop) to warm-start the control path "
                                 "instead of zero-initializing it. Default None = original zero-init.")
        parser.add_argument("--condition_mode", choices=["pose_video", "native_trajectory"],
                            default="pose_video")
        parser.add_argument("--native_adapter_init", type=str, default=None)
        parser.add_argument("--native_baseline_init", type=str, default=None,
                            help="V1 native encoder checkpoint used only as a frozen V3 video baseline")
        parser.add_argument("--native_reset_action_outputs", action="store_true",
                            help="Reset native input-residual and AdaLN projections after loading "
                            "an adapter. Useful when a single-clip adapter would leak a memorized "
                            "action-to-video mapping into a multi-episode run.")
        parser.add_argument("--native_trajectory_dim", type=int, default=None,
                            help="Override NativeTrajectoryEncoder input_dim (default None = 24, "
                                 "the original task_362 trajectory dim). Gate D's 37D rot6d core "
                                 "requires --native_trajectory_dim 37.")
        parser.add_argument("--native_conditioner_version", choices=["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11"],
                            default="v1", help="v2 adds full-sequence encoding and "
                            "per-block action AdaLN modulation; v1 preserves historical behavior.")
        parser.add_argument("--action_dropout_prob", type=float, default=0.0,
                            help="Probability of dropping the complete native action condition. "
                            "Required for action classifier-free guidance; never drops time points independently.")
        parser.add_argument("--action_contrastive_weight", type=float, default=0.0)
        parser.add_argument("--action_contrastive_margin", type=float, default=0.02)
        parser.add_argument("--action_ranking_weight", type=float, default=0.0,
                            help="Weight for hinge loss requiring the correct action to denoise "
                                 "better than a temporally wrong action.")
        parser.add_argument("--action_ranking_margin", type=float, default=0.02,
                            help="Required MSE gap: wrong_action_mse >= correct_action_mse + margin.")
        parser.add_argument("--action_motion_weight", type=float, default=0.0,
                            help="Weight for V4 action-to-ground-truth latent-motion supervision.")
        parser.add_argument("--action_local_motion_weight", type=float, default=0.0,
                            help="Weight for direct x0 latent-motion reconstruction, focused on moving regions.")
        parser.add_argument("--action_local_motion_focus", type=float, default=4.0,
                            help="Additional weight assigned to high-motion latent locations.")
        parser.add_argument("--action_spatial_gate_weight", type=float, default=0.0,
                            help="Weight for supervising V9's action spatial gate with GT latent motion.")

        parser.add_argument("--reg_weight_init", type=float, default=0.01, help="Initial weight for regularization loss")
        parser.add_argument("--reg_weight_decay_steps", type=int, default=0, help="Steps to linearly decay reg_weight to 0; 0 means no decay")

        args = parser.parse_args()

        # Convert video_resolution_buckets string to list of tuples
        frames, height, width = args.train_resolution.split("x")
        args.train_resolution = (int(frames), int(height), int(width))

        return cls(**vars(args))


def main():
    args = Args.parse_args()
    trainer_cls = get_model_cls(args.model_name, args.training_type)
    trainer = trainer_cls(args)
    trainer.fit()


if __name__ == "__main__":
    import torch
    main()
