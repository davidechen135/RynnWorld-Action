from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from diffusers import (
    AutoencoderKLWan,              
    UniPCMultistepScheduler,
    WanImageToVideoPipeline,    
    WanTransformer3DModel,
)

from diffusers.models.embeddings import get_3d_rotary_pos_embed
from PIL import Image
import numpy as np
from numpy import dtype
from transformers import T5TokenizerFast, UMT5EncoderModel
from typing_extensions import override
from transformers import AutoConfig

from core.finetune.schemas import Wan_Components as Components
from core.finetune.trainer import Trainer
from core.finetune.models.wan_i2v.wan_trainer import WanI2VTrainer
from core.finetune.utils import unwrap_model
from core.finetune.models.wan_i2v.sft_trainer import generate_uniform_pointmap, retrieve_latents

from ..utils import register
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.optimization import get_scheduler
from diffusers.configuration_utils import register_to_config
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import get_1d_rotary_pos_embed
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.normalization import FP32LayerNorm
from diffusers.models.modeling_utils import ModelMixin
import json
from core.finetune.schemas import Args, Components, State
from accelerate.accelerator import Accelerator, DistributedType
from torch.utils.data import DataLoader, Dataset
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
import types
from termcolor import cprint
from core.finetune.utils import (
    cast_training_params,
    free_memory,
    get_intermediate_ckpt_path,
    get_latest_ckpt_path_to_resume_from,
    get_memory_statistics,
    get_optimizer,
    string_to_filename,
    unload_model,
    unwrap_model,
)
from tqdm import tqdm
from itertools import chain
from accelerate import init_empty_weights
from accelerate import load_checkpoint_and_dispatch
import deepspeed
from einops import rearrange
import functools
import random

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class EMA:
    """Exponential Moving Average for trainable parameters."""

    def __init__(self, parameters, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in parameters:
            if param.requires_grad:
                # Store without 'module.' prefix for consistent matching
                clean_name = name.replace("module.", "", 1) if name.startswith("module.") else name
                self.shadow[clean_name] = param.data.clone()

    def _clean_name(self, name):
        """Strip DeepSpeed 'module.' prefix for consistent key matching."""
        return name.replace("module.", "", 1) if name.startswith("module.") else name

    @torch.no_grad()
    def update(self, parameters):
        for name, param in parameters:
            clean_name = self._clean_name(name)
            if param.requires_grad and clean_name in self.shadow:
                if self.shadow[clean_name].device != param.data.device:
                    self.shadow[clean_name] = self.shadow[clean_name].to(param.data.device)
                self.shadow[clean_name].lerp_(param.data.to(self.shadow[clean_name].dtype), 1.0 - self.decay)

    def apply_shadow(self, parameters):
        """Replace model params with EMA shadow params (for saving/inference)."""
        for name, param in parameters:
            clean_name = self._clean_name(name)
            if param.requires_grad and clean_name in self.shadow:
                self.backup[clean_name] = param.data.clone()
                param.data.copy_(self.shadow[clean_name].to(param.data.device))

    def restore(self, parameters):
        """Restore model params from backup (after saving/inference)."""
        for name, param in parameters:
            clean_name = self._clean_name(name)
            if clean_name in self.backup:
                param.data.copy_(self.backup[clean_name])
        self.backup = {}

    def state_dict(self):
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state_dict):
        self.shadow = {k: v.clone() for k, v in state_dict.items()}


try:
    from diffusers.models.transformers.transformer_wan import WanTimeTextImageEmbedding
except ImportError as e:
    cprint("❌ Critical Error: Could not import `WanTimeTextImageEmbedding` for monkey-patching.", 'red')
    cprint("   The structure of the `diffusers` library may have changed.", 'red')
    raise e

def patched_wan_time_text_image_embedding_forward(
    self,  # The first argument must be `self`
    timestep: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    encoder_hidden_states_image: Optional[torch.Tensor] = None,
    timestep_seq_len: Optional[int] = None,
) -> tuple:
    timestep = self.timesteps_proj(timestep.to(torch.float32))
    if timestep_seq_len is not None:
        timestep = timestep.unflatten(0, (-1, timestep_seq_len))

    with torch.autocast(device_type=timestep.device.type, dtype=torch.float32, enabled=True):
        temb = self.time_embedder(timestep)

        timestep_proj = self.time_proj(self.act_fn(temb))
    temb_casted = temb.type_as(encoder_hidden_states)
    timestep_proj = timestep_proj.type_as(encoder_hidden_states)
    
    encoder_hidden_states = self.text_embedder(encoder_hidden_states)

    if encoder_hidden_states_image is not None:
        encoder_hidden_states_image = self.image_embedder(encoder_hidden_states_image)

    return temb_casted, timestep_proj, encoder_hidden_states, encoder_hidden_states_image

WanTimeTextImageEmbedding.forward = patched_wan_time_text_image_embedding_forward
cprint("✅ [Monkey Patch Applied] `WanTimeTextImageEmbedding.forward` has been replaced to ensure float32 stability during mixed-precision training.", "green")

def wan_forward(
    self,
    hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    encoder_hidden_states: torch.Tensor,
    encoder_hidden_states_image: torch.Tensor | None = None,
    return_dict: bool = True,
    attention_kwargs: dict[str, Any] | None = None,
    control_video_latent: torch.Tensor | None = None,
    control_type: str = 'add',
    null_condition=False,
    robot_trajectory: torch.Tensor | None = None,
) -> torch.Tensor | dict[str, torch.Tensor]:
    batch_size, num_channels, num_frames, height, width = hidden_states.shape
    p_t, p_h, p_w = self.config.patch_size
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p_h
    post_patch_width = width // p_w

    rotary_emb = self.rope(hidden_states)

    # hidden_states = self.patch_embedding(hidden_states)
    has_native = robot_trajectory is not None and hasattr(self, "native_trajectory_encoder")
    has_control = control_video_latent is not None and hasattr(self, 'control_patch_embedding')
    native_modulation = None
    if has_native and not null_condition:
        hidden_states = self.patch_embedding(hidden_states)
        native_output = self.native_trajectory_encoder(
            robot_trajectory, post_patch_num_frames
        )
        if isinstance(native_output, tuple):
            native_features, native_modulation = native_output
        else:
            native_features = native_output
        if native_features is not None:
            hidden_states = hidden_states + native_features.to(hidden_states.dtype)
    elif not has_control or null_condition:
        hidden_states = self.patch_embedding(hidden_states)
    elif control_type in ['add', 'add-plus']:
        hidden_states = self.patch_embedding(hidden_states)
        hidden_states_control = self.control_patch_embedding(control_video_latent)
        control_scale = getattr(self, 'control_scale', None)
        if control_scale is not None:
            hidden_states_control = hidden_states_control * control_scale
        hidden_states = hidden_states + hidden_states_control
    elif control_type == 'concat':
        hidden_states = torch.cat([hidden_states, control_video_latent], dim=1)
        hidden_states = self.control_patch_embedding(hidden_states)

    hidden_states = hidden_states.flatten(2).transpose(1, 2)

    # timestep shape: batch_size, or batch_size, seq_len (wan 2.2 ti2v)
    if timestep.ndim == 2:
        ts_seq_len = timestep.shape[1]
        timestep = timestep.flatten()  # batch_size * seq_len
    else:
        ts_seq_len = None

    temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
        timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=ts_seq_len
    )
    if ts_seq_len is not None:
        # batch_size, seq_len, 6, inner_dim
        timestep_proj = timestep_proj.unflatten(2, (6, -1))
    else:
        # batch_size, 6, inner_dim
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

    if native_modulation is not None:
        if ts_seq_len is None:
            raise ValueError("native action modulation requires per-token Wan timesteps")
        native_modulation = native_modulation.to(
            device=timestep_proj.device, dtype=timestep_proj.dtype
        )
        if native_modulation.shape[1] != post_patch_num_frames:
            raise ValueError(
                "action modulation time mismatch: "
                f"{native_modulation.shape[1]} != {post_patch_num_frames}"
            )
        native_modulation = native_modulation[:, :, None, None].expand(
            -1, -1, post_patch_height, post_patch_width, -1, -1
        )
        native_modulation = native_modulation.reshape(
            batch_size, ts_seq_len, 6, -1
        )
        timestep_proj = timestep_proj + native_modulation

    if encoder_hidden_states_image is not None:
        encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

    # 4. Transformer blocks
    if torch.is_grad_enabled() and self.gradient_checkpointing:
        for block in self.blocks:
            hidden_states = self._gradient_checkpointing_func(
                block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
            )
    else:
        for block in self.blocks:
            hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)

    # 5. Output norm, projection & unpatchify
    if temb.ndim == 3:
        # batch_size, seq_len, inner_dim (wan 2.2 ti2v)
        shift, scale = (self.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).chunk(2, dim=2)
        shift = shift.squeeze(2)
        scale = scale.squeeze(2)
    else:
        # batch_size, inner_dim
        shift, scale = (self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)

    # Move the shift and scale tensors to the same device as hidden_states.
    # When using multi-GPU inference via accelerate these will be on the
    # first device rather than the last device, which hidden_states ends up
    # on.
    shift = shift.to(hidden_states.device)
    scale = scale.to(hidden_states.device)

    hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
    hidden_states = self.proj_out(hidden_states)

    hidden_states = hidden_states.reshape(
        batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
    )
    hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
    output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

    if not return_dict:
        return (output,)

    return Transformer2DModelOutput(sample=output)

WanTransformer3DModel.forward = wan_forward
cprint("✅ [Monkey Patch Applied] `WanTransformer3DModel.forward` has been added control video latent.", "green")


class Wan_Components(Components):
    high_noise_model : Any = None
    # low_noise_model : Any = None

        
class RynnWorldTeleopTrainer(WanI2VTrainer):
    UNLOAD_LIST = ["text_encoder","vae"]
    @override
    def __init__(self, args: Args) -> None:
        self.args = args
        self.control_type = getattr(args, 'control_type', 'add')
        self.condition_mode = getattr(args, "condition_mode", "pose_video")
        if self.control_type == 'add-plus':
            # For add-plus, we use add mode in forward, but do distribution alignment in compute_loss
            self.is_concat = False
        else:
            self.is_concat = (self.control_type == 'concat')

        self.state = State(
            weight_dtype=self.get_training_dtype(),
            train_frames=self.args.train_resolution[0],
            train_height=self.args.train_resolution[1],
            train_width=self.args.train_resolution[2],
        )

        self._init_distributed()
        self.state.weight_dtype = self.get_training_dtype()
        self.components: Components = self.load_components()
        self.dataset: Dataset = None
        self.data_loader: DataLoader = None

        self.optimizer = None
        self.lr_scheduler = None
        self.ema_updates = 0

        if self.accelerator.is_main_process:
            print("\n" + "="*60)
            print("   ACCELERATE DISTRIBUTED SANITY CHECK (from Python code)")
            print(f"   - Total number of processes found: {self.accelerator.num_processes}")
            print(f"   - Current process is the main process: {self.accelerator.is_main_process}")
            print(f"   - Device for the main process: {self.accelerator.device}")
            print("="*60 + "\n")
        self._init_logging()
        self._init_directories()
        self.state.using_deepspeed = self.accelerator.state.deepspeed_plugin is not None

        # Running statistics for control latent distribution alignment
        # Using EMA to maintain stable mean/std even with batch_size=1
        self.control_running_stats = {
            "mean": None,  # Will be initialized on first batch
            "std": None,
            "count": 0,
            "momentum": 0.01,  # EMA momentum: new_value * momentum + old_value * (1 - momentum)
        }

    @override
    def load_components(self) -> Dict[str, Any]:
        components = Wan_Components()
        model_path = str(self.args.model_path)

        cprint(f"Loading components from: {model_path}",'green')
        components.pipeline_cls = WanImageToVideoPipeline
        components.tokenizer = T5TokenizerFast.from_pretrained(model_path, subfolder="tokenizer")
        components.text_encoder = UMT5EncoderModel.from_pretrained(model_path, subfolder="text_encoder")
        components.scheduler = UniPCMultistepScheduler.from_pretrained(model_path, subfolder="scheduler")

        ds_plugin = self.accelerator.state.deepspeed_plugin
        is_zero3 = ds_plugin is not None and ds_plugin.zero_stage == 3
        if is_zero3:
            import deepspeed
            with deepspeed.zero.Init(config_dict_or_path=ds_plugin.deepspeed_config):
                components.text_encoder = UMT5EncoderModel.from_pretrained(model_path, subfolder="text_encoder")
                components.vae = AutoencoderKLWan.from_pretrained(model_path, subfolder="vae")
                components.high_noise_model = WanTransformer3DModel.from_pretrained(
                    model_path, subfolder="transformer", eps=1e-5
                )
        else:
            components.text_encoder = UMT5EncoderModel.from_pretrained(model_path, subfolder="text_encoder")
            components.vae = AutoencoderKLWan.from_pretrained(model_path, subfolder="vae")
            components.high_noise_model = WanTransformer3DModel.from_pretrained(
                model_path, subfolder="transformer", eps=1e-5
            )

        boundary_ratio = 0
        num_train_timesteps = components.scheduler.config.num_train_timesteps
        self.state.moe_boundary = int(num_train_timesteps * boundary_ratio)
        cprint('MoE boundary set to timestep {self.state.moe_boundary}', 'green')

        try:
            components.high_noise_model.enable_xformers_memory_efficient_attention()
            cprint("✅ Successfully enabled xformers memory efficient attention.", "green")
        except Exception as e:
            cprint(f"Could not enable xformers. Fallback might be used automatically by PyTorch 2.0+. Error: {e}", "yellow")

        return components

    @override
    def prepare_for_training(self) -> None:
        high_noise_model, self.optimizer, self.data_loader, self.lr_scheduler = self.accelerator.prepare(
            self.components.high_noise_model, 
            self.optimizer, 
            self.data_loader, 
            self.lr_scheduler
        )
        # self.components.vae = self.accelerator.prepare(
        #     self.components.vae
        # )

        # self.components.vae.to(self.accelerator.device)
        self.components.high_noise_model = high_noise_model
        # self.components.low_noise_model = low_noise_model

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(self.data_loader) / self.args.gradient_accumulation_steps)
        if self.state.overwrote_max_train_steps:
            self.args.train_steps = self.args.train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        self.args.train_epochs = math.ceil(self.args.train_steps / num_update_steps_per_epoch)
        self.state.num_update_steps_per_epoch = num_update_steps_per_epoch

    @override
    def __prepare_saving_loading_hooks(self, transformer_lora_config):
        def save_model_hook(models: list, weights: list, output_dir: str):
            if self.accelerator.is_main_process:
                unwrapped_high_noise_model = unwrap_model(self.accelerator, self.components.high_noise_model)
                
                # Save LoRA weights only if LoRA adapter is present
                if transformer_lora_config is not None:
                    high_noise_lora_layers_to_save = get_peft_model_state_dict(
                        unwrapped_high_noise_model, adapter_name="high_noise"
                    )
                    self.components.pipeline_cls.save_lora_weights(
                        save_directory=os.path.join(output_dir, "high_noise_lora"),
                        transformer_lora_layers=high_noise_lora_layers_to_save,
                    )

                if hasattr(unwrapped_high_noise_model, "native_trajectory_encoder"):
                    torch.save(
                        unwrapped_high_noise_model.native_trajectory_encoder.state_dict(),
                        os.path.join(output_dir, "native_trajectory_encoder.bin"),
                    )
                if hasattr(unwrapped_high_noise_model, "control_patch_embedding"):
                    control_patch_embedding_state_dict = unwrapped_high_noise_model.control_patch_embedding.state_dict()
                    control_patch_embedding_save_path = os.path.join(output_dir, "control_patch_embedding.bin")
                    torch.save(control_patch_embedding_state_dict, control_patch_embedding_save_path)

                # Save control_scale if it exists (add mode)
                if hasattr(unwrapped_high_noise_model, 'control_scale'):
                    control_scale_save_path = os.path.join(output_dir, "control_scale.bin")
                    torch.save(unwrapped_high_noise_model.control_scale.data, control_scale_save_path)

                # Save running stats for control latent distribution alignment
                if hasattr(self, 'control_running_stats') and self.control_running_stats['mean'] is not None:
                    running_stats_save_path = os.path.join(output_dir, "control_running_stats.bin")
                    torch.save({
                        'mean': self.control_running_stats['mean'],
                        'std': self.control_running_stats['std'],
                        'count': self.control_running_stats['count'],
                        'momentum': self.control_running_stats['momentum'],
                    }, running_stats_save_path)
                    mean_val = self.control_running_stats['mean'].mean().item()
                    std_val = self.control_running_stats['std'].mean().item()
                    count_val = self.control_running_stats['count']
                    logger.info(f"[CONTROL_RUNNING_STATS] Saved to {running_stats_save_path} | mean={mean_val:.6f} std={std_val:.6f} count={count_val}")
                    print(f"[CONTROL_RUNNING_STATS] Saved | mean={mean_val:.6f} std={std_val:.6f} count={count_val}", flush=True)
                else:
                    has_attr = hasattr(self, 'control_running_stats')
                    mean_is_none = (not has_attr) or self.control_running_stats.get('mean') is None
                    logger.warning(f"[CONTROL_RUNNING_STATS] NOT SAVED | has_attr={has_attr} mean_is_none={mean_is_none} control_type={getattr(self, 'control_type', 'N/A')}")
                    print(f"[CONTROL_RUNNING_STATS] NOT SAVED | has_attr={has_attr} mean_is_none={mean_is_none} control_type={getattr(self, 'control_type', 'N/A')}", flush=True)

                # Save EMA weights
                if hasattr(self, 'ema'):
                    ema_save_path = os.path.join(output_dir, "ema_weights.bin")
                    torch.save(self.ema.state_dict(), ema_save_path)
                    logger.info(f"Successfully saved EMA weights to {ema_save_path}")
                
                logger.info(f"Successfully saved control_patch_embedding weights to {output_dir}")

            # For LoRA mode, pop model from list to prevent accelerate from saving the full model
            # For SFT mode, let accelerate/DeepSpeed save the full model state
            if transformer_lora_config is not None:
                models.clear()
                weights.clear()
                
        def load_model_hook(models: list, input_dir: str):
            high_noise_model_ = unwrap_model(self.accelerator, self.components.high_noise_model)

            # Load LoRA weights only if LoRA adapter is present
            if transformer_lora_config is not None:
                high_noise_lora_path = os.path.join(input_dir, "high_noise_lora")
                if os.path.exists(high_noise_lora_path):
                    high_noise_state_dict = self.components.pipeline_cls.lora_state_dict(high_noise_lora_path)
                    set_peft_model_state_dict(high_noise_model_, high_noise_state_dict, adapter_name="high_noise")
                    logger.info(f"Successfully loaded LoRA weights from {high_noise_lora_path} into high_noise_model")
                else:
                    logger.warning(f"Could not find LoRA weights for high_noise_model at {high_noise_lora_path}")
            
            control_patch_embedding_path = os.path.join(input_dir, "control_patch_embedding.bin")
            if os.path.exists(control_patch_embedding_path):
                try:
                    saved_weights = torch.load(control_patch_embedding_path, map_location="cpu")
                    high_noise_model_.control_patch_embedding.load_state_dict(saved_weights)
                    logger.info(f"Successfully loaded weights from {control_patch_embedding_path} into control_patch_embedding layer")
                except Exception as e:
                    logger.error(f"Failed to load control_patch_embedding weights from {control_patch_embedding_path}: {e}")

            native_path = os.path.join(input_dir, "native_trajectory_encoder.bin")
            if os.path.exists(native_path) and hasattr(high_noise_model_, "native_trajectory_encoder"):
                try:
                    high_noise_model_.native_trajectory_encoder.load_state_dict(
                        torch.load(native_path, map_location="cpu", weights_only=True)
                    )
                    logger.info(f"Loaded native trajectory encoder from {native_path}")
                except Exception as e:
                    logger.error(f"Failed to load native trajectory encoder: {e}")

            # Load control_scale if it exists (add mode)
            control_scale_path = os.path.join(input_dir, "control_scale.bin")
            if os.path.exists(control_scale_path):
                try:
                    control_scale_data = torch.load(control_scale_path, map_location="cpu")
                    high_noise_model_.control_scale.data = control_scale_data.squeeze()
                    logger.info(f"Successfully loaded control_scale from {control_scale_path}")
                except Exception as e:
                    logger.error(f"Failed to load control_scale from {control_scale_path}: {e}")

            # Load running stats for control latent distribution alignment
            running_stats_path = os.path.join(input_dir, "control_running_stats.bin")
            if os.path.exists(running_stats_path):
                try:
                    running_stats = torch.load(running_stats_path, map_location="cpu")
                    self.control_running_stats['mean'] = running_stats['mean']
                    self.control_running_stats['std'] = running_stats['std']
                    self.control_running_stats['count'] = running_stats['count']
                    self.control_running_stats['momentum'] = running_stats.get('momentum', 0.01)
                    logger.info(f"Successfully loaded control running stats from {running_stats_path} (count={self.control_running_stats['count']})")
                except Exception as e:
                    logger.error(f"Failed to load control running stats from {running_stats_path}: {e}")

            # Load EMA weights if available
            ema_path = os.path.join(input_dir, "ema_weights.bin")
            if os.path.exists(ema_path) and hasattr(self, 'ema'):
                try:
                    ema_state = torch.load(ema_path, map_location="cpu")
                    self.ema.load_state_dict(ema_state)
                    logger.info(f"Successfully loaded EMA weights from {ema_path}")
                except Exception as e:
                    logger.error(f"Failed to load EMA weights from {ema_path}: {e}")

            # For LoRA mode, pop model from list to prevent accelerate from loading the full model
            if transformer_lora_config is not None:
                indices_to_pop = []
                for i, model in enumerate(models):
                    if model is self.components.high_noise_model:
                        indices_to_pop.append(i)
                
                for i in sorted(indices_to_pop, reverse=True):
                    models.pop(i)

        self.accelerator.register_save_state_pre_hook(save_model_hook)
        self.accelerator.register_load_state_pre_hook(load_model_hook)

    @override
    def prepare_trainable_parameters(self):
        logger.info("Initializing trainable parameters")
        weight_dtype = self.state.weight_dtype

        if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
            raise ValueError("Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead.")

        # Load pretrained weights from init_from_checkpoint (e.g., SFT pretrain checkpoint-3000)
        init_ckpt = getattr(self.args, 'init_from_checkpoint', None)
        if init_ckpt is not None:
            ema_path = os.path.join(str(init_ckpt), "ema_weights.pt")
            ema_bin_path = os.path.join(str(init_ckpt), "ema_weights.bin")
            raw_path = os.path.join(str(init_ckpt), "pytorch_model", "mp_rank_00_model_states.pt")

            if os.path.exists(ema_path):
                state_dict = torch.load(ema_path, map_location="cpu", weights_only=False)
                cprint(f"Loading init weights from EMA: {ema_path} ({len(state_dict)} keys)", "green")
            elif os.path.exists(ema_bin_path):
                state_dict = torch.load(ema_bin_path, map_location="cpu", weights_only=False)
                cprint(f"Loading init weights from EMA: {ema_bin_path} ({len(state_dict)} keys)", "green")
            elif os.path.exists(raw_path):
                raw_ckpt = torch.load(raw_path, map_location="cpu", weights_only=False)
                state_dict = raw_ckpt.get("module", raw_ckpt)
                cprint(f"Loading init weights from model_states: {raw_path} ({len(state_dict)} keys)", "green")
            else:
                raise FileNotFoundError(f"No weights found in init_from_checkpoint: {init_ckpt}")

            missing, unexpected = self.components.high_noise_model.load_state_dict(state_dict, strict=False)
            if self.accelerator.is_main_process:
                if missing:
                    cprint(f"  Missing keys ({len(missing)}): {missing[:5]}...", "yellow")
                if unexpected:
                    cprint(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}...", "yellow")
                cprint(f"  Successfully loaded init_from_checkpoint weights.", "green")
            del state_dict
            torch.cuda.empty_cache()

        # For LoRA, we freeze all the parameters
        # For SFT, we train all the parameters in transformer model
        for attr_name, component in vars(self.components).items():
            if hasattr(component, "requires_grad_"):
                if self.args.training_type == "sft" and ("high_noise_model" in attr_name or "low_noise_model" in attr_name):
                    component.requires_grad_(True)
                else:
                    component.requires_grad_(False)

        transformer_lora_config = None
        if self.args.training_type == "lora":
            transformer_lora_config = LoraConfig(
                r=self.args.rank,
                lora_alpha=self.args.lora_alpha,
                init_lora_weights=True,
                target_modules=self.args.target_modules,
            )
            self.components.high_noise_model.add_adapter(transformer_lora_config, adapter_name="high_noise")
            self.components.high_noise_model.requires_grad_(False)
            for name, param in self.components.high_noise_model.named_parameters():
                if 'lora' in name:
                    param.requires_grad = True

        if self.condition_mode == "native_trajectory":
            from core.control import (
                NativeTrajectoryConditionerV2,
                NativeTrajectoryConditionerV3,
                NativeTrajectoryConditionerV4,
                NativeTrajectoryConditionerV5,
                NativeTrajectoryConditionerV6,
                NativeTrajectoryEncoder,
            )

            native_dim = getattr(self.args, "native_trajectory_dim", None)
            native_version = getattr(self.args, "native_conditioner_version", "v1")
            if native_version in ("v2", "v3", "v4", "v5", "v6"):
                if native_dim is None:
                    raise ValueError("native conditioner v2/v3 requires --native_trajectory_dim")
                self.components.high_noise_model.native_trajectory_encoder = (
                    {
                        "v2": NativeTrajectoryConditionerV2,
                        "v3": NativeTrajectoryConditionerV3,
                        "v4": NativeTrajectoryConditionerV4,
                        "v5": NativeTrajectoryConditionerV5,
                        "v6": NativeTrajectoryConditionerV6,
                    }[native_version](input_dim=native_dim)
                )
                baseline_init = getattr(self.args, "native_baseline_init", None)
                if native_version in ("v3", "v4", "v5", "v6") and baseline_init:
                    old = NativeTrajectoryEncoder(input_dim=33)
                    old.load_state_dict(torch.load(
                        baseline_init, map_location="cpu", weights_only=True
                    ))
                    old.eval()
                    with torch.no_grad():
                        baseline = old(torch.zeros(1, 1, 33), 1)
                        self.components.high_noise_model.native_trajectory_encoder.base_residual.copy_(baseline)
                    cprint(f"V3 frozen video baseline initialized from {baseline_init}", "green")
            else:
                self.components.high_noise_model.native_trajectory_encoder = (
                    NativeTrajectoryEncoder(input_dim=native_dim) if native_dim else NativeTrajectoryEncoder()
                )
            native_init = getattr(self.args, "native_adapter_init", None)
            if native_init:
                native_path = os.path.join(native_init, "native_trajectory_encoder.bin")
                missing, unexpected = self.components.high_noise_model.native_trajectory_encoder.load_state_dict(
                    torch.load(native_path, map_location="cpu", weights_only=True),
                    strict=native_version != "v4",
                )
                if native_version == "v4" and (unexpected or set(missing) != {"motion_projection.weight"}):
                    raise RuntimeError(
                        f"unexpected V3->V4 adapter load result: missing={missing}, unexpected={unexpected}"
                    )
                cprint(f"Native trajectory adapter initialized from {native_path}", "green")

        # Create control_patch_embedding and control_scale (shared for both LoRA and SFT)
        patch_embedding = self.components.high_noise_model.patch_embedding
        in_channels = patch_embedding.in_channels
        out_channels = patch_embedding.out_channels
        kernel_size = patch_embedding.kernel_size
        device = patch_embedding.weight.device
        dtype = patch_embedding.weight.dtype

        if self.is_concat:
            control_patch_embedding = nn.Conv3d(
                in_channels + in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=kernel_size
            ).to(device=device, dtype=dtype)
            with torch.no_grad():
                control_patch_embedding.weight[:, :in_channels, :, :, :].copy_(patch_embedding.weight)
                control_patch_embedding.weight[:, in_channels:, :, :, :].zero_()
                if patch_embedding.bias is not None:
                    control_patch_embedding.bias.copy_(patch_embedding.bias)

            self.components.high_noise_model.control_patch_embedding = control_patch_embedding
            self.components.high_noise_model.control_patch_embedding.requires_grad_(True)

            self.original_patch_weight = patch_embedding.weight.data.clone().detach()
            self.original_patch_bias = patch_embedding.bias.data.clone().detach() if patch_embedding.bias is not None else None
            cprint("Concat Mode: Original patch_embedding expanded to 2x channels, control part zero-initialized.", "green")
        else:
            control_patch_embedding = nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=kernel_size
            ).to(device=device, dtype=dtype)

            with torch.no_grad():
                control_patch_embedding.weight.zero_()
                if control_patch_embedding.bias is not None:
                    control_patch_embedding.bias.zero_()

            control_scale = nn.Parameter(torch.tensor(0.1, dtype=dtype)).to(device=device)

            self.components.high_noise_model.control_patch_embedding = control_patch_embedding
            self.components.high_noise_model.control_patch_embedding.requires_grad_(True)
            self.components.high_noise_model.control_scale = control_scale
            self.components.high_noise_model.control_scale.requires_grad_(True)
            cprint("Add Mode: control_patch_embedding zero-initialized, control_scale=0.1.", "green")

            # Optionally warm-start the control path from an already-trained SFT
            # checkpoint instead of from zero. Zero-init means the control signal
            # contributes nothing at step 0 and has to be relearned from scratch; on a
            # short run (~100 steps) it never gets back to the SFT magnitude
            # (measured: absmax 0.0012 after 100 steps vs 0.0364 in the released SFT
            # checkpoint, 31x smaller), so control barely reaches the denoiser and the
            # rollout collapses a couple of frames in. Loading the SFT control path
            # keeps the pretrained control behaviour on step 0 and lets LoRA adapt from
            # there. Off by default: without the flag this branch is byte-identical to
            # the original zero-init.
            init_from = getattr(self.args, "control_init_from", None)
            if init_from:
                cpe_path = os.path.join(init_from, "control_patch_embedding.bin")
                cs_path = os.path.join(init_from, "control_scale.bin")
                if os.path.exists(cpe_path):
                    sd = torch.load(cpe_path, map_location="cpu", weights_only=False)
                    self.components.high_noise_model.control_patch_embedding.load_state_dict(
                        {k: v.to(device=device, dtype=dtype) for k, v in sd.items()})
                    cprint(f"  control_patch_embedding warm-started from {cpe_path}", "green")
                else:
                    raise FileNotFoundError(f"--control_init_from given but missing {cpe_path}")
                if os.path.exists(cs_path):
                    v = torch.load(cs_path, map_location="cpu", weights_only=False)
                    v = v.item() if hasattr(v, "item") else float(v)
                    with torch.no_grad():
                        self.components.high_noise_model.control_scale.fill_(v)
                    cprint(f"  control_scale warm-started to {v}", "green")

        if self.args.training_type == "lora":
            freeze_lora = getattr(self.args, 'freeze_lora', False)
            for name, param in self.components.high_noise_model.named_parameters():
                if 'lora' in name:
                    param.requires_grad = not freeze_lora
            if freeze_lora:
                cprint("LoRA weights FROZEN - only training control_patch_embedding", "yellow")
            else:
                cprint("LoRA weights TRAINABLE - training both LoRA and control_patch_embedding", "green")
        elif self.args.training_type == "sft":
            cprint("SFT Mode: All transformer + control parameters trainable", "green")

        if self.condition_mode == "native_trajectory":
            self.components.high_noise_model.control_patch_embedding.requires_grad_(False)
            if hasattr(self.components.high_noise_model, "control_scale"):
                self.components.high_noise_model.control_scale.requires_grad_(False)
            self.components.high_noise_model.native_trajectory_encoder.requires_grad_(True)
            if getattr(self.args, "native_baseline_init", None):
                native = self.components.high_noise_model.native_trajectory_encoder
                if hasattr(native, "base_residual"):
                    native.base_residual.requires_grad_(False)
                    native.base_modulation.requires_grad_(False)
            if getattr(self.args, "native_conditioner_version", "v1") == "v6":
                native = self.components.high_noise_model.native_trajectory_encoder
                native.requires_grad_(False)
                native.input_projection.requires_grad_(True)
                cprint("V6: only the local action input MLP is trainable; injection projections are frozen.", "yellow")
            cprint("Native trajectory mode: old control path frozen and unused.", "green")

        self.__prepare_saving_loading_hooks(transformer_lora_config)

        ignore_list = ["high_noise_model"] + self.UNLOAD_LIST
        self.move_components_to_device(dtype=weight_dtype, ignore_list=ignore_list)

        if self.args.gradient_checkpointing:
            self.components.high_noise_model.enable_gradient_checkpointing()
            cprint("Gradient checkpointing enabled.", "green")

    @override
    def prepare_optimizer(self) -> None:
        logger.info("Initializing optimizer and lr scheduler")

        if self.accelerator.is_main_process:
            trainable_params_count = 0
            print("\n" + "="*60)
            print("   CHECKING TRAINABLE PARAMETERS")
            for name, param in self.components.high_noise_model.named_parameters():
                if param.requires_grad:
                    print(f"   - Trainable: {name}, shape: {param.shape}")
                    trainable_params_count += param.numel()
            print(f"   >>> Total Trainable Parameters: {trainable_params_count / 1_000_000:.2f} M")
            print("="*60 + "\n")


        # Make sure the trainable params are in float32
        # cast_training_params([self.components.high_noise_model, self.components.low_noise_model], dtype=torch.float32)
        cast_training_params([self.components.high_noise_model], dtype=self.components.high_noise_model.dtype)
        
        # For LoRA, we only want to train the LoRA weights
        # For SFT, we want to train all the 
        # Split params: control_patch_embedding and control_scale get a smaller lr
        control_patch_params = []
        lora_params = []
        for name, param in self.components.high_noise_model.named_parameters():
            if param.requires_grad:
                if (
                    'control_patch_embedding' in name
                    or 'native_trajectory_encoder' in name
                    or name == 'control_scale'
                ):
                    control_patch_params.append(param)
                else:
                    lora_params.append(param)

        # Use different learning rates for control and LoRA
        control_lr = getattr(self.args, 'control_lr', self.args.learning_rate)
        params_to_optimize = [
            {"params": lora_params, "lr": self.args.learning_rate},
            {"params": control_patch_params, "lr": control_lr},
        ]
        trainable_parameters = lora_params + control_patch_params
        self.state.num_trainable_parameters = sum(p.numel() for p in trainable_parameters)
        
        if self.accelerator.is_main_process:
            print(f"\n   >>> Control LR: {control_lr}, LoRA LR: {self.args.learning_rate}")

        use_deepspeed_opt = (
            self.accelerator.state.deepspeed_plugin is not None
            and "optimizer" in self.accelerator.state.deepspeed_plugin.deepspeed_config
        )
        optimizer = get_optimizer(
            params_to_optimize=params_to_optimize,
            optimizer_name=self.args.optimizer,
            learning_rate=self.args.learning_rate,
            beta1=self.args.beta1,
            beta2=self.args.beta2,
            beta3=self.args.beta3,
            epsilon=self.args.epsilon,
            weight_decay=self.args.weight_decay,
            use_deepspeed=use_deepspeed_opt,
        )

        num_update_steps_per_epoch = math.ceil(len(self.data_loader) / self.args.gradient_accumulation_steps)
        if self.args.train_steps is None:
            self.args.train_steps = self.args.train_epochs * num_update_steps_per_epoch
            self.state.overwrote_max_train_steps = True

        use_deepspeed_lr_scheduler = (
            self.accelerator.state.deepspeed_plugin is not None
            and "scheduler" in self.accelerator.state.deepspeed_plugin.deepspeed_config
        )
        # total_training_steps = self.args.train_steps * self.accelerator.num_processes
        # num_warmup_steps = self.args.lr_warmup_steps * self.accelerator.num_processes
        total_training_steps = self.args.train_steps
        num_warmup_steps = self.args.lr_warmup_steps

        if use_deepspeed_lr_scheduler:
            from accelerate.utils import DummyScheduler

            lr_scheduler = DummyScheduler(
                name=self.args.lr_scheduler,
                optimizer=optimizer,
                total_num_steps=total_training_steps,
                num_warmup_steps=num_warmup_steps,
            )
        else:
            lr_scheduler = get_scheduler(
                name=self.args.lr_scheduler,
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_training_steps,
                num_cycles=self.args.lr_num_cycles,
                power=self.args.lr_power,
            )

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        # Initialize EMA for trainable parameters
        self.ema = EMA(
            self.components.high_noise_model.named_parameters(),
            decay=self.args.ema_decay,
        )
        cprint(f"✅ EMA initialized with decay={self.args.ema_decay}, start_step={self.args.ema_start_step}", "green")

    @override
    def collate_fn(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        ret = {"encoded_videos": [], "img_latent": [], "null_embedding": []}
        if self.condition_mode == "native_trajectory":
            ret["robot_trajectory"] = []
        else:
            ret.update({"control_video": [], "null_control_video": []})
        for sample in samples:
            encoded_video = sample["encoded_video"]
            img_latent = sample["img_latent"]
            null_embedding = sample["null_embedding"]

            ret["encoded_videos"].append(encoded_video)
            ret["img_latent"].append(img_latent)
            ret["null_embedding"].append(null_embedding)
            if self.condition_mode == "native_trajectory":
                ret["robot_trajectory"].append(sample["robot_trajectory"])
            else:
                ret["control_video"].append(sample["control_video"])
                ret["null_control_video"].append(sample["null_control_video"])

        ret["encoded_videos"] = torch.stack(ret["encoded_videos"])
        ret["img_latent"] = torch.stack(ret["img_latent"])
        ret["null_embedding"] = torch.stack(ret["null_embedding"])
        if self.condition_mode == "native_trajectory":
            ret["robot_trajectory"] = torch.stack(ret["robot_trajectory"])
        else:
            ret["control_video"] = torch.stack(ret["control_video"])
            ret["null_control_video"] = torch.stack(ret["null_control_video"])
        return ret

    @override
    def prepare_dataset(self) -> None:
        self.components.vae = self.components.vae.to(self.accelerator.device, dtype=self.state.weight_dtype)
        self.components.text_encoder = self.components.text_encoder.to(
            self.accelerator.device, dtype=self.state.weight_dtype
        )
        logger.info("Initializing dataset and dataloader")
        from core.finetune.datasets import EgoVerseDataset22

        self.dataset = EgoVerseDataset22(
            data_root=self.args.validation_dir, 
            max_num_frames=self.args.train_resolution[0], 
            height=self.args.train_resolution[1],        
            width=self.args.train_resolution[2],      
            device=self.accelerator.device,           
            trainer=self,
            cache_dir=self.args.cache_dir,
            prompt=self.args.prompt,
        )

        # Prepare VAE and text encoder for encoding
        # self.components.vae.requires_grad_(False)
        # self.components.text_encoder.requires_grad_(False)


        # Precompute latent for video and prompt embedding
        logger.info("Precomputing latent for video and prompt embedding ... Done")

        # unload_model(self.components.vae)
        unload_model(self.components.text_encoder)
        free_memory()

        sampler = None
        shuffle = True
        with open(self.args.validation_dir, "r", encoding="utf-8") as f:
            index_records = json.load(f)
        task_ids = [rec.get("task_id") for rec in index_records]
        if all(t is not None for t in task_ids) and len(set(task_ids)) > 1:
            # Task-balanced sampling (Gate D): each task contributes equal expected
            # gradient mass regardless of raw per-task window count.
            task_counts: Dict[Any, int] = {}
            for t in task_ids:
                task_counts[t] = task_counts.get(t, 0) + 1
            sample_weights = [1.0 / task_counts[t] for t in task_ids]
            sampler = torch.utils.data.WeightedRandomSampler(
                sample_weights, num_samples=len(sample_weights), replacement=True
            )
            shuffle = False
            logger.info(f"Using WeightedRandomSampler for task-balanced sampling: {task_counts}")

        self.data_loader = torch.utils.data.DataLoader(
            self.dataset,
            collate_fn=self.collate_fn,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
            shuffle=shuffle,
            sampler=sampler,
        )
        if hasattr(self.components, "text_encoder"):
            self.components.text_encoder.to("cpu")
        
        if hasattr(self.components, "vae"):
            self.components.vae.to("cpu")

        import gc
        gc.collect()
        torch.cuda.empty_cache()

    @override
    def compute_loss(self, batch, global_step: int = 0) -> torch.Tensor:
        target_module = getattr(self.components.high_noise_model, "module", self.components.high_noise_model)
        model_dtype = target_module.patch_embedding.weight.dtype
        device = self.components.high_noise_model.device

        # latent torch.Size([1, 48, 7, 30, 52])
        # img_latent torch.Size([1, 48, 1, 30, 52])
        # control_video torch.Size([1, 48, 7, 30, 52])
        video_latent = batch["encoded_videos"].to(model_dtype) # [B, 16, 21, 30, 52]
        control_video_latent = (
            batch["control_video"].to(model_dtype)
            if self.condition_mode == "pose_video"
            else None
        )
        robot_trajectory = (
            batch["robot_trajectory"].to(device=device, dtype=torch.float32)
            if self.condition_mode == "native_trajectory"
            else None
        )
        img_latent = batch["img_latent"].to(model_dtype)
        null_embedding = batch["null_embedding"].to(model_dtype)
        batch_size, num_channels, num_frames, height, width = video_latent.shape

        # Align control_latent distribution to video_latent distribution using running statistics
        # Only for add-plus mode; add and concat modes skip this for backward compatibility.
        # Skeleton videos have mostly white background, causing very different VAE latent distribution.
        # Using EMA running stats for stable alignment even with batch_size=1.
        if self.condition_mode == "pose_video" and self.control_type == 'add-plus':
            with torch.no_grad():
                v_mean = video_latent.mean(dim=(0, 2, 3, 4), keepdim=True)
                v_std = video_latent.std(dim=(0, 2, 3, 4), keepdim=True) + 1e-8
                c_mean_batch = control_video_latent.mean(dim=(0, 2, 3, 4), keepdim=True)
                c_std_batch = control_video_latent.std(dim=(0, 2, 3, 4), keepdim=True) + 1e-8

                # Update running statistics with EMA
                momentum = self.control_running_stats["momentum"]
                if self.control_running_stats["mean"] is None:
                    # Initialize running stats on first batch
                    self.control_running_stats["mean"] = c_mean_batch.detach().clone()
                    self.control_running_stats["std"] = c_std_batch.detach().clone()
                    self.control_running_stats["count"] = 1
                else:
                    # EMA update: new * momentum + old * (1 - momentum)
                    self.control_running_stats["mean"].lerp_(c_mean_batch.detach(), momentum)
                    self.control_running_stats["std"].lerp_(c_std_batch.detach(), momentum)
                    self.control_running_stats["count"] += 1

                c_mean = self.control_running_stats["mean"]
                c_std = self.control_running_stats["std"]
                control_video_latent = (control_video_latent - c_mean) / c_std * v_std + v_mean

        # Debug: print latent distributions to diagnose distribution mismatch
        if global_step % 50 == 0 and self.accelerator.is_main_process:
            print(f"\n[Step {global_step}] video_latent: mean={video_latent.mean().item():.4f}, std={video_latent.std().item():.4f}, min={video_latent.min().item():.4f}, max={video_latent.max().item():.4f}")
            if self.control_type == 'add-plus' and self.control_running_stats['mean'] is not None:
                c_mean = self.control_running_stats['mean']
                c_std = self.control_running_stats['std']
                print(f"[Step {global_step}] control_latent (after align): mean={control_video_latent.mean().item():.4f}, std={control_video_latent.std().item():.4f}, min={control_video_latent.min().item():.4f}, max={control_video_latent.max().item():.4f}")
                print(f"[Step {global_step}] running control mean: {c_mean.mean().item():.4f}, std: {c_std.mean().item():.4f} (count={self.control_running_stats['count']})")
            print()
        

        # =============== only sample 50 timesteps ===============
        # noise = torch.randn_like(video_latent).to(model_dtype)
        # idx = torch.randint(0, 50, (batch_size,))  # CPU, to match scheduler.timesteps device
        # timesteps = self.components.scheduler.timesteps[idx].to(device)
        # noisy_latents = self.components.scheduler.add_noise(original_samples=video_latent, noise=noise, timesteps=timesteps)
        # target = noise - video_latent
        # ======================================================
        
        # ===============  sample 1000 timesteps ===============
        noise = torch.randn_like(video_latent).to(model_dtype)
        timesteps_idx = torch.randint(0, self.components.scheduler.config.num_train_timesteps, (batch_size,), device=device).long()
        s = timesteps_idx.float() / self.components.scheduler.config.num_train_timesteps
        flow_shift = self.components.scheduler.config.flow_shift 

        sigma_t = flow_shift * s / (1 + (flow_shift - 1) * s)                                                                                                                                                                                                   
        sigma_view = sigma_t.view(batch_size, 1, 1, 1, 1).to(model_dtype)                                                                                                                                                                                       
        shifted_timesteps = sigma_t * self.components.scheduler.config.num_train_timesteps

        noisy_latents = (1.0 - sigma_view) * video_latent + sigma_view * noise
        target = noise - video_latent
        # ======================================================

        noisy_latents[:, :, 0:1, :, :] = img_latent.clone().to(model_dtype)
        high_noise_input = noisy_latents
        # high_noise_input torch.Size([1, 48, 7, 30, 52])
        # control_video_latent torch.Size([1, 48, 7, 30, 52])

        # ===============cfg===============
        conditioning_dropout_prob = (
            float(getattr(self.args, "action_dropout_prob", 0.0))
            if self.condition_mode == "native_trajectory" else 0.0
        )
        if not 0.0 <= conditioning_dropout_prob < 1.0:
            raise ValueError(
                f"action_dropout_prob must be in [0, 1), got {conditioning_dropout_prob}"
            )
        mask = (torch.rand((batch_size,), device=device) >= conditioning_dropout_prob).to(dtype=model_dtype)
        mask = mask.view(batch_size, 1, 1, 1, 1)
        if control_video_latent is not None:
            control_video_latent = control_video_latent * mask

        import random
        null_condition = random.random() < conditioning_dropout_prob
        action_was_dropped = self.condition_mode == "native_trajectory" and null_condition
        if self.condition_mode == "native_trajectory" and null_condition:
            robot_trajectory = torch.zeros_like(robot_trajectory)
            null_condition = False
        # =================================

        first_frame_mask = torch.ones(1, 1, video_latent.shape[2], video_latent.shape[3], video_latent.shape[4], device=device)
        first_frame_mask[:, :, 0] = 0

        # temp_ts = (first_frame_mask[0][0][:, ::2, ::2] * timesteps).flatten()
        # timestep = temp_ts.unsqueeze(0).expand(video_latent.shape[0], -1)
        # temp_ts = (first_frame_mask[0][0][:, ::2, ::2] * timesteps_idx.view(-1, 1, 1, 1).float()).flatten(1)
        temp_ts = (first_frame_mask[0][0][:, ::2, ::2] * shifted_timesteps.view(-1, 1, 1, 1).float()).flatten(1)
        timestep_input = temp_ts.to(model_dtype)

        high_noise_pred = self.components.high_noise_model(
            hidden_states=high_noise_input,
            control_video_latent=control_video_latent,
            timestep=timestep_input,
            encoder_hidden_states=null_embedding,
            encoder_hidden_states_image=None,
            attention_kwargs=None,
            return_dict=False,
            control_type=self.control_type,
            null_condition=null_condition,
            robot_trajectory=robot_trajectory,
        )[0]

        # high_noise_pred_uncondition = self.components.high_noise_model(
        #     hidden_states=high_noise_input,
        #     control_video_latent=control_video_latent,
        #     timestep=timestep_input,
        #     encoder_hidden_states=null_embedding,
        #     encoder_hidden_states_image=None,
        #     attention_kwargs=None,
        #     return_dict=False,
        #     is_concat=self.is_concat,
        #     null_condition=True
        # )[0]

        # Timestep-weighted loss: 1 / (sigma * (1 - sigma))
        # Rebalances gradients toward perceptually important middle-sigma timesteps.
        sigma_scalar = sigma_t.float()
        timestep_weight = 1.0 / (sigma_scalar * (1.0 - sigma_scalar) + 1e-5)
        timestep_weight = timestep_weight.clamp(max=10.0)
        timestep_weight = timestep_weight / timestep_weight.mean()

        per_sample_loss = ((high_noise_pred[:, :, 1:].float() - target[:, :, 1:].float()) ** 2).mean(dim=(1, 2, 3, 4))
        loss = (per_sample_loss * timestep_weight).mean()

        # V4 learns an explicit per-frame correspondence between action and
        # observed video motion.  This auxiliary head cannot improve its loss
        # by corrupting a counterfactual denoiser output, unlike response/rank
        # objectives.  A 4x4 latent grid retains local arm/object motion while
        # keeping the target dimension equal to the 768D action representation.
        motion_weight = float(getattr(self.args, "action_motion_weight", 0.0))
        native_encoder = self.components.high_noise_model.native_trajectory_encoder
        if (
            self.condition_mode == "native_trajectory"
            and motion_weight > 0.0
            and not action_was_dropped
            and hasattr(native_encoder, "predict_video_motion")
        ):
            motion_pred = native_encoder.predict_video_motion(
                robot_trajectory, num_frames
            ).float()
            with torch.no_grad():
                pooled_video = F.adaptive_avg_pool3d(
                    video_latent.float(), (num_frames, 4, 4)
                ).permute(0, 2, 1, 3, 4).flatten(2)
                motion_target = torch.zeros_like(pooled_video)
                motion_target[:, 1:] = pooled_video[:, 1:] - pooled_video[:, :-1]
            motion_loss = F.smooth_l1_loss(
                motion_pred[:, 1:], motion_target[:, 1:]
            )
            loss = loss + motion_weight * motion_loss

        # Reconstruction can be minimized from the video prior while ignoring
        # action. Compare the same noisy latent under a temporally wrong action.
        # The response loss merely asks the outputs to differ; the ranking loss
        # additionally requires the correct action to be the better prediction.
        contrastive_weight = float(getattr(self.args, "action_contrastive_weight", 0.0))
        ranking_weight = float(getattr(self.args, "action_ranking_weight", 0.0))
        if (
            self.condition_mode == "native_trajectory"
            and (contrastive_weight > 0.0 or ranking_weight > 0.0)
            and global_step >= 5
            and not action_was_dropped
        ):
            # Alternate two hard negatives. Reversal tests temporal direction;
            # a half-window roll tests action-to-frame alignment.
            if global_step % 2 == 0:
                wrong_trajectory = robot_trajectory.flip(dims=(1,))
            else:
                wrong_trajectory = torch.roll(
                    robot_trajectory, shifts=robot_trajectory.shape[1] // 2, dims=1
                )
            wrong_pred = self.components.high_noise_model(
                hidden_states=high_noise_input,
                control_video_latent=None,
                timestep=timestep_input,
                encoder_hidden_states=null_embedding,
                encoder_hidden_states_image=None,
                attention_kwargs=None,
                return_dict=False,
                control_type=self.control_type,
                null_condition=False,
                robot_trajectory=wrong_trajectory,
            )[0]
            if contrastive_weight > 0.0:
                response_mse = (
                    high_noise_pred[:, :, 1:].float() - wrong_pred[:, :, 1:].float()
                ).square().mean()
                response_rms = (response_mse + 1e-12).sqrt()
                margin = float(getattr(self.args, "action_contrastive_margin", 0.02))
                loss = loss + contrastive_weight * torch.relu(
                    response_rms.new_tensor(margin) - response_rms
                )

            if ranking_weight > 0.0:
                wrong_per_sample_loss = (
                    wrong_pred[:, :, 1:].float() - target[:, :, 1:].float()
                ).square().mean(dim=(1, 2, 3, 4))
                ranking_margin = float(getattr(self.args, "action_ranking_margin", 0.02))
                # Detach the correct loss in this term so the hinge cannot be
                # satisfied by deliberately making the correct branch worse.
                ranking_loss = torch.relu(
                    per_sample_loss.detach()
                    + wrong_per_sample_loss.new_tensor(ranking_margin)
                    - wrong_per_sample_loss
                )
                loss = loss + ranking_weight * (ranking_loss * timestep_weight).mean()

        return loss

    @override
    def train(self) -> None:
        logger.info("Starting training")

        memory_statistics = get_memory_statistics()
        logger.info(f"Memory before training start: {json.dumps(memory_statistics, indent=4)}")

        self.state.total_batch_size_count = (self.args.batch_size * self.accelerator.num_processes * self.args.gradient_accumulation_steps)
        info = {
            "trainable parameters": self.state.num_trainable_parameters,
            "total samples": len(self.dataset),
            "train epochs": self.args.train_epochs,
            "train steps": self.args.train_steps,
            "batches per device": self.args.batch_size,
            "total batches observed per epoch": len(self.data_loader),
            "train batch size total count": self.state.total_batch_size_count,
            "gradient accumulation steps": self.args.gradient_accumulation_steps,
        }
        logger.info(f"Training configuration: {json.dumps(info, indent=4)}")

        global_step = 0
        first_epoch = 0
        initial_global_step = 0

        # Potentially load in the weights and states from a previous save
        (
            resume_from_checkpoint_path,
            initial_global_step,
            global_step,
            first_epoch,
        ) = get_latest_ckpt_path_to_resume_from(
            resume_from_checkpoint=self.args.resume_from_checkpoint,
            num_update_steps_per_epoch=self.state.num_update_steps_per_epoch,
        )
        if resume_from_checkpoint_path is not None:
            self.accelerator.load_state(resume_from_checkpoint_path)

        progress_bar = tqdm(
            range(0, self.args.train_steps),
            initial=initial_global_step,
            desc="Training steps",
            disable=not self.accelerator.is_local_main_process,
        )
    
        import time
        
        step_start_time = time.time()
        
        total_samples_processed = 0
        training_start_time = time.time()
        
        if self.accelerator.is_main_process:
            print("\n" + "="*80)
            print(f"{'Global Step':<15} | {'Samples This Step':<20} | {'Instant Throughput':<25} | {'Average Throughput':<25}")
            print("="*80)

        accelerator = self.accelerator
        generator = torch.Generator(device=accelerator.device)
        if self.args.seed is not None:
            generator = generator.manual_seed(self.args.seed)
        self.state.generator = generator

        free_memory()
        for epoch in range(first_epoch, self.args.train_epochs):
            logger.debug(f"Starting epoch ({epoch + 1}/{self.args.train_epochs})")

            self.components.high_noise_model.train()
            # models_to_accumulate = [self.components.transformer]
            models_to_accumulate = [self.components.high_noise_model]

            for step, batch in enumerate(self.data_loader):
                logger.debug(f"Starting step {step + 1}")
                logs = {}

                with accelerator.accumulate(models_to_accumulate):
                    # These weighting schemes use a uniform timestep sampling and instead post-weight the loss
                    loss = self.compute_loss(batch, global_step)
                    accelerator.backward(loss)

                    if (
                        self.condition_mode == "native_trajectory"
                        and accelerator.is_main_process
                        and accelerator.sync_gradients
                        and global_step < 3
                    ):
                        native = unwrap_model(
                            accelerator, self.components.high_noise_model
                        ).native_trajectory_encoder
                        projections = (
                            [native.output_projection]
                            if hasattr(native, "output_projection")
                            else [native.input_residual_projection, native.adaln_projection]
                        )
                        projection_grads = [p.weight.grad for p in projections]
                        grad_max = max(
                            (0.0 if g is None else g.float().abs().max().item())
                            for g in projection_grads
                        )
                        weight_max = max(
                            p.weight.float().abs().max().item() for p in projections
                        )
                        print(
                            "[NATIVE_DIAG] before_step "
                            f"grad_max={grad_max:.8f} weight_max={weight_max:.8f}",
                            flush=True,
                        )

                    if accelerator.sync_gradients:
                        if accelerator.distributed_type == DistributedType.DEEPSPEED:
                            # grad_norm = self.components.transformer.get_global_grad_norm()
                            grad_norm_high = self.components.high_noise_model.get_global_grad_norm()
                            grad_norm = (grad_norm_high**2)**0.5
                            # In some cases the grad norm may not return a float
                            if torch.is_tensor(grad_norm):
                                grad_norm = grad_norm.item()
                        else:
                            unwrapped_model = self.accelerator.unwrap_model(self.components.high_noise_model)
                            grad_norm = accelerator.clip_grad_norm_(
                                unwrapped_model.parameters(), 
                                self.args.max_grad_norm
                            )
                            if torch.is_tensor(grad_norm):
                                grad_norm = grad_norm.item()

                        logs["grad_norm"] = grad_norm

                    self.optimizer.step()
                    self.lr_scheduler.step()
                    if (
                        self.condition_mode == "native_trajectory"
                        and accelerator.is_main_process
                        and accelerator.sync_gradients
                        and global_step < 3
                    ):
                        native = unwrap_model(
                            accelerator, self.components.high_noise_model
                        ).native_trajectory_encoder
                        projections = (
                            [native.output_projection]
                            if hasattr(native, "output_projection")
                            else [native.input_residual_projection, native.adaln_projection]
                        )
                        print(
                            "[NATIVE_DIAG] after_step "
                            f"weight_max={max(p.weight.float().abs().max().item() for p in projections):.8f}",
                            flush=True,
                        )
                    self.optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    # Update EMA after warmup period
                    if global_step >= self.args.ema_start_step:
                        self.ema.update(self.components.high_noise_model.named_parameters())
                        self.ema_updates += 1

                    self._maybe_save_checkpoint(global_step)

                    samples_in_this_step = batch['encoded_videos'].shape[0] * self.accelerator.num_processes

                    step_end_time = time.time()
                    step_duration = step_end_time - step_start_time
                    instant_throughput = samples_in_this_step / step_duration if step_duration > 0 else 0
                    step_start_time = step_end_time

                    total_samples_processed += samples_in_this_step
                    total_training_time = time.time() - training_start_time
                    average_throughput = total_samples_processed / total_training_time if total_training_time > 0 else 0

                    if self.accelerator.is_main_process and (global_step % 10 == 0 or global_step == 1):
                        print(f"{global_step:<15} | {samples_in_this_step:<20} | {instant_throughput:<25.2f} samples/sec | {average_throughput:<25.2f} samples/sec")


                logs["loss"] = loss.detach().item()
                logs["lr"] = self.lr_scheduler.get_last_lr()[0]
                progress_bar.set_postfix(logs)

                if self.accelerator.is_main_process and (global_step % 10 == 0 or global_step == 1):
                    log_str = f"Epoch: {epoch+1}, Step: {global_step}/{self.args.train_steps}, "
                    log_str += f"Loss: {logs['loss']:.4f}, LR: {logs['lr']:.2e}"
                    if "grad_norm" in logs:
                        log_str += f", Grad Norm: {logs['grad_norm']:.4f}"
                    
                    logger.info(log_str)

                accelerator.log(logs, step=global_step)

                if global_step >= self.args.train_steps:
                    break

            memory_statistics = get_memory_statistics()
            logger.info(f"Memory after epoch {epoch + 1}: {json.dumps(memory_statistics, indent=4)}")

        accelerator.wait_for_everyone()
        self._maybe_save_checkpoint(global_step, must_save=True)

        # Save final EMA weights for inference
        if self.accelerator.is_main_process and hasattr(self, 'ema'):
            ema_output_dir = os.path.join(str(self.args.output_dir), "ema_final")
            os.makedirs(ema_output_dir, exist_ok=True)

            unwrapped = unwrap_model(self.accelerator, self.components.high_noise_model)
            if self.ema_updates > 0:
                self.ema.apply_shadow(unwrapped.named_parameters())
            else:
                logger.info("EMA had no updates; saving raw final adapter weights.")

            if hasattr(unwrapped, 'peft_config'):
                ema_lora_layers = get_peft_model_state_dict(unwrapped, adapter_name="high_noise")
                self.components.pipeline_cls.save_lora_weights(
                    save_directory=os.path.join(ema_output_dir, "high_noise_lora"),
                    transformer_lora_layers=ema_lora_layers,
                )
            else:
                torch.save(self.ema.state_dict(), os.path.join(ema_output_dir, "ema_weights.bin"))

            if hasattr(unwrapped, "native_trajectory_encoder"):
                torch.save(
                    unwrapped.native_trajectory_encoder.state_dict(),
                    os.path.join(ema_output_dir, "native_trajectory_encoder.bin"),
                )
            if hasattr(unwrapped, "control_patch_embedding"):
                torch.save(
                    unwrapped.control_patch_embedding.state_dict(),
                    os.path.join(ema_output_dir, "control_patch_embedding.bin"),
                )
            if hasattr(unwrapped, 'control_scale'):
                torch.save(
                    unwrapped.control_scale.data,
                    os.path.join(ema_output_dir, "control_scale.bin"),
                )

            # Save running stats for inference
            if hasattr(self, 'control_running_stats') and self.control_running_stats['mean'] is not None:
                torch.save({
                    'mean': self.control_running_stats['mean'],
                    'std': self.control_running_stats['std'],
                    'count': self.control_running_stats['count'],
                    'momentum': self.control_running_stats['momentum'],
                }, os.path.join(ema_output_dir, "control_running_stats.bin"))
                logger.info(f"Saved final control running stats to {ema_output_dir}")

            logger.info(f"Saved final EMA weights to {ema_output_dir}")

            self.ema.restore(unwrapped.named_parameters())

        if self.args.do_validation:
            free_memory()
            self.validate(global_step)

        del self.components
        free_memory()
        memory_statistics = get_memory_statistics()
        logger.info(f"Memory after training end: {json.dumps(memory_statistics, indent=4)}")

        accelerator.end_training()

    @override
    def _maybe_save_checkpoint(self, global_step: int, must_save: bool = False):
        if not (must_save or global_step % self.args.checkpointing_steps == 0):
            return
        save_path = get_intermediate_ckpt_path(
            checkpointing_limit=self.args.checkpointing_limit,
            step=global_step,
            output_dir=self.args.output_dir,
        )
        if self.accelerator.is_main_process:
            logger.info(f"Checkpointing at step {global_step}")
            logger.info(f"Saving state to {save_path}")
            os.makedirs(save_path, exist_ok=True)
        self.accelerator.wait_for_everyone()
        if (
            self.condition_mode == "native_trajectory"
            and getattr(self.args, "freeze_lora", False)
        ):
            if self.accelerator.is_main_process:
                unwrapped = unwrap_model(
                    self.accelerator, self.components.high_noise_model
                )
                torch.save(
                    unwrapped.native_trajectory_encoder.state_dict(),
                    os.path.join(save_path, "native_trajectory_encoder.bin"),
                )
                torch.save(
                    {
                        "global_step": global_step,
                        "condition_mode": self.condition_mode,
                        "ema_updates": self.ema_updates,
                    },
                    os.path.join(save_path, "training_meta.bin"),
                )
            self.accelerator.wait_for_everyone()
            return
        self.accelerator.save_state(save_path)

register("rynnworld_teleop", "lora", RynnWorldTeleopTrainer)
register("rynnworld_teleop", "sft", RynnWorldTeleopTrainer)
