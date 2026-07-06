from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import (
    AutoencoderKLWan,
    UniPCMultistepScheduler,
    WanImageToVideoPipeline,
    WanTransformer3DModel,
)
from diffusers.training_utils import EMAModel
from transformers import T5TokenizerFast, UMT5EncoderModel
from typing_extensions import override

from core.finetune.models.wan_i2v.wan_trainer import Wan_Components as Components
from core.finetune.models.wan_i2v.wan_trainer import WanI2VTrainer
from core.finetune.utils import (
    cast_training_params,
    free_memory,
    get_intermediate_ckpt_path,
    get_latest_ckpt_path_to_resume_from,
    get_memory_statistics,
    get_optimizer,
    unload_model,
    unwrap_model,
)
from ..utils import register

from diffusers.optimization import get_scheduler
from diffusers.utils import logging
from accelerate.accelerator import DistributedType
from torch.utils.data import DataLoader
from termcolor import cprint
from tqdm import tqdm

import gc
import json
import math
import os
import random
import time

logger = logging.get_logger(__name__)


class RynnWorldTeleopPretrainTrainer(WanI2VTrainer):
    UNLOAD_LIST = ["text_encoder", "vae"]

    @override
    def __init__(self, args) -> None:
        from core.finetune.schemas import Args, State
        self.args = args
        self.state = State(
            weight_dtype=self.get_training_dtype(),
            train_frames=self.args.train_resolution[0],
            train_height=self.args.train_resolution[1],
            train_width=self.args.train_resolution[2],
        )

        self._init_distributed()
        self.state.weight_dtype = self.get_training_dtype()
        self.components = self.load_components()
        self.dataset = None
        self.data_loader = None
        self.optimizer = None
        self.lr_scheduler = None
        self.ema_model = None
        self._ema_param_names = []

        if self.accelerator.is_main_process:
            print("\n" + "=" * 60)
            print("   RynnWorldTeleop Pretrain Trainer Initialized")
            print(f"   - Total processes: {self.accelerator.num_processes}")
            print(f"   - Device: {self.accelerator.device}")
            print("=" * 60 + "\n")

        self._init_logging()
        self._init_directories()
        self.state.using_deepspeed = self.accelerator.state.deepspeed_plugin is not None

    @override
    def load_components(self) -> Dict[str, Any]:
        components = Components()
        model_path = str(self.args.model_path)

        cprint(f"Loading components from: {model_path}", 'green')
        components.pipeline_cls = WanImageToVideoPipeline
        components.tokenizer = T5TokenizerFast.from_pretrained(model_path, subfolder="tokenizer")
        components.text_encoder = UMT5EncoderModel.from_pretrained(model_path, subfolder="text_encoder")
        components.scheduler = UniPCMultistepScheduler.from_pretrained(model_path, subfolder="scheduler")
        components.vae = AutoencoderKLWan.from_pretrained(model_path, subfolder="vae")

        ds_plugin = self.accelerator.state.deepspeed_plugin
        is_zero3 = ds_plugin is not None and ds_plugin.zero_stage == 3
        if is_zero3:
            import deepspeed
            with deepspeed.zero.Init(config_dict_or_path=ds_plugin.deepspeed_config):
                components.high_noise_model = WanTransformer3DModel.from_pretrained(
                    model_path, subfolder="transformer", eps=1e-5
                )
        else:
            components.high_noise_model = WanTransformer3DModel.from_pretrained(
                model_path, subfolder="transformer", eps=1e-5
            )

        try:
            components.high_noise_model.enable_xformers_memory_efficient_attention()
            cprint("xformers memory efficient attention enabled.", "green")
        except Exception as e:
            cprint(f"Could not enable xformers: {e}", "yellow")

        return components

    @override
    def prepare_trainable_parameters(self):
        logger.info("Initializing trainable parameters for SFT")
        weight_dtype = self.state.weight_dtype

        for attr_name, component in vars(self.components).items():
            if hasattr(component, "requires_grad_"):
                if "high_noise_model" in attr_name:
                    component.requires_grad_(True)
                else:
                    component.requires_grad_(False)

        ignore_list = ["high_noise_model"] + self.UNLOAD_LIST
        self.move_components_to_device(dtype=weight_dtype, ignore_list=ignore_list)

        if self.args.gradient_checkpointing:
            self.components.high_noise_model.enable_gradient_checkpointing()
            cprint("Gradient checkpointing enabled.", "green")

    @override
    def prepare_optimizer(self) -> None:
        logger.info("Initializing optimizer and lr scheduler")

        trainable_params = [p for p in self.components.high_noise_model.parameters() if p.requires_grad]
        self.state.num_trainable_parameters = sum(p.numel() for p in trainable_params)

        if self.accelerator.is_main_process:
            print(f"Total Trainable Parameters: {self.state.num_trainable_parameters / 1_000_000:.2f} M")

        params_to_optimize = [{"params": trainable_params, "lr": self.args.learning_rate}]

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

        total_training_steps = self.args.train_steps
        num_warmup_steps = self.args.lr_warmup_steps

        use_deepspeed_lr_scheduler = (
            self.accelerator.state.deepspeed_plugin is not None
            and "scheduler" in self.accelerator.state.deepspeed_plugin.deepspeed_config
        )
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

    @override
    def prepare_dataset(self) -> None:
        self.components.vae = self.components.vae.to(self.accelerator.device, dtype=self.state.weight_dtype)
        self.components.text_encoder = self.components.text_encoder.to(
            self.accelerator.device, dtype=self.state.weight_dtype
        )
        logger.info("Initializing dataset and dataloader")
        from core.finetune.datasets import EgoVerseDataset22SFT

        self.dataset = EgoVerseDataset22SFT(
            data_root=self.args.validation_dir,
            max_num_frames=self.args.train_resolution[0],
            height=self.args.train_resolution[1],
            width=self.args.train_resolution[2],
            device=self.accelerator.device,
            trainer=self,
            cache_dir=self.args.cache_dir,
            prompt=self.args.prompt,
        )

        unload_model(self.components.vae)
        unload_model(self.components.text_encoder)
        free_memory()

        self.data_loader = torch.utils.data.DataLoader(
            self.dataset,
            collate_fn=self.collate_fn,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
            shuffle=True,
        )

        self.components.text_encoder.to("cpu")
        self.components.vae.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

    @override
    def prepare_for_training(self) -> None:
        high_noise_model, self.optimizer, self.data_loader, self.lr_scheduler = self.accelerator.prepare(
            self.components.high_noise_model,
            self.optimizer,
            self.data_loader,
            self.lr_scheduler,
        )
        self.components.high_noise_model = high_noise_model

        num_update_steps_per_epoch = math.ceil(len(self.data_loader) / self.args.gradient_accumulation_steps)
        if self.state.overwrote_max_train_steps:
            self.args.train_steps = self.args.train_epochs * num_update_steps_per_epoch
        self.args.train_epochs = math.ceil(self.args.train_steps / num_update_steps_per_epoch)
        self.state.num_update_steps_per_epoch = num_update_steps_per_epoch

        # Initialize EMA (aligned with FlowWorld)
        use_ema = getattr(self.args, 'use_ema', True)
        if use_ema:
            unwrapped = unwrap_model(self.accelerator, self.components.high_noise_model)
            trainable_params = [p for p in unwrapped.parameters() if p.requires_grad]
            self.ema_model = EMAModel(trainable_params, decay=self.args.ema_decay)
            self._ema_param_names = [n for n, p in unwrapped.named_parameters() if p.requires_grad]
            cprint(f"EMA initialized with decay={self.args.ema_decay}", "green")

    @override
    def collate_fn(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        ret = {
            "encoded_videos": [],
            "img_latent": [],
            "prompt_embedding": [],
            "null_embedding": [],
        }
        for sample in samples:
            ret["encoded_videos"].append(sample["encoded_video"])
            ret["img_latent"].append(sample["img_latent"])
            ret["prompt_embedding"].append(sample["prompt_embedding"])
            ret["null_embedding"].append(sample["null_embedding"])

        ret["encoded_videos"] = torch.stack(ret["encoded_videos"])
        ret["img_latent"] = torch.stack(ret["img_latent"])
        ret["prompt_embedding"] = torch.stack(ret["prompt_embedding"])
        ret["null_embedding"] = torch.stack(ret["null_embedding"])
        return ret

    @override
    def encode_text(self, prompt: str, max_sequence_length: int = 226) -> torch.Tensor:
        import ftfy
        import html
        import regex as re

        def basic_clean(text):
            text = ftfy.fix_text(text)
            text = html.unescape(html.unescape(text))
            return text.strip()

        def whitespace_clean(text):
            text = re.sub(r"\s+", " ", text)
            text = text.strip()
            return text

        prompt = whitespace_clean(basic_clean(prompt))
        dtype = self.components.text_encoder.dtype

        text_inputs = self.components.tokenizer(
            [prompt],
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.accelerator.device)
        mask = text_inputs.attention_mask.to(self.accelerator.device)
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.components.text_encoder(text_input_ids, mask).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
        )
        return prompt_embeds

    @override
    def compute_loss(self, batch) -> torch.Tensor:
        target_module = getattr(self.components.high_noise_model, "module", self.components.high_noise_model)
        model_dtype = target_module.patch_embedding.weight.dtype
        device = self.components.high_noise_model.device

        video_latent = batch["encoded_videos"].to(model_dtype)
        img_latent = batch["img_latent"].to(model_dtype)
        prompt_embedding = batch["prompt_embedding"].to(model_dtype)
        null_embedding = batch["null_embedding"].to(model_dtype)

        batch_size, num_channels, num_frames, height, width = video_latent.shape

        # 20% prompt dropout for CFG (FlowWorld uses 15%)
        if torch.rand(1).item() < 0.2:
            encoder_hidden_states = null_embedding
        else:
            encoder_hidden_states = prompt_embedding

        # Flow matching noise schedule with flow_shift
        noise = torch.randn_like(video_latent)
        num_train_timesteps = self.components.scheduler.config.num_train_timesteps
        timesteps_idx = torch.randint(0, num_train_timesteps, (batch_size,), device=device).long()
        s = timesteps_idx.float() / num_train_timesteps
        flow_shift = self.components.scheduler.config.flow_shift
        sigma_t = flow_shift * s / (1 + (flow_shift - 1) * s)
        sigma_view = sigma_t.view(batch_size, 1, 1, 1, 1).to(model_dtype)
        shifted_timesteps = sigma_t * num_train_timesteps

        noisy_latents = (1.0 - sigma_view) * video_latent + sigma_view * noise
        target = noise - video_latent

        # First frame conditioning
        noisy_latents[:, :, 0:1, :, :] = img_latent.clone().to(model_dtype)

        # Per-token timestep (first frame = 0, rest = shifted_timesteps)
        first_frame_mask = torch.ones(1, 1, num_frames, height, width, device=device)
        first_frame_mask[:, :, 0] = 0
        temp_ts = (first_frame_mask[0][0][:, ::2, ::2] * shifted_timesteps.view(-1, 1, 1, 1).float()).flatten(1)
        timestep_input = temp_ts.to(model_dtype)

        # Forward pass
        predicted = self.components.high_noise_model(
            hidden_states=noisy_latents,
            timestep=timestep_input,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_image=None,
            attention_kwargs=None,
            return_dict=False,
        )[0]

        # MSE loss excluding first frame (aligned with FlowWorld)
        loss = F.mse_loss(predicted[:, :, 1:].float(), target[:, :, 1:].float(), reduction="mean")

        return loss

    @override
    def train(self) -> None:
        logger.info("Starting SFT training")

        memory_statistics = get_memory_statistics()
        logger.info(f"Memory before training start: {json.dumps(memory_statistics, indent=4)}")

        self.state.total_batch_size_count = (
            self.args.batch_size * self.accelerator.num_processes * self.args.gradient_accumulation_steps
        )
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
            # Resume EMA weights
            if self.ema_model is not None:
                ema_path = os.path.join(resume_from_checkpoint_path, "ema_weights.pt")
                if os.path.exists(ema_path):
                    ema_state_dict = torch.load(ema_path, map_location="cpu", weights_only=False)
                    for i, name in enumerate(self._ema_param_names):
                        if name in ema_state_dict:
                            self.ema_model.shadow_params[i].copy_(ema_state_dict[name])
                    cprint(f"Resumed EMA weights from {ema_path}", "green")

        progress_bar = tqdm(
            range(0, self.args.train_steps),
            initial=initial_global_step,
            desc="Training steps",
            disable=not self.accelerator.is_local_main_process,
        )

        step_start_time = time.time()
        total_samples_processed = 0
        training_start_time = time.time()

        if self.accelerator.is_main_process:
            print("\n" + "=" * 80)
            print(f"  Training: {initial_global_step}/{self.args.train_steps} steps completed | "
                  f"{self.args.train_epochs} epochs | "
                  f"{len(self.dataset)} samples | "
                  f"batch_size={self.args.batch_size} x {self.accelerator.num_processes} GPUs x {self.args.gradient_accumulation_steps} accum")
            print("=" * 80)
            print(f"{'Global Step':<15} | {'Loss':<10} | {'Instant Throughput':<25} | {'Average Throughput':<25}")
            print("-" * 80)

        accelerator = self.accelerator
        generator = torch.Generator(device=accelerator.device)
        if self.args.seed is not None:
            generator = generator.manual_seed(self.args.seed)
        self.state.generator = generator

        free_memory()
        for epoch in range(first_epoch, self.args.train_epochs):
            logger.debug(f"Starting epoch ({epoch + 1}/{self.args.train_epochs})")

            self.components.high_noise_model.train()
            models_to_accumulate = [self.components.high_noise_model]

            for step, batch in enumerate(self.data_loader):
                logs = {}

                with accelerator.accumulate(models_to_accumulate):
                    loss = self.compute_loss(batch)
                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        if accelerator.distributed_type == DistributedType.DEEPSPEED:
                            grad_norm = (self.components.high_noise_model.get_global_grad_norm() ** 2) ** 0.5
                            if torch.is_tensor(grad_norm):
                                grad_norm = grad_norm.item()
                        else:
                            unwrapped_model = accelerator.unwrap_model(self.components.high_noise_model)
                            grad_norm = accelerator.clip_grad_norm_(
                                unwrapped_model.parameters(),
                                self.args.max_grad_norm
                            )
                            if torch.is_tensor(grad_norm):
                                grad_norm = grad_norm.item()
                        logs["grad_norm"] = grad_norm

                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                    # Update EMA (every optimizer step, aligned with FlowWorld)
                    if self.ema_model is not None:
                        unwrapped = unwrap_model(accelerator, self.components.high_noise_model)
                        trainable_params = [p for p in unwrapped.parameters() if p.requires_grad]
                        self.ema_model.step(trainable_params)

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    self._maybe_save_checkpoint(global_step)

                    samples_in_this_step = batch['encoded_videos'].shape[0] * accelerator.num_processes
                    step_end_time = time.time()
                    step_duration = step_end_time - step_start_time
                    instant_throughput = samples_in_this_step / step_duration if step_duration > 0 else 0
                    step_start_time = step_end_time

                    total_samples_processed += samples_in_this_step
                    total_training_time = time.time() - training_start_time
                    average_throughput = total_samples_processed / total_training_time if total_training_time > 0 else 0

                    if accelerator.is_main_process and (global_step % 10 == 0 or global_step == 1):
                        remaining_steps = self.args.train_steps - global_step
                        eta_seconds = remaining_steps * step_duration if step_duration > 0 else 0
                        eta_h, eta_m = int(eta_seconds // 3600), int((eta_seconds % 3600) // 60)
                        print(f"[{global_step}/{self.args.train_steps}] loss={loss.item():.4f} | "
                              f"{instant_throughput:.2f} samples/sec | avg {average_throughput:.2f} samples/sec | "
                              f"ETA {eta_h}h{eta_m:02d}m")

                logs["loss"] = loss.detach().item()
                logs["lr"] = self.lr_scheduler.get_last_lr()[0]
                progress_bar.set_postfix(logs)
                accelerator.log(logs, step=global_step)

                if global_step >= self.args.train_steps:
                    break

            memory_statistics = get_memory_statistics()
            logger.info(f"Memory after epoch {epoch + 1}: {json.dumps(memory_statistics, indent=4)}")

        accelerator.wait_for_everyone()
        self._maybe_save_checkpoint(global_step, must_save=True)

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
        self.accelerator.save_state(save_path, safe_serialization=True)

        # Save EMA weights alongside checkpoint (aligned with FlowWorld)
        if self.ema_model is not None and self.accelerator.is_main_process:
            ema_save_path = os.path.join(save_path, "ema_weights.pt")
            ema_state_dict = {}
            for name, shadow_param in zip(self._ema_param_names, self.ema_model.shadow_params):
                ema_state_dict[name] = shadow_param.detach().cpu().clone()
            torch.save(ema_state_dict, ema_save_path)
            logger.info(f"Saved EMA weights to {ema_save_path}")

    @override
    def initialize_pipeline(self):
        return None

    @override
    def validation_step(self, eval_data, pipe):
        return []

    @override
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        vae = self.components.vae
        video = video.to(vae.device, dtype=vae.dtype)
        with torch.no_grad():
            latent_dist = vae.encode(video).latent_dist
            latent = latent_dist.mode()
            latents_mean = (
                torch.tensor(vae.config.latents_mean)
                .view(1, vae.config.z_dim, 1, 1, 1)
                .to(latent.device, latent.dtype)
            )
            latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
                latent.device, latent.dtype
            )
            latent = (latent - latents_mean) * latents_std
        return latent


register("rynnworld_teleop_pretrain", "sft", RynnWorldTeleopPretrainTrainer)
