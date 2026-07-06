import logging
from typing import Literal, Optional
import os
import shutil
import tempfile
import numpy as np
from torchvision import transforms
import torch
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch.nn as nn
import hashlib
from pathlib import Path
from safetensors.torch import load_file, save_file
from diffusers import (
    WanImageToVideoPipeline,
    WanPipeline,
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.utils import (
    BaseOutput,
    export_to_video,
    load_image,
    load_video,
    replace_example_docstring,
)
from termcolor import cprint
logging.basicConfig(level=logging.INFO)
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from pathlib import Path
from safetensors.torch import load_file, save_file
from torchvision.transforms import ToPILImage

def safe_export_to_video(frames, output_path, fps=16):
    """Write video to /tmp first then move, avoiding NFS seek issues."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        export_to_video(frames, tmp_path, fps=fps)
        shutil.move(tmp_path, output_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

def wan_forward(
    self,
    hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    encoder_hidden_states: torch.Tensor,
    encoder_hidden_states_image: torch.Tensor | None = None,
    return_dict: bool = True,
    attention_kwargs: dict[str, Any] | None = None,
    control_video_latent: torch.Tensor | None = None,
    control_type: str='add',
    null_condition=False,
) -> torch.Tensor | dict[str, torch.Tensor]:
    batch_size, num_channels, num_frames, height, width = hidden_states.shape
    p_t, p_h, p_w = self.config.patch_size
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p_h
    post_patch_width = width // p_w

    rotary_emb = self.rope(hidden_states)

    # Force cast embedding layers to match input dtype (accelerate hooks may change it)
    target_dtype = hidden_states.dtype
    if self.patch_embedding.weight.dtype != target_dtype:
        self.patch_embedding = self.patch_embedding.to(dtype=target_dtype)
    if hasattr(self, 'control_patch_embedding') and self.control_patch_embedding.weight.dtype != target_dtype:
        self.control_patch_embedding = self.control_patch_embedding.to(dtype=target_dtype)
    if hasattr(self, 'control_scale') and self.control_scale.dtype != target_dtype:
        self.control_scale = self.control_scale.to(dtype=target_dtype)
    if self.condition_embedder.time_proj.weight.dtype != target_dtype:
        self.condition_embedder = self.condition_embedder.to(dtype=target_dtype)

    # hidden_states = self.patch_embedding(hidden_states)
    if control_type in ['add', 'add-plus']:
        if null_condition:
            hidden_states = self.patch_embedding(hidden_states)
        else:
            hidden_states = self.patch_embedding(hidden_states)
            hidden_states_control = self.control_patch_embedding(control_video_latent)
            control_scale = getattr(self, 'control_scale', None)
            if control_scale is not None:
                hidden_states_control = hidden_states_control * control_scale
            hidden_states = hidden_states + hidden_states_control
    elif control_type=='concat':
        if null_condition:
            hidden_states = self.patch_embedding(hidden_states)
        else:
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

def randn_tensor(
    shape: Union[Tuple, List],
    generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None,
    device: Optional[Union[str, "torch.device"]] = None,
    dtype: Optional["torch.dtype"] = None,
    layout: Optional["torch.layout"] = None,
):
    # device on which tensor is created defaults to device
    if isinstance(device, str):
        device = torch.device(device)
    rand_device = device
    batch_size = shape[0]

    layout = layout or torch.strided
    device = device or torch.device("cpu")

    if generator is not None:
        gen_device_type = generator.device.type if not isinstance(generator, list) else generator[0].device.type
        if gen_device_type != device.type and gen_device_type == "cpu":
            rand_device = "cpu"
            if device != "mps":
                print(
                    f"The passed generator was created on 'cpu' even though a tensor on {device} was expected."
                    f" Tensors will be created on 'cpu' and then moved to {device}. Note that one can probably"
                    f" slightly speed up this function by passing a generator that was created on the {device} device."
                )
        elif gen_device_type != device.type and gen_device_type == "cuda":
            raise ValueError(f"Cannot generate a {device} tensor from a generator of type {gen_device_type}.")

    # make sure generator list of length 1 is treated like a non-list
    if isinstance(generator, list) and len(generator) == 1:
        generator = generator[0]

    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        latents = [
            torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype, layout=layout)
            for i in range(batch_size)
        ]
        latents = torch.cat(latents, dim=0).to(device)
    else:
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)

    return latents

def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")

class WanImagePipeline(WanImageToVideoPipeline):
    def video_transform(self, frames: torch.Tensor) -> torch.Tensor:
        self.__transforms = transforms.Compose([transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)])
        return torch.stack([self.__transforms(f) for f in frames], dim=0)

    @torch.no_grad()
    def prepare_latents(
        self,
        image,
        batch_size: int,
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        last_image: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial

        shape = (batch_size, num_channels_latents, num_latent_frames, latent_height, latent_width)
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        image = image.unsqueeze(2)  # [batch_size, channels, 1, height, width]

        if self.config.expand_timesteps:
            video_condition = image
        elif last_image is None:
            video_condition = torch.cat(
                [image, image.new_zeros(image.shape[0], image.shape[1], num_frames - 1, height, width)], dim=2
            )
        else:
            last_image = last_image.unsqueeze(2)
            video_condition = torch.cat(
                [image, image.new_zeros(image.shape[0], image.shape[1], num_frames - 2, height, width), last_image],
                dim=2,
            )
        video_condition = video_condition.to(device=device, dtype=self.vae.dtype)

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )

        if isinstance(generator, list):
            latent_condition = [
                retrieve_latents(self.vae.encode(video_condition), sample_mode="argmax") for _ in generator
            ]
            latent_condition = torch.cat(latent_condition)
        else:
            latent_condition = retrieve_latents(self.vae.encode(video_condition), sample_mode="argmax")
            latent_condition = latent_condition.repeat(batch_size, 1, 1, 1, 1)

        latent_condition = latent_condition.to(dtype)
        latent_condition = (latent_condition - latents_mean) * latents_std

        first_frame_mask = torch.ones(
            1, 1, num_latent_frames, latent_height, latent_width, dtype=dtype, device=device
        )
        first_frame_mask[:, :, 0] = 0

        return latents, latent_condition, first_frame_mask

    def make_null_video(self, num_frames):
        num_frames = int(num_frames) 
        height = 480
        width = 832

        device = self._execution_device
        vae_dtype = self.vae.dtype

        self.vae.to(device)

        short_video = torch.full((1, 3, num_frames, height, width), -1.0, device=device, dtype=vae_dtype)

        with torch.no_grad():
            latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, -1, 1, 1, 1).to(device, vae_dtype)
            latents_std = torch.tensor(self.vae.config.latents_std).view(1, -1, 1, 1, 1).to(device, vae_dtype)

            short_video = short_video.contiguous()
            
            short_video_latents = self.vae.encode(short_video).latent_dist.mode()
            short_video_latents = (short_video_latents - latents_mean) / latents_std

        return short_video_latents

    @torch.no_grad()
    def __call__(
        self,
        video_latent_path=None,
        image = None,
        prompt: str | list[str] = None,
        negative_prompt: str | list[str] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        guidance_scale_2: float | None = None,
        num_videos_per_prompt: int | None = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        image_embeds: torch.Tensor | None = None,
        last_image: torch.Tensor | None = None,
        output_type: str | None = "np",
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        max_sequence_length: int = 512,
        control_type: str = None,
        gt_inject_steps: int = 15,
    ):
        video_latent_path = Path(video_latent_path)
        device = self._execution_device

        cache_data = load_file(video_latent_path)
        video_latents = cache_data["video_latents"]
        control_video_latent = cache_data["control_video_latents"]
        img_latent = cache_data["img_latent"]
        
        num_frames = (control_video_latent.shape[1] - 1) * 4 + 1

        # video_latents torch.Size([48, 7, 30, 52]) -> [1, 48, 7, 30, 52]
        # control_video_latents torch.Size([48, 7, 30, 52]) -> [1, 48, 7, 30, 52]
        # img_latent torch.Size([48, 1, 30, 52])
        transformer_dtype = self.transformer.dtype if self.transformer is not None else self.transformer_2.dtype
        video_latents = video_latents.unsqueeze(0).to(device=device, dtype=transformer_dtype)
        control_video_latent = control_video_latent.unsqueeze(0).to(device=device, dtype=transformer_dtype)
        img_latent = img_latent.to(device=device, dtype=transformer_dtype)
        
        # ==========================================
        # 'add-plus': Align control_latent distribution to match video_latent distribution
        # This matches the training behavior when running stats alignment is enabled.
        # Regular 'add' and 'concat' modes skip this for backward compatibility.
        # ==========================================
        if control_type == 'add-plus':
            with torch.no_grad():
                control_video_latent_raw = control_video_latent.clone()

                v_mean = video_latents.mean(dim=(0, 2, 3, 4), keepdim=True)
                v_std = video_latents.std(dim=(0, 2, 3, 4), keepdim=True) + 1e-8
                
                running_stats = getattr(self.transformer, 'control_running_stats', None)
                if running_stats is not None and 'mean' in running_stats:
                    c_mean = running_stats['mean'].to(device=device)
                    c_std = running_stats['std'].to(device=device)
                else:
                    c_mean = control_video_latent.mean(dim=(0, 2, 3, 4), keepdim=True)
                    c_std = control_video_latent.std(dim=(0, 2, 3, 4), keepdim=True) + 1e-8
                
                control_video_latent = (control_video_latent - c_mean) / c_std * v_std + v_mean
        else:
            control_video_latent_raw = control_video_latent
        
        # Use VAE-encoded zero video as null control (matching training behavior)
        null_control_video_latent = self.make_null_video(num_frames)
        null_control_video_latent = null_control_video_latent.to(device=device, dtype=control_video_latent.dtype)
        num_frames = max(num_frames, 1)

        patch_size = (
            self.transformer.config.patch_size
            if self.transformer is not None
            else self.transformer_2.config.patch_size
        )
        h_multiple_of = self.vae_scale_factor_spatial * patch_size[1]
        w_multiple_of = self.vae_scale_factor_spatial * patch_size[2]

        self._guidance_scale = guidance_scale
        self._guidance_scale_2 = guidance_scale_2
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        
        batch_size = 1

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        # Encode image embedding
        transformer_dtype = self.transformer.dtype if self.transformer is not None else self.transformer_2.dtype
        prompt_embeds = prompt_embeds.to(device=device, dtype=transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(device=device, dtype=transformer_dtype)


        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.vae.config.z_dim
        # image = self.video_processor.preprocess(image, height=height, width=width).to(device, dtype=torch.float32)
        # image torch.Size([1, 3, 480, 832])

        # latents, condition, first_frame_mask = latents_outputs
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial
        shape = (batch_size, num_channels_latents, num_latent_frames, latent_height, latent_width)
        latents = randn_tensor(shape, generator=generator, device=device)
        first_frame_mask = torch.ones(1, 1, num_latent_frames, latent_height, latent_width, device=device)
        first_frame_mask[:, :, 0] = 0
        # latents torch.Size([1, 48, 21, 30, 52])
 
        # 6. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                current_model = self.transformer
                current_guidance_scale = guidance_scale

                # latent_model_input = (1 - first_frame_mask) * condition + first_frame_mask * latents
                latent_model_input = latents
                latent_model_input[:, :, 0:1, :, :] = img_latent.clone()
                latent_model_input = latent_model_input.to(transformer_dtype)

                temp_ts = (first_frame_mask[0][0][:, ::2, ::2] * t).flatten()
                timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)

                with current_model.cache_context("cond"):
                    noise_pred = current_model(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        control_video_latent=control_video_latent,
                        encoder_hidden_states_image=None,
                        attention_kwargs=None,
                        return_dict=False,
                        control_type=control_type,
                        null_condition=False,
                    )[0]

                if self.do_classifier_free_guidance:
                    with current_model.cache_context("uncond"):
                        noise_uncond = current_model(
                            hidden_states=latent_model_input,
                            timestep=timestep,
                            encoder_hidden_states=negative_prompt_embeds,
                            control_video_latent=control_video_latent,
                            encoder_hidden_states_image=None,
                            attention_kwargs=None,
                            return_dict=False,
                            control_type=control_type,
                            null_condition=False,
                        )[0]
                        noise_pred = noise_uncond + current_guidance_scale * (noise_pred - noise_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
    
                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        self._current_timestep = None

        # latents = (1 - first_frame_mask) * condition + first_frame_mask * latents
        latents[:, :, 0:1, :, :] = img_latent.clone()


        latents = latents.to(self.vae.dtype)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = latents / latents_std + latents_mean

        video_latents = video_latents.to(self.vae.dtype)
        control_video_latent = control_video_latent.to(self.vae.dtype)
        video_latents = video_latents.to(device=latents.device)
        control_video_latent = control_video_latent.to(device=latents.device)
        video_latents = video_latents / latents_std + latents_mean
        control_video_latent = control_video_latent / latents_std + latents_mean

        gt = self.vae.decode(video_latents, return_dict=False)[0]
        control_video = self.vae.decode(control_video_latent, return_dict=False)[0]
        control_video_latent_raw = control_video_latent_raw.to(self.vae.dtype).to(device=latents.device)
        control_video_latent_raw = control_video_latent_raw / latents_std + latents_mean
        control_video_raw = self.vae.decode(control_video_latent_raw, return_dict=False)[0]
        video = self.vae.decode(latents, return_dict=False)[0]

        video = self.video_processor.postprocess_video(video, output_type=output_type)
        gt = self.video_processor.postprocess_video(gt, output_type=output_type)
        control_video = self.video_processor.postprocess_video(control_video, output_type=output_type)
        control_video_raw = self.video_processor.postprocess_video(control_video_raw, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,), (gt,), (control_video,), (control_video_raw,)

        return BaseOutput(frames=video), BaseOutput(frames=gt), BaseOutput(frames=control_video), BaseOutput(frames=control_video_raw)
