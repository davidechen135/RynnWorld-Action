import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import torch
from accelerate.logging import get_logger
from safetensors.torch import load_file, save_file
from torch.utils.data import Dataset
from torchvision import transforms
from typing_extensions import override
import PIL
import json
from core.finetune.constants import LOG_LEVEL, LOG_NAME
import torch.nn.functional as F
from .utils import (
    preprocess_image_with_resize,
    preprocess_video_with_resize,
)
import random
import time
import os

if TYPE_CHECKING:
    from core.finetune.trainer import Trainer

# Must import after torch because this can sometimes lead to a nasty segmentation fault, or stack smashing error
# Very few bug reports but it happens. Look in decord Github issues for more relevant information.
import decord  # isort:skip

decord.bridge.set_bridge("torch")

logger = get_logger(LOG_NAME, LOG_LEVEL)


class I2VDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        max_num_frames: int,
        height: int,
        width: int,
        device: torch.device,
        trainer: "Trainer" = None,
    ) -> None:
        super().__init__()

        data_root = Path(data_root)
        if not data_root.exists():
            raise FileNotFoundError(f"Data root not found at: {data_root}")
        
        with open(data_root, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.prompts: List[str] = [item['caption'] for item in data]
        self.videos: List[Path] = [Path(item['video']) for item in data]
        self.images: List[Path] = [Path(item['image']) for item in data]

        self.trainer = trainer
        self.device = device
        if self.trainer is None:
            raise ValueError("A `trainer` object with `encode_video` and `encode_text` methods must be provided.")
        
        self.encode_video = trainer.encode_video
        self.encode_text = trainer.encode_text

        self.max_num_frames = max_num_frames
        self.height = height
        self.width = width
        self.__transforms = transforms.Compose([transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)])

        if not (len(self.videos) == len(self.prompts)):
            raise ValueError("Lengths of prompts and videos do not match.")

    def __len__(self) -> int:
        return len(self.videos)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        prompt = self.prompts[index]
        video_path = Path(self.videos[index])
        image_path = self.images[index]

        cache_dir = Path(os.environ.get('CACHE_DIR', 'data/cache'))
        filename = video_path.stem
        parent= video_path.parent.name

        video_latent_dir = cache_dir
        prompt_embeddings_dir = cache_dir / "prompt_embeddings"
        video_latent_dir.mkdir(parents=True, exist_ok=True)
        prompt_embeddings_dir.mkdir(parents=True, exist_ok=True)

        null_prompt = ""
        null_prompt_hash = str(hashlib.sha256(null_prompt.encode()).hexdigest())
        null_prompt_embedding_path = prompt_embeddings_dir / (null_prompt_hash + ".safetensors")
        if null_prompt_embedding_path.exists():
            null_prompt_embedding = load_file(null_prompt_embedding_path)["null_prompt_embedding"]
        else:
            null_prompt_embedding = self.encode_text(null_prompt)
            null_prompt_embedding = null_prompt_embedding.cpu().squeeze(0) # [1, L, C] -> [L, C]
            save_file({"null_prompt_embedding": null_prompt_embedding}, null_prompt_embedding_path)

        prompt_hash = str(hashlib.sha256(prompt.encode()).hexdigest())
        prompt_embedding_path = prompt_embeddings_dir / (prompt_hash + ".safetensors")
        # prompt_embedding_path = prompt_embeddings_dir / parent/ (filename + ".safetensors")
        if prompt_embedding_path.exists():
            prompt_embedding = load_file(prompt_embedding_path)["prompt_embedding"]
        else:
            prompt_embedding = self.encode_text(prompt)
            prompt_embedding = prompt_embedding.cpu().squeeze(0) # [1, L, C] -> [L, C]
            save_file({"prompt_embedding": prompt_embedding}, prompt_embedding_path)

        encoded_video_path = video_latent_dir / parent/ (filename + ".safetensors")
        if encoded_video_path.exists():
            encoded_video = load_file(encoded_video_path)["encoded_video"]
        else:
            encoded_video_path.parent.mkdir(parents=True, exist_ok=True)
            video_frames, _ = self.preprocess(video_path, None)
            print(f"Processing video: {encoded_video_path}")
            video_frames = self.video_transform(video_frames)
            video_frames = video_frames.permute(1, 0, 2, 3) 
            video_frames = video_frames.unsqueeze(0) 
            encoded_video = self.encode_video(video_frames).cpu().squeeze(0) # 移除 batch 维度
            save_file({"encoded_video": encoded_video}, encoded_video_path)

        _, image = self.preprocess(None, image_path)
        image = self.image_transform(image)

        # image torch.Size([3, 480, 720])
        # encoded_video torch.Size([16, 21, 60, 90])
        # null_prompt_embedding torch.Size([512, 4096])
        # prompt_embedding torch.Size([512, 4096])
        
        return {
            "image": image,
            "prompt_embedding": prompt_embedding,
            "encoded_video": encoded_video,
            "null_embedding": null_prompt_embedding,
        }

    def preprocess(self, video_path: Path | None, image_path: Path | None) -> Tuple[torch.Tensor, torch.Tensor]:
        video = None
        if video_path is not None:
            video = preprocess_video_with_resize(video_path, self.max_num_frames, self.height, self.width)
        
        image = None
        if image_path is not None:
            image = preprocess_image_with_resize(image_path, self.height, self.width)
        
        return video, image

    def video_transform(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: [F, C, H, W]
        return torch.stack([self.__transforms(f) for f in frames], dim=0)

    def image_transform(self, image: torch.Tensor) -> torch.Tensor:
        # image: [C, H, W]
        return self.__transforms(image)

class EgoVerseDataset22(Dataset):
    def __init__(
        self,
        data_root: str,
        cache_dir: str,
        max_num_frames: int,
        height: int,
        width: int,
        device: torch.device,
        trainer: "Trainer" = None,
        prompt: str = "",
    ) -> None:
        super().__init__()
        data_root = Path(data_root)
        if not data_root.exists():
            raise FileNotFoundError(f"Data root not found at: {data_root}")
        with open(data_root, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.video_latent_path: List[str] = [item['video_latent_path'] for item in data]
        self.text_embedding_path: List[str] = [item.get('text_embedding_path', None) for item in data]
        self.has_text_embeddings = self.text_embedding_path[0] is not None if len(self.text_embedding_path) > 0 else False

        from termcolor import cprint
        cprint(f"[EgoVerseDataset22] Loaded {len(self.video_latent_path)} samples (text_embeddings={'yes' if self.has_text_embeddings else 'no'})", 'green')
        # self.frame: List[str] = [item['frame'] for item in data]
        # self.videos: List[Path] = [Path(item['video']) for item in data]
        # self.control_videos: List[Path] = [Path(item['control_video']) for item in data]
        # self.images: List[Path] = [Path(item['image']) for item in data]
        # self.islong: List[bool] = [bool(item['islong']) for item in data]
        self.trainer = trainer
        self.device = device
        if self.trainer is None:
            raise ValueError("A `trainer` object with `encode_video` and `encode_text` methods must be provided.")
        self.encode_video = trainer.encode_video
        self.encode_text = trainer.encode_text
        cache_dir = Path(cache_dir)
        # self.video_latent_dir = cache_dir / "video_latents"
        self.prompt_embeddings_dir = cache_dir / "prompt_embeddings"
        # self.video_latent_dir.mkdir(parents=True, exist_ok=True)
        self.prompt_embeddings_dir.mkdir(parents=True, exist_ok=True)
        self.max_num_frames = max_num_frames
        self.height = height
        self.width = width
        self.__transforms = transforms.Compose([transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)])

        # null_prompt = "First-person egocentric perspective, high-quality video, human hands performing natural and dexterous interactions with the environment. Realistic physics, consistent lighting and shadows on hand-object contact, fluid motion, photorealistic textures, immersive atmosphere."
        null_prompt = prompt
        null_prompt_hash = str(hashlib.sha256(null_prompt.encode()).hexdigest())
        null_prompt_embedding_path = Path(os.environ.get('PROMPT_EMBEDDINGS_DIR', 'data/prompt_embeddings')) / (null_prompt_hash + ".safetensors")
        if null_prompt_embedding_path.exists():
            self.null_prompt_embedding = load_file(null_prompt_embedding_path)["null_prompt_embedding"]
        else:
            with torch.no_grad():
                null_prompt_embedding = self.encode_text(null_prompt)
                self.null_prompt_embedding = null_prompt_embedding.cpu().squeeze(0) # [1, L, C] -> [L, C]
            save_file({"null_prompt_embedding": self.null_prompt_embedding}, null_prompt_embedding_path)

        cprint(f"✅  Number of slice: {len(self.video_latent_path)}", 'green')

        self.make_null_video()

        # try:
        #     self.null_prompt_embedding = load_file(null_prompt_embedding_path)["null_prompt_embedding"]
        #     cprint(f"✅  Null prompt embedding loaded from {null_prompt_embedding_path}", 'green')
        # except Exception as e:
        #     cprint(f"❌  Failed to load null prompt: {e}", 'red')
        #     raise e

    def __len__(self) -> int:
        return len(self.video_latent_path)

    def make_null_video(self):
        batch_size = 1
        num_channels = 3
        height = 480
        width = 832
        short_video = torch.zeros((25, height, width, 3), dtype=torch.float32)
        long_video = torch.zeros((81, height, width, 3), dtype=torch.float32)
        short_video = short_video.permute(0, 3, 1, 2) 
        long_video = long_video.permute(0, 3, 1, 2)
        long_video = self.video_transform(long_video.float())
        short_video = self.video_transform(short_video.float())
        vae_dtype = self.trainer.components.vae.dtype

        long_video = long_video.permute(1, 0, 2, 3).unsqueeze(0).to(self.device, dtype=vae_dtype)
        short_video = short_video.permute(1, 0, 2, 3).unsqueeze(0).to(self.device, dtype=vae_dtype)

        with torch.no_grad():
            latents_mean = torch.tensor(self.trainer.components.vae.config.latents_mean).view(1, self.trainer.components.vae.config.z_dim, 1, 1, 1).to(self.device)
            latents_std = torch.tensor(self.trainer.components.vae.config.latents_std).view(1, self.trainer.components.vae.config.z_dim, 1, 1, 1).to(self.device)

            long_video_latents = self.trainer.components.vae.encode(long_video).latent_dist.mode()
            long_video_latents = (long_video_latents - latents_mean) / latents_std

            short_video_latents = self.trainer.components.vae.encode(short_video).latent_dist.mode()
            short_video_latents = (short_video_latents - latents_mean) / latents_std

        self.short_video_latents = short_video_latents.squeeze(0).cpu()
        self.long_video_latents = long_video_latents.squeeze(0).cpu()

    def __getitem__(self, index: int) -> Dict[str, Any]:
        video_latent_path = self.video_latent_path[index]
        video_latent_path = Path(video_latent_path)
        # frame = self.frame[index]
        max_retries = 5
        cache_data = None

        for i in range(max_retries):
            try:
                cache_data = load_file(video_latent_path)
                break  
            except Exception as e:
                rank = os.environ.get('RANK', '0')
                if i < max_retries - 1:
                    print(f"[Rank {rank}] Warning: Load failed (attempt {i+1}/{max_retries}). Retrying... Path: {video_latent_path}")
                    time.sleep(1) # 等待1秒后重试
                else:
                    print(f"[Rank {rank}] ERROR: Permanent failure on {video_latent_path}. Skipping to random sample.")
                    return self.__getitem__(random.randint(0, len(self) - 1))
        try:
            encoded_video = cache_data["video_latents"]
            encoded_control_video = cache_data["control_video_latents"]
            img_latent = cache_data["img_latent"]
        except Exception as e:
            print(f"Error parsing keys in {video_latent_path}: {e}")
            return self.__getitem__(random.randint(0, len(self) - 1))

        if encoded_video.shape[1] == 7:
            null_control_video = self.short_video_latents
        elif encoded_video.shape[1] == 21:
            null_control_video = self.long_video_latents

        ret = {
            "null_embedding": self.null_prompt_embedding,
            "img_latent": img_latent,
            "encoded_video": encoded_video,
            "control_video": encoded_control_video,
            "null_control_video": null_control_video,
        }

        if self.has_text_embeddings:
            text_embedding_path = Path(self.text_embedding_path[index])
            try:
                text_data = load_file(text_embedding_path)
                ret["prompt_embedding"] = text_data["text_embedding"]
            except Exception as e:
                ret["prompt_embedding"] = self.null_prompt_embedding

        return ret

    def video_transform(self, frames: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.__transforms(f) for f in frames], dim=0)

    def load_video_chunk(self, video_path, start_frame, num_frames, height, width):
        video_reader = decord.VideoReader(uri=str(video_path), width=width, height=height)
        indices = list(range(start_frame, start_frame + num_frames))
        frames = video_reader.get_batch(indices)
        frames = frames.permute(0, 3, 1, 2)
        return frames

    def preprocess_short_video(self, video_path, max_num_frames, height, width):
        video_reader = decord.VideoReader(uri=str(video_path), width=width, height=height)
        frames_raw = video_reader.get_batch(range(len(video_reader))) # [F, H, W, C]
        
        frames_permuted = frames_raw.float().permute(3, 0, 1, 2)
        
        interpolated_frames = F.interpolate(
            frames_permuted.unsqueeze(0),       
            size=(max_num_frames, height, width),
            mode='trilinear',                 
            align_corners=False               
        )
        final_frames = interpolated_frames.squeeze(0).permute(1, 2, 3, 0)
        return final_frames.byte() 
    
    def image_transform(self, image: torch.Tensor) -> torch.Tensor:
        return self.__transforms(image)


class EgoVerseDataset22SFT(Dataset):
    def __init__(
        self,
        data_root: str,
        cache_dir: str,
        max_num_frames: int,
        height: int,
        width: int,
        device: torch.device,
        trainer: "Trainer" = None,
        prompt: str = "",
    ) -> None:
        super().__init__()
        data_root = Path(data_root)
        if not data_root.exists():
            raise FileNotFoundError(f"Data root not found at: {data_root}")
        with open(data_root, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.video_latent_path: List[str] = [item['video_latent_path'] for item in data]
        self.text_embedding_path: List[str] = [item['text_embedding_path'] for item in data]

        from termcolor import cprint
        cprint(f"[EgoVerseDataset22SFT] Loaded {len(self.video_latent_path)} samples", 'green')

        self.trainer = trainer
        self.device = device
        if self.trainer is None:
            raise ValueError("A `trainer` object with `encode_text` method must be provided.")
        self.encode_text = trainer.encode_text

        self.max_num_frames = max_num_frames
        self.height = height
        self.width = width

        # Precompute null prompt embedding for CFG
        cache_dir = Path(cache_dir)
        self.prompt_embeddings_dir = cache_dir / "prompt_embeddings"
        self.prompt_embeddings_dir.mkdir(parents=True, exist_ok=True)

        null_prompt = ""
        null_prompt_hash = str(hashlib.sha256(null_prompt.encode()).hexdigest())
        null_prompt_embedding_path = self.prompt_embeddings_dir / (null_prompt_hash + ".safetensors")
        if null_prompt_embedding_path.exists():
            self.null_prompt_embedding = load_file(null_prompt_embedding_path)["null_prompt_embedding"]
        else:
            with torch.no_grad():
                null_prompt_embedding = self.encode_text(null_prompt)
                self.null_prompt_embedding = null_prompt_embedding.cpu().squeeze(0)
            save_file({"null_prompt_embedding": self.null_prompt_embedding}, null_prompt_embedding_path)

        cprint(f"[EgoVerseDataset22SFT] null_prompt_embedding shape: {self.null_prompt_embedding.shape}", 'green')

    def __len__(self) -> int:
        return len(self.video_latent_path)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        video_latent_path = Path(self.video_latent_path[index])
        text_embedding_path = Path(self.text_embedding_path[index])

        max_retries = 5
        cache_data = None
        for i in range(max_retries):
            try:
                cache_data = load_file(video_latent_path)
                break
            except Exception as e:
                rank = os.environ.get('RANK', '0')
                if i < max_retries - 1:
                    print(f"[Rank {rank}] Warning: Load failed (attempt {i+1}/{max_retries}). Path: {video_latent_path}")
                    time.sleep(1)
                else:
                    print(f"[Rank {rank}] ERROR: Permanent failure on {video_latent_path}. Skipping.")
                    return self.__getitem__(random.randint(0, len(self) - 1))

        try:
            encoded_video = cache_data["video_latents"]
            img_latent = cache_data["img_latent"]
        except Exception as e:
            print(f"Error parsing keys in {video_latent_path}: {e}")
            return self.__getitem__(random.randint(0, len(self) - 1))

        try:
            text_data = load_file(text_embedding_path)
            text_embedding = text_data["text_embedding"]
        except Exception as e:
            print(f"Error loading text embedding {text_embedding_path}: {e}")
            return self.__getitem__(random.randint(0, len(self) - 1))

        return {
            "encoded_video": encoded_video,
            "img_latent": img_latent,
            "prompt_embedding": text_embedding,
            "null_embedding": self.null_prompt_embedding,
        }
