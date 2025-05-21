from enum import Enum
from pathlib import Path
from typing import Protocol, Union

import torch
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    LlavaNextVideoForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)
import numpy as np
import imageio.v3 as iio
import decord
from huggingface_hub import snapshot_download
from huggingface_hub.utils import HFValidationError

decord.bridge.set_bridge("torch")

DEFAULT_VLM_CAPTION_INSTRUCTION = (
    "Shortly describe the content of this video in one or two sentences."
)

class CaptionerType(str, Enum):
    """Enum for different types of video captioners."""

    LLAVA_NEXT_7B = "llava_next_7b"
    QWEN_25_VL = "qwen_25_vl"


# Add this constant, matching model_loader.py
LOCAL_MODELS_CACHE = Path("models")
LOCAL_MODELS_CACHE.mkdir(parents=True, exist_ok=True)


def create_captioner(
    captioner_type: CaptionerType,
    device: str = None,
    use_8bit: bool = True,  # Default True, matches TransformersVlmCaptioner
    vlm_instruction: str = DEFAULT_VLM_CAPTION_INSTRUCTION,
    llava_model_id_or_path: str = "llava-hf/LLaVA-NeXT-Video-7B-hf",
    qwen_model_id_or_path: str = "Qwen/Qwen2.5-VL-7B-Instruct",  # Default to HF ID
) -> "MediaCaptioningModel":
    """Factory function to create a media captioning model."""
    return TransformersVlmCaptioner(
        model_type=captioner_type,
        device=device,
        use_8bit=use_8bit,
        vlm_instruction=vlm_instruction,
        llava_model_id_or_path=llava_model_id_or_path,
        qwen_model_id_or_path=qwen_model_id_or_path,
    )


class MediaCaptioningModel(Protocol):
    def caption(
        self,
        path: Union[str, Path],
        fps: int = 3,
        clean_caption: bool = True,
        max_new_tokens: int = 100,
    ) -> str:
        ...


class TransformersVlmCaptioner(MediaCaptioningModel):
    """Video and image captioning model using models implemented in HuggingFace's `transformers`."""

    def __init__(
        self,
        model_type: CaptionerType,
        device: str = None,
        use_8bit: bool = True,  # Default to True as requested
        vlm_instruction: str = DEFAULT_VLM_CAPTION_INSTRUCTION,
        llava_model_id_or_path: str = "llava-hf/LLaVA-NeXT-Video-7B-hf",
        qwen_model_id_or_path: str = "Qwen/Qwen2.5-VL-7B-Instruct",  # Default to HF ID
    ):
        self.model_type = model_type
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.vlm_instruction = vlm_instruction
        self.model = None
        self.processor = None

        model_torch_dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        if self.model_type == CaptionerType.LLAVA_NEXT_7B:
            repo_id = llava_model_id_or_path
            local_cache_dir = LOCAL_MODELS_CACHE / repo_id.replace("/", "--")

            load_kwargs = {
                "quantization_config": BitsAndBytesConfig(load_in_8bit=True) if use_8bit else None,
                "torch_dtype": model_torch_dtype if not use_8bit else None,
                "device_map": self.device.type if use_8bit else None, # Use device_map for 8-bit, otherwise handle with .to()
                "low_cpu_mem_usage": True, # Often helpful
            }
            load_kwargs_local = load_kwargs.copy()
            load_kwargs_local["local_files_only"] = True

            try:
                self.model = LlavaNextVideoForConditionalGeneration.from_pretrained(
                    local_cache_dir,
                    **load_kwargs_local
                )
                self.processor = AutoProcessor.from_pretrained(local_cache_dir, use_fast=True, local_files_only=True)
                print(f"Loaded LLaVA model from local cache: {local_cache_dir}")

            except (OSError, HFValidationError) as e:
                 print(f"LLaVA model not found in local cache ({local_cache_dir}).")
                 print(f"Attempting to download {repo_id} from Hugging Face...")
                 snapshot_download(
                     repo_id=repo_id,
                     local_dir=local_cache_dir,
                     # force_download=True # Uncomment to force fresh download
                 )
                 print(f"Download of {repo_id} complete.")
                 print(f"Downloaded LLaVA model to local cache: {local_cache_dir}")
                 self.model = LlavaNextVideoForConditionalGeneration.from_pretrained(
                     local_cache_dir,
                     **load_kwargs # Use base load_kwargs here
                 )
                 self.processor = AutoProcessor.from_pretrained(local_cache_dir, use_fast=True, local_files_only=True)

            if not use_8bit and self.model is not None and load_kwargs.get("device_map") is None:
                 self.model.to(self.device)

            if self.processor is not None:
                print(f"Updating LLaVA processor config format in {local_cache_dir}...")
                self.processor.save_pretrained(local_cache_dir)
                print("Processor config updated.")

        elif self.model_type == CaptionerType.QWEN_25_VL:
            repo_id = qwen_model_id_or_path
            local_cache_dir = LOCAL_MODELS_CACHE / repo_id.replace("/", "--")

            load_kwargs = {
                "quantization_config": BitsAndBytesConfig(load_in_8bit=True) if use_8bit else None,
                "torch_dtype": model_torch_dtype if not use_8bit else None,
                "device_map": self.device.type if use_8bit else None, # Use device_map for 8-bit
                "trust_remote_code": True, # Qwen specific
                "low_cpu_mem_usage": True, # Often helpful
            }
            load_kwargs_local = load_kwargs.copy()
            load_kwargs_local["local_files_only"] = True

            try:
                 self.processor = AutoProcessor.from_pretrained(local_cache_dir, use_fast=True, local_files_only=True, trust_remote_code=True)
                 self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    local_cache_dir,
                    **load_kwargs_local # Use local kwargs for initial load
                 )
                 print(f"Loaded Qwen model from local cache: {local_cache_dir}")

            except (OSError, HFValidationError) as e:
                 print(f"Qwen model not found in local cache ({local_cache_dir}).")
                 print(f"Attempting to download {repo_id} from Hugging Face...")
                 snapshot_download(
                     repo_id=repo_id,
                     local_dir=local_cache_dir,
                     trust_remote_code=True, # Need this for download too? Probably not, but included for safety.
                     # force_download=True # Uncomment to force fresh download
                 )
                 print(f"Download of {repo_id} complete.")
                 self.processor = AutoProcessor.from_pretrained(local_cache_dir, use_fast=True, local_files_only=True, trust_remote_code=True)
                 self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                     local_cache_dir,
                     **load_kwargs # Use base load_kwargs
                 )

            if not use_8bit and self.model is not None and load_kwargs.get("device_map") is None:
                 self.model.to(self.device)

            if self.processor is not None:
                print(f"Updating Qwen processor config format in {local_cache_dir}...")
                self.processor.save_pretrained(local_cache_dir)
                print("Processor config updated.")

        else:
             raise ValueError(f"Unsupported model_type: {self.model_type}")

        if self.model is None or self.processor is None:
             raise RuntimeError(f"Failed to load model components for {self.model_type}")

    def _is_image_file(self, path: Union[str, Path]) -> bool:
        """Check if the path points to a common image file."""
        return str(path).lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif"))

    def _clean_raw_caption(self, raw_caption: str) -> str:
        """Clean up the raw caption from the VLM by removing common patterns."""
        if self.vlm_instruction and raw_caption.startswith(self.vlm_instruction):
            caption = raw_caption[len(self.vlm_instruction) :].strip()
        else:
            caption = raw_caption.strip()

        assistant_prefix = "ASSISTANT:"
        if caption.startswith(assistant_prefix):
            caption = caption[len(assistant_prefix):].strip()
        
        parts = caption.split("ASSISTANT:")
        if len(parts) > 1:
            caption = parts[-1].strip()

        if caption.startswith('"') and caption.endswith('"'):
            caption = caption[1:-1]
        if caption.startswith("'") and caption.endswith("'"):
            caption = caption[1:-1]

        return caption.strip()

    def caption(
        self,
        path: Union[str, Path],
        fps: int = 3,
        clean_caption: bool = True,
        max_new_tokens: int = 100,
    ) -> str:
        is_image = self._is_image_file(path)
        media_data = None

        if is_image:
            media_data = iio.imread(str(path))
            if media_data.ndim == 3 and media_data.shape[2] == 4: # RGBA to RGB
                media_data = media_data[:, :, :3]
        else:  # It's a video
            video_reader = decord.VideoReader(str(path))
            num_frames_total = len(video_reader)

            if num_frames_total == 0:
                raise ValueError(f"Video file {path} has 0 frames. Cannot process.")

            if self.model_type == CaptionerType.LLAVA_NEXT_7B:
                num_frames_to_sample = 8  # LLaVA-NeXT is trained on 8 frames
                if num_frames_total <= num_frames_to_sample:
                    indices = np.arange(num_frames_total)
                else:
                    indices = np.linspace(0, num_frames_total - 1, num_frames_to_sample, dtype=int)
                media_data = video_reader.get_batch(indices).numpy()
            elif self.model_type == CaptionerType.QWEN_25_VL:
                media_data = video_reader.get_batch(np.arange(num_frames_total)).numpy()
            else:  # Fallback for other/undefined model types
                media_data = video_reader.get_batch(np.arange(num_frames_total)).numpy()

        if media_data is None:
            raise ValueError(f"Could not load media from path: {path}")

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.vlm_instruction},
                    {"type": "image" if is_image else "video"}, # Use correct type
                ],
            },
        ]

        if self.model_type == CaptionerType.QWEN_25_VL and hasattr(self.processor, 'tokenizer'):
             prompt = self.processor.tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
             inputs = self.processor(
                 text=[prompt],
                 videos=media_data if not is_image else None,
                 images=media_data if is_image else None,
                 padding=True,
                 truncation=True,
                 return_tensors="pt",
                 video_fps=fps if not is_image else None,
             ).to(self.device)
        else: # LLaVA and others
             prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
             inputs = self.processor(
                 text=prompt,
                 videos=media_data if not is_image else None,
                 images=media_data if is_image else None,
                 padding=True,
                 truncation=True,
                 return_tensors="pt",
                 video_fps=fps if not is_image else None,
             ).to(self.device)

        raw_caption = ""
        generated_ids = None

        inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        if self.model_type == CaptionerType.LLAVA_NEXT_7B:
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=4,
                no_repeat_ngram_size=2,
                early_stopping=True,
            )
        elif self.model_type == CaptionerType.QWEN_25_VL:
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens
            )
        else:
            raise ValueError(f"Unsupported model_type for caption generation: {self.model_type}")

        if generated_ids is not None and inputs is not None and 'input_ids' in inputs and inputs['input_ids'] is not None:
             input_token_len = inputs['input_ids'].shape[1]
             generated_ids_trimmed = generated_ids[:, input_token_len:]
        else:
             print("Warning: Could not determine input token length for trimming.")
             generated_ids_trimmed = generated_ids

        if generated_ids_trimmed is not None:
            if self.model_type == CaptionerType.QWEN_25_VL:
                raw_caption = self.processor.tokenizer.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0]
            else: # LLaVA and potentially others
                raw_caption = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True
                )[0]
        else:
             raw_caption = ""

        if clean_caption:
            final_caption = self._clean_raw_caption(raw_caption)
        else:
            final_caption = raw_caption.strip()

        return final_caption


def example() -> None:
    import sys

    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <video_path>")  # noqa: T201
        sys.exit(1)

    model = TransformersVlmCaptioner()
    caption = model.caption(sys.argv[1])
    print(caption)  # noqa: T201


if __name__ == "__main__":
    example()
