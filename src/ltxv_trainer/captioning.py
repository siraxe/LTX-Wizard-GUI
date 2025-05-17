import itertools  # noqa: I001
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Union

import torch
from diffusers import BitsAndBytesConfig
from transformers import AutoModel, AutoProcessor, LlavaNextVideoForConditionalGeneration
import numpy as np

# Should be imported after `torch` to avoid compatibility issues.import decord
import decord  # type: ignore
from ltxv_trainer.utils import open_image_as_srgb

decord.bridge.set_bridge("torch")

DEFAULT_VLM_CAPTION_INSTRUCTION = "Shortly describe the content of this video in two sentences."


class CaptionerType(str, Enum):
    """Enum for different types of video captioners."""

    LLAVA_NEXT_7B = "llava_next_7b"


def create_captioner(
    captioner_type: CaptionerType,
    models_dir: Union[str, Path] = "models",  # Added models_dir
    **kwargs,
) -> "MediaCaptioningModel":
    """Factory function to create a video captioner.

    Args:
        captioner_type: The type of captioner to create
        models_dir: The directory to store/load models from.
        **kwargs: Additional arguments to pass to the captioner constructor

    Returns:
        An instance of a MediaCaptioningModel
    """
    if captioner_type == CaptionerType.LLAVA_NEXT_7B:
        return TransformersVlmCaptioner(
            model_id="llava-hf/LLaVA-NeXT-Video-7B-hf",
            models_dir=models_dir,  # Pass models_dir
            **kwargs,
        )
    else:
        raise ValueError(f"Unsupported captioner type: {captioner_type}")


class MediaCaptioningModel(ABC):
    """Abstract base class for video and image captioning models."""

    @abstractmethod
    def caption(self, path: Union[str, Path]) -> str:
        """Generate a caption for the given video or image.

        Args:
            path: Path to the video/image file to caption

        Returns:
            A string containing the generated caption
        """

    @staticmethod
    def _read_video_frames(video_path: Union[str, Path], sampling_factor: int = 8) -> torch.Tensor:
        """Read frames from a video file."""
        video_reader = decord.VideoReader(uri=str(video_path))
        total_frames = len(video_reader)
        indices = list(range(0, total_frames, total_frames // sampling_factor))
        frames = video_reader.get_batch(indices)
        return frames

    @staticmethod
    def _read_image(image_path: Union[str, Path]) -> torch.Tensor:
        """Read an image file and convert to tensor format."""
        image = open_image_as_srgb(image_path)
        image_tensor = torch.from_numpy(np.array(image))
        return image_tensor

    @staticmethod
    def _is_image_file(path: Union[str, Path]) -> bool:
        """Check if the file is an image based on extension."""
        return str(path).lower().endswith((".png", ".jpg", ".jpeg", ".heic", ".heif", ".webp"))

    @staticmethod
    def _clean_raw_caption(caption: str) -> str:
        """Clean up the raw caption."""
        start = ["The", "This"]
        kind = ["video", "image", "scene", "animated sequence"]
        act = ["displays", "shows", "features", "depicts", "presents", "showcases", "captures"]

        for x, y, z in itertools.product(start, kind, act):
            caption = caption.replace(f"{x} {y} {z} ", "", 1)

        return caption


class TransformersVlmCaptioner(MediaCaptioningModel):
    """Video and image captioning model using models implemented in HuggingFace's `transformers`."""

    def __init__(
        self,
        model_id: str = "llava-hf/LLaVA-NeXT-Video-7B-hf",
        models_dir: Union[str, Path] = "models",  # Added models_dir with default
        device: str | torch.device = None,
        use_8bit: bool = False,
        vlm_instruction: str = DEFAULT_VLM_CAPTION_INSTRUCTION,
    ):
        """Initialize the captioner.

        Args:
            model_id: HuggingFace model ID
            models_dir: Directory to store/load HuggingFace models.
            device: torch.device to use for the model
            use_8bit: Whether to load the model in 8-bit.
            vlm_instruction: The instruction prompt for the VLM.
        """
        self.device = torch.device(device or "cuda" if torch.cuda.is_available() else "cpu")
        self.vlm_instruction = vlm_instruction
        self.models_dir = Path(models_dir)  # Store as Path object
        self._load_model(model_id, use_8bit=use_8bit, models_dir=self.models_dir) # Pass models_dir

    def caption(
        self,
        path: Union[str, Path],
        frames_sampling_factor: int = 8,
        clean_caption: bool = True,
    ) -> str:
        """Generate a caption for the given video or image.

        Args:
            path: Path to the video/image file to caption
            frames_sampling_factor: Factor to sample frames from the video, i.e. every
                `frames_sampling_factor`-th frame will be used (ignored for images).
            clean_caption: Whether to clean up the raw caption by removing common VLM patterns.

        Returns:
            A string containing the generated caption
        """
        # Determine if input is image or video
        is_image = self._is_image_file(path)

        # Read input file
        media = self._read_image(path) if is_image else self._read_video_frames(path, frames_sampling_factor)
        # Prepare inputs
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.vlm_instruction},
                    {"type": "video" if not is_image else "image"},
                ],
            },
        ]

        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(
            text=prompt,
            videos=media if not is_image else None,
            images=media if is_image else None,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        # Generate caption
        output_tokens = self.model.generate(**inputs, max_new_tokens=200, do_sample=False)
        output = self.processor.decode(output_tokens[0], skip_special_tokens=True)
        caption_raw = output.split("ASSISTANT: ")[1]

        # Clean up caption
        caption = self._clean_raw_caption(caption_raw) if clean_caption else caption_raw
        return caption

    def _load_model(self, model_id: str, use_8bit: bool, models_dir: Path) -> None:
        model_name = model_id.split("/")[-1]
        local_model_path = models_dir / model_name

        if model_id == "llava-hf/LLaVA-NeXT-Video-7B-hf":
            model_cls = LlavaNextVideoForConditionalGeneration
        else:
            # For other models, ensure their specific class is used or AutoModelForConditionalGeneration
            # For now, this follows the original logic for non-LLAVA IDs.
            model_cls = AutoModel

        quantization_config = BitsAndBytesConfig(load_in_8bit=True) if use_8bit else None

        # Attempt to load from local path first
        if local_model_path.exists() and local_model_path.is_dir():
            try:
                print(f"Attempting to load model from local path: {local_model_path}")  # noqa: T201
                self.model = model_cls.from_pretrained(
                    local_model_path,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    quantization_config=quantization_config,
                    device_map=self.device.type,
                )
                self.processor = AutoProcessor.from_pretrained(local_model_path)
                print(f"Successfully loaded model and processor from {local_model_path}")  # noqa: T201
                return  # Successfully loaded
            except Exception as e:
                print(f"Failed to load model from {local_model_path}: {e}. Will attempt to download from Hugging Face Hub.")  # noqa: T201

        # If not found locally or local loading failed, download from Hub
        print(f"Model not found at {local_model_path} or local loading failed. Downloading '{model_id}' from Hugging Face Hub.")  # noqa: T201
        try:
            # Ensure the target directory for saving models exists
            models_dir.mkdir(parents=True, exist_ok=True)

            self.model = model_cls.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                quantization_config=quantization_config,
                device_map=self.device.type,
            )
            self.processor = AutoProcessor.from_pretrained(model_id)
            print(f"Successfully downloaded '{model_id}'.")  # noqa: T201

            # Save the downloaded model and processor to the local path for future use
            try:
                print(f"Saving model and processor to {local_model_path}...")  # noqa: T201
                self.model.save_pretrained(local_model_path)
                self.processor.save_pretrained(local_model_path)
                print(f"Successfully saved model and processor to {local_model_path}.")  # noqa: T201
            except Exception as e:
                # Log if saving fails, but the model is already loaded in memory, so a warning is sufficient.
                print(f"Warning: Failed to save model/processor to {local_model_path}: {e}")  # noqa: T201

        except Exception as e:
            print(f"Fatal error: Could not download or load model '{model_id}' from Hugging Face Hub: {e}")  # noqa: T201
            # Depending on desired behavior, could set model/processor to None or raise
            raise # Re-raise the exception if download or initial load from hub fails


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
