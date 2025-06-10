import json
import tempfile
from enum import Enum
from pathlib import Path
from typing import Union
from urllib.parse import urlparse
import os
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.utils import HfHubHTTPError, HFValidationError

import torch
from diffusers import (
    AutoencoderKLLTXVideo,
    BitsAndBytesConfig,
    FlowMatchEulerDiscreteScheduler,
    LTXVideoTransformer3DModel,
)
from pydantic import BaseModel, ConfigDict
from transformers import T5EncoderModel, T5Tokenizer

# The main HF repo to load scheduler, tokenizer, and text encoder from
HF_MAIN_REPO = "Lightricks/LTX-Video"
LOCAL_MODELS_CACHE = Path("models")
LOCAL_MODELS_CACHE.mkdir(parents=True, exist_ok=True)


class LtxvModelVersion(str, Enum):
    """Available LTXV model versions."""

    LTXV_2B_090 = "LTXV_2B_0.9.0"
    LTXV_2B_091 = "LTXV_2B_0.9.1"
    LTXV_2B_095 = "LTXV_2B_0.9.5"
    LTXV_2B_096_DEV = "LTXV_2B_0.9.6_DEV"
    LTXV_2B_096_DISTILLED = "LTXV_2B_0.9.6_DISTILLED"
    LTXV_13B_097_DEV = "LTXV_13B_097_DEV"
    LTXV_13B_097_DEV_FP8 = "LTXV_13B_097_DEV_FP8"
    LTXV_13B_097_DISTILLED = "LTXV_13B_097_DISTILLED"
    LTXV_13B_097_DISTILLED_FP8 = "LTXV_13B_097_DISTILLED_FP8"

    def __str__(self) -> str:
        """Return the version string."""
        return self.value

    @classmethod
    def latest(cls) -> "LtxvModelVersion":
        """Get the latest available version."""
        return cls.LTXV_13B_097_DISTILLED

    @property
    def hf_repo(self) -> str:
        """Get the HuggingFace repo for this version."""
        match self:
            case LtxvModelVersion.LTXV_2B_090:
                return "Lightricks/LTX-Video"
            case LtxvModelVersion.LTXV_2B_091:
                return "Lightricks/LTX-Video-0.9.1"
            case LtxvModelVersion.LTXV_2B_095:
                return "Lightricks/LTX-Video-0.9.5"
            case LtxvModelVersion.LTXV_2B_096_DEV:
                raise ValueError("LTXV_2B_096_DEV does not have a HuggingFace repo")
            case LtxvModelVersion.LTXV_2B_096_DISTILLED:
                raise ValueError("LTXV_2B_096_DISTILLED does not have a HuggingFace repo")
            case LtxvModelVersion.LTXV_13B_097_DEV:
                raise ValueError("LTXV_13B_097_DEV does not have a HuggingFace repo")
            case LtxvModelVersion.LTXV_13B_097_DEV_FP8:
                raise ValueError("LTXV_13B_097_DEV_FP8 does not have a HuggingFace repo")
            case LtxvModelVersion.LTXV_13B_097_DISTILLED:
                raise ValueError("LTXV_13B_097_DISTILLED does not have a HuggingFace repo")
            case LtxvModelVersion.LTXV_13B_097_DISTILLED_FP8:
                raise ValueError("LTXV_13B_097_DISTILLED_FP8 does not have a HuggingFace repo")
        raise ValueError(f"Unknown version: {self}")

    @property
    def safetensors_url(self) -> str:
        """Get the safetensors URL for this version."""
        match self:
            case LtxvModelVersion.LTXV_2B_090:
                return "https://huggingface.co/Lightricks/LTX-Video/blob/main/ltx-video-2b-v0.9.safetensors"
            case LtxvModelVersion.LTXV_2B_091:
                return "https://huggingface.co/Lightricks/LTX-Video/blob/main/ltx-video-2b-v0.9.1.safetensors"
            case LtxvModelVersion.LTXV_2B_095:
                return "https://huggingface.co/Lightricks/LTX-Video/blob/main/ltx-video-2b-v0.9.5.safetensors"
            case LtxvModelVersion.LTXV_2B_096_DEV:
                return "https://huggingface.co/Lightricks/LTX-Video/blob/main/ltxv-2b-0.9.6-dev-04-25.safetensors"
            case LtxvModelVersion.LTXV_2B_096_DISTILLED:
                return "https://huggingface.co/Lightricks/LTX-Video/blob/main/ltxv-2b-0.9.6-distilled-04-25.safetensors"
            case LtxvModelVersion.LTXV_13B_097_DEV:
                return "https://huggingface.co/Lightricks/LTX-Video/blob/main/ltxv-13b-0.9.7-dev.safetensors"
            case LtxvModelVersion.LTXV_13B_097_DEV_FP8:
                return "https://huggingface.co/Lightricks/LTX-Video/blob/main/ltxv-13b-0.9.7-dev-fp8.safetensors"
            case LtxvModelVersion.LTXV_13B_097_DISTILLED:
                return "https://huggingface.co/Lightricks/LTX-Video/blob/main/ltxv-13b-0.9.7-distilled.safetensors"
            case LtxvModelVersion.LTXV_13B_097_DISTILLED_FP8:
                return "https://huggingface.co/Lightricks/LTX-Video/blob/main/ltxv-13b-0.9.7-distilled-fp8.safetensors"
        raise ValueError(f"Unknown version: {self}")


# Type for model sources - can be:
# 1. HuggingFace repo ID (str)
# 2. Local path (str or Path)
# 3. Direct version specification (LtxvModelVersion)
ModelSource = Union[str, Path, LtxvModelVersion]


class LtxvModelComponents(BaseModel):
    """Container for all LTXV model components."""

    scheduler: FlowMatchEulerDiscreteScheduler
    tokenizer: T5Tokenizer
    text_encoder: T5EncoderModel
    vae: AutoencoderKLLTXVideo
    transformer: LTXVideoTransformer3DModel

    model_config = ConfigDict(arbitrary_types_allowed=True)


def load_scheduler() -> FlowMatchEulerDiscreteScheduler:
    """
    Load the Flow Matching scheduler component from the main HF repo.
    It will first check the local `models/` cache, then download if not found.

    Returns:
        Loaded scheduler
    """
    repo_id = HF_MAIN_REPO
    subfolder = "scheduler"
    # download_destination_parent is where snapshot_download will place the 'scheduler' folder from the repo.
    download_destination_parent = LOCAL_MODELS_CACHE / (repo_id.replace("/", "--") + f"--{subfolder}-root")
    actual_model_files_dir = download_destination_parent / subfolder # e.g. models/Lightricks--LTX-Video--scheduler-root/scheduler/

    try:
        # Attempt to load from where files are expected after download
        return FlowMatchEulerDiscreteScheduler.from_pretrained(
            actual_model_files_dir,
            local_files_only=True,
        )
    except (OSError, HFValidationError): 
        # Download: this will place e.g. repo/scheduler/* into download_destination_parent/scheduler/*
        snapshot_download(
            repo_id=repo_id,
            allow_patterns=[f"{subfolder}/*", f"{subfolder}/**/*"], # Get files and files in subdirectories of the target subfolder
            local_dir=download_destination_parent, 
            local_dir_use_symlinks=False,
            # force_download=True # Uncomment to force fresh download
        )
        # Load from the now populated local cache
        return FlowMatchEulerDiscreteScheduler.from_pretrained(
            actual_model_files_dir, 
            local_files_only=True,
        )


def load_tokenizer() -> T5Tokenizer:
    """
    Load the T5 tokenizer component from the main HF repo.
    It will first check the local `models/` cache, then download if not found.

    Returns:
        Loaded tokenizer
    """
    repo_id = HF_MAIN_REPO
    subfolder = "tokenizer"
    download_destination_parent = LOCAL_MODELS_CACHE / (repo_id.replace("/", "--") + f"--{subfolder}-root")
    actual_model_files_dir = download_destination_parent / subfolder

    try:
        # Attempt to load from the directory
        return T5Tokenizer.from_pretrained(
            str(actual_model_files_dir),
            local_files_only=True,
            legacy=False,
        )
    except (OSError, HFValidationError):
        # If loading from local files fails, download them
        snapshot_download(
            repo_id=repo_id,
            allow_patterns=[f"{subfolder}/*", f"{subfolder}/**/*"],
            local_dir=download_destination_parent,
            local_dir_use_symlinks=False,
        )
        # After download, try loading again from the directory
        return T5Tokenizer.from_pretrained(
            str(actual_model_files_dir),
            local_files_only=True,
            legacy=False,
        )


def load_text_encoder(*, load_in_8bit: bool = False) -> T5EncoderModel:
    """
    Load the T5 text encoder component from the main HF repo.
    It will first check the local `models/` cache, then download if not found.

    Args:
        load_in_8bit: Whether to load in 8-bit precision

    Returns:
        Loaded text encoder
    """
    repo_id = HF_MAIN_REPO
    subfolder = "text_encoder"
    download_destination_parent = LOCAL_MODELS_CACHE / (repo_id.replace("/", "--") + f"--{subfolder}-root")
    actual_model_files_dir = download_destination_parent / subfolder
    
    kwargs = (
        {"quantization_config": BitsAndBytesConfig(load_in_8bit=True)}
        if load_in_8bit
        else {"torch_dtype": torch.bfloat16}
    )
    try:
        return T5EncoderModel.from_pretrained(
            actual_model_files_dir,
            local_files_only=True,
            **kwargs,
        )
    except (OSError, HFValidationError):
        snapshot_download(
            repo_id=repo_id,
            allow_patterns=[f"{subfolder}/*", f"{subfolder}/**/*"],
            local_dir=download_destination_parent,
            local_dir_use_symlinks=False,
        )
        return T5EncoderModel.from_pretrained(
            actual_model_files_dir,
            local_files_only=True,
            **kwargs,
        )


def load_vae(
    source: ModelSource,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> AutoencoderKLLTXVideo:
    """
    Load the VAE component.
    It will first check the local `models/` cache, then download if not found.

    Args:
        source: Model source (HF repo, local path, or version)
        dtype: Data type for the VAE

    Returns:
        Loaded VAE
    """
    if isinstance(source, str): 
        if version := _try_parse_version(source):
            source = version

    if isinstance(source, LtxvModelVersion):
        if source in (
            LtxvModelVersion.LTXV_2B_095,
            LtxvModelVersion.LTXV_2B_096_DEV,
            LtxvModelVersion.LTXV_2B_096_DISTILLED,
            LtxvModelVersion.LTXV_13B_097_DEV,
            LtxvModelVersion.LTXV_13B_097_DEV_FP8,
            LtxvModelVersion.LTXV_13B_097_DISTILLED,
            LtxvModelVersion.LTXV_13B_097_DISTILLED_FP8,
        ):
            repo_id_to_load = LtxvModelVersion.LTXV_2B_095.hf_repo
            subfolder = "vae"
            download_destination_parent = LOCAL_MODELS_CACHE / (repo_id_to_load.replace("/", "--") + f"--{subfolder}-root")
            actual_model_files_dir = download_destination_parent / subfolder
            try:
                return AutoencoderKLLTXVideo.from_pretrained(
                    actual_model_files_dir,
                    torch_dtype=dtype,
                    local_files_only=True,
                )
            except (OSError, HFValidationError):
                snapshot_download(
                    repo_id=repo_id_to_load,
                    allow_patterns=[f"{subfolder}/*", f"{subfolder}/**/*"],
                    local_dir=download_destination_parent,
                    local_dir_use_symlinks=False,
                )
                return AutoencoderKLLTXVideo.from_pretrained(
                    actual_model_files_dir,
                    torch_dtype=dtype,
                    local_files_only=True,
                )
        else: 
            file_url = source.safetensors_url
            filename = file_url.split("/")[-1]
            
            parsed_url = urlparse(file_url)
            path_parts = parsed_url.path.strip("/").split("/")
            repo_id_of_file = f"{path_parts[0]}/{path_parts[1]}" if len(path_parts) > 1 else HF_MAIN_REPO
            actual_filename_on_hub = path_parts[-1]

            model_name_for_path = filename.replace(".safetensors", "")
            # For single files, target_local_file_dir is where the file itself will be placed.
            target_local_file_dir = LOCAL_MODELS_CACHE / "vae" / model_name_for_path 
            target_local_file_path = target_local_file_dir / actual_filename_on_hub
            target_local_file_dir.mkdir(parents=True, exist_ok=True)

            if not target_local_file_path.exists():
                try:
                    hf_hub_download(
                        repo_id=repo_id_of_file,
                        filename=actual_filename_on_hub,
                        local_dir=target_local_file_dir, # hf_hub_download puts the file directly in local_dir
                        local_dir_use_symlinks=False,
                    )
                except Exception as e:
                    raise IOError(f"Failed to download VAE {actual_filename_on_hub} from repo {repo_id_of_file} for version {source}: {e}") from e

            try:
                return AutoencoderKLLTXVideo.from_single_file(
                    str(target_local_file_path),
                    torch_dtype=dtype,
                    local_files_only=True,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load VAE from local file {target_local_file_path} "
                    f"(version {source}, dtype: {dtype}). File was expected to exist. Original error: {e}"
                ) from e

    elif isinstance(source, (str, Path)):
        source_str = str(source)
        if _is_huggingface_repo(source_str): 
            repo_id = source_str
            subfolder = "vae" 
            download_destination_parent = LOCAL_MODELS_CACHE / (repo_id.replace("/", "--") + f"--{subfolder}-root")
            actual_model_files_dir = download_destination_parent / subfolder
            
            try:
                return AutoencoderKLLTXVideo.from_pretrained(
                    actual_model_files_dir, 
                    torch_dtype=dtype,
                    local_files_only=True,
                )
            except (OSError, HFValidationError):
                snapshot_download(
                    repo_id=repo_id,
                    allow_patterns=[f"{subfolder}/*", f"{subfolder}/**/*"], 
                    local_dir=download_destination_parent, 
                    local_dir_use_symlinks=False,
                )
                return AutoencoderKLLTXVideo.from_pretrained(
                    actual_model_files_dir, 
                    torch_dtype=dtype,
                    local_files_only=True,
                )
        elif _is_safetensors_url(source_str) or Path(source_str).is_file(): 
            is_local_file = Path(source_str).is_file()
            if is_local_file:
                return AutoencoderKLLTXVideo.from_single_file(source_str, torch_dtype=dtype)

            file_url = source_str
            filename = file_url.split("/")[-1]
            parsed_url = urlparse(file_url)
            path_parts = parsed_url.path.strip("/").split("/")
            repo_id_of_file = f"{path_parts[0]}/{path_parts[1]}" if len(path_parts) >= 2 else None 
            actual_filename_on_hub = path_parts[-1]

            model_name_for_path = filename.replace(".safetensors", "")
            target_local_file_dir = LOCAL_MODELS_CACHE / "vae_single_files" / model_name_for_path
            target_local_file_path = target_local_file_dir / actual_filename_on_hub
            target_local_file_dir.mkdir(parents=True, exist_ok=True)

            if not target_local_file_path.exists():
                try:
                    hf_hub_download(
                        repo_id=repo_id_of_file,
                        filename=actual_filename_on_hub,
                        local_dir=target_local_file_dir,
                        local_dir_use_symlinks=False,
                    )
                except Exception as e:
                    raise IOError(f"Failed to download VAE {actual_filename_on_hub} from repo {repo_id_of_file} for version {source}: {e}") from e

            try:
                return AutoencoderKLLTXVideo.from_single_file(
                    str(target_local_file_path),
                    torch_dtype=dtype,
                    local_files_only=True,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load VAE from local file {target_local_file_path} "
                    f"(version {source}, dtype: {dtype}). File was expected to exist. Original error: {e}"
                ) from e

    raise ValueError(f"Invalid model source for VAE: {source}")


def load_transformer(
    source: ModelSource,
    *,
    dtype: torch.dtype = torch.float32,
) -> LTXVideoTransformer3DModel:
    """
    Load the transformer component.
    It will first check the local `models/` cache, then download if not found.

    Args:
        source: Model source (HF repo, local path, or version)
        dtype: Data type for the transformer

    Returns:
        Loaded transformer
    """
    if isinstance(source, str):  
        if version := _try_parse_version(source):
            source = version

    if isinstance(source, LtxvModelVersion):
        file_url = source.safetensors_url # All LtxvModelVersion for transformers use a safetensors_url
        filename = file_url.split("/")[-1]
        parsed_url = urlparse(file_url) 
        path_parts = parsed_url.path.strip("/").split("/")
        repo_id_of_file = f"{path_parts[0]}/{path_parts[1]}" if len(path_parts) > 1 else HF_MAIN_REPO
        actual_filename_on_hub = path_parts[-1]

        model_name_for_path = filename.replace(".safetensors", "")
        # For single files, target_local_file_dir is where the file itself will be placed.
        target_local_file_dir = LOCAL_MODELS_CACHE / "transformer" / model_name_for_path
        target_local_file_path = target_local_file_dir / actual_filename_on_hub
        target_local_file_dir.mkdir(parents=True, exist_ok=True)
        
        # Special handling for LTXV_13B_097_DEV which uses a custom loader function
        if source in (
            LtxvModelVersion.LTXV_13B_097_DEV,
            LtxvModelVersion.LTXV_13B_097_DISTILLED,
            LtxvModelVersion.LTXV_13B_097_DEV_FP8,
            LtxvModelVersion.LTXV_13B_097_DISTILLED_FP8,
        ):
            if not target_local_file_path.exists(): # Check if file needs download
                 try:
                    hf_hub_download(
                        repo_id=repo_id_of_file,
                        filename=actual_filename_on_hub,
                        local_dir=target_local_file_dir,
                        local_dir_use_symlinks=False,
                    )
                 except Exception as e:
                    raise IOError(f"Failed to download {actual_filename_on_hub} for {source} from {repo_id_of_file}: {e}") from e
            return _load_ltxv_13b_transformer(str(target_local_file_path), dtype=dtype)

        # For other LtxvModelVersion transformers (single .safetensors file)
        if not target_local_file_path.exists():
            try:
                hf_hub_download(
                    repo_id=repo_id_of_file,
                    filename=actual_filename_on_hub,
                    local_dir=target_local_file_dir,
                    local_dir_use_symlinks=False,
                )
            except Exception as e:
                raise IOError(f"Failed to download transformer {actual_filename_on_hub} from repo {repo_id_of_file} for version {source}: {e}") from e
        
        try:
            return LTXVideoTransformer3DModel.from_single_file(
                str(target_local_file_path),
                torch_dtype=dtype,
                local_files_only=True,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load transformer from local file {target_local_file_path} "
                f"(version {source}, dtype: {dtype}). File was expected to exist. Original error: {e}"
            ) from e

    elif isinstance(source, (str, Path)):
        source_str = str(source)
        if _is_huggingface_repo(source_str): 
            repo_id = source_str
            subfolder = "transformer" 
            download_destination_parent = LOCAL_MODELS_CACHE / (repo_id.replace("/", "--") + f"--{subfolder}-root")
            actual_model_files_dir = download_destination_parent / subfolder
            
            try:
                return LTXVideoTransformer3DModel.from_pretrained(
                    actual_model_files_dir, 
                    torch_dtype=dtype,
                    local_files_only=True,
                )
            except (OSError, HFValidationError): 
                snapshot_download(
                    repo_id=repo_id,
                    allow_patterns=[f"{subfolder}/*", f"{subfolder}/**/*"], 
                    local_dir=download_destination_parent, 
                    local_dir_use_symlinks=False,
                )
                return LTXVideoTransformer3DModel.from_pretrained(
                    actual_model_files_dir, 
                    torch_dtype=dtype,
                    local_files_only=True,
                )

        elif _is_safetensors_url(source_str) or Path(source_str).is_file(): 
            is_local_file = Path(source_str).is_file()
            if is_local_file:
                return LTXVideoTransformer3DModel.from_single_file(str(source_str), torch_dtype=dtype)

            file_url = source_str
            filename = file_url.split("/")[-1]
            parsed_url = urlparse(file_url)
            path_parts = parsed_url.path.strip("/").split("/")
            repo_id_of_file = f"{path_parts[0]}/{path_parts[1]}" if len(path_parts) >= 2 else None
            actual_filename_on_hub = path_parts[-1]

            model_name_for_path = filename.replace(".safetensors", "")
            target_local_file_dir = LOCAL_MODELS_CACHE / "transformer_single_files" / model_name_for_path
            target_local_file_path = target_local_file_dir / actual_filename_on_hub
            target_local_file_dir.mkdir(parents=True, exist_ok=True)

            if not target_local_file_path.exists():
                try:
                    hf_hub_download(
                        repo_id=repo_id_of_file,
                        filename=actual_filename_on_hub,
                        local_dir=target_local_file_dir,
                        local_dir_use_symlinks=False,
                    )
                except Exception as e:
                    raise IOError(f"Failed to download Transformer {actual_filename_on_hub} from repo {repo_id_of_file} for version {source}: {e}") from e

            try:
                return LTXVideoTransformer3DModel.from_single_file(
                    str(target_local_file_path),
                    torch_dtype=dtype,
                    local_files_only=True,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load transformer from local file {target_local_file_path} "
                    f"(version {source}, dtype: {dtype}). File was expected to exist. Original error: {e}"
                ) from e

    raise ValueError(f"Invalid model source for Transformer: {source}")


def load_ltxv_components(
    model_source: ModelSource | None = None,
    *,
    load_text_encoder_in_8bit: bool = False,
    transformer_dtype: torch.dtype = torch.float32,
    vae_dtype: torch.dtype = torch.bfloat16,
) -> LtxvModelComponents:
    """
    Load all components of the LTXV model from a specified source.
    Note: scheduler, tokenizer, and text encoder are always loaded from the main HF repo.

    Args:
        model_source: Source to load the VAE and transformer from. Can be:
            - HuggingFace repo ID (e.g. "Lightricks/LTX-Video")
            - Local path to model files (str or Path)
            - LtxvModelVersion enum value
            - None (will use the latest version)
        load_text_encoder_in_8bit: Whether to load text encoder in 8-bit precision
        transformer_dtype: Data type for transformer model
        vae_dtype: Data type for VAE model

    Returns:
        LtxvModelComponents containing all loaded model components
    """

    if model_source is None:
        model_source = LtxvModelVersion.latest()

    return LtxvModelComponents(
        scheduler=load_scheduler(),
        tokenizer=load_tokenizer(),
        text_encoder=load_text_encoder(load_in_8bit=load_text_encoder_in_8bit),
        vae=load_vae(model_source, dtype=vae_dtype),
        transformer=load_transformer(model_source, dtype=transformer_dtype),
    )


def _try_parse_version(source: str | Path) -> LtxvModelVersion | None:
    """
    Try to parse a string as an LtxvModelVersion.

    Args:
        source: String to parse

    Returns:
        LtxvModelVersion if successful, None otherwise
    """
    try:
        return LtxvModelVersion(str(source))
    except ValueError:
        return None


def _is_huggingface_repo(source: str | Path) -> bool:
    """
    Check if a string is a valid HuggingFace repo ID.

    Args:
        source: String or Path to check

    Returns:
        True if the string looks like a HF repo ID
    """
    # Basic check: contains slash, no URL components
    return "/" in source and not urlparse(source).scheme


def _is_safetensors_url(source: str | Path) -> bool:
    """
    Check if a string is a valid safetensors URL.
    """
    return source.endswith(".safetensors")


def _load_ltxv_13b_transformer(safetensors_url_or_path: str, *, dtype: torch.dtype) -> LTXVideoTransformer3DModel:
    """A specific loader for LTXV-13B's transformer which doesn't yet have a Diffusers config"""
    transformer_13b_config = {
        "_class_name": "LTXVideoTransformer3DModel",
        "_diffusers_version": "0.33.0.dev0",
        "activation_fn": "gelu-approximate",
        "attention_bias": True,
        "attention_head_dim": 128,
        "attention_out_bias": True,
        "caption_channels": 4096,
        "cross_attention_dim": 4096,
        "in_channels": 128,
        "norm_elementwise_affine": False,
        "norm_eps": 1e-06,
        "num_attention_heads": 32,
        "num_layers": 48,
        "out_channels": 128,
        "patch_size": 1,
        "patch_size_t": 1,
        "qk_norm": "rms_norm_across_heads",
    }

    temp_config_file = None
    try:
        # Create NamedTemporaryFile with delete=False
        temp_config_file_obj = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        temp_config_file_path = temp_config_file_obj.name
        
        json.dump(transformer_13b_config, temp_config_file_obj)
        temp_config_file_obj.flush()
        # Explicitly close the file object before passing its name
        temp_config_file_obj.close()

        return LTXVideoTransformer3DModel.from_single_file(
            safetensors_url_or_path,
            config=temp_config_file_path, # Pass the path to the now-closed temporary file
            torch_dtype=dtype,
        )
    finally:
        # Ensure the temporary file is deleted if it was created
        if temp_config_file_path and Path(temp_config_file_path).exists():
            try:
                os.remove(temp_config_file_path)
            except OSError:
                # Log or handle deletion error if necessary, but don't let it crash the main flow
                # For example: print(f"Warning: Could not delete temporary config file {temp_config_file_path}")
                pass
