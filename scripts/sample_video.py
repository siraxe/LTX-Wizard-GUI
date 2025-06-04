import argparse
import torch
from pathlib import Path
from diffusers import LTXImageToVideoPipeline, LTXPipeline
from diffusers.utils import export_to_video
from loguru import logger
from safetensors.torch import load_file
from ltxv_trainer.model_loader import load_ltxv_components
from ltxv_trainer.config import LtxvTrainerConfig
from ltxv_trainer.utils import open_image_as_srgb
from copy import deepcopy # Needed for copying scheduler
import yaml

# Add necessary imports here based on your needs
# For example, if you need specific config details or utility functions:
# from ltxv_trainer.config import LtxvTrainerConfig
# from ltxv_trainer.utils import some_utility_function

def load_model_and_pipeline(checkpoint_path: str, config: LtxvTrainerConfig, use_image_to_video_pipeline: bool = False):
    """
    Loads the model components and checkpoint, and sets up the inference pipeline.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Loading model components to device: {device}")

    # Load model components based on config (might need adjustments depending on your setup)
    # For sampling, we typically load in full precision or quantized, not LoRA init
    # Adjust dtype and quantization based on the loaded checkpoint and config
    # This part might need refinement based on how your checkpoints are structured
    logger.debug(f"Calling load_ltxv_components with model_source: {config.model.model_source}, load_text_encoder_in_8bit: {config.acceleration.load_text_encoder_in_8bit}, transformer_dtype: {torch.float16 if torch.cuda.is_available() else torch.float32}, vae_dtype: {torch.bfloat16}")
    components = load_ltxv_components(
        model_source=config.model.model_source,
        load_text_encoder_in_8bit=config.acceleration.load_text_encoder_in_8bit, # Use config setting
        transformer_dtype=torch.float16 if torch.cuda.is_available() else torch.float32, # Use float16 for faster inference on GPU
        vae_dtype=torch.bfloat16, # VAE can often be bfloat16
    )

    scheduler = components.scheduler
    tokenizer = components.tokenizer
    text_encoder = components.text_encoder
    vae = components.vae
    transformer = components.transformer

    logger.debug(f"Loaded components. Text Encoder type: {type(text_encoder)}, VAE type: {type(vae)}, Transformer type: {type(transformer)}")
    logger.debug(f"Text Encoder device: {text_encoder.device}, VAE device: {vae.device}, Transformer device: {transformer.device}")
    logger.debug(f"Is Text Encoder 8-bit? {hasattr(text_encoder, 'quantization_config') and text_encoder.quantization_config is not None}")

    # Explicitly move components to the target device
    # Removed explicit .to(device) calls as models should be on the correct device after accelerator.prepare
    # vae.to(device)
    # # Check if text_encoder was loaded in 8-bit by looking for quantization_config
    # if not (hasattr(text_encoder, 'quantization_config') and text_encoder.quantization_config is not None):
    #     text_encoder.to(device)
    # transformer.to(device)

    # Load checkpoint weights
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    logger.info(f"Loading checkpoint from {checkpoint_path}")
    state_dict = load_file(checkpoint_path) # Load directly to device if possible
    logger.debug(f"Loaded state_dict with {len(state_dict)} keys.")

    # Determine if it's a full model or LoRA checkpoint based on key names
    is_lora_checkpoint = any("lora_A" in k or "lora_B" in k for k in state_dict.keys())

    if is_lora_checkpoint:
        logger.info("Detected LoRA checkpoint. Applying LoRA weights.")
        # Assuming the base model was loaded without LoRA, now add adapter and load weights
        from peft import LoraConfig, set_peft_model_state_dict
        logger.debug(f"Applying LoRA config with rank: {config.lora.rank}, alpha: {config.lora.alpha}, target_modules: {config.lora.target_modules}")
        try:
             # Attempt to add adapter if not already present - depends on how base model is loaded
             transformer.add_adapter(LoraConfig(
                 r=config.lora.rank, # Assuming rank is in config
                 lora_alpha=config.lora.alpha, # Assuming alpha is in config
                 target_modules=config.lora.target_modules, # Assuming target_modules is in config
                 lora_dropout=config.lora.dropout,
                 init_lora_weights=False # Do not re-initialize, we are loading
             ))
             logger.debug("LoRA adapter added.")
        except ValueError as e:
             # Handle cases where adapter might already be present
             if "Adapter with name 'default' already exists" not in str(e):
                  raise e
             logger.info("LoRA adapter already exists, proceeding with loading state dict.")


        # Adjust key names if needed based on how state_dict was saved (e.g., with or without "_orig_mod.")
        # Assuming saved with clean names (no "_orig_mod.") as in trainer._save_checkpoint
        state_dict = {k.replace("transformer.", "", 1): v for k, v in state_dict.items()}
        # PEFT might expect 'default' adapter name in state_dict keys if saved directly
        # The trainer saves with 'lora_A.default', 'lora_B.default' if using save_lora_weights
        # The `set_peft_model_state_dict` handles this if keys match PEFT format
        # If state_dict keys are 'lora_A', 'lora_B', you might need to adjust them here
        # Example adjustment if keys are 'lora_A', 'lora_B' and need 'default':
        # state_dict = {k.replace("lora_A", "lora_A.default", 1): v for k, v in state_dict.items()}
        # state_dict = {k.replace("lora_B", "lora_B.default", 1): v for k, v in state_dict.items()}

        logger.debug("Applying LoRA state dict...")
        missing_keys, unexpected_keys = set_peft_model_state_dict(transformer, state_dict, adapter_name="default")
        if missing_keys:
             logger.warning(f"Missing keys in checkpoint: {missing_keys}")
        if unexpected_keys:
             logger.warning(f"Unexpected keys in checkpoint: {unexpected_keys}")
        logger.debug("LoRA state dict applied.")


    else:
        logger.info("Detected full model checkpoint. Loading full model weights.")
        # Adjust key names if saved with a prefix like "transformer."
        state_dict = {k.replace("transformer.", "", 1): v for k, v in state_dict.items()}
        # Attempt to load state dict
        logger.debug("Applying full model state dict...")
        missing_keys, unexpected_keys = transformer.load_state_dict(state_dict, strict=False)
        if missing_keys:
             logger.warning(f"Missing keys in checkpoint: {missing_keys}")
        if unexpected_keys:
             logger.warning(f"Unexpected keys in checkpoint: {unexpected_keys}")
        logger.debug("Full model state dict applied.")


    # Set models to evaluation mode
    transformer.eval()
    text_encoder.eval()
    vae.eval()

    # Create the pipeline
    # Determine pipeline type based on whether image input is expected
    # For simplicity here, we'll use LTXPipeline. If you need image conditioning,
    # you'll need to add logic to choose LTXImageToVideoPipeline and provide images.
    logger.debug("Creating pipeline...")
    if use_image_to_video_pipeline:
        pipeline = LTXImageToVideoPipeline(
            scheduler=deepcopy(scheduler),
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
        )
    else:
        pipeline = LTXPipeline(
            scheduler=deepcopy(scheduler), # Pass a deepcopy to avoid modifying the original
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
        )
    logger.debug("Pipeline created.")
    pipeline.set_progress_bar_config(disable=False) # Enable progress bar for sampling

    logger.debug("Finished load_model_and_pipeline.")
    return pipeline, device

def sample_videos(
    pipeline: LTXPipeline | LTXImageToVideoPipeline, # Update type hint
    device: torch.device,
    prompts: list[str],
    output_dir: Path,
    video_dims: tuple[int, int, int], # width, height, frames
    num_inference_steps: int = 50,
    seed: int = 42,
    negative_prompt: str | None = None,
    image_path: str | None = None, # New parameter for image input
    decode_timestep=0.05, # Aligned with official config
    decode_noise_scale=0.025 # Aligned with official config
):
    """
    Runs the sampling process for a list of prompts.
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    width, height, frames = video_dims

    logger.info(f"Generating {len(prompts)} videos...")

    video_paths = []
    # Move VAE and Text Encoder to the correct device for sampling, if not 8-bit
    pipeline.vae.to(device)
    # Check if text_encoder was loaded in 8-bit by looking for quantization_config
    if not (hasattr(pipeline.text_encoder, 'quantization_config') and pipeline.text_encoder.quantization_config is not None):
        pipeline.text_encoder.to(device)

    for i, prompt in enumerate(prompts):
        logger.info(f"Generating video for prompt {i+1}/{len(prompts)}: '{prompt}'")
        generator = torch.Generator(device=device).manual_seed(seed + i) # Use seed + index for variation

        pipeline_inputs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "num_frames": frames,
            "num_inference_steps": num_inference_steps,
            "generator": generator,
        }

        # Add image handling here if using LTXImageToVideoPipeline
        if image_path:
            logger.info(f"Using image {image_path} as initial frame.")
            pipeline_inputs["image"] = open_image_as_srgb(image_path).to(device)

        with torch.no_grad(): # Ensure no gradients are computed during inference
             # Use autocast for mixed precision inference if desired and available
             with torch.autocast(device.type, dtype=torch.float16 if device.type == 'cuda' else torch.float32):
                result = pipeline(**pipeline_inputs)
                videos = result.frames # Assumes pipeline returns 'frames'

        # Assuming result.frames is a list of video tensors or similar
        # Export each video generated for this prompt (usually just one)
        for j, video in enumerate(videos):
            video_path = output_dir / f"sample_{i+1}_{j+1}.mp4"
            try:
                 export_to_video(video, str(video_path), fps=25) # Adjust FPS as needed
                 video_paths.append(video_path)
                 logger.info(f"Saved video to {video_path}")
            except Exception as e:
                 logger.error(f"Failed to save video {i+1}_{j+1} to {video_path}: {e}")

    # Optionally move VAE and Text Encoder back to CPU after sampling to free up GPU memory
    # This is done in trainer.py's validation step, but might not be strictly necessary for a simple sample script
    # pipeline.vae.to("cpu")
    # if not hasattr(pipeline.text_encoder, 'quantization_config') or pipeline.text_encoder.quantization_config is None:
    #      pipeline.text_encoder.to("cpu")

    logger.info("Sampling complete.")
    return video_paths

def main():
    parser = argparse.ArgumentParser(description="Sample videos using a trained LTXV model checkpoint.")
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the trainer configuration YAML file."
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="Path to an initial image for image-to-video generation."
    )
    parser.add_argument(
        "--decode_timestep",
        type=float,
        default=0.05,
        help="Timestep for decoding. Default: 0.05, aligned with official config."
    )
    parser.add_argument(
        "--decode_noise_scale",
        type=float,
        default=0.025,
        help="Noise scale for decoding. Default: 0.025, aligned with official config."
    )
    # Remove or modify arguments that will be read from config
    # parser.add_argument(
    #     "--checkpoint_path",
    #     type=str,
    #     required=True,
    #     help="Path to the model checkpoint (.safetensors file)."
    # )
    # parser.add_argument(
    #     "--output_dir",
    #     type=str,
    #     default="manual_samples",
    #     help="Directory to save the generated videos.",
    # )
    # parser.add_argument(
    #     "--prompts",
    #     type=str,
    #     nargs='+',
    #     required=True,
    #     help="List of prompts to generate videos for. Enclose prompts with spaces in quotes.",
    # )
    # parser.add_argument(
    #     "--width",
    #     type=int,
    #     default=512,
    #     help="Width of the output video frames.",
    # )
    # parser.add_argument(
    #     "--height",
    #     type=int,
    #     default=512,
    #     help="Height of the output video frames.",
    # )
    # parser.add_argument(
    #     "--frames",
    #     type=int,
    #     default=24,
    #     help="Number of frames in the output video.",
    # )
    # parser.add_argument(
    #     "--inference_steps",
    #     type=int,
    #     default=50,
    #     help="Number of inference steps to run.",
    # )
    # parser.add_argument(
    #     "--seed",
    #     type=int,
    #     default=42,
    #     help="Random seed for generation.",
    # )
    # parser.add_argument(
    #     "--negative_prompt",
    #     type=str,
    #     default=None,
    #     help="Negative prompt for generation.",
    # )
    # Add argument for image paths if using image conditioning
    # parser.add_argument(
    #     "--images",
    #     type=str,
    #     nargs='+',
    #     help="List of paths to conditioning images (must match number of prompts if provided).",
    # )


    args = parser.parse_args()

    # Load configuration from file
    try:
        with open(args.config_path, "r") as file:
            config_data = yaml.safe_load(file)

        # Remove the 'misc' section if it exists to avoid validation errors (if applicable)
        if 'misc' in config_data:
            del config_data['misc']

        config = LtxvTrainerConfig(**config_data)
        logger.info(f"Loaded configuration from {args.config_path}")
    except Exception as e:
        logger.error(f"Error loading config file {args.config_path}: {e}")
        # Add more detailed logging of the exception
        import traceback
        logger.error(f"Detailed error: {traceback.format_exc()}")
        return

    # Read sampling parameters from config
    prompts = config.validation.prompts
    negative_prompt = config.validation.negative_prompt
    video_dims = tuple(config.validation.video_dims) # Expecting a list [w, h, f] in config
    seed = config.validation.seed
    inference_steps = config.validation.inference_steps
    checkpoint_path = config.model.load_checkpoint
    output_dir = Path(config.output_dir)
    # Note: guidance_scale and videos_per_prompt are in config.validation but not used in sample_videos function yet
    # image_paths would also come from config if implemented

    if not checkpoint_path:
         logger.error("Error: No checkpoint path specified in config.")
         return

    if not prompts:
         logger.error("Error: No prompts specified in config.")
         return

    # Load model and pipeline using the checkpoint path from config
    try:
        pipeline, device = load_model_and_pipeline(checkpoint_path, config, use_image_to_video_pipeline=True)
    except Exception as e:
        logger.error(f"Error loading model or pipeline: {e}")
        return


    # Run sampling using parameters read from config
    try:
        sample_videos(
            pipeline=pipeline,
            device=device,
            prompts=prompts,
            output_dir=output_dir,
            video_dims=video_dims,
            num_inference_steps=inference_steps,
            seed=seed,
            negative_prompt=negative_prompt,
            image_path=args.image_path, # Pass image_path to sample_videos
            decode_timestep=args.decode_timestep,
            decode_noise_scale=args.decode_noise_scale
        )
    except Exception as e:
        logger.error(f"Error during video sampling: {e}")


if __name__ == "__main__":
    main()
