import torch
from pathlib import Path
from PIL import Image # Keep for open_image_as_srgb, though it might not be directly used here
from copy import deepcopy

# LTX-Video-Trainer specific imports
from src.ltxv_trainer.model_loader import load_ltxv_components, LtxvModelVersion # Corrected import
from src.ltxv_trainer.quantization import QuantizationOptions, quantize_model # Corrected import
from diffusers import LTXImageToVideoPipeline
from src.ltxv_trainer.utils import open_image_as_srgb
from diffusers.utils import export_to_video

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # General torch_dtype for operations if not specified by model loading
    # However, dtypes for components will be set by load_ltxv_components args

    # --- Configuration - Adjust these as needed ---
    # CRITICAL: Specify the source for your LTXV model components.
    # This can be a HuggingFace repo ID (e.g., "Lightricks/LTX-Video-S")
    # or a local path to a directory containing model files (transformer.pt, vae/, etc.)
    # e.g., model_source_input = "path/to/your/local_ltxv_model_directory"
    model_source_input = LtxvModelVersion.LTXV_13B_097_DEV  # <-- Using specific model version

    # CRITICAL: This is a placeholder. You MUST provide a valid image path.
    input_image_path = "sunset.png" # <-- USER MUST VERIFY/CHANGE THIS
    output_video_path = Path("test_output_image_to_video.mp4")

    prompt = "A girl in pool wearing black bikini , she smiles and dances"
    negative_prompt = "blurry, low quality, ugly, watermark, text, signature"
    width = 512
    height = 512 # Example height, adjust as needed
    guidance_scale = 7.5 # Aligned with common values in official config
    num_frames = 49 # Example number of frames
    num_inference_steps = 25 # Increased from 20 for potentially more refined motion
    frame_rate = 8 # Example frame rate
    seed = 1337

    # Load components
    print(f"Using device: {device}")
    print(f"Loading LTXV components from source: {model_source_input}...")
    transformer_quantization: QuantizationOptions = "fp8-quanto"
    try:
        components = load_ltxv_components(
            model_source=model_source_input,
            load_text_encoder_in_8bit=False, # Keep False for simplicity unless 8-bit is specifically needed
            transformer_dtype=torch.bfloat16 if device.type == 'cuda' else torch.float32,
            vae_dtype=torch.bfloat16 if device.type == 'cuda' else torch.float32
        )
        scheduler = components.scheduler
        tokenizer = components.tokenizer
        text_encoder = components.text_encoder
        vae = components.vae
        transformer = components.transformer

    except Exception as e:
        print(f"Error loading components with load_ltxv_components: {e}")
        print("Please ensure 'model_source_input' is correct and points to a valid model source.")
        print("If using a local path, ensure the directory structure matches what 'load_ltxv_components' expects.")
        print("Check that all necessary model files (transformer, VAE, etc.) are present at the source.")
        return

    print("LTXV components loaded successfully.")

    # Apply quantization to the transformer if specified
    if transformer_quantization and transformer_quantization != "no_change":
        if hasattr(components, 'transformer') and components.transformer is not None:
            print(f"Applying {transformer_quantization} quantization to the transformer...")
            components.transformer = quantize_model(components.transformer, transformer_quantization)
            # Ensure the quantized model is on the correct device if quantize_model doesn't handle it
            if device.type == 'cuda':
                components.transformer.to(device)
            print("Transformer quantized and moved to device.")
        else:
            print("Warning: Transformer component not found or is None, skipping quantization.")

    print(f"  Scheduler: {type(scheduler).__name__}")
    print(f"  Tokenizer: {type(tokenizer).__name__}")
    print(f"  Text Encoder: {type(text_encoder).__name__} on {text_encoder.device}")
    print(f"  VAE: {type(vae).__name__} on {vae.device}")
    print(f"  Transformer: {type(transformer).__name__} on {transformer.device}")

    # Prepare the pipeline
    pipeline = LTXImageToVideoPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer,
        scheduler=scheduler,
    )
    pipeline.to(device) # Ensure pipeline is on the correct device
    pipeline.set_progress_bar_config(disable=False)

    print("Pipeline initialized.")

    # Encode prompts and then unload text_encoder to save VRAM
    print("Encoding prompts...")
    prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = pipeline.encode_prompt(
        prompt=prompt,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=True,
        num_videos_per_prompt=1, # Assuming 1 video per prompt for this script
        device=device, # Text encoder is already on this device via pipeline.to(device)
        dtype=pipeline.text_encoder.dtype, # Use the dtype of the text_encoder
        max_sequence_length=pipeline.tokenizer_max_length
    )

    print("Unloading text encoder to free VRAM...")
    del text_encoder # text_encoder variable from the outer scope
    # The pipeline object still holds its reference if needed, but we are done with it directly.
    # For maximum memory saving, if tokenizer is also not needed by diffusion loop, it could be removed from pipeline too.
    # However, text_encoder is the larger component.
    if hasattr(pipeline, 'text_encoder'):
        del pipeline.text_encoder # Remove from pipeline itself if possible
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Load image
    try:
        image = open_image_as_srgb(input_image_path)
        print(f"Input image '{input_image_path}' loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Input image not found at '{input_image_path}'. Please set 'input_image_path' correctly.")
        return
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # Generate video
    print(f"Generating video with prompt: '{prompt}'")
    generator = torch.Generator(device=device).manual_seed(seed)

    try:
        with torch.no_grad(): # Important for inference
            video_frames = pipeline(
                image=image,
                prompt_embeds=prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_prompt_attention_mask=negative_prompt_attention_mask,
                width=width,
                height=height,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale, # Use the defined guidance_scale
                generator=generator,
                frame_rate=frame_rate,  # Match output FPS
                decode_timestep=0.05, # Aligned with official config
                decode_noise_scale=0.025 # Aligned with official config
            ).frames
            
        print("Video frames generated.")
    except Exception as e:
        print(f"Error during video generation: {e}")
        import traceback
        traceback.print_exc()
        return

    # 5. Save Video
    print(f"Saving video to: {output_video_path}")
    frames_to_save = video_frames
    # Check if video_frames is a list containing another list (e.g., batch output)
    if isinstance(video_frames, list) and len(video_frames) > 0 and isinstance(video_frames[0], list):
        print("Info: video_frames is a list of lists. Using the first list of frames.")
        frames_to_save = video_frames[0]
    elif not isinstance(video_frames, list) or (isinstance(video_frames, list) and len(video_frames) > 0 and not hasattr(video_frames[0], 'mode')):
        # This case tries to catch if it's not a list of PIL images or list of list of PIL images
        print(f"Warning: video_frames format might be unexpected. Type: {type(video_frames)}. Attempting to save directly.")

    try:
        # export_to_video expects a tensor of shape (num_frames, height, width, channels)
        # The pipeline output might be (batch_size, num_frames, channels, height, width)
        # or (num_frames, channels, height, width) if batch_size is 1 and squeezed.
        # Adjust processing as needed. Assuming batch_size=1 and we take the first video.


        export_to_video(frames_to_save, str(output_video_path), fps=8) # fps can be adjusted
        print(f"Video saved successfully to {output_video_path}")
    except Exception as e:
        print(f"Error saving video: {e}")
        print("Ensure 'video_frames' has the correct format for 'export_to_video'.")
        return

if __name__ == "__main__":
    main()
