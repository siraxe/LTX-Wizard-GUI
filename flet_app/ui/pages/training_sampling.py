import flet as ft
import os
# from rich.color import Color # Not used
import yaml 
import shlex 
import json 
import subprocess 
import asyncio 
from .._styles import create_textfield, add_section_title,create_dropdown ,create_styled_button
from settings import settings
from loguru import logger 
from ui.tab_dataset_view import _build_expansion_tile 
import traceback # Import traceback

# Global variables for Flet controls and state
model_source_dropdown = None
lora_dropdown = None
lora_str_textfield = None
prompts_textfield = None
negative_prompt_textfield = None
video_dims_textfield = None
seed_textfield = None
inference_steps_textfield = None
guidance_scale_textfield = None
interval_textfield = None
videos_per_prompt_textfield = None
sampling_status_text = None 

sampling_console_output = None
sampling_progress_bar = None

file_picker = None # Global file picker instance
active_image_target_control = None # To track which image (c1 or c2) is being picked

# Image display controls - initialized once
image_display_c1 = ft.Image(
    # src="/images/image_placeholder.png", # Initial src can be None or placeholder
    src=None,
    width=200, height=200, fit=ft.ImageFit.CONTAIN, visible=False # Start not visible
)
image_display_c2 = ft.Image(
    # src="/images/image_placeholder.png",
    src=None,
    width=200, height=200, fit=ft.ImageFit.CONTAIN, visible=False # Start not visible
)

# These globals store the *current* selected path, primarily for UI interaction.
# The authoritative source when saving/loading YAML should be page attributes or YAML content itself.
selected_image_path_c1 = None 
selected_image_path_c2 = None 

def _create_image_selector_container(page: ft.Page, image_control: ft.Image, 
                                     image_path_attr_name_on_page: str, # e.g., "selected_image_path_c1"
                                     col_span: int):
    
    def _on_image_area_click(e):
        nonlocal page, image_control # Capture from outer scope
        global active_image_target_control, file_picker
        active_image_target_control = image_control # Set which image display (c1 or c2) is being targeted
        
        # Ensure file_picker is initialized and on the page
        if file_picker is None:
            return # Avoid proceeding if picker isn't ready.

        file_picker.pick_files(allow_multiple=False, allowed_extensions=["png", "jpg", "jpeg", "gif", "webp"])

    # Define remove_btn here so it's in scope for _remove_image_click and show/hide methods
    remove_btn = ft.IconButton(
        icon=ft.Icons.CLOSE, icon_color=ft.Colors.WHITE, icon_size=20,
        width=30, height=30,
        style=ft.ButtonStyle(bgcolor=ft.Colors.RED_700, shape=ft.CircleBorder(), padding=ft.padding.all(0)),
        visible=False, # Start hidden
        tooltip="Remove image"
    )

    def _remove_image_click(e):
        nonlocal page, image_control, image_path_attr_name_on_page, remove_btn # Capture from outer scope
        try:
            image_control.src = None # Or placeholder
            image_control.visible = False
            
            # Update the corresponding page attribute that stores the path
            if hasattr(page, image_path_attr_name_on_page):
                setattr(page, image_path_attr_name_on_page, None)

            # Update global selected_image_path_cx if they are still in use for direct UI logic
            if image_control == image_display_c1:
                global selected_image_path_c1; selected_image_path_c1 = None
            elif image_control == image_display_c2:
                global selected_image_path_c2; selected_image_path_c2 = None
            
            remove_btn.visible = False # Hide the button itself
            
            # Update controls
            remove_btn.update()
            image_control.update()
            if hasattr(image_control, 'parent') and image_control.parent and hasattr(image_control.parent, 'update'):
                image_control.parent.update()
            page.update()
        except Exception as ex:
            logger.error(f"Error in _remove_image_click for {image_path_attr_name_on_page}: {ex}")
            logger.error(traceback.format_exc())

    remove_btn.on_click = _remove_image_click # Assign the handler

    image_container = ft.Container(
        content=ft.Stack([image_control, ft.Container(content=remove_btn, alignment=ft.alignment.top_right, padding=5)]),
        width=200, height=200, border_radius=ft.border_radius.all(5)
    )
    
    def show_remove_button_impl():
        if not image_control.src or not image_control.visible:
            if remove_btn.visible: # Ensure it's hidden if no image
                 remove_btn.visible = False
            return

        if not remove_btn.visible:
            remove_btn.visible = True
            if page: page.update()

    container = ft.Container(
        content=ft.Column(
            [image_container, ft.Text("Click to select image", size=12, color=ft.Colors.GREY_600)],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER, alignment=ft.MainAxisAlignment.CENTER, spacing=5,
        ),
        width=200, height=250,
        border=ft.border.all(1, ft.Colors.BLUE_GREY_400), border_radius=ft.border_radius.all(5),
        alignment=ft.alignment.center,
        on_click=_on_image_area_click, # Click the whole area to pick
        col=col_span
    )

    image_control.show_remove_button = show_remove_button_impl

    def hide_remove_button_impl():
        if remove_btn.visible:
            remove_btn.visible = False
            if page: page.update()
            
    image_control.hide_remove_button = hide_remove_button_impl
    
    # Initial state check: if image is already visible (e.g. loaded from YAML before this container is built/rebuilt)
    if image_control.src and image_control.visible:
        show_remove_button_impl()
    else:
        hide_remove_button_impl() # Ensures button is hidden if no image initially

    return container

# This on_dialog_result needs to be defined at a scope where `page` is accessible,
# or `page` needs to be passed to it. It's best defined inside `build_training_sampling_page_content`.
# def on_dialog_result(e: ft.FilePickerResultEvent, page_ref: ft.Page): ... (see below)


def _get_lora_options() -> dict[str, str]:
    lora_options = {"none": "none"} # Key: display name, Value: actual value/path part
    output_dir = "workspace/output"
    if not os.path.exists(output_dir):
        logger.warning(f"LoRA Output directory not found: {output_dir}")
        return lora_options

    for name in os.listdir(output_dir):
        checkpoints_dir = os.path.join(output_dir, name, "checkpoints")
        if os.path.isdir(checkpoints_dir):
            for filename in os.listdir(checkpoints_dir):
                if filename.endswith(".safetensors"):
                    # Using folder name as key for simplicity, and relative path as value for dropdown text
                    option_key = name # This is what will be stored if selected
                    option_text = f"{name} ({filename})" # More descriptive text
                    lora_options[option_key] = option_text 
                    # The actual path construction will use 'name' and 'filename' later
    return lora_options


def _collect_sampling_parameters(page: ft.Page) -> dict: # Added page parameter
    # Get image paths from page attributes, which should be the single source of truth
    # after YAML load or file picking.
    current_c1_path = getattr(page, 'selected_image_path_c1', None)
    current_c2_path = getattr(page, 'selected_image_path_c2', None)
    logger.debug(f"Collecting params: C1 Path='{current_c1_path}', C2 Path='{current_c2_path}'")

    return {
        "selected_lora_key": lora_dropdown.value if lora_dropdown else "none",
        "single_prompt": prompts_textfield.value.strip() if prompts_textfield else "",
        "negative_prompt": negative_prompt_textfield.value.strip() if negative_prompt_textfield else None,
        "video_dims_text": video_dims_textfield.value if video_dims_textfield else "[512, 512, 49]",
        "seed_text": seed_textfield.value if seed_textfield else "42",
        "inference_steps_text": inference_steps_textfield.value if inference_steps_textfield else "25",
        "guidance_scale_text": guidance_scale_textfield.value if guidance_scale_textfield else "3.5",
        "videos_per_prompt_text": videos_per_prompt_textfield.value if videos_per_prompt_textfield else "1",
        "selected_image_path_c1": current_c1_path,
        "selected_image_path_c2": current_c2_path
    }


def _validate_and_parse_parameters(params: dict) -> tuple[dict | None, str | None]: # Return dict or None
    single_prompt = params["single_prompt"]
    # ... (rest of the validation logic is likely fine, ensure error messages are clear)
    if not single_prompt and not params.get("selected_image_path_c1") and not params.get("selected_image_path_c2"):
        return None, "Error: No prompt provided and no images selected for sampling."
    # Allow sampling with only images if prompts are empty, or only prompts if images are empty.
    # The script sample_video.py should handle these cases.

    try:
        video_dims_list = json.loads(params["video_dims_text"])
        if not (isinstance(video_dims_list, list) and len(video_dims_list) == 3 and all(isinstance(x, int) for x in video_dims_list)):
             raise ValueError("Expected a list of 3 integers like [W, H, Frames].")
    except (json.JSONDecodeError, ValueError) as e:
        return None, f"Error parsing video dimensions: {e}"

    # ... (other validations for seed, inference_steps, etc.)
    try: seed = int(params["seed_text"])
    except ValueError: return None, "Error: Invalid seed. Must be an integer."
    try: inference_steps = int(params["inference_steps_text"])
    except ValueError: return None, "Error: Invalid inference steps. Must be an integer."
    try: guidance_scale = float(params["guidance_scale_text"])
    except ValueError: return None, "Error: Invalid guidance scale. Must be a number."
    try: videos_per_prompt = int(params["videos_per_prompt_text"]); assert videos_per_prompt > 0
    except (ValueError, AssertionError): return None, "Error: Invalid videos per prompt. Must be a positive integer."

    return {
        "prompts": [single_prompt] if single_prompt else [], # Ensure list, even if empty
        "negative_prompt": params["negative_prompt"],
        "video_dims": video_dims_list,
        "seed": seed,
        "inference_steps": inference_steps,
        "guidance_scale": guidance_scale,
        "videos_per_prompt": videos_per_prompt,
        "selected_lora_key": params["selected_lora_key"],
        # Pass image paths directly, sample_video.py will decide how to use them
        "images_for_sampling": [p for p in [params.get("selected_image_path_c1"), params.get("selected_image_path_c2")] if p]
    }, None


def _get_checkpoint_path_from_selection(selected_lora_key: str, lora_options_map: dict) -> tuple[str | None, str | None]:
    """
    Determines the full checkpoint path based on the selected LoRA key and the options map.
    The lora_options_map should have {key: "folder_name (filename.safetensors)"}.
    The actual checkpoint path is constructed from the folder_name and filename.
    """
    if not selected_lora_key or selected_lora_key == "none":
        return None, "No LoRA checkpoint selected. Proceeding without LoRA." # This is a warning, not an error to stop.

    # The lora_dropdown.value (selected_lora_key) is the 'name' (folder name).
    # We need to find the corresponding .safetensors file.
    # The _get_lora_options now stores "name: name (filename.safetensors)"
    # So, selected_lora_key is the 'name'. We need to find the filename.
    
    # This logic needs to be robust if lora_options_map changes format.
    # Assuming selected_lora_key is the FOLDER name.
    # We need to find the actual .safetensors file within that folder.
    # For simplicity, if a folder has multiple, we might pick the first or require more specific selection.
    # The current _get_lora_options might oversimplify if a folder has multiple .safetensors.
    # Let's assume for now _get_lora_options provides enough info or the structure is one .safetensors per folder.

    # If selected_lora_key is the folder name:
    checkpoint_folder_path = os.path.join("workspace", "output", selected_lora_key, "checkpoints")
    if os.path.isdir(checkpoint_folder_path):
        found_files = [f for f in os.listdir(checkpoint_folder_path) if f.endswith(".safetensors")]
        if found_files:
            # Taking the first one if multiple (could be improved)
            checkpoint_filename = found_files[0]
            full_path = os.path.join(checkpoint_folder_path, checkpoint_filename).replace("\\", "/")
            logger.info(f"Using checkpoint: {full_path}")
            return full_path, None
        else:
            return None, f"Error: No .safetensors file found in {checkpoint_folder_path} for LoRA key '{selected_lora_key}'."
    else:
        return None, f"Error: Checkpoint directory not found for LoRA key '{selected_lora_key}': {checkpoint_folder_path}"


def _update_and_save_config_for_sampling(config_path: str, parameters: dict, checkpoint_path: str | None) -> str | None:
    try:
        # It's often better to start with a minimal or default config structure
        # rather than reading an existing one that might have unrelated settings.
        # However, if the script `sample_video.py` expects a full config, reading might be needed.
        # For now, creating a focused config for sampling:
        
        sampling_config = {'model': {}, 'validation': {}, 'output_dir': "workspace/output/manual_samples"}

        if checkpoint_path:
            sampling_config['model']['load_checkpoint'] = checkpoint_path
        
        # Map parsed_parameters to the sampling_config structure expected by sample_video.py
        sampling_config['validation']['prompts'] = parameters.get("prompts", [])
        sampling_config['validation']['negative_prompt'] = parameters.get("negative_prompt", "")
        sampling_config['validation']['video_dims'] = parameters.get("video_dims", [512,512,49])
        sampling_config['validation']['seed'] = parameters.get("seed", 42)
        sampling_config['validation']['inference_steps'] = parameters.get("inference_steps", 25)
        sampling_config['validation']['guidance_scale'] = parameters.get("guidance_scale", 3.5)
        sampling_config['validation']['videos_per_prompt'] = parameters.get("videos_per_prompt", 1)
        
        # Handle images for sampling
        images_for_sampling = parameters.get("images_for_sampling", [])
        if images_for_sampling:
            sampling_config['validation']['images'] = images_for_sampling
        else:
            # Ensure 'images' key is None or absent if no images are provided for sampling
            sampling_config['validation']['images'] = None 

        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(sampling_config, f, indent=2, sort_keys=False, allow_unicode=True, Dumper=yaml.SafeDumper)
        logger.info(f"Sampling configuration saved to {config_path}")
        return None
    except Exception as e:
        logger.error(f"Error creating/saving sampling config {config_path}: {e}")
        logger.error(traceback.format_exc())
        return f"Error saving sampling config: {e}"


async def _run_sampling_command_async(page: ft.Page, config_file_path: str) -> None:
    global sampling_console_output, sampling_progress_bar, sampling_status_text
    # The command should point to sample_video.py and pass the generated config
    # Assuming sample_video.py is in the project root or accessible via PATH
    # And it accepts a --config argument
    
    # Determine the path to sample_video.py relative to this file or project root
    # This needs to be robust. For example, if flet_app is the root for execution:
    sample_script_path = "sample_video.py" # Adjust if it's elsewhere, e.g., "scripts/sample_video.py"
    
    # Command construction: python sample_video.py --config path/to/config.yaml
    # Ensure paths are handled correctly (absolute or relative to script's CWD)
    # Using shlex.quote for safety with paths
    quoted_config_path = shlex.quote(os.path.abspath(config_file_path))
    command_parts = ["python", sample_script_path, "--config", quoted_config_path]
    command_str = " ".join(command_parts)

    logger.info(f"Executing sampling command asynchronously: {command_str}")

    if sampling_status_text:
        sampling_status_text.value = "Running sampling script..."
        sampling_status_text.color = ft.Colors.BLUE_ACCENT_700
        try: sampling_status_text.update()
        except: pass


    if sampling_console_output:
        sampling_console_output.value = f"Executing: {command_str}\n\n" # Show the command
        try: sampling_console_output.update()
        except: pass


    if sampling_progress_bar:
        sampling_progress_bar.value = None # Indeterminate progress
        sampling_progress_bar.visible = True
        try: sampling_progress_bar.update()
        except: pass

    try:
        process = await asyncio.create_subprocess_shell(
            command_str, # Use the constructed string command
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            # Set cwd if sample_script_path needs a specific working directory
            # cwd=os.path.abspath(".") # Example: run from project root
        )

        async def read_stream(stream, output_field, is_stderr=False):
            while True:
                line = await stream.readline()
                if not line: break
                output = line.decode('utf-8', errors='ignore').strip()
                if output and output_field:
                    prefix = "[ERR] " if is_stderr else "[OUT] "
                    logger.info(f"{prefix}{output}")
                    output_field.value += output + "\n"
                    try: output_field.update()
                    except: pass # Ignore update errors in async loop

        await asyncio.gather(
            read_stream(process.stdout, sampling_console_output),
            read_stream(process.stderr, sampling_console_output, is_stderr=True)
        )
        await process.wait()

        if sampling_status_text:
            if process.returncode == 0:
                sampling_status_text.value = "Sampling finished successfully."
                sampling_status_text.color = ft.Colors.GREEN_ACCENT_700
            else:
                sampling_status_text.value = f"Sampling failed (code {process.returncode}). Check console."
                sampling_status_text.color = ft.Colors.RED_ACCENT_700
            try: sampling_status_text.update()
            except: pass

    except FileNotFoundError: # Specifically for "python" or "sample_video.py" not found
        logger.error(f"Error: 'python' interpreter or script '{sample_script_path}' not found. Ensure Python is in PATH and script path is correct.")
        if sampling_status_text:
            sampling_status_text.value = "Error: Python or sampling script not found."
            sampling_status_text.color = ft.Colors.RED_ACCENT_700
            try: sampling_status_text.update()
            except: pass
        if sampling_console_output:
            sampling_console_output.value += "Error: 'python' or sampling script not found. Check setup.\n"
            try: sampling_console_output.update()
            except: pass
    except Exception as e:
        logger.error(f"Error running sampling command: {e}")
        logger.error(traceback.format_exc())
        if sampling_status_text:
            sampling_status_text.value = f"Error: {e}"
            sampling_status_text.color = ft.Colors.RED_ACCENT_700
            try: sampling_status_text.update()
            except: pass
        if sampling_console_output and hasattr(sampling_console_output, 'value'):
             sampling_console_output.value += f"\n--- Python Execution Error ---\n{traceback.format_exc()}\n"
             try: sampling_console_output.update();
             except: pass


    finally:
        if sampling_progress_bar:
            sampling_progress_bar.visible = False
            sampling_progress_bar.value = 0.0 # Reset
            try: sampling_progress_bar.update()
            except: pass


def _build_validation_config_section(page: ft.Page):
    global prompts_textfield, negative_prompt_textfield, video_dims_textfield, seed_textfield, \
           inference_steps_textfield, interval_textfield, videos_per_prompt_textfield, guidance_scale_textfield
    
    controls = []
    controls.extend(add_section_title("Validation Configuration (for Training Config File)")) # Clarify purpose
    
    prompts_textfield = create_textfield("Prompts",
                                            "CAKEIFY a person using a knife to cut a cake shaped like bottle of mouthwash\nCAKEIFY a person using a knife to cut a cake shaped like owl head",
                                            hint_text="Enter each prompt on a new line (used in training config)",
                                            multiline=True, min_lines=3, max_lines=5, expand=True, col=8)
    negative_prompt_textfield = create_textfield("Negative Prompt", "worst quality, inconsistent motion, blurry, jittery, distorted", 
                                                 multiline=True, min_lines=2, max_lines=5, expand=True, col=4)
    video_dims_textfield = create_textfield("Video Dims", "[512, 512, 49]", hint_text="[W, H, Frames]", expand=True)
    seed_textfield = create_textfield("Seed (Validation)", "42", expand=True) # Ensure string for textfield
    inference_steps_textfield = create_textfield("Inference Steps", "25", expand=True)
    interval_textfield = create_textfield("Interval (Validation)", "250", hint_text="0 or null to disable", expand=True)
    videos_per_prompt_textfield = create_textfield("Videos Per Prompt", "1", expand=True)
    guidance_scale_textfield = create_textfield("Guidance Scale", "3.5", expand=True)

    controls.append(ft.ResponsiveRow(controls=[prompts_textfield, negative_prompt_textfield], vertical_alignment=ft.CrossAxisAlignment.START))
    
    # Image selectors C1 and C2
    # Pass the page attribute name for storing the selected path
    c1_selector = _create_image_selector_container(page, image_display_c1, "selected_image_path_c1", 6)
    c2_selector = _create_image_selector_container(page, image_display_c2, "selected_image_path_c2", 6)

    # Layout for image selectors and other validation fields
    validation_params_col = ft.Column(
        controls=[
            ft.Row([video_dims_textfield, seed_textfield, inference_steps_textfield]),
            ft.Row([interval_textfield, videos_per_prompt_textfield, guidance_scale_textfield])
        ],
        col=6, expand=True, spacing=5
    )
    image_selectors_row = ft.ResponsiveRow(
        controls=[c1_selector, c2_selector],
        # vertical_alignment=ft.CrossAxisAlignment.START,
        alignment=ft.MainAxisAlignment.SPACE_AROUND,
        col=6, expand=True
    )
    controls.append(ft.Divider(thickness=1))
    controls.append(ft.ResponsiveRow(controls=[image_selectors_row, validation_params_col], vertical_alignment=ft.CrossAxisAlignment.START))
    return controls


def _build_sample_videos_section(page: ft.Page): # Added page
    global model_source_dropdown, lora_dropdown, lora_str_textfield, sampling_console_output, sampling_progress_bar
    
    # Video Placeholders (Grid) - mostly decorative for now
    video_placeholders = [
        ft.Container(
            content=ft.Column([ft.Icon(ft.Icons.VIDEO_LIBRARY, size=48, color=ft.Colors.GREY_400),
                               ft.Text(f"Video {i+1}", color=ft.Colors.GREY_500, size=14, italic=True)],
                              alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
            width=180, height=120, bgcolor=ft.Colors.GREY_100, border=ft.border.all(1, ft.Colors.GREY_300),
            border_radius=8, alignment=ft.alignment.center, margin=ft.margin.all(5),
        ) for i in range(3)
    ]
    
    initial_lora_options_map = _get_lora_options()

    lora_dropdown = create_dropdown("Select LoRA (for Sampling)", "none", options=initial_lora_options_map,
                                    hint_text="Select LoRA model for sampling", col=6, expand=True)

    def update_lora_dropdown_options(_):
        latest_lora_map = _get_lora_options()
        lora_dropdown.options = [ft.dropdown.Option(key=k, text=v) for k, v in latest_lora_map.items()]
        try: lora_dropdown.update()
        except Exception as e: logger.error(f"Error updating lora_dropdown: {e}")


    update_lora_btn = ft.IconButton(icon=ft.Icons.REFRESH, tooltip="Refresh LoRA list", on_click=update_lora_dropdown_options,
                                   icon_size=20, col=1) # Adjusted col span

    lora_str_textfield = create_textfield("LoRA Strength (Not Used)", "1.0", hint_text="Currently not passed to script", col=2, expand=True, disabled=True)
    model_source_dropdown = create_dropdown("Base Model (Not Used)", settings.ltx_def_model, settings.ltx_model_dict,
                                            hint_text="Base model selection (currently not passed to sampling script)", col=3, expand=True, disabled=True)

    sampling_progress_bar = ft.ProgressBar(value=0.0, visible=False, bar_height=10)
    sampling_console_output = ft.TextField(label="Sampling Script Output", text_size=10, multiline=True, read_only=True,
                                           expand=True, min_lines=10, max_lines=20, border_color=ft.Colors.BLUE_GREY_800,
                                           bgcolor=ft.Colors.BLACK12, color=ft.Colors.WHITE70)

    return _build_expansion_tile(
        title="Manual Video Sampling",
        controls=[
            ft.Text("Generate sample videos using selected prompts, images, and LoRA.", size=12, italic=True),
            ft.Divider(height=5, color=ft.Colors.TRANSPARENT),
            # ft.GridView(controls=video_placeholders, max_extent=200, child_aspect_ratio=1.5, spacing=5, run_spacing=10, expand=False, height=140), # Optional: keep placeholders
            ft.ResponsiveRow(controls=[lora_dropdown, update_lora_btn, lora_str_textfield, model_source_dropdown], spacing=5, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            ft.Row([create_styled_button("Generate Sample Video", lambda e: run_manual_sample_video(e.page))], alignment=ft.MainAxisAlignment.END), # Pass page
            ft.Divider(height=10, color=ft.Colors.TRANSPARENT),
            ft.Text("Console Output:", weight=ft.FontWeight.BOLD),
            sampling_progress_bar,
            sampling_console_output
        ],
        initially_expanded=False # Expand by default
    )


def build_training_sampling_page_content(page: ft.Page):
    global file_picker, sampling_status_text # Ensure file_picker is global

    # Initialize page attributes for selected image paths if not already present
    # These will be the single source of truth for which image is selected for C1/C2.
    if not hasattr(page, 'selected_image_path_c1'): page.selected_image_path_c1 = None
    if not hasattr(page, 'selected_image_path_c2'): page.selected_image_path_c2 = None
    
    # Attach the actual Flet Image controls to the page for access by utils_top_menu.py
    page.image_display_c1 = image_display_c1 
    page.image_display_c2 = image_display_c2
    
    page_controls = []

    # Define the on_dialog_result handler within this scope to capture `page`
    def on_dialog_result_for_page(e: ft.FilePickerResultEvent):
        nonlocal page # Explicitly capture page from the outer scope
        global active_image_target_control # This is fine as global
        
        if e.files and active_image_target_control:
            selected_file_path = e.files[0].path.replace('\\', '/') # Normalize path
            
            target_page_attr_name = None
            if active_image_target_control == image_display_c1:
                target_page_attr_name = "selected_image_path_c1"
                global selected_image_path_c1; selected_image_path_c1 = selected_file_path # Update global if needed for other logic
            elif active_image_target_control == image_display_c2:
                target_page_attr_name = "selected_image_path_c2"
                global selected_image_path_c2; selected_image_path_c2 = selected_file_path # Update global
            else:
                logger.warning("File picked but active_image_target_control is not c1 or c2. Ignoring.")
                active_image_target_control = None # Reset
                return

            try:
                # Update the page attribute (single source of truth for path)
                setattr(page, target_page_attr_name, selected_file_path)

                # Update the Flet Image control
                active_image_target_control.src = selected_file_path
                active_image_target_control.visible = True
                active_image_target_control.update()

                if hasattr(active_image_target_control, 'show_remove_button'):
                    active_image_target_control.show_remove_button()
                
                if hasattr(active_image_target_control, 'parent') and active_image_target_control.parent and hasattr(active_image_target_control.parent, 'update'):
                    active_image_target_control.parent.update()
                
                page.update() # Update the whole page
                logger.info(f"Image {target_page_attr_name} successfully updated from file picker.")
            except Exception as ex:
                logger.error(f"Error processing selected file for {target_page_attr_name}: {ex}")
                logger.error(traceback.format_exc())
            finally:
                active_image_target_control = None # Reset for next pick
        else:
            active_image_target_control = None # Reset

    # Initialize file_picker ONCE and add to overlay
    if file_picker is None:
        file_picker = ft.FilePicker(on_result=on_dialog_result_for_page) # Use the page-scoped handler
        if file_picker not in page.overlay: # Check before adding
            page.overlay.append(file_picker)
    else: # If already exists, ensure its on_result is correctly set (e.g. if page reloads)
        file_picker.on_result = on_dialog_result_for_page
        if file_picker not in page.overlay: # Still check and add if somehow removed
             page.overlay.append(file_picker)
    
    page.update() # Ensure overlay is processed

    page_controls.extend(_build_validation_config_section(page))
    page_controls.append(ft.Divider(height=5, color=ft.Colors.TRANSPARENT))
    page_controls.append(_build_sample_videos_section(page)) # Pass page

    sampling_status_text = ft.Text("", color=ft.Colors.BLUE_GREY_400, italic=True, size=12) # Initialize
    page_controls.append(ft.Row([sampling_status_text], alignment=ft.MainAxisAlignment.CENTER))

    return ft.Container(
        content=ft.Column(
            controls=page_controls, spacing=10, # Increased spacing
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH, # Stretch to fill width
            scroll=ft.ScrollMode.ADAPTIVE, # Adaptive scrolling
        ),
        expand=True, padding=ft.padding.all(10) # Increased padding
    )


def run_manual_sample_video(page: ft.Page): # Changed from event `e` to `page`
    global sampling_status_text # Ensure this is the global status text control
    
    sampling_config_filename = "temp_manual_sample_config.yaml"
    # Place in a known, writable directory, e.g., workspace/temp
    temp_config_dir = os.path.abspath(os.path.join("workspace", "temp"))
    os.makedirs(temp_config_dir, exist_ok=True)
    config_path_for_sampling = os.path.join(temp_config_dir, sampling_config_filename)

    logger.info(f"Running manual sample video. Config will be saved to: {config_path_for_sampling}")

    if sampling_status_text:
        sampling_status_text.value = "Preparing for manual sampling..."
        sampling_status_text.color = ft.Colors.BLUE_ACCENT_700
        try: sampling_status_text.update()
        except: pass

    parameters_from_ui = _collect_sampling_parameters(page) # Pass page

    parsed_parameters, validation_error = _validate_and_parse_parameters(parameters_from_ui)
    if validation_error:
        logger.error(f"Parameter validation failed for manual sampling: {validation_error}")
        if sampling_status_text:
            sampling_status_text.value = validation_error
            sampling_status_text.color = ft.Colors.RED_ACCENT_700
            try: sampling_status_text.update()
            except: pass
        return
    logger.debug(f"Parsed parameters for manual sampling: {parsed_parameters}")

    # Get LoRA options again to ensure the map is current for path construction
    current_lora_options_map = _get_lora_options()
    selected_lora_key = parsed_parameters["selected_lora_key"]
    checkpoint_path, checkpoint_error = _get_checkpoint_path_from_selection(selected_lora_key, current_lora_options_map)
    
    if checkpoint_error and selected_lora_key != "none": # Error only if a LoRA was selected but not found
        logger.warning(f"Checkpoint path determination failed for manual sampling: {checkpoint_error}")
        if sampling_status_text:
            sampling_status_text.value = checkpoint_error
            sampling_status_text.color = ft.Colors.ORANGE_ACCENT_700
            try: sampling_status_text.update()
            except: pass
        return # Stop if a selected LoRA is problematic
    elif selected_lora_key == "none":
        logger.info("No LoRA selected for manual sampling, proceeding without specific checkpoint.")
        checkpoint_path = None # Explicitly None

    logger.debug(f"Checkpoint path for manual sampling: {checkpoint_path}")

    save_error = _update_and_save_config_for_sampling(config_path_for_sampling, parsed_parameters, checkpoint_path)
    if save_error:
        logger.error(f"Failed to save temporary config for manual sampling: {save_error}")
        if sampling_status_text:
            sampling_status_text.value = f"Config save error: {save_error}"
            sampling_status_text.color = ft.Colors.RED_ACCENT_700
            try: sampling_status_text.update()
            except: pass
        return

    logger.info(f"Temporary config for manual sampling saved. Starting async task.")
    page.run_task(_run_sampling_command_async, page, config_path_for_sampling) # Pass page and config_path

    if sampling_status_text:
        sampling_status_text.value = "Manual sampling command sent. Check console output below."
        sampling_status_text.color = ft.Colors.GREEN_ACCENT_700
        try: sampling_status_text.update()
        except: pass

