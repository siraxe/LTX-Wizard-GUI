import flet as ft
import os
from rich.color import Color
import yaml # Import yaml for potential future use or reference, though not strictly needed for the CLI approach
import shlex # For splitting command strings safely
import json # To parse video_dims list
import subprocess # Import subprocess module
import asyncio # Import asyncio for asynchronous operations
from .._styles import create_textfield, add_section_title,create_dropdown ,create_styled_button
from settings import settings
from loguru import logger # Import logger for error logging
from ui.tab_dataset_view import _build_expansion_tile # Import the necessary helper function

# Global variables to store references to Flet controls and console output
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
sampling_status_text = None # For displaying the sampling status

# Console output controls
sampling_console_output = None
sampling_progress_bar = None

# List to store paths of images dropped from OS
dropped_image_paths = []

# Add a flag to track pending save
pending_save = {'should_save': False, 'yaml_dict': None, 'out_path': None}

# New global variables for file picker and image display
file_picker = None
active_image_target_control = None # To keep track of which image display to update

image_display_c1 = ft.Image(
    src="/images/image_placeholder.png",  # Placeholder image
    width=200,
    height=200,
    fit=ft.ImageFit.CONTAIN
)
image_display_c2 = ft.Image(
    src="/images/image_placeholder.png",  # Placeholder image
    width=200,
    height=200,
    fit=ft.ImageFit.CONTAIN
)

selected_image_path_c1 = None # New global variable to store the selected image path for c1
selected_image_path_c2 = None # New global variable to store the selected image path for c2

# Global on_pick_files_result removed as it's superseded by the inner on_dialog_result
# defined within build_training_sampling_page_content and assigned to the global file_picker instance.

def _create_image_selector_container(image_control: ft.Image, image_path_var: str, col_span: int):
    def _on_click(e):
        global active_image_target_control
        active_image_target_control = image_control
        file_picker.pick_files(allow_multiple=False, allowed_extensions=["png", "jpg", "jpeg", "gif"])

    return ft.Container(
        content=ft.Column(
            [image_control, ft.Text("Click to select image")],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            alignment=ft.MainAxisAlignment.CENTER,
        ),
        width=200,
        height=250,
        border=ft.border.all(1, ft.Colors.BLUE_GREY_400),
        border_radius=ft.border_radius.all(5),
        alignment=ft.alignment.center,
        on_click=_on_click,
        col=col_span
    )

def _get_lora_options() -> dict[str, str]:
    """
    Searches for .safetensors files in workspace/output/<name>/checkpoints/
    and returns a dictionary for the LoRA dropdown.
    """
    lora_options = {"none": "none"}
    output_dir = "workspace/output"
    if not os.path.exists(output_dir):
        logger.warning(f"Output directory not found: {output_dir}")
        return lora_options

    for name in os.listdir(output_dir):
        checkpoints_dir = os.path.join(output_dir, name, "checkpoints")
        if os.path.isdir(checkpoints_dir):
            for filename in os.listdir(checkpoints_dir):
                if filename.endswith(".safetensors"):
                    # Key is the folder name (name), value is checkpoints/filename
                    lora_options[name] = os.path.join("checkpoints", filename).replace("\\", "/") # Use forward slashes for path consistency

    logger.debug(f"Found {len(lora_options) - 1} LoRA checkpoints.")
    return lora_options

def _collect_sampling_parameters() -> dict:
    """
    Collects sampling parameters from the UI controls.
    Returns a dictionary of collected parameters.
    """
    return {
        # model_source is not directly used by sample_video.py CLI args anymore
        "selected_lora_key": lora_dropdown.value if lora_dropdown else "none",
        "single_prompt": prompts_textfield.value.strip() if prompts_textfield else "",
        "negative_prompt": negative_prompt_textfield.value.strip() if negative_prompt_textfield else None,
        "video_dims_text": video_dims_textfield.value if video_dims_textfield else "[512, 512, 49]",
        "seed_text": seed_textfield.value if seed_textfield else "42",
        "inference_steps_text": inference_steps_textfield.value if inference_steps_textfield else "25",
        "guidance_scale_text": guidance_scale_textfield.value if guidance_scale_textfield else "3.5",
        "videos_per_prompt_text": videos_per_prompt_textfield.value if videos_per_prompt_textfield else "1",
        "selected_image_path_c1": selected_image_path_c1, # Include the selected image path for c1
        "selected_image_path_c2": selected_image_path_c2 # Include the selected image path for c2
    }


def _validate_and_parse_parameters(params: dict) -> tuple[dict, str | None]:
    """
    Validates and parses collected parameters. Returns a dictionary of parsed values
    and an error message if validation fails.
    """
    single_prompt = params["single_prompt"]
    video_dims_text = params["video_dims_text"]
    seed_text = params["seed_text"]
    inference_steps_text = params["inference_steps_text"]
    guidance_scale_text = params["guidance_scale_text"]
    videos_per_prompt_text = params["videos_per_prompt_text"]

    if not single_prompt:
        return {}, "Error: No prompt provided."

    try:
        video_dims_list = json.loads(video_dims_text)
        if not (isinstance(video_dims_list, list) and len(video_dims_list) == 3 and all(isinstance(x, int) for x in video_dims_list)):
             raise ValueError("Expected a list of 3 integers.")
    except (json.JSONDecodeError, ValueError) as e:
        return {}, f"Error parsing video dimensions: {e}"

    try:
        seed = int(seed_text)
    except ValueError:
        return {}, "Error: Invalid seed value. Must be an integer."

    try:
        inference_steps = int(inference_steps_text)
    except ValueError:
        return {}, "Error: Invalid inference steps value. Must be an integer."

    try:
        guidance_scale = float(guidance_scale_text)
    except ValueError:
         return {}, "Error: Invalid guidance scale value. Must be a number."

    try:
        videos_per_prompt = int(videos_per_prompt_text)
        if videos_per_prompt <= 0:
             raise ValueError("Must be a positive integer.")
    except ValueError:
         return {}, "Error: Invalid videos per prompt value. Must be a positive integer."

    return {
        "prompts": [single_prompt], # sample_video.py expects a list
        "negative_prompt": params["negative_prompt"],
        "video_dims": video_dims_list,
        "seed": seed,
        "inference_steps": inference_steps,
        "guidance_scale": guidance_scale,
        "videos_per_prompt": videos_per_prompt,
        "selected_lora_key": params["selected_lora_key"], # Pass through for checkpoint path logic
    }, None


def _get_checkpoint_path(selected_lora_key: str) -> tuple[str | None, str | None]:
    """
    Determines the full checkpoint path based on the selected LoRA key.
    Returns the full path and an error message if selection is invalid.
    """
    if selected_lora_key and selected_lora_key != "none":
         try:
            # Assuming lora_dropdown.options are populated with key=folder_name, text=relative_path
            selected_option = next(opt for opt in lora_dropdown.options if opt.key == selected_lora_key)
            lora_relative_path = selected_option.text
            folder_name = selected_lora_key # Key is the folder name
            full_checkpoint_path = os.path.join("workspace", "output", folder_name, lora_relative_path).replace("\\", "/")
            return full_checkpoint_path, None
         except StopIteration:
             return None, f"Error: Could not find selected LoRA option key: {selected_lora_key}"
    elif selected_lora_key == "none":
         return None, "Please select a LoRA checkpoint to sample."
    else:
         return None, "Error: Invalid LoRA selection."


def _update_and_save_config(config_path: str, parameters: dict, checkpoint_path: str | None) -> str | None:
    """
    Loads the existing config, updates it with parameters and checkpoint path,
    and saves it back to the specified path.
    Returns an error message if saving fails, otherwise None.
    """
    try:
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)

        # Update the yaml_config dictionary with UI values
        # These map to the 'validation' section or other top-level keys in sample_video.py's config loading
        if 'validation' not in yaml_config:
            yaml_config['validation'] = {}

        yaml_config['validation']['prompts'] = parameters["prompts"]
        yaml_config['validation']['negative_prompt'] = parameters["negative_prompt"]
        yaml_config['validation']['video_dims'] = parameters["video_dims"]
        yaml_config['validation']['seed'] = parameters["seed"]
        yaml_config['validation']['inference_steps'] = parameters["inference_steps"]
        yaml_config['validation']['guidance_scale'] = parameters["guidance_scale"]
        yaml_config['validation']['videos_per_prompt'] = parameters["videos_per_prompt"]
        
        # Add selected image path to validation if available
        if parameters.get("selected_image_path"):
            yaml_config['validation']['images'] = [parameters["selected_image_path"]]
        else:
            # Ensure 'images' key is not present or is None if no image is selected
            yaml_config['validation']['images'] = None

        # Handle checkpoint path for model loading
        if checkpoint_path:
            if 'model' not in yaml_config:
                yaml_config['model'] = {}
            yaml_config['model']['load_checkpoint'] = checkpoint_path

        # Update output directory (if needed, sample_video.py now reads this)
        yaml_config['output_dir'] = "workspace/output/manual_samples" # Use a fixed output dir for manual samples

        with open(config_path, 'w') as f:
            yaml.dump(yaml_config, f, indent=2)

        logger.info(f"Updated sampling configuration saved to {config_path}")
        return None # No error

    except Exception as e:
        logger.error(f"Error updating or saving config file {config_path}: {e}")
        return f"Error updating or saving config file {config_path}: {e}"


def _run_sampling_command() -> None:
    """
    Constructs and signals the terminal command to run start_sampling.bat.
    """
    # The command now just calls the batch file which uses the updated config
    command = f"call start_sampling.bat"

    # Print the command for the user to execute manually
    print("Please run the following command in your terminal:")
    print(command)
    logger.info(f"Generated command for manual execution: {command}")

# Add the async command runner function here, rewritten cleanly
async def _run_sampling_command_async(page: ft.Page) -> None:
    """
    Runs the start_sampling.bat command asynchronously and updates the console output.
    """
    global sampling_console_output, sampling_progress_bar, sampling_status_text

    command = "start_sampling.bat"
    logger.info(f"Executing command asynchronously: {command}")

    if sampling_status_text:
        sampling_status_text.value = "Running sampling script... Check console below for output."
        sampling_status_text.color = ft.Colors.BLUE_ACCENT_700
        sampling_status_text.update()

    if sampling_console_output:
        sampling_console_output.value = ""
        sampling_console_output.update()

    if sampling_progress_bar:
        sampling_progress_bar.value = 0.0 # Reset progress bar
        sampling_progress_bar.visible = True
        sampling_progress_bar.update()

    try:
        # Use asyncio.create_subprocess_shell for running batch files
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Read stdout and stderr concurrently
        async def read_stream(stream, output_field, is_stderr=False):
            while True:
                line = await stream.readline()
                if not line:
                    break
                output = line.decode('utf-8', errors='ignore').strip()
                if output:
                    logger.info(f"[CMD {'ERR' if is_stderr else 'OUT'}] {output}")
                    if output_field:
                        output_field.value += output + "\n"
                        output_field.update()

        await asyncio.gather(
            read_stream(process.stdout, sampling_console_output, is_stderr=False),
            read_stream(process.stderr, sampling_console_output, is_stderr=True)
        )

        # Wait for the process to finish
        await process.wait()

        if sampling_status_text:
            if process.returncode == 0:
                sampling_status_text.value = "Sampling script finished successfully."
                sampling_status_text.color = ft.Colors.GREEN_ACCENT_700
                logger.info("Sampling script finished successfully.")
            else:
                sampling_status_text.value = f"Sampling script failed with return code {process.returncode}. Check console for errors."
                sampling_status_text.color = ft.Colors.RED_ACCENT_700
                logger.error(f"Sampling script failed with return code {process.returncode}.")
            sampling_status_text.update()

    except Exception as e:
        logger.error(f"Error running sampling command: {e}")
        if sampling_status_text:
            sampling_status_text.value = f"Error running sampling command: {e}"
            sampling_status_text.color = ft.Colors.RED_ACCENT_700
            sampling_status_text.update()

    finally:
        if sampling_progress_bar:
            sampling_progress_bar.value = 1.0 # Complete progress bar
            sampling_progress_bar.visible = False # Optionally hide progress bar on completion
            sampling_progress_bar.update()

# --- UI Building Functions ---

def _build_validation_config_section(page: ft.Page):
    """Builds the Flet controls for the Validation Configuration section."""
    global prompts_textfield, negative_prompt_textfield, video_dims_textfield, seed_textfield, inference_steps_textfield, interval_textfield, videos_per_prompt_textfield, guidance_scale_textfield
    controls = []
    controls.extend(add_section_title("Validation Configuration"))
    
    # Assign controls to global variables and set default value for prompts as a single string
    prompts_textfield = create_textfield("Prompts",
                                            "CAKEIFY a person using a knife to cut a cake shaped like bottle of mouthwash \nCAKEIFY a person using a knife to cut a cake shaped like owl head",
                                            hint_text="Enter each prompt on a new line",
                                            multiline=True,
                                            min_lines=3, max_lines=3,
                                            expand=True, col=8) # Modified to single line input
    negative_prompt_textfield = create_textfield("Negative Prompt", "worst quality, inconsistent motion, blurry, jittery, distorted", multiline=True, min_lines=2, max_lines=5, expand=True, col=4)
    video_dims_textfield = create_textfield("Video Dims", "[512, 512, 49]", hint_text="width, height, frames", expand=True)
    seed_textfield = create_textfield("Seed (Validation)", 42, expand=True)
    inference_steps_textfield = create_textfield("Inference Steps", 25, expand=True)
    interval_textfield = create_textfield("Interval (Validation)", 250, hint_text="Set to null to disable validation", expand=True)
    videos_per_prompt_textfield = create_textfield("Videos Per Prompt", 1, expand=True)
    guidance_scale_textfield = create_textfield("Guidance Scale", 3.5, expand=True)

    controls.append(ft.ResponsiveRow(controls=[
        prompts_textfield,
        negative_prompt_textfield
    ], vertical_alignment=ft.CrossAxisAlignment.START))

    r1= ft.Row(controls=[
        video_dims_textfield,
        seed_textfield,
        inference_steps_textfield
    ],col=4, expand=True)

    r2 = ft.Row(controls=[
        interval_textfield,
        videos_per_prompt_textfield,
        guidance_scale_textfield
    ],col=4, expand=True)

    r_sum = ft.Column(controls=[
        r1,
        r2
    ],col=6, expand=True)

    # Define the drag and drop area container
    image_drop_area_content = ft.Container(
        content=ft.Text("Drag and drop images here"),
        alignment=ft.alignment.center,
        width=400,
        height=200,
        border=ft.border.all(1, ft.Colors.BLUE_GREY_400),
        border_radius=5,
    )

    # New visual drop target using ft.Container
    image_drop_area = ft.Container(
        content=image_drop_area_content,
        width=400,
        height=200,
        border=ft.border.all(1, ft.Colors.BLUE_GREY_400), # Initial border
        border_radius=5,
        alignment=ft.alignment.center,
        on_click=lambda e: file_picker.pick_files(
            allow_multiple=False, allowed_extensions=["png", "jpg", "jpeg", "gif"]
        ),
    )
    c1 = _create_image_selector_container(image_display_c1, "selected_image_path_c1", 6)
    c2 = _create_image_selector_container(image_display_c2, "selected_image_path_c2", 6)

    controls.append(ft.Divider(thickness=1))
    controls.append(ft.ResponsiveRow(controls=[
        ft.Column(controls=[
                ft.ResponsiveRow(
                    controls=[c1,c2]
                ),
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            alignment=ft.MainAxisAlignment.CENTER,
            expand=True,col=6,
        ),
        r_sum
    ]))

    return controls

def _get_lora_options():
    """Fetches available LoRA models from the config and formats them for the dropdown."""
    lora_models_path = settings.LORA_MODELS_DIR
    lora_models = {}
    if os.path.exists(lora_models_path):
        for item in os.listdir(lora_models_path):
            item_path = os.path.join(lora_models_path, item)
            if os.path.isdir(item_path):
                lora_models[item] = item
    
    options = {name: name for name in lora_models.keys()}
    options["none"] = "None"
    return options

def _build_sample_videos_section():
    """Builds the Flet controls for the Sample videos section, including the video placeholders grid."""
    global model_source_dropdown, lora_dropdown, lora_str_textfield, sampling_console_output, sampling_progress_bar
    
    # --- Video Placeholders (Grid) ---
    video_placeholders = [
        ft.Container(
            content=ft.Column([
                ft.Icon(ft.Icons.VIDEO_LIBRARY, size=48, color=ft.Colors.GREY_400),
                ft.Text(f"Video Placeholder {i+1}", color=ft.Colors.GREY_500, size=14, italic=True),
            ], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
            width=180,
            height=120,
            bgcolor=ft.Colors.GREY_100,
            border=ft.border.all(1, ft.Colors.GREY_300),
            border_radius=8,
            alignment=ft.alignment.center,
            margin=ft.margin.all(5),
        ) for i in range(3)
    ]
    
    # Get initial LoRA options
    initial_lora_options = _get_lora_options()

    lora_dropdown = create_dropdown(
        "Select LoRA",
        "none", # Default value
        initial_lora_options,
        hint_text="Select model or specify path below",
        col=6, expand=True
    )

    def update_lora_dropdown(_):
        """Updates the LoRA dropdown options."""
        latest_lora_options = _get_lora_options()
        lora_dropdown.options = [ft.dropdown.Option(key, text=value) for key, value in latest_lora_options.items()]
        lora_dropdown.update()

    update_button = ft.IconButton(
        icon=ft.Icons.REFRESH,
        tooltip="Update dataset list",
        on_click=update_lora_dropdown, # Use the new update function
        style=ft.ButtonStyle(padding=ft.padding.symmetric(horizontal=5)),
        icon_size=20,col=1.5, expand=True
    )

    lora_str_textfield = create_textfield("LoRA srt", "1.0", hint_text="lora strength", col=1.5, expand=True)

    model_source_dropdown = create_dropdown(
        "Model Source",
        settings.ltx_def_model,
        settings.ltx_model_dict,
        hint_text="Select model or specify path below",col=3, expand=True
    )

    sampling_progress_bar = ft.ProgressBar(value=0.0, visible=False) # Initialize progress bar

    sampling_console_output = ft.TextField(
        label="Script Output",
        text_size=10,
        multiline=True,
        read_only=True,
        expand=True,
        min_lines=10, # Adjust as needed
        max_lines=20, # Adjust as needed
        border_color=ft.Colors.BLUE_GREY_800,
        bgcolor=ft.Colors.BLUE_GREY_900,
        color=ft.Colors.WHITE70,
    ) # Initialize console output field

    # Wrap all controls in an ExpansionTile
    return _build_expansion_tile(
        title="Sample Videos (WIP , not working)",
        controls=[
            ft.Divider(height=5, color=ft.Colors.TRANSPARENT),
            ft.GridView(
                controls=video_placeholders,
                max_extent=200,
                child_aspect_ratio=1.5,
                spacing=5,
                run_spacing=10,
                expand=False,
                height=140
            ),
            ft.ResponsiveRow(controls=[
                model_source_dropdown,
                lora_dropdown,
                lora_str_textfield,
                update_button
                ],spacing=3,
            ),
            ft.Row(
                controls=[
                    create_styled_button("Sample Video", run_sample_video)
                ],
                alignment=ft.MainAxisAlignment.END # Align content to the right
            ),
            ft.Divider(height=5, color=ft.Colors.TRANSPARENT),
            ft.Text("Console Output", weight=ft.FontWeight.BOLD),
            sampling_progress_bar,
            sampling_console_output
        ],
        initially_expanded=False
    )


def build_training_sampling_page_content(page: ft.Page):
    """
    Generates and assembles Flet controls for the training and sampling page.
    Returns a Container with all page content.
    """
    global file_picker, sampling_status_text

    # Initialize selected image paths and display controls on the page object
    page.selected_image_path_c1 = None
    page.selected_image_path_c2 = None
    page.image_display_c1 = image_display_c1
    page.image_display_c2 = image_display_c2
    logger.info("[training_sampling.py] Initialized page image paths and display controls")
    page_controls = []

    def on_dialog_result(e: ft.FilePickerResultEvent):
        # This 'page' is from the outer scope of build_training_sampling_page_content
        global active_image_target_control, selected_image_path_c1, selected_image_path_c2 
        
        if e.files and active_image_target_control:
            selected_file_path = e.files[0].path
            
            # 'page' is lexically scoped from build_training_sampling_page_content

            if active_image_target_control == image_display_c1:
                selected_image_path_c1 = selected_file_path # Update global if still used elsewhere directly
                page.selected_image_path_c1 = selected_file_path # Set on page object
                logger.info(f"[training_sampling.py on_dialog_result] page.selected_image_path_c1 set to: {page.selected_image_path_c1}")
                # Update both the global and page's image display control
                image_display_c1.src = selected_file_path
                page.image_display_c1.src = selected_file_path
                image_display_c1.update()
            elif active_image_target_control == image_display_c2:
                selected_image_path_c2 = selected_file_path # Update global if still used elsewhere directly
                page.selected_image_path_c2 = selected_file_path # Set on page object
                logger.info(f"[training_sampling.py on_dialog_result] page.selected_image_path_c2 set to: {page.selected_image_path_c2}")
                # Update both the global and page's image display control
                image_display_c2.src = selected_file_path
                page.image_display_c2.src = selected_file_path
                image_display_c2.update()
            
            page.update() # Update the page to reflect image changes
        
        # Reset active_image_target_control after the dialog interaction is complete,
        # regardless of whether files were selected. This ensures the next image selection
        # process starts with a clean state for active_image_target_control.
        active_image_target_control = None
    global file_picker # Ensure we are using the global file_picker instance
    if file_picker is None:
        file_picker = ft.FilePicker(on_result=on_dialog_result) # Assign the inner on_dialog_result
        page.overlay.append(file_picker)
    else: # If already initialized, ensure its on_result is this function instance
        file_picker.on_result = on_dialog_result # CRITICAL: Ensure this specific on_dialog_result is used
        if file_picker not in page.overlay:
             page.overlay.append(file_picker)

    # Also, ensure the global on_pick_files_result (if it exists and is different) is not interfering.
    # For clarity, it would be best to have only one on_result handler for the file_picker.
    # The one defined within this function now correctly uses the 'page' object from its scope.
    # The global file_picker instance, with its on_result correctly set to the inner on_dialog_result,
    # is already added to page.overlay, so no need to re-initialize or add to page_controls here.

    # Add Validation Configuration section
    page_controls.extend(_build_validation_config_section(page))

    page_controls.append(ft.Divider(height=5, color=ft.Colors.TRANSPARENT))

    # Add Sample videos section
    page_controls.append(_build_sample_videos_section())

    # Add status text
    sampling_status_text = ft.Text("", color=ft.Colors.BLUE_ACCENT_700, italic=True)
    page_controls.append(ft.Row([sampling_status_text]))

    return ft.Container(
        content=ft.Column(
            controls=page_controls,
            spacing=5,
            run_spacing=5,
            tight=True,
            horizontal_alignment=ft.CrossAxisAlignment.START,
            scroll=ft.ScrollMode.AUTO,
        ),
        expand=True,
        padding=ft.padding.all(5)
    )


# --- Event Handlers ---

def run_sample_video(e):
    """
    Event handler for the 'Sample Video' button.
    Collects UI parameters, validates them, updates the config file, and runs the sampling command.
    """
    global sampling_status_text
    config_path = "flet_app/assets/config_to_sample.yaml"

    if sampling_status_text:
        sampling_status_text.value = f"Collecting parameters and updating config file {config_path}..."
        sampling_status_text.color = ft.Colors.BLUE_ACCENT_700
        sampling_status_text.update()

    # Collect parameters from UI
    parameters = _collect_sampling_parameters()
    logger.debug(f"Collected parameters: {parameters}")

    # Validate and parse parameters
    parsed_parameters, validation_error = _validate_and_parse_parameters(parameters)
    if validation_error:
        if sampling_status_text:
            sampling_status_text.value = validation_error
            sampling_status_text.color = ft.Colors.RED_ACCENT_700
            sampling_status_text.update()
        logger.error(f"Parameter validation failed: {validation_error}")
        return
    logger.debug(f"Parsed parameters: {parsed_parameters}")

    # Determine checkpoint path
    selected_lora_key = parsed_parameters["selected_lora_key"]
    checkpoint_path, checkpoint_error = _get_checkpoint_path(selected_lora_key)
    if checkpoint_error:
        if sampling_status_text:
            sampling_status_text.value = checkpoint_error
            # Use orange for a warning about no LoRA selected, red for other errors
            sampling_status_text.color = ft.Colors.ORANGE_ACCENT_700 if selected_lora_key == "none" else ft.Colors.RED_ACCENT_700
            sampling_status_text.update()
        logger.warning(f"Checkpoint path determination failed: {checkpoint_error}")
        if selected_lora_key == "none":
            # Allow proceeding without a checkpoint if 'none' is explicitly selected,
            # although the sampling script might require one depending on its logic.
            logger.info("Proceeding without a checkpoint as 'none' was selected.")
            checkpoint_path = None # Explicitly set to None if 'none' was selected and allowed.
        else:
             # For other errors, stop the process
             return
    logger.debug(f"Determined checkpoint path: {checkpoint_path}")

    # Update and save the config file
    save_error = _update_and_save_config(config_path, parsed_parameters, checkpoint_path)
    if save_error:
         # Status text was already updated by _update_and_save_config
         return

    # Run the sampling command asynchronously and update the console
    e.page.run_task(_run_sampling_command_async, e.page)

    # Update status text
    if sampling_status_text:
        sampling_status_text.value = "Config updated and command sent for execution. Check terminal for progress."
        sampling_status_text.color = ft.Colors.GREEN_ACCENT_700 # Indicate success
        sampling_status_text.update()
