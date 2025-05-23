import flet as ft
import os
import json
import asyncio
import signal # Added signal import
from settings import config # Import only the config class


from ui.popups.video_player_dialog import open_video_captions_dialog
from ui.popups.delete_caption_dialog import show_delete_caption_dialog
from ui._styles import create_dropdown, create_styled_button, create_textfield, BTN_STYLE2
from ui.utils.utils_datasets import (
    get_dataset_folders,
    get_videos_and_thumbnails,
)



# ======================================================================================
# Global State (Keep track of UI controls and running processes)
# ======================================================================================

# References to UI controls
selected_dataset = {"value": None}
video_files_list = {"value": []}
thumbnails_grid_ref = ft.Ref[ft.GridView]()
processed_progress_bar = ft.ProgressBar(visible=False)
processed_output_field = ft.TextField(
    label="Processed Output", text_size=10, multiline=True, read_only=True,
    visible=False, min_lines=6, max_lines=15, expand=True)
bottom_app_bar_ref = None

# Global controls (defined here but created in _create_global_controls)
bucket_size_textfield: ft.TextField = None
rename_textfield: ft.TextField = None
model_name_dropdown: ft.Dropdown = None
trigger_word_textfield: ft.TextField = None

# References to controls created in dataset_tab_layout that need external access
# These refs will hold the actual control instances once created in dataset_tab_layout
dataset_dropdown_control_ref = ft.Ref[ft.Dropdown]()
dataset_add_captions_button_ref = ft.Ref[ft.ElevatedButton]()
dataset_delete_captions_button_ref = ft.Ref[ft.ElevatedButton]()
dataset_preprocess_button_ref = ft.Ref[ft.ElevatedButton]()
caption_model_dropdown_ref = ft.Ref[ft.Dropdown]()
captions_checkbox_ref = ft.Ref[ft.Checkbox]() # This one is a direct ft.Checkbox, so ref is valid
cap_command_textfield_ref = ft.Ref[ft.TextField]()
max_tokens_textfield_ref = ft.Ref[ft.TextField]()

# Process tracking for stopping script execution
current_caption_process = {"proc": None}


# ======================================================================================
# Data & Utility Functions (File I/O, data parsing, validation)
# ======================================================================================

def parse_bucket_string_to_list(raw_bucket_str: str) -> list[int] | None:
    """Parse a bucket size string to a list of integers."""
    raw_bucket_str = raw_bucket_str.strip()
    try:
        if raw_bucket_str.startswith('[') and raw_bucket_str.endswith(']'):
            parsed_list = json.loads(raw_bucket_str)
            if isinstance(parsed_list, list) and len(parsed_list) == 3 and all(isinstance(i, int) for i in parsed_list):
                return parsed_list
        elif 'x' in raw_bucket_str.lower():
            parts_x = raw_bucket_str.lower().split('x')
            if len(parts_x) == 3 and all(p.strip().isdigit() for p in parts_x):
                return [int(p.strip()) for p in parts_x]
    except (json.JSONDecodeError, ValueError):
        return None
    return None

def format_bucket_list_to_string(bucket_list: list) -> str:
    """Format a list of bucket size values to a string."""
    if isinstance(bucket_list, list) and len(bucket_list) == 3 and all(isinstance(i, (int, float)) for i in bucket_list):
        return f"[{bucket_list[0]}, {bucket_list[1]}, {bucket_list[2]}]"
    return config.DEFAULT_BUCKET_SIZE_STR

def load_dataset_config(dataset_name: str | None) -> tuple[str, str, str]:
    """Load bucket size, model name, and trigger word from dataset info.json, or return defaults."""
    bucket_to_set = config.DEFAULT_BUCKET_SIZE_STR
    model_to_set = config.ltx_def_model
    trigger_word_to_set = ''
    if dataset_name:
        dataset_info_json_path = os.path.join(config.DATASETS_DIR, dataset_name, "info.json")
        if os.path.exists(dataset_info_json_path):
            try:
                with open(dataset_info_json_path, 'r') as f:
                    loaded_data = json.load(f)
                    if isinstance(loaded_data, dict) and "bucket_model" in loaded_data:
                        bucket_model_data = loaded_data["bucket_model"]
                        if isinstance(bucket_model_data, dict):
                            size_val = bucket_model_data.get("size")
                            type_val = bucket_model_data.get("type")
                            trigger_word_val = bucket_model_data.get("trigger_word")
                            if isinstance(size_val, list):
                                bucket_to_set = format_bucket_list_to_string(size_val)
                            if isinstance(type_val, str):
                                model_to_set = type_val
                            if isinstance(trigger_word_val, str):
                                trigger_word_to_set = trigger_word_val
            except (json.JSONDecodeError, IOError):
                pass
    return bucket_to_set, model_to_set, trigger_word_to_set

def save_dataset_config(dataset_name: str, bucket_str: str, model_name: str, trigger_word: str) -> bool:
    """Save bucket size, model name, and trigger word to dataset info.json."""
    dataset_info_json_path = os.path.join(config.DATASETS_DIR, dataset_name, "info.json")
    parsed_bucket_list = parse_bucket_string_to_list(bucket_str)
    if parsed_bucket_list is None:
        # Attempt to parse default if user input is invalid
        parsed_bucket_list = parse_bucket_string_to_list(config.DEFAULT_BUCKET_SIZE_STR)
        # Fallback to hardcoded default if parsing default fails
        if parsed_bucket_list is None:
            parsed_bucket_list = [512, 512, 49]
    bucket_model_to_save = {
        "size": parsed_bucket_list,
        "type": model_name,
        "trigger_word": trigger_word or ''
    }
    full_config_data = {}
    try:
        if os.path.exists(dataset_info_json_path):
            with open(dataset_info_json_path, 'r') as f:
                content = f.read()
                if content.strip():
                    full_config_data = json.loads(content)
                    # Ensure the loaded data is a dictionary
                    if not isinstance(full_config_data, dict):
                        full_config_data = {}
                else:
                    full_config_data = {}
        full_config_data["bucket_model"] = bucket_model_to_save
        os.makedirs(os.path.dirname(dataset_info_json_path), exist_ok=True)
        with open(dataset_info_json_path, 'w') as f:
            json.dump(full_config_data, f, indent=4)
        return True
    except Exception:
        return False

def load_processed_map(dataset_name: str) -> dict | None:
    """Load processed.json for a dataset, or return None."""
    processed_json_path = os.path.join(config.DATASETS_DIR, dataset_name, "preprocessed_data", "processed.json")
    if os.path.exists(processed_json_path):
        try:
            with open(processed_json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def load_dataset_captions(dataset_name: str) -> list:
    """Load captions.json for a dataset, or return empty list."""
    dataset_captions_json_path = os.path.join(config.DATASETS_DIR, dataset_name, "captions.json")
    if os.path.exists(dataset_captions_json_path):
        try:
            with open(dataset_captions_json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return []
    return []

def delete_captions_file(dataset_name: str) -> bool:
    """Delete captions.json for a dataset."""
    captions_file_path = os.path.join(config.DATASETS_DIR, dataset_name, "captions.json")
    if os.path.exists(captions_file_path):
        try:
            os.remove(captions_file_path)
            return True
        except Exception:
            return False
    return False

def validate_bucket_values(W_val, H_val, F_val) -> list[str]:
    """Validate bucket size values and return a list of error messages."""
    errors = []
    if W_val is None or not isinstance(W_val, int) or W_val <= 0 or W_val % 32 != 0:
        errors.append(f"Width ({W_val}) must be a positive integer divisible by 32.")
    if H_val is None or not isinstance(H_val, int) or H_val <= 0 or H_val % 32 != 0:
        errors.append(f"Height ({H_val}) must be a positive integer divisible by 32.")
    # Adjusted validation for Frames based on typical dataset preprocessing requirements
    if F_val is None or not isinstance(F_val, int) or F_val <= 0:
         errors.append(f"Frames ({F_val}) must be a positive integer.")
    # Add specific validation if needed (e.g., F_val >= 5 and (F_val - 1) % 4 == 0) - keeping original for now
    if F_val is not None and (F_val < 5 or (F_val - 1) % 4 != 0):
         errors.append(f"Frames ({F_val}) invalid (must be >= 5 and 4n+1).")
    return errors

# ======================================================================================
# Script Command Building Functions (Generate CLI commands)
# ======================================================================================

def build_caption_command(
    dataset_folder_path: str,
    output_json_path: str,
    selected_model: str,
    use_8bit: bool,
    instruction: str,
    max_new_tokens: int,
) -> str:
    """Build the command string for the captioning script."""
    python_exe = os.path.normpath(os.path.join("venv", "Scripts", "python.exe"))
    script_file = os.path.normpath("scripts/caption_videos.py")

    command = f'"{python_exe}" -u "{script_file}" "{dataset_folder_path}/" --output "{output_json_path}" --captioner-type {selected_model}'

    if use_8bit:
        command += " --use-8bit"

    # Ensure instruction and max_new_tokens are included
    command += f' --instruction "{instruction}"'
    command += f' --max-new-tokens {max_new_tokens}'

    return command

def build_preprocess_command(
    input_captions_json_path: str,
    preprocess_output_dir: str,
    resolution_buckets_str: str,
    model_name_val: str,
    trigger_word: str,
) -> str:
    """Build the command string for the preprocessing script."""
    python_exe = os.path.normpath(os.path.join("venv", "Scripts", "python.exe"))
    script_file = os.path.normpath("scripts/preprocess_dataset.py")

    command = (
        f'"{python_exe}" -u "{script_file}" "{input_captions_json_path}" --output-dir "{preprocess_output_dir}" '
        f'--resolution-buckets "{resolution_buckets_str}" --caption-column "caption" '
        f'--video-column "media_path" --model-source "{model_name_val}"'
    )

    if trigger_word:
        command += f' --id-token "{trigger_word}"'

    return command

# ======================================================================================
# Async Task Runner & Process State Management (Handles running external scripts)
# ======================================================================================

def set_bottom_app_bar_height():
    """Adjust the height of the bottom app bar based on output field visibility."""
    global bottom_app_bar_ref
    if bottom_app_bar_ref is not None and bottom_app_bar_ref.page:
        if processed_output_field.visible:
            bottom_app_bar_ref.height = 240
        else:
            bottom_app_bar_ref.height = 0
        bottom_app_bar_ref.update()

async def run_dataset_script_command(
    command_str: str,
    page_ref: ft.Page,
    button_ref: ft.ElevatedButton,
    progress_bar_ref: ft.ProgressBar,
    output_field_ref: ft.TextField,
    original_button_text: str,
    delete_button_ref=None,
    thumbnails_grid_control=None,
    on_success_callback=None,
):
    """Run a dataset script command asynchronously and update UI with output. Supports stopping."""
    def append_output(text):
        output_field_ref.value += text
        output_field_ref.visible = True
        set_bottom_app_bar_height()
        if page_ref.client_storage:
            output_field_ref.update()
            page_ref.update()

    try:
        output_field_ref.value = ""
        output_field_ref.visible = True
        progress_bar_ref.visible = True
        set_bottom_app_bar_height()
        button_ref.disabled = True
        button_ref.text = f"{original_button_text.replace('Add', 'Adding')}..." # Dynamic button text
        if page_ref.client_storage:
            page_ref.update()

        process = await asyncio.create_subprocess_shell(
            command_str,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT
        )
        current_caption_process["proc"] = process

        assert process.stdout is not None
        while True:
            line = await process.stdout.readline()
            if not line:
                break
            append_output(line.decode(errors='replace'))

        rc = await process.wait()
        current_caption_process["proc"] = None

        if rc == 0:
            success_message = f"Script successful: {output_field_ref.value.splitlines()[-1] if output_field_ref.value else 'No output'}..."
            if page_ref.client_storage:
                page_ref.snack_bar = ft.SnackBar(content=ft.Text(success_message), open=True)
            if on_success_callback:
                on_success_callback()
        else:
            err_msg = f"Script failed with exit code {rc}\nLast output:\n{output_field_ref.value.splitlines()[-50:] if output_field_ref.value else 'N/A'}" # Show last 50 lines of error
            append_output(err_msg)
            if page_ref.client_storage:
                page_ref.snack_bar = ft.SnackBar(content=ft.Text(f"Script failed (code {rc}). Check output field for details."), open=True)

    except Exception as e:
        error_trace = f"Cmd failed: {e}\n"
        append_output(error_trace)
        if page_ref.client_storage:
            page_ref.snack_bar = ft.SnackBar(content=ft.Text(f"Command execution failed: {e}"), open=True)

    finally:
        current_caption_process["proc"] = None
        button_ref.text = original_button_text
        button_ref.disabled = False
        progress_bar_ref.visible = False
        set_bottom_app_bar_height() # Re-evaluate height in case output field is no longer visible
        # --- Restore Delete button after captioning completes or fails ---
        if delete_button_ref is not None and thumbnails_grid_control is not None:
            delete_button_ref.text = "Delete"
            delete_button_ref.on_click = lambda evt: on_delete_captions_click(evt, thumbnails_grid_control)
            delete_button_ref.tooltip = "Delete the captions.json file"
            # Only re-enable if there's a dataset selected (handled in on_dataset_dropdown_change, but good to be safe)
            delete_button_ref.disabled = not selected_dataset.get("value")
            delete_button_ref.update()

        if page_ref.client_storage:
            page_ref.update()

# ======================================================================================
# GUI Event Handlers (Handle user interactions)
# ======================================================================================

def on_rename_files_click(e: ft.ControlEvent):
    """
    Rename all video files in the selected dataset according to the rename_textfield value,
    appending _01, _02, etc. Update captions.json and info.json if they exist.
    Provide user feedback via snackbar.
    """
    current_dataset_name = selected_dataset.get("value")
    if not current_dataset_name:
        if e.page:
            e.page.snack_bar = ft.SnackBar(content=ft.Text("Error: No dataset selected for renaming."), open=True)
            e.page.update()
        return

    base_name = rename_textfield.value.strip()
    if not base_name:
        if e.page:
            e.page.snack_bar = ft.SnackBar(content=ft.Text("Error: Enter a base name in the rename field."), open=True)
            e.page.update()
        return

    dataset_folder_path = os.path.abspath(os.path.join(config.DATASETS_DIR, current_dataset_name))
    video_exts = ['.mp4', '.mov', '.avi', '.webm', '.mkv']
    video_files = [f for f in os.listdir(dataset_folder_path) if os.path.splitext(f)[1].lower() in video_exts]
    video_files.sort()  # Ensure consistent order

    if not video_files:
        if e.page:
            e.page.snack_bar = ft.SnackBar(content=ft.Text("No video files found to rename."), open=True)
            e.page.update()
        return

    # Prepare new names and check for collisions
    new_names = []
    for idx, old_name in enumerate(video_files, 1):
        ext = os.path.splitext(old_name)[1]
        new_name = f"{base_name}_{idx:02d}{ext}"
        new_names.append(new_name)

    # Check for duplicate new names
    if len(set(new_names)) != len(new_names):
        if e.page:
            e.page.snack_bar = ft.SnackBar(content=ft.Text("Naming collision detected. Aborting."), open=True)
            e.page.update()
        return

    # Ensure no existing file will be overwritten (excluding files being renamed)
    existing_files_set = set(video_files)
    for new_name in new_names:
        if new_name in existing_files_set and new_name not in [old for old, new in zip(video_files, new_names) if new == new_name]:
            if e.page:
                e.page.snack_bar = ft.SnackBar(content=ft.Text(f"File {new_name} already exists and is not part of this renaming batch. Aborting."), open=True)
                e.page.update()
            return

    # Rename files and build old_to_new map
    old_to_new = {}
    try:
        for old_name, new_name in zip(video_files, new_names):
            old_path = os.path.join(dataset_folder_path, old_name)
            new_path = os.path.join(dataset_folder_path, new_name)
            os.rename(old_path, new_path)
            old_to_new[old_name] = new_name
    except Exception as ex:
        if e.page:
            e.page.snack_bar = ft.SnackBar(content=ft.Text(f"Failed to rename {old_name} to {new_name}: {ex}"), open=True)
            e.page.update()
        return # Stop if any rename fails

    # Update captions.json if exists
    captions_path = os.path.join(dataset_folder_path, "captions.json")
    if os.path.exists(captions_path):
        try:
            with open(captions_path, "r", encoding="utf-8") as f:
                captions_data = json.load(f)
            changed = False
            # Update filename fields in-place (never duplicate entries)
            for entry in captions_data:
                for field in ("media_path", "video"): # Check relevant fields
                    if field in entry and entry[field] in old_to_new:
                        entry[field] = old_to_new[entry[field]]
                        changed = True
            if changed:
                with open(captions_path, "w", encoding="utf-8") as f:
                    json.dump(captions_data, f, indent=2, ensure_ascii=False)
        except Exception as ex:
            if e.page:
                e.page.snack_bar = ft.SnackBar(content=ft.Text(f"Failed to update captions.json after renaming: {ex}"), open=True)
                e.page.update()
            # Continue as file renaming was successful, but notify user

    # Update info.json if exists (rename keys, preserve values, never duplicate)
    info_path = os.path.join(dataset_folder_path, "info.json")
    if os.path.exists(info_path):
        try:
            with open(info_path, "r", encoding="utf-8") as f:
                info_data = json.load(f)
            changed = False
            # If info_data is a dict with video filename keys, rename keys
            if isinstance(info_data, dict):
                new_info_data = {}
                for k, v in info_data.items():
                    new_key = old_to_new.get(k, k)
                    new_info_data[new_key] = v
                    if new_key != k:
                        changed = True
                info_data = new_info_data
            # No need for recursive update for other types if structure is standardized
            if changed:
                with open(info_path, "w", encoding="utf-8") as f:
                    json.dump(info_data, f, indent=4, ensure_ascii=False)
        except Exception as ex:
            if e.page:
                e.page.snack_bar = ft.SnackBar(content=ft.Text(f"Failed to update info.json after renaming: {ex}"), open=True)
                e.page.update()
            # Continue as file renaming was successful, but notify user

    # Success feedback and UI update
    if e.page:
        e.page.snack_bar = ft.SnackBar(content=ft.Text(f"Renamed {len(video_files)} files successfully."), open=True)
        update_thumbnails(page_ctx=e.page, grid_control=thumbnails_grid_ref.current, force_refresh=True) # Force refresh to update image sources
        e.page.update()


def on_bucket_or_model_change(e: ft.ControlEvent):
    """Handle changes to bucket size, model name, or trigger word controls."""
    current_dataset_name = selected_dataset.get("value")
    if not current_dataset_name:
        if e.page:
            e.page.snack_bar = ft.SnackBar(
                content=ft.Text("No dataset selected. Configuration not saved."),
                open=True
            )
            e.page.update()
        return

    # Validate bucket size format before saving
    bucket_str_val = bucket_size_textfield.value.strip()
    parsed_bucket_list = parse_bucket_string_to_list(bucket_str_val)
    if parsed_bucket_list is None:
         if e.page:
            e.page.snack_bar = ft.SnackBar(
                content=ft.Text(f"Invalid Bucket Size format: '{bucket_str_val}'. Using default."),
                open=True
            )
            e.page.update()
            # Continue with default or previous valid value for saving config
            bucket_str_val = config.DEFAULT_BUCKET_SIZE_STR # Or load previous valid? Using default for simplicity.

    success = save_dataset_config(
        current_dataset_name,
        bucket_str_val, # Use validated/default value
        model_name_dropdown.value,
        trigger_word_textfield.value
    )
    if e.page:
        if success:
            e.page.snack_bar = ft.SnackBar(
                content=ft.Text(f"Config saved for {current_dataset_name}."),
                open=True
            )
        else:
            e.page.snack_bar = ft.SnackBar(
                content=ft.Text(f"Error saving config for {current_dataset_name}"),
                open=True
            )
        e.page.update()


def on_dataset_dropdown_change(ev: ft.ControlEvent, thumbnails_grid_control: ft.GridView, dataset_delete_captions_button_control: ft.ElevatedButton):
    """Handle dataset dropdown selection change."""
    # Hide output and progress bar when changing dataset
    if processed_output_field.page:
        processed_output_field.visible = False
        set_bottom_app_bar_height() # Recalculate height
    if processed_progress_bar.page:
        processed_progress_bar.visible = False

    selected_dataset["value"] = ev.control.value

    # Load and update config fields based on selected dataset
    bucket_val, model_val, trigger_word_val = load_dataset_config(selected_dataset["value"])
    if bucket_size_textfield: bucket_size_textfield.value = bucket_val
    # Ensure loaded model value is a valid choice, otherwise use default
    if model_name_dropdown: model_name_dropdown.value = model_val if model_val in config.ltx_models else config.ltx_def_model
    if trigger_word_textfield: trigger_word_textfield.value = trigger_word_val or ''

    # Update thumbnails for the new dataset
    update_thumbnails(page_ctx=ev.page, grid_control=thumbnails_grid_control)

    # Update delete captions button state
    if dataset_delete_captions_button_control:
        pass # Keep button always enabled


def on_update_button_click(e: ft.ControlEvent, dataset_dropdown_control, thumbnails_grid_control, add_button, delete_button):
    """Handle the update dataset list button click."""
    reload_current_dataset(e.page, dataset_dropdown_control, thumbnails_grid_control, add_button, delete_button)


# Need access to caption_model_dropdown, captions_checkbox, cap_command, max_tokens_textfield
# These are currently local to dataset_tab_layout. Let's define them globally for now
# or pass them explicitly if we keep them local to the layout function.
# Given the complexity, let's get them by reference or make them accessible.

# Let's create references or access them via their parent container if needed
# Assuming they will be created within the layout function and need to be accessible here.
# A simpler approach for now is to access them via the page if they are defined with refs or IDs.
# Or, pass them as arguments to the handler if it's defined where they are created.
# Let's redefine on_add_captions_click_with_model to accept needed controls.

def on_add_captions_click_with_model(e: ft.ControlEvent,
                                     caption_model_dropdown: ft.Dropdown,
                                     captions_checkbox: ft.Checkbox,
                                     cap_command_textfield: ft.TextField,
                                     max_tokens_textfield: ft.TextField,
                                     dataset_add_captions_button_control: ft.ElevatedButton,
                                     dataset_delete_captions_button_control: ft.ElevatedButton,
                                     thumbnails_grid_control: ft.GridView):
    """Handle the 'Add Captions' button click, starting the captioning process."""
    # If a process is running, treat as stop
    proc = current_caption_process.get("proc")
    if proc is not None and proc.returncode is None:
        stop_captioning(
            e,
            dataset_add_captions_button_control,
            dataset_delete_captions_button_control,
            thumbnails_grid_control
        )
        return

    selected_model = caption_model_dropdown.value or "llava_next_7b"
    current_dataset_name = selected_dataset.get("value")

    if not current_dataset_name:
        if e.page:
            e.page.snack_bar = ft.SnackBar(content=ft.Text("Error: No dataset selected."), open=True)
            e.page.update()
        return

    dataset_folder_path = os.path.abspath(os.path.join(config.DATASETS_DIR, current_dataset_name))
    output_json_path = os.path.join(dataset_folder_path, "captions.json")

    # --- Build the command string using the dedicated helper function ---
    command = build_caption_command(
        dataset_folder_path=dataset_folder_path,
        output_json_path=output_json_path,
        selected_model=selected_model,
        use_8bit=(selected_model == "qwen_25_vl" and captions_checkbox.value),
        instruction=cap_command_textfield.value.strip(),
        max_new_tokens=int(max_tokens_textfield.value.strip() or 100),
    )
    # ---------------------------------------------------------------------

    # Change Delete button to Stop
    dataset_delete_captions_button_control.text = "Stop"
    dataset_delete_captions_button_control.on_click = lambda evt: stop_captioning(
        evt,
        dataset_add_captions_button_control,
        dataset_delete_captions_button_control,
        thumbnails_grid_control
    )
    dataset_delete_captions_button_control.tooltip = "Stop captioning process"
    dataset_delete_captions_button_control.disabled = False
    dataset_delete_captions_button_control.update()


    if e.page:
        e.page.update()
        # Run the command asynchronously
        e.page.run_task(
            run_dataset_script_command,
            command,
            e.page,
            dataset_add_captions_button_control,
            processed_progress_bar,
            processed_output_field,
            "Add Captions", # Original button text
            delete_button_ref=dataset_delete_captions_button_control,
            thumbnails_grid_control=thumbnails_grid_control,
            on_success_callback=lambda: update_thumbnails(page_ctx=e.page, grid_control=thumbnails_grid_control, force_refresh=True) # Force refresh after captioning
        )

def stop_captioning(e: ft.ControlEvent,
                    add_button: ft.ElevatedButton,
                    delete_button: ft.ElevatedButton,
                    thumbnails_grid_control: ft.GridView):
    """Attempt to stop the running captioning process."""
    # Try to kill by PID file (more reliable for spawned processes)
    pid_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../scripts/caption_pid.txt')
    killed = False
    if os.path.exists(pid_file):
        try:
            with open(pid_file, 'r') as f:
                pid = int(f.read().strip())
            # Try to kill the process group
            os.kill(pid, signal.SIGTERM) # Or signal.SIGKILL
            killed = True
            print(f"Sent SIGTERM to process {pid}") # Debug
        except Exception as ex:
            print(f"Error sending signal to PID {pid}: {ex}") # Debug
        # Clean up PID file regardless
        try:
            os.remove(pid_file)
            print(f"Removed PID file: {pid_file}") # Debug
        except Exception as ex:
            print(f"Error removing PID file {pid_file}: {ex}") # Debug


    # Fallback to killing the tracked process tree if PID file failed or wasn't used
    if not killed:
        proc = current_caption_process.get("proc")
        if proc is not None and proc.returncode is None:
            print("Attempting to terminate/kill process tree...") # Debug
            try:
                # Terminate process tree on Windows, terminate single process on others
                if os.name == 'nt': # Windows
                    import subprocess
                    # Use taskkill /F /T /PID to force kill process tree
                    subprocess.run(['taskkill', '/F', '/T', '/PID', str(proc.pid)])
                else: # Linux/macOS
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                killed = True # Mark as killed if command ran without error
                print("Process tree termination/kill command sent.") # Debug
            except Exception as ex:
                print(f"Error terminating/killing process tree: {ex}") # Debug
                try:
                    proc.kill() # Final fallback
                    killed = True # Mark as killed if kill works
                    print("Used proc.kill() as final fallback.") # Debug
                except Exception as ex_kill:
                    print(f"Error with proc.kill() fallback: {ex_kill}") # Debug
            finally:
                 current_caption_process["proc"] = None # Clear tracked process


    # Restore Delete button to its original state
    delete_button.text = "Delete"
    delete_button.on_click = lambda evt: on_delete_captions_click(evt, thumbnails_grid_control)
    delete_button.tooltip = "Delete the captions.json file"
    delete_button.disabled = not selected_dataset.get("value") # Disable if no dataset selected
    delete_button.update()

    # Restore Add Captions button state (if it was disabled)
    add_button.text = "Add Captions" # Restore original text
    add_button.disabled = False # Re-enable button
    add_button.update()

    processed_progress_bar.visible = False # Hide progress bar
    if processed_output_field.page:
        processed_output_field.value += "\n--- Process Stopped ---\n"
        processed_output_field.update()
        set_bottom_app_bar_height() # Adjust height after adding text

    if e.page:
        e.page.snack_bar = ft.SnackBar(content=ft.Text("Captioning process stopped."), open=True)
        e.page.update()


def on_delete_captions_click(e: ft.ControlEvent, thumbnails_grid_control: ft.GridView):
    """Handle the 'Delete Captions' button click, showing a confirmation dialog."""
    page_for_dialog = e.page
    button_control = e.control
    current_dataset_name = selected_dataset.get("value")

    if not current_dataset_name:
        if page_for_dialog:
            page_for_dialog.snack_bar = ft.SnackBar(content=ft.Text("No dataset selected."), open=True)
            page_for_dialog.update()
        return

    captions_file_path = os.path.join(config.DATASETS_DIR, current_dataset_name, "captions.json")
    if not os.path.exists(captions_file_path):
        if page_for_dialog:
            page_for_dialog.snack_bar = ft.SnackBar(content=ft.Text(f"Captions for '{current_dataset_name}' not found."), open=True)
            page_for_dialog.update()
        return

    # Disable button while dialog is open
    if button_control:
        button_control.disabled = True
    if page_for_dialog:
        page_for_dialog.update()

    try:
        # Show confirmation dialog before deleting
        show_delete_caption_dialog(
            page_for_dialog,
            current_dataset_name,
            lambda: perform_delete_captions(page_for_dialog, thumbnails_grid_control)
        )
    finally:
        # Re-enable button after dialog is closed
        if button_control:
             # Only re-enable if a dataset is still selected
            button_control.disabled = not selected_dataset.get("value")
        if page_for_dialog:
            page_for_dialog.update()


def perform_delete_captions(page_context: ft.Page, thumbnails_grid_control: ft.GridView):
    """Perform the actual deletion of the captions file after confirmation."""
    current_dataset_name = selected_dataset.get("value")
    if not current_dataset_name:
        return

    if delete_captions_file(current_dataset_name):
        if page_context:
            page_context.snack_bar = ft.SnackBar(content=ft.Text(f"Deleted captions for {current_dataset_name}."), open=True)
        update_thumbnails(page_ctx=page_context, grid_control=thumbnails_grid_control, force_refresh=True) # Force refresh after deleting captions
    else:
        if page_context:
            page_context.snack_bar = ft.SnackBar(content=ft.Text(f"Error deleting captions for {current_dataset_name}."), open=True)
            page_context.update()


def on_preprocess_dataset_click(e: ft.ControlEvent,
                                model_name_dropdown: ft.Dropdown,
                                bucket_size_textfield: ft.TextField,
                                trigger_word_textfield: ft.TextField):
    """Handle the 'Start Preprocess' button click."""
    current_dataset_name = selected_dataset.get("value")
    if not current_dataset_name:
        if e.page:
            e.page.snack_bar = ft.SnackBar(content=ft.Text("Error: No dataset selected for preprocessing."), open=True)
            e.page.update()
        return

    input_captions_json_path = os.path.abspath(os.path.join(config.DATASETS_DIR, current_dataset_name, "captions.json"))
    preprocess_output_dir = os.path.abspath(os.path.join(config.DATASETS_DIR, current_dataset_name, "preprocessed_data"))

    # Check if captions file exists
    if not os.path.exists(input_captions_json_path):
        error_msg = f"Error: Preprocessing input file not found: {input_captions_json_path}"
        if e.page:
            e.page.snack_bar = ft.SnackBar(content=ft.Text(error_msg), open=True)
            processed_output_field.value = error_msg + "\n" # Display error in output field too
            processed_output_field.visible = True
            set_bottom_app_bar_height()
            processed_output_field.update()
            e.page.update()
        return

    model_name_val = model_name_dropdown.value.strip()
    raw_bucket_str_val = bucket_size_textfield.value.strip()
    trigger_word = trigger_word_textfield.value.strip()

    if not model_name_val or not raw_bucket_str_val:
        error_msg = "Error: Model Name or Bucket Size cannot be empty."
        if e.page:
            e.page.snack_bar = ft.SnackBar(content=ft.Text(error_msg), open=True)
            processed_output_field.value = error_msg + "\n"
            processed_output_field.visible = True
            set_bottom_app_bar_height()
            processed_output_field.update()
            e.page.update()
        return

    # Validate and parse bucket size string
    parsed_bucket_list = parse_bucket_string_to_list(raw_bucket_str_val)
    if parsed_bucket_list is None:
        error_msg = f"Error parsing Bucket Size format: '{raw_bucket_str_val}'. Expected '[W, H, F]' or 'WxHxF'."
        if e.page:
             e.page.snack_bar = ft.SnackBar(content=ft.Text(error_msg), open=True)
             processed_output_field.value = error_msg + "\n"
             processed_output_field.visible = True
             set_bottom_app_bar_height()
             processed_output_field.update()
             e.page.update()
        return

    # Validate bucket values (divisible by 32, etc.)
    error_messages = validate_bucket_values(*parsed_bucket_list)
    if error_messages:
        error_msg = "Bucket Size validation errors:\n" + "\n".join(error_messages)
        if e.page:
            e.page.snack_bar = ft.SnackBar(content=ft.Text("Bucket Size validation failed. Check output."), open=True)
            processed_output_field.value = error_msg + "\n"
            processed_output_field.visible = True
            set_bottom_app_bar_height()
            processed_output_field.update()
            e.page.update()
        return

    # Format validated values for the script
    resolution_buckets_str = f"{parsed_bucket_list[0]}x{parsed_bucket_list[1]}x{parsed_bucket_list[2]}"

    # --- Build the command string using the dedicated helper function ---
    command = build_preprocess_command(
        input_captions_json_path=input_captions_json_path,
        preprocess_output_dir=preprocess_output_dir,
        resolution_buckets_str=resolution_buckets_str,
        model_name_val=model_name_val,
        trigger_word=trigger_word,
    )
    # ---------------------------------------------------------------------

    if e.page:
        # Run the command asynchronously
        e.page.run_task(
            run_dataset_script_command,
            command,
            e.page,
            e.control, # Pass the preprocess button itself
            processed_progress_bar,
            processed_output_field,
            "Start Preprocess", # Original button text
            on_success_callback=lambda: update_thumbnails(page_ctx=e.page, grid_control=thumbnails_grid_ref.current, force_refresh=True) # Force refresh after preprocessing
        )

# ======================================================================================
# GUI Update/Utility Functions (Functions that update the UI state)
# ======================================================================================

def update_thumbnails(page_ctx: ft.Page | None, grid_control: ft.GridView | None, force_refresh: bool = False):
    """Update the thumbnails grid for the selected dataset. If force_refresh is True, use cache-busting temp files."""
    if not grid_control:
        return

    current_selection = selected_dataset.get("value")
    grid_control.controls.clear()
    processed_map = load_processed_map(current_selection) if current_selection else None

    if not current_selection:
        folders_exist = get_dataset_folders() is not None and len(get_dataset_folders()) > 0
        grid_control.controls.append(ft.Text("Select a dataset to view videos." if folders_exist else "No datasets found."))
    else:
        # Fetch video and thumbnail information
        thumbnail_paths_map, video_info = get_videos_and_thumbnails(current_selection)
        video_files_list["value"] = list(thumbnail_paths_map.keys()) # Update global list
        dataset_captions = load_dataset_captions(current_selection)

        if not thumbnail_paths_map:
            grid_control.controls.append(ft.Text(f"No videos found in dataset '{current_selection}'."))
        else:
            for video_path, thumb_path in thumbnail_paths_map.items():
                video_name = os.path.basename(video_path)
                info = video_info.get(video_name, {})
                width, height, frames = info.get("width", "?"), info.get("height", "?"), info.get("frames", "?")
                has_caption = any(entry.get("media_path") == video_name and entry.get("caption", "").strip() for entry in dataset_captions)
                cap_val, cap_color = ("yes", ft.Colors.GREEN) if has_caption else ("no", ft.Colors.RED)
                proc_val, proc_color = ("yes", ft.Colors.GREEN) if processed_map and video_name in processed_map else ("no", ft.Colors.RED)

                # Use a temporary file path for the image source if force_refresh is true and thumbnail exists
                image_src = thumb_path
                temp_thumb_path = None
                if force_refresh and thumb_path and os.path.exists(thumb_path):
                     try:
                        temp_thumb_name = f"{os.path.splitext(os.path.basename(thumb_path))[0]}.tmp_{int(__import__('time').time())}{os.path.splitext(thumb_path)[1]}"
                        temp_thumb_path = os.path.join(os.path.dirname(thumb_path), temp_thumb_name)
                        __import__('shutil').copy2(thumb_path, temp_thumb_path)
                        image_src = temp_thumb_path
                     except Exception as e:
                         print(f"Error creating temp thumbnail for {thumb_path}: {e}") # Debug
                         image_src = thumb_path # Fallback if temp creation fails

                grid_control.controls.append(
                    ft.Container(
                        content=ft.Column([
                            ft.Image(
                                src=image_src, # Use temp path if created, otherwise original
                                width=config.THUMB_TARGET_W,
                                height=config.THUMB_TARGET_H,
                                fit=ft.ImageFit.COVER,
                                border_radius=ft.border_radius.all(5)
                            ),
                            ft.Text(spans=[
                                ft.TextSpan("[cap - ", style=ft.TextStyle(color=ft.Colors.GREY_500, size=10)),
                                ft.TextSpan(cap_val, style=ft.TextStyle(color=cap_color, size=10)),
                                ft.TextSpan(", proc - ", style=ft.TextStyle(color=ft.Colors.GREY_500, size=10)),
                                ft.TextSpan(proc_val, style=ft.TextStyle(color=proc_color, size=10)),
                                ft.TextSpan("]", style=ft.TextStyle(color=ft.Colors.GREY_500, size=10)),
                            ], size=10),
                            ft.Text(f"[{width}x{height} - {frames} frames]", size=10, color=ft.Colors.GREY_500),
                        ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, tight=True),
                        data=video_path, # Store original video path in data
                        on_click=lambda e_click, vp=video_path, current_grid=grid_control, page_ctx=page_ctx: (
                             open_video_captions_dialog( # Pass page_ctx explicitly
                                 page_ctx,
                                 vp,
                                 video_files_list["value"],
                                 on_caption_updated_callback=lambda: update_thumbnails(page_ctx=page_ctx, grid_control=current_grid, force_refresh=True) # Force refresh on caption update
                             ) if page_ctx else None
                         ),
                        tooltip=video_name,
                        width=config.THUMB_TARGET_W + 10,
                        height=config.THUMB_TARGET_H + 45, # Adjust height to fit text
                        padding=5,
                        border=ft.border.all(1, ft.Colors.OUTLINE),
                        border_radius=ft.border_radius.all(5)
                    )
                )

    # Update the grid view itself
    if grid_control and grid_control.page:
        grid_control.update()

    # Clean up temporary thumbnail files (basic approach - might need more robust handling)
    if force_refresh and current_selection:
        dataset_folder_path = os.path.abspath(os.path.join(config.DATASETS_DIR, current_selection))
        for file_name in os.listdir(dataset_folder_path):
            if file_name.endswith('.jpg') and '.tmp_' in file_name:
                try:
                    os.remove(os.path.join(dataset_folder_path, file_name))
                    #print(f"Cleaned up temp thumbnail: {file_name}") # Debug
                except Exception as e:
                    print(f"Error cleaning up temp thumbnail {file_name}: {e}") # Debug


def update_dataset_dropdown(
    p_page: ft.Page | None,
    current_dataset_dropdown: ft.Dropdown,
    current_thumbnails_grid: ft.GridView,
    delete_button: ft.ElevatedButton # Pass delete button
):
    """Update the dataset dropdown and thumbnails grid when dataset list might have changed."""
    folders = get_dataset_folders()
    folder_names = list(folders.keys()) if folders else []

    current_dataset_dropdown.options = [ft.dropdown.Option(name) for name in folder_names]
    current_dataset_dropdown.value = None # Clear selection
    selected_dataset["value"] = None # Clear global state

    # Load and set default config values
    bucket_val, model_val, trigger_word_val = load_dataset_config(None) # Load defaults
    if bucket_size_textfield: bucket_size_textfield.value = bucket_val
    if model_name_dropdown: model_name_dropdown.value = model_val
    if trigger_word_textfield: trigger_word_textfield.value = trigger_word_val or ''

    # Update thumbnails (will show "Select a dataset...")
    update_thumbnails(page_ctx=p_page, grid_control=current_thumbnails_grid)

     # Update delete captions button state (disable)
    if delete_button:
        pass # Keep button always enabled


    if current_dataset_dropdown.page:
        current_dataset_dropdown.update()

    if p_page: # Use page ref for snackbar
        p_page.snack_bar = ft.SnackBar(ft.Text("Dataset list updated! Select a dataset."))
        p_page.snack_bar.open = True


def reload_current_dataset(
    p_page: ft.Page | None,
    current_dataset_dropdown: ft.Dropdown,
    current_thumbnails_grid: ft.GridView,
    add_button: ft.ElevatedButton,
    delete_button: ft.ElevatedButton
):
    """Reload the currently selected dataset, preserving selection if possible."""
    # Hide output and progress bar on reload
    if processed_output_field.page:
        processed_output_field.visible = False
        set_bottom_app_bar_height() # This function updates
    if processed_progress_bar.page:
        processed_progress_bar.visible = False

    folders = get_dataset_folders()
    folder_names = list(folders.keys()) if folders else []

    current_dataset_dropdown.options = [ft.dropdown.Option(name) for name in folder_names]
    current_dataset_dropdown.disabled = len(folder_names) == 0

    prev_selected_name = selected_dataset.get("value")

    if prev_selected_name and prev_selected_name in folder_names:
        # If previously selected dataset still exists, keep it selected
        current_dataset_dropdown.value = prev_selected_name
        selected_dataset["value"] = prev_selected_name
        bucket_val, model_val, trigger_word_val = load_dataset_config(prev_selected_name)
        if bucket_size_textfield: bucket_size_textfield.value = bucket_val
        # Ensure loaded model value is a valid choice, otherwise use default
        if model_name_dropdown: model_name_dropdown.value = model_val if model_val in config.ltx_models else config.ltx_def_model
        if trigger_word_textfield: trigger_word_textfield.value = trigger_word_val or ''
        update_thumbnails(page_ctx=p_page, grid_control=current_thumbnails_grid) # Update thumbnails for the current dataset
        snack_bar_text = f"Dataset '{prev_selected_name}' reloaded."
    else:
        # If previous dataset is gone or none was selected, clear selection and show default
        current_dataset_dropdown.value = None
        selected_dataset["value"] = None
        bucket_val, model_val, trigger_word_val = load_dataset_config(None) # Load defaults
        if bucket_size_textfield: bucket_size_textfield.value = bucket_val
        if model_name_dropdown: model_name_dropdown.value = model_val
        if trigger_word_textfield: trigger_word_textfield.value = trigger_word_val or ''
        update_thumbnails(page_ctx=p_page, grid_control=current_thumbnails_grid) # Update thumbnails (will show "Select a dataset...")
        snack_bar_text = "Dataset list reloaded. Select a dataset."

    # Update delete captions button state
    if delete_button:
        pass # Keep button always enabled

    # Update add captions button state (always enabled if a dataset is selected)
    if add_button:
         pass # Keep button always enabled


    if p_page: # Use page ref for snackbar
        p_page.snack_bar = ft.SnackBar(ft.Text(snack_bar_text))
        p_page.snack_bar.open = True
        p_page.update() # Keep page update for snackbar and potentially other layout changes


# ======================================================================================
# GUI Control Creation Functions (Build individual controls or groups)
# ======================================================================================

EXPANSION_TILE_HEADER_BG_COLOR = ft.CupertinoColors.with_opacity(0.08, ft.CupertinoColors.ACTIVE_BLUE) # Define a variable for header background color
EXPANSION_TILE_INSIDE_BG_COLOR = ft.Colors.TRANSPARENT

def _create_global_controls():
    """Creates and initializes the global UI controls."""
    global bucket_size_textfield, rename_textfield, model_name_dropdown, trigger_word_textfield
    # Note: processed_progress_bar and processed_output_field are already global and initialized

    bucket_size_textfield = create_textfield(
        label="Bucket Size (e.g., [W, H, F] or WxHxF)",
        value=config.DEFAULT_BUCKET_SIZE_STR,
        expand=True
    )

    rename_textfield = create_textfield(
        label="Rename all files",
        value="",
        hint_text="Name of videos + _num will be added",
        expand=True,
    )

    model_name_dropdown = create_dropdown(
        "Model Name",
        config.ltx_def_model, # Use config directly
        {name: name for name in config.ltx_models}, # Use config directly
        "Select model",
        expand=True
    )

    trigger_word_textfield = create_textfield(
        "Trigger WORD", "", col=9, expand=True, hint_text="e.g. 'CAKEIFY' , leave empty for none"
    )

    # Assign change handlers to the global config controls
    bucket_size_textfield.on_change = on_bucket_or_model_change
    model_name_dropdown.on_change = on_bucket_or_model_change
    trigger_word_textfield.on_change = on_bucket_or_model_change


def _build_dataset_selection_section(dataset_dropdown_control: ft.Dropdown, update_button_control: ft.IconButton):
    """Builds the section for selecting and updating the dataset."""
    return ft.Row([
        ft.Container(content=dataset_dropdown_control, expand=True, width=160),
        ft.Container(content=update_button_control, alignment=ft.alignment.center_right, width=40)
    ])


def _build_captioning_section(
    caption_model_dropdown: ft.Dropdown,
    captions_checkbox_container: ft.Container, # Container for 8-bit checkbox
    cap_command_textfield: ft.TextField,
    max_tokens_textfield: ft.TextField,
    dataset_add_captions_button_control: ft.ElevatedButton,
    dataset_delete_captions_button_control: ft.ElevatedButton,
    thumbnails_grid_control: ft.GridView # Needed for event handlers
):
    """Builds the captioning controls section as an ExpansionTile."""
    # Assign on_click handlers here, as they need access to local controls
    dataset_add_captions_button_control.on_click = lambda e: on_add_captions_click_with_model(
        e,
        caption_model_dropdown,
        captions_checkbox_container.content, # Pass the actual checkbox control
        cap_command_textfield,
        max_tokens_textfield,
        dataset_add_captions_button_control,
        dataset_delete_captions_button_control,
        thumbnails_grid_control
    )
    dataset_delete_captions_button_control.on_click = lambda e: on_delete_captions_click(e, thumbnails_grid_control)

    return ft.ExpansionTile(
        title=ft.Text("1. Captions", size=12),
        bgcolor=EXPANSION_TILE_INSIDE_BG_COLOR, # Apply color
        collapsed_bgcolor=EXPANSION_TILE_HEADER_BG_COLOR, # Apply color
        controls=[ # Wrap controls in a Column for spacing
            ft.Column([
                ft.ResponsiveRow([captions_checkbox_container, caption_model_dropdown]),
                ft.ResponsiveRow([max_tokens_textfield, cap_command_textfield]),
                ft.Row([
                    ft.Container(content=dataset_add_captions_button_control, expand=True),
                    ft.Container(content=dataset_delete_captions_button_control, expand=True)
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                 ft.Container(height=10), # Spacer
            ], spacing=10)
        ],
        initially_expanded=True, # Set to True
    )


def _build_preprocessing_section(
    model_name_dropdown: ft.Dropdown,
    bucket_size_textfield: ft.TextField,
    trigger_word_textfield: ft.TextField,
    dataset_preprocess_button_control: ft.ElevatedButton,
    thumbnails_grid_control: ft.GridView # Needed for event handler callback
):
    """Builds the preprocessing controls section as an ExpansionTile."""
    # Assign event handler for preprocess button here, as it needs access to local controls
    dataset_preprocess_button_control.on_click = lambda e: on_preprocess_dataset_click(
        e,
        model_name_dropdown,
        bucket_size_textfield,
        trigger_word_textfield
    )
    return ft.ExpansionTile(
        title=ft.Text("2. Preprocess Dataset", size=12),
        bgcolor=EXPANSION_TILE_INSIDE_BG_COLOR, # Apply color
        collapsed_bgcolor=EXPANSION_TILE_HEADER_BG_COLOR, # Apply color
        controls=[ # Wrap controls in a Column for spacing
            ft.Column([
                model_name_dropdown,
                bucket_size_textfield,
                trigger_word_textfield,
                dataset_preprocess_button_control,
                ft.Container(height=10), # Spacer
            ], spacing=10)
        ],
        initially_expanded=False, # Set to False
    )


def _build_latent_test_section():
    """Builds the optional latent test section (currently a placeholder) as an ExpansionTile."""
    return ft.ExpansionTile(
        title=ft.Text("3. Test latent", size=12),
        bgcolor=EXPANSION_TILE_INSIDE_BG_COLOR, # Apply color
        collapsed_bgcolor=EXPANSION_TILE_HEADER_BG_COLOR, # Apply color
        controls=[ # Wrap controls in a Column for spacing
            ft.Column([
                ft.Text("Test here", size=12),
                 ft.Container(height=10), # Spacer
            ], spacing=10)
        ],
        initially_expanded=False, # Set to False
    )


def _build_rename_section(rename_textfield: ft.TextField, thumbnails_grid_control: ft.GridView):
    """Builds the file renaming section as an ExpansionTile."""
    # Assign event handler for rename button
    rename_button = create_styled_button(
        "Rename files",
        # Pass thumbnails_grid_control to on_rename_files_click
        on_click=lambda e: on_rename_files_click(e), # on_rename_files_click uses thumbnails_grid_ref.current
        tooltip="Rename files",
        expand=True,
        button_style=BTN_STYLE2
    )
    return ft.ExpansionTile(
        title=ft.Text("Rename files", size=12),
        bgcolor=EXPANSION_TILE_INSIDE_BG_COLOR, # Apply color
        collapsed_bgcolor=EXPANSION_TILE_HEADER_BG_COLOR, # Apply color
        controls=[ # Wrap controls in a Column for spacing
            ft.Column([
                rename_textfield,
                rename_button
            ], spacing=10)
        ],
        initially_expanded=False, # Set to False
    )

def _build_bottom_status_bar():
    """Builds the bottom app bar for status updates."""
    global bottom_app_bar_ref # Need to assign to the global ref
    bottom_app_bar = ft.BottomAppBar(
        bgcolor=ft.Colors.BLUE_GREY_900,
        height=0, # Start hidden
        content=ft.Row([
            ft.Container(
                content=ft.Column([
                    processed_progress_bar, # Global progress bar
                    processed_output_field, # Global output field
                ], expand=True),
                expand=True,
            ),
        ], expand=True),
    )
    bottom_app_bar_ref = bottom_app_bar # Assign to global ref
    return bottom_app_bar


# ======================================================================================
# Main GUI Layout Builder (Assembles the sections)
# ======================================================================================

def dataset_tab_layout(page=None):
    """Build the main dataset tab layout."""
    global bottom_app_bar_ref # Need to assign to the global ref
    p_page = page # Alias for clarity

    # Create global controls if not already created (should happen once per app lifecycle)
    if bucket_size_textfield is None: # Check one of the global controls
        _create_global_controls()

    # Create controls specific to this layout function scope
    folders = get_dataset_folders()
    folder_names = list(folders.keys()) if folders else []
    dataset_dropdown_control = create_dropdown(
        "Select dataset",
        selected_dataset["value"],
        {name: name for name in folder_names}, # Options from folder names
        "Select your dataset",
        expand=True,
    )
    # Assign the created control to the global ref AFTER creation
    dataset_dropdown_control_ref.current = dataset_dropdown_control

    dataset_dropdown_control.disabled = len(folder_names) == 0
     # Assign dataset dropdown change handler
    dataset_dropdown_control.on_change = lambda ev: on_dataset_dropdown_change(
        ev,
        thumbnails_grid_ref.current,
        dataset_delete_captions_button_ref.current # Pass delete button ref's current value
    )


    update_button_control = ft.IconButton(
        icon=ft.Icons.REFRESH, tooltip="Update dataset list",
        on_click=lambda e: reload_current_dataset( # Call reload_current_dataset directly
            e.page,
            dataset_dropdown_control_ref.current,
            thumbnails_grid_ref.current,
            dataset_add_captions_button_ref.current,
            dataset_delete_captions_button_ref.current
        ),
        style=ft.ButtonStyle(padding=ft.padding.symmetric(horizontal=8)), icon_size=20
    )

    # Thumbnail Grid (global ref)
    thumbnails_grid_control = ft.GridView(
        ref=thumbnails_grid_ref, # Assign the global ref
        runs_count=5, max_extent=config.THUMB_TARGET_W + 20,
        child_aspect_ratio=(config.THUMB_TARGET_W + 10) / (config.THUMB_TARGET_H + 80), # Adjusted aspect ratio
        spacing=7, run_spacing=7, controls=[], expand=True
    )


    # Captioning specific controls
    caption_model_dropdown = create_dropdown(
        "Captioning Model", 
        config.captions_def_model,  # default
        config.captions, # Use config.captions dictionary directly
        "Select a captioning model",
        expand=True,col=9,
    )
    # Assign to ref
    caption_model_dropdown_ref.current = caption_model_dropdown

    captions_checkbox = ft.Checkbox(
            label="8-bit", value=True,scale=1,
            visible=True,
            left="left",
            expand=True,
    )
    # Assign to ref
    captions_checkbox_ref.current = captions_checkbox

    captions_checkbox_container = ft.Container(captions_checkbox,
            expand=True,col=3,scale=0.8,
            alignment=ft.alignment.bottom_center,    # Align to bottom
            margin=ft.margin.only(top=10)
    )

    # Text fields for caption command and max tokens (ensure correct variable names used)
    cap_command_textfield = create_textfield("Command", "Shortly describe the content of this video in one or two sentences.", # Renamed
                                    expand=True,
                                    hint_text="command for captioning",col=8,
    )
    # Assign to ref
    cap_command_textfield_ref.current = cap_command_textfield

    max_tokens_textfield = create_textfield("Max Tokens", "100",
                                    expand=True,
                                    hint_text="max tokens",col=4,
    )
    # Assign to ref
    max_tokens_textfield_ref.current = max_tokens_textfield


    # Captioning action buttons
    # Need to define delete button before add button for the stop logic reference
    dataset_delete_captions_button_control = create_styled_button(
        "Delete",
        ref=dataset_delete_captions_button_ref, # Assign ref - Valid for ft.ElevatedButton (assuming create_styled_button returns one and passes kwargs)
        # on_click handler assigned in _build_captioning_section
        tooltip="Delete the captions.json file",
        expand=True, # Make it expandable for layout
        button_style=BTN_STYLE2,
    )
    # Assign to ref
    dataset_delete_captions_button_ref.current = dataset_delete_captions_button_control


    dataset_add_captions_button_control = create_styled_button(
        "Add Captions",
        ref=dataset_add_captions_button_ref, # Assign ref - Valid for ft.ElevatedButton (assuming create_styled_button returns one and passes kwargs)
        # on_click handler assigned in _build_captioning_section
        button_style=BTN_STYLE2,
        expand=True
    )
    # Assign to ref
    dataset_add_captions_button_ref.current = dataset_add_captions_button_control


    # Preprocessing specific button (uses global controls bucket_size_textfield, model_name_dropdown, trigger_word_textfield)
    dataset_preprocess_button_control = create_styled_button(
        "Start Preprocess ",
        ref=dataset_preprocess_button_ref, # Assign ref - Valid for ft.ElevatedButton
        # on_click handler assigned in _build_preprocessing_section
        tooltip="Preprocess dataset using captions.json",
        expand=True,
        button_style=BTN_STYLE2
    )
    # Assign to ref
    dataset_preprocess_button_ref.current = dataset_preprocess_button_control


    # Assemble sections into the left column
    lc_content = ft.Column([
        _build_dataset_selection_section(dataset_dropdown_control_ref.current, update_button_control),
        _build_captioning_section(
            caption_model_dropdown_ref.current,
            captions_checkbox_container,
            cap_command_textfield_ref.current,
            max_tokens_textfield_ref.current,
            dataset_add_captions_button_ref.current,
            dataset_delete_captions_button_ref.current,
            thumbnails_grid_ref.current
        ),
        _build_preprocessing_section(
            model_name_dropdown, # Global control
            bucket_size_textfield, # Global control
            trigger_word_textfield, # Global control
            dataset_preprocess_button_ref.current, # Use ref
            thumbnails_grid_ref.current # Use ref
        ),
        _build_latent_test_section(), # Placeholder
        _build_rename_section(rename_textfield, thumbnails_grid_ref.current), # Global control, pass grid ref
    ], spacing=0, width=200, alignment=ft.MainAxisAlignment.START) # Set spacing to 0 for ExpansionTiles

    # Build the bottom status bar
    bottom_app_bar = _build_bottom_status_bar() # Assigns to global bottom_app_bar_ref

    # Assemble the right column
    rc_content = ft.Column([
        thumbnails_grid_ref.current, # Global ref grid
        bottom_app_bar, # Bottom app bar
    ], alignment=ft.CrossAxisAlignment.STRETCH, expand=True, spacing=10)

    # Containers for layout structure
    lc = ft.Container(
        content=lc_content,
        padding=ft.padding.only(top=0, right=0, left=5),
    )
    rc = ft.Container(
        content=rc_content,
        padding=ft.padding.only(top=5, left=0, right=0),
        expand=True
    )

    reload_current_dataset(
        p_page,
        dataset_dropdown_control_ref.current,
        thumbnails_grid_ref.current,
        dataset_add_captions_button_ref.current,
        dataset_delete_captions_button_ref.current
    )


    # Main layout row
    main_container = ft.Row(
        controls=[
            lc,
            ft.VerticalDivider(color=ft.Colors.GREY_500, width=1),
            rc,
        ],
        alignment=ft.MainAxisAlignment.START,
        vertical_alignment=ft.CrossAxisAlignment.START,
        expand=True
    )

    return main_container # Return the main layout container


# Note: Global on_change handlers for bucket/model/trigger word are assigned in _create_global_controls
