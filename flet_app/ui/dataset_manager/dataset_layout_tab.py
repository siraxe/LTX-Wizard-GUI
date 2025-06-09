import flet as ft
import os
import json
import asyncio
import signal # Added signal import
import subprocess
import shutil
import time
from settings import settings # Import only the config class

from ui_popups.delete_caption_dialog import show_delete_caption_dialog
from ui._styles import create_dropdown, create_styled_button, create_textfield, BTN_STYLE2
from ui.dataset_manager.dataset_utils import (
    get_dataset_folders,
    get_videos_and_thumbnails,
    apply_affix_from_textfield, find_and_replace_in_captions # New functions to be created
)
from ui.dataset_manager.dataset_thumb_layout import create_thumbnail_container, set_thumbnail_selection_state # Import the new function
from ui_popups.image_player_dialog import open_image_captions_dialog # Keep this import
from ui_popups.video_player_dialog import open_video_captions_dialog # Keep this import
from ui.flet_hotkeys import is_d_key_pressed_global # Import global D key state

# ======================================================================================
# Global State (Keep track of UI controls and running processes)
# ======================================================================================

# References to UI controls
selected_dataset = {"value": None}
DATASETS_TYPE = {"value": None} # "image" or "video"
video_files_list = {"value": []}
thumbnails_grid_ref = ft.Ref[ft.GridView]()
processed_progress_bar = ft.ProgressBar(visible=False)
processed_output_field = ft.TextField(
    label="Processed Output", text_size=10, multiline=True, read_only=True,
    visible=False, min_lines=6, max_lines=15, expand=True)
bottom_app_bar_ref = None

# Multi-selection state
selected_thumbnails_set = set() # Stores video_path of selected thumbnails
last_clicked_thumbnail_index = -1 # Stores the index of the last clicked checkbox
# is_d_key_pressed = False # Tracks if 'D' hotkey is pressed - REMOVED, now global in flet_hotkeys.py

# Global controls (defined here but created in _create_global_controls)
bucket_size_textfield: ft.TextField = None
rename_textfield: ft.TextField = None
model_name_dropdown: ft.Dropdown = None
trigger_word_textfield: ft.TextField = None

# References to controls created in dataset_tab_layout that need external access
dataset_dropdown_control_ref = ft.Ref[ft.Dropdown]()
dataset_add_captions_button_ref = ft.Ref[ft.ElevatedButton]()
dataset_delete_captions_button_ref = ft.Ref[ft.ElevatedButton]()
dataset_preprocess_button_ref = ft.Ref[ft.ElevatedButton]()
caption_model_dropdown_ref = ft.Ref[ft.Dropdown]()
captions_checkbox_ref = ft.Ref[ft.Checkbox]() # This one is a direct ft.Checkbox, so ref is valid
cap_command_textfield_ref = ft.Ref[ft.TextField]()
max_tokens_textfield_ref = ft.Ref[ft.TextField]()
change_fps_textfield_ref = ft.Ref[ft.TextField]() # Ref for the Change FPS textfield

# Process tracking for stopping script execution
current_caption_process = {"proc": None}

# Constant for ExpansionTile border radius
EXPANSION_TILE_BORDER_RADIUS = 10

# ======================================================================================
# Data & Utility Functions (File I/O, data parsing, validation)
# ======================================================================================

def parse_bucket_string_to_list(raw_bucket_str: str) -> list[int] | None:
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
    if isinstance(bucket_list, list) and len(bucket_list) == 3 and all(isinstance(i, (int, float)) for i in bucket_list):
        return f"[{bucket_list[0]}, {bucket_list[1]}, {bucket_list[2]}]"
    return settings.DEFAULT_BUCKET_SIZE_STR

def load_dataset_config(dataset_name: str | None) -> tuple[str, str, str]:
    bucket_to_set = settings.DEFAULT_BUCKET_SIZE_STR
    model_to_set = settings.ltx_def_model
    trigger_word_to_set = ''
    if dataset_name:
        base_dir, _ = _get_dataset_base_dir(dataset_name) # Unpack the tuple
        dataset_info_json_path = os.path.join(base_dir, dataset_name.replace('(img) ', '').replace(' (img)', ''), "info.json")
        if os.path.exists(dataset_info_json_path):
            try:
                with open(dataset_info_json_path, 'r') as f:
                    loaded_data = json.load(f)
                    if isinstance(loaded_data, dict):
                        # Read from top level of loaded_data
                        bucket_res_val = loaded_data.get("bucket_resolution") # Corresponds to 'size_val'
                        model_name_val = loaded_data.get("model_name")       # Corresponds to 'type_val'
                        trigger_word_val = loaded_data.get("trigger_word")

                        # bucket_resolution can be a list or a string representation of a list
                        if isinstance(bucket_res_val, list):
                            bucket_to_set = format_bucket_list_to_string(bucket_res_val)
                        elif isinstance(bucket_res_val, str):
                            # Attempt to parse if it's a string like "[512, 512, 49]"
                            parsed_list = parse_bucket_string_to_list(bucket_res_val)
                            if parsed_list:
                                bucket_to_set = format_bucket_list_to_string(parsed_list)
                            else: # If string is not parsable, keep it as is or use default
                                bucket_to_set = bucket_res_val # Or settings.DEFAULT_BUCKET_SIZE_STR if strict parsing needed
                        
                        if isinstance(model_name_val, str):
                            model_to_set = model_name_val
                        if isinstance(trigger_word_val, str):
                            trigger_word_to_set = trigger_word_val
            except (json.JSONDecodeError, IOError):
                pass
    return bucket_to_set, model_to_set, trigger_word_to_set

def save_dataset_config(dataset_name: str, bucket_str: str, model_name: str, trigger_word: str) -> bool:
    base_dir = _get_dataset_base_dir(dataset_name)
    dataset_info_json_path = os.path.join(base_dir, dataset_name.replace('(img) ', '').replace(' (img)', ''), "info.json")
    
    bucket_str_val = bucket_str # Initialize with the input value

    parsed_bucket_list = parse_bucket_string_to_list(bucket_str)
    if parsed_bucket_list is None:
         if e.page:
            e.page.snack_bar = ft.SnackBar(
                content=ft.Text(f"Invalid Bucket Size format: '{bucket_str}'. Using default."),
                open=True
            )
            e.page.update()
            # Continue with default or previous valid value for saving config
            bucket_str_val = settings.DEFAULT_BUCKET_SIZE_STR # Or load previous valid? Using default for simplicity.

    dataset_config = {
        "bucket_resolution": bucket_str_val,
        "model_name": model_name,
        "trigger_word": trigger_word
    }

    try:
        os.makedirs(os.path.dirname(dataset_info_json_path), exist_ok=True)
        with open(dataset_info_json_path, "w") as f:
            json.dump(dataset_config, f, indent=4)
        return True
    except Exception as e:
        print(f"Error saving dataset config: {e}")
    return False

def load_processed_map(dataset_name: str) -> dict | None:
    base_dir, _ = _get_dataset_base_dir(dataset_name) # Unpack the tuple
    processed_json_path = os.path.join(base_dir, dataset_name.replace('(img) ', '').replace(' (img)', ''), "preprocessed_data", "processed.json")
    if os.path.exists(processed_json_path):
        try:
            with open(processed_json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def load_dataset_captions(dataset_name: str) -> list:
    base_dir, _ = _get_dataset_base_dir(dataset_name) # Unpack the tuple
    dataset_captions_json_path = os.path.join(base_dir, dataset_name.replace('(img) ', '').replace(' (img)', ''), "captions.json")
    if os.path.exists(dataset_captions_json_path):
        try:
            with open(dataset_captions_json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return []
    return []

def delete_captions_file(dataset_name: str) -> bool:
    base_dir, _ = _get_dataset_base_dir(dataset_name) # Unpack the tuple
    captions_file_path = os.path.join(base_dir, dataset_name.replace('(img) ', '').replace(' (img)', ''), "captions.json")
    if os.path.exists(captions_file_path):
        try:
            os.remove(captions_file_path)
            return True
        except Exception:
            return False
    return False

def validate_bucket_values(W_val, H_val, F_val) -> list[str]:
    errors = []
    if W_val is None or not isinstance(W_val, int) or W_val <= 0 or W_val % 32 != 0:
        errors.append(f"Width ({W_val}) must be a positive integer divisible by 32.")
    if H_val is None or not isinstance(H_val, int) or H_val <= 0 or H_val % 32 != 0:
        errors.append(f"Height ({H_val}) must be a positive integer divisible by 32.")
    # Adjusted validation for Frames based on typical dataset preprocessing requirements
    if F_val is None or not isinstance(F_val, int) or F_val <= 0:
         errors.append(f"Frames ({F_val}) must be a positive integer.")
    # Special case: Allow 1 frame for images, otherwise validate as video (4n+1 and >=5)
    if F_val is not None and F_val != 1:
        if F_val < 5 or (F_val - 1) % 4 != 0:
            errors.append(f"Frames ({F_val}) invalid (must be 1 for images or ≥5 and 4n+1 for videos).")
    return errors

def _get_dataset_base_dir(dataset_name: str) -> tuple[str, str]:
    clean_name = dataset_name.replace('(img) ', '').replace(' (img)', '')
    # Check if it's explicitly marked as image or if the folder exists in the image datasets directory
    if dataset_name.startswith('(img) ') or dataset_name.endswith(' (img)') or \
       os.path.exists(os.path.join(settings.DATASETS_IMG_DIR, clean_name)):
        return settings.DATASETS_IMG_DIR, "image"
    return settings.DATASETS_DIR, "video"

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

def on_change_fps_click(e: ft.ControlEvent):
    current_dataset_name = selected_dataset.get("value")
    if not current_dataset_name:
        if e.page:
            e.page.snack_bar = ft.SnackBar(content=ft.Text("Error: No dataset selected."), open=True)
            e.page.update()
        return

    if not change_fps_textfield_ref.current or not change_fps_textfield_ref.current.value:
        if e.page:
            e.page.snack_bar = ft.SnackBar(content=ft.Text("Error: FPS textfield not available or empty."), open=True)
            e.page.update()
        return
        
    target_fps_str = change_fps_textfield_ref.current.value.strip()
    try:
        target_fps_float = float(target_fps_str)
        if target_fps_float <= 0:
            raise ValueError("FPS must be positive")
    except ValueError:
        if e.page:
            e.page.snack_bar = ft.SnackBar(content=ft.Text(f"Error: Invalid FPS value '{target_fps_str}'. Must be a positive number."), open=True)
            e.page.update()
        return

    base_dir, _ = _get_dataset_base_dir(current_dataset_name)
    dataset_folder_path = os.path.abspath(os.path.join(base_dir, current_dataset_name.replace('(img) ', '').replace(' (img)', '')))
    if not os.path.isdir(dataset_folder_path):
        if e.page:
            e.page.snack_bar = ft.SnackBar(content=ft.Text(f"Error: Dataset folder not found: {dataset_folder_path}"), open=True)
            e.page.update()
        return

    ffmpeg_exe = settings.FFMPEG_PATH
    ffprobe_exe = ""
    if os.path.isabs(ffmpeg_exe) or os.path.sep in ffmpeg_exe:
        ffmpeg_dir = os.path.dirname(ffmpeg_exe)
        ffprobe_basename = "ffprobe.exe" if ffmpeg_exe.lower().endswith(".exe") else "ffprobe"
        ffprobe_exe = os.path.join(ffmpeg_dir, ffprobe_basename)
    else: # Assumed to be a command in PATH
        ffprobe_exe = "ffprobe.exe" if ffmpeg_exe.lower().endswith(".exe") else "ffprobe"

    video_exts = ['.mp4', '.mov', '.avi', '.webm', '.mkv', '.flv', '.gif']
    processed_files = 0
    failed_files = 0
    skipped_files = 0

    # Show initial processing message
    if e.page:
        e.page.snack_bar = ft.SnackBar(content=ft.Text(f"Processing videos in {current_dataset_name} to {target_fps_float} FPS..."), open=True)
        e.page.update()

    video_files_in_dataset = [f for f in os.listdir(dataset_folder_path) if os.path.splitext(f)[1].lower() in video_exts]

    if not video_files_in_dataset:
        if e.page:
            e.page.snack_bar = ft.SnackBar(content=ft.Text("No video files found in the dataset."), open=True)
            e.page.update()
        return

    for video_file_name in video_files_in_dataset:
        input_video_path = os.path.join(dataset_folder_path, video_file_name)
        base, ext = os.path.splitext(video_file_name)
        temp_output_video_path = os.path.join(dataset_folder_path, f"{base}_tempfps{ext}")
        original_fps = None

        try:
            # Get original FPS
            ffprobe_cmd = [
                ffprobe_exe, "-v", "error", "-select_streams", "v:0", 
                "-show_entries", "stream=r_frame_rate", "-of", 
                "default=noprint_wrappers=1:nokey=1", input_video_path
            ]
            result = subprocess.run(ffprobe_cmd, capture_output=True, text=True, check=False)
            if result.returncode != 0:
                print(f"ffprobe error for {video_file_name}: {result.stderr}")
                failed_files += 1
                continue
            
            raw_r_frame_rate = result.stdout.strip()
            if not raw_r_frame_rate:
                print(f"Could not determine r_frame_rate for {video_file_name}.")
                failed_files +=1
                continue
            
            if '/' in raw_r_frame_rate:
                num, den = map(float, raw_r_frame_rate.split('/'))
                if den == 0: # Avoid division by zero
                    print(f"Invalid r_frame_rate denominator for {video_file_name}: {raw_r_frame_rate}")
                    failed_files += 1
                    continue
                original_fps = num / den
            else:
                original_fps = float(raw_r_frame_rate)

            if abs(original_fps - target_fps_float) < 0.01: # Comparing floats
                print(f"Skipping {video_file_name}, already at target FPS ({original_fps:.2f}).")
                skipped_files += 1
                continue

            # Change FPS using ffmpeg
            ffmpeg_cmd = [
                ffmpeg_exe, "-y", "-i", input_video_path, 
                "-r", str(target_fps_float), 
                temp_output_video_path
            ]
            print(f"Running: {' '.join(ffmpeg_cmd)}")
            result_ffmpeg = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=False)

            if result_ffmpeg.returncode == 0:
                shutil.move(temp_output_video_path, input_video_path)
                print(f"Successfully changed FPS for {video_file_name}")
                processed_files += 1
            else:
                print(f"ffmpeg error for {video_file_name}: {result_ffmpeg.stderr}")
                failed_files += 1
                if os.path.exists(temp_output_video_path):
                    os.remove(temp_output_video_path)
        except Exception as ex:
            print(f"Error processing {video_file_name}: {ex}")
            failed_files += 1
            if os.path.exists(temp_output_video_path):
                try:
                    os.remove(temp_output_video_path)
                except OSError as ose:
                    print(f"Could not remove temp file {temp_output_video_path}: {ose}")
    
    summary_message = f"FPS change complete. Processed: {processed_files}, Skipped: {skipped_files}, Failed: {failed_files}."
    if e.page:
        e.page.snack_bar = ft.SnackBar(content=ft.Text(summary_message), open=True)
        e.page.update()

    if processed_files > 0 and thumbnails_grid_ref.current and e.page:
        update_thumbnails(e.page, thumbnails_grid_ref.current, force_refresh=True)

def on_rename_files_click(e: ft.ControlEvent):
    print("\n=== RENAME FUNCTION CALLED (dataset_layout_tab.py) ===")
    current_dataset_name = selected_dataset.get("value")
    print(f"[DEBUG] Current dataset name: {current_dataset_name}")
    
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

    # Clean the dataset name by removing any image markers (both old and new formats)
    clean_current_name = current_dataset_name.replace('(img) ', '').replace(' (img)', '')
    clean_base_name = base_name.replace('(img) ', '').replace(' (img)', '')
    
    print(f"[DEBUG] Clean current name: {clean_current_name}")
    print(f"[DEBUG] Clean base name: {clean_base_name}")

    # Determine dataset type using the global variable
    dataset_type = DATASETS_TYPE["value"]
    
    print(f"[DEBUG] Dataset type: {dataset_type}")
    
    if dataset_type == "image":
        source_dir = os.path.join(settings.DATASETS_IMG_DIR, clean_current_name)
        target_dir = os.path.join(settings.DATASETS_IMG_DIR, clean_base_name)
        file_extensions = settings.IMAGE_EXTENSIONS
        file_type = 'image'
    else: # dataset_type == "video"
        source_dir = os.path.join(settings.DATASETS_DIR, clean_current_name)
        target_dir = os.path.join(settings.DATASETS_DIR, clean_base_name)
        file_extensions = settings.VIDEO_EXTENSIONS
        file_type = 'video'
    
    print(f"[DEBUG] Source directory: {source_dir}")
    print(f"[DEBUG] Target directory: {target_dir}")
    
    if not os.path.exists(source_dir):
        error_msg = f"Error: Source directory not found: {source_dir}"
        print(f"[ERROR] {error_msg}")
        if e.page:
            e.page.snack_bar = ft.SnackBar(content=ft.Text(error_msg), open=True)
            e.page.update()
        return
    
    # Get existing files with appropriate extensions
    existing_files = set()
    try:
        for f in os.listdir(source_dir):
            file_ext = os.path.splitext(f)[1].lower().lstrip('.')
            if file_ext in file_extensions:
                existing_files.add(f)
        print(f"[DEBUG] Found {len(existing_files)} files to rename")
        print(f"[DEBUG] Looking for extensions: {file_extensions}")
        print(f"[DEBUG] Files in directory: {os.listdir(source_dir)[:10]}")  # Print first 10 files for debugging
    except Exception as ex:
        error_msg = f"Error reading files from {source_dir}: {str(ex)}"
        print(f"[ERROR] {error_msg}")
        if e.page:
            e.page.snack_bar = ft.SnackBar(content=ft.Text(error_msg), open=True)
            e.page.update()
        return
    
    if not existing_files:
        error_msg = f"No {file_type} files found in {source_dir}"
        print(f"[ERROR] {error_msg}")
        if e.page:
            e.page.snack_bar = ft.SnackBar(content=ft.Text(error_msg), open=True)
            e.page.update()
        return
    
    files_to_rename = sorted(list(existing_files))  # Sort for consistent renaming order

    if not files_to_rename:
        if e.page:
            e.page.snack_bar = ft.SnackBar(
                content=ft.Text(f"No {file_type} files found to rename in '{clean_dataset_name}'."), 
                open=True
            )
            e.page.update()
        return

    # --- Validate and clean up captions.json before renaming ---
    captions_path = os.path.join(source_dir, "captions.json")
    captions_data = []
    if os.path.exists(captions_path):
        try:
            with open(captions_path, "r", encoding="utf-8") as f:
                captions_data = json.load(f)

            initial_caption_count = len(captions_data)
            cleaned_captions_data = []
            removed_entries_count = 0

            for entry in captions_data:
                # Check if the entry is a dictionary and has a 'media_path' key
                if isinstance(entry, dict) and "media_path" in entry:
                    media_path_value = entry.get("media_path")
                    # Check if the media_path value exists as a file in the dataset folder
                    if media_path_value and os.path.exists(os.path.join(source_dir, media_path_value)):
                        cleaned_captions_data.append(entry)
                    else:
                        # Log a warning and increment removed count
                        print(f"Warning: Removing caption entry with invalid media_path: {media_path_value}")
                        removed_entries_count += 1
                else:
                    # Log a warning for invalid entry format
                    print(f"Warning: Removing invalid caption entry format: {entry}")
                    removed_entries_count += 1

            if removed_entries_count > 0:
                # Save the cleaned data back to captions.json
                with open(captions_path, "w", encoding="utf-8") as f:
                    json.dump(cleaned_captions_data, f, indent=2, ensure_ascii=False)
                if e.page:
                     e.page.snack_bar = ft.SnackBar(content=ft.Text(f"Cleaned up captions.json: Removed {removed_entries_count} invalid entr(y/ies)."), open=True)
                     e.page.update()
                captions_data = cleaned_captions_data # Update captions_data to the cleaned version for renaming

        except Exception as ex:
            if e.page:
                e.page.snack_bar = ft.SnackBar(content=ft.Text(f"Error validating/cleaning captions.json: {ex}"), open=True)
                e.page.update()
            # Continue with potentially problematic captions_data, but warn the user
            print(f"Error validating/cleaning captions.json: {ex}")
            # If loading failed, initialize captions_data as empty to prevent further errors
            if 'captions_data' not in locals() or captions_data is None:
                 captions_data = []
    else:
         captions_data = [] # Initialize as empty list if captions.json doesn't exist
    # -------------------------------------------------------------

    # Prepare new names and check for collisions
    new_names = []
    for idx, old_name in enumerate(files_to_rename, 1):
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
    # We use the initial set of existing files for this check
    for new_name in new_names:
        if new_name in existing_files and new_name not in new_names: # Check against original existing files, exclude the new names being created
            if e.page:
                e.page.snack_bar = ft.SnackBar(content=ft.Text(f"File {new_name} already exists and is not part of this renaming batch. Aborting."), open=True)
                e.page.update()
            return

    # Rename files and build old_to_new map
    old_to_new = {}
    renaming_successful = True
    for old_name, new_name in zip(files_to_rename, new_names):
        old_path = os.path.join(source_dir, old_name)
        new_path = os.path.join(source_dir, new_name)
        try:
            os.rename(old_path, new_path)
            old_to_new[old_name] = new_name
            print(f"[DEBUG] Renamed {old_name} to {new_name}")
        except Exception as ex:
            renaming_successful = False
            error_msg = f"Failed to rename {old_name} to {new_name}: {ex}"
            print(f"[ERROR] {error_msg}")
            if e.page:
                e.page.snack_bar = ft.SnackBar(content=ft.Text(error_msg), open=True)
                e.page.update()
            # Decide whether to stop or continue. Stopping is safer to avoid partial renames.
            return # Stop if any rename fails

    # Update captions.json if exists (using the potentially cleaned data)
    if captions_data: # Only proceed if captions_data is not empty after cleanup
        try:
            changed = False
            # Update filename fields in-place (never duplicate entries)
            for entry in captions_data:
                # Check if the entry is a dictionary before accessing keys
                if isinstance(entry, dict):
                    for field in ("media_path", "video"): # Check relevant fields
                        if field in entry and entry[field] in old_to_new:
                            entry[field] = old_to_new[entry[field]]
                            changed = True

            # --- Sort captions_data by the new media_path ---
            # Ensure all entries have a 'media_path' before sorting
            sortable_entries = [entry for entry in captions_data if isinstance(entry, dict) and "media_path" in entry]
            non_sortable_entries = [entry for entry in captions_data if not (isinstance(entry, dict) and "media_path" in entry)]

            # Sort the sortable entries
            sortable_entries.sort(key=lambda x: x.get("media_path", ""))

            # Combine sorted sortable entries and non-sortable entries (though non-sortable should be removed by cleanup)
            captions_data = sortable_entries + non_sortable_entries
            # ---------------------------------------------
            if changed or removed_entries_count > 0: # Save if renamed paths or if entries were removed during cleanup
                 with open(captions_path, "w", encoding="utf-8") as f:
                    json.dump(captions_data, f, indent=2, ensure_ascii=False)

        except Exception as ex:
            if e.page:
                e.page.snack_bar = ft.SnackBar(content=ft.Text(f"Failed to update/sort captions.json after renaming: {ex}"), open=True)
                e.page.update()
            # Continue as file renaming was successful, but notify user

    # Update info.json if exists (rename keys, preserve values, never duplicate)
    info_path = os.path.join(source_dir, "info.json")
    print(f"[DEBUG] Checking for info.json at: {info_path}")
    if os.path.exists(info_path):
        try:
            with open(info_path, "r", encoding="utf-8") as f:
                content = f.read()
                if content.strip():
                    info_data = json.loads(content)
                    # Ensure the loaded data is a dictionary
                    if not isinstance(info_data, dict):
                        info_data = {}
                else:
                    info_data = {}

            changed = False
            # If info_data is a dict with video filename keys, rename keys
            if isinstance(info_data, dict):
                new_info_data = {}
                # Sort the original keys to maintain some order, though info.json structure might vary
                for k in sorted(info_data.keys()):
                    v = info_data[k]
                    new_key = old_to_new.get(k, k)
                    new_info_data[new_key] = v
                    if new_key != k:
                        changed = True
                info_data = new_info_data
            # No need for recursive update for other types if structure is standardized
            # Check for cap and proc directories and set flags
            cap_dir = os.path.join(source_dir, "cap")
            proc_dir = os.path.join(source_dir, "proc")
            
            if os.path.isdir(cap_dir):
                print(f"[DEBUG] Found 'cap' directory, setting 'cap' to 'yes'")
                info_data["cap"] = "yes"
            
            if os.path.isdir(proc_dir):
                print(f"[DEBUG] Found 'proc' directory, setting 'proc' to 'yes'")
                info_data["proc"] = "yes"
            
            # Save the updated info.json
            with open(info_path, "w", encoding="utf-8") as f:
                json.dump(info_data, f, indent=2, ensure_ascii=False)
                print(f"[DEBUG] Updated info.json with cap/proc settings")
        except Exception as ex:
            print(f"Error updating info.json: {ex}")
            # Non-critical, continue

    # Success feedback and UI update
    if renaming_successful:
        # Determine the correct thumbnail directory
        if dataset_type == "image":
            thumbnails_dir = os.path.join(settings.THUMBNAILS_IMG_BASE_DIR, clean_current_name)
        else:
            thumbnails_dir = os.path.join(settings.THUMBNAILS_BASE_DIR, clean_current_name)

        # Clean up all existing thumbnails in the directory
        if os.path.exists(thumbnails_dir):
            for thumb_file in os.listdir(thumbnails_dir):
                try:
                    os.remove(os.path.join(thumbnails_dir, thumb_file))
                    print(f"[DEBUG] Deleted old thumbnail: {thumb_file}")
                except Exception as ex:
                    print(f"[ERROR] Failed to delete old thumbnail {thumb_file}: {ex}")

        if e.page:
            e.page.snack_bar = ft.SnackBar(content=ft.Text(f"Renamed {len(files_to_rename)} files successfully and cleaned up old thumbnails."), open=True)
            update_thumbnails(page_ctx=e.page, grid_control=thumbnails_grid_ref.current, force_refresh=True) # Force refresh to update image sources
            e.page.update()


def on_bucket_or_model_change(e: ft.ControlEvent):
    current_dataset_name = selected_dataset.get("value")
    if not current_dataset_name:
        return

    bucket_str_val = bucket_size_textfield.value # Initialize bucket_str_val

    # Validate bucket format and update bucket_str_val if invalid
    parsed_bucket_list = parse_bucket_string_to_list(bucket_str_val)
    if parsed_bucket_list is None:
        if e.page:
            e.page.snack_bar = ft.SnackBar(
                content=ft.Text(f"Invalid Bucket Size format: '{bucket_str_val}'. Using default."),
                open=True
            )
            e.page.update()
        bucket_str_val = settings.DEFAULT_BUCKET_SIZE_STR # Use default if invalid

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


async def on_dataset_dropdown_change(
    ev: ft.ControlEvent,
    thumbnails_grid_control: ft.GridView,
    dataset_delete_captions_button_control: ft.ElevatedButton,
    bucket_size_textfield_control: ft.TextField, # New argument
    model_name_dropdown_control: ft.Dropdown,    # New argument
    trigger_word_textfield_control: ft.TextField # New argument
):
    # Hide output and progress bar when changing dataset
    if processed_output_field.page:
        processed_output_field.visible = False
        set_bottom_app_bar_height() # Recalculate height
    if processed_progress_bar.page:
        processed_progress_bar.visible = False

    selected_dataset["value"] = ev.control.value
    
    # Determine dataset type and store it globally
    base_dir, dataset_type = _get_dataset_base_dir(selected_dataset["value"])
    DATASETS_TYPE["value"] = dataset_type

    # Load and update config fields based on selected dataset
    bucket_val, model_val, trigger_word_val = load_dataset_config(selected_dataset["value"])
    
    # Use the passed control arguments directly
    bucket_size_textfield_control.value = bucket_val
    # Ensure loaded model value is a valid choice, otherwise use default
    model_name_dropdown_control.value = model_val if model_val in settings.ltx_models else settings.ltx_def_model
    trigger_word_textfield_control.value = trigger_word_val or ''

    # Add update calls for the controls
    bucket_size_textfield_control.update()
    model_name_dropdown_control.update()
    trigger_word_textfield_control.update()

    # Update thumbnails for the new dataset
    update_thumbnails(page_ctx=ev.page, grid_control=thumbnails_grid_control)

    # Update delete captions button state
    if dataset_delete_captions_button_control:
        pass # Keep button always enabled


    if ev.page:
        ev.page.update() # Use await for page.update()


async def on_update_button_click(e: ft.ControlEvent, dataset_dropdown_control, thumbnails_grid_control, add_button, delete_button):
    await update_dataset_dropdown(e.page, dataset_dropdown_control, thumbnails_grid_control, delete_button)
    await reload_current_dataset(e.page, dataset_dropdown_control, thumbnails_grid_control, add_button, delete_button)


def on_add_captions_click_with_model(e: ft.ControlEvent,
                                     caption_model_dropdown: ft.Dropdown,
                                     captions_checkbox: ft.Checkbox,
                                     cap_command_textfield: ft.TextField,
                                     max_tokens_textfield: ft.TextField,
                                     dataset_add_captions_button_control: ft.ElevatedButton,
                                     dataset_delete_captions_button_control: ft.ElevatedButton,
                                     thumbnails_grid_control: ft.GridView):
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

    # Determine dataset type using the global variable
    dataset_type = DATASETS_TYPE["value"]
    
    # Remove the ' (img)' suffix if present for the actual folder name
    clean_dataset_name = current_dataset_name.replace(' (img)', '')
    
    if dataset_type == "image":
        dataset_folder_path = os.path.abspath(os.path.join(settings.DATASETS_IMG_DIR, clean_dataset_name))
    else: # dataset_type == "video"
        dataset_folder_path = os.path.abspath(os.path.join(settings.DATASETS_DIR, clean_dataset_name))
    
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
    page_for_dialog = e.page
    button_control = e.control
    current_dataset_name = selected_dataset.get("value")

    if not current_dataset_name:
        if page_for_dialog:
            page_for_dialog.snack_bar = ft.SnackBar(content=ft.Text("No dataset selected."), open=True)
            page_for_dialog.update()
        return

    base_dir, _ = _get_dataset_base_dir(current_dataset_name)
    captions_file_path = os.path.join(base_dir, current_dataset_name.replace('(img) ', '').replace(' (img)', ''), "captions.json")
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
    current_dataset_name = selected_dataset.get("value")
    if not current_dataset_name:
        if e.page:
            e.page.snack_bar = ft.SnackBar(content=ft.Text("Error: No dataset selected for preprocessing."), open=True)
            e.page.update()
        return

    # Clean the dataset name by removing any image markers (both old and new formats)
    clean_dataset_name = current_dataset_name.replace('(img) ', '').replace(' (img)', '')

    # Determine dataset type using the global variable
    dataset_type = DATASETS_TYPE["value"]
    
    if dataset_type == "image":
        dataset_dir = settings.DATASETS_IMG_DIR
    else: # dataset_type == "video"
        dataset_dir = settings.DATASETS_DIR
    
    input_captions_json_path = os.path.abspath(os.path.join(dataset_dir, clean_dataset_name, "captions.json"))
    preprocess_output_dir = os.path.abspath(os.path.join(dataset_dir, clean_dataset_name, "preprocessed_data"))

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

# Global dictionary to track temporary thumbnail paths and their creation times
_temp_thumbnails = {}

def _on_thumbnail_checkbox_change(video_path: str, is_checked: bool, thumbnail_index: int):
    global selected_thumbnails_set, last_clicked_thumbnail_index
    # Use the global state from flet_hotkeys.py
    import ui.flet_hotkeys # Re-import to get a mutable reference to the module itself

    if ui.flet_hotkeys.is_d_key_pressed_global and last_clicked_thumbnail_index != -1:
        # D key is pressed, perform range selection
        start_index = min(last_clicked_thumbnail_index, thumbnail_index)
        end_index = max(last_clicked_thumbnail_index, thumbnail_index)

        for i in range(start_index, end_index + 1):
            if i < len(thumbnails_grid_ref.current.controls):
                control = thumbnails_grid_ref.current.controls[i]
                if isinstance(control, ft.Container) and isinstance(control.content, ft.Stack) and len(control.content.controls) > 1:
                    # Call the new function from dataset_thumb_layout to update the visual state
                    set_thumbnail_selection_state(control, is_checked)
                    
                    if is_checked:
                        selected_thumbnails_set.add(control.data) # Add video_path to set
                    else:
                        selected_thumbnails_set.discard(control.data) # Remove video_path from set
        
        # Reset the global D key state after processing the range selection
        ui.flet_hotkeys.is_d_key_pressed_global = False

    else:
        # Normal single selection (the visual update for this single click is handled in dataset_thumb_layout.py)
        if is_checked:
            selected_thumbnails_set.add(video_path)
        else:
            selected_thumbnails_set.discard(video_path)

    last_clicked_thumbnail_index = thumbnail_index # Update last clicked index

    # print(f"Selected: {len(selected_thumbnails_set)} items") # Debugging

def cleanup_old_temp_thumbnails(thumb_dir: str, max_age_seconds: int = 3600):
    """Clean up temporary thumbnails older than max_age_seconds"""
    current_time = time.time()
    if not os.path.exists(thumb_dir):
        return
        
    for filename in os.listdir(thumb_dir):
        if filename.endswith('.tmp_'):
            try:
                file_path = os.path.join(thumb_dir, filename)
                file_mtime = os.path.getmtime(file_path)
                if current_time - file_mtime > max_age_seconds:
                    os.remove(file_path)
            except Exception as e:
                print(f"Error cleaning up old temp thumbnail {filename}: {e}")

def update_thumbnails(page_ctx: ft.Page | None, grid_control: ft.GridView | None, force_refresh: bool = False):
    global _temp_thumbnails, selected_thumbnails_set, last_clicked_thumbnail_index
    
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
        thumbnail_paths_map, video_info = get_videos_and_thumbnails(current_selection, DATASETS_TYPE["value"])
        video_files_list["value"] = list(thumbnail_paths_map.keys()) # Update global list
        dataset_captions = load_dataset_captions(current_selection)

        if not thumbnail_paths_map:
            grid_control.controls.append(ft.Text(f"No videos found in dataset '{current_selection}'."))
        else:
            # Clear selection state if dataset changes or force refresh
            if force_refresh or (selected_dataset.get("value") != current_selection):
                selected_thumbnails_set.clear()
                last_clicked_thumbnail_index = -1

            # Sort thumbnails by video_path for consistent indexing
            sorted_thumbnail_items = sorted(thumbnail_paths_map.items(), key=lambda item: item[0])

            for i, (video_path, thumb_path) in enumerate(sorted_thumbnail_items):
                has_caption = any(entry.get("media_path") == os.path.basename(video_path) and entry.get("caption", "").strip() for entry in dataset_captions)
                
                grid_control.controls.append(
                    create_thumbnail_container(
                        page_ctx=page_ctx,
                        video_path=video_path,
                        thumb_path=thumb_path,
                        video_info=video_info,
                        has_caption=has_caption,
                        processed_map=processed_map,
                        video_files_list=video_files_list["value"],
                        update_thumbnails_callback=update_thumbnails,
                        grid_control=grid_control,
                        on_checkbox_change_callback=_on_thumbnail_checkbox_change, # Pass the new handler
                        thumbnail_index=i, # Pass the index
                        is_selected_initially=(video_path in selected_thumbnails_set) # Pass initial selection state
                    )
                )

    # Update the grid view itself
    if grid_control and grid_control.page:
        grid_control.update()

    # Clean up old temporary thumbnails if we're doing a refresh
    if force_refresh and current_selection:
        # Use the global DATASETS_TYPE to determine the base directory
        dataset_type = DATASETS_TYPE["value"]
        base_dir = settings.DATASETS_IMG_DIR if dataset_type == "image" else settings.DATASETS_DIR
        dataset_folder_path = os.path.abspath(os.path.join(base_dir, current_selection.replace('(img) ', '').replace(' (img)', '')))
        
        # Clean up any old temp thumbnails in this directory
        cleanup_old_temp_thumbnails(dataset_folder_path)
        
        # Also clean up any thumbnails in the thumbnails directory
        thumb_dir = os.path.join(settings.THUMBNAILS_IMG_BASE_DIR if dataset_type == "image" else settings.THUMBNAILS_BASE_DIR, current_selection.replace('(img) ', '').replace(' (img)', ''))
        if os.path.exists(thumb_dir):
            cleanup_old_temp_thumbnails(thumb_dir)


def update_dataset_dropdown(
    p_page: ft.Page | None,
    current_dataset_dropdown: ft.Dropdown,
    current_thumbnails_grid: ft.GridView,
    delete_button: ft.ElevatedButton # Pass delete button
):
    folders = get_dataset_folders()
    # Use dictionary items to get both value (display name) and key (actual folder name)
    current_dataset_dropdown.options = [ft.dropdown.Option(key=name, text=display_name) for name, display_name in folders.items()] if folders else []
    current_dataset_dropdown.value = None # Clear selection
    selected_dataset["value"] = None # Clear global state

    # Load and set default config values
    bucket_val, model_val, trigger_word_val = load_dataset_config(None) # Load defaults
    if bucket_size_textfield: bucket_size_textfield.value = bucket_val
    if model_name_dropdown: model_name_dropdown.value = model_val
    if trigger_word_textfield: trigger_word_textfield.value = trigger_word_val or ''

    # Add update calls for the controls
    if bucket_size_textfield: bucket_size_textfield.update()
    if model_name_dropdown: model_name_dropdown.update()
    if trigger_word_textfield: trigger_word_textfield.update()

    # Update thumbnails (will show "Select a dataset...")
    update_thumbnails(page_ctx=p_page, grid_control=current_thumbnails_grid)

    # Update delete captions button state
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
    # Hide output and progress bar on reload
    if processed_output_field.page:
        processed_output_field.visible = False
        set_bottom_app_bar_height() # This function updates
    if processed_progress_bar.page:
        processed_progress_bar.visible = False

    folders = get_dataset_folders()
    # Use dictionary items to get both value (display name) and key (actual folder name)
    current_dataset_dropdown.options = [ft.dropdown.Option(key=name, text=display_name) for name, display_name in folders.items()] if folders else []
    current_dataset_dropdown.disabled = len(folders) == 0

    prev_selected_name = selected_dataset.get("value")

    if prev_selected_name and prev_selected_name in folders:
        # If previously selected dataset still exists, keep it selected
        current_dataset_dropdown.value = prev_selected_name # Set value to the key (actual folder name)
        selected_dataset["value"] = prev_selected_name
        bucket_val, model_val, trigger_word_val = load_dataset_config(prev_selected_name)
        if bucket_size_textfield: bucket_size_textfield.value = bucket_val
        # Ensure loaded model value is a valid choice, otherwise use default
        if model_name_dropdown: model_name_dropdown.value = model_val if model_val in settings.ltx_models else settings.ltx_def_model
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

def _build_expansion_tile(
    title: str,
    controls: list[ft.Control],
    initially_expanded: bool = False,
):
    return ft.ExpansionTile(
        title=ft.Text(title, size=12),
        bgcolor=EXPANSION_TILE_INSIDE_BG_COLOR,
        collapsed_bgcolor=EXPANSION_TILE_HEADER_BG_COLOR,
        controls=[ft.Divider(), ft.Column(controls, spacing=10) ,ft.Divider()], # Wrap controls in a Column
        initially_expanded=initially_expanded,
        collapsed_shape=ft.RoundedRectangleBorder(radius=EXPANSION_TILE_BORDER_RADIUS),
        shape=ft.RoundedRectangleBorder(radius=EXPANSION_TILE_BORDER_RADIUS),
        enable_feedback=False,
    )

def _create_global_controls():
    global bucket_size_textfield, rename_textfield, model_name_dropdown, trigger_word_textfield
    # Note: processed_progress_bar and processed_output_field are already global and initialized

    # Check if controls are already initialized to prevent re-creation
    if bucket_size_textfield is not None:
        return # Controls already exist

    bucket_size_textfield = create_textfield(
        label="Bucket Size (e.g., [W, H, F] or WxHxF)",
        value=settings.DEFAULT_BUCKET_SIZE_STR,
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
        settings.ltx_def_model, # Use config directly
        {name: name for name in settings.ltx_models}, # Use config directly
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
    return ft.Column([
        ft.Container(height=10), # Add a divider above the first row
        ft.Row([
            ft.Container(content=dataset_dropdown_control, expand=True, width=160),
            ft.Container(content=update_button_control, alignment=ft.alignment.center_right, width=40),
        ], expand=True), # Make the Row expand horizontally
        ft.Container(height=3), # Spacer
        ft.Divider(), # Add a divider below the first row
    ], spacing=0) # Set spacing to 0 as Row handles spacing

def _build_captioning_section(
    caption_model_dropdown: ft.Dropdown,
    captions_checkbox_container: ft.Container, # Container for 8-bit checkbox (pass container for layout)
    cap_command_textfield: ft.TextField,
    max_tokens_textfield: ft.TextField,
    dataset_add_captions_button_control: ft.ElevatedButton,
    dataset_delete_captions_button_control: ft.ElevatedButton,):
    # No on_click handlers assigned here anymore
    return _build_expansion_tile(
        title="1. Captions",
        controls=[ # Pass the controls list directly
            ft.ResponsiveRow([captions_checkbox_container, caption_model_dropdown]), # Removed extra Container around dropdown
            ft.ResponsiveRow([max_tokens_textfield, cap_command_textfield]),
            ft.Row([
                ft.Container(content=dataset_add_captions_button_control, expand=True),
                ft.Container(content=dataset_delete_captions_button_control, expand=True)
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
        ],
        initially_expanded=True, # Set to True
    )

def _build_preprocessing_section(
    model_name_dropdown: ft.Dropdown,
    bucket_size_textfield: ft.TextField,
    trigger_word_textfield: ft.TextField,
    dataset_preprocess_button_control: ft.ElevatedButton,
):
    # No on_click handlers assigned here anymore
    # Use the generic function
    return _build_expansion_tile(
        title="2. Preprocess Dataset",
        controls=[ # Pass the controls list directly
            model_name_dropdown,
            bucket_size_textfield,
            trigger_word_textfield,
            dataset_preprocess_button_control,
            ft.Container(height=10), # Spacer
        ],
        initially_expanded=False, # Set to False
    )

affix_text_field_ref = ft.Ref[ft.TextField]()
find_text_field_ref = ft.Ref[ft.TextField]()
replace_text_field_ref = ft.Ref[ft.TextField]()

def _build_latent_test_section(update_thumbnails_func):
    find_replace=ft.Column([
        create_textfield(label="Find",value="",expand=True, ref=find_text_field_ref),
        create_textfield(label="Replace", value="", expand=True, ref=replace_text_field_ref),
        create_styled_button("Find and Replace", on_click=lambda e: e.page.run_task(find_and_replace_in_captions,
            e, selected_dataset, DATASETS_TYPE["value"], find_text_field_ref, replace_text_field_ref, update_thumbnails_func, thumbnails_grid_ref
        ))
    ])
    prefix_suffix_replace=ft.Column([
        create_textfield(label="Text",value="",expand=True, ref=affix_text_field_ref),
        ft.ResponsiveRow([
            create_styled_button("Add prefix",col=6, on_click=lambda e: e.page.run_task(apply_affix_from_textfield,
                e, "prefix", selected_dataset, DATASETS_TYPE["value"], update_thumbnails_func, thumbnails_grid_ref, affix_text_field_ref
            )),
            create_styled_button("Add suffix",col=6, on_click=lambda e: e.page.run_task(apply_affix_from_textfield,
                e, "suffix", selected_dataset, DATASETS_TYPE["value"], update_thumbnails_func, thumbnails_grid_ref, affix_text_field_ref
            ))
        ])
    ])
    # Use the generic function
    return _build_expansion_tile(
        title="Batch captions",
        controls=[ # Pass the controls list directly
            find_replace,
            ft.Divider(thickness=1,height=3),
            prefix_suffix_replace
        ],
        initially_expanded=False, # Set to False
    )

def _build_batch_section(change_fps_section: ft.ResponsiveRow, rename_textfield: ft.TextField, rename_files_button: ft.ElevatedButton):

    # No on_click handlers assigned here anymore
    # Use the generic function
    return _build_expansion_tile(
        title="Batch files",
        controls=[ # Pass the controls list directly
            change_fps_section,
            ft.Divider(thickness=1), # Spacer
            rename_textfield,
            rename_files_button,
        ],
        initially_expanded=False, # Set to False
    )

def _build_bottom_status_bar():
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
    global bottom_app_bar_ref # No longer need to reference is_d_key_pressed here
    p_page = page # Alias for clarity

    # The global keyboard event handler is now managed in flet_hotkeys.py
    # No need for a local on_keyboard_event here.

    # Create global controls if not already created (should happen once per app lifecycle)
    if bucket_size_textfield is None: # Check one of the global controls
        _create_global_controls()

    # --- Create Controls ---
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

    thumbnails_grid_control = ft.GridView(
        ref=thumbnails_grid_ref, # Assign the global ref
        runs_count=5, max_extent=settings.THUMB_TARGET_W + 20,
        child_aspect_ratio=(settings.THUMB_TARGET_W + 10) / (settings.THUMB_TARGET_H + 80), # Adjusted aspect ratio
        spacing=7, run_spacing=7, controls=[], expand=True
    )

    dataset_dropdown_control.on_change = lambda ev: ev.page.run_task(
        on_dataset_dropdown_change, # Pass the function itself
        ev,
        thumbnails_grid_control,
        dataset_delete_captions_button_ref.current,
        bucket_size_textfield,
        model_name_dropdown,
        trigger_word_textfield
    )

    update_button_control = ft.IconButton(
        icon=ft.Icons.REFRESH, 
        tooltip="Update dataset list and refresh thumbnails",
        style=ft.ButtonStyle(padding=ft.padding.symmetric(horizontal=8)), 
        icon_size=20,
        on_click=lambda e: update_thumbnails(page_ctx=e.page, grid_control=thumbnails_grid_control, force_refresh=True)
    )

    # Captioning specific controls
    caption_model_dropdown = create_dropdown(
        "Captioning Model",
        settings.captions_def_model,  # default
        settings.captions, # Use config.captions dictionary directly
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
    dataset_delete_captions_button_control = create_styled_button(
        "Delete",
        ref=dataset_delete_captions_button_ref, # Assign ref - Valid for ft.ElevatedButton (assuming create_styled_button returns one and passes kwargs)
        tooltip="Delete the captions.json file",
        expand=True, # Make it expandable for layout
        button_style=BTN_STYLE2,
    )
    # Assign to ref
    dataset_delete_captions_button_ref.current = dataset_delete_captions_button_control

    dataset_add_captions_button_control = create_styled_button(
        "Add Captions",
        ref=dataset_add_captions_button_ref, # Assign ref - Valid for ft.ElevatedButton (assuming create_styled_button returns one and passes kwargs)
        button_style=BTN_STYLE2,
        expand=True
    )
    # Assign to ref
    dataset_add_captions_button_ref.current = dataset_add_captions_button_control

    # Preprocessing specific button (uses global controls bucket_size_textfield, model_name_dropdown, trigger_word_textfield)
    dataset_preprocess_button_control = create_styled_button(
        "Start Preprocess ",
        ref=dataset_preprocess_button_ref, # Assign ref - Valid for ft.ElevatedButton
        tooltip="Preprocess dataset using captions.json",
        expand=True,
        button_style=BTN_STYLE2
    )
    # Assign to ref
    dataset_preprocess_button_ref.current = dataset_preprocess_button_control

    # Change fps
    change_fps_textfield = create_textfield("Change fps", "24",
                                    expand=True,
                                    hint_text="fps",col=4,
    )
    change_fps_textfield_ref.current = change_fps_textfield # Assign to global ref

    change_fps_button = create_styled_button(
        "Change fps",
        tooltip="Change fps",
        expand=True,
        on_click=lambda e: e.page.run_task(on_change_fps_click,
            e, selected_dataset, DATASETS_TYPE["value"], change_fps_textfield_ref, thumbnails_grid_ref, update_thumbnails_func, settings
        ),
        button_style=BTN_STYLE2
    )
    change_fps_section = ft.ResponsiveRow([
        ft.Container(content=change_fps_textfield, col=4,),
        ft.Container(content=change_fps_button, col=8,),
    ], spacing=5)


    # Rename specific controls
    rename_files_button = create_styled_button(
        "Rename files",
        tooltip="Rename files",
        expand=True,
        on_click=lambda e: e.page.run_task(on_rename_files_click,
            e, selected_dataset, DATASETS_TYPE["value"], rename_textfield, thumbnails_grid_ref, update_thumbnails_func, settings
        ),
        button_style=BTN_STYLE2
    )

    # --- Assign Event Handlers ---
    update_button_control.on_click = lambda e: reload_current_dataset( # Call reload_current_dataset directly
        e.page,
        dataset_dropdown_control_ref.current,
        thumbnails_grid_control,
        dataset_add_captions_button_ref.current,
        dataset_delete_captions_button_ref.current
    )
    # Assign on_click handlers for captioning buttons
    dataset_add_captions_button_control.on_click = lambda e: on_add_captions_click_with_model(
        e,
        caption_model_dropdown_ref.current, # Use ref's current value
        captions_checkbox_ref.current, # Use ref's current value
        cap_command_textfield_ref.current, # Use ref's current value
        max_tokens_textfield_ref.current, # Use ref's current value
        dataset_add_captions_button_ref.current, # Use ref's current value
        dataset_delete_captions_button_ref.current, # Use ref's current value
        thumbnails_grid_control # Use ref's current value
    )
    dataset_delete_captions_button_control.on_click = lambda e: on_delete_captions_click(e, thumbnails_grid_control) # Use ref's current value

    # Assign on_click handler for preprocess button
    dataset_preprocess_button_control.on_click = lambda e: on_preprocess_dataset_click(
        e,
        model_name_dropdown, # Global control
        bucket_size_textfield, # Global control
        trigger_word_textfield # Global control
    )

    # Assign on_click handler for rename button
    change_fps_button.on_click = lambda e: on_change_fps_click(e) # Uses rename_textfield and thumbnails_grid_ref.current internally
    rename_files_button.on_click = lambda e: on_rename_files_click(e) # Uses rename_textfield and thumbnails_grid_ref.current internally

    # --- Assemble Sections ---
    dataset_selection_section = _build_dataset_selection_section(dataset_dropdown_control_ref.current, update_button_control)

    captioning_section = _build_captioning_section(
        caption_model_dropdown_ref.current,
        captions_checkbox_container, # Still passing container as it includes styling/layout
        cap_command_textfield_ref.current,
        max_tokens_textfield_ref.current,
        dataset_add_captions_button_ref.current,
        dataset_delete_captions_button_ref.current,
    )

    preprocessing_section = _build_preprocessing_section(
        model_name_dropdown, # Global control
        bucket_size_textfield, # Global control
        trigger_word_textfield, # Global control
        dataset_preprocess_button_ref.current, # Use ref
    )

    latent_test_section = _build_latent_test_section(update_thumbnails) # Placeholder

    batch_section = _build_batch_section(change_fps_section, rename_textfield, rename_files_button) # Global control, pass grid ref

    # Assemble sections into the left column
    lc_content = ft.Column([
        dataset_selection_section,
        captioning_section,
        preprocessing_section,
        latent_test_section, # Placeholder
        batch_section, # Global control, pass grid ref
    ], spacing=3, width=200, alignment=ft.MainAxisAlignment.START) # Set spacing to 5 for ExpansionTiles

    # Build the bottom status bar
    bottom_app_bar = _build_bottom_status_bar() # Assigns to global bottom_app_bar_ref

    # Assemble the right column
    rc_content = ft.Column([
        thumbnails_grid_control, # Global ref grid
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
        thumbnails_grid_control,
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
