# This file will contain functions related to dataset actions,
# such as captioning, preprocessing, renaming, and FPS changes.
# It will also house the event handlers for these actions.

import flet as ft
import os
import json
import asyncio
import signal
import subprocess
import shutil
import time

from settings import settings
from ui_popups.delete_caption_dialog import show_delete_caption_dialog
from ui.dataset_manager.dataset_utils import (
    load_dataset_config, save_dataset_config, load_processed_map,
    load_dataset_captions, delete_captions_file, validate_bucket_values,
    _get_dataset_base_dir, get_videos_and_thumbnails, get_dataset_folders,
    parse_bucket_string_to_list # Add this import
)
from ui.dataset_manager.dataset_thumb_layout import create_thumbnail_container, set_thumbnail_selection_state
from ui.flet_hotkeys import is_d_key_pressed_global # Import global D key state

# Global references from dataset_layout_tab.py that need to be accessed here
# These will be passed as arguments to functions or accessed via global state if necessary
# For now, I'll assume they are passed as arguments where needed.
# If a global reference is truly needed, it should be imported or managed carefully.

# Process tracking for stopping script execution
current_caption_process = {"proc": None}

def _get_selected_filenames(thumbnails_grid_control: ft.GridView) -> list[str]:
    """
    Helper function to get a list of base filenames for selected thumbnails.
    """
    selected_filenames = []
    if thumbnails_grid_control and thumbnails_grid_control.current and thumbnails_grid_control.current.controls:
        for thumbnail_container in thumbnails_grid_control.current.controls:
            if isinstance(thumbnail_container, ft.Container) and \
               isinstance(thumbnail_container.content, ft.Stack):
                
                checkbox = None
                for control in thumbnail_container.content.controls:
                    if isinstance(control, ft.Checkbox):
                        checkbox = control
                        break
                
                if checkbox and checkbox.value:
                    # thumbnail_container.data holds the original video_path/image_path
                    selected_filenames.append(os.path.basename(thumbnail_container.data))
    return selected_filenames

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
    command += f' --max-new_tokens {max_new_tokens}'

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

async def run_dataset_script_command(
    command_str: str,
    page_ref: ft.Page,
    button_ref: ft.ElevatedButton,
    progress_bar_ref: ft.ProgressBar,
    output_field_ref: ft.TextField,
    original_button_text: str,
    set_bottom_app_bar_height_func, # Pass this function from layout_tab
    delete_button_ref=None,
    thumbnails_grid_control=None,
    on_success_callback=None,
):
    def append_output(text):
        output_field_ref.value += text
        output_field_ref.visible = True
        set_bottom_app_bar_height_func()
        if page_ref.client_storage:
            output_field_ref.update()
            page_ref.update()

    try:
        output_field_ref.value = ""
        output_field_ref.visible = True
        progress_bar_ref.visible = True
        set_bottom_app_bar_height_func()
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
        set_bottom_app_bar_height_func() # Re-evaluate height in case output field is no longer visible
        # --- Restore Delete button after captioning completes or fails ---
        if delete_button_ref is not None and thumbnails_grid_control is not None:
            delete_button_ref.text = "Delete"
            # The on_click for delete needs to be re-assigned to the correct handler in dataset_actions
            # This will be handled by passing the correct function reference from dataset_layout_tab
            # For now, keep it as a placeholder or assume it's set externally.
            # delete_button_ref.on_click = lambda evt: on_delete_captions_click(evt, thumbnails_grid_control)
            delete_button_ref.tooltip = "Delete the captions.json file"
            # Only re-enable if there's a dataset selected (handled in on_dataset_dropdown_change, but good to be safe)
            # This requires selected_dataset to be passed or accessed globally.
            # For now, assume it's passed or handled by the caller.
            # delete_button_ref.disabled = not selected_dataset.get("value")
            delete_button_ref.update()

        if page_ref.client_storage:
            page_ref.update()

# ======================================================================================
# GUI Event Handlers (Handle user interactions)
# ======================================================================================

async def on_change_fps_click(e: ft.ControlEvent, selected_dataset_ref, DATASETS_TYPE_ref, change_fps_textfield_ref_obj, thumbnails_grid_ref_obj, update_thumbnails_func):
    current_dataset_name = selected_dataset_ref.get("value")
    if not current_dataset_name:
        if e.page:
            e.page.snack_bar = ft.SnackBar(content=ft.Text("Error: No dataset selected."), open=True)
            e.page.update()
        return

    if not change_fps_textfield_ref_obj.current or not change_fps_textfield_ref_obj.current.value:
        if e.page:
            e.page.snack_bar = ft.SnackBar(content=ft.Text("Error: FPS textfield not available or empty."), open=True)
            e.page.update()
        return
        
    target_fps_str = change_fps_textfield_ref_obj.current.value.strip()
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

    if processed_files > 0 and thumbnails_grid_ref_obj.current and e.page:
        update_thumbnails_func(e.page, thumbnails_grid_ref_obj.current, force_refresh=True)

async def on_rename_files_click(e: ft.ControlEvent, selected_dataset_ref, DATASETS_TYPE_ref, rename_textfield_obj, thumbnails_grid_ref_obj, update_thumbnails_func):
    print("\n=== RENAME FUNCTION CALLED (dataset_actions.py) ===")
    current_dataset_name = selected_dataset_ref.get("value")
    print(f"[DEBUG] Current dataset name: {current_dataset_name}")
    
    if not current_dataset_name:
        if e.page:
            e.page.snack_bar = ft.SnackBar(content=ft.Text("Error: No dataset selected for renaming."), open=True)
            e.page.update()
        return
    
    if not rename_textfield_obj or not rename_textfield_obj.value:
        if e.page:
            e.page.snack_bar = ft.SnackBar(content=ft.Text("Error: Rename textfield not available or empty."), open=True)
            e.page.update()
        return

    base_name = rename_textfield_obj.value.strip()
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
    dataset_type = DATASETS_TYPE_ref["value"]
    
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
    
    selected_files_from_thumbnails = _get_selected_filenames(thumbnails_grid_ref_obj)
    
    if selected_files_from_thumbnails:
        print(f"[DEBUG] Renaming {len(selected_files_from_thumbnails)} selected files.")
        # Ensure selected files are part of existing_files to prevent renaming non-existent files
        files_to_rename = sorted([f for f in selected_files_from_thumbnails if f in existing_files])
    else:
        print("[DEBUG] No thumbnails selected, renaming all existing files.")
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
            update_thumbnails_func(page_ctx=e.page, grid_control=thumbnails_grid_ref_obj.current, force_refresh=True) # Force refresh to update image sources
            e.page.update()

async def on_bucket_or_model_change(e: ft.ControlEvent, selected_dataset_ref, bucket_size_textfield_obj, model_name_dropdown_obj, trigger_word_textfield_obj):
    current_dataset_name = selected_dataset_ref.get("value")
    if not current_dataset_name:
        return

    bucket_str_val = bucket_size_textfield_obj.value # Initialize bucket_str_val

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
        model_name_dropdown_obj.value,
        trigger_word_textfield_obj.value
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

async def on_add_captions_click_with_model(e: ft.ControlEvent,
                                     caption_model_dropdown: ft.Dropdown,
                                     captions_checkbox: ft.Checkbox,
                                     cap_command_textfield: ft.TextField,
                                     max_tokens_textfield: ft.TextField,
                                     dataset_add_captions_button_control: ft.ElevatedButton,
                                     dataset_delete_captions_button_control: ft.ElevatedButton,
                                     thumbnails_grid_control: ft.GridView,
                                     selected_dataset_ref,
                                     DATASETS_TYPE_ref,
                                     processed_progress_bar_ref,
                                     processed_output_field_ref,
                                     set_bottom_app_bar_height_func,
                                     update_thumbnails_func):
    # If a process is running, treat as stop
    proc = current_caption_process.get("proc")
    if proc is not None and proc.returncode is None:
        stop_captioning(
            e,
            dataset_add_captions_button_control,
            dataset_delete_captions_button_control,
            thumbnails_grid_control,
            selected_dataset_ref,
            processed_progress_bar_ref,
            processed_output_field_ref,
            set_bottom_app_bar_height_func
        )
        return

    selected_model = caption_model_dropdown.value or "llava_next_7b"
    current_dataset_name = selected_dataset_ref.get("value")

    if not current_dataset_name:
        if e.page:
            e.page.snack_bar = ft.SnackBar(content=ft.Text("Error: No dataset selected."), open=True)
            e.page.update()
        return

    # Determine dataset type using the global variable
    dataset_type = DATASETS_TYPE_ref["value"]
    
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
        thumbnails_grid_control,
        selected_dataset_ref,
        processed_progress_bar_ref,
        processed_output_field_ref,
        set_bottom_app_bar_height_func
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
            processed_progress_bar_ref,
            processed_output_field_ref,
            "Add Captions", # Original button text
            set_bottom_app_bar_height_func,
            delete_button_ref=dataset_delete_captions_button_control,
            thumbnails_grid_control=thumbnails_grid_control,
            on_success_callback=lambda: update_thumbnails_func(page_ctx=e.page, grid_control=thumbnails_grid_control, force_refresh=True) # Force refresh after captioning
        )

def stop_captioning(e: ft.ControlEvent,
                    add_button: ft.ElevatedButton,
                    delete_button: ft.ElevatedButton,
                    thumbnails_grid_control: ft.GridView,
                    selected_dataset_ref,
                    processed_progress_bar_ref,
                    processed_output_field_ref,
                    set_bottom_app_bar_height_func):
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
    # Re-assign on_click to the correct handler in dataset_actions
    delete_button.on_click = lambda evt: on_delete_captions_click(evt, thumbnails_grid_control, selected_dataset_ref, processed_progress_bar_ref, processed_output_field_ref, set_bottom_app_bar_height_func, update_thumbnails_func)
    delete_button.tooltip = "Delete the captions.json file"
    delete_button.disabled = not selected_dataset_ref.get("value") # Disable if no dataset selected
    delete_button.update()

    # Restore Add Captions button state (if it was disabled)
    add_button.text = "Add Captions" # Restore original text
    add_button.disabled = False # Re-enable button
    add_button.update()

    processed_progress_bar_ref.visible = False # Hide progress bar
    if processed_output_field_ref.page:
        processed_output_field_ref.value += "\n--- Process Stopped ---\n"
        processed_output_field_ref.update()
        set_bottom_app_bar_height_func() # Adjust height after adding text

    if e.page:
        e.page.snack_bar = ft.SnackBar(content=ft.Text("Captioning process stopped."), open=True)
        e.page.update()


def on_delete_captions_click(e: ft.ControlEvent, thumbnails_grid_control: ft.GridView, selected_dataset_ref, processed_progress_bar_ref, processed_output_field_ref, set_bottom_app_bar_height_func, update_thumbnails_func):
    page_for_dialog = e.page
    button_control = e.control
    current_dataset_name = selected_dataset_ref.get("value")

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
            lambda: perform_delete_captions(page_for_dialog, thumbnails_grid_control, selected_dataset_ref, processed_progress_bar_ref, processed_output_field_ref, set_bottom_app_bar_height_func, update_thumbnails_func)
        )
    finally:
        # Re-enable button after dialog is closed
        if button_control:
             # Only re-enable if a dataset is still selected
            button_control.disabled = not selected_dataset_ref.get("value")
        if page_for_dialog:
            page_for_dialog.update()


def perform_delete_captions(page_context: ft.Page, thumbnails_grid_control: ft.GridView, selected_dataset_ref, processed_progress_bar_ref, processed_output_field_ref, set_bottom_app_bar_height_func, update_thumbnails_func):
    current_dataset_name = selected_dataset_ref.get("value")
    if not current_dataset_name:
        return

    if delete_captions_file(current_dataset_name):
        if page_context:
            page_context.snack_bar = ft.SnackBar(content=ft.Text(f"Deleted captions for {current_dataset_name}."), open=True)
        update_thumbnails_func(page_ctx=page_context, grid_control=thumbnails_grid_control, force_refresh=True) # Force refresh after deleting captions
    else:
        if page_context:
            page_context.snack_bar = ft.SnackBar(content=ft.Text(f"Error deleting captions for {current_dataset_name}."), open=True)
            page_context.update()


async def on_preprocess_dataset_click(e: ft.ControlEvent,
                                model_name_dropdown: ft.Dropdown,
                                bucket_size_textfield: ft.TextField,
                                trigger_word_textfield: ft.TextField,
                                selected_dataset_ref,
                                DATASETS_TYPE_ref,
                                processed_progress_bar_ref,
                                processed_output_field_ref,
                                set_bottom_app_bar_height_func,
                                update_thumbnails_func,
                                thumbnails_grid_control: ft.GridView): # Add this argument
    current_dataset_name = selected_dataset_ref.get("value")
    if not current_dataset_name:
        if e.page:
            e.page.snack_bar = ft.SnackBar(content=ft.Text("Error: No dataset selected for preprocessing."), open=True)
            e.page.update()
        return

    # Clean the dataset name by removing any image markers (both old and new formats)
    clean_dataset_name = current_dataset_name.replace('(img) ', '').replace(' (img)', '')

    # Determine dataset type using the global variable
    dataset_type = DATASETS_TYPE_ref["value"]
    
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
            processed_output_field_ref.value = error_msg + "\n" # Display error in output field too
            processed_output_field_ref.visible = True
            set_bottom_app_bar_height_func()
            processed_output_field_ref.update()
            e.page.update()
        return

    model_name_val = model_name_dropdown.value.strip()
    raw_bucket_str_val = bucket_size_textfield.value.strip()
    trigger_word = trigger_word_textfield.value.strip()

    if not model_name_val or not raw_bucket_str_val:
        error_msg = "Error: Model Name or Bucket Size cannot be empty."
        if e.page:
            e.page.snack_bar = ft.SnackBar(content=ft.Text(error_msg), open=True)
            processed_output_field_ref.value = error_msg + "\n"
            processed_output_field_ref.visible = True
            set_bottom_app_bar_height_func()
            processed_output_field_ref.update()
            e.page.update()
        return

    # Validate and parse bucket size string
    parsed_bucket_list = parse_bucket_string_to_list(raw_bucket_str_val)
    if parsed_bucket_list is None:
        error_msg = f"Error parsing Bucket Size format: '{raw_bucket_str_val}'. Expected '[W, H, F]' or 'WxHxF'."
        if e.page:
             e.page.snack_bar = ft.SnackBar(content=ft.Text(error_msg), open=True)
             processed_output_field_ref.value = error_msg + "\n"
             processed_output_field_ref.visible = True
             set_bottom_app_bar_height_func()
             processed_output_field_ref.update()
             e.page.update()
        return

    # Validate bucket values (divisible by 32, etc.)
    error_messages = validate_bucket_values(*parsed_bucket_list)
    if error_messages:
        error_msg = "Bucket Size validation errors:\n" + "\n".join(error_messages)
        if e.page:
            e.page.snack_bar = ft.SnackBar(content=ft.Text("Bucket Size validation failed. Check output."), open=True)
            processed_output_field_ref.value = error_msg + "\n"
            processed_output_field_ref.visible = True
            set_bottom_app_bar_height_func()
            processed_output_field_ref.update()
            e.page.update()
        return

    # Format validated values for the script
    resolution_buckets_str = f"{parsed_bucket_list[0]}x{parsed_bucket_list[1]}x{parsed_bucket_list[2]}"

    # --- Build the command string using the dedicated helper function ---
    command = build_preprocess_command(
        input_captions_json_path=input_captions_json_path,
        preprocess_output_dir=preprocess_output_dir,
        resolution_buckets_str=resolution_buckets_str,
        model_name_val=model_name_dropdown.value,
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
            processed_progress_bar_ref,
            processed_output_field_ref,
            "Start Preprocess", # Original button text
            set_bottom_app_bar_height_func,
            on_success_callback=lambda: update_thumbnails_func(page_ctx=e.page, grid_control=thumbnails_grid_control, force_refresh=True) # Force refresh after preprocessing
        )

async def apply_affix_from_textfield(e: ft.ControlEvent, affix_type: str, selected_dataset_ref, DATASETS_TYPE_ref, update_thumbnails_func, thumbnails_grid_ref_obj, affix_text_field_ref: ft.Ref[ft.TextField]):
    if not selected_dataset_ref["value"]:
        e.page.snack_bar = ft.SnackBar(content=ft.Text("Please select a dataset first."), open=True)
        if e.page: e.page.update()
        return

    if not affix_text_field_ref.current or not affix_text_field_ref.current.value:
        e.page.snack_bar = ft.SnackBar(content=ft.Text("Affix text cannot be empty. Please enter text in the 'Text' field."), open=True)
        if e.page: e.page.update()
        return
    
    affix_text = affix_text_field_ref.current.value.strip()
    current_dataset_name = selected_dataset_ref["value"]
    
    # Remove the ' (img)' suffix if present for the actual folder name
    clean_dataset_name = current_dataset_name.replace(' (img)', '')
    
    if DATASETS_TYPE_ref["value"] == "image":
        dataset_dir = settings.DATASETS_IMG_DIR
    else:
        dataset_dir = settings.DATASETS_DIR
        
    captions_json_path = os.path.join(dataset_dir, clean_dataset_name, "captions.json")

    if not os.path.exists(captions_json_path):
        # Create an empty captions.json file if it doesn't exist
        try:
            with open(captions_json_path, 'w', encoding='utf-8') as f:
                json.dump([], f, indent=4)
            e.page.snack_bar = ft.SnackBar(content=ft.Text(f"Created empty captions.json for dataset '{current_dataset_name}'."), open=True)
            if e.page: e.page.update()
            captions_data = [] # Initialize captions_data as empty list
        except Exception as ex:
            e.page.snack_bar = ft.SnackBar(content=ft.Text(f"Error creating captions.json: {ex}"), open=True)
            if e.page: e.page.update()
            return # Exit if file creation fails

    try:
        with open(captions_json_path, 'r', encoding='utf-8') as f:
            captions_data = json.load(f)
    except json.JSONDecodeError:
        e.page.snack_bar = ft.SnackBar(content=ft.Text(f"Error reading captions.json for '{current_dataset_name}'. Invalid JSON."), open=True)
        if e.page: e.page.update()
        return
    except Exception as ex:
        e.page.snack_bar = ft.SnackBar(content=ft.Text(f"Error reading captions.json: {ex}"), open=True)
        if e.page: e.page.update()
        return

    if not isinstance(captions_data, list):
        e.page.snack_bar = ft.SnackBar(content=ft.Text(f"Captions data for '{current_dataset_name}' is not a list."), open=True)
        if e.page: e.page.update()
        return

    # --- NEW LOGIC: Check for missing videos/images and add them to captions.json ---
    dataset_folder_path = os.path.join(dataset_dir, clean_dataset_name)
    
    # Get actual media files in the dataset folder
    # get_videos_and_thumbnails returns (video_paths, thumbnail_paths)
    # We only need the video_paths (or image_paths)
    all_media_paths_in_folder, _ = get_videos_and_thumbnails(dataset_folder_path, DATASETS_TYPE_ref["value"])
    
    # Extract base filenames from existing captions_data
    existing_captioned_basenames = {os.path.basename(item["media_path"]) for item in captions_data if isinstance(item, dict) and "media_path" in item}
    
    new_entries_added = False
    for media_full_path in all_media_paths_in_folder:
        media_basename = os.path.basename(media_full_path)
        if media_basename not in existing_captioned_basenames:
            # Add new entry for the missing video/image
            captions_data.append({
                "media_path": media_basename,
                "caption": "" # Default empty caption
            })
            new_entries_added = True
            print(f"Added missing media to captions.json: {media_basename}") # For debugging

    if new_entries_added:
        # Sort the captions_data by media_path after adding new entries
        captions_data.sort(key=lambda x: x.get("media_path", ""))
        try:
            with open(captions_json_path, 'w', encoding='utf-8') as f:
                json.dump(captions_data, f, indent=4)
            e.page.snack_bar = ft.SnackBar(content=ft.Text(f"Updated captions.json with new media entries."), open=True)
            if e.page: e.page.update()
        except Exception as ex:
            e.page.snack_bar = ft.SnackBar(content=ft.Text(f"Error saving updated captions.json: {ex}"), open=True)
            if e.page: e.page.update()
    # --- END NEW LOGIC ---

    selected_files_from_thumbnails = _get_selected_filenames(thumbnails_grid_ref_obj)
    
    captions_to_process = []
    if selected_files_from_thumbnails:
        # Filter captions_data to only include entries for selected files
        for item in captions_data:
            if isinstance(item, dict) and "media_path" in item and \
               os.path.basename(item["media_path"]) in selected_files_from_thumbnails:
                captions_to_process.append(item)
        if not captions_to_process:
            e.page.snack_bar = ft.SnackBar(content=ft.Text(f"No selected files have captions to modify."), open=True)
            if e.page: e.page.update()
            return
    else:
        # If no thumbnails selected, process all captions
        captions_to_process = captions_data
        if not captions_to_process:
            e.page.snack_bar = ft.SnackBar(content=ft.Text(f"Captions.json is empty. No captions to modify."), open=True)
            if e.page: e.page.update()
            return

    modified_count = 0
    for item in captions_to_process:
        if isinstance(item, dict) and "caption" in item and isinstance(item["caption"], str):
            if affix_type == "prefix":
                item["caption"] = f"{affix_text} {item['caption']}"
            elif affix_type == "suffix":
                item["caption"] = f"{item['caption']} {affix_text}"
            modified_count += 1
    
    if modified_count == 0: # This check is now for captions_to_process
        e.page.snack_bar = ft.SnackBar(content=ft.Text(f"No captions were modified. Check format."), open=True)
        if e.page: e.page.update()
        return
    
    # If selected_files_from_thumbnails was used, we need to merge back the modified captions
    if selected_files_from_thumbnails:
        updated_captions_data = []
        processed_media_paths = {os.path.basename(item["media_path"]) for item in captions_to_process if "media_path" in item}
        for original_item in captions_data:
            if isinstance(original_item, dict) and "media_path" in original_item and \
               os.path.basename(original_item["media_path"]) in processed_media_paths:
                # Find the modified version of this item from captions_to_process
                found_modified = False
                for modified_item in captions_to_process:
                    if os.path.basename(modified_item.get("media_path")) == os.path.basename(original_item.get("media_path")):
                        updated_captions_data.append(modified_item)
                        found_modified = True
                        break
                if not found_modified: # Should not happen if logic is correct
                    updated_captions_data.append(original_item)
            else:
                updated_captions_data.append(original_item)
        captions_data = updated_captions_data

    try:
        with open(captions_json_path, 'w', encoding='utf-8') as f:
            json.dump(captions_data, f, indent=4)
        e.page.snack_bar = ft.SnackBar(content=ft.Text(f"Successfully {affix_type}ed to {modified_count} captions."), open=True)
        if affix_text_field_ref.current:
            affix_text_field_ref.current.value = ""
        
        if asyncio.iscoroutinefunction(update_thumbnails_func):
            await update_thumbnails_func(e.page, thumbnails_grid_ref_obj.current, force_refresh=True)
        else:
            update_thumbnails_func(e.page, thumbnails_grid_ref_obj.current, force_refresh=True)
            
    except Exception as ex:
        e.page.snack_bar = ft.SnackBar(content=ft.Text(f"Error saving captions: {ex}"), open=True)
    finally:
        if e.page: e.page.update()

async def find_and_replace_in_captions(e: ft.ControlEvent, selected_dataset_ref, DATASETS_TYPE_ref, find_text_field_ref: ft.Ref[ft.TextField], replace_text_field_ref: ft.Ref[ft.TextField], update_thumbnails_func, thumbnails_grid_ref_obj):
    if not selected_dataset_ref["value"]:
        e.page.snack_bar = ft.SnackBar(content=ft.Text("Please select a dataset first."), open=True)
        if e.page: e.page.update()
        return

    if not find_text_field_ref.current or not find_text_field_ref.current.value:
        e.page.snack_bar = ft.SnackBar(content=ft.Text("Find text cannot be empty."), open=True)
        if e.page: e.page.update()
        return
    
    find_text = find_text_field_ref.current.value
    replace_text = replace_text_field_ref.current.value if replace_text_field_ref.current and replace_text_field_ref.current.value is not None else ""
    current_dataset_name = selected_dataset_ref["value"]
    
    # Remove the ' (img)' suffix if present for the actual folder name
    clean_dataset_name = current_dataset_name.replace(' (img)', '')
    
    if DATASETS_TYPE_ref["value"] == "image":
        dataset_dir = settings.DATASETS_IMG_DIR
    else:
        dataset_dir = settings.DATASETS_DIR
        
    captions_json_path = os.path.join(dataset_dir, clean_dataset_name, "captions.json")

    if not os.path.exists(captions_json_path):
        e.page.snack_bar = ft.SnackBar(content=ft.Text(f"No captions.json found for '{current_dataset_name}'."), open=True)
        if e.page: e.page.update()
        return

    try:
        with open(captions_json_path, 'r', encoding='utf-8') as f:
            captions_data = json.load(f)
    except json.JSONDecodeError:
        e.page.snack_bar = ft.SnackBar(content=ft.Text(f"Error reading captions.json for '{current_dataset_name}'. Invalid JSON."), open=True)
        if e.page: e.page.update()
        return
    except Exception as ex:
        e.page.snack_bar = ft.SnackBar(content=ft.Text(f"Error reading captions.json: {ex}"), open=True)
        if e.page: e.page.update()
        return

    if not isinstance(captions_data, list):
        e.page.snack_bar = ft.SnackBar(content=ft.Text(f"Captions data for '{current_dataset_name}' is not a list."), open=True)
        if e.page: e.page.update()
        return

    selected_files_from_thumbnails = _get_selected_filenames(thumbnails_grid_ref_obj)
    
    captions_to_process = []
    if selected_files_from_thumbnails:
        # Filter captions_data to only include entries for selected files
        for item in captions_data:
            if isinstance(item, dict) and "media_path" in item and \
               os.path.basename(item["media_path"]) in selected_files_from_thumbnails:
                captions_to_process.append(item)
        if not captions_to_process:
            e.page.snack_bar = ft.SnackBar(content=ft.Text(f"No selected files have captions to modify."), open=True)
            if e.page: e.page.update()
            return
    else:
        # If no thumbnails selected, process all captions
        captions_to_process = captions_data
        if not captions_to_process:
            e.page.snack_bar = ft.SnackBar(content=ft.Text(f"Captions.json is empty. No changes made."), open=True)
            if e.page: e.page.update()
            return

    modified_count = 0
    replacements_made = 0
    for item in captions_to_process:
        if isinstance(item, dict) and "caption" in item and isinstance(item["caption"], str):
            original_caption = item["caption"]
            item["caption"] = original_caption.replace(find_text, replace_text)
            if original_caption != item["caption"]:
                modified_count += 1
                replacements_made += original_caption.count(find_text) if find_text else 0
    
    if modified_count == 0: # This check is now for captions_to_process
        e.page.snack_bar = ft.SnackBar(content=ft.Text(f"Text '{find_text}' not found. No changes made."), open=True)
        if e.page: e.page.update()
        return
    
    # If selected_files_from_thumbnails was used, we need to merge back the modified captions
    if selected_files_from_thumbnails:
        updated_captions_data = []
        processed_media_paths = {os.path.basename(item["media_path"]) for item in captions_to_process if "media_path" in item}
        for original_item in captions_data:
            if isinstance(original_item, dict) and "media_path" in original_item and \
               os.path.basename(original_item["media_path"]) in processed_media_paths:
                # Find the modified version of this item from captions_to_process
                found_modified = False
                for modified_item in captions_to_process:
                    if os.path.basename(modified_item.get("media_path")) == os.path.basename(original_item.get("media_path")):
                        updated_captions_data.append(modified_item)
                        found_modified = True
                        break
                if not found_modified: # Should not happen if logic is correct
                    updated_captions_data.append(original_item)
            else:
                updated_captions_data.append(original_item)
        captions_data = updated_captions_data

    try:
        with open(captions_json_path, 'w', encoding='utf-8') as f:
            json.dump(captions_data, f, indent=4)
        e.page.snack_bar = ft.SnackBar(content=ft.Text(f"Made {replacements_made} replacement(s) in {modified_count} caption(s)."), open=True)
        if find_text_field_ref.current: find_text_field_ref.current.value = ""
        if replace_text_field_ref.current: replace_text_field_ref.current.value = ""
        
        if asyncio.iscoroutinefunction(update_thumbnails_func):
            await update_thumbnails_func(e.page, thumbnails_grid_ref_obj.current, force_refresh=True)
        else:
            update_thumbnails_func(e.page, thumbnails_grid_ref_obj.current, force_refresh=True)
            
    except Exception as ex:
        e.page.snack_bar = ft.SnackBar(content=ft.Text(f"Error saving captions: {ex}"), open=True)
    finally:
        if e.page: e.page.update()
