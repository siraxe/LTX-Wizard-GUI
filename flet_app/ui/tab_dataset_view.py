import flet as ft
import os
import json
import asyncio

from ui.popups.video_player_dialog import open_video_captions_dialog
from ui.popups.delete_caption_dialog import show_delete_caption_dialog
from ui._styles import create_dropdown, create_styled_button, create_textfield, BTN_STYLE2
from ui.utils.utils_datasets import (
    get_dataset_folders,
    get_videos_and_thumbnails,
    DATASETS_DIR, 
    THUMB_TARGET_W, 
    THUMB_TARGET_H,
    DEFAULT_BUCKET_SIZE_STR,
    DEFAULT_MODEL_NAME,
)

# =====================
# Data/Utility Functions
# =====================

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
    return DEFAULT_BUCKET_SIZE_STR

def load_dataset_config(dataset_name: str | None) -> tuple[str, str, str]:
    """Load bucket size, model name, and trigger word from dataset info.json, or return defaults."""
    bucket_to_set = DEFAULT_BUCKET_SIZE_STR
    model_to_set = DEFAULT_MODEL_NAME
    trigger_word_to_set = ''
    if dataset_name:
        dataset_info_json_path = os.path.join(DATASETS_DIR, dataset_name, "info.json")
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
    dataset_info_json_path = os.path.join(DATASETS_DIR, dataset_name, "info.json")
    parsed_bucket_list = parse_bucket_string_to_list(bucket_str)
    if parsed_bucket_list is None:
        parsed_bucket_list = parse_bucket_string_to_list(DEFAULT_BUCKET_SIZE_STR)
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

def load_processed_map(dataset_name: str):
    """Load processed.json for a dataset, or return None."""
    processed_json_path = os.path.join(DATASETS_DIR, dataset_name, "preprocessed_data", "processed.json")
    if os.path.exists(processed_json_path):
        try:
            with open(processed_json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def load_dataset_captions(dataset_name: str):
    """Load captions.json for a dataset, or return empty list."""
    dataset_captions_json_path = os.path.join(DATASETS_DIR, dataset_name, "captions.json")
    if os.path.exists(dataset_captions_json_path):
        try:
            with open(dataset_captions_json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return []
    return []

def delete_captions_file(dataset_name: str) -> bool:
    """Delete captions.json for a dataset."""
    captions_file_path = os.path.join(DATASETS_DIR, dataset_name, "captions.json")
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
    if W_val is None or W_val % 32 != 0:
        errors.append(f"Width ({W_val}) must be div by 32.")
    if H_val is None or H_val % 32 != 0:
        errors.append(f"Height ({H_val}) must be div by 32.")
    if F_val is None or not (F_val >= 5 and (F_val - 1) % 4 == 0):
        errors.append(f"Frames ({F_val}) invalid (must be >= 5 and 4n+1).")
    return errors

# =====================
# GUI Control Creation
# =====================

# Global controls (used in multiple places)
bucket_size_textfield = create_textfield(
    label="Bucket Size (e.g., [W, H, F] or WxHxF)",
    value=DEFAULT_BUCKET_SIZE_STR,
    expand=True
)

rename_textfield = create_textfield(
    label="Rename all files",
    value="",
    hint_text="Name of videos + _num will be added",
    expand=True,
)

MODEL_NAME_CHOICES = [
    "LTXV_2B_0.9.0",
    "LTXV_2B_0.9.1",
    "LTXV_2B_0.9.5",
    "LTXV_2B_0.9.6_DEV",
    "LTXV_2B_0.9.6_DISTILLED",
    "LTXV_13B_097_DEV",
    "LTXV_13B_097_DEV_FP8",
    "LTXV_13B_097_DISTILLED",
    "LTXV_13B_097_DISTILLED_FP8",
]

model_name_dropdown = create_dropdown(
    "Model Name",
    DEFAULT_MODEL_NAME,
    {name: name for name in MODEL_NAME_CHOICES},
    "Select model",
    expand=True
)

trigger_word_textfield = create_textfield(
    "Trigger WORD", "", col=9, expand=True, hint_text="e.g. 'CAKEIFY' , leave empty for none"
)

# =====================
# GUI Event Handlers
# =====================

def on_rename_files_click(e: ft.ControlEvent):
    """
    Rename all video files in the selected dataset according to the rename_textfield value,
    appending _01, _02, etc. Update captions.json and info.json if they exist.
    Provide user feedback via snackbar.
    """
    current_dataset_name = selected_dataset.get("value")
    if not current_dataset_name:
        if e.page and e.page.client_storage:
            e.page.snack_bar = ft.SnackBar(content=ft.Text("Error: No dataset selected for renaming."), open=True)
            e.page.update()
        return

    base_name = rename_textfield.value.strip()
    if not base_name:
        if e.page and e.page.client_storage:
            e.page.snack_bar = ft.SnackBar(content=ft.Text("Error: Enter a base name in the rename field."), open=True)
            e.page.update()
        return

    dataset_folder_path = os.path.abspath(os.path.join(DATASETS_DIR, current_dataset_name))
    video_exts = ['.mp4', '.mov', '.avi', '.webm', '.mkv']
    video_files = [f for f in os.listdir(dataset_folder_path) if os.path.splitext(f)[1].lower() in video_exts]
    video_files.sort()  # Ensure consistent order
    if not video_files:
        if e.page and e.page.client_storage:
            e.page.snack_bar = ft.SnackBar(content=ft.Text("No video files found to rename."), open=True)
            e.page.update()
        return

    # Prepare new names and check for collisions
    new_names = []
    for idx, old_name in enumerate(video_files, 1):
        ext = os.path.splitext(old_name)[1]
        new_name = f"{base_name}_{idx:02d}{ext}"
        new_names.append(new_name)
    if len(set(new_names)) != len(new_names):
        if e.page and e.page.client_storage:
            e.page.snack_bar = ft.SnackBar(content=ft.Text("Naming collision detected. Aborting."), open=True)
            e.page.update()
        return
    # Ensure no existing file will be overwritten
    for new_name in new_names:
        if new_name in video_files:
            if e.page and e.page.client_storage:
                e.page.snack_bar = ft.SnackBar(content=ft.Text(f"File {new_name} already exists. Aborting."), open=True)
                e.page.update()
            return

    # Rename files
    old_to_new = {}
    for old_name, new_name in zip(video_files, new_names):
        old_path = os.path.join(dataset_folder_path, old_name)
        new_path = os.path.join(dataset_folder_path, new_name)
        try:
            os.rename(old_path, new_path)
            old_to_new[old_name] = new_name
        except Exception as ex:
            if e.page and e.page.client_storage:
                e.page.snack_bar = ft.SnackBar(content=ft.Text(f"Failed to rename {old_name}: {ex}"), open=True)
                e.page.update()
            return

    # Update captions.json if exists
    captions_path = os.path.join(dataset_folder_path, "captions.json")
    if os.path.exists(captions_path):
        try:
            with open(captions_path, "r", encoding="utf-8") as f:
                captions_data = json.load(f)
            changed = False
            # Update filename fields in-place (never duplicate entries)
            for entry in captions_data:
                for field in ("media_path", "video"):
                    if field in entry and entry[field] in old_to_new:
                        entry[field] = old_to_new[entry[field]]
                        changed = True
            if changed:
                with open(captions_path, "w", encoding="utf-8") as f:
                    json.dump(captions_data, f, indent=2, ensure_ascii=False)
        except Exception as ex:
            if e.page and e.page.client_storage:
                e.page.snack_bar = ft.SnackBar(content=ft.Text(f"Failed to update captions.json: {ex}"), open=True)
                e.page.update()
            return

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
            else:
                # If not a dict, fallback to recursive update (legacy)
                def update_dict(d):
                    nonlocal changed
                    for k, v in d.items():
                        if isinstance(v, str) and v in old_to_new:
                            d[k] = old_to_new[v]
                            changed = True
                        elif isinstance(v, list):
                            for i, item in enumerate(v):
                                if isinstance(item, str) and item in old_to_new:
                                    v[i] = old_to_new[item]
                                    changed = True
                                elif isinstance(item, dict):
                                    update_dict(item)
                        elif isinstance(v, dict):
                            update_dict(v)
                update_dict(info_data)
            if changed:
                with open(info_path, "w", encoding="utf-8") as f:
                    json.dump(info_data, f, indent=4, ensure_ascii=False)
        except Exception as ex:
            if e.page and e.page.client_storage:
                e.page.snack_bar = ft.SnackBar(content=ft.Text(f"Failed to update info.json: {ex}"), open=True)
                e.page.update()
            return

    # Success feedback and UI update
    if e.page and e.page.client_storage:
        e.page.snack_bar = ft.SnackBar(content=ft.Text(f"Renamed {len(video_files)} files successfully."), open=True)
        update_thumbnails(page_ctx=e.page, grid_control=thumbnails_grid_ref.current)
        e.page.update()

def on_bucket_or_model_change(e: ft.ControlEvent):
    """Handle changes to bucket size or model name controls."""
    current_dataset_name = selected_dataset.get("value")
    if not current_dataset_name:
        if e.page:
            e.page.snack_bar = ft.SnackBar(
                content=ft.Text("No dataset selected. Configuration not saved."),
                open=True
            )
            e.page.update()
        return
    success = save_dataset_config(
        current_dataset_name,
        bucket_size_textfield.value,
        model_name_dropdown.value,
        trigger_word_textfield.value
    )
    if e.page:
        if success:
            e.page.snack_bar = ft.SnackBar(
                content=ft.Text(f"Config saved: {current_dataset_name}{os.sep}info.json."),
                open=True
            )
        else:
            e.page.snack_bar = ft.SnackBar(
                content=ft.Text(f"Error saving config for {current_dataset_name}"),
                open=True
            )
        e.page.update()

def on_dataset_dropdown_change(ev: ft.ControlEvent, thumbnails_grid_control: ft.GridView):
    """Handle dataset dropdown selection change."""
    if processed_output_field.page:
        processed_output_field.visible = False
        processed_output_field.update()
        set_bottom_app_bar_height()
    if processed_progress_bar.page:
        processed_progress_bar.visible = False
        processed_progress_bar.update()
    selected_dataset["value"] = ev.control.value
    bucket_val, model_val, trigger_word_val = load_dataset_config(selected_dataset["value"])
    bucket_size_textfield.value = bucket_val
    model_name_dropdown.value = model_val if model_val in MODEL_NAME_CHOICES else DEFAULT_MODEL_NAME
    trigger_word_textfield.value = trigger_word_val or ''
    if bucket_size_textfield.page:
        bucket_size_textfield.update()
    if model_name_dropdown.page:
        model_name_dropdown.update()
    if trigger_word_textfield.page:
        trigger_word_textfield.update()
    update_thumbnails(page_ctx=ev.page, grid_control=thumbnails_grid_control)
    if ev.page:
        ev.page.update()

def on_update_button_click(e: ft.ControlEvent, dataset_dropdown_control, thumbnails_grid_control):
    reload_current_dataset(e.page, dataset_dropdown_control, thumbnails_grid_control)

def on_add_captions_click(e: ft.ControlEvent, thumbnails_grid_control: ft.GridView):
    current_dataset_name = selected_dataset.get("value")
    if not current_dataset_name:
        if e.page and e.page.client_storage:
            e.page.snack_bar = ft.SnackBar(content=ft.Text("Error: No dataset selected."), open=True)
            e.page.update()
        return
    dataset_folder_path = os.path.abspath(os.path.join(DATASETS_DIR, current_dataset_name))
    output_json_path = os.path.join(dataset_folder_path, "captions.json")
    python_exe = os.path.normpath(os.path.join("venv", "Scripts", "python.exe"))
    script_file = os.path.normpath("scripts/caption_videos.py")
    command = f'"{python_exe}" "{script_file}" "{dataset_folder_path}/" --output "{output_json_path}" --captioner-type llava_next_7b'
    if e.page:
        e.page.run_task(run_dataset_script_command, command, e.page, e.control, processed_progress_bar, processed_output_field, "Add Captions", on_success_callback=lambda: update_thumbnails(page_ctx=e.page, grid_control=thumbnails_grid_control))

def on_delete_captions_click(e: ft.ControlEvent, thumbnails_grid_control: ft.GridView):
    page_for_dialog = e.page
    button_control = e.control
    current_dataset_name = selected_dataset.get("value")
    if not current_dataset_name:
        if page_for_dialog and page_for_dialog.client_storage:
            page_for_dialog.snack_bar = ft.SnackBar(content=ft.Text("No dataset selected."), open=True)
            page_for_dialog.update()
        return
    captions_file_path = os.path.join(DATASETS_DIR, current_dataset_name, "captions.json")
    if not os.path.exists(captions_file_path):
        if page_for_dialog and page_for_dialog.client_storage:
            page_for_dialog.snack_bar = ft.SnackBar(content=ft.Text(f"Captions for '{current_dataset_name}' not found."), open=True)
            page_for_dialog.update()
        return
    if button_control:
        button_control.disabled = True
    if page_for_dialog and page_for_dialog.client_storage:
        page_for_dialog.update()
    try:
        show_delete_caption_dialog(page_for_dialog, current_dataset_name, lambda: perform_delete_captions(page_for_dialog, thumbnails_grid_control))
    finally:
        if button_control:
            button_control.disabled = False
        if page_for_dialog and page_for_dialog.client_storage:
            page_for_dialog.update()

def perform_delete_captions(page_context: ft.Page, thumbnails_grid_control: ft.GridView):
    current_dataset_name = selected_dataset.get("value")
    if not current_dataset_name:
        return
    if delete_captions_file(current_dataset_name):
        if page_context and page_context.client_storage:
            page_context.snack_bar = ft.SnackBar(content=ft.Text(f"Deleted captions for {current_dataset_name}."), open=True)
        update_thumbnails(page_ctx=page_context, grid_control=thumbnails_grid_control)
    else:
        if page_context and page_context.client_storage:
            page_context.snack_bar = ft.SnackBar(content=ft.Text(f"Error deleting captions."), open=True)
            page_context.update()

def on_preprocess_dataset_click(e: ft.ControlEvent):
    current_dataset_name = selected_dataset.get("value")
    if not current_dataset_name:
        if e.page and e.page.client_storage:
            e.page.snack_bar = ft.SnackBar(content=ft.Text("Error: No dataset selected for preprocessing."), open=True)
            e.page.update()
        return
    input_captions_json_path = os.path.abspath(os.path.join(DATASETS_DIR, current_dataset_name, "captions.json"))
    preprocess_output_dir = os.path.abspath(os.path.join(DATASETS_DIR, current_dataset_name, "preprocessed_data"))
    if not os.path.exists(input_captions_json_path):
        if e.page and e.page.client_storage:
            e.page.snack_bar = ft.SnackBar(content=ft.Text(f"Error: Preprocessing input file not found: {input_captions_json_path}"), open=True)
            e.page.update()
        return
    python_exe = os.path.normpath(os.path.join("venv", "Scripts", "python.exe"))
    script_file = os.path.normpath("scripts/preprocess_dataset.py")
    model_name_val = model_name_dropdown.value.strip()
    raw_bucket_str_val = bucket_size_textfield.value.strip()
    if not model_name_val or not raw_bucket_str_val:
        if e.page and e.page.client_storage:
            e.page.snack_bar = ft.SnackBar(content=ft.Text("Error: Model Name or Bucket Size cannot be empty."), open=True)
            e.page.update()
        return
    parsed_list_for_script = parse_bucket_string_to_list(raw_bucket_str_val)
    if parsed_list_for_script:
        W_val, H_val, F_val = parsed_list_for_script
    else:
        try:
            if raw_bucket_str_val.startswith('[') and raw_bucket_str_val.endswith(']'):
                parsed_list_detailed = json.loads(raw_bucket_str_val)
                if not (isinstance(parsed_list_detailed, list) and len(parsed_list_detailed) == 3 and all(isinstance(i, int) for i in parsed_list_detailed)):
                    raise ValueError("Invalid list format")
                W_val, H_val, F_val = parsed_list_detailed
            elif 'x' in raw_bucket_str_val.lower():
                parts = [p.strip() for p in raw_bucket_str_val.lower().split('x') if p.strip()]
                if not (len(parts) == 3 and all(part.isdigit() for part in parts)):
                    raise ValueError("Invalid WxHxF format")
                W_val, H_val, F_val = int(parts[0]), int(parts[1]), int(parts[2])
            else:
                raise ValueError("Unrecognized format")
        except Exception as ex_parse:
            if e.page and e.page.client_storage:
                error_msg = f"Error parsing Bucket Size '{raw_bucket_str_val}': {ex_parse}"
                e.page.snack_bar = ft.SnackBar(content=ft.Text(error_msg), open=True)
                processed_output_field.value = error_msg
                processed_output_field.visible = True
                set_bottom_app_bar_height()
                processed_output_field.update()
                e.page.update()
            return
    error_messages = validate_bucket_values(W_val, H_val, F_val)
    if error_messages:
        if e.page and e.page.client_storage:
            error_msg = "Bucket Size errors: " + " ".join(error_messages)
            e.page.snack_bar = ft.SnackBar(content=ft.Text(error_msg), open=True)
            processed_output_field.value = error_msg
            processed_output_field.visible = True
            set_bottom_app_bar_height()
            processed_output_field.update()
            e.page.update()
        return
    resolution_buckets_str = f"{W_val}x{H_val}x{F_val}"
    trigger_word = trigger_word_textfield.value.strip()
    command = (f'"{python_exe}" "{script_file}" "{input_captions_json_path}" --output-dir "{preprocess_output_dir}" '
               f'--resolution-buckets "{resolution_buckets_str}" --caption-column "caption" '
               f'--video-column "media_path" --model-source "{model_name_val}"')
    if trigger_word:
        command += f' --id-token "{trigger_word}"'
    if e.page:
        e.page.run_task(
            run_dataset_script_command,
            command,
            e.page,
            e.control,
            processed_progress_bar,
            processed_output_field,
            "Preprocess Dataset",
            on_success_callback=lambda: update_thumbnails(page_ctx=e.page, grid_control=thumbnails_grid_ref.current)
        )

# =====================
# GUI Update/Utility Functions
# =====================

def update_thumbnails(page_ctx: ft.Page | None, grid_control: ft.GridView | None, force_refresh: bool = False):
    """Update the thumbnails grid for the selected dataset. If force_refresh is True, use cache-busting temp files."""
    if not grid_control:
        return
    current_selection = selected_dataset["value"]
    grid_control.controls.clear()
    processed_map = load_processed_map(current_selection) if current_selection else None
    if not current_selection:
        folders_exist = get_dataset_folders()
        grid_control.controls.append(ft.Text("Select a dataset to view videos." if folders_exist else "No datasets found."))
    else:
        thumbnail_paths_map, video_info = get_videos_and_thumbnails(current_selection)
        video_files_list["value"] = list(thumbnail_paths_map.keys())
        dataset_captions = load_dataset_captions(current_selection)
        if not thumbnail_paths_map:
            grid_control.controls.append(ft.Text("No videos found in this dataset."))
        else:
            for video_path, thumb_path in thumbnail_paths_map.items():
                video_name = os.path.basename(video_path)
                info = video_info.get(video_name, {})
                width, height, frames = info.get("width", "?"), info.get("height", "?"), info.get("frames", "?")
                has_caption = any(entry.get("media_path") == video_name and entry.get("caption", "").strip() for entry in dataset_captions)
                cap_val, cap_color = ("yes", ft.Colors.GREEN) if has_caption else ("no", ft.Colors.RED)
                proc_val, proc_color = ("yes", ft.Colors.GREEN) if processed_map and video_name in processed_map else ("no", ft.Colors.RED)
                grid_control.controls.append(
                    ft.Container(
                        content=ft.Column([
                            # Only use cache-busting temp files if force_refresh is True
                            ft.Image(
                                src=(
                                    (
                                        __import__('shutil').copy2(thumb_path, thumb_path + f'.tmp_{int(__import__("time").time())}.jpg')
                                        or thumb_path + f'.tmp_{int(__import__("time").time())}.jpg'
                                    ) if force_refresh and thumb_path and os.path.exists(thumb_path) else thumb_path
                                ),
                                width=THUMB_TARGET_W,
                                height=THUMB_TARGET_H,
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
                        data=video_path,
                        on_click=lambda e_click, vp=video_path, current_grid=grid_control: open_video_captions_dialog(
                            page_ctx, 
                            vp, 
                            video_files_list["value"],
                            on_caption_updated_callback=lambda: update_thumbnails(page_ctx=page_ctx, grid_control=current_grid)
                        ) if page_ctx else None, 
                        tooltip=video_name, width=THUMB_TARGET_W + 10, height=THUMB_TARGET_H + 45,
                        padding=5, border=ft.border.all(1, ft.Colors.OUTLINE), border_radius=ft.border_radius.all(5)
                    )
                )
    if grid_control and grid_control.page:
        grid_control.update()
    if page_ctx and page_ctx.client_storage:
        page_ctx.update()

def update_dataset_dropdown(p_page: ft.Page | None, current_dataset_dropdown: ft.Dropdown, current_thumbnails_grid: ft.GridView):
    """Update the dataset dropdown and thumbnails grid."""
    folders = get_dataset_folders()
    current_dataset_dropdown.options = [ft.dropdown.Option(name) for name in folders.keys()] if folders else []
    current_dataset_dropdown.value = None
    current_dataset_dropdown.disabled = len(folders) == 0
    selected_dataset["value"] = None
    bucket_val, model_val, trigger_word_val = load_dataset_config(None)
    bucket_size_textfield.value = bucket_val
    model_name_dropdown.value = model_val
    trigger_word_textfield.value = trigger_word_val or ''
    if bucket_size_textfield.page:
        bucket_size_textfield.update()
    if model_name_dropdown.page:
        model_name_dropdown.update()
    if trigger_word_textfield.page:
        trigger_word_textfield.update()
    update_thumbnails(page_ctx=p_page, grid_control=current_thumbnails_grid)
    if current_dataset_dropdown.page:
        current_dataset_dropdown.update()
    if p_page and p_page.client_storage:
        p_page.snack_bar = ft.SnackBar(ft.Text("Dataset list updated! Select a dataset."))
        p_page.snack_bar.open = True
        p_page.update()

def reload_current_dataset(p_page: ft.Page | None, current_dataset_dropdown: ft.Dropdown, current_thumbnails_grid: ft.GridView):
    if processed_output_field.page:
        processed_output_field.visible = False
        processed_output_field.update()
        set_bottom_app_bar_height()
    if processed_progress_bar.page:
        processed_progress_bar.visible = False
        processed_progress_bar.update()
    folders = get_dataset_folders()
    current_dataset_dropdown.options = [ft.dropdown.Option(name) for name in folders.keys()] if folders else []
    current_dataset_dropdown.disabled = len(folders) == 0
    prev_selected = selected_dataset["value"]
    if prev_selected and prev_selected in folders:
        current_dataset_dropdown.value = prev_selected
        selected_dataset["value"] = prev_selected
        bucket_val, model_val, trigger_word_val = load_dataset_config(prev_selected)
        bucket_size_textfield.value = bucket_val
        model_name_dropdown.value = model_val if model_val in MODEL_NAME_CHOICES else DEFAULT_MODEL_NAME
        trigger_word_textfield.value = trigger_word_val or ''
    else:
        current_dataset_dropdown.value = None
        selected_dataset["value"] = None
        bucket_val, model_val, trigger_word_val = load_dataset_config(None)
        bucket_size_textfield.value = bucket_val
        model_name_dropdown.value = model_val
        trigger_word_textfield.value = trigger_word_val or ''
    if bucket_size_textfield.page:
        bucket_size_textfield.update()
    if model_name_dropdown.page:
        model_name_dropdown.update()
    if trigger_word_textfield.page:
        trigger_word_textfield.update()
    if current_dataset_dropdown.page:
        current_dataset_dropdown.update()
    if p_page and p_page.client_storage:
        p_page.update()
    update_thumbnails(page_ctx=p_page, grid_control=current_thumbnails_grid)
    if not (prev_selected and prev_selected in folders):
        if p_page and p_page.client_storage:
            p_page.snack_bar = ft.SnackBar(ft.Text("Dataset list reloaded. Select a dataset if previous one is gone."))
            p_page.snack_bar.open = True

# =====================
# Async Task Runner & Process State
# =====================

current_caption_process = {"proc": None}  # Track running process for stop functionality

async def run_dataset_script_command(command_str: str, page_ref: ft.Page, button_ref: ft.ElevatedButton, progress_bar_ref: ft.ProgressBar, output_field_ref: ft.TextField, original_button_text: str, on_success_callback=None):
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
        button_ref.text = f"{original_button_text.replace('Add', 'Adding')}..."
        if page_ref.client_storage:
            page_ref.update()
        process = await asyncio.create_subprocess_shell(
            command_str,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT
        )
        current_caption_process["proc"] = process
        assert process.stdout is not None
        async for line in process.stdout:
            append_output(line.decode(errors='replace'))
        rc = await process.wait()
        current_caption_process["proc"] = None
        if rc == 0:
            if page_ref.client_storage:
                page_ref.snack_bar = ft.SnackBar(content=ft.Text(f"Script successful: {output_field_ref.value[:50]}..."), open=True)
            if on_success_callback:
                on_success_callback()
        else:
            err_msg = f"Script failed (code {rc})"
            if page_ref.client_storage:
                page_ref.snack_bar = ft.SnackBar(content=ft.Text(err_msg), open=True)
    except Exception as e:
        output_field_ref.value = f"Cmd failed: {e}"
        if page_ref.client_storage:
            page_ref.snack_bar = ft.SnackBar(content=ft.Text(f"Cmd failed: {e}"), open=True)
    finally:
        current_caption_process["proc"] = None
        button_ref.text = original_button_text
        button_ref.disabled = False
        progress_bar_ref.visible = False
        if page_ref.client_storage:
            page_ref.update()


# =====================
# Global State
# =====================

folders = get_dataset_folders()
selected_dataset = {"value": None}
video_files_list = {"value": []}
thumbnails_grid_ref = ft.Ref[ft.GridView]()
processed_progress_bar = ft.ProgressBar(visible=False)
processed_output_field = ft.TextField(
    label="Processed Output", text_size=10, multiline=True, read_only=True, 
    visible=False, min_lines=6, max_lines=15, expand=True)
bottom_app_bar_ref = None

def set_bottom_app_bar_height():
    global bottom_app_bar_ref
    if bottom_app_bar_ref is not None:
        if processed_output_field.visible:
            bottom_app_bar_ref.height = 240
        else:
            bottom_app_bar_ref.height = 0
        if bottom_app_bar_ref.page:
            bottom_app_bar_ref.update()

# =====================
# Main GUI Layout Builder
# =====================

def dataset_tab_layout(page=None):
    """Build the main dataset tab layout."""
    global bottom_app_bar_ref
    p_page = page
    dataset_dropdown_control = create_dropdown(
        "Select dataset",
        selected_dataset["value"],
        folders,
        "Select your dataset",
        expand=True
    )
    thumbnails_grid_control = ft.GridView(
        ref=thumbnails_grid_ref,
        runs_count=5, max_extent=THUMB_TARGET_W + 20,
        child_aspect_ratio=(THUMB_TARGET_W + 10) / (THUMB_TARGET_H + 80),
        spacing=7, run_spacing=7, controls=[], expand=True
    )
    dataset_dropdown_control.on_change = lambda ev: on_dataset_dropdown_change(ev, thumbnails_grid_control)
    update_button_control = ft.IconButton(
        icon=ft.Icons.REFRESH, tooltip="Update dataset list",
        on_click=lambda e: on_update_button_click(e, dataset_dropdown_control, thumbnails_grid_control),
        style=ft.ButtonStyle(padding=ft.padding.symmetric(horizontal=8)), icon_size=20
    )
    # Dropdown for selecting captioning model
    CAPTION_MODEL_CHOICES = {
        "qwen_25_vl": "Qwen2.5-VL-7B",
        "llava_next_7b": "LLaVA-NeXT-7B"
    }
    caption_model_dropdown = create_dropdown(
        "Captioning Model", "llava_next_7b",  # default
        CAPTION_MODEL_CHOICES,
        "Select a captioning model",
        expand=True,col=9,
    )
    captions_checkbox = ft.Checkbox(
            label="8-bit", value=True,scale=1,
            visible=False,
            left="left",
            expand=True,
    )
    captions_checkbox_c = ft.Container(captions_checkbox,
            expand=True,col=3,scale=0.8,
            alignment=ft.alignment.bottom_center,    # Align to bottom
            margin=ft.margin.only(top=10)
    )


    def on_caption_model_change(ev):
        if caption_model_dropdown.value == "qwen_25_vl":
            captions_checkbox.visible = True
        else:
            captions_checkbox.visible = False
        if hasattr(ev, 'page') and ev.page:
            ev.page.update()

    caption_model_dropdown.on_change = on_caption_model_change

    import signal

    def stop_captioning(e, button, thumbnails_grid_control):
        # Try to kill by PID file
        pid_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../scripts/caption_pid.txt')
        killed = False
        if os.path.exists(pid_file):
            try:
                with open(pid_file, 'r') as f:
                    pid = int(f.read().strip())
                # Try to kill the process
                os.kill(pid, signal.SIGTERM)
                killed = True
            except Exception:
                pass
            try:
                os.remove(pid_file)
            except Exception:
                pass
        # Fallback to killing the tracked process tree if PID file failed
        if not killed:
            proc = current_caption_process.get("proc")
            if proc is not None and proc.returncode is None:
                try:
                    proc.terminate()
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                current_caption_process["proc"] = None
        # Restore Delete button
        dataset_delete_captions_button_control.text = "Delete"
        dataset_delete_captions_button_control.on_click = lambda evt: on_delete_captions_click(evt, thumbnails_grid_control)
        dataset_delete_captions_button_control.tooltip = "Delete the captions.json file"
        dataset_delete_captions_button_control.disabled = False
        dataset_delete_captions_button_control.update()
        processed_progress_bar.visible = False
        processed_output_field.value += "Stopped\n"
        processed_output_field.update()
        if e.page:
            e.page.snack_bar = ft.SnackBar(content=ft.Text("Captioning stopped."), open=True)
            e.page.update()

    def on_add_captions_click_with_model(e, thumbnails_grid_control=thumbnails_grid_control):
        # If a process is running, treat as stop
        proc = current_caption_process.get("proc")
        if proc is not None and proc.returncode is None:
            stop_captioning(e, dataset_add_captions_button_control, thumbnails_grid_control)
            return
        selected_model = caption_model_dropdown.value or "qwen_25_vl"
        current_dataset_name = selected_dataset.get("value")
        if not current_dataset_name:
            if e.page and e.page.client_storage:
                e.page.snack_bar = ft.SnackBar(content=ft.Text("Error: No dataset selected."), open=True)
                e.page.update()
            return
        dataset_folder_path = os.path.abspath(os.path.join(DATASETS_DIR, current_dataset_name))
        output_json_path = os.path.join(dataset_folder_path, "captions.json")
        python_exe = os.path.normpath(os.path.join("venv", "Scripts", "python.exe"))
        script_file = os.path.normpath("scripts/caption_videos.py")
        command = f'"{python_exe}" "{script_file}" "{dataset_folder_path}/" --output "{output_json_path}" --captioner-type {selected_model}'
        # Add --use-8bit if Qwen model and checkbox is checked
        if selected_model == "qwen_25_vl" and captions_checkbox.value:
            command += " --use-8bit"
        # Change Delete button to Stop
        dataset_delete_captions_button_control.text = "Stop"
        dataset_delete_captions_button_control.on_click = lambda evt: stop_captioning(evt, dataset_delete_captions_button_control, thumbnails_grid_control)
        dataset_delete_captions_button_control.tooltip = "Stop captioning process"
        dataset_delete_captions_button_control.disabled = False
        dataset_delete_captions_button_control.update()
        if e.page:
            e.page.update()
            e.page.run_task(run_dataset_script_command, command, e.page, dataset_add_captions_button_control, processed_progress_bar, processed_output_field, "Add Captions", on_success_callback=lambda: update_thumbnails(page_ctx=e.page, grid_control=thumbnails_grid_control))

    dataset_add_captions_button_control = create_styled_button(
        "Add Captions",
        on_click=on_add_captions_click_with_model,
        button_style=BTN_STYLE2,
        expand=True
    )
    dataset_delete_captions_button_control = create_styled_button(
        "Delete",
        on_click=lambda e: on_delete_captions_click(e, thumbnails_grid_control),
        tooltip="Delete the captions.json file",
        expand=False,
        button_style=BTN_STYLE2
    )
    dataset_preprocess_button_control = create_styled_button(
        "Start Preprocess ",
        on_click=on_preprocess_dataset_click,
        tooltip="Preprocess dataset using captions.json",
        expand=True,
        button_style=BTN_STYLE2
    )
    bucket_val, model_val, trigger_word_val = load_dataset_config(selected_dataset["value"])
    bucket_size_textfield.value = bucket_val
    model_name_dropdown.value = model_val if model_val in MODEL_NAME_CHOICES else DEFAULT_MODEL_NAME
    trigger_word_textfield.value = trigger_word_val or ''
    current_folders = get_dataset_folders()
    dataset_dropdown_control.options = [ft.dropdown.Option(name) for name in current_folders.keys()] if current_folders else []
    dataset_dropdown_control.disabled = len(current_folders) == 0
    lc_content = ft.Column([
        ft.Row([
            ft.Container(content=dataset_dropdown_control, expand=True, width=160),
            ft.Container(content=update_button_control, alignment=ft.alignment.center_right, width=40)
        ]),
        ft.Text("1. Captions", size=12),
        ft.Divider(height=1, thickness=1),
        ft.ResponsiveRow([captions_checkbox_c,caption_model_dropdown]),
        ft.Row([
            ft.Container(content=dataset_add_captions_button_control, expand=True),
            ft.Container(content=dataset_delete_captions_button_control, expand=True)
        ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
        ft.Container(height=10),
        ft.Column([
            ft.Text("2. Preprocess Dataset", size=12),
            ft.Divider(height=1, thickness=1),
            model_name_dropdown,
            bucket_size_textfield,
            trigger_word_textfield,
            dataset_preprocess_button_control,
        ]),
        ft.Container(height=10),
        ft.Column([
            ft.Text("3. Test latent (optional)", size=12),
            ft.Divider(height=1, thickness=1),
            ft.Text("Test here", size=12),
        ]),
        ft.Container(height=10),
        ft.Column([
            ft.Text("Rename files", size=12),
            ft.Divider(height=1, thickness=1),
            rename_textfield,
            create_styled_button("Rename files",
                                    on_click=on_rename_files_click,
                                    tooltip="Rename files",
                                    expand=True,
                                    button_style=BTN_STYLE2)
        ])
    ], spacing=10, width=200, alignment=ft.MainAxisAlignment.START)
    bottom_app_bar = ft.BottomAppBar(
        bgcolor=ft.Colors.BLUE_GREY_900,
        height=0,
        content=ft.Row([
            ft.Container(
                content=ft.Column([
                    processed_progress_bar,
                    processed_output_field,
                ], expand=True),
                expand=True,
            ),
        ], expand=True),
    )
    bottom_app_bar_ref = bottom_app_bar
    rc_content = ft.Column([
        thumbnails_grid_control,
        bottom_app_bar
    ], alignment=ft.CrossAxisAlignment.STRETCH, expand=True, spacing=10)
    lc = ft.Container(
        content=lc_content,
        padding=ft.padding.only(top=10, right=0, left=10),
    )
    rc = ft.Container(
        content=rc_content,
        padding=ft.padding.only(top=10, left=0, right=0),
        expand=True
    )
    update_thumbnails(page_ctx=p_page, grid_control=thumbnails_grid_control)
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
    return main_container

# Assign on_change handlers for global controls
bucket_size_textfield.on_change = on_bucket_or_model_change
model_name_dropdown.on_change = on_bucket_or_model_change
trigger_word_textfield.on_change = on_bucket_or_model_change

