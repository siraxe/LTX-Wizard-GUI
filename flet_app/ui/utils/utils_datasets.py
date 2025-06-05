import os
import glob
import cv2
import json
import flet as ft
from settings import settings
import asyncio
import subprocess
import shutil

#Helper to generate thumbnail for a video
def regenerate_all_thumbnails_for_dataset(dataset_name):
    # import glob # This import can be at top level
    dataset_path = os.path.join(settings.DATASETS_DIR, dataset_name)
    thumbnails_dir = os.path.join(settings.THUMBNAILS_BASE_DIR, dataset_name)
    if not os.path.exists(dataset_path):
        return
    os.makedirs(thumbnails_dir, exist_ok=True)
    for ext in settings.VIDEO_EXTENSIONS:
        for video_path in glob.glob(os.path.join(dataset_path, f"*{ext}")) + glob.glob(os.path.join(dataset_path, f"*{ext.upper()}")):
            video_name = os.path.basename(video_path)
            thumbnail_name = f"{os.path.splitext(video_name)[0]}.jpg"
            thumbnail_path = os.path.join(thumbnails_dir, thumbnail_name)
            if os.path.exists(thumbnail_path):
                try:
                    os.remove(thumbnail_path)
                except Exception:
                    pass
            generate_thumbnail(video_path, thumbnail_path)

def generate_thumbnail(video_path, thumbnail_path):
    try:
        vid = cv2.VideoCapture(video_path)
        if not vid.isOpened():
            return False
        success, image = vid.read()
        if success:
            orig_h, orig_w = image.shape[:2]
            orig_aspect_ratio = orig_w / orig_h
            crop_w, crop_h = orig_w, orig_h
            x_offset, y_offset = 0, 0
            if orig_aspect_ratio > settings.TARGET_ASPECT_RATIO:
                crop_w = int(orig_h * settings.TARGET_ASPECT_RATIO)
                x_offset = int((orig_w - crop_w) / 2)
            elif orig_aspect_ratio < settings.TARGET_ASPECT_RATIO:
                crop_h = int(orig_w / settings.TARGET_ASPECT_RATIO)
                y_offset = int((orig_h - crop_h) / 2)
            
            if crop_w <= 0 or crop_h <= 0 or x_offset < 0 or y_offset < 0 or \
               (y_offset + crop_h) > orig_h or (x_offset + crop_w) > orig_w:
                cropped_image = image
            else:
                cropped_image = image[y_offset:y_offset+crop_h, x_offset:x_offset+crop_w]

            if cropped_image.size == 0:
                vid.release()
                return False
            
            thumbnail_image = cv2.resize(cropped_image, (settings.THUMB_TARGET_W, settings.THUMB_TARGET_H), interpolation=cv2.INTER_AREA)
            os.makedirs(os.path.dirname(thumbnail_path), exist_ok=True)
            cv2.imwrite(thumbnail_path, thumbnail_image)
            vid.release()
            return True
        vid.release()
    except Exception: # General exception for OpenCV/file errors
        pass 
    return False

def get_dataset_folders():
    """Gets folder names from both DATASETS_DIR and DATASETS_IMG_DIR."""
    dataset_folders = {}

    # Add folders from DATASETS_DIR
    if os.path.exists(settings.DATASETS_DIR):
        for name in os.listdir(settings.DATASETS_DIR):
            folder_path = os.path.join(settings.DATASETS_DIR, name)
            if os.path.isdir(folder_path):
                dataset_folders[name] = name

    # Add folders from DATASETS_IMG_DIR
    if os.path.exists(settings.DATASETS_IMG_DIR):
        for name in os.listdir(settings.DATASETS_IMG_DIR):
            folder_path = os.path.join(settings.DATASETS_IMG_DIR, name)
            if os.path.isdir(folder_path):
                # Only add if not already present from DATASETS_DIR to avoid duplicates
                if name not in dataset_folders:
                    dataset_folders[name] = name

    return dataset_folders

def get_videos_and_thumbnails(dataset_name):
    dataset_path = os.path.join(settings.DATASETS_DIR, dataset_name)
    thumbnails_dir = os.path.join(settings.THUMBNAILS_BASE_DIR, dataset_name)
    os.makedirs(thumbnails_dir, exist_ok=True)
    info_path = os.path.join(dataset_path, "info.json")
    video_files = []
    if not os.path.exists(dataset_path):
        return {}, {}
    for ext in settings.VIDEO_EXTENSIONS:
        video_files.extend(glob.glob(os.path.join(dataset_path, f"*{ext}")))
        video_files.extend(glob.glob(os.path.join(dataset_path, f"*{ext.upper()}")))
    video_files = sorted(list(set(video_files)))
    thumbnail_paths = {}
    video_info = {}
    if os.path.exists(info_path):
        try:
            with open(info_path, 'r', encoding='utf-8') as f:
                video_info = json.load(f)
        except json.JSONDecodeError:
            video_info = {}
        except Exception:
            video_info = {}
            
    info_changed = False
    for video_path in video_files:
        video_name = os.path.basename(video_path)
        thumbnail_name = f"{os.path.splitext(video_name)[0]}.jpg"
        thumbnail_path = os.path.join(thumbnails_dir, thumbnail_name)
        if video_name not in video_info:
            try:
                vid = cv2.VideoCapture(video_path)
                if vid.isOpened():
                    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
                    if width > 0 and height > 0 and frames >= 0:
                        video_info[video_name] = {"width": width, "height": height, "frames": frames}
                        info_changed = True
                vid.release()
            except Exception:
                pass
        if not os.path.exists(thumbnail_path):
            generate_thumbnail(video_path, thumbnail_path)
        if os.path.exists(thumbnail_path):
            thumbnail_paths[video_path] = thumbnail_path
            
    if info_changed:
        try:
            os.makedirs(dataset_path, exist_ok=True)
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(video_info, f, indent=4)
        except Exception:
            pass
            
    return thumbnail_paths, video_info

async def apply_affix_from_textfield(e: ft.ControlEvent, affix_type: str, selected_dataset_ref, update_thumbnails_func, thumbnails_grid_ref_obj, affix_text_field_ref: ft.Ref[ft.TextField]):
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
    captions_json_path = os.path.join(settings.DATASETS_DIR, current_dataset_name, "captions.json")

    if not os.path.exists(captions_json_path):
        e.page.snack_bar = ft.SnackBar(content=ft.Text(f"No captions.json found for dataset '{current_dataset_name}'."), open=True)
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

    modified_count = 0
    for item in captions_data:
        if isinstance(item, dict) and "caption" in item and isinstance(item["caption"], str):
            if affix_type == "prefix":
                item["caption"] = f"{affix_text} {item['caption']}"
            elif affix_type == "suffix":
                item["caption"] = f"{item['caption']} {affix_text}"
            modified_count += 1
    
    if modified_count == 0 and captions_data:
        e.page.snack_bar = ft.SnackBar(content=ft.Text(f"No captions were modified. Check format."), open=True)
        if e.page: e.page.update()
        return
    elif modified_count == 0 and not captions_data:
        e.page.snack_bar = ft.SnackBar(content=ft.Text(f"Captions.json is empty. No captions to modify."), open=True)
        if e.page: e.page.update()
        return

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

async def find_and_replace_in_captions(e: ft.ControlEvent, selected_dataset_ref, find_text_field_ref: ft.Ref[ft.TextField], replace_text_field_ref: ft.Ref[ft.TextField], update_thumbnails_func, thumbnails_grid_ref_obj):
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
    captions_json_path = os.path.join(settings.DATASETS_DIR, current_dataset_name, "captions.json")

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

    modified_count = 0
    replacements_made = 0
    for item in captions_data:
        if isinstance(item, dict) and "caption" in item and isinstance(item["caption"], str):
            original_caption = item["caption"]
            item["caption"] = original_caption.replace(find_text, replace_text)
            if original_caption != item["caption"]:
                modified_count += 1
                replacements_made += original_caption.count(find_text) if find_text else 0
    
    if modified_count == 0 and captions_data:
        e.page.snack_bar = ft.SnackBar(content=ft.Text(f"Text '{find_text}' not found. No changes made."), open=True)
        if e.page: e.page.update()
        return
    elif modified_count == 0 and not captions_data:
        e.page.snack_bar = ft.SnackBar(content=ft.Text(f"Captions.json is empty. No changes made."), open=True)
        if e.page: e.page.update()
        return

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

async def on_change_fps_click(e: ft.ControlEvent, selected_dataset_ref, change_fps_textfield_ref_obj, thumbnails_grid_ref_obj, update_thumbnails_func, settings_obj):
    current_dataset_name = selected_dataset_ref.get("value")
    if not current_dataset_name:
        if e.page:
            e.page.snack_bar = ft.SnackBar(content=ft.Text("Error: No dataset selected."), open=True)
            e.page.update()
        return

    if not change_fps_textfield_ref_obj or not change_fps_textfield_ref_obj.current or not change_fps_textfield_ref_obj.current.value:
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
            e.page.snack_bar = ft.SnackBar(content=ft.Text("Error: Invalid FPS. Please enter a positive number."), open=True)
            e.page.update()
        return

    dataset_folder = os.path.join(settings_obj.DATASETS_DIR, current_dataset_name)
    if not os.path.isdir(dataset_folder):
        if e.page:
            e.page.snack_bar = ft.SnackBar(content=ft.Text(f"Error: Dataset folder '{current_dataset_name}' not found."), open=True)
            e.page.update()
        return

    output_folder = os.path.join(dataset_folder, f"{current_dataset_name}_{target_fps_str}fps")
    os.makedirs(output_folder, exist_ok=True)

    ffmpeg_path = settings_obj.FFMPEG_PATH
    if not ffmpeg_path or not os.path.exists(ffmpeg_path):
        if e.page:
            e.page.snack_bar = ft.SnackBar(content=ft.Text("Error: ffmpeg path not configured or invalid in settings."), open=True)
            e.page.update()
        return
        
    processed_files = 0
    failed_files = 0
    
    video_files_to_process = []
    for ext in settings_obj.VIDEO_EXTENSIONS:
        video_files_to_process.extend(glob.glob(os.path.join(dataset_folder, f"*{ext}")))
        video_files_to_process.extend(glob.glob(os.path.join(dataset_folder, f"*{ext.upper()}")))
    
    video_files_to_process = [f for f in list(set(video_files_to_process)) if os.path.isfile(f)]


    if not video_files_to_process:
        if e.page:
            e.page.snack_bar = ft.SnackBar(content=ft.Text(f"No video files found in '{current_dataset_name}'."), open=True)
            e.page.update()
        return

    for video_file in video_files_to_process:
        base_name = os.path.basename(video_file)
        output_file = os.path.join(output_folder, base_name)
        
        command = [
            ffmpeg_path, "-y",
            "-i", video_file,
            "-vf", f"fps={target_fps_float}",
            "-c:v", "libx264", # Or user-defined codec
            "-preset", "medium", # Or user-defined
            "-crf", "18", # Or user-defined
            "-c:a", "aac", # Or user-defined
            "-b:a", "128k", # Or user-defined
            output_file
        ]
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
            stdout, stderr = process.communicate()
            if process.returncode == 0:
                processed_files += 1
            else:
                failed_files += 1
                print(f"ffmpeg error for {video_file}: {stderr}") # Log error
        except Exception as ex:
            failed_files += 1
            print(f"Error processing {video_file}: {ex}") # Log error

    result_message = f"FPS change: {processed_files} processed, {failed_files} failed. Output in '{output_folder}'."
    if e.page:
        e.page.snack_bar = ft.SnackBar(content=ft.Text(result_message), open=True)
        # Optionally, refresh thumbnails if the operation modifies the current dataset directly
        # or if the user is expected to switch to the new dataset folder.
        # For now, we assume the user will manually select the new dataset if needed.
        # if asyncio.iscoroutinefunction(update_thumbnails_func):
        #     await update_thumbnails_func(e.page, thumbnails_grid_ref_obj.current, force_refresh=True)
        # else:
        #     update_thumbnails_func(e.page, thumbnails_grid_ref_obj.current, force_refresh=True)
        e.page.update()

async def on_rename_files_click(e: ft.ControlEvent, selected_dataset_ref, rename_textfield_obj, thumbnails_grid_ref_obj, update_thumbnails_func, settings_obj):
    current_dataset_name = selected_dataset_ref.get("value")
    if not current_dataset_name:
        if e.page:
            e.page.snack_bar = ft.SnackBar(content=ft.Text("Error: No dataset selected."), open=True)
            e.page.update()
        return

    if not rename_textfield_obj or not rename_textfield_obj.current or not rename_textfield_obj.current.value:
        if e.page:
            e.page.snack_bar = ft.SnackBar(content=ft.Text("Error: Base name for renaming not provided."), open=True)
            e.page.update()
        return

    base_name_template = rename_textfield_obj.current.value.strip()
    if not base_name_template: # Ensure it's not just whitespace
        if e.page:
            e.page.snack_bar = ft.SnackBar(content=ft.Text("Error: Base name cannot be empty."), open=True)
            e.page.update()
        return

    dataset_folder = os.path.join(settings_obj.DATASETS_DIR, current_dataset_name)
    if not os.path.isdir(dataset_folder):
        if e.page:
            e.page.snack_bar = ft.SnackBar(content=ft.Text(f"Error: Dataset folder '{current_dataset_name}' not found."), open=True)
            e.page.update()
        return

    captions_json_path = os.path.join(dataset_folder, "captions.json")
    captions_data = []
    if os.path.exists(captions_json_path):
        try:
            with open(captions_json_path, 'r', encoding='utf-8') as f:
                captions_data = json.load(f)
            if not isinstance(captions_data, list):
                captions_data = [] # Or handle error
        except (json.JSONDecodeError, Exception):
            captions_data = [] # Or handle error

    video_files_to_rename = []
    for ext in settings_obj.VIDEO_EXTENSIONS:
        video_files_to_rename.extend(glob.glob(os.path.join(dataset_folder, f"*{ext}")))
        video_files_to_rename.extend(glob.glob(os.path.join(dataset_folder, f"*{ext.upper()}")))
    
    video_files_to_rename = sorted([f for f in list(set(video_files_to_rename)) if os.path.isfile(f)])

    if not video_files_to_rename:
        if e.page:
            e.page.snack_bar = ft.SnackBar(content=ft.Text(f"No video files found in '{current_dataset_name}' to rename."), open=True)
            e.page.update()
        return
        
    renamed_files_count = 0
    failed_files_count = 0
    updated_captions_map = {} # To store new filenames for captions

    for idx, old_video_path in enumerate(video_files_to_rename):
        original_basename = os.path.basename(old_video_path)
        _, ext = os.path.splitext(original_basename)
        new_basename = f"{base_name_template}_{idx+1:04d}{ext}" # e.g., myvideo_0001.mp4
        new_video_path = os.path.join(dataset_folder, new_basename)

        if old_video_path == new_video_path: # Skip if name is already correct
            updated_captions_map[original_basename] = new_basename # Still need for caption update
            continue

        try:
            shutil.move(old_video_path, new_video_path)
            renamed_files_count += 1
            updated_captions_map[original_basename] = new_basename
            
            # Rename corresponding thumbnail if it exists
            old_thumb_name = f"{os.path.splitext(original_basename)[0]}.jpg"
            new_thumb_name = f"{os.path.splitext(new_basename)[0]}.jpg"
            old_thumb_path = os.path.join(settings_obj.THUMBNAILS_BASE_DIR, current_dataset_name, old_thumb_name)
            new_thumb_path = os.path.join(settings_obj.THUMBNAILS_BASE_DIR, current_dataset_name, new_thumb_name)
            if os.path.exists(old_thumb_path):
                shutil.move(old_thumb_path, new_thumb_path)

        except Exception as ex:
            failed_files_count += 1
            print(f"Error renaming {original_basename} to {new_basename}: {ex}")

    # Update captions.json
    if captions_data and updated_captions_map:
        modified_captions = False
        for item in captions_data:
            if isinstance(item, dict) and "video_filename" in item:
                if item["video_filename"] in updated_captions_map:
                    item["video_filename"] = updated_captions_map[item["video_filename"]]
                    modified_captions = True
        if modified_captions:
            try:
                with open(captions_json_path, 'w', encoding='utf-8') as f:
                    json.dump(captions_data, f, indent=4)
            except Exception as ex:
                print(f"Error updating captions.json: {ex}")
                if e.page:
                    e.page.snack_bar = ft.SnackBar(content=ft.Text(f"Files renamed, but error updating captions.json: {ex}"), open=True)
                    # No page update here, it's part of the main finally block

    result_message = f"Renamed {renamed_files_count} files. Failed: {failed_files_count}."
    if e.page:
        e.page.snack_bar = ft.SnackBar(content=ft.Text(result_message), open=True)
        if asyncio.iscoroutinefunction(update_thumbnails_func):
            await update_thumbnails_func(e.page, thumbnails_grid_ref_obj.current, force_refresh=True)
        else:
            update_thumbnails_func(e.page, thumbnails_grid_ref_obj.current, force_refresh=True)
        e.page.update()
