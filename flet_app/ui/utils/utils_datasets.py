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

def generate_image_thumbnail(image_path, thumbnail_path):
    """Generates a thumbnail for an image file."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            return False

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
             return False

        thumbnail_image = cv2.resize(cropped_image, (settings.THUMB_TARGET_W, settings.THUMB_TARGET_H), interpolation=cv2.INTER_AREA)
        os.makedirs(os.path.dirname(thumbnail_path), exist_ok=True)
        cv2.imwrite(thumbnail_path, thumbnail_image)
        return True
    except Exception: # General exception for OpenCV/file errors
        pass
    return False

def get_dataset_folders():
    """Gets folder names from both DATASETS_DIR and DATASETS_IMG_DIR.
    
    Returns:
        dict: A dictionary where keys are folder names and values are display names.
        Folders from DATASETS_IMG_DIR will have "(img) " prepended to their display names.
    """
    dataset_folders = {}

    # Add folders from DATASETS_DIR
    if os.path.exists(settings.DATASETS_DIR):
        for name in os.listdir(settings.DATASETS_DIR):
            if name == "_bak": # Ignore _bak folder
                continue
            folder_path = os.path.join(settings.DATASETS_DIR, name)
            if os.path.isdir(folder_path):
                # Only add if not already present (shouldn't happen as we're iterating DATASETS_DIR first)
                if name not in dataset_folders:
                    dataset_folders[name] = name

    # Add folders from DATASETS_IMG_DIR
    if os.path.exists(settings.DATASETS_IMG_DIR):
        for name in os.listdir(settings.DATASETS_IMG_DIR):
            if name == "_bak": # Ignore _bak folder
                continue
            folder_path = os.path.join(settings.DATASETS_IMG_DIR, name)
            if os.path.isdir(folder_path):
                # Always add with (img) prefix for DATASETS_IMG_DIR
                # Remove any existing (img) prefix to prevent duplicates
                clean_name = name.replace('(img) ', '').replace(' (img)', '')
                dataset_folders[clean_name] = f"(img) {clean_name}"

    return dataset_folders

def get_videos_and_thumbnails(dataset_name, dataset_type):
    """
    Gets media files and generates/retrieves thumbnails for a dataset.
    Handles both video and image datasets based on their location.
    """
    # Clean the dataset name by removing any image markers
    clean_dataset_name = dataset_name.replace('(img) ', '').replace(' (img)', '')
    
    if dataset_type == "image":
        # Use clean_dataset_name for path construction to handle both old and new formats
        dataset_path = os.path.join(settings.DATASETS_IMG_DIR, clean_dataset_name)
        thumbnails_dir = os.path.join(settings.THUMBNAILS_IMG_BASE_DIR, clean_dataset_name)
        media_extensions = settings.IMAGE_EXTENSIONS
        # Create and use info.json for image datasets to store dimensions
        info_path = os.path.join(dataset_path, "info.json")
    else: # dataset_type == "video"
        dataset_path = os.path.join(settings.DATASETS_DIR, clean_dataset_name)
        thumbnails_dir = os.path.join(settings.THUMBNAILS_BASE_DIR, clean_dataset_name)
        media_extensions = settings.VIDEO_EXTENSIONS
        info_path = os.path.join(dataset_path, "info.json")

    os.makedirs(thumbnails_dir, exist_ok=True)

    if not os.path.exists(dataset_path):
        print(f"Dataset path not found: {dataset_path}") # Debugging print
        return {}, {} # Return empty dictionaries if dataset path doesn't exist

    # List all media files with specified extensions
    media_files = []
    for ext in media_extensions:
        media_files.extend(glob.glob(os.path.join(dataset_path, f"*{ext}")))
        media_files.extend(glob.glob(os.path.join(dataset_path, f"*{ext.upper()}")))

    # Ensure file paths are normalized and unique, then sort
    media_files = sorted(list(set(os.path.normpath(f) for f in media_files)))

    thumbnail_paths = {}
    media_info = {} # Use media_info to be general for video/image

    # Load existing info for video datasets
    if info_path and os.path.exists(info_path):
        try:
            with open(info_path, 'r', encoding='utf-8') as f:
                loaded_info = json.load(f)
                if isinstance(loaded_info, dict):
                    media_info = loaded_info
        except (json.JSONDecodeError, Exception) as e:
            print(f"Error loading info.json for {dataset_name}: {e}") # Debugging print
            media_info = {}

    info_changed = False # To track if media_info was updated and needs saving

    for media_path in media_files:
        media_name = os.path.basename(media_path)
        thumbnail_name = f"{os.path.splitext(media_name)[0]}.jpg"
        thumbnail_path = os.path.join(thumbnails_dir, thumbnail_name)

        # Get media dimensions/info if not already in media_info or if it's an image dataset
        if media_name not in media_info or dataset_type == "image": # Always try to get info for images
            try:
                if dataset_type == "image":
                    img = cv2.imread(media_path)
                    if img is not None:
                        height, width = img.shape[:2]
                        # For images, frames can be considered 1 or omitted, using 1 for now
                        media_info[media_name] = {"width": width, "height": height, "frames": 1}
                        info_changed = True
                        # Ensure the dataset directory exists before trying to save info.json
                        os.makedirs(dataset_path, exist_ok=True)
                    else:
                        print(f"Could not read image file: {media_path}") # Debugging print
                else: # Video dataset
                    vid = cv2.VideoCapture(media_path)
                    if vid.isOpened():
                        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
                        if width > 0 and height > 0 and frames >= 0:
                            media_info[media_name] = {"width": width, "height": height, "frames": frames}
                            info_changed = True
                    vid.release()
            except Exception as e:
                print(f"Error getting media info for {media_name}: {e}") # Debugging print
                pass # Continue even if info retrieval fails

        # Generate thumbnail if it doesn't exist
        if not os.path.exists(thumbnail_path):
            print(f"Generating thumbnail for: {media_path}") # Debugging print
            if dataset_type == "image":
                generate_image_thumbnail(media_path, thumbnail_path)
            else:
                generate_thumbnail(media_path, thumbnail_path)

        # Add thumbnail path to the map if it exists
        if os.path.exists(thumbnail_path):
            thumbnail_paths[media_path] = thumbnail_path
        else:
             print(f"Thumbnail not found after generation attempt: {thumbnail_path}") # Debugging print


    # Save updated info.json for video datasets if changes occurred
    if info_changed and info_path:
        try:
            os.makedirs(os.path.dirname(info_path), exist_ok=True)
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(media_info, f, indent=4)
        except Exception as e:
             print(f"Error saving info.json for {dataset_name}: {e}") # Debugging print
             pass # Continue even if saving fails

    return thumbnail_paths, media_info

async def apply_affix_from_textfield(e: ft.ControlEvent, affix_type: str, selected_dataset_ref, dataset_type: str, update_thumbnails_func, thumbnails_grid_ref_obj, affix_text_field_ref: ft.Ref[ft.TextField]):
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
    
    if dataset_type == "image":
        dataset_dir = settings.DATASETS_IMG_DIR
    else:
        dataset_dir = settings.DATASETS_DIR
        
    captions_json_path = os.path.join(dataset_dir, clean_dataset_name, "captions.json")

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

async def find_and_replace_in_captions(e: ft.ControlEvent, selected_dataset_ref, dataset_type: str, find_text_field_ref: ft.Ref[ft.TextField], replace_text_field_ref: ft.Ref[ft.TextField], update_thumbnails_func, thumbnails_grid_ref_obj):
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
    
    if dataset_type == "image":
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

async def on_change_fps_click(e: ft.ControlEvent, selected_dataset_ref, dataset_type: str, change_fps_textfield_ref_obj, thumbnails_grid_ref_obj, update_thumbnails_func, settings_obj):
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

    clean_dataset_name = current_dataset_name.replace('(img) ', '').replace(' (img)', '')
    if dataset_type == "image":
        dataset_folder = os.path.join(settings_obj.DATASETS_IMG_DIR, clean_dataset_name)
    else:
        dataset_folder = os.path.join(settings_obj.DATASETS_DIR, clean_dataset_name)

    if not os.path.isdir(dataset_folder):
        if e.page:
            e.page.snack_bar = ft.SnackBar(content=ft.Text(f"Error: Dataset folder '{clean_dataset_name}' not found."), open=True)
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

async def on_rename_files_click(e: ft.ControlEvent, selected_dataset_ref, dataset_type: str, rename_textfield_obj, thumbnails_grid_ref_obj, update_thumbnails_func, settings_obj):
    print("\n=== RENAME FUNCTION CALLED ===")
    print("=== Starting rename operation ===")
    print(f"[DEBUG] Event type: {type(e)}")
    print(f"[DEBUG] Selected dataset ref: {selected_dataset_ref}")
    current_dataset_name = selected_dataset_ref.get("value")
    print(f"[DEBUG] Current dataset name: {current_dataset_name}")
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

    # Clean the dataset name by removing any image markers (both old and new formats)
    clean_dataset_name = current_dataset_name.replace('(img) ', '').replace(' (img)', '')
    print(f"[DEBUG] Cleaned dataset name: {clean_dataset_name}")
    
    print(f"[DEBUG] Dataset type: {dataset_type}")
    
    if dataset_type == "image":
        dataset_dir = settings_obj.DATASETS_IMG_DIR
        file_extensions = settings_obj.IMAGE_EXTENSIONS
    else:
        dataset_dir = settings_obj.DATASETS_DIR
        file_extensions = settings_obj.VIDEO_EXTENSIONS
    
    dataset_folder = os.path.join(dataset_dir, clean_dataset_name)
    if not os.path.isdir(dataset_folder):
        if e.page:
            e.page.snack_bar = ft.SnackBar(content=ft.Text(f"Error: Dataset folder '{clean_dataset_name}' not found in {dataset_dir}."), open=True)
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

    files_to_rename = []
    for ext in file_extensions:
        files_to_rename.extend(glob.glob(os.path.join(dataset_folder, f"*{ext}")))
        files_to_rename.extend(glob.glob(os.path.join(dataset_folder, f"*{ext.upper()}")))
    
    files_to_rename = sorted([f for f in list(set(files_to_rename)) if os.path.isfile(f)])

    if not files_to_rename:
        file_type = "image" if dataset_type == "image" else "video"
        if e.page:
            e.page.snack_bar = ft.SnackBar(content=ft.Text(f"No {file_type} files found in '{clean_dataset_name}' to rename."), open=True)
            e.page.update()
        return
        
    renamed_files_count = 0
    failed_files_count = 0
    updated_captions_map = {} # To store new filenames for captions

    for idx, old_file_path in enumerate(files_to_rename):
        original_basename = os.path.basename(old_file_path)
        _, ext = os.path.splitext(original_basename)
        new_basename = f"{base_name_template}_{idx+1:04d}{ext}" # e.g., myimage_0001.jpg or myvideo_0001.mp4
        new_file_path = os.path.join(dataset_folder, new_basename)

        if old_file_path == new_file_path: # Skip if name is already correct
            updated_captions_map[original_basename] = new_basename # Still need for caption update
            continue

        try:
            shutil.move(old_file_path, new_file_path)
            renamed_files_count += 1
            updated_captions_map[original_basename] = new_basename
            
            # Rename corresponding thumbnail if it exists
            old_thumb_name = f"{os.path.splitext(original_basename)[0]}.jpg"
            new_thumb_name = f"{os.path.splitext(new_basename)[0]}.jpg"
            
            if dataset_type == "image":
                old_thumb_path = os.path.join(settings_obj.THUMBNAILS_IMG_BASE_DIR, clean_dataset_name, old_thumb_name)
                new_thumb_path = os.path.join(settings_obj.THUMBNAILS_IMG_BASE_DIR, clean_dataset_name, new_thumb_name)
            else:
                old_thumb_path = os.path.join(settings_obj.THUMBNAILS_BASE_DIR, clean_dataset_name, old_thumb_name)
                new_thumb_path = os.path.join(settings_obj.THUMBNAILS_BASE_DIR, clean_dataset_name, new_thumb_name)

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
