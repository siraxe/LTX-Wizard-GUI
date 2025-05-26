import os
import glob
import cv2
import json
from settings import settings


# Helper to generate thumbnail for a video
def regenerate_all_thumbnails_for_dataset(dataset_name):
    import glob
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
            # print(f"Error: Could not open video {video_path}")
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
            
            # Ensure crop dimensions are valid
            if crop_w <= 0 or crop_h <= 0 or x_offset < 0 or y_offset < 0 or \
               (y_offset + crop_h) > orig_h or (x_offset + crop_w) > orig_w:
                # Fallback to using the original image if crop parameters are invalid
                # This can happen with very small or unusually dimensioned videos
                # print(f"Warning: Invalid crop parameters for {video_path}. Using original image for thumbnail.")
                cropped_image = image
            else:
                cropped_image = image[y_offset:y_offset+crop_h, x_offset:x_offset+crop_w]

            if cropped_image.size == 0:
                # print(f"Error: Cropped image is empty for {video_path}")
                vid.release()
                return False
            
            thumbnail_image = cv2.resize(cropped_image, (settings.THUMB_TARGET_W, settings.THUMB_TARGET_H), interpolation=cv2.INTER_AREA)
            os.makedirs(os.path.dirname(thumbnail_path), exist_ok=True)
            cv2.imwrite(thumbnail_path, thumbnail_image)
            vid.release()
            return True
        else:
            # print(f"Error: Could not read frame from video {video_path}")
            pass 
        vid.release()
    except Exception as e:
        # print(f"Error generating thumbnail for {video_path}: {e}")
        pass 
    return False

# Helper to get dataset folders
def get_dataset_folders():
    if not os.path.exists(settings.DATASETS_DIR):
        os.makedirs(settings.DATASETS_DIR, exist_ok=True) # Create if it doesn't exist
        return {}
    return {name: name for name in os.listdir(settings.DATASETS_DIR) if os.path.isdir(os.path.join(settings.DATASETS_DIR, name))}

# Helper to get video files and thumbnails for a dataset
def get_videos_and_thumbnails(dataset_name):
    dataset_path = os.path.join(settings.DATASETS_DIR, dataset_name)
    thumbnails_dir = os.path.join(settings.THUMBNAILS_BASE_DIR, dataset_name)
    os.makedirs(thumbnails_dir, exist_ok=True)

    info_path = os.path.join(dataset_path, "info.json")
    
    video_files = []
    if not os.path.exists(dataset_path):
        # print(f"Warning: Dataset path {dataset_path} does not exist.")
        return {}, {}

    for ext in settings.VIDEO_EXTENSIONS:
        video_files.extend(glob.glob(os.path.join(dataset_path, f"*{ext}")))
        video_files.extend(glob.glob(os.path.join(dataset_path, f"*{ext.upper()}"))) # Case-insensitive

    video_files = sorted(list(set(video_files))) # Remove duplicates and sort

    thumbnail_paths = {}
    video_info = {}
    
    if os.path.exists(info_path):
        try:
            with open(info_path, 'r', encoding='utf-8') as f:
                video_info = json.load(f)
        except json.JSONDecodeError:
            # print(f"Warning: Could not decode JSON from {info_path}. Initializing empty video_info.")
            video_info = {}
        except Exception as e:
            # print(f"Error reading {info_path}: {e}. Initializing empty video_info.")
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
                    # Basic validation for extracted info
                    if width > 0 and height > 0 and frames >= 0:
                        video_info[video_name] = {
                            "width": width,
                            "height": height,
                            "frames": frames
                        }
                        info_changed = True
                    else:
                        # print(f"Warning: Invalid metadata extracted for {video_name} (W:{width}, H:{height}, F:{frames}). Skipping info update.")
                        pass
                else:
                    # print(f"Warning: Could not open video {video_name} to extract info.")
                    pass
                vid.release()
            except Exception as e:
                # print(f"Error extracting info for {video_name}: {e}")
                pass
                
        # Only generate thumbnail if it does not exist
        if not os.path.exists(thumbnail_path):
            generate_thumbnail(video_path, thumbnail_path)

        if os.path.exists(thumbnail_path):
            thumbnail_paths[video_path] = thumbnail_path
            
    if info_changed:
        try:
            os.makedirs(dataset_path, exist_ok=True)
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(video_info, f, indent=4)
        except Exception as e:
            # print(f"Error writing info.json to {info_path}: {e}")
            pass
            
    return thumbnail_paths, video_info
