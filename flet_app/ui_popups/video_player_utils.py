# video_player_utils.py
import os
import json
import cv2
import subprocess
import math
from typing import Tuple, List, Dict, Any, Optional

# Attempt to import settings for FFMPEG_PATH
try:
    from settings import settings as app_settings
except ImportError:
    # Fallback if settings module is not found or FFMPEG_PATH is not there
    class FallbackSettings:
        FFMPEG_PATH = "ffmpeg" # Default to ffmpeg in PATH
        # Define other settings if needed, or handle their absence
    app_settings = FallbackSettings()

MIN_OVERLAY_SIZE = 20 # Minimum size for the overlay during resize (used in calculations)

# Caption and Data Handling
def load_caption_for_video(video_path: str) -> Tuple[str, str, Optional[str]]:
    """
    Load caption and negative caption for a given video from its captions.json file.
    Returns: (caption, negative_caption, message_string_or_none)
    """
    video_dir = os.path.dirname(video_path)
    dataset_json_path = os.path.join(video_dir, "captions.json")
    video_filename = os.path.basename(video_path)
    no_caption_text = "No captions found, add it here and press Update"
    
    caption_value = ""
    negative_caption_value = ""
    message_to_display = no_caption_text

    if os.path.exists(dataset_json_path):
        try:
            with open(dataset_json_path, 'r', encoding='utf-8') as f:
                dataset_json_data = json.load(f)
            for entry in dataset_json_data:
                if entry.get("media_path") == video_filename:
                    caption_value = entry.get("caption", "")
                    negative_caption_value = entry.get("negative_caption", "")
                    message_to_display = None  # Captions found
                    break
        except Exception as e:
            print(f"Error reading captions file {dataset_json_path}: {e}")
            message_to_display = f"Error loading captions: {e}"
            
    return caption_value, negative_caption_value, message_to_display

def save_caption_for_video(video_path: str, new_caption: str, field_name: str = "caption") -> Tuple[bool, str]:
    """
    Save the new caption for the given video to its captions.json file.
    field_name can be "caption" or "negative_caption".
    Returns: (success_bool, message_string)
    """
    video_dir = os.path.dirname(video_path)
    dataset_json_path = os.path.join(video_dir, "captions.json")
    video_filename = os.path.basename(video_path)
    current_dataset_json_data = []

    if os.path.exists(dataset_json_path):
        try:
            with open(dataset_json_path, 'r', encoding='utf-8') as f:
                current_dataset_json_data = json.load(f)
        except Exception as ex:
            msg = f"Error re-reading captions file before save: {ex}"
            print(msg)
            return False, msg

    found_entry = False
    for entry in current_dataset_json_data:
        if entry.get("media_path") == video_filename:
            entry[field_name] = new_caption
            found_entry = True
            break
    
    if not found_entry:
        new_entry = {"media_path": video_filename, "caption": "", "negative_caption": ""}
        new_entry[field_name] = new_caption
        current_dataset_json_data.append(new_entry)

    try:
        os.makedirs(video_dir, exist_ok=True)
        with open(dataset_json_path, 'w', encoding='utf-8') as f:
            json.dump(current_dataset_json_data, f, indent=2, ensure_ascii=False)
        
        friendly_field_name = field_name.replace('_', ' ').title()
        return True, f"{friendly_field_name} updated!"
    except Exception as ex_write:
        msg = f"Error writing captions file {dataset_json_path}: {ex_write}"
        print(msg)
        return False, f"Failed to update {friendly_field_name}: {ex_write}"

def get_next_video_path(video_list: List[str], current_video_path: str, offset: int) -> Optional[str]:
    """
    Return the next video path in the list given the current path and offset.
    Wraps around if at the end/start.
    """
    if not video_list or not current_video_path:
        return None
    try:
        current_idx = video_list.index(current_video_path)
        new_idx = (current_idx + offset) % len(video_list)
        return video_list[new_idx]
    except ValueError:
        print(f"Error: Current video {current_video_path} not in list.")
        return None

# Video Metadata

def get_video_metadata(video_path: str) -> Optional[Dict[str, Any]]:
    """
    Extracts metadata (fps, total_frames, width, height) from a video file.
    Returns a dictionary or None if an error occurs.
    """
    if not video_path or not os.path.exists(video_path):
        print(f"Video path does not exist: {video_path}")
        return None
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Could not open video: {video_path}")
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        if fps <= 0 or total_frames <= 0 or width <= 0 or height <= 0:
            print(f"Invalid metadata for video {video_path}: fps={fps}, frames={total_frames}, w={width}, h={height}")
            # Allow partial metadata if some values are valid, but log it.
            # For critical operations, the caller should check for all necessary fields.

        return {'fps': fps, 'total_frames': total_frames, 'width': width, 'height': height}
    except Exception as e:
        print(f"Error getting video metadata for {video_path}: {e}")
        return None

def update_video_info_json(video_path: str) -> bool:
    """
    Updates the info.json for a given video path with its current metadata.
    Creates info.json if it doesn't exist.
    Returns True on success, False on failure.
    """
    video_dir = os.path.dirname(video_path)
    info_json_path = os.path.join(video_dir, 'info.json')
    info_data: Dict[str, Any] = {}

    if os.path.exists(info_json_path):
        try:
            with open(info_json_path, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
            if isinstance(loaded_data, dict):
                info_data = loaded_data
            else:
                print(f"Warning: info.json at {info_json_path} was not a dictionary. Reinitializing.")
        except json.JSONDecodeError:
            print(f"Warning: info.json at {info_json_path} is corrupted. Reinitializing.")
        except Exception as e:
            print(f"Error loading info.json: {e}. Reinitializing.")
    
    video_filename = os.path.basename(video_path)
    metadata = get_video_metadata(video_path)

    if metadata:
        if video_filename not in info_data:
            info_data[video_filename] = {}
        info_data[video_filename]['frames'] = metadata.get('total_frames')
        info_data[video_filename]['fps'] = metadata.get('fps')
        info_data[video_filename]['width'] = metadata.get('width')
        info_data[video_filename]['height'] = metadata.get('height')
    else:
        print(f"Could not get metadata for {video_filename} to update info.json.")
        # Optionally remove entry or leave stale, depending on desired behavior
        # For now, we'll proceed to write even if metadata is missing, to preserve other entries.

    try:
        os.makedirs(video_dir, exist_ok=True)
        with open(info_json_path, 'w', encoding='utf-8') as f:
            json.dump(info_data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error writing info.json: {e}")
        return False

# Dimension Calculations

def _closest_divisible_by(value: int, divisor: int) -> int:
    """Helper to find the closest multiple of 'divisor' to 'value', ensuring minimum of 'divisor'."""
    return max(divisor, int(round(value / divisor)) * divisor)

def calculate_adjusted_size(
    current_w_str: str, 
    current_h_str: str, 
    video_path: Optional[str], 
    adjustment_type: str, # 'add' or 'sub'
    step: int = 32
) -> Tuple[str, str]:
    """
    Adjusts width and height by a step (default 32), maintaining aspect ratio if video_path is provided.
    Returns new (width_str, height_str).
    """
    try:
        w = int(current_w_str) if current_w_str else step
        h = int(current_h_str) if current_h_str else step
    except ValueError:
        return str(step), str(step)

    aspect_ratio = None
    if video_path:
        metadata = get_video_metadata(video_path)
        if metadata and metadata['width'] > 0 and metadata['height'] > 0:
            aspect_ratio = metadata['width'] / metadata['height']

    if adjustment_type == 'add':
        w = ((w // step) + 1) * step
        if aspect_ratio:
            h = _closest_divisible_by(w / aspect_ratio, step)
        else:
            h = ((h // step) + 1) * step
    elif adjustment_type == 'sub':
        w = max(step, ((w - 1) // step) * step)
        if aspect_ratio:
            h = _closest_divisible_by(w / aspect_ratio, step)
        else:
            h = max(step, ((h - 1) // step) * step)
    
    return str(w), str(h)

def calculate_closest_div32_dimensions(video_path: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Calculates dimensions for a video that are closest to original and divisible by 32.
    Returns (width_str, height_str) or (None, None).
    """
    metadata = get_video_metadata(video_path)
    if not metadata or metadata['width'] <= 0 or metadata['height'] <= 0:
        return None, None
    
    width32 = _closest_divisible_by(metadata['width'], 32)
    height32 = _closest_divisible_by(metadata['height'], 32)
    
    return str(width32), str(height32)

# FFmpeg Utilities

def _get_ffmpeg_exe_path() -> str:
    """Gets the FFmpeg executable path from settings or defaults."""
    ffmpeg_path = getattr(app_settings, 'FFMPEG_PATH', 'ffmpeg')
    # If the path from settings is just "ffmpeg", assume it's in PATH.
    # If it's a relative path, it should be relative to the project root.
    # For simplicity, this example assumes if it's not "ffmpeg", it's an absolute path or resolvable.
    # A more robust solution might involve checking os.path.isabs and joining with a base dir.
    if ffmpeg_path != "ffmpeg" and not os.path.isabs(ffmpeg_path):
         # This assumes 'settings.py' is at a level where this relative path makes sense
         # Or that FFMPEG_PATH is configured to be absolute or findable in PATH
        pass # Keep as is, or try to resolve relative to a known project root
    return ffmpeg_path

def _run_ffmpeg_process(command: List[str]) -> Tuple[bool, str, str]:
    """
    Runs an FFmpeg command using subprocess.
    Returns: (success_bool, stdout_str, stderr_str)
    """
    try:
        print(f"Running FFmpeg command: {' '.join(command)}")
        process = subprocess.Popen(
            command, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True, 
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        )
        stdout, stderr = process.communicate()
        if process.returncode == 0:
            return True, stdout, stderr
        else:
            print(f"FFmpeg error (code {process.returncode}): {stderr.strip()}")
            return False, stdout, stderr
    except FileNotFoundError:
        msg = f"Error: FFmpeg executable not found at '{command[0]}'. Please check FFMPEG_PATH in settings."
        print(msg)
        return False, "", msg
    except Exception as e:
        msg = f"Error during FFmpeg execution: {e}"
        print(msg)
        return False, "", msg

def _get_temp_output_path(input_video_path: str, operation_suffix: str) -> str:
    """Generates a temporary output path for FFmpeg operations."""
    base_name = os.path.basename(input_video_path)
    name, ext = os.path.splitext(base_name)
    # Place temp files in a 'temp' subdirectory of the input video's directory
    # or a global temp directory if preferred. For this example, using video's dir.
    temp_dir = os.path.join(os.path.dirname(input_video_path), "temp_processing")
    os.makedirs(temp_dir, exist_ok=True)
    return os.path.join(temp_dir, f"{name}_{operation_suffix}{ext}")


def crop_video_from_overlay(
    current_video_path: str,
    overlay_x_norm: float, overlay_y_norm: float, # Normalized 0-1 relative to displayed video
    overlay_w_norm: float, overlay_h_norm: float, # Normalized 0-1 relative to displayed video
    displayed_video_w: int, displayed_video_h: int, # Actual pixel size of video as displayed
    video_orig_w: int, video_orig_h: int,
    player_content_w: int, player_content_h: int # Dimensions of the container holding the video player
) -> Tuple[bool, str, Optional[str]]:
    """
    Crops a video based on normalized overlay coordinates.
    The overlay coordinates are relative to the *displayed* video within the player.
    Handles scaling calculations to map displayed coordinates to original video coordinates.
    Returns (success, message, output_path_or_none).
    """
    ffmpeg_exe = _get_ffmpeg_exe_path()
    if not os.path.exists(current_video_path):
        return False, "Input video not found.", None

    # Calculate the scale factor and offsets of the displayed video within the player_content area
    # This logic mirrors how a video player might fit content (letterboxing/pillarboxing)
    video_aspect = video_orig_w / video_orig_h
    player_aspect = player_content_w / player_content_h

    actual_disp_w: float
    actual_disp_h: float
    offset_x_player: float = 0
    offset_y_player: float = 0

    if video_aspect > player_aspect: # Video is wider than player area (letterboxed if height matches)
        actual_disp_w = player_content_w
        actual_disp_h = actual_disp_w / video_aspect
        offset_y_player = (player_content_h - actual_disp_h) / 2
    else: # Video is taller than player area (pillarboxed if width matches)
        actual_disp_h = player_content_h
        actual_disp_w = actual_disp_h * video_aspect
        offset_x_player = (player_content_w - actual_disp_w) / 2
    
    # Convert normalized overlay coordinates (relative to player_content_w/h)
    # to pixel coordinates on the *actual displayed video*
    # The provided overlay_x/y/w/h are assumed to be relative to player_content_w/h
    # and the overlay is drawn on top of the (potentially letter/pillarboxed) video.

    # Crop coordinates in terms of the original video dimensions
    # First, map overlay coordinates from player_content space to displayed_video space
    # The overlay_x_norm, etc. are actually pixel values from the GestureDetector
    # which is sized to player_content_w/h.
    
    # Let's assume overlay_x_norm etc. are already pixel values from the GestureDetector (e.g., control.left)
    # These are overlay_x_px, overlay_y_px, overlay_w_px, overlay_h_px
    
    # Scale factor from original video to displayed video
    scale_to_displayed_w = actual_disp_w / video_orig_w
    scale_to_displayed_h = actual_disp_h / video_orig_h
    
    # Crop coordinates in original video pixels
    crop_orig_x = (overlay_x_norm - offset_x_player) / scale_to_displayed_w
    crop_orig_y = (overlay_y_norm - offset_y_player) / scale_to_displayed_h
    crop_orig_w = overlay_w_norm / scale_to_displayed_w
    crop_orig_h = overlay_h_norm / scale_to_displayed_h

    # Ensure crop dimensions are positive and within bounds
    crop_orig_x = max(0, crop_orig_x)
    crop_orig_y = max(0, crop_orig_y)
    crop_orig_w = min(crop_orig_w, video_orig_w - crop_orig_x)
    crop_orig_h = min(crop_orig_h, video_orig_h - crop_orig_y)

    # Ensure width and height are even and at least 2x2
    target_crop_w = math.floor(crop_orig_w / 2) * 2
    target_crop_h = math.floor(crop_orig_h / 2) * 2
    target_crop_x = math.floor(crop_orig_x / 2) * 2
    target_crop_y = math.floor(crop_orig_y / 2) * 2

    if target_crop_w < 2 or target_crop_h < 2:
        return False, f"Crop dimensions too small ({target_crop_w}x{target_crop_h}). Min 2x2.", None

    temp_output_path = _get_temp_output_path(current_video_path, "cropped_overlay")
    
    vf_filters = []
    # Pre-scaling if original video is smaller than target crop (unlikely with overlay but for consistency)
    # This part is more relevant for crop_video_to_dimensions. For overlay, we crop what's selected.
    # However, if the selected area *implies* an upscale, FFmpeg handles it with 'crop'.

    vf_filters.append(f"crop={target_crop_w}:{target_crop_h}:{target_crop_x}:{target_crop_y}")
    vf_filters.append("pad=ceil(iw/2)*2:ceil(ih/2)*2") # Ensure final dimensions are even

    command = [
        ffmpeg_exe, "-y", "-i", current_video_path,
        "-vf", ",".join(vf_filters),
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-c:a", "aac", "-b:a", "128k", # Or copy audio: "-c:a", "copy"
        temp_output_path
    ]

    success, _, stderr = _run_ffmpeg_process(command)
    if success and os.path.exists(temp_output_path):
        return True, "Video cropped successfully from overlay selection!", temp_output_path
    else:
        if os.path.exists(temp_output_path): os.remove(temp_output_path) # Clean up
        return False, f"FFmpeg error during overlay crop: {stderr.strip()}", None


def crop_video_to_dimensions(
    current_video_path: str, 
    target_width: int, 
    target_height: int
) -> Tuple[bool, str, Optional[str]]:
    """
    Crops and scales a video to target_width and target_height.
    Scales video so smallest side matches target, then center crops.
    Returns (success, message, output_path_or_none).
    """
    ffmpeg_exe = _get_ffmpeg_exe_path()
    metadata = get_video_metadata(current_video_path)
    if not metadata or metadata['width'] <= 0 or metadata['height'] <= 0:
        return False, "Could not get valid original video dimensions.", None

    orig_width, orig_height = metadata['width'], metadata['height']
    
    # Scaling logic: scale so that the video covers the target area, then crop.
    scale_w_factor = target_width / orig_width
    scale_h_factor = target_height / orig_height
    scale_factor = max(scale_w_factor, scale_h_factor) # Ensure video covers target area

    scaled_width = math.ceil(orig_width * scale_factor / 2) * 2 # Ensure even
    scaled_height = math.ceil(orig_height * scale_factor / 2) * 2 # Ensure even

    temp_output_path = _get_temp_output_path(current_video_path, "cropped_dim")
    
    filters = [
        f"scale={scaled_width}:{scaled_height}:flags=lanczos",
        f"crop={target_width}:{target_height}", # Center crop from scaled video
        "pad=ceil(iw/2)*2:ceil(ih/2)*2" # Ensure final dimensions are even
    ]
    scale_crop_filter = ','.join(filters)

    command = [
        ffmpeg_exe, "-y", "-i", current_video_path,
        "-vf", scale_crop_filter,
        "-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k", # Or copy: "-c:a", "copy"
        temp_output_path
    ]

    success, _, stderr = _run_ffmpeg_process(command)
    if success and os.path.exists(temp_output_path):
        return True, "Video cropped successfully to dimensions!", temp_output_path
    else:
        if os.path.exists(temp_output_path): os.remove(temp_output_path)
        return False, f"FFmpeg error during dimension crop: {stderr.strip()}", None


def flip_video_horizontal(current_video_path: str) -> Tuple[bool, str, Optional[str]]:
    """Flips video horizontally. Returns (success, message, output_path_or_none)."""
    ffmpeg_exe = _get_ffmpeg_exe_path()
    temp_output_path = _get_temp_output_path(current_video_path, "flipped")
    command = [
        ffmpeg_exe, "-y", "-i", current_video_path,
        "-vf", "hflip",
        "-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        "-c:a", "copy", # Copy audio stream
        temp_output_path
    ]
    success, _, stderr = _run_ffmpeg_process(command)
    if success and os.path.exists(temp_output_path):
        return True, "Video flipped horizontally.", temp_output_path
    else:
        if os.path.exists(temp_output_path): os.remove(temp_output_path)
        return False, f"FFmpeg error during flip: {stderr.strip()}", None

def rotate_video_90(current_video_path: str, direction: str) -> Tuple[bool, str, Optional[str]]:
    """
    Rotates video by 90 degrees.
    direction: 'plus' for 90 degrees clockwise, 'minus' for 90 degrees counter-clockwise.
    Returns (success, message, output_path_or_none).
    """
    ffmpeg_exe = _get_ffmpeg_exe_path()
    temp_output_path = _get_temp_output_path(current_video_path, f"rotated_{direction}")
    
    transpose_value = ""
    message = ""
    if direction == 'plus':
        transpose_value = "transpose=1" # Rotate 90 degrees clockwise
        message = "Video rotated 90 degrees clockwise."
    elif direction == 'minus':
        transpose_value = "transpose=2" # Rotate 90 degrees counter-clockwise
        message = "Video rotated 90 degrees counter-clockwise."
    else:
        return False, "Invalid rotation direction. Use 'plus' or 'minus'.", None

    command = [
        ffmpeg_exe, "-y", "-i", current_video_path,
        "-vf", transpose_value,
        "-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        "-c:a", "copy", # Copy audio stream
        temp_output_path
    ]
    success, _, stderr = _run_ffmpeg_process(command)
    if success and os.path.exists(temp_output_path):
        return True, message, temp_output_path
    else:
        if os.path.exists(temp_output_path): os.remove(temp_output_path)
        return False, f"FFmpeg error during rotation: {stderr.strip()}", None

def reverse_video(current_video_path: str) -> Tuple[bool, str, Optional[str]]:
    """Reverses video. Returns (success, message, output_path_or_none)."""
    ffmpeg_exe = _get_ffmpeg_exe_path()
    temp_output_path = _get_temp_output_path(current_video_path, "reversed")
    command = [
        ffmpeg_exe, "-y", "-i", current_video_path,
        "-vf", "reverse", 
        # Reversing audio can be complex and sometimes undesirable.
        # "-af", "areverse", # Optionally reverse audio
        "-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        "-c:a", "copy", # Typically copy audio or omit if audio reversal is not needed / problematic
        temp_output_path
    ]
    success, _, stderr = _run_ffmpeg_process(command)
    if success and os.path.exists(temp_output_path):
        return True, "Video reversed.", temp_output_path
    else:
        if os.path.exists(temp_output_path): os.remove(temp_output_path)
        return False, f"FFmpeg error during reverse: {stderr.strip()}", None

def time_remap_video_by_speed(current_video_path: str, speed_multiplier: float) -> Tuple[bool, str, Optional[str]]:
    """
    Remaps video timing by a speed multiplier (e.g., 0.5 for half speed, 2.0 for double speed).
    Returns (success, message, output_path_or_none).
    """
    if speed_multiplier <= 0:
        return False, "Speed multiplier must be positive.", None

    ffmpeg_exe = _get_ffmpeg_exe_path()
    metadata = get_video_metadata(current_video_path)
    if not metadata or not metadata.get('fps'):
        return False, "Could not get original video FPS for remapping.", None
    
    original_fps = metadata['fps']
    
    # setpts changes presentation timestamps. 1/speed_multiplier for setpts.
    # e.g. speed_multiplier = 2 (double speed) => setpts=0.5*PTS
    # e.g. speed_multiplier = 0.5 (half speed) => setpts=2*PTS
    pts_factor = 1.0 / speed_multiplier
    
    # atempo filter for audio, range 0.5-100.0. Chain for larger changes.
    # For simplicity, handling common cases. More complex audio speed adjustment might be needed.
    audio_filter = f"atempo={speed_multiplier}"
    if speed_multiplier > 2.0: # Chain atempo for > 2x speed up
        audio_filter = f"atempo=2.0,atempo={speed_multiplier/2.0}" # Example for up to 4x
    elif speed_multiplier < 0.5: # Chain atempo for < 0.5x slow down
         audio_filter = f"atempo=0.5,atempo={speed_multiplier/0.5}" # Example for down to 0.25x


    temp_output_path = _get_temp_output_path(current_video_path, "remapped")
    command = [
        ffmpeg_exe, "-y", "-i", current_video_path,
        "-vf", f"setpts={pts_factor}*PTS",
        "-af", audio_filter,
        "-r", str(original_fps), # Keep original FPS, duration changes
        "-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        # Audio codec might need to be specified if not aac, or if issues with atempo
        "-c:a", "aac", "-b:a", "128k", 
        temp_output_path
    ]
    success, _, stderr = _run_ffmpeg_process(command)
    if success and os.path.exists(temp_output_path):
        return True, f"Video time remapped by factor {speed_multiplier}.", temp_output_path
    else:
        if os.path.exists(temp_output_path): os.remove(temp_output_path)
        return False, f"FFmpeg error during time remap: {stderr.strip()}", None

def cut_video_by_frames(
    current_video_path: str, 
    start_frame: int, 
    end_frame: int
) -> Tuple[bool, str, Optional[str]]:
    """
    Cuts video from start_frame to end_frame.
    Returns (success, message, output_path_or_none).
    """
    ffmpeg_exe = _get_ffmpeg_exe_path()
    metadata = get_video_metadata(current_video_path)
    if not metadata or not metadata.get('fps') or metadata['fps'] <= 0:
        return False, "Could not get valid FPS for cutting by frames.", None
    
    fps = metadata['fps']
    if start_frame >= end_frame:
        return False, "Start frame must be less than end frame.", None

    start_time = start_frame / fps
    # To make the cut inclusive of the end_frame, the duration or -to time needs careful calculation.
    # Using -to (end time) is often more intuitive.
    end_time = end_frame / fps 
    # If end_frame is the very last frame, ensure end_time is slightly beyond it or use frame numbers directly if ffmpeg supports.
    # For -to, it specifies the time to stop writing output.

    temp_output_path = _get_temp_output_path(current_video_path, "cut")
    command = [
        ffmpeg_exe, "-y", 
        "-i", current_video_path, # Input after -ss for faster seeking for some formats
        "-ss", str(start_time),   # Seek to start time
        "-to", str(end_time),     # Cut up to end time
        "-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        "-c:a", "copy", # Copy audio stream
        temp_output_path
    ]
    
    success, _, stderr = _run_ffmpeg_process(command)
    if success and os.path.exists(temp_output_path):
        return True, f"Video cut from frame {start_frame} to {end_frame}.", temp_output_path
    else:
        if os.path.exists(temp_output_path): os.remove(temp_output_path)
        return False, f"FFmpeg error during cut: {stderr.strip()}", None


def split_video_by_frame(
    current_video_path: str, 
    split_frame: int
) -> Tuple[bool, str, Optional[str], Optional[str]]:
    """
    Splits a video into two parts at split_frame.
    The first part is from frame 0 to split_frame-1.
    The second part is from split_frame to the end.
    Returns (success, message, output_path_part1_or_none, output_path_part2_or_none).
    """
    ffmpeg_exe = _get_ffmpeg_exe_path()
    metadata = get_video_metadata(current_video_path)
    if not metadata or not metadata.get('fps') or metadata['fps'] <= 0 or not metadata.get('total_frames'):
        return False, "Could not get valid metadata for splitting.", None, None
    
    fps = metadata['fps']
    total_frames = metadata['total_frames']

    if split_frame <= 0 or split_frame >= total_frames:
        return False, "Split frame must be within the video's frame range (exclusive of 0 and total_frames).", None, None

    # Part 1: 0 to split_frame (exclusive of split_frame itself for time calc)
    start_time_1 = 0
    end_time_1 = split_frame / fps 
    temp_output_path_1 = _get_temp_output_path(current_video_path, "split_part1")
    command1 = [
        ffmpeg_exe, "-y", "-i", current_video_path,
        "-ss", str(start_time_1), "-to", str(end_time_1),
        "-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        "-c:a", "copy", temp_output_path_1
    ]
    success1, _, stderr1 = _run_ffmpeg_process(command1)
    if not (success1 and os.path.exists(temp_output_path_1)):
        if os.path.exists(temp_output_path_1): os.remove(temp_output_path_1)
        return False, f"FFmpeg error during split (part 1): {stderr1.strip()}", None, None

    # Part 2: split_frame to end
    start_time_2 = split_frame / fps
    # No -to needed, goes to end of stream
    temp_output_path_2 = _get_temp_output_path(current_video_path, "split_part2")
    command2 = [
        ffmpeg_exe, "-y", "-i", current_video_path,
        "-ss", str(start_time_2),
        "-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        "-c:a", "copy", temp_output_path_2
    ]
    success2, _, stderr2 = _run_ffmpeg_process(command2)
    if not (success2 and os.path.exists(temp_output_path_2)):
        if os.path.exists(temp_output_path_1): os.remove(temp_output_path_1) # Clean up part 1
        if os.path.exists(temp_output_path_2): os.remove(temp_output_path_2)
        return False, f"FFmpeg error during split (part 2): {stderr2.strip()}", temp_output_path_1, None
        
    return True, f"Video split at frame {split_frame}.", temp_output_path_1, temp_output_path_2
