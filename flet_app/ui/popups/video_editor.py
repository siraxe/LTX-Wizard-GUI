import flet as ft
from ui._styles import create_textfield, create_styled_button, BTN_STYLE2
import os
import shutil
import subprocess
import cv2

def handle_size_add(width_field, height_field, current_video_path, page=None):
    """
    Increase both width and height fields by the closest 32 multiple, maintaining aspect ratio if possible.
    """
    try:
        w = int(width_field.value) if width_field.value else 32
        h = int(height_field.value) if height_field.value else 32
        # Try to get aspect ratio from video
        aspect = None
        if current_video_path and os.path.exists(current_video_path):
            cap = cv2.VideoCapture(current_video_path)
            if cap.isOpened():
                orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if orig_height > 0:
                    aspect = orig_width / orig_height
                cap.release()
        if aspect:
            # Increase width, calculate height to keep aspect
            w = ((w // 32) + 1) * 32
            h = max(32, int(round(w / aspect / 32)) * 32)
        else:
            w = ((w // 32) + 1) * 32
            h = ((h // 32) + 1) * 32
        width_field.value = str(w)
        height_field.value = str(h)
    except Exception:
        width_field.value = "32"
        height_field.value = "32"
    if page:
        page.update()

def handle_size_sub(width_field, height_field, current_video_path, page=None):
    """
    Decrease both width and height fields by the closest 32 multiple, maintaining aspect ratio if possible.
    """
    try:
        w = int(width_field.value) if width_field.value else 32
        h = int(height_field.value) if height_field.value else 32
        # Try to get aspect ratio from video
        aspect = None
        if current_video_path and os.path.exists(current_video_path):
            cap = cv2.VideoCapture(current_video_path)
            if cap.isOpened():
                orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if orig_height > 0:
                    aspect = orig_width / orig_height
                cap.release()
        if aspect:
            # Decrease width, calculate height to keep aspect
            w = max(32, ((w - 1) // 32) * 32)
            h = max(32, int(round(w / aspect / 32)) * 32)
        else:
            w = max(32, ((w - 1) // 32) * 32)
            h = max(32, ((h - 1) // 32) * 32)
        width_field.value = str(w)
        height_field.value = str(h)
    except Exception:
        width_field.value = "32"
        height_field.value = "32"
    if page:
        page.update()

def build_crop_controls_row(page, current_video_path, on_crop, on_crop_all=None, on_get_closest=None):
    """
    Build a ResponsiveRow with width/height text fields and Crop/Crop All/Closest buttons.
    Returns (returnBox, width_field, height_field).
    This function always returns a tuple; it never returns None.
    """

    width_field = create_textfield(
        label="Width",
        value="",
        hint_text=None,
        multiline=False,
        min_lines=1,
        max_lines=1,
        expand=True,
        col=8,
        on_change=None,
        tooltip=None
    )
    width_field.keyboard_type = ft.KeyboardType.NUMBER

    height_field = create_textfield(
        label="Height",
        value="",
        hint_text=None,
        multiline=False,
        min_lines=1,
        max_lines=1,
        expand=True,
        col=8,
        on_change=None,
        tooltip=None
    )
    height_field.keyboard_type = ft.KeyboardType.NUMBER

    # Define add/sub callbacks that close over the required values
    def add_callback(e):
        handle_size_add(width_field, height_field, current_video_path, page)
    def sub_callback(e):
        handle_size_sub(width_field, height_field, current_video_path, page)

    add_button = create_styled_button(
        text="+",
        on_click=add_callback,
        col=4,
        button_style=BTN_STYLE2
    )

    substract_button = create_styled_button(
        text="-",
        on_click=sub_callback,
        col=4,
        button_style=BTN_STYLE2
    )

    crop_button = create_styled_button(
        text="Crop",
        on_click=on_crop,
        col=3,
        button_style=BTN_STYLE2
    )

    crop_all_button = create_styled_button(
        text="Crop All",
        on_click=on_crop_all,
        col=5,
        button_style=BTN_STYLE2
    )
    closes_button = create_styled_button(
        text="Closest",
        on_click=on_get_closest if on_get_closest is not None else (lambda e: None),
        col=4,  
        button_style=BTN_STYLE2
    )

    crop_buttons_row = ft.ResponsiveRow(
        controls=[crop_all_button, crop_button, closes_button])

    crop_controls_row = ft.ResponsiveRow(
        controls=[  ft.Column([ft.ResponsiveRow([width_field, add_button]),
                    ft.ResponsiveRow([height_field, substract_button])]),
                    ft.Column([crop_buttons_row])],
        alignment=ft.MainAxisAlignment.CENTER,
        spacing=10,
        col=5
    )
    time_controls_row = ft.ResponsiveRow(
        controls=[ft.Text("Time")],
        alignment=ft.MainAxisAlignment.CENTER,
        spacing=10,
        col=7
    )
    returnBox = ft.ResponsiveRow(
        controls=[crop_controls_row, time_controls_row])
    return returnBox, width_field, height_field


def handle_set_closest_div32(width_field, height_field, current_video_path, page=None):
    """
    Reads the video's resolution, finds the closest width and height divisible by 32,
    and populates width_field and height_field with those values.
    """
    if not current_video_path or not os.path.exists(current_video_path):
        return
    cap = cv2.VideoCapture(current_video_path)
    if not cap.isOpened():
        return
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    # Find closest divisible by 32
    def closest_div32(val):
        return max(32, int(round(val / 32)) * 32)
    width32 = closest_div32(width)
    height32 = closest_div32(height)
    width_field.value = str(width32)
    height_field.value = str(height32)
    if page:
        page.update()

def handle_crop_all_videos(page: ft.Page, width_field, height_field, video_list):
    """
    Crops all videos in video_list using the current width/height field values.
    """
    if not video_list:
        return
    for video_path in video_list:
        handle_crop_video_click(page, width_field, height_field, video_path)

def handle_crop_video_click(page: ft.Page, width_field, height_field, current_video_path):
    """
    Handles cropping logic: reads dimensions, runs FFmpeg to crop/scale and encode video, and overwrites the original file.
    """


    if not width_field or not height_field or not current_video_path:
        return
    try:
        target_width = int(width_field.value)
        target_height = int(height_field.value)
        if target_width <= 0 or target_height <= 0:
            raise ValueError("Dimensions must be positive integers.")
    except ValueError:
        return

    temp_output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "storage", "temp")
    os.makedirs(temp_output_dir, exist_ok=True)
    ffmpeg_exe = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "ffmpeg", "bin", "ffmpeg.exe"))
    if not os.path.exists(ffmpeg_exe):
        return
    ffmpeg_output_path = os.path.join(temp_output_dir, f"ffmpeg_cropped_{os.path.basename(current_video_path)}")
    input_abs_path = os.path.abspath(current_video_path)
    output_abs_path = os.path.abspath(ffmpeg_output_path)
    # First scale to fit within target dimensions (preserve aspect), then crop center to target size
    # This ensures the result always fills the box without stretching
    scale_crop_filter = (
        f"scale='if(gt(a,{target_width}/{target_height}),{target_width},-1)':'if(gt(a,{target_width}/{target_height}),-1,{target_height})',"
        f"crop={target_width}:{target_height}"
    )
    cmd_list = [
        ffmpeg_exe, "-y", "-i", input_abs_path,
        "-vf", scale_crop_filter,
        "-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        output_abs_path
    ]
    def log_debug(msg):
        with open("crop_debug.log", "a", encoding="utf-8") as f:
            f.write(msg + "\n")
        print(msg)

    log_debug(f"Running FFmpeg: {' '.join(cmd_list)}")
    try:
        process = subprocess.Popen(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        stdout, stderr = process.communicate()
        ffmpeg_result = process.returncode
        log_debug(f"FFmpeg return code: {ffmpeg_result}")
        log_debug(f"FFmpeg stderr: {stderr.decode('utf-8', errors='ignore')}")
    except FileNotFoundError:
        log_debug("FFmpeg executable not found!")
        return
    except Exception as e:
        log_debug(f"Exception running FFmpeg: {e}")
        return
    if ffmpeg_result != 0 or not os.path.exists(output_abs_path):
        log_debug("FFmpeg failed or output file not created.")
        return
    try:
        shutil.move(output_abs_path, input_abs_path)
    except Exception:
        if os.path.exists(output_abs_path):
            try:
                os.remove(output_abs_path)
            except Exception:
                pass
    # --- Update info.json with new width/height ---
    import json
    video_dir = os.path.dirname(input_abs_path)
    info_json_path = os.path.join(video_dir, 'info.json')
    info_data = {}
    if os.path.exists(info_json_path):
        try:
            with open(info_json_path, 'r', encoding='utf-8') as f:
                info_data = json.load(f)
        except Exception:
            info_data = {}
    video_filename = os.path.basename(input_abs_path)
    if video_filename not in info_data:
        info_data[video_filename] = {}
    info_data[video_filename]['width'] = target_width
    info_data[video_filename]['height'] = target_height
    # Optionally, update frame count
    try:
        cap = cv2.VideoCapture(input_abs_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        info_data[video_filename]['frames'] = frame_count
    except Exception:
        pass
    try:
        with open(info_json_path, 'w', encoding='utf-8') as f:
            json.dump(info_data, f, indent=2, ensure_ascii=False)
    except Exception:
        pass
    # --- End update info.json ---
    if page:
        # --- Force video player popup to reload the current video ---
        try:
            from ui.popups.video_player_dialog import open_video_captions_dialog
            from ui.tab_dataset_view import update_thumbnails, thumbnails_grid_ref, selected_dataset
            # Try to get the dataset name from the video path
            dataset_name = None
            if current_video_path:
                # e.g., workspace/datasets/<dataset_name>/<video_file>
                parts = os.path.normpath(current_video_path).split(os.sep)
                if "datasets" in parts:
                    idx = parts.index("datasets")
                    if idx + 1 < len(parts):
                        dataset_name = parts[idx + 1]
            # Regenerate all thumbnails for this dataset
            try:
                from ui.utils.utils_datasets import regenerate_all_thumbnails_for_dataset
                regenerate_all_thumbnails_for_dataset(dataset_name)
            except Exception as thumb_exc:
                print(f"Could not regenerate all thumbnails: {thumb_exc}")
            # Refresh the video popup if possible
            open_video_captions_dialog(page, current_video_path, [current_video_path])
            # Update thumbnails for the dataset if possible
            if dataset_name and thumbnails_grid_ref.current:
                update_thumbnails(page_ctx=page, grid_control=thumbnails_grid_ref.current, force_refresh=True)
                thumbnails_grid_ref.current.update()
                page.update()
        except Exception as e:
            print(f"Could not refresh video player or thumbnails: {e}")
        page.update()
