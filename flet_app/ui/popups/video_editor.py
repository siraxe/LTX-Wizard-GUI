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

def build_crop_controls_row(page, current_video_path, on_crop, on_crop_all=None, on_get_closest=None, video_list: list = None, on_caption_updated_callback: callable = None):
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


    # Get video frame count for RangeSlider
    frame_count = 100 # Default value if video cannot be read
    if current_video_path and os.path.exists(current_video_path):
        cap = cv2.VideoCapture(current_video_path)
        if cap.isOpened():
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

    # Range slider for selecting frame range
    frame_range_slider = ft.RangeSlider(
        min=0,
        max=frame_count,
        start_value=0,
        end_value=frame_count,
        divisions=frame_count, # Optional: shows markers for each frame
        label="{value}",  # Hide label above thumbs
        round=0, # Display integer frame numbers
        expand=True,
        on_change_start=None,
        on_change=None,
        on_change_end=None,
    )

    # Text controls to show start and end values
    start_value_text = ft.Text(f"Start: {frame_range_slider.start_value}", size=12)
    end_value_text = ft.Text(f"End: {frame_range_slider.end_value}", size=12)
    # Text control to show total frames
    total_frames_text = ft.Text(f"Total: {int(frame_range_slider.end_value - frame_range_slider.start_value)}", size=12)
    # time_remap_value_text will be updated by time_slider
    time_remap_value_text = ft.Text(f"Frames after Remap: {frame_count}", size=12)

    # Handlers for RangeSlider events (must be after frame_range_slider is defined)
    def slider_is_changing(e):
        start_value_text.value = f"Start: {int(e.control.start_value)}"
        end_value_text.value = f"End: {int(e.control.end_value)}"
        total_frames_text.value = f"Total: {int(e.control.end_value - e.control.start_value)}"
        start_value_text.update()
        end_value_text.update()
        total_frames_text.update()
        frame_range_slider.update()

    frame_range_slider.on_change = slider_is_changing

    # Handler for time_slider
    def time_slider_is_changing(e):
        percent = e.control.value
        remapped_frames = int(frame_count * (percent / 100))
        time_remap_value_text.value = f"Frames after Remap: {remapped_frames}"
        time_remap_value_text.update()

    time_slider = ft.Slider(
        value=100,
        min=1,
        max=200,
        divisions=200,
        label="{value}%",
        on_change=time_slider_is_changing,
    )

    frame_slider = ft.Column(
        controls=[
            frame_range_slider,
            ft.Row([start_value_text, total_frames_text, end_value_text], alignment=ft.MainAxisAlignment.SPACE_BETWEEN)
        ],
        alignment=ft.MainAxisAlignment.CENTER,
        spacing=5,
        col=4
    )

    # Flip horizontally button
    flip_horizontal_button = create_styled_button(
        text="Flip Horizontal",
        on_click=lambda e: on_flip_horizontal(e, page, current_video_path, video_list, on_caption_updated_callback),
        col=6,
        button_style=BTN_STYLE2
    )
    cut_to_frames_button = create_styled_button(
        text="Cut to Frames",
        on_click=lambda e: cut_to_frames(e, page, current_video_path, frame_range_slider, video_list, on_caption_updated_callback),
        col=6,
        button_style=BTN_STYLE2
    )
    # Reverse button
    reverse_button = create_styled_button(
        text="Reverse",
        on_click=lambda e: on_reverse(e, page, current_video_path, video_list, on_caption_updated_callback),
        col=6,
        button_style=BTN_STYLE2
    )
    time_remap_button = create_styled_button(
        text="Time Remap",
        on_click=lambda e: on_time_remap(e, page, current_video_path, time_slider, video_list, on_caption_updated_callback),
        col=6,
        button_style=BTN_STYLE2
    )

    crop_controls_row = ft.ResponsiveRow(
        controls=[  ft.Column([ft.ResponsiveRow([width_field, add_button]),
                    ft.ResponsiveRow([height_field, substract_button])]),
                    ft.Column([crop_buttons_row])],
        alignment=ft.MainAxisAlignment.CENTER,
        spacing=5,
        col=4,
        scale=0.9
    )

    frame_controls_column = ft.Column(
        controls=[frame_slider,ft.ResponsiveRow([flip_horizontal_button, cut_to_frames_button])],
        alignment=ft.MainAxisAlignment.CENTER,
        spacing=5,
        col=4,
        scale=0.9
    )

    # Restore original time_controls_column layout with slider, remap text, and buttons
    time_controls_column = ft.Column(
        controls=[
            ft.ResponsiveRow([
                time_slider,
                ft.Row([time_remap_value_text], 
                alignment=ft.MainAxisAlignment.CENTER,
                spacing=5
                )
            ]),
            ft.ResponsiveRow([reverse_button, time_remap_button])
        ],
        alignment=ft.MainAxisAlignment.CENTER,
        spacing=5,
        col=4,
        scale=0.9
    )
    
    returnBox = ft.ResponsiveRow(
        controls=[crop_controls_row,
                  frame_controls_column,
                  time_controls_column],
                  spacing=3
                  )
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

def on_flip_horizontal(e, page: ft.Page, current_video_path: str, video_list: list, on_caption_updated_callback: callable):
    """
    Flips the current video horizontally using FFmpeg and replaces the original file.
    """
    if not current_video_path or not os.path.exists(current_video_path):
        print("No video path or file not found.")
        return

    temp_output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "storage", "temp")
    os.makedirs(temp_output_dir, exist_ok=True)
    ffmpeg_exe = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "ffmpeg", "bin", "ffmpeg.exe"))

    if not os.path.exists(ffmpeg_exe):
        print("FFmpeg executable not found!")
        return

    input_abs_path = os.path.abspath(current_video_path)
    ffmpeg_output_path = os.path.join(temp_output_dir, f"ffmpeg_flipped_{os.path.basename(current_video_path)}")
    output_abs_path = os.path.abspath(ffmpeg_output_path)

    cmd_list = [
        ffmpeg_exe, "-y", "-i", input_abs_path,
        "-vf", "hflip", # Horizontal flip filter
        "-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        output_abs_path
    ]

    print(f"Running FFmpeg: {' '.join(cmd_list)}")

    try:
        process = subprocess.Popen(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        stdout, stderr = process.communicate()
        ffmpeg_result = process.returncode
        print(f"FFmpeg return code: {ffmpeg_result}")
        print(f"FFmpeg stderr: {stderr.decode('utf-8', errors='ignore')}")

        if ffmpeg_result != 0 or not os.path.exists(output_abs_path):
            print("FFmpeg failed or output file not created.")
            return

        # Replace original file with flipped video
        shutil.move(output_abs_path, input_abs_path)
        print(f"Successfully flipped and replaced {current_video_path}")

    except FileNotFoundError:
        print("FFmpeg executable not found!")
        return
    except Exception as e:
        print(f"Exception running FFmpeg: {e}")
        if os.path.exists(output_abs_path):
            try:
                os.remove(output_abs_path)
            except Exception:
                pass
        return

    # --- Update UI: Refresh video player and thumbnails ---
    if page:
        try:
            from ui.popups.video_player_dialog import open_video_captions_dialog
            from ui.tab_dataset_view import update_thumbnails, thumbnails_grid_ref, selected_dataset
            from ui.utils.utils_datasets import regenerate_all_thumbnails_for_dataset

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
            if dataset_name:
                try:
                    regenerate_all_thumbnails_for_dataset(dataset_name)
                except Exception as thumb_exc:
                    print(f"Could not regenerate all thumbnails: {thumb_exc}")

            # Refresh the video popup if possible
            open_video_captions_dialog(page, current_video_path, video_list, on_caption_updated_callback)

            # Update thumbnails for the dataset if possible
            if dataset_name and thumbnails_grid_ref.current:
                update_thumbnails(page_ctx=page, grid_control=thumbnails_grid_ref.current, force_refresh=True)
                thumbnails_grid_ref.current.update()
                page.update()

        except Exception as e:
            print(f"Could not refresh video player or thumbnails: {e}")

        page.update()

def on_reverse(e, page: ft.Page, current_video_path: str, video_list: list, on_caption_updated_callback: callable):
    """
    Reverses the current video using FFmpeg and replaces the original file.
    """
    import json
    if not current_video_path or not os.path.exists(current_video_path):
        print("No video path or file not found.")
        return

    temp_output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "storage", "temp")
    os.makedirs(temp_output_dir, exist_ok=True)
    ffmpeg_exe = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "ffmpeg", "bin", "ffmpeg.exe"))

    if not os.path.exists(ffmpeg_exe):
        print("FFmpeg executable not found!")
        return

    input_abs_path = os.path.abspath(current_video_path)
    ffmpeg_output_path = os.path.join(temp_output_dir, f"ffmpeg_reversed_{os.path.basename(current_video_path)}")
    output_abs_path = os.path.abspath(ffmpeg_output_path)

    # FFmpeg command to reverse video
    cmd_list = [
        ffmpeg_exe, "-y", "-i", input_abs_path,
        "-vf", "reverse", # Reverse video filter
        "-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        output_abs_path
    ]

    print(f"Running FFmpeg: {' '.join(cmd_list)}")

    try:
        process = subprocess.Popen(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        stdout, stderr = process.communicate()
        ffmpeg_result = process.returncode
        print(f"FFmpeg return code: {ffmpeg_result}")
        print(f"FFmpeg stderr: {stderr.decode('utf-8', errors='ignore')}")

        if ffmpeg_result != 0 or not os.path.exists(output_abs_path):
            print("FFmpeg failed or output file not created.")
            return

        # Replace original file with reversed video
        shutil.move(output_abs_path, input_abs_path)
        print(f"Successfully reversed and replaced {current_video_path}")

    except FileNotFoundError:
        print("FFmpeg executable not found!")
        return
    except Exception as e:
        print(f"Exception running FFmpeg: {e}")
        if os.path.exists(output_abs_path):
            try:
                os.remove(output_abs_path)
            except Exception:
                pass
        return

    # --- Update info.json (optional, but good practice to re-read/write) ---
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
    # Re-read frames and fps just in case (reversing shouldn't change them)
    try:
        cap = cv2.VideoCapture(input_abs_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        info_data[video_filename]['frames'] = frame_count
        info_data[video_filename]['fps'] = fps
        info_data[video_filename]['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        info_data[video_filename]['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    except Exception:
        pass
    try:
        with open(info_json_path, 'w', encoding='utf-8') as f:
            json.dump(info_data, f, indent=2, ensure_ascii=False)
    except Exception:
        pass
    # --- End update info.json ---

    # --- Update UI: Refresh video player and thumbnails ---
    if page:
        try:
            from ui.popups.video_player_dialog import open_video_captions_dialog
            from ui.tab_dataset_view import update_thumbnails, thumbnails_grid_ref, selected_dataset
            from ui.utils.utils_datasets import regenerate_all_thumbnails_for_dataset

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
            if dataset_name:
                try:
                    regenerate_all_thumbnails_for_dataset(dataset_name)
                except Exception as thumb_exc:
                    print(f"Could not regenerate all thumbnails: {thumb_exc}")

            # Refresh the video popup if possible
            open_video_captions_dialog(page, current_video_path, video_list, on_caption_updated_callback)

            # Update thumbnails for the dataset if possible
            if dataset_name and thumbnails_grid_ref.current:
                update_thumbnails(page_ctx=page, grid_control=thumbnails_grid_ref.current, force_refresh=True)
                thumbnails_grid_ref.current.update()
                page.update()

        except Exception as e:
            print(f"Could not refresh video player or thumbnails: {e}")

        page.update()

def on_time_remap(e, page: ft.Page, current_video_path: str, time_slider, video_list: list, on_caption_updated_callback: callable):
    """
    Remap the video duration by changing its speed to match the remapped frame count (from time_slider).
    If remapped_frames < original: speed up (drop frames). If remapped_frames > original: slow down (duplicate frames).
    """
    import json
    if not current_video_path or not os.path.exists(current_video_path):
        print("No video path or file not found.")
        return

    # Get original frame count and FPS
    cap = cv2.VideoCapture(current_video_path)
    if not cap.isOpened():
        print("Could not open video file.")
        return
    original_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if original_frames <= 0 or original_fps <= 0:
        print("Could not get video frame count or FPS.")
        return

    # Get remapped_frames from the slider
    percent = time_slider.value
    remapped_frames = int(original_frames * (percent / 100))
    if remapped_frames <= 0:
        print("Remapped frame count is invalid.")
        return
    if remapped_frames == original_frames:
        print("Remapped frame count is the same as the original. No remapping needed.")
        return

    # Calculate duration and new fps for remapping - these are not directly used in filter
    # but helpful for understanding
    # duration = original_frames / original_fps
    # new_fps_if_duration_same = remapped_frames / duration

    # Use setpts filter to change speed (duration), keeping output FPS the same as original
    speed_factor_pts = remapped_frames / original_frames
    filter_str = f"setpts=PTS*{speed_factor_pts}"

    temp_output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "storage", "temp")
    os.makedirs(temp_output_dir, exist_ok=True)
    ffmpeg_exe = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "ffmpeg", "bin", "ffmpeg.exe"))
    if not os.path.exists(ffmpeg_exe):
        print("FFmpeg executable not found!")
        return
    input_abs_path = os.path.abspath(current_video_path)
    ffmpeg_output_path = os.path.join(temp_output_dir, f"ffmpeg_remap_{os.path.basename(current_video_path)}")
    output_abs_path = os.path.abspath(ffmpeg_output_path)

    cmd_list = [
        ffmpeg_exe, "-y", "-i", input_abs_path,
        "-vf", filter_str,
        "-r", str(original_fps), # Explicitly set output FPS to original
        "-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        output_abs_path
    ]
    print(f"Running FFmpeg: {' '.join(cmd_list)}")
    try:
        process = subprocess.Popen(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        stdout, stderr = process.communicate()
        ffmpeg_result = process.returncode
        print(f"FFmpeg return code: {ffmpeg_result}")
        print(f"FFmpeg stderr: {stderr.decode('utf-8', errors='ignore')}")
        if ffmpeg_result != 0 or not os.path.exists(output_abs_path):
            print("FFmpeg failed or output file not created.")
            return
        # Replace original file with remapped video
        shutil.move(output_abs_path, input_abs_path)
        print(f"Successfully remapped and replaced {current_video_path}")
    except FileNotFoundError:
        print("FFmpeg executable not found!")
        return
    except Exception as ex:
        print(f"Exception running FFmpeg: {ex}")
        if os.path.exists(output_abs_path):
            try:
                os.remove(output_abs_path)
            except Exception:
                pass
        return

    # --- Update info.json with new frame count and fps ---
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
    try:
        cap = cv2.VideoCapture(input_abs_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        info_data[video_filename]['frames'] = frame_count
        info_data[video_filename]['fps'] = fps
    except Exception:
        pass
    try:
        with open(info_json_path, 'w', encoding='utf-8') as f:
            json.dump(info_data, f, indent=2, ensure_ascii=False)
    except Exception:
        pass
    # --- End update info.json ---
    if page:
        try:
            from ui.popups.video_player_dialog import open_video_captions_dialog
            from ui.tab_dataset_view import update_thumbnails, thumbnails_grid_ref, selected_dataset
            from ui.utils.utils_datasets import regenerate_all_thumbnails_for_dataset
            # Try to get the dataset name from the video path
            dataset_name = None
            if current_video_path:
                parts = os.path.normpath(current_video_path).split(os.sep)
                if "datasets" in parts:
                    idx = parts.index("datasets")
                    if idx + 1 < len(parts):
                        dataset_name = parts[idx + 1]
            # Regenerate all thumbnails for this dataset
            if dataset_name:
                try:
                    regenerate_all_thumbnails_for_dataset(dataset_name)
                except Exception as thumb_exc:
                    print(f"Could not regenerate all thumbnails: {thumb_exc}")
            # Refresh the video popup if possible
            open_video_captions_dialog(page, current_video_path, video_list, on_caption_updated_callback)
            # Update thumbnails for the dataset if possible
            if dataset_name and thumbnails_grid_ref.current:
                update_thumbnails(page_ctx=page, grid_control=thumbnails_grid_ref.current, force_refresh=True)
                thumbnails_grid_ref.current.update()
                page.update()
        except Exception as e:
            print(f"Could not refresh video player or thumbnails: {e}")
        page.update()

def handle_crop_all_videos(page: ft.Page, width_field, height_field, video_list, on_caption_updated_callback: callable):
    """
    Crops all videos in video_list using the current width/height field values, with scaling up if needed (see handle_crop_video_click).
    """
    if not video_list:
        return
    for video_path in video_list:
        handle_crop_video_click(page, width_field, height_field, video_path, video_list=video_list, on_caption_updated_callback=on_caption_updated_callback)

def handle_crop_video_click(page: ft.Page, width_field, height_field, current_video_path, video_list: list = None, on_caption_updated_callback: callable = None):
    """
    Handles cropping logic: reads dimensions, runs FFmpeg to crop/scale and encode video, and overwrites the original file.
    Now also scales up proportionally if video is smaller than the target dimensions, so smallest side matches target, before cropping.
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

    # Get original video dimensions
    cap = cv2.VideoCapture(current_video_path)
    if not cap.isOpened():
        return
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if orig_width <= 0 or orig_height <= 0:
        return

    # Always scale so that the smallest side matches the corresponding target (up or down)
    scale_w = target_width / orig_width
    scale_h = target_height / orig_height
    scale_factor = max(scale_w, scale_h)
    scaled_width = int(round(orig_width * scale_factor))
    scaled_height = int(round(orig_height * scale_factor))

    temp_output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "storage", "temp")
    os.makedirs(temp_output_dir, exist_ok=True)
    ffmpeg_exe = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "ffmpeg", "bin", "ffmpeg.exe"))
    if not os.path.exists(ffmpeg_exe):
        return
    ffmpeg_output_path = os.path.join(temp_output_dir, f"ffmpeg_cropped_{os.path.basename(current_video_path)}")
    input_abs_path = os.path.abspath(current_video_path)
    output_abs_path = os.path.abspath(ffmpeg_output_path)

    # Compose FFmpeg filter chain
    filters = []
    filters.append(f"scale={scaled_width}:{scaled_height}")
    # Always crop to target size, center
    filters.append(f"crop={target_width}:{target_height}")
    scale_crop_filter = ','.join(filters)

    cmd_list = [
        ffmpeg_exe, "-y", "-i", input_abs_path,
        "-vf", scale_crop_filter,
        "-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        output_abs_path
    ]
    def log_debug(msg):
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
            open_video_captions_dialog(page, current_video_path, video_list, on_caption_updated_callback)
            # Update thumbnails for the dataset if possible
            if dataset_name and thumbnails_grid_ref.current:
                update_thumbnails(page_ctx=page, grid_control=thumbnails_grid_ref.current, force_refresh=True)
                thumbnails_grid_ref.current.update()
                page.update()
        except Exception as e:
            print(f"Could not refresh video player or thumbnails: {e}")
        page.update()

def cut_to_frames(e, page: ft.Page, current_video_path: str, frame_range_slider: ft.RangeSlider, video_list: list, on_caption_updated_callback: callable):
    """
    Cuts the current video to the specified frame range using FFmpeg and replaces the original file.
    """
    if not current_video_path or not os.path.exists(current_video_path):
        print("No video path or file not found.")
        return

    try:
        start_frame = int(frame_range_slider.start_value)
        end_frame = int(frame_range_slider.end_value)
        if start_frame >= end_frame:
            print("Start frame must be less than end frame.")
            return
    except ValueError:
        print("Invalid frame numbers.")
        return

    # Get frame rate
    cap = cv2.VideoCapture(current_video_path)
    if not cap.isOpened():
        print("Could not open video file.")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if fps <= 0:
        print("Could not get video frame rate.")
        return

    temp_output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "storage", "temp")
    os.makedirs(temp_output_dir, exist_ok=True)
    ffmpeg_exe = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "ffmpeg", "bin", "ffmpeg.exe"))

    if not os.path.exists(ffmpeg_exe):
        print("FFmpeg executable not found!")
        return

    input_abs_path = os.path.abspath(current_video_path)
    ffmpeg_output_path = os.path.join(temp_output_dir, f"ffmpeg_cut_{os.path.basename(current_video_path)}")
    output_abs_path = os.path.abspath(ffmpeg_output_path)

    # Calculate start and end time in seconds for -ss and -to/t flags
    # Using -ss after -i for accurate seeking
    start_time = start_frame / fps
    end_time = end_frame / fps # Using -to with end_time for inclusive cutting up to that time

    cmd_list = [
        ffmpeg_exe, "-y", "-i", input_abs_path,
        "-ss", str(start_time),
        "-to", str(end_time),
        "-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        output_abs_path
    ]

    print(f"Running FFmpeg: {' '.join(cmd_list)}")

    try:
        process = subprocess.Popen(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        stdout, stderr = process.communicate()
        ffmpeg_result = process.returncode
        print(f"FFmpeg return code: {ffmpeg_result}")
        print(f"FFmpeg stderr: {stderr.decode('utf-8', errors='ignore')}")

        if ffmpeg_result != 0 or not os.path.exists(output_abs_path):
            print("FFmpeg failed or output file not created.")
            return

        # Replace original file with cut video
        shutil.move(output_abs_path, input_abs_path)
        print(f"Successfully cut and replaced {current_video_path}")

    except FileNotFoundError:
        print("FFmpeg executable not found!")
        return
    except Exception as e:
        print(f"Exception running FFmpeg: {e}")
        if os.path.exists(output_abs_path):
            try:
                os.remove(output_abs_path)
            except Exception:
                pass
        return

    # --- Update info.json with new frame count ---
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
    # Update frame count
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

    # --- Update UI: Refresh video player and thumbnails ---
    if page:
        try:
            from ui.popups.video_player_dialog import open_video_captions_dialog
            from ui.tab_dataset_view import update_thumbnails, thumbnails_grid_ref, selected_dataset
            from ui.utils.utils_datasets import regenerate_all_thumbnails_for_dataset

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
            if dataset_name:
                try:
                    regenerate_all_thumbnails_for_dataset(dataset_name)
                except Exception as thumb_exc:
                    print(f"Could not regenerate all thumbnails: {thumb_exc}")

            # Refresh the video popup if possible
            open_video_captions_dialog(page, current_video_path, video_list, on_caption_updated_callback)

            # Update thumbnails for the dataset if possible
            if dataset_name and thumbnails_grid_ref.current:
                update_thumbnails(page_ctx=page, grid_control=thumbnails_grid_ref.current, force_refresh=True)
                thumbnails_grid_ref.current.update()
                page.update()

        except Exception as e:
            print(f"Could not refresh video player or thumbnails: {e}")

        page.update()
