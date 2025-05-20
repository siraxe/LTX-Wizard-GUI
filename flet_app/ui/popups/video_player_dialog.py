import flet as ft
import os
import json
import cv2
import numpy as np
import shutil # For file operations
import subprocess # For running FFmpeg more robustly
from ui._styles import create_textfield, create_styled_button, VIDEO_PLAYER_DIALOG_WIDTH, VIDEO_PLAYER_DIALOG_HEIGHT, BTN_STYLE2
from flet_video.video import Video, VideoMedia

# === Data & Utility Functions ===

def load_caption_for_video(video_path: str) -> tuple[str, ft.Text | None]:
    """Load caption for a given video from its captions.json file."""
    video_dir = os.path.dirname(video_path)
    dataset_json_path = os.path.join(video_dir, "captions.json")
    video_filename = os.path.basename(video_path)
    no_caption_text = "No captions found, add it here and press Update"
    loaded_message = ft.Text(no_caption_text, color=ft.Colors.RED_400, size=12)
    caption_value = ""
    if os.path.exists(dataset_json_path):
        try:
            with open(dataset_json_path, 'r', encoding='utf-8') as f:
                dataset_json_data = json.load(f)
            for entry in dataset_json_data:
                if entry.get("media_path") == video_filename:
                    caption_value = entry.get("caption", "")
                    loaded_message = None
                    break
        except Exception as e:
            print(f"Error reading captions file {dataset_json_path}: {e}")
            loaded_message = ft.Text("Error loading captions.", color=ft.Colors.RED_400, size=12)
    return caption_value, loaded_message

def save_caption_for_video(page: ft.Page, video_path: str, new_caption: str, on_caption_updated_callback: callable = None) -> bool:
    """Save the new caption for the given video to its captions.json file. Show a snackbar on the page for success/failure. Calls callback if provided."""
    video_dir = os.path.dirname(video_path)
    dataset_json_path = os.path.join(video_dir, "captions.json")
    video_filename = os.path.basename(video_path)
    current_dataset_json_data = []
    if os.path.exists(dataset_json_path):
        try:
            with open(dataset_json_path, 'r', encoding='utf-8') as f:
                current_dataset_json_data = json.load(f)
        except Exception as ex:
            print(f"Error re-reading captions file before save: {ex}")
            if page: page.snack_bar = ft.SnackBar(ft.Text(f"Error re-reading captions: {ex}"), open=True); page.update()
            return False
    found_entry = False
    for entry in current_dataset_json_data:
        if entry.get("media_path") == video_filename:
            entry["caption"] = new_caption
            found_entry = True
            break
    if not found_entry:
        current_dataset_json_data.append({"media_path": video_filename, "caption": new_caption})
    try:
        os.makedirs(video_dir, exist_ok=True)
        with open(dataset_json_path, 'w', encoding='utf-8') as f:
            json.dump(current_dataset_json_data, f, indent=2, ensure_ascii=False)
        if page: page.snack_bar = ft.SnackBar(ft.Text("Caption updated!"), open=True)
        if on_caption_updated_callback:
            on_caption_updated_callback()
        if page: page.update()
        return True
    except Exception as ex_write:
        print(f"Error writing captions file {dataset_json_path}: {ex_write}")
        if page: page.snack_bar = ft.SnackBar(ft.Text(f"Failed to update caption: {ex_write}"), open=True); page.update()
        return False

def get_next_video_path(video_list: list, current_video_path: str, offset: int) -> str | None:
    """Return the next video path in the list given the current path and offset. Wraps around if at the end/start."""
    if not video_list or not current_video_path:
        return None
    try:
        current_idx = video_list.index(current_video_path)
        new_idx = (current_idx + offset) % len(video_list)
        return video_list[new_idx]
    except ValueError:
        print(f"Error: Current video {current_video_path} not in list.")
        return None

# === Globals (for dialog state) ===
_active_video_player_instance = None
_active_caption_field_instance = None
_active_message_container_instance = None
_active_on_caption_updated_callback = None
_current_video_list_for_dialog = []
_current_video_path_for_dialog = ""
_last_video_loading_path = ""
_active_width_field_instance = None
_active_height_field_instance = None

# === GUI-Building Functions ===
def _video_on_completed(e):
    # Workaround: ensure video is playing after looping
    try:
        if e.control:
            e.control.play()
    except Exception:
        pass

def _video_on_click(e):
    try:
        if e.control:
            e.control.play_or_pause()
    except Exception:
        pass

from threading import Timer

# --- Visual feedback state ---
_video_feedback_overlay = None
_video_feedback_timer = None

def _show_video_feedback_icon(stack, icon, color):
    global _video_feedback_overlay, _video_feedback_timer
    if _video_feedback_overlay is not None:
        _video_feedback_overlay.visible = True
        _video_feedback_overlay.content = ft.Icon(icon, size=48, color=color)
        stack.update()
        if _video_feedback_timer:
            _video_feedback_timer.cancel()
        def hide():
            # Use Flet's run_on_main if available for thread-safe update
            try:
                page = getattr(stack, 'page', None)
                if page and hasattr(page, 'run_on_main'):
                    page.run_on_main(lambda: _hide_feedback_overlay(stack))
                else:
                    _hide_feedback_overlay(stack)
            except Exception:
                pass
        _video_feedback_timer = Timer(0.7, hide)
        _video_feedback_timer.start()

def _hide_feedback_overlay(stack):
    global _video_feedback_overlay
    _video_feedback_overlay.visible = False
    stack.update()

def build_video_player(video_path: str, autoplay: bool = False):
    """Create and return a Video player control for the given video path, wrapped in a clickable Container with play/pause feedback."""
    video = Video(
        playlist=[VideoMedia(resource=video_path)],
        autoplay=autoplay,
        width=VIDEO_PLAYER_DIALOG_WIDTH - 40,
        height=VIDEO_PLAYER_DIALOG_HEIGHT - 40,
        expand=False,
        show_controls=True,
        playlist_mode="loop",  # Loop the video when it reaches the end
        on_completed=_video_on_completed,
    )
    feedback_overlay = ft.Container(
        content=ft.Icon(ft.Icons.PLAY_ARROW, size=48, color=ft.Colors.WHITE70),
        alignment=ft.alignment.top_right,
        margin=ft.margin.only(top=16, right=16),
        bgcolor=ft.Colors.with_opacity(0.0, ft.Colors.BLACK),
        visible=False,
        width=VIDEO_PLAYER_DIALOG_WIDTH - 40,
        height=VIDEO_PLAYER_DIALOG_HEIGHT - 40,
        expand=False,
        animate_opacity=300
    )
    global _video_feedback_overlay
    _video_feedback_overlay = feedback_overlay

    # Track play/pause state manually
    _video_is_playing = [autoplay]  # Use list for mutability in closure

    def show_feedback(is_playing):
        icon = ft.Icons.PAUSE if is_playing else ft.Icons.PLAY_ARROW
        color = ft.Colors.WHITE70
        _show_video_feedback_icon(stack, icon, color)

    def play_or_pause_with_feedback():
        # Toggle our state
        _video_is_playing[0] = not _video_is_playing[0]
        orig_play_or_pause()
        show_feedback(_video_is_playing[0])

    def container_on_click(e):
        try:
            play_or_pause_with_feedback()
        except Exception:
            pass
    overlay = ft.Container(
        width=VIDEO_PLAYER_DIALOG_WIDTH - 40,
        height=VIDEO_PLAYER_DIALOG_HEIGHT - 40,
        opacity=0.01,  # nearly invisible, but clickable
        bgcolor=None,
        on_click=container_on_click
    )
    stack = ft.Stack([
        video,
        feedback_overlay,
        overlay
    ], width=VIDEO_PLAYER_DIALOG_WIDTH - 40, height=VIDEO_PLAYER_DIALOG_HEIGHT - 40)
    # Patch video.play_or_pause to also show feedback
    orig_play_or_pause = video.play_or_pause
    video.play_or_pause = play_or_pause_with_feedback
    return stack

# --- Global flag for caption field focus state
_caption_field_is_focused = False

def _caption_field_on_focus(e):
    global _caption_field_is_focused
    _caption_field_is_focused = True

def _caption_field_on_blur(e):
    global _caption_field_is_focused
    _caption_field_is_focused = False

def build_caption_field(initial_value: str = "") -> ft.TextField:
    """Create and return a styled TextField for video captions."""
    tf = create_textfield(
        label="Video Caption",
        value=initial_value,
        hint_text="Enter/edit caption for this video...",
        multiline=True, min_lines=3, max_lines=6,
        expand=True
    )
    tf.on_focus = _caption_field_on_focus
    tf.on_blur = _caption_field_on_blur
    return tf

def build_message_container(content=None) -> ft.Container:
    """Create a container for displaying messages (e.g., no caption found)."""
    return ft.Container(content=content, expand=True)

def build_navigation_controls(on_prev, on_next) -> list[ft.Control]:
    """Create navigation arrow controls for previous/next video."""
    left_arrow = ft.IconButton(ft.Icons.ARROW_LEFT, on_click=on_prev, tooltip="Previous video", icon_size=20)
    right_arrow = ft.IconButton(ft.Icons.ARROW_RIGHT, on_click=on_next, tooltip="Next video", icon_size=20)
    return [left_arrow, right_arrow]

def update_video_player_source(video_player: Video, new_video_path: str):
    """Update the video player's source to the new video path."""
    video_player.stop()
    while video_player.playlist:
        video_player.playlist_remove(0)
    video_player.playlist_add(VideoMedia(resource=new_video_path))
    if video_player.page:
        video_player.update()
    video_player.jump_to(0)

def update_dialog_title(page: ft.Page, new_video_path: str):
    """Update the dialog title to the new video's filename."""
    if hasattr(page, 'base_dialog') and page.base_dialog:
        page.base_dialog.title_text_control.value = f"{os.path.basename(new_video_path)}"
        if page.base_dialog.title_text_control.page:
            page.base_dialog.title_text_control.update()

def update_caption_and_message(video_path: str, caption_field: ft.TextField, message_container: ft.Container):
    """Load and update the caption field and message container for the given video."""
    caption_value, message_control = load_caption_for_video(video_path)
    caption_field.value = caption_value
    message_container.content = message_control

# === End GUI-Building Functions ===

# === Dialog Logic & Event Handlers ===

def switch_video_in_dialog(page: ft.Page, new_video_offset: int):
    """
    Switches the dialog to show a different video (prev/next) and updates all relevant controls.
    Loads the new video player in a background thread to prevent UI freezing.
    """
    global _active_video_player_instance, _active_caption_field_instance, _active_message_container_instance
    global _current_video_list_for_dialog, _current_video_path_for_dialog, _active_on_caption_updated_callback
    global _last_video_loading_path

    if not _current_video_list_for_dialog or not _current_video_path_for_dialog:
        return

    # Save the current caption before switching
    if _active_caption_field_instance is not None and _current_video_path_for_dialog:
        current_caption = _active_caption_field_instance.value.strip()
        save_caption_for_video(page, _current_video_path_for_dialog, current_caption, _active_on_caption_updated_callback)

    idx = _current_video_list_for_dialog.index(_current_video_path_for_dialog)
    new_idx = (idx + new_video_offset) % len(_current_video_list_for_dialog)
    new_video_path = _current_video_list_for_dialog[new_idx]
    _last_video_loading_path = new_video_path
    _current_video_path_for_dialog = new_video_path

    def load_and_replace_video():
        global _active_video_player_instance
        global _current_video_path_for_dialog, _last_video_loading_path

        if _current_video_path_for_dialog != _last_video_loading_path:
            return

        try:
            # Fully rebuild the dialog content and navigation controls
            main_content_ui, nav_controls = create_video_player_with_captions_content(
                page, _current_video_path_for_dialog, _current_video_list_for_dialog, _active_on_caption_updated_callback
            )

            if page and hasattr(page, 'base_dialog') and getattr(page.base_dialog, 'show_dialog', None):
                page.base_dialog.show_dialog(content=main_content_ui, title=os.path.basename(_current_video_path_for_dialog), new_width=VIDEO_PLAYER_DIALOG_WIDTH, title_prefix_controls=nav_controls)
                page.dialog = page.base_dialog
                page.dialog.update()
            elif page:
                fallback_alert = ft.AlertDialog(title=ft.Text(os.path.basename(_current_video_path_for_dialog)), content=main_content_ui, actions=[ft.TextButton("Close", on_click=lambda e: fallback_alert.close())], on_dismiss=handle_dialog_dismiss)
                page.dialog = fallback_alert; fallback_alert.open = True; page.update()
        except Exception as e:
            print(f"Error rebuilding dialog content for video switch: {e}")
            if page:
                def show_error_snackbar(error_message):
                    if page.snack_bar is None:
                        page.snack_bar = ft.SnackBar(ft.Text(f"Error loading video: {error_message}"), open=True)
                    else:
                        page.snack_bar.content = ft.Text(f"Error loading video: {error_message}")
                        page.snack_bar.open = True
                    page.update()
                show_error_snackbar(e)

    if page:
        load_and_replace_video()

def handle_update_caption_click(page: ft.Page):
    """
    Handles the Update button click: saves the caption and updates UI.
    """
    global _active_caption_field_instance, _active_message_container_instance, _active_on_caption_updated_callback, _current_video_path_for_dialog
    new_caption = _active_caption_field_instance.value.strip()
    save_caption_for_video(page, _current_video_path_for_dialog, new_caption, _active_on_caption_updated_callback)
    if _active_caption_field_instance and _active_message_container_instance:
        update_caption_and_message(_current_video_path_for_dialog, _active_caption_field_instance, _active_message_container_instance)
    if page:
        page.update()

def handle_dialog_dismiss(e):
    """
    Handles the dialog dismissal event: saves the current caption if the caption field exists.
    """
    global _active_caption_field_instance, _current_video_path_for_dialog, _active_on_caption_updated_callback
    if _active_caption_field_instance and _current_video_path_for_dialog:
        current_caption = _active_caption_field_instance.value.strip()
        # Pass page=None here as page may not be valid after dialog is dismissed
        save_caption_for_video(page=None, video_path=_current_video_path_for_dialog, new_caption=current_caption, on_caption_updated_callback=_active_on_caption_updated_callback)

def handle_crop_video_click(page: ft.Page):
    """
    Handles the Crop button click: reads dimensions, uses FFmpeg to crop/scale and encode video, and overwrites the original file.
    """
    global _active_width_field_instance, _active_height_field_instance, _current_video_path_for_dialog

    if not _active_width_field_instance or not _active_height_field_instance or not _current_video_path_for_dialog:
        return

    try:
        target_width = int(_active_width_field_instance.value)
        target_height = int(_active_height_field_instance.value)
        if target_width <= 0 or target_height <= 0:
            raise ValueError("Dimensions must be positive integers.")
    except ValueError as e:
        return

    current_video_path = _current_video_path_for_dialog

    # --- FFmpeg crop/resize and encode ---
    temp_output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "storage", "temp")
    os.makedirs(temp_output_dir, exist_ok=True)
    
    # Construct path to ffmpeg.exe
    ffmpeg_exe = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ffmpeg", "bin", "ffmpeg.exe"))

    if not os.path.exists(ffmpeg_exe):
        return

    ffmpeg_output_path = os.path.join(temp_output_dir, f"ffmpeg_cropped_{os.path.basename(current_video_path)}")
    input_abs_path = os.path.abspath(current_video_path)
    output_abs_path = os.path.abspath(ffmpeg_output_path)
    
    # Use subprocess for FFmpeg for better path handling and error reporting
    # Use crop filter instead of scale to crop the center of the video
    crop_filter = f"crop={target_width}:{target_height}"
    cmd_list = [
        ffmpeg_exe, "-y", "-i", input_abs_path,
        "-vf", crop_filter,
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
    except Exception as e_subproc:
        log_debug(f"Exception running FFmpeg: {e_subproc}")
        return

    if ffmpeg_result != 0 or not os.path.exists(output_abs_path):
        log_debug("FFmpeg failed or output file not created.")
        return

    move_success = False
    try:
        shutil.move(output_abs_path, input_abs_path)
        move_success = True
    except Exception as e_move:
        if os.path.exists(output_abs_path):
            try:
                os.remove(output_abs_path)
            except Exception as e_remove:
                pass
        # Do not return here; attempt info.json update for debugging

    # --- Update info.json --- 
    info_json_dir = os.path.dirname(input_abs_path)
    info_json_path = os.path.join(info_json_dir, "info.json")

    new_frame_count = 0
    try:
        new_cap = cv2.VideoCapture(input_abs_path)
        if new_cap.isOpened():
            new_frame_count = int(new_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            new_cap.release()
        else:
            pass
    except Exception as e_cap:
        pass

    info_data = {}
    if os.path.exists(info_json_path):
        try:
            with open(info_json_path, 'r', encoding='utf-8') as f:
                info_data = json.load(f)
        except Exception as e_read:
            pass

    # Preserve specified fields if they exist
    preserved_type = info_data.get("type")
    preserved_trigger_word = info_data.get("trigger_word")

    # Update only the entry for the current video file
    video_filename = os.path.basename(input_abs_path)
    info_data[video_filename] = {
        "width": target_width,
        "height": target_height,
        "frames": new_frame_count
    }

    # Restore preserved fields at the root if needed (optional, only if you want to keep them)
    if preserved_type is not None:
        info_data["type"] = preserved_type
    if preserved_trigger_word is not None:
        info_data["trigger_word"] = preserved_trigger_word

    try:
        with open(info_json_path, 'w', encoding='utf-8') as f:
            json.dump(info_data, f, indent=2, ensure_ascii=False)
    except Exception as e_write:
        pass
    # --- End update info.json ---

    # Refresh video player in popup by closing and reopening the dialog
    try:
        from .video_player_dialog import open_video_captions_dialog
        if page and '_current_video_path_for_dialog' in globals() and _current_video_path_for_dialog:
            # Try to preserve the video list if available
            video_list = _current_video_list_for_dialog if '_current_video_list_for_dialog' in globals() else None
            on_caption_updated_callback = _active_on_caption_updated_callback if '_active_on_caption_updated_callback' in globals() else None
            # Close dialog if possible
            if hasattr(page, 'base_dialog') and page.base_dialog:
                page.base_dialog.hide_dialog()
            # Reopen dialog for the same video
            open_video_captions_dialog(page, _current_video_path_for_dialog, video_list, on_caption_updated_callback)
    except Exception as e:
        log_debug(f"Failed to refresh video player by reopening dialog: {e}")


# === Keyboard Event Handler ===
from ui.flet_hotkeys import AUTO_VIDEO_PLAYBACK, VIDEO_PLAY_PAUSE_KEY, VIDEO_NEXT_KEY, VIDEO_PREV_KEY


# === Keyboard Event Handler ===
def handle_caption_dialog_keyboard(page: ft.Page, e: ft.KeyboardEvent):
    """
    Handles keyboard events for the video captions dialog, including play/pause toggle and navigation.
    Uses keybindings from ui.flet_hotkeys.
    """
    try:
        global _caption_field_is_focused
        if _caption_field_is_focused:
            return  # Do not trigger any hotkeys while typing in the caption field
        # Play/pause
        if hasattr(e, 'key') and e.key == VIDEO_PLAY_PAUSE_KEY:
            if _active_video_player_instance is not None:
                video_ctrl = None
                try:
                    video_ctrl = _active_video_player_instance.controls[0]
                except Exception:
                    pass
                if video_ctrl and hasattr(video_ctrl, 'play_or_pause'):
                    video_ctrl.play_or_pause()
        # Previous video
        elif hasattr(e, 'key') and e.key == VIDEO_PREV_KEY:
            switch_video_in_dialog(page, -1)
        # Next video
        elif hasattr(e, 'key') and e.key == VIDEO_NEXT_KEY:
            switch_video_in_dialog(page, 1)
    except Exception as ex:
        print(f"[ERROR] Exception in handle_caption_dialog_keyboard: {ex}")

# --- Main Dialog Construction ---
def create_video_player_with_captions_content(page: ft.Page, video_path: str, video_list: list, on_caption_updated_callback: callable = None) -> tuple[ft.Column, list[ft.Control]]:
    """
    Builds the main content column and navigation controls for the video player with captions dialog.
    """
    global _active_video_player_instance, _active_caption_field_instance, _active_message_container_instance
    global _current_video_list_for_dialog, _current_video_path_for_dialog, _active_on_caption_updated_callback
    global _last_video_loading_path, _active_width_field_instance, _active_height_field_instance

    _current_video_path_for_dialog = video_path
    _current_video_list_for_dialog = video_list
    _active_on_caption_updated_callback = on_caption_updated_callback
    _last_video_loading_path = video_path

    nav_controls = build_navigation_controls(
        lambda e: switch_video_in_dialog(page, -1),
        lambda e: switch_video_in_dialog(page, 1)
    )

    # Video player: always visible, no overlay/thumbnail logic
    _active_video_player_instance = build_video_player(video_path, autoplay=AUTO_VIDEO_PLAYBACK)

    caption_value, message_control = load_caption_for_video(video_path)
    _active_caption_field_instance = build_caption_field(initial_value=caption_value)

    # Import crop controls from video_editor.py
    from .video_editor import build_crop_controls_row, handle_crop_video_click, handle_crop_all_videos, handle_set_closest_div32

    def on_crop_click(e):
        page.run_thread(lambda: handle_crop_video_click(page, _active_width_field_instance, _active_height_field_instance, _current_video_path_for_dialog))

    def on_crop_all_click(e):
        page.run_thread(lambda: handle_crop_all_videos(page, _active_width_field_instance, _active_height_field_instance, _current_video_list_for_dialog))

    def on_get_closest(e):
        handle_set_closest_div32(_active_width_field_instance, _active_height_field_instance, _current_video_path_for_dialog, page)

    crop_controls_row, _active_width_field_instance, _active_height_field_instance = build_crop_controls_row(
        page, _current_video_path_for_dialog, on_crop_click, on_crop_all=on_crop_all_click, on_get_closest=on_get_closest)

    _active_message_container_instance = build_message_container(content=message_control)

    content_column = ft.Column(
        controls=[
            ft.Row([_active_video_player_instance], alignment=ft.MainAxisAlignment.CENTER),
            _active_caption_field_instance,
            crop_controls_row,
            ft.Row([
                _active_message_container_instance
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN, vertical_alignment=ft.CrossAxisAlignment.CENTER)
        ],
        spacing=10, tight=True,
        scroll=ft.ScrollMode.ADAPTIVE
    )
    return content_column, nav_controls

def open_video_captions_dialog(page: ft.Page, video_path: str, video_list=None, on_caption_updated_callback: callable = None):
    """
    Opens the video captions dialog for the given video and list.
    """
    global _active_caption_field_instance
    if not video_path:
        return
    if video_list is None:
        video_list = [video_path]

    video_filename = os.path.basename(video_path)
    dialog_title_text = f"{video_filename}"

    main_content_ui, nav_prefix_controls = create_video_player_with_captions_content(page, video_path, video_list, on_caption_updated_callback)

    desired_width = VIDEO_PLAYER_DIALOG_WIDTH

    if hasattr(page, 'base_dialog') and page.base_dialog:
        # Set the on_dismiss callback for the base_dialog instance before showing
        # This assumes base_dialog is a PopupDialogBase instance or similar
        page.base_dialog._on_dismiss_callback = handle_dialog_dismiss
        # Pass the handle_dialog_dismiss callback to the PopupDialogBase instance
        page.base_dialog.show_dialog(content=main_content_ui, title=dialog_title_text, new_width=desired_width, title_prefix_controls=nav_prefix_controls)
    else:
        print("Error: Base dialog (PopupDialogBase) not found on page.")
        fallback_alert = ft.AlertDialog(title=ft.Text(dialog_title_text), content=main_content_ui, actions=[ft.TextButton("Close", on_click=lambda e: fallback_alert.close())], on_dismiss=handle_dialog_dismiss)
        page.dialog = fallback_alert; fallback_alert.open = True; page.update()

    # Attach dialog-specific hotkey handler using keybindings from flet_hotkeys
    page.video_dialog_hotkey_handler = lambda e: handle_caption_dialog_keyboard(page, e)
    page.video_dialog_open = True
