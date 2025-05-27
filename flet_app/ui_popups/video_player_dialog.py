# video_player_dialog.py
import flet as ft
import os
import json
import cv2
import numpy as np
import shutil # For file operations
import subprocess # For running FFmpeg more robustly
from ui._styles import create_textfield, create_styled_button, VIDEO_PLAYER_DIALOG_WIDTH, VIDEO_PLAYER_DIALOG_HEIGHT, BTN_STYLE2
from flet_video.video import Video, VideoMedia
import threading  # Add this import for frame counter
from . import video_editor

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
_video_is_playing = [False]  # Global play/pause state, always a list for mutability
_active_page_ref: ft.Page = None # New global to hold page reference

# Global state for video playback range and reframing
reframed_playback = False
_playback_start_frame = -1
_playback_end_frame = -1

# --- Frame counter globals ---
_frame_update_timer: threading.Timer | None = None
_dialog_is_open = False  # Add dialog open flag for frame counter

# --- Video Editor UI Globals ---
_frame_range_slider_instance: ft.RangeSlider = None
_start_value_text_instance: ft.Text = None
_end_value_text_instance: ft.Text = None
_total_frames_text_instance: ft.Text = None

# === GUI-Building Functions ===
def _video_on_completed(e):
    # This is called when the video naturally ends. We want to loop within the defined range.
    handle_video_completed(e.control)

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
    """Create and return a Video player control for the given video path, wrapped in a clickable Container with play/pause feedback and frame counter."""
    global _video_feedback_overlay, _video_is_playing, _frame_update_timer
    import time
    video = Video(
        playlist=[VideoMedia(resource=video_path)],
        autoplay=autoplay,
        width=VIDEO_PLAYER_DIALOG_WIDTH - 40,
        height=VIDEO_PLAYER_DIALOG_HEIGHT - 40,
        expand=False,
        show_controls=True,
        playlist_mode="none",
        on_completed=lambda e: handle_video_completed(e.control),
    )
    feedback_overlay = ft.Container(
        content=ft.Icon(ft.Icons.PAUSE, size=48, color=ft.Colors.WHITE70),
        alignment=ft.alignment.top_right,
        margin=ft.margin.only(top=16, right=16),
        bgcolor=ft.Colors.with_opacity(0.0, ft.Colors.BLACK),
        visible=False,
        width=VIDEO_PLAYER_DIALOG_WIDTH - 40,
        height=VIDEO_PLAYER_DIALOG_HEIGHT - 40,
        expand=False,
        animate_opacity=300
    )
    _video_feedback_overlay = feedback_overlay

    # --- Frame counter logic ---
    video_fps = 30.0
    total_frames = 1
    try:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps > 0:
                video_fps = fps
            tf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if tf > 0:
                total_frames = tf
            cap.release()
    except Exception as e:
        print(f"Error getting video FPS or total frame count for {video_path}: {e}")

    frame_counter_text = ft.Text("001 / 001", color=ft.Colors.WHITE70, size=12)
    video._video_fps = video_fps
    video._frame_counter_text = frame_counter_text
    video._total_frames = total_frames

    def _update_frame_counter(video_player: Video, frame_counter_text: ft.Text):
        global _dialog_is_open, _playback_start_frame, _playback_end_frame
        global reframed_playback
        import time
        if video_player is None or frame_counter_text is None:
            return
        if not hasattr(video_player, '_video_fps') or video_player._video_fps is None:
            return
        effective_fps = video_player._video_fps
        if not isinstance(effective_fps, (int, float)) or effective_fps <= 0:
            effective_fps = 30.0
        while _dialog_is_open:
            if video_player is None or video_player.page is None:
                break
            try:
                current_position_ms = video_player.get_current_position(wait_timeout=0.2)
                if current_position_ms is not None:
                    current_position_ms = max(0, current_position_ms)
                    current_frame = int((current_position_ms / 1000.0) * effective_fps)
                    current_frame = max(0, min(current_frame, video_player._total_frames-1))

                    # Trigger frame right away if triggered by _on_frame_range_change_end from video_editor.py
                    if reframed_playback:
                        video_player.seek(int((_playback_start_frame / effective_fps) * 1000)) # Jump to the beginning of the defined range
                        video_player.play() # Ensure playback continues after seek
                        if video_player.page: video_player.update()
                        reframed_playback = False

                    # Check if current frame exceeds end_frame and loop (could be outdates if triggered by _on_frame_range_change_end from video_editor.py)
                    if _playback_end_frame != -1 and current_frame >= _playback_end_frame:
                        video_player.seek(int((_playback_start_frame / effective_fps) * 1000)) # Jump to the beginning of the defined range
                        video_player.play() # Ensure playback continues after seek
                        if video_player.page: # Ensure control is mounted before updating
                             video_player.update() # Update the control state if necessary
                        current_frame = _playback_start_frame # Update current frame for display immediately

                    # Display as 001 / 048
                    frame_str = f"{current_frame+1:03d} / {video_player._total_frames:03d}"
                    if frame_counter_text.page:
                        frame_counter_text.value = frame_str
                        frame_counter_text.update()
                else:
                    if frame_counter_text.page:
                        frame_counter_text.value = f"--- / {video_player._total_frames:03d}"
                        frame_counter_text.update()
            except Exception as e:
                pass
                #print(f"Error in _update_frame_counter: {e}")
            time.sleep(0.1)

    # Track play/pause state globally
    _video_is_playing[0] = autoplay

    def show_feedback(is_playing):
        icon = ft.Icons.PLAY_ARROW if is_playing else ft.Icons.PAUSE
        color = ft.Colors.WHITE70
        _show_video_feedback_icon(stack, icon, color)

    def play_or_pause_with_feedback():
        global _video_is_playing
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
        overlay,
        ft.Container(
            content=frame_counter_text,
            alignment=ft.alignment.bottom_right,
            margin=ft.margin.only(bottom=16, right=16),
            bgcolor=ft.Colors.with_opacity(0.0, ft.Colors.BLACK),
            visible=True,
            expand=False
        )
    ], width=VIDEO_PLAYER_DIALOG_WIDTH - 40, height=VIDEO_PLAYER_DIALOG_HEIGHT - 40)
    orig_play_or_pause = video.play_or_pause
    video.play_or_pause = play_or_pause_with_feedback

    # --- Start frame counter thread ---
    global _frame_update_timer
    if _frame_update_timer is not None and hasattr(_frame_update_timer, 'is_alive') and _frame_update_timer.is_alive():
        _frame_update_timer.cancel()
    _frame_update_timer = threading.Timer(0.1, lambda: _update_frame_counter(video, frame_counter_text))
    _frame_update_timer.daemon = True
    _frame_update_timer.start()

    return stack

# Add a new handler function for video reframing to implement manual looping
def handle_video_reframing(video_control: Video):
    """Handler for video reframing to implement manual looping."""
    global _playback_start_frame, _playback_end_frame
    try:
        # This function is now primarily used to seek to the start of the defined range
        # when the range is updated from the slider.
        video_control.seek(int((_playback_start_frame / video_control._video_fps) * 1000)) # Jump to the beginning of the defined range
        video_control.play()
        if video_control.page: # Ensure control is mounted before updating
             video_control.update() # Update the control state if necessary
    except Exception as e:
        pass
        #print(f"Error seeking/restarting video on reframing: {e}")

# Add a new handler function for video completion to implement manual looping
def handle_video_completed(video_control: Video):
    """Handler for video completion to implement manual looping."""
    global _playback_start_frame, _playback_end_frame
    try:
        # When the video naturally completes, loop back to the start of the defined range
        video_control.seek(int((_playback_start_frame / video_control._video_fps) * 1000)) # Jump to the beginning of the defined range
        video_control.play()
        if video_control.page: # Ensure control is mounted before updating
             video_control.update() # Update the control state if necessary
    except Exception as e:
        pass

def update_video_player_source(video_player: Video, new_video_path: str):
    """Update the video player's source to the new video path and update its FPS and total frames."""
    video_player.playlist = [VideoMedia(resource=new_video_path)]
    video_player.update()

    # Recalculate and update FPS and total frames for the video player instance
    video_fps = 30.0
    total_frames = 1
    try:
        cap = cv2.VideoCapture(new_video_path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps > 0:
                video_fps = fps
            tf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if tf > 0:
                total_frames = tf
            cap.release()
    except Exception as e:
        print(f"Error getting video FPS or total frame count for {new_video_path}: {e}")

    video_player._video_fps = video_fps
    video_player._total_frames = total_frames

    # Also, reset playback range globals to default for the new video
    global _playback_start_frame, _playback_end_frame
    _playback_start_frame = 0
    _playback_end_frame = -1 # -1 indicates no specific end frame set

    # Seek to the beginning of the new video
    video_player.seek(0)

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
    # Ensure updates if controls are on page
    if caption_field.page:
        caption_field.update()
    if message_container.page:
        message_container.update()

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
    global _frame_range_slider_instance, _start_value_text_instance, _end_value_text_instance, _total_frames_text_instance
    global _active_page_ref # Add to global declaration

    if not _current_video_list_for_dialog or not _current_video_path_for_dialog:
        return

    # Save the current caption before switching
    if _active_caption_field_instance is not None and _current_video_path_for_dialog:
        current_caption = _active_caption_field_instance.value.strip()
        # Pass None for the callback when saving on video switch to prevent thumbnail refresh
        save_caption_for_video(page, _current_video_path_for_dialog, current_caption, None)

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
                page.video_dialog_open = True
                page.video_dialog_hotkey_handler = lambda event: handle_caption_dialog_keyboard(page, event)
                page.dialog.update()
            elif page:
                fallback_alert = ft.AlertDialog(title=ft.Text(os.path.basename(_current_video_path_for_dialog)), content=main_content_ui, actions=[ft.TextButton("Close", on_click=lambda e: fallback_alert.close())], on_dismiss=lambda e: handle_dialog_dismiss(e))
                page.dialog = fallback_alert; fallback_alert.open = True; page.video_dialog_open = True; page.video_dialog_hotkey_handler = lambda event: handle_caption_dialog_keyboard(page, event); page.update()
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

    # Update the frame range slider and text controls
    frame_count = 100 # Default
    if new_video_path and os.path.exists(new_video_path):
        cap = cv2.VideoCapture(new_video_path)
        if cap.isOpened():
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

    if _frame_range_slider_instance:
        _frame_range_slider_instance.max = frame_count
        _frame_range_slider_instance.start_value = 0
        _frame_range_slider_instance.end_value = frame_count
        _frame_range_slider_instance.divisions = frame_count
        if _frame_range_slider_instance.page: _frame_range_slider_instance.update() # Update the slider

    if _start_value_text_instance:
        _start_value_text_instance.value = f"Start: {0}"
        if _start_value_text_instance.page: _start_value_text_instance.update()
    if _end_value_text_instance:
        _end_value_text_instance.value = f"End: {frame_count}"
        if _end_value_text_instance.page: _end_value_text_instance.update()
    if _total_frames_text_instance:
        _total_frames_text_instance.value = f"Total: {frame_count}"
        if _total_frames_text_instance.page: _total_frames_text_instance.update()

    page.update()

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
    Handles the dialog dismissal event: saves the current caption if the caption field exists and stops the frame counter thread.
    """
    global _active_caption_field_instance, _current_video_path_for_dialog, _active_on_caption_updated_callback, _dialog_is_open, _frame_update_timer
    global _active_page_ref # Add to global declaration
    _dialog_is_open = False  # Stop frame counter thread
    if _frame_update_timer is not None and hasattr(_frame_update_timer, 'is_alive') and _frame_update_timer.is_alive():
        _frame_update_timer.cancel()
    if _active_caption_field_instance and _current_video_path_for_dialog:
        current_caption = _active_caption_field_instance.value.strip()
        # Pass page=None here as page may not be valid after dialog is dismissed
        save_caption_for_video(page=None, video_path=_current_video_path_for_dialog, new_caption=current_caption, on_caption_updated_callback=_active_on_caption_updated_callback)
    
    if _active_page_ref:
        _active_page_ref.video_dialog_open = False
        _active_page_ref.video_dialog_hotkey_handler = None
        _active_page_ref.update()

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
                    video_ctrl = _active_video_player_instance.controls[0] # Assuming stack structure
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
        else:
            # If the key is not handled by the video dialog, pass it to the global hotkey handler
            from ui.flet_hotkeys import global_hotkey_handler
            global_hotkey_handler(page, e)
    except Exception as ex:
        print(f"[ERROR] Exception in handle_caption_dialog_keyboard: {ex}")
        return False

# --- Main Dialog Construction ---
def create_video_player_with_captions_content(page: ft.Page, video_path: str, video_list: list, on_caption_updated_callback: callable = None) -> tuple[ft.Column, list[ft.Control]]:
    """
    Builds the main content column and navigation controls for the video player with captions dialog.
    """
    global _active_video_player_instance, _active_caption_field_instance, _active_message_container_instance
    global _current_video_list_for_dialog, _current_video_path_for_dialog, _active_on_caption_updated_callback
    global _last_video_loading_path
    global _active_width_field_instance, _active_height_field_instance
    global _frame_range_slider_instance, _start_value_text_instance, _end_value_text_instance, _total_frames_text_instance
    global _active_page_ref # Add to global declaration

    _current_video_path_for_dialog = video_path
    _current_video_list_for_dialog = video_list
    _active_on_caption_updated_callback = on_caption_updated_callback
    _last_video_loading_path = video_path
    _active_page_ref = page # Set the global page reference

    nav_controls = build_navigation_controls(
        lambda e: switch_video_in_dialog(page, -1),
        lambda e: switch_video_in_dialog(page, 1)
    )

    _active_video_player_instance = build_video_player(video_path, autoplay=AUTO_VIDEO_PLAYBACK)

    caption_value, message_control = load_caption_for_video(video_path)
    _active_caption_field_instance = build_caption_field(initial_value=caption_value)

    # Define callbacks for video_editor functions that will be run in a thread
    def on_crop_click(e):
        # This will use the handle_crop_video_click from video_editor
        page.run_thread(lambda: video_editor.handle_crop_video_click(page, _active_width_field_instance, _active_height_field_instance, _current_video_path_for_dialog, video_list=_current_video_list_for_dialog, on_caption_updated_callback=_active_on_caption_updated_callback))

    def on_crop_all_click(e):
        page.run_thread(lambda: video_editor.handle_crop_all_videos(page, _active_width_field_instance, _active_height_field_instance, _current_video_list_for_dialog, _active_on_caption_updated_callback))

    def on_get_closest(e):
        # This function is typically synchronous and updates UI fields directly
        video_editor.handle_set_closest_div32(_active_width_field_instance, _active_height_field_instance, _current_video_path_for_dialog, page)

    (
        crop_controls_row,
        _active_width_field_instance, # These are assigned by build_crop_controls_row
        _active_height_field_instance, # These are assigned by build_crop_controls_row
        frame_range_slider,
        start_value_text,
        end_value_text,
        total_frames_text,
    ) = video_editor.build_crop_controls_row(
        page,
        video_path,
        on_crop=on_crop_click, # Uses the on_crop_click defined above
        on_crop_all=on_crop_all_click,
        on_get_closest=on_get_closest,
        video_list=_current_video_list_for_dialog,
        on_caption_updated_callback=_active_on_caption_updated_callback,
        video_player_instance=_active_video_player_instance, # Pass the video player instance itself
    )

    # Store the instances globally
    _frame_range_slider_instance = frame_range_slider
    _start_value_text_instance = start_value_text
    _end_value_text_instance = end_value_text
    _total_frames_text_instance = total_frames_text

    _active_message_container_instance = build_message_container(content=message_control)

    content_column = ft.Column(
        controls=[
            ft.Row([_active_video_player_instance], alignment=ft.MainAxisAlignment.CENTER),
            _active_caption_field_instance,
            crop_controls_row, # This row now contains controls built by video_editor
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
    global _active_caption_field_instance, _dialog_is_open
    if not video_path:
        return
    if video_list is None:
        video_list = [video_path]

    video_filename = os.path.basename(video_path)
    dialog_title_text = f"{video_filename}"

    # Set dialog open flag for frame counter
    _dialog_is_open = True

    main_content_ui, nav_prefix_controls = create_video_player_with_captions_content(page, video_path, video_list, on_caption_updated_callback)

    desired_width = VIDEO_PLAYER_DIALOG_WIDTH

    if hasattr(page, 'base_dialog') and page.base_dialog:
        page.base_dialog._on_dismiss_callback = handle_dialog_dismiss
        page.base_dialog.show_dialog(content=main_content_ui, title=dialog_title_text, new_width=desired_width, title_prefix_controls=nav_prefix_controls)
        page.dialog = page.base_dialog
        page.video_dialog_open = True
        page.video_dialog_hotkey_handler = lambda event: handle_caption_dialog_keyboard(page, event)
        page.dialog.update()

        # Now that the video player is on the page, update its source (if it's not already playing the correct one)
        # _active_video_player_instance is built with the correct video_path in create_video_player_with_captions_content
        # update_video_player_source(_active_video_player_instance, video_path) # May not be needed if autoplay handles it

        update_dialog_title(page, video_path) # Ensure title is correct
    else:
        print("Error: Base dialog (PopupDialogBase) not found on page.")
        fallback_alert = ft.AlertDialog(title=ft.Text(dialog_title_text), content=main_content_ui, actions=[ft.TextButton("Close", on_click=lambda e: fallback_alert.close())], on_dismiss=lambda e: handle_dialog_dismiss(e))
        page.dialog = fallback_alert; fallback_alert.open = True; page.video_dialog_open = True; page.video_dialog_hotkey_handler = lambda event: handle_caption_dialog_keyboard(page, event); page.update()


def update_playback_range(video_player: Video, start_frame: int, end_frame: int):
    """Updates the global playback range and seeks the video to the start frame."""
    global _playback_start_frame, _playback_end_frame, reframed_playback
    _playback_start_frame = start_frame
    _playback_end_frame = end_frame
    reframed_playback = True 
    if video_player and hasattr(video_player, '_video_fps') and video_player._video_fps > 0:
        seek_time_ms = int((_playback_start_frame / video_player._video_fps) * 1000)
        video_player.seek(seek_time_ms)

# --- Global flag for caption field focus state ---
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

# Removed duplicated update_video_player_source, update_dialog_title, update_caption_and_message
# as they are defined earlier in the file.