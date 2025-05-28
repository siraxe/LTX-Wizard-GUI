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
import time # Import time for sleep

# === Data & Utility Functions ===

def load_caption_for_video(video_path: str) -> tuple[str, str, ft.Text | None]:
    """Load caption for a given video from its captions.json file."""
    video_dir = os.path.dirname(video_path)
    dataset_json_path = os.path.join(video_dir, "captions.json")
    video_filename = os.path.basename(video_path)
    no_caption_text = "No captions found, add it here and press Update"
    loaded_message = ft.Text(no_caption_text, color=ft.Colors.RED_400, size=12)
    caption_value = ""
    negative_caption_value = ""
    if os.path.exists(dataset_json_path):
        try:
            with open(dataset_json_path, 'r', encoding='utf-8') as f:
                dataset_json_data = json.load(f)
            for entry in dataset_json_data:
                if entry.get("media_path") == video_filename:
                    caption_value = entry.get("caption", "")
                    negative_caption_value = entry.get("negative_caption", "") # Load negative caption
                    loaded_message = None
                    break
        except Exception as e:
            print(f"Error reading captions file {dataset_json_path}: {e}")
            loaded_message = ft.Text("Error loading captions.", color=ft.Colors.RED_400, size=12)
    return caption_value, negative_caption_value, loaded_message # Return both captions

def save_caption_for_video(page: ft.Page, video_path: str, new_caption: str, on_caption_updated_callback: callable = None, field_name: str = "caption") -> bool:
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
            entry[field_name] = new_caption # Use field_name here
            found_entry = True
            break
    if not found_entry:
        # Initialize both fields when adding a new entry
        new_entry = {"media_path": video_filename}
        if field_name == "caption":
            new_entry["caption"] = new_caption
            new_entry["negative_caption"] = ""
        else:
            new_entry["caption"] = ""
            new_entry["negative_caption"] = new_caption
        current_dataset_json_data.append(new_entry)

    try:
        os.makedirs(video_dir, exist_ok=True)
        with open(dataset_json_path, 'w', encoding='utf-8') as f:
            json.dump(current_dataset_json_data, f, indent=2, ensure_ascii=False)
        # Only show snackbar for the main caption update, or modify to be clearer
        # if field_name == "caption":
        #     if page: page.snack_bar = ft.SnackBar(ft.Text("Caption updated!"), open=True)
        if page: page.snack_bar = ft.SnackBar(ft.Text(f"{field_name.replace('_', ' ').title()} updated!"), open=True)

        if on_caption_updated_callback:
            on_caption_updated_callback()
        if page: page.update()
        return True
    except Exception as ex_write:
        print(f"Error writing captions file {dataset_json_path}: {ex_write}")
        if page: page.snack_bar = ft.SnackBar(ft.Text(f"Failed to update {field_name.replace('_', ' ')}: {ex_write}"), open=True); page.update()
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
# Refactored into VideoDialogState class

class VideoDialogState:
    """Manages the state for the Video Player Dialog."""
    def __init__(self):
        self.active_video_player_instance: Video | None = None
        self.active_caption_field_instance: ft.TextField | None = None
        self.active_caption_neg_field_instance: ft.TextField | None = None
        self.active_message_container_instance: ft.Container | None = None
        self.active_on_caption_updated_callback: callable | None = None
        self.current_video_list_for_dialog: list = []
        self.current_video_path_for_dialog: str = ""
        self.last_video_loading_path: str = ""
        self.active_width_field_instance: ft.TextField | None = None
        self.active_height_field_instance: ft.TextField | None = None
        self.video_is_playing: list[bool] = [False]  # List for mutability
        self.active_page_ref: ft.Page | None = None

        # Global state for video playback range and reframing
        self.reframed_playback: bool = False
        self.playback_start_frame: int = -1
        self.playback_end_frame: int = -1

        # --- Frame counter globals ---
        self.frame_update_timer: threading.Timer | None = None
        self.dialog_is_open: bool = False

        # --- Video Editor UI Globals ---
        self.frame_range_slider_instance: ft.RangeSlider | None = None
        self.start_value_text_instance: ft.Text | None = None
        self.end_value_text_instance: ft.Text | None = None
        self.total_frames_text_instance: ft.Text | None = None

        # --- Visual feedback state ---
        self.video_feedback_overlay: ft.Container | None = None
        self.video_feedback_timer: threading.Timer | None = None

        # --- Global flag for caption field focus state ---
        self.caption_field_is_focused: bool = False

# Create a single instance of the state class
dialog_state = VideoDialogState()

# --- Visual feedback state ---
# Moved into VideoDialogState class

def _show_video_feedback_icon(stack, icon, color):
    if dialog_state.video_feedback_overlay is not None:
        dialog_state.video_feedback_overlay.visible = True
        dialog_state.video_feedback_overlay.content = ft.Icon(icon, size=48, color=color)
        stack.update()
        if dialog_state.video_feedback_timer:
            dialog_state.video_feedback_timer.cancel()
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
        dialog_state.video_feedback_timer = threading.Timer(0.7, hide)
        dialog_state.video_feedback_timer.start()

def _hide_feedback_overlay(stack):
    dialog_state.video_feedback_overlay.visible = False
    stack.update()

def build_video_player(video_path: str, autoplay: bool = False):
    """Create and return a Video player control for the given video path, wrapped in a clickable Container with play/pause feedback and frame counter."""
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
    dialog_state.video_feedback_overlay = feedback_overlay

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
        if video_player is None or frame_counter_text is None:
            return
        if not hasattr(video_player, '_video_fps') or video_player._video_fps is None:
            return
        effective_fps = video_player._video_fps
        if not isinstance(effective_fps, (int, float)) or effective_fps <= 0:
            effective_fps = 30.0
        while dialog_state.dialog_is_open:
            if video_player is None or video_player.page is None:
                break
            try:
                current_position_ms = video_player.get_current_position(wait_timeout=0.2)
                if current_position_ms is not None:
                    current_position_ms = max(0, current_position_ms)
                    current_frame = int((current_position_ms / 1000.0) * effective_fps)
                    current_frame = max(0, min(current_frame, video_player._total_frames-1))

                    # Trigger frame right away if triggered by _on_frame_range_change_end from video_editor.py
                    if dialog_state.reframed_playback:
                        video_player.seek(int((dialog_state.playback_start_frame / effective_fps) * 1000)) # Jump to the beginning of the defined range
                        video_player.play() # Ensure playback continues after seek
                        if video_player.page: video_player.update()
                        dialog_state.reframed_playback = False

                    # Check if current frame exceeds end_frame and loop (could be outdates if triggered by _on_frame_range_change_end from video_editor.py)
                    if dialog_state.playback_end_frame != -1 and current_frame >= dialog_state.playback_end_frame:
                        video_player.seek(int((dialog_state.playback_start_frame / effective_fps) * 1000)) # Jump to the beginning of the defined range
                        video_player.play() # Ensure playback continues after seek
                        if video_player.page: # Ensure control is mounted before updating
                             video_player.update() # Update the control state if necessary
                        current_frame = dialog_state.playback_start_frame # Update current frame for display immediately

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
            time.sleep(0.1)

    # Track play/pause state globally
    dialog_state.video_is_playing[0] = autoplay

    def show_feedback(is_playing):
        icon = ft.Icons.PLAY_ARROW if is_playing else ft.Icons.PAUSE
        color = ft.Colors.WHITE70
        _show_video_feedback_icon(stack, icon, color)

    def play_or_pause_with_feedback():
        dialog_state.video_is_playing[0] = not dialog_state.video_is_playing[0]
        orig_play_or_pause()
        show_feedback(dialog_state.video_is_playing[0])

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
    if dialog_state.frame_update_timer is not None and hasattr(dialog_state.frame_update_timer, 'is_alive') and dialog_state.frame_update_timer.is_alive():
        dialog_state.frame_update_timer.cancel()
    dialog_state.frame_update_timer = threading.Timer(0.1, lambda: _update_frame_counter(video, frame_counter_text))
    dialog_state.frame_update_timer.daemon = True
    dialog_state.frame_update_timer.start()

    return stack

# Add a new handler function for video reframing to implement manual looping
def handle_video_reframing(video_control: Video):
    """Handler for video reframing to implement manual looping."""
    try:
        # This function is now primarily used to seek to the start of the defined range
        # when the range is updated from the slider.
        # Access the Video control from the stack if needed, although it should be the Video control already
        video_ctrl = video_control.controls[0] if isinstance(video_control, ft.Stack) and video_control.controls else video_control
        if video_ctrl and hasattr(video_ctrl, '_video_fps') and video_ctrl._video_fps > 0:
            video_ctrl.seek(int((dialog_state.playback_start_frame / video_ctrl._video_fps) * 1000)) # Jump to the beginning of the defined range
            video_ctrl.play()
            if video_ctrl.page: # Ensure control is mounted before updating
                 video_ctrl.update() # Update the control state if necessary
    except Exception as e:
        pass

# Add a new handler function for video completion to implement manual looping
def handle_video_completed(video_control: Video):
    """Handler for video completion to implement manual looping."""
    try:
        # When the video naturally completes, loop back to the start of the defined range
        # Access the Video control from the stack if needed, although it should be the Video control already
        video_ctrl = video_control.controls[0] if isinstance(video_control, ft.Stack) and video_control.controls else video_control
        if video_ctrl and hasattr(video_ctrl, '_video_fps') and video_ctrl._video_fps > 0:
            video_ctrl.seek(int((dialog_state.playback_start_frame / video_ctrl._video_fps) * 1000)) # Jump to the beginning of the defined range
            video_ctrl.play()
            if video_ctrl.page: # Ensure control is mounted before updating
                 video_ctrl.update() # Update the control state if necessary
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
    dialog_state.playback_start_frame = 0
    dialog_state.playback_end_frame = -1 # -1 indicates no specific end frame set

    # Seek to the beginning of the new video
    video_player.seek(0)

def update_dialog_title(page: ft.Page, new_video_path: str):
    """Update the dialog title to the new video's filename."""
    if hasattr(page, 'base_dialog') and page.base_dialog:
        page.base_dialog.title_text_control.value = f"{os.path.basename(new_video_path)}"
        if page.base_dialog.title_text_control.page:
            page.base_dialog.title_text_control.update()

def update_caption_and_message(video_path: str, caption_field: ft.TextField, neg_caption_field: ft.TextField, message_container: ft.Container):
    """Load and update the caption field and message container for the given video."""
    caption_value, neg_caption_value, message_control = load_caption_for_video(video_path) # Get both captions
    caption_field.value = caption_value
    neg_caption_field.value = neg_caption_value # Update negative caption field
    message_container.content = message_control
    # Ensure updates if controls are on page
    if caption_field.page:
        caption_field.update()
    if neg_caption_field.page:
        neg_caption_field.update()
    if message_container.page:
        message_container.update()

# === End GUI-Building Functions ===

# === Dialog Logic & Event Handlers ===

def switch_video_in_dialog(page: ft.Page, new_video_offset: int):
    """
    Switches the dialog to show a different video (prev/next) and updates all relevant controls.
    Loads the new video player in a background thread to prevent UI freezing.
    """
    # Stop the existing frame counter timer if it's running
    if dialog_state.frame_update_timer is not None and dialog_state.frame_update_timer.is_alive():
        dialog_state.dialog_is_open = False # Signal the thread to stop
        dialog_state.frame_update_timer.cancel()
        time.sleep(0.05) # Small sleep to allow the thread to exit its loop

    if not dialog_state.current_video_list_for_dialog or not dialog_state.current_video_path_for_dialog:
        return

    # Save the current caption before switching
    if dialog_state.active_caption_field_instance is not None and dialog_state.current_video_path_for_dialog:
        current_caption = dialog_state.active_caption_field_instance.value.strip()
        # Pass None for the callback when saving on video switch to prevent thumbnail refresh
        save_caption_for_video(page, dialog_state.current_video_path_for_dialog, current_caption, None, field_name='caption') # Specify field_name
        
    if dialog_state.active_caption_neg_field_instance is not None and dialog_state.current_video_path_for_dialog:
        current_neg_caption = dialog_state.active_caption_neg_field_instance.value.strip()
        # Pass None for the callback when saving on video switch to prevent thumbnail refresh
        save_caption_for_video(page, dialog_state.current_video_path_for_dialog, current_neg_caption, None, field_name='negative_caption') # Specify field_name

    idx = dialog_state.current_video_list_for_dialog.index(dialog_state.current_video_path_for_dialog)
    new_idx = (idx + new_video_offset) % len(dialog_state.current_video_list_for_dialog)
    new_video_path = dialog_state.current_video_list_for_dialog[new_idx]
    dialog_state.last_video_loading_path = new_video_path
    dialog_state.current_video_path_for_dialog = new_video_path

    # Set dialog open flag to True before loading the new video and starting the new timer
    dialog_state.dialog_is_open = True

    def load_and_replace_video():
        if dialog_state.current_video_path_for_dialog != dialog_state.last_video_loading_path:
            return

        try:
            # Fully rebuild the dialog content and navigation controls
            main_content_ui, nav_controls = create_video_player_with_captions_content(
                page, dialog_state.current_video_path_for_dialog, dialog_state.current_video_list_for_dialog, dialog_state.active_on_caption_updated_callback
            )

            if page and hasattr(page, 'base_dialog'):
                page.base_dialog.show_dialog(content=main_content_ui, title=os.path.basename(dialog_state.current_video_path_for_dialog), new_width=VIDEO_PLAYER_DIALOG_WIDTH, title_prefix_controls=nav_controls)
                # page.dialog = page.base_dialog # This line might not be necessary if base_dialog.show_dialog handles it
                page.video_dialog_open = True
                page.video_dialog_hotkey_handler = lambda event: handle_caption_dialog_keyboard(page, event)
                # page.dialog.update() # Update is called later after seeking
            elif page:
                fallback_alert = ft.AlertDialog(title=ft.Text(os.path.basename(dialog_state.current_video_path_for_dialog)), content=main_content_ui, actions=[ft.TextButton("Close", on_click=lambda e: fallback_alert.close())], on_dismiss=lambda e: handle_dialog_dismiss(e))
                page.dialog = fallback_alert; fallback_alert.open = True; page.video_dialog_open = True; page.video_dialog_hotkey_handler = lambda event: handle_caption_dialog_keyboard(page, event); # page.update() # Update is called later after seeking

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

        # Now that the video player is on the page and dialog_state.active_video_player_instance is set,
        # ensure its source is updated and then update the slider based on its frames.
        # The build_video_player already sets the source, but this ensures consistency.
        if dialog_state.active_video_player_instance:
             # Access the Video control from the stack before updating source and seeking
            video_ctrl = dialog_state.active_video_player_instance.controls[0] if dialog_state.active_video_player_instance.controls else None
            if video_ctrl:
                # Removed the call to update_video_player_source as playlist is not settable
                # update_video_player_source(video_ctrl, dialog_state.current_video_path_for_dialog) # Update the source explicitly
                video_ctrl.seek(0) # Seek to the very beginning

        # Update the page after all changes are made to the dialog content and video player state
        if page: page.update()

    if page:
        # Run the loading and replacement logic directly
        load_and_replace_video()

# === Keyboard Event Handler ===
from ui.flet_hotkeys import AUTO_VIDEO_PLAYBACK, VIDEO_PLAY_PAUSE_KEY, VIDEO_NEXT_KEY, VIDEO_PREV_KEY


# === Keyboard Event Handler ===
def handle_caption_dialog_keyboard(page: ft.Page, e: ft.KeyboardEvent):
    """
    Handles keyboard events for the video captions dialog, including play/pause toggle and navigation.
    Uses keybindings from ui.flet_hotkeys.
    """
    try:
        if dialog_state.caption_field_is_focused:
            return  # Do not trigger any hotkeys while typing in the caption field
        # Play/pause
        if hasattr(e, 'key') and e.key == VIDEO_PLAY_PAUSE_KEY:
            if dialog_state.active_video_player_instance is not None:
                video_ctrl = None
                try:
                    # Access the Video control from the stack
                    video_ctrl = dialog_state.active_video_player_instance.controls[0]
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
        return False

# --- Main Dialog Construction ---
def create_video_player_with_captions_content(page: ft.Page, video_path: str, video_list: list, on_caption_updated_callback: callable = None) -> tuple[ft.Column, list[ft.Control]]:
    """
    Builds the main content column and navigation controls for the video player with captions dialog.
    """
    dialog_state.current_video_path_for_dialog = video_path
    dialog_state.current_video_list_for_dialog = video_list
    dialog_state.active_on_caption_updated_callback = on_caption_updated_callback
    dialog_state.last_video_loading_path = video_path
    dialog_state.active_page_ref = page # Set the global page reference

    nav_controls = build_navigation_controls(
        lambda e: switch_video_in_dialog(page, -1),
        lambda e: switch_video_in_dialog(page, 1)
    )

    _active_video_player_instance = build_video_player(video_path, autoplay=AUTO_VIDEO_PLAYBACK)
    dialog_state.active_video_player_instance = _active_video_player_instance # Update the instance in the state

    # Explicitly reset playback range for the new video
    dialog_state.playback_start_frame = 0
    dialog_state.playback_end_frame = -1
    dialog_state.reframed_playback = True # Signal frame counter to seek to start

    caption_value, neg_caption_value, message_control = load_caption_for_video(video_path) # Get both captions
    _active_caption_field_instance = build_caption_field(initial_value=caption_value)
    _active_caption_neg_field_instance = build_caption_neg_field(initial_value=neg_caption_value) # Pass neg_caption_value
    dialog_state.active_caption_field_instance = _active_caption_field_instance # Update the instance in the state
    dialog_state.active_caption_neg_field_instance = _active_caption_neg_field_instance # Update the instance in the state

    # Define callbacks for video_editor functions that will be run in a thread
    def on_crop_click(e):
        # This will use the handle_crop_video_click from video_editor
        page.run_thread(lambda: video_editor.handle_crop_video_click(page, dialog_state.active_width_field_instance, dialog_state.active_height_field_instance, dialog_state.current_video_path_for_dialog, video_list=dialog_state.current_video_list_for_dialog, on_caption_updated_callback=dialog_state.active_on_caption_updated_callback))

    def on_crop_all_click(e):
        page.run_thread(lambda: video_editor.handle_crop_all_videos(page, dialog_state.active_width_field_instance, dialog_state.active_height_field_instance, dialog_state.current_video_list_for_dialog, dialog_state.active_on_caption_updated_callback))

    def on_get_closest(e):
        # This function is typically synchronous and updates UI fields directly
        video_editor.handle_set_closest_div32(dialog_state.active_width_field_instance, dialog_state.active_height_field_instance, dialog_state.current_video_path_for_dialog, page)

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
        video_list=dialog_state.current_video_list_for_dialog,
        on_caption_updated_callback=dialog_state.active_on_caption_updated_callback,
        video_player_instance=dialog_state.active_video_player_instance, # Pass the video player instance itself
    )

    # Store the instances globally
    dialog_state.frame_range_slider_instance = frame_range_slider
    dialog_state.start_value_text_instance = start_value_text
    dialog_state.end_value_text_instance = end_value_text
    dialog_state.total_frames_text_instance = total_frames_text
    dialog_state.active_width_field_instance = _active_width_field_instance # Update the instance in the state
    dialog_state.active_height_field_instance = _active_height_field_instance # Update the instance in the state

    _active_message_container_instance = build_message_container(content=message_control)
    dialog_state.active_message_container_instance = _active_message_container_instance # Update the instance in the state

    content_column = ft.Column(
        controls=[
            ft.Row([dialog_state.active_video_player_instance], alignment=ft.MainAxisAlignment.CENTER),
            # Wrap both caption fields in a responsive row
            ft.ResponsiveRow([
                dialog_state.active_caption_field_instance,
                dialog_state.active_caption_neg_field_instance,
            ], spacing=10),
            crop_controls_row, # This row now contains controls built by video_editor
            ft.Row([
                dialog_state.active_message_container_instance
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN, vertical_alignment=ft.CrossAxisAlignment.CENTER)
        ],
        spacing=10, tight=True,
        scroll=ft.ScrollMode.ADAPTIVE
    )

    # Set dialog open flag for frame counter should be handled in switch_video_in_dialog or open_video_captions_dialog
    # dialog_state.dialog_is_open = True # Removed from here

    return content_column, nav_controls

def open_video_captions_dialog(page: ft.Page, video_path: str, video_list=None, on_caption_updated_callback: callable = None):
    """
    Opens the video captions dialog for the given video and list.
    """
    global dialog_state # Need global here because we are setting dialog_state.dialog_is_open
    if not video_path:
        return
    if video_list is None:
        video_list = [video_path]

    video_filename = os.path.basename(video_path)
    dialog_title_text = f"{video_filename}"

    # Set dialog open flag for frame counter before creating content
    dialog_state.dialog_is_open = True

    main_content_ui, nav_prefix_controls = create_video_player_with_captions_content(page, video_path, video_list, on_caption_updated_callback)

    desired_width = VIDEO_PLAYER_DIALOG_WIDTH

    if hasattr(page, 'base_dialog') and page.base_dialog:
        page.base_dialog._on_dismiss_callback = handle_dialog_dismiss
        page.base_dialog.show_dialog(content=main_content_ui, title=dialog_title_text, new_width=desired_width, title_prefix_controls=nav_prefix_controls)
        page.dialog = page.base_dialog
        page.video_dialog_open = True
        page.video_dialog_hotkey_handler = lambda event: handle_caption_dialog_keyboard(page, event)
        page.dialog.update()

        # Now that the video player is on the page and dialog_state.active_video_player_instance is set,
        # ensure its source is updated and then update the slider based on its frames.
        # The build_video_player already sets the source, but this ensures consistency.
        # update_video_player_source(dialog_state.active_video_player_instance, video_path) # May not be needed if autoplay handles it correctly initially

        update_dialog_title(page, video_path) # Ensure title is correct
    else:
        print("Error: Base dialog (PopupDialogBase) not found on page.")
        fallback_alert = ft.AlertDialog(title=ft.Text(dialog_title_text), content=main_content_ui, actions=[ft.TextButton("Close", on_click=lambda e: fallback_alert.close())], on_dismiss=lambda e: handle_dialog_dismiss(e))
        page.dialog = fallback_alert; fallback_alert.open = True; page.video_dialog_open = True; page.video_dialog_hotkey_handler = lambda event: handle_caption_dialog_keyboard(page, event); page.update()


def update_playback_range(video_player_stack: ft.Stack, start_frame: int, end_frame: int):
    """Updates the global playback range and seeks the video to the start frame."""
    global dialog_state
    dialog_state.playback_start_frame = start_frame
    dialog_state.playback_end_frame = end_frame
    dialog_state.reframed_playback = True
    # Access the Video control from the stack
    video_ctrl = video_player_stack.controls[0] if video_player_stack and video_player_stack.controls else None
    if video_ctrl and hasattr(video_ctrl, '_video_fps') and video_ctrl._video_fps > 0:
        seek_time_ms = int((dialog_state.playback_start_frame / video_ctrl._video_fps) * 1000)
        video_ctrl.seek(seek_time_ms)

# --- Global flag for caption field focus state ---

def _caption_field_on_focus(e):
    dialog_state.caption_field_is_focused = True

def _caption_field_on_blur(e):
    dialog_state.caption_field_is_focused = False

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
    tf.col = 9
    return tf

def build_caption_neg_field(initial_value: str = "") -> ft.TextField:
    """Create and return a styled TextField for neg video captions."""
    tf = create_textfield(
        label="Video Negative Caption",
        value=initial_value,
        hint_text="Enter/edit negative caption for this video...",
        multiline=True, min_lines=3, max_lines=6,
        expand=True
    )
    tf.on_focus = _caption_field_on_focus
    tf.on_blur = _caption_field_on_blur
    tf.col = 3
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

def handle_update_caption_click(page: ft.Page):
    """
    Handles the Update button click: saves the caption and updates UI.
    """
    global dialog_state
    new_caption = dialog_state.active_caption_field_instance.value.strip()
    new_neg_caption = dialog_state.active_caption_neg_field_instance.value.strip()
    save_caption_for_video(page, dialog_state.current_video_path_for_dialog, new_caption, dialog_state.active_on_caption_updated_callback, field_name='caption') # Specify field_name
    save_caption_for_video(page, dialog_state.current_video_path_for_dialog, new_neg_caption, dialog_state.active_on_caption_updated_callback, field_name='negative_caption') # Specify field_name
    if dialog_state.active_caption_field_instance and dialog_state.active_message_container_instance:
        update_caption_and_message(dialog_state.current_video_path_for_dialog, dialog_state.active_caption_field_instance, dialog_state.active_caption_neg_field_instance, dialog_state.active_message_container_instance) # Pass neg field
    if page:
        page.update()

def handle_dialog_dismiss(e):
    """
    Handles the dialog dismissal event: saves the current caption if the caption field exists and stops the frame counter thread.
    """
    global dialog_state
    dialog_state.dialog_is_open = False  # Stop frame counter thread
    if dialog_state.frame_update_timer is not None and hasattr(dialog_state.frame_update_timer, 'is_alive') and dialog_state.frame_update_timer.is_alive():
        dialog_state.frame_update_timer.cancel()
    if dialog_state.active_caption_field_instance and dialog_state.current_video_path_for_dialog:
        current_caption = dialog_state.active_caption_field_instance.value.strip()
        current_neg_caption = dialog_state.active_caption_neg_field_instance.value.strip()
        # Pass page=None here as page may not be valid after dialog is dismissed
        save_caption_for_video(page=None, video_path=dialog_state.current_video_path_for_dialog, new_caption=current_caption, on_caption_updated_callback=dialog_state.active_on_caption_updated_callback, field_name='caption') # Specify field_name
        save_caption_for_video(page=None, video_path=dialog_state.current_video_path_for_dialog, new_caption=current_neg_caption, on_caption_updated_callback=dialog_state.active_on_caption_updated_callback, field_name='negative_caption') # Specify field_name
    if dialog_state.active_page_ref:
        dialog_state.active_page_ref.video_dialog_open = False
        dialog_state.active_page_ref.video_dialog_hotkey_handler = None
        dialog_state.active_page_ref.update()