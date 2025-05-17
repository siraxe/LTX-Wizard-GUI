import flet as ft
import os
import json
from ui._styles import create_textfield, create_styled_button, VIDEO_PLAYER_DIALOG_WIDTH, VIDEO_PLAYER_DIALOG_HEIGHT
from flet_video.video import Video, VideoMedia

# --- Data/Utility Helper Functions ---

def load_caption_for_video(video_path: str) -> tuple[str, ft.Text | None]:
    """
    Loads the caption for a given video from its captions.json file.
    Returns a tuple of (caption_text, message_control_if_no_caption).
    """
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
    """
    Saves the new caption for the given video to its captions.json file.
    Shows a snackbar on the page for success/failure. Calls callback if provided.
    Returns True if successful, False otherwise.
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
    """
    Returns the next video path in the list given the current path and offset.
    Wraps around if at the end/start. Returns None if not found.
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

# --- End Data/Utility Helper Functions ---

_active_video_player_instance = None
_active_caption_field_instance = None
_active_message_container_instance = None
_active_on_caption_updated_callback = None
_current_video_list_for_dialog = []
_current_video_path_for_dialog = ""
_last_video_loading_path = ""


# --- GUI-Building Functions ---
def build_video_player(video_path: str) -> Video:
    """Creates and returns a Video player control for the given video path."""
    return Video(
        playlist=[VideoMedia(resource=video_path)],
        autoplay=True,
        width=VIDEO_PLAYER_DIALOG_WIDTH - 40,
        height=VIDEO_PLAYER_DIALOG_HEIGHT - 40,
        expand=False,
        show_controls=True,
    )

def build_caption_field(initial_value: str = "") -> ft.TextField:
    """Creates and returns a styled TextField for video captions."""
    return create_textfield(
        label="Video Caption",
        value=initial_value,
        hint_text="Enter/edit caption for this video...",
        multiline=True, min_lines=3, max_lines=6,
        expand=True
    )

def build_message_container(content=None) -> ft.Container:
    """Creates a container for displaying messages (e.g., no caption found)."""
    return ft.Container(content=content, expand=True)

def build_navigation_controls(on_prev, on_next) -> list[ft.Control]:
    """Creates navigation arrow controls for previous/next video."""
    left_arrow = ft.IconButton(ft.Icons.ARROW_LEFT, on_click=on_prev, tooltip="Previous video", icon_size=20)
    right_arrow = ft.IconButton(ft.Icons.ARROW_RIGHT, on_click=on_next, tooltip="Next video", icon_size=20)
    return [left_arrow, right_arrow]

def build_update_button(on_click) -> ft.Control:
    """Creates the Update button for saving captions."""
    return create_styled_button("Update", on_click=on_click, width=100)

def update_video_player_source(video_player: Video, new_video_path: str):
    """Updates the video player's source to the new video path."""
    video_player.stop()
    # Clear existing playlist items
    while video_player.playlist:
        video_player.playlist_remove(0)
    # Add the new video
    video_player.playlist_add(VideoMedia(resource=new_video_path))

    # Ensure the control updates its visual representation
    if video_player.page:
        video_player.update()

    video_player.jump_to(0) # Start from the beginning
    video_player.play()

def update_dialog_title(page: ft.Page, new_video_path: str):
    """Updates the dialog title to the new video's filename."""
    if hasattr(page, 'base_dialog') and page.base_dialog:
        page.base_dialog.title_text_control.value = f"{os.path.basename(new_video_path)}"
        if page.base_dialog.title_text_control.page:
            page.base_dialog.title_text_control.update()

def update_caption_and_message(video_path: str, caption_field: ft.TextField, message_container: ft.Container):
    """
    Loads and updates the caption field and message container for the given video.
    """
    caption_value, message_control = load_caption_for_video(video_path)
    caption_field.value = caption_value
    message_container.content = message_control

# --- End GUI-Building Functions ---

# --- Dialog Logic and Event Handlers ---

def switch_video_in_dialog(page: ft.Page, new_video_offset: int):
    """
    Switches the dialog to show a different video (prev/next) and updates all relevant controls.
    Loads the new video player in a background thread to prevent UI freezing.
    """
    global _active_video_player_instance, _active_caption_field_instance, _active_message_container_instance
    global _current_video_list_for_dialog, _current_video_path_for_dialog, _active_on_caption_updated_callback
    global _last_video_loading_path

    new_video_path = get_next_video_path(_current_video_list_for_dialog, _current_video_path_for_dialog, new_video_offset)
    if not new_video_path or new_video_path == _current_video_path_for_dialog:
        return

    _current_video_path_for_dialog = new_video_path
    update_dialog_title(page, new_video_path)
    _last_video_loading_path = new_video_path

    if _active_caption_field_instance and _active_message_container_instance:
        update_caption_and_message(new_video_path, _active_caption_field_instance, _active_message_container_instance)

    def load_and_replace_video():
        global _active_video_player_instance
        global _current_video_path_for_dialog, _last_video_loading_path

        if _current_video_path_for_dialog != _last_video_loading_path:
            return

        if _active_video_player_instance:
            try:
                _active_video_player_instance.stop()
                while _active_video_player_instance.playlist:
                    _active_video_player_instance.playlist_remove(0)
                _active_video_player_instance.playlist_add(VideoMedia(resource=new_video_path))
                _active_video_player_instance.jump_to(0)
                _active_video_player_instance.play()

                if _active_video_player_instance.page:
                     _active_video_player_instance.update()
                if page:
                    page.update()

            except Exception as e:
                print(f"Error updating video player in background thread: {e}")
                if page:
                    def show_error_snackbar(error_message):
                        if page.snack_bar is None:
                            page.snack_bar = ft.SnackBar(ft.Text(f"Error loading video: {error_message}"), open=True)
                        else:
                            page.snack_bar.content = ft.Text(f"Error loading video: {error_message}")
                            page.snack_bar.open = True
                        page.update()

                    page.run_sync(lambda: show_error_snackbar(e))

    if page:
        page.run_thread(load_and_replace_video)

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

def handle_caption_dialog_keyboard(page: ft.Page, e: ft.KeyboardEvent):
    """
    Handles keyboard events for the video captions dialog (navigation, save, close).
    """
    global _active_caption_field_instance
    caption_field_is_focused = _active_caption_field_instance is not None and getattr(_active_caption_field_instance, 'focused', False)
    if e.key == 'Escape' and not caption_field_is_focused:
        if hasattr(page, 'base_dialog') and getattr(page.base_dialog, 'visible', False):
            page.base_dialog.hide_dialog()
            return
    if not caption_field_is_focused:
        if e.key == "[":
            switch_video_in_dialog(page, -1)
        elif e.key == "]":
            switch_video_in_dialog(page, 1)
    if e.ctrl and e.key.lower() == "s" and _active_caption_field_instance:
        handle_update_caption_click(page)

# --- Main Dialog Construction ---
def create_video_player_with_captions_content(page: ft.Page, video_path: str, video_list: list, on_caption_updated_callback: callable = None) -> tuple[ft.Column, list[ft.Control]]:
    """
    Builds the main content column and navigation controls for the video player with captions dialog.
    """
    global _active_video_player_instance, _active_caption_field_instance, _active_message_container_instance
    global _current_video_list_for_dialog, _current_video_path_for_dialog, _active_on_caption_updated_callback
    global _last_video_loading_path

    _current_video_path_for_dialog = video_path
    _current_video_list_for_dialog = video_list
    _active_on_caption_updated_callback = on_caption_updated_callback
    _last_video_loading_path = video_path

    nav_controls = build_navigation_controls(
        lambda e: switch_video_in_dialog(page, -1),
        lambda e: switch_video_in_dialog(page, 1)
    )

    _active_video_player_instance = build_video_player(video_path)

    caption_value, message_control = load_caption_for_video(video_path)
    _active_caption_field_instance = build_caption_field(initial_value=caption_value)

    _active_message_container_instance = build_message_container(content=message_control)

    update_button = build_update_button(on_click=lambda e: handle_update_caption_click(page))

    content_column = ft.Column(
        controls=[
            ft.Row([_active_video_player_instance], alignment=ft.MainAxisAlignment.CENTER),
            _active_caption_field_instance,
            ft.Row([
                _active_message_container_instance, update_button
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
        print("Error: video_path is required for open_video_captions_dialog.")
        return
    if video_list is None:
        video_list = [video_path]

    video_filename = os.path.basename(video_path)
    dialog_title_text = f"{video_filename}"

    main_content_ui, nav_prefix_controls = create_video_player_with_captions_content(page, video_path, video_list, on_caption_updated_callback)

    desired_width = VIDEO_PLAYER_DIALOG_WIDTH

    if hasattr(page, 'base_dialog') and page.base_dialog:
        page.base_dialog.show_dialog(content=main_content_ui, title=dialog_title_text, new_width=desired_width, title_prefix_controls=nav_prefix_controls)
    else:
        print("Error: Base dialog (PopupDialogBase) not found on page.")
        fallback_alert = ft.AlertDialog(title=ft.Text(dialog_title_text), content=main_content_ui, actions=[ft.TextButton("Close", on_click=lambda e: fallback_alert.close())])
        page.dialog = fallback_alert; fallback_alert.open = True; page.update()

    def caption_dialog_keyboard_handler(e: ft.KeyboardEvent):
        handle_caption_dialog_keyboard(page, e)

    page.video_dialog_hotkey_handler = caption_dialog_keyboard_handler
    page.video_dialog_open = True
