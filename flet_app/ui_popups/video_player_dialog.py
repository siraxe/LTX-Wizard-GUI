# video_player_dialog.py
import flet as ft
import os
from ui._styles import create_textfield, create_styled_button, VIDEO_PLAYER_DIALOG_WIDTH, VIDEO_PLAYER_DIALOG_HEIGHT, BTN_STYLE2
from ui.flet_hotkeys import AUTO_VIDEO_PLAYBACK, VIDEO_PLAY_PAUSE_KEY, VIDEO_NEXT_KEY, VIDEO_PREV_KEY
from flet_video.video import Video, VideoMedia
import threading
import time
from typing import Optional, List, Callable, Tuple

from .video_dialog_class import dialog_state
from . import video_player_utils as vpu
from . import video_editor

MIN_OVERLAY_SIZE = 20

# --- UI Element Builders (Specific to this dialog, including moved controls) ---

def _caption_field_on_focus(e: ft.ControlEvent):
    dialog_state.caption_field_is_focused = True

def _caption_field_on_blur(e: ft.ControlEvent):
    dialog_state.caption_field_is_focused = False

def build_caption_field(initial_value: str = "") -> ft.TextField:
    tf = create_textfield(
        label="Video Caption", value=initial_value,
        hint_text="Enter/edit caption...",
        multiline=True, min_lines=3, max_lines=5,
        expand=True, col={'md': 8, 'sm': 12}
    )
    tf.on_focus = _caption_field_on_focus
    tf.on_blur = _caption_field_on_blur
    return tf

def build_caption_neg_field(initial_value: str = "") -> ft.TextField:
    tf = create_textfield(
        label="Negative Caption", value=initial_value,
        hint_text="Enter/edit negative caption...",
        multiline=True, min_lines=3, max_lines=5,
        expand=True, col={'md': 4, 'sm': 12}
    )
    tf.on_focus = _caption_field_on_focus
    tf.on_blur = _caption_field_on_blur
    return tf

def build_message_container(content: Optional[ft.Control] = None) -> ft.Container:
    return ft.Container(content=content, expand=True, padding=ft.padding.only(top=5))

def build_navigation_controls(on_prev: Callable, on_next: Callable) -> List[ft.Control]:
    return [
        ft.IconButton(ft.Icons.ARROW_LEFT, on_click=on_prev, tooltip="Previous video", icon_size=20),
        ft.IconButton(ft.Icons.ARROW_RIGHT, on_click=on_next, tooltip="Next video", icon_size=20)
    ]

def build_crop_controls_row(
    page: ft.Page,
    current_video_path: str,
    # Actions passed from the calling context (create_video_player_with_captions_content)
    on_crop_dimensions_action: Callable, # Corresponds to on_crop from _bak
    on_crop_all_action: Callable,       # Corresponds to on_crop_all from _bak
    on_get_closest_action: Callable,    # Corresponds to on_get_closest from _bak
    on_crop_editor_overlay_action: Callable, # For "Apply" crop from editor
    on_toggle_crop_editor_visibility: Callable, # Corresponds to on_crop_editor (to open/toggle) from _bak
    on_flip_horizontal_action: Callable,
    on_reverse_action: Callable,
    on_time_remap_action: Callable, # Takes speed_value
    on_cut_to_frames_action: Callable, # Takes start_frame, end_frame
    on_split_to_video_action: Callable, # Takes split_frame
    on_cut_all_videos_to_max_action: Callable, # Takes num_frames
    video_list: Optional[List[str]] = None,
    on_caption_updated_callback: Optional[Callable] = None):
    metadata = vpu.get_video_metadata(current_video_path)
    original_frames = metadata.get('total_frames', 100) if metadata else 100
    original_fps = metadata.get('fps', 30.0) if metadata else 30.0
    if original_frames <= 0: original_frames = 100 # Ensure positive for slider
    if original_fps <= 0: original_fps = 30.0

    initial_width_val = ""
    initial_height_val = ""

    if metadata:
        w_str, h_str = vpu.calculate_closest_div32_dimensions(current_video_path)
        if w_str and h_str:
            initial_width_val, initial_height_val = w_str, h_str
        elif metadata.get('width') and metadata.get('height'):
            initial_width_val, initial_height_val = str(metadata['width']), str(metadata['height'])

    width_field = create_textfield(label="Width", value=initial_width_val, col=8, keyboard_type=ft.KeyboardType.NUMBER)
    height_field = create_textfield(label="Height", value=initial_height_val, col=8, keyboard_type=ft.KeyboardType.NUMBER)

    dialog_state.active_width_field_instance = width_field
    dialog_state.active_height_field_instance = height_field

    add_button = create_styled_button(text="+", on_click=lambda e: video_editor.handle_size_add(width_field, height_field, current_video_path, page), col=4, button_style=BTN_STYLE2)
    # Using "substract_button" text and wiring as per _bak.py structure
    substract_button = create_styled_button(text="-", on_click=lambda e: video_editor.handle_size_sub(width_field, height_field, current_video_path, page), col=4, button_style=BTN_STYLE2)

    crop_button = create_styled_button(text="Crop", on_click=on_crop_dimensions_action, col=3, button_style=BTN_STYLE2)
    crop_all_button = create_styled_button(text="Crop All", on_click=on_crop_all_action, col=5, button_style=BTN_STYLE2)
    closes_button = create_styled_button(text="Closest", on_click=on_get_closest_action, col=4, button_style=BTN_STYLE2)

    crop_buttons_row_internal = ft.ResponsiveRow(controls=[crop_all_button, crop_button, closes_button],spacing=3,expand=True) # Renamed to avoid conflict

    # Using "Crop Editor" for toggle, "Apply" for applying overlay crop
    crop_editor_button = create_styled_button(text="Crop Editor", on_click=on_toggle_crop_editor_visibility, col=6, button_style=BTN_STYLE2)
    crop_editor_apply_button = create_styled_button(text="Apply Crop", on_click=on_crop_editor_overlay_action, col=6, button_style=BTN_STYLE2, tooltip="Apply crop based on the visual editor overlay")

    # Frame Slider components
    frame_range_slider = ft.RangeSlider(
        min=0, max=original_frames, start_value=0, end_value=original_frames,
        divisions=original_frames if original_frames > 0 else None, label="{value}", round=0, expand=True)
    dialog_state.frame_range_slider_instance = frame_range_slider

    start_value_text = ft.Text(f"Start: {int(frame_range_slider.start_value or 0)}", size=12)
    dialog_state.start_value_text_instance = start_value_text

    end_value_text = ft.Text(f"End: {int(frame_range_slider.end_value or original_frames)}", size=12)
    dialog_state.end_value_text_instance = end_value_text

    total_val = int((frame_range_slider.end_value or original_frames) - (frame_range_slider.start_value or 0))
    total_frames_text = ft.Text(f"Total: {total_val}", size=12)
    dialog_state.total_frames_text_instance = total_frames_text
    
    # Connect to the existing function in video_player_dialog.py for slider changes
    frame_range_slider.on_change_end = lambda e_slider: update_playback_range_and_seek_video(int(e_slider.control.start_value), int(e_slider.control.end_value))
    frame_range_slider.on_change = lambda e_slider: None # As in _bak.py

    frame_slider_col = ft.Column( # As in _bak.py structure
        controls=[frame_range_slider,
                    ft.Row([start_value_text, total_frames_text, end_value_text], alignment=ft.MainAxisAlignment.SPACE_BETWEEN)
        ],
        alignment=ft.MainAxisAlignment.CENTER,
        spacing=5,
    )

    # Time Remap components
    time_remap_value_text = ft.Text(f"Remapped: {original_frames}", size=12) # Initial text from _bak.py was "Remapped frames: "
    time_slider = ft.Slider(
        min=0.1, max=2.0, value=1.0, divisions=99, # divisions 19 from current dialog, _bak.py had 99
        label="{value}x Speed", round=1, expand=True, # label from current dialog, _bak.py had {value}x
        on_change=lambda e_slider: (
            setattr(time_remap_value_text, 'value', f"Remapped: {int(original_frames / e_slider.control.value)}"), # Text from current dialog
            time_remap_value_text.update() if time_remap_value_text.page else None
        )
    )

    # Action Buttons, wired to new action parameters
    flip_horizontal_button = create_styled_button(text="Flip Horizontal", on_click=on_flip_horizontal_action, col=6, button_style=BTN_STYLE2)
    cut_to_frames_button = create_styled_button(text="Cut to Frames", on_click=lambda e: on_cut_to_frames_action(int(frame_range_slider.start_value or 0), int(frame_range_slider.end_value or 0)), col=6, button_style=BTN_STYLE2)
    split_to_video_button = create_styled_button(text="Split to Video", on_click=lambda e: on_split_to_video_action(int(frame_range_slider.start_value or 0)), col=6, button_style=BTN_STYLE2) # col=6 from _bak
    reverse_button = create_styled_button(text="Reverse", on_click=on_reverse_action, col=6, button_style=BTN_STYLE2)
    time_remap_button = create_styled_button(text="Time Remap", on_click=lambda e: on_time_remap_action(time_slider.value), col=6, button_style=BTN_STYLE2)

    num_to_cut_to = create_textfield(label="num", value=str(original_frames // 2 if original_frames > 1 else 150), col=4, keyboard_type=ft.KeyboardType.NUMBER) # col=4 from _bak
    cut_all_videos_to_max_button = create_styled_button(text="Cut All Videos to", on_click=lambda e: on_cut_all_videos_to_max_action(int(num_to_cut_to.value or 0)), col=8, button_style=BTN_STYLE2) # col=8 from _bak

    # Column 1: Crop Operations (structure from _bak.py)
    crop_column_restored = ft.Column( # Changed from ResponsiveRow in _bak.py to Column for proper layout with col=4
        controls=[
            ft.ResponsiveRow([width_field, add_button]),
            ft.ResponsiveRow([height_field, substract_button]), # Use substract_button
            ft.ResponsiveRow([crop_buttons_row_internal]), # Row of CropAll/Crop/Closest
            ft.Divider(height=1),
            ft.ResponsiveRow([crop_editor_button, crop_editor_apply_button]) # Row of EditorOpen/EditorApply
        ],
        # alignment=ft.MainAxisAlignment.CENTER, # Columns don't take MainAxisAlignment
        spacing=3,
        col={'md': 4}, # Use dict for responsive col
        scale=0.9 # scale not directly applicable to ft.Column, apply to ft.Container if needed or adjust padding/margins
    )

    # Column 2: Frame Operations (structure from _bak.py)
    frame_controls_column_restored = ft.Column(
        controls=[
            frame_slider_col,
            ft.ResponsiveRow([flip_horizontal_button, cut_to_frames_button]),
            split_to_video_button # Directly in column as per _bak
        ],
        spacing=5,
        col={'md': 4},
        scale=0.9
    )

    # Column 3: Time Operations (structure from _bak.py)
    time_controls_column_restored = ft.Column(
        controls=[
            ft.ResponsiveRow([time_slider, ft.Row([time_remap_value_text], alignment=ft.MainAxisAlignment.CENTER, spacing=5)]), # Wrapped text in Row for alignment
            ft.ResponsiveRow([reverse_button, time_remap_button]),
            ft.ResponsiveRow([cut_all_videos_to_max_button, num_to_cut_to]),
        ],
        spacing=5,
        col={'md': 4},
        scale=0.9
    )
    
    # The main container row, matching _bak.py's returnBox
    # Applying scale here if needed, or on individual columns through wrappers. Flet's scale is on individual controls.
    # For simplicity, scale is omitted here but can be added by wrapping columns in ft.Container(..., scale=0.9)
    returnBox = ft.ResponsiveRow(
        controls=[crop_column_restored, frame_controls_column_restored, time_controls_column_restored],
        spacing=10, # spacing from current dialog's main row, _bak.py had 3 for its final row
        vertical_alignment=ft.CrossAxisAlignment.START # from current dialog
    )
    return returnBox


# === Data & Utility Functions (UI Focused Wrappers) ===
def load_caption_ui_elements(video_path: str) -> Tuple[str, str, Optional[ft.Text]]:
    caption_value, neg_caption_value, message_str = vpu.load_caption_for_video(video_path)
    loaded_message_control = None
    if message_str:
        color = ft.Colors.RED_400 if "Error" in message_str else ft.Colors.AMBER_700
        loaded_message_control = ft.Text(message_str, color=color, size=12)
    return caption_value, neg_caption_value, loaded_message_control

def save_caption_with_ui_feedback(page: ft.Page, video_path: str, new_caption: str, on_caption_updated_callback: Optional[Callable] = None, field_name: str = "caption") -> bool:
    success, message = vpu.save_caption_for_video(video_path, new_caption, field_name)
    if page:
        page.snack_bar = ft.SnackBar(ft.Text(message), open=True)
        if success and on_caption_updated_callback: on_caption_updated_callback() # Removed () for direct call if it was a method
        page.update()
    return success

def update_dialog_title(page: ft.Page, video_path: str):
    if page and hasattr(page, 'base_dialog') and page.base_dialog and hasattr(page.base_dialog, 'title'):
        try:
            base_name = os.path.basename(video_path)
            if hasattr(page.base_dialog, 'title_text_control') and page.base_dialog.title_text_control:
                 page.base_dialog.title_text_control.value = f"Video: {base_name}"
                 if page.base_dialog.title_text_control.page: page.base_dialog.title_text_control.update()
            elif isinstance(page.base_dialog.title, ft.Text):
                page.base_dialog.title.value = f"Video: {base_name}"
                if page.base_dialog.title.page: page.base_dialog.title.update()
        except Exception as e: print(f"Error updating dialog title: {e}")

def update_video_player_source(video_player: Video, new_path: str, autoplay: bool = False):
    if video_player:
        video_player.playlist = [VideoMedia(resource=new_path)]
        video_player.autoplay = autoplay
        if video_player.page: video_player.update()
        if autoplay: video_player.play()

def update_caption_and_message_ui(caption_text_val: str, neg_caption_text_val: str, message_control_val: Optional[ft.Control] = None):
    if dialog_state.active_caption_field_instance:
        dialog_state.active_caption_field_instance.value = caption_text_val
        if dialog_state.active_caption_field_instance.page: dialog_state.active_caption_field_instance.update()
    if dialog_state.active_caption_neg_field_instance:
        dialog_state.active_caption_neg_field_instance.value = neg_caption_text_val
        if dialog_state.active_caption_neg_field_instance.page: dialog_state.active_caption_neg_field_instance.update()
    if dialog_state.active_message_container_instance:
        dialog_state.active_message_container_instance.content = message_control_val
        if dialog_state.active_message_container_instance.page: dialog_state.active_message_container_instance.update()

# === Video Player Callbacks ===
def _update_frame_counter(video_player: Video, frame_counter_text_ctrl: ft.Text):
    if video_player is None or frame_counter_text_ctrl is None: return
    effective_fps = getattr(video_player, '_video_fps', 30.0)
    total_frames_attr = getattr(video_player, '_total_frames', 1)
    if not isinstance(effective_fps, (int, float)) or effective_fps <= 0: effective_fps = 30.0

    while dialog_state.dialog_is_open:
        if not video_player.page: break
        try:
            current_position_ms = video_player.get_current_position(wait_timeout=0.2)
            if current_position_ms is not None:
                current_position_ms = max(0, current_position_ms)
                current_frame = int((current_position_ms / 1000.0) * effective_fps)
                current_frame = max(0, min(current_frame, total_frames_attr -1 if total_frames_attr > 0 else 0 ))

                if dialog_state.reframed_playback:
                    start_ms = int((dialog_state.playback_start_frame / effective_fps) * 1000)
                    video_player.seek(start_ms)
                    if video_player.page: video_player.update()
                    dialog_state.reframed_playback = False
                    current_frame = dialog_state.playback_start_frame

                if dialog_state.playback_end_frame != -1 and current_frame >= dialog_state.playback_end_frame:
                    start_ms = int((dialog_state.playback_start_frame / effective_fps) * 1000)
                    video_player.seek(start_ms)
                    if video_player.page: video_player.update()
                    current_frame = dialog_state.playback_start_frame

                frame_str = f"{current_frame+1:03d} / {total_frames_attr:03d}"
                if frame_counter_text_ctrl.page:
                    frame_counter_text_ctrl.value = frame_str
                    frame_counter_text_ctrl.update()
            else:
                if frame_counter_text_ctrl.page:
                    frame_counter_text_ctrl.value = f"--- / {total_frames_attr:03d}"
                    frame_counter_text_ctrl.update()
        except Exception: pass
        time.sleep(0.1)

def _reset_video_restarting_flag_for_overlay():
    dialog_state.is_video_restarting = False
    if dialog_state.last_video_error: dialog_state.last_video_error = ""

def _show_video_feedback_icon(stack: ft.Stack, icon_name: ft.Icons, color: ft.Colors):
    if dialog_state.video_feedback_overlay is not None:
        dialog_state.video_feedback_overlay.visible = True
        dialog_state.video_feedback_overlay.content = ft.Icon(icon_name, size=48, color=color)
        if stack.page: stack.update()
        if dialog_state.video_feedback_timer: dialog_state.video_feedback_timer.cancel()

        page_ref = getattr(stack, 'page', None)
        hide_action = lambda: _hide_feedback_overlay(stack) if stack.page else None
        if page_ref and hasattr(page_ref, 'run_on_main'):
            dialog_state.video_feedback_timer = threading.Timer(0.7, lambda: page_ref.run_on_main(hide_action))
        else:
            dialog_state.video_feedback_timer = threading.Timer(0.7, hide_action)
        dialog_state.video_feedback_timer.daemon = True
        dialog_state.video_feedback_timer.start()

def _hide_feedback_overlay(stack: ft.Stack):
    if dialog_state.video_feedback_overlay is not None:
        dialog_state.video_feedback_overlay.visible = False
        if stack.page: stack.update()

def build_video_player(video_path: str, autoplay: bool = False) -> Tuple[ft.Stack, Video]:
    dialog_state.dialog_is_open = True
    metadata = vpu.get_video_metadata(video_path)
    video_fps = metadata.get('fps', 30.0) if metadata else 30.0
    total_frames = metadata.get('total_frames', 1) if metadata else 1
    if video_fps <= 0: video_fps = 30.0
    if total_frames <=0: total_frames = 1

    video_player_control = Video(
        playlist=[VideoMedia(resource=video_path)], autoplay=autoplay,
        width=VIDEO_PLAYER_DIALOG_WIDTH - 40, height=VIDEO_PLAYER_DIALOG_HEIGHT - 40,
        expand=False, show_controls=True, playlist_mode="none",
        on_completed=lambda e: handle_video_completed(e.control),
    )
    video_player_control._video_fps = video_fps
    video_player_control._total_frames = total_frames

    initial_overlay_width, initial_overlay_height = 200, 200
    dialog_state.overlay_visual_instance = ft.Container(
        width=initial_overlay_width, height=initial_overlay_height,
        border=ft.border.all(2, ft.Colors.RED_ACCENT_700),
        bgcolor=ft.Colors.with_opacity(0.2, ft.Colors.WHITE),
        visible=dialog_state.overlay_is_visible,
    )
    content_w, content_h = VIDEO_PLAYER_DIALOG_WIDTH - 40, VIDEO_PLAYER_DIALOG_HEIGHT - 40
    initial_left, initial_top = (content_w - initial_overlay_width) / 2, (content_h - initial_overlay_height) / 2

    dialog_state.overlay_control_instance = ft.GestureDetector(
        content=dialog_state.overlay_visual_instance,
        left=initial_left, top=initial_top,
        width=initial_overlay_width, height=initial_overlay_height,
        on_pan_start=on_overlay_pan_start, on_pan_update=on_overlay_pan_update, on_pan_end=on_overlay_pan_end,
        drag_interval=0, visible=dialog_state.overlay_is_visible,
    )
    frame_counter_text = ft.Text(f"001 / {total_frames:03d}", color=ft.Colors.WHITE70, size=12)
    dialog_state.video_feedback_overlay = ft.Container(
        content=ft.Icon(ft.Icons.PLAY_ARROW, size=48, color=ft.Colors.WHITE70),
        alignment=ft.alignment.center, visible=False,
        width=content_w, height=content_h,
    )
    stack_elements = [
        video_player_control, dialog_state.video_feedback_overlay,
        ft.Container(content=frame_counter_text, alignment=ft.alignment.bottom_right, padding=ft.padding.only(bottom=5, right=10)),
        dialog_state.overlay_control_instance,
    ]
    video_stack = ft.Stack(stack_elements, width=content_w, height=content_h)

    dialog_state.video_is_playing[0] = autoplay
    def play_or_pause_with_feedback_local():
        is_playing = video_player_control.is_playing()
        if is_playing: video_player_control.pause(); dialog_state.video_is_playing[0] = False
        else: video_player_control.play(); dialog_state.video_is_playing[0] = True
        icon = ft.Icons.PAUSE if dialog_state.video_is_playing[0] else ft.Icons.PLAY_ARROW
        _show_video_feedback_icon(video_stack, icon, ft.Colors.WHITE70)
    video_player_control.play_or_pause = play_or_pause_with_feedback_local

    if dialog_state.frame_update_timer and dialog_state.frame_update_timer.is_alive():
        dialog_state.frame_update_timer.cancel()
    dialog_state.frame_update_timer = threading.Timer(0.1, lambda: _update_frame_counter(video_player_control, frame_counter_text))
    dialog_state.frame_update_timer.daemon = True
    dialog_state.frame_update_timer.start()
    return video_stack, video_player_control

def handle_video_reframing(video_control_maybe_stack):
    try:
        video_ctrl = video_control_maybe_stack
        if isinstance(video_control_maybe_stack, ft.Stack): video_ctrl = video_control_maybe_stack.controls[0]
        if video_ctrl and hasattr(video_ctrl, '_video_fps') and video_ctrl._video_fps > 0:
            start_frame = max(0, dialog_state.playback_start_frame)
            seek_pos_ms = int((start_frame / video_ctrl._video_fps) * 1000)
            video_ctrl.seek(seek_pos_ms)
            if video_ctrl.page: video_ctrl.update()
    except Exception as e: print(f"Error in handle_video_reframing: {e}")

def handle_video_completed(video_control: Video):
    current_time = time.time()
    if (current_time - dialog_state.last_completion_processed_time < 0.5): return
    if dialog_state.is_processing_completion: return

    try:
        dialog_state.is_processing_completion = True
        dialog_state.last_completion_processed_time = current_time
        dialog_state.is_video_restarting = True
        dialog_state.last_video_error = ""
        if dialog_state.video_restart_timer and dialog_state.video_restart_timer.is_alive():
            dialog_state.video_restart_timer.cancel()
        dialog_state.video_restart_timer = threading.Timer(0.3, _reset_video_restarting_flag_for_overlay)
        dialog_state.video_restart_timer.daemon = True
        dialog_state.video_restart_timer.start()

        if video_control and hasattr(video_control, '_video_fps') and video_control._video_fps > 0:
            start_frame = max(0, dialog_state.playback_start_frame)
            seek_position_ms = int((start_frame / video_control._video_fps) * 1000)
            video_control.seek(seek_position_ms)
            video_control.play()
            if video_control.page: video_control.update()
        else:
            dialog_state.last_video_error = "Video control invalid for restart on completion."
    except Exception as e:
        dialog_state.last_video_error = str(e)
    finally:
        dialog_state.is_processing_completion = False

# === Dialog Logic & Event Handlers ===
def switch_video_in_dialog(page: ft.Page, new_video_offset: int):
    if dialog_state.frame_update_timer and dialog_state.frame_update_timer.is_alive():
        dialog_state.dialog_is_open = False
        dialog_state.frame_update_timer.join(timeout=0.2)
        dialog_state.frame_update_timer = None
    if not dialog_state.current_video_list_for_dialog or not dialog_state.current_video_path_for_dialog: return

    if dialog_state.active_caption_field_instance:
        save_caption_with_ui_feedback(page, dialog_state.current_video_path_for_dialog, dialog_state.active_caption_field_instance.value.strip(), None, 'caption')
    if dialog_state.active_caption_neg_field_instance:
        save_caption_with_ui_feedback(page, dialog_state.current_video_path_for_dialog, dialog_state.active_caption_neg_field_instance.value.strip(), None, 'negative_caption')

    new_video_path = vpu.get_next_video_path(dialog_state.current_video_list_for_dialog, dialog_state.current_video_path_for_dialog, new_video_offset)
    if not new_video_path: return

    dialog_state.current_video_path_for_dialog = new_video_path
    dialog_state.dialog_is_open = True

    main_content_ui, nav_controls = create_video_player_with_captions_content(
        page, dialog_state.current_video_path_for_dialog,
        dialog_state.current_video_list_for_dialog,
        dialog_state.active_on_caption_updated_callback
    )
    if hasattr(page, 'base_dialog') and page.base_dialog:
        page.base_dialog.show_dialog(
            content=main_content_ui, title=os.path.basename(dialog_state.current_video_path_for_dialog),
            new_width=VIDEO_PLAYER_DIALOG_WIDTH, title_prefix_controls=nav_controls
        )
        page.dialog = page.base_dialog
        page.video_dialog_open = True
        page.video_dialog_hotkey_handler = lambda event: handle_caption_dialog_keyboard(page, event)
    else:
        fallback_alert = ft.AlertDialog(title=ft.Text(os.path.basename(dialog_state.current_video_path_for_dialog)), content=main_content_ui, actions=[ft.TextButton("Close", on_click=lambda e: fallback_alert.close())], on_dismiss=lambda e: handle_dialog_dismiss(page))
        page.dialog = fallback_alert; fallback_alert.open = True
        page.video_dialog_open = True; page.video_dialog_hotkey_handler = lambda event: handle_caption_dialog_keyboard(page, event)

    if dialog_state.active_video_player_instance:
        dialog_state.active_video_player_instance.seek(0)
        if dialog_state.active_video_player_instance.page: dialog_state.active_video_player_instance.update()
    if page: page.update()


def handle_caption_dialog_keyboard(page: ft.Page, e: ft.KeyboardEvent):
    try:
        if dialog_state.caption_field_is_focused: return
        key = getattr(e, 'key', None)
        if key == VIDEO_PLAY_PAUSE_KEY and dialog_state.active_video_player_instance:
            dialog_state.active_video_player_instance.play_or_pause()
        elif key == VIDEO_PREV_KEY: switch_video_in_dialog(page, -1)
        elif key == VIDEO_NEXT_KEY: switch_video_in_dialog(page, 1)
        elif key == "C" and not dialog_state.caption_field_is_focused:
            dialog_state.c_key_scaling_active = True
    except Exception as ex: print(f"Keyboard handler error: {ex}")

# --- Main Dialog Construction ---
def create_video_player_with_captions_content(page: ft.Page, video_path: str, video_list: List[str], on_caption_updated_callback: Optional[Callable] = None) -> Tuple[ft.Column, List[ft.Control]]:
    dialog_state.current_video_path_for_dialog = video_path
    dialog_state.current_video_list_for_dialog = video_list
    dialog_state.active_on_caption_updated_callback = on_caption_updated_callback
    dialog_state.active_page_ref = page

    def toggle_crop_editor_overlay_visibility(e=None):
        dialog_state.overlay_is_visible = not dialog_state.overlay_is_visible
        if dialog_state.overlay_control_instance:
            dialog_state.overlay_control_instance.visible = dialog_state.overlay_is_visible
            if dialog_state.overlay_control_instance.page: dialog_state.overlay_control_instance.update()
        if dialog_state.overlay_visual_instance:
            dialog_state.overlay_visual_instance.visible = dialog_state.overlay_is_visible
            if dialog_state.overlay_visual_instance.page: dialog_state.overlay_visual_instance.update()
        if page: page.update()
    dialog_state.active_toggle_crop_visibility_func = toggle_crop_editor_overlay_visibility

    nav_controls = build_navigation_controls(lambda e: switch_video_in_dialog(page, -1), lambda e: switch_video_in_dialog(page, 1))
    video_player_stack, actual_video_player = build_video_player(video_path, autoplay=AUTO_VIDEO_PLAYBACK)
    dialog_state.active_video_player_instance = actual_video_player

    metadata = vpu.get_video_metadata(video_path)
    dialog_state.playback_start_frame = 0
    dialog_state.playback_end_frame = metadata.get('total_frames', -1) if metadata else -1
    dialog_state.reframed_playback = True

    caption_val, neg_caption_val, message_ui_element = load_caption_ui_elements(video_path)
    dialog_state.active_caption_field_instance = build_caption_field(initial_value=caption_val)
    dialog_state.active_caption_neg_field_instance = build_caption_neg_field(initial_value=neg_caption_val)

    on_crop_dim_action = lambda e: page.run_thread(
        video_editor.handle_crop_video_click,
        page, dialog_state.active_width_field_instance, dialog_state.active_height_field_instance,
        dialog_state.current_video_path_for_dialog, dialog_state.current_video_list_for_dialog,
        dialog_state.active_on_caption_updated_callback
    )
    on_crop_all_act = lambda e: page.run_thread(
        video_editor.handle_crop_all_videos,
        page, 
        dialog_state.current_video_path_for_dialog, # Added
        dialog_state.active_width_field_instance, 
        dialog_state.active_height_field_instance,
        dialog_state.current_video_list_for_dialog, 
        dialog_state.active_on_caption_updated_callback
    )
    on_get_closest_act = lambda e: video_editor.handle_set_closest_div32(
        dialog_state.active_width_field_instance, dialog_state.active_height_field_instance,
        dialog_state.current_video_path_for_dialog, page
    )
    on_crop_overlay_apply_act = lambda e: page.run_thread(
        video_editor._perform_crop_from_editor_overlay,
        page, dialog_state.current_video_path_for_dialog,
        dialog_state.current_video_list_for_dialog, dialog_state.active_on_caption_updated_callback
    )
    on_flip_act = lambda e: page.run_thread(video_editor.on_flip_horizontal, page, dialog_state.current_video_path_for_dialog, dialog_state.current_video_list_for_dialog, dialog_state.active_on_caption_updated_callback)
    on_rev_act = lambda e: page.run_thread(video_editor.on_reverse, page, dialog_state.current_video_path_for_dialog, dialog_state.current_video_list_for_dialog, dialog_state.active_on_caption_updated_callback)
    on_remap_act = lambda speed_val: page.run_thread(video_editor.on_time_remap, page, dialog_state.current_video_path_for_dialog, speed_val, dialog_state.current_video_list_for_dialog, dialog_state.active_on_caption_updated_callback)
    on_cut_act = lambda start_f, end_f: page.run_thread(video_editor.cut_to_frames, page, dialog_state.current_video_path_for_dialog, start_f, end_f, dialog_state.current_video_list_for_dialog, dialog_state.active_on_caption_updated_callback)
    on_split_act = lambda split_f: page.run_thread(video_editor.split_to_video, page, dialog_state.current_video_path_for_dialog, split_f, dialog_state.current_video_list_for_dialog, dialog_state.active_on_caption_updated_callback, dialog_state.active_video_player_instance)
    on_cut_all_max_act = lambda num_frames: page.run_thread(video_editor.cut_all_videos_to_max, page, dialog_state.current_video_path_for_dialog, dialog_state.current_video_list_for_dialog, num_frames, dialog_state.active_on_caption_updated_callback)

    editing_controls_row = build_crop_controls_row(
        page=page, current_video_path=video_path,
        on_crop_dimensions_action=on_crop_dim_action,
        on_crop_all_action=on_crop_all_act,
        on_get_closest_action=on_get_closest_act,
        on_crop_editor_overlay_action=on_crop_overlay_apply_act,
        on_toggle_crop_editor_visibility=toggle_crop_editor_overlay_visibility,
        on_flip_horizontal_action=on_flip_act,
        on_reverse_action=on_rev_act,
        on_time_remap_action=on_remap_act,
        on_cut_to_frames_action=on_cut_act,
        on_split_to_video_action=on_split_act,
        on_cut_all_videos_to_max_action=on_cut_all_max_act,
        video_list=video_list,
        on_caption_updated_callback=on_caption_updated_callback)
    dialog_state.active_message_container_instance = build_message_container(content=message_ui_element)

    content_column = ft.Column(
        controls=[
            ft.Row([video_player_stack], alignment=ft.MainAxisAlignment.CENTER),
            ft.ResponsiveRow([dialog_state.active_caption_field_instance, dialog_state.active_caption_neg_field_instance], spacing=10),
            editing_controls_row,
            ft.Row([dialog_state.active_message_container_instance], alignment=ft.MainAxisAlignment.START, vertical_alignment=ft.CrossAxisAlignment.CENTER)
        ],
        spacing=10, tight=True, scroll=ft.ScrollMode.ADAPTIVE
    )
    return content_column, nav_controls

def open_video_captions_dialog(page: ft.Page, video_path: str, video_list: Optional[List[str]]=None, on_caption_updated_callback: Optional[Callable] = None):
    if not video_path: return
    if video_list is None: video_list = [video_path]
    dialog_state.dialog_is_open = True
    main_content_ui, nav_prefix_controls = create_video_player_with_captions_content(page, video_path, video_list, on_caption_updated_callback)

    dialog_title_text = os.path.basename(video_path)
    desired_width = VIDEO_PLAYER_DIALOG_WIDTH

    if hasattr(page, 'base_dialog') and page.base_dialog:
        page.base_dialog._on_dismiss_callback = lambda e: handle_dialog_dismiss(page)
        page.base_dialog.show_dialog(content=main_content_ui, title=dialog_title_text, new_width=desired_width, title_prefix_controls=nav_prefix_controls)
        page.dialog = page.base_dialog
        page.video_dialog_open = True
        page.video_dialog_hotkey_handler = lambda event: handle_caption_dialog_keyboard(page, event)
        update_dialog_title(page, video_path)
    else:
        fallback_alert = ft.AlertDialog(title=ft.Text(dialog_title_text), content=main_content_ui, actions=[ft.TextButton("Close", on_click=lambda e: fallback_alert.close())], on_dismiss=lambda e: handle_dialog_dismiss(page))
        page.dialog = fallback_alert; fallback_alert.open = True
        page.video_dialog_open = True; page.video_dialog_hotkey_handler = lambda event: handle_caption_dialog_keyboard(page, event)

    if dialog_state.active_video_player_instance:
        dialog_state.active_video_player_instance.seek(0)
        if dialog_state.active_video_player_instance.page: dialog_state.active_video_player_instance.update()
    if page: page.update()

def update_playback_range_and_seek_video(start_frame: int, end_frame: int):
    dialog_state.playback_start_frame = start_frame
    dialog_state.playback_end_frame = end_frame
    dialog_state.reframed_playback = True
    active_player = dialog_state.active_video_player_instance
    if active_player and hasattr(active_player, '_video_fps') and active_player._video_fps > 0:
        seek_time_ms = int((start_frame / active_player._video_fps) * 1000)
        active_player.seek(seek_time_ms)
        if active_player.page: active_player.update()

    if dialog_state.start_value_text_instance and dialog_state.start_value_text_instance.page:
        dialog_state.start_value_text_instance.value = f"Start: {start_frame}"; dialog_state.start_value_text_instance.update()
    if dialog_state.end_value_text_instance and dialog_state.end_value_text_instance.page:
        dialog_state.end_value_text_instance.value = f"End: {end_frame}"; dialog_state.end_value_text_instance.update()
    if dialog_state.total_frames_text_instance and dialog_state.total_frames_text_instance.page:
        dialog_state.total_frames_text_instance.value = f"Total: {end_frame - start_frame}"; dialog_state.total_frames_text_instance.update()

def handle_dialog_dismiss(page: ft.Page):
    dialog_state.dialog_is_open = False
    if dialog_state.frame_update_timer and dialog_state.frame_update_timer.is_alive():
        dialog_state.frame_update_timer.join(timeout=0.2)
        dialog_state.frame_update_timer = None
    if dialog_state.active_caption_field_instance and dialog_state.current_video_path_for_dialog:
        vpu.save_caption_for_video(dialog_state.current_video_path_for_dialog, dialog_state.active_caption_field_instance.value.strip(), 'caption')
        if dialog_state.active_on_caption_updated_callback: dialog_state.active_on_caption_updated_callback() # Call if callable
    if dialog_state.active_caption_neg_field_instance and dialog_state.current_video_path_for_dialog:
        vpu.save_caption_for_video(dialog_state.current_video_path_for_dialog, dialog_state.active_caption_neg_field_instance.value.strip(), 'negative_caption')
        if dialog_state.active_on_caption_updated_callback: dialog_state.active_on_caption_updated_callback() # Call if callable
    if page:
        page.video_dialog_open = False
        page.video_dialog_hotkey_handler = None
    if dialog_state.overlay_is_visible and hasattr(dialog_state, 'active_toggle_crop_visibility_func') and callable(dialog_state.active_toggle_crop_visibility_func):
        dialog_state.active_toggle_crop_visibility_func()
    if hasattr(dialog_state, 'active_toggle_crop_visibility_func'):
        dialog_state.active_toggle_crop_visibility_func = None

    dialog_state.active_video_player_instance = None
    dialog_state.overlay_visual_instance = None
    dialog_state.overlay_control_instance = None

def on_overlay_pan_start(e: ft.DragStartEvent):
    if not dialog_state.overlay_control_instance or not dialog_state.overlay_visual_instance: return
    control = dialog_state.overlay_control_instance
    control_x, control_y, control_width, control_height = control.left or 0, control.top or 0, control.width or 0, control.height or 0
    dialog_state.overlay_pan_start_x, dialog_state.overlay_pan_start_y = e.global_x, e.global_y
    dialog_state.overlay_initial_box_left, dialog_state.overlay_initial_box_top = control_x, control_y
    dialog_state.overlay_initial_box_width, dialog_state.overlay_initial_box_height = control_width, control_height
    corner_threshold = 20
    if e.local_x < corner_threshold and e.local_y < corner_threshold: dialog_state.overlay_interaction_mode = "resize_tl"
    elif e.local_x > control_width - corner_threshold and e.local_y < corner_threshold: dialog_state.overlay_interaction_mode = "resize_tr"
    elif e.local_x < corner_threshold and e.local_y > control_height - corner_threshold: dialog_state.overlay_interaction_mode = "resize_bl"
    elif e.local_x > control_width - corner_threshold and e.local_y > control_height - corner_threshold: dialog_state.overlay_interaction_mode = "resize_br"
    else: dialog_state.overlay_interaction_mode = "pan"
    if dialog_state.overlay_interaction_mode.startswith("resize_") and dialog_state.c_key_scaling_active and control_width > 0 and control_height > 0:
        dialog_state.aspect_ratio_locked, dialog_state.locked_aspect_ratio = True, control_width / control_height
    else: dialog_state.aspect_ratio_locked = False

def on_overlay_pan_update(e: ft.DragUpdateEvent):
    if not dialog_state.overlay_control_instance or not dialog_state.overlay_visual_instance or dialog_state.overlay_interaction_mode == "none" or dialog_state.is_video_restarting: return
    control, visual = dialog_state.overlay_control_instance, dialog_state.overlay_visual_instance
    video_player_width, video_player_height = VIDEO_PLAYER_DIALOG_WIDTH - 40, VIDEO_PLAYER_DIALOG_HEIGHT - 40
    delta_x, delta_y = e.global_x - dialog_state.overlay_pan_start_x, e.global_y - dialog_state.overlay_pan_start_y
    new_left, new_top, new_width, new_height = dialog_state.overlay_initial_box_left, dialog_state.overlay_initial_box_top, dialog_state.overlay_initial_box_width, dialog_state.overlay_initial_box_height
    mode = dialog_state.overlay_interaction_mode

    if mode == "pan": new_left += delta_x; new_top += delta_y
    elif mode.startswith("resize_"):
        potential_w, potential_h = new_width, new_height
        if mode == "resize_br": potential_w += delta_x; potential_h += delta_y
        elif mode == "resize_bl": potential_w -= delta_x; potential_h += delta_y
        elif mode == "resize_tr": potential_w += delta_x; potential_h -= delta_y
        elif mode == "resize_tl": potential_w -= delta_x; potential_h -= delta_y

        if dialog_state.aspect_ratio_locked and dialog_state.locked_aspect_ratio > 0:
            if mode in ["resize_br", "resize_tr"]: new_width, new_height = potential_w, potential_w / dialog_state.locked_aspect_ratio
            else: new_height, new_width = potential_h, potential_h * dialog_state.locked_aspect_ratio
        else: new_width, new_height = potential_w, potential_h

        if mode == "resize_bl": new_left = dialog_state.overlay_initial_box_left + (dialog_state.overlay_initial_box_width - new_width)
        elif mode == "resize_tr": new_top = dialog_state.overlay_initial_box_top + (dialog_state.overlay_initial_box_height - new_height)
        elif mode == "resize_tl":
            new_left = dialog_state.overlay_initial_box_left + (dialog_state.overlay_initial_box_width - new_width)
            new_top = dialog_state.overlay_initial_box_top + (dialog_state.overlay_initial_box_height - new_height)

    new_width, new_height = max(MIN_OVERLAY_SIZE, new_width), max(MIN_OVERLAY_SIZE, new_height)
    new_left = max(0, min(new_left, video_player_width - new_width))
    new_top = max(0, min(new_top, video_player_height - new_height))
    new_width = min(new_width, video_player_width - new_left)
    new_height = min(new_height, video_player_height - new_top)
    new_width, new_height = max(MIN_OVERLAY_SIZE, new_width), max(MIN_OVERLAY_SIZE, new_height)
    if new_left + new_width > video_player_width: new_left = video_player_width - new_width
    if new_top + new_height > video_player_height: new_top = video_player_height - new_height
    new_left, new_top = max(0, new_left), max(0, new_top)

    control.left, control.top, control.width, control.height = new_left, new_top, new_width, new_height
    visual.width, visual.height = new_width, new_height
    if control.page: control.update()
    if visual.page: visual.update()

def on_overlay_pan_end(e: ft.DragEndEvent):
    dialog_state.overlay_interaction_mode = "none"
    dialog_state.aspect_ratio_locked = False
    if dialog_state.c_key_scaling_active: dialog_state.c_key_scaling_active = False