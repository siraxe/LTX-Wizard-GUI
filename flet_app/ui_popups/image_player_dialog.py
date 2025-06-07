# image_player_dialog.py
import flet as ft
import base64 # Added import for base64
import os
import json
import subprocess
from ui._styles import create_textfield, create_styled_button, IMAGE_PLAYER_DIALOG_WIDTH, IMAGE_PLAYER_DIALOG_HEIGHT, BTN_STYLE2
from ui.flet_hotkeys import AUTO_PLAYBACK, NEXT_KEY, PREV_KEY
from typing import Optional, List, Callable, Tuple

from .image_dialog_class import image_dialog_state
from . import image_player_utils as ipu
from . import image_editor
from .batch_crop_warning import show_batch_crop_warning_dialog
from .image_player_utils import calculate_contained_image_dimensions, get_image_metadata

MIN_OVERLAY_SIZE = 20

# --- UI Element Builders (Specific to this dialog, including moved controls) ---

def _caption_field_on_focus(e: ft.ControlEvent):
    image_dialog_state.caption_field_is_focused = True

def _caption_field_on_blur(e: ft.ControlEvent):
    image_dialog_state.caption_field_is_focused = False

def build_caption_field(initial_value: str = "") -> ft.TextField:
    tf = create_textfield(
        label="Image Caption", value=initial_value,
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
        ft.IconButton(ft.Icons.ARROW_LEFT, on_click=on_prev, tooltip="Previous image", icon_size=20),
        ft.IconButton(ft.Icons.ARROW_RIGHT, on_click=on_next, tooltip="Next image", icon_size=20),
        ft.IconButton(ft.Icons.IMAGE, on_click=lambda e: open_in_image_editor(), tooltip="Open in image editor", icon_size=20)
    ]

def build_crop_controls_row(
    page: ft.Page,
    current_image_path: str,
    # Actions passed from the calling context
    on_crop_dimensions_action: Callable,
    on_crop_all_action: Callable,
    on_get_closest_action: Callable,
    on_crop_editor_overlay_action: Callable,
    on_toggle_crop_editor_visibility: Callable,
    image_list: Optional[List[str]] = None,
    on_caption_updated_callback: Optional[Callable] = None):

    # Get metadata directly from the image file to populate initial width/height fields
    metadata = ipu.get_image_metadata(current_image_path)
    
    initial_width_val = ""
    initial_height_val = ""

    if metadata:
        # Calculate closest divisible by 32 dimensions based on actual image metadata
        w_str, h_str = ipu.calculate_closest_div32_dimensions(current_image_path)
        if w_str and h_str:
            initial_width_val, initial_height_val = w_str, h_str
        elif metadata.get('width') and metadata.get('height'):
            initial_width_val, initial_height_val = str(metadata['width']), str(metadata['height'])

    width_field = create_textfield(label="Width", value=initial_width_val, col=8, keyboard_type=ft.KeyboardType.NUMBER)
    height_field = create_textfield(label="Height", value=initial_height_val, col=8, keyboard_type=ft.KeyboardType.NUMBER)

    image_dialog_state.active_width_field_instance = width_field
    image_dialog_state.active_height_field_instance = height_field

    add_button = create_styled_button(text="+", on_click=lambda e: image_editor.handle_size_add(width_field, height_field, current_image_path, page), col=4, button_style=BTN_STYLE2)
    substract_button = create_styled_button(text="-", on_click=lambda e: image_editor.handle_size_sub(width_field, height_field, current_image_path, page), col=4, button_style=BTN_STYLE2)

    crop_button = create_styled_button(text="Crop", on_click=on_crop_dimensions_action, col=3, button_style=BTN_STYLE2)
    crop_all_button = create_styled_button(text="Crop All", on_click=on_crop_all_action, col=5, button_style=BTN_STYLE2)
    closes_button = create_styled_button(text="Closest", on_click=on_get_closest_action, col=4, button_style=BTN_STYLE2)

    crop_buttons_row_internal = ft.ResponsiveRow(controls=[crop_all_button, crop_button, closes_button],spacing=3,expand=True)

    crop_editor_button = create_styled_button(text="Crop Editor", on_click=on_toggle_crop_editor_visibility, col=6, button_style=BTN_STYLE2)
    crop_editor_apply_button = create_styled_button(text="Apply Crop", on_click=on_crop_editor_overlay_action, col=6, button_style=BTN_STYLE2, tooltip="Apply crop based on the visual editor overlay")

    crop_column_restored = ft.Column(
        controls=[
            ft.ResponsiveRow([width_field, add_button]),
            ft.ResponsiveRow([height_field, substract_button]),
            ft.ResponsiveRow([crop_buttons_row_internal]),
            ft.Divider(height=1),
            ft.ResponsiveRow([crop_editor_button, crop_editor_apply_button])
        ],
        spacing=3,
        col={'md': 12},
    )

    returnBox = ft.ResponsiveRow(
        controls=[crop_column_restored],
        spacing=10,
        vertical_alignment=ft.CrossAxisAlignment.START,
        col=6
    )
    return returnBox

def build_other_controls_row(
    page: ft.Page,
    current_image_path: str,
    # Placeholder actions for now
    on_flip_image_action: Callable,
    on_rotate_plus_90_action: Callable,
    on_rotate_minus_90_action: Callable):

    flip_button = create_styled_button(text="Flip Image", on_click=on_flip_image_action, col=4, button_style=BTN_STYLE2)
    rotate_plus_90_button = create_styled_button(text="+90", on_click=on_rotate_plus_90_action, col=4, button_style=BTN_STYLE2)
    rotate_minus_90_button = create_styled_button(text="-90", on_click=on_rotate_minus_90_action, col=4, button_style=BTN_STYLE2)

    other_buttons_row_internal = ft.ResponsiveRow(controls=[flip_button, rotate_plus_90_button, rotate_minus_90_button], spacing=3, expand=True)

    other_column = ft.Column(
        controls=[
            ft.ResponsiveRow([other_buttons_row_internal]),
        ],
        spacing=3,
        col={'md': 12},
    )

    returnBox = ft.ResponsiveRow(
        controls=[other_column],
        spacing=10,
        vertical_alignment=ft.CrossAxisAlignment.START,
        col=6
    )
    return returnBox


# === Data & Utility Functions (UI Focused Wrappers) ===
def load_caption_ui_elements(image_path: str) -> Tuple[str, str, Optional[ft.Text]]:
    caption_value, neg_caption_value, message_str = ipu.load_caption_for_image(image_path)
    loaded_message_control = None
    if message_str:
        color = ft.Colors.RED_400 if "Error" in message_str else ft.Colors.AMBER_700
        loaded_message_control = ft.Text(message_str, color=color, size=12)
    return caption_value, neg_caption_value, loaded_message_control

def save_caption_with_ui_feedback(page: ft.Page, image_path: str, new_caption: str, on_caption_updated_callback: Optional[Callable] = None, field_name: str = "caption") -> bool:
    success, message = ipu.save_caption_for_image(image_path, new_caption, field_name)
    if page:
        page.snack_bar = ft.SnackBar(ft.Text(message), open=True)
        if success and on_caption_updated_callback: on_caption_updated_callback(image_path) # Pass image_path to the callback
        page.update()
    return success

def update_dialog_title(page: ft.Page, image_path: str):
    if page and hasattr(page, 'base_dialog') and page.base_dialog:
        try:
            base_name = os.path.basename(image_path)
            new_title_value = f"{base_name}"
            
            if hasattr(page.base_dialog, 'title_text_control') and page.base_dialog.title_text_control:
                 page.base_dialog.title_text_control.value = new_title_value
                 if page.base_dialog.title_text_control.page: 
                     page.base_dialog.title_text_control.update()
            elif isinstance(page.base_dialog.title, ft.Text):
                page.base_dialog.title.value = new_title_value
                if page.base_dialog.title.page: 
                    page.base_dialog.title.update()
            
            # Ensure the base dialog itself is updated to reflect title changes
            if page.base_dialog.page:
                page.base_dialog.update()
        except Exception as e: print(f"Error updating dialog title: {e}")

def update_caption_and_message_ui(caption_text_val: str, neg_caption_text_val: str, message_control_val: Optional[ft.Control] = None):
    if image_dialog_state.active_caption_field_instance:
        image_dialog_state.active_caption_field_instance.value = caption_text_val
        if image_dialog_state.active_caption_field_instance.page: image_dialog_state.active_caption_field_instance.update()
    if image_dialog_state.active_caption_neg_field_instance:
        image_dialog_state.active_caption_neg_field_instance.value = neg_caption_text_val
        if image_dialog_state.active_caption_neg_field_instance.page: image_dialog_state.active_caption_neg_field_instance.update()
    if image_dialog_state.active_message_container_instance:
        image_dialog_state.active_message_container_instance.content = message_control_val
        if image_dialog_state.active_message_container_instance.page: image_dialog_state.active_message_container_instance.update()

def build_image_display(image_path: str) -> Tuple[ft.Stack, ft.Image]:
    """
    Build the image display with cropping overlay.
    
    Args:
        image_path: Path to the image to display
        
    Returns:
        Tuple containing:
        - The image stack with overlay controls
        - The image control for updating the displayed image
    """
    image_dialog_state.dialog_is_open = True
    
    # Get original image dimensions
    metadata = get_image_metadata(image_path)
    image_orig_w = metadata.get('width', 1) if metadata else 1
    image_orig_h = metadata.get('height', 1) if metadata else 1

    # Define the content area for the image player
    player_content_w = IMAGE_PLAYER_DIALOG_WIDTH - 40
    player_content_h = IMAGE_PLAYER_DIALOG_HEIGHT - 40

    # Calculate effective displayed dimensions and offsets for the image
    effective_image_w, effective_image_h, offset_x, offset_y = \
        calculate_contained_image_dimensions(image_orig_w, image_orig_h, player_content_w, player_content_h)

    # Store these in dialog state for pan handlers
    image_dialog_state.effective_image_w = effective_image_w
    image_dialog_state.effective_image_h = effective_image_h
    image_dialog_state.offset_x = offset_x
    image_dialog_state.offset_y = offset_y
    image_dialog_state.image_orig_w = image_orig_w
    image_dialog_state.image_orig_h = image_orig_h
    image_dialog_state.player_content_w = player_content_w
    image_dialog_state.player_content_h = player_content_h

    # Load image as base64
    encoded_string = None
    if os.path.exists(image_path):
        try:
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        except Exception as e:
            print(f"Error loading initial image as base64: {e}")

    image_control = ft.Image(
        src_base64=encoded_string,
        width=player_content_w, # Image control takes up full content area
        height=player_content_h,
        fit=ft.ImageFit.CONTAIN, # This scales the image proportionally
        expand=False,
    )
    
    # Create overlay for cropping
    # Initial overlay size relative to the effective image size, not the full container
    initial_overlay_width = min(200, effective_image_w)
    initial_overlay_height = min(200, effective_image_h)

    image_dialog_state.overlay_visual_instance = ft.Container(
        width=initial_overlay_width, 
        height=initial_overlay_height,
        border=ft.border.all(2, ft.Colors.RED_ACCENT_700),
        bgcolor=ft.Colors.with_opacity(0.2, ft.Colors.WHITE),
        visible=image_dialog_state.overlay_is_visible,
    )
    
    # Calculate initial position for the overlay (centered within the effective image area)
    initial_left = offset_x + (effective_image_w - initial_overlay_width) / 2
    initial_top = offset_y + (effective_image_h - initial_overlay_height) / 2
    
    # Create gesture detector for the overlay
    image_dialog_state.overlay_control_instance = ft.GestureDetector(
        content=image_dialog_state.overlay_visual_instance,
        left=initial_left, 
        top=initial_top,
        width=initial_overlay_width, 
        height=initial_overlay_height,
        on_pan_start=on_overlay_pan_start, 
        on_pan_update=on_overlay_pan_update, 
        on_pan_end=on_overlay_pan_end,
        drag_interval=0, 
        visible=image_dialog_state.overlay_is_visible,
    )
    
    # Feedback overlay (for operations like crop)
    image_dialog_state.image_feedback_overlay = ft.Container(
        content=ft.Icon(ft.Icons.IMAGE, size=48, color=ft.Colors.WHITE70),
        alignment=ft.alignment.center, 
        visible=False,
        width=player_content_w, # This overlay covers the whole player content area
        height=player_content_h,
    )
    
    # Build the stack with all elements
    stack_elements = [
        image_control,
        image_dialog_state.image_feedback_overlay,
        image_dialog_state.overlay_control_instance,
    ]
    
    image_stack = ft.Stack(
        stack_elements, 
        width=player_content_w, 
        height=player_content_h
    )
    
    return image_stack, image_control


def update_image_player_source(image_control: ft.Image, new_image_path: str):
    """
    Updates the source of the image player control using base64 encoding.
    """
    if image_control and os.path.exists(new_image_path):
        try:
            with open(new_image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
            
            image_control.src_base64 = encoded_string
            image_control.src = None # Clear src if it was set
            if image_control.page:
                image_control.update()
        except Exception as e:
            print(f"Error loading image as base64: {e}")
            image_control.src_base64 = None # Clear in case of error
            if image_control.page:
                image_control.update()

# === Dialog Logic & Event Handlers ===
def switch_image_in_dialog(page: ft.Page, new_image_offset: int):
    """
    Switch to the next or previous image in the image list.
    
    Args:
        page: The Flet page instance
        new_image_offset: The offset to move in the image list (1 for next, -1 for previous)
    """
    # Save current captions if fields exist
    if hasattr(image_dialog_state, 'active_caption_field_instance') and image_dialog_state.active_caption_field_instance:
        save_caption_with_ui_feedback(
            page, 
            image_dialog_state.image_path, 
            image_dialog_state.active_caption_field_instance.value.strip(), 
            None, 
            'caption'
        )
    
    if hasattr(image_dialog_state, 'active_caption_neg_field_instance') and image_dialog_state.active_caption_neg_field_instance:
        save_caption_with_ui_feedback(
            page, 
            image_dialog_state.image_path, 
            image_dialog_state.active_caption_neg_field_instance.value.strip(), 
            None, 
            'negative_caption'
        )
    
    # Get the new image path
    if not hasattr(image_dialog_state, 'image_list') or not image_dialog_state.image_list:
        print("No image list available for navigation")
        return

    current_image_path = image_dialog_state.image_path
    current_index = image_dialog_state.image_list.index(current_image_path) if current_image_path in image_dialog_state.image_list else -1

    if current_index == -1:
        print(f"Current image {current_image_path} not found in the image list")
        return

    new_index = (current_index + new_image_offset) % len(image_dialog_state.image_list)
    new_image_path = image_dialog_state.image_list[new_index]
    
    if not os.path.exists(new_image_path):
        print(f"Image not found: {new_image_path}")
        return

    # Update the image source using the robust update function
    if hasattr(image_dialog_state, 'active_image_player_instance') and image_dialog_state.active_image_player_instance:
        update_image_player_source(image_dialog_state.active_image_player_instance, new_image_path)
    
    # Update dialog state
    image_dialog_state.image_path = new_image_path
    
    # Update captions and UI
    caption_text, neg_caption_text, message_ui_element = load_caption_ui_elements(new_image_path)
    update_caption_and_message_ui(caption_text, neg_caption_text, message_ui_element)
    
    # Update dialog title
    update_dialog_title(page, new_image_path)
    
    # Reset overlay position based on new image's effective dimensions
    if hasattr(image_dialog_state, 'overlay_control_instance'):
        metadata = get_image_metadata(new_image_path)
        image_orig_w = metadata.get('width', 1) if metadata else 1
        image_orig_h = metadata.get('height', 1) if metadata else 1

        player_content_w = IMAGE_PLAYER_DIALOG_WIDTH - 40
        player_content_h = IMAGE_PLAYER_DIALOG_HEIGHT - 40

        effective_image_w, effective_image_h, offset_x, offset_y = \
            calculate_contained_image_dimensions(image_orig_w, image_orig_h, player_content_w, player_content_h)

        # Update dialog state with new effective dimensions
        image_dialog_state.effective_image_w = effective_image_w
        image_dialog_state.effective_image_h = effective_image_h
        image_dialog_state.offset_x = offset_x
        image_dialog_state.offset_y = offset_y
        image_dialog_state.image_orig_w = image_orig_w
        image_dialog_state.image_orig_h = image_orig_h
        image_dialog_state.player_content_w = player_content_w
        image_dialog_state.player_content_h = player_content_h

        # Recalculate initial overlay position based on new effective image size
        initial_overlay_width = min(200, effective_image_w)
        initial_overlay_height = min(200, effective_image_h)
        
        initial_left = offset_x + (effective_image_w - initial_overlay_width) / 2
        initial_top = offset_y + (effective_image_h - initial_overlay_height) / 2
        
        image_dialog_state.overlay_control_instance.left = initial_left
        image_dialog_state.overlay_control_instance.top = initial_top
        image_dialog_state.overlay_control_instance.width = initial_overlay_width
        image_dialog_state.overlay_control_instance.height = initial_overlay_height
        image_dialog_state.overlay_control_instance.update()

        if image_dialog_state.overlay_visual_instance:
            image_dialog_state.overlay_visual_instance.width = initial_overlay_width
            image_dialog_state.overlay_visual_instance.height = initial_overlay_height
            image_dialog_state.overlay_visual_instance.update()
    
    # Update the page
    if page: 
        page.update()

def handle_caption_dialog_keyboard(page: ft.Page, e: ft.KeyboardEvent):
    """
    Handle keyboard events for the image caption dialog.
    
    Args:
        page: The Flet page instance
        e: The keyboard event
    """
    try:
        # Skip if a caption field is focused
        if hasattr(image_dialog_state, 'caption_field_is_focused') and image_dialog_state.caption_field_is_focused:
            return
            
        key = getattr(e, 'key', None)
        ctrl = getattr(e, 'ctrl', False)
        shift = getattr(e, 'shift', False)
        
        # Navigation between images
        if key == NEXT_KEY:
            switch_image_in_dialog(page, 1)
            return
            
        if key == PREV_KEY:
            switch_image_in_dialog(page, -1)
            return
            
        # Close dialog with Escape
        if key == 'escape':
            if hasattr(image_dialog_state, 'overlay_is_visible') and image_dialog_state.overlay_is_visible:
                # Toggle overlay visibility if open
                if hasattr(image_dialog_state, 'active_toggle_crop_visibility_func'):
                    image_dialog_state.active_toggle_crop_visibility_func()
            else:
                # Close the dialog
                if hasattr(page, 'dialog') and page.dialog:
                    if hasattr(page.dialog, 'close'):
                        page.dialog.close()
                    else:
                        page.dialog.open = False
                        page.update()
            return
            
        # Toggle crop overlay with 'c'
        if key == 'c' and not ctrl and not shift:
            if hasattr(image_dialog_state, 'active_toggle_crop_visibility_func'):
                image_dialog_state.active_toggle_crop_visibility_func()
            image_dialog_state.c_key_scaling_active = True # Activate C-key scaling
            return
            
        # Save with Ctrl+S (removed as per user request)
        # if ctrl and key == 's':
        #     if hasattr(image_dialog_state, 'active_caption_field_instance') and image_dialog_state.active_caption_field_instance:
        #         save_caption_with_ui_feedback(
        #             page,
        #             image_dialog_state.image_path,
        #             image_dialog_state.active_caption_field_instance.value or "",
        #             image_dialog_state.on_caption_updated_callback,
        #             'caption'
        #         )
        #     return
            
    except Exception as ex:
        print(f"Error in handle_caption_dialog_keyboard: {ex}")
        import traceback
        traceback.print_exc()

# --- Main Dialog Construction ---
def create_image_player_with_captions_content(page: ft.Page, image_path: str, image_list: List[str], on_caption_updated_callback: Optional[Callable] = None) -> Tuple[ft.Column, List[ft.Control]]:
    image_dialog_state.current_image_path_for_dialog = image_path
    image_dialog_state.current_image_list_for_dialog = image_list
    image_dialog_state.active_on_caption_updated_callback = on_caption_updated_callback
    image_dialog_state.active_page_ref = page

    def toggle_crop_editor_overlay_visibility(e=None):
        image_dialog_state.overlay_is_visible = not image_dialog_state.overlay_is_visible
        if image_dialog_state.overlay_control_instance:
            image_dialog_state.overlay_control_instance.visible = image_dialog_state.overlay_is_visible
            if image_dialog_state.overlay_control_instance.page: image_dialog_state.overlay_control_instance.update()
        if image_dialog_state.overlay_visual_instance:
            image_dialog_state.overlay_visual_instance.visible = image_dialog_state.overlay_is_visible
            if image_dialog_state.overlay_visual_instance.page: image_dialog_state.overlay_visual_instance.update()
        if page: page.update()
    image_dialog_state.active_toggle_crop_visibility_func = toggle_crop_editor_overlay_visibility

    nav_controls = build_navigation_controls(lambda e: switch_image_in_dialog(page, -1), lambda e: switch_image_in_dialog(page, 1))
    image_player_stack, actual_image_player = build_image_display(image_path)
    image_dialog_state.active_image_player_instance = actual_image_player
    image_dialog_state.active_image_player_stack = image_player_stack # Store the stack instance

    # Removed video-specific metadata and playback state
    # metadata = ipu.get_video_metadata(video_path)
    # image_dialog_state.playback_start_frame = 0
    # image_dialog_state.playback_end_frame = metadata.get('total_frames', -1) if metadata else -1
    # image_dialog_state.reframed_playback = True

    caption_val, neg_caption_val, message_ui_element = load_caption_ui_elements(image_path)
    image_dialog_state.active_caption_field_instance = build_caption_field(initial_value=caption_val)
    image_dialog_state.active_caption_neg_field_instance = build_caption_neg_field(initial_value=neg_caption_val)

    on_crop_dim_action = lambda e: page.run_thread(
        image_editor.handle_crop_image_click,
        page, image_dialog_state.active_width_field_instance, image_dialog_state.active_height_field_instance,
        image_dialog_state.current_image_path_for_dialog, image_dialog_state.current_image_list_for_dialog,
        image_dialog_state.active_on_caption_updated_callback
    )
    on_crop_all_act = lambda e: show_batch_crop_warning_dialog(
        page,
        on_confirm=lambda: page.run_thread(
            image_editor.handle_crop_all_images,
            page, 
            image_dialog_state.current_image_path_for_dialog,
            image_dialog_state.active_width_field_instance, 
            image_dialog_state.active_height_field_instance,
            image_dialog_state.current_image_list_for_dialog,
            image_dialog_state.active_on_caption_updated_callback
        )
    )
    on_get_closest_act = lambda e: image_editor.handle_set_closest_div32(
        image_dialog_state.active_width_field_instance, image_dialog_state.active_height_field_instance,
        image_dialog_state.current_image_path_for_dialog, page
    )
    on_crop_overlay_apply_act = lambda e: page.run_thread(
        image_editor._perform_crop_from_editor_overlay,
        page, image_dialog_state.current_image_path_for_dialog,
        image_dialog_state.current_image_list_for_dialog, image_dialog_state.active_on_caption_updated_callback
    )

    crop_controls_row = build_crop_controls_row(
        page=page, current_image_path=image_path,
        on_crop_dimensions_action=on_crop_dim_action,
        on_crop_all_action=on_crop_all_act,
        on_get_closest_action=on_get_closest_act,
        on_crop_editor_overlay_action=on_crop_overlay_apply_act,
        on_toggle_crop_editor_visibility=toggle_crop_editor_overlay_visibility,
        image_list=image_list,
        on_caption_updated_callback=on_caption_updated_callback)
    
    on_flip_image_act = lambda e: page.run_thread(
        image_editor.handle_flip_image,
        page, image_dialog_state.current_image_path_for_dialog,
        image_dialog_state.current_image_list_for_dialog,
        image_dialog_state.active_on_caption_updated_callback
    )
    on_rotate_plus_90_act = lambda e: page.run_thread(
        image_editor.handle_rotate_image,
        page, image_dialog_state.current_image_path_for_dialog,
        image_dialog_state.current_image_list_for_dialog,
        image_dialog_state.active_on_caption_updated_callback,
        90
    )
    on_rotate_minus_90_act = lambda e: page.run_thread(
        image_editor.handle_rotate_image,
        page, image_dialog_state.current_image_path_for_dialog,
        image_dialog_state.current_image_list_for_dialog,
        image_dialog_state.active_on_caption_updated_callback,
        -90
    )

    other_control_row = build_other_controls_row(
        page=page, current_image_path=image_path,
        on_flip_image_action=on_flip_image_act,
        on_rotate_plus_90_action=on_rotate_plus_90_act,
        on_rotate_minus_90_action=on_rotate_minus_90_act
    )

    image_dialog_state.active_message_container_instance = build_message_container(content=message_ui_element)

    content_column = ft.Column(
        controls=[
            ft.Row([image_player_stack], alignment=ft.MainAxisAlignment.CENTER),
            ft.ResponsiveRow([image_dialog_state.active_caption_field_instance, image_dialog_state.active_caption_neg_field_instance], spacing=10),
            ft.ResponsiveRow([crop_controls_row, other_control_row]),
            ft.Row([image_dialog_state.active_message_container_instance], alignment=ft.MainAxisAlignment.START, vertical_alignment=ft.CrossAxisAlignment.CENTER)
        ],
        spacing=10, tight=True, scroll=ft.ScrollMode.ADAPTIVE
    )
    return content_column, nav_controls

def open_image_captions_dialog(page: ft.Page, image_path: str, image_list: Optional[List[str]] = None, on_caption_updated_callback: Optional[Callable] = None):
    """
    Open a dialog to view and edit an image with captions.
    
    Args:
        page: The Flet page instance
        image_path: Path to the image to display
        image_list: Optional list of image paths for navigation
        on_caption_updated_callback: Callback function when captions are updated
    """
    if not image_path or not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
        
    if image_list is None:
        image_list = [image_path]
        
    # Initialize dialog state
    image_dialog_state.dialog_is_open = True
    image_dialog_state.page = page
    image_dialog_state.image_path = image_path
    image_dialog_state.image_list = image_list
    image_dialog_state.on_caption_updated_callback = on_caption_updated_callback
    image_dialog_state.overlay_is_visible = False
    image_dialog_state.c_key_scaling_active = False # Initialize c_key_scaling_active
    
    # Load captions if available
    caption_text, neg_caption_text, message = ipu.load_caption_for_image(image_path)
    
    # Create UI elements
    main_content_ui, nav_prefix_controls = create_image_player_with_captions_content(page, image_path, image_list, on_caption_updated_callback)
    
    # Get the base filename for the dialog title
    dialog_title_text = os.path.basename(image_path)
    desired_width = IMAGE_PLAYER_DIALOG_WIDTH
    
    # Show the dialog using the base dialog if available
    if hasattr(page, 'base_dialog') and page.base_dialog:
        # Set the on_dismiss callback on the base dialog instance
        if hasattr(page.base_dialog, '_on_dismiss_callback'):
            page.base_dialog._on_dismiss_callback = lambda e: handle_dialog_dismiss(page)
            
        page.base_dialog.show_dialog(
            content=main_content_ui,
            title=dialog_title_text,
            title_prefix_controls=nav_prefix_controls,
            new_width=desired_width
        )
        page.dialog = page.base_dialog
        page.image_dialog_open = True
        page.image_dialog_hotkey_handler = lambda event: handle_caption_dialog_keyboard(page, event)
    else:
        # Fallback to a simple alert dialog
        fallback_alert = ft.AlertDialog(
            title=ft.Text(dialog_title_text),
            content=main_content_ui,
            actions=[ft.TextButton("Close", on_click=lambda e: fallback_alert.close())],
            on_dismiss=lambda e: handle_dialog_dismiss(page)
        )
        page.dialog = fallback_alert
        fallback_alert.open = True
        page.image_dialog_open = True
        page.image_dialog_hotkey_handler = lambda event: handle_caption_dialog_keyboard(page, event)
    
    # Update the page to show the dialog
    if page:
        page.update()

def open_in_image_editor():
    """
    Opens the currently selected image in the external image editor specified in settings.json.
    """
    settings_path = os.path.join(os.path.dirname(__file__), '..', 'settings.json')
    image_editor_path = None
    try:
        with open(settings_path, 'r') as f:
            settings = json.load(f)
            image_editor_path = settings.get("IMAGE_EDITOR_PATH")
    except FileNotFoundError:
        print(f"Error: settings.json not found at {settings_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode settings.json at {settings_path}")
        return

    if not image_editor_path:
        print("Error: IMAGE_EDITOR_PATH not found in settings.json")
        return

    current_image_path = image_dialog_state.image_path
    if not current_image_path or not os.path.exists(current_image_path):
        print(f"Error: No image selected or image not found at {current_image_path}")
        return

    try:
        # Use subprocess.Popen to open the image with the specified editor
        # This assumes the editor can be launched directly with the image path as an argument
        subprocess.Popen([image_editor_path, current_image_path], shell=True)
        print(f"Attempting to open {current_image_path} with {image_editor_path}")
    except Exception as e:
        print(f"Error opening image with external editor: {e}")
        print(f"Please ensure '{image_editor_path}' is a valid executable and can open '{current_image_path}'.")

def handle_dialog_dismiss(page: ft.Page):
    """
    Clean up resources when the dialog is dismissed.
    Optimized for better performance.
    """
    try:
        captions_saved = False # Initialize captions_saved
        # Save any unsaved captions if they've been modified
        if hasattr(image_dialog_state, 'image_path') and image_dialog_state.image_path:
            # Only save if there are unsaved changes
            if (hasattr(image_dialog_state, 'active_caption_field_instance') and 
                image_dialog_state.active_caption_field_instance and 
                getattr(image_dialog_state.active_caption_field_instance, 'dirty', True)):
                success = save_caption_with_ui_feedback( # Initialize success here
                    page, 
                    image_dialog_state.image_path, 
                    image_dialog_state.active_caption_field_instance.value.strip() if image_dialog_state.active_caption_field_instance.value else "", 
                    image_dialog_state.active_on_caption_updated_callback, # Pass the callback
                    'caption'
                )
                if success: captions_saved = True
            
            if (hasattr(image_dialog_state, 'active_caption_neg_field_instance') and 
                image_dialog_state.active_caption_neg_field_instance and 
                getattr(image_dialog_state.active_caption_neg_field_instance, 'dirty', True)):
                success = save_caption_with_ui_feedback(
                    page, 
                    image_dialog_state.image_path, 
                    image_dialog_state.active_caption_neg_field_instance.value.strip() if image_dialog_state.active_caption_neg_field_instance.value else "", 
                    image_dialog_state.active_on_caption_updated_callback, # Pass the callback
                    'negative_caption'
                )
                if success: captions_saved = True
        
        # Batch clear page attributes
        page_attrs = ['image_dialog_open', 'image_dialog_hotkey_handler']
        for attr in page_attrs:
            if hasattr(page, attr):
                delattr(page, attr)
        
        # Reset dialog state attributes to their default/initial values
        image_dialog_state.dialog = None
        image_dialog_state.overlay_visual_instance = None
        image_dialog_state.overlay_control_instance = None
        image_dialog_state.image_feedback_overlay = None
        image_dialog_state.active_caption_field_instance = None
        image_dialog_state.active_caption_neg_field_instance = None
        image_dialog_state.active_message_container_instance = None
        image_dialog_state.current_image_path_for_dialog = None
        image_dialog_state.current_image_list_for_dialog = None
        image_dialog_state.active_on_caption_updated_callback = None
        image_dialog_state.active_image_player_instance = None
        image_dialog_state.page = None
        image_dialog_state.image_path = None
        image_dialog_state.image_list = None
        image_dialog_state.on_caption_updated_callback = None
        image_dialog_state.active_toggle_crop_visibility_func = None
        image_dialog_state.overlay_is_visible = False
        image_dialog_state.caption_field_is_focused = False
        image_dialog_state.dialog_is_open = False
        image_dialog_state.c_key_scaling_active = False
        image_dialog_state.aspect_ratio_locked = False
        image_dialog_state.locked_aspect_ratio = 0.0
        image_dialog_state.overlay_pan_start_x = 0.0
        image_dialog_state.overlay_pan_start_y = 0.0
        image_dialog_state.overlay_initial_box_left = 0.0
        image_dialog_state.overlay_initial_box_top = 0.0
        image_dialog_state.overlay_initial_box_width = 0.0
        image_dialog_state.overlay_initial_box_height = 0.0
        image_dialog_state.overlay_interaction_mode = "none"
        image_dialog_state.active_width_field_instance = None
        image_dialog_state.active_height_field_instance = None
        image_dialog_state.active_image_player_stack = None # Reset the stack instance
        
        # Close dialog if still open - simplified logic
        if hasattr(page, 'dialog') and page.dialog:
            page.dialog.open = False
        
        # Single page update at the end
        try:
            page.update()
        except Exception as update_error:
            print(f"Error updating page: {update_error}")
            
    except Exception as ex:
        print(f"Error in handle_dialog_dismiss: {ex}")
        import traceback
        traceback.print_exc()

def on_overlay_pan_start(e: ft.DragStartEvent):
    if not image_dialog_state.overlay_control_instance or not image_dialog_state.overlay_visual_instance: return
    control = image_dialog_state.overlay_control_instance
    control_x, control_y, control_width, control_height = control.left or 0, control.top or 0, control.width or 0, control.height or 0
    image_dialog_state.overlay_pan_start_x, image_dialog_state.overlay_pan_start_y = e.global_x, e.global_y
    image_dialog_state.overlay_initial_box_left, image_dialog_state.overlay_initial_box_top = control_x, control_y
    image_dialog_state.overlay_initial_box_width, image_dialog_state.overlay_initial_box_height = control_width, control_height
    corner_threshold = 20
    if e.local_x < corner_threshold and e.local_y < corner_threshold: image_dialog_state.overlay_interaction_mode = "resize_tl"
    elif e.local_x > control_width - corner_threshold and e.local_y < corner_threshold: image_dialog_state.overlay_interaction_mode = "resize_tr"
    elif e.local_x < corner_threshold and e.local_y > control_height - corner_threshold: image_dialog_state.overlay_interaction_mode = "resize_bl"
    elif e.local_x > control_width - corner_threshold and e.local_y > control_height - corner_threshold: image_dialog_state.overlay_interaction_mode = "resize_br"
    else: image_dialog_state.overlay_interaction_mode = "pan"
    
    if image_dialog_state.overlay_interaction_mode.startswith("resize_") and image_dialog_state.c_key_scaling_active:
        # Lock aspect ratio to the original image's aspect ratio
        if image_dialog_state.image_orig_h > 0:
            image_dialog_state.aspect_ratio_locked = True
            image_dialog_state.locked_aspect_ratio = image_dialog_state.image_orig_w / image_dialog_state.image_orig_h
        else:
            image_dialog_state.aspect_ratio_locked = False
    else: 
        image_dialog_state.aspect_ratio_locked = False

def on_overlay_pan_update(e: ft.DragUpdateEvent):
    if not image_dialog_state.overlay_control_instance or not image_dialog_state.overlay_visual_instance or image_dialog_state.overlay_interaction_mode == "none": return
    control, visual = image_dialog_state.overlay_control_instance, image_dialog_state.overlay_visual_instance
    
    # Use effective image dimensions for boundary checks
    effective_image_w = image_dialog_state.effective_image_w
    effective_image_h = image_dialog_state.effective_image_h
    offset_x = image_dialog_state.offset_x
    offset_y = image_dialog_state.offset_y

    delta_x, delta_y = e.global_x - image_dialog_state.overlay_pan_start_x, e.global_y - image_dialog_state.overlay_pan_start_y
    new_left, new_top, new_width, new_height = image_dialog_state.overlay_initial_box_left, image_dialog_state.overlay_initial_box_top, image_dialog_state.overlay_initial_box_width, image_dialog_state.overlay_initial_box_height
    mode = image_dialog_state.overlay_interaction_mode

    if mode == "pan": 
        new_left += delta_x
        new_top += delta_y
    elif mode.startswith("resize_"):
        potential_w, potential_h = new_width, new_height
        if mode == "resize_br": 
            potential_w += delta_x
            potential_h += delta_y
        elif mode == "resize_bl": 
            potential_w -= delta_x
            potential_h += delta_y
            new_left = image_dialog_state.overlay_initial_box_left + (image_dialog_state.overlay_initial_box_width - potential_w)
        elif mode == "resize_tr": 
            potential_w += delta_x
            potential_h -= delta_y
            new_top = image_dialog_state.overlay_initial_box_top + (image_dialog_state.overlay_initial_box_height - potential_h)
        elif mode == "resize_tl": 
            potential_w -= delta_x
            potential_h -= delta_y
            new_left = image_dialog_state.overlay_initial_box_left + (image_dialog_state.overlay_initial_box_width - potential_w)
            new_top = image_dialog_state.overlay_initial_box_top + (image_dialog_state.overlay_initial_box_height - potential_h)

        if image_dialog_state.aspect_ratio_locked and image_dialog_state.locked_aspect_ratio > 0:
            # Calculate new dimensions maintaining aspect ratio
            if mode in ["resize_br", "resize_tr"]:
                new_width = potential_w
                new_height = int(potential_w / image_dialog_state.locked_aspect_ratio)
            else: # resize_bl, resize_tl
                new_height = potential_h
                new_width = int(potential_h * image_dialog_state.locked_aspect_ratio)
        else: 
            new_width, new_height = potential_w, potential_h

    # Clamp dimensions to minimum size
    new_width = max(MIN_OVERLAY_SIZE, new_width)
    new_height = max(MIN_OVERLAY_SIZE, new_height)

    # Clamp position and size to within the effective image area
    # Adjust for the offset of the effective image within the player content area
    clamped_left = max(offset_x, min(new_left, offset_x + effective_image_w - new_width))
    clamped_top = max(offset_y, min(new_top, offset_y + effective_image_h - new_height))
    
    clamped_width = min(new_width, (offset_x + effective_image_w) - clamped_left)
    clamped_height = min(new_height, (offset_y + effective_image_h) - clamped_top)

    # Re-clamp dimensions to minimum size after clamping to boundaries
    clamped_width = max(MIN_OVERLAY_SIZE, clamped_width)
    clamped_height = max(MIN_OVERLAY_SIZE, clamped_height)

    # Final check to ensure overlay doesn't go out of bounds if resizing caused it to shrink too much
    if clamped_left + clamped_width > offset_x + effective_image_w:
        clamped_left = offset_x + effective_image_w - clamped_width
    if clamped_top + clamped_height > offset_y + effective_image_h:
        clamped_top = offset_y + effective_image_h - clamped_height

    # Ensure final position is not less than offset_x/y
    clamped_left = max(offset_x, clamped_left)
    clamped_top = max(offset_y, clamped_top)

    control.left, control.top, control.width, control.height = clamped_left, clamped_top, clamped_width, clamped_height
    visual.width, visual.height = clamped_width, clamped_height
    if control.page: control.update()
    if visual.page: visual.update()

def on_overlay_pan_end(e: ft.DragEndEvent):
    image_dialog_state.overlay_interaction_mode = "none"
    image_dialog_state.aspect_ratio_locked = False
    if image_dialog_state.c_key_scaling_active: image_dialog_state.c_key_scaling_active = False
