import flet as ft
import threading
from typing import Optional, List, Callable

class ImageDialogState:
    """Manages the state for the Image Player Dialog."""
    def __init__(self):
        self.active_image_player_instance: Optional[ft.Image] = None
        self.active_image_player_stack: Optional[ft.Stack] = None
        self.active_caption_field_instance: Optional[ft.TextField] = None
        self.active_caption_neg_field_instance: Optional[ft.TextField] = None
        self.active_message_container_instance: Optional[ft.Container] = None
        self.active_on_caption_updated_callback: Optional[Callable] = None
        self.current_image_list_for_dialog: List[str] = []
        self.current_image_path_for_dialog: str = ""
        self.active_width_field_instance: Optional[ft.TextField] = None
        self.active_height_field_instance: Optional[ft.TextField] = None
        self.active_page_ref: Optional[ft.Page] = None

        # Overlay box references
        self.overlay_visual_instance: Optional[ft.Container] = None
        self.overlay_control_instance: Optional[ft.GestureDetector] = None

        # Dialog state
        self.dialog_is_open: bool = False

        # Caption field focus state
        self.caption_field_is_focused: bool = False

        # Overlay Resize/Pan Interaction State
        self.overlay_interaction_mode: str = "none"
        self.overlay_pan_start_x: float = 0
        self.overlay_pan_start_y: float = 0
        self.overlay_initial_box_left: float = 0
        self.overlay_initial_box_top: float = 0
        self.overlay_initial_box_width: float = 0
        self.overlay_initial_box_height: float = 0

        self.overlay_is_visible: bool = False
        self.aspect_ratio_locked: bool = False
        self.locked_aspect_ratio: float = 1.0
        self.c_key_scaling_active: bool = False

# Create a single instance of the state class
image_dialog_state = ImageDialogState()
