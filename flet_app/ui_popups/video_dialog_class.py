import flet as ft
import threading
from flet_video.video import Video # For type hinting Video player instances
from typing import Optional, List, Callable # Import necessary typing tools

class VideoDialogState:
    """Manages the state for the Video Player Dialog."""
    def __init__(self):
        self.active_video_player_instance: Optional[Video] = None
        self.active_caption_field_instance: Optional[ft.TextField] = None
        self.active_caption_neg_field_instance: Optional[ft.TextField] = None
        self.active_message_container_instance: Optional[ft.Container] = None
        self.active_on_caption_updated_callback: Optional[Callable] = None
        self.current_video_list_for_dialog: List[str] = [] # Assuming list of strings
        self.current_video_path_for_dialog: str = ""
        self.last_video_loading_path: str = ""
        self.active_width_field_instance: Optional[ft.TextField] = None
        self.active_height_field_instance: Optional[ft.TextField] = None
        self.video_is_playing: List[bool] = [False]  # List for mutability
        self.active_page_ref: Optional[ft.Page] = None

        # Overlay box references
        self.overlay_visual_instance: Optional[ft.Container] = None
        self.overlay_control_instance: Optional[ft.GestureDetector] = None

        # Video playback range and reframing state
        self.reframed_playback: bool = False
        self.playback_start_frame: int = -1
        self.playback_end_frame: int = -1

        # Frame counter state
        self.frame_update_timer: Optional[threading.Timer] = None
        self.dialog_is_open: bool = False

        # Video Editor UI element references
        self.frame_range_slider_instance: Optional[ft.RangeSlider] = None
        self.start_value_text_instance: Optional[ft.Text] = None
        self.end_value_text_instance: Optional[ft.Text] = None
        self.total_frames_text_instance: Optional[ft.Text] = None

        # Visual feedback state
        self.video_feedback_overlay: Optional[ft.Container] = None
        self.video_feedback_timer: Optional[threading.Timer] = None

        # Caption field focus state
        self.caption_field_is_focused: bool = False

        # Overlay Resize/Pan Interaction State
        self.overlay_interaction_mode: str = "none" # e.g., "pan", "resize_tl", "resize_tr", "resize_bl", "resize_br"
        self.overlay_pan_start_x: float = 0
        self.overlay_pan_start_y: float = 0
        self.overlay_initial_box_left: float = 0
        self.overlay_initial_box_top: float = 0
        self.overlay_initial_box_width: float = 0
        self.overlay_initial_box_height: float = 0

        # Debouncing overlay pan updates during video restart
        self.is_video_restarting: bool = False
        self.video_restart_timer: Optional[threading.Timer] = None
        self.last_video_error: str = ""
        self.is_processing_completion: bool = False # Re-entrancy guard
        self.last_completion_processed_time: float = 0.0 # Cool-down timestamp

        self.overlay_is_visible: bool = False
        self.aspect_ratio_locked: bool = False
        self.locked_aspect_ratio: float = 1.0 # Stores width / height
        self.c_key_scaling_active: bool = False # Flag for 'C' key based scaling

# Create a single instance of the state class
dialog_state = VideoDialogState()
