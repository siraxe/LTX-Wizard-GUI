# =====================
# Hotkey Functions
# =====================

from ui.utils.utils_top_menu import TopBarUtils

# --- Video Player Hotkey Logic ---
AUTO_VIDEO_PLAYBACK = True  # Set to True to auto-play video on open/switch

# --- Video Player Hotkey Keybindings ---
VIDEO_PLAY_PAUSE_KEY = " "  # Spacebar
VIDEO_NEXT_KEY = "]"
VIDEO_PREV_KEY = "["

def handle_global_keyboard_event(page, e):
    """
    Handles global keyboard shortcuts and dialog hotkeys.
    - Esc: Close dialog
    - Ctrl+S, Ctrl+Shift+S, Ctrl+O, Ctrl+F: Menu hotkeys
    """
    # If the video dialog is open and has a handler, call it
    if getattr(page, 'video_dialog_open', False) and getattr(page, 'video_dialog_hotkey_handler', None):
        page.video_dialog_hotkey_handler(e)
        return
    # Esc key closes base dialog if open
    if hasattr(e, 'key') and e.key == 'Escape':
        if hasattr(page, 'base_dialog') and getattr(page.base_dialog, 'visible', False):
            page.base_dialog.hide_dialog()
            return
    # Global hotkeys for menu actions
    if hasattr(e, 'ctrl') and e.ctrl:
        # Ctrl+Shift+S (Save As)
        if hasattr(e, 'shift') and e.shift and hasattr(e, 'key') and e.key.lower() == 's':
            TopBarUtils.handle_save_as(page)
        # Ctrl+S (Save)
        elif hasattr(e, 'key') and e.key.lower() == 's':
            TopBarUtils.handle_save(page)
        # Ctrl+O (Open)
        elif hasattr(e, 'key') and e.key.lower() == 'o':
            TopBarUtils.handle_open(page)
        # Ctrl+F (Open Base Dialog)
        elif hasattr(e, 'key') and e.key.lower() == 'f':
            if hasattr(page, 'base_dialog'):
                page.base_dialog.show_dialog()
