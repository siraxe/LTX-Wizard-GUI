# =====================
# Hotkey Functions
# =====================

from ui.utils.utils_top_menu import TopBarUtils

# --- Global Hotkey Logic ---
AUTO_PLAYBACK = True  # Set to True to auto-play media on open/switch

# --- Global Hotkey Keybindings ---
PLAY_PAUSE_KEY = " "  # Spacebar
NEXT_KEY = "]"
PREV_KEY = "["

def global_hotkey_handler(page, e):
    """
    Handles global keyboard shortcuts and dialog hotkeys.
    - Esc: Close dialog
    - Ctrl+S, Ctrl+Shift+S, Ctrl+O, Ctrl+F: Menu hotkeys
    """
    # If a media dialog is open and has a handler, call it
    if getattr(page, 'image_dialog_open', False) and getattr(page, 'image_dialog_hotkey_handler', None):
        page.image_dialog_hotkey_handler(e)
        return
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
