import flet as ft
from ui.flet_hotkeys import global_hotkey_handler
from ui.tab_training_view import get_training_tab_content
from ui.dataset_manager.dataset_layout_tab import dataset_tab_layout
from ui.tab_tools_view import get_tools_tab_content
from flet_app_top_menu import create_app_menu_bar
from ui.utils.utils_top_menu import TopBarUtils
from ui_popups.popup_dialog_base import PopupDialogBase

# =====================
# Helper/Data Functions
# =====================

def refresh_menu_bar(page, menu_bar_column):
    """Refreshes the application menu bar."""
    new_menu_bar = create_app_menu_bar(page)
    menu_bar_column.controls.clear()
    menu_bar_column.controls.append(new_menu_bar)
    menu_bar_column.update()

# =====================
# GUI-Building Functions
# =====================

def build_menu_bar_column(page):
    """Creates the menu bar column control."""
    app_menu_bar = create_app_menu_bar(page)
    menu_bar_column = ft.Column(
        controls=[app_menu_bar],
        horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
        spacing=0
    )
    return menu_bar_column

def build_main_tabs(page):
    """Creates the main tab control with Training, Datasets, and Tools tabs."""
    training_tab_container = get_training_tab_content(page)
    main_tabs = ft.Tabs(
        selected_index=0,
        animation_duration=100,
        tabs=[
            ft.Tab(
                text="Training",
                content=training_tab_container
            ),
            ft.Tab(
                text="Datasets",
                content=dataset_tab_layout(page)
            ),
            ft.Tab(
                text="Tools",
                content=get_tools_tab_content(page)
            ),
        ],
        expand=True,
    )
    return main_tabs, training_tab_container

def build_tabs_column(main_tabs):
    """Wraps the main tabs in a column that expands vertically and stretches horizontally."""
    return ft.Column(
        controls=[main_tabs],
        expand=True,
        horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
        spacing=0
    )

def build_base_dialog(page):
    """Creates the base popup dialog instance."""
    popup_content = ft.Text("This is a base dialog. Populate it with specific content and title.")
    return PopupDialogBase(page, content=popup_content, title="Info")

# =====================
# Main Application Entry
# =====================

def main(page: ft.Page):
    """Main entry point for the LTX Trainer Flet application."""
    # Window setup
    page.title = "LTX GUI Wizard"
    page.padding = 0
    page.window.center()
    page.window.height = 950
    page.window.width = 900

    # Build GUI controls
    menu_bar_column = build_menu_bar_column(page)
    main_tabs, training_tab_container = build_main_tabs(page)
    tabs_column = build_tabs_column(main_tabs)

    # Expose for Save As handler
    page.training_tab_container = training_tab_container

    # Attach menu bar refresh function
    page.refresh_menu_bar = lambda: refresh_menu_bar(page, menu_bar_column)

    # Add controls to page
    page.add(menu_bar_column)
    page.add(tabs_column)

    # Dialogs and overlays
    page.base_dialog = build_base_dialog(page)
    page.overlay.append(page.base_dialog)
    page.video_dialog_hotkey_handler = None
    page.video_dialog_open = False

    # Keyboard event handler
    page.on_keyboard_event = lambda e: global_hotkey_handler(page, e)
    page.update()

ft.app(target=main)
