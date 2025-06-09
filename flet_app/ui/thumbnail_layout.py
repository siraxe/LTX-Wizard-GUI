import flet as ft
import os
from settings import settings
from ui_popups.image_player_dialog import open_image_captions_dialog
from ui_popups.video_player_dialog import open_video_captions_dialog

def set_thumbnail_selection_state(thumbnail_container: ft.Container, is_selected: bool):
    """
    Updates the visual state (background color and checkbox opacity) of a thumbnail container.
    This function is designed to be called programmatically from outside the thumbnail_layout module.
    """
    checkbox = None
    if isinstance(thumbnail_container.content, ft.Stack):
        for control in thumbnail_container.content.controls:
            if isinstance(control, ft.Checkbox):
                checkbox = control
                break
    
    if checkbox:
        checkbox.value = is_selected # Set the value
        checkbox.opacity = 1 if is_selected else 0 # Update opacity based on selection
        thumbnail_container.bgcolor = ft.Colors.with_opacity(0.3, ft.Colors.BLUE_100) if is_selected else ft.Colors.TRANSPARENT
        
        if thumbnail_container.page:
            thumbnail_container.update()
            checkbox.update()

def create_thumbnail_container(
    page_ctx: ft.Page,
    video_path: str,
    thumb_path: str,
    video_info: dict,
    has_caption: bool,
    processed_map: dict,
    video_files_list: list,
    update_thumbnails_callback,
    grid_control: ft.GridView,
    on_checkbox_change_callback,
    thumbnail_index: int,
    is_selected_initially: bool
):
    video_name = os.path.basename(video_path)
    info = video_info.get(video_name, {})
    width, height, frames = info.get("width", "?"), info.get("height", "?"), info.get("frames", "?")
    cap_val, cap_color = ("yes", ft.Colors.GREEN) if has_caption else ("no", ft.Colors.RED)
    proc_val, proc_color = ("yes", ft.Colors.GREEN) if processed_map and video_name in processed_map else ("no", ft.Colors.RED)

    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']

    def _handle_thumbnail_click(e_click):
        file_ext = os.path.splitext(video_path)[1].lower()
        if file_ext in image_extensions:
            if page_ctx:
                open_image_captions_dialog(
                    page_ctx,
                    video_path,
                    video_files_list,
                    on_caption_updated_callback=lambda _: update_thumbnails_callback(page_ctx=page_ctx, grid_control=grid_control, force_refresh=True)
                )
        else: # It's a video
            if page_ctx:
                open_video_captions_dialog(
                    page_ctx,
                    video_path,
                    video_files_list,
                    on_caption_updated_callback=lambda: update_thumbnails_callback(page_ctx=page_ctx, grid_control=grid_control, force_refresh=True)
                )

    is_hovered = False

    checkbox = ft.Checkbox(
        value=is_selected_initially,
        check_color=ft.Colors.WHITE,
        fill_color=ft.Colors.BLUE_GREY_500,
        overlay_color=ft.Colors.TRANSPARENT,
        active_color=ft.Colors.BLUE_GREY_500,
        right=0,
        bottom=0,
        opacity=0,
        animate_opacity=ft.Animation(200, ft.AnimationCurve.EASE_IN_OUT),
        data=thumbnail_index
    )

    # Define the handler functions BEFORE creating thumbnail_container
    def _on_hover_container(e: ft.HoverEvent):
        nonlocal is_hovered
        is_hovered = (e.data == "true")
        # When hovered, update opacity based on hover state and current checkbox value
        checkbox.opacity = 1 if is_hovered or checkbox.value else 0
        if checkbox.page:
            checkbox.page.update()

    def _on_checkbox_change(e: ft.ControlEvent):
        # This is triggered by user click. Update visuals and then call external callback.
        set_thumbnail_selection_state(thumbnail_container, checkbox.value)
        on_checkbox_change_callback(video_path, checkbox.value, thumbnail_index)

    thumbnail_container = ft.Container(
        content=ft.Stack(
            [
                ft.Column([
                    ft.Image(
                        src=thumb_path,
                        width=settings.THUMB_TARGET_W,
                        height=settings.THUMB_TARGET_H,
                        fit=ft.ImageFit.COVER,
                        border_radius=ft.border_radius.all(5)
                    ),
                    ft.Text(spans=[
                        ft.TextSpan("[cap - ", style=ft.TextStyle(color=ft.Colors.GREY_500, size=10)),
                        ft.TextSpan(cap_val, style=ft.TextStyle(color=cap_color, size=10)),
                        ft.TextSpan(", proc - ", style=ft.TextStyle(color=ft.Colors.GREY_500, size=10)),
                        ft.TextSpan(proc_val, style=ft.TextStyle(color=proc_color, size=10)),
                        ft.TextSpan("]", style=ft.TextStyle(color=ft.Colors.GREY_500, size=10)),
                    ], size=10),
                    ft.Text(f"[{width}x{height} - {frames} frames]", size=10, color=ft.Colors.GREY_500),
                ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, tight=True),
                checkbox
            ]
        ),
        data=video_path,
        on_click=_handle_thumbnail_click,
        on_hover=_on_hover_container, # Now _on_hover_container is defined
        tooltip=video_name,
        width=settings.THUMB_TARGET_W + 10,
        height=settings.THUMB_TARGET_H + 45,
        padding=5,
        border=ft.border.all(1, ft.Colors.OUTLINE),
        border_radius=ft.border_radius.all(5),
        bgcolor=ft.Colors.with_opacity(0.1, ft.Colors.GREEN_ACCENT_700) if is_selected_initially else ft.Colors.TRANSPARENT
    )

    # Initial visual update based on is_selected_initially
    set_thumbnail_selection_state(thumbnail_container, is_selected_initially)

    # Assign on_change to checkbox
    checkbox.on_change = _on_checkbox_change

    return thumbnail_container
