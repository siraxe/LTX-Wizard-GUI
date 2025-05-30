import flet as ft 
import os
import shutil
from typing import Optional, List, Callable 
from flet_video.video import Video # <--- ADDED THIS IMPORT

from . import video_player_dialog 
from .video_dialog_class import dialog_state as player_dialog_state 
from . import video_player_utils as vpu


def handle_size_add(width_field: ft.TextField, height_field: ft.TextField, current_video_path: str, page: Optional[ft.Page] = None): 
    new_w_str, new_h_str = vpu.calculate_adjusted_size(
        width_field.value, height_field.value, current_video_path, 'add'
    )
    width_field.value = new_w_str
    height_field.value = new_h_str
    if page:
        width_field.update()
        height_field.update()

def handle_size_sub(width_field: ft.TextField, height_field: ft.TextField, current_video_path: str, page: Optional[ft.Page] = None): 
    new_w_str, new_h_str = vpu.calculate_adjusted_size(
        width_field.value, height_field.value, current_video_path, 'sub'
    )
    width_field.value = new_w_str
    height_field.value = new_h_str
    if page:
        width_field.update()
        height_field.update()

def _perform_crop_from_editor_overlay(page: ft.Page, current_video_path: str, video_list: Optional[List[str]], on_caption_updated_callback: Optional[Callable]): 
    if not current_video_path or not os.path.exists(current_video_path):
        if page: page.snack_bar = ft.SnackBar(ft.Text("Error: Invalid video path for cropping."), open=True); page.update()
        return

    overlay_control = player_dialog_state.overlay_control_instance
    if not overlay_control:
        if page: page.snack_bar = ft.SnackBar(ft.Text("Error: Crop overlay not initialized."), open=True); page.update()
        return

    if not player_dialog_state.overlay_is_visible: 
        if page: page.snack_bar = ft.SnackBar(ft.Text("Crop overlay is not visible. Please open the Crop Editor."), open=True); page.update()
        return

    metadata = vpu.get_video_metadata(current_video_path)
    if not metadata or not metadata.get('width') or not metadata.get('height'):
        if page: page.snack_bar = ft.SnackBar(ft.Text("Error: Could not get video dimensions for crop."), open=True); page.update()
        return
    video_orig_w, video_orig_h = metadata['width'], metadata['height']

    overlay_x_px = overlay_control.left if overlay_control.left is not None else 0
    overlay_y_px = overlay_control.top if overlay_control.top is not None else 0
    overlay_w_px = overlay_control.width if overlay_control.width is not None else 0
    overlay_h_px = overlay_control.height if overlay_control.height is not None else 0
    
    from ui._styles import VIDEO_PLAYER_DIALOG_WIDTH, VIDEO_PLAYER_DIALOG_HEIGHT 
    player_content_w = VIDEO_PLAYER_DIALOG_WIDTH - 40 
    player_content_h = VIDEO_PLAYER_DIALOG_HEIGHT - 40

    success, msg, temp_output_path = vpu.crop_video_from_overlay(
        current_video_path=current_video_path,
        overlay_x_norm=overlay_x_px, 
        overlay_y_norm=overlay_y_px,
        overlay_w_norm=overlay_w_px,
        overlay_h_norm=overlay_h_px,
        displayed_video_w=player_content_w, 
        displayed_video_h=player_content_h,
        video_orig_w=video_orig_w,
        video_orig_h=video_orig_h,
        player_content_w=player_content_w, 
        player_content_h=player_content_h 
    )

    if success and temp_output_path:
        try:
            shutil.move(temp_output_path, current_video_path) 
            vpu.update_video_info_json(current_video_path) 
            if page: page.snack_bar = ft.SnackBar(ft.Text(msg), open=True)
            
            if player_dialog_state.active_video_player_instance:
                current_autoplay = player_dialog_state.active_video_player_instance.autoplay
                video_player_dialog.update_video_player_source(
                    player_dialog_state.active_video_player_instance, 
                    current_video_path,
                    autoplay=current_autoplay 
                )
            if on_caption_updated_callback: 
                on_caption_updated_callback(current_video_path) 
        except Exception as e:
            msg = f"Error moving cropped file: {e}"
            if page: page.snack_bar = ft.SnackBar(ft.Text(msg), open=True)
        finally:
            if os.path.exists(temp_output_path): 
                try: os.remove(temp_output_path)
                except Exception as e_del: print(f"Error deleting temp file {temp_output_path}: {e_del}")
    else:
        if page: page.snack_bar = ft.SnackBar(ft.Text(msg), open=True)
    
    if page: page.update()

def handle_set_closest_div32(width_field: ft.TextField, height_field: ft.TextField, current_video_path: str, page: Optional[ft.Page] = None): 
    w_str, h_str = vpu.calculate_closest_div32_dimensions(current_video_path)
    if w_str and h_str:
        width_field.value = w_str
        height_field.value = h_str
        if page:
            width_field.update()
            height_field.update()

def _generic_video_operation_ui_update(
    page: ft.Page, 
    processed_video_path: str, 
    video_list: Optional[List[str]] = None, 
    on_caption_updated_callback: Optional[Callable] = None, 
    operation_message: str = "Video operation successful."
    ):
    if page:
        page.snack_bar = ft.SnackBar(ft.Text(operation_message), open=True)
        vpu.update_video_info_json(processed_video_path)
        video_player_dialog.open_video_captions_dialog(page, processed_video_path, video_list, on_caption_updated_callback)
        if on_caption_updated_callback:
            on_caption_updated_callback(processed_video_path) 
        page.update()

def on_flip_horizontal(page: ft.Page, current_video_path: str, video_list: Optional[List[str]], on_caption_updated_callback: Optional[Callable]): 
    success, msg, temp_output_path = vpu.flip_video_horizontal(current_video_path)
    if success and temp_output_path:
        try:
            shutil.move(temp_output_path, current_video_path)
            _generic_video_operation_ui_update(page, current_video_path, video_list, on_caption_updated_callback, msg)
        except Exception as e:
            if page: page.snack_bar = ft.SnackBar(ft.Text(f"Error moving flipped file: {e}"), open=True); page.update()
            if os.path.exists(temp_output_path): os.remove(temp_output_path)
    else:
        if page: page.snack_bar = ft.SnackBar(ft.Text(msg), open=True); page.update()

def on_reverse(page: ft.Page, current_video_path: str, video_list: Optional[List[str]], on_caption_updated_callback: Optional[Callable]): 
    success, msg, temp_output_path = vpu.reverse_video(current_video_path)
    if success and temp_output_path:
        try:
            shutil.move(temp_output_path, current_video_path)
            _generic_video_operation_ui_update(page, current_video_path, video_list, on_caption_updated_callback, msg)
        except Exception as e:
            if page: page.snack_bar = ft.SnackBar(ft.Text(f"Error moving reversed file: {e}"), open=True); page.update()
            if os.path.exists(temp_output_path): os.remove(temp_output_path)
    else:
        if page: page.snack_bar = ft.SnackBar(ft.Text(msg), open=True); page.update()

def on_time_remap(page: ft.Page, current_video_path: str, speed_multiplier: float, video_list: Optional[List[str]], on_caption_updated_callback: Optional[Callable]): 
    success, msg, temp_output_path = vpu.time_remap_video_by_speed(current_video_path, speed_multiplier)
    if success and temp_output_path:
        try:
            shutil.move(temp_output_path, current_video_path)
            _generic_video_operation_ui_update(page, current_video_path, video_list, on_caption_updated_callback, msg)
        except Exception as e:
            if page: page.snack_bar = ft.SnackBar(ft.Text(f"Error moving remapped file: {e}"), open=True); page.update()
            if os.path.exists(temp_output_path): os.remove(temp_output_path)
    else:
        if page: page.snack_bar = ft.SnackBar(ft.Text(msg), open=True); page.update()

def handle_crop_video_click(page: ft.Page, width_field: Optional[ft.TextField], height_field: Optional[ft.TextField], current_video_path: str, video_list: Optional[List[str]] = None, on_caption_updated_callback: Optional[Callable] = None, should_update_ui: bool = True): 
    if not width_field or not width_field.value or not height_field or not height_field.value : 
        if page and should_update_ui: page.snack_bar = ft.SnackBar(ft.Text("Width and Height must be specified."), open=True); page.update()
        return
    try:
        target_width = int(width_field.value)
        target_height = int(height_field.value)
    except ValueError:
        if page and should_update_ui: page.snack_bar = ft.SnackBar(ft.Text("Invalid Width or Height value."), open=True); page.update()
        return

    success, msg, temp_output_path = vpu.crop_video_to_dimensions(current_video_path, target_width, target_height)
    
    if success and temp_output_path:
        try:
            shutil.move(temp_output_path, current_video_path)
            if should_update_ui:
                 _generic_video_operation_ui_update(page, current_video_path, video_list, on_caption_updated_callback, msg)
            else: 
                vpu.update_video_info_json(current_video_path)
                if page: page.snack_bar = ft.SnackBar(ft.Text(f"{os.path.basename(current_video_path)}: {msg}" if msg else f"{os.path.basename(current_video_path)} cropped (batch)."), duration=2000, open=True)
        except Exception as e:
            final_msg = f"Error moving cropped file for {os.path.basename(current_video_path)}: {e}"
            if page and should_update_ui: page.snack_bar = ft.SnackBar(ft.Text(final_msg), open=True)
            print(final_msg)
            if os.path.exists(temp_output_path): os.remove(temp_output_path)
    else:
        final_msg = msg if msg else f"Failed to crop {os.path.basename(current_video_path)}."
        if page and should_update_ui: page.snack_bar = ft.SnackBar(ft.Text(final_msg), open=True)
        print(final_msg)

    if page and not should_update_ui: 
        page.update()

def handle_crop_all_videos(
    page: ft.Page, 
    current_video_path_in_dialog: Optional[str],
    width_field: Optional[ft.TextField], 
    height_field: Optional[ft.TextField], 
    video_list: Optional[List[str]], 
    on_caption_updated_callback: Optional[Callable]
): 
    if not video_list:
        if page: page.snack_bar = ft.SnackBar(ft.Text("No videos in the list to crop."), open=True); page.update()
        return
    
    if not width_field or not width_field.value or not height_field or not height_field.value:
        if page: page.snack_bar = ft.SnackBar(ft.Text("Width and Height must be specified for batch crop."), open=True); page.update()
        return

    processed_count = 0
    dialog_refreshed_for_current_video = False

    for video_path_item in video_list: 
        processed_count +=1
        # Determine if the UI (dialog) should be fully updated for this specific item
        should_refresh_dialog_for_this_item = (video_path_item == current_video_path_in_dialog)
        
        # Call handle_crop_video_click, which might reopen the dialog if should_refresh_dialog_for_this_item is True
        handle_crop_video_click(
            page, width_field, height_field, 
            video_path_item, video_list, 
            on_caption_updated_callback, 
            should_update_ui=should_refresh_dialog_for_this_item
        )
        
        if should_refresh_dialog_for_this_item:
            dialog_refreshed_for_current_video = True # Mark that the dialog was handled

    if page:
        # Display a summary snackbar
        page.snack_bar = ft.SnackBar(ft.Text(f"Batch crop attempt finished for {processed_count} videos. Review individual messages."), open=True)
        
        # Call general callback and update page only if dialog wasn't specifically refreshed for the current video
        if on_caption_updated_callback and not dialog_refreshed_for_current_video:
            on_caption_updated_callback()
        
        if not dialog_refreshed_for_current_video:
            page.update()

def cut_to_frames(page: ft.Page, current_video_path: str, start_frame: int, end_frame: int, video_list: Optional[List[str]], on_caption_updated_callback: Optional[Callable]): 
    success, msg, temp_output_path = vpu.cut_video_by_frames(current_video_path, start_frame, end_frame)
    if success and temp_output_path:
        try:
            shutil.move(temp_output_path, current_video_path)
            _generic_video_operation_ui_update(page, current_video_path, video_list, on_caption_updated_callback, msg)
        except Exception as e:
            if page: page.snack_bar = ft.SnackBar(ft.Text(f"Error moving cut file: {e}"), open=True); page.update()
            if os.path.exists(temp_output_path): os.remove(temp_output_path)
    else:
        if page: page.snack_bar = ft.SnackBar(ft.Text(msg), open=True); page.update()

def split_to_video(page: ft.Page, current_video_path: str, split_frame: int, video_list: Optional[List[str]], on_caption_updated_callback: Optional[Callable], video_player_instance_from_dialog_state: Optional[Video]): 
    if split_frame <= 0: 
        if page: page.snack_bar = ft.SnackBar(ft.Text("Split frame must be greater than 0."), open=True); page.update()
        return

    success, msg, temp_path1, temp_path2 = vpu.split_video_by_frame(current_video_path, split_frame)

    if success and temp_path1 and temp_path2:
        original_dir = os.path.dirname(current_video_path)
        base_name, ext = os.path.splitext(os.path.basename(current_video_path))
        
        final_path1 = current_video_path 
        
        counter = 1
        final_path2_base = os.path.join(original_dir, f"{base_name}_splitP2")
        final_path2 = f"{final_path2_base}{ext}"
        while os.path.exists(final_path2):
            final_path2 = f"{final_path2_base}_{counter}{ext}"
            counter += 1
            if counter > 100: 
                if page: page.snack_bar = ft.SnackBar(ft.Text("Could not find a unique name for split part 2."), open=True); page.update()
                if os.path.exists(temp_path1): os.remove(temp_path1)
                if os.path.exists(temp_path2): os.remove(temp_path2)
                return
        try:
            shutil.move(temp_path1, final_path1)
            shutil.move(temp_path2, final_path2)

            vpu.update_video_info_json(final_path1)
            vpu.update_video_info_json(final_path2)

            if page: page.snack_bar = ft.SnackBar(ft.Text(f"Video split. Original updated, new: {os.path.basename(final_path2)}"), open=True)
            
            video_player_dialog.open_video_captions_dialog(page, final_path1, video_list, on_caption_updated_callback) 

            if on_caption_updated_callback:
                on_caption_updated_callback()

            page.update()

        except Exception as e:
            if page: page.snack_bar = ft.SnackBar(ft.Text(f"Error moving split files: {e}"), open=True); page.update()
            if os.path.exists(temp_path1): os.remove(temp_path1) 
            if os.path.exists(temp_path2): os.remove(temp_path2) 
    else:
        if page: page.snack_bar = ft.SnackBar(ft.Text(msg), open=True); page.update()
        if temp_path1 and os.path.exists(temp_path1): os.remove(temp_path1)
        if temp_path2 and os.path.exists(temp_path2): os.remove(temp_path2)

def cut_all_videos_to_max(
    page: ft.Page, 
    current_video_path_in_dialog: Optional[str], 
    video_list: Optional[List[str]], 
    num_to_cut_to: int, 
    on_caption_updated_callback: Optional[Callable]
): 
    if not video_list:
        if page: page.snack_bar = ft.SnackBar(ft.Text("No videos in list to cut."), open=True); page.update()
        return
    if num_to_cut_to <= 0:
        if page: page.snack_bar = ft.SnackBar(ft.Text("Number of frames to cut to must be positive."), open=True); page.update()
        return

    processed_count = 0
    failed_count = 0
    successfully_processed_paths = [] # New: track successful paths

    for video_path_item in video_list: 
        metadata = vpu.get_video_metadata(video_path_item)
        if not metadata or not metadata.get('total_frames') or metadata['total_frames'] <= num_to_cut_to :
            print(f"Skipping {os.path.basename(video_path_item)}: already short enough or no metadata.")
            continue 

        success, msg, temp_output_path = vpu.cut_video_by_frames(video_path_item, 0, num_to_cut_to) 
        if success and temp_output_path:
            try:
                # Ensure the original file exists before attempting to move over it
                if not os.path.exists(video_path_item):
                    print(f"Error: Original file {video_path_item} not found for move.")
                    failed_count += 1
                    if os.path.exists(temp_output_path): os.remove(temp_output_path)
                    continue

                shutil.move(temp_output_path, video_path_item)
                vpu.update_video_info_json(video_path_item) # Update JSON after successful move
                processed_count += 1
                successfully_processed_paths.append(video_path_item) # New: add to list
                if page: page.snack_bar = ft.SnackBar(ft.Text(f"Cut {os.path.basename(video_path_item)} to {num_to_cut_to} frames."), duration=2000, open=True); page.update()
            except Exception as e:
                failed_count += 1
                print(f"Error moving cut file for {os.path.basename(video_path_item)}: {e}")
                if os.path.exists(temp_output_path): os.remove(temp_output_path) # Clean up temp file on error
        else:
            failed_count +=1
            print(f"Failed to cut {os.path.basename(video_path_item)}: {msg}")
            if page: page.snack_bar = ft.SnackBar(ft.Text(f"Failed to cut {os.path.basename(video_path_item)}: {msg}"), open=True); page.update()
    
    if page:
        page.snack_bar = ft.SnackBar(ft.Text(f"Batch cut complete: {processed_count} succeeded, {failed_count} failed."), open=True)
        
        refreshed_current_video = False
        if current_video_path_in_dialog and current_video_path_in_dialog in successfully_processed_paths:
            # Assumes: from flet_app.ui_popups import video_player_dialog
            video_player_dialog.open_video_captions_dialog(
                page, 
                current_video_path_in_dialog, 
                video_list, 
                on_caption_updated_callback
            )
            refreshed_current_video = True

        if on_caption_updated_callback and not refreshed_current_video:
            on_caption_updated_callback()
        
        if not refreshed_current_video: 
            page.update()
