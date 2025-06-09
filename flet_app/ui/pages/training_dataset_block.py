import flet as ft
import os
import cv2
import numpy as np
import json
from .._styles import create_textfield, create_dropdown # Import helper functions
from ..dataset_manager.dataset_utils import get_dataset_folders, _get_dataset_base_dir  # Reuse the helper
from settings import settings


# =====================
# Data/Utility Functions
# =====================
def load_dataset_summary(dataset):
    """
    Loads summary statistics for a dataset: number of videos, captioned, processed, and total frames.
    """
    if not dataset or dataset == "Select your dataset": # Explicitly handle the problematic string
        return {
            "Files": 0,
            "Captioned": 0,
            "Processed": 0,
            "Total frames/images": 0
        }
    
    base_dir, dataset_type = _get_dataset_base_dir(dataset)
    clean_dataset_name = dataset.replace('(img) ', '').replace(' (img)', '')
    dataset_full_path = os.path.join(base_dir, clean_dataset_name)

    info_path = os.path.join(dataset_full_path, "info.json")
    captions_path = os.path.join(dataset_full_path, "captions.json")
    processed_path = os.path.join(dataset_full_path, "preprocessed_data", "processed.json")
    
    num_files = 0
    num_captioned = 0
    num_processed = 0
    total_frames_or_images = 0

    if dataset_type == "image":
        file_extensions = settings.IMAGE_EXTENSIONS
    else:
        file_extensions = settings.VIDEO_EXTENSIONS

    media_files = [f for ext in file_extensions for f in os.listdir(dataset_full_path) if f.lower().endswith(ext)]
    num_files = len(media_files)

    if os.path.exists(info_path):
        try:
            with open(info_path, 'r') as f:
                info = json.load(f)
            # Sum frames for videos, or count images for image datasets
            if dataset_type == "video":
                total_frames_or_images = sum(v.get("frames", 0) for v in info.values() if isinstance(v, dict))
            else: # image dataset
                total_frames_or_images = num_files # Each image is a "frame" in this context
        except Exception:
            pass
    
    if os.path.exists(captions_path):
        try:
            with open(captions_path, 'r', encoding='utf-8') as f:
                captions = json.load(f)
                num_captioned = sum(1 for entry in captions if entry.get("caption", "").strip())
        except Exception:
            pass
    
    if os.path.exists(processed_path):
        try:
            with open(processed_path, 'r', encoding='utf-8') as f:
                processed_map = json.load(f)
            num_processed = len(processed_map)
        except Exception:
            num_processed = 0
    
    return {
        "Files": num_files,
        "Captioned": num_captioned,
        "Processed": num_processed,
        "Total frames/images": total_frames_or_images
    }

def generate_collage(thumbnails_dir, summary_path, target_w=settings.COLLAGE_WIDTH, target_h=settings.COLLAGE_HEIGHT):
    """
    Generates a collage image from all jpg thumbnails in a directory (except summary.jpg).
    """
    images = [os.path.join(thumbnails_dir, f) for f in os.listdir(thumbnails_dir)
              if f.endswith('.jpg') and f != 'summary.jpg']
    if not images:
        return False
    thumbs = [cv2.imread(img) for img in images if cv2.imread(img) is not None]
    if not thumbs:
        return False
    n = len(thumbs)
    best_cols = max(1, int(np.round(np.sqrt(n * (target_w/target_h)))))
    rows = (n + best_cols - 1) // best_cols
    scaled_thumbs = []
    for t in thumbs:
        h, w = t.shape[:2]
        scale = min(settings.THUMB_CELL_W / w, settings.THUMB_CELL_H / h, 1.0)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(t, (new_w, new_h), interpolation=cv2.INTER_AREA)
        pad_top = (settings.THUMB_CELL_H - new_h) // 2
        pad_bottom = settings.THUMB_CELL_H - new_h - pad_top
        pad_left = (settings.THUMB_CELL_W - new_w) // 2
        pad_right = settings.THUMB_CELL_W - new_w - pad_left
        padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
        scaled_thumbs.append(padded)
    grid_rows = []
    for r in range(rows):
        row_imgs = scaled_thumbs[r*best_cols:(r+1)*best_cols]
        while len(row_imgs) < best_cols:
            row_imgs.append(np.zeros((settings.THUMB_CELL_H, settings.THUMB_CELL_W, 3), dtype=np.uint8))
        grid_rows.append(np.hstack(row_imgs))
    collage = np.vstack(grid_rows)
    ch, cw = collage.shape[:2]
    if ch < target_h:
        pad_h = target_h - ch
        collage = np.pad(collage, ((pad_h//2, pad_h - pad_h//2), (0,0), (0,0)), mode='constant', constant_values=0)
    if cw < target_w:
        pad_w = target_w - cw
        collage = np.pad(collage, ((0,0), (pad_w//2, pad_w - pad_w//2), (0,0)), mode='constant', constant_values=0)
    ch, cw = collage.shape[:2]
    y0 = (ch - target_h) // 2
    x0 = (cw - target_w) // 2
    collage = collage[y0:y0+target_h, x0:x0+target_w]
    cv2.imwrite(summary_path, collage)
    return True

# =====================
# GUI-Building Functions
# =====================
def build_training_dataset_page_content():
    """
    Builds the main container for the training dataset selection page, including dropdown, summary, and controls.
    """
    selected_dataset = {"value": None}
    content_column_ref = ft.Ref[ft.Column]()
    dataset_dropdown_ref = ft.Ref[ft.Dropdown]()
    num_workers_field_ref = ft.Ref[ft.TextField]()

    def reload_current_dataset():
        col = content_column_ref.current
        if col is None:
            return
        folders = get_dataset_folders()
        dataset_dropdown = dataset_dropdown_ref.current
        prev_selected = selected_dataset["value"]
        if dataset_dropdown:
            # Rebuild options with correct key/text mapping
            dropdown_options_map = {name: display_name for name, display_name in folders.items()}
            dataset_dropdown.options = [ft.dropdown.Option(key=name, text=display_name) for name, display_name in dropdown_options_map.items()]
            dataset_dropdown.disabled = len(folders) == 0
            
            # prev_selected now holds the clean name (or None)
            if prev_selected and prev_selected in folders.keys(): # Check against clean names (keys)
                dataset_dropdown.value = prev_selected
                selected_dataset["value"] = prev_selected # Ensure selected_dataset also holds clean name
            else:
                # If no previous valid selection, try to select the first available dataset
                if folders:
                    first_dataset_key = list(folders.keys())[0]
                    dataset_dropdown.value = first_dataset_key
                    selected_dataset["value"] = first_dataset_key
                else:
                    dataset_dropdown.value = None # No datasets available, clear selection
                    selected_dataset["value"] = None
            dataset_dropdown.update()
        update_summary_row()

    def build_controls():
        """
        Builds the top row controls: dataset dropdown, refresh button, and num workers field.
        """
        folders = get_dataset_folders()
        # Prepare options for dropdown: key is clean name, text is display name
        dropdown_options_map = {name: display_name for name, display_name in folders.items()}
        dataset_dropdown = create_dropdown(
            "Select dataset",
            selected_dataset["value"], # This should store the clean name
            dropdown_options_map,
            hint_text="Select your dataset",
            expand=True
        )
        dataset_dropdown_ref.current = dataset_dropdown
        dataset_dropdown.disabled = len(folders) == 0
        def on_dataset_change(e):
            selected_dataset["value"] = e.control.value if e.control.value else None
            update_summary_row()
        dataset_dropdown.on_change = on_dataset_change
        update_button = ft.IconButton(
            icon=ft.Icons.REFRESH,
            tooltip="Update dataset list",
            on_click=lambda e: reload_current_dataset(),
            style=ft.ButtonStyle(padding=ft.padding.symmetric(horizontal=8)),
            icon_size=20
        )
        if not hasattr(build_controls, "num_workers"):
            build_controls.num_workers = 2
        def _handle_num_workers_update(e):
            current_input_value = e.control.value
            previous_valid_workers_value = getattr(build_controls, 'num_workers', 2)
            try:
                val = int(current_input_value)
                if val >= 0:
                    build_controls.num_workers = val
                else:
                    e.control.value = str(previous_valid_workers_value)
            except ValueError:
                e.control.value = str(previous_valid_workers_value)
        num_workers_field = create_textfield(
            "Num Dataloader Workers",
            str(getattr(build_controls, 'num_workers', 2)),
            col=None,
            expand=False
        )
        num_workers_field_ref.current = num_workers_field
        num_workers_field.width = 200
        num_workers_field.on_change = _handle_num_workers_update
        return ft.Row([
            dataset_dropdown,
            update_button,
            num_workers_field
        ], expand=True)

    def build_summary_display_row():
        """
        Builds the row that displays the dataset summary and collage image.
        """
        return ft.Row([
            ft.Container(key="summary_img_container", width=settings.COLLAGE_WIDTH, height=settings.COLLAGE_HEIGHT),
            ft.Column(key="summary_text_column", spacing=8, expand=True)
        ], spacing=10)

    def update_summary_row():
        """
        Updates the summary display row with the current dataset's summary and collage image.
        """
        page_col = content_column_ref.current
        if not page_col: return

        summary_img_container = None
        summary_text_column = None
        if len(page_col.controls) > 1 and isinstance(page_col.controls[1], ft.Row):
            summary_row_candidate = page_col.controls[1]
            if len(summary_row_candidate.controls) == 2 and \
               isinstance(summary_row_candidate.controls[0], ft.Container) and \
               getattr(summary_row_candidate.controls[0], 'key', None) == 'summary_img_container' and \
               isinstance(summary_row_candidate.controls[1], ft.Column) and \
               getattr(summary_row_candidate.controls[1], 'key', None) == 'summary_text_column':
                summary_img_container = summary_row_candidate.controls[0]
                summary_text_column = summary_row_candidate.controls[1]

        if not summary_img_container or not summary_text_column:
            return

        summary_img_container.content = None # Clear image first
        summary_text_column.controls.clear() # Clear text first

        current_selected_dataset = selected_dataset["value"]

        if not current_selected_dataset or str(current_selected_dataset).lower() == "none": # Handle None and "None" string
            summary_text_column.controls.append(
                ft.Text("Select a dataset", key="placeholder_select_dataset")
            )
        else:
            # Determine dataset type and base directory for thumbnails
            base_dir, dataset_type = _get_dataset_base_dir(current_selected_dataset)
            clean_dataset_name = current_selected_dataset.replace('(img) ', '').replace(' (img)', '')
            
            if dataset_type == "image":
                thumbnails_base_dir = settings.THUMBNAILS_IMG_BASE_DIR
            else:
                thumbnails_base_dir = settings.THUMBNAILS_BASE_DIR

            thumbnails_dir = os.path.join(thumbnails_base_dir, clean_dataset_name)
            summary_path = os.path.join(thumbnails_dir, "summary.jpg")
            
            if not os.path.exists(thumbnails_dir):
                os.makedirs(thumbnails_dir, exist_ok=True)
            
            if not os.path.exists(thumbnails_dir):
                summary_text_column.controls.append(ft.Text(f"Thumbnails directory for {current_selected_dataset} not found or couldn't be created.", size=12))
            else:
                # Try to generate summary if it doesn't exist
                if not os.path.exists(summary_path):
                    generate_collage(thumbnails_dir, summary_path)
                
                if os.path.exists(summary_path):
                    summary_img_container.content = ft.Image(
                        src=summary_path,
                        width=settings.COLLAGE_WIDTH - 2,
                        height=settings.COLLAGE_HEIGHT,
                        fit=ft.ImageFit.CONTAIN
                    )
                
                summary_data = load_dataset_summary(current_selected_dataset)
                
                # Check for "Files" instead of "Videos"
                if not summary_data.get("Files") and not os.path.exists(summary_path):
                    summary_text_column.controls.append(ft.Text(f"No files or summary image found for {current_selected_dataset}.", size=12))
                
                summary_text_column.controls.append(ft.Text("Dataset summary", weight=ft.FontWeight.BOLD, size=14))
                for k, v in summary_data.items():
                    summary_text_column.controls.append(ft.Text(f"{k} - {v}", size=12))

        if summary_img_container.page: summary_img_container.update()
        if summary_text_column.page: summary_text_column.update()

    content_column = ft.Column(
        ref=content_column_ref,
        controls=[
            build_controls(),
            build_summary_display_row()
        ],
        scroll=ft.ScrollMode.ADAPTIVE,
        expand=True
    )
    container = ft.Container(content=content_column, expand=True, )
    def _on_mount_actions(e):
        update_summary_row()
    container.on_mount = _on_mount_actions

    # Expose selected dataset and num_workers for Save/Open
    container.get_selected_dataset = lambda: (print("Selected dataset:", selected_dataset["value"]), selected_dataset["value"])[1]
    container.get_num_workers = lambda: getattr(build_controls, 'num_workers', 2)

    # Add set_selected_dataset method
    def set_selected_dataset(dataset_name, page_ctx=None):
        dropdown = dataset_dropdown_ref.current
        folders = get_dataset_folders() # Get the clean_name: display_name map

        # dataset_name passed to this function should be the clean name
        if dataset_name is None or (dataset_name not in folders.keys()):
            selected_dataset["value"] = None
            if dropdown:
                dropdown.value = ""  # Key for "None" option
                if dropdown.page:
                    dropdown.update()
                    if page_ctx:
                        page_ctx.update()
        elif dataset_name in folders.keys(): # dataset_name is not None and is a valid clean name
            selected_dataset["value"] = dataset_name # Store the clean name
            if dropdown:
                dropdown.value = str(dataset_name) # Set dropdown value to the clean name (key)
                if dropdown.page:
                    dropdown.update()
                    if page_ctx:
                        page_ctx.update()
        
        update_summary_row()
    container.set_selected_dataset = set_selected_dataset

    # Add set_num_workers method
    def set_num_workers(num_workers, page_ctx=None):
        try:
            val = int(num_workers)
            if val >= 0:
                build_controls.num_workers = val
                # Update textfield value in UI
                num_workers_field = num_workers_field_ref.current
                if num_workers_field:
                    num_workers_field.value = str(val)
                    num_workers_field.update()
        except Exception as e:
            print(f"Error in set_num_workers: {e}")
    container.set_num_workers = set_num_workers

    return container

# =====================
# Entry Point
# =====================
get_training_dataset_page_content = build_training_dataset_page_content
