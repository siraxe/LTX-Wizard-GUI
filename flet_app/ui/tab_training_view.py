import flet as ft
from .pages.training_config import get_training_config_page_content
from .pages.training_sampling import build_training_sampling_page_content
from ui_popups import dataset_not_selected
import os
import yaml
import subprocess
import shutil
import asyncio
from PIL import Image

# =====================
# Data/Utility Functions
# =====================

def _process_and_save_image(yaml_dict, source_image_path, target_filename):
    """Helper function to process and save an image to the dataset's sample_images directory."""
    try:
        # Extract the output directory from the YAML dictionary
        output_dir = yaml_dict.get('output_dir')
        if not output_dir:
            print("Warning: output_dir not found in YAML config, cannot copy image.")
            return
            
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Get video dimensions for scaling
        video_dims = yaml_dict.get('validation', {}).get('video_dims')
        if not video_dims or len(video_dims) < 2:
            print("Warning: Invalid video dimensions in config, using default 512x512")
            target_width, target_height = 512, 512
        else:
            target_width, target_height = video_dims[0], video_dims[1]
            
        # Open and process the image
        img = Image.open(source_image_path)
        original_width, original_height = img.size

        # Calculate scaling factor to fit the smallest side
        width_ratio = target_width / original_width
        height_ratio = target_height / original_height
        scale_factor = min(width_ratio, height_ratio)

        # Calculate new dimensions
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)

        # Resize the image
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Calculate cropping box (center crop)
        left = (new_width - target_width) / 2
        top = (new_height - target_height) / 2
        right = (new_width + target_width) / 2
        bottom = (new_height + target_height) / 2

        # Crop the image
        img = img.crop((left, top, right, bottom))

        # Save the processed image
        img.save(os.path.join(output_dir, target_filename))
        print(f"Saved processed image to {os.path.join(output_dir, target_filename)}")
        
    except Exception as e:
        print(f"Error processing image {source_image_path}: {e}")

async def save_training_config_to_yaml(training_tab_container, selected_image_path_c1: str = None, selected_image_path_c2: str = None):
    """
    Extracts the config from the UI and saves it as a YAML file.
    Returns the output path and the YAML dictionary.
    """
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    
    def _save_config():
        from ui.utils.utils_top_menu import TopBarUtils
        yaml_dict = TopBarUtils.build_yaml_config_from_ui(
            training_tab_container, 
            current_selected_image_path_c1=selected_image_path_c1,
            current_selected_image_path_c2=selected_image_path_c2
        )
        
        out_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "assets", "config_to_train.yaml")
        )
        
        # Save the YAML config
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_dict, f, sort_keys=False, allow_unicode=True, Dumper=yaml.SafeDumper)
        
        # Process images in a non-blocking way
        if selected_image_path_c1 and os.path.exists(selected_image_path_c1):
            _process_and_save_image(yaml_dict, selected_image_path_c1, 'img1.png')
        if selected_image_path_c2 and os.path.exists(selected_image_path_c2):
            _process_and_save_image(yaml_dict, selected_image_path_c2, 'img2')
            
        return out_path, yaml_dict
    
    # Run the blocking operations in a thread
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        result = await loop.run_in_executor(pool, _save_config)
    
    print(f"Saved training config to {result[0]}")
    return result

async def run_training_batch_file(use_multi_gpu: bool):
    """
    Runs the appropriate training batch file based on the multi_gpu flag.
    Returns the batch file path.
    """
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    
    def _run_batch():
        if use_multi_gpu:
            bat_filename = "start_last_multi.bat"
        else:
            bat_filename = "start_last.bat"
            
        bat_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", bat_filename)
        )
        
        # Run the batch file in a non-blocking way
        import subprocess
        print(f"DEBUG: Executing batch file: {bat_path}")
        subprocess.Popen([bat_path], shell=True, cwd=os.path.dirname(bat_path))
        return bat_path
    
    # Run the blocking operation in a thread
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        bat_path = await loop.run_in_executor(pool, _run_batch)
    
    print(f"Started training with batch file: {bat_path}")
    return bat_path

def is_dataset_selected(config_page_content):
    """
    Checks if a dataset is selected in the config page content.
    Returns the selected dataset or None.
    """
    dataset_block = getattr(config_page_content, 'dataset_block', None)
    if dataset_block and hasattr(dataset_block, 'get_selected_dataset'):
        return dataset_block.get_selected_dataset()
    return None

# =====================
# GUI-Building Functions
# =====================

def build_navigation_rail(on_nav_change):
    """
    Builds the sub-navigation rail for the training tab.
    """
    config_dest_content = ft.Container(
        content=ft.Text("Config", size=10),
        padding=ft.padding.symmetric(vertical=0, horizontal=0),
        alignment=ft.alignment.center,
        expand=True
    )
    sampling_dest_content = ft.Container(
        content=ft.Text("Sampling", size=10),
        padding=ft.padding.symmetric(vertical=0, horizontal=0),
        alignment=ft.alignment.center,
        expand=True
    )
    return ft.NavigationRail(
        selected_index=0,
        on_change=on_nav_change,
        bgcolor=ft.Colors.TRANSPARENT,
        indicator_color=ft.Colors.TRANSPARENT,
        indicator_shape=ft.RoundedRectangleBorder(radius=0),
        label_type=ft.NavigationRailLabelType.NONE,
        destinations=[
            ft.NavigationRailDestination(icon=config_dest_content),
            ft.NavigationRailDestination(icon=sampling_dest_content),
        ]
    )

def build_main_content_row(sub_navigation_rail, content_area):
    """
    Builds the main content row containing the navigation rail and content area.
    """
    return ft.Row(
        [
            sub_navigation_rail,
            ft.VerticalDivider(),
            content_area,
        ],
        expand=True,
        spacing=0,
        vertical_alignment=ft.CrossAxisAlignment.START
    )

def build_bottom_app_bar(on_start_click, multi_gpu_checkbox):
    """
    Builds the bottom app bar with the Start button and Multi-GPU checkbox.
    """
    return ft.BottomAppBar(
        bgcolor=ft.Colors.BLUE_GREY_900,
        height=60,
        content=ft.Row(
            [
                # Container for Multi-GPU checkbox
                ft.Container(
                    content=multi_gpu_checkbox,
                    alignment=ft.alignment.center_left,
                    expand=True,
                    padding=ft.padding.only(left=20) # Add some padding
                ),
                # Container for Start button
                ft.Container(
                    ft.ElevatedButton(
                        "Start",
                        on_click=on_start_click,
                        style=ft.ButtonStyle(
                            text_style=ft.TextStyle(size=14),
                            shape=ft.RoundedRectangleBorder(radius=0)
                        ),
                        width=150,
                        height=40,
                    ),
                    alignment=ft.alignment.center_right,
                    padding=ft.padding.only(right=20), # Add some padding
                ),
            ],
            expand=True,
        ),
    )

def build_main_container(main_content_row, bottom_app_bar):
    """
    Wraps the main content and bottom bar in a Stack for sticky footer.
    """
    return ft.Container(
        padding=ft.padding.only(top=0, bottom=0),
        content=ft.Stack(
            [
                ft.Container(main_content_row, expand=True),
                ft.Container(
                    bottom_app_bar,
                    left=0,
                    right=0,
                    bottom=0,
                    alignment=ft.alignment.bottom_center
                ),
            ],
            expand=True,
        ),
        expand=True
    )

# =====================
# Main Entry Point
# =====================

def get_training_tab_content(page: ft.Page):
    """
    Entry point for building the Training tab content. Sets up navigation, content, and event handlers.
    """
    page.snack_bar = ft.SnackBar(content=ft.Text("Training tab loaded! (debug)"), open=True)
    page.update()

    # Initialize config and sampling page content
    config_page_content = get_training_config_page_content()
    sampling_page_content = build_training_sampling_page_content(page)

    # Content area container
    content_area = ft.Container(
        content=config_page_content,
        expand=True,
        alignment=ft.alignment.top_left,
        padding=ft.padding.all(0)
    )

    def on_nav_change(e):
        try:
            # Switch content
            selected_idx = e.control.selected_index
            if selected_idx == 0:
                content_area.content = config_page_content
            elif selected_idx == 1:
                content_area.content = sampling_page_content
            page.update()
        except Exception as ex:
            print(f"Error switching tabs: {ex}")
            page.update()

    sub_navigation_rail = build_navigation_rail(on_nav_change)
    main_content_row = build_main_content_row(sub_navigation_rail, content_area)

    # Add Multi-GPU checkbox
    multi_gpu_checkbox = ft.Checkbox(label="Multi-GPU", value=False) # False by default

    async def handle_training_output(page=None, training_tab_container=None):
        """
        Handles saving config and running training, with error handling and user feedback.
        """
        print("DEBUG: Entering handle_training_output")
        try:
            if training_tab_container is None:
                msg = "Error: training_tab_container not passed to handle_training_output."
                print(f"DEBUG: {msg}")
                if page is not None:
                    page.snack_bar = ft.SnackBar(content=ft.Text(msg), open=True)
                    page.update()
                return

            print("DEBUG: Getting selected image paths")
            # Get the selected image paths from the page object
            current_selected_image_path_c1 = getattr(page, 'selected_image_path_c1', None)
            current_selected_image_path_c2 = getattr(page, 'selected_image_path_c2', None)
            print(f"DEBUG: Image paths - c1: {current_selected_image_path_c1}, c2: {current_selected_image_path_c2}")
            
            print("DEBUG: Saving training config to YAML")
            # Pass both image paths to the save function
            out_path, _ = await save_training_config_to_yaml(
                training_tab_container, 
                selected_image_path_c1=current_selected_image_path_c1,
                selected_image_path_c2=current_selected_image_path_c2
            )
            print(f"DEBUG: Config saved to {out_path}")

            # Determine which batch file to run based on checkbox state
            use_multi_gpu = multi_gpu_checkbox.value
            print(f"DEBUG: Running batch file (multi_gpu={use_multi_gpu})")
            bat_path = await run_training_batch_file(use_multi_gpu)
            print(f"DEBUG: Batch file executed: {bat_path}")

            msg = f"Saved config to {out_path}\nBatch file: {bat_path}\nTraining started."
            print(f"DEBUG: Showing success message: {msg}")
            if page is not None:
                page.snack_bar = ft.SnackBar(content=ft.Text(msg), open=True)
                page.update()
            print("DEBUG: handle_training_output completed successfully")
        except Exception as e:
            msg = f"Error: {e}"
            print(f"ERROR in handle_training_output: {e}", exc_info=True)
            if page is not None:
                page.snack_bar = ft.SnackBar(content=ft.Text(msg), open=True)
                page.update()

    async def handle_start_click(e, training_tab_container_arg):
        """
        Handles the Start button click: checks dataset selection and triggers training.
        """
        print("DEBUG: Start button clicked")
        try:
            print("DEBUG: Checking if dataset is selected")
            dataset_selected = is_dataset_selected(config_page_content)
            print(f"DEBUG: Dataset selected: {dataset_selected}")
            
            async def run_training():
                print("DEBUG: Running training process")
                try:
                    await handle_training_output(e.page, training_tab_container_arg)
                    print("DEBUG: Training process completed")
                except Exception as ex:
                    print(f"ERROR in run_training: {ex}", exc_info=True)
                    if e.page:
                        e.page.snack_bar = ft.SnackBar(
                            content=ft.Text(f"Error starting training: {ex}"), 
                            open=True
                        )
                        e.page.update()
            
            if not dataset_selected:
                print("DEBUG: No dataset selected, showing confirmation dialog")
                def on_confirm():
                    print("DEBUG: User confirmed to proceed without dataset")
                    e.page.run_task(run_training)
                
                dataset_not_selected.show_dataset_not_selected_dialog(
                    e.page, "Dataset not selected, proceed?", on_confirm
                )
            else:
                print("DEBUG: Dataset selected, starting training")
                await run_training()
                
        except Exception as ex:
            error_msg = f"Error in handle_start_click: {ex}"
            print(error_msg, exc_info=True)
            if e and e.page:
                e.page.snack_bar = ft.SnackBar(
                    content=ft.Text(error_msg),
                    open=True
                )
                e.page.update()

    # Create a wrapper function that can be called synchronously
    def start_button_click(e):
        print("DEBUG: Start button clicked (wrapper)")
        e.page.run_task(handle_start_click, e, main_container)
    
    # Build the bottom app bar with the wrapper function
    bottom_bar = build_bottom_app_bar(start_button_click, multi_gpu_checkbox)
    main_container = build_main_container(main_content_row, bottom_bar)

    # Attach references for later extraction
    main_container.config_page_content = config_page_content
    
    main_container.sampling_page_content = sampling_page_content
    if hasattr(config_page_content, 'dataset_block'):
        main_container.dataset_page_content = config_page_content.dataset_block
    page.training_tab_container = main_container
    return main_container