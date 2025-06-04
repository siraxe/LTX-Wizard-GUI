import os
import yaml
from pathlib import Path
import flet as ft
from loguru import logger
from PIL import Image
import re
import traceback # Import traceback for detailed error logging

# Initialize logger if not already set up
try:
    logger
except NameError:
    import sys
    logger.remove() # Remove default handler
    logger.add(sys.stderr, level="DEBUG") # Add new handler with DEBUG level

# selected_image_path_c1 and selected_image_path_c2 will be imported dynamically
# within handle_save and handle_save_as to get current values.

class TopBarUtils:
    @staticmethod
    def extract_config_from_controls(control):
        result = {}
        if hasattr(control, 'controls') and control.controls:
            for child in control.controls:
                child_result = TopBarUtils.extract_config_from_controls(child)
                if child_result:
                    result.update(child_result)
        elif hasattr(control, 'content') and control.content:
            child_result = TopBarUtils.extract_config_from_controls(control.content)
            if child_result:
                result.update(child_result)
        elif isinstance(control, ft.TextField):
            result[control.label] = control.value
        elif isinstance(control, ft.Dropdown):
            result[control.label] = control.value
        elif isinstance(control, ft.Checkbox):
            result[control.label] = control.value
        return result

    @staticmethod
    def _load_cropped_image_into_ui(page: ft.Page, image_path: str, target_control_key: str,
                                  image_display_c1=None, image_display_c2=None):
        """
        Load a cropped image into the UI control and update the remove button visibility.
        Args:
            page: The Flet page object
            image_path: Path to the image to load
            target_control_key: Either 'c1' or 'c2' to specify which control to update
            image_display_c1: Optional reference to the C1 image display control (can be derived from page)
            image_display_c2: Optional reference to the C2 image display control (can be derived from page)
        """

        image_display_target = None
        if target_control_key.lower() == 'c1':
            image_display_target = image_display_c1 if image_display_c1 else getattr(page, 'image_display_c1', None)
        elif target_control_key.lower() == 'c2':
            image_display_target = image_display_c2 if image_display_c2 else getattr(page, 'image_display_c2', None)

        if not image_display_target:
            logger.error(f"Target display control for {target_control_key} not found in _load_cropped_image_into_ui.")
            return

        path_exists = os.path.exists(image_path) if image_path else False

        if not image_path or not path_exists:
            logger.warning(f"Image not found at '{image_path}' in _load_cropped_image_into_ui. Clearing display for {target_control_key}.")
            image_display_target.src = None # Or a placeholder like "/images/image_placeholder.png"
            image_display_target.visible = False
            if hasattr(page, f'selected_image_path_{target_control_key.lower()}'):
                 setattr(page, f'selected_image_path_{target_control_key.lower()}', None)

            try:
                if hasattr(image_display_target, 'hide_remove_button'):
                    image_display_target.hide_remove_button()
                image_display_target.update()
            except Exception as e_clear_update:
                logger.error(f"Error updating display or hide_remove_button for {target_control_key} (path not found): {e_clear_update}")
                logger.error(traceback.format_exc())
            return

        try:
            image_path_norm = image_path.replace('\\', '/')

            if hasattr(page, f'selected_image_path_{target_control_key.lower()}'):
                setattr(page, f'selected_image_path_{target_control_key.lower()}', image_path_norm)

            image_display_target.src = image_path_norm
            image_display_target.visible = True

            # Optional: Ensure the image is visible and properly sized (Flet's fit usually handles this)
            # if hasattr(image_display_target, 'width'): image_display_target.width = 200
            # if hasattr(image_display_target, 'height'): image_display_target.height = 200

            image_display_target.update()

            if hasattr(image_display_target, 'show_remove_button'):
                image_display_target.show_remove_button()
            else:
                logger.warning(f"image_display_{target_control_key} does not have show_remove_button method (called from _load_cropped_image_into_ui).")

            if hasattr(image_display_target, 'parent') and hasattr(image_display_target.parent, 'update'):
                image_display_target.parent.update()

            if page: page.update()
            logger.info(f"Successfully loaded image for {target_control_key} from '{image_path_norm}' in _load_cropped_image_into_ui.")

        except Exception as e:
            logger.error(f"Error in _load_cropped_image_into_ui for {target_control_key} with path '{image_path}': {e}")
            logger.error(traceback.format_exc())

    @staticmethod
    def _save_and_scale_image(source_image_path: str, video_dims_tuple: tuple, dataset_name: str, 
                            target_filename: str, page: ft.Page = None, target_control: str = None,
                            image_display_c1=None, image_display_c2=None):
        if not source_image_path or not dataset_name:
            logger.warning("_save_and_scale_image: Missing source_image_path or dataset_name.")
            return None

        try:
            img = Image.open(source_image_path)
        except FileNotFoundError:
            logger.error(f"Error: Source image not found at {source_image_path} in _save_and_scale_image.")
            return None
        except Exception as e:
            logger.error(f"Error opening image {source_image_path} in _save_and_scale_image: {e}")
            return None

        dataset_sample_images_dir = Path("workspace") / "datasets" / dataset_name / "sample_images"
        dataset_sample_images_dir.mkdir(parents=True, exist_ok=True)
        target_path = dataset_sample_images_dir / target_filename

        original_width, original_height = img.size
        target_width, target_height = video_dims_tuple[0], video_dims_tuple[1]

        if original_width == 0 or original_height == 0 or target_width == 0 or target_height == 0:
            logger.warning(f"Invalid dimensions for scaling. Original: {img.size}, Target: {(target_width, target_height)}. Saving original.")
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            img.save(target_path)
            return str(target_path).replace('\\', '/')

        width_ratio = target_width / original_width
        height_ratio = target_height / original_height
        scale_factor = max(width_ratio, height_ratio)
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)
        
        if new_width <= 0 or new_height <= 0:
            logger.warning(f"Calculated new dimensions are invalid ({new_width}x{new_height}). Saving original.")
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            img.save(target_path)
            return str(target_path).replace('\\', '/')

        try:
            img_resized = img.resize((new_width, new_height), Image.LANCZOS)
            left = (new_width - target_width) // 2
            top = (new_height - target_height) // 2
            right = left + target_width
            bottom = top + target_height
            img_cropped = img_resized.crop((left, top, right, bottom))
            
            if img_cropped.mode == 'RGBA':
                img_cropped = img_cropped.convert('RGB')
            
            img_cropped.save(target_path)
            result_path = str(target_path).replace('\\', '/')
            logger.info(f"Image saved and scaled to: {result_path}")
            
            if page is not None and target_control and target_control.lower() in ['c1', 'c2']:
                logger.debug(f"Calling _load_cropped_image_into_ui from _save_and_scale_image for {target_control} with path {result_path}")
                TopBarUtils._load_cropped_image_into_ui(
                    page=page,
                    image_path=result_path,
                    target_control_key=target_control, # Pass as target_control_key
                    image_display_c1=image_display_c1, # Pass along if available
                    image_display_c2=image_display_c2  # Pass along if available
                )
            return result_path
        except Exception as e:
            logger.error(f"Error resizing or saving image {source_image_path} to {target_path}: {e}")
            logger.error(traceback.format_exc())
            return None

    @staticmethod
    def build_yaml_config_from_ui(training_tab_container, 
                                current_selected_image_path_c1: str = None, 
                                current_selected_image_path_c2: str = None,
                                image_display_c1=None, # Pass these along
                                image_display_c2=None):
        logger.debug("Building YAML config from UI...")
        config_controls = training_tab_container.config_page_content
        sampling_controls = training_tab_container.sampling_page_content
        dataset_controls = getattr(training_tab_container, 'dataset_page_content', None)
        
        config = TopBarUtils.extract_config_from_controls(config_controls)
        sampling = TopBarUtils.extract_config_from_controls(sampling_controls)
        
        dataset_path = None
        num_workers = 2 
        selected_dataset_name_for_images = None
        
        if dataset_controls and hasattr(dataset_controls, 'get_selected_dataset'):
            selected_dataset_value = dataset_controls.get_selected_dataset()
            if selected_dataset_value:
                dataset_path = os.path.join('workspace', 'datasets', selected_dataset_value, 'preprocessed_data').replace('\\', '/')
                selected_dataset_name_for_images = selected_dataset_value
                
        if dataset_controls and hasattr(dataset_controls, 'get_num_workers'):
            try:
                num_workers = int(dataset_controls.get_num_workers())
            except (ValueError, TypeError):
                num_workers = 2
        
        sampling_checkbox = config.get('Sampling', True)
        match_checkbox = config.get('Match', False)
        interval_checkpoints = int(config.get('Interval (Checkpoints)', 250))
        interval_validation = int(sampling.get('Interval (Validation)', 250))
        
        if not sampling_checkbox:
            interval_validation_final = 0
        elif match_checkbox:
            interval_validation_final = interval_checkpoints
        else:
            interval_validation_final = interval_validation
        interval_for_yaml = None if interval_validation_final == 0 else interval_validation_final
        
        blocks_to_swap_value_str = config.get('Block to swap')
        blocks_to_swap_final = None
        if blocks_to_swap_value_str is not None and str(blocks_to_swap_value_str).strip() != "":
            try:
                blocks_to_swap_final = int(blocks_to_swap_value_str)
                if blocks_to_swap_final < 0: blocks_to_swap_final = 0
            except ValueError: blocks_to_swap_final = None
                
        prompts_input = sampling.get('Prompts', '')
        prompt_list = [p.strip() for p in prompts_input.splitlines() if p.strip()]
        
        processed_image_paths_for_yaml = [] # Changed variable name for clarity
        video_dims_str = sampling.get('Video Dims', '[512, 512, 49]')
        video_dims_tuple_for_scaling = (512, 512)
        video_dims_for_yaml = [512, 512, 49] # Default for YAML

        try:
            cleaned_dims_str = video_dims_str.strip()
            if cleaned_dims_str.startswith('[') and cleaned_dims_str.endswith(']'):
                cleaned_dims_str = cleaned_dims_str[1:-1]
            parts = [p.strip() for p in cleaned_dims_str.split(',')]
            if len(parts) >= 2:
                width = int(parts[0])
                height = int(parts[1])
                video_dims_tuple_for_scaling = (width, height)
                video_dims_for_yaml = [width, height, 49]
                if len(parts) >= 3: video_dims_for_yaml[2] = int(parts[2])
            else: raise ValueError("Video Dims format error.")
        except Exception as e:
            logger.warning(f"Error parsing Video Dims '{video_dims_str}': {e}. Using defaults.")
            # Defaults are already set

        page = getattr(training_tab_container, 'page', None)
        if selected_dataset_name_for_images:
            if current_selected_image_path_c1 and os.path.exists(current_selected_image_path_c1):
                img1_rel_path = TopBarUtils._save_and_scale_image(
                    source_image_path=current_selected_image_path_c1, 
                    video_dims_tuple=video_dims_tuple_for_scaling, 
                    dataset_name=selected_dataset_name_for_images, 
                    target_filename="img1.png",
                    page=page, target_control='c1', # Pass target_control
                    image_display_c1=image_display_c1, image_display_c2=image_display_c2 # Pass displays
                )
                if img1_rel_path: processed_image_paths_for_yaml.append(img1_rel_path)
            
            if current_selected_image_path_c2 and os.path.exists(current_selected_image_path_c2):
                img2_rel_path = TopBarUtils._save_and_scale_image(
                    source_image_path=current_selected_image_path_c2, 
                    video_dims_tuple=video_dims_tuple_for_scaling, 
                    dataset_name=selected_dataset_name_for_images, 
                    target_filename="img2.png",
                    page=page, target_control='c2', # Pass target_control
                    image_display_c1=image_display_c1, image_display_c2=image_display_c2 # Pass displays
                )
                if img2_rel_path: processed_image_paths_for_yaml.append(img2_rel_path)
        
        final_image_list_for_yaml = None
        num_prompts = len(prompt_list)
        num_available_images = len(processed_image_paths_for_yaml)

        if num_prompts > 0 and num_available_images > 0:
            final_image_list_for_yaml = []
            if num_prompts == 1:
                final_image_list_for_yaml.append(processed_image_paths_for_yaml[0])
            else: 
                if num_available_images == 1:
                    for _ in range(num_prompts): final_image_list_for_yaml.append(processed_image_paths_for_yaml[0])
                else: 
                    for i in range(num_prompts): final_image_list_for_yaml.append(processed_image_paths_for_yaml[i % num_available_images])
        
        yaml_dict = {
            'model': {
                'model_source': config.get('Model Source'), 'training_mode': config.get('Training Mode'),
                'load_checkpoint': config.get('Last Checkpoint') or None, 'blocks_to_swap': blocks_to_swap_final,
            },
            'lora': {
                'rank': int(config.get('Rank', 128)), 'alpha': int(config.get('Alpha', 128)),
                'dropout': float(config.get('Dropout', 0.0)),
                'target_modules': [s.strip() for s in (config.get('Target Modules') or '').split(',') if s.strip()],
            },
            'optimization': {
                'learning_rate': float(config.get('Learning Rate', 2e-4)), 'steps': int(config.get('Steps', 2000)),
                'batch_size': int(config.get('Batch Size', 1)), 
                'gradient_accumulation_steps': int(config.get('Gradient Accumulation Steps', 1)),
                'max_grad_norm': float(config.get('Max Grad Norm', 1.0)), 'optimizer_type': config.get('Optimizer Type'),
                'scheduler_type': config.get('Scheduler Type'), 'scheduler_params': {},
                'enable_gradient_checkpointing': bool(config.get('Gradient Checkpointing', True)),
                'first_frame_conditioning_p': float(config.get('First Frame Conditioning P', 0.5)),
            },
            'acceleration': {
                'mixed_precision_mode': config.get('Mixed Precision Mode'), 'quantization': config.get('Quantization'),
                'load_text_encoder_in_8bit': bool(config.get('Load Text Encoder in 8bit', True)),
                'compile_with_inductor': bool(config.get('Compile with Inductor', False)),
                'compilation_mode': config.get('Compilation Mode'),
            },
            'data': { 'preprocessed_data_root': dataset_path, 'num_dataloader_workers': int(num_workers), },
            'validation': {
                'prompts': prompt_list, 'negative_prompt': sampling.get('Negative Prompt', ''),
                'video_dims': video_dims_for_yaml, 'seed': int(sampling.get('Seed (Validation)', 42)),
                'inference_steps': int(sampling.get('Inference Steps', 50)), 'interval': interval_for_yaml,
                'videos_per_prompt': int(sampling.get('Videos Per Prompt', 1)),
                'guidance_scale': float(sampling.get('Guidance Scale', 3.5)),
                'images': final_image_list_for_yaml
            },
            'checkpoints': {
                'interval': int(config.get('Interval (Checkpoints)', 250)),
                'keep_last_n': int(config.get('Keep Last N', -1)),
            },
            'seed': int(config.get('Seed (General)', 42)), 'output_dir': config.get('Output Directory'),
            'misc': { 'sampling_enabled': sampling_checkbox, 'match_enabled': match_checkbox, }
        }
        logger.debug("Finished building YAML config from UI.")
        return yaml_dict

    @staticmethod
    def set_yaml_path_and_title(page, path, set_as_current=True):
        if set_as_current:
            page.y_name = path
        filename = os.path.basename(path) if path else None
        if filename and filename.lower() == "config_default.yaml":
            page.title = "LTX Trainer"
        elif filename:
            page.title = f"LTX Trainer - {filename}"
        else:
            page.title = "LTX Trainer"
        page.update()

    @staticmethod
    def handle_save(page: ft.Page):
        logger.debug("Handle Save triggered.")
        path = getattr(page, 'y_name', None)
        is_default_loaded = getattr(page, 'is_default_config_loaded', False)

        if is_default_loaded or (path and os.path.basename(path).lower() == "config_default.yaml"):
            logger.debug("Default config loaded or path is default, redirecting to Save As.")
            TopBarUtils.handle_save_as(page)
            return
        
        if path:
            if not path.lower().endswith(('.yaml', '.yml')): path += '.yaml'
            training_tab = getattr(page, 'training_tab_container', None)
            if not training_tab: logger.error("Training tab container not found in handle_save."); return
            
            current_c1 = getattr(page, 'selected_image_path_c1', None)
            current_c2 = getattr(page, 'selected_image_path_c2', None)
            image_display_c1 = getattr(page, 'image_display_c1', None) # Get image display controls
            image_display_c2 = getattr(page, 'image_display_c2', None)
            
            yaml_dict = TopBarUtils.build_yaml_config_from_ui(
                training_tab, 
                current_selected_image_path_c1=current_c1, current_selected_image_path_c2=current_c2,
                image_display_c1=image_display_c1, image_display_c2=image_display_c2 # Pass them
            )
            try:
                with open(path, 'w', encoding='utf-8') as f:
                    yaml.dump(yaml_dict, f, sort_keys=False, allow_unicode=True, Dumper=yaml.SafeDumper)
                logger.info(f"Config saved to {path}")
                TopBarUtils.set_yaml_path_and_title(page, path)
                TopBarUtils.add_recent_file(path, page)
                page.is_default_config_loaded = False
            except Exception as e:
                logger.error(f"Error saving file to {path}: {e}")
                logger.error(traceback.format_exc())
        else:
            logger.debug("No path set, redirecting to Save As.")
            TopBarUtils.handle_save_as(page)

    @staticmethod
    def handle_save_as(page: ft.Page):
        logger.debug("Handle Save As triggered.")
        file_picker = ft.FilePicker()
        page.overlay.append(file_picker)
        page.update()
        def on_save_result(e: ft.FilePickerResultEvent):
            if e.path:
                path = e.path
                if not path.lower().endswith(('.yaml', '.yml')): path += '.yaml'
                training_tab = getattr(page, 'training_tab_container', None)
                if not training_tab: logger.error("Training tab container not found in on_save_result."); return

                current_c1 = getattr(page, 'selected_image_path_c1', None)
                current_c2 = getattr(page, 'selected_image_path_c2', None)
                image_display_c1 = getattr(page, 'image_display_c1', None)
                image_display_c2 = getattr(page, 'image_display_c2', None)
                
                yaml_dict = TopBarUtils.build_yaml_config_from_ui(
                    training_tab, 
                    current_selected_image_path_c1=current_c1, current_selected_image_path_c2=current_c2,
                    image_display_c1=image_display_c1, image_display_c2=image_display_c2
                )
                try:
                    with open(path, 'w', encoding='utf-8') as f:
                        yaml.dump(yaml_dict, f, sort_keys=False, allow_unicode=True, Dumper=yaml.SafeDumper)
                    logger.info(f"Config saved as to {path}")
                    TopBarUtils.set_yaml_path_and_title(page, path)
                    TopBarUtils.add_recent_file(path, page)
                    page.is_default_config_loaded = False
                except Exception as ex_save:
                    logger.error(f"Error saving file (Save As) to {path}: {ex_save}")
                    logger.error(traceback.format_exc())
            else:
                logger.debug("Save As dialog cancelled or no path selected.")
        file_picker.on_result = on_save_result
        default_name = "ltxv_config.yaml"
        default_dir = os.path.abspath(os.path.join(os.getcwd(), "workspace", "configs"))
        if not os.path.exists(default_dir): os.makedirs(default_dir)
        file_picker.save_file(
            dialog_title="Save config as YAML", file_name=default_name,
            initial_directory=default_dir, allowed_extensions=["yaml", "yml"]
        )

    @staticmethod
    def handle_open(page: ft.Page, file_path=None, set_as_current=True):
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f: config_data = yaml.safe_load(f)
                training_tab = getattr(page, 'training_tab_container', None)
                if not training_tab: logger.error("Training tab container not found in handle_open (direct path)."); return
                
                TopBarUtils.update_ui_from_yaml(training_tab, config_data)
                TopBarUtils.set_yaml_path_and_title(page, file_path, set_as_current=set_as_current)
                TopBarUtils.add_recent_file(file_path, page)
                if os.path.basename(file_path).lower() != "config_default.yaml" and set_as_current:
                    page.is_default_config_loaded = False
                elif os.path.basename(file_path).lower() == "config_default.yaml":
                     page.is_default_config_loaded = True
            except Exception as e:
                logger.error(f"Error opening file {file_path}: {e}")
                logger.error(traceback.format_exc())
            return

        file_picker = ft.FilePicker()
        page.overlay.append(file_picker)
        page.file_picker_open_ref = file_picker # Store ref if needed elsewhere
        page.update()
        def on_open_result(e: ft.FilePickerResultEvent):
            if e.files and len(e.files) > 0:
                path = e.files[0].path
                try:
                    with open(path, 'r', encoding='utf-8') as f: config_data = yaml.safe_load(f)
                    training_tab = getattr(page, 'training_tab_container', None)
                    if not training_tab: logger.error("Training tab container not found in on_open_result."); return
                    
                    TopBarUtils.update_ui_from_yaml(training_tab, config_data)
                    TopBarUtils.set_yaml_path_and_title(page, path, set_as_current=set_as_current)
                    TopBarUtils.add_recent_file(path, page)
                    if os.path.basename(path).lower() != "config_default.yaml" and set_as_current:
                        page.is_default_config_loaded = False
                    logger.info(f"Opened config from dialog: {path}")
                except Exception as ex_open:
                    logger.error(f"Error opening file from dialog {path}: {ex_open}")
                    logger.error(traceback.format_exc())
            else:
                logger.debug("Open file dialog cancelled or no file selected.")
        file_picker.on_result = on_open_result
        default_dir = os.path.abspath(os.path.join(os.getcwd(), "workspace", "configs"))
        if not os.path.exists(default_dir): os.makedirs(default_dir)
        file_picker.pick_files(
            dialog_title="Open config YAML", initial_directory=default_dir,
            allowed_extensions=["yaml", "yml"], allow_multiple=False
        )

    @staticmethod
    def update_ui_from_yaml(training_tab_container, config_data):
        page = getattr(training_tab_container, 'page', None)
        if not page:
            logger.error("Page object not found in training_tab_container for update_ui_from_yaml.")
            return

        # --- Image loading section ---
        validation_config = config_data.get('validation', {})
        loaded_image_paths_from_yaml = []
        if isinstance(validation_config, dict):
            validation_images_from_yaml = validation_config.get('images', []) # Expect a list
            if isinstance(validation_images_from_yaml, list):
                loaded_image_paths_from_yaml = validation_images_from_yaml
            elif isinstance(validation_images_from_yaml, str) and validation_images_from_yaml: # Handle if it's a single string
                loaded_image_paths_from_yaml = [validation_images_from_yaml]

        image_controls_map = {
            'c1': getattr(page, 'image_display_c1', None),
            'c2': getattr(page, 'image_display_c2', None)
        }

        for i, target_control_key in enumerate(['c1', 'c2']):
            image_display_control = image_controls_map[target_control_key]
            img_path_from_yaml = None
            if i < len(loaded_image_paths_from_yaml) and loaded_image_paths_from_yaml[i]: # Check if path is not None/empty
                img_path_from_yaml = loaded_image_paths_from_yaml[i]

            if image_display_control:
                path_exists = os.path.exists(img_path_from_yaml) if img_path_from_yaml else False

                if img_path_from_yaml and path_exists:
                    norm_img_path = img_path_from_yaml.replace('\\', '/')
                    image_display_control.src = norm_img_path
                    image_display_control.visible = True
                    setattr(page, f'selected_image_path_{target_control_key}', norm_img_path)

                    # Call show_remove_button - it should ONLY set the .visible property, not call .update()
                    if hasattr(image_display_control, 'show_remove_button'):
                        image_display_control.show_remove_button()
                    else:
                        logger.warning(f"show_remove_button method not found on {target_control_key}.")

                else:
                    if img_path_from_yaml and not path_exists:
                        logger.warning(f"Image path for {target_control_key} ('{img_path_from_yaml}') not found. Clearing display.")
                    else:
                        logger.debug(f"No valid image path for {target_control_key} in YAML. Clearing display.")

                    image_display_control.src = None # Or placeholder "/images/image_placeholder.png"
                    image_display_control.visible = False
                    setattr(page, f'selected_image_path_{target_control_key}', None)

                    # Call hide_remove_button - it should ONLY set the .visible property, not call .update()
                    if hasattr(image_display_control, 'hide_remove_button'):
                        logger.debug(f"Calling hide_remove_button for {target_control_key}.")
                        image_display_control.hide_remove_button()
                    else:
                        logger.warning(f"hide_remove_button method not found on {target_control_key}.")
            else:
                logger.warning(f"Image display control for {target_control_key} not found on page object.")

        # --- End of Image loading section ---

        YAML_TO_LABEL = {
            "model_source": "Model Source", "training_mode": "Training Mode", "load_checkpoint": "Last Checkpoint",
            "blocks_to_swap": "Block to swap", "rank": "Rank", "alpha": "Alpha", "dropout": "Dropout",
            "target_modules": "Target Modules", "learning_rate": "Learning Rate", "steps": "Steps",
            "batch_size": "Batch Size", "gradient_accumulation_steps": "Gradient Accumulation Steps",
            "max_grad_norm": "Max Grad Norm", "optimizer_type": "Optimizer Type", "scheduler_type": "Scheduler Type",
            "scheduler_params": "Scheduler Params", "enable_gradient_checkpointing": "Gradient Checkpointing",
            "first_frame_conditioning_p": "First Frame Conditioning P", "mixed_precision_mode": "Mixed Precision Mode",
            "quantization": "Quantization", "load_text_encoder_in_8bit": "Load Text Encoder in 8bit",
            "compile_with_inductor": "Compile with Inductor", "compilation_mode": "Compilation Mode",
            "preprocessed_data_root": "Preprocessed Data Root", "num_dataloader_workers": "Num Dataloader Workers",
            "prompts": "Prompts", "negative_prompt": "Negative Prompt", "video_dims": "Video Dims",
            "seed": "Seed (Validation)", "inference_steps": "Inference Steps", "interval": "Interval (Validation)",
            "videos_per_prompt": "Videos Per Prompt", "guidance_scale": "Guidance Scale",
            "interval_checkpoints": "Interval (Checkpoints)", "keep_last_n": "Keep Last N",
            "timestep_sampling_mode": "Timestep Sampling Mode", "timestep_sampling_params": "Timestep Sampling Params",
            "seed_general": "Seed (General)", "output_dir": "Output Directory",
        }
        def flatten_yaml(cfg, parent_key=""):
            flat = {}
            for k, v in cfg.items():
                key = k
                if parent_key == "validation" and k == "seed": key = "seed"
                elif parent_key == "checkpoints" and k == "interval": key = "interval_checkpoints"
                elif parent_key == "seed": key = "seed_general"
                elif parent_key == "misc" and k == "sampling_enabled": key = "Sampling"
                elif parent_key == "misc" and k == "match_enabled": key = "Match"
                if isinstance(v, dict): flat.update(flatten_yaml(v, k))
                else: flat[key] = v
            return flat
        
        flat_config = flatten_yaml(config_data)

        def update_controls_recursive(control_container):
            if hasattr(control_container, 'controls') and control_container.controls:
                for child_control in control_container.controls: update_controls_recursive(child_control)
            if hasattr(control_container, 'content') and control_container.content:
                update_controls_recursive(control_container.content)
            
            label = getattr(control_container, 'label', None)
            if not label: return

            yaml_key_from_map = next((k for k, v in YAML_TO_LABEL.items() if v == label), None)
            
            # Special handling for misc keys that map directly to labels
            if label == "Sampling" and "Sampling" in flat_config:
                val = flat_config["Sampling"]
                if isinstance(control_container, ft.Checkbox): control_container.value = bool(val)
            elif label == "Match" and "Match" in flat_config:
                val = flat_config["Match"]
                if isinstance(control_container, ft.Checkbox): control_container.value = bool(val)
            elif label == "Prompts" and yaml_key_from_map and yaml_key_from_map in flat_config: # Handle prompts specifically
                val = flat_config[yaml_key_from_map]
                if isinstance(control_container, ft.TextField):
                     if isinstance(val, list): control_container.value = '\n'.join(str(x) for x in val)
                     else: control_container.value = str(val) if val is not None else ""
            elif yaml_key_from_map and yaml_key_from_map in flat_config:
                val = flat_config[yaml_key_from_map]
                if yaml_key_from_map == "interval" and val is None: val = "0" # Interval (Validation)
                
                if isinstance(control_container, ft.TextField):
                    if isinstance(val, list): control_container.value = ', '.join(str(x) for x in val)
                    else: control_container.value = str(val) if val is not None else ""
                elif isinstance(control_container, ft.Dropdown):
                    control_container.value = str(val) if val is not None else None
                elif isinstance(control_container, ft.Checkbox):
                    control_container.value = bool(val)
            # else:
                # logger.debug(f"No matching YAML key for label '{label}' or key not in flat_config.")

        update_controls_recursive(training_tab_container.config_page_content)
        update_controls_recursive(training_tab_container.sampling_page_content)

        dataset_page_content = getattr(training_tab_container, 'dataset_page_content', None)
        if dataset_page_content:
            page_ctx = getattr(training_tab_container, 'page', None) # Should be same as page var above
            num_workers_yaml = flat_config.get("num_dataloader_workers")
            if num_workers_yaml is not None and hasattr(dataset_page_content, 'set_num_workers'):
                try:
                    num_workers_field = getattr(dataset_page_content, 'num_workers_field_ref', None)
                    if num_workers_field and hasattr(num_workers_field, 'current') and getattr(num_workers_field.current, 'page', None):
                        dataset_page_content.set_num_workers(num_workers_yaml, page_ctx)
                except Exception as e_num_workers: logger.error(f"Error setting num_workers: {e_num_workers}")
            
            dataset_root_yaml = flat_config.get("preprocessed_data_root")
            if (dataset_root_yaml is None or str(dataset_root_yaml).lower() == "null" or str(dataset_root_yaml).strip() == "") and hasattr(dataset_page_content, 'set_selected_dataset'):
                try: dataset_page_content.set_selected_dataset(None, page_ctx)
                except Exception as e_set_ds_none: logger.error(f"Error setting dataset to None: {e_set_ds_none}")
            elif dataset_root_yaml and hasattr(dataset_page_content, 'set_selected_dataset'):
                try:
                    dataset_path_str = str(dataset_root_yaml)
                    dataset_folder_name = os.path.basename(os.path.dirname(dataset_path_str)) if dataset_path_str.endswith('preprocessed_data') else os.path.basename(dataset_path_str)
                    dataset_page_content.set_selected_dataset(dataset_folder_name, page_ctx)
                except Exception as e_set_ds: logger.error(f"Error setting dataset from path '{dataset_root_yaml}': {e_set_ds}")
        
        # Finally, update the page AFTER all control properties have been set.
        # This single page.update() should be sufficient to render all changes
        # when the relevant controls become visible.
        if page:
            try:
                page.update()
            except Exception as e_page_update:
                # Log any errors during the final page update
                logger.error(f"Error during final page update in update_ui_from_yaml: {e_page_update}")
                logger.error(traceback.format_exc())

    @staticmethod
    def handle_load_default(page: ft.Page):
        logger.debug("Handle Load Default triggered.")
        default_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'assets', 'config_default.yaml')
        if not os.path.exists(default_path):
            logger.error(f"Default config file not found at: {default_path}")
            # Optionally show a message to the user in the UI
            if hasattr(page, "show_snackbar"): # Assuming a snackbar method
                page.show_snackbar(ft.SnackBar(ft.Text(f"Error: Default config not found."), open=True))
            return

        TopBarUtils.handle_open(page, file_path=default_path, set_as_current=False) # set_as_current=False for default
        page.is_default_config_loaded = True 

        training_tab = getattr(page, 'training_tab_container', None)
        if training_tab:
            dataset_page_content = getattr(training_tab, 'dataset_page_content', None)
            if dataset_page_content and hasattr(dataset_page_content, 'set_selected_dataset'):
                try:
                    dataset_page_content.set_selected_dataset(None, page) # Clear dataset selection
                    logger.debug("Dataset selection cleared after loading default config.")
                except Exception as e_clear_ds:
                    logger.error(f"Error clearing dataset after loading default: {e_clear_ds}")
        else:
            logger.warning("Training tab container not found after loading default config.")
        page.update() # Ensure page updates after clearing dataset

    @staticmethod
    def get_recent_files_path():
        # Ensure the path is relative to this file's directory if it's intended to be bundled
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'recent_files.txt')

    @staticmethod
    def load_recent_files():
        path = TopBarUtils.get_recent_files_path()
        if not os.path.exists(path): return []
        try:
            with open(path, 'r', encoding='utf-8') as f:
                files = [line.strip() for line in f if line.strip() and os.path.exists(line.strip())] # Also check existence
            return files
        except Exception as e:
            logger.error(f"Error loading recent files from {path}: {e}")
            return []

    @staticmethod
    def save_recent_files(files):
        path = TopBarUtils.get_recent_files_path()
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                for file_path in files[:5]: # Save top 5 recent files
                    f.write(file_path + '\n')
        except Exception as e:
            logger.error(f"Error saving recent files to {path}: {e}")

    @staticmethod
    def add_recent_file(filepath, page=None):
        if not filepath or not os.path.exists(filepath): # Do not add non-existent files
            logger.warning(f"Attempted to add non-existent or empty filepath to recents: '{filepath}'")
            return
        files = TopBarUtils.load_recent_files()
        if filepath in files: files.remove(filepath)
        files.insert(0, filepath)
        TopBarUtils.save_recent_files(files)
        if page and hasattr(page, 'refresh_menu_bar'):
            page.refresh_menu_bar()

    @staticmethod
    def get_recent_files_menu_items(on_click_handler, text_size=10): # Renamed on_click to on_click_handler
        files = TopBarUtils.load_recent_files()
        if not files:
            return [ft.MenuItemButton(content=ft.Text("None", size=text_size, italic=True), disabled=True)]
        items = []
        for f_path in files:
            # Display a shorter version of the path, e.g., "configs/my_config.yaml"
            try:
                # Try to make it relative to 'workspace' or show last two parts
                workspace_dir = os.path.abspath("workspace")
                if f_path.startswith(workspace_dir):
                    display_name = os.path.relpath(f_path, os.path.dirname(workspace_dir))
                else:
                    display_name = os.path.join(os.path.basename(os.path.dirname(f_path)), os.path.basename(f_path))
            except Exception:
                display_name = os.path.basename(f_path) # Fallback

            items.append(ft.MenuItemButton(content=ft.Text(display_name, size=text_size, tooltip=f_path), 
                                          on_click=on_click_handler, data=f_path))
        return items

