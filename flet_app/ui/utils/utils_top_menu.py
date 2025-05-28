import os
import yaml
from pathlib import Path
import flet as ft
from loguru import logger
from PIL import Image
import re

# Initialize logger if not already set up
try:
    logger
except NameError:
    import sys
    from loguru import logger
    logger.remove()
    logger.add(sys.stderr, level="INFO")

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
    def _load_cropped_image_into_ui(page: ft.Page, image_path: str, target_control: str, 
                                  image_display_c1=None, image_display_c2=None):
        """
        Load a cropped image into the UI control.
        
        Args:
            page: The Flet page object
            image_path: Path to the image to load
            target_control: Either 'c1' or 'c2' to specify which control to update
            image_display_c1: Optional reference to the C1 image display control
            image_display_c2: Optional reference to the C2 image display control
        """
        if not image_path or not os.path.exists(image_path):
            print(f"Error: Image not found at {image_path}")
            return
            
        try:
            # Ensure we're using forward slashes for consistency
            image_path = image_path.replace('\\', '/')
            
            # Update the page attribute if it exists
            if target_control.lower() == 'c1':
                # Update the page attribute
                if hasattr(page, 'selected_image_path_c1'):
                    page.selected_image_path_c1 = image_path
                
                # Update the image display if control is provided
                if image_display_c1 is not None:
                    image_display_c1.src = image_path
                    image_display_c1.update()
                    
            elif target_control.lower() == 'c2':
                # Update the page attribute
                if hasattr(page, 'selected_image_path_c2'):
                    page.selected_image_path_c2 = image_path
                
                # Update the image display if control is provided
                if image_display_c2 is not None:
                    image_display_c2.src = image_path
                    image_display_c2.update()
            
            # Update the page to reflect changes
            if page is not None:
                page.update()
                
            print(f"Successfully loaded image for {target_control} from {image_path}")
        except Exception as e:
            print(f"Error loading image into UI: {e}")
            import traceback
            traceback.print_exc()

    @staticmethod
    def _save_and_scale_image(source_image_path: str, video_dims_tuple: tuple, dataset_name: str, 
                            target_filename: str, page: ft.Page = None, target_control: str = None,
                            image_display_c1=None, image_display_c2=None):
        """
        Save and scale an image, optionally updating the UI with the result.
        
        Args:
            source_image_path: Path to the source image
            video_dims_tuple: Target dimensions as (width, height)
            dataset_name: Name of the dataset
            target_filename: Output filename (e.g., 'img1.png')
            page: Optional Flet page object for UI updates
            target_control: Optional control to update ('c1' or 'c2')
            image_display_c1: Optional reference to the C1 image display control
            image_display_c2: Optional reference to the C2 image display control
        """
        if not source_image_path or not dataset_name:
            return None

        try:
            img = Image.open(source_image_path)
        except FileNotFoundError:
            print(f"Error: Source image not found at {source_image_path}")
            return None
        except Exception as e:
            print(f"Error opening image {source_image_path}: {e}")
            return None

        # Ensure output directory exists
        dataset_sample_images_dir = Path("workspace") / "datasets" / dataset_name / "sample_images"
        dataset_sample_images_dir.mkdir(parents=True, exist_ok=True)
        target_path = dataset_sample_images_dir / target_filename

        # Scaling logic - scale smallest side to match target, then center crop
        original_width, original_height = img.size
        target_width, target_height = video_dims_tuple[0], video_dims_tuple[1]

        if original_width == 0 or original_height == 0 or target_width == 0 or target_height == 0:
            print(f"Warning: Invalid dimensions for scaling. Original: {img.size}, Target: {(target_width, target_height)}")
            # Save without scaling if dimensions are problematic
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            img.save(target_path)
            return str(target_path).replace('\\', '/')

        # Calculate scaling factor based on the smallest side
        width_ratio = target_width / original_width
        height_ratio = target_height / original_height
        
        # Use the larger scaling factor to ensure the smallest side matches target
        scale_factor = max(width_ratio, height_ratio)
        
        # Calculate new dimensions after scaling
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)
        
        if new_width <= 0 or new_height <= 0:
            print(f"Warning: Calculated new dimensions are invalid ({new_width}x{new_height}). Saving original.")
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            img.save(target_path)
            return str(target_path).replace('\\', '/')

        try:
            # Resize the image (this will make one side match target and the other side be larger)
            img_resized = img.resize((new_width, new_height), Image.LANCZOS)
            
            # Calculate center crop coordinates
            left = (new_width - target_width) // 2
            top = (new_height - target_height) // 2
            right = left + target_width
            bottom = top + target_height
            
            # Perform center crop
            img_cropped = img_resized.crop((left, top, right, bottom))
            
            # Convert to RGB if needed before saving
            if img_cropped.mode == 'RGBA':
                img_cropped = img_cropped.convert('RGB')
            
            # Save the cropped image
            img_cropped.save(target_path)
            result_path = str(target_path).replace('\\', '/')
            
            # Update the UI if page and target_control are provided
            if page is not None and target_control and target_control.lower() in ['c1', 'c2']:
                print(f"Calling _load_cropped_image_into_ui with page={page is not None}, path={result_path}, target={target_control}")
                TopBarUtils._load_cropped_image_into_ui(
                    page=page,
                    image_path=result_path,
                    target_control=target_control,
                    image_display_c1=image_display_c1,
                    image_display_c2=image_display_c2
                )
                
            # Ensure the path is returned with forward slashes for consistency
            return result_path.replace('\\', '/')
        except Exception as e:
            print(f"Error resizing or saving image {source_image_path} to {target_path}: {e}")
            return None

    @staticmethod
    def build_yaml_config_from_ui(training_tab_container, 
                                current_selected_image_path_c1: str = None, 
                                current_selected_image_path_c2: str = None,
                                image_display_c1=None,
                                image_display_c2=None):
        """
        Build a YAML configuration dictionary from the UI state.
        
        Args:
            training_tab_container: The container with all the UI controls
            current_selected_image_path_c1: Path to the first selected image (C1)
            current_selected_image_path_c2: Path to the second selected image (C2)
            image_display_c1: Reference to the C1 image display control
            image_display_c2: Reference to the C2 image display control
            
        Returns:
            dict: The complete YAML configuration as a dictionary
        """
        # Get references to the main UI control containers
        config_controls = training_tab_container.config_page_content
        sampling_controls = training_tab_container.sampling_page_content
        dataset_controls = getattr(training_tab_container, 'dataset_page_content', None)
        
        # Extract configuration from UI controls
        config = TopBarUtils.extract_config_from_controls(config_controls)
        sampling = TopBarUtils.extract_config_from_controls(sampling_controls)
        
        # Get dataset and num_workers from dataset page
        dataset_path = None
        num_workers = 2  # Default value
        selected_dataset_name_for_images = None
        
        if dataset_controls and hasattr(dataset_controls, 'get_selected_dataset'):
            selected_dataset_value = dataset_controls.get_selected_dataset()
            if selected_dataset_value:
                # Ensure forward slashes for consistency
                dataset_path = os.path.join('workspace', 'datasets', selected_dataset_value, 'preprocessed_data').replace('\\', '/')
                selected_dataset_name_for_images = selected_dataset_value
                
        if dataset_controls and hasattr(dataset_controls, 'get_num_workers'):
            try:
                num_workers = int(dataset_controls.get_num_workers())
            except (ValueError, TypeError):
                num_workers = 2  # Fallback to default if conversion fails
        
        # --- Custom logic for Interval (Validation) override ---
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
            
        # Write null to YAML if interval is 0
        interval_for_yaml = None if interval_validation_final == 0 else interval_validation_final
        
        # --- Logic for blocks_to_swap ---
        blocks_to_swap_value_str = config.get('Block to swap')
        blocks_to_swap_final = None  # Default to None if not set or invalid
        if blocks_to_swap_value_str is not None and str(blocks_to_swap_value_str).strip() != "":
            try:
                blocks_to_swap_final = int(blocks_to_swap_value_str)
                if blocks_to_swap_final < 0:
                    blocks_to_swap_final = 0
            except ValueError:
                blocks_to_swap_final = None
                
        # --- Process prompts from UI ---
        prompts_input = sampling.get('Prompts', '')
        prompt_list = [p.strip() for p in prompts_input.splitlines() if p.strip()]
        
        # --- Process validation images ---
        processed_image_paths_for_toml = []
        video_dims_str = sampling.get('Video Dims', '[512, 512, 49]')
        video_dims_tuple_for_scaling = (512, 512)  # Default value

        try:
            # Parse video dimensions
            cleaned_dims_str = video_dims_str.strip()
            if cleaned_dims_str.startswith('[') and cleaned_dims_str.endswith(']'):
                cleaned_dims_str = cleaned_dims_str[1:-1]
            
            parts = [p.strip() for p in cleaned_dims_str.split(',')]
            
            if len(parts) >= 2:
                width = int(parts[0])
                height = int(parts[1])
                video_dims_tuple_for_scaling = (width, height)
                video_dims = [width, height, 49]  # Default to 49 frames if not specified
                if len(parts) >= 3:
                    video_dims[2] = int(parts[2])
            else:
                raise ValueError("Video Dims format requires at least width and height (e.g., '512, 512' or '[512, 512, 49]').")
        except Exception as e:
            print(f"Error parsing Video Dims '{video_dims_str}': {e}. Using default 512x512 for scaling.")
            video_dims = [512, 512, 49]  # Default values
            video_dims_tuple_for_scaling = (512, 512)
            
        # Process and save validation images if a dataset is selected
        if selected_dataset_name_for_images:
            # Get the page object from the training tab container if available
            page = getattr(training_tab_container, 'page', None)
            
            # Process first image (C1) if provided
            if current_selected_image_path_c1 and os.path.exists(current_selected_image_path_c1):
                img1_rel_path = TopBarUtils._save_and_scale_image(
                    source_image_path=current_selected_image_path_c1, 
                    video_dims_tuple=video_dims_tuple_for_scaling, 
                    dataset_name=selected_dataset_name_for_images, 
                    target_filename="img1.png",
                    page=page,
                    target_control='c1',
                    image_display_c1=image_display_c1,
                    image_display_c2=image_display_c2
                )
                if img1_rel_path:
                    processed_image_paths_for_toml.append(img1_rel_path)
            
            # Process second image (C2) if provided
            if current_selected_image_path_c2 and os.path.exists(current_selected_image_path_c2):
                img2_rel_path = TopBarUtils._save_and_scale_image(
                    source_image_path=current_selected_image_path_c2, 
                    video_dims_tuple=video_dims_tuple_for_scaling, 
                    dataset_name=selected_dataset_name_for_images, 
                    target_filename="img2.png",
                    page=page,
                    target_control='c2',
                    image_display_c1=image_display_c1,
                    image_display_c2=image_display_c2
                )
                if img2_rel_path:
                    processed_image_paths_for_toml.append(img2_rel_path)
        
        # Build the final YAML dictionary
        yaml_dict = {
            'model': {
                'model_source': config.get('Model Source'),
                'training_mode': config.get('Training Mode'),
                'load_checkpoint': config.get('Last Checkpoint') or None,
                'blocks_to_swap': blocks_to_swap_final,
            },
            'lora': {
                'rank': int(config.get('Rank', 128)),
                'alpha': int(config.get('Alpha', 128)),
                'dropout': float(config.get('Dropout', 0.0)),
                'target_modules': [s.strip() for s in (config.get('Target Modules') or '').split(',') if s.strip()],
            },
            'optimization': {
                'learning_rate': float(config.get('Learning Rate', 2e-4)),
                'steps': int(config.get('Steps', 2000)),
                'batch_size': int(config.get('Batch Size', 1)),
                'gradient_accumulation_steps': int(config.get('Gradient Accumulation Steps', 1)),
                'max_grad_norm': float(config.get('Max Grad Norm', 1.0)),
                'optimizer_type': config.get('Optimizer Type'),
                'scheduler_type': config.get('Scheduler Type'),
                'scheduler_params': {},
                'enable_gradient_checkpointing': bool(config.get('Gradient Checkpointing', True)),
                'first_frame_conditioning_p': float(config.get('First Frame Conditioning P', 0.5)),
            },
            'acceleration': {
                'mixed_precision_mode': config.get('Mixed Precision Mode'),
                'quantization': config.get('Quantization'),
                'load_text_encoder_in_8bit': bool(config.get('Load Text Encoder in 8bit', True)),
                'compile_with_inductor': bool(config.get('Compile with Inductor', False)),
                'compilation_mode': config.get('Compilation Mode'),
            },
            'data': {
                'preprocessed_data_root': dataset_path,
                'num_dataloader_workers': int(num_workers),
            },
            'validation': {
                'prompts': prompt_list,
                'negative_prompt': sampling.get('Negative Prompt', ''),
                'video_dims': video_dims,
                'seed': int(sampling.get('Seed (Validation)', 42)),
                'inference_steps': int(sampling.get('Inference Steps', 50)),
                'interval': interval_for_yaml,
                'videos_per_prompt': int(sampling.get('Videos Per Prompt', 1)),
                'guidance_scale': float(sampling.get('Guidance Scale', 3.5)),
                'images': [img for img in processed_image_paths_for_toml if img] if processed_image_paths_for_toml else None
            },
            'checkpoints': {
                'interval': int(config.get('Interval (Checkpoints)', 250)),
                'keep_last_n': int(config.get('Keep Last N', -1)),
            },
            'seed': int(config.get('Seed (General)', 42)),
            'output_dir': config.get('Output Directory'),
            'misc': {
                'sampling_enabled': sampling_checkbox,
                'match_enabled': match_checkbox,
            }
        }
        prompts_input = sampling.get('Prompts') or ''
        # Split by lines and remove empty lines and strip whitespace from each line
        # This assumes each line in the textfield is a separate prompt.
        prompt_list = [p.strip() for p in prompts_input.splitlines() if p.strip()]
        yaml_dict['validation']['prompts'] = prompt_list

        # New logic for handling two validation images from training_sampling page
        processed_image_paths_for_toml = []
        selected_dataset_name_for_images = None
        if dataset_controls and hasattr(dataset_controls, 'get_selected_dataset'):
            selected_dataset_name_for_images = dataset_controls.get_selected_dataset()
        
        video_dims_str = sampling.get('Video Dims', '[512, 512, 49]')
        video_dims_tuple_for_scaling = (512, 512) # Default value

        try:
            cleaned_dims_str = video_dims_str.strip()
            if cleaned_dims_str.startswith('[') and cleaned_dims_str.endswith(']'):
                cleaned_dims_str = cleaned_dims_str[1:-1]
            
            parts = [p.strip() for p in cleaned_dims_str.split(',')]
            
            if len(parts) >= 2:
                width = int(parts[0])
                height = int(parts[1])
                video_dims_tuple_for_scaling = (width, height)
            else:
                raise ValueError("Video Dims format requires at least width and height (e.g., '512, 512' or '[512, 512, 49]').")
        except Exception as e:
            print(f"Error parsing Video Dims '{video_dims_str}': {e}. Using default 512x512 for scaling.")
            # Default is already set to (512, 512)

        if selected_dataset_name_for_images:
            # Get the page object from the training tab container if available
            page = getattr(training_tab_container, 'page', None)
            
            if current_selected_image_path_c1:
                img1_rel_path = TopBarUtils._save_and_scale_image(
                    source_image_path=current_selected_image_path_c1, 
                    video_dims_tuple=video_dims_tuple_for_scaling, 
                    dataset_name=selected_dataset_name_for_images, 
                    target_filename="img1.png",
                    page=page,
                    target_control='c1',
                    image_display_c1=image_display_c1,
                    image_display_c2=image_display_c2
                )
                if img1_rel_path:
                    processed_image_paths_for_toml.append(img1_rel_path)
            
            if current_selected_image_path_c2:
                img2_rel_path = TopBarUtils._save_and_scale_image(
                    source_image_path=current_selected_image_path_c2, 
                    video_dims_tuple=video_dims_tuple_for_scaling, 
                    dataset_name=selected_dataset_name_for_images, 
                    target_filename="img2.png",
                    page=page,
                    target_control='c2',
                    image_display_c1=image_display_c1,
                    image_display_c2=image_display_c2
                )
                if img2_rel_path:
                    processed_image_paths_for_toml.append(img2_rel_path)

        # Assign images to prompts based on specified logic
        final_image_list_for_toml = None
        num_prompts = len(prompt_list) # prompt_list is already defined earlier
        num_available_images = len(processed_image_paths_for_toml)

        if num_prompts > 0 and num_available_images > 0:
            final_image_list_for_toml = []
            if num_prompts == 1:
                final_image_list_for_toml.append(processed_image_paths_for_toml[0])
            else: # num_prompts > 1
                if num_available_images == 1:
                    for _ in range(num_prompts):
                        final_image_list_for_toml.append(processed_image_paths_for_toml[0])
                else: # num_available_images > 1 (typically 2)
                    for i in range(num_prompts):
                        final_image_list_for_toml.append(processed_image_paths_for_toml[i % num_available_images])
        elif num_prompts == 0 and num_available_images > 0:
             # If no prompts but images are present, one could decide to add the first image
             # For now, sticking to None if no prompts, as per original logic for single image
             final_image_list_for_toml = None 
        
        yaml_dict['validation']['images'] = final_image_list_for_toml

        return yaml_dict

    @staticmethod
    def set_yaml_path_and_title(page, path, set_as_current=True):
        if set_as_current:
            page.y_name = path
        filename = os.path.basename(path) if path else None
        # Do not change title if config_default.yaml is opened
        if filename and filename.lower() == "config_default.yaml":
            page.title = "LTX Trainer"
        elif filename:
            page.title = f"LTX Trainer - {filename}"
        else:
            page.title = "LTX Trainer"
        page.update()

    @staticmethod
    def handle_save(page: ft.Page):
        path = getattr(page, 'y_name', None)
        is_default_loaded = getattr(page, 'is_default_config_loaded', False)

        if is_default_loaded: # If UI shows default, always Save As
            TopBarUtils.handle_save_as(page)
            return
        
        # If the current config is the default config, force Save As (this is a fallback)
        if path and os.path.basename(path).lower() == "config_default.yaml":
            TopBarUtils.handle_save_as(page)
            return
        
        if path:
            if not path.lower().endswith(('.yaml', '.yml')):
                path += '.yaml'
            training_tab = getattr(page, 'training_tab_container', None)
            if not training_tab:
                return
            
            # Retrieve image paths and display controls from the page object
            current_c1 = getattr(page, 'selected_image_path_c1', None)
            current_c2 = getattr(page, 'selected_image_path_c2', None)
            image_display_c1 = getattr(page, 'image_display_c1', None)
            image_display_c2 = getattr(page, 'image_display_c2', None)
            
            yaml_dict = TopBarUtils.build_yaml_config_from_ui(
                training_tab, 
                current_selected_image_path_c1=current_c1, 
                current_selected_image_path_c2=current_c2,
                image_display_c1=image_display_c1,
                image_display_c2=image_display_c2
            )

            with open(path, 'w', encoding='utf-8') as f:
                yaml.dump(yaml_dict, f, sort_keys=False, allow_unicode=True, Dumper=yaml.SafeDumper)
            TopBarUtils.set_yaml_path_and_title(page, path)
            TopBarUtils.add_recent_file(path, page)
            page.is_default_config_loaded = False # Reset flag after successful save
        else:
            TopBarUtils.handle_save_as(page)

    @staticmethod
    def handle_save_as(page: ft.Page):
        file_picker = ft.FilePicker()
        page.overlay.append(file_picker)
        page.update()
        def on_save_result(e):
            if e.path:
                path = e.path
                if not path.lower().endswith(('.yaml', '.yml')):
                    path += '.yaml'
                training_tab = getattr(page, 'training_tab_container', None)
                if not training_tab:
                    return

                # Retrieve image paths and display controls from the page object
                current_c1 = getattr(page, 'selected_image_path_c1', None)
                current_c2 = getattr(page, 'selected_image_path_c2', None)
                image_display_c1 = getattr(page, 'image_display_c1', None)
                image_display_c2 = getattr(page, 'image_display_c2', None)
                
                yaml_dict = TopBarUtils.build_yaml_config_from_ui(
                    training_tab, 
                    current_selected_image_path_c1=current_c1, 
                    current_selected_image_path_c2=current_c2,
                    image_display_c1=image_display_c1,
                    image_display_c2=image_display_c2
                )

                with open(path, 'w', encoding='utf-8') as f:
                    yaml.dump(yaml_dict, f, sort_keys=False, allow_unicode=True, Dumper=yaml.SafeDumper)
                TopBarUtils.set_yaml_path_and_title(page, path)
                TopBarUtils.add_recent_file(path, page)
                page.is_default_config_loaded = False # Reset flag after successful save as
        file_picker.on_result = on_save_result
        default_name = "ltxv_config.yaml"
        default_dir = os.path.abspath(os.path.join(os.getcwd(), "workspace", "configs"))
        file_picker.save_file(
            dialog_title="Save config as YAML",
            file_name=default_name,
            initial_directory=default_dir,
            allowed_extensions=["yaml", "yml"]
        )

    @staticmethod
    def handle_open(page: ft.Page, file_path=None, set_as_current=True):
        if file_path:
            # Open the file directly
            path = file_path
            with open(path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            training_tab = getattr(page, 'training_tab_container', None)
            if not training_tab:
                return
            TopBarUtils.update_ui_from_yaml(training_tab, config_data)
            TopBarUtils.set_yaml_path_and_title(page, path, set_as_current=set_as_current)
            TopBarUtils.add_recent_file(path, page)
            if os.path.basename(path).lower() != "config_default.yaml" and set_as_current:
                page.is_default_config_loaded = False
            elif os.path.basename(path).lower() == "config_default.yaml":
                 page.is_default_config_loaded = True # Explicitly set for default load via this path
            return
        # Original file picker dialog
        file_picker = ft.FilePicker()
        page.overlay.append(file_picker)
        page.file_picker_open_ref = file_picker
        page.update()
        def on_open_result(e):
            if e.files and len(e.files) > 0:
                path = e.files[0].path
                with open(path, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
                training_tab = getattr(page, 'training_tab_container', None)
                if not training_tab:
                    return
                TopBarUtils.update_ui_from_yaml(training_tab, config_data)
                TopBarUtils.set_yaml_path_and_title(page, path, set_as_current=set_as_current)
                TopBarUtils.add_recent_file(path, page)
                if os.path.basename(path).lower() != "config_default.yaml" and set_as_current:
                    page.is_default_config_loaded = False
        file_picker.on_result = on_open_result
        default_dir = os.path.abspath(os.path.join(os.getcwd(), "workspace", "configs"))
        file_picker.pick_files(
            dialog_title="Open config YAML",
            initial_directory=default_dir,
            allowed_extensions=["yaml", "yml"],
            allow_multiple=False
        )

    @staticmethod
    def update_ui_from_yaml(training_tab_container, config_data):
        # First, handle validation images if they exist
        validation_config = config_data.get('validation', {})
        if isinstance(validation_config, dict):
            validation_images = validation_config.get('images', [])
            if isinstance(validation_images, list) and validation_images:
                # Get the page object from the training tab container
                page = getattr(training_tab_container, 'page', None)
                if page:
                    # Update image display controls if they exist
                    for i, img_path in enumerate(validation_images[:2]):  # Only handle first 2 images
                        if not img_path or not os.path.exists(img_path):
                            logger.warning(f"Image not found at path: {img_path}")
                            continue
                        target_control = f'c{i+1}'
                        try:
                            # Get the image display control from the page
                            image_display = getattr(page, f'image_display_{target_control}', None)
                            if image_display:
                                # Update the image source
                                img_path = img_path.replace('\\', '/')  # Ensure forward slashes
                                image_display.src = img_path
                                try:
                                    image_display.update()
                                except :
                                    pass
                                # Update the selected image path
                                setattr(page, f'selected_image_path_{target_control}', img_path)
                                logger.info(f"Loaded validation image {i+1} from config: {img_path}")
                        except Exception as e:
                            logger.error(f"Error loading validation image {img_path}: {e}")
                            import traceback
                            traceback.print_exc()

        YAML_TO_LABEL = {
            "model_source": "Model Source",
            "training_mode": "Training Mode",
            "load_checkpoint": "Last Checkpoint",
            "blocks_to_swap": "Block to swap",
            "rank": "Rank",
            "alpha": "Alpha",
            "dropout": "Dropout",
            "target_modules": "Target Modules",
            "learning_rate": "Learning Rate",
            "steps": "Steps",
            "batch_size": "Batch Size",
            "gradient_accumulation_steps": "Gradient Accumulation Steps",
            "max_grad_norm": "Max Grad Norm",
            "optimizer_type": "Optimizer Type",
            "scheduler_type": "Scheduler Type",
            "scheduler_params": "Scheduler Params",
            "enable_gradient_checkpointing": "Gradient Checkpointing",
            "first_frame_conditioning_p": "First Frame Conditioning P",
            "mixed_precision_mode": "Mixed Precision Mode",
            "quantization": "Quantization",
            "load_text_encoder_in_8bit": "Load Text Encoder in 8bit",
            "compile_with_inductor": "Compile with Inductor",
            "compilation_mode": "Compilation Mode",
            "preprocessed_data_root": "Preprocessed Data Root",
            "num_dataloader_workers": "Num Dataloader Workers",
            "prompts": "Prompts",
            "negative_prompt": "Negative Prompt",
            "video_dims": "Video Dims",
            "seed": "Seed (Validation)",
            "inference_steps": "Inference Steps",
            "interval": "Interval (Validation)",
            "videos_per_prompt": "Videos Per Prompt",
            "guidance_scale": "Guidance Scale",
            "interval_checkpoints": "Interval (Checkpoints)",
            "keep_last_n": "Keep Last N",
            "timestep_sampling_mode": "Timestep Sampling Mode",
            "timestep_sampling_params": "Timestep Sampling Params",
            "seed_general": "Seed (General)",
            "output_dir": "Output Directory",
        }
        def flatten_yaml(cfg, parent_key=""):
            flat = {}
            for k, v in cfg.items():
                key = k
                if parent_key == "validation" and k == "seed":
                    key = "seed"
                elif parent_key == "checkpoints" and k == "interval":
                    key = "interval_checkpoints"
                elif parent_key == "seed":
                    key = "seed_general"
                # Handle misc section keys
                elif parent_key == "misc" and k == "sampling_enabled":
                    key = "Sampling" # Map to UI control label
                elif parent_key == "misc" and k == "match_enabled":
                    key = "Match" # Map to UI control label
                if isinstance(v, dict):
                    flat.update(flatten_yaml(v, k))
                else:
                    flat[key] = v
            return flat
        flat = flatten_yaml(config_data)
        def update_controls(control):
            if hasattr(control, 'controls') and control.controls:
                for child in control.controls:
                    update_controls(child)
            if hasattr(control, 'content') and control.content:
                update_controls(control.content)
            label = getattr(control, 'label', None)
            yaml_key = None
            for k, v in YAML_TO_LABEL.items():
                if v == label:
                    yaml_key = k
                    break
            # Check if the label matches the keys from the misc section
            if label == "Sampling" and "Sampling" in flat:
                 val = flat["Sampling"]
                 if isinstance(control, ft.Checkbox):
                    control.value = bool(val)
            elif label == "Match" and "Match" in flat:
                 val = flat["Match"]
                 if isinstance(control, ft.Checkbox):
                    control.value = bool(val)
            elif label == "Prompts" and yaml_key and yaml_key in flat:
                val = flat[yaml_key]
                if isinstance(control, ft.TextField):
                     if isinstance(val, list):
                        # Join prompt list with newlines, as each line is a separate prompt
                        control.value = '\n'.join(str(x) for x in val)
                     else:
                        # If it's not a list (e.g., single prompt string), set directly
                        control.value = str(val) if val is not None else ""
            elif yaml_key and yaml_key in flat:
                val = flat[yaml_key]
                # If loading interval and it's None, set UI to '0'
                if yaml_key == "interval" and val is None:
                    val = "0"
                if isinstance(control, ft.TextField):
                    # Keep original logic for other text fields (joining lists with comma space)
                    if isinstance(val, list):
                        control.value = ', '.join(str(x) for x in val)
                    else:
                        control.value = str(val) if val is not None else ""
        update_controls(training_tab_container.config_page_content)
        update_controls(training_tab_container.sampling_page_content)

        # --- Handle Dataset Page Content ---
        dataset_page_content = getattr(training_tab_container, 'dataset_page_content', None)
        if dataset_page_content:
            page_ctx = training_tab_container.page

            num_workers_yaml = flat.get("num_dataloader_workers")
            if num_workers_yaml is not None and hasattr(dataset_page_content, 'set_num_workers'):
                try:
                    # Only call if the num_workers_field is attached to the page
                    num_workers_field = getattr(dataset_page_content, 'num_workers_field_ref', None)
                    if num_workers_field and hasattr(num_workers_field, 'current') and getattr(num_workers_field.current, 'page', None):
                        dataset_page_content.set_num_workers(num_workers_yaml, page_ctx)
                except Exception as e:
                    pass
            
            dataset_root_yaml = flat.get("preprocessed_data_root")
            if (dataset_root_yaml is None or str(dataset_root_yaml).lower() == "null") and hasattr(dataset_page_content, 'set_selected_dataset'):
                try:
                    dataset_page_content.set_selected_dataset(None, page_ctx)
                except Exception as e:
                    pass
            elif dataset_root_yaml and hasattr(dataset_page_content, 'set_selected_dataset'):
                try:
                    dataset_path = str(dataset_root_yaml)
                    if dataset_path.endswith('preprocessed_data'):
                        dataset_folder_name = os.path.basename(os.path.dirname(dataset_path))
                    else:
                        dataset_folder_name = os.path.basename(dataset_path)
                    dataset_page_content.set_selected_dataset(dataset_folder_name, page_ctx)
                except Exception as e:
                    pass

        # --- Original Update Logic for other tabs (condensed for brevity) ---
        parent = training_tab_container.content.controls[0]
        if hasattr(parent, "controls"):
            content_area = parent.controls[2]
        elif hasattr(parent, "content"):
            content_area = parent.content
        else:
            content_area = parent  # fallback

        visible_content = getattr(content_area, "content", content_area)
        if visible_content is training_tab_container.config_page_content:
            training_tab_container.config_page_content.update()
        elif visible_content is training_tab_container.sampling_page_content:
            training_tab_container.sampling_page_content.update()
        
        training_tab_container.update()
        content_area.update()

    @staticmethod
    def handle_load_default(page: ft.Page):
        default_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'assets', 'config_default.yaml')
        TopBarUtils.handle_open(page, file_path=default_path, set_as_current=False)
        page.is_default_config_loaded = True # Set flag when default is loaded

        # Directly set dataset dropdown to None after loading defaults
        training_tab = getattr(page, 'training_tab_container', None)
        if training_tab:
            dataset_page_content = getattr(training_tab, 'dataset_page_content', None)
            if dataset_page_content and hasattr(dataset_page_content, 'set_selected_dataset'):
                try:
                    dataset_page_content.set_selected_dataset(None, page)
                except Exception as e:
                    pass
        else:
            pass

    @staticmethod
    def get_recent_files_path():
        return os.path.join(os.path.dirname(__file__), 'recent_files.txt')

    @staticmethod
    def load_recent_files():
        path = TopBarUtils.get_recent_files_path()
        if not os.path.exists(path):
            return []
        with open(path, 'r', encoding='utf-8') as f:
            files = [line.strip() for line in f if line.strip()]
        return files

    @staticmethod
    def save_recent_files(files):
        path = TopBarUtils.get_recent_files_path()
        with open(path, 'w', encoding='utf-8') as f:
            for file in files[:3]:
                f.write(file + '\n')

    @staticmethod
    def add_recent_file(filepath, page=None):
        files = TopBarUtils.load_recent_files()
        if filepath in files:
            files.remove(filepath)
        files.insert(0, filepath)
        TopBarUtils.save_recent_files(files)
        if page and hasattr(page, 'refresh_menu_bar'):
            page.refresh_menu_bar()

    @staticmethod
    def get_recent_files_menu_items(on_click, text_size=10):
        files = TopBarUtils.load_recent_files()
        if not files:
            return [ft.MenuItemButton(content=ft.Text("none", size=text_size), on_click=on_click)]
        items = []
        for f in files:
            short = os.path.join(os.path.basename(os.path.dirname(f)), os.path.basename(f))
            items.append(ft.MenuItemButton(content=ft.Text(short, size=text_size), on_click=on_click, data=f))
        return items
