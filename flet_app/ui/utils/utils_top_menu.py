import os
import yaml
import flet as ft
import re

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
    def build_yaml_config_from_ui(training_tab_container):
        config_controls = training_tab_container.config_page_content
        sampling_controls = training_tab_container.sampling_page_content
        dataset_controls = getattr(training_tab_container, 'dataset_page_content', None)
        config = TopBarUtils.extract_config_from_controls(config_controls)
        sampling = TopBarUtils.extract_config_from_controls(sampling_controls)
        # Get dataset and num_workers from dataset page
        dataset_path = None
        num_workers = 2  # Default value
        if dataset_controls and hasattr(dataset_controls, 'get_selected_dataset'):
            selected_dataset_value = dataset_controls.get_selected_dataset()
            if selected_dataset_value:
                # Ensure forward slashes for consistency - Corrected escaping
                dataset_path = os.path.join('workspace', 'datasets', selected_dataset_value, 'preprocessed_data').replace('\\', '/')
        if dataset_controls and hasattr(dataset_controls, 'get_num_workers'):
            try:
                num_workers = int(dataset_controls.get_num_workers())
            except (ValueError, TypeError):
                num_workers = 2 # Fallback to default if conversion fails
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
        # --- Write null to YAML if interval is 0 ---
        interval_for_yaml = None if interval_validation_final == 0 else interval_validation_final
        # --- Logic for blocks_to_swap ---
        blocks_to_swap_value_str = config.get('Block to swap')
        blocks_to_swap_final = None # Default to None if not set or invalid
        if blocks_to_swap_value_str is not None and str(blocks_to_swap_value_str).strip() != "":
            try:
                blocks_to_swap_final = int(blocks_to_swap_value_str)
                if blocks_to_swap_final < 0:
                    blocks_to_swap_final = 0
            except ValueError:
                blocks_to_swap_final = None
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
                'prompts': [],
                'negative_prompt': sampling.get('Negative Prompt'),
                'video_dims': eval(sampling.get('Video Dims', '[512, 512, 49]')),
                'seed': int(sampling.get('Seed (Validation)', 42)),
                'inference_steps': int(sampling.get('Inference Steps', 50)),
                'interval': interval_for_yaml,
                'videos_per_prompt': int(sampling.get('Videos Per Prompt', 1)),
                'guidance_scale': float(sampling.get('Guidance Scale', 3.5)),
            },
            'checkpoints': {
                'interval': int(config.get('Interval (Checkpoints)', 250)),
                'keep_last_n': int(config.get('Keep Last N', -1)),
            },
            'flow_matching': {
                'timestep_sampling_mode': config.get('Timestep Sampling Mode'),
                'timestep_sampling_params': {},
            },
            'seed': int(config.get('Seed (General)', 42)),
            'output_dir': config.get('Output Directory'),
            'misc': {
                'sampling_enabled': sampling_checkbox,
                'match_enabled': match_checkbox,
            }
        }
        prompt_list = []
        prompts_input = sampling.get('Prompts') or ''

        # --- Revised logic to split prompts by commas and newlines, respecting quotes ---
        # This regex finds segments which are either:
        # 1. A quoted string: \"...\" allowing escaped quotes and other escaped characters inside.
        #    \"(?:[^\"\\\\]|\\.)*\"  - Corrected escaping of backslash inside character set
        # 2. Or, a sequence of characters that are not commas or newlines.
        #    [^,\\n]+\
        # It also handles potential whitespace and empty matches by filtering later.
        matches = re.findall(r'\"(?:[^\"\\\\]|\\.)*\"|[^,\\n]+', prompts_input)

        # Process the matches to strip whitespace and remove external quotes
        prompt_list = []
        for match in matches:
            cleaned_prompt = match.strip()
            if cleaned_prompt:
                # If it\'s a quoted string from the regex match, remove the outer quotes
                if cleaned_prompt.startswith('\"') and cleaned_prompt.endswith('\"'):
                    cleaned_prompt = cleaned_prompt[1:-1]
                    # Unescape escaped quotes and backslashes within the cleaned string
                    cleaned_prompt = cleaned_prompt.replace('\\\"', '\"').replace('\\\\', '\\\\')
                prompt_list.append(cleaned_prompt)
        # --- End of Revised logic ---\

        yaml_dict['validation']['prompts'] = prompt_list

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
            yaml_dict = TopBarUtils.build_yaml_config_from_ui(training_tab)
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
                yaml_dict = TopBarUtils.build_yaml_config_from_ui(training_tab)
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
                        # Join prompt list with quotes, commas, and newlines for multiline TextField
                        quoted_prompts = [f'"{str(x)}"'+ ',' for x in val] # Add comma after each quoted prompt
                        # Remove the last comma before joining with newline
                        if quoted_prompts:
                            quoted_prompts[-1] = quoted_prompts[-1].rstrip(',')
                        control.value = '\n'.join(quoted_prompts)
                     else:
                        # If it's not a list (e.g., single prompt string), set directly, potentially quoting it
                        if isinstance(val, str):
                             # If a single string, put quotes around it
                            control.value = f'"{val}"'
                        else:
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
