import json
import os


class Config:
    _instance = None
    _settings = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_settings()
        return cls._instance

    def _load_settings(self):
        # Construct the path to settings.json relative to the current file
        settings_file_path = os.path.join(os.path.dirname(__file__), 'settings.json')
        try:
            with open(settings_file_path, 'r') as f:
                self._settings = json.load(f)
            self._post_process_settings()
        except FileNotFoundError:
            print(f"Error: settings.json not found at {settings_file_path}")
            # Depending on your application's needs, you might want to:
            # 1. Load default settings from a hardcoded dictionary
            # 2. Raise an exception to stop the application
            # 3. Log the error and continue with an empty config (might lead to further errors)
            # For now, we'll just print an error.
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {settings_file_path}. Check for syntax errors.")
            # Handle malformed JSON

    def _post_process_settings(self):
        # Reconstruct derived values
        self._settings['ltx_model_dict'] = {model: model for model in self._settings['ltx_models']}
        
        # Calculate TARGET_ASPECT_RATIO dynamically
        if 'THUMB_TARGET_W' in self._settings and 'THUMB_TARGET_H' in self._settings:
            self._settings['TARGET_ASPECT_RATIO'] = self._settings['THUMB_TARGET_W'] / self._settings['THUMB_TARGET_H']
        else:
            self._settings['TARGET_ASPECT_RATIO'] = None # Or a default value

        # Combine media extensions
        self._settings['MEDIA_EXTENSIONS'] = self._settings['VIDEO_EXTENSIONS'] + self._settings['IMAGE_EXTENSIONS']

        # Ensure paths are OS-agnostic by splitting and rejoining
        # This assumes paths in JSON use forward slashes as separators
        for key in ['DATASETS_DIR', 'THUMBNAILS_BASE_DIR', 'LORA_MODELS_DIR', 'FFMPEG_PATH']:
            if key in self._settings and isinstance(self._settings[key], str):
                self._settings[key] = os.path.join(*self._settings[key].split('/'))

    def __getattr__(self, name):
        if name in self._settings:
            return self._settings[name]
        # If the attribute is not found in _settings, raise an AttributeError
        # This is standard Python behavior for missing attributes
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")


# Instantiate the Config class to make it accessible globally
# Other modules can now import and use 'settings' directly:
# from .settings import settings
# print(settings.THUMB_TARGET_W)
settings = Config()