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
        if 'ltx_models' in self._settings: # Check if key exists
            self._settings['ltx_model_dict'] = {model: model for model in self._settings['ltx_models']}
        
        # Calculate TARGET_ASPECT_RATIO dynamically
        if 'THUMB_TARGET_W' in self._settings and 'THUMB_TARGET_H' in self._settings and \
           self._settings['THUMB_TARGET_H'] != 0: # Avoid division by zero
            self._settings['TARGET_ASPECT_RATIO'] = self._settings['THUMB_TARGET_W'] / self._settings['THUMB_TARGET_H']
        else:
            self._settings['TARGET_ASPECT_RATIO'] = 16/9 # Default aspect ratio

        # Combine media extensions
        if 'VIDEO_EXTENSIONS' in self._settings and 'IMAGE_EXTENSIONS' in self._settings:
            self._settings['MEDIA_EXTENSIONS'] = self._settings['VIDEO_EXTENSIONS'] + self._settings['IMAGE_EXTENSIONS']
        else:
            self._settings['MEDIA_EXTENSIONS'] = [] # Default to empty list

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        paths_to_absolutize = ['DATASETS_DIR', 'DATASETS_IMG_DIR', 'THUMBNAILS_BASE_DIR', 'LORA_MODELS_DIR', 'THUMBNAILS_IMG_BASE_DIR']
        
        for key in paths_to_absolutize:
            if key in self._settings and isinstance(self._settings[key], str):
                path_val = str(self._settings[key]) # Ensure it's a string
                # Normalize separators first (e.g. "path/to/something" -> "path\to\something" on Windows)
                normalized_path_val = os.path.join(*path_val.split('/'))
                if not os.path.isabs(normalized_path_val):
                    normalized_path_val = os.path.join(project_root, normalized_path_val)
                self._settings[key] = os.path.normpath(normalized_path_val)

        # Handle FFMPEG_PATH separately
        if 'FFMPEG_PATH' in self._settings and isinstance(self._settings['FFMPEG_PATH'], str):
            ffmpeg_path_val_raw = str(self._settings['FFMPEG_PATH'])

            if ffmpeg_path_val_raw.lower() not in ['ffmpeg', 'ffmpeg.exe']:
                # Check if the raw path from JSON is absolute
                if os.path.isabs(ffmpeg_path_val_raw):
                    # It's an absolute path, just normalize it (e.g. D:/path -> D:\\path on Win)
                    self._settings['FFMPEG_PATH'] = os.path.normpath(ffmpeg_path_val_raw)
                else:
                    # It's a relative path, join with project_root and normalize
                    # Normalize slashes in the relative path before joining
                    relative_path_normalized_slashes = os.path.join(*ffmpeg_path_val_raw.replace('\\', '/').split('/'))
                    self._settings['FFMPEG_PATH'] = os.path.normpath(os.path.join(project_root, relative_path_normalized_slashes))
                
            else:
                # It's 'ffmpeg' or 'ffmpeg.exe', ensure it's stored as such.
                self._settings['FFMPEG_PATH'] = ffmpeg_path_val_raw 

    def get(self, key, default=None):
        return self._settings.get(key, default)

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