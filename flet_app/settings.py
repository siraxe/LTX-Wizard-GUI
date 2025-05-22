import os
class config:
    ltx_models = [
        "LTXV_2B_0.9.0",
        "LTXV_2B_0.9.1",
        "LTXV_2B_0.9.5",
        "LTXV_2B_0.9.6_DEV",
        "LTXV_2B_0.9.6_DISTILLED",
        "LTXV_13B_097_DEV",
        "LTXV_13B_097_DEV_FP8",
        "LTXV_13B_097_DISTILLED",
        "LTXV_13B_097_DISTILLED_FP8",
    ]
    
    captions = {
        "qwen_25_vl": "Qwen2.5-VL-7B",
        "llava_next_7b": "LLaVA-NeXT-7B"
    }

    # Default models / dict
    ltx_model_dict = {model: model for model in ltx_models}
    captions_def_model = captions["llava_next_7b"]
    ltx_def_model = ltx_models[7]


    # Thumbnail settings
    THUMB_TARGET_W = 160
    THUMB_TARGET_H = 90
    TARGET_ASPECT_RATIO = THUMB_TARGET_W / THUMB_TARGET_H
    # Collage settings
    COLLAGE_WIDTH = 270
    COLLAGE_HEIGHT = 152
    THUMB_CELL_W = 91
    THUMB_CELL_H = 51

    
    # Dataset settings
    DATASETS_DIR = os.path.join("workspace", "datasets")
    THUMBNAILS_BASE_DIR = os.path.join("workspace", "thumbnails")
    DEFAULT_BUCKET_SIZE_STR = "[512, 512, 49]"
    VIDEO_EXTENSIONS = ["mp4", "avi", "mov", "mkv", "webm"]
    IMAGE_EXTENSIONS = ["jpg", "jpeg", "png"]
    MEDIA_EXTENSIONS = VIDEO_EXTENSIONS + IMAGE_EXTENSIONS
    