import flet as ft
# import yaml # Removed as hardcoded config data is reduced
from .._styles import create_textfield, create_dropdown, add_section_title # Import helper functions
from .training_dataset_block import get_training_dataset_page_content


def get_training_config_page_content():
    """Generates Flet controls with hardcoded configuration values, grouped by section."""

    page_controls = []

    # --- Model Configuration & Dataset Selection (Side by Side) ---
    dataset_block = get_training_dataset_page_content()
    page_controls.append(
        ft.ResponsiveRow([
            ft.Column([
                *add_section_title("Model Configuration"),
                ft.Column([
                    ft.ResponsiveRow(controls=[
                        create_dropdown(
                            "Model Source",
                            "LTXV_13B_097_DEV",
                            {
                                "LTXV_13B_097_DEV": "LTXV_13B_097_DEV",
                                "LTXV_13B_097_DEV_FP8": "LTXV_13B_097_DEV_FP8",
                                "LTXV_13B_097_DISTILLED": "LTXV_13B_097_DISTILLED",
                                "LTXV_13B_097_DISTILLED_FP8": "LTXV_13B_097_DISTILLED_FP8",
                                "LTXV_2B_0.9.6_DEV": "LTXV_2B_0.9.6_DEV",
                                "LTXV_2B_0.9.5": "LTXV_2B_0.9.5",
                                "LTXV_2B_0.9.1": "LTXV_2B_0.9.1",
                                "LTXV_2B_0.9.0": "LTXV_2B_0.9.0"
                            },hint_text="Select model or specify path below",col=8, expand=True
                        ),
                        create_dropdown(
                            "Training Mode",
                            "lora",
                            {
                                "lora": "lora",
                                "full": "full"
                            },col=4, expand=True
                        ),
                    ]),
                    create_textfield("Output Directory", "workspace/output/cakeify_lora_13b", col=3, expand=True),
                    ft.ResponsiveRow(controls=[
                        create_textfield("Block to swap", 0, col=3,  hint_text="Max 47 , need at least 1 in vram",expand=True),
                        create_textfield("Custom Model Path (if not in dropdown)", None, hint_text="HF repo or local path", col=9, expand=True),
                    ]),
                    ft.ResponsiveRow(controls=[
                        create_textfield("Seed (General)", 42, col=3, expand=True),
                        create_textfield("Last Checkpoint", None, hint_text="Path to file/directory to resume from. If directory, latest checkpoint will be used", col=9, expand=True),
                    ]),
                    ft.ResponsiveRow(controls=[
                        ft.Checkbox(label="Sampling", value=True,col=3,scale=0.8, tooltip="Enable sampling for training, change settings in Sampling tab"),
                        ft.Checkbox(label="Match", value=True,col=3,scale=0.8, tooltip="Override settings in Sampling using Interval (Checkpoints)"),
                        create_textfield("Interval (Checkpoints)", 250, hint_text="Save every N steps, or null",col=3.5, expand=True, tooltip="Save a checkpoint every N steps, set to null to disable"),
                        create_textfield("Keep Last N", -1, hint_text="-1 to keep all", col=2.5, expand=True)
                    ]),
                ])
            ], col=5), # Adjusted col for Model Configuration
            ft.Column([
                *add_section_title("Dataset Selection"),
                dataset_block,
            ], col=7), # Adjusted col for Dataset Selection
        ], spacing=20)
    )
    page_controls.append(ft.Divider(height=5, color=ft.Colors.TRANSPARENT))

    # --- LoRA Configuration ---
    page_controls.extend(add_section_title("LoRA Configuration")) # Use extend
    page_controls.append(ft.ResponsiveRow(controls=[
        create_textfield("Rank", 128, col=2, expand=True),
        create_textfield("Alpha", 128, col=2, expand=True),
        create_textfield("Dropout", 0.0, col=2, expand=True ),
        create_textfield("Target Modules", "to_k, to_q, to_v, to_out.0", hint_text="Comma-separated list", col=6, expand=True)
    ]))

    page_controls.append(ft.Divider(height=5, color=ft.Colors.TRANSPARENT))

    # --- Optimization Configuration ---
    page_controls.extend(add_section_title("Optimization Configuration")) # Use extend
    page_controls.append(ft.ResponsiveRow(controls=[
        create_textfield("Learning Rate", 2e-4, col=2, expand=True),
        create_textfield("Steps", 2000, col=2, expand=True),
        create_textfield("Batch Size", 1, col=2, expand=True),
        create_textfield("Gradient Accumulation Steps", 1, col=2, expand=True),
        create_textfield("Max Grad Norm", 1.0, col=3, expand=True)
    ]))
    page_controls.append(ft.ResponsiveRow(controls=[
        create_dropdown(
            "Optimizer Type",
            "adamw",
            {
                "adamw": "adamw",
                "adamw8bit": "adamw8bit"
            }, col=2, expand=True
        ),
        create_dropdown(
            "Scheduler Type",
            "linear",
            {
                "constant": "constant",
                "linear": "linear",
                "cosine": "cosine",
                "cosine_with_restarts": "cosine_with_restarts",
                "polynomial": "polynomial"
            }, col=2, expand=True
        ),
        create_textfield("Scheduler Params", "{}", col=2, expand=True),
        create_textfield("First Frame Conditioning P", 0.5, col=2, expand=True),
        ft.Checkbox(label="Gradient Checkpointing", value=True,col=3,scale=0.8),
    ]))
    page_controls.append(ft.Divider(height=5, color=ft.Colors.TRANSPARENT))


    # --- Acceleration Optimization & Flow Matching Configuration (Side by Side) ---
    page_controls.append(
        ft.ResponsiveRow([
            ft.Column([
                *add_section_title("Acceleration Optimization"),
                ft.ResponsiveRow(controls=[
                    create_dropdown(
                        "Mixed Precision Mode",
                        "bf16",
                        {
                            "no": "no",
                            "fp16": "fp16",
                            "bf16": "bf16"
                        }, col=3,expand=True
                    ),
                    create_dropdown(
                        "Quantization",
                        "fp8-quanto",
                        {
                            "no_change": "no_change",
                            "int8-quanto": "int8-quanto",
                            "int4-quanto": "int4-quanto",
                            "int2-quanto": "int2-quanto",
                            "fp8-quanto": "fp8-quanto",
                            "fp8uz-quanto": "fp8uz-quanto"
                        }, col=4,expand=True
                    ),
                    create_dropdown(
                        "Compilation Mode",
                        "reduce-overhead",
                        {
                            "default": "default",
                            "reduce-overhead": "reduce-overhead",
                            "max-autotune": "max-autotune"
                        }, col=5,expand=True
                    ),
                ]),
                ft.ResponsiveRow(controls=[
                    ft.Checkbox(label="Load Text Encoder in 8bit", value=True, col=6,scale=0.8,tooltip="# Load text encoder in 8-bit precision to save memory"),
                    ft.Checkbox(label="Compile with Inductor", value=False, col=6,scale=0.8),
                ])
            ], col=7),
            ft.Column([
                *add_section_title("Flow Matching Configuration"),
                ft.ResponsiveRow(controls=[
                    create_dropdown(
                        "Timestep Sampling Mode",
                        "shifted_logit_normal",
                        {
                            "uniform": "uniform",
                            "shifted_logit_normal": "shifted_logit_normal"
                        }, col=6
                    ),
                    create_textfield("Timestep Sampling Params", "{}", col=6)
                ]),
            ], col=5),
        ], spacing=20)
    )
    page_controls.append(ft.Divider(height=5, color=ft.Colors.TRANSPARENT))

    container = ft.Container(
        content=ft.Column(
            controls=page_controls,
            spacing=8, # Slightly reduced spacing between controls/sections
            scroll=ft.ScrollMode.AUTO,
        ),
        expand=True, # Allow container to take full height
        padding=ft.padding.all(5)
    )
    container.dataset_block = dataset_block
    return container
