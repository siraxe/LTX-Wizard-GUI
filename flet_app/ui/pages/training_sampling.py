import flet as ft
from .._styles import create_textfield, add_section_title

def get_training_sampling_page_content():
    """Generates Flet controls for validation and checkpoint configuration with hardcoded values."""
    page_controls = []

    page_controls.extend(add_section_title("Validation Configuration"))
    page_controls.append(ft.ResponsiveRow(controls=[
        create_textfield("Prompts",'"CAKEIFY a person using a knife to cut a cake shaped like bottle of mouthwash", "CAKEIFY a person using a knife to cut a cake shaped like potted plant", "CAKEIFY a person using a knife to cut a cake shaped like a jar of Nutella"',
                                           hint_text="One prompt per line",multiline=True, min_lines=3, max_lines=7, expand=True, col=8),
        create_textfield("Negative Prompt", "worst quality, inconsistent motion, blurry, jittery, distorted", multiline=True, min_lines=2, max_lines=5, expand=True, col=4)
    ], vertical_alignment=ft.CrossAxisAlignment.START))
    page_controls.append(ft.Row(controls=[
        create_textfield("Video Dims", "[512, 512, 49]", hint_text="width, height, frames", expand=True),
        create_textfield("Seed (Validation)", 42, expand=True),
        create_textfield("Inference Steps", 25, expand=True)
    ]))
    page_controls.append(ft.Row(controls=[
        create_textfield("Interval (Validation)", 250, hint_text="Set to null to disable validation", expand=True),
        create_textfield("Videos Per Prompt", 1, expand=True),
        create_textfield("Guidance Scale", 3.5, expand=True)
    ]))

    page_controls.append(ft.Divider(height=5, color=ft.Colors.TRANSPARENT))
    
    # --- Checkpoint Configuration ---
    page_controls.extend(add_section_title("Sample videos"))
    page_controls.append(ft.Divider(height=5, color=ft.Colors.TRANSPARENT))
    # --- Video Placeholders (Grid) ---
    video_placeholders = [
        ft.Container(
            content=ft.Column([
                ft.Icon(ft.Icons.VIDEO_LIBRARY, size=48, color=ft.Colors.GREY_400),
                ft.Text(f"Video Placeholder {i+1}", color=ft.Colors.GREY_500, size=14, italic=True),
            ], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
            width=180,
            height=120,
            bgcolor=ft.Colors.GREY_100,
            border=ft.border.all(1, ft.Colors.GREY_300),
            border_radius=8,
            alignment=ft.alignment.center,
            margin=ft.margin.all(5),
        ) for i in range(3)
    ]
    page_controls.append(
        ft.GridView(
            controls=video_placeholders,
            max_extent=200,
            child_aspect_ratio=1.5,
            spacing=10,
            run_spacing=10,
            expand=False,
            height=140
        )
    )

    return ft.Container(
        content=ft.Column(
            controls=page_controls,
            spacing=8,
            run_spacing=8,
            tight=True,
            horizontal_alignment=ft.CrossAxisAlignment.START,
            scroll=ft.ScrollMode.AUTO,
        ),
        expand=True,
        padding=ft.padding.all(5)
    )

# Add a flag to track pending save
pending_save = {'should_save': False, 'yaml_dict': None, 'out_path': None}
