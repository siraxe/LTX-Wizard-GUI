import flet as ft
from flet import FilePicker, FilePickerResultEvent # Import FilePicker related classes
from ._styles import create_textfield, add_section_title, create_styled_button
import os # Import os module for path manipulation
import subprocess # Import subprocess
import asyncio # Import asyncio

# =====================
# Helper/Data/Utility Functions
# =====================

def get_abs_path(path: str) -> str:
    """Return the absolute path for a given path string."""
    return os.path.abspath(path)

def get_stem_with_comfy_suffix(input_path: str, custom_stem: str = None) -> str:
    """Generate the output filename stem, ensuring it ends with '_comfy'."""
    if custom_stem:
        stem_candidate = os.path.splitext(custom_stem)[0]
    else:
        stem_candidate = os.path.splitext(os.path.basename(input_path))[0]
    if stem_candidate.endswith("_comfy"):
        return stem_candidate
    return f"{stem_candidate}_comfy"

def build_convert_command(input_lora: str, output_dir: str, output_stem: str) -> str:
    """Build the command string to run the Lora conversion script."""
    python_exe = os.path.normpath(os.path.join("venv", "Scripts", "python.exe"))
    script_file = os.path.normpath("scripts/convert_checkpoint.py")
    output_filename = f"{output_stem}.safetensors"
    abs_full_output_path = os.path.join(output_dir, output_filename)
    return f'"{python_exe}" "{script_file}" "{input_lora}" --output-path "{abs_full_output_path}" --to-comfy'

# Function to execute the command using subprocess
async def run_script_command(command_str: str, page_ref: ft.Page, button_ref: ft.ElevatedButton, progress_bar_ref: ft.ProgressBar, output_field_ref: ft.TextField, original_button_text: str):
    """Run a shell command asynchronously and update the UI with the result."""
    print(f"--- run_script_command entered ---")
    print(f"Executing command: {command_str}")
    try:
        # Using shell=True because the command is a single string with quoted paths.
        # Ensure paths in command_str are correctly quoted if they contain spaces.
        # Run the blocking subprocess call in a separate thread
        print("Attempting to run subprocess...")
        result = await asyncio.to_thread( # Use asyncio.to_thread
            subprocess.run,
            command_str,
            shell=True,
            capture_output=True,
            text=True,
            check=False
        )
        print(f"Subprocess finished. Return code: {result.returncode}")
        if result.returncode == 0:
            output_field_ref.value = result.stdout if result.stdout else "Script successful (no stdout)."
            snack_bar_instance = ft.SnackBar(content=ft.Text(f"Script successful: {result.stdout[:50]}..."), open=True)
            page_ref.snack_bar = snack_bar_instance
            print(f"Script stdout: {result.stdout}")
        else:
            error_message = f"Script failed (code {result.returncode})"
            output_field_ref.value = f"Exit Code: {result.returncode}\n\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
            if not result.stderr and result.stdout and result.returncode != 0: # Sometimes errors go to stdout
                 error_message += f" - See STDOUT in output box."
            elif result.stderr:
                error_message += f": {result.stderr[:50]}..."
            snack_bar_instance = ft.SnackBar(content=ft.Text(error_message), open=True)
            page_ref.snack_bar = snack_bar_instance
            print(f"Script failed. STDOUT: {result.stdout}\nSTDERR: {result.stderr}")
    except Exception as e:
        print(f"Exception in run_script_command: {str(e)}")
        output_field_ref.value = f"Failed to run command: {str(e)}"
        snack_bar_instance = ft.SnackBar(content=ft.Text(f"Failed to run command: {str(e)}"), open=True)
        page_ref.snack_bar = snack_bar_instance
    finally:
        print(f"--- run_script_command finally block ---")
        output_field_ref.visible = True
        button_ref.text = original_button_text
        button_ref.disabled = False
        progress_bar_ref.visible = False
        print(f"Updating page in finally. Output field should be visible. Output text: {output_field_ref.value}")
        page_ref.update()

# =====================
# GUI-Building Functions
# =====================

def build_file_picker_row(lora_input_path_field, lora_output_path_field, file_picker_input, file_picker_output):
    """Build the row for file pickers (input file and output directory)."""
    return ft.ResponsiveRow(
        controls=[
            lora_input_path_field,
            ft.IconButton(
                ft.Icons.FOLDER_OPEN,
                tooltip="Select input Lora file",
                col=1,
                on_click=lambda _: file_picker_input.pick_files(
                    allow_multiple=False,
                    allowed_extensions=["safetensors"],
                    initial_directory=os.path.join(os.getcwd(), "workspace", "output")
                )
            ),
            lora_output_path_field,
            ft.IconButton(
                ft.Icons.FOLDER_OPEN,
                tooltip="Select output directory",
                col=1,
                on_click=lambda _: file_picker_output.get_directory_path(
                     initial_directory=os.path.join(os.getcwd(), "workspace", "output_converted")
                )
            )
        ],
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
        spacing=8
    )

def build_convert_lora_section(page, file_picker_input, file_picker_output):
    """Build the Convert Lora section controls and event handlers."""
    # --- Controls ---
    lora_input_path_field = create_textfield("Lora Input path", None, hint_text="Path to Lora file", expand=True, col=5)
    lora_output_path_field = create_textfield("Lora Output Path", None, hint_text="Directory for output", expand=True, col=5)
    output_name_field = create_textfield("Output name", None, hint_text="Leave empty to use the original name (suffix '_comfy' will be added)", expand=True, col=9)
    convert_button = create_styled_button("Convert", col=3)
    progress_bar_convert = ft.ProgressBar(visible=False, col=12)
    output_display_field = ft.TextField(
        label="Script Output",
        text_size=10,
        multiline=True,
        read_only=True,
        visible=False,
        min_lines=3,
        max_lines=10,
        expand=True,
        col=12
    )

    # --- Event Handlers ---
    def pick_input_file_result(e: FilePickerResultEvent):
        if e.files:
            lora_input_path_field.value = e.files[0].path
            lora_input_path_field.update()

    def pick_output_directory_result(e: FilePickerResultEvent):
        if e.path:
            lora_output_path_field.value = e.path
            lora_output_path_field.update()

    def handle_convert_lora_click(e):
        input_lora = lora_input_path_field.value
        output_dir = lora_output_path_field.value
        custom_name_stem_user_input = output_name_field.value

        if not input_lora:
            page.show_snack_bar(ft.SnackBar(content=ft.Text("Error: Lora Input path is required."), open=True))
            return
        if not output_dir:
            page.show_snack_bar(ft.SnackBar(content=ft.Text("Error: Lora Output Path (directory) is required."), open=True))
            return

        abs_input_lora = get_abs_path(input_lora)
        abs_output_dir = get_abs_path(output_dir)
        final_stem = get_stem_with_comfy_suffix(abs_input_lora, custom_name_stem_user_input)
        command = build_convert_command(abs_input_lora, abs_output_dir, final_stem)

        convert_button.text = "Converting..."
        convert_button.disabled = True
        progress_bar_convert.visible = True
        output_display_field.value = ""
        output_display_field.visible = False
        page.update()

        page.run_task(run_script_command, command, page, convert_button, progress_bar_convert, output_display_field, "Convert")

    # Assign event handlers
    convert_button.on_click = handle_convert_lora_click
    file_picker_input.on_result = pick_input_file_result
    file_picker_output.on_result = pick_output_directory_result

    # --- Layout ---
    controls = []
    controls.extend(add_section_title("Convert Lora"))
    controls.append(
        build_file_picker_row(lora_input_path_field, lora_output_path_field, file_picker_input, file_picker_output)
    )
    controls.append(
        ft.ResponsiveRow(
            controls=[output_name_field, convert_button],
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=8
        )
    )
    controls.append(ft.ResponsiveRow([progress_bar_convert]))
    controls.append(ft.ResponsiveRow([output_display_field]))

    return controls

# =====================
# Main Tab Content Entrypoint
# =====================

def get_tools_tab_content(page: ft.Page):
    """Entrypoint for the Tools tab content. Builds and returns the tab's UI container."""
    page_controls = []

    # File pickers (must be added to page.overlay)
    file_picker_input = FilePicker()
    page.overlay.append(file_picker_input)
    file_picker_output = FilePicker()
    page.overlay.append(file_picker_output)

    # Build Convert Lora section
    page_controls.extend(build_convert_lora_section(page, file_picker_input, file_picker_output))

    return ft.Container(
        content=ft.Column(
            controls=page_controls,
            spacing=8,
            scroll=ft.ScrollMode.AUTO,
        ),
        expand=True,
        padding=ft.padding.all(10)
    )