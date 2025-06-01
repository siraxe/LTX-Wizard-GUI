import flet as ft
from flet import FilePicker, FilePickerResultEvent # Import FilePicker related classes
from ._styles import create_textfield, add_section_title, create_styled_button
import os # Import os module for path manipulation
import subprocess # Import subprocess
import asyncio # Import asyncio
import base64
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend for Matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import json # Added for parsing script output
from pathlib import Path
from datetime import datetime
import functools

# =====================
# Helper/Data/Utility Functions
# =====================

# Placeholder for backend LoRA comparison logic
async def backend_compare_loras(lora_a_path: str, lora_b_path: str, page: ft.Page, output_plot_path_str: str):
    # This function will now primarily be a wrapper to call the script
    # The actual logic is in scripts/compare_loras.py
    # It will return the stdout of the script for parsing by the caller.

    python_exe = os.path.normpath(os.path.join("venv", "Scripts", "python.exe"))
    script_path = os.path.normpath(os.path.join("scripts", "compare_loras.py"))

    command_parts = [
        python_exe,
        script_path,
        "--lora-a-path", lora_a_path,
        "--lora-b-path", lora_b_path,
        "--output-plot-path", output_plot_path_str
    ]
    command_str = " ".join(f'"{part}"' if ' ' in part else part for part in command_parts)
    
    # We'll run this command using subprocess directly here for simplicity in returning stdout
    # or adapt run_script_command if more UI interaction during run is needed.
    # For now, let's assume a blocking call for simplicity of integrating, 
    # but run_script_command is better for non-blocking UI.
    
    # This is a simplified execution for now. Will be replaced by run_script_command call.
    # For the purpose of this change, we are modifying how _async_handle_compare_loras_click works.
    # This backend_compare_loras will be removed or significantly changed when integrating with run_script_command.
    process = await asyncio.to_thread(
        subprocess.run, command_str, capture_output=True, text=True, shell=True, check=False
    )
    
    
    
    if process.returncode != 0:
        # Log the error or raise an exception with stderr
        error_message = f"Error in compare_loras.py: {process.stderr}"
        print(error_message)
        # Return something to indicate failure, or raise exception
        return {"stdout": process.stdout, "stderr": process.stderr, "plot_path": output_plot_path_str, "success": False}
        
    return {"stdout": process.stdout, "stderr": process.stderr, "plot_path": output_plot_path_str, "success": True}

    """Simulates backend LoRA comparison. 
    In a real scenario, this would load LoRA files, compute delta weights, 
    and calculate similarities/differences per layer.
    """
    # Simulate network/computation delay
    await asyncio.sleep(1) # Keep it short for UI responsiveness

    if not lora_a_path or not lora_b_path:
        raise ValueError("Both LoRA A and LoRA B paths are required for comparison.")
    
    # Check if files exist (basic check)
    if not os.path.exists(lora_a_path):
        raise FileNotFoundError(f"LoRA A file not found: {lora_a_path}")
    if not os.path.exists(lora_b_path):
        raise FileNotFoundError(f"LoRA B file not found: {lora_b_path}")

    # Dummy data for demonstration
    dummy_results = [
        {"layer_name": "down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q", "similarity": 0.95, "difference": 0.0523},
        {"layer_name": "down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_k", "similarity": 0.20, "difference": 0.8512},
        {"layer_name": "mid_block_attentions_0_transformer_blocks_0_attn2_to_out_0", "similarity": 0.65, "difference": 0.3589},
        {"layer_name": "up_blocks_3_attentions_2_transformer_blocks_0_ff_net_0_proj", "similarity": -0.50, "difference": 1.5077}, 
        {"layer_name": "up_blocks_3_attentions_2_transformer_blocks_0_norm3", "similarity": 0.0, "difference": 1.0000},
    ]
    # Simulate a case where comparison might yield no common layers or an issue
    # if "empty" in lora_a_path:
    #     return []

    plot_path = None
    try:
        # Generate dummy 3D plot data
        X = np.arange(-5, 5, 0.25)
        Y = np.arange(-5, 5, 0.25)
        X, Y = np.meshgrid(X, Y)
        R = np.sqrt(X**2 + Y**2)
        Z = np.sin(R) # Example: a sinc function

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='viridis')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Difference Value')
        ax.set_title('Dummy 3D LoRA Layer Difference')
        
        # Save plot to a temporary file
        with open(output_plot_path_str, 'wb') as plot_file:
            plt.savefig(plot_file)
        plt.close(fig) # Close the figure to free memory

    except Exception as e:
        print(f"Error generating 3D plot: {e}")
        plot_path = None # Ensure plot_path is None if generation fails

    return {"table_data": dummy_results, "plot_path": output_plot_path_str}

# Helper for color-coding similarity scores in the DataTable
def get_similarity_color(similarity_score: float) -> str:
    if similarity_score > 0.75:
        return ft.Colors.GREEN_ACCENT_700
    elif similarity_score > 0.25:
        return ft.Colors.ORANGE_ACCENT_700
    elif similarity_score < -0.25: # Strong divergence/conflict
        return ft.Colors.RED_ACCENT_700
    return ft.Colors.BLUE_GREY_500 # Neutral or mild similarity/divergence

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


async def _async_handle_compare_loras_click(
    page: ft.Page, 
    lora_a_field: ft.TextField, 
    lora_b_field: ft.TextField, 
    lora_comparison_status: ft.Text, 
    lora_comparison_table: ft.DataTable,
    lora_3d_plot_image_display: ft.Container,
    # Add references to UI elements needed by run_script_command if we use it directly here
    # For now, we'll call a modified backend_compare_loras and parse its results.
    # compare_button: ft.ElevatedButton, # Example if run_script_command is used
    # progress_bar_compare: ft.ProgressBar # Example
):
    lora_a_path = lora_a_field.value
    lora_b_path = lora_b_field.value

    if not lora_a_path or not lora_b_path:
        page.show_snack_bar(ft.SnackBar(content=ft.Text("Error: Please select both LoRA A and LoRA B files for comparison."), open=True))
        return

    lora_comparison_status.value = "Comparing LoRAs... please wait."
    lora_comparison_table.rows.clear()
    lora_3d_plot_image_display.content.src = None
    lora_3d_plot_image_display.content.visible = False
    page.update()

    persistent_plot_path_str = None # Initialize
    try:
        # Define a persistent directory for plots within the flet_app structure
        base_path = Path(__file__).resolve().parent.parent # Should be flet_app directory
        plots_dir = base_path / "generated_data" / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        

        # Generate a unique filename for the plot
        lora_a_name_stem = Path(lora_a_path).stem
        lora_b_name_stem = Path(lora_b_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"plot_{lora_a_name_stem}_vs_{lora_b_name_stem}_{timestamp}.png"
        persistent_plot_path_str = str(plots_dir / plot_filename)
        

        # Call the backend function which now runs the script
        script_results = await backend_compare_loras(lora_a_path, lora_b_path, page, persistent_plot_path_str)
        
        
        stdout_str = script_results.get("stdout", "")
        stderr_str = script_results.get("stderr", "")
        returned_plot_path = script_results.get("plot_path") # This should be temp_plot_path_str
        success = script_results.get("success", False)

        lora_comparison_table.rows.clear() # Clear previous table data
        parsed_table_data = []
        if stdout_str:
            for line in stdout_str.splitlines():
                
                if line.startswith("LAYER_DATA::"):
                    try:
                        json_str = line.replace("LAYER_DATA::", "")
                        item = json.loads(json_str) # Make sure to import json
                        
                        parsed_table_data.append(item)
                        similarity_str = f"{item['similarity']:.4f}"
                        diff_str = f"{item['difference']:.4f}"
                        sim_color = get_similarity_color(item['similarity'])
                        lora_comparison_table.rows.append(
                            ft.DataRow(cells=[
                                ft.DataCell(ft.Text(item['layer_name'], tooltip=item['layer_name'])),
                                ft.DataCell(ft.Text(similarity_str, color=sim_color)),
                                ft.DataCell(ft.Text(diff_str)),
                            ])
                        )
                    except json.JSONDecodeError as json_err:
                        print(f"Error decoding JSON from script output: {json_err} on line: {line}")
                    except KeyError as key_err:
                        print(f"Missing key in JSON from script output: {key_err} on line: {line}")
        
        if not success:
            lora_comparison_status.value = f"Comparison script failed. Check console/script errors."
            if stderr_str:
                lora_comparison_status.value += f" Details: {stderr_str[:200]}..."
            page.show_snack_bar(ft.SnackBar(content=ft.Text("Comparison script error. See status for details."), open=True))
        elif not parsed_table_data and not (persistent_plot_path_str and os.path.exists(persistent_plot_path_str)):
            lora_comparison_status.value = "Comparison complete. No data or plot to display."
        elif not parsed_table_data:
            lora_comparison_status.value = "Comparison complete. No tabular data to display."
        else:
            lora_comparison_status.value = f"Comparison complete. Displaying {len(parsed_table_data)} layers."

        # Update page after table modifications, before plot display logic
        page.update()

        if persistent_plot_path_str and os.path.exists(persistent_plot_path_str):
            try:
                with open(persistent_plot_path_str, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
                lora_3d_plot_image_display.content.src = None # Clear path-based src
                lora_3d_plot_image_display.content.src_base64 = f"data:image/png;base64,{encoded_string}"
                lora_3d_plot_image_display.content.visible = True
                lora_comparison_status.value = "Comparison complete. Image displayed." # Visual cue
            except Exception as img_ex:
                lora_comparison_status.value = f"Error displaying image: {img_ex}" # Visual cue for error
                lora_3d_plot_image_display.content.visible = False
                lora_3d_plot_image_display.content.src = None
                lora_3d_plot_image_display.content.src_base64 = None
            page.update() # Crucial: Update the page to reflect image src_base64 and visibility change
        else:
            lora_comparison_status.value = "Comparison complete. Plot image not found." # Visual cue
            lora_3d_plot_image_display.content.visible = False
            lora_3d_plot_image_display.content.src = None # Clear src if image not found
            lora_3d_plot_image_display.content.src_base64 = None
            page.update()

    except FileNotFoundError as fnf_err: # This would be for files _async_handle_compare_loras_click itself tries to access
        lora_comparison_status.value = f"Error: {str(fnf_err)}"
        page.show_snack_bar(ft.SnackBar(content=ft.Text(f"Error: {str(fnf_err)}"), open=True))
    except ValueError as val_err:
        lora_comparison_status.value = f"Error: {str(val_err)}"
        page.show_snack_bar(ft.SnackBar(content=ft.Text(f"Error: {str(val_err)}"), open=True))
    except Exception as ex:
        lora_comparison_status.value = f"An unexpected error occurred: {str(ex)}"
        page.show_snack_bar(ft.SnackBar(content=ft.Text(f"An unexpected error occurred: {str(ex)}.{type(ex).__name__}"), open=True))
        print(f"DEBUG: [_async_handle_compare_loras_click] Exception: {ex}", type(ex))
        import traceback
        traceback.print_exc()
    finally:
        # No longer cleaning up the plot file here as it's persistent.
        # Consider a separate cleanup mechanism for the plots_dir if it grows too large.
        page.update()

def build_convert_command(input_lora: str, output_dir: str, output_stem: str) -> str:
    """Build the command string to run the Lora conversion script."""
    python_exe = os.path.normpath(os.path.join("venv", "Scripts", "python.exe"))
    script_file = os.path.normpath("scripts/convert_checkpoint.py")
    output_filename = f"{output_stem}.safetensors"
    abs_full_output_path = os.path.join(output_dir, output_filename)
    return f'"{python_exe}" "{script_file}" "{input_lora}" --output-path "{abs_full_output_path}" --to-comfy'

async def run_script_command(command_str: str, page_ref: ft.Page, button_ref: ft.ElevatedButton, progress_bar_ref: ft.ProgressBar, output_field_ref: ft.TextField, original_button_text: str):
    """Run a shell command asynchronously and update the UI with the result."""
    
    output_field_ref.value = "" # Clear previous output
    output_field_ref.visible = True # Ensure output field is visible from the start
    page_ref.update() # Update UI to show empty output field

    process = None # Initialize process variable
    try:
        # Using shell=True because the command is a single string with quoted paths.
        # Ensure paths in command_str are correctly quoted if they contain spaces.
        print("Attempting to run subprocess with Popen...")
        # Use Popen to allow streaming output
        process = await asyncio.create_subprocess_shell(
            command_str,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print(f"Subprocess started with PID: {process.pid}")

        # Read output in real-time
        async def read_stream(stream, output_field, stream_name):
            print(f"Starting to read {stream_name}...")
            while True:
                # Read a line (or chunk) from the stream
                line = await stream.readline()
                if not line:
                    print(f"End of {stream_name}.")
                    break
                # Decode the line and append to the output field
                decoded_line = line.decode('utf-8', errors='replace').strip()
                print(f"{stream_name}: {decoded_line}")
                # Use await asyncio.to_thread to update UI, as it might be blocking
                await asyncio.to_thread(
                    lambda: setattr(output_field, 'value', output_field.value + decoded_line + "\n") # Append line with newline
                )
                await asyncio.to_thread(page_ref.update) # Update the UI after appending

        # Create tasks to read stdout and stderr concurrently
        stdout_task = asyncio.create_task(read_stream(process.stdout, output_field_ref, "STDOUT"))
        stderr_task = asyncio.create_task(read_stream(process.stderr, output_field_ref, "STDERR"))

        # Wait for both streams to finish and the process to complete
        await asyncio.gather(stdout_task, stderr_task)

        # Wait for the process to terminate and get the return code
        returncode = await process.wait()
        print(f"Subprocess finished. Return code: {returncode}")

        # Check return code for success/failure
        if returncode == 0:
            snack_bar_instance = ft.SnackBar(content=ft.Text(f"Script successful."), open=True)
            page_ref.snack_bar = snack_bar_instance
        else:
            error_message = f"Script failed (code {returncode})"
            # Note: stderr and stdout are already streamed, so the final value of output_field_ref
            # will contain the combined output. We might want to add a specific error indicator
            # or just rely on the snackbar and the content of the output field.
            snack_bar_instance = ft.SnackBar(content=ft.Text(error_message), open=True)
            page_ref.snack_bar = snack_bar_instance
            print(f"Script failed. Exit Code: {returncode}")

    except Exception as e:
        print(f"Exception in run_script_command: {str(e)}")
        # Use await asyncio.to_thread to update UI
        await asyncio.to_thread(
            lambda: setattr(output_field_ref, 'value', output_field_ref.value + f"\nFailed to run command: {str(e)}")
        )
        snack_bar_instance = ft.SnackBar(content=ft.Text(f"Failed to run command: {str(e)}"), open=True)
        page_ref.snack_bar = snack_bar_instance
    finally:
        print(f"--- run_script_command finally block ---")
        # Ensure the output field is visible regardless of success/failure
        await asyncio.to_thread(
            lambda: setattr(output_field_ref, 'visible', True)
        )
        await asyncio.to_thread(
             lambda: setattr(button_ref, 'text', original_button_text)
        )
        await asyncio.to_thread(
            lambda: setattr(button_ref, 'disabled', False)
        )
        await asyncio.to_thread(
            lambda: setattr(progress_bar_ref, 'visible', False)
        )
        print(f"Updating page in finally.")
        await asyncio.to_thread(page_ref.update)

def parse_weights_string(weights_str: str) -> list[float] | None:
    """Parse the comma-separated weights string into a list of floats."""
    if not weights_str:
        return None # Return None if string is empty
    try:
        weights_list = [float(w.strip()) for w in weights_str.split(',') if w.strip()]
        if len(weights_list) != 2:
            raise ValueError("Exactly two comma-separated weights are required.")
        return weights_list
    except ValueError as e:
        # Re-raise with a more specific message or handle as needed
        raise ValueError(f"Invalid format for weights: {e}. Use comma-separated numbers (e.g., 1.0,0.5).") from e

def parse_density_string(density_str: str) -> float | None:
    """Parse the density string into a float and validate its range."""
    if not density_str:
        return None
    try:
        density = float(density_str)
        if not (0.0 <= density <= 1.0):
            raise ValueError("Density must be between 0.0 and 1.0.")
        return density
    except ValueError as e:
        raise ValueError(f"Invalid format for density: {e}. Use a number between 0.0 and 1.0.") from e

def build_merge_command(lora_a_path: str, lora_b_path: str, output_path: str, technique: str, weights: list[float] | None = None, density: float | None = None, ties_method: str | None = None, dare_density: float | None = None, dare: bool = False) -> str:
    """Build the command string to run the LoRA merging script."""
    python_exe = os.path.normpath(os.path.join("venv", "Scripts", "python.exe"))
    script_file = os.path.normpath("scripts/merge_loras.py")

    command_parts = [f'"{python_exe}"', f'"{script_file}"']

    # Add input paths (A and B) and output path as positional arguments (quoted)
    command_parts.extend([f'"{lora_a_path}"', f'"{lora_b_path}"'])
    command_parts.append(f'"{output_path}"')

    # Add merge technique option
    command_parts.extend(["--technique", technique])

    # Add optional weights by repeating the flag for each weight
    if weights is not None:
         for w in weights:
              command_parts.extend(["--weights", str(w)])

    # Add density option if applicable
    if density is not None:
         command_parts.extend(["--density", str(density)])

    # Add TIES method option if applicable
    if technique == "ties" and ties_method:
         command_parts.extend(["--majority-sign-method", ties_method])

    # Add DARE options if applicable
    if dare and dare_density is not None:
        command_parts.extend(["--dare"]) # Add the flag if True
        command_parts.extend(["--dare-density", str(dare_density)])

    return " ".join(command_parts)

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
                    initial_directory=os.path.join(os.getcwd(), "workspace", "output", "checkpoints")
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
    lora_output_path_field = create_textfield("Lora Output Path (optional)", None, hint_text="Directory for output", expand=True, col=5)
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
        # Check if output_dir is empty, if so, use the input file's directory
        if not output_dir:
            if not input_lora:
                 # This case should ideally not happen due to the check above, but good for safety
                 page.show_snack_bar(ft.SnackBar(content=ft.Text("Error: Both input and output paths are empty."), open=True))
                 return
            output_dir = os.path.dirname(input_lora) # Use directory of input file

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
# New Merge LoRAs Section
# =====================

def build_merge_loras_section(page, file_picker_merge_input: FilePicker, file_picker_merge_output: FilePicker, merge_lora_a_field_ref: ft.TextField, merge_lora_b_field_ref: ft.TextField):
    """Build the Merge LoRAs section controls and event handlers.
    Uses passed-in TextField references for LoRA A and LoRA B paths.
    """
    # --- Controls ---
    # Buttons for LoRA A and B are still defined here, but they will update the passed-in TextFields.
    merge_lora_a_button = ft.IconButton(
        ft.Icons.FOLDER_OPEN,
        tooltip="Select input LoRA A file",
        col=1,
    )

    merge_lora_b_button = ft.IconButton(
        ft.Icons.FOLDER_OPEN,
        tooltip="Select input LoRA B file",
        col=1,
    )

    # Output file
    merge_lora_output_path_field = create_textfield(
        "Merged Output File",
        None,
        hint_text="Path to save the merged file",
        expand=True,
        col=5
    )
    merge_lora_output_button = ft.IconButton(
        ft.Icons.SAVE,
        tooltip="Select output file path",
        col=1,
    )

    merge_weights_field = create_textfield(
        "Weights (Optional if none)",
        "0.5,0.5", # Added default value
        hint_text="Comma-separated list for A,B (e.g., 1.0,0.5)",
        expand=True,
        col=2
    )

    merge_density_field = create_textfield(
        "Density (Optional)",
        "0.0", # Changed default to string
        hint_text="Number between 0 and 1 (e.g., 0.5)",
        expand=True,
        col=2
    )

    merge_technique_dropdown = ft.Dropdown(
        label="Merge Technique",
        options=[
            ft.dropdown.Option(key="linear", text="linear"),
            ft.dropdown.Option(key="cat", text="cat"),
            ft.dropdown.Option(key="ties", text="ties"),
            ft.dropdown.Option(key="svd", text="svd"),
            # Add more as implemented in scripts/merge_loras.py
        ],
        value="linear", # Default value
        expand=True,
        col=2
    )

    merge_dare_density_field = create_textfield(
        "DARE Density (Optional)",
        "0.7", # Changed default to string
        hint_text="Number between 0 and 1 (e.g., 0.5)",
        expand=True,
        col=2
    )
    merge_dare_density_checkbox = ft.Checkbox(
        label="Use DARE",
        value=False,
        expand=True,
        col=2
    )
    dare_field_container = ft.Container(
        content=ft.Row([merge_dare_density_field, merge_dare_density_checkbox]),
        expand=True,
        col=4,
        visible=True
    )

    merge_ties_technique_dropdown = ft.Dropdown(
        label="Method",
        options=[
            ft.dropdown.Option(key="frequency", text="frequency"),
            ft.dropdown.Option(key="total", text="total"),
        ],
        value="frequency", # Default value
        expand=True,
        visible=False, # Hidden by default
        col=2,
        tooltip="Majority sign method for TIES"
    )

    merge_button = create_styled_button("Merge LoRAs", col=3)

    progress_bar_merge = ft.ProgressBar(visible=False, col=12)

    output_display_field_merge = ft.TextField(
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
    # Variable to track which input field to update after file selection
    current_picker_target_field = None

    def pick_merge_input_files_result(e: FilePickerResultEvent):
        nonlocal current_picker_target_field # Access the variable from the outer scope
        if e.files and len(e.files) > 0 and current_picker_target_field:
            current_picker_target_field.value = e.files[0].path
            current_picker_target_field.update()
        # Reset target after selection (or cancellation)
        current_picker_target_field = None


    def pick_merge_output_file_result(e: FilePickerResultEvent):
        # This handler is now used for save_file for the output path
        if e.path:
            # save_file provides a single path in e.path
            output_path = e.path
            # Manually ensure the .safetensors extension is present
            if not output_path.lower().endswith(".safetensors"):
                output_path += ".safetensors"
            merge_lora_output_path_field.value = output_path
        else:
            # Clear the field if no file was selected
            merge_lora_output_path_field.value = ""
        merge_lora_output_path_field.update()

    def handle_merge_technique_change(e):
        """Show/hide the TIES method dropdown and DARE density fields based on the selected merge technique."""
        # Update TIES method dropdown visibility
        if merge_technique_dropdown.value == "ties":
            merge_ties_technique_dropdown.visible = True
        else:
            merge_ties_technique_dropdown.visible = False

        # Update DARE field container visibility
        if merge_technique_dropdown.value in ["linear", "ties"]:
            dare_field_container.visible = True
        else:
            dare_field_container.visible = False

        # Need to update the parent row or the entire section to reflect visibility changes
        # Updating the page is the safest bet in this structure
        page.update()


    async def handle_merge_loras_click(e):
        lora_a_path_str = merge_lora_a_field_ref.value
        lora_b_path_str = merge_lora_b_field_ref.value
        output_path_str = merge_lora_output_path_field.value
        weights_str = merge_weights_field.value
        technique = merge_technique_dropdown.value
        density_str = merge_density_field.value
        ties_method = merge_ties_technique_dropdown.value if merge_ties_technique_dropdown.visible else None
        # Get DARE values
        dare_density_str = merge_dare_density_field.value # Read DARE density field
        use_dare = merge_dare_density_checkbox.value # Read DARE checkbox value

        if not lora_a_path_str:
            page.show_snack_bar(ft.SnackBar(content=ft.Text("Error: Input LoRA A file is required."), open=True))
            return
        if not lora_b_path_str:
            page.show_snack_bar(ft.SnackBar(content=ft.Text("Error: Input LoRA B file is required."), open=True))
            return
        if not output_path_str:
            page.show_snack_bar(ft.SnackBar(content=ft.Text("Error: Merged Output File path is required."), open=True))
            return

        input_lora_paths = [get_abs_path(lora_a_path_str), get_abs_path(lora_b_path_str)]
        abs_output_path = get_abs_path(output_path_str)

        # Parse weights using the helper function
        merge_weights = None
        if weights_str:
            try:
                merge_weights = parse_weights_string(weights_str)
            except ValueError as e:
                page.show_snack_bar(ft.SnackBar(content=ft.Text(f"Error: {e}"), open=True))
                return

        # Parse density using the helper function
        merge_density = None
        # Only parse density if the field is not empty or if the technique requires it
        if density_str or technique in ["ties"]:
             try:
                 merge_density = parse_density_string(density_str)
             except ValueError as e:
                 page.show_snack_bar(ft.SnackBar(content=ft.Text(f"Error: {e}"), open=True))
                 return

        # Check if density is required for the selected technique but not provided
        if technique in ["ties"] and merge_density is None:
             page.show_snack_bar(ft.SnackBar(content=ft.Text(f"Error: Density is required for '{technique}' technique."), open=True))
             return

        # Parse DARE density using the helper function
        dare_density_value = None
        if dare_density_str:
             try:
                 dare_density_value = parse_density_string(dare_density_str)
             except ValueError as e:
                 page.show_snack_bar(ft.SnackBar(content=ft.Text(f"Error: {e}"), open=True))
                 return

        # Build the command string using the helper function
        try:
            command_str = build_merge_command(
                input_lora_paths[0],
                input_lora_paths[1],
                abs_output_path,
                technique,
                weights=merge_weights,
                density=merge_density,
                ties_method=ties_method,
                dare_density=dare_density_value, # Pass parsed DARE density
                dare=use_dare # Pass the boolean value from the checkbox
            )
        except ValueError as e:
             page.show_snack_bar(ft.SnackBar(content=ft.Text(f"Error building command: {e}"), open=True))
             return


        # Update UI and run command using the existing helper
        original_button_text = merge_button.text
        merge_button.text = "Merging..."
        merge_button.disabled = True
        progress_bar_merge.visible = True
        output_display_field_merge.value = ""
        output_display_field_merge.visible = False
        page.update()

        # Run the script command and wait for it to finish
        await run_script_command(
            command_str,
            page,
            merge_button,
            progress_bar_merge,
            output_display_field_merge,
            original_button_text # Pass original text back
        )


    # Assign event handlers
    file_picker_merge_input.on_result = pick_merge_input_files_result
    file_picker_merge_output.on_result = pick_merge_output_file_result
    merge_technique_dropdown.on_change = handle_merge_technique_change # Assign the change handler


    # Configure file picker calls for A and B
    initial_lora_dir = os.path.join(os.getcwd(), "workspace", "output", "checkpoints")
    initial_merged_output_dir = os.path.join(os.getcwd(), "workspace", "output_merged")


    def handle_pick_lora_a(e):
        nonlocal current_picker_target_field
        current_picker_target_field = merge_lora_a_field_ref # Use passed-in reference
        file_picker_merge_input.pick_files(
            allow_multiple=False,
            allowed_extensions=["safetensors"],
            initial_directory=initial_lora_dir
        )

    def handle_pick_lora_b(e):
        nonlocal current_picker_target_field
        current_picker_target_field = merge_lora_b_field_ref # Use passed-in reference
        file_picker_merge_input.pick_files(
            allow_multiple=False,
            allowed_extensions=["safetensors"],
            initial_directory=initial_lora_dir
        )

    def handle_pick_merge_output_file(e):
         # This will call save_file, and the result is handled by pick_merge_output_file_result
         file_picker_merge_output.save_file(
            allowed_extensions=["safetensors"],
            initial_directory=initial_merged_output_dir,
            file_name="merged_lora.safetensors" # Suggest a default filename
         )


    merge_lora_a_button.on_click = handle_pick_lora_a
    merge_lora_b_button.on_click = handle_pick_lora_b
    merge_lora_output_button.on_click = handle_pick_merge_output_file # Use the new handler


    merge_button.on_click = handle_merge_loras_click # Assign the async handler

    # --- Layout ---
    controls = []
    controls.extend(add_section_title("Merge LoRAs"))
    # Input files A row
    controls.append(
        ft.ResponsiveRow(
            controls=[merge_lora_a_field_ref, merge_lora_a_button, merge_lora_b_field_ref, merge_lora_b_button],
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=8
        )
    )

    # Output file row
    controls.append(
        ft.ResponsiveRow(
            controls=[merge_density_field,merge_technique_dropdown,merge_weights_field,merge_lora_output_path_field, merge_lora_output_button],
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=8
        )
    )

    # Button row
    controls.append(
        ft.ResponsiveRow(
            controls=[merge_ties_technique_dropdown,dare_field_container,merge_button],
            alignment=ft.MainAxisAlignment.END # Align button to the right
        )
    )
    # Progress bar row
    controls.append(ft.ResponsiveRow([progress_bar_merge]))
    # Output display row
    controls.append(ft.ResponsiveRow([output_display_field_merge]))
    return controls


async def on_compare_click_wrapper(page_arg, lora_a_field_arg, lora_b_field_arg, lora_comparison_status_arg, lora_comparison_table_arg, lora_3d_plot_image_display_arg, e):
    # 'e' is the event object from Flet, passed by functools.partial as the last argument
    await _async_handle_compare_loras_click(
        page=page_arg,
        lora_a_field=lora_a_field_arg,
        lora_b_field=lora_b_field_arg,
        lora_comparison_status=lora_comparison_status_arg,
        lora_comparison_table=lora_comparison_table_arg,
        lora_3d_plot_image_display=lora_3d_plot_image_display_arg
    )

# =====================
# Main Tab Content Entrypoint
# =====================

def get_tools_tab_content(page: ft.Page):
    """Entrypoint for the Tools tab content. Builds and returns the tab's UI container."""
    page_controls = []

    # File pickers (must be added to page.overlay)
    # File pickers for Convert Lora section
    file_picker_input = FilePicker()
    page.overlay.append(file_picker_input)
    file_picker_output = FilePicker()
    page.overlay.append(file_picker_output)

    # File pickers for Merge LoRAs section
    # Reusing file_picker_merge_input for both LoRA A and B pickers
    file_picker_merge_input = FilePicker() # This will be used by merge_lora_a_button and merge_lora_b_button
    page.overlay.append(file_picker_merge_input)
    file_picker_merge_output = FilePicker() # For the output of the merge operation
    page.overlay.append(file_picker_merge_output)

    # Create TextFields for LoRA A and B paths here, to be shared with Merge and Compare sections
    merge_lora_a_field = create_textfield(
        "Input LoRA A (for Merge/Compare)", None, hint_text="Path to LoRA A .safetensors file", expand=True, col=5
    )
    merge_lora_b_field = create_textfield(
        "Input LoRA B (for Merge/Compare)", None, hint_text="Path to LoRA B .safetensors file", expand=True, col=5
    )

    # Build Convert Lora section
    page_controls.extend(build_convert_lora_section(page, file_picker_input, file_picker_output))

    # Add a separator for visual distinction
    page_controls.append(ft.Divider(height=20, color=ft.Colors.with_opacity(0.2, ft.Colors.ON_SURFACE)))

    # Build Merge LoRAs section, passing the shared TextFields
    page_controls.extend(build_merge_loras_section(page, file_picker_merge_input, file_picker_merge_output, merge_lora_a_field, merge_lora_b_field))

    # Define the button handler using functools.partial to pass arguments
    lora_comparison_status = ft.Text("Select LoRA A and LoRA B above, then click 'Compare LoRAs'.", col=12)

    lora_3d_plot_image_display_image = ft.Image(visible=False, width=700, height=500, fit=ft.ImageFit.CONTAIN)
    lora_3d_plot_image_display = ft.Container(
        content=lora_3d_plot_image_display_image,
        width=700, # Match image width
        height=500, # Match image height
        alignment=ft.alignment.center,
        visible=False, # Container visibility will control overall overall visibility
        border=ft.border.all(1, ft.Colors.BLUE_GREY_400), # Optional: add a border for visual clarity
        border_radius=ft.border_radius.all(5)
    )
    # Initialize the comparison table
    lora_comparison_table = ft.DataTable(
    columns=[
        ft.DataColumn(ft.Text("Layer Name")),
        ft.DataColumn(ft.Text("Cosine Similarity"), numeric=True),
        ft.DataColumn(ft.Text("Mag. of Difference"), numeric=True),
    ],
    rows=[],
    column_spacing=20,
    expand=True,)

    compare_button_on_click_handler = functools.partial(
        on_compare_click_wrapper,  # The external function we defined earlier
        page,                      # Pass the page object
        merge_lora_a_field,        # Pass LoRA A text field
        merge_lora_b_field,        # Pass LoRA B text field
        lora_comparison_status,    # Pass status text UI element
        lora_comparison_table,     # Pass table UI element
        lora_3d_plot_image_display # Pass image display UI element
    )

    # --- LoRA Comparison Section UI --- 
    page_controls.append(ft.Divider(height=20, color=ft.Colors.with_opacity(0.2, ft.Colors.ON_SURFACE)))
    compare_button = create_styled_button("Compare LoRAs", on_click=compare_button_on_click_handler, col=3)
    # Button row for Compare LoRAs
    compare_row = ft.Row(controls=[ft.ResponsiveRow(controls=[compare_button],alignment=ft.MainAxisAlignment.END)
        ]
    )


    page_controls.extend(compare_row.controls)
    
    page_controls.append(ft.ResponsiveRow([lora_comparison_status]))
    page_controls.append(ft.ResponsiveRow(
        controls=[ft.Container(content=lora_3d_plot_image_display, alignment=ft.alignment.center, expand=True)]
    ))
    page_controls.append(ft.ResponsiveRow(
        controls=[ft.Container(content=lora_comparison_table, expand=True, border=ft.border.all(1, ft.Colors.OUTLINE), border_radius=5, padding=10)], 
        # The container helps manage scrolling if the table is very wide/long
    ))

    return ft.Container(
        content=ft.Column(
            controls=page_controls,
            spacing=8,
            scroll=ft.ScrollMode.AUTO,
        ),
        expand=True,
        padding=ft.padding.all(10)
    )