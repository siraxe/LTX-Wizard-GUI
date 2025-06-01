import typer
from pathlib import Path
from safetensors.torch import load_file
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

app = typer.Typer(
    pretty_exceptions_enable=False,
    help="Compare two LoRA models, generate a 3D difference plot, and output layer similarities."
)

def get_tensor_keys(state_dict):
    """Helper to list all tensor keys, typically for debugging or selection."""
    return [key for key, value in state_dict.items() if isinstance(value, torch.Tensor)]

@app.command()
def main(
    lora_a_path: Path = typer.Option(..., "--lora-a-path", "-a", help="Path to the first LoRA file (.safetensors)", exists=True, file_okay=True, dir_okay=False, readable=True),
    lora_b_path: Path = typer.Option(..., "--lora-b-path", "-b", help="Path to the second LoRA file (.safetensors)", exists=True, file_okay=True, dir_okay=False, readable=True),
    output_plot_path: Path = typer.Option(..., "--output-plot-path", "-p", help="Path to save the generated 3D plot image (.png)"),
    # For now, we'll automatically try to find comparable layers.
    # Later, we can add specific key selection:
    # layer_key_a: str = typer.Option("", "--key-a", help="Specific tensor key from LoRA A to compare (e.g., 'base_model.model.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora.down.weight')"),
    # layer_key_b: str = typer.Option("", "--key-b", help="Specific tensor key from LoRA B to compare (e.g., 'base_model.model.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora.down.weight')"),
):
    """Compares two LoRA models, generates a 3D plot of their differences for a sample layer, and prints layer info."""
    typer.echo(f"DEBUG: Loading LoRA A from: {lora_a_path}")
    lora_a_state_dict = load_file(str(lora_a_path))
    typer.echo(f"DEBUG: Loading LoRA B from: {lora_b_path}")
    lora_b_state_dict = load_file(str(lora_b_path))

    typer.echo("\n--- LoRA A Keys ---")
    keys_a = get_tensor_keys(lora_a_state_dict)
    typer.echo(f"DEBUG: Found {len(keys_a)} keys in LoRA A.")
    # for key in keys_a:
    #     typer.echo(key) # Keep this commented unless very detailed key listing is needed for a specific debug run

    typer.echo("\n--- LoRA B Keys ---")
    keys_b = get_tensor_keys(lora_b_state_dict)
    typer.echo(f"DEBUG: Found {len(keys_b)} keys in LoRA B.")
    # for key in keys_b:
    #     typer.echo(key) # Keep this commented

    # Placeholder for finding common/comparable keys and selecting one pair
    # For this initial version, let's just try to find the first common '.lora_down.weight' or similar
    # This is a very naive approach and will need refinement.
    common_suffix_candidates = [
        ".lora_down.weight", 
        ".lora_A.weight", 
        ".lora.up.weight", 
        ".lora_B.weight"
    ]
    
    tensor_a = None
    tensor_b = None
    selected_key_a = None
    selected_key_b = None
    typer.echo("DEBUG: Starting search for comparable layers for 3D plot...")

    for key_a_full in keys_a:
        for suffix in common_suffix_candidates:
            if key_a_full.endswith(suffix):
                typer.echo(f"DEBUG: Potential match: key_a_full='{key_a_full}', suffix='{suffix}'")
                # Try to find a corresponding key in LoRA B
                # This assumes a somewhat similar naming convention or structure
                key_b_candidate_stem = key_a_full[:-len(suffix)]
                for key_b_full in keys_b:
                    if key_b_full.startswith(key_b_candidate_stem) and key_b_full.endswith(suffix):
                        typer.echo(f"DEBUG: Found corresponding key_b_full='{key_b_full}' for key_a_full='{key_a_full}'")
                        selected_key_a = key_a_full
                        selected_key_b = key_b_full
                        tensor_a = lora_a_state_dict[selected_key_a]
                        tensor_b = lora_b_state_dict[selected_key_b]
                        break
            if tensor_a is not None:
                break
        if tensor_a is not None:
            break

    typer.echo("DEBUG: Finished search for comparable layers for 3D plot.")
    if tensor_a is None or tensor_b is None:
        typer.echo(typer.style("Could not find a common layer with known suffixes to compare for 3D plot.", fg=typer.colors.RED))
        # Still proceed to generate table data if possible (future enhancement)
    else:
        typer.echo(f"DEBUG: Selected for 3D plot: '{selected_key_a}' (A) vs '{selected_key_b}' (B)")
        typer.echo(f"DEBUG: Tensor A shape: {tensor_a.shape}, Tensor B shape: {tensor_b.shape}")
        if tensor_a.shape != tensor_b.shape:
            typer.echo(typer.style(f"Selected tensors have different shapes: A{tensor_a.shape} vs B{tensor_b.shape}. Cannot directly compare for 3D plot.", fg=typer.colors.RED))
        else:
            typer.echo(f"Tensor shapes: A{tensor_a.shape}, B{tensor_b.shape}")
            # Ensure tensors are on CPU and convert to NumPy
            diff_tensor = (tensor_a.cpu().float() - tensor_b.cpu().float()).numpy()
            typer.echo(f"DEBUG: Calculated diff_tensor shape: {diff_tensor.shape}")

            # For 3D plot, we might need to handle >2D tensors (e.g., take a slice or aggregate)
            # For now, if it's 2D, plot it. If 1D, make it 2D. If >2D, take a slice.
            if diff_tensor.ndim == 1:
                # Make 1D array into a 2D array (e.g., a row vector)
                diff_tensor_2d = diff_tensor.reshape(1, -1)
            elif diff_tensor.ndim == 2:
                diff_tensor_2d = diff_tensor
            elif diff_tensor.ndim > 2:
                typer.echo(f"DEBUG: Tensor is >2D ({diff_tensor.ndim}D). Attempting to take a 2D slice for plotting.")
                typer.echo(f"Tensor is >2D ({diff_tensor.ndim}D). Taking a 2D slice (first two dimensions) for plotting.")
                # Example: take the first slice along the third dimension, and all of the first two.
                # This is a simplification; actual slicing might need to be more intelligent.
                slicing_indices = [0] * (diff_tensor.ndim - 2)
                diff_tensor_2d = diff_tensor[tuple(slicing_indices + [slice(None), slice(None)])]
                if diff_tensor_2d.ndim < 2: # if slicing resulted in 1D or 0D
                    diff_tensor_2d = diff_tensor[tuple([slice(None), slice(None)] + [0]*(diff_tensor.ndim-2))]
                typer.echo(f"Sliced to shape: {diff_tensor_2d.shape}")
            else: # 0D tensor
                typer.echo(typer.style("Tensor is 0D, cannot create 3D plot.", fg=typer.colors.RED))
                diff_tensor_2d = None

            if diff_tensor_2d is not None and diff_tensor_2d.ndim == 2:
                X = np.arange(diff_tensor_2d.shape[1])
                Y = np.arange(diff_tensor_2d.shape[0])
                X, Y = np.meshgrid(X, Y)
                Z = diff_tensor_2d
                typer.echo(f"DEBUG: Final 2D data for plot (Z) shape: {Z.shape}")

                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                ax.plot_surface(X, Y, Z, cmap='viridis') # Or 'coolwarm' for differences
                ax.set_xlabel('Dimension 1 Index')
                ax.set_ylabel('Dimension 0 Index')
                ax.set_zlabel('Weight Difference')
                ax.set_title(f'Difference: {selected_key_a[:30]}... vs {selected_key_b[:30]}...')
                
                try:
                    typer.echo(f"DEBUG: Attempting to save plot to: {output_plot_path}")
                    plt.savefig(str(output_plot_path))
                    typer.echo(f"DEBUG: 3D difference plot saved to: {output_plot_path}")
                except Exception as e:
                    typer.echo(typer.style(f"DEBUG: Error saving plot: {e}", fg=typer.colors.RED))
                finally:
                    plt.close(fig)
            elif diff_tensor_2d is not None:
                 typer.echo(typer.style(f"Could not reduce tensor to 2D for plotting. Final shape: {diff_tensor_2d.shape}", fg=typer.colors.RED))


    # --- Placeholder for tabular data generation (cosine similarity, etc.) ---
    # This part would iterate through common layers and print structured output
    # that the Flet app can parse for the table.
    # Example output format for one layer (to be printed to stdout):
    # LAYER_DATA::{ "layer_name": "some_common_layer_key", "similarity": 0.85, "difference": 0.15 }
    typer.echo("\nDEBUG: Generating tabular comparison data for ALL comparable layers...")
    table_data_generated_count = 0
    cos_sim_func = torch.nn.CosineSimilarity(dim=0) # Initialize once

    # Iterate through all keys in LoRA A to find matches in LoRA B for table data
    for key_a_table in keys_a:
        for suffix_table in common_suffix_candidates: # Using the same suffixes as for plot
            if key_a_table.endswith(suffix_table):
                key_b_candidate_stem_table = key_a_table[:-len(suffix_table)]
                for key_b_table in keys_b:
                    if key_b_table.startswith(key_b_candidate_stem_table) and key_b_table.endswith(suffix_table):
                        # Found a potential pair for table data
                        tensor_a_table = lora_a_state_dict.get(key_a_table)
                        tensor_b_table = lora_b_state_dict.get(key_b_table)

                        if tensor_a_table is not None and tensor_b_table is not None and tensor_a_table.shape == tensor_b_table.shape:
                            # Calculate similarity and difference
                            sim_table = cos_sim_func(
                                tensor_a_table.cpu().float().flatten(), 
                                tensor_b_table.cpu().float().flatten()
                            ).item()
                            mag_diff_table = torch.norm(tensor_a_table.cpu().float() - tensor_b_table.cpu().float()).item()
                            
                            layer_name_for_table_iter = key_a_table if key_a_table == key_b_table else f"{key_a_table} | {key_b_table}"
                            
                            layer_data_json_str = f"LAYER_DATA::{{\"layer_name\": \"{layer_name_for_table_iter}\", \"similarity\": {sim_table:.4f}, \"difference\": {mag_diff_table:.4f}}}"
                            typer.echo(f"DEBUG: Generated table data line: {layer_data_json_str}")
                            typer.echo(layer_data_json_str)
                            table_data_generated_count += 1
                        elif tensor_a_table is not None and tensor_b_table is not None:
                            typer.echo(f"DEBUG: Skipping table data for {key_a_table} & {key_b_table} due to shape mismatch: A{tensor_a_table.shape} vs B{tensor_b_table.shape}")
                        break # Found corresponding key_b_table for this key_a_table and suffix_table, move to next key_a_table or suffix_table
    
    if table_data_generated_count == 0:
        # Fallback if no comparable layers were found for table
        typer.echo("DEBUG: No comparable layers found for table data. Emitting dummy data.")
        dummy_line_1 = "LAYER_DATA::{{\"layer_name\": \"dummy_layer_1 (no comparable found)\", \"similarity\": 0.00, \"difference\": 0.00}}"
        typer.echo(f"DEBUG: Generated dummy table data line: {dummy_line_1}")
        typer.echo(dummy_line_1)

    typer.echo("DEBUG: Comparison script finished.")

if __name__ == "__main__":
    app()
