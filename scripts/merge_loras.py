import sys
from pathlib import Path
from typing import List, Optional

import torch
import typer
from safetensors.torch import load_file, save_file
# You might need other imports from PEFT or Diffusers if using their specific merge functions
# from peft import ...
# from diffusers import .

app = typer.Typer(
    pretty_exceptions_enable=False,
    help="Merge multiple LoRA adapters into a single LoRA checkpoint file.",
)

def apply_dare_preprocessing(
    state_dict: dict, 
    density: float
) -> dict:
    """
    Applies DARE (Drop And REscale) preprocessing to a single LoRA state dictionary.
    This is a simplified implementation applying thresholding and rescaling to A and B factors.

    Args:
        state_dict (dict): The LoRA state dictionary.
        density (float): The density parameter (0.0 to 1.0) for DARE.

    Returns:
        dict: The preprocessed LoRA state dictionary.
    """
    typer.echo(f"Applying DARE preprocessing with density={density}")
    processed_state_dict = {}
    scaling_factor = 1.0 / density if density > 0 else 0.0 # Simple rescaling

    for key, tensor in state_dict.items():
        is_lora_a = ".lora_A.default.weight" in key
        is_lora_b = ".lora_B.default.weight" in key

        if is_lora_a or is_lora_b:
            try:
                num_elements = tensor.numel()
                k = int(density * num_elements)

                if k == 0 and density > 0:
                    # If density > 0 but k=0, drop all elements
                    processed_tensor = torch.zeros_like(tensor)
                elif k >= num_elements:
                     # If density is 1.0 or more, keep all elements
                     processed_tensor = tensor
                else:
                    # Threshold based on magnitude
                    flattened_tensor = tensor.flatten()
                    magnitudes = torch.abs(flattened_tensor)
                    threshold = torch.topk(magnitudes, k)[0][-1]
                    threshold_mask = torch.abs(tensor) >= threshold
                    thresholded_tensor = tensor * threshold_mask

                    # Apply rescaling
                    processed_tensor = thresholded_tensor * scaling_factor

                processed_state_dict[key] = processed_tensor

            except Exception as e:
                typer.echo(f"Error applying DARE to key '{key}': {{e}}. Skipping DARE for this key.", err=True)
                processed_state_dict[key] = tensor
        else:
             processed_state_dict[key] = tensor

    return processed_state_dict

def _load_lora_files(
    lora_paths: List[Path],
    dare: bool,
    dare_density: Optional[float]
) -> List[dict]:
    """
    Loads LoRA state dictionaries from specified paths and applies DARE if enabled.

    Args:
        lora_paths (List[Path]): Paths to the LoRA checkpoint files.
        dare (bool): Whether to apply DARE preprocessing.
        dare_density (Optional[float]): Density for DARE preprocessing.

    Returns:
        List[dict]: A list of loaded and potentially preprocessed state dictionaries.
    """
    lora_state_dicts = []
    for path in lora_paths:
        if not path.exists():
            typer.echo(f"Error: LoRA file not found: {{path}}", err=True)
            raise typer.Exit(code=1)
        if path.suffix.lower() != ".safetensors":
             typer.echo(f"Warning: File {{path}} does not have a .safetensors extension. Attempting to load anyway.", err=True)
        try:
            sd = load_file(str(path))
            if dare and dare_density is not None:
                 sd = apply_dare_preprocessing(sd, dare_density)
            lora_state_dicts.append(sd)
        except Exception as e:
            typer.echo(f"Error loading or preprocessing {{path}}: {{e}}", err=True)
            raise typer.Exit(code=1)
    return lora_state_dicts

def _validate_merge_parameters(
    lora_paths: List[Path],
    weights: Optional[List[float]],
    merge_technique: str,
    density: Optional[float],
    majority_sign_method: Optional[str],
    dare_density: Optional[float],
    dare: bool
) -> Optional[List[float]]:
    """
    Validates the provided merging parameters based on the chosen technique.

    Args:
        lora_paths (List[Path]): Paths to the LoRA checkpoint files.
        weights (Optional[List[float]]): Optional weights for each LoRA adapter.
        merge_technique (str): Merging technique to use.
        density (Optional[float]): Density parameter for techniques like TIES.
        majority_sign_method (Optional[str]): Majority sign method for TIES.
        dare_density (Optional[float]): Density parameter for DARE preprocessing.
        dare (bool): Whether to apply DARE preprocessing.

    Returns:
        Optional[List[float]]: The validated or default weights if applicable, otherwise None.
    """
    if len(lora_paths) < 2:
        typer.echo("Error: Please provide at least two LoRA paths to merge.", err=True)
        raise typer.Exit(code=1)

    if weights and len(weights) != len(lora_paths):
        typer.echo("Error: Number of weights must match the number of LoRA paths.", err=True)
        raise typer.Exit(code=1)

    # Validate parameters based on merge technique
    if merge_technique == "ties":
        if len(lora_paths) != 2:
             typer.echo("Error: 'ties' merge technique currently only supports merging exactly two LoRAs.", err=True)
             raise typer.Exit(code=1)
        if density is None:
            typer.echo("Error: Density is required for 'ties' technique.", err=True)
            raise typer.Exit(code=1)
        if not (0.0 <= density <= 1.0):
            typer.echo("Error: Density must be between 0.0 and 1.0 for 'ties' technique.", err=True)
            raise typer.Exit(code=1)
        if majority_sign_method not in ["total", "frequency"]:
             typer.echo("Error: majority_sign_method must be 'total' or 'frequency' for 'ties' technique.", err=True)
             raise typer.Exit(code=1)
        if weights is None:
             # TIES usually uses equal weighting if not specified
             weights = [0.5, 0.5]
        elif len(weights) != 2:
             typer.echo("Error: Please provide exactly two weights for 'ties' technique.", err=True)
             raise typer.Exit(code=1)

    # Validate DARE density if provided
    if dare_density is not None:
        if merge_technique not in ["linear", "ties"]:
            typer.echo(f"Warning: DARE preprocessing is typically used with 'linear' or 'ties' merge techniques. It may not be suitable for '{{merge_technique}}'.", err=True)
        if not (0.0 <= dare_density <= 1.0):
            typer.echo("Error: DARE density must be between 0.0 and 1.0.", err=True)
            raise typer.Exit(code=1)

    # Use default weights if none provided (for linear/cat when not specified) and return them
    if weights is None and merge_technique in ["linear", "cat"]:
        return [1.0] * len(lora_paths)

    return weights # Return the provided or default TIES/validated weights

def _merge_linear(
    lora_state_dicts: List[dict],
    weights: List[float]
) -> dict:
    """
    Performs linear merging (weighted sum) of LoRA state dictionaries.

    Args:
        lora_state_dicts (List[dict]): List of loaded LoRA state dictionaries.
        weights (List[float]): List of weights for linear merging.

    Returns:
        dict: The merged state dictionary.
    """
    typer.echo("Using 'linear' merge technique (weighted sum of A and B matrices).")
    merged_state_dict = {}
    first_sd = lora_state_dicts[0]

    for key in first_sd.keys():
        if not all(key in sd for sd in lora_state_dicts):
            typer.echo(f"Warning: Key '{{key}}' not found in all LoRA files. Copying from first.", err=True)
            merged_state_dict[key] = first_sd[key]
            continue

        is_lora_a = ".lora_A.default.weight" in key
        is_lora_b = ".lora_B.default.weight" in key

        if is_lora_a or is_lora_b:
            try:
                merged_tensor = torch.zeros_like(first_sd[key])
                for i, sd in enumerate(lora_state_dicts):
                    merged_tensor += weights[i] * sd[key]
                merged_state_dict[key] = merged_tensor
            except Exception as e:
                typer.echo(f"Error merging key '{{key}}': {{e}}. Skipping.", err=True)
                merged_state_dict[key] = first_sd[key]
        else:
             merged_state_dict[key] = first_sd[key]

    return merged_state_dict

def _merge_cat(
    lora_state_dicts: List[dict],
    weights: List[float]
) -> dict:
    """
    Performs concatenation ('cat') merging of exactly two LoRA state dictionaries.

    Args:
        lora_state_dicts (List[dict]): List containing exactly two loaded LoRA state dictionaries.
        weights (List[float]): List containing exactly two weights for concatenation.

    Returns:
        dict: The merged state dictionary.
    """
    typer.echo("Using 'cat' merge technique (concatenation of A and B matrices).")
    merged_state_dict = {}

    # Validation for 'cat' (already partly done in _validate_merge_parameters, but defensive here)
    if len(lora_state_dicts) != 2:
        raise ValueError("'cat' merge requires exactly two LoRAs.")
    if len(weights) != 2:
         raise ValueError("'cat' merge requires exactly two weights.")

    lora1_sd = lora_state_dicts[0]
    lora2_sd = lora_state_dicts[1]
    weight1 = weights[0]
    weight2 = weights[1]

    for key in lora1_sd.keys():
         if key not in lora2_sd:
             typer.echo(f"Warning: Key '{{key}}' not found in the second LoRA file for 'cat' merge. Copying from first.", err=True)
             merged_state_dict[key] = lora1_sd[key]
             continue

         is_lora_a = ".lora_A.default.weight" in key
         is_lora_b = ".lora_B.default.weight" in key

         if is_lora_a:
             try:
                 tensor1 = weight1 * lora1_sd[key]
                 tensor2 = weight2 * lora2_sd[key]
                 merged_tensor = torch.cat((tensor1, tensor2), dim=0)
                 merged_state_dict[key] = merged_tensor
             except Exception as e:
                 typer.echo(f"Error concatenating key '{{key}}' (A): {{e}}. Skipping.", err=True)
                 merged_state_dict[key] = lora1_sd[key]
         elif is_lora_b:
              try:
                 tensor1 = weight1 * lora1_sd[key]
                 tensor2 = weight2 * lora2_sd[key]
                 merged_tensor = torch.cat((tensor1, tensor2), dim=1)
                 merged_state_dict[key] = merged_tensor
              except Exception as e:
                 typer.echo(f"Error concatenating key '{{key}}' (B): {{e}}. Skipping.", err=True)
                 merged_state_dict[key] = lora1_sd[key]
         else:
             merged_state_dict[key] = lora1_sd[key]

    return merged_state_dict

def _merge_ties(
    lora_state_dicts: List[dict],
    weights: List[float],
    density: float,
    majority_sign_method: str
) -> dict:
    """
    Performs TIES merging of exactly two LoRA state dictionaries.

    Args:
        lora_state_dicts (List[dict]): List containing exactly two loaded LoRA state dictionaries.
        weights (List[float]): List containing exactly two weights for TIES.
        density (float): Density parameter for TIES.
        majority_sign_method (str): Majority sign method for TIES ('total' or 'frequency').

    Returns:
        dict: The merged state dictionary.
    """
    typer.echo(f"Using 'ties' merge technique with density={density} and method='{majority_sign_method}'.")
    typer.echo("Note: This implementation is a simplified adaptation of TIES principles applied directly to LoRA A and B factors.")
    merged_state_dict = {}

    # Validation for 'ties' (already partly done in _validate_merge_parameters, but defensive here)
    if len(lora_state_dicts) != 2:
        raise ValueError("'ties' merge requires exactly two LoRAs.")
    if len(weights) != 2:
         raise ValueError("'ties' merge requires exactly two weights.")
    if not (0.0 <= density <= 1.0):
         raise ValueError("Density must be between 0.0 and 1.0 for 'ties'.")
    if majority_sign_method not in ["total", "frequency"]:
         raise ValueError("majority_sign_method must be 'total' or 'frequency' for 'ties'.")

    lora1_sd = lora_state_dicts[0]
    lora2_sd = lora_state_dicts[1]
    weight1 = weights[0]
    weight2 = weights[1]

    for key in lora1_sd.keys(): # Iterate through keys of the first SD (assuming same structure)
         if key not in lora2_sd:
             typer.echo(f"Warning: Key '{{key}}' not found in both LoRA files for 'ties' merge. Copying from first.", err=True)
             merged_state_dict[key] = lora1_sd[key]
             continue

         is_lora_a = ".lora_A.default.weight" in key
         is_lora_b = ".lora_B.default.weight" in key

         if is_lora_a or is_lora_b:
                 try:
                     tensor1 = lora1_sd[key]
                     tensor2 = lora2_sd[key]

                     # Apply weights
                     weighted_tensor1 = weight1 * tensor1
                     weighted_tensor2 = weight2 * tensor2

                     # Apply density thresholding to weighted tensors separately based on magnitude
                     num_elements = weighted_tensor1.numel() # Assuming same size for both
                     k = int(density * num_elements)

                     if k == 0 and density > 0:
                         typer.echo(f"Warning: Calculated number of elements to keep for density {{density}} is 0 for key '{{key}}'. No elements will be kept.", err=True)
                         thresholded_tensor1 = torch.zeros_like(weighted_tensor1)
                         thresholded_tensor2 = torch.zeros_like(weighted_tensor2)
                     elif k >= num_elements:
                         # If density is 1.0 or more, keep all elements (no thresholding)
                         thresholded_tensor1 = weighted_tensor1
                         thresholded_tensor2 = weighted_tensor2
                     else:
                          # Threshold tensor 1
                          flattened_t1 = weighted_tensor1.flatten()
                          magnitudes_t1 = torch.abs(flattened_t1)
                          threshold_t1 = torch.topk(magnitudes_t1, k)[0][-1]
                          threshold_mask_t1 = torch.abs(weighted_tensor1) >= threshold_t1
                          thresholded_tensor1 = weighted_tensor1 * threshold_mask_t1

                          # Threshold tensor 2
                          flattened_t2 = weighted_tensor2.flatten()
                          magnitudes_t2 = torch.abs(flattened_t2)
                          threshold_t2 = torch.topk(magnitudes_t2, k)[0][-1]
                          threshold_mask_t2 = torch.abs(weighted_tensor2) >= threshold_t2
                          thresholded_tensor2 = weighted_tensor2 * threshold_mask_t2


                     # Compute the majority sign mask based on the chosen method
                     if majority_sign_method == 'frequency':
                         sign_mask = torch.zeros_like(thresholded_tensor1)
                         positive_mask = (torch.sign(weighted_tensor1) == 1) & (torch.sign(weighted_tensor2) == 1)
                         negative_mask = (torch.sign(weighted_tensor1) == -1) & (torch.sign(weighted_tensor2) == -1)
                         zero_mask_t1 = (torch.sign(weighted_tensor1) == 0)
                         zero_mask_t2 = (torch.sign(weighted_tensor2) == 0)

                         sign_mask[positive_mask] = 1
                         sign_mask[negative_mask] = -1
                         sign_mask[zero_mask_t1 & ~zero_mask_t2] = torch.sign(weighted_tensor2[zero_mask_t1 & ~zero_mask_t2])
                         sign_mask[~zero_mask_t1 & zero_mask_t2] = torch.sign(weighted_tensor1[~zero_mask_t1 & zero_mask_t2])

                         disagree_mask = (torch.sign(weighted_tensor1) != torch.sign(weighted_tensor2)) & (~zero_mask_t1 & ~zero_mask_t2)
                         sign_mask[disagree_mask] = 0

                     elif majority_sign_method == 'total':
                          total_sign_t1 = torch.sum(torch.sign(weighted_tensor1))
                          total_sign_t2 = torch.sum(torch.sign(weighted_tensor2))

                          sign_sum_total = total_sign_t1 + total_sign_t2
                          if sign_sum_total > 0:
                               sign_mask = torch.ones_like(thresholded_tensor1)
                          elif sign_sum_total < 0:
                               sign_mask = -torch.ones_like(thresholded_tensor1)
                          else:
                               sign_mask = torch.zeros_like(thresholded_tensor1)


                     # Sum the individually thresholded tensors
                     sum_thresholded = thresholded_tensor1 + thresholded_tensor2

                     # Apply the sign mask to the sum of thresholded tensors
                     merged_tensor = sum_thresholded * sign_mask

                     merged_state_dict[key] = merged_tensor

                 except Exception as e:
                     typer.echo(f"Error merging key '{key}' with 'ties' technique: {{e}}. Skipping or copying from first.", err=True)
                     merged_state_dict[key] = lora1_sd[key]

         else:
            # Non-LoRA key, copy from the first state dict
            merged_state_dict[key] = lora1_sd[key]

    return merged_state_dict


@app.command()
def main(
    lora_paths: List[Path] = typer.Argument(
        ...,
        help="Paths to the LoRA checkpoint files to merge (e.g., lora1.safetensors lora2.safetensors). "
        "Provide at least two paths.",
    ),
    output_path: Path = typer.Argument(
        ...,
        help="Path to save the merged LoRA checkpoint file (e.g., merged_lora.safetensors).",
    ),
    weights: Optional[List[float]] = typer.Option(
        None,
        "--weights",
        "-w",
        help="Optional weights for each LoRA adapter. Must match the number of lora_paths. "
        "If not provided, equal weights (1.0) are assumed.",
    ),
    merge_technique: str = typer.Option(
        "linear",
        "--technique",
        "-t",
        help="Merging technique to use. Options: 'linear', 'cat', 'ties'. "
        "Note: 'dare' (as a standalone technique) and 'svd' are more complex and not implemented.",
    ),
    density: Optional[float] = typer.Option(
        None,
        "--density",
        "-d",
        help="Density parameter (0.0 to 1.0) for techniques like TIES.",
    ),
    majority_sign_method: Optional[str] = typer.Option(
        None,
        "--majority-sign-method",
        "-m",
        help="Majority sign method for TIES ('total' or 'frequency').",
    ),
    dare_density: Optional[float] = typer.Option(
        None,
        "--dare-density",
        help="Optional Density parameter (0.0 to 1.0) for DARE preprocessing, applied before merging.",
    ),
    dare: bool = typer.Option(
        False,
        "--dare",
        help="Whether to apply DARE preprocessing before merging.",
    ),

) -> None:
    """
    Merges multiple LoRA adapters into a single file.

    Inputs:
    - lora_paths (List[Path]): List of paths to the input LoRA .safetensors files.
    - output_path (Path): Path where the merged .safetensors file will be saved.
    - weights (Optional[List[float]]): List of weights for weighted merging. Defaults to [1.0, 1.0, ...].
    - merge_technique (str): The method to use for merging ('linear', 'cat', 'ties').
    - density (Optional[float]): Density parameter (0.0 to 1.0) for techniques like TIES.
    - majority_sign_method (Optional[str]): Majority sign method for TIES ('total' or 'frequency').
    - dare_density (Optional[float]): Density parameter (0.0 to 1.0) for DARE preprocessing.
    - dare (bool): Whether to apply DARE preprocessing before merging.

    Outputs:
    - A single .safetensors file at the specified output_path containing the merged LoRA weights.
    """

    # 1. Validate Parameters
    # _validate_merge_parameters will exit if validation fails
    # and return the potentially adjusted weights for linear/cat
    weights = _validate_merge_parameters(
        lora_paths,
        weights,
        merge_technique,
        density,
        majority_sign_method,
        dare_density,
        dare
    )


    typer.echo(f"Merging LoRAs: {lora_paths} into {output_path} using technique: {merge_technique}")

    # 2. Load LoRA state dictionaries (with optional DARE preprocessing)
    lora_state_dicts = _load_lora_files(lora_paths, dare, dare_density)


    # 3. Perform Merging based on technique
    merged_state_dict = {}
    if merge_technique == "linear":
        merged_state_dict = _merge_linear(lora_state_dicts, weights)

    elif merge_technique == "cat":
        # Cat merge requires exactly two LoRAs, already validated.
        merged_state_dict = _merge_cat(lora_state_dicts, weights)

    elif merge_technique == "ties":
        # TIES merge requires exactly two LoRAs, density, and method, already validated.
        merged_state_dict = _merge_ties(
            lora_state_dicts,
            weights,
            density,
            majority_sign_method
        )

    elif merge_technique in ["dare", "svd"]:
        typer.echo(f"Using '{merge_technique}' merge technique.")
        typer.echo(f"TODO: Implement '{merge_technique}' merging logic.")
        typer.echo("This typically requires more complex operations (like SVD on delta weights) and might need more arguments.")
        raise NotImplementedError(f"'{merge_technique}' merging logic is a TODO and not implemented in this basic script.")


    else:
        # Should not happen if _validate_merge_parameters passes, but defensive.
        typer.echo(f"Error: Unsupported merge technique: {merge_technique}", err=True)
        typer.echo("Supported techniques are: linear, cat, ties.", err=True)
        raise typer.Exit(code=1)


    # 4. Save the merged state dictionary
    typer.echo(f"Saving merged LoRA weights to {output_path}")
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_file(merged_state_dict, str(output_path))
        typer.echo("Merging complete. Merged file saved.")
    except Exception as e:
        typer.echo(f"Error saving merged file to {output_path}: {{e}}", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
