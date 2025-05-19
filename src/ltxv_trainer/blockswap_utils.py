import torch
import gc
from functools import wraps
from accelerate import init_empty_weights, dispatch_model, infer_auto_device_map
from torch.utils.checkpoint import checkpoint
import bitsandbytes.nn as bnb
from accelerate.big_modeling import attach_layerwise_casting_hooks

# Recursively move all submodules and buffers to the target device
def move_module_and_submodules_to_device(module, device):
    module.to(device)
    for child in module.children():
        move_module_and_submodules_to_device(child, device)
    for name, buf in module.named_buffers(recurse=False):
        setattr(module, name, buf.to(device))

# Helper to recursively move tensors in args/kwargs to a device
def _move_item_to_device(item, device):
    if isinstance(item, torch.Tensor):
        return item.to(device)
    elif isinstance(item, tuple):
        return tuple(_move_item_to_device(x, device) for x in item)
    elif isinstance(item, list):
        return [_move_item_to_device(x, device) for x in item]
    # Add dict handling if necessary, though typically model inputs aren't complex dicts of tensors
    # elif isinstance(item, dict):
    #     return {k: _move_item_to_device(v, device) for k, v in item.items()}
    return item

# Helper to create the patched forward method
def _create_patched_forward_for_cpu_module(original_forward):
    cpu_device = torch.device("cpu")

    @wraps(original_forward)
    def patched_forward(*args, **kwargs):
        input_device = None # We'll try to determine this to send output back correctly
        cpu_device = torch.device("cpu")

        # Determine input_device from the first *original* tensor encountered
        # We need to iterate through the original args/kwargs before they are moved to new_args/new_kwargs
        for original_arg in args:
            if isinstance(original_arg, torch.Tensor):
                input_device = original_arg.device
                break
            elif isinstance(original_arg, (tuple, list)):
                for item_in_coll in original_arg:
                    if isinstance(item_in_coll, torch.Tensor):
                        input_device = item_in_coll.device
                        break
                if input_device is not None:
                    break
        if input_device is None:
            for k_orig, v_orig in kwargs.items():
                if isinstance(v_orig, torch.Tensor):
                    input_device = v_orig.device
                    break
                elif isinstance(v_orig, (tuple, list)):
                    for item_in_coll in v_orig:
                        if isinstance(item_in_coll, torch.Tensor):
                            input_device = item_in_coll.device
                            break
                    if input_device is not None:
                        break
        
        if input_device is None and torch.cuda.is_available():
            print(f"[Blockswap Debug] No input tensor device found, defaulting to cuda:0 for {original_forward.__qualname__}")
            input_device = torch.device("cuda:0")
        elif input_device is None:
            print(f"[Blockswap Debug] No input tensor device found and CUDA not available, defaulting to CPU for {original_forward.__qualname__}")
            input_device = cpu_device # Fallback if no tensors and no CUDA

        print(f"[Blockswap Debug] Patched forward for {original_forward.__qualname__}: Determined original input_device: {input_device}")

        # Move tensor arguments and contents of tuples/lists to CPU
        new_args = tuple(_move_item_to_device(arg, cpu_device) for arg in args)
        new_kwargs = {k: _move_item_to_device(v, cpu_device) for k, v in kwargs.items()}
        
        # Special dtype handling for encoder_attention_mask for offloaded blocks on CPU
        # The LTXVideoTransformerBlock.forward receives encoder_attention_mask as a kwarg.
        # F.scaled_dot_product_attention on CPU expects mask to be bool, float32, or match query (bfloat16).
        # If mask becomes float16 internally on CPU from an initial bfloat16, it causes an error.
        # Casting to float32 here makes it robust.
        if 'encoder_attention_mask' in new_kwargs:
            mask_tensor = new_kwargs['encoder_attention_mask']
            if isinstance(mask_tensor, torch.Tensor):
                # Get query dtype (bfloat16) from hidden_states (new_args[1] for LTXVideoTransformerBlock)
                query_dtype = None
                if len(new_args) > 1 and isinstance(new_args[1], torch.Tensor):
                    query_dtype = new_args[1].dtype
                
                # If mask is not bool and not matching query_dtype, cast to float32
                # This specifically targets the bfloat16 -> float16 issue on CPU for the mask.
                if mask_tensor.dtype != torch.bool and (query_dtype is None or mask_tensor.dtype != query_dtype):
                    print(f"[Blockswap Debug] Casting encoder_attention_mask from {mask_tensor.dtype} to torch.float32 for CPU block {original_forward.__qualname__}.")
                    new_kwargs['encoder_attention_mask'] = mask_tensor.to(torch.float32)

        print(f"[Blockswap Debug] Calling original_forward on CPU for {original_forward.__qualname__}...")
        output = original_forward(*new_args, **new_kwargs)

        # Log device of output before moving back
        if isinstance(output, torch.Tensor):
            print(f"[Blockswap Debug] Output from {original_forward.__qualname__} (on CPU) device: {output.device}")
        elif isinstance(output, (tuple, list)):
            for i_out, out_item in enumerate(output):
                if isinstance(out_item, torch.Tensor):
                    print(f"[Blockswap Debug] Output item {i_out} from {original_forward.__qualname__} (on CPU) device: {out_item.device}")
                    break # Just need one to confirm general location

        # Move tensor outputs and contents of tuples/lists back to the original input_device
        if input_device and input_device != cpu_device:
            print(f"[Blockswap Debug] Moving output of {original_forward.__qualname__} from CPU back to {input_device}")
            output = _move_item_to_device(output, input_device)
            # Verification print after moving back
            if isinstance(output, torch.Tensor):
                print(f"[Blockswap Debug] Output of {original_forward.__qualname__} (after move) device: {output.device}")
            elif isinstance(output, (tuple, list)):
                for i_out, out_item in enumerate(output):
                    if isinstance(out_item, torch.Tensor):
                        print(f"[Blockswap Debug] Output item {i_out} of {original_forward.__qualname__} (after move) device: {out_item.device}")
                        break # Just need one
        elif not input_device:
            print(f"[Blockswap Warning] No input_device determined for {original_forward.__qualname__}, output remains on CPU.")
        elif input_device == cpu_device:
            print(f"[Blockswap Debug] Input_device for {original_forward.__qualname__} was CPU, output remains on CPU.")
        
        return output
    return patched_forward

# Helper to create a patched forward method with checkpointing
def _create_patched_forward_with_checkpointing(original_forward):
    @wraps(original_forward)
    def patched_forward(*args, **kwargs):
        # Apply torch.utils.checkpoint.checkpoint to the original forward pass
        # set use_reentrant=False for better compatibility and performance with complex models/autograd
        return checkpoint(original_forward, *args, use_reentrant=False, **kwargs)
    return patched_forward

def blockswap_transformer_blocks(model, transformer_blocks_to_swap=0, device="cpu", offload_txt_in=False, offload_img_in=False, method_index=5):
    list_of_functions = [
        # Add your different blockswap implementations here as you create them
        # For now, this list contains the original logic renamed to _01
        blockswap_transformer_blocks_01,
        blockswap_transformer_blocks_02, # Accelerate offloading
        blockswap_transformer_blocks_03, # Gradient Checkpointing
        blockswap_transformer_blocks_04, # AMP placeholder
        blockswap_transformer_blocks_05, # BitsAndBytes 8-bit Quantization
        blockswap_transformer_blocks_06, # Accelerate Layerwise Casting
        # ...
    ]

    if not (0 <= method_index < len(list_of_functions)):
        print(f"[Blockswap Error] Invalid method_index: {method_index}. Available methods: 0 to {len(list_of_functions) - 1}. Using method 0.")
        method_index = 0

    selected_function = list_of_functions[method_index]
    print(f"[Blockswap Info] Using method index {method_index}: {selected_function.__name__}")

    return selected_function(model, transformer_blocks_to_swap, device, offload_txt_in, offload_img_in)

# Rename the original function to _01
def blockswap_transformer_blocks_01(model, transformer_blocks_to_swap=0, device="cpu", offload_txt_in=False, offload_img_in=False):
    if device != "cpu":
        # Simplified parameter move for non-CPU targets (or handle error)
        if hasattr(model, "transformer_blocks"):
            num_blocks_total = len(model.transformer_blocks)
            for i in range(num_blocks_total - transformer_blocks_to_swap, num_blocks_total):
                if 0 <= i < num_blocks_total:
                    move_module_and_submodules_to_device(model.transformer_blocks[i], device)
        if offload_txt_in and hasattr(model, "txt_in"): move_module_and_submodules_to_device(model.txt_in, device)
        if offload_img_in and hasattr(model, "img_in"): move_module_and_submodules_to_device(model.img_in, device)
        return

    # --- CPU Offloading with Forward Pass Patching ---
    modules_offloaded_and_patched = []

    if hasattr(model, "transformer_blocks") and transformer_blocks_to_swap > 0:
        num_blocks_total = len(model.transformer_blocks)
        start_index_to_offload = num_blocks_total - transformer_blocks_to_swap
        
        for i in range(start_index_to_offload, num_blocks_total):
            if 0 <= i < num_blocks_total:
                block = model.transformer_blocks[i]
                move_module_and_submodules_to_device(block, device) # Move block to CPU
                if not hasattr(block, 'original_forward'): 
                    block.original_forward = block.forward
                    block.forward = _create_patched_forward_for_cpu_module(block.original_forward)
                    modules_offloaded_and_patched.append(f"block_{i}")

    if offload_txt_in and hasattr(model, "txt_in"):
        module_to_offload = model.txt_in
        move_module_and_submodules_to_device(module_to_offload, device)
        if not hasattr(module_to_offload, 'original_forward'):
            module_to_offload.original_forward = module_to_offload.forward
            module_to_offload.forward = _create_patched_forward_for_cpu_module(module_to_offload.forward)
            modules_offloaded_and_patched.append("txt_in")

    if offload_img_in and hasattr(model, "img_in"):
        module_to_offload = model.img_in
        move_module_and_submodules_to_device(module_to_offload, device)
        if not hasattr(module_to_offload, 'original_forward'):
            module_to_offload.original_forward = module_to_offload.forward
            module_to_offload.forward = _create_patched_forward_for_cpu_module(module_to_offload.forward)
            modules_offloaded_and_patched.append("img_in")

    if modules_offloaded_and_patched:
        print(f"Offloaded and patched modules: {', '.join(modules_offloaded_and_patched)} to {device}.")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    else:
        print("No modules were offloaded or patched for CPU blockswap.") 

def blockswap_transformer_blocks_02(model, transformer_blocks_to_swap=0, device="cpu", offload_txt_in=False, offload_img_in=False):
    print("[Blockswap Info] Using Accelerate offloading method (blockswap_transformer_blocks_02).")

    if device != "cpu":
        print("[Blockswap Warning] Accelerate offloading method 02 is primarily for CPU offloading. For non-CPU devices, manual move is assumed (handled by method 01 if needed).")
        # Optionally, you could still use accelerate's dispatch for non-CPU multi-GPU setups if needed.
        # For this example, we'll focus on the CPU offload case as per the user's issue.
        return # Exit if not targeting CPU offload

    # Accelerate needs an empty model to dispatch weights into
    # We assume the model is already loaded, so we'll try to dispatch the existing model
    # This might not be the standard accelerate workflow (which often loads into empty weights)
    # but attempts to adapt it to your existing loaded model.

    try:
        # Determine device map. 'auto' lets accelerate figure out the best mapping
        # based on available memory. We prioritize CPU offloading here.
        # We can try to guide accelerate to offload the specified blocks
        # This requires knowing the structure of your model and module names.
        # Assuming 'transformer_blocks' is a list/ModuleList and 'txt_in', 'img_in' are submodules.

        device_map = None
        if transformer_blocks_to_swap > 0 or offload_txt_in or offload_img_in:
            print(f"[Blockswap Info] Attempting to create a device map for offloading...")
            # Create a device map that suggests offloading specified parts to 'cpu'
            device_map_dict = {}
            if hasattr(model, "transformer_blocks"):
                num_blocks_total = len(model.transformer_blocks)
                start_index_to_offload = num_blocks_total - transformer_blocks_to_swap
                for i in range(start_index_to_offload, num_blocks_total):
                     if 0 <= i < num_blocks_total:
                          # Assuming transformer blocks are named something like model.transformer_blocks.N
                          # You might need to adjust the key based on your actual model's naming
                          block_name = f"transformer_blocks.{i}"
                          device_map_dict[block_name] = "cpu"

            if offload_txt_in and hasattr(model, "txt_in"):
                # Assuming txt_in is named 'txt_in' within the model
                device_map_dict["txt_in"] = "cpu"

            if offload_img_in and hasattr(model, "img_in"):
                # Assuming img_in is named 'img_in' within the model
                device_map_dict["img_in"] = "cpu"

            if not device_map_dict:
                print("[Blockswap Info] No specific modules specified for offloading. Accelerate will attempt automatic offloading.")
                # Use 'auto' if no specific offload targets are given
                calculated_device_map = "auto"
            else:
                print(f"[Blockswap Info] Generated explicit device map suggestions: {device_map_dict}")
                # Use infer_auto_device_map with explicit suggestions and 'balanced' strategy
                # This requires the model to be on a device initially to calculate sizes.
                # Assuming model is on GPU before calling this function.
                # This might still place some non-specified layers on CPU if needed for balance.
                try:
                    calculated_device_map = infer_auto_device_map(model, device_map=device_map_dict, no_split_module_classes=["LTXVideoTransformerBlock"], dtype=model.dtype if hasattr(model, 'dtype') else None, style="balanced")
                    print(f"[Blockswap Info] Final balanced device map: {calculated_device_map}")
                except Exception as e:
                    print(f"[Blockswap Error] Could not create balanced device map using infer_auto_device_map: {e}")
                    calculated_device_map = None # Indicate failure

        else:
             # If no specific modules are targeted, let accelerate decide based on 'auto'
             print("[Blockswap Info] No specific modules specified for offloading. Accelerate will attempt automatic offloading ('auto').")
             calculated_device_map = "auto"

        if calculated_device_map is None:
             print("[Blockswap Error] Could not determine a device map for Accelerate offloading.")
             return

         # Dispatch the model according to the device map
         # This moves the model parameters to the specified devices
         # Accelerate also patches the forward pass internally to handle device movements during computation.
        model = dispatch_model(model, device_map=calculated_device_map)
        print("[Blockswap Info] Model dispatched using Accelerate.")

    except Exception as e:
        print(f"[Blockswap Error] Failed to apply Accelerate offloading: {e}")
        # You might want to fall back to the original method or raise an error

    # Accelerate handles cache clearing internally to some extent, but manual clear is still good practice.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    print("[Blockswap Info] Accelerate offloading setup complete.") 

def blockswap_transformer_blocks_03(model, transformer_blocks_to_swap=0, device="cpu", offload_txt_in=False, offload_img_in=False):
    print("[Blockswap Info] Using Gradient Checkpointing method (blockswap_transformer_blocks_03).")

    if device != "cpu":
        print("[Blockswap Warning] Gradient Checkpointing method 03 is typically for saving GPU memory during training. The 'device' parameter is less relevant here.")

    modules_checkpointed = []

    if hasattr(model, "transformer_blocks") and transformer_blocks_to_swap > 0:
        num_blocks_total = len(model.transformer_blocks)
        # Apply checkpointing to the *first* transformer_blocks_to_swap blocks for maximum activation memory saving
        end_index_to_checkpoint = min(transformer_blocks_to_swap, num_blocks_total)

        for i in range(end_index_to_checkpoint):
            if 0 <= i < num_blocks_total:
                block = model.transformer_blocks[i]
                if not hasattr(block, 'original_forward'): # Avoid double patching if method 01 was used previously
                    block.original_forward = block.forward
                    block.forward = _create_patched_forward_with_checkpointing(block.original_forward)
                    modules_checkpointed.append(f"block_{i}")
                else:
                    print(f"[Blockswap Info] Block {i} already has a patched forward (likely from method 01). Skipping checkpointing patch.")

    if offload_txt_in or offload_img_in:
         print("[Blockswap Info] Checkpointing method 03 does not apply to txt_in or img_in directly. Only transformer blocks are checkpointed.")

    if modules_checkpointed:
        print(f"Checkpointing applied to modules: {', '.join(modules_checkpointed)}.")
    else:
        print("No modules were checkpointed using method 03.")

    # Checkpointing doesn't typically require explicit cache clearing/GC like offloading weights
    # as it saves activation memory during the backward pass. 

def blockswap_transformer_blocks_04(model, transformer_blocks_to_swap=0, device="cpu", offload_txt_in=False, offload_img_in=False):
    print("[Blockswap Info] Using Mixed Precision Training (AMP) method (blockswap_transformer_blocks_04).")

    print("[Blockswap Info] Note: This method requires modifications in your training loop.")
    print("To enable AMP, you will need to:")
    print("1. Initialize a GradScaler: `scaler = torch.cuda.amp.GradScaler()`")
    print("2. Wrap your forward pass in `autocast`: `with torch.cuda.amp.autocast(): output = model(input)`")
    print("3. Scale the loss before backward: `scaler.scale(loss).backward()`")
    print("4. Step the optimizer and update the scaler: `scaler.step(optimizer); scaler.update()`")
    print("[Blockswap Info] The arguments transformer_blocks_to_swap, device, offload_txt_in, offload_img_in are not directly used by this method.")

    # Return a flag or the scaler object if needed by the training script structure
    # For now, just returning None as it's primarily an instructional method here.
    return None

def blockswap_transformer_blocks_05(model, transformer_blocks_to_swap=0, device="cpu", offload_txt_in=False, offload_img_in=False):
    print("[Blockswap Info] Using BitsAndBytes 8-bit Quantization method (blockswap_transformer_blocks_05).")

    if device != "cpu":
         print("[Blockswap Warning] BitsAndBytes 8-bit quantization is typically applied to layers intended for GPU computation to save VRAM.")

    modules_quantized = []

    target_modules = []
    if hasattr(model, "transformer_blocks") and transformer_blocks_to_swap > 0:
        # Quantize the *entire* transformer_blocks ModuleList or iterate? Let's iterate for granularity.
        # We'll quantize the last 'transformer_blocks_to_swap' blocks as per the original method's logic.
        num_blocks_total = len(model.transformer_blocks)
        start_index_to_quantize = num_blocks_total - transformer_blocks_to_swap
        for i in range(start_index_to_quantize, num_blocks_total):
            if 0 <= i < num_blocks_total:
                target_modules.append((model.transformer_blocks, i, f"transformer_blocks[{i}]"))

    if offload_txt_in and hasattr(model, "txt_in"):
         target_modules.append((model, "txt_in", "txt_in")) # Assuming txt_in is a direct attribute of model

    if offload_img_in and hasattr(model, "img_in"):
         target_modules.append((model, "img_in", "img_in")) # Assuming img_in is a direct attribute of model

    # Recursively search for Linear layers within the target modules and replace them
    def find_and_quantize_linear(module, name=""):
        for child_name, child_module in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name
            if isinstance(child_module, torch.nn.Linear):
                try:
                    # Replace Linear with bnb.Linear8bitLt
                    # Note: This requires bitsandbytes to be correctly installed and configured
                    # on a CUDA-enabled system for 8-bit ops.
                    quantized_linear = bnb.Linear8bitLt(child_module.in_features,
                                                      child_module.out_features,
                                                      bias=child_module.bias is not None,
                                                      has_fp16_weights=False, # Typically True if using mixed precision elsewhere
                                                      threshold=6.0)
                    # Copy weights
                    quantized_linear.weight.data = child_module.weight.data
                    if child_module.bias is not None:
                        quantized_linear.bias.data = child_module.bias.data

                    # Set the quantized layer back to the parent module
                    setattr(module, child_name, quantized_linear)
                    modules_quantized.append(full_name)
                    print(f"[Blockswap Info] Quantized layer: {full_name}")
                except Exception as e:
                    print(f"[Blockswap Error] Could not quantize layer {full_name}: {e}")
            else:
                find_and_quantize_linear(child_module, full_name)

    if not target_modules:
        print("[Blockswap Info] No specific modules targeted for quantization. Applying quantization to the entire model.")
        # Apply to the entire model if no specific modules are given
        find_and_quantize_linear(model, "model")
    else:
        print(f"[Blockswap Info] Applying quantization to specified modules: {[name for _, _, name in target_modules]}")
        for parent_module, key, name in target_modules:
             # If target is a specific block in a list/ModuleList
             if isinstance(parent_module, torch.nn.ModuleList) and isinstance(key, int):
                  find_and_quantize_linear(parent_module[key], name)
             # If target is a direct attribute of the parent module
             elif isinstance(parent_module, torch.nn.Module) and isinstance(key, str):
                  if hasattr(parent_module, key):
                       find_and_quantize_linear(getattr(parent_module, key), name)
                  else:
                       print(f"[Blockswap Warning] Module {key} not found in {parent_module.__class__.__name__}.")

    if modules_quantized:
        print(f"BitsAndBytes 8-bit quantization applied to: {', '.join(modules_quantized)}.")
        print("[Blockswap Info] Model size should now be significantly smaller in VRAM.")
    else:
        print("[Blockswap Info] No quantizable layers found in the target modules, or quantization failed.")

    # Quantization modifies the model in place and doesn't require explicit cache clearing for memory savings
    # as the layer weights themselves are stored in 8-bit.

def blockswap_transformer_blocks_06(model, transformer_blocks_to_swap=0, device="cpu", offload_txt_in=False, offload_img_in=False):
    print("[Blockswap Info] Using Accelerate Layerwise Casting method (blockswap_transformer_blocks_06).")

    # Determine compute dtype. Assuming bfloat16 is preferred if available, otherwise float16.
    compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    storage_dtype = torch.float8_e4m3fn # Use FP8 for storage as mentioned in Accelerate docs

    print(f"[Blockswap Info] Applying layerwise casting with storage_dtype={storage_dtype} and compute_dtype={compute_dtype}.")
    print("[Blockswap Info] This method modifies layer weights for memory savings directly.")

    try:
        # Applying layerwise casting to the entire model.
        # The arguments transformer_blocks_to_swap, offload_txt_in, offload_img_in
        # are not directly used to target specific modules in this general implementation.
        # If you need to apply this only to specific blocks, you would need a more complex
        # selection logic before applying the hook.
        attach_layerwise_casting_hooks(
            model,
            storage_dtype=storage_dtype,
            compute_dtype=compute_dtype,
            # You can add skip_modules_pattern or skip_modules_classes here if needed
            # skip_modules_pattern=["norm"], # Example: skip normalization layers
        )
        print("[Blockswap Info] Layerwise casting hooks attached.")

    except Exception as e:
        print(f"[Blockswap Error] Failed to apply Layerwise Casting: {e}")
        print("[Blockswap Error] Ensure your Accelerate version supports this feature and dtypes are compatible.")

    # Layerwise casting primarily affects how weights are handled internally and doesn't
    # typically require explicit cache clearing like full model offloading.
