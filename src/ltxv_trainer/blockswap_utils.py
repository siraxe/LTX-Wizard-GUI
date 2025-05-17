import torch
import gc
from functools import wraps

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

def blockswap_transformer_blocks(model, transformer_blocks_to_swap=0, device="cpu", offload_txt_in=False, offload_img_in=False):
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
            module_to_offload.forward = _create_patched_forward_for_cpu_module(module_to_offload.original_forward)
            modules_offloaded_and_patched.append("txt_in")

    if offload_img_in and hasattr(model, "img_in"):
        module_to_offload = model.img_in
        move_module_and_submodules_to_device(module_to_offload, device)
        if not hasattr(module_to_offload, 'original_forward'):
            module_to_offload.original_forward = module_to_offload.forward
            module_to_offload.forward = _create_patched_forward_for_cpu_module(module_to_offload.original_forward)
            modules_offloaded_and_patched.append("img_in")

    if modules_offloaded_and_patched:
        print(f"Offloaded and patched modules: {', '.join(modules_offloaded_and_patched)} to {device}.")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    else:
        print("No modules were offloaded or patched for CPU blockswap.") 