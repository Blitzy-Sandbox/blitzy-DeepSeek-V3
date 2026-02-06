"""FP8-to-BF16 weight dequantization utility for DeepSeek-V3 inference checkpoints.

This module converts FP8 (float8_e4m3fn) quantized checkpoint files to BF16
(bfloat16) precision by performing block-wise dequantization. It processes
safetensors shard files from a HuggingFace-format checkpoint directory, applies
per-block scale factor correction to recover BF16 fidelity, and writes a new
checkpoint directory with all weights in BF16 format.

The actual dequantization is delegated to ``kernel.weight_dequant()``, which
launches a Triton GPU kernel that multiplies each 128×128 block of the FP8
weight matrix by its corresponding reciprocal scale factor stored in the
companion ``_scale_inv`` tensor. The 128×128 block granularity and the FP8
E4M3 format (4-bit exponent, 3-bit mantissa, max representable value 448) are
documented in ``README_WEIGHTS.md`` and originate from the block-wise
quantization scheme described in the DeepSeek-V3 technical report.

This utility is needed when the target deployment environment lacks FP8 kernel
support (e.g., pre-Hopper GPUs) and BF16 inference is preferred. Users running
on hardware with native FP8 support can skip this conversion and load the FP8
checkpoint directly.

Module dependency:
    - Imports ``weight_dequant`` from ``kernel.py`` for GPU-accelerated
      block-wise dequantization via Triton kernels.

Reference:
    arXiv:2412.19437, Section 3.3 — FP8 mixed-precision training framework
    and block-wise quantization scheme used to produce the FP8 checkpoints
    that this utility converts.

See also:
    - ``README_WEIGHTS.md`` — Documents the FP8 weight file structure,
      E4M3 format, 128×128 block size, and ``_scale_inv`` tensor layout.
    - ``kernel.py`` — Contains the ``weight_dequant()`` wrapper and
      ``weight_dequant_kernel`` Triton kernel that perform the actual
      block-wise dequantization.
"""

import os
import json
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm

import torch
from safetensors.torch import load_file, save_file

from kernel import weight_dequant

def main(fp8_path, bf16_path):
    """Converts an FP8 (E4M3) quantized checkpoint to BF16 precision.

    Orchestrates a full checkpoint conversion pipeline that reads FP8
    safetensors shard files, dequantizes every FP8 weight tensor to BF16
    via block-wise scale factor correction, and writes a clean BF16
    checkpoint that can be loaded without FP8 kernel support.

    Pipeline stages:
        1. Reads ``model.safetensors.index.json`` from *fp8_path* to obtain
           the ``weight_map`` dictionary that maps tensor names to their
           containing safetensors shard file names.
        2. Iterates through each ``*.safetensors`` shard file in sorted order,
           loading the full shard to GPU memory.
        3. For each tensor in the shard:
           - **_scale_inv suffix**: Skipped — these are FP8 quantization
             metadata consumed during dequantization, not independent weights.
           - **FP8 tensor** (``element_size() == 1`` byte): Looks up the
             corresponding ``<weight_name>_scale_inv`` tensor (which may
             reside in a different shard file) via ``get_tensor()``, then
             calls ``kernel.weight_dequant(weight, scale_inv)`` to produce
             a BF16 tensor.
           - **Non-FP8 tensor** (BF16/FP32, ``element_size() > 1``): Passed
             through to the output unchanged.
        4. Saves the converted state dict to *bf16_path* with the same shard
           file name.
        5. Rewrites ``model.safetensors.index.json`` in *bf16_path* with all
           ``_scale_inv`` entries removed from ``weight_map`` and ``metadata``
           reset to an empty dict (FP8 quantization_config is no longer
           applicable).

    Args:
        fp8_path (str): Path to the directory containing the FP8 checkpoint.
            Must contain ``model.safetensors.index.json`` and one or more
            ``*.safetensors`` shard files.
        bf16_path (str): Path to the output directory where the converted BF16
            checkpoint will be written. Created automatically if it does not
            exist.

    Raises:
        FileNotFoundError: If *fp8_path* does not contain
            ``model.safetensors.index.json`` or any safetensors shard files.
        KeyError: If a required ``_scale_inv`` tensor is missing from the
            ``weight_map`` (handled gracefully with a warning; the weight is
            passed through unconverted).

    Notes:
        - **FP8 tensor identification**: Tensors with ``element_size() == 1``
          byte are identified as ``float8_e4m3fn`` — the only 1-byte floating
          point dtype present in DeepSeek-V3 checkpoints. Standard BF16
          tensors use 2 bytes and FP32 tensors use 4 bytes.
        - **Block-wise dequantization**: ``kernel.weight_dequant()`` applies
          128×128 block-level scale factors stored in ``_scale_inv`` tensors.
          Each block's FP8 values are multiplied by the corresponding
          reciprocal scale factor to recover approximate BF16 magnitudes.
          Reference: arXiv:2412.19437, Section 3.3; ``README_WEIGHTS.md``.
        - **Cross-shard scale lookup**: A ``_scale_inv`` tensor may reside in
          a different safetensors shard file than its associated weight tensor.
          The ``get_tensor()`` helper resolves this by consulting the
          ``weight_map`` index and lazily loading the required shard.
        - **Memory management**: At most 2 shard files are kept in GPU memory
          simultaneously. After processing each shard, if more than 2 are
          cached, the oldest is evicted and ``torch.cuda.empty_cache()`` is
          called to release GPU memory. This bounds peak memory usage during
          conversion of large checkpoints (the 671B model spans many shards).
        - **Index rewrite**: The output ``model.safetensors.index.json`` has
          its ``metadata`` field reset to ``{}`` (the source FP8
          ``quantization_config`` metadata is no longer relevant) and all
          ``_scale_inv`` entries are pruned from ``weight_map`` since those
          tensors are not present in the BF16 output.
    """
    # [Assumptions Made] Set default dtype to bfloat16 — ensures any newly created
    # tensors (e.g., the output of weight_dequant) default to BF16 precision,
    # matching the target output format of this conversion utility.
    torch.set_default_dtype(torch.bfloat16)
    os.makedirs(bf16_path, exist_ok=True)
    model_index_file = os.path.join(fp8_path, "model.safetensors.index.json")
    with open(model_index_file, "r") as f:
        model_index = json.load(f)
    weight_map = model_index["weight_map"]
    
    # [Trade-offs] Cache loaded safetensor files in memory — enables cross-file
    # scale_inv lookups without repeated disk I/O, at the cost of GPU memory;
    # bounded to 2 files by cleanup logic below (see FIFO eviction after save).
    loaded_files = {}
    fp8_weight_names = []

    # Helper function to get tensor from the correct file
    def get_tensor(tensor_name):
        """Retrieves a tensor by name, loading its shard file on demand.

        Looks up which safetensors shard file contains the requested tensor
        using the ``weight_map`` index (parsed from
        ``model.safetensors.index.json``), loads that shard to GPU if it is
        not already cached in ``loaded_files``, and returns the tensor.

        Args:
            tensor_name (str): The fully-qualified tensor name as it appears
                in ``weight_map`` (e.g.,
                ``"model.layers.0.self_attn.q_proj.weight_scale_inv"``).

        Returns:
            torch.Tensor: The requested tensor, resident on CUDA.

        Raises:
            KeyError: If *tensor_name* is not present in ``weight_map`` or
                not found in the loaded shard file.

        Notes:
            - Uses ``weight_map`` (from ``model.safetensors.index.json``) to
              resolve which shard file contains the requested tensor, enabling
              cross-file lookups when a ``_scale_inv`` tensor resides in a
              different shard than its associated weight.
            - Caches entire shard files in the ``loaded_files`` dict to avoid
              redundant disk I/O when multiple tensors from the same shard are
              needed in quick succession (e.g., a weight and its scale_inv
              both queried during the same iteration).
            - This lazy-loading-with-caching strategy is critical because the
              ``_scale_inv`` tensor for a given weight may be stored in a
              different shard file than the weight itself, due to the way
              HuggingFace serializes large models across multiple shards.
        """
        file_name = weight_map[tensor_name]
        if file_name not in loaded_files:
            file_path = os.path.join(fp8_path, file_name)
            loaded_files[file_name] = load_file(file_path, device="cuda")
        return loaded_files[file_name][tensor_name]

    safetensor_files = list(glob(os.path.join(fp8_path, "*.safetensors")))
    safetensor_files.sort()
    for safetensor_file in tqdm(safetensor_files):
        file_name = os.path.basename(safetensor_file)
        current_state_dict = load_file(safetensor_file, device="cuda")
        loaded_files[file_name] = current_state_dict
        
        new_state_dict = {}
        for weight_name, weight in current_state_dict.items():
            # [Assumptions Made] Skip _scale_inv tensors from output — these are FP8
            # quantization metadata consumed during dequantization but not needed in
            # the BF16 output; they are also removed from the index file below.
            if weight_name.endswith("_scale_inv"):
                continue
            # [Alternatives Considered] Identify FP8 tensors by byte size (1 byte per
            # element) rather than checking dtype — this is a robust heuristic because
            # float8_e4m3fn is the only 1-byte floating point type in the checkpoint;
            # standard BF16/FP32 tensors are 2+ bytes.
            # Reference: arXiv:2412.19437, Section 3.3
            elif weight.element_size() == 1:  # FP8 weight
                scale_inv_name = f"{weight_name}_scale_inv"
                try:
                    # Get scale_inv from the correct file
                    scale_inv = get_tensor(scale_inv_name)
                    fp8_weight_names.append(weight_name)
                    # [Trade-offs] Dequantize on GPU via kernel.weight_dequant — performs
                    # block-wise (128×128) scale multiplication using Triton kernel for
                    # throughput; the scale_inv tensor stores per-block reciprocal scale
                    # factors. Reference: README_WEIGHTS.md FP8 format specification
                    new_state_dict[weight_name] = weight_dequant(weight, scale_inv)
                except KeyError:
                    # [Assumptions Made] Non-fatal handling of missing scale_inv — some
                    # tensors may not have quantization metadata (e.g., bias terms);
                    # warn and pass through unchanged rather than failing the entire
                    # conversion pipeline.
                    print(f"Warning: Missing scale_inv tensor for {weight_name}, skipping conversion")
                    new_state_dict[weight_name] = weight
            else:
                new_state_dict[weight_name] = weight
                
        new_safetensor_file = os.path.join(bf16_path, file_name)
        save_file(new_state_dict, new_safetensor_file)
        
        # [Trade-offs] Keep only the 2 most recently loaded shard files in memory —
        # bounds GPU memory usage during conversion of large checkpoints (671B model
        # has many shards); 2 is sufficient because scale_inv tensors are typically
        # in the same or adjacent shard as their associated weights. FIFO eviction
        # with explicit CUDA cache clearing to prevent OOM.
        if len(loaded_files) > 2:
            oldest_file = next(iter(loaded_files))
            del loaded_files[oldest_file]
            torch.cuda.empty_cache()
    
    # [Refactoring Rationale] Rewrite model index with empty metadata and pruned
    # weight_map — removes all _scale_inv entries that no longer have corresponding
    # tensors in the BF16 output; resets metadata to empty dict because FP8
    # quantization_config metadata is no longer applicable to the converted checkpoint.
    new_model_index_file = os.path.join(bf16_path, "model.safetensors.index.json")
    for weight_name in fp8_weight_names:
        scale_inv_name = f"{weight_name}_scale_inv"
        if scale_inv_name in weight_map:
            weight_map.pop(scale_inv_name)
    with open(new_model_index_file, "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f, indent=2)
        

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-fp8-hf-path", type=str, required=True)
    parser.add_argument("--output-bf16-hf-path", type=str, required=True)
    args = parser.parse_args()
    main(args.input_fp8_hf_path, args.output_bf16_hf_path)
    
