"""HuggingFace-to-model-parallel checkpoint conversion pipeline for DeepSeek-V3.

This module converts HuggingFace safetensors checkpoint files into rank-specific
model-parallel shards suitable for distributed inference via ``torchrun``. It
bridges the gap between the HuggingFace checkpoint naming convention (e.g.,
``model.layers.{N}.self_attn.q_proj.weight``) and the internal naming convention
used by the ``Transformer`` class in ``model.py`` (e.g.,
``layers.{N}.attn.wq.weight``).

The module contains two core components:

1. **mapping** dictionary — A lookup table that translates each HuggingFace
   parameter suffix to its corresponding internal name and specifies the
   sharding dimension for model-parallel partitioning. Entries with
   ``shard_dim=0`` are column-parallel (output features split), entries with
   ``shard_dim=1`` are row-parallel (input features split), and entries with
   ``shard_dim=None`` are replicated in full on every rank.

2. **main()** function — The conversion pipeline that iterates over all
   HuggingFace safetensors files, remaps parameter names, shards weights
   along the appropriate dimension, partitions routed experts across ranks by
   contiguous index range, and writes per-rank safetensors files in the format
   ``model{rank}-mp{mp}.safetensors``.

The output checkpoint files are consumed by ``generate.py``'s model loading
path (``safetensors.torch.load_model``) during distributed inference.

Usage::

    python convert.py \\
        --hf-ckpt-path /path/to/hf/DeepSeek-V3 \\
        --save-path /path/to/output \\
        --n-experts 256 \\
        --model-parallel 8

Reference:
    DeepSeek-V3 Technical Report — arXiv:2412.19437
    Model architecture context: Sections 2.1 (MLA), 2.1.2 (DeepSeekMoE), 2.2 (MTP)

See Also:
    - ``inference/model.py``: Defines the ``Transformer`` class and weight naming
      conventions that this conversion targets.
    - ``inference/generate.py``: Loads the converted model-parallel shards for
      distributed inference.
    - ``inference/fp8_cast_bf16.py``: Alternative conversion path for FP8-to-BF16
      weight dequantization (does not perform model-parallel sharding).
"""
import os
import shutil
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm, trange

import torch
from safetensors.torch import safe_open, save_file


# HuggingFace-to-internal parameter name mapping with model-parallel sharding directives.
#
# Each entry has the form:
#     hf_suffix: (internal_suffix, shard_dim)
#
# where:
#     - hf_suffix:       The second-to-last dot-separated component of the HuggingFace
#                         parameter name, used as the lookup key after name remapping.
#     - internal_suffix:  The replacement name used in model.py's Transformer class.
#     - shard_dim:        The tensor dimension along which to shard for model parallelism:
#                           0    = column-parallel (output features split across ranks)
#                           1    = row-parallel (input features split across ranks)
#                           None = replicated (full copy on every rank)
#
# [Alternatives Considered] A config-driven approach was considered but a static dictionary
# is sufficient given the fixed architecture and avoids runtime overhead for a one-time
# conversion utility.
mapping = {
    "embed_tokens": ("embed", 0),          # Vocabulary embedding — column-parallel: each rank holds vocab_size/mp rows
    "input_layernorm": ("attn_norm", None), # Pre-attention RMSNorm — replicated: normalization weights are small and identical across ranks
    "post_attention_layernorm": ("ffn_norm", None),  # Pre-FFN RMSNorm — replicated: same rationale as input_layernorm
    "q_proj": ("wq", 0),                   # Query projection (non-LoRA layers) — column-parallel: output heads split across ranks
    "q_a_proj": ("wq_a", None),            # Query LoRA down-projection — replicated: low-rank bottleneck shared across ranks
    "q_a_layernorm": ("q_norm", None),     # Query LoRA normalization — replicated
    "q_b_proj": ("wq_b", 0),              # Query LoRA up-projection — column-parallel: output heads split across ranks
    "kv_a_proj_with_mqa": ("wkv_a", None), # KV joint LoRA down-projection (MLA latent compression) — replicated: shared latent representation
    "kv_a_layernorm": ("kv_norm", None),   # KV LoRA normalization — replicated
    "kv_b_proj": ("wkv_b", 0),            # KV LoRA up-projection — column-parallel: output heads split across ranks
    "o_proj": ("wo", 1),                   # Output projection — row-parallel (dim=1): input features (head outputs) split across ranks, all_reduce after
    "gate": ("gate", None),                # MoE gate/router weights — replicated: all ranks need full routing decisions
    "gate_proj": ("w1", 0),                # Expert/MLP gate projection (SwiGLU w1) — column-parallel: output features split
    "down_proj": ("w2", 1),                # Expert/MLP down projection (SwiGLU w2) — row-parallel: input features split, all_reduce after
    "up_proj": ("w3", 0),                  # Expert/MLP up projection (SwiGLU w3) — column-parallel: output features split
    "norm": ("norm", None),                # Final RMSNorm — replicated
    "lm_head": ("head", 0),               # Language model head — column-parallel: vocab output split across ranks
    "scale": ("scale", None),              # FP8 quantization scale factors — replicated: scales must match their associated weight shards
}


def main(hf_ckpt_path, save_path, n_experts, mp):
    """
    Converts HuggingFace safetensors checkpoints into model-parallel shards.

    Iterates through all ``*.safetensors`` files in the HuggingFace checkpoint
    directory, applies name remapping from HuggingFace conventions to internal
    conventions (see module-level ``mapping`` dictionary), shards weight tensors
    along the specified dimension for model parallelism, partitions routed
    experts across ranks by contiguous index range, and saves one safetensors
    file per rank.

    The conversion pipeline performs four key transformations:

    1. **Name remapping**: Replaces HuggingFace naming prefixes with shorter
       internal names (``self_attn`` → ``attn``, ``mlp`` → ``ffn``,
       ``weight_scale_inv`` → ``scale``, ``e_score_correction_bias`` → ``bias``)
       to align with ``model.py``'s ``Transformer`` class attribute naming.
    2. **Weight sharding**: Splits weight tensors along the dimension specified
       in ``mapping`` (0 for column-parallel, 1 for row-parallel) into ``mp``
       equal-sized shards, one per rank.
    3. **Expert partitioning**: Assigns each routed expert to exactly one rank
       by contiguous index range — rank ``i`` receives experts
       ``[i * n_local_experts, (i+1) * n_local_experts)``. Shared experts are
       replicated on all ranks.
    4. **Tokenizer copying**: Copies all tokenizer-related files (matching
       ``*token*``) from the source checkpoint to the output directory for
       inference convenience.

    Args:
        hf_ckpt_path (str): Path to the directory containing the HuggingFace
            checkpoint files (``*.safetensors`` and tokenizer files).
        save_path (str): Path to the output directory where rank-specific
            model-parallel shards will be saved. Created if it does not exist.
        n_experts (int): Total number of routed experts in the model (e.g., 256
            for DeepSeek-V3 671B). Must be evenly divisible by ``mp``.
        mp (int): Model parallelism degree — the number of ranks/GPUs across
            which to shard the model weights.

    Returns:
        None: Writes ``mp`` safetensors files to ``save_path``, each named
        ``model{rank}-mp{mp}.safetensors``, plus copies of tokenizer files.

    Raises:
        AssertionError: If a parameter's sharding dimension is not evenly
            divisible by ``mp``.
        AssertionError: If a parameter name's second-to-last dot-separated
            component is not found in the ``mapping`` dictionary.

    Notes:
        - **Layer 61 skip**: Parameters belonging to ``model.layers.61`` are
          skipped during conversion. Layer 61 contains the Multi-Token
          Prediction (MTP) module, which is used during training for auxiliary
          prediction but is not required for standard autoregressive inference.
          Reference: arXiv:2412.19437, Section 2.2.
        - **Expert partitioning**: Each rank receives
          ``n_local_experts = n_experts // mp`` consecutive experts by index.
          For example, with 256 experts and 8-way parallelism, rank 0 gets
          experts 0–31, rank 1 gets experts 32–63, and so on. The CLI enforces
          that ``n_experts`` is divisible by ``mp``.
        - **Output format**: Files are named ``model{rank}-mp{mp}.safetensors``
          (e.g., ``model0-mp8.safetensors``). This naming convention is expected
          by ``generate.py``'s model loading logic.
        - **Tokenizer co-location**: Tokenizer files are copied alongside model
          shards so that ``generate.py`` can load both from the same directory,
          simplifying the inference deployment path.

    See Also:
        - ``inference/model.py``: Defines the ``Transformer`` class whose weight
          names correspond to the ``internal_suffix`` values in ``mapping``.
        - ``inference/generate.py``: Loads the output shards via
          ``safetensors.torch.load_model`` for distributed inference.
    """
    # [Assumptions Made] Fixed CPU thread count to 8 — matches generate.py convention;
    # bounds thread contention during parallel tensor operations on CPU
    torch.set_num_threads(8)
    # [Trade-offs] Experts partitioned evenly across model-parallel ranks — each rank
    # handles n_experts/mp consecutive experts, requiring n_experts to be divisible by
    # mp (enforced in CLI via assertion before this function is called)
    n_local_experts = n_experts // mp
    state_dicts = [{} for _ in range(mp)]

    for file_path in tqdm(glob(os.path.join(hf_ckpt_path, "*.safetensors"))):
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for name in f.keys():
                # [Assumptions Made] Skip layer 61 — this layer contains the Multi-Token
                # Prediction (MTP) module used during training but excluded from standard
                # autoregressive inference. Reference: arXiv:2412.19437, Section 2.2
                if "model.layers.61" in name:
                    continue
                param: torch.Tensor = f.get_tensor(name)
                if name.startswith("model."):
                    name = name[len("model."):]
                # [Refactoring Rationale] Remap HuggingFace naming conventions to internal
                # conventions — self_attn→attn, mlp→ffn, weight_scale_inv→scale,
                # e_score_correction_bias→bias; the internal names are shorter and align
                # with model.py's Transformer class attribute naming
                name = name.replace("self_attn", "attn")
                name = name.replace("mlp", "ffn")
                name = name.replace("weight_scale_inv", "scale")
                name = name.replace("e_score_correction_bias", "bias")
                # [Assumptions Made] Extract the second-to-last dot-separated component as
                # the mapping key — relies on consistent HuggingFace naming convention where
                # parameter names follow the pattern model.layers.{N}.{module}.{param}.weight
                key = name.split(".")[-2]
                assert key in mapping, f"Key {key} not found in mapping"
                new_key, dim = mapping[key]
                name = name.replace(key, new_key)
                for i in range(mp):
                    new_param = param
                    # [Trade-offs] Expert assignment by contiguous index range — expert i
                    # goes to rank floor(i / n_local_experts); this ensures each rank's
                    # experts are contiguous in the original checkpoint ordering,
                    # simplifying the conversion logic at the cost of potentially uneven
                    # activation patterns across ranks
                    if "experts" in name and "shared_experts" not in name:
                        idx = int(name.split(".")[-3])
                        if idx < i * n_local_experts or idx >= (i + 1) * n_local_experts:
                            continue
                    elif dim is not None:
                        assert param.size(dim) % mp == 0, f"Dimension {dim} must be divisible by {mp}"
                        shard_size = param.size(dim) // mp
                        new_param = param.narrow(dim, i * shard_size, shard_size).contiguous()
                    state_dicts[i][name] = new_param

    os.makedirs(save_path, exist_ok=True)

    for i in trange(mp):
        save_file(state_dicts[i], os.path.join(save_path, f"model{i}-mp{mp}.safetensors"))

    # [Assumptions Made] Copy all files matching *token* from the HuggingFace checkpoint —
    # captures tokenizer.json, tokenizer_config.json, and any special_tokens files;
    # co-locating tokenizer with model shards simplifies the inference loading path
    # in generate.py
    for file_path in glob(os.path.join(hf_ckpt_path, "*token*")):
        new_file_path = os.path.join(save_path, os.path.basename(file_path))
        shutil.copyfile(file_path, new_file_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--hf-ckpt-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--n-experts", type=int, required=True)
    parser.add_argument("--model-parallel", type=int, required=True)
    args = parser.parse_args()
    assert args.n_experts % args.model_parallel == 0, "Number of experts must be divisible by model parallelism"
    main(args.hf_ckpt_path, args.save_path, args.n_experts, args.model_parallel)
