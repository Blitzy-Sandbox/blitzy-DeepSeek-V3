# DeepSeek-V3 Configuration Parameter Reference

> **Source of Truth:** [`ModelArgs` dataclass](../model.py) (lines 20–86)
> **External Reference:** [arXiv:2412.19437](https://arxiv.org/abs/2412.19437), Table 1

## Table of Contents

- [Overview](#overview)
- [General Parameters](#general-parameters)
- [MoE Parameters](#moe-parameters)
- [MLA Parameters](#mla-parameters)
- [YaRN/RoPE Parameters](#yarnrope-parameters)
- [Model Variant Comparison](#model-variant-comparison)
- [Parameter Relationships](#parameter-relationships)
- [Default Values and Rationale](#default-values-and-rationale)
- [JSON-to-ModelArgs Mapping](#json-to-modelargs-mapping)
- [Cross-References](#cross-references)

---

## Overview

The DeepSeek-V3 inference configuration system uses JSON files deserialized into the `ModelArgs` dataclass defined in [`inference/model.py`](../model.py) (line 20). The loading mechanism is straightforward:

1. `generate.py` reads a JSON configuration file via `json.load()`
2. The resulting dictionary is unpacked into `ModelArgs` via `ModelArgs(**config)`
3. JSON keys map **1:1** to `ModelArgs` field names — no key translation or renaming occurs
4. Parameters **not present** in a JSON config file automatically fall back to the `ModelArgs` default values

### Configuration Variants

Four JSON configuration files are provided under `inference/configs/`, each targeting a different model variant:

| Config File | Model Size | Description |
|-------------|-----------|-------------|
| [`config_16B.json`](config_16B.json) | 16B | Smallest variant; uses direct query projection (no LoRA), softmax routing, BF16 weights |
| [`config_236B.json`](config_236B.json) | 236B | Mid-size variant; enables query LoRA, hierarchical expert grouping, BF16 weights |
| [`config_671B.json`](config_671B.json) | 671B | Full model; FP8 E4M3 quantized weights, sigmoid routing with auxiliary-loss-free balancing |
| [`config_v3.1.json`](config_v3.1.json) | 671B (v3.1) | v3.1 refinement of 671B; adds `ue8m0` scale format constraining FP8 scale factors to powers of 2 |

---

## General Parameters

These parameters control the overall model architecture dimensions, data types, and capacity limits.

| Parameter | Type | Default | Purpose | Architecture Relationship |
|-----------|------|---------|---------|--------------------------|
| `max_batch_size` | `int` | `8` | Maximum number of sequences processed simultaneously in a single forward pass. | Dimensions KV cache pre-allocation: all cache buffers in `MLA.__init__()` are sized as `(max_batch_size, max_seq_len, ...)`. |
| `max_seq_len` | `int` | `16384` | Maximum sequence length for generation, including both prompt and generated tokens. Computed as `4096 × 4`. | Determines `freqs_cis` precomputation length in `precompute_freqs_cis()` and KV cache sequence dimension. When `max_seq_len > original_seq_len`, YaRN frequency correction is activated. |
| `dtype` | `Literal["bf16", "fp8"]` | `"bf16"` | Weight data type for all `Linear` layers. | `"fp8"` enables FP8 E4M3 quantized weights with block-wise scale factors (128×128 blocks). Set via `Linear.dtype = torch.float8_e4m3fn` in `Transformer.__init__()`. Set to `"fp8"` in 671B and v3.1 configs. |
| `scale_fmt` | `Optional[str]` | `None` | FP8 scale factor format specifier. | `"ue8m0"` constrains scale factors to powers of 2 (unsigned 8-bit exponent, 0-bit mantissa). Propagated via `Linear.scale_fmt` class attribute and passed to `act_quant()` in `kernel.py`. Used only by v3.1 config. |
| `vocab_size` | `int` | `102400` | Size of the token vocabulary. | Partitioned across ranks as `part_vocab_size = vocab_size // world_size` via `ParallelEmbedding`. Values: 102,400 for 16B/236B; 129,280 for 671B/v3.1. |
| `dim` | `int` | `2048` | Model hidden dimension (embedding size). | Determines embedding output size, attention input/output dimensions, and all intermediate projections. Must be divisible by `n_heads`. |
| `inter_dim` | `int` | `10944` | Dense MLP intermediate dimension. | Used by `MLP` layers in the first `n_dense_layers` Transformer blocks. Split across ranks via `ColumnParallelLinear` (gate/up projections) and `RowParallelLinear` (down projection). |
| `moe_inter_dim` | `int` | `1408` | MoE expert intermediate dimension. | Each individual expert's hidden size within the SwiGLU MLP (`Expert` class). Intentionally smaller than `inter_dim` since many experts operate in parallel (only `n_activated_experts` per token). |
| `n_layers` | `int` | `27` | Total number of Transformer blocks. | Includes both dense layers (first `n_dense_layers`) and MoE layers (remaining `n_layers - n_dense_layers`). Iterated in `Transformer.__init__()` to construct the `layers` ModuleList. |
| `n_dense_layers` | `int` | `1` | Number of initial dense (non-MoE) layers. | Layers with `layer_id < n_dense_layers` use standard `MLP`; remaining layers use `MoE`. Controlled in `Block.__init__()`. |
| `n_heads` | `int` | `16` | Number of attention heads. | Split across ranks as `n_local_heads = n_heads // world_size` in `MLA.__init__()`. Determines the number of parallel attention computations. |

---

## MoE Parameters

These parameters configure the Mixture-of-Experts routing and expert allocation.
Reference: [arXiv:2412.19437, Section 2.1.2](https://arxiv.org/abs/2412.19437).

| Parameter | Type | Default | Purpose | Architecture Relationship |
|-----------|------|---------|---------|--------------------------|
| `n_routed_experts` | `int` | `64` | Total number of routed experts across all ranks. | Partitioned across ranks as `n_local_experts = n_routed_experts // world_size`. Each rank instantiates only its local experts; other slots are `None` in `MoE.experts` ModuleList. |
| `n_shared_experts` | `int` | `2` | Number of shared experts that process **all** tokens regardless of routing. | Implemented as a single `MLP` with intermediate dimension `n_shared_experts × moe_inter_dim` in `MoE.__init__()`. Provides a baseline representation that every token receives. |
| `n_activated_experts` | `int` | `6` | Number of experts activated per token (top-k). | Referred to as `topk` in the `Gate` class. Determines the `k` in `torch.topk(scores, self.topk)` during expert selection. |
| `n_expert_groups` | `int` | `1` | Number of expert groups for hierarchical two-stage routing. | When `> 1`, experts are divided into groups; first, `n_limited_groups` groups are selected via top-k on group scores, then top-k experts within those groups. Set to `8` for 236B/671B/v3.1. |
| `n_limited_groups` | `int` | `1` | Number of groups selected in the first routing stage. | Limits cross-group expert activation in hierarchical routing. Non-selected groups have their scores masked to `-inf` before final expert top-k. |
| `score_func` | `Literal["softmax", "sigmoid"]` | `"softmax"` | Expert routing scoring function. | `"softmax"` normalizes scores to a probability distribution over all experts. `"sigmoid"` enables auxiliary-loss-free load balancing via the learned `Gate.bias` parameter (used for 671B/v3.1). With `"sigmoid"`, routing weights are renormalized: `weights /= weights.sum()`. Reference: [arXiv:2412.19437, Section 2.1.2](https://arxiv.org/abs/2412.19437). |
| `route_scale` | `float` | `1.0` | Multiplicative scaling factor applied to expert routing weights after selection. | Scales the contribution magnitude of selected experts via `weights *= self.route_scale` in `Gate.forward()`. |

---

## MLA Parameters

These parameters configure Multi-Head Latent Attention (MLA), the core attention mechanism that jointly compresses keys and values into a low-rank latent space to reduce KV cache memory.
Reference: [arXiv:2412.19437, Section 2.1.1](https://arxiv.org/abs/2412.19437).

| Parameter | Type | Default | Purpose | Architecture Relationship |
|-----------|------|---------|---------|--------------------------|
| `q_lora_rank` | `int` | `0` | LoRA rank for query compression. | `0` disables LoRA: queries are projected directly via a single `ColumnParallelLinear(dim, n_heads × qk_head_dim)`. `> 0` enables two-stage query projection: `wq_a(dim → q_lora_rank)` → `RMSNorm` → `wq_b(q_lora_rank → n_heads × qk_head_dim)`. 16B uses `0`; 236B/671B/v3.1 use `1536`. |
| `kv_lora_rank` | `int` | `512` | Rank of the joint KV latent compression. | All variants use `512`. Determines the compressed KV cache size per layer in absorb mode: `(max_batch_size, max_seq_len, kv_lora_rank)`. Keys and values are jointly compressed into this latent space via `wkv_a`, then expanded per-head via `wkv_b`. |
| `qk_nope_head_dim` | `int` | `128` | Dimension of the non-positional (content-dependent) component of each attention head's query and key. | Combined with `qk_rope_head_dim` to form `qk_head_dim = qk_nope_head_dim + qk_rope_head_dim = 192`. Used in the content-dependent attention score computation. All variants use `128`. |
| `qk_rope_head_dim` | `int` | `64` | Dimension of the positional (RoPE) component of each attention head's query and key. | Carries the decoupled Rotary Position Embedding signal, separated from content dimensions to allow independent positional encoding. Shared across all ranks (not split by `world_size`). All variants use `64`. |
| `v_head_dim` | `int` | `128` | Value head dimension. | Determines the output size per attention head before the output projection `wo`. In MLA, values are extracted from the joint KV projection: `wkv_b` projects from `kv_lora_rank` to `n_heads × (qk_nope_head_dim + v_head_dim)`. All variants use `128`. |

---

## YaRN/RoPE Parameters

These parameters configure Rotary Position Embeddings (RoPE) and YaRN-based sequence length extension. All YaRN/RoPE parameters except `mscale` are **not present in any of the 4 config files** — they always use `ModelArgs` defaults unless overridden programmatically at runtime. The parameter `mscale` is set only in `config_16B.json`.

| Parameter | Type | Default | Purpose | Architecture Relationship |
|-----------|------|---------|---------|--------------------------|
| `original_seq_len` | `int` | `4096` | Pre-training sequence length. | When `max_seq_len > original_seq_len`, YaRN frequency correction is applied in `precompute_freqs_cis()`. This is the length at which base RoPE frequencies were calibrated during training. |
| `rope_theta` | `float` | `10000.0` | Base frequency for Rotary Position Embeddings. | Standard value from the original RoPE formulation (Su et al., 2021). Used to compute base frequencies: `freqs = 1.0 / (rope_theta ^ (2i / dim))`. |
| `rope_factor` | `float` | `40` | Sequence length extension factor for RoPE. | Used in YaRN frequency correction when `max_seq_len > original_seq_len`. Low-frequency dimension frequencies are divided by this factor for interpolation. |
| `beta_fast` | `int` | `32` | Fast correction dimension threshold for YaRN. | High-frequency RoPE dimensions above this rotation threshold retain their original scale without interpolation. Corresponds to `high_rot` in `find_correction_range()`. |
| `beta_slow` | `int` | `1` | Slow correction dimension threshold for YaRN. | Low-frequency RoPE dimensions below this rotation threshold are fully interpolated (scaled by `1/rope_factor`). Corresponds to `low_rot` in `find_correction_range()`. |
| `mscale` | `float` | `1.0` | Softmax attention scale correction for extended sequences. | When `max_seq_len > original_seq_len`, the softmax scale is adjusted: `mscale_factor = 0.1 × mscale × log(rope_factor) + 1.0`, then `softmax_scale *= mscale_factor²`. Set to `0.707` (≈√0.5) in 16B config; all others use default `1.0`. |

---

## Model Variant Comparison

Side-by-side comparison of all parameters across the four configuration variants. Parameters not explicitly set in a config file fall back to `ModelArgs` defaults, indicated by **(default)**.

### Full Parameter Table

| Parameter | 16B | 236B | 671B | v3.1 |
|-----------|-----|------|------|------|
| **General** | | | | |
| `max_batch_size` | *(default: 8)* | *(default: 8)* | *(default: 8)* | *(default: 8)* |
| `max_seq_len` | *(default: 16384)* | *(default: 16384)* | *(default: 16384)* | *(default: 16384)* |
| `dtype` | *(default: "bf16")* | *(default: "bf16")* | **"fp8"** | **"fp8"** |
| `scale_fmt` | *(default: None)* | *(default: None)* | *(default: None)* | **"ue8m0"** |
| `vocab_size` | 102400 | 102400 | **129280** | **129280** |
| `dim` | 2048 | 5120 | **7168** | **7168** |
| `inter_dim` | 10944 | 12288 | **18432** | **18432** |
| `moe_inter_dim` | 1408 | 1536 | **2048** | **2048** |
| `n_layers` | 27 | 60 | **61** | **61** |
| `n_dense_layers` | 1 | 1 | **3** | **3** |
| `n_heads` | 16 | 128 | 128 | 128 |
| **MoE** | | | | |
| `n_routed_experts` | 64 | 160 | **256** | **256** |
| `n_shared_experts` | 2 | 2 | **1** | **1** |
| `n_activated_experts` | 6 | 6 | **8** | **8** |
| `n_expert_groups` | *(default: 1)* | 8 | 8 | 8 |
| `n_limited_groups` | *(default: 1)* | 3 | 4 | 4 |
| `score_func` | *(default: "softmax")* | *(default: "softmax")* | **"sigmoid"** | **"sigmoid"** |
| `route_scale` | 1.0 | 16.0 | 2.5 | 2.5 |
| **MLA** | | | | |
| `q_lora_rank` | **0** | 1536 | 1536 | 1536 |
| `kv_lora_rank` | 512 | 512 | 512 | 512 |
| `qk_nope_head_dim` | 128 | 128 | 128 | 128 |
| `qk_rope_head_dim` | 64 | 64 | 64 | 64 |
| `v_head_dim` | 128 | 128 | 128 | 128 |
| **YaRN/RoPE** | | | | |
| `original_seq_len` | *(default: 4096)* | *(default: 4096)* | *(default: 4096)* | *(default: 4096)* |
| `rope_theta` | *(default: 10000.0)* | *(default: 10000.0)* | *(default: 10000.0)* | *(default: 10000.0)* |
| `rope_factor` | *(default: 40)* | *(default: 40)* | *(default: 40)* | *(default: 40)* |
| `beta_fast` | *(default: 32)* | *(default: 32)* | *(default: 32)* | *(default: 32)* |
| `beta_slow` | *(default: 1)* | *(default: 1)* | *(default: 1)* | *(default: 1)* |
| `mscale` | **0.707** | *(default: 1.0)* | *(default: 1.0)* | *(default: 1.0)* |

### Key Differences Across Variants

| Characteristic | 16B | 236B | 671B / v3.1 |
|---------------|-----|------|-------------|
| **Model scale** | Smallest (2048-dim, 16 heads) | Mid-size (5120-dim, 128 heads) | Full-size (7168-dim, 128 heads) |
| **Weight format** | BF16 (default) | BF16 (default) | FP8 E4M3 quantized |
| **FP8 scale format** | N/A | N/A | None (671B) / `ue8m0` (v3.1) |
| **Query LoRA** | Disabled (`q_lora_rank=0`) | Enabled (`q_lora_rank=1536`) | Enabled (`q_lora_rank=1536`) |
| **Routing function** | Softmax (default) | Softmax (default) | Sigmoid (auxiliary-loss-free) |
| **Expert count** | 64 routed, 6 active | 160 routed, 6 active | 256 routed, 8 active |
| **Hierarchical routing** | Disabled (1 group) | 8 groups, top-3 selected | 8 groups, top-4 selected |
| **Dense layers** | 1 initial dense layer | 1 initial dense layer | 3 initial dense layers |
| **Vocabulary size** | 102,400 | 102,400 | 129,280 |
| **Attention scale** | `mscale=0.707` (≈√0.5) | Default `mscale=1.0` | Default `mscale=1.0` |

---

## Parameter Relationships

Several parameters are **computed or derived** at runtime rather than configured directly. Understanding these relationships is essential for capacity planning and distributed deployment.

### Attention Dimensions

```
qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
            = 128 + 64
            = 192                              (all variants)
```

The `qk_head_dim` in MLA combines content-dependent and positional components. This is distinct from the traditional `head_dim = dim / n_heads` used in standard multi-head attention:

| Variant | `dim / n_heads` (traditional) | `qk_head_dim` (MLA) |
|---------|-------------------------------|---------------------|
| 16B     | 2048 / 16 = 128               | 192                 |
| 236B    | 5120 / 128 = 40               | 192                 |
| 671B    | 7168 / 128 = 56               | 192                 |

### Distributed Partitioning

All partitioning requires exact divisibility. For a given `world_size` (number of tensor-parallel GPUs):

```
n_local_heads   = n_heads / world_size           (attention heads per rank)
n_local_experts = n_routed_experts / world_size   (routed experts per rank)
part_vocab_size = vocab_size / world_size          (vocabulary partition per rank)
```

Divisibility constraints enforced by assertions in `model.py`:

- `vocab_size % world_size == 0` — enforced in `ParallelEmbedding.__init__()`
- `n_routed_experts % world_size == 0` — enforced in `MoE.__init__()`
- `out_features % world_size == 0` — enforced in `ColumnParallelLinear.__init__()` (applies to `n_heads`-derived dimensions)
- `in_features % world_size == 0` — enforced in `RowParallelLinear.__init__()`

### KV Cache Memory

KV cache size per layer depends on the attention implementation mode (`attn_impl` global variable in `model.py`):

**Absorb mode** (default, `attn_impl="absorb"`):

```
Elements per layer = max_batch_size × max_seq_len × (kv_lora_rank + qk_rope_head_dim)
                   = 8 × 16384 × (512 + 64)
                   = 8 × 16384 × 576
                   = 75,497,472 elements (~75.5M)
```

Two buffers are allocated: `kv_cache` with shape `(batch, seq, kv_lora_rank)` and `pe_cache` with shape `(batch, seq, qk_rope_head_dim)`.

**Naive mode** (`attn_impl="naive"`):

```
Elements per layer = max_batch_size × max_seq_len × n_local_heads × (qk_head_dim + v_head_dim)
                   = 8 × 16384 × n_local_heads × (192 + 128)
                   = 8 × 16384 × n_local_heads × 320
```

Two buffers are allocated: `k_cache` with shape `(batch, seq, n_local_heads, qk_head_dim)` and `v_cache` with shape `(batch, seq, n_local_heads, v_head_dim)`.

Absorb mode is significantly more memory-efficient because it caches in the compressed latent space (`kv_lora_rank=512`) rather than the full head-expanded space, which scales with `n_local_heads`.

### Dimensional Divisibility Verification

`dim` must be divisible by `n_heads` for attention head splitting:

| Variant | `dim` | `n_heads` | `dim / n_heads` | Valid |
|---------|-------|-----------|-----------------|-------|
| 16B     | 2048  | 16        | 128             | ✓     |
| 236B    | 5120  | 128       | 40              | ✓     |
| 671B    | 7168  | 128       | 56              | ✓     |

`inter_dim` and `dim` are split by `world_size` for tensor-parallel execution via `ColumnParallelLinear` and `RowParallelLinear`.

---

## Default Values and Rationale

Parameters that are not set in any configuration file use `ModelArgs` defaults. The following explains the rationale behind key default choices.

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| `max_batch_size` | `8` | Conservative default balancing throughput and memory. KV cache is pre-allocated to this size at model initialization — larger values consume proportionally more GPU memory. |
| `max_seq_len` | `16384` | Set to 4× the `original_seq_len` (4096), enabling YaRN-extended context window. This triggers the YaRN frequency correction path in `precompute_freqs_cis()`. |
| `dtype` | `"bf16"` | Safe default for broad GPU compatibility (A100, H100, consumer GPUs). `"fp8"` is used for the 671B model to reduce memory footprint but requires FP8-capable hardware (H100/H200). |
| `scale_fmt` | `None` | No special scale format constraint by default. `"ue8m0"` (v3.1 only) constrains FP8 scale factors to exact powers of 2 for hardware-friendly dequantization. |
| `score_func` | `"softmax"` | Standard scoring for smaller models where auxiliary-loss-free balancing is not needed. `"sigmoid"` enables the learned bias mechanism (`Gate.bias`) for load balancing in the 671B model. Reference: [arXiv:2412.19437, Section 2.1.2](https://arxiv.org/abs/2412.19437). |
| `q_lora_rank` | `0` | Disables query LoRA compression for the smallest model (16B), using a direct query projection instead. Larger models (236B, 671B, v3.1) set this to `1536` for parameter-efficient query compression. Reference: [arXiv:2412.19437, Section 2.1.1](https://arxiv.org/abs/2412.19437). |
| `kv_lora_rank` | `512` | Shared across **all** variants — a core MLA design choice determining the compressed KV cache dimension. Balances attention quality with memory efficiency for the latent compression scheme. |
| `original_seq_len` | `4096` | The sequence length used during pre-training. Serves as the baseline for YaRN frequency correction when `max_seq_len` exceeds this value. |
| `rope_theta` | `10000.0` | Standard base frequency from the original RoPE formulation (Su et al., 2021). |
| `rope_factor` | `40` | Enables 40× sequence extension relative to `original_seq_len` when YaRN correction is active. |
| `mscale` | `1.0` | No attention scale correction by default. The 16B model overrides to `0.707` (≈√0.5) to correct softmax entropy at extended sequence lengths. |
| `block_size` | `128` | **Hardcoded in `model.py` (not configurable via JSON).** Matches the FP8 training block granularity of 128×128 tiles. Reference: [arXiv:2412.19437, Section 3.3](https://arxiv.org/abs/2412.19437). |

---

## JSON-to-ModelArgs Mapping

### Mapping Mechanism

Configuration JSON keys map **exactly 1:1** to `ModelArgs` attribute names:

- Any key present in a JSON config file **overrides** the corresponding `ModelArgs` default
- Keys absent from a config file **fall back** to the `ModelArgs` default value defined in [`inference/model.py`](../model.py) lines 20–86
- No key translation, renaming, or transformation occurs — the JSON dictionary is unpacked directly via `ModelArgs(**config)`

### Loading Code Path

In `generate.py`, the configuration is loaded and applied as follows:

```python
with open(config_path) as f:
    config = json.load(f)
args = ModelArgs(**config)
model = Transformer(args)
```

### Parameters Never Set in Any Config File

The following parameters are **not present in any of the 4 configuration files** and always use their `ModelArgs` defaults unless overridden programmatically at runtime:

| Parameter | Default Value | Reason Not Configured |
|-----------|--------------|----------------------|
| `max_batch_size` | `8` | Runtime-dependent; may be adjusted based on available GPU memory |
| `max_seq_len` | `16384` | Runtime-dependent; may be adjusted based on sequence length needs |
| `original_seq_len` | `4096` | Fixed pre-training parameter; shared across all model variants |
| `rope_theta` | `10000.0` | Standard RoPE base; no variant-specific override needed |
| `rope_factor` | `40` | Shared extension factor; consistent across all variants |
| `beta_fast` | `32` | YaRN correction hyperparameter; same for all variants |
| `beta_slow` | `1` | YaRN correction hyperparameter; same for all variants |

### Parameters Set in Only Some Config Files

| Parameter | Set In | Value When Set | Default When Absent |
|-----------|--------|---------------|---------------------|
| `dtype` | 671B, v3.1 | `"fp8"` | `"bf16"` |
| `scale_fmt` | v3.1 only | `"ue8m0"` | `None` |
| `score_func` | 671B, v3.1 | `"sigmoid"` | `"softmax"` |
| `mscale` | 16B only | `0.707` | `1.0` |
| `n_expert_groups` | 236B, 671B, v3.1 | `8` | `1` |
| `n_limited_groups` | 236B (`3`), 671B (`4`), v3.1 (`4`) | varies | `1` |

---

## Cross-References

### Internal Documentation

- **ModelArgs source of truth:** [`inference/model.py`](../model.py), lines 20–86 — the `ModelArgs` dataclass defining all configurable parameters with their default values and type annotations
- **FP8 format details:** [`README_WEIGHTS.md`](../../README_WEIGHTS.md) — documents the FP8 E4M3 quantization format, 128×128 block size, and `scale_inv` tensor structure used by `dtype="fp8"` configurations
- **Architecture overview:** [`inference/ARCHITECTURE.md`](../ARCHITECTURE.md) — high-level pipeline overview, module dependency graph, and architectural glossary

### External References

- **Architecture hyperparameters:** [arXiv:2412.19437](https://arxiv.org/abs/2412.19437), Table 1 — canonical source for DeepSeek-V3 model dimension choices across variants
- **MLA parameters:** [arXiv:2412.19437](https://arxiv.org/abs/2412.19437), Section 2.1.1 — Multi-Head Latent Attention design and latent compression rationale
- **MoE parameters:** [arXiv:2412.19437](https://arxiv.org/abs/2412.19437), Section 2.1.2 — DeepSeekMoE architecture, auxiliary-loss-free load balancing, and sigmoid routing
- **FP8 quantization parameters:** [arXiv:2412.19437](https://arxiv.org/abs/2412.19437), Section 3.3 — block-wise FP8 training with 128×128 tile granularity

### Configuration File Locations

| Config File | Path |
|-------------|------|
| 16B variant | [`inference/configs/config_16B.json`](config_16B.json) |
| 236B variant | [`inference/configs/config_236B.json`](config_236B.json) |
| 671B variant | [`inference/configs/config_671B.json`](config_671B.json) |
| v3.1 variant | [`inference/configs/config_v3.1.json`](config_v3.1.json) |
