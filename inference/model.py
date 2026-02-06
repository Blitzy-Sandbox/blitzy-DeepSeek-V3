"""Core Transformer architecture implementation for DeepSeek-V3 inference.

This module defines the complete DeepSeek-V3 model architecture, implementing several
key innovations from the DeepSeek-V3 technical report:

- **Multi-Head Latent Attention (MLA):** Compresses key-value representations into a
  low-rank latent space, reducing KV cache memory from O(n·d_head·n_heads) to
  O(n·kv_lora_rank + n·qk_rope_head_dim) per layer. Supports two compute strategies:
  "naive" (explicit K,V materialization) and "absorb" (fused projection into query space).
  Reference: arXiv:2412.19437, Section 2.1.1

- **DeepSeekMoE with Auxiliary-Loss-Free Load Balancing:** Routes tokens to a sparse
  subset of experts using a learned bias mechanism that avoids representational damage
  from auxiliary loss terms. Includes shared experts for baseline representation.
  Reference: arXiv:2412.19437, Section 2.1.2

- **YaRN-based RoPE for Extended Context:** Applies frequency-selective scaling to
  rotary positional embeddings, enabling context extension beyond the pre-training
  sequence length while preserving local attention patterns.

- **FP8 Block-wise Quantization Support:** Supports inference with FP8 (E4M3) quantized
  weights using 128×128 block granularity, with switchable GEMM backends (BF16
  dequantization fallback or native FP8 Triton kernels).
  Reference: arXiv:2412.19437, Section 3.3

Classes:
    ModelArgs: Dataclass holding all model hyperparameters across 4 config variants.
    ParallelEmbedding: Vocabulary-partitioned embedding with all_reduce aggregation.
    Linear: Base linear layer with FP8 quantization and per-block scale factor support.
    ColumnParallelLinear: Output-feature-partitioned linear for tensor parallelism.
    RowParallelLinear: Input-feature-partitioned linear with all_reduce aggregation.
    RMSNorm: Root Mean Square normalization (no mean-centering).
    MLA: Multi-Head Latent Attention with compressed KV cache.
    MLP: SwiGLU feed-forward network with tensor parallelism.
    Gate: Expert routing with auxiliary-loss-free bias-based load balancing.
    Expert: Individual expert MLP within the MoE layer.
    MoE: Mixture-of-Experts layer with expert parallelism and shared experts.
    Block: Pre-norm Transformer block (RMSNorm → MLA → residual → RMSNorm → FFN → residual).
    Transformer: Top-level model class managing global state and full forward pass.

Standalone Functions:
    linear(): Dispatch function for BF16/FP8 linear operations.
    precompute_freqs_cis(): YaRN-corrected rotary positional embedding frequencies.
    apply_rotary_emb(): Applies rotary embeddings via complex multiplication.

Module-level Global State:
    world_size, rank: Distributed process topology, set by Transformer.__init__().
    block_size: FP8 quantization tile size (128), matching training granularity.
    gemm_impl: GEMM backend selector ("bf16" or "fp8").
    attn_impl: Attention strategy selector ("naive" or "absorb").

Dependencies:
    kernel.py: Provides act_quant(), weight_dequant(), fp8_gemm() for FP8 operations.

Reference: arXiv:2412.19437 — DeepSeek-V3 Technical Report
    Section 2.1.1: Multi-Head Latent Attention (MLA)
    Section 2.1.2: DeepSeekMoE Architecture
    Section 2.2: Multi-Token Prediction (MTP)
    Section 3.3: FP8 Mixed Precision Training
"""

import math
from dataclasses import dataclass
from typing import Tuple, Optional, Literal

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from kernel import act_quant, weight_dequant, fp8_gemm


# [Assumptions Made] Module-level globals for distributed state — set by Transformer.__init__()
# once at model construction; using globals instead of passing through every forward() call
# avoids API complexity in a research codebase where distributed state is fixed for the
# lifetime of the process.
world_size = 1
rank = 0
# [Trade-offs] Block size 128 for FP8 quantization — matches the 128×128 tile granularity
# used during FP8 training (arXiv:2412.19437, Section 3.3); aligns with GPU tensor core
# tile sizes for efficient quantized GEMM execution.
block_size = 128
# [Alternatives Considered] Switchable GEMM implementation — "bf16" dequantizes FP8 weights
# to BF16 then uses standard F.linear (broader GPU compatibility); "fp8" uses native FP8
# GEMM via Triton kernel (higher throughput on H100+); set by external configuration.
gemm_impl: Literal["bf16", "fp8"] = "bf16"
# [Alternatives Considered] Two attention strategies — "naive" materializes full K,V tensors
# per head (straightforward but memory-heavy); "absorb" fuses KV projection into query space
# avoiding explicit key materialization (memory-efficient, the default).
# Reference: arXiv:2412.19437, Section 2.1.1
attn_impl: Literal["naive", "absorb"] = "absorb"

@dataclass
class ModelArgs:
    """Data class for defining model architecture hyperparameters for DeepSeek-V3.

    Groups all configuration parameters needed to construct a DeepSeek-V3 Transformer
    model. Default values correspond to the 16B model variant; larger model variants
    (236B, 671B, v3.1) override these via JSON configuration files.

    Parameter values are loaded from JSON config files (config_16B.json, config_236B.json,
    config_671B.json, config_v3.1.json) in inference/configs/. See CONFIG_REFERENCE.md
    for full parameter documentation.

    Attributes:
        General Parameters:
            max_batch_size (int): Maximum batch size for KV cache pre-allocation.
            max_seq_len (int): Maximum sequence length (context window) in tokens.
            dtype (Literal["bf16", "fp8"]): Weight storage format — "bf16" for BFloat16,
                "fp8" for FP8 E4M3 quantized weights.
            scale_fmt (Optional[str]): FP8 scale tensor format — None for standard float32
                scales, "ue8m0" for unsigned E8M0 power-of-2 scales (used by v3.1).
            vocab_size (int): Total vocabulary size for token embeddings.
            dim (int): Model hidden dimension (embedding size). 16B=2048, 236B=5120, 671B=7168.
            inter_dim (int): Intermediate dimension for dense MLP layers (SwiGLU hidden size).
            moe_inter_dim (int): Intermediate dimension for each MoE expert's MLP.
            n_layers (int): Total number of Transformer blocks. 16B=27, 236B=60, 671B=61.
            n_dense_layers (int): Number of initial layers using dense MLP instead of MoE.
            n_heads (int): Number of attention heads. 16B=16, 236B=40, 671B=128.

        MoE Parameters (Reference: arXiv:2412.19437, Section 2.1.2):
            n_routed_experts (int): Total number of routed experts across all ranks.
            n_shared_experts (int): Number of shared experts applied to every token.
            n_activated_experts (int): Number of experts activated per token (top-k).
            n_expert_groups (int): Number of expert groups for hierarchical routing.
            n_limited_groups (int): Number of groups selected in the first routing stage.
            score_func (Literal["softmax", "sigmoid"]): Expert scoring function.
            route_scale (float): Multiplicative scaling factor for final routing weights.

        MLA Parameters (Reference: arXiv:2412.19437, Section 2.1.1):
            q_lora_rank (int): LoRA rank for query compression (0 disables LoRA).
            kv_lora_rank (int): Rank for joint KV latent compression.
            qk_nope_head_dim (int): Per-head dimension for non-positional Q/K components.
            qk_rope_head_dim (int): Per-head dimension for RoPE-encoded Q/K components.
            v_head_dim (int): Per-head dimension for value projections.

        YaRN RoPE Extension Parameters:
            original_seq_len (int): Pre-training context length before YaRN extension.
            rope_theta (float): Base frequency for rotary positional encoding.
            rope_factor (float): Context extension scaling factor for YaRN.
            beta_fast (int): High-frequency rotation count threshold for YaRN correction.
            beta_slow (int): Low-frequency rotation count threshold for YaRN correction.
            mscale (float): Softmax scale adjustment factor for extended context.

    Reference: arXiv:2412.19437, Table 1 for architecture hyperparameters.
    """
    max_batch_size: int = 8
    # [Assumptions Made] Default 16K context — the 16B model default; overridden by config
    # JSON for larger models (671B supports 128K).
    max_seq_len: int = 4096 * 4
    dtype: Literal["bf16", "fp8"] = "bf16"
    scale_fmt: Optional[str] = None
    vocab_size: int = 102400
    dim: int = 2048
    inter_dim: int = 10944
    moe_inter_dim: int = 1408
    n_layers: int = 27
    n_dense_layers: int = 1
    n_heads: int = 16
    # [Assumptions Made] MoE defaults correspond to the 16B model variant; larger models
    # override these via config JSON (e.g., 671B uses 256 routed experts, 8 activated).
    # moe
    n_routed_experts: int = 64
    n_shared_experts: int = 2
    n_activated_experts: int = 6
    n_expert_groups: int = 1
    n_limited_groups: int = 1
    score_func: Literal["softmax", "sigmoid"] = "softmax"
    route_scale: float = 1.
    # [Trade-offs] q_lora_rank=0 disables query LoRA compression for 16B model — smaller
    # models use direct query projection; larger models (236B, 671B) use LoRA with rank 1536
    # to reduce query parameter count. Reference: arXiv:2412.19437, Section 2.1.1
    # mla
    q_lora_rank: int = 0
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    # [Assumptions Made] YaRN defaults for RoPE extension — original_seq_len is the
    # pre-training context length; when max_seq_len exceeds this, YaRN frequency correction
    # is applied to extend context without retraining.
    # yarn
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.


class ParallelEmbedding(nn.Module):
    """Vocabulary-partitioned embedding layer with distributed all_reduce aggregation.

    Partitions the embedding table across distributed ranks so that each rank holds
    vocab_size / world_size rows. During forward pass, each rank performs a local
    embedding lookup for tokens in its partition range and zeros out tokens belonging
    to other ranks. An all_reduce(SUM) then reconstructs the complete embedding.

    Args:
        vocab_size (int): Total vocabulary size (must be divisible by world_size).
        dim (int): Embedding dimension (model hidden size).

    Attributes:
        weight (nn.Parameter): Local embedding matrix of shape ``(part_vocab_size, dim)``
            where part_vocab_size = vocab_size // world_size.
        vocab_start_idx (int): First vocabulary index owned by this rank.
        vocab_end_idx (int): One past the last vocabulary index owned by this rank.
    """
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        assert vocab_size % world_size == 0, f"Vocabulary size must be divisible by world size (world_size={world_size})"
        self.part_vocab_size = (vocab_size // world_size)
        self.vocab_start_idx = rank * self.part_vocab_size
        self.vocab_end_idx = self.vocab_start_idx + self.part_vocab_size
        self.weight = nn.Parameter(torch.empty(self.part_vocab_size, self.dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for parallel embedding layer.

        Performs distributed embedding lookup: masks out-of-range tokens, applies
        local embedding, zeros masked positions, and all_reduces across ranks.

        Args:
            x (torch.Tensor): Input token indices of shape ``(batch_size, seq_len)``
                with integer values in [0, vocab_size).

        Returns:
            torch.Tensor: Embedded representations of shape ``(batch_size, seq_len, dim)``.

        Notes:
            When world_size > 1, distributed communication is performed via
            dist.all_reduce(SUM) to aggregate partial embeddings from all ranks.
        """
        if world_size > 1:
            # [Trade-offs] Mask out-of-range tokens to zero before embedding lookup — this
            # avoids index-out-of-bounds errors when tokens belong to another rank's vocabulary
            # partition; the masking + all_reduce pattern ensures each token's embedding comes
            # from exactly the rank that owns it.
            mask = (x < self.vocab_start_idx) | (x >= self.vocab_end_idx)
            x = x - self.vocab_start_idx
            x[mask] = 0
        y = F.embedding(x, self.weight)
        if world_size > 1:
            y[mask] = 0
            # [Distributed] all_reduce(SUM) over all ranks — each rank contributes the
            # embedding for tokens in its vocabulary partition (non-zero) and zeros for tokens
            # outside its range; summing reconstructs the complete embedding. Participating
            # ranks: all ranks in the default process group. Data: (batch_size, seq_len, dim)
            # tensor.
            dist.all_reduce(y)
        return y


def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, scale_fmt: Optional[str] = None) -> torch.Tensor:
    """Dispatch function for linear transformation with FP8/BF16 quantization support.

    Applies y = xA^T + b with three execution paths depending on weight dtype and
    the global ``gemm_impl`` setting, enabling transparent switching between standard
    and quantized computation.

    Args:
        x (torch.Tensor): Input activation tensor of shape ``(*, in_features)``.
        weight (torch.Tensor): Weight matrix of shape ``(out_features, in_features)``.
            May be FP8 (element_size==1) or BF16/FP32 (element_size>1).
        bias (Optional[torch.Tensor]): Bias vector of shape ``(out_features,)``.
            Default is None.
        scale_fmt (Optional[str]): FP8 scale format for activation quantization.
            None for standard float32 scales, "ue8m0" for power-of-2 scales.

    Returns:
        torch.Tensor: Output tensor of shape ``(*, out_features)``.

    Notes:
        Three dispatch paths based on weight format and gemm_impl setting:

        1. **Standard BF16/FP32 path** (weight.element_size() > 1): Direct F.linear
           with no quantization overhead.
        2. **BF16 GEMM fallback** (FP8 weight + gemm_impl=="bf16"): Dequantizes FP8
           weights to BF16 via ``weight_dequant()`` from kernel.py, then F.linear.
           Provides compatibility on GPUs without native FP8 GEMM support (pre-H100).
        3. **Native FP8 GEMM** (FP8 weight + gemm_impl=="fp8"): Quantizes activations
           on-the-fly via ``act_quant()`` from kernel.py, then computes via Triton
           ``fp8_gemm()`` kernel. Highest throughput on H100+ GPUs.
           Reference: arXiv:2412.19437, Section 3.3
    """
    # [Trade-offs] Standard BF16/FP32 path — when weights are not quantized
    # (element_size > 1 byte), use PyTorch's optimized F.linear directly; no
    # quantization overhead.
    if weight.element_size() > 1:
        return F.linear(x, weight, bias)
    # [Alternatives Considered] BF16 GEMM fallback for FP8 weights — dequantizes
    # weights to BF16 for GPUs without native FP8 GEMM support (pre-H100); trades
    # compute for compatibility.
    elif gemm_impl == "bf16":
        weight = weight_dequant(weight, weight.scale)
        return F.linear(x, weight, bias)
    else:
        # [Trade-offs] Native FP8 GEMM path — quantizes activations on-the-fly via
        # act_quant and uses Triton FP8 GEMM kernel; highest throughput on H100+ GPUs
        # but requires FP8 hardware support. Reference: arXiv:2412.19437, Section 3.3
        x, scale = act_quant(x, block_size, scale_fmt)
        y = fp8_gemm(x, scale, weight, weight.scale)
        if bias is not None:
            y += bias
        return y


class Linear(nn.Module):
    """Base linear layer with FP8 quantization and per-block scale factor support.

    Serves as the base class for ColumnParallelLinear and RowParallelLinear. Manages
    weight allocation in either BF16 or FP8 format, with automatic allocation of
    per-block scale tensors when using FP8 weights.

    The class-level ``dtype`` and ``scale_fmt`` attributes are shared mutable state set
    once by ``Transformer.__init__()``. All Linear instances created after that point
    use the same dtype, enabling transparent FP8 support without per-instance config.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type override for this instance. Defaults to the
            class-level ``Linear.dtype`` (typically torch.bfloat16 or torch.float8_e4m3fn).

    Attributes:
        weight (nn.Parameter): Weight matrix of shape ``(out_features, in_features)``.
        scale (nn.Parameter or None): Per-block FP8 scale factors of shape
            ``(ceil(out_features/block_size), ceil(in_features/block_size))`` when using
            FP8 weights, or None for BF16/FP32 weights.
        bias (nn.Parameter or None): Bias vector of shape ``(out_features,)`` if enabled.

    Notes:
        When weight is FP8 (element_size == 1), a scale tensor is allocated with one
        float32 scale factor per 128×128 weight block. The scale is attached to the
        weight parameter as an attribute (``weight.scale``) AND registered as a separate
        parameter (``self.scale``) for checkpoint serialization compatibility.
    """
    # [Assumptions Made] Class-level dtype and scale_fmt — shared mutable state set once by
    # Transformer.__init__(); all Linear instances use the same dtype, avoiding per-instance
    # configuration. This is a non-standard pattern chosen for simplicity in a research codebase.
    dtype = torch.bfloat16
    scale_fmt: Optional[str] = None

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype or Linear.dtype))
        if self.weight.element_size() == 1:
            # [Trade-offs] Allocate per-block FP8 scale tensor — ceil division handles weight
            # dimensions not evenly divisible by block_size (128); scale tensor stores one
            # float32 scale factor per 128×128 weight block. The scale is attached to the
            # weight parameter as an attribute AND registered as a separate parameter for
            # serialization.
            scale_out_features = (out_features + block_size - 1) // block_size
            scale_in_features = (in_features + block_size - 1) // block_size
            self.weight.scale = self.scale = nn.Parameter(torch.empty(scale_out_features, scale_in_features, dtype=torch.float32))
        else:
            self.register_parameter("scale", None)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass dispatching to the ``linear()`` function.

        Args:
            x (torch.Tensor): Input tensor of shape ``(*, in_features)``.

        Returns:
            torch.Tensor: Output tensor of shape ``(*, out_features)``.
        """
        return linear(x, self.weight, self.bias, self.scale_fmt)


class ColumnParallelLinear(Linear):
    """Linear layer with column (output) parallelism across distributed ranks.

    Splits output features across ranks so each rank computes
    ``out_features / world_size`` output features from the full input. No all_reduce
    is needed because each rank independently computes its partition of the output.

    Used for the first projection in parallel patterns (e.g., MLA query/KV projections,
    MLP w1/w3 gates, vocabulary head).

    Args:
        in_features (int): Number of input features (full, not partitioned).
        out_features (int): Total number of output features (must be divisible by world_size).
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to ``Linear.dtype``.

    Attributes:
        part_out_features (int): Number of output features on this rank
            (out_features // world_size).
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        assert out_features % world_size == 0, f"Output features must be divisible by world size (world_size={world_size})"
        self.part_out_features = out_features // world_size
        super().__init__(in_features, self.part_out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass computing this rank's output feature partition.

        Args:
            x (torch.Tensor): Input tensor of shape ``(*, in_features)``.

        Returns:
            torch.Tensor: Output tensor of shape ``(*, part_out_features)`` where
                part_out_features = out_features // world_size.

        Notes:
            No distributed communication is required — each rank independently
            computes its partition of the output features.
        """
        y = linear(x, self.weight, self.bias)
        return y


class RowParallelLinear(Linear):
    """Linear layer with row (input) parallelism across distributed ranks.

    Splits input features across ranks so each rank holds ``in_features / world_size``
    columns of the weight matrix. Each rank computes a partial matrix product, and
    an all_reduce(SUM) aggregates the partial results to produce the full output.

    Used for the second projection in parallel patterns (e.g., MLA output projection,
    MLP w2 down-projection, shared expert output).

    Args:
        in_features (int): Total number of input features (must be divisible by world_size).
        out_features (int): Number of output features (full, not partitioned).
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to ``Linear.dtype``.

    Attributes:
        part_in_features (int): Number of input features on this rank
            (in_features // world_size).
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        assert in_features % world_size == 0, f"Input features must be divisible by world size (world_size={world_size})"
        self.part_in_features = in_features // world_size
        super().__init__(self.part_in_features, out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with all_reduce aggregation of partial matrix products.

        Args:
            x (torch.Tensor): Input tensor of shape ``(*, part_in_features)`` where
                part_in_features = in_features // world_size.

        Returns:
            torch.Tensor: Output tensor of shape ``(*, out_features)`` after
                all_reduce aggregation and optional bias addition.

        Notes:
            When world_size > 1, dist.all_reduce(SUM) aggregates partial outputs
            from all ranks. Bias is added after all_reduce to avoid redundant
            application on each rank.
        """
        y = linear(x, self.weight)
        if world_size > 1:
            # [Distributed] all_reduce(SUM) over all ranks — each rank computes a partial
            # matrix product with its weight partition; summing across ranks reconstructs the
            # full output. Participating ranks: all ranks in default process group.
            # Data: (*, out_features) tensor.
            dist.all_reduce(y)
        if self.bias is not None:
            # [Assumptions Made] Bias added after all_reduce — ensures bias is applied once
            # to the aggregated result rather than redundantly on each rank.
            y += self.bias
        return y


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (RMSNorm).

    Normalizes the input tensor by dividing by the root mean square of its elements
    along the last dimension, then scales by a learnable weight parameter.

    Args:
        dim (int): Dimension of the normalization (last dimension size).
        eps (float): Epsilon value for numerical stability. Defaults to 1e-6.

    Attributes:
        weight (nn.Parameter): Learnable scale parameter of shape ``(dim,)``.

    Notes:
        Uses RMSNorm instead of LayerNorm — RMSNorm omits the mean-centering step,
        reducing computation by ~50% while achieving comparable training stability
        for Transformer architectures. Reference: Zhang & Sennrich, 2019.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        """Forward pass for RMSNorm.

        Args:
            x (torch.Tensor): Input tensor of shape ``(*, dim)``.

        Returns:
            torch.Tensor: Normalized tensor of shape ``(*, dim)``, same as input.
        """
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)


def precompute_freqs_cis(args: ModelArgs) -> torch.Tensor:
    """Precomputes YaRN-corrected rotary positional embedding frequency tensors.

    Generates complex exponential values for Rotary Position Embedding (RoPE),
    with optional YaRN frequency correction when the target sequence length exceeds
    the original pre-training context length. YaRN applies dimension-selective scaling:
    low-frequency components (many rotations in the original context) are interpolated,
    high-frequency components (few rotations) are kept at original scale, with a smooth
    ramp in between.

    Args:
        args (ModelArgs): Model arguments containing RoPE parameters:
            qk_rope_head_dim, max_seq_len, original_seq_len, rope_theta, rope_factor,
            beta_fast, beta_slow.

    Returns:
        torch.Tensor: Complex tensor of shape ``(max_seq_len, qk_rope_head_dim // 2)``
            where each element is a unit-magnitude complex exponential encoding position
            and frequency information.

    Notes:
        When max_seq_len <= original_seq_len, standard RoPE frequencies are used without
        YaRN correction. The returned complex tensor is used by ``apply_rotary_emb()``
        via complex multiplication.

        Reference: arXiv:2412.19437 for DeepSeek-V3's RoPE parameters;
        YaRN (Yet another RoPE extensioN) paper for the correction algorithm.
    """
    dim = args.qk_rope_head_dim
    seqlen = args.max_seq_len
    beta_fast = args.beta_fast
    beta_slow = args.beta_slow
    base = args.rope_theta
    factor = args.rope_factor

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        """Finds the frequency dimension index for a given rotation count threshold.

        Computes which frequency dimension would complete exactly ``num_rotations``
        full rotations within ``max_seq_len`` positions. Used to determine the
        boundary between dimensions that need frequency interpolation and those
        that can be extrapolated safely.

        Args:
            num_rotations (float): Target number of full rotations within the context.
            dim (int): Total dimensionality of the RoPE embedding space.
            base (float): Base value for the geometric frequency sequence.
            max_seq_len (int): Maximum sequence length to evaluate against.

        Returns:
            float: The (continuous) dimension index at which the rotation threshold is met.
        """
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        """Defines the transition zone for YaRN frequency correction.

        Identifies the range of frequency dimensions between high-frequency (extrapolated)
        and low-frequency (interpolated) regimes. Dimensions below ``low`` are fully
        interpolated, dimensions above ``high`` are fully extrapolated, and dimensions
        in between receive a smoothly ramped blend.

        Args:
            low_rot (float): Lower rotation count threshold (beta_fast — high-frequency boundary).
            high_rot (float): Upper rotation count threshold (beta_slow — low-frequency boundary).
            dim (int): Total dimensionality of the RoPE embedding space.
            base (float): Base value for the geometric frequency sequence.
            max_seq_len (int): Original pre-training sequence length.

        Returns:
            Tuple[int, int]: The range of correction dimension indices (low, high),
                clamped to valid dimension indices [0, dim-1].
        """
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim-1)

    def linear_ramp_factor(min, max, dim):
        """Creates a smooth transition mask for blending original and scaled frequencies.

        Generates a linear ramp from 0 to 1 across the specified dimension range,
        clamped to [0, 1]. Used to smoothly blend between interpolated and original
        frequency values in the YaRN correction.

        Args:
            min (float): Start dimension index for the ramp (value = 0 at this point).
            max (float): End dimension index for the ramp (value = 1 at this point).
            dim (int): Total number of frequency dimensions (qk_rope_head_dim // 2).

        Returns:
            torch.Tensor: Ramp tensor of shape ``(dim,)`` with values in [0, 1].
        """
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    # [Assumptions Made] Standard RoPE frequency computation — geometric sequence of
    # frequencies from 1/base^0 to 1/base^1 with dim/2 elements; this is the standard
    # RoPE formulation from Su et al., 2021.
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if seqlen > args.original_seq_len:
        # [Alternatives Considered] YaRN frequency correction instead of simple NTK scaling —
        # YaRN applies different scaling factors to different frequency dimensions: low
        # frequencies (covering many rotations) are interpolated, high frequencies (few
        # rotations) are kept original, with a smooth ramp in between. This preserves local
        # attention patterns while extending context. Reference: arXiv:2412.19437
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, args.original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    # [Assumptions Made] Complex exponential representation — torch.polar creates complex
    # numbers from magnitude (1) and angle (freq*position), enabling efficient rotary
    # embedding application via complex multiplication.
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Applies rotary positional embeddings via complex number multiplication.

    Converts adjacent pairs of real dimensions into complex numbers, multiplies by
    the precomputed complex exponentials (encoding position and frequency), then
    converts back to real representation. This is mathematically equivalent to applying
    2×2 rotation matrices but leverages PyTorch's optimized complex arithmetic.

    Args:
        x (torch.Tensor): Input tensor of shape
            ``(batch_size, seq_len, n_heads, qk_rope_head_dim)`` containing the
            RoPE-eligible portion of query or key vectors.
        freqs_cis (torch.Tensor): Precomputed complex exponential tensor of shape
            ``(seq_len, qk_rope_head_dim // 2)`` from ``precompute_freqs_cis()``.

    Returns:
        torch.Tensor: Tensor with rotary embeddings applied, same shape as input
            ``(batch_size, seq_len, n_heads, qk_rope_head_dim)``.
    """
    dtype = x.dtype
    # [Trade-offs] Convert to complex representation for RoPE — pairs adjacent real
    # dimensions into complex numbers, enabling rotation via complex multiplication; this
    # is mathematically equivalent to the 2x2 rotation matrix formulation but leverages
    # PyTorch's optimized complex arithmetic.
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    # [Assumptions Made] Convert back to real representation and flatten — restores the
    # original tensor layout after complex multiplication; the float() → dtype cast handles
    # potential precision differences.
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    return y.to(dtype)


class MLA(nn.Module):
    """Multi-Head Latent Attention (MLA) — DeepSeek-V3's key architectural innovation.

    Implements the latent compression scheme where key-value representations are jointly
    compressed into a low-rank latent space, dramatically reducing KV cache memory from
    O(n · d_head · n_heads) to O(n · kv_lora_rank + n · qk_rope_head_dim) per layer.

    The architecture has three key components:
    1. **Query compression (optional):** When q_lora_rank > 0 (236B, 671B models), queries
       are compressed via LoRA: dim → q_lora_rank → n_heads * qk_head_dim. When q_lora_rank
       == 0 (16B model), a direct projection is used.
    2. **Joint KV compression:** Input is projected to kv_lora_rank + qk_rope_head_dim dims.
       The first kv_lora_rank dimensions form the compressed KV latent; the remaining
       qk_rope_head_dim dimensions provide position-dependent features for decoupled RoPE.
    3. **Two attention strategies:**
       - "naive": Materializes full K, V tensors per head from the latent projection.
         Straightforward but memory-intensive KV cache: O(bsz * seq * n_heads * head_dim).
       - "absorb": Fuses the K projection into the query space via matrix absorption,
         caching only the compressed latent + RoPE component. Memory-efficient KV cache:
         O(bsz * seq * (kv_lora_rank + qk_rope_head_dim)). This is the default and
         recommended mode.

    Reference: arXiv:2412.19437, Section 2.1.1

    Attributes:
        dim (int): Model hidden dimension.
        n_heads (int): Total number of attention heads across all ranks.
        n_local_heads (int): Attention heads on this rank (n_heads // world_size).
        q_lora_rank (int): Query LoRA rank (0 disables LoRA compression).
        kv_lora_rank (int): Joint KV latent compression rank.
        qk_nope_head_dim (int): Per-head non-positional Q/K dimension.
        qk_rope_head_dim (int): Per-head RoPE-encoded Q/K dimension.
        qk_head_dim (int): Total Q/K per-head dimension (nope + rope).
        v_head_dim (int): Per-head value dimension.
        softmax_scale (float): Attention score scaling factor (1/sqrt(qk_head_dim),
            adjusted by mscale for extended context).

        KV cache buffers (depends on attn_impl):
            For "naive": k_cache ``(max_batch_size, max_seq_len, n_local_heads, qk_head_dim)``,
                v_cache ``(max_batch_size, max_seq_len, n_local_heads, v_head_dim)``
            For "absorb": kv_cache ``(max_batch_size, max_seq_len, kv_lora_rank)``,
                pe_cache ``(max_batch_size, max_seq_len, qk_rope_head_dim)``
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads // world_size
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim

        # [Trade-offs] Conditional query LoRA — when q_lora_rank == 0 (16B model), use direct
        # projection for simplicity; when > 0 (236B, 671B), use LoRA decomposition
        # (wq_a → q_norm → wq_b) to reduce parameter count from dim*n_heads*qk_head_dim
        # to dim*q_lora_rank + q_lora_rank*n_heads*qk_head_dim.
        # Reference: arXiv:2412.19437, Section 2.1.1
        if self.q_lora_rank == 0:
            self.wq = ColumnParallelLinear(self.dim, self.n_heads * self.qk_head_dim)
        else:
            self.wq_a = Linear(self.dim, self.q_lora_rank)
            self.q_norm = RMSNorm(self.q_lora_rank)
            self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.qk_head_dim)
        # [Trade-offs] Joint KV compression with decoupled RoPE — projects input to
        # kv_lora_rank + qk_rope_head_dim dimensions; the first kv_lora_rank dimensions form
        # the compressed KV latent, the remaining qk_rope_head_dim dimensions provide
        # position-dependent features for RoPE. Reference: arXiv:2412.19437, Section 2.1.1
        self.wkv_a = Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        self.wkv_b = ColumnParallelLinear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim))
        self.wo = RowParallelLinear(self.n_heads * self.v_head_dim, self.dim)
        self.softmax_scale = self.qk_head_dim ** -0.5
        if args.max_seq_len > args.original_seq_len:
            # [Trade-offs] Adjusted softmax scale for extended sequences — standard scale is
            # 1/sqrt(qk_head_dim); when using YaRN-extended context (max_seq_len >
            # original_seq_len), the mscale factor compensates for the entropy increase in
            # longer attention distributions. The formula 0.1 * mscale * log(rope_factor) + 1.0
            # is empirically tuned.
            mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

        # [Alternatives Considered] Two cache strategies — "naive" caches explicit K,V tensors
        # per head (standard but memory-intensive: O(bsz * seq_len * n_heads * head_dim));
        # "absorb" caches the compressed KV latent + RoPE component (memory-efficient:
        # O(bsz * seq_len * (kv_lora_rank + qk_rope_head_dim))), requiring on-the-fly
        # projection during attention but dramatically reducing memory for long sequences.
        if attn_impl == "naive":
            self.register_buffer("k_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.qk_head_dim), persistent=False)
            self.register_buffer("v_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.v_head_dim), persistent=False)
        else:
            self.register_buffer("kv_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.kv_lora_rank), persistent=False)
            self.register_buffer("pe_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.qk_rope_head_dim), persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        """Forward pass for MLA with latent KV compression and two attention strategies.

        Processes input through query projection, joint KV compression, RoPE application,
        and attention computation using either the "naive" or "absorb" strategy.

        Tensor shape flow:
            1. Input: x ``(batch_size, seq_len, dim)``
            2. Query projection: q ``(bsz, seqlen, n_local_heads, qk_head_dim)``
            3. Query split: q_nope ``(bsz, seqlen, n_local_heads, qk_nope_head_dim)``,
               q_pe ``(bsz, seqlen, n_local_heads, qk_rope_head_dim)``
            4. KV projection: kv ``(bsz, seqlen, kv_lora_rank)``,
               k_pe ``(bsz, seqlen, 1, qk_rope_head_dim)``
            5a. Naive path — k ``(bsz, seqlen, n_local_heads, qk_head_dim)``,
                v ``(bsz, seqlen, n_local_heads, v_head_dim)``
            5b. Absorb path — q_nope absorbed to ``(bsz, seqlen, n_local_heads, kv_lora_rank)``
            6. Attention scores: ``(bsz, seqlen, n_local_heads, total_len)``
            7. Output: ``(batch_size, seq_len, dim)``

        Args:
            x (torch.Tensor): Input tensor of shape ``(batch_size, seq_len, dim)``.
            start_pos (int): Starting position in the KV cache for this step.
            freqs_cis (torch.Tensor): Precomputed RoPE complex exponentials of shape
                ``(seq_len, qk_rope_head_dim // 2)``.
            mask (Optional[torch.Tensor]): Causal attention mask of shape
                ``(seq_len, seq_len)`` with -inf for masked positions, or None
                for single-token decode steps.

        Returns:
            torch.Tensor: Output tensor of shape ``(batch_size, seq_len, dim)``.
        """
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            q = self.wq_b(self.q_norm(self.wq_a(x)))
        q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb(q_pe, freqs_cis)
        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)
        if attn_impl == "naive":
            # [Trade-offs] Naive attention — materializes full K,V per head from the latent
            # projection; straightforward implementation for debugging/validation but consumes
            # more memory for the KV cache. k_pe is broadcast-expanded across heads since
            # position encoding is shared.
            q = torch.cat([q_nope, q_pe], dim=-1)
            kv = self.wkv_b(self.kv_norm(kv))
            kv = kv.view(bsz, seqlen, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)], dim=-1)
            self.k_cache[:bsz, start_pos:end_pos] = k
            self.v_cache[:bsz, start_pos:end_pos] = v
            scores = torch.einsum("bshd,bthd->bsht", q, self.k_cache[:bsz, :end_pos]) * self.softmax_scale
        else:
            # [Trade-offs] Absorbed attention — the key computational trick of MLA: instead of
            # projecting latent→K then computing Q·K^T, absorb the K projection (wkv_b) into
            # the query by computing Q·W_kv_b^T, then dot with the cached latent directly.
            # This avoids materializing the full K tensor and reduces KV cache to just the
            # latent + RoPE components. Scores are computed as two separate terms:
            # (q_nope absorbed via wkv_b) · kv_cache + q_pe · pe_cache.
            # Reference: arXiv:2412.19437, Section 2.1.1
            wkv_b = self.wkv_b.weight if self.wkv_b.scale is None else weight_dequant(self.wkv_b.weight, self.wkv_b.scale, block_size) 
            wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
            q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])
            self.kv_cache[:bsz, start_pos:end_pos] = self.kv_norm(kv)
            self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)
            scores = (torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos]) +
                      torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])) * self.softmax_scale
        if mask is not None:
            scores += mask.unsqueeze(1)
        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
        if attn_impl == "naive":
            x = torch.einsum("bsht,bthd->bshd", scores, self.v_cache[:bsz, :end_pos])
        else:
            # [Trade-offs] Absorbed value projection — similarly to keys, the value extraction
            # is absorbed: first compute attention-weighted sum of cached latents, then project
            # through the value portion of wkv_b. This avoids ever materializing the full V
            # tensor in absorb mode.
            x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos])
            x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])
        x = self.wo(x.flatten(2))
        return x


class MLP(nn.Module):
    """SwiGLU feed-forward network with tensor parallelism.

    Implements the gated linear unit activation pattern used in DeepSeek-V3's dense
    feed-forward layers. Formula: output = w2(SiLU(w1(x)) * w3(x)), where w1 provides
    the gate signal, w3 provides the value, and w2 projects back to model dimension.

    Uses ColumnParallelLinear for w1/w3 (output-partitioned) and RowParallelLinear for
    w2 (input-partitioned with all_reduce), enabling tensor-parallel execution.

    Args:
        dim (int): Input and output dimensionality (model hidden size).
        inter_dim (int): Hidden layer dimensionality (SwiGLU intermediate size).

    Attributes:
        w1 (ColumnParallelLinear): Gate projection, shape ``(dim, inter_dim/world_size)``.
        w2 (RowParallelLinear): Down-projection, shape ``(inter_dim/world_size, dim)``.
        w3 (ColumnParallelLinear): Value projection, shape ``(dim, inter_dim/world_size)``.

    Notes:
        Tensor shapes through the forward pass:
            Input x: ``(*, dim)`` → w1/w3: ``(*, inter_dim/world_size)`` → w2: ``(*, dim)``
    """
    def __init__(self, dim: int, inter_dim: int):
        """Initializes the SwiGLU MLP layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality (partitioned across ranks).
        """
        super().__init__()
        self.w1 = ColumnParallelLinear(dim, inter_dim)
        self.w2 = RowParallelLinear(inter_dim, dim)
        self.w3 = ColumnParallelLinear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass applying SwiGLU activation.

        Args:
            x (torch.Tensor): Input tensor of shape ``(*, dim)``.

        Returns:
            torch.Tensor: Output tensor of shape ``(*, dim)``.
        """
        # [Alternatives Considered] SwiGLU activation (SiLU-gated linear unit) instead of
        # standard ReLU/GELU FFN — SwiGLU provides smoother gradients and better training
        # dynamics for large Transformers; the gated formulation (w1 for gate, w3 for value)
        # uses 3 projections instead of 2 but yields better quality-compute tradeoffs.
        # Reference: Shazeer, 2020 (GLU variants)
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Gate(nn.Module):
    """Expert routing gate with auxiliary-loss-free load balancing for DeepSeekMoE.

    Implements the two-stage routing mechanism from DeepSeekMoE:
    1. Compute expert affinity scores via linear projection + scoring function.
    2. (Optional) Add learned per-expert bias for load balancing without auxiliary loss.
    3. (If n_groups > 1) Select top-k expert groups via hierarchical group routing.
    4. Select top-k experts from the remaining candidates.
    5. Extract routing weights from the original (pre-bias) scores for the selected experts.

    The bias parameter enables load balancing without adding auxiliary loss terms to the
    training objective, avoiding the representational damage that auxiliary losses can cause.
    The bias is only enabled for the 671B model (dim == 7168) where 256 routed experts
    make load balancing most critical.

    Reference: arXiv:2412.19437, Section 2.1.2

    Attributes:
        dim (int): Model hidden dimension (used to determine bias enablement).
        topk (int): Number of experts activated per token (n_activated_experts).
        n_groups (int): Number of expert groups for hierarchical routing.
        topk_groups (int): Number of groups selected in the first routing stage.
        score_func (str): Expert scoring function ("softmax" or "sigmoid").
        route_scale (float): Multiplicative scaling for final routing weights.
        weight (nn.Parameter): Gate projection matrix of shape ``(n_routed_experts, dim)``.
        bias (nn.Parameter or None): Per-expert bias of shape ``(n_routed_experts,)``
            for auxiliary-loss-free load balancing. Only allocated when dim == 7168.
    """
    def __init__(self, args: ModelArgs):
        """Initializes the Gate module with routing parameters.

        Args:
            args (ModelArgs): Model arguments containing gating parameters including
                dim, n_activated_experts, n_expert_groups, n_limited_groups,
                score_func, route_scale, and n_routed_experts.
        """
        super().__init__()
        self.dim = args.dim
        self.topk = args.n_activated_experts
        self.n_groups = args.n_expert_groups
        self.topk_groups = args.n_limited_groups
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
        # [Assumptions Made] Bias enabled only when dim == 7168 (671B model) — the
        # auxiliary-loss-free bias mechanism is specifically designed for the largest model
        # variant where expert load balancing is most critical due to 256 routed experts;
        # smaller models (16B with dim=2048, 236B with dim=5120) use fewer experts and
        # don't require bias-based balancing. Reference: arXiv:2412.19437, Section 2.1.2
        self.bias = nn.Parameter(torch.empty(args.n_routed_experts, dtype=torch.float32)) if self.dim == 7168 else None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass computing expert routing weights and indices.

        Tensor shape flow:
            1. Input x: ``(num_tokens, dim)`` (flattened from batch)
            2. Raw scores: ``(num_tokens, n_routed_experts)``
            3. If n_groups > 1: reshaped to ``(num_tokens, n_groups, experts_per_group)``
            4. Output weights: ``(num_tokens, topk)``, indices: ``(num_tokens, topk)``

        Args:
            x (torch.Tensor): Input tensor of shape ``(num_tokens, dim)`` where
                num_tokens = batch_size * seq_len (pre-flattened by MoE.forward).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - weights: Routing weights of shape ``(num_tokens, topk)`` scaled by
                  route_scale, cast to input dtype.
                - indices: Selected expert indices of shape ``(num_tokens, topk)``
                  with values in [0, n_routed_experts).
        """
        scores = linear(x, self.weight)
        # [Alternatives Considered] Softmax vs sigmoid scoring — softmax normalizes scores
        # to sum to 1 (competitive routing, used by smaller models); sigmoid allows
        # independent expert activation probabilities (non-competitive, used by 671B with
        # score_func="sigmoid"), which is more compatible with the bias-based load balancing
        # mechanism. Reference: arXiv:2412.19437, Section 2.1.2
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        else:
            scores = scores.sigmoid()
        original_scores = scores
        if self.bias is not None:
            # [Trade-offs] Auxiliary-loss-free load balancing via learned bias — adds a
            # per-expert bias to routing scores to influence expert selection without modifying
            # the training loss; the bias is adjusted during training to promote balanced
            # expert utilization. This avoids the representational damage caused by auxiliary
            # load-balancing losses. Reference: arXiv:2412.19437, Section 2.1.2
            scores = scores + self.bias
        if self.n_groups > 1:
            # [Trade-offs] Two-stage routing: group selection then expert selection — first
            # selects topk_groups expert groups, then selects topk experts from those groups;
            # this hierarchical approach constrains cross-node communication in distributed
            # MoE by limiting which groups of experts can be activated. Masked groups receive
            # -inf scores to prevent selection.
            scores = scores.view(x.size(0), self.n_groups, -1)
            if self.bias is None:
                group_scores = scores.amax(dim=-1)
            else:
                group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            mask = scores.new_ones(x.size(0), self.n_groups, dtype=bool).scatter_(1, indices, False)
            scores = scores.masked_fill_(mask.unsqueeze(-1), float("-inf")).flatten(1)
        indices = torch.topk(scores, self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func == "sigmoid":
            # [Assumptions Made] Normalize sigmoid weights to sum to 1 — since sigmoid scores
            # are independent (not inherently normalized like softmax), explicit normalization
            # ensures the expert-weighted combination maintains consistent scale.
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale
        return weights.type_as(x), indices


class Expert(nn.Module):
    """Individual expert MLP within the Mixture-of-Experts layer.

    Uses the same SwiGLU activation pattern as MLP (output = w2(SiLU(w1(x)) * w3(x)))
    but with non-parallel base Linear layers instead of ColumnParallel/RowParallel
    variants, since experts are already partitioned across ranks by the MoE module.

    Args:
        dim (int): Input and output dimensionality (model hidden size).
        inter_dim (int): Hidden layer dimensionality (MoE intermediate size,
            typically different from the dense MLP inter_dim).

    Attributes:
        w1 (Linear): Gate projection, shape ``(dim, inter_dim)``.
        w2 (Linear): Down-projection, shape ``(inter_dim, dim)``.
        w3 (Linear): Value projection, shape ``(dim, inter_dim)``.

    Notes:
        Tensor shapes through the forward pass:
            Input x: ``(num_assigned_tokens, dim)`` → w1/w3: ``(num_assigned_tokens,
            moe_inter_dim)`` → w2: ``(num_assigned_tokens, dim)``
    """
    def __init__(self, dim: int, inter_dim: int):
        """Initializes the Expert MLP layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        # [Trade-offs] Uses base Linear instead of ColumnParallel/RowParallelLinear — experts
        # are already partitioned across ranks (each rank owns n_local_experts), so
        # intra-expert parallelism is unnecessary; this avoids redundant all_reduce operations
        # within each expert.
        self.w1 = Linear(dim, inter_dim)
        self.w2 = Linear(inter_dim, dim)
        self.w3 = Linear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass applying SwiGLU activation within a single expert.

        Args:
            x (torch.Tensor): Input tensor of shape ``(num_assigned_tokens, dim)``
                containing only the tokens routed to this expert.

        Returns:
            torch.Tensor: Output tensor of shape ``(num_assigned_tokens, dim)``.
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoE(nn.Module):
    """Mixture-of-Experts (MoE) layer with expert parallelism and shared experts.

    Implements the DeepSeekMoE architecture where experts are partitioned across
    distributed ranks (expert parallelism): each rank owns n_local_experts =
    n_routed_experts / world_size experts and computes outputs only for tokens
    routed to those experts. An all_reduce aggregates partial results across ranks.

    Additionally, shared experts (a standard MLP) process ALL tokens regardless of
    routing, providing a baseline representation that supplements the routed experts.
    The shared expert output is added after the all_reduce of routed expert outputs.

    Reference: arXiv:2412.19437, Section 2.1.2

    Attributes:
        dim (int): Model hidden dimension.
        n_routed_experts (int): Total number of routed experts across all ranks.
        n_local_experts (int): Experts on this rank (n_routed_experts // world_size).
        n_activated_experts (int): Number of experts activated per token.
        experts_start_idx (int): Global index of the first expert on this rank.
        experts_end_idx (int): Global index past the last expert on this rank.
        gate (Gate): Expert routing gate with optional bias-based load balancing.
        experts (nn.ModuleList): List of Expert modules, with None placeholders for
            experts not owned by this rank (length = n_routed_experts).
        shared_experts (MLP): Shared expert MLP processing all tokens, with
            intermediate dimension = n_shared_experts * moe_inter_dim.
    """
    def __init__(self, args: ModelArgs):
        """Initializes the MoE module with expert partitioning.

        Args:
            args (ModelArgs): Model arguments containing MoE parameters including
                n_routed_experts, n_activated_experts, moe_inter_dim, and
                n_shared_experts.

        Raises:
            AssertionError: If n_routed_experts is not divisible by world_size.
        """
        super().__init__()
        self.dim = args.dim
        assert args.n_routed_experts % world_size == 0, f"Number of experts must be divisible by world size (world_size={world_size})"
        self.n_routed_experts = args.n_routed_experts
        self.n_local_experts = args.n_routed_experts // world_size
        self.n_activated_experts = args.n_activated_experts
        self.experts_start_idx = rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.gate = Gate(args)
        self.experts = nn.ModuleList([Expert(args.dim, args.moe_inter_dim) if self.experts_start_idx <= i < self.experts_end_idx else None
                                      for i in range(self.n_routed_experts)])
        self.shared_experts = MLP(args.dim, args.n_shared_experts * args.moe_inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with token-level expert routing and aggregation.

        Tensor shape flow:
            1. Input x: ``(batch_size, seq_len, dim)``
            2. Flatten: ``(batch_size * seq_len, dim)`` for per-token routing
            3. Gate returns weights ``(num_tokens, topk)``, indices ``(num_tokens, topk)``
            4. Each expert processes its assigned tokens: ``(num_assigned, dim)``
            5. Routed output y: ``(num_tokens, dim)`` (accumulated weighted expert outputs)
            6. Shared expert output z: ``(num_tokens, dim)``
            7. Output: ``(batch_size, seq_len, dim)`` after reshape

        Args:
            x (torch.Tensor): Input tensor of shape ``(batch_size, seq_len, dim)``.

        Returns:
            torch.Tensor: Output tensor of shape ``(batch_size, seq_len, dim)``
                combining routed expert outputs and shared expert output.
        """
        shape = x.size()
        # [Assumptions Made] Flatten batch and sequence dimensions for token-level expert
        # routing — MoE routing operates on individual tokens, not sequences; reshaping to
        # (num_tokens, dim) enables per-token expert assignment.
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x)
        y = torch.zeros_like(x)
        # [Trade-offs] Pre-compute expert assignment counts via bincount — enables skipping
        # experts with zero assigned tokens (count == 0), avoiding unnecessary computation;
        # the tolist() converts to Python list for efficient iteration.
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0:
                continue
            # [Trade-offs] Sequential expert dispatch with index-gather — iterates through
            # local experts, gathers assigned tokens by index, computes expert output, and
            # accumulates weighted results; this approach is simple but sequential. Production
            # systems often use grouped GEMM for parallelism, but sequential dispatch is
            # sufficient for inference.
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            y[idx] += expert(x[idx]) * weights[idx, top, None]
        # [Trade-offs] Shared experts process ALL tokens — provides a baseline representation
        # independent of routing, ensuring every token receives some expert computation even
        # if routing is suboptimal. Shared expert output is added after the routed expert
        # all_reduce. Reference: arXiv:2412.19437, Section 2.1.2
        z = self.shared_experts(x)
        if world_size > 1:
            # [Distributed] all_reduce(SUM) over all ranks — aggregates routed expert outputs
            # from all ranks; each rank computes outputs only for its local experts
            # (experts_start_idx to experts_end_idx), and all_reduce sums the partial results
            # so every rank has the complete routed expert output. Participating ranks: all
            # ranks in default process group. Data: (num_tokens, dim) tensor.
            # Note: shared expert output (z) is NOT included in the all_reduce because
            # shared_experts uses RowParallelLinear which handles its own all_reduce internally.
            dist.all_reduce(y)
        return (y + z).view(shape)


class Block(nn.Module):
    """Pre-norm Transformer block: RMSNorm → MLA → residual → RMSNorm → FFN → residual.

    Each block follows the pre-normalization architecture where layer normalization is
    applied before each sub-layer (attention and FFN), with residual connections around
    each. The FFN sub-layer is either a dense MLP (for the first n_dense_layers layers)
    or a Mixture-of-Experts layer (for all remaining layers).

    Reference: arXiv:2412.19437, Section 2.1.2 (MoE layer allocation)

    Args:
        layer_id (int): Zero-based layer index in the Transformer stack.
        args (ModelArgs): Model configuration parameters.

    Attributes:
        attn (MLA): Multi-Head Latent Attention sub-layer.
        ffn (MLP or MoE): Feed-forward sub-layer — MLP for dense layers, MoE otherwise.
        attn_norm (RMSNorm): Pre-attention normalization of shape ``(dim,)``.
        ffn_norm (RMSNorm): Pre-FFN normalization of shape ``(dim,)``.

    Notes:
        Tensor shapes: input and output are both ``(batch_size, seq_len, dim)``.
    """
    def __init__(self, layer_id: int, args: ModelArgs):
        """Initializes the Transformer block with attention and FFN sub-layers.

        Args:
            layer_id (int): Zero-based layer index determining MLP vs MoE selection.
            args (ModelArgs): Model arguments containing block parameters.
        """
        super().__init__()
        self.attn = MLA(args)
        # [Trade-offs] Dense MLP for initial layers, MoE for remaining — the first
        # n_dense_layers layers (1-3 depending on model size) use standard MLP to
        # establish foundational representations before expert specialization; subsequent
        # layers use MoE for capacity-efficient scaling.
        # Reference: arXiv:2412.19437, Section 2.1.2
        self.ffn = MLP(args.dim, args.inter_dim) if layer_id < args.n_dense_layers else MoE(args)
        self.attn_norm = RMSNorm(args.dim)
        self.ffn_norm = RMSNorm(args.dim)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Forward pass through the Transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape ``(batch_size, seq_len, dim)``.
            start_pos (int): Starting position in the KV cache for this step.
            freqs_cis (torch.Tensor): Precomputed RoPE complex exponentials of shape
                ``(seq_len, qk_rope_head_dim // 2)``.
            mask (Optional[torch.Tensor]): Causal attention mask of shape
                ``(seq_len, seq_len)`` or None for single-token decode.

        Returns:
            torch.Tensor: Output tensor of shape ``(batch_size, seq_len, dim)``.
        """
        x = x + self.attn(self.attn_norm(x), start_pos, freqs_cis, mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class Transformer(nn.Module):
    """Top-level DeepSeek-V3 Transformer model for autoregressive language modeling.

    Constructs the full model from embedding, Transformer blocks, final normalization,
    and language model head. Critically, ``__init__`` sets global mutable state that
    affects all subsequently-created modules:
        - ``world_size`` and ``rank``: Distributed process topology globals.
        - ``Linear.dtype``: Weight storage dtype for all Linear layers (FP8 or BF16).
        - ``Linear.scale_fmt``: FP8 scale tensor format.

    This non-standard global-state initialization pattern assumes only one Transformer
    is instantiated per process (valid for inference serving).

    The forward pass performs: embed → Transformer blocks → RMSNorm → select last token →
    vocabulary projection → all_gather (if distributed).

    Reference: arXiv:2412.19437

    Attributes:
        max_seq_len (int): Maximum supported sequence length.
        embed (ParallelEmbedding): Token embedding with vocabulary parallelism.
        layers (nn.ModuleList): Stack of ``n_layers`` Transformer Blocks.
        norm (RMSNorm): Final layer normalization of shape ``(dim,)``.
        head (ColumnParallelLinear): Language model head projecting to vocabulary,
            shape ``(dim, vocab_size/world_size)`` per rank. Always uses BF16 dtype.
        freqs_cis (torch.Tensor): Precomputed RoPE frequency complex exponentials
            of shape ``(max_seq_len, qk_rope_head_dim // 2)``.
    """
    def __init__(self, args: ModelArgs):
        """Initializes the Transformer and sets global distributed/dtype state.

        Args:
            args (ModelArgs): Complete model configuration parameters. See ModelArgs
                docstring and CONFIG_REFERENCE.md for parameter documentation.

        Notes:
            Side effects — sets module-level globals (world_size, rank) and class-level
            attributes (Linear.dtype, Linear.scale_fmt) that affect ALL subsequently
            constructed Linear, ColumnParallelLinear, RowParallelLinear, and
            ParallelEmbedding modules.
        """
        # [Assumptions Made] Set global mutable state during model construction — world_size,
        # rank, Linear.dtype, and Linear.scale_fmt are set once here and affect ALL
        # subsequently-created Linear layers and distributed operations. This non-standard
        # pattern avoids threading distributed config through every constructor but assumes
        # only one Transformer is instantiated per process.
        global world_size, rank
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        # [Trade-offs] FP8 dtype for weight storage when config specifies "fp8" —
        # torch.float8_e4m3fn is a 1-byte format (4-bit exponent, 3-bit mantissa) that
        # halves weight memory vs BF16; the linear() function handles dispatch based on
        # this dtype. Reference: arXiv:2412.19437, Section 3.3
        Linear.dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
        Linear.scale_fmt = args.scale_fmt
        super().__init__()
        self.max_seq_len = args.max_seq_len
        self.embed = ParallelEmbedding(args.vocab_size, args.dim)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(Block(layer_id, args))
        self.norm = RMSNorm(args.dim)
        # [Assumptions Made] Language model head always uses default dtype (BF16) — the
        # vocabulary projection is not quantized to FP8 because precision in the final logit
        # computation is critical for generation quality; using torch.get_default_dtype()
        # explicitly overrides the Linear.dtype class attribute.
        self.head = ColumnParallelLinear(args.dim, args.vocab_size, dtype=torch.get_default_dtype())
        self.register_buffer("freqs_cis", precompute_freqs_cis(args), persistent=False)

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int = 0):
        """Forward pass producing next-token logits over the full vocabulary.

        Tensor shape flow:
            1. tokens input: ``(batch_size, seq_len)``
            2. After embed: h ``(batch_size, seq_len, dim)``
            3. After all Transformer blocks: h ``(batch_size, seq_len, dim)``
            4. After norm + last token selection: h ``(batch_size, dim)``
            5. After head: logits ``(batch_size, part_vocab_size)`` per rank
            6. After all_gather + cat: logits ``(batch_size, vocab_size)``

        Args:
            tokens (torch.Tensor): Input token IDs of shape ``(batch_size, seq_len)``.
            start_pos (int, optional): Starting position in the KV cache and RoPE
                frequencies. Defaults to 0 (prefill mode).

        Returns:
            torch.Tensor: Logits tensor of shape ``(batch_size, vocab_size)``
                representing unnormalized log-probabilities over the full vocabulary.
        """
        seqlen = tokens.size(1)
        h = self.embed(tokens)
        freqs_cis = self.freqs_cis[start_pos:start_pos+seqlen]
        # [Assumptions Made] Causal attention mask only needed for prefill (seqlen > 1) —
        # during single-token decode steps, the KV cache already excludes future positions;
        # the upper-triangular -inf mask enforces causal attention during multi-token prefill.
        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device).triu_(1)
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        # [Assumptions Made] Select only the last token's hidden state for logit computation —
        # in autoregressive generation, only the last position produces the next-token
        # prediction; selecting h[:, -1] avoids computing logits for all positions.
        h = self.norm(h)[:, -1]
        logits = self.head(h)
        if world_size > 1:
            # [Distributed] all_gather across all ranks — each rank computes logits for its
            # vocabulary partition (part_vocab_size = vocab_size / world_size) via
            # ColumnParallelLinear head; all_gather collects all partitions and concatenates
            # along the vocabulary dimension to reconstruct full vocab logits. Participating
            # ranks: all ranks in default process group. Data: (batch_size, part_vocab_size)
            # per rank → (batch_size, vocab_size) after cat.
            all_logits = [torch.empty_like(logits) for _ in range(world_size)]
            dist.all_gather(all_logits, logits)
            logits = torch.cat(all_logits, dim=-1)
        return logits


if __name__ == "__main__":
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.manual_seed(0)
    args = ModelArgs()
    x = torch.randint(0, args.vocab_size, (2, 128))
    model = Transformer(args)
    print(model(x).size())
