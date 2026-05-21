"""FP8 quantization, dequantization, and matrix multiplication kernel library for DeepSeek-V3 inference.

This module provides Triton GPU kernels and their Python wrapper functions for three core
FP8 (8-bit floating point) operations used throughout the DeepSeek-V3 inference pipeline:

1. **Activation Quantization** (``act_quant_kernel`` / ``act_quant``):
   Quantizes BF16/FP32 activations to FP8 E4M3 format using per-block scaling.
   Each block of 128 elements receives its own scale factor computed from the
   block's absolute maximum value.

2. **Weight Dequantization** (``weight_dequant_kernel`` / ``weight_dequant``):
   Dequantizes FP8 E4M3 weight matrices back to BF16/FP32 using 2D block-wise
   scale factors. Each (128 x 128) tile of the weight matrix has a corresponding
   scale factor stored in the scale_inv tensor.

3. **FP8 Matrix Multiplication** (``fp8_gemm_kernel`` / ``fp8_gemm``):
   Performs blocked matrix multiplication on FP8 inputs with per-block scale
   accumulation, computing C = A x B^T where each K-block's dot product is
   corrected by the corresponding A and B scale factors before FP32 accumulation.

FP8 E4M3 Format:
    The E4M3 (4-bit exponent, 3-bit mantissa) variant of FP8 provides a dynamic
    range of approximately +/-448 with reduced precision compared to BF16. The
    maximum representable positive value is 448.0. This format is used for both
    activations and weights in the DeepSeek-V3 FP8 quantized inference mode.

Block-Level Quantization Granularity:
    All operations use a 128-element (1D) or 128x128-element (2D) block granularity,
    matching the block-wise quantization scheme used during DeepSeek-V3 training.
    This granularity provides a balance between quantization accuracy (finer blocks
    reduce quantization error) and scale factor storage overhead.

Module Relationships:
    - Imported by ``model.py``: The ``linear()`` function dispatches to ``fp8_gemm()``
      for FP8 weight inference and ``act_quant()`` for activation quantization.
    - Imported by ``fp8_cast_bf16.py``: The ``weight_dequant()`` function is used to
      convert FP8 checkpoint weights back to BF16 for non-FP8 inference.

Reference:
    arXiv:2412.19437, Section 3.3 -- FP8 Training: Documents the block-wise FP8
    quantization scheme with 128x128 tile granularity used during DeepSeek-V3 training,
    which these kernels mirror for inference.
    See also: README_WEIGHTS.md for the FP8 weight file format specification
    (e4m3 format, 128x128 block size, scale_inv tensor layout).
"""
from typing import Tuple, Optional

import torch
import triton
import triton.language as tl
from triton import Config


@triton.jit
def act_quant_kernel(x_ptr, y_ptr, s_ptr, BLOCK_SIZE: tl.constexpr, scale_fmt: tl.constexpr):
    """Quantizes a contiguous block of activations from BF16/FP32 to FP8 E4M3 format with per-block scaling.

    This Triton kernel processes one block of BLOCK_SIZE contiguous elements per program
    instance. For each block, it:
    1. Loads BLOCK_SIZE elements from the input activation tensor.
    2. Computes the per-block absolute maximum (amax) via reduction.
    3. Derives a scale factor as amax / 448 (where 448 is the max representable FP8 E4M3 value).
    4. Optionally rounds the scale to the nearest power of 2 (ue8m0 format).
    5. Divides activations by the scale and casts to FP8 E4M3.

    Args:
        x_ptr (triton.Pointer): Pointer to contiguous input activation tensor in BF16 or FP32.
            Shape: flat block of BLOCK_SIZE elements starting at offset ``pid * BLOCK_SIZE``.
        y_ptr (triton.Pointer): Pointer to output tensor for FP8 E4M3 quantized activations.
            Shape: same as x_ptr; receives quantized values with dtype float8_e4m3fn.
        s_ptr (triton.Pointer): Pointer to output tensor for per-block float32 scale factors.
            Shape: one scalar per program instance (indexed by pid); scale = amax / 448.
        BLOCK_SIZE (tl.constexpr): Number of elements processed per program instance.
            Typically 128, matching the training-time quantization block size.
        scale_fmt (tl.constexpr): Optional scale format string. When set to ``"ue8m0"``
            (unsigned-exponent-8-mantissa-0), the scale factor is rounded to the nearest
            power of 2 via ceil(log2(s)) then exp2. When None, the raw float32 scale is used.

    Returns:
        None: Results written to y_ptr (quantized activations) and s_ptr (scale factors).

    Notes:
        The FP8 E4M3 format has 4 exponent bits and 3 mantissa bits, providing a
        representable range of approximately [-448, 448]. The scale factor maps each
        block's dynamic range into this representable range.
        Reference: arXiv:2412.19437, Section 3.3 -- block-wise FP8 quantization scheme.
    """
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs).to(tl.float32)
    # [Assumptions Made] Per-block absolute maximum for scale derivation — each block of BLOCK_SIZE
    # elements gets its own scale factor, providing finer granularity than per-tensor quantization
    # while amortizing scale storage overhead. Reference: arXiv:2412.19437, Section 3.3
    amax = tl.max(tl.abs(x)) # reduction
    # [Trade-offs] Clamp amax floor to 1e-4 — prevents division by zero for all-zero blocks and
    # avoids extreme scale factors for near-zero blocks; 1e-4 is small enough to not affect normal
    # activation ranges while providing numerical safety
    amax = tl.maximum(amax, 1e-4) # clamp to 1e-4
    # [Assumptions Made] Scale factor = amax / 448 — 448 is the maximum positive value representable
    # in FP8 E4M3 format (2^8 x (1 + 0.875) = 480 is theoretical max, but 448 = 2^8 x 1.75 is the
    # practical max after accounting for special values); dividing by this maps the block's maximum
    # to the format's representable range
    s = amax / 448.
    # [Alternatives Considered] Optional ue8m0 scale format — rounds scale to nearest power-of-2
    # by taking ceil(log2(s)) then exp2; this constrains scales to powers of 2 which enables
    # integer-only scale arithmetic in hardware, potentially improving throughput on specialized
    # accelerators. Used by config_v3.1.json (scale_fmt: "ue8m0")
    if scale_fmt == "ue8m0":
        exp = tl.math.ceil(tl.math.log2(s))
        s = tl.math.exp2(exp)
    y = x / s
    y = y.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, y)
    tl.store(s_ptr + pid, s)


def act_quant(x: torch.Tensor, block_size: int = 128, scale_fmt: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantizes an input activation tensor to FP8 E4M3 format using block-wise scaling.

    Wraps the ``act_quant_kernel`` Triton kernel, handling tensor allocation and grid
    launch configuration. Each contiguous block of ``block_size`` elements in the flattened
    input is independently quantized with its own scale factor.

    Args:
        x (torch.Tensor): Input activation tensor to quantize. Must be contiguous.
            Shape: ``(*, K)`` where ``*`` is any number of leading dimensions and ``K``
            (the last dimension) must be divisible by ``block_size``.
            Dtype: BF16 or FP32.
        block_size (int, optional): Number of elements per quantization block. Default is 128,
            matching the DeepSeek-V3 training-time quantization granularity.
        scale_fmt (Optional[str], optional): Scale format override. Set to ``"ue8m0"`` for
            power-of-2 scale factors (used by v3.1 config). Default is None (raw float32 scales).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple of:
            - Quantized activations: shape ``(*, K)``, dtype ``torch.float8_e4m3fn``.
            - Scale factors: shape ``(*, K // block_size)``, dtype ``torch.float32``.
              One scale factor per block of ``block_size`` elements.

    Raises:
        AssertionError: If ``x`` is not contiguous.
        AssertionError: If ``x.size(-1)`` is not divisible by ``block_size``.

    Notes:
        Grid launch: one Triton program instance per block of ``block_size`` elements;
        total programs = ``x.numel() // block_size``.
        Called by ``model.py:linear()`` for FP8 activation quantization during inference.
        Reference: arXiv:2412.19437, Section 3.3.
    """
    # [Assumptions Made] Input must be contiguous — Triton kernels operate on flat memory with
    # pointer arithmetic; non-contiguous tensors would require stride-aware indexing not
    # implemented in this kernel
    assert x.is_contiguous(), 'Input tensor must be contiguous'
    # [Assumptions Made] Last dimension must be divisible by block_size — ensures each
    # quantization block is complete; partial blocks would require padding logic not
    # implemented here
    assert x.size(-1) % block_size == 0, f'Last dimension size must be divisible by block_size (block_size={block_size})'
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s = x.new_empty(*x.size()[:-1], x.size(-1) // block_size, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(x.numel(), meta['BLOCK_SIZE']), )
    act_quant_kernel[grid](x, y, s, BLOCK_SIZE=block_size, scale_fmt=scale_fmt)
    return y, s


@triton.jit
def weight_dequant_kernel(x_ptr, s_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    """Dequantizes a 2D FP8 weight matrix tile using block-wise scale factors.

    This Triton kernel processes one (BLOCK_SIZE x BLOCK_SIZE) tile of the weight matrix
    per program instance. For each tile, it:
    1. Loads a 2D block of FP8 quantized weights at position (pid_m, pid_n).
    2. Looks up the corresponding scale factor from the 2D scale tensor at index
       ``s[pid_m, pid_n]`` (where the scale tensor has dimensions
       ``(ceil(M/BLOCK_SIZE), ceil(N/BLOCK_SIZE))``).
    3. Multiplies the FP8 values (cast to FP32) by the scale factor.
    4. Stores the dequantized result (in the output dtype).

    Args:
        x_ptr (tl.pointer): Pointer to the quantized weight matrix in FP8 E4M3 format.
            Logical shape: ``(M, N)``, stored in row-major layout.
        s_ptr (tl.pointer): Pointer to the 2D scale factor tensor (scale_inv values).
            Logical shape: ``(ceil(M/BLOCK_SIZE), ceil(N/BLOCK_SIZE))``, stored in
            row-major layout. Each entry is the scale factor for the corresponding
            (BLOCK_SIZE x BLOCK_SIZE) tile of the weight matrix.
        y_ptr (tl.pointer): Pointer to the output buffer for dequantized weights.
            Logical shape: ``(M, N)``, same layout as x_ptr, in the default dtype (BF16).
        M (int): Number of rows in the weight matrix.
        N (int): Number of columns in the weight matrix.
        BLOCK_SIZE (tl.constexpr): Tile dimension for 2D blocking. Typically 128,
            matching the 128x128 block-wise quantization granularity from training.

    Returns:
        None: Dequantized results written to y_ptr.

    Notes:
        The scale tensor is indexed as ``s[pid_m, pid_n]`` where pid_m and pid_n are
        the block row and column indices respectively. The number of block columns
        ``n = ceil(N/BLOCK_SIZE)`` is used to compute the linear offset into the
        row-major scale tensor: ``s_ptr + pid_m * n + pid_n``.
        Boundary masking handles edge tiles where M or N is not a multiple of BLOCK_SIZE.
        Reference: arXiv:2412.19437, Section 3.3 -- block-wise FP8 quantization format.
        See also: README_WEIGHTS.md for scale_inv tensor format documentation.
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    # [Assumptions Made] Compute number of block columns for scale tensor indexing — scale tensor
    # is stored in row-major layout with dimensions (ceil(M/BLOCK_SIZE), ceil(N/BLOCK_SIZE))
    n = tl.cdiv(N, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]
    # [Trade-offs] Boundary masking for non-aligned dimensions — handles weight matrices where M
    # or N is not a multiple of BLOCK_SIZE by masking out-of-bounds accesses; zero-padded loads
    # for safety. Reference: README_WEIGHTS.md documents the weight matrix layout
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    # [Assumptions Made] One scale factor per BLOCK_SIZE x BLOCK_SIZE tile — the scale tensor
    # stores reciprocal scale factors (scale_inv) with 2D indexing: row index = block row (pid_m),
    # column index = block column (pid_n)
    s = tl.load(s_ptr + pid_m * n + pid_n)
    y = x * s
    tl.store(y_ptr + offs, y, mask=mask)


def weight_dequant(x: torch.Tensor, s: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """Dequantizes an FP8 E4M3 weight tensor to the default dtype using block-wise scale factors.

    Wraps the ``weight_dequant_kernel`` Triton kernel, handling contiguity validation,
    output allocation, and 2D grid launch. Each (block_size x block_size) tile of the
    weight matrix is independently dequantized using its corresponding scale factor.

    Args:
        x (torch.Tensor): Quantized weight tensor.
            Shape: ``(M, N)``, dtype ``torch.float8_e4m3fn``.
            Must be contiguous and 2-dimensional.
        s (torch.Tensor): Block-wise scale factor tensor (scale_inv values).
            Shape: ``(ceil(M/block_size), ceil(N/block_size))``, dtype ``torch.float32``.
            Must be contiguous and 2-dimensional.
        block_size (int, optional): Tile dimension for 2D blocking. Default is 128,
            matching the 128x128 training-time quantization granularity.

    Returns:
        torch.Tensor: Dequantized weight tensor.
            Shape: ``(M, N)``, dtype ``torch.get_default_dtype()`` (typically BF16 as set
            by ``generate.py`` or ``fp8_cast_bf16.py`` during initialization).

    Raises:
        AssertionError: If ``x`` or ``s`` are not contiguous.
        AssertionError: If ``x`` or ``s`` do not have exactly 2 dimensions.

    Notes:
        Grid launch is 2D: ``(ceil(M/block_size), ceil(N/block_size))`` blocks, one
        program instance per (block_size x block_size) tile.
        Called by ``fp8_cast_bf16.py:main()`` for FP8-to-BF16 checkpoint conversion and
        by ``model.py:linear()`` when weight dequantization is needed.
        Reference: arXiv:2412.19437, Section 3.3.
    """
    assert x.is_contiguous() and s.is_contiguous(), 'Input tensors must be contiguous'
    assert x.dim() == 2 and s.dim() == 2, 'Input tensors must have 2 dimensions'
    M, N = x.size()
    y = torch.empty_like(x, dtype=torch.get_default_dtype())
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE']), triton.cdiv(N, meta['BLOCK_SIZE']))
    weight_dequant_kernel[grid](x, s, y, M, N, BLOCK_SIZE=block_size)
    return y


# [Trade-offs] Autotuning configuration space for FP8 GEMM — 36 configurations exploring:
#   BLOCK_SIZE_M in {16, 32, 64}: varied for different M dimensions (smaller M for decode phase
#     where batch_size is small, larger M for prefill phase with longer sequences)
#   BLOCK_SIZE_N in {32, 64, 128}: varied for different output feature dimensions
#   BLOCK_SIZE_K = 128 (fixed): matches the 128-element quantization block size, ensuring each
#     K-block aligns exactly with one scale factor — this is critical for per-block scale
#     accumulation correctness
#   num_stages in {3, 4, 5, 6}: software pipelining stages for memory latency hiding; more
#     stages increase register pressure but improve throughput for memory-bound workloads
#   num_warps = 8 (fixed): standard for H100/A100 SM occupancy, balancing parallelism within
#     each thread block
# Triton autotune profiles these 36 configurations on first invocation and caches the best
# config per (N, K) pair (the autotune key). M is excluded from the key because it varies
# between prefill (large M) and decode (small M) within a single session.
fp8_gemm_configs = [
    Config({'BLOCK_SIZE_M': block_m, 'BLOCK_SIZE_N': block_n, 'BLOCK_SIZE_K': 128}, num_stages=num_stages, num_warps=8)
    for block_m in [16, 32, 64] for block_n in [32, 64, 128] for num_stages in [3, 4, 5, 6]
]

@triton.autotune(configs=fp8_gemm_configs, key=['N', 'K'])
@triton.jit
def fp8_gemm_kernel(a_ptr, b_ptr, c_ptr,
                    a_s_ptr, b_s_ptr,
                    M, N: tl.constexpr, K: tl.constexpr,
                    BLOCK_SIZE_M: tl.constexpr,
                    BLOCK_SIZE_N: tl.constexpr,
                    BLOCK_SIZE_K: tl.constexpr):
    """Performs blocked FP8 matrix multiplication C = A x B^T with per-block scale accumulation.

    This Triton kernel computes a matrix product where both input matrices are in FP8 E4M3
    format with associated per-block scale factors. The computation proceeds as:
    1. For each output tile (BLOCK_SIZE_M x BLOCK_SIZE_N), iterate over K in blocks of
       BLOCK_SIZE_K.
    2. At each K-block step ``i``:
       - Load A tile: ``(BLOCK_SIZE_M, BLOCK_SIZE_K)`` from A at K-offset ``i * BLOCK_SIZE_K``.
       - Load B tile: ``(BLOCK_SIZE_K, BLOCK_SIZE_N)`` from B (transposed access: B is stored
         as (N, K) row-major, accessed as B^T).
       - Load per-block scales: ``a_s[m, i]`` and ``b_s[n // BLOCK_SIZE_K, i]``.
       - Accumulate: ``accumulator += dot(A_tile, B_tile) * a_scale * b_scale`` in FP32.
    3. Cast the FP32 accumulator to the output dtype and store with boundary masking.

    Matrix layout:
        - A: ``(M, K)`` row-major — activation matrix
        - B: ``(N, K)`` row-major — weight matrix (accessed as B^T, so effectively (K, N))
        - C: ``(M, N)`` row-major — output matrix

    Args:
        a_ptr (tl.tensor): Pointer to matrix A (activations) in FP8 E4M3.
            Logical shape: ``(M, K)``, row-major layout.
        b_ptr (tl.tensor): Pointer to matrix B (weights) in FP8 E4M3.
            Logical shape: ``(N, K)``, row-major layout (transposed during access to
            compute A x B^T).
        c_ptr (tl.tensor): Pointer to output matrix C.
            Logical shape: ``(M, N)``, row-major layout, in the default dtype (BF16).
        a_s_ptr (tl.tensor): Pointer to per-block scale factors for matrix A.
            Logical shape: ``(M, K // BLOCK_SIZE_K)`` — one scale per K-block per row.
        b_s_ptr (tl.tensor): Pointer to per-block scale factors for matrix B.
            Logical shape: ``(ceil(N / BLOCK_SIZE_K), K // BLOCK_SIZE_K)`` — one scale
            per K-block per N-block group.
        M (int): Number of rows in A and C. Varies per invocation (batch * seq_len).
        N (tl.constexpr): Number of rows in B / columns in C (output features).
            Constexpr for autotune keying.
        K (tl.constexpr): Shared inner dimension (input features).
            Constexpr for autotune keying.
        BLOCK_SIZE_M (tl.constexpr): Tile size for the M dimension. Autotuned from {16, 32, 64}.
        BLOCK_SIZE_N (tl.constexpr): Tile size for the N dimension. Autotuned from {32, 64, 128}.
        BLOCK_SIZE_K (tl.constexpr): Tile size for the K dimension. Fixed at 128 to align
            with the FP8 quantization block size.

    Returns:
        None: Output written to c_ptr.

    Notes:
        Per-block scale accumulation is the key mechanism for maintaining numerical accuracy
        in blocked FP8 GEMM: rather than applying a single global scale post-hoc, each
        K-block's contribution is individually scale-corrected in FP32 before accumulation.
        This matches the training-time quantization granularity.
        Reference: arXiv:2412.19437, Section 3.3 -- block-wise FP8 computation.
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    k = tl.cdiv(K, BLOCK_SIZE_K)
    # [Trade-offs] Modulo indexing for M and N offsets — wraps around to handle cases where
    # the grid dimensions don't exactly cover M and N; this is a Triton pattern for safe
    # out-of-bounds handling that trades potential redundant computation for simpler boundary logic
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    # [Assumptions Made] B matrix stored in row-major (N, K) layout but accessed as transposed —
    # b_ptr uses offs_n for row and offs_k for column, effectively computing A @ B^T; this matches
    # the standard weight layout convention (out_features, in_features)
    b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]
    a_s_ptrs = a_s_ptr + offs_m * k
    # [Assumptions Made] B scale tensor indexed by (n // BLOCK_SIZE_K, k_block) — the scale factor
    # for B's block at column n and K-block i is found by dividing n by the K-block size (which
    # equals the quantization block size); this accounts for B's transposed access pattern
    b_s_ptrs = b_s_ptr + (offs_n // BLOCK_SIZE_K) * k

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(k):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - i * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - i * BLOCK_SIZE_K, other=0.0)
        a_s = tl.load(a_s_ptrs)
        b_s = tl.load(b_s_ptrs)
        # [Trade-offs] Per-block scale accumulation — multiply each FP8 block's dot product by
        # corresponding A and B scale factors before accumulating into FP32; this maintains
        # numerical accuracy by applying scale corrections at block granularity rather than
        # post-hoc scaling of the entire result. The FP32 accumulator prevents precision loss
        # during the summation. Reference: arXiv:2412.19437, Section 3.3
        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K
        a_s_ptrs += 1
        b_s_ptrs += 1
    c = accumulator.to(c_ptr.dtype.element_ty)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


def fp8_gemm(a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor):
    """Performs FP8 matrix multiplication with per-block scaling: C = A x B^T.

    Wraps the ``fp8_gemm_kernel`` autotuned Triton kernel, handling shape extraction,
    output allocation, and 2D grid launch. Supports batched inputs by flattening all
    leading dimensions of A into the M dimension.

    Args:
        a (torch.Tensor): Activation matrix in FP8 format. Must be contiguous.
            Shape: ``(*, K)`` where ``*`` is any number of leading dimensions
            (e.g., ``(batch_size, seq_len, K)`` or ``(batch_size * seq_len, K)``).
            Dtype: ``torch.float8_e4m3fn``.
        a_s (torch.Tensor): Per-block scale factors for activations. Must be contiguous.
            Shape: ``(*, K // 128)`` — one float32 scale per 128-element block along K.
        b (torch.Tensor): Weight matrix in FP8 format. Must be contiguous.
            Shape: ``(N, K)`` stored as ``(out_features, in_features)``.
            Dtype: ``torch.float8_e4m3fn``.
        b_s (torch.Tensor): Per-block scale factors for weights. Must be contiguous.
            Shape: ``(ceil(N / 128), K // 128)`` — one float32 scale per
            (128 x 128) weight tile.

    Returns:
        torch.Tensor: Matrix product result.
            Shape: ``(*, N)`` where ``*`` matches the leading dimensions of ``a``.
            Dtype: ``torch.get_default_dtype()`` (typically BF16).

    Raises:
        AssertionError: If ``a`` or ``b`` are not contiguous.
        AssertionError: If ``a_s`` or ``b_s`` are not contiguous.

    Notes:
        M is computed as ``a.numel() // K`` to handle batched inputs — all leading
        dimensions are collapsed into a single M dimension for the GEMM, then the
        output is reshaped to ``(*, N)`` via ``a.new_empty(*a.size()[:-1], N)``.
        Grid is 2D: ``(ceil(M / BLOCK_SIZE_M), ceil(N / BLOCK_SIZE_N))`` where
        BLOCK_SIZE_M and BLOCK_SIZE_N are selected by Triton autotune.
        Called by ``model.py:linear()`` for FP8 weight dispatch during inference.
        Reference: arXiv:2412.19437, Section 3.3.
    """
    assert a.is_contiguous() and b.is_contiguous(), 'Input tensors must be contiguous'
    assert a_s.is_contiguous() and b_s.is_contiguous(), 'Scaling factor tensors must be contiguous'
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    c = a.new_empty(*a.size()[:-1], N, dtype=torch.get_default_dtype())
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(N, META['BLOCK_SIZE_N']))
    fp8_gemm_kernel[grid](a, b, c, a_s, b_s, M, N, K)
    return c
