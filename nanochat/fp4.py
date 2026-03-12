"""Minimal NVFP4 training for nanochat — block-scaled FP4 with two-level scaling.

Drop-in replacement for TransformerEngine's NVFP4 training (~10k lines) with ~200
lines. This is the simplest NVFP4 training implementation we know of.

How NVFP4 differs from FP8
===========================
FP8 uses one scalar scale per tensor ("tensorwise scaling").
NVFP4 uses two-level scaling:
  1. Per-tensor scale (FP32): maps the global range into a manageable zone
  2. Per-block scale (FP8 E4M3): one scale per 16 elements along the reduction dim

The FP4 E2M1 format (1 sign, 2 exponent, 1 mantissa) has only 16 representable values:
  {0, +/-0.5, +/-1, +/-1.5, +/-2, +/-3, +/-4, +/-6}

Two-level scaling ensures these few values can represent diverse magnitudes.
Block size 16 with E4M3 scales is NVIDIA's proprietary format (NVFP4), distinct
from the OCP standard MXFP4 (block size 32, E8M0 scales).

The three GEMMs of training
============================
  forward:   Y      = X      @ W^T    (both quantized to NVFP4)
  backward:  grad_X = grad_Y @ W      (re-quantize with transposed layout)
             grad_W = grad_Y^T @ X    (re-quantize with transposed layout)

Unlike FP8 where transposing a quantized tensor is trivial (each value is one
byte), FP4 packs two values per byte and has block-wise scales tied to the
matrix layout. Transposing requires re-quantization from full precision, so we
save full-precision tensors in the forward pass for backward (6 quantizations
per layer — 2 per GEMM — vs 2 for FP8 which reuses saved quantized tensors).

torch._scaled_mm requirements for NVFP4
========================================
  - Input dtype: torch.float4_e2m1fn_x2 (packed, 2 values per byte)
  - Scale dtype: torch.float8_e4m3fn in "swizzled" blocked layout for cuBLASLt
  - Second operand must be column-major
  - use_fast_accum must be False (required for NVFP4, unlike FP8 where it's optional)
  - Reduction dimension (K) must be divisible by 32
  - N (output dim of matmul) must be divisible by 16
  - Per-tensor scales are applied as post-multiply (not passed to _scaled_mm)

Performance notes
==================
The heavy lifting (block amax + scaling + E2M1 encoding) is fused into a single
Triton kernel. Uint4 packing and scale swizzle are done in PyTorch afterward
(2-3 kernel launches each). Without the Triton fusion, the ~20 separate kernel
launches per quantization make pure-PyTorch FP4 training ~35x slower than BF16.

SM121 (DGX Spark) compatibility
================================
cuBLASLt NVFP4 GEMMs work on SM121 via the extended mma.sync instruction path.
The Triton quantization kernel uses only elementwise ops (no tl.dot_scaled),
which work fine on SM121. TransformerEngine, CUTLASS Python DSL, and Triton
tl.dot_scaled do NOT work on SM121 — cuBLASLt (via torch._scaled_mm) is the
only viable matmul path.
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl

from nanochat.common import COMPUTE_DTYPE

# The largest representable magnitude in each format. Used to compute scales.
F4_E2M1_MAX = 6.0    # E2M1: sign(1) exp(2) mantissa(1), max = 2^2 * 1.5 = 6
F8_E4M3_MAX = 448.0  # E4M3: sign(1) exp(4) mantissa(3), max = 2^8 * 1.75 = 448
# NVFP4 groups elements into blocks of 16 along the reduction dimension.
# Each block gets its own E4M3 scale factor.
BLOCK_SIZE = 16


def _ceil_div(a, b):
    return (a + b - 1) // b


# =============================================================================
# Triton kernel: fused NVFP4 quantization + E2M1 encoding
# =============================================================================

@triton.jit
def _nvfp4_quantize_kernel(
    x_ptr,              # input tensor (M, K) in bf16/fp32
    enc_ptr,            # output encoded uint8 (M, K), one E2M1 nibble per byte
    scales_ptr,         # output block scales uint8 (M, K//16)
    inv_pts_ptr,        # pointer to 1.0 / per_tensor_scale (scalar, on GPU)
    M, K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,  # must be 16 (NVFP4 block size)
):
    """Fused quantization kernel: one program handles one (BLOCK_M, 16) tile.

    For each block of 16 elements along K:
      1. Load 16 values, multiply by inv_per_tensor_scale
      2. Compute block amax -> block_scale (E4M3)
      3. Scale values by 1/block_scale
      4. Encode to E2M1 (round-to-nearest via boundary comparison)
      5. Store encoded uint8 + block scale
    Packing pairs of nibbles into uint4 bytes is done in the caller.
    """
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    # Offsets for this (BLOCK_M, 16) tile
    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    k_offs = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    m_mask = m_offs < M

    # Load tile — each thread block processes BLOCK_M rows × 16 columns
    x_offsets = m_offs[:, None] * K + k_offs[None, :]
    x = tl.load(x_ptr + x_offsets, mask=m_mask[:, None], other=0.0).to(tl.float32)

    # Level 1: apply per-tensor scale (computed once in the caller)
    inv_pts = tl.load(inv_pts_ptr)
    x = x * inv_pts

    # Level 2: compute per-block scale from the max absolute value
    amax = tl.max(tl.abs(x), axis=1)  # (BLOCK_M,) — one amax per row
    # Scale maps [0, amax] -> [0, F4_E2M1_MAX=6.0], then the scale itself
    # is stored as E4M3, so clamp to E4M3's representable range
    block_scale = amax / 6.0
    block_scale = tl.maximum(block_scale, 1.9531e-03)  # float8_e4m3fn.tiny (2^-9)
    block_scale = tl.minimum(block_scale, 448.0)        # float8_e4m3fn.max

    # Apply block scale and clamp to E2M1 range [-6, 6]
    x = x / block_scale[:, None]
    x = tl.maximum(tl.minimum(x, 6.0), -6.0)

    # E2M1 round-to-nearest encoding via boundary comparisons.
    # The 8 positive E2M1 values and their midpoint boundaries:
    #   value:    0    0.5   1.0   1.5   2.0   3.0   4.0   6.0
    #   encoding: 0    1     2     3     4     5     6     7
    #   boundary:   0.25  0.75  1.25  1.75  2.50  3.50  5.00
    abs_x = tl.abs(x)
    enc = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.uint8)
    enc = tl.where(abs_x >= 0.25, 1, enc)   # >= 0.25 -> 0.5
    enc = tl.where(abs_x >= 0.75, 2, enc)   # >= 0.75 -> 1.0
    enc = tl.where(abs_x >= 1.25, 3, enc)   # >= 1.25 -> 1.5
    enc = tl.where(abs_x >= 1.75, 4, enc)   # >= 1.75 -> 2.0
    enc = tl.where(abs_x >= 2.50, 5, enc)   # >= 2.50 -> 3.0
    enc = tl.where(abs_x >= 3.50, 6, enc)   # >= 3.50 -> 4.0
    enc = tl.where(abs_x >= 5.00, 7, enc)   # >= 5.00 -> 6.0

    # E2M1 encoding is 4 bits: [sign | exp1 | exp0 | mantissa] = [bit3 | bits 2:0]
    sign = tl.where(x < 0, 8, 0).to(tl.uint8)  # bit 3 = sign
    enc = enc | sign

    # Store one encoded nibble per byte (caller packs pairs into uint4)
    tl.store(enc_ptr + x_offsets, enc, mask=m_mask[:, None])

    # Store block scale as E4M3 (reinterpreted as uint8 for raw byte storage)
    scale_fp8 = block_scale.to(tl.float8e4nv)
    scale_uint8 = scale_fp8.to(tl.uint8, bitcast=True)
    scale_offs = m_offs * (K // BLOCK_K) + pid_k
    tl.store(scales_ptr + scale_offs, scale_uint8, mask=m_mask)


# =============================================================================
# Scale swizzle for cuBLASLt
# =============================================================================

@torch.no_grad()
def _to_blocked(scale_uint8, rows, cols):
    """Swizzle (rows, cols) scale matrix to cuBLASLt blocked layout.

    cuBLASLt requires NVFP4 block scales in a specific "swizzled" memory layout
    (CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3). The hardware reads scales in
    128×4 tiles, which we rearrange into 32×16 tiles to match its access pattern.

    The transformation:
      1. Pad to multiples of (128 rows, 4 cols) for tile alignment
      2. Split into 128×4 tiles
      3. Rearrange each tile: split 128 rows into 4 groups of 32, interleave
         with the 4 columns → 32×16 output tile
    """
    n_row_blocks = _ceil_div(rows, 128)
    n_col_blocks = _ceil_div(cols, 4)
    # Pad to tile-aligned dimensions
    padded = torch.zeros(
        n_row_blocks * 128, n_col_blocks * 4,
        device=scale_uint8.device, dtype=torch.uint8,
    )
    padded[:rows, :cols] = scale_uint8.reshape(rows, cols)
    # Split into (n_row_blocks, 128, n_col_blocks, 4) tiles, then swap axes
    # so tiles are indexed as (row_block, col_block, 128, 4)
    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    # Rearrange each 128×4 tile into 32×16: split 128 into (4, 32), transpose
    # the 4 with 32, then merge the 4 cols with the 4 groups → 32×16
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
    return rearranged.flatten().contiguous()


# =============================================================================
# Quantization entry point
# =============================================================================

@torch.no_grad()
def _nvfp4_quantize(x):
    """Quantize a 2D tensor to NVFP4 with two-level scaling via fused Triton kernel.

    Returns (fp4_packed, swizzled_scales, per_tensor_scale).
    - fp4_packed: (M, K//2) in float4_e2m1fn_x2 dtype (2 values per byte)
    - swizzled_scales: block scales in cuBLASLt layout, as float8_e4m3fn
    - per_tensor_scale: scalar FP32 tensor, applied as post-multiply after _scaled_mm
    """
    M, K = x.shape
    assert K % 32 == 0, f"K={K} must be divisible by 32"

    # Level 1: per-tensor scale. This maps the tensor's dynamic range into the
    # zone representable by E4M3 block scales × E2M1 values. The max representable
    # magnitude after both scaling levels is F8_E4M3_MAX * F4_E2M1_MAX = 448 * 6.
    x = x.contiguous()
    amax = x.float().abs().max().clamp(min=1e-12)
    per_tensor_scale = amax / (F8_E4M3_MAX * F4_E2M1_MAX)
    # Keep inverse scale on GPU as a tensor — loading it in the Triton kernel
    # via tl.load() avoids a CPU-GPU sync that .item() would cause
    inv_pts = 1.0 / per_tensor_scale

    # Allocate outputs for the Triton kernel
    encoded = torch.empty(M, K, device=x.device, dtype=torch.uint8)
    scales = torch.empty(M, K // BLOCK_SIZE, device=x.device, dtype=torch.uint8)

    # Launch: one program per (BLOCK_M rows, 1 block-of-16 along K)
    BLOCK_M = 4
    grid = (_ceil_div(M, BLOCK_M), K // BLOCK_SIZE)
    _nvfp4_quantize_kernel[grid](
        x, encoded, scales, inv_pts,
        M, K,
        BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_SIZE,
    )

    # Pack pairs of nibbles into bytes: even index → low nibble, odd → high nibble.
    # This matches torch.float4_e2m1fn_x2's packing convention.
    packed = encoded[:, 0::2] | (encoded[:, 1::2] << 4)  # (M, K//2)
    fp4_data = packed.view(torch.float4_e2m1fn_x2)

    # Swizzle block scales into cuBLASLt's expected memory layout
    scales_swizzled = _to_blocked(scales, M, K // BLOCK_SIZE).view(torch.float8_e4m3fn)

    return fp4_data, scales_swizzled, per_tensor_scale


# =============================================================================
# FP4 matmul and autograd
# =============================================================================

def _fp4_matmul(a_hp, b_hp, out_dtype):
    """Quantize A and B to NVFP4 and compute A @ B^T via torch._scaled_mm.

    A: (M, K) row-major, B: (N, K) row-major. Result: (M, N).
    Both operands are independently quantized with their own two-level scales.
    """
    a_fp4, a_scales, a_pts = _nvfp4_quantize(a_hp)
    b_fp4, b_scales, b_pts = _nvfp4_quantize(b_hp)

    # A @ B^T: a_fp4 is (M, K_packed) row-major, b_fp4.t() is (K_packed, N) col-major
    result = torch._scaled_mm(
        a_fp4,
        b_fp4.t(),
        scale_a=a_scales,
        scale_b=b_scales,
        out_dtype=out_dtype,
        # NVFP4 requires use_fast_accum=False (unlike FP8 where True is optional).
        # cuBLASLt only supports the precise accumulation path for FP4.
        use_fast_accum=False,
    )
    # _scaled_mm handles block scales internally but NOT per-tensor scales.
    # We apply them as a post-multiply: result * (scale_a * scale_b).
    result = result * (a_pts * b_pts).to(out_dtype)
    return result


class _Float4Matmul(torch.autograd.Function):
    """Custom autograd for the three NVFP4 GEMMs of a Linear layer.

    Unlike FP8 (which uses @allow_in_graph), we do NOT let torch.compile trace
    into this. Inductor's memory estimator cannot handle float4_e2m1fn_x2 tensors
    and will crash with an AssertionError in simd_kernel_features.py. Instead,
    Float4Linear.forward is wrapped with @torch.compiler.disable, creating a
    graph break. The actual matmul (cuBLAS) dominates runtime anyway.

    We save full-precision tensors for backward because FP4's packed format and
    block-wise scales are layout-dependent — transposing a quantized FP4 tensor
    would require unpacking, re-blocking, and re-swizzling the scales.
    Re-quantizing from full precision is simpler and correct.
    """

    @staticmethod
    def forward(ctx, input_2d, weight):
        ctx.save_for_backward(input_2d, weight)
        # Y = X @ W^T: both (M,K) and (N,K) are row-major, matching _fp4_matmul's
        # TN layout (it computes A @ B^T internally)
        return _fp4_matmul(input_2d, weight, out_dtype=input_2d.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input_2d, weight = ctx.saved_tensors

        # GEMM 2: grad_input = grad_output @ weight   [shapes: (M,N) @ (N,K) -> (M,K)]
        # _fp4_matmul computes A @ B^T, so we pass B = weight^T to get:
        #   grad_output @ (weight^T)^T = grad_output @ weight
        weight_t = weight.t().contiguous()  # (K, N) row-major
        grad_input = _fp4_matmul(grad_output, weight_t, out_dtype=grad_output.dtype)

        # GEMM 3: grad_weight = grad_output^T @ input  [shapes: (N,M) @ (M,K) -> (N,K)]
        # _fp4_matmul computes A @ B^T, so we pass A = grad_output^T, B = input^T:
        #   grad_output^T @ (input^T)^T = grad_output^T @ input
        grad_output_t = grad_output.t().contiguous()  # (N, M) row-major
        input_t = input_2d.t().contiguous()            # (K, M) row-major
        grad_weight = _fp4_matmul(grad_output_t, input_t, out_dtype=grad_output.dtype)

        return grad_input, grad_weight


class Float4Linear(nn.Linear):
    """Drop-in nn.Linear replacement that does NVFP4 compute.

    Weights and biases remain in their original precision (e.g. fp32/bf16).
    Only the matmul is performed in NVFP4 via the _Float4Matmul autograd function.
    """

    @torch.compiler.disable
    def forward(self, input):
        # Cast input to COMPUTE_DTYPE (typically bf16) — _scaled_mm expects
        # reduced precision input, and we no longer rely on autocast to do this
        input = input.to(COMPUTE_DTYPE)
        # _scaled_mm only works on 2D tensors, so flatten batch dimensions.
        # Note: the flattened batch dim (M) must be divisible by 32 because it
        # becomes the reduction dim in the grad_weight GEMM during backward.
        # In practice this is always true (e.g. batch_size=32 × seq_len=2048).
        orig_shape = input.shape
        input_2d = input.reshape(-1, orig_shape[-1])
        output = _Float4Matmul.apply(input_2d, self.weight)
        output = output.reshape(*orig_shape[:-1], output.shape[-1])
        if self.bias is not None:
            output = output + self.bias.to(output.dtype)
        return output

    @classmethod
    def from_float(cls, mod):
        """Create Float4Linear from nn.Linear, sharing the same weight and bias.

        Uses meta device to avoid allocating a temporary weight tensor — we
        create the module shell on meta (shapes/dtypes only, no memory), then
        point .weight and .bias to the original module's parameters.
        """
        with torch.device("meta"):
            new_mod = cls(mod.in_features, mod.out_features, bias=False)
        new_mod.weight = mod.weight
        new_mod.bias = mod.bias
        return new_mod


def convert_to_float4_training(module, *, module_filter_fn=None):
    """Replace nn.Linear layers with Float4Linear throughout a module.

    Walks the module tree in post-order (children before parents) and swaps
    each nn.Linear that passes the optional filter. The new Float4Linear shares
    the original weight and bias tensors — no copies, no extra memory.

    Args:
        module: Root module to convert.
        module_filter_fn: Optional filter(module, fqn) -> bool. Only matching Linears
            are converted. Common use: skip layers with dims not divisible by 32
            (NVFP4 requires K % 32 == 0 for the packed format).
    """

    def _convert(mod, prefix=""):
        for name, child in mod.named_children():
            fqn = f"{prefix}.{name}" if prefix else name
            _convert(child, fqn)
            if isinstance(child, nn.Linear) and not isinstance(child, Float4Linear):
                if module_filter_fn is None or module_filter_fn(child, fqn):
                    setattr(mod, name, Float4Linear.from_float(child))

    _convert(module)
    return module
