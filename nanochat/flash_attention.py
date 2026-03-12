"""
Unified attention interface for nanochat.

Training: flex_attention with precomputed block masks (efficient sliding window on all GPUs)
Inference: PyTorch SDPA with KV cache support

Usage:
    from nanochat.flash_attention import flex_attn_func, create_sliding_window_block_mask
    from nanochat.flash_attention import sdpa_attn_with_kvcache
"""
import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask


# =============================================================================
# Training: flex_attention
# =============================================================================
def create_sliding_window_block_mask(seq_len, window_size, device):
    """Create a flex_attention block mask for causal sliding window attention.

    Args:
        seq_len: sequence length
        window_size: number of past tokens to attend to (full context if >= seq_len)
        device: torch device
    """
    def mask_fn(b, h, q_idx, kv_idx, _w=window_size):
        return (q_idx >= kv_idx) & (q_idx - kv_idx <= _w)
    return create_block_mask(mask_fn, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len, device=device)


def flex_attn_func(q, k, v, block_mask=None):
    """
    Training attention using flex_attention.

    Args:
        q, k, v: Tensors of shape (B, T, H, D)
        block_mask: precomputed flex_attention BlockMask

    Returns:
        Output tensor of shape (B, T, H, D)
    """
    # flex_attention expects (B, H, T, D)
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    enable_gqa = q.size(1) != k.size(1)
    y = flex_attention(q, k, v, block_mask=block_mask, enable_gqa=enable_gqa)
    return y.transpose(1, 2)  # back to (B, T, H, D)


# =============================================================================
# Inference: SDPA with KV cache
# =============================================================================
def _sdpa_attention(q, k, v, window_size, enable_gqa):
    """
    SDPA attention with sliding window support.
    q, k, v are (B, H, T, D) format.
    """
    Tq = q.size(2)
    Tk = k.size(2)
    window = window_size[0]

    # Full context, same length
    if (window < 0 or window >= Tq) and Tq == Tk:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)

    # Single token generation
    if Tq == 1:
        if window >= 0 and window < Tk:
            # window is "left" tokens we need to include (window + 1) keys total
            start = max(0, Tk - (window + 1))
            k = k[:, :, start:, :]
            v = v[:, :, start:, :]
        return F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)

    # Need explicit mask for sliding window/chunk inference
    device = q.device
    # For chunk inference (Tq != Tk), is_causal is not aligned to cache position => build an explicit bool mask
    row_idx = (Tk - Tq) + torch.arange(Tq, device=device).unsqueeze(1)
    col_idx = torch.arange(Tk, device=device).unsqueeze(0)
    mask = col_idx <= row_idx

    # sliding window (left)
    if window >= 0 and window < Tk:
        mask = mask & ((row_idx - col_idx) <= window)

    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=enable_gqa)


def sdpa_attn_func(q, k, v, window_size=(-1, -1)):
    """
    Training attention using SDPA (fallback for CPU/MPS when flex_attention isn't available).

    Args:
        q, k, v: Tensors of shape (B, T, H, D)
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (B, T, H, D)
    """
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    enable_gqa = q.size(1) != k.size(1)
    y = _sdpa_attention(q, k, v, window_size, enable_gqa)
    return y.transpose(1, 2)


def sdpa_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, cache_seqlens=None,
                            causal=False, window_size=(-1, -1)):
    """
    SDPA attention with KV cache for inference.

    Args:
        q: Queries, shape (B, T_new, H, D)
        k_cache, v_cache: Pre-allocated cache tensors, shape (B, T_max, H_kv, D)
        k, v: New keys/values to insert, shape (B, T_new, H_kv, D)
        cache_seqlens: Current position in cache, shape (B,) int32
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (B, T_new, H, D)
    """
    # Manually manage KV cache
    B, T_new, H, D = q.shape
    pos = cache_seqlens[0].item()  # assume uniform position across batch

    # Insert new k, v into cache (in-place)
    if k is not None and v is not None:
        k_cache[:, pos:pos+T_new, :, :] = k
        v_cache[:, pos:pos+T_new, :, :] = v

    # Get full cache up to current position + new tokens
    end_pos = pos + T_new
    k_full = k_cache[:, :end_pos, :, :]
    v_full = v_cache[:, :end_pos, :, :]

    # Transpose to SDPA layout: (B, T, H, D) -> (B, H, T, D)
    q_sdpa = q.transpose(1, 2)
    k_sdpa = k_full.transpose(1, 2)
    v_sdpa = v_full.transpose(1, 2)

    enable_gqa = q_sdpa.size(1) != k_sdpa.size(1)
    y_sdpa = _sdpa_attention(q_sdpa, k_sdpa, v_sdpa, window_size, enable_gqa)

    return y_sdpa.transpose(1, 2)  # back to (B, T, H, D)
