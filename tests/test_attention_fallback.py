"""
Test attention implementations - flex_attention (training) and SDPA (inference).

Run: python -m pytest tests/test_attention_fallback.py -v -s
"""
import torch
import pytest
from nanochat.flash_attention import (
    flex_attn_func,
    sdpa_attn_func,
    sdpa_attn_with_kvcache,
    create_sliding_window_block_mask,
)
from nanochat.engine import KVCache


# =============================================================================
# flex_attention tests (training path, requires CUDA)
# =============================================================================
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for flex_attention")
class TestFlexAttention:
    """Test flex_attention training path."""

    DEVICE = "cuda"
    DTYPE = torch.bfloat16

    def test_basic_causal(self):
        """Basic causal attention with block mask."""
        B, T, H, D = 2, 64, 4, 32
        q = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)

        block_mask = create_sliding_window_block_mask(T, T, self.DEVICE)
        y = flex_attn_func(q, k, v, block_mask=block_mask)

        assert y.shape == (B, T, H, D)
        assert not torch.isnan(y).any(), "Output contains NaN"

    def test_sliding_window(self):
        """Sliding window attention with block mask."""
        B, T, H, D = 2, 128, 4, 32
        window = 32
        q = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)

        block_mask = create_sliding_window_block_mask(T, window, self.DEVICE)
        y = flex_attn_func(q, k, v, block_mask=block_mask)

        assert y.shape == (B, T, H, D)
        assert not torch.isnan(y).any(), "Output contains NaN"

    def test_gqa(self):
        """Group Query Attention (fewer KV heads than Q heads)."""
        B, T, D = 2, 64, 32
        n_heads = 8
        n_kv_heads = 2

        q = torch.randn(B, T, n_heads, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T, n_kv_heads, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, T, n_kv_heads, D, device=self.DEVICE, dtype=self.DTYPE)

        block_mask = create_sliding_window_block_mask(T, T, self.DEVICE)
        y = flex_attn_func(q, k, v, block_mask=block_mask)

        assert y.shape == (B, T, n_heads, D)
        assert not torch.isnan(y).any(), "Output contains NaN"

    def test_backward_gradients(self):
        """Verify gradients flow through flex_attention."""
        B, T, H, D = 2, 32, 4, 16
        q = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE, requires_grad=True)
        k = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE, requires_grad=True)
        v = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE, requires_grad=True)

        block_mask = create_sliding_window_block_mask(T, T, self.DEVICE)
        y = flex_attn_func(q, k, v, block_mask=block_mask)
        loss = y.sum()
        loss.backward()

        assert q.grad is not None, "No gradient for q"
        assert k.grad is not None, "No gradient for k"
        assert v.grad is not None, "No gradient for v"
        assert not torch.isnan(q.grad).any(), "NaN in q gradient"

    def test_flex_vs_sdpa_match(self):
        """Verify flex_attention and SDPA produce similar results for full context."""
        B, T, H, D = 2, 64, 4, 32
        q = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)

        block_mask = create_sliding_window_block_mask(T, T, self.DEVICE)
        y_flex = flex_attn_func(q, k, v, block_mask=block_mask)
        y_sdpa = sdpa_attn_func(q, k, v, window_size=(T, 0))

        max_diff = (y_flex - y_sdpa).abs().max().item()
        assert torch.allclose(y_flex, y_sdpa, atol=1e-2, rtol=1e-2), \
            f"flex vs sdpa max_diff={max_diff:.6f}"


# =============================================================================
# SDPA-only tests (run on any device)
# =============================================================================
class TestSDPAOnly:
    """Test SDPA fallback works correctly. Runs on any device."""

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    def test_basic_forward(self):
        """Test SDPA forward pass produces valid output."""
        B, T, H, D = 2, 64, 4, 32
        q = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)

        y = sdpa_attn_func(q, k, v, window_size=(T, 0))

        assert y.shape == (B, T, H, D)
        assert not torch.isnan(y).any(), "Output contains NaN"

    def test_sliding_window(self):
        """Test SDPA with sliding window."""
        B, T, H, D = 2, 128, 4, 32
        window = 32
        q = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)

        y = sdpa_attn_func(q, k, v, window_size=(window, 0))

        assert y.shape == (B, T, H, D)
        assert not torch.isnan(y).any(), "Output contains NaN"

    def test_backward(self):
        """Test gradients flow through SDPA."""
        B, T, H, D = 2, 32, 4, 16
        q = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE, requires_grad=True)
        k = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE, requires_grad=True)
        v = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE, requires_grad=True)

        y = sdpa_attn_func(q, k, v, window_size=(T, 0))
        loss = y.sum()
        loss.backward()

        assert q.grad is not None, "No gradient for q"
        assert k.grad is not None, "No gradient for k"
        assert v.grad is not None, "No gradient for v"
        assert not torch.isnan(q.grad).any(), "NaN in q gradient"

    def test_kvcache(self):
        """Test SDPA with KV cache."""
        B, T_max, H, D = 2, 64, 4, 32
        n_layers = 1

        cache = KVCache(
            batch_size=B, num_heads=H, seq_len=T_max, head_dim=D,
            num_layers=n_layers, device=self.DEVICE, dtype=self.DTYPE
        )
        k_cache, v_cache = cache.get_layer_cache(0)

        # Prefill
        T_prefill = 16
        q = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)

        y = sdpa_attn_with_kvcache(
            q, k_cache, v_cache, k=k, v=v,
            cache_seqlens=cache.cache_seqlens,
            causal=True, window_size=(T_max, 0)
        )
        cache.advance(T_prefill)

        assert y.shape == (B, T_prefill, H, D)
        assert cache.get_pos() == T_prefill

        # Generate single token
        q_single = torch.randn(B, 1, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k_single = torch.randn(B, 1, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v_single = torch.randn(B, 1, H, D, device=self.DEVICE, dtype=self.DTYPE)

        y_single = sdpa_attn_with_kvcache(
            q_single, k_cache, v_cache, k=k_single, v=v_single,
            cache_seqlens=cache.cache_seqlens,
            causal=True, window_size=(T_max, 0)
        )
        cache.advance(1)

        assert y_single.shape == (B, 1, H, D)
        assert cache.get_pos() == T_prefill + 1

    def test_kvcache_sliding_window_decode(self):
        """Test single token decode with sliding window smaller than cache size."""
        B, T_max, H, D = 2, 64, 4, 32
        T_prefill = 32
        window = 8

        cache = KVCache(
            batch_size=B, num_heads=H, seq_len=T_max, head_dim=D,
            num_layers=1, device=self.DEVICE, dtype=self.DTYPE
        )
        k_cache, v_cache = cache.get_layer_cache(0)

        # Fill cache
        k_init = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v_init = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k_cache[:, :T_prefill, :, :] = k_init
        v_cache[:, :T_prefill, :, :] = v_init
        cache.cache_seqlens.fill_(T_prefill)

        # Decode with sliding window
        q_single = torch.randn(B, 1, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k_single = torch.randn(B, 1, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v_single = torch.randn(B, 1, H, D, device=self.DEVICE, dtype=self.DTYPE)

        y = sdpa_attn_with_kvcache(
            q_single, k_cache, v_cache, k=k_single, v=v_single,
            cache_seqlens=cache.cache_seqlens,
            causal=True, window_size=(window, 0)
        )

        assert y.shape == (B, 1, H, D)
        assert not torch.isnan(y).any(), "Output contains NaN"


if __name__ == "__main__":
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        major, minor = torch.cuda.get_device_capability()
        print(f"Compute capability: {major}.{minor}")
    print()

    pytest.main([__file__, "-v", "-s"])
