import torch

from labs.lab00_positional_encoding import sinusoidal_position_encoding
from labs.lab01_attention_basics import scaled_dot_product_attention
from labs.lab02_multi_head_attention import merge_heads, split_heads
from labs.lab04_tiny_copy_task import make_copy_batch
from labs.lab05_tiny_language_model import make_pattern_batch
from labs.lab06_kv_cache import compare_full_and_cached
from labs.lab07_modern_blocks import RMSNorm, SwiGLU, apply_rope
from labs.lab08_moe_routing import TopKMoE
from labs.lab09_lora_linear import run_demo as run_lora_demo
from labs.lab10_mha_mqa_gqa import GroupedQuerySelfAttention
from labs.lab11_moe_variants import DenseMoE, SharedExpertSparseMoE, SparseMoE


def test_attention_mask_and_probability_rows():
    torch.manual_seed(0)
    query = torch.randn(1, 4, 8)
    mask = torch.tril(torch.ones(4, 4, dtype=torch.bool))
    output, weights = scaled_dot_product_attention(query, query, query, mask)
    assert output.shape == query.shape
    assert torch.allclose(weights.sum(-1), torch.ones(1, 4))
    assert torch.count_nonzero(weights[0].triu(1)) == 0


def test_sinusoidal_position_zero_has_expected_pattern():
    encoding = sinusoidal_position_encoding(6, 8)
    assert encoding.shape == (6, 8)
    assert torch.allclose(encoding[0, 0::2], torch.zeros(4))
    assert torch.allclose(encoding[0, 1::2], torch.ones(4))


def test_split_and_merge_heads_are_inverse():
    x = torch.randn(2, 5, 16)
    assert torch.allclose(merge_heads(split_heads(x, 4)), x)


def test_synthetic_task_batches_have_next_token_alignment():
    source, decoder_input, labels = make_copy_batch(3, 5, 16)
    assert source.shape == (3, 5)
    assert decoder_input.shape == labels.shape == (3, 6)

    inputs, next_tokens = make_pattern_batch(3, 7, 12)
    assert torch.equal((inputs + 1) % 12, next_tokens)


def test_kv_cache_matches_full_attention():
    assert compare_full_and_cached() < 1e-6


def test_modern_blocks_preserve_expected_shapes():
    x = torch.randn(2, 3, 5, 8)
    rotated = apply_rope(x)
    assert rotated.shape == x.shape
    assert torch.allclose(rotated.norm(dim=-1), x.norm(dim=-1), atol=1e-5)

    hidden = torch.randn(2, 5, 16)
    assert RMSNorm(16)(hidden).shape == hidden.shape
    assert SwiGLU(16, 32)(hidden).shape == hidden.shape


def test_moe_routes_every_token_to_top_k_experts():
    x = torch.randn(2, 5, 16)
    output, auxiliary_loss, routes, counts = TopKMoE(top_k=2)(x)
    assert output.shape == x.shape
    assert routes.shape == (2, 5, 2)
    assert sum(counts) == 20
    assert torch.isfinite(auxiliary_loss)


def test_lora_starts_equal_to_base_and_can_fuse():
    initial_difference, fused_difference = run_lora_demo()
    assert initial_difference == 0.0
    assert fused_difference < 1e-5


def test_mha_mqa_gqa_share_one_implementation():
    x = torch.randn(2, 5, 32)
    mha = GroupedQuerySelfAttention(32, 8, 8)
    gqa = GroupedQuerySelfAttention(32, 8, 2)
    mqa = GroupedQuerySelfAttention(32, 8, 1)

    for module in (mha, gqa, mqa):
        output, cache, weights = module(x)
        assert output.shape == x.shape
        assert cache[0].shape[1] == module.num_kv_heads
        assert weights.shape == (2, 8, 5, 5)

    assert (
        mha.kv_cache_elements_per_token()
        > gqa.kv_cache_elements_per_token()
        > mqa.kv_cache_elements_per_token()
    )


def test_dense_sparse_and_shared_expert_moe_shapes():
    x = torch.randn(2, 5, 16)
    dense_output, dense_weights = DenseMoE(num_experts=4)(x)
    sparse_output, routes, counts, _ = SparseMoE(num_experts=8, top_k=2)(x)
    shared_output, shared_routes, shared_counts, _ = SharedExpertSparseMoE(
        num_shared_experts=1, num_routed_experts=8, top_k=2
    )(x)

    assert dense_output.shape == sparse_output.shape == shared_output.shape == x.shape
    assert dense_weights.shape == (2, 5, 4)
    assert routes.shape == shared_routes.shape == (2, 5, 2)
    assert sum(counts) == sum(shared_counts) == 20
