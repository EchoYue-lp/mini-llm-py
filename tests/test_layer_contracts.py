import pytest
import torch
import torch.nn as nn

from models.layers import LayerNorm, MultiHeadAttention, ScaledDotProductAttention
from models.transformer_models import DecoderOnlyModel, EncoderDecoderModel
from utils.mask_utils import (
    collate_fn_lm,
    collate_fn_mt,
    combine_masks,
    create_causal_mask,
    create_padding_mask,
)


def test_custom_layer_norm_matches_pytorch_formula():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 8)
    custom = LayerNorm(8, eps=1e-6)
    reference = nn.LayerNorm(8, eps=1e-6)
    with torch.no_grad():
        reference.weight.copy_(custom.gamma)
        reference.bias.copy_(custom.beta)
    assert torch.allclose(custom(x), reference(x), atol=1e-6)


def test_attention_rejects_non_boolean_mask_and_zeros_fully_masked_rows():
    attention = ScaledDotProductAttention(d_k=4, dropout=0.0)
    query = torch.randn(1, 2, 3, 4)
    with pytest.raises(TypeError, match="boolean"):
        attention(query, query, query, torch.ones(3, 3))

    _, weights = attention(
        query,
        query,
        query,
        torch.zeros(1, 1, 3, 3, dtype=torch.bool),
    )
    assert torch.count_nonzero(weights) == 0


def test_multi_head_attention_uses_runtime_validation():
    with pytest.raises(ValueError, match="divisible"):
        MultiHeadAttention(d_model=10, num_heads=3)


def test_mask_helpers_validate_shape_dtype_and_length():
    with pytest.raises(ValueError, match="positive"):
        create_causal_mask(0)
    with pytest.raises(ValueError, match=r"\[B,T\]"):
        create_padding_mask(torch.ones(3))
    with pytest.raises(TypeError, match="boolean"):
        combine_masks(torch.ones(1, 1, 2, 2), torch.ones(1, 1, 1, 2))

    causal = create_causal_mask(3)
    padding = create_padding_mask(torch.tensor([[4, 5, 0]]))
    assert combine_masks(causal, padding).shape == (1, 1, 3, 3)


def test_collate_helpers_reject_ambiguous_empty_inputs():
    with pytest.raises(ValueError, match="at least one"):
        collate_fn_lm([])
    with pytest.raises(ValueError, match="at least two"):
        collate_fn_lm([[1]])
    with pytest.raises(ValueError, match="batch sizes"):
        collate_fn_mt([[1]], [[2], [3]])


def test_model_entrypoints_validate_token_shape_dtype_and_batch():
    with pytest.raises(ValueError, match="divisible"):
        DecoderOnlyModel(vocab_size=32, d_model=10, num_heads=3)

    decoder = DecoderOnlyModel(
        vocab_size=32,
        d_model=8,
        num_layers=1,
        num_heads=2,
        d_ff=16,
        max_len=8,
        dropout=0.0,
    )
    with pytest.raises(TypeError, match="torch.long"):
        decoder(torch.randn(2, 4))

    encoder_decoder = EncoderDecoderModel(
        src_vocab_size=32,
        tgt_vocab_size=32,
        d_model=8,
        num_layers=1,
        num_heads=2,
        d_ff=16,
        max_len=8,
        dropout=0.0,
    )
    with pytest.raises(ValueError, match="batch sizes"):
        encoder_decoder(
            torch.ones(2, 4, dtype=torch.long),
            torch.ones(3, 4, dtype=torch.long),
        )
