"""
测试 Pre-LN 架构的正确性
"""
import torch
from models.transformer_models import DecoderOnlyModel, EncoderDecoderModel
from utils.mask_utils import create_causal_mask, create_padding_mask, combine_masks

def test_decoder_only_model():
    """测试 Decoder-Only 模型（Pre-LN）"""
    print("=" * 60)
    print("测试 Decoder-Only 模型 (Pre-LN 架构)")
    print("=" * 60)

    vocab_size = 1000
    d_model = 128
    batch_size = 4
    seq_len = 16

    model = DecoderOnlyModel(vocab_size, d_model, num_layers=2, num_heads=4)
    model.eval()

    # 创建输入
    x = torch.randint(0, vocab_size, (batch_size, seq_len))

    # 创建 mask
    causal_mask = create_causal_mask(seq_len)
    padding_mask = create_padding_mask(x)
    mask = combine_masks(causal_mask, padding_mask)

    # 前向传播
    with torch.no_grad():
        logits, attn_weights = model(x, mask=mask)

    # 验证输出形状
    assert logits.shape == (batch_size, seq_len, vocab_size), \
        f"期望 logits 形状为 {(batch_size, seq_len, vocab_size)}，实际为 {logits.shape}"

    assert len(attn_weights) == 2, \
        f"期望 2 层注意力权重，实际为 {len(attn_weights)}"

    # 验证输出不包含 NaN 或 Inf
    assert not torch.isnan(logits).any(), "logits 包含 NaN"
    assert not torch.isinf(logits).any(), "logits 包含 Inf"

    print(f"✓ 输入形状: {x.shape}")
    print(f"✓ 输出形状: {logits.shape}")
    print(f"✓ 注意力权重层数: {len(attn_weights)}")
    print(f"✓ 输出范围: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
    print(f"✓ Decoder-Only 模型测试通过！\n")

def test_encoder_decoder_model():
    """测试 Encoder-Decoder 模型（Pre-LN）"""
    print("=" * 60)
    print("测试 Encoder-Decoder 模型 (Pre-LN 架构)")
    print("=" * 60)

    src_vocab_size = 1000
    tgt_vocab_size = 1000
    d_model = 128
    batch_size = 4
    src_len = 16
    tgt_len = 12

    model = EncoderDecoderModel(src_vocab_size, tgt_vocab_size, d_model, num_layers=2, num_heads=4)
    model.eval()

    # 创建输入
    src = torch.randint(0, src_vocab_size, (batch_size, src_len))
    tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_len))

    # 创建 mask
    src_mask = create_padding_mask(src)
    tgt_causal_mask = create_causal_mask(tgt_len)
    tgt_padding_mask = create_padding_mask(tgt)
    tgt_mask = combine_masks(tgt_causal_mask, tgt_padding_mask)
    cross_mask = create_padding_mask(src)

    # 前向传播
    with torch.no_grad():
        logits, attn_weights = model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask, cross_mask=cross_mask)

    # 验证输出形状
    assert logits.shape == (batch_size, tgt_len, tgt_vocab_size), \
        f"期望 logits 形状为 {(batch_size, tgt_len, tgt_vocab_size)}，实际为 {logits.shape}"

    assert len(attn_weights) == 2, \
        f"期望 2 层注意力权重，实际为 {len(attn_weights)}"

    # 验证输出不包含 NaN 或 Inf
    assert not torch.isnan(logits).any(), "logits 包含 NaN"
    assert not torch.isinf(logits).any(), "logits 包含 Inf"

    print(f"✓ 源序列形状: {src.shape}")
    print(f"✓ 目标序列形状: {tgt.shape}")
    print(f"✓ 输出形状: {logits.shape}")
    print(f"✓ 注意力权重层数: {len(attn_weights)}")
    print(f"✓ 输出范围: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
    print(f"✓ Encoder-Decoder 模型测试通过！\n")

def test_gradient_flow():
    """测试梯度流动（Pre-LN 应该有更好的梯度流动）"""
    print("=" * 60)
    print("测试梯度流动")
    print("=" * 60)

    vocab_size = 1000
    d_model = 128
    batch_size = 4
    seq_len = 16

    model = DecoderOnlyModel(vocab_size, d_model, num_layers=4, num_heads=4)
    model.train()

    # 创建输入和目标
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    target = torch.randint(0, vocab_size, (batch_size, seq_len))

    # 创建 mask
    causal_mask = create_causal_mask(seq_len)
    padding_mask = create_padding_mask(x)
    mask = combine_masks(causal_mask, padding_mask)

    # 前向传播和反向传播
    logits, _ = model(x, mask=mask)
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, vocab_size),
        target.view(-1)
    )
    loss.backward()

    # 检查梯度
    has_grad = False
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_grad = True
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)
            if grad_norm > 100:  # 梯度过大警告
                print(f"⚠ 警告: {name} 的梯度范数较大: {grad_norm:.4f}")

    assert has_grad, "模型没有梯度"
    avg_grad_norm = sum(grad_norms) / len(grad_norms)

    print(f"✓ 损失值: {loss.item():.4f}")
    print(f"✓ 平均梯度范数: {avg_grad_norm:.4f}")
    print(f"✓ 梯度范数范围: [{min(grad_norms):.4f}, {max(grad_norms):.4f}]")
    print(f"✓ 梯度流动测试通过！\n")

if __name__ == "__main__":
    test_decoder_only_model()
    test_encoder_decoder_model()
    test_gradient_flow()

    print("=" * 60)
    print("🎉 所有 Pre-LN 架构测试通过！")
    print("=" * 60)
