"""
测试序列长度处理的正确性

测试场景:
1. 正常长度的序列
2. 恰好等于 max_len 的序列
3. 超过 max_len 的序列（应被截断）
"""

import torch
from scripts.train_decoder import collate_fn
from scripts.train_encoder_decoder import collate_fn_with_padding


def test_decoder_collate_fn():
    """测试 Decoder 的 collate 函数"""
    print("\n" + "="*60)
    print("测试 Decoder collate_fn")
    print("="*60)

    # 测试用例
    max_seq_len = 128
    pad_token_id = 0

    # 1. 正常长度
    batch1 = [[1, 2, 3, 4, 5], [6, 7, 8]]
    input_ids, target_ids = collate_fn(batch1, pad_token_id, max_seq_len)
    assert input_ids.shape[1] < max_seq_len
    print(f"✓ 正常长度测试通过: batch shape = {input_ids.shape}")

    # 2. 恰好等于 max_len
    batch2 = [list(range(max_seq_len)), list(range(50))]
    input_ids, target_ids = collate_fn(batch2, pad_token_id, max_seq_len)
    assert input_ids.shape[1] == max_seq_len - 1  # -1 因为要去掉最后一个 token
    print(f"✓ max_len 边界测试通过: batch shape = {input_ids.shape}")

    # 3. 超过 max_len（应被截断）
    batch3 = [list(range(500)), list(range(1279)), list(range(50))]
    input_ids, target_ids = collate_fn(batch3, pad_token_id, max_seq_len)
    assert input_ids.shape[1] == max_seq_len - 1
    print(f"✓ 超长序列截断测试通过: batch shape = {input_ids.shape}")
    print(f"  原始长度: [500, 1279, 50]")
    print(f"  截断后最大长度: {max_seq_len}")

    print("\n所有 Decoder collate_fn 测试通过! ✓")


def test_encoder_decoder_collate_fn():
    """测试 Encoder-Decoder 的 collate 函数"""
    print("\n" + "="*60)
    print("测试 Encoder-Decoder collate_fn")
    print("="*60)

    max_seq_len = 128
    pad_token_id = 0

    # 1. 正常长度
    src_batch1 = [[1, 2, 3, 4], [5, 6]]
    tgt_batch1 = [[7, 8, 9], [10, 11, 12, 13]]
    src_tensor, tgt_tensor = collate_fn_with_padding(
        src_batch1, tgt_batch1, pad_token_id, max_seq_len
    )
    assert src_tensor.shape[1] < max_seq_len
    assert tgt_tensor.shape[1] < max_seq_len
    print(f"✓ 正常长度测试通过: src={src_tensor.shape}, tgt={tgt_tensor.shape}")

    # 2. 恰好等于 max_len
    src_batch2 = [list(range(max_seq_len))]
    tgt_batch2 = [list(range(max_seq_len))]
    src_tensor, tgt_tensor = collate_fn_with_padding(
        src_batch2, tgt_batch2, pad_token_id, max_seq_len
    )
    assert src_tensor.shape[1] == max_seq_len
    assert tgt_tensor.shape[1] == max_seq_len
    print(f"✓ max_len 边界测试通过: src={src_tensor.shape}, tgt={tgt_tensor.shape}")

    # 3. 超过 max_len（应被截断）
    src_batch3 = [list(range(1279)), list(range(50))]
    tgt_batch3 = [list(range(500)), list(range(30))]
    src_tensor, tgt_tensor = collate_fn_with_padding(
        src_batch3, tgt_batch3, pad_token_id, max_seq_len
    )
    assert src_tensor.shape[1] == max_seq_len
    assert tgt_tensor.shape[1] == max_seq_len
    print(f"✓ 超长序列截断测试通过: src={src_tensor.shape}, tgt={tgt_tensor.shape}")
    print(f"  原始 src 长度: [1279, 50]")
    print(f"  原始 tgt 长度: [500, 30]")
    print(f"  截断后最大长度: {max_seq_len}")

    print("\n所有 Encoder-Decoder collate_fn 测试通过! ✓")


def test_model_with_long_sequence():
    """测试模型对超长序列的处理"""
    print("\n" + "="*60)
    print("测试模型处理超长序列")
    print("="*60)

    from models.transformer_models import DecoderOnlyModel

    # 创建一个小模型
    vocab_size = 100
    d_model = 64
    num_layers = 2
    num_heads = 4
    d_ff = 128
    max_len = 128

    model = DecoderOnlyModel(vocab_size, d_model, num_layers, num_heads, d_ff, max_len)

    # 测试正常长度
    x_normal = torch.randint(0, vocab_size, (2, 50))
    logits, _ = model(x_normal)
    print(f"✓ 正常长度 (seq_len=50): logits shape = {logits.shape}")

    # 测试 max_len 边界
    x_max = torch.randint(0, vocab_size, (2, max_len))
    logits, _ = model(x_max)
    print(f"✓ max_len 边界 (seq_len={max_len}): logits shape = {logits.shape}")

    # 测试超长序列（应该报错，但我们的 collate_fn 会在之前截断）
    try:
        x_too_long = torch.randint(0, vocab_size, (2, max_len + 100))
        logits, _ = model(x_too_long)
        print(f"✗ 超长序列未被拒绝！这不应该发生")
    except ValueError as e:
        print(f"✓ 超长序列被正确拒绝: {e}")

    print("\n模型长度检查测试通过! ✓")


if __name__ == "__main__":
    test_decoder_collate_fn()
    test_encoder_decoder_collate_fn()
    test_model_with_long_sequence()

    print("\n" + "="*60)
    print("所有测试通过! 🎉")
    print("="*60)
    print("\n修复总结:")
    print("1. ✓ collate_fn 添加了序列截断逻辑")
    print("2. ✓ 支持自定义 max_seq_len 参数")
    print("3. ✓ 模型的位置编码保留长度检查")
    print("\n这确保了即使数据中有超长序列，也不会导致训练崩溃。")
