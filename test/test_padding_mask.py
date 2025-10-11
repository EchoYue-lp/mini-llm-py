"""
测试 Padding Mask 修复是否正确
验证训练时的 padding mask 是否被正确应用
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from utils.mask_utils import (
    create_causal_mask,
    create_padding_mask,
    combine_masks,
    collate_fn_lm,
    collate_fn_mt
)


def test_collate_fn_with_different_lengths():
    """测试 collate_fn 是否正确处理不同长度的序列"""
    print("=" * 60)
    print("测试 collate_fn 动态 padding")
    print("=" * 60)

    # 创建不同长度的序列
    batch = [
        [1, 2, 3, 4, 5, 6, 7],      # 长度 7
        [10, 11, 12, 13],            # 长度 4
        [20, 21, 22, 23, 24, 25],    # 长度 6
        [30, 31],                     # 长度 2
    ]

    pad_token_id = 0
    input_ids, target_ids = collate_fn_lm(batch, pad_token_id)

    print(f"原始序列长度: {[len(x) for x in batch]}")
    print(f"Padding 后形状: {input_ids.shape}")
    print(f"\nInput IDs (前 6 列):")
    print(input_ids)
    print(f"\nTarget IDs (前 6 列):")
    print(target_ids)

    # 验证 padding 是否正确
    assert input_ids.shape[0] == 4, "Batch size 应该为 4"
    assert input_ids.shape[1] == 6, "序列长度应该为 max_len - 1 = 7 - 1 = 6"

    # 验证 padding token
    assert input_ids[1, 3] == 13, "第 2 个序列的第 4 个 token 应该是 13"
    assert input_ids[1, 4] == 0, "第 2 个序列的第 5 个位置应该是 padding (0)"
    assert input_ids[1, 5] == 0, "第 2 个序列的第 6 个位置应该是 padding (0)"

    print("\n✓ collate_fn 正确处理了不同长度的序列")


def test_padding_mask_creation():
    """测试 padding mask 是否正确创建"""
    print("\n" + "=" * 60)
    print("测试 Padding Mask 创建")
    print("=" * 60)

    # 创建包含 padding 的序列
    seq = torch.tensor([
        [1, 2, 3, 4, 5, 0, 0],    # 后两个是 padding
        [10, 11, 12, 0, 0, 0, 0],  # 后四个是 padding
        [20, 21, 22, 23, 24, 25, 26],  # 没有 padding
    ])

    pad_token_id = 0
    padding_mask = create_padding_mask(seq, pad_token_id)

    print(f"序列形状: {seq.shape}")
    print(f"Padding Mask 形状: {padding_mask.shape}")
    print(f"\nPadding Mask (batch=0):")
    print(padding_mask[0, 0, 0, :])
    print(f"Padding Mask (batch=1):")
    print(padding_mask[1, 0, 0, :])
    print(f"Padding Mask (batch=2):")
    print(padding_mask[2, 0, 0, :])

    # 验证 mask 是否正确
    assert padding_mask[0, 0, 0, 0] == True, "第 1 个序列的第 1 个位置应该是 True"
    assert padding_mask[0, 0, 0, 5] == False, "第 1 个序列的第 6 个位置（padding）应该是 False"
    assert padding_mask[1, 0, 0, 3] == False, "第 2 个序列的第 4 个位置（padding）应该是 False"
    assert padding_mask[2, 0, 0, 6] == True, "第 3 个序列的第 7 个位置应该是 True（无 padding）"

    print("\n✓ Padding Mask 创建正确")


def test_combined_mask():
    """测试组合 mask（causal + padding）"""
    print("\n" + "=" * 60)
    print("测试组合 Mask（Causal + Padding）")
    print("=" * 60)

    seq_len = 5
    seq = torch.tensor([
        [1, 2, 3, 0, 0],  # 后两个是 padding
    ])

    pad_token_id = 0
    causal_mask = create_causal_mask(seq_len)
    padding_mask = create_padding_mask(seq, pad_token_id)
    combined = combine_masks(causal_mask, padding_mask)

    print(f"Causal Mask (下三角):")
    print(causal_mask[0, 0, :, :])
    print(f"\nPadding Mask:")
    print(padding_mask[0, 0, :, :])
    print(f"\nCombined Mask (Causal AND Padding):")
    print(combined[0, 0, :, :])

    # 验证组合 mask
    # 第 1 行：只能看到第 1 个位置（因为是因果 mask）
    assert combined[0, 0, 0, 0] == True, "位置 (0,0) 应该是 True"
    assert combined[0, 0, 0, 1] == False, "位置 (0,1) 应该是 False（因果）"

    # 第 3 行：可以看到前 3 个位置（因果），但不能看到后 2 个（padding）
    assert combined[0, 0, 2, 0] == True, "位置 (2,0) 应该是 True"
    assert combined[0, 0, 2, 1] == True, "位置 (2,1) 应该是 True"
    assert combined[0, 0, 2, 2] == True, "位置 (2,2) 应该是 True"
    assert combined[0, 0, 2, 3] == False, "位置 (2,3) 应该是 False（padding）"
    assert combined[0, 0, 2, 4] == False, "位置 (2,4) 应该是 False（padding）"

    # 第 4 行（padding 位置）：
    # 注意：padding mask 只 mask 掉 key 的 padding 位置，query 位置仍然会计算
    # 这是因为 loss 计算时会通过 ignore_index 忽略 padding 位置的输出
    # 所以位置 (3,0) 仍然是 True（query=3 可以看到 key=0）
    # 但位置 (3,3) 和 (3,4) 是 False（因为 key 位置 3,4 是 padding）
    assert combined[0, 0, 3, 0] == True, "位置 (3,0) 应该是 True（query 位置即使是 padding 也会计算，loss 时通过 ignore_index 忽略）"
    assert combined[0, 0, 3, 3] == False, "位置 (3,3) 应该是 False（key 是 padding）"
    assert combined[0, 0, 3, 4] == False, "位置 (3,4) 应该是 False（key 是 padding）"

    print("\n✓ 组合 Mask 正确（同时满足因果关系和忽略 padding）")
    print("  注意：padding mask 只 mask 掉 key 的 padding，query 的 padding 通过 loss ignore_index 处理")


def test_collate_fn_mt():
    """测试机器翻译的 collate_fn"""
    print("\n" + "=" * 60)
    print("测试机器翻译 collate_fn")
    print("=" * 60)

    src_batch = [
        [1, 2, 3, 4, 5],
        [10, 11, 12],
        [20, 21, 22, 23],
    ]

    tgt_batch = [
        [101, 102, 103, 104],
        [201, 202],
        [301, 302, 303, 304, 305, 306],
    ]

    pad_token_id = 0
    src_tensor, tgt_tensor = collate_fn_mt(src_batch, tgt_batch, pad_token_id)

    print(f"源序列长度: {[len(x) for x in src_batch]}")
    print(f"目标序列长度: {[len(x) for x in tgt_batch]}")
    print(f"\nPadding 后源序列形状: {src_tensor.shape}")
    print(f"Padding 后目标序列形状: {tgt_tensor.shape}")
    print(f"\n源序列 tensor:")
    print(src_tensor)
    print(f"\n目标序列 tensor:")
    print(tgt_tensor)

    # 验证
    assert src_tensor.shape == (3, 5), "源序列形状应该是 (3, 5)"
    assert tgt_tensor.shape == (3, 6), "目标序列形状应该是 (3, 6)"

    # 验证 padding
    assert src_tensor[1, 2] == 12, "第 2 个源序列的第 3 个 token 应该是 12"
    assert src_tensor[1, 3] == 0, "第 2 个源序列的第 4 个位置应该是 padding"
    assert tgt_tensor[1, 1] == 202, "第 2 个目标序列的第 2 个 token 应该是 202"
    assert tgt_tensor[1, 2] == 0, "第 2 个目标序列的第 3 个位置应该是 padding"

    print("\n✓ 机器翻译 collate_fn 正确")


def test_loss_with_padding():
    """测试 CrossEntropyLoss 是否正确忽略 padding"""
    print("\n" + "=" * 60)
    print("测试 Loss 计算时忽略 Padding")
    print("=" * 60)

    import torch.nn as nn

    vocab_size = 100
    pad_token_id = 0

    # 创建模拟的 logits 和 targets
    # Shape: (batch=2, seq_len=4, vocab_size=100)
    logits = torch.randn(2, 4, vocab_size)
    targets = torch.tensor([
        [10, 20, 30, 0],  # 最后一个是 padding
        [40, 50, 0, 0],   # 最后两个是 padding
    ])

    # 不忽略 padding 的损失
    criterion_no_ignore = nn.CrossEntropyLoss(reduction='sum')
    loss_no_ignore = criterion_no_ignore(
        logits.view(-1, vocab_size),
        targets.view(-1)
    )

    # 忽略 padding 的损失
    criterion_with_ignore = nn.CrossEntropyLoss(ignore_index=pad_token_id, reduction='sum')
    loss_with_ignore = criterion_with_ignore(
        logits.view(-1, vocab_size),
        targets.view(-1)
    )

    print(f"不忽略 padding 的损失: {loss_no_ignore.item():.4f}")
    print(f"忽略 padding 的损失: {loss_with_ignore.item():.4f}")

    # 手动计算只有非 padding 位置的损失
    non_pad_mask = targets != pad_token_id
    num_non_pad = non_pad_mask.sum().item()
    print(f"\n非 padding token 数量: {num_non_pad} / {targets.numel()}")

    # 损失应该不同（忽略 padding 后损失应该更小）
    assert loss_no_ignore > loss_with_ignore, "忽略 padding 后损失应该更小"

    print("\n✓ CrossEntropyLoss 正确忽略了 padding token")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("开始测试 Padding Mask 修复")
    print("=" * 60 + "\n")

    try:
        test_collate_fn_with_different_lengths()
        test_padding_mask_creation()
        test_combined_mask()
        test_collate_fn_mt()
        test_loss_with_padding()

        print("\n" + "=" * 60)
        print("🎉 所有 Padding Mask 测试通过！修复成功！")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 测试失败: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
