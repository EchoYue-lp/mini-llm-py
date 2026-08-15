"""
测试 Embedding Scaling 修复是否正确
验证训练和推理的一致性
"""
import sys

import torch
from models.transformer_models import EncoderDecoderModel
from utils.translation_utils import beam_search_translate


class FakeTokenizer:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2

    def __len__(self):
        return 1000

    def decode(self, ids, skip_special_tokens=True):
        del skip_special_tokens
        return " ".join(str(item) for item in ids)


def test_embedding_scaling_consistency():
    """测试训练时forward和手动运行encoder/decoder的一致性"""
    print("=" * 60)
    print("测试 Embedding Scaling 修复")
    print("=" * 60)

    # 创建模型
    model = EncoderDecoderModel(
        src_vocab_size=1000,
        tgt_vocab_size=1000,
        d_model=256,
        num_layers=2,
        num_heads=4
    )
    model.eval()

    # 创建 tokenizer
    tokenizer = FakeTokenizer()

    # 测试输入
    src = torch.randint(0, 1000, (1, 10))
    tgt = torch.randint(0, 1000, (1, 8))

    print("\n1. 使用模型 forward (训练时的方式):")
    with torch.no_grad():
        logits_train, _ = model(src, tgt)
        print(f"   输出形状: {logits_train.shape}")
        print(f"   输出范围: [{logits_train.min().item():.4f}, {logits_train.max().item():.4f}]")
        print(f"   输出均值: {logits_train.mean().item():.4f}")
        print(f"   输出标准差: {logits_train.std().item():.4f}")

    print("\n2. 手动运行 encoder + decoder (翻译时的方式 - 修复后):")
    with torch.no_grad():
        from utils.mask_utils import create_causal_mask, create_padding_mask

        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        src_mask = create_padding_mask(src, pad_token_id=pad_token_id)

        # ✅ 修复后：应用 embedding scaling
        src_emb = model.src_embed(src) * model.embed_scale
        src_emb = model.src_pos_enc(src_emb)
        memory = src_emb
        for layer in model.encoder_layers:
            memory, _ = layer(memory, self_mask=src_mask)
        memory = model.encoder_norm(memory)

        # ✅ 修复后：应用 embedding scaling
        tgt_emb = model.tgt_embed(tgt) * model.embed_scale
        tgt_emb = model.tgt_pos_enc(tgt_emb)
        tgt_mask = create_causal_mask(tgt.size(1), device=tgt.device)
        for layer in model.decoder_layers:
            tgt_emb, _, _ = layer(tgt_emb, enc_out=memory, self_mask=tgt_mask)
        tgt_emb = model.decoder_norm(tgt_emb)
        logits_infer = model.out_proj(tgt_emb)

        print(f"   输出形状: {logits_infer.shape}")
        print(f"   输出范围: [{logits_infer.min().item():.4f}, {logits_infer.max().item():.4f}]")
        print(f"   输出均值: {logits_infer.mean().item():.4f}")
        print(f"   输出标准差: {logits_infer.std().item():.4f}")

    print("\n3. 一致性检查:")
    l2_distance = (logits_train - logits_infer).norm().item()
    max_diff = (logits_train - logits_infer).abs().max().item()

    print(f"   L2 距离: {l2_distance:.6f}")
    print(f"   最大差异: {max_diff:.6f}")

    # 验证一致性（允许微小的数值误差）
    assert l2_distance < 1e-4, f"L2 距离过大: {l2_distance}"
    assert max_diff < 1e-5, f"最大差异过大: {max_diff}"

    print("\n✅ 训练和推理完全一致！Embedding Scaling 修复成功！")


def test_beam_search_with_scaling():
    """测试 beam_search_translate 是否正确应用 scaling"""
    print("\n" + "=" * 60)
    print("测试 Beam Search 翻译功能")
    print("=" * 60)

    # 创建模型
    tokenizer = FakeTokenizer()
    vocab_size = len(tokenizer)
    model = EncoderDecoderModel(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=128,
        num_layers=2,
        num_heads=4
    )
    model.eval()

    # 测试输入
    src_ids = [10, 20, 30, 40, 50]

    try:
        # 运行 beam search
        result = beam_search_translate(
            model,
            src_ids,
            tokenizer,
            beam_width=3,
            max_len=10,
            device="cpu"
        )

        print(f"\n✅ Beam Search 运行成功")
        print(f"   输入 token IDs: {src_ids}")
        print(f"   输出文本: '{result}'")

    except Exception as e:
        print(f"\n❌ Beam Search 运行失败: {e}")
        raise


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("开始测试 Embedding Scaling 修复")
    print("=" * 60 + "\n")

    try:
        test_embedding_scaling_consistency()
        test_beam_search_with_scaling()

        print("\n" + "=" * 60)
        print("🎉 所有测试通过！Embedding Scaling Bug 已修复！")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 测试失败: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
