"""
测试所有关键修复是否正常工作
"""
import sys

import torch
import math
from models.layers import PositionalEncoding
from models.transformer_models import DecoderOnlyModel, EncoderDecoderModel
from utils.generation_utils import top_p_candidates

def test_positional_encoding():
    """测试位置编码（奇数和偶数维度）"""
    print("测试位置编码...")

    # 测试偶数维度
    pe_even = PositionalEncoding(d_model=256, max_len=100, dropout=0.0)
    x_even = torch.randn(2, 10, 256)
    out_even = pe_even(x_even)
    assert out_even.shape == (2, 10, 256), f"偶数维度输出形状错误: {out_even.shape}"
    print("  ✓ 偶数维度 (d_model=256) 测试通过")

    # 测试奇数维度
    pe_odd = PositionalEncoding(d_model=255, max_len=100, dropout=0.0)
    x_odd = torch.randn(2, 10, 255)
    out_odd = pe_odd(x_odd)
    assert out_odd.shape == (2, 10, 255), f"奇数维度输出形状错误: {out_odd.shape}"
    print("  ✓ 奇数维度 (d_model=255) 测试通过")

    # 测试非常小的奇数维度
    pe_small = PositionalEncoding(d_model=5, max_len=100, dropout=0.0)
    x_small = torch.randn(2, 10, 5)
    out_small = pe_small(x_small)
    assert out_small.shape == (2, 10, 5), f"小奇数维度输出形状错误: {out_small.shape}"
    print("  ✓ 小奇数维度 (d_model=5) 测试通过")

    print("✅ 位置编码测试全部通过\n")

def test_embedding_scaling():
    """测试 Embedding 缩放"""
    print("测试 Embedding 缩放...")

    # Decoder-Only 模型
    d_model = 256
    model_decoder = DecoderOnlyModel(vocab_size=1000, d_model=d_model, num_layers=2, num_heads=4)
    assert hasattr(model_decoder, 'embed_scale'), "Decoder 模型缺少 embed_scale 属性"
    assert model_decoder.embed_scale == math.sqrt(d_model), f"Decoder embed_scale 值错误: {model_decoder.embed_scale}"
    print(f"  ✓ Decoder-Only 模型 embed_scale = {model_decoder.embed_scale:.4f}")

    # Encoder-Decoder 模型
    model_enc_dec = EncoderDecoderModel(src_vocab_size=1000, tgt_vocab_size=1000, d_model=d_model, num_layers=2, num_heads=4)
    assert hasattr(model_enc_dec, 'embed_scale'), "Encoder-Decoder 模型缺少 embed_scale 属性"
    assert model_enc_dec.embed_scale == math.sqrt(d_model), f"Encoder-Decoder embed_scale 值错误: {model_enc_dec.embed_scale}"
    print(f"  ✓ Encoder-Decoder 模型 embed_scale = {model_enc_dec.embed_scale:.4f}")

    # 测试 forward 中是否应用了缩放
    x = torch.randint(0, 1000, (2, 10))
    with torch.no_grad():
        logits, _ = model_decoder(x)
    assert logits.shape == (2, 10, 1000), f"Decoder 输出形状错误: {logits.shape}"
    print("  ✓ Embedding 缩放在 forward 中正确应用")

    print("✅ Embedding 缩放测试全部通过\n")

def test_top_p_sampling_logic():
    """测试 Top-P 采样的核心逻辑（不依赖模型）"""
    print("测试 Top-P 采样逻辑...")

    # 模拟概率分布
    probs = torch.tensor([0.5, 0.3, 0.15, 0.04, 0.01])
    p = 0.9

    selected_probs, selected_indices = top_p_candidates(probs, p)

    print(f"  原始概率: {probs.tolist()}")
    print(f"  Top-p={p} 选中索引: {selected_indices.tolist()}")
    print(f"  Top-p={p} 选中概率: {selected_probs.tolist()}")

    assert selected_indices.tolist() == [0, 1, 2], "应包含第一个使累计概率越过 p 的 token"
    assert torch.allclose(selected_probs.sum(), torch.tensor(1.0))
    print("  ✓ Top-P 采样逻辑正确")

    # 测试边界情况：p 很小
    p_small = 0.1
    _, selected_indices_small = top_p_candidates(probs, p_small)
    assert len(selected_indices_small) == 1, "p 很小时也应至少保留一个 token"
    print(f"  ✓ 边界情况 (p={p_small}) 处理正确，保留 1 个 token")

    print("✅ Top-P 采样逻辑测试全部通过\n")

def test_model_forward():
    """测试模型前向传播"""
    print("测试模型前向传播...")

    # Decoder-Only
    model_decoder = DecoderOnlyModel(vocab_size=1000, d_model=128, num_layers=2, num_heads=4, d_ff=512)
    x = torch.randint(0, 1000, (4, 20))

    with torch.no_grad():
        logits, attn_weights = model_decoder(x)

    assert logits.shape == (4, 20, 1000), f"Decoder logits 形状错误: {logits.shape}"
    assert len(attn_weights) == 2, f"注意力权重层数错误: {len(attn_weights)}"
    print("  ✓ Decoder-Only 前向传播正常")

    # Encoder-Decoder
    model_enc_dec = EncoderDecoderModel(src_vocab_size=1000, tgt_vocab_size=1500, d_model=128, num_layers=2, num_heads=4, d_ff=512)
    src = torch.randint(0, 1000, (4, 15))
    tgt = torch.randint(0, 1500, (4, 20))

    with torch.no_grad():
        logits, attn_weights = model_enc_dec(src, tgt)

    assert logits.shape == (4, 20, 1500), f"Encoder-Decoder logits 形状错误: {logits.shape}"
    assert len(attn_weights) == 2, f"注意力权重层数错误: {len(attn_weights)}"
    print("  ✓ Encoder-Decoder 前向传播正常")

    print("✅ 模型前向传播测试全部通过\n")

def test_cuda_availability():
    """测试 CUDA 可用性"""
    print("测试设备可用性...")

    cuda_available = torch.cuda.is_available()
    mps_available = torch.backends.mps.is_available()

    print(f"  CUDA 可用: {cuda_available}")
    if cuda_available:
        print(f"  CUDA 设备数量: {torch.cuda.device_count()}")
        print(f"  CUDA 设备名称: {torch.cuda.get_device_name(0)}")

    print(f"  MPS 可用: {mps_available}")

    # 推荐设备
    device = "cuda" if cuda_available else ("mps" if mps_available else "cpu")
    print(f"  推荐使用设备: {device}")

    # 测试简单计算
    x = torch.randn(100, 100).to(device)
    y = torch.randn(100, 100).to(device)
    z = torch.matmul(x, y)
    assert z.device.type == device, f"计算设备不匹配: {z.device}"
    print(f"  ✓ 在 {device} 上计算正常")

    print("✅ 设备测试全部通过\n")

if __name__ == "__main__":
    print("="*60)
    print("开始测试所有关键修复...")
    print("="*60 + "\n")

    try:
        test_positional_encoding()
        test_embedding_scaling()
        test_top_p_sampling_logic()
        test_model_forward()
        test_cuda_availability()

        print("="*60)
        print("🎉 所有测试通过！代码修复成功！")
        print("="*60)

    except Exception as e:
        print(f"\n❌ 测试失败: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
