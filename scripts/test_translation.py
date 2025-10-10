"""
测试不同翻译策略的效果
对比 Greedy、Beam Search、Top-K、Top-P
"""

import torch
from models.transformer_models import EncoderDecoderModel
from transformers import GPT2TokenizerFast
from utils.translation_utils import (
    beam_search_translate,
    greedy_translate,
    top_k_translate,
    top_p_translate
)
import time


def test_all_methods(model, tokenizer, test_sentences, device="cpu"):
    """
    对比所有翻译方法
    """
    print("=" * 80)
    print("翻译策略对比测试")
    print("=" * 80)

    for i, sentence in enumerate(test_sentences, 1):
        print(f"\n测试句子 {i}: {sentence}")
        print("-" * 80)

        src_ids = tokenizer.encode(sentence, add_special_tokens=False)

        methods = [
            ("Greedy", lambda: greedy_translate(model, src_ids, tokenizer, max_len=50, device=device)),
            ("Beam (k=3)", lambda: beam_search_translate(model, src_ids, tokenizer, beam_width=3, max_len=50, device=device)),
            ("Beam (k=5)", lambda: beam_search_translate(model, src_ids, tokenizer, beam_width=5, max_len=50, device=device)),
            ("Top-K", lambda: top_k_translate(model, src_ids, tokenizer, k=10, max_len=50, device=device)),
            ("Top-P", lambda: top_p_translate(model, src_ids, tokenizer, p=0.9, max_len=50, device=device)),
        ]

        results = []
        for name, method in methods:
            try:
                start = time.time()
                translation = method()
                elapsed = time.time() - start
                results.append((name, translation, elapsed))
            except Exception as e:
                results.append((name, f"Error: {e}", 0))

        # 打印结果
        for name, translation, elapsed in results:
            time_str = f"({elapsed*1000:.1f}ms)" if elapsed > 0 else ""
            print(f"{name:12s} {time_str:12s} → {translation}")

        print()


def main():
    model_path = "encoder_decoder_best.pt"
    tokenizer_dir = "tokenization/gpt2"
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model from {model_path}...")
    print(f"Device: {device}\n")

    tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_dir)
    src_vocab_size = tgt_vocab_size = tokenizer.vocab_size

    # 加载模型
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) else checkpoint
    max_len = state_dict['src_pos_enc.pe'].shape[0] if 'src_pos_enc.pe' in state_dict else 128

    model = EncoderDecoderModel(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=256,
        num_layers=4,
        num_heads=4,
        d_ff=1024,
        max_len=max_len
    )
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    if isinstance(checkpoint, dict) and 'epoch' in checkpoint:
        print(f"✓ Model loaded (Epoch {checkpoint['epoch']}, Val Loss: {checkpoint['val_loss']:.4f})\n")

    # 测试句子
    test_sentences = [
        "Hello, how are you?",
        "I love programming.",
        "The weather is nice today.",
        "Machine learning is fascinating.",
        "Can you help me with this problem?",
    ]

    test_all_methods(model, tokenizer, test_sentences, device)

    print("\n" + "=" * 80)
    print("总结:")
    print("=" * 80)
    print("""
观察要点:
1. Beam Search 通常生成更流畅、更自然的翻译
2. Beam width 越大，质量可能越好，但速度越慢
3. Greedy 最快，但可能错过更好的翻译
4. Top-K/Top-P 每次运行结果可能不同（随机性）
5. 对于翻译任务，推荐使用 Beam Search (k=4-6)
    """)


if __name__ == "__main__":
    main()
