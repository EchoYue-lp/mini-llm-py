import torch
from models.transformer_models import EncoderDecoderModel
from transformers import GPT2TokenizerFast
from utils.translation_utils import (
    beam_search_translate,
    greedy_translate,
    top_k_translate,
    top_p_translate
)
import os

def main():
    model_path = "encoder_decoder_best.pt"  # 使用最佳模型
    tokenizer_dir = "tokenization/gpt2"
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_dir)
    src_vocab_size = tgt_vocab_size = tokenizer.vocab_size

    print(f"正在从 {model_path} 加载模型...")
    print(f"使用设备: {device}\n")

    # 使用 checkpoint_utils 自动加载模型配置
    from utils.checkpoint_utils import load_model_from_checkpoint

    model, checkpoint_info = load_model_from_checkpoint(
        model_path,
        EncoderDecoderModel,
        model_type='encoder_decoder',
        device=device
    )

    if not checkpoint_info:
        print("警告：使用默认配置，如果训练时修改了配置，翻译结果可能不正确\n")

    print("=" * 60)
    print("翻译模式说明:")
    print("  1. Greedy    - 快速，每步选最优 token")
    print("  2. Beam      - 高质量，搜索多个候选序列 (推荐)")
    print("  3. Top-K     - 随机性，从 top-k 中采样")
    print("  4. Top-P     - 随机性，nucleus 采样")
    print("=" * 60)

    print("\n请输入英文句子，回车翻译，输入 exit 退出：")

    while True:
        src = input("\nEN> ").strip()
        if src.lower() == "exit":
            break

        if not src:
            continue

        print("选择生成方式 [1-Greedy/2-Beam/3-Top-K/4-Top-P] (默认2): ", end="")
        mode = input().strip() or "2"

        src_ids = tokenizer.encode(src, add_special_tokens=False)

        try:
            if mode == "1":
                print("使用 Greedy Decoding...")
                zh = greedy_translate(model, src_ids, tokenizer, max_len=50, device=device)
            elif mode == "2":
                print("使用 Beam Search (beam_width=5)...")
                zh = beam_search_translate(model, src_ids, tokenizer, beam_width=5, max_len=50, device=device)
            elif mode == "3":
                print("使用 Top-K Sampling (k=10)...")
                zh = top_k_translate(model, src_ids, tokenizer, k=10, max_len=50, device=device)
            elif mode == "4":
                print("使用 Top-P Sampling (p=0.9)...")
                zh = top_p_translate(model, src_ids, tokenizer, p=0.9, max_len=50, device=device)
            else:
                print("无效选择，使用默认 Beam Search...")
                zh = beam_search_translate(model, src_ids, tokenizer, beam_width=5, max_len=50, device=device)

            print(f"ZH> {zh}")

        except Exception as e:
            print(f"翻译出错: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
