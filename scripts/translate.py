import torch
from models.transformer_models import EncoderDecoderModel
from utils.sentencepiece_tokenizer import SentencePieceTokenizer
from utils.translation_utils import (
    beam_search_translate,
    greedy_translate,
    top_k_translate,
    top_p_translate
)
import os

def main():
    model_path = "encoder_decoder_best.pt"  # 使用最佳模型
    tokenizer_path = "tokenization/sentencepiece_enzh.model"
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = SentencePieceTokenizer.from_pretrained(tokenizer_path)
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

    print("\n解码策略: [1] Beam Search ⭐ [2] Top-P [3] Greedy [4] Top-K")
    print("输入英文句子翻译，exit 退出\n")

    while True:
        src = input("\nEN> ").strip()
        if src.lower() == "exit":
            break

        if not src:
            continue

        mode = input("策略 [1-Beam/2-TopP/3-Greedy/4-TopK] (默认1): ").strip() or "1"
        src_ids = tokenizer.encode(src, add_special_tokens=False)

        try:
            if mode == "1":
                zh = beam_search_translate(model, src_ids, tokenizer, beam_width=5, max_len=50, device=device)
            elif mode == "2":
                zh = top_p_translate(model, src_ids, tokenizer, p=0.9, max_len=50, device=device)
            elif mode == "3":
                zh = greedy_translate(model, src_ids, tokenizer, max_len=50, device=device)
            elif mode == "4":
                zh = top_k_translate(model, src_ids, tokenizer, k=10, max_len=50, device=device)
            else:
                zh = beam_search_translate(model, src_ids, tokenizer, beam_width=5, max_len=50, device=device)

            print(f"ZH> {zh}")

        except Exception as e:
            print(f"翻译出错: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
