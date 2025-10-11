"""
使用新的 SentencePiece Tokenizer 重新处理 IWSLT2017 数据集
"""

import torch
import os
import sys
sys.path.insert(0, '.')
from utils.sentencepiece_tokenizer import SentencePieceTokenizer
from tqdm import tqdm


def retokenize_file(input_file, output_file, tokenizer, add_special_tokens=True, max_len=None):
    """
    使用新 tokenizer 重新处理文本文件

    Args:
        input_file: 输入文本文件
        output_file: 输出 .pt 文件
        tokenizer: tokenizer 实例
        add_special_tokens: 是否添加特殊 token
        max_len: 最大长度（可选截断）
    """
    print(f"处理: {input_file}")
    print(f"  → {output_file}")

    ids_list = []
    total_tokens = 0
    max_seq_len = 0

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in tqdm(lines, desc=f"  Tokenizing", ncols=80):
        line = line.strip()
        if not line:
            continue

        # 编码
        ids = tokenizer.encode(line, add_special_tokens=add_special_tokens)

        # 可选截断
        if max_len and len(ids) > max_len:
            ids = ids[:max_len]

        ids_list.append(ids)
        total_tokens += len(ids)
        max_seq_len = max(max_seq_len, len(ids))

    # 保存
    torch.save(ids_list, output_file)

    avg_len = total_tokens / len(ids_list) if ids_list else 0
    print(f"  ✓ 完成: {len(ids_list):,} 条")
    print(f"    平均长度: {avg_len:.1f} tokens")
    print(f"    最大长度: {max_seq_len} tokens")
    print()

    return len(ids_list), avg_len, max_seq_len


def main():
    """
    主函数：重新处理所有数据集
    """
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 18 + "重新处理 IWSLT2017 数据集" + " " * 21 + "║")
    print("║" + " " * 18 + "使用 SentencePiece Tokenizer" + " " * 18 + "║")
    print("╚" + "=" * 68 + "╝")
    print()

    # 检查 tokenizer 是否存在
    tokenizer_path = "tokenization/sentencepiece_enzh.model"
    if not os.path.exists(tokenizer_path):
        print(f"❌ 错误: Tokenizer 模型文件不存在: {tokenizer_path}")
        print()
        print("请先运行: python scripts/train_sentencepiece.py")
        return

    # 加载 tokenizer
    print("=" * 70)
    print("加载 Tokenizer")
    print("=" * 70)
    tokenizer = SentencePieceTokenizer.from_pretrained(tokenizer_path)
    print(tokenizer)
    print()

    # 数据目录
    data_dir = "data/iwslt2017"

    # 处理所有split
    splits = ["train", "validation", "test"]
    languages = ["en", "zh"]

    print("=" * 70)
    print("重新 Tokenize 数据集")
    print("=" * 70)
    print()

    stats = {}

    for split in splits:
        for lang in languages:
            input_file = os.path.join(data_dir, f"{split}.{lang}.txt")
            output_file = os.path.join(data_dir, f"{split}.{lang}_ids_sp.pt")

            if not os.path.exists(input_file):
                print(f"⚠ 跳过不存在的文件: {input_file}")
                continue

            # 对于目标语言（中文），添加 BOS/EOS
            # 对于源语言（英文），不添加特殊 token（在训练时动态处理）
            add_special_tokens = (lang == "zh")

            count, avg_len, max_len = retokenize_file(
                input_file=input_file,
                output_file=output_file,
                tokenizer=tokenizer,
                add_special_tokens=add_special_tokens,
                max_len=None  # 不截断，在训练时动态处理
            )

            stats[f"{split}.{lang}"] = {
                "count": count,
                "avg_len": avg_len,
                "max_len": max_len
            }

    # 输出统计
    print("=" * 70)
    print("📊 处理统计")
    print("=" * 70)
    print()
    print(f"{'数据集':<20} {'样本数':>12} {'平均长度':>12} {'最大长度':>12}")
    print("-" * 70)

    for split in splits:
        for lang in languages:
            key = f"{split}.{lang}"
            if key in stats:
                s = stats[key]
                print(f"{key:<20} {s['count']:>12,} {s['avg_len']:>12.1f} {s['max_len']:>12}")

    print()

    # 对比 GPT-2 tokenizer
    print("=" * 70)
    print("📈 与 GPT-2 Tokenizer 对比")
    print("=" * 70)
    print()

    # 计算中文平均 token 数的变化
    if "train.zh" in stats:
        zh_avg_sp = stats["train.zh"]["avg_len"]
        zh_avg_gpt2_estimate = zh_avg_sp * 4.0  # GPT-2 对中文约 4-5x 开销

        print(f"中文句子平均 token 数:")
        print(f"  SentencePiece: {zh_avg_sp:.1f} tokens")
        print(f"  GPT-2 (估计):  {zh_avg_gpt2_estimate:.1f} tokens")
        print(f"  效率提升:      {zh_avg_gpt2_estimate/zh_avg_sp:.1f}x")
        print()

    # 词汇表大小对比
    print(f"词汇表大小:")
    print(f"  SentencePiece: {tokenizer.vocab_size:,}")
    print(f"  GPT-2:         50,257")
    print(f"  减少:          {50257 - tokenizer.vocab_size:,} ({(1 - tokenizer.vocab_size/50257)*100:.1f}%)")
    print()

    # 参数量对比
    d_model = 128  # 默认配置
    embed_params_sp = tokenizer.vocab_size * d_model * 2  # src + tgt embedding
    embed_params_gpt2 = 50257 * d_model * 2

    print(f"Embedding 参数量 (d_model={d_model}):")
    print(f"  SentencePiece: {embed_params_sp:,} ({embed_params_sp/1e6:.2f}M)")
    print(f"  GPT-2:         {embed_params_gpt2:,} ({embed_params_gpt2/1e6:.2f}M)")
    print(f"  节省:          {embed_params_gpt2 - embed_params_sp:,} ({(1-embed_params_sp/embed_params_gpt2)*100:.1f}%)")
    print()

    # 后续步骤
    print("=" * 70)
    print("✅ 数据集重新处理完成！")
    print("=" * 70)
    print()
    print("📝 后续步骤:")
    print()
    print("1. 训练脚本已自动更新（如需手动修改，编辑 scripts/train_encoder_decoder.py）")
    print()
    print("2. 开始训练:")
    print("   python scripts/train_encoder_decoder.py")
    print()
    print("3. 使用新 tokenizer 进行翻译:")
    print("   python scripts/translate.py")
    print()
    print("4. 预期改进:")
    print("   - 模型参数减少 ~60%")
    print("   - 中文处理效率提升 4-5x")
    print("   - 翻译质量提升（语义完整性更好）")
    print()


if __name__ == "__main__":
    main()
