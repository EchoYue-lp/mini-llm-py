"""
训练自定义 SentencePiece Tokenizer
针对英中翻译任务优化

使用 IWSLT2017 英中数据集训练一个统一的 tokenizer
"""

import sentencepiece as spm
import os

def prepare_training_data(
    data_dir="data/iwslt2017",
    output_file="data/iwslt2017/combined_corpus.txt",
    max_sentences=1000000
):
    """
    合并英文和中文训练语料，用于训练 tokenizer

    Args:
        data_dir: 数据目录
        output_file: 输出合并语料文件
        max_sentences: 最大句子数（防止内存溢出）
    """
    print("=" * 70)
    print("准备训练数据")
    print("=" * 70)

    # 读取训练集（英文和中文）
    en_file = os.path.join(data_dir, "train.en.txt")
    zh_file = os.path.join(data_dir, "train.zh.txt")

    print(f"读取英文语料: {en_file}")
    print(f"读取中文语料: {zh_file}")

    with open(output_file, 'w', encoding='utf-8') as out_f:
        # 读取英文
        with open(en_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= max_sentences:
                    break
                line = line.strip()
                if line:
                    out_f.write(line + '\n')

        en_count = i + 1

        # 读取中文
        with open(zh_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= max_sentences:
                    break
                line = line.strip()
                if line:
                    out_f.write(line + '\n')

        zh_count = i + 1

    total_lines = en_count + zh_count
    print(f"\n✓ 合并完成:")
    print(f"  英文句子: {en_count:,}")
    print(f"  中文句子: {zh_count:,}")
    print(f"  总计: {total_lines:,}")
    print(f"  输出文件: {output_file}")
    print()

    return output_file, total_lines


def train_tokenizer(
    input_file,
    model_prefix="tokenization/sentencepiece_enzh",
    vocab_size=16000,
    character_coverage=0.9995,
    model_type='bpe'
):
    """
    训练 SentencePiece tokenizer

    Args:
        input_file: 输入语料文件
        model_prefix: 模型输出前缀
        vocab_size: 词汇表大小
        character_coverage: 字符覆盖率（对中文很重要）
        model_type: 模型类型 (bpe 或 unigram)
    """
    print("=" * 70)
    print("训练 SentencePiece Tokenizer")
    print("=" * 70)
    print(f"配置:")
    print(f"  输入文件: {input_file}")
    print(f"  模型前缀: {model_prefix}")
    print(f"  词汇表大小: {vocab_size:,}")
    print(f"  模型类型: {model_type}")
    print(f"  字符覆盖率: {character_coverage}")
    print()

    # 创建输出目录
    os.makedirs(os.path.dirname(model_prefix), exist_ok=True)

    # 训练参数
    train_args = {
        'input': input_file,
        'model_prefix': model_prefix,
        'vocab_size': vocab_size,
        'character_coverage': character_coverage,
        'model_type': model_type,
        # 特殊 token
        'pad_id': 0,
        'unk_id': 1,
        'bos_id': 2,
        'eos_id': 3,
        'pad_piece': '<pad>',
        'unk_piece': '<unk>',
        'bos_piece': '<s>',
        'eos_piece': '</s>',
        # 其他设置
        'max_sentence_length': 4096,
        'shuffle_input_sentence': True,
        'train_extremely_large_corpus': False,
        # 用户自定义 token（可选）
        'user_defined_symbols': [],
        # 数字处理
        'byte_fallback': True,  # 处理未知字符
    }

    print("开始训练... (可能需要 5-10 分钟)")
    print()

    # 训练
    spm.SentencePieceTrainer.train(**{k: v for k, v in train_args.items()})

    print()
    print("=" * 70)
    print("✓ 训练完成！")
    print("=" * 70)
    print(f"模型文件: {model_prefix}.model")
    print(f"词汇表文件: {model_prefix}.vocab")
    print()

    return f"{model_prefix}.model", f"{model_prefix}.vocab"


def test_tokenizer(model_path):
    """
    测试训练好的 tokenizer
    """
    print("=" * 70)
    print("测试 Tokenizer")
    print("=" * 70)

    sp = spm.SentencePieceProcessor()
    sp.load(model_path)

    # 测试样例
    test_cases = [
        ("Hello, how are you?", "英文简单句"),
        ("I love machine translation.", "英文专业句"),
        ("你好，你好吗？", "中文简单句"),
        ("我喜欢机器翻译。", "中文专业句"),
        ("The quick brown fox jumps over the lazy dog.", "英文长句"),
        ("人工智能和深度学习正在改变世界。", "中文长句"),
    ]

    print(f"词汇表大小: {sp.vocab_size():,}")
    print(f"PAD token: '{sp.id_to_piece(0)}' (id={0})")
    print(f"UNK token: '{sp.id_to_piece(1)}' (id={1})")
    print(f"BOS token: '{sp.id_to_piece(2)}' (id={2})")
    print(f"EOS token: '{sp.id_to_piece(3)}' (id={3})")
    print()

    print("-" * 70)
    print("测试样例:")
    print("-" * 70)

    for text, desc in test_cases:
        ids = sp.encode(text)
        tokens = sp.encode(text, out_type=str)
        decoded = sp.decode(ids)

        print(f"\n{desc}:")
        print(f"  原文: {text}")
        print(f"  Token 数: {len(ids)}")
        print(f"  Tokens: {tokens[:10]}{'...' if len(tokens) > 10 else ''}")
        print(f"  解码: {decoded}")

    print()
    print("=" * 70)

    # 对比 GPT-2 tokenizer 效率
    print("\n与 GPT-2 Tokenizer 的效率对比:")
    print("-" * 70)

    zh_sentence = "我喜欢机器翻译，它让不同语言的人能够交流。"
    en_sentence = "I love machine translation, it enables people to communicate."

    zh_tokens = len(sp.encode(zh_sentence))
    en_tokens = len(sp.encode(en_sentence))

    print(f"中文: {zh_sentence}")
    print(f"  SentencePiece: {zh_tokens} tokens")
    print(f"  GPT-2 估计: ~80 tokens")
    print(f"  效率提升: {80/zh_tokens:.1f}x")
    print()
    print(f"英文: {en_sentence}")
    print(f"  SentencePiece: {en_tokens} tokens")
    print(f"  GPT-2 估计: ~15 tokens")
    print(f"  差异: {en_tokens/15:.1f}x")
    print()


def main():
    """
    主函数：准备数据 → 训练 tokenizer → 测试
    """
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "训练自定义 SentencePiece Tokenizer" + " " * 19 + "║")
    print("║" + " " * 22 + "英中翻译任务专用" + " " * 24 + "║")
    print("╚" + "=" * 68 + "╝")
    print()

    # 1. 准备训练数据
    corpus_file, total_lines = prepare_training_data(
        data_dir="data/iwslt2017",
        output_file="data/iwslt2017/combined_corpus.txt",
        max_sentences=1000000
    )

    # 2. 训练 tokenizer
    model_path, vocab_path = train_tokenizer(
        input_file=corpus_file,
        model_prefix="tokenization/sentencepiece_enzh",
        vocab_size=16000,
        character_coverage=0.9995,
        model_type='bpe'
    )

    # 3. 测试 tokenizer
    test_tokenizer(model_path)

    # 4. 输出使用说明
    print("=" * 70)
    print("📝 后续步骤")
    print("=" * 70)
    print()
    print("1. Tokenizer 已保存到: tokenization/sentencepiece_enzh.model")
    print("2. 使用新 tokenizer 重新处理数据集:")
    print("   python scripts/retokenize_dataset.py")
    print()
    print("3. 使用新 tokenizer 训练模型:")
    print("   python scripts/train_encoder_decoder.py")
    print()
    print("4. 参数量对比:")
    print("   - GPT-2 (50k vocab):        12.9M embedding 参数")
    print("   - SentencePiece (16k vocab): 4.1M embedding 参数")
    print("   - 节省: 8.8M 参数 (68% 减少)")
    print()


if __name__ == "__main__":
    main()
