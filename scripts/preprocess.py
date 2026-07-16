import os
import argparse
from tqdm import tqdm
import torch

from utils.tokenizer_utils import load_gpt2_tokenizer

def preprocess_text_file(text_file, tokenizer, out_file, seq_len=96, min_length=10):
    # 读取所有文本
    with open(text_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    # 在文档/段落边界插入 EOS，让生成模型真正学习停止信号。
    input_ids = []
    for line in lines:
        input_ids.extend(tokenizer.encode(line, add_special_tokens=False))
        if tokenizer.eos_token_id is not None:
            input_ids.append(tokenizer.eos_token_id)
    # 按 seq_len 切分
    chunks = []
    for i in range(0, len(input_ids), seq_len):
        chunk = input_ids[i:i+seq_len]
        if len(chunk) >= min_length:
            chunks.append(chunk)
    # 保存为 pt 文件
    torch.save(chunks, out_file)
    print(f"已保存 {len(chunks)} 个样本到 {out_file}")

def preprocess_wikitext2(data_dir="data/wikitext2", tokenizer_dir="tokenization/gpt2", seq_len=96):
    # 设置 model_max_length 避免 tokenizer 警告
    tokenizer = load_gpt2_tokenizer(tokenizer_dir)
    tokenizer.model_max_length = int(1e30)
    for split in ["train", "validation", "test"]:
        text_file = os.path.join(data_dir, f"{split}.txt")
        out_file = os.path.join(data_dir, f"{split}_ids.pt")
        preprocess_text_file(text_file, tokenizer, out_file, seq_len=seq_len)

def preprocess_parallel_corpus(src_file, tgt_file, tokenizer, src_out, tgt_out, max_len=96):
    """
    预处理平行语料库，确保源和目标句对一一对应

    Args:
        src_file: 源语言文件
        tgt_file: 目标语言文件
        tokenizer: tokenizer
        src_out: 源语言输出文件
        tgt_out: 目标语言输出文件
        max_len: 最大序列长度（会过滤掉超过此长度的句对）
    """
    # 读取源和目标文本
    with open(src_file, 'r', encoding='utf-8') as f:
        src_lines = [line.strip() for line in f if line.strip()]
    with open(tgt_file, 'r', encoding='utf-8') as f:
        tgt_lines = [line.strip() for line in f if line.strip()]

    if len(src_lines) != len(tgt_lines):
        print(f"警告：源文件和目标文件的行数不匹配！src: {len(src_lines)}, tgt: {len(tgt_lines)}")
        min_len = min(len(src_lines), len(tgt_lines))
        src_lines = src_lines[:min_len]
        tgt_lines = tgt_lines[:min_len]

    src_ids_list = []
    tgt_ids_list = []

    # 逐对处理
    for src_line, tgt_line in tqdm(zip(src_lines, tgt_lines), total=len(src_lines), desc="Tokenizing"):
        src_ids = tokenizer.encode(src_line, add_special_tokens=False)
        tgt_ids = tokenizer.encode(tgt_line, add_special_tokens=False)

        # 过滤掉过长的句对
        if len(src_ids) <= max_len and len(tgt_ids) <= max_len:
            src_ids_list.append(src_ids)
            tgt_ids_list.append(tgt_ids)

    # 保存
    torch.save(src_ids_list, src_out)
    torch.save(tgt_ids_list, tgt_out)
    print(f"已保存 {len(src_ids_list)} 个平行句对")
    print(f"  源: {src_out}")
    print(f"  目标: {tgt_out}")

def preprocess_opus100_en_zh(data_dir="data/iwslt2017", tokenizer_dir="tokenization/gpt2", seq_len=96):
    # 设置 model_max_length 避免 tokenizer 警告
    tokenizer = load_gpt2_tokenizer(tokenizer_dir)
    tokenizer.model_max_length = int(1e30)
    for split in ["train", "validation", "test"]:
        src_file = os.path.join(data_dir, f"{split}.en.txt")
        tgt_file = os.path.join(data_dir, f"{split}.zh.txt")
        src_out = os.path.join(data_dir, f"{split}.en_ids.pt")
        tgt_out = os.path.join(data_dir, f"{split}.zh_ids.pt")
        print(f"\n处理 {split} 集...")
        preprocess_parallel_corpus(src_file, tgt_file, tokenizer, src_out, tgt_out, max_len=seq_len)

def main():
    parser = argparse.ArgumentParser(description="预处理生成或旧版 GPT-2 翻译数据")
    parser.add_argument("--generation", action="store_true")
    parser.add_argument("--legacy-translation", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if not (args.generation or args.legacy_translation or args.all):
        parser.print_help()
        return
    if args.generation or args.all:
        preprocess_wikitext2()
    if args.legacy_translation or args.all:
        preprocess_opus100_en_zh()


if __name__ == "__main__":
    main()
