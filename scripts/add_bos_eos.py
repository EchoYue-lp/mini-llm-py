from transformers import GPT2TokenizerFast
import torch
import os

def add_bos_eos_to_file(in_file, out_file, tokenizer_dir, add_bos=True, add_eos=True):
    tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_dir)
    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.bos_token_id
    with open(in_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    new_lines = []
    for line in lines:
        ids = tokenizer.encode(line, add_special_tokens=False)
        if add_bos:
            ids = [bos_id] + ids
        if add_eos:
            ids = ids + [eos_id]
        new_lines.append(ids)
    torch.save(new_lines, out_file)
    print(f"已保存带 BOS/EOS 的 token ids 到 {out_file}")

if __name__ == "__main__":
    # 示例：为翻译任务目标序列添加 BOS/EOS
    add_bos_eos_to_file(
        in_file="data/iwslt2017/train.zh.txt",
        out_file="data/iwslt2017/train.zh_bos_eos.pt",
        tokenizer_dir="tokenization/gpt2",
        add_bos=True,
        add_eos=True
    )
