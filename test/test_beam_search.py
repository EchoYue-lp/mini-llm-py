"""
测试 beam search 翻译功能
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from transformers import GPT2TokenizerFast
from utils.translation_utils import beam_search_translate, greedy_translate

# 简单测试
tokenizer = GPT2TokenizerFast.from_pretrained("tokenization/gpt2")

# 测试源句子
src = "Hello world"
src_ids = tokenizer.encode(src, add_special_tokens=False)

print(f"源句子: {src}")
print(f"源 token ids: {src_ids}")
print(f"BOS token id: {tokenizer.bos_token_id}")
print(f"EOS token id: {tokenizer.eos_token_id}")
print(f"PAD token id: {tokenizer.pad_token_id}")

# 检查问题
bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
print(f"\n使用的 BOS id: {bos_id}")

# 初始 beam
initial_seq = torch.tensor([bos_id], dtype=torch.long)
print(f"初始序列: {initial_seq.tolist()}")
print(f"初始序列最后一个 token: {initial_seq[-1].item()}")
print(f"是否等于 EOS? {initial_seq[-1].item() == tokenizer.eos_token_id}")

# 问题诊断
if initial_seq[-1].item() == tokenizer.eos_token_id:
    print("\n❌ 问题找到了！")
    print("   BOS token 和 EOS token 相同，导致第一步就认为序列完成了！")
    print("   GPT2 tokenizer 没有 BOS token，需要特殊处理")
