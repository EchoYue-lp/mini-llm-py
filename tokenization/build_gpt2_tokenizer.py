from transformers import GPT2TokenizerFast
import os

def get_tokenizer(save_dir="tokenization/gpt2"):
    os.makedirs(save_dir, exist_ok=True)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.save_pretrained(save_dir)
    print(f"GPT2 分词器已保存到 {save_dir}")
    return tokenizer

if __name__ == "__main__":
    get_tokenizer()
