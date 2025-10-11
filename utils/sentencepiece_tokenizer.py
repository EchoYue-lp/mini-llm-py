"""
SentencePiece Tokenizer 封装类
提供与 HuggingFace tokenizer 兼容的接口
"""

import sentencepiece as spm
import os


class SentencePieceTokenizer:
    """
    SentencePiece Tokenizer 封装
    兼容 GPT2TokenizerFast 的接口，便于无缝替换
    """

    def __init__(self, model_path):
        """
        初始化 tokenizer

        Args:
            model_path: SentencePiece 模型文件路径 (.model)
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Tokenizer 模型文件不存在: {model_path}")

        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        self.model_path = model_path

        # 特殊 token IDs（与训练时保持一致）
        self._pad_token_id = 0
        self._unk_token_id = 1
        self._bos_token_id = 2
        self._eos_token_id = 3

    @property
    def vocab_size(self):
        """词汇表大小"""
        return self.sp.vocab_size()

    @property
    def pad_token_id(self):
        """Padding token ID"""
        return self._pad_token_id

    @property
    def unk_token_id(self):
        """Unknown token ID"""
        return self._unk_token_id

    @property
    def bos_token_id(self):
        """Begin of sequence token ID"""
        return self._bos_token_id

    @property
    def eos_token_id(self):
        """End of sequence token ID"""
        return self._eos_token_id

    def encode(self, text, add_special_tokens=True):
        """
        将文本编码为 token IDs

        Args:
            text: 输入文本 (str)
            add_special_tokens: 是否添加特殊 token (BOS/EOS)

        Returns:
            List[int]: token ID 列表
        """
        if isinstance(text, list):
            # 如果是列表，递归编码
            return [self.encode(t, add_special_tokens=add_special_tokens) for t in text]

        ids = self.sp.encode(text)

        if add_special_tokens:
            # 添加 BOS 和 EOS
            ids = [self.bos_token_id] + ids + [self.eos_token_id]

        return ids

    def decode(self, ids, skip_special_tokens=True):
        """
        将 token IDs 解码为文本

        Args:
            ids: token ID 列表 (List[int] 或 List[List[int]])
            skip_special_tokens: 是否跳过特殊 token

        Returns:
            str 或 List[str]: 解码后的文本
        """
        if isinstance(ids, list) and len(ids) > 0 and isinstance(ids[0], list):
            # 批量解码
            return [self.decode(batch_ids, skip_special_tokens=skip_special_tokens) for batch_ids in ids]

        if skip_special_tokens:
            # 过滤特殊 token
            ids = [
                id for id in ids
                if id not in [self.pad_token_id, self.bos_token_id, self.eos_token_id, self.unk_token_id]
            ]

        return self.sp.decode(ids)

    def tokenize(self, text):
        """
        将文本转换为 token 字符串列表

        Args:
            text: 输入文本

        Returns:
            List[str]: token 列表
        """
        return self.sp.encode(text, out_type=str)

    def convert_tokens_to_ids(self, tokens):
        """
        将 token 字符串转换为 IDs

        Args:
            tokens: token 列表 (List[str])

        Returns:
            List[int]: ID 列表
        """
        if isinstance(tokens, str):
            return self.sp.piece_to_id(tokens)
        return [self.sp.piece_to_id(token) for token in tokens]

    def convert_ids_to_tokens(self, ids):
        """
        将 IDs 转换为 token 字符串

        Args:
            ids: ID 列表 (List[int])

        Returns:
            List[str]: token 列表
        """
        if isinstance(ids, int):
            return self.sp.id_to_piece(ids)
        return [self.sp.id_to_piece(id) for id in ids]

    def __call__(self, text, add_special_tokens=True, **kwargs):
        """
        简化调用接口

        Args:
            text: 输入文本
            add_special_tokens: 是否添加特殊 token

        Returns:
            dict: 包含 'input_ids' 的字典（兼容 HuggingFace）
        """
        ids = self.encode(text, add_special_tokens=add_special_tokens)
        return {'input_ids': ids}

    @classmethod
    def from_pretrained(cls, model_path):
        """
        类方法：加载预训练模型（兼容 HuggingFace 接口）

        Args:
            model_path: 模型路径（目录或 .model 文件）

        Returns:
            SentencePieceTokenizer 实例
        """
        # 如果是目录，查找 .model 文件
        if os.path.isdir(model_path):
            model_files = [f for f in os.listdir(model_path) if f.endswith('.model')]
            if not model_files:
                raise FileNotFoundError(f"目录 {model_path} 中没有找到 .model 文件")
            model_path = os.path.join(model_path, model_files[0])

        return cls(model_path)

    def save_pretrained(self, save_directory):
        """
        保存 tokenizer 到目录

        Args:
            save_directory: 保存目录
        """
        os.makedirs(save_directory, exist_ok=True)

        # 复制模型文件
        import shutil
        model_name = os.path.basename(self.model_path)
        dest_path = os.path.join(save_directory, model_name)
        shutil.copy(self.model_path, dest_path)

        # 如果有 vocab 文件也复制
        vocab_path = self.model_path.replace('.model', '.vocab')
        if os.path.exists(vocab_path):
            vocab_dest = os.path.join(save_directory, os.path.basename(vocab_path))
            shutil.copy(vocab_path, vocab_dest)

        print(f"Tokenizer 已保存到: {save_directory}")

    def get_vocab(self):
        """
        获取完整词汇表

        Returns:
            dict: {token: id}
        """
        vocab = {}
        for i in range(self.vocab_size):
            token = self.sp.id_to_piece(i)
            vocab[token] = i
        return vocab

    def __repr__(self):
        return (
            f"SentencePieceTokenizer(\n"
            f"  vocab_size={self.vocab_size},\n"
            f"  model_path='{self.model_path}',\n"
            f"  pad_token_id={self.pad_token_id},\n"
            f"  bos_token_id={self.bos_token_id},\n"
            f"  eos_token_id={self.eos_token_id}\n"
            f")"
        )


# 测试代码
if __name__ == "__main__":
    print("=" * 70)
    print("SentencePiece Tokenizer 测试")
    print("=" * 70)
    print()

    # 测试加载
    tokenizer = SentencePieceTokenizer.from_pretrained("tokenization/sentencepiece_enzh.model")
    print(tokenizer)
    print()

    # 测试编码/解码
    test_cases = [
        "Hello, how are you?",
        "你好，你好吗？",
        "I love machine translation.",
        "我喜欢机器翻译。"
    ]

    print("编码/解码测试:")
    print("-" * 70)
    for text in test_cases:
        ids = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(ids)
        print(f"原文: {text}")
        print(f"IDs: {ids}")
        print(f"Token数: {len(ids)}")
        print(f"解码: {decoded}")
        print()
