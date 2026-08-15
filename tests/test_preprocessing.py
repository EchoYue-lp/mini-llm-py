import torch
import pytest

from scripts.preprocess import preprocess_text_file
from utils.tokenizer_utils import load_gpt2_tokenizer


class FakeTokenizer:
    eos_token_id = 99

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        table = {"a": [1], "bc": [2, 3]}
        return table[text]


def test_preprocess_keeps_eos_boundaries_and_final_chunk(tmp_path):
    source = tmp_path / "sample.txt"
    output = tmp_path / "sample.pt"
    source.write_text("a\nbc\n", encoding="utf-8")

    preprocess_text_file(
        source,
        FakeTokenizer(),
        output,
        seq_len=2,
        min_length=1,
    )

    chunks = torch.load(output, weights_only=False)
    assert chunks == [[1, 99], [2, 3], [99]]


def test_gpt2_uses_a_dedicated_padding_token():
    pytest.importorskip("transformers")
    tokenizer = load_gpt2_tokenizer("tokenization/gpt2")
    assert tokenizer.pad_token_id is not None
    assert tokenizer.pad_token_id != 0
    assert tokenizer.pad_token_id != tokenizer.eos_token_id
