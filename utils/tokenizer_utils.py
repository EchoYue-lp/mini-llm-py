"""Tokenizer helpers shared by training, preprocessing, and generation."""

from transformers import GPT2TokenizerFast


GPT2_PAD_TOKEN = "<|pad|>"


def load_gpt2_tokenizer(tokenizer_dir="tokenization/gpt2"):
    """Load GPT-2 with a dedicated padding token.

    GPT-2 token id 0 is a real vocabulary item, so using 0 as padding silently
    masks valid text. Adding one dedicated token keeps EOS trainable and makes
    padding behavior explicit.
    """

    tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": GPT2_PAD_TOKEN})
    return tokenizer
