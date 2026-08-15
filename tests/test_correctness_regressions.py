import torch
import pytest

from evaluation import compare_tool_router_models
from evaluation.tool_router_metrics import (
    extract_json,
    is_schema_valid,
    parse_raw_json,
)
from models.layers import PositionalEncoding
from models.transformer_models import DecoderOnlyModel, EncoderDecoderModel
from scripts.retokenize_dataset import retokenize_parallel_files
from scripts.train_encoder_decoder import collate_fn_with_padding
from utils.checkpoint_utils import load_model_from_checkpoint
from utils.generation_utils import top_k_sampling
from utils.scheduler_utils import WarmupLRScheduler


class TinyTokenizer:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2

    def encode(self, text, add_special_tokens=True):
        ids = [3 + len(piece) for piece in text.split()]
        return [self.bos_token_id, *ids, self.eos_token_id] if add_special_tokens else ids

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(item) for item in ids)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_positional_encoding_preserves_input_dtype(dtype):
    encoding = PositionalEncoding(8, max_len=16, dropout=0.0)
    result = encoding(torch.zeros((1, 4, 8), dtype=dtype))
    assert result.dtype == dtype


def test_decoder_default_mask_blocks_future_tokens():
    torch.manual_seed(3)
    model = DecoderOnlyModel(32, 16, 1, 4, 32, dropout=0.0).eval()
    first = torch.tensor([[4, 5, 6, 7]])
    changed_future = torch.tensor([[4, 5, 20, 21]])

    with torch.inference_mode():
        first_logits, _ = model(first)
        changed_logits, _ = model(changed_future)
        explicit_logits, _ = model(
            changed_future,
            mask=torch.ones((1, 1, 4, 4), dtype=torch.bool),
        )

    torch.testing.assert_close(first_logits[:, :2], changed_logits[:, :2])
    torch.testing.assert_close(first_logits[:, :2], explicit_logits[:, :2])


def test_encoder_decoder_default_target_mask_blocks_future_tokens():
    torch.manual_seed(5)
    model = EncoderDecoderModel(32, 32, 16, 1, 4, 32, dropout=0.0).eval()
    source = torch.tensor([[8, 9, 10]])
    first = torch.tensor([[1, 4, 5, 6]])
    changed_future = torch.tensor([[1, 4, 20, 21]])

    with torch.inference_mode():
        first_logits, _ = model(source, first)
        changed_logits, _ = model(source, changed_future)
        explicit_logits, _ = model(
            source,
            changed_future,
            tgt_mask=torch.ones((1, 1, 4, 4), dtype=torch.bool),
        )

    torch.testing.assert_close(first_logits[:, :2], changed_logits[:, :2])
    torch.testing.assert_close(first_logits[:, :2], explicit_logits[:, :2])


def test_target_truncation_preserves_eos():
    _, target = collate_fn_with_padding(
        [[8, 9]],
        [[1, 10, 11, 12, 2]],
        pad_token_id=0,
        max_seq_len=4,
        tgt_eos_token_id=2,
    )
    assert target.tolist() == [[1, 10, 11, 2]]


def test_parallel_tokenization_rejects_one_sided_blank_line(tmp_path):
    source = tmp_path / "sample.en.txt"
    target = tmp_path / "sample.zh.txt"
    source.write_text("hello world\n\nlast row\n", encoding="utf-8")
    target.write_text("你好 世界\n不应出现\n最后 一行\n", encoding="utf-8")

    with pytest.raises(ValueError, match="只有一侧为空"):
        retokenize_parallel_files(
            source,
            target,
            tmp_path / "source.pt",
            tmp_path / "target.pt",
            TinyTokenizer(),
        )


def test_parallel_tokenization_preserves_target_eos(tmp_path):
    source = tmp_path / "sample.en.txt"
    target = tmp_path / "sample.zh.txt"
    source.write_text("one two three four five\n", encoding="utf-8")
    target.write_text("一 二 三 四 五\n", encoding="utf-8")
    _, target_rows = retokenize_parallel_files(
        source,
        target,
        tmp_path / "source.pt",
        tmp_path / "target.pt",
        TinyTokenizer(),
        max_len=4,
    )
    assert target_rows[0][-1] == TinyTokenizer.eos_token_id


def test_cosine_schedule_stays_at_zero_after_training_horizon():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    scheduler = WarmupLRScheduler(
        optimizer,
        scheduler_type="cosine",
        num_training_steps=4,
        num_warmup_steps=0,
    )
    learning_rates = []
    for _ in range(8):
        optimizer.step()
        scheduler.step()
        learning_rates.append(optimizer.param_groups[0]["lr"])
    assert learning_rates[3:] == [0.0] * 5


def test_weight_only_checkpoint_requires_explicit_num_heads(tmp_path):
    model = DecoderOnlyModel(32, 16, 1, 4, 32, max_len=8, dropout=0.0)
    checkpoint = tmp_path / "weights.pt"
    torch.save(model.state_dict(), checkpoint)

    with pytest.raises(ValueError, match="num_heads"):
        load_model_from_checkpoint(checkpoint, DecoderOnlyModel)

    restored, _ = load_model_from_checkpoint(
        checkpoint,
        DecoderOnlyModel,
        num_heads=4,
        dropout=0.0,
    )
    assert restored.layers[0].self_attn.num_heads == 4


def test_generation_validates_context_and_restores_training_mode():
    model = DecoderOnlyModel(16, 16, 1, 4, 32, max_len=4, dropout=0.0)
    tokenizer = TinyTokenizer()
    model.train()

    with pytest.raises(ValueError, match="prompt plus generated"):
        top_k_sampling(model, [3, 4, 5], tokenizer, k=1, max_len=2)
    assert model.training

    top_k_sampling(model, [3, 4], tokenizer, k=1, max_len=1)
    assert model.training


def test_tool_router_metrics_distinguish_raw_extractable_and_schema_valid():
    valid = (
        '{"action":"no_tool","intent":"chitchat","tool":null,'
        '"arguments":{},"missing_arguments":[]}'
    )
    wrapped = f"结果如下：{valid}"
    invalid_schema = '{"action":"no_tool","intent":"made_up"}'

    assert parse_raw_json(valid) is not None
    assert parse_raw_json(wrapped) is None
    assert extract_json(wrapped) is not None
    assert is_schema_valid(extract_json(wrapped))
    assert not is_schema_valid(parse_raw_json(invalid_schema))


def test_comparison_conclusions_are_derived_from_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(
        compare_tool_router_models,
        "TOOL_ROUTER_RESULTS_DIR",
        tmp_path,
    )
    expected = {
        "action": "no_tool",
        "intent": "chitchat",
        "tool": None,
        "arguments": {},
        "missing_arguments": [],
    }
    metrics = {key: 1.0 for key in compare_tool_router_models.METRIC_NAMES}
    reports = []
    for label, exact_match in (("base", 0.0), ("measured-winner", 0.75)):
        report_metrics = dict(metrics, exact_match=exact_match)
        reports.append(
            {
                "label": label,
                "metrics": report_metrics,
                "predictions": [
                    {
                        "input": "你好",
                        "expected": expected,
                        "actual": expected,
                        "raw": "{}",
                    }
                ],
            }
        )

    output = compare_tool_router_models.write_markdown(reports)
    text = output.read_text(encoding="utf-8")
    assert "measured-winner（75%）" in text
    assert "长训练的整条正确率更高" not in text
