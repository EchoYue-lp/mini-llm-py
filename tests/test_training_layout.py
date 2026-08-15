import json

from evaluation import validate_tool_router_data
from scripts import prepare_tool_router_data
from utils.project_paths import (
    MLX_MODEL_DIR,
    MLX_MODEL_MANIFEST,
    PROJECT_ROOT,
    TOOL_ROUTER_DATA_DIR,
    TOOL_ROUTER_RESULTS_DIR,
    TOOL_ROUTER_SHORT_ADAPTER_DIR,
)


def test_training_artifacts_live_outside_source_packages():
    assert MLX_MODEL_DIR == PROJECT_ROOT / "artifacts/models/Qwen3-0.6B"
    assert MLX_MODEL_MANIFEST == MLX_MODEL_DIR / "mini_llm_download_manifest.json"
    assert TOOL_ROUTER_DATA_DIR == PROJECT_ROOT / "data/tool_router"
    assert TOOL_ROUTER_SHORT_ADAPTER_DIR == (
        PROJECT_ROOT / "artifacts/adapters/tool-router-short"
    )
    assert TOOL_ROUTER_RESULTS_DIR == (
        PROJECT_ROOT / "artifacts/results/tool-router"
    )


def test_tool_router_data_generation_and_validation(tmp_path, monkeypatch):
    monkeypatch.setattr(
        prepare_tool_router_data,
        "TOOL_ROUTER_DATA_DIR",
        tmp_path,
    )
    monkeypatch.setattr(
        validate_tool_router_data,
        "TOOL_ROUTER_DATA_DIR",
        tmp_path,
    )

    prepare_tool_router_data.main()
    validate_tool_router_data.main()

    assert len((tmp_path / "train.jsonl").read_text().splitlines()) == 34
    assert len((tmp_path / "valid.jsonl").read_text().splitlines()) == 8
    assert len((tmp_path / "test.jsonl").read_text().splitlines()) == 8

    for split in ("valid", "test"):
        rows = [
            json.loads(line)
            for line in (tmp_path / f"{split}.jsonl").read_text().splitlines()
        ]
        intents = {
            json.loads(row["messages"][-1]["content"])["intent"]
            for row in rows
        }
        actions = {
            json.loads(row["messages"][-1]["content"])["action"]
            for row in rows
        }
        assert len(intents) == 8
        assert actions == {"call_tool", "ask_clarification", "no_tool"}
