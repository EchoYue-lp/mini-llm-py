from evaluation import validate_tool_router_data
from scripts import prepare_tool_router_data
from utils.project_paths import (
    MLX_MODEL_DIR,
    PROJECT_ROOT,
    TOOL_ROUTER_DATA_DIR,
    TOOL_ROUTER_RESULTS_DIR,
    TOOL_ROUTER_SHORT_ADAPTER_DIR,
)


def test_training_artifacts_live_outside_source_packages():
    assert MLX_MODEL_DIR == PROJECT_ROOT / "artifacts/models/Qwen3-0.6B"
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

    assert len((tmp_path / "train.jsonl").read_text().splitlines()) == 38
    assert len((tmp_path / "valid.jsonl").read_text().splitlines()) == 5
    assert len((tmp_path / "test.jsonl").read_text().splitlines()) == 5
