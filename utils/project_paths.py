"""Canonical repository paths shared by root-level command modules."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MLX_MODEL_DIR = ARTIFACTS_DIR / "models" / "Qwen3-0.6B"
MLX_MODEL_MANIFEST = MLX_MODEL_DIR / "mini_llm_download_manifest.json"
TOOL_ROUTER_DATA_DIR = PROJECT_ROOT / "data" / "tool_router"
TOOL_ROUTER_SHORT_ADAPTER_DIR = (
    ARTIFACTS_DIR / "adapters" / "tool-router-short"
)
TOOL_ROUTER_LONG_ADAPTER_DIR = (
    ARTIFACTS_DIR / "adapters" / "tool-router-long"
)
TOOL_ROUTER_RESULTS_DIR = ARTIFACTS_DIR / "results" / "tool-router"


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a user path relative to the repository root."""

    value = Path(path).expanduser()
    return value if value.is_absolute() else PROJECT_ROOT / value
