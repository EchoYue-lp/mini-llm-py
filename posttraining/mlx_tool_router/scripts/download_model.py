#!/usr/bin/env python3
"""Download the official Qwen3-0.6B snapshot into the project."""

from pathlib import Path

from huggingface_hub import snapshot_download


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models" / "Qwen3-0.6B"


def main() -> None:
    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id="Qwen/Qwen3-0.6B",
        local_dir=MODEL_DIR,
    )
    print(f"Model downloaded to: {path}")


if __name__ == "__main__":
    main()
