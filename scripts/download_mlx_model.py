#!/usr/bin/env python3
"""Download the official Qwen3-0.6B snapshot into the project."""

from huggingface_hub import snapshot_download

from utils.project_paths import MLX_MODEL_DIR


def main() -> None:
    MLX_MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id="Qwen/Qwen3-0.6B",
        local_dir=MLX_MODEL_DIR,
    )
    print(f"Model downloaded to: {path}")


if __name__ == "__main__":
    main()
