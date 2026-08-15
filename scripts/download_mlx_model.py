#!/usr/bin/env python3
"""Download a recorded Qwen3-0.6B revision into the project."""

import argparse
import json
from importlib.metadata import version

from huggingface_hub import HfApi, snapshot_download

from utils.project_paths import MLX_MODEL_DIR, MLX_MODEL_MANIFEST


REPO_ID = "Qwen/Qwen3-0.6B"
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--revision",
        help="Immutable commit hash or tag; defaults to the recorded local revision",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    MLX_MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    recorded_revision = None
    if MLX_MODEL_MANIFEST.exists() and args.revision is None:
        recorded_revision = json.loads(
            MLX_MODEL_MANIFEST.read_text(encoding="utf-8")
        )["resolved_revision"]
    requested_revision = args.revision or recorded_revision or "main"
    resolved_revision = HfApi().model_info(
        REPO_ID,
        revision=requested_revision,
    ).sha
    path = snapshot_download(
        repo_id=REPO_ID,
        local_dir=MLX_MODEL_DIR,
        revision=resolved_revision,
    )
    MLX_MODEL_MANIFEST.write_text(
        json.dumps(
            {
                "repo_id": REPO_ID,
                "requested_revision": requested_revision,
                "resolved_revision": resolved_revision,
                "huggingface_hub": version("huggingface-hub"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Model downloaded to: {path}")
    print(f"Pinned revision: {resolved_revision}")


if __name__ == "__main__":
    main()
