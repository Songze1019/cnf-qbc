#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download


DEFAULT_REPO_ID = "SandboxAQ/aqvolt26-dataset-subset"
DEFAULT_FILENAME = "test.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the gated AQVolt26 subset into the local data directory."
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"Hugging Face dataset repo id (default: {DEFAULT_REPO_ID})",
    )
    parser.add_argument(
        "--filename",
        default=DEFAULT_FILENAME,
        help=f"File to download from the dataset repo (default: {DEFAULT_FILENAME})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/aqvolt26"),
        help="Target directory for the downloaded file.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face read token. Falls back to HF_TOKEN/HUGGINGFACE_HUB_TOKEN.",
    )
    return parser.parse_args()


def resolve_token(cli_token: str | None) -> str | None:
    return cli_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def main() -> int:
    args = parse_args()
    token = resolve_token(args.token)
    if not token:
        print(
            "Missing Hugging Face token. Set HF_TOKEN or pass --token after accepting "
            "the AQVolt26 dataset access terms on Hugging Face.",
            file=sys.stderr,
        )
        return 2

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / args.filename

    try:
        cached_path = Path(
            hf_hub_download(
                repo_id=args.repo_id,
                filename=args.filename,
                repo_type="dataset",
                token=token,
            )
        )
    except Exception as exc:
        print(
            "AQVolt26 download failed. This dataset is gated and requires an authenticated "
            "account with accepted access terms.\n"
            f"Underlying error: {exc}",
            file=sys.stderr,
        )
        return 1

    shutil.copy2(cached_path, destination)
    print(f"Downloaded {args.repo_id}:{args.filename} -> {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
