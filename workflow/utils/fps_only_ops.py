from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def select_all_local_indices(candidate_count: int) -> np.ndarray:
    if candidate_count < 0:
        raise ValueError(f"candidate_count must be non-negative, got {candidate_count}")
    return np.arange(candidate_count, dtype=np.int64)


def cmd_select_all_candidates(args: argparse.Namespace) -> None:
    candidate_pool_indices_npy = Path(args.candidate_pool_indices_npy)
    selected_local_out = Path(args.selected_local_out)

    candidate_pool_indices = np.load(candidate_pool_indices_npy).astype(
        np.int64, copy=False
    )
    if candidate_pool_indices.ndim != 1:
        raise ValueError("candidate_pool_indices must be a 1D array")

    selected_local = select_all_local_indices(candidate_pool_indices.shape[0])
    selected_local_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(selected_local_out, selected_local)
    print(f"Selected all candidates: count={selected_local.shape[0]}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Helper ops for FPS-only workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_select_all = subparsers.add_parser(
        "select-all-candidates",
        help="Create selected_local indices that keep all FPS-selected candidates",
    )
    p_select_all.add_argument("--candidate_pool_indices_npy", required=True)
    p_select_all.add_argument("--selected_local_out", required=True)
    p_select_all.set_defaults(func=cmd_select_all_candidates)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
