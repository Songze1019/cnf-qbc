from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write


def read_frames(path: Path) -> list[Atoms]:
    frames = read(str(path), index=":")
    if isinstance(frames, Atoms):
        return [frames]
    return list(frames)


def sample_candidate_indices(n_pool: int, candidate_k: int, seed: int) -> np.ndarray:
    if n_pool <= 0:
        raise ValueError(f"n_pool must be positive, got {n_pool}")
    if candidate_k <= 0:
        raise ValueError(f"candidate_k must be positive, got {candidate_k}")

    n_pick = min(int(candidate_k), int(n_pool))
    rng = np.random.default_rng(int(seed))
    return rng.choice(n_pool, size=n_pick, replace=False).astype(np.int64)


def cmd_random_subset(args: argparse.Namespace) -> None:
    pool_xyz = Path(args.pool_xyz)
    candidate_xyz = Path(args.candidate_xyz)
    candidate_pool_indices = Path(args.candidate_pool_indices)

    pool_frames = read_frames(pool_xyz)
    if len(pool_frames) == 0:
        raise RuntimeError(f"No pool structures in {pool_xyz}")

    selected_indices = sample_candidate_indices(
        n_pool=len(pool_frames),
        candidate_k=int(args.candidate_k),
        seed=int(args.seed),
    )
    candidate_frames = [pool_frames[int(i)] for i in selected_indices]

    candidate_xyz.parent.mkdir(parents=True, exist_ok=True)
    write(str(candidate_xyz), candidate_frames, format="extxyz")
    np.save(candidate_pool_indices, selected_indices)

    print(
        f"Random candidate subset: candidate_k={len(candidate_frames)} "
        f"pool_size={len(pool_frames)}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Helper ops for random-subset QBC workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_random = subparsers.add_parser(
        "random-subset",
        help="Sample a random candidate subset from pool and export structures + pool indices",
    )
    p_random.add_argument("--pool_xyz", required=True)
    p_random.add_argument("--candidate_k", required=True, type=int)
    p_random.add_argument("--seed", required=True, type=int)
    p_random.add_argument("--candidate_xyz", required=True)
    p_random.add_argument("--candidate_pool_indices", required=True)
    p_random.set_defaults(func=cmd_random_subset)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
