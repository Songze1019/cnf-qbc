from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write

ROUNDS = (0, 4, 8, 12, 16, 20)


def read_frames(path: Path) -> list[Atoms]:
    frames = read(str(path), index=":")
    if isinstance(frames, Atoms):
        return [frames]
    return list(frames)


def round_size(round_idx: int, init_size: int, select_size: int) -> int:
    return init_size + select_size * round_idx


def create_random_rounds(
    input_xyz: Path,
    output_dir: Path,
    init_size: int,
    select_size: int,
    seed: int,
) -> None:
    frames = read_frames(input_xyz)
    n_total = len(frames)
    max_size = round_size(max(ROUNDS), init_size, select_size)
    if max_size > n_total:
        raise ValueError(
            f"Requested max random round size {max_size}, but input only has {n_total} frames"
        )

    rng = np.random.default_rng(seed)
    permutation = rng.permutation(n_total)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "random_permutation.npy", permutation.astype(np.int64))

    for round_idx in ROUNDS:
        n_keep = round_size(round_idx, init_size, select_size)
        selected_indices = permutation[:n_keep].astype(np.int64, copy=False)
        selected_frames = [frames[int(i)] for i in selected_indices]
        output_path = output_dir / f"train_round{round_idx}.xyz"
        index_path = output_dir / f"train_round{round_idx}_indices.npy"
        write(str(output_path), selected_frames, format="extxyz")
        np.save(index_path, selected_indices)
        print(f"round={round_idx} frames={n_keep} -> {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create nested random Transition1x training rounds."
    )
    parser.add_argument(
        "--input_xyz",
        type=Path,
        default=Path("data/transition1x/train.xyz"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("workflow/random"),
    )
    parser.add_argument("--init_size", type=int, default=1024)
    parser.add_argument("--select_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_random_rounds(
        input_xyz=args.input_xyz,
        output_dir=args.output_dir,
        init_size=args.init_size,
        select_size=args.select_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
