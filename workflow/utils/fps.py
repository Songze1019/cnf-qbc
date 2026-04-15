import argparse
import time
from pathlib import Path

from ase import Atoms

import numba
import numpy as np
from ase.io import read, write
from tqdm.auto import tqdm


@numba.jit(nopython=True, parallel=True, fastmath=True)
def distances_to_center_numba_parallel(x, center_features):
    n_points = x.shape[0]
    n_features = x.shape[1]
    distances = np.zeros(n_points, dtype=np.float32)

    for i in numba.prange(n_points):
        d = 0.0
        for k in range(n_features):
            tmp = x[i, k] - center_features[k]
            d += tmp * tmp
        distances[i] = np.sqrt(d)

    return distances


class CoreSetSelectorMemoryEfficient:
    def __init__(self, features: np.ndarray):
        self.features = features.astype(np.float32)
        self.n_points = int(self.features.shape[0])
        print(f"Initialized selector with {self.n_points} points.")

    def select_greedy_batch(self, k: int, s0=None) -> np.ndarray:
        if s0 is None:
            s0 = []
        if k <= 0:
            raise ValueError("k must be positive")
        if k > self.n_points:
            raise ValueError(f"k={k} is larger than number of points={self.n_points}")

        print(f"Running K-Center-Greedy to select {k} centers...")
        start_time = time.time()

        min_distances = np.full(self.n_points, np.inf, dtype=np.float32)
        selected_indices = list(s0)

        if not selected_indices:
            selected_indices.append(int(np.random.randint(self.n_points)))

        for idx in selected_indices:
            center_features = self.features[idx]
            dists = distances_to_center_numba_parallel(self.features, center_features)
            min_distances = np.minimum(min_distances, dists)

        progress = tqdm(
            total=k - len(selected_indices),
            desc="FPS selection",
            unit="center",
        )
        while len(selected_indices) < k:
            next_center_index = int(np.argmax(min_distances))
            selected_indices.append(next_center_index)

            center_features = self.features[next_center_index]
            dists_to_new_center = distances_to_center_numba_parallel(
                self.features, center_features
            )
            min_distances = np.minimum(min_distances, dists_to_new_center)
            progress.update(1)
        progress.close()

        elapsed = time.time() - start_time
        print(f"Greedy selection completed in {elapsed:.2f} s.")
        print(f"SELECTION_WALL_TIME_SECONDS={elapsed:.6f}")
        return np.asarray(selected_indices, dtype=np.int64)

    def select_batch(self, k: int, s0=None) -> np.ndarray:
        return self.select_greedy_batch(k, s0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FPS in extracted MACE feature space"
    )
    parser.add_argument(
        "--feature_file",
        type=str,
        default="workflow/utils/matpes_mace_hidden_features.npz",
        help="Feature NPZ file from workflow/utils/extract.py",
    )
    parser.add_argument(
        "--feature_key",
        type=str,
        default="structure_features",
        help="Key in NPZ for structure-level features",
    )
    parser.add_argument(
        "--input_structures",
        type=str,
        default="data/matpes/MatPES-PBE-2025.1.extxyz",
        help="Input structure trajectory (EXTXYZ)",
    )
    parser.add_argument(
        "--k", type=int, default=100, help="Number of selected structures"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for first center"
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="workflow/utils/matpes_k_center_sampling",
        help="Output file prefix",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    feature_path = Path(args.feature_file)
    structure_path = Path(args.input_structures)
    output_prefix = Path(args.output_prefix)

    if not feature_path.exists():
        raise FileNotFoundError(f"Feature file not found: {feature_path}")
    if not structure_path.exists():
        raise FileNotFoundError(f"Input structures not found: {structure_path}")

    with np.load(feature_path, allow_pickle=True) as data:
        if args.feature_key not in data:
            raise KeyError(
                f"feature_key '{args.feature_key}' not found in {feature_path}. "
                f"Available keys: {list(data.keys())}"
            )
        features = data[args.feature_key]

    if features.ndim != 2:
        raise ValueError(
            f"Expected 2D features, got shape={features.shape} from key={args.feature_key}"
        )

    raw_images = read(structure_path, index=":")
    images: list[Atoms] = []
    if isinstance(raw_images, Atoms):
        images = [raw_images]
    else:
        for item in raw_images:
            if not isinstance(item, Atoms):
                raise TypeError("Expected trajectory entries to be ASE Atoms objects")
            images.append(item)
    if len(images) != features.shape[0]:
        raise ValueError(
            "Number of structures does not match number of feature vectors: "
            f"{len(images)} vs {features.shape[0]}"
        )

    selector = CoreSetSelectorMemoryEfficient(features)
    selected_indices = selector.select_batch(k=args.k)

    selected_mask = np.zeros(features.shape[0], dtype=bool)
    selected_mask[selected_indices] = True
    selected_structures = [img for i, img in enumerate(images) if selected_mask[i]]
    selected_features = features[selected_indices]

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    np.save(f"{output_prefix}_{args.k}_indices.npy", selected_indices)
    np.save(f"{output_prefix}_{args.k}_features.npy", selected_features)
    write(f"{output_prefix}_{args.k}.extxyz", selected_structures)

    print(f"Saved indices: {output_prefix}_{args.k}_indices.npy")
    print(f"Saved selected features: {output_prefix}_{args.k}_features.npy")
    print(f"Saved selected structures: {output_prefix}_{args.k}.extxyz")


if __name__ == "__main__":
    main()
