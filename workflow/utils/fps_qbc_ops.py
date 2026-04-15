from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence, cast

import numpy as np
from ase import Atoms
from ase.io import read, write
from tqdm.auto import tqdm


def read_frames(path: Path) -> list[Atoms]:
    frames = read(str(path), index=":")
    if isinstance(frames, Atoms):
        return [frames]
    return list(frames)


def load_feature_matrix(path: Path, feature_key: str) -> np.ndarray:
    with np.load(path, allow_pickle=True) as data:
        if feature_key not in data:
            raise KeyError(
                f"feature_key '{feature_key}' not found in {path}. "
                f"Available keys: {list(data.keys())}"
            )
        features = np.asarray(data[feature_key])

    if features.ndim != 2:
        raise ValueError(
            f"Expected 2D features from key={feature_key}, got shape={features.shape}"
        )
    return features


def save_feature_matrix(path: Path, feature_key: str, features: np.ndarray) -> None:
    if features.ndim != 2:
        raise ValueError(f"Expected 2D features to save, got shape={features.shape}")

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{feature_key: features})


def normalize_feature_matrices(
    train_features: np.ndarray, pool_features: np.ndarray, mode: str
) -> tuple[np.ndarray, np.ndarray]:
    if mode == "none":
        return train_features, pool_features

    if mode != "zscore":
        raise ValueError(f"Unknown FPS normalization mode: {mode}")

    if train_features.ndim != 2 or pool_features.ndim != 2:
        raise ValueError(
            "Expected 2D feature matrices for FPS normalization, "
            f"got train={train_features.shape}, pool={pool_features.shape}"
        )

    combined = np.concatenate([train_features, pool_features], axis=0)
    mean = combined.mean(axis=0, keepdims=True)
    std = combined.std(axis=0, keepdims=True)
    safe_std = np.where(std > 1e-12, std, 1.0)

    normalized_train = (train_features - mean) / safe_std
    normalized_pool = (pool_features - mean) / safe_std
    return normalized_train, normalized_pool


def parse_enable_cueq(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Invalid boolean value for enable_cueq: {value}")


def build_calculators(
    model_paths: Sequence[str], device: str, enable_cueq_raw: str
) -> list:
    from mace.calculators import MACECalculator

    enable_cueq = parse_enable_cueq(enable_cueq_raw)
    return [
        MACECalculator(model_paths=model_path, device=device, enable_cueq=enable_cueq)
        for model_path in model_paths
    ]


def _build_atomic_data_for_frames(frames: list[Atoms], calculator) -> list:
    from mace import data as mace_data

    key_specification = mace_data.KeySpecification(
        info_keys=calculator.info_keys,
        arrays_keys=calculator.arrays_keys,
    )
    head_name = (
        calculator.head[0] if isinstance(calculator.head, list) else calculator.head
    )

    atomic_dataset = []
    for atoms in frames:
        at = atoms.copy()
        at.info.setdefault("charge", 0.0)
        at.info.setdefault("spin", 0.0)

        config = mace_data.config_from_atoms(
            at,
            key_specification=key_specification,
            head_name=head_name,
        )
        atomic_data = mace_data.AtomicData.from_config(
            config,
            z_table=calculator.z_table,
            cutoff=calculator.r_max,
            heads=calculator.available_heads,
        )
        atomic_dataset.append(atomic_data)

    return atomic_dataset


def _predict_committee_batch(
    frames: list[Atoms],
    calculators: list,
    device: str,
    batch_size: int,
    progress_desc: str | None = None,
) -> tuple[np.ndarray, list[list[np.ndarray]]]:
    from mace.tools import torch_geometric

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    n_models = len(calculators)
    n_frames = len(frames)
    if n_frames == 0:
        return np.zeros((n_models, 0), dtype=float), [[] for _ in range(n_models)]

    atomic_dataset = _build_atomic_data_for_frames(frames, calculators[0])
    loader = torch_geometric.dataloader.DataLoader(
        dataset=cast(Any, atomic_dataset),
        batch_size=min(batch_size, n_frames),
        shuffle=False,
        drop_last=False,
    )

    models = []
    for calculator in calculators:
        model = cast(Any, calculator.models[0])
        model.eval()
        models.append(model)

    energy_predictions = np.zeros((n_models, n_frames), dtype=float)
    force_predictions: list[list[np.ndarray]] = [
        [np.zeros((0, 3), dtype=float) for _ in range(n_frames)]
        for _ in range(n_models)
    ]

    frame_offset = 0
    pbar = (
        tqdm(total=n_frames, desc=progress_desc, unit="structure")
        if progress_desc
        else None
    )

    for batch in loader:
        batch = batch.to(device)
        ptr = batch.ptr.detach().cpu().numpy().astype(np.int64, copy=False)
        batch_size_now = int(ptr.shape[0] - 1)

        for model_idx, model in enumerate(models):
            output = model(batch.to_dict(), compute_force=True)

            energies = (
                output["energy"]
                .detach()
                .cpu()
                .numpy()
                .astype(float, copy=False)
                .reshape(-1)
            )
            forces = output["forces"].detach().cpu().numpy().astype(float, copy=False)

            if energies.shape[0] != batch_size_now:
                raise RuntimeError(
                    "Batched energy count mismatch: "
                    f"got {energies.shape[0]}, expected {batch_size_now}"
                )

            energy_predictions[
                model_idx, frame_offset : frame_offset + batch_size_now
            ] = energies

            for local_idx in range(batch_size_now):
                start = int(ptr[local_idx])
                end = int(ptr[local_idx + 1])
                force_predictions[model_idx][frame_offset + local_idx] = forces[
                    start:end
                ]

        frame_offset += batch_size_now
        if pbar is not None:
            pbar.update(batch_size_now)

    if pbar is not None:
        pbar.close()

    if frame_offset != n_frames:
        raise RuntimeError(
            f"Prediction count mismatch: got {frame_offset}, expected {n_frames}"
        )

    return energy_predictions, force_predictions


def _compute_uncertainty_from_predictions(
    energy_predictions: np.ndarray,
    force_predictions: list[list[np.ndarray]],
) -> dict[str, np.ndarray]:
    n_models, n_frames = energy_predictions.shape
    mean_force_std = np.zeros(n_frames, dtype=float)
    p95_force_std = np.zeros(n_frames, dtype=float)
    max_force_std = np.zeros(n_frames, dtype=float)
    energy_std = np.zeros(n_frames, dtype=float)

    for frame_idx in range(n_frames):
        force_ensemble = np.asarray(
            [force_predictions[model_idx][frame_idx] for model_idx in range(n_models)],
            dtype=float,
        )
        force_std = np.std(force_ensemble, axis=0)
        force_std_norm = np.linalg.norm(force_std, axis=1)

        mean_force_std[frame_idx] = float(force_std_norm.mean())
        p95_force_std[frame_idx] = float(np.quantile(force_std_norm, 0.95))
        max_force_std[frame_idx] = float(force_std_norm.max())
        energy_std[frame_idx] = float(np.std(energy_predictions[:, frame_idx]))

    return {
        "mean_force_std": mean_force_std,
        "p95_force_std": p95_force_std,
        "max_force_std": max_force_std,
        "energy_std": energy_std,
    }


def compute_committee_uncertainty_arrays(
    frames: list[Atoms],
    calculators: list,
    device: str,
    batch_size: int,
    progress_desc: str | None = None,
) -> dict[str, np.ndarray]:
    energy_predictions, force_predictions = _predict_committee_batch(
        frames=frames,
        calculators=calculators,
        device=device,
        batch_size=batch_size,
        progress_desc=progress_desc,
    )
    return _compute_uncertainty_from_predictions(energy_predictions, force_predictions)


def select_metric_scores(metric: str, arrays: dict[str, np.ndarray]) -> np.ndarray:
    if metric == "force_std_p95":
        return arrays["p95_force_std"]
    if metric == "force_std_mean":
        return arrays["mean_force_std"]
    if metric == "energy_std_abs":
        return arrays["energy_std"]
    raise ValueError(f"Unknown metric: {metric}")


def get_reference_energy(atoms: Atoms) -> float:
    for key in ("REF_energy", "energy"):
        if key in atoms.info:
            return float(atoms.info[key])
    raise KeyError(
        "Missing reference energy in atoms.info (expected REF_energy or energy)"
    )


def get_reference_forces(atoms: Atoms) -> np.ndarray:
    for key in ("REF_forces", "forces"):
        if key in atoms.arrays:
            return np.asarray(atoms.arrays[key], dtype=float)
    raise KeyError(
        "Missing reference forces in atoms.arrays (expected REF_forces or forces)"
    )


def cmd_init_split(args: argparse.Namespace) -> None:
    input_xyz = Path(args.input_xyz)
    train_out = Path(args.train_out)
    pool_out = Path(args.pool_out)

    frames = read_frames(input_xyz)
    n_total = len(frames)
    if n_total == 0:
        raise RuntimeError(f"No structures in {input_xyz}")

    init_size = int(args.init_size)
    if not (0 < init_size < n_total):
        raise RuntimeError(
            f"init_size must satisfy 0 < init_size < n_total ({n_total}), got {init_size}"
        )

    rng = np.random.default_rng(int(args.seed))
    perm = rng.permutation(n_total)
    train_idx = set(int(i) for i in perm[:init_size])

    train_frames = [at for i, at in enumerate(frames) if i in train_idx]
    pool_frames = [at for i, at in enumerate(frames) if i not in train_idx]

    train_out.parent.mkdir(parents=True, exist_ok=True)
    write(str(train_out), train_frames, format="extxyz")
    write(str(pool_out), pool_frames, format="extxyz")

    print(f"Initialized train={len(train_frames)} pool={len(pool_frames)}")


def cmd_count_frames(args: argparse.Namespace) -> None:
    n = len(read_frames(Path(args.input_xyz)))
    print(n)


def cmd_candidate_k(args: argparse.Namespace) -> None:
    n_pool = int(args.pool_size)
    pct = float(args.candidate_pct)
    frac = pct / 100.0 if pct > 1.0 else pct
    frac = min(max(frac, 0.0), 1.0)
    k = int(math.floor(n_pool * frac))
    k = max(1, k)
    k = min(n_pool, k)
    print(k)


def cmd_committee_uncertainty(args: argparse.Namespace) -> None:
    candidate_xyz = Path(args.candidate_xyz)
    select_local_out = Path(args.selected_local_out)
    stats_out = Path(args.stats_out)

    metric = str(args.metric)
    if metric not in {"force_std_p95", "force_std_mean", "energy_std_abs"}:
        raise ValueError(f"Unknown metric: {metric}")

    candidate_frames = read_frames(candidate_xyz)
    if len(candidate_frames) == 0:
        raise RuntimeError(f"No candidate structures in {candidate_xyz}")

    calculators = build_calculators(args.model_paths, args.device, args.enable_cueq)
    arrays = compute_committee_uncertainty_arrays(
        candidate_frames,
        calculators,
        device=args.device,
        batch_size=int(args.batch_size),
        progress_desc=f"candidate uncertainty ({metric})",
    )
    scores = select_metric_scores(metric, arrays)

    n = len(candidate_frames)
    n_pick = min(int(args.select_size), n)
    rank_desc = np.argsort(scores)[::-1]
    selected_local = rank_desc[:n_pick].astype(np.int64)

    select_local_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(select_local_out, selected_local)
    np.save(
        stats_out,
        {
            "scores": scores,
            "mean_force_std": arrays["mean_force_std"],
            "p95_force_std": arrays["p95_force_std"],
            "max_force_std": arrays["max_force_std"],
            "energy_std": arrays["energy_std"],
            "metric": np.array(metric),
            "selected_local": selected_local,
        },
        allow_pickle=True,
    )

    print(f"{float(scores.max()):.12f}")


def cmd_pool_mean_uncertainty(args: argparse.Namespace) -> None:
    metric = str(args.metric)
    if metric not in {"force_std_p95", "force_std_mean", "energy_std_abs"}:
        raise ValueError(f"Unknown metric: {metric}")

    pool_frames = read_frames(Path(args.pool_xyz))
    if len(pool_frames) == 0:
        print("nan")
        return

    calculators = build_calculators(args.model_paths, args.device, args.enable_cueq)
    arrays = compute_committee_uncertainty_arrays(
        pool_frames,
        calculators,
        device=args.device,
        batch_size=int(args.batch_size),
        progress_desc=f"pool uncertainty ({metric})",
    )
    scores = select_metric_scores(metric, arrays)
    print(f"{float(np.mean(scores)):.12f}")


def cmd_test_uncertainty(args: argparse.Namespace) -> None:
    metric = str(args.metric)
    if metric not in {"force_std_p95", "force_std_mean", "energy_std_abs"}:
        raise ValueError(f"Unknown metric: {metric}")

    test_xyz = Path(args.test_xyz)
    stats_out = Path(args.stats_out)

    test_frames = read_frames(test_xyz)
    if len(test_frames) == 0:
        stats_out.parent.mkdir(parents=True, exist_ok=True)
        np.save(
            stats_out,
            {
                "scores": np.zeros(0, dtype=float),
                "mean_force_std": np.zeros(0, dtype=float),
                "p95_force_std": np.zeros(0, dtype=float),
                "max_force_std": np.zeros(0, dtype=float),
                "energy_std": np.zeros(0, dtype=float),
                "metric": np.array(metric),
                "mean_score": np.array(float("nan")),
            },
            allow_pickle=True,
        )
        print("nan")
        return

    calculators = build_calculators(args.model_paths, args.device, args.enable_cueq)
    arrays = compute_committee_uncertainty_arrays(
        test_frames,
        calculators,
        device=args.device,
        batch_size=int(args.batch_size),
        progress_desc=f"test uncertainty ({metric})",
    )
    scores = select_metric_scores(metric, arrays)
    mean_score = float(np.mean(scores))

    stats_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(
        stats_out,
        {
            "scores": scores,
            "mean_force_std": arrays["mean_force_std"],
            "p95_force_std": arrays["p95_force_std"],
            "max_force_std": arrays["max_force_std"],
            "energy_std": arrays["energy_std"],
            "metric": np.array(metric),
            "mean_score": np.array(mean_score),
        },
        allow_pickle=True,
    )
    print(f"{mean_score:.12f}")


def cmd_test_rmse(args: argparse.Namespace) -> None:
    test_frames = read_frames(Path(args.test_xyz))
    if len(test_frames) == 0:
        print("nan\tnan\tnan\tnan")
        return

    calculators = build_calculators(args.model_paths, args.device, args.enable_cueq)
    energy_predictions, force_predictions = _predict_committee_batch(
        frames=test_frames,
        calculators=calculators,
        device=args.device,
        batch_size=int(args.batch_size),
        progress_desc="test rmse/l4 inference",
    )
    n_models = int(energy_predictions.shape[0])

    ref_energies = np.asarray(
        [get_reference_energy(atoms) for atoms in test_frames], dtype=float
    )
    ref_forces = [get_reference_forces(atoms) for atoms in test_frames]

    energy_rmse_list = []
    force_rmse_list = []
    energy_l4_list = []
    force_l4_list = []
    for model_idx in range(n_models):
        energy_diffs = energy_predictions[model_idx] - ref_energies
        force_diff_squares_sum = 0.0
        force_diff_fourth_sum = 0.0
        force_comp_count = 0

        for frame_idx, ref_force in enumerate(ref_forces):
            pred_forces = np.asarray(
                force_predictions[model_idx][frame_idx], dtype=float
            )
            diff = pred_forces - ref_force
            force_diff_squares_sum += float(np.sum(diff * diff))
            force_diff_fourth_sum += float(np.sum(np.abs(diff) ** 4))
            force_comp_count += int(diff.size)

        e_rmse = float(np.sqrt(np.mean(energy_diffs * energy_diffs)))
        e_l4 = float(np.power(np.mean(np.abs(energy_diffs) ** 4), 0.25))
        f_rmse = float(np.sqrt(force_diff_squares_sum / force_comp_count))
        f_l4 = float(np.power(force_diff_fourth_sum / force_comp_count, 0.25))
        energy_rmse_list.append(e_rmse)
        force_rmse_list.append(f_rmse)
        energy_l4_list.append(e_l4)
        force_l4_list.append(f_l4)

    print(
        f"{float(np.mean(energy_rmse_list)):.12f}\t"
        f"{float(np.mean(force_rmse_list)):.12f}\t"
        f"{float(np.mean(energy_l4_list)):.12f}\t"
        f"{float(np.mean(force_l4_list)):.12f}"
    )


def cmd_append_round_metrics(args: argparse.Namespace) -> None:
    metrics_path = Path(args.metrics_file)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "round",
        "candidate_k",
        "selected_size",
        "pool_size_after_select",
        "uncertainty_metric",
        "max_candidate_uncertainty",
        "test_mean_uncertainty",
        "test_energy_rmse",
        "test_force_rmse",
        "test_energy_l4",
        "test_force_l4",
    ]
    row = [
        str(args.round),
        str(args.candidate_k),
        str(args.selected_size),
        str(args.pool_size_after_select),
        str(args.uncertainty_metric),
        str(args.max_candidate_uncertainty),
        str(args.test_mean_uncertainty),
        str(args.test_energy_rmse),
        str(args.test_force_rmse),
        str(args.test_energy_l4),
        str(args.test_force_l4),
    ]

    write_header = not metrics_path.exists() or metrics_path.stat().st_size == 0
    with metrics_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)


def cmd_update_datasets(args: argparse.Namespace) -> None:
    train_xyz = Path(args.train_xyz)
    pool_xyz = Path(args.pool_xyz)
    candidate_pool_indices_npy = Path(args.candidate_pool_indices_npy)
    selected_local_npy = Path(args.selected_local_npy)
    next_train_xyz = Path(args.next_train_xyz)
    next_pool_xyz = Path(args.next_pool_xyz)

    train_frames = read_frames(train_xyz)
    pool_frames = read_frames(pool_xyz)

    candidate_pool_indices = np.load(candidate_pool_indices_npy)
    selected_local = np.load(selected_local_npy)

    selected_pool_indices = candidate_pool_indices[selected_local].astype(np.int64)
    selected_pool_set = set(int(i) for i in selected_pool_indices.tolist())

    new_train = train_frames + [
        at for i, at in enumerate(pool_frames) if i in selected_pool_set
    ]
    new_pool = [at for i, at in enumerate(pool_frames) if i not in selected_pool_set]

    next_train_xyz.parent.mkdir(parents=True, exist_ok=True)
    write(str(next_train_xyz), new_train, format="extxyz")
    write(str(next_pool_xyz), new_pool, format="extxyz")

    print(f"Updated datasets: train={len(new_train)} pool={len(new_pool)}")


def cmd_update_feature_sets(args: argparse.Namespace) -> None:
    train_feature_file = Path(args.train_feature_file)
    pool_feature_file = Path(args.pool_feature_file)
    candidate_pool_indices_npy = Path(args.candidate_pool_indices_npy)
    selected_local_npy = Path(args.selected_local_npy)
    next_train_feature_file = Path(args.next_train_feature_file)
    next_pool_feature_file = Path(args.next_pool_feature_file)
    feature_key = str(args.feature_key)

    train_features = load_feature_matrix(train_feature_file, feature_key)
    pool_features = load_feature_matrix(pool_feature_file, feature_key)

    candidate_pool_indices = np.load(candidate_pool_indices_npy).astype(
        np.int64, copy=False
    )
    selected_local = np.load(selected_local_npy).astype(np.int64, copy=False)

    if candidate_pool_indices.ndim != 1 or selected_local.ndim != 1:
        raise ValueError("candidate_pool_indices and selected_local must be 1D arrays")

    if selected_local.size == 0:
        save_feature_matrix(next_train_feature_file, feature_key, train_features)
        save_feature_matrix(next_pool_feature_file, feature_key, pool_features)
        print(
            "Updated feature sets: "
            f"train={train_features.shape[0]} pool={pool_features.shape[0]}"
        )
        return

    if np.any(selected_local < 0) or np.any(
        selected_local >= candidate_pool_indices.shape[0]
    ):
        raise IndexError(
            "selected_local contains invalid indices for candidate_pool_indices "
            f"(candidate_count={candidate_pool_indices.shape[0]})"
        )

    selected_pool_indices = candidate_pool_indices[selected_local]
    n_pool = int(pool_features.shape[0])
    if np.any(selected_pool_indices < 0) or np.any(selected_pool_indices >= n_pool):
        raise IndexError(
            "selected_pool_indices out of bounds for pool features "
            f"(pool_size={n_pool})"
        )

    selected_mask = np.zeros(n_pool, dtype=bool)
    selected_mask[selected_pool_indices] = True
    selected_pool_ordered = np.flatnonzero(selected_mask)

    if selected_pool_ordered.size > 0:
        selected_features = pool_features[selected_pool_ordered]
        new_train_features = np.concatenate([train_features, selected_features], axis=0)
    else:
        new_train_features = train_features
    new_pool_features = pool_features[~selected_mask]

    save_feature_matrix(next_train_feature_file, feature_key, new_train_features)
    save_feature_matrix(next_pool_feature_file, feature_key, new_pool_features)

    print(
        f"Updated feature sets: train={new_train_features.shape[0]} "
        f"pool={new_pool_features.shape[0]}"
    )


def cmd_threshold_stop(args: argparse.Namespace) -> None:
    max_uncert = float(args.max_uncert)
    threshold = float(args.threshold)
    print("1" if max_uncert <= threshold else "0")


def cmd_anchored_fps(args: argparse.Namespace) -> None:
    from fps import CoreSetSelectorMemoryEfficient

    train_feature_file = Path(args.train_feature_file)
    pool_feature_file = Path(args.pool_feature_file)
    pool_xyz = Path(args.pool_xyz)
    output_prefix = Path(args.output_prefix)
    feature_key = str(args.feature_key)
    candidate_k = int(args.candidate_k)
    normalization = str(args.normalization)

    train_features = load_feature_matrix(train_feature_file, feature_key)
    pool_features = load_feature_matrix(pool_feature_file, feature_key)

    if train_features.shape[1] != pool_features.shape[1]:
        raise ValueError(
            "Feature dimension mismatch between train and pool: "
            f"{train_features.shape[1]} vs {pool_features.shape[1]}"
        )

    pool_frames = read_frames(pool_xyz)
    if len(pool_frames) != pool_features.shape[0]:
        raise ValueError(
            "Pool xyz size does not match pool feature size: "
            f"{len(pool_frames)} vs {pool_features.shape[0]}"
        )

    train_features, pool_features = normalize_feature_matrices(
        train_features, pool_features, normalization
    )

    n_train = int(train_features.shape[0])
    n_pool = int(pool_features.shape[0])
    if n_pool <= 0:
        raise RuntimeError("Pool is empty, cannot run anchored FPS")

    k = min(max(candidate_k, 1), n_pool)
    all_features = np.concatenate([train_features, pool_features], axis=0)
    initial_selected = list(range(n_train))

    selector = CoreSetSelectorMemoryEfficient(all_features)
    selected_global = selector.select_batch(k=n_train + k, s0=initial_selected)
    selected_global = np.asarray(selected_global, dtype=np.int64)

    selected_pool_local = selected_global[selected_global >= n_train] - n_train
    if selected_pool_local.shape[0] > k:
        selected_pool_local = selected_pool_local[:k]

    selected_pool_local = selected_pool_local.astype(np.int64, copy=False)
    selected_structures = [pool_frames[int(i)] for i in selected_pool_local]
    selected_features = pool_features[selected_pool_local]

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    np.save(f"{output_prefix}_{k}_indices.npy", selected_pool_local)
    np.save(f"{output_prefix}_{k}_features.npy", selected_features)
    write(f"{output_prefix}_{k}.extxyz", selected_structures, format="extxyz")

    print(f"Saved indices: {output_prefix}_{k}_indices.npy")
    print(f"Saved selected features: {output_prefix}_{k}_features.npy")
    print(f"Saved selected structures: {output_prefix}_{k}.extxyz")


CONFIG_KEYS = {
    "input_xyz",
    "init_size",
    "candidate_pct",
    "select_size",
    "max_rounds",
    "workdir",
    "train_work_dir",
    "log_dir",
    "model_dir",
    "checkpoints_dir",
    "results_dir",
    "downloads_dir",
    "valid_xyz",
    "test_xyz",
    "fps_model",
    "seeds",
    "seed",
    "device",
    "batch_size",
    "fps_batch_size",
    "eval_batch_size",
    "max_num_epochs",
    "r_max",
    "hidden_irreps",
    "force_weight",
    "uncertainty_metric",
    "uncertainty_threshold",
    "fps_feature_key",
    "fps_normalization",
    "enable_cueq",
    "cueq",
    "metrics_file",
    "metrics_interval",
}


def parse_simple_config(path: Path) -> dict[str, str]:
    cfg: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "#" in line:
            line = line.split("#", 1)[0].rstrip()
        if ":" in line:
            key, value = line.split(":", 1)
        elif "=" in line:
            key, value = line.split("=", 1)
        else:
            continue
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if value and ((value[0] == value[-1]) and value[0] in {"'", '"'}):
            value = value[1:-1]
        cfg[key] = value
    return cfg


def load_config(path: Path) -> dict[str, str]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            raise ValueError("JSON config must be an object")
        return {str(k): str(v) for k, v in obj.items()}

    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore

            obj = yaml.safe_load(path.read_text(encoding="utf-8"))
            if obj is None:
                return {}
            if not isinstance(obj, dict):
                raise ValueError("YAML config must be a mapping/object")
            return {str(k): str(v) for k, v in obj.items()}
        except ModuleNotFoundError:
            return parse_simple_config(path)

    return parse_simple_config(path)


def cmd_config_env(args: argparse.Namespace) -> None:
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = load_config(config_path)
    for key, value in cfg.items():
        if key not in CONFIG_KEYS:
            raise ValueError(
                f"Unknown config key: {key}. Supported keys: {sorted(CONFIG_KEYS)}"
            )
        print(f"{key}\t{value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Helper ops for FPS + QBC workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_init = subparsers.add_parser(
        "init-split", help="Randomly split initial train and pool"
    )
    p_init.add_argument("--input_xyz", required=True)
    p_init.add_argument("--init_size", required=True, type=int)
    p_init.add_argument("--seed", required=True, type=int)
    p_init.add_argument("--train_out", required=True)
    p_init.add_argument("--pool_out", required=True)
    p_init.set_defaults(func=cmd_init_split)

    p_count = subparsers.add_parser(
        "count-frames", help="Count number of frames in xyz"
    )
    p_count.add_argument("--input_xyz", required=True)
    p_count.set_defaults(func=cmd_count_frames)

    p_k = subparsers.add_parser(
        "candidate-k", help="Convert candidate pct to candidate count"
    )
    p_k.add_argument("--pool_size", required=True, type=int)
    p_k.add_argument("--candidate_pct", required=True, type=float)
    p_k.set_defaults(func=cmd_candidate_k)

    p_unc = subparsers.add_parser(
        "committee-uncertainty", help="Evaluate committee uncertainty on candidates"
    )
    p_unc.add_argument("--candidate_xyz", required=True)
    p_unc.add_argument("--select_size", required=True, type=int)
    p_unc.add_argument("--metric", required=True)
    p_unc.add_argument("--selected_local_out", required=True)
    p_unc.add_argument("--stats_out", required=True)
    p_unc.add_argument("--device", required=True)
    p_unc.add_argument("--batch_size", type=int, default=1)
    p_unc.add_argument("--enable_cueq", default="True")
    p_unc.add_argument("--model_paths", nargs="+", required=True)
    p_unc.set_defaults(func=cmd_committee_uncertainty)

    p_upd = subparsers.add_parser(
        "update-datasets", help="Move selected structures from pool to train"
    )
    p_upd.add_argument("--train_xyz", required=True)
    p_upd.add_argument("--pool_xyz", required=True)
    p_upd.add_argument("--candidate_pool_indices_npy", required=True)
    p_upd.add_argument("--selected_local_npy", required=True)
    p_upd.add_argument("--next_train_xyz", required=True)
    p_upd.add_argument("--next_pool_xyz", required=True)
    p_upd.set_defaults(func=cmd_update_datasets)

    p_upd_feat = subparsers.add_parser(
        "update-feature-sets",
        help="Update train/pool feature matrices using selected pool indices",
    )
    p_upd_feat.add_argument("--train_feature_file", required=True)
    p_upd_feat.add_argument("--pool_feature_file", required=True)
    p_upd_feat.add_argument("--feature_key", required=True)
    p_upd_feat.add_argument("--candidate_pool_indices_npy", required=True)
    p_upd_feat.add_argument("--selected_local_npy", required=True)
    p_upd_feat.add_argument("--next_train_feature_file", required=True)
    p_upd_feat.add_argument("--next_pool_feature_file", required=True)
    p_upd_feat.set_defaults(func=cmd_update_feature_sets)

    p_stop = subparsers.add_parser(
        "threshold-stop", help="Check uncertainty stop condition"
    )
    p_stop.add_argument("--max_uncert", required=True, type=float)
    p_stop.add_argument("--threshold", required=True, type=float)
    p_stop.set_defaults(func=cmd_threshold_stop)

    p_cfg = subparsers.add_parser(
        "config-env", help="Read config file and output key-value pairs"
    )
    p_cfg.add_argument("--config", required=True)
    p_cfg.set_defaults(func=cmd_config_env)

    p_afps = subparsers.add_parser(
        "anchored-fps",
        help="Run FPS on pool using train structures as pre-selected anchors",
    )
    p_afps.add_argument("--train_feature_file", required=True)
    p_afps.add_argument("--pool_feature_file", required=True)
    p_afps.add_argument("--feature_key", required=True)
    p_afps.add_argument("--pool_xyz", required=True)
    p_afps.add_argument("--candidate_k", required=True, type=int)
    p_afps.add_argument("--output_prefix", required=True)
    p_afps.add_argument("--normalization", default="zscore")
    p_afps.set_defaults(func=cmd_anchored_fps)

    p_pool = subparsers.add_parser(
        "pool-mean-uncertainty",
        help="Compute mean uncertainty on the remaining pool",
    )
    p_pool.add_argument("--pool_xyz", required=True)
    p_pool.add_argument("--metric", required=True)
    p_pool.add_argument("--device", required=True)
    p_pool.add_argument("--batch_size", type=int, default=1)
    p_pool.add_argument("--enable_cueq", default="True")
    p_pool.add_argument("--model_paths", nargs="+", required=True)
    p_pool.set_defaults(func=cmd_pool_mean_uncertainty)

    p_test_unc = subparsers.add_parser(
        "test-uncertainty",
        help="Compute uncertainty on the test set and save per-structure stats",
    )
    p_test_unc.add_argument("--test_xyz", required=True)
    p_test_unc.add_argument("--metric", required=True)
    p_test_unc.add_argument("--stats_out", required=True)
    p_test_unc.add_argument("--device", required=True)
    p_test_unc.add_argument("--batch_size", type=int, default=1)
    p_test_unc.add_argument("--enable_cueq", default="True")
    p_test_unc.add_argument("--model_paths", nargs="+", required=True)
    p_test_unc.set_defaults(func=cmd_test_uncertainty)

    p_test = subparsers.add_parser(
        "test-rmse",
        help="Compute test energy/force RMSE and L4 averaged over committee models",
    )
    p_test.add_argument("--test_xyz", required=True)
    p_test.add_argument("--device", required=True)
    p_test.add_argument("--batch_size", type=int, default=1)
    p_test.add_argument("--enable_cueq", default="True")
    p_test.add_argument("--model_paths", nargs="+", required=True)
    p_test.set_defaults(func=cmd_test_rmse)

    p_append = subparsers.add_parser(
        "append-round-metrics",
        help="Append one round metrics row to CSV",
    )
    p_append.add_argument("--metrics_file", required=True)
    p_append.add_argument("--round", required=True, type=int)
    p_append.add_argument("--candidate_k", required=True, type=int)
    p_append.add_argument("--selected_size", required=True, type=int)
    p_append.add_argument("--pool_size_after_select", required=True, type=int)
    p_append.add_argument("--uncertainty_metric", required=True)
    p_append.add_argument("--max_candidate_uncertainty", required=True)
    p_append.add_argument("--test_mean_uncertainty", required=True)
    p_append.add_argument("--test_energy_rmse", required=True)
    p_append.add_argument("--test_force_rmse", required=True)
    p_append.add_argument("--test_energy_l4", required=True)
    p_append.add_argument("--test_force_l4", required=True)
    p_append.set_defaults(func=cmd_append_round_metrics)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
