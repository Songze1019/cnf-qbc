"""Extract structure-level hidden features from MACE for MatPES.

Default use case:
    python workflow/utils/extract.py

This reads the MatPES EXTXYZ dataset, extracts `node_feats` from MACE for each
structure, applies structure-level pooling, and saves one NPZ file in `fps/`.
"""

import argparse
import os
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from ase.io import iread
from mace import data as mace_data
from mace.calculators import MACECalculator
from mace.tools import torch_geometric
from tqdm.auto import tqdm


os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract MACE hidden features for FPS")
    parser.add_argument(
        "--input",
        type=str,
        default="data/matpes/MatPES-PBE-2025.1.extxyz",
        help="Input EXTXYZ path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="workflow/utils/matpes_mace_hidden_features.npz",
        help="Output feature file (.npz)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/home/xmcao/huosongze/.mace/model/mace-omat-0-small.model",
        help="MACE model path",
    )
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--pool",
        type=str,
        default="mean",
        choices=["mean", "sum"],
        help="Pooling for per-structure features from node features",
    )
    parser.add_argument(
        "--save_per_atom",
        action="store_true",
        help="Also save per-atom node features for each structure",
    )
    parser.add_argument(
        "--tmp_feature_file",
        type=str,
        default="workflow/utils/.tmp_structure_features.npy",
        help="Temporary memmap file to avoid RAM blow-up",
    )
    return parser.parse_args()


def _build_calculator(model_path: str, device: str) -> MACECalculator:
    return MACECalculator(
        model_paths=model_path,
        device=device,
        enable_cueq=True,
    )


def _count_structures(input_path: Path) -> int:
    count = 0
    for _ in iread(input_path, index=":"):
        count += 1
    return count


def _prepare_atomic_data_for_atoms(atoms, calc: MACECalculator):
    keyspec = mace_data.KeySpecification(
        info_keys=calc.info_keys,
        arrays_keys=calc.arrays_keys,
    )

    head_name = calc.head[0] if isinstance(calc.head, list) else calc.head

    atoms.info.setdefault("charge", 0.0)
    atoms.info.setdefault("spin", 0.0)

    config = mace_data.config_from_atoms(
        atoms,
        key_specification=keyspec,
        head_name=head_name,
    )
    atomic_data = mace_data.AtomicData.from_config(
        config,
        z_table=calc.z_table,
        cutoff=calc.r_max,
        heads=calc.available_heads,
    )
    return atomic_data


def _batched_atomic_data(input_path: Path, calc: MACECalculator, batch_size: int):
    current_batch = []
    for atoms in iread(input_path, index=":"):
        current_batch.append(_prepare_atomic_data_for_atoms(atoms, calc))
        if len(current_batch) == batch_size:
            yield current_batch
            current_batch = []
    if current_batch:
        yield current_batch


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    tmp_feature_path = Path(args.tmp_feature_file)

    model_candidates = [
        Path(args.model),
        Path(os.path.expanduser(args.model)),
        Path("/home/xmcao/huosongze/.mace/model/mace-omat-0-small.model"),
    ]
    model_path = None
    for candidate in model_candidates:
        if candidate.exists():
            model_path = candidate
            break
    if model_path is None:
        tried = "\n".join(str(p) for p in model_candidates)
        raise FileNotFoundError(f"Couldn't find MACE model files. Tried:\n{tried}")

    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")

    print(f"Counting structures in: {input_path}")
    total_frames = _count_structures(input_path)
    print(f"Total frames: {total_frames}")

    calc = _build_calculator(str(model_path), args.device)
    model = cast(Any, calc.models[0])
    model.eval()
    print(f"Loaded model: {model_path} on {args.device}")

    if total_frames == 0:
        raise ValueError(f"No frames found in input file: {input_path}")

    structure_features_memmap = None
    per_atom_features = []
    n_atoms = np.empty(total_frames, dtype=np.int32)
    write_index = 0

    tmp_feature_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for atomic_batch in tqdm(
            _batched_atomic_data(input_path, calc, args.batch_size),
            total=(total_frames + args.batch_size - 1) // args.batch_size,
            desc="Extracting MACE features",
            unit="batch",
        ):
            loader = torch_geometric.dataloader.DataLoader(
                dataset=cast(Any, atomic_batch),
                batch_size=len(atomic_batch),
                shuffle=False,
                drop_last=False,
            )
            batch = next(iter(loader))
            batch = batch.to(args.device)
            output = model(batch.to_dict(), compute_force=False)

            node_feats = output["node_feats"]
            batch_ids = batch.batch
            batch_size_now = int(batch_ids.max().item()) + 1

            for local_idx in range(batch_size_now):
                mask = batch_ids == local_idx
                feats = node_feats[mask].detach().cpu()
                n_atoms[write_index] = int(feats.shape[0])

                if args.pool == "mean":
                    pooled = feats.mean(dim=0)
                else:
                    pooled = feats.sum(dim=0)
                pooled_np = pooled.numpy().astype(np.float32, copy=False)

                if structure_features_memmap is None:
                    structure_features_memmap = np.lib.format.open_memmap(
                        tmp_feature_path,
                        mode="w+",
                        dtype=np.float32,
                        shape=(total_frames, pooled_np.shape[0]),
                    )
                structure_features_memmap[write_index] = pooled_np

                if args.save_per_atom:
                    per_atom_features.append(feats.numpy())

                write_index += 1

    if structure_features_memmap is None:
        raise RuntimeError("No features extracted; memmap was not initialized")
    if write_index != total_frames:
        raise RuntimeError(
            f"Extracted structure count mismatch: got {write_index}, expected {total_frames}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {
        "structure_features": structure_features_memmap,
        "n_atoms": n_atoms,
        "pool": np.array(args.pool),
        "model_path": np.array(str(model_path)),
        "input_path": np.array(str(input_path)),
    }
    if args.save_per_atom:
        save_kwargs["per_atom_features"] = np.array(per_atom_features, dtype=object)

    np.savez_compressed(output_path, **save_kwargs)

    if tmp_feature_path.exists():
        tmp_feature_path.unlink()

    print(f"Saved features to: {output_path}")
    print(f"structure_features shape: {structure_features_memmap.shape}")


if __name__ == "__main__":
    main()
