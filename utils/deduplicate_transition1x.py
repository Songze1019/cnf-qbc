import os

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

import argparse
import shutil
import time
import urllib.request
from pathlib import Path

import h5py
import numpy as np
import torch
from ase import Atoms
from mace import data as mace_data
from mace.calculators import MACECalculator
from mace.tools import torch_geometric
from e3nn import o3


ENERGY_KEY = "wB97x_6-31G(d).energy"
FORCES_KEY = "wB97x_6-31G(d).forces"
DEFAULT_INPUT = Path("data/Transition1x.h5")
DEFAULT_OUTPUT = Path("data/Transition1x_maceoff_small_dedup20.h5")
DEFAULT_MODEL = Path.home() / ".mace" / "MACE-OFF23_small.model"
MACE_OFF_SMALL_URL = (
    "https://github.com/ACEsuit/mace-off/blob/main/"
    "mace_off23/MACE-OFF23_small.model?raw=true"
)
ATOMIC_NUMBERS_TO_SYMBOLS = {
    1: "H",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    15: "P",
    16: "S",
    17: "Cl",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Deduplicate Transition1x by MACE-OFF23 small descriptors."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Input HDF5 file")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output HDF5 file")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="MACE model path")
    parser.add_argument("--keep-count", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="", help="cuda, cpu, or empty for auto")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--max-reactions",
        type=int,
        default=None,
        help="Process only the first N reactions for smoke testing.",
    )
    return parser.parse_args()


def ensure_mace_off_small(model_path):
    model_path = Path(model_path).expanduser()
    if model_path.exists():
        print(f"Using existing model: {model_path}")
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = model_path.with_suffix(model_path.suffix + ".tmp")
    print(f"Downloading MACE-OFF23 small model to {model_path}")
    print("Model license: ASL; see https://github.com/gabor1/ASL")
    urllib.request.urlretrieve(MACE_OFF_SMALL_URL, tmp_path)
    tmp_path.replace(model_path)
    return model_path


def copy_attrs(src, dst):
    for key, value in src.attrs.items():
        dst.attrs[key] = value


def copy_group_recursive(src, dst):
    copy_attrs(src, dst)
    for key, obj in src.items():
        if isinstance(obj, h5py.Dataset):
            src.copy(key, dst)
        elif isinstance(obj, h5py.Group):
            child = dst.create_group(key)
            copy_group_recursive(obj, child)
        else:
            raise TypeError(f"Unsupported HDF5 object at {obj.name}: {type(obj)}")


def find_matching_frame_indices(positions, endpoint_positions, atol=1e-10):
    positions = np.asarray(positions)
    matched = []
    for endpoint in endpoint_positions:
        endpoint = np.asarray(endpoint)
        if endpoint.ndim == 3 and endpoint.shape[0] == 1:
            endpoint = endpoint[0]
        if endpoint.shape != positions.shape[1:]:
            raise ValueError(
                f"Endpoint shape {endpoint.shape} does not match frame shape "
                f"{positions.shape[1:]}"
            )
        distances = np.max(np.abs(positions - endpoint[None, :, :]), axis=(1, 2))
        best_idx = int(np.argmin(distances))
        if not np.isclose(distances[best_idx], 0.0, atol=atol):
            raise ValueError(
                f"Could not match endpoint positions; closest max abs diff is "
                f"{distances[best_idx]:.3e}"
            )
        if best_idx not in matched:
            matched.append(best_idx)
    return matched


def farthest_point_indices(descriptors, keep_count, forced_indices=()):
    descriptors = np.asarray(descriptors, dtype=np.float64)
    n_frames = descriptors.shape[0]
    if keep_count >= n_frames:
        return list(range(n_frames))
    if keep_count <= 0:
        return []

    forced = []
    for idx in forced_indices:
        idx = int(idx)
        if idx < 0 or idx >= n_frames:
            raise IndexError(f"Forced index {idx} out of range for {n_frames} frames")
        if idx not in forced:
            forced.append(idx)

    selected = forced[:keep_count]
    if not selected:
        selected = [0]

    selected_mask = np.zeros(n_frames, dtype=bool)
    selected_mask[selected] = True
    min_sq_dist = np.full(n_frames, np.inf, dtype=np.float64)

    for idx in selected:
        diff = descriptors - descriptors[idx]
        min_sq_dist = np.minimum(min_sq_dist, np.einsum("ij,ij->i", diff, diff))
    min_sq_dist[selected_mask] = -np.inf

    while len(selected) < keep_count:
        next_idx = int(np.argmax(min_sq_dist))
        if selected_mask[next_idx]:
            break
        selected.append(next_idx)
        selected_mask[next_idx] = True
        diff = descriptors - descriptors[next_idx]
        min_sq_dist = np.minimum(min_sq_dist, np.einsum("ij,ij->i", diff, diff))
        min_sq_dist[selected_mask] = -np.inf

    return sorted(selected)


def select_diverse_indices(descriptors, endpoint_indices, keep_count=20):
    descriptors = np.asarray(descriptors)
    if descriptors.shape[0] <= keep_count:
        return list(range(descriptors.shape[0]))
    return farthest_point_indices(
        descriptors=descriptors,
        keep_count=keep_count,
        forced_indices=endpoint_indices,
    )


def reaction_to_atoms_list(reaction_group):
    atomic_numbers = reaction_group["atomic_numbers"][...]
    symbols = [ATOMIC_NUMBERS_TO_SYMBOLS[int(z)] for z in atomic_numbers]
    positions = reaction_group["positions"]
    energies = reaction_group[ENERGY_KEY]
    forces = reaction_group[FORCES_KEY]
    atoms_list = []
    for idx in range(positions.shape[0]):
        atoms = Atoms(symbols=symbols, positions=positions[idx])
        energy = float(energies[idx])
        frame_forces = forces[idx]
        atoms.info["energy"] = energy
        atoms.info["REF_energy"] = energy
        atoms.info["transition1x_frame_index"] = idx
        atoms.arrays["forces"] = frame_forces.copy()
        atoms.arrays["REF_forces"] = frame_forces.copy()
        atoms_list.append(atoms)
    return atoms_list


def descriptor_dimension(model):
    irreps_out = o3.Irreps(str(model.products[0].linear.irreps_out))
    l_max = irreps_out.lmax
    num_inv = irreps_out.dim // (l_max + 1) ** 2
    num_interactions = int(model.num_interactions)
    per_layer_dims = [irreps_out.dim] * num_interactions
    per_layer_dims[-1] = num_inv
    return sum(per_layer_dims)


def extract_descriptors_batched(atoms_list, calc, batch_size, device):
    model = calc.models[0]
    model.eval()
    keyspec = mace_data.KeySpecification(
        info_keys=calc.info_keys, arrays_keys=calc.arrays_keys
    )
    atomic_data_list = []
    for atoms in atoms_list:
        config = mace_data.config_from_atoms(
            atoms,
            key_specification=keyspec,
            head_name=calc.head,
        )
        atomic_data_list.append(
            mace_data.AtomicData.from_config(
                config,
                z_table=calc.z_table,
                cutoff=calc.r_max,
                heads=calc.available_heads,
            )
        )

    loader = torch_geometric.dataloader.DataLoader(
        dataset=atomic_data_list,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    descriptors = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch.to_dict(), compute_force=False)
            node_feats = output["node_feats"]
            batch_ids = batch.batch
            cur_batch_size = int(batch_ids.max().item()) + 1
            for local_idx in range(cur_batch_size):
                mask = batch_ids == local_idx
                descriptors.append(node_feats[mask].mean(dim=0).cpu())
    return torch.stack(descriptors, dim=0).numpy()


def selected_endpoint_indices(reaction_group):
    positions = reaction_group["positions"][...]
    endpoints = [
        reaction_group["reactant"]["positions"][...],
        reaction_group["transition_state"]["positions"][...],
        reaction_group["product"]["positions"][...],
    ]
    return find_matching_frame_indices(positions, endpoints)


def copy_deduplicated_reaction(src_group, dst_parent, name, selected_indices):
    dst_group = dst_parent.create_group(name)
    copy_attrs(src_group, dst_group)
    selected_indices = np.asarray(selected_indices, dtype=np.int64)
    for key, obj in src_group.items():
        if key in {"positions", ENERGY_KEY, FORCES_KEY}:
            dst_group.create_dataset(
                key,
                data=obj[selected_indices],
                compression=obj.compression,
                compression_opts=obj.compression_opts,
                shuffle=obj.shuffle,
            )
        elif key == "atomic_numbers":
            src_group.copy(key, dst_group)
        elif isinstance(obj, h5py.Group):
            child = dst_group.create_group(key)
            copy_group_recursive(obj, child)
        else:
            src_group.copy(key, dst_group)
    dst_group.create_dataset("selected_frame_indices", data=selected_indices)
    dst_group.attrs["dedup_keep_count"] = int(len(selected_indices))
    return dst_group


def iter_reaction_paths(root_group):
    for formula in root_group.keys():
        for rxn in root_group[formula].keys():
            yield formula, rxn


def create_split_links(src_file, dst_file):
    for split in ["train", "val", "test"]:
        split_group = dst_file.create_group(split)
        copy_attrs(src_file[split], split_group)
        for formula, rxn in iter_reaction_paths(src_file[split]):
            if f"data/{formula}/{rxn}" not in dst_file:
                continue
            if formula not in split_group:
                split_group.create_group(formula)
            split_group[formula][rxn] = dst_file[f"data/{formula}/{rxn}"]


def run_deduplication(args):
    input_path = Path(args.input)
    output_path = Path(args.output)
    model_path = ensure_mace_off_small(args.model)
    if output_path.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_path} exists; pass --overwrite to replace")
        output_path.unlink()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    calc = MACECalculator(
        model_paths=str(model_path),
        device=device,
        default_dtype="float32",
        enable_cueq=True,
    )
    print(f"Descriptor dimension: {descriptor_dimension(calc.models[0])}")
    print(f"Device: {device}; batch size: {args.batch_size}")

    tmp_output = output_path.with_suffix(output_path.suffix + ".tmp")
    if tmp_output.exists():
        tmp_output.unlink()

    n_reactions = 0
    n_input_frames = 0
    n_output_frames = 0
    start = time.time()
    with h5py.File(input_path, "r") as src, h5py.File(tmp_output, "w") as dst:
        copy_attrs(src, dst)
        dst.attrs["dedup_method"] = "mace_off23_small_descriptor_farthest_point"
        dst.attrs["dedup_keep_count"] = int(args.keep_count)
        dst.attrs["dedup_model_path"] = str(model_path)
        dst.attrs["dedup_source_file"] = str(input_path)
        dst.attrs["dedup_forced_endpoints"] = "reactant,transition_state,product"

        dst_data = dst.create_group("data")
        copy_attrs(src["data"], dst_data)

        for formula, rxn in iter_reaction_paths(src["data"]):
            if args.max_reactions is not None and n_reactions >= args.max_reactions:
                break
            src_formula = src["data"][formula]
            if formula not in dst_data:
                dst_data.create_group(formula)
                copy_attrs(src_formula, dst_data[formula])
            src_reaction = src_formula[rxn]
            n_frames = int(src_reaction[ENERGY_KEY].shape[0])
            endpoint_indices = selected_endpoint_indices(src_reaction)
            if n_frames <= args.keep_count:
                selected = list(range(n_frames))
            else:
                atoms_list = reaction_to_atoms_list(src_reaction)
                descriptors = extract_descriptors_batched(
                    atoms_list=atoms_list,
                    calc=calc,
                    batch_size=args.batch_size,
                    device=device,
                )
                selected = select_diverse_indices(
                    descriptors=descriptors,
                    endpoint_indices=endpoint_indices,
                    keep_count=args.keep_count,
                )
            copy_deduplicated_reaction(src_reaction, dst_data[formula], rxn, selected)
            n_reactions += 1
            n_input_frames += n_frames
            n_output_frames += len(selected)
            elapsed = time.time() - start
            rate = n_reactions / elapsed if elapsed else 0.0
            print(
                f"[{n_reactions}] {formula}/{rxn}: "
                f"{n_frames} -> {len(selected)} frames; "
                f"elapsed {elapsed:.1f}s; {rate:.2f} rxn/s",
                flush=True,
            )

        create_split_links(src, dst)
        dst.attrs["dedup_reactions_written"] = int(n_reactions)
        dst.attrs["dedup_input_frames_seen"] = int(n_input_frames)
        dst.attrs["dedup_output_frames_written"] = int(n_output_frames)

    if output_path.exists():
        output_path.unlink()
    shutil.move(tmp_output, output_path)
    print(f"Wrote {output_path}")
    print(f"Reactions: {n_reactions}")
    print(f"Frames: {n_input_frames} -> {n_output_frames}")


def main():
    run_deduplication(parse_args())


if __name__ == "__main__":
    main()
