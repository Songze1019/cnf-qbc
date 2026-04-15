from __future__ import annotations

import argparse
from pathlib import Path

import h5py
from ase import Atoms
from ase.io import write

ENERGY_KEY = "wB97x_6-31G(d).energy"
FORCES_KEY = "wB97x_6-31G(d).forces"


def build_atoms_frames(split: str, formula: str, reaction: str, reaction_group) -> list[Atoms]:
    atomic_numbers = reaction_group["atomic_numbers"][...]
    positions = reaction_group["positions"][...]
    energies = reaction_group[ENERGY_KEY][...]
    forces = reaction_group[FORCES_KEY][...]
    selected_frame_indices = reaction_group["selected_frame_indices"][...]

    frames: list[Atoms] = []
    for frame_index in range(positions.shape[0]):
        atoms = Atoms(numbers=atomic_numbers, positions=positions[frame_index])
        energy = float(energies[frame_index])
        frame_forces = forces[frame_index]

        atoms.info["split"] = split
        atoms.info["formula"] = formula
        atoms.info["reaction"] = reaction
        atoms.info["frame_index"] = int(frame_index)
        atoms.info["original_frame_index"] = int(selected_frame_indices[frame_index])
        atoms.info["energy_unit"] = "eV"
        atoms.info["force_unit"] = "eV/Angstrom"
        atoms.info["energy"] = energy
        atoms.info["REF_energy"] = energy
        atoms.arrays["forces"] = frame_forces.copy()
        atoms.arrays["REF_forces"] = frame_forces.copy()
        frames.append(atoms)

    return frames


def export_split(h5_path: Path, split: str, output_path: Path) -> tuple[int, int]:
    n_reactions = 0
    frames: list[Atoms] = []
    with h5py.File(h5_path, "r") as handle:
        split_group = handle[split]
        for formula in split_group.keys():
            formula_group = split_group[formula]
            for reaction in formula_group.keys():
                frames.extend(
                    build_atoms_frames(
                        split=split,
                        formula=formula,
                        reaction=reaction,
                        reaction_group=formula_group[reaction],
                    )
                )
                n_reactions += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write(str(output_path), frames, format="extxyz")
    return n_reactions, len(frames)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Transition1x HDF5 splits to extxyz with energy/force metadata."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/Transition1x_maceoff_small_dedup20.h5"),
        help="Input HDF5 dataset.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/transition1x"),
        help="Output directory for train.xyz, val.xyz, test.xyz.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for split in ("train", "val", "test"):
        output_path = args.output_dir / f"{split}.xyz"
        n_reactions, n_frames = export_split(args.input, split, output_path)
        print(f"{split}: reactions={n_reactions} frames={n_frames} -> {output_path}")


if __name__ == "__main__":
    main()
