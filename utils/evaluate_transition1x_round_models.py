from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from ase import Atoms

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflow.utils.fps_qbc_ops import _predict_committee_batch, build_calculators

DATASET_PATH = Path("data/Transition1x_maceoff_small_dedup20.h5")
OUTPUT_DIR = Path("rets/transition1x_round_model_reaction_eval")
ROUNDS = (0, 4, 8, 12, 16, 20)
ENERGY_KEY = "wB97x_6-31G(d).energy"
MODEL_FAMILIES = {
    "fps_qbc": Path("rets/transition1x-fps_qbc-rounds-lr1e-2-stage2"),
    "qbc": Path("rets/transition1x-qbc-rounds-lr1e-2-stage2"),
    "fps": Path("rets/transition1x-fps-rounds-lr1e-2-stage2"),
    "random": Path("rets/transition1x-random-rounds-lr1e-2-stage2"),
}


@dataclass
class EndpointRecord:
    split: str
    formula: str
    reaction: str
    endpoint: str
    n_atoms: int
    dft_energy_eV: float


@dataclass
class ModelSpec:
    family: str
    round_idx: int
    model_path: Path


def iter_reactions(handle: h5py.File, split: str):
    for formula in handle[split].keys():
        for reaction in handle[split][formula].keys():
            yield formula, reaction, handle[split][formula][reaction]


def build_endpoint_dataset(dataset_path: Path) -> tuple[list[EndpointRecord], list[Atoms]]:
    endpoint_records: list[EndpointRecord] = []
    endpoint_atoms: list[Atoms] = []
    with h5py.File(dataset_path, "r") as handle:
        for split in ("val", "test"):
            for formula, reaction, group in iter_reactions(handle, split):
                atomic_numbers = group["atomic_numbers"][...]
                n_atoms = int(atomic_numbers.shape[0])
                for endpoint in ("reactant", "transition_state", "product"):
                    endpoint_group = group[endpoint]
                    positions = endpoint_group["positions"][0]
                    dft_energy = float(endpoint_group[ENERGY_KEY][0])
                    atoms = Atoms(numbers=atomic_numbers, positions=positions)
                    endpoint_atoms.append(atoms)
                    endpoint_records.append(
                        EndpointRecord(
                            split=split,
                            formula=formula,
                            reaction=reaction,
                            endpoint=endpoint,
                            n_atoms=n_atoms,
                            dft_energy_eV=dft_energy,
                        )
                    )
    return endpoint_records, endpoint_atoms


def build_model_specs() -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    for family, root in MODEL_FAMILIES.items():
        for round_idx in ROUNDS:
            model_path = root / f"round{round_idx}" / "checkpoints" / f"transition1x_round{round_idx}_run-0_stagetwo.model"
            if not model_path.exists():
                raise FileNotFoundError(f"Missing model: {model_path}")
            specs.append(ModelSpec(family=family, round_idx=round_idx, model_path=model_path))
    return specs


def batched_predict_energies(model_path: Path, atoms_list: list[Atoms], device: str, batch_size: int) -> np.ndarray:
    calculators = build_calculators([str(model_path)], device, "True")
    energy_predictions, _ = _predict_committee_batch(
        frames=atoms_list,
        calculators=calculators,
        device=device,
        batch_size=batch_size,
        progress_desc=f"predict {model_path.parent.parent.name}/{model_path.stem}",
    )
    return energy_predictions[0]


def reaction_rows_for_model(spec: ModelSpec, endpoint_records: list[EndpointRecord], predicted_energies: np.ndarray) -> list[dict[str, object]]:
    if len(endpoint_records) != int(predicted_energies.shape[0]):
        raise ValueError("Endpoint record count does not match predicted energy count")

    rows: list[dict[str, object]] = []
    for idx in range(0, len(endpoint_records), 3):
        reactant = endpoint_records[idx]
        transition_state = endpoint_records[idx + 1]
        product = endpoint_records[idx + 2]

        if not (
            reactant.endpoint == "reactant"
            and transition_state.endpoint == "transition_state"
            and product.endpoint == "product"
        ):
            raise ValueError(f"Unexpected endpoint ordering at index {idx}")

        dft_reactant = reactant.dft_energy_eV
        dft_ts = transition_state.dft_energy_eV
        dft_product = product.dft_energy_eV
        pred_reactant = float(predicted_energies[idx])
        pred_ts = float(predicted_energies[idx + 1])
        pred_product = float(predicted_energies[idx + 2])

        dft_forward = dft_ts - dft_reactant
        pred_forward = pred_ts - pred_reactant
        dft_reaction = dft_product - dft_reactant
        pred_reaction = pred_product - pred_reactant
        dft_reverse = dft_ts - dft_product
        pred_reverse = pred_ts - pred_product

        rows.append(
            {
                "family": spec.family,
                "round": spec.round_idx,
                "split": reactant.split,
                "formula": reactant.formula,
                "reaction": reactant.reaction,
                "n_atoms": reactant.n_atoms,
                "dft_reactant_energy_eV": dft_reactant,
                "dft_transition_state_energy_eV": dft_ts,
                "dft_product_energy_eV": dft_product,
                "model_reactant_energy_eV": pred_reactant,
                "model_transition_state_energy_eV": pred_ts,
                "model_product_energy_eV": pred_product,
                "dft_forward_barrier_eV": dft_forward,
                "model_forward_barrier_eV": pred_forward,
                "forward_barrier_error_eV": pred_forward - dft_forward,
                "dft_reaction_energy_eV": dft_reaction,
                "model_reaction_energy_eV": pred_reaction,
                "reaction_energy_error_eV": pred_reaction - dft_reaction,
                "dft_reverse_barrier_eV": dft_reverse,
                "model_reverse_barrier_eV": pred_reverse,
                "reverse_barrier_error_eV": pred_reverse - dft_reverse,
            }
        )
    return rows


def summarize_metric(errors: np.ndarray) -> tuple[float, float, float]:
    abs_errors = np.abs(errors)
    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(errors * errors)))
    l4 = float(np.power(np.mean(abs_errors**4), 0.25))
    return mae, rmse, l4


def summary_rows(detail_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    families_rounds = sorted({(row["family"], row["round"]) for row in detail_rows}, key=lambda x: (str(x[0]), int(x[1])))
    for family, round_idx in families_rounds:
        for split in ("val", "test"):
            subset = [row for row in detail_rows if row["family"] == family and row["round"] == round_idx and row["split"] == split]
            if not subset:
                continue
            forward_errors = np.asarray([float(row["forward_barrier_error_eV"]) for row in subset], dtype=float)
            reaction_errors = np.asarray([float(row["reaction_energy_error_eV"]) for row in subset], dtype=float)
            reverse_errors = np.asarray([float(row["reverse_barrier_error_eV"]) for row in subset], dtype=float)
            forward_mae, forward_rmse, forward_l4 = summarize_metric(forward_errors)
            reaction_mae, reaction_rmse, reaction_l4 = summarize_metric(reaction_errors)
            reverse_mae, reverse_rmse, reverse_l4 = summarize_metric(reverse_errors)
            rows.append(
                {
                    "family": family,
                    "round": int(round_idx),
                    "split": split,
                    "n_reactions": len(subset),
                    "forward_mae_eV": forward_mae,
                    "forward_rmse_eV": forward_rmse,
                    "forward_l4_eV": forward_l4,
                    "reaction_mae_eV": reaction_mae,
                    "reaction_rmse_eV": reaction_rmse,
                    "reaction_l4_eV": reaction_l4,
                    "reverse_mae_eV": reverse_mae,
                    "reverse_rmse_eV": reverse_rmse,
                    "reverse_l4_eV": reverse_l4,
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summary: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Transition1x Round Model Reaction Energy Evaluation",
        "",
        f"- Dataset: `{DATASET_PATH}`",
        "- Splits: `val`, `test`",
        "- Definitions: `forward barrier = E_TS - E_reactant`; `reaction energy = E_product - E_reactant`; `reverse barrier = E_TS - E_product`; error = `model - DFT`",
        "- Unit: `eV`",
        "",
    ]
    for split in ("val", "test"):
        lines.append(f"## {split.title()} Split")
        lines.append("")
        lines.append("| Family | Round | Fwd MAE | Fwd RMSE | Fwd L4 | Rxn MAE | Rxn RMSE | Rxn L4 | Rev MAE | Rev RMSE | Rev L4 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in [r for r in summary if r["split"] == split]:
            lines.append(
                f"| {row['family']} | {row['round']} | "
                f"{row['forward_mae_eV']:.3f} | {row['forward_rmse_eV']:.3f} | {row['forward_l4_eV']:.3f} | "
                f"{row['reaction_mae_eV']:.3f} | {row['reaction_rmse_eV']:.3f} | {row['reaction_l4_eV']:.3f} | "
                f"{row['reverse_mae_eV']:.3f} | {row['reverse_rmse_eV']:.3f} | {row['reverse_l4_eV']:.3f} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    device = "cuda"
    batch_size = 64
    endpoint_records, endpoint_atoms = build_endpoint_dataset(DATASET_PATH)
    specs = build_model_specs()

    detail_rows: list[dict[str, object]] = []
    for spec in specs:
        predicted_energies = batched_predict_energies(spec.model_path, endpoint_atoms, device=device, batch_size=batch_size)
        detail_rows.extend(reaction_rows_for_model(spec, endpoint_records, predicted_energies))

    summary = summary_rows(detail_rows)
    write_csv(OUTPUT_DIR / "transition1x_round_model_reaction_details.csv", detail_rows)
    write_csv(OUTPUT_DIR / "transition1x_round_model_reaction_summary.csv", summary)
    write_markdown(OUTPUT_DIR / "transition1x_round_model_reaction_summary.md", summary)
    print(f"Wrote {len(detail_rows)} detail rows and {len(summary)} summary rows to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
