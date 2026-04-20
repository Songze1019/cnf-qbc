from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from ase import Atoms
from ase.io import write
from ase.mep import NEB
from ase.optimize import FIRE
from mace.calculators import MACECalculator
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATASET_PATH = Path("data/Transition1x_maceoff_small_dedup20.h5")
OUTPUT_DIR = Path("rets/transition1x-neb")
ENERGY_KEY = "wB97x_6-31G(d).energy"
ROUNDS = (0, 4, 8, 12, 16, 20)
DEFAULT_ROUNDS = (4, 12, 20)
DEFAULT_SAMPLE_REACTIONS = 50
DEFAULT_SAMPLE_SEED = 42
NEB_PROTOCOL = "idpp_two_stage_v1"
ROUND_MODEL_FAMILIES = {
    "fps_qbc": Path("rets/transition1x-fps_qbc-rounds-lr1e-2-stage2"),
    "qbc": Path("rets/transition1x-qbc-rounds-lr1e-2-stage2"),
    "fps": Path("rets/transition1x-fps-rounds-lr1e-2-stage2"),
    "random": Path("rets/transition1x-random-rounds-lr1e-2-stage2"),
}
ALL_DATA_MODEL = Path(
    "rets/transition1x-mace-alldata-lr1e-2-stage2/checkpoints/"
    "transition1x_scratch_run-0_stagetwo.model"
)
SUMMARY_FIELDNAMES = [
    "family",
    "round",
    "split",
    "n_reactions",
    "n_success",
    "success_rate",
    "all_ts_rmsd_mean_A",
    "all_ts_rmsd_median_A",
    "all_ts_rmsd_rmse_A",
    "all_forward_barrier_me_eV",
    "all_forward_barrier_mae_eV",
    "all_forward_barrier_rmse_eV",
    "all_forward_barrier_l4_eV",
    "all_reverse_barrier_me_eV",
    "all_reverse_barrier_mae_eV",
    "all_reverse_barrier_rmse_eV",
    "all_reverse_barrier_l4_eV",
    "success_ts_rmsd_mean_A",
    "success_ts_rmsd_median_A",
    "success_ts_rmsd_rmse_A",
    "success_forward_barrier_me_eV",
    "success_forward_barrier_mae_eV",
    "success_forward_barrier_rmse_eV",
    "success_forward_barrier_l4_eV",
    "success_reverse_barrier_me_eV",
    "success_reverse_barrier_mae_eV",
    "success_reverse_barrier_rmse_eV",
    "success_reverse_barrier_l4_eV",
]


@dataclass(frozen=True)
class ReactionRecord:
    split: str
    formula: str
    reaction: str
    atomic_numbers: np.ndarray
    reactant_positions: np.ndarray
    ts_positions: np.ndarray
    product_positions: np.ndarray
    dft_reactant_energy_eV: float
    dft_ts_energy_eV: float
    dft_product_energy_eV: float


@dataclass(frozen=True)
class ModelSpec:
    family: str
    round_label: str
    model_path: Path


def iter_reaction_records(dataset_path: Path, splits: set[str]) -> list[ReactionRecord]:
    records: list[ReactionRecord] = []
    with h5py.File(dataset_path, "r") as handle:
        for split in ("val", "test"):
            if split not in splits:
                continue
            for formula in handle[split].keys():
                for reaction in handle[split][formula].keys():
                    group = handle[split][formula][reaction]
                    records.append(
                        ReactionRecord(
                            split=split,
                            formula=formula,
                            reaction=reaction,
                            atomic_numbers=group["atomic_numbers"][...],
                            reactant_positions=group["reactant"]["positions"][0],
                            ts_positions=group["transition_state"]["positions"][0],
                            product_positions=group["product"]["positions"][0],
                            dft_reactant_energy_eV=float(
                                group["reactant"][ENERGY_KEY][0]
                            ),
                            dft_ts_energy_eV=float(
                                group["transition_state"][ENERGY_KEY][0]
                            ),
                            dft_product_energy_eV=float(
                                group["product"][ENERGY_KEY][0]
                            ),
                        )
                    )
    return records


def build_model_specs(families: set[str], rounds: set[int], include_all_data: bool) -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    for family, root in ROUND_MODEL_FAMILIES.items():
        if family not in families:
            continue
        for round_idx in ROUNDS:
            if round_idx not in rounds:
                continue
            model_path = (
                root
                / f"round{round_idx}"
                / "checkpoints"
                / f"transition1x_round{round_idx}_run-0_stagetwo.model"
            )
            if not model_path.exists():
                raise FileNotFoundError(f"Missing model: {model_path}")
            specs.append(
                ModelSpec(family=family, round_label=str(round_idx), model_path=model_path)
            )

    if include_all_data:
        if not ALL_DATA_MODEL.exists():
            raise FileNotFoundError(f"Missing all-data model: {ALL_DATA_MODEL}")
        specs.append(
            ModelSpec(family="all_data", round_label="all", model_path=ALL_DATA_MODEL)
        )
    return specs


def row_key(row: dict[str, object]) -> tuple[str, str, str, str, str]:
    return (
        str(row["family"]),
        str(row["round"]),
        str(row["split"]),
        str(row["formula"]),
        str(row["reaction"]),
    )


def load_existing_details(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def settings_signature(args: argparse.Namespace) -> str:
    return "|".join(
        [
            f"protocol={NEB_PROTOCOL}",
            f"device={args.device}",
            f"enable_cueq={bool(args.enable_cueq and args.device != 'cpu')}",
            f"num_images={args.num_images}",
            f"fmax={args.fmax}",
            f"steps={args.steps}",
            f"climb={args.climb}",
            f"splits={','.join(sorted(args.splits))}",
            f"sample_reactions={args.sample_reactions}",
            f"sample_seed={args.sample_seed}",
            f"rounds={','.join(map(str, sorted(args.rounds)))}",
            f"include_all_data={bool(args.include_all_data or args.only_all_data)}",
        ]
    )


def sample_records_by_split(
    records: list[ReactionRecord],
    sample_reactions: int | None,
    seed: int,
) -> list[ReactionRecord]:
    if sample_reactions is None:
        return records
    if sample_reactions < 1:
        raise ValueError(f"sample_reactions must be positive, got {sample_reactions}")

    rng = np.random.default_rng(seed)
    sampled: list[ReactionRecord] = []
    for split in ("val", "test"):
        split_records = [record for record in records if record.split == split]
        if not split_records:
            continue
        n_select = min(sample_reactions, len(split_records))
        selected = set(rng.choice(len(split_records), size=n_select, replace=False))
        sampled.extend(
            record for idx, record in enumerate(split_records) if idx in selected
        )
    return sampled


def kabsch_rmsd(positions: np.ndarray, reference: np.ndarray) -> float:
    p = np.asarray(positions, dtype=float)
    q = np.asarray(reference, dtype=float)
    p0 = p - p.mean(axis=0, keepdims=True)
    q0 = q - q.mean(axis=0, keepdims=True)
    covariance = p0.T @ q0
    v, _, wt = np.linalg.svd(covariance)
    handedness = np.sign(np.linalg.det(v @ wt))
    rotation = v @ np.diag([1.0, 1.0, handedness]) @ wt
    p_aligned = p0 @ rotation
    return float(np.sqrt(np.mean(np.sum((p_aligned - q0) ** 2, axis=1))))


def make_endpoint_atoms(record: ReactionRecord) -> tuple[Atoms, Atoms]:
    reactant = Atoms(numbers=record.atomic_numbers, positions=record.reactant_positions)
    product = Atoms(numbers=record.atomic_numbers, positions=record.product_positions)
    return reactant, product


def compute_neb_max_force(neb: NEB) -> float:
    try:
        neb_forces = neb.get_forces()
        return float(np.max(np.linalg.norm(neb_forces.reshape(-1, 3), axis=1)))
    except Exception:
        return float("nan")


def compute_neb_outputs(record: ReactionRecord, images: list[Atoms], final_max_force: float) -> tuple[dict[str, object], Atoms]:
    energies = np.asarray([float(image.get_potential_energy()) for image in images])
    ts_image_idx = int(np.argmax(energies))
    ts_atoms = images[ts_image_idx].copy()
    ts_atoms.info["split"] = record.split
    ts_atoms.info["formula"] = record.formula
    ts_atoms.info["reaction"] = record.reaction
    ts_atoms.info["neb_ts_image_index"] = ts_image_idx

    model_reactant_energy = float(energies[0])
    model_ts_energy = float(energies[ts_image_idx])
    model_product_energy = float(energies[-1])
    model_forward_barrier = model_ts_energy - model_reactant_energy
    model_reverse_barrier = model_ts_energy - model_product_energy
    dft_forward_barrier = record.dft_ts_energy_eV - record.dft_reactant_energy_eV
    dft_reverse_barrier = record.dft_ts_energy_eV - record.dft_product_energy_eV
    rmsd = kabsch_rmsd(ts_atoms.positions, record.ts_positions)

    row = {
        "split": record.split,
        "formula": record.formula,
        "reaction": record.reaction,
        "n_atoms": int(record.atomic_numbers.shape[0]),
        "ts_image_index": ts_image_idx,
        "ts_rmsd_A": rmsd,
        "dft_forward_barrier_eV": dft_forward_barrier,
        "model_forward_barrier_eV": model_forward_barrier,
        "forward_barrier_error_eV": model_forward_barrier - dft_forward_barrier,
        "dft_reverse_barrier_eV": dft_reverse_barrier,
        "model_reverse_barrier_eV": model_reverse_barrier,
        "reverse_barrier_error_eV": model_reverse_barrier - dft_reverse_barrier,
        "dft_reactant_energy_eV": record.dft_reactant_energy_eV,
        "dft_ts_energy_eV": record.dft_ts_energy_eV,
        "dft_product_energy_eV": record.dft_product_energy_eV,
        "model_reactant_energy_eV": model_reactant_energy,
        "model_ts_energy_eV": model_ts_energy,
        "model_product_energy_eV": model_product_energy,
        "neb_max_force_eV_per_A": final_max_force,
        "final_max_force_eV_per_A": final_max_force,
    }
    return row, ts_atoms


def run_neb_for_reaction(
    record: ReactionRecord,
    calc: MACECalculator,
    num_images: int,
    fmax: float,
    steps: int,
    climb: bool,
) -> tuple[dict[str, object], Atoms]:
    if num_images < 3:
        raise ValueError(f"num_images must include endpoints and be >= 3, got {num_images}")

    reactant, product = make_endpoint_atoms(record)
    images = [reactant]
    images.extend(reactant.copy() for _ in range(num_images - 2))
    images.append(product)

    for image in images:
        image.calc = calc

    stage1_neb = NEB(images, climb=False, allow_shared_calculator=True)
    stage1_neb.interpolate(method="idpp")
    stage1_opt = FIRE(stage1_neb, logfile=None)
    stage1_converged = bool(stage1_opt.run(fmax=fmax, steps=steps))
    stage1_steps = int(getattr(stage1_opt, "nsteps", -1))
    stage1_max_force = compute_neb_max_force(stage1_neb)

    stage2_converged = stage1_converged
    stage2_steps = 0
    stage2_max_force = stage1_max_force
    if climb:
        stage2_neb = NEB(images, climb=True, allow_shared_calculator=True)
        stage2_opt = FIRE(stage2_neb, logfile=None)
        stage2_converged = bool(stage2_opt.run(fmax=fmax, steps=steps))
        stage2_steps = int(getattr(stage2_opt, "nsteps", -1))
        stage2_max_force = compute_neb_max_force(stage2_neb)

    success = stage2_converged if climb else stage1_converged
    final_max_force = stage2_max_force if climb else stage1_max_force
    row, ts_atoms = compute_neb_outputs(record, images, final_max_force)
    row.update(
        {
            "neb_protocol": NEB_PROTOCOL,
            "success": success,
            "converged": success,
            "stage1_converged": stage1_converged,
            "stage2_converged": stage2_converged,
            "stage1_steps": stage1_steps,
            "stage2_steps": stage2_steps,
            "optimizer_steps": stage1_steps + max(stage2_steps, 0),
            "stage1_max_force_eV_per_A": stage1_max_force,
            "stage2_max_force_eV_per_A": stage2_max_force,
            "num_images": num_images,
            "climb": climb,
        }
    )
    return row, ts_atoms


def summarize(values: np.ndarray) -> dict[str, float]:
    abs_values = np.abs(values)
    return {
        "mean": float(np.mean(values)),
        "mae": float(np.mean(abs_values)),
        "rmse": float(np.sqrt(np.mean(values * values))),
        "l4": float(np.power(np.mean(abs_values**4), 0.25)),
    }


def summarize_subset(subset: list[dict[str, object]]) -> dict[str, float]:
    if not subset:
        return {
            "ts_rmsd_mean_A": float("nan"),
            "ts_rmsd_median_A": float("nan"),
            "ts_rmsd_rmse_A": float("nan"),
            "forward_barrier_me_eV": float("nan"),
            "forward_barrier_mae_eV": float("nan"),
            "forward_barrier_rmse_eV": float("nan"),
            "forward_barrier_l4_eV": float("nan"),
            "reverse_barrier_me_eV": float("nan"),
            "reverse_barrier_mae_eV": float("nan"),
            "reverse_barrier_rmse_eV": float("nan"),
            "reverse_barrier_l4_eV": float("nan"),
        }

    rmsd = np.asarray([float(row["ts_rmsd_A"]) for row in subset], dtype=float)
    forward_barrier_error = np.asarray(
        [float(row["forward_barrier_error_eV"]) for row in subset], dtype=float
    )
    reverse_barrier_error = np.asarray(
        [float(row["reverse_barrier_error_eV"]) for row in subset], dtype=float
    )
    forward_barrier_stats = summarize(forward_barrier_error)
    reverse_barrier_stats = summarize(reverse_barrier_error)
    return {
        "ts_rmsd_mean_A": float(np.mean(rmsd)),
        "ts_rmsd_median_A": float(np.median(rmsd)),
        "ts_rmsd_rmse_A": float(np.sqrt(np.mean(rmsd * rmsd))),
        "forward_barrier_me_eV": forward_barrier_stats["mean"],
        "forward_barrier_mae_eV": forward_barrier_stats["mae"],
        "forward_barrier_rmse_eV": forward_barrier_stats["rmse"],
        "forward_barrier_l4_eV": forward_barrier_stats["l4"],
        "reverse_barrier_me_eV": reverse_barrier_stats["mean"],
        "reverse_barrier_mae_eV": reverse_barrier_stats["mae"],
        "reverse_barrier_rmse_eV": reverse_barrier_stats["rmse"],
        "reverse_barrier_l4_eV": reverse_barrier_stats["l4"],
    }


def summarize_rows(detail_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    groups = sorted(
        {
            (row["family"], row["round"], row["split"])
            for row in detail_rows
            if row["status"] == "ok"
        },
        key=lambda x: (str(x[0]), str(x[1]), str(x[2])),
    )
    for family, round_label, split in groups:
        subset = [
            row
            for row in detail_rows
            if row["family"] == family
            and row["round"] == round_label
            and row["split"] == split
            and row["status"] == "ok"
        ]
        success_subset = [
            row
            for row in subset
            if str(row.get("success", row.get("converged", ""))).lower() == "true"
        ]
        all_stats = summarize_subset(subset)
        success_stats = summarize_subset(success_subset)
        summary.append(
            {
                "family": family,
                "round": round_label,
                "split": split,
                "n_reactions": len(subset),
                "n_success": len(success_subset),
                "success_rate": float(len(success_subset) / len(subset)) if subset else float("nan"),
                **{f"all_{key}": value for key, value in all_stats.items()},
                **{f"success_{key}": value for key, value in success_stats.items()},
            }
        )
    return summary


def write_csv(
    path: Path,
    rows: list[dict[str, object]],
    fieldnames: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        if not rows:
            raise ValueError(f"No rows to write for {path}")
        fieldnames = []
        for row in rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summary_rows: list[dict[str, object]]) -> None:
    lines = [
        "# Transition1x NEB TS Evaluation",
        "",
        f"- Dataset: `{DATASET_PATH}`",
        f"- NEB protocol: `{NEB_PROTOCOL}`",
        "- NEB endpoints: DFT reactant and product geometries",
        "- Initialization: IDPP interpolation, then stage-1 plain NEB and stage-2 climbing-image NEB",
        "- Success rate (SR): fraction of reactions with stage-2 NEB convergence",
        "- TS geometry: highest-energy image on the final model NEB path",
        "- TS RMSD and barrier metrics below are computed on successful NEB runs only",
        "",
    ]
    for split in ("val", "test"):
        lines.append(f"## {split.title()} Split")
        lines.append("")
        lines.append(
            "| Family | Round | N | N Success | SR | TS RMSD Mean | TS RMSD Median | Fwd Barrier MAE | Fwd Barrier RMSE | Rev Barrier MAE | Rev Barrier RMSE |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in [r for r in summary_rows if r["split"] == split]:
            lines.append(
                f"| {row['family']} | {row['round']} | {row['n_reactions']} | {row['n_success']} | "
                f"{row['success_rate']:.3f} | {row['success_ts_rmsd_mean_A']:.3f} | {row['success_ts_rmsd_median_A']:.3f} | "
                f"{row['success_forward_barrier_mae_eV']:.3f} | {row['success_forward_barrier_rmse_eV']:.3f} | "
                f"{row['success_reverse_barrier_mae_eV']:.3f} | {row['success_reverse_barrier_rmse_eV']:.3f} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ASE NEB with Transition1x round models and compare TS RMSD/barrier errors."
    )
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--output_dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--enable_cueq", action="store_true", default=True)
    parser.add_argument("--no_enable_cueq", dest="enable_cueq", action="store_false")
    parser.add_argument("--splits", nargs="+", default=["test"], choices=["val", "test"])
    parser.add_argument(
        "--families",
        nargs="+",
        default=["fps_qbc", "qbc", "fps", "random"],
        choices=["fps_qbc", "qbc", "fps", "random"],
    )
    parser.add_argument("--rounds", nargs="+", type=int, default=list(DEFAULT_ROUNDS))
    parser.add_argument("--include_all_data", action="store_true", default=True)
    parser.add_argument("--no_all_data", dest="include_all_data", action="store_false")
    parser.add_argument(
        "--only_all_data",
        action="store_true",
        help="Evaluate only the all-data baseline model.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Load existing details CSV, skip completed OK rows, and rewrite summary.",
    )
    parser.add_argument("--num_images", type=int, default=9)
    parser.add_argument("--fmax", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--no_climb", dest="climb", action="store_false")
    parser.add_argument("--climb", dest="climb", action="store_true", default=True)
    parser.add_argument(
        "--max_reactions",
        type=int,
        default=None,
        help="Optional per-split cap for quick tests.",
    )
    parser.add_argument(
        "--sample_reactions",
        type=int,
        default=DEFAULT_SAMPLE_REACTIONS,
        help="Randomly sample this many reactions per split. Use --sample_reactions 0 to disable.",
    )
    parser.add_argument(
        "--sample_seed",
        type=int,
        default=DEFAULT_SAMPLE_SEED,
        help="Random seed used by --sample_reactions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    current_signature = settings_signature(args)
    splits = set(args.splits)
    rounds = set(args.rounds)
    records = iter_reaction_records(args.dataset, splits)
    if args.max_reactions is not None:
        capped: list[ReactionRecord] = []
        for split in ("val", "test"):
            split_records = [record for record in records if record.split == split]
            capped.extend(split_records[: args.max_reactions])
        records = capped
    sample_reactions = None if args.sample_reactions == 0 else args.sample_reactions
    records = sample_records_by_split(
        records=records,
        sample_reactions=sample_reactions,
        seed=args.sample_seed,
    )
    print(
        f"[INFO] selected {len(records)} reactions from splits={','.join(args.splits)} "
        f"sample_reactions={sample_reactions} sample_seed={args.sample_seed}",
        flush=True,
    )
    print(f"[INFO] neb_protocol={NEB_PROTOCOL}", flush=True)

    families = set() if args.only_all_data else set(args.families)
    specs = build_model_specs(
        families=families,
        rounds=rounds,
        include_all_data=bool(args.include_all_data or args.only_all_data),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ts_dir = args.output_dir / "predicted_ts"
    ts_dir.mkdir(parents=True, exist_ok=True)

    details_path = args.output_dir / "transition1x_neb_details.csv"
    detail_rows: list[dict[str, object]] = (
        load_existing_details(details_path) if args.resume else []
    )
    expected_keys = {
        (
            spec.family,
            spec.round_label,
            record.split,
            record.formula,
            record.reaction,
        )
        for spec in specs
        for record in records
    }
    if args.resume:
        original_detail_count = len(detail_rows)
        detail_rows = [
            row
            for row in detail_rows
            if row_key(row) in expected_keys
            and row.get("settings_signature") == current_signature
        ]
        print(
            f"[INFO] resume loaded {original_detail_count} rows, "
            f"kept {len(detail_rows)} rows for current selection",
            flush=True,
        )
    completed_ok = {
        row_key(row)
        for row in detail_rows
        if row.get("status") == "ok"
    }
    for spec in specs:
        print(
            f"[INFO] model={spec.family} round={spec.round_label} path={spec.model_path}",
            flush=True,
        )
        print(
            f"[INFO] loading MACECalculator device={args.device} "
            f"enable_cueq={bool(args.enable_cueq and args.device != 'cpu')}",
            flush=True,
        )
        calc_start = time.perf_counter()
        calc = MACECalculator(
            model_paths=str(spec.model_path),
            device=args.device,
            default_dtype="float32",
            enable_cueq=bool(args.enable_cueq and args.device != "cpu"),
        )
        print(
            f"[INFO] calculator loaded in {time.perf_counter() - calc_start:.1f} s; "
            f"starting NEB for {len(records)} reactions",
            flush=True,
        )
        ts_atoms_by_split: dict[str, list[Atoms]] = {split: [] for split in splits}
        progress = tqdm(
            records,
            desc=f"{spec.family} round={spec.round_label}",
            total=len(records),
            unit="rxn",
        )
        for record in progress:
            progress.set_postfix(
                split=record.split,
                formula=record.formula,
                reaction=record.reaction,
                refresh=True,
            )
            base = {
                "family": spec.family,
                "round": spec.round_label,
                "model_path": str(spec.model_path),
                "split": record.split,
                "formula": record.formula,
                "reaction": record.reaction,
                "settings_signature": current_signature,
            }
            if row_key(base) in completed_ok:
                tqdm.write("[INFO] skipping completed row")
                continue
            try:
                row, ts_atoms = run_neb_for_reaction(
                    record=record,
                    calc=calc,
                    num_images=args.num_images,
                    fmax=args.fmax,
                    steps=args.steps,
                    climb=args.climb,
                )
                ts_atoms.info.update(
                    {
                        "family": spec.family,
                        "round": spec.round_label,
                        "ts_rmsd_A": row["ts_rmsd_A"],
                        "forward_barrier_error_eV": row["forward_barrier_error_eV"],
                        "reverse_barrier_error_eV": row["reverse_barrier_error_eV"],
                    }
                )
                detail_rows.append({**base, "status": "ok", "error_message": "", **row})
                if row["success"]:
                    ts_atoms_by_split[record.split].append(ts_atoms)
            except Exception as exc:
                detail_rows.append(
                    {
                        **base,
                        "status": "failed",
                        "error_message": repr(exc),
                    }
                )
        for split, atoms_list in ts_atoms_by_split.items():
            if atoms_list:
                write(
                    str(ts_dir / f"{spec.family}_round{spec.round_label}_{split}_ts.xyz"),
                    atoms_list,
                    format="extxyz",
                )

    summary = summarize_rows(detail_rows)
    write_csv(details_path, detail_rows)
    write_csv(
        args.output_dir / "transition1x_neb_summary.csv",
        summary,
        fieldnames=SUMMARY_FIELDNAMES,
    )
    write_markdown(args.output_dir / "transition1x_neb_summary.md", summary)
    print(f"[INFO] wrote {len(detail_rows)} detail rows and {len(summary)} summary rows to {args.output_dir}")


if __name__ == "__main__":
    main()
