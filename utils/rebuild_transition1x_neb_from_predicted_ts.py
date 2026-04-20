from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import numpy as np
from ase.io import iread

DEFAULT_OUTPUT_DIR = Path("rets/transition1x-neb-idpp2stage")
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
FILENAME_RE = re.compile(r"^(?P<family>.+)_round(?P<round>.+)_(?P<split>val|test)_ts\.xyz$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild Transition1x NEB summary from predicted_ts extxyz files."
    )
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--expected_reactions",
        type=int,
        default=50,
        help="Expected number of reactions per spec for success-rate reconstruction.",
    )
    return parser.parse_args()


def parse_float(value: object) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text in ("", "nan", "NaN", "None"):
        return float("nan")
    return float(text)


def summarize(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    abs_arr = np.abs(arr)
    return {
        "mean": float(np.mean(arr)),
        "mae": float(np.mean(abs_arr)),
        "rmse": float(np.sqrt(np.mean(arr * arr))),
        "l4": float(np.power(np.mean(abs_arr**4), 0.25)),
    }


def summarize_success_rows(rows: list[dict[str, object]]) -> dict[str, float]:
    if not rows:
        return {
            "success_ts_rmsd_mean_A": float("nan"),
            "success_ts_rmsd_median_A": float("nan"),
            "success_ts_rmsd_rmse_A": float("nan"),
            "success_forward_barrier_me_eV": float("nan"),
            "success_forward_barrier_mae_eV": float("nan"),
            "success_forward_barrier_rmse_eV": float("nan"),
            "success_forward_barrier_l4_eV": float("nan"),
            "success_reverse_barrier_me_eV": float("nan"),
            "success_reverse_barrier_mae_eV": float("nan"),
            "success_reverse_barrier_rmse_eV": float("nan"),
            "success_reverse_barrier_l4_eV": float("nan"),
        }

    ts_rmsd = [parse_float(row["ts_rmsd_A"]) for row in rows]
    forward_error = [parse_float(row["forward_barrier_error_eV"]) for row in rows]
    reverse_error = [parse_float(row["reverse_barrier_error_eV"]) for row in rows]
    forward_stats = summarize(forward_error)
    reverse_stats = summarize(reverse_error)
    ts_rmsd_arr = np.asarray(ts_rmsd, dtype=float)
    return {
        "success_ts_rmsd_mean_A": float(np.mean(ts_rmsd_arr)),
        "success_ts_rmsd_median_A": float(np.median(ts_rmsd_arr)),
        "success_ts_rmsd_rmse_A": float(np.sqrt(np.mean(ts_rmsd_arr * ts_rmsd_arr))),
        "success_forward_barrier_me_eV": forward_stats["mean"],
        "success_forward_barrier_mae_eV": forward_stats["mae"],
        "success_forward_barrier_rmse_eV": forward_stats["rmse"],
        "success_forward_barrier_l4_eV": forward_stats["l4"],
        "success_reverse_barrier_me_eV": reverse_stats["mean"],
        "success_reverse_barrier_mae_eV": reverse_stats["mae"],
        "success_reverse_barrier_rmse_eV": reverse_stats["rmse"],
        "success_reverse_barrier_l4_eV": reverse_stats["l4"],
    }


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summary_rows: list[dict[str, object]], expected_reactions: int) -> None:
    lines = [
        "# Transition1x NEB TS Evaluation",
        "",
        "- Rebuilt from `predicted_ts/*.xyz` only",
        f"- Expected reactions per spec: `{expected_reactions}`",
        "- TS RMSD and barrier metrics are computed on successful NEB runs only",
        "",
    ]
    for split in ("val", "test"):
        split_rows = [row for row in summary_rows if row["split"] == split]
        lines.append(f"## {split.title()} Split")
        lines.append("")
        lines.append(
            "| Family | Round | N | N Success | SR | TS RMSD Mean | TS RMSD Median | Fwd Barrier MAE | Fwd Barrier RMSE | Rev Barrier MAE | Rev Barrier RMSE |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in split_rows:
            lines.append(
                f"| {row['family']} | {row['round']} | {row['n_reactions']} | {row['n_success']} | "
                f"{row['success_rate']:.3f} | {row['success_ts_rmsd_mean_A']:.3f} | {row['success_ts_rmsd_median_A']:.3f} | "
                f"{row['success_forward_barrier_mae_eV']:.3f} | {row['success_forward_barrier_rmse_eV']:.3f} | "
                f"{row['success_reverse_barrier_mae_eV']:.3f} | {row['success_reverse_barrier_rmse_eV']:.3f} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    predicted_dir = args.output_dir / "predicted_ts"
    if not predicted_dir.exists():
        raise FileNotFoundError(f"Missing predicted_ts directory: {predicted_dir}")

    success_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for path in sorted(predicted_dir.glob("*_ts.xyz")):
        match = FILENAME_RE.match(path.name)
        if not match:
            continue
        family = match.group("family")
        round_label = match.group("round")
        split = match.group("split")
        atoms_list = list(iread(path, index=":"))
        for atoms in atoms_list:
            success_rows.append(
                {
                    "family": family,
                    "round": round_label,
                    "split": split,
                    "formula": atoms.info.get("formula", ""),
                    "reaction": atoms.info.get("reaction", ""),
                    "ts_rmsd_A": parse_float(atoms.info.get("ts_rmsd_A")),
                    "forward_barrier_error_eV": parse_float(atoms.info.get("forward_barrier_error_eV")),
                    "reverse_barrier_error_eV": parse_float(atoms.info.get("reverse_barrier_error_eV")),
                }
            )
        success_stats = summarize_success_rows([row for row in success_rows if row["family"] == family and row["round"] == round_label and row["split"] == split])
        summary_rows.append(
            {
                "family": family,
                "round": round_label,
                "split": split,
                "n_reactions": args.expected_reactions,
                "n_success": len(atoms_list),
                "success_rate": float(len(atoms_list) / args.expected_reactions) if args.expected_reactions else float("nan"),
                "all_ts_rmsd_mean_A": float("nan"),
                "all_ts_rmsd_median_A": float("nan"),
                "all_ts_rmsd_rmse_A": float("nan"),
                "all_forward_barrier_me_eV": float("nan"),
                "all_forward_barrier_mae_eV": float("nan"),
                "all_forward_barrier_rmse_eV": float("nan"),
                "all_forward_barrier_l4_eV": float("nan"),
                "all_reverse_barrier_me_eV": float("nan"),
                "all_reverse_barrier_mae_eV": float("nan"),
                "all_reverse_barrier_rmse_eV": float("nan"),
                "all_reverse_barrier_l4_eV": float("nan"),
                **success_stats,
            }
        )

    success_details_path = args.output_dir / "transition1x_neb_success_details.csv"
    summary_csv_path = args.output_dir / "transition1x_neb_summary.csv"
    summary_md_path = args.output_dir / "transition1x_neb_summary.md"
    write_csv(
        success_details_path,
        success_rows,
        fieldnames=[
            "family",
            "round",
            "split",
            "formula",
            "reaction",
            "ts_rmsd_A",
            "forward_barrier_error_eV",
            "reverse_barrier_error_eV",
        ],
    )
    write_csv(summary_csv_path, summary_rows, fieldnames=SUMMARY_FIELDNAMES)
    write_markdown(summary_md_path, summary_rows, expected_reactions=args.expected_reactions)
    print(success_details_path)
    print(summary_csv_path)
    print(summary_md_path)


if __name__ == "__main__":
    main()
