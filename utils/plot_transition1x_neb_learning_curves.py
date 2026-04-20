from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

SUMMARY_CSV = Path("rets/transition1x-neb/transition1x_neb_summary.csv")
OUTPUT_DIR = Path("rets/transition1x-neb/plots")
ROUNDS = (4, 12, 20)
FAMILIES = ["fps_qbc", "qbc", "fps", "random"]
FAMILY_LABELS = {
    "fps_qbc": "FPS+QBC",
    "qbc": "QBC",
    "fps": "FPS",
    "random": "Random",
}
FAMILY_COLORS = {
    "fps_qbc": "#1f77b4",
    "qbc": "#d62728",
    "fps": "#2ca02c",
    "random": "#9467bd",
}
FAMILY_MARKERS = {
    "fps_qbc": "o",
    "qbc": "s",
    "fps": "^",
    "random": "D",
}


def load_rows(summary_csv: Path) -> list[dict[str, str]]:
    with summary_csv.open() as handle:
        return list(csv.DictReader(handle))


def round_sort_key(row: dict[str, str]) -> int:
    return int(row["round"])


def baseline_value(rows: list[dict[str, str]], split: str, key: str) -> float | None:
    matches = [
        row for row in rows if row["family"] == "all_data" and row["split"] == split
    ]
    if not matches:
        return None
    value = matches[0][key]
    if value in ("", "nan", "NaN"):
        return None
    return float(value)


def plot_panel(
    rows: list[dict[str, str]],
    metric_key: str,
    ylabel: str,
    output_name: str,
    output_dir: Path,
) -> Path:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.linewidth": 1.0,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "savefig.dpi": 600,
            "figure.dpi": 600,
        }
    )

    splits = [split for split in ("val", "test") if any(row["split"] == split for row in rows)]
    if not splits:
        raise ValueError("No val/test rows found in summary CSV")
    figsize = (4.8, 3.4) if len(splits) == 1 else (7.2, 3.4)
    fig, axes_obj = plt.subplots(
        1,
        len(splits),
        figsize=figsize,
        dpi=600,
        sharex=True,
        squeeze=False,
    )
    axes = axes_obj[0]
    for ax, split in zip(axes, splits):
        for family in FAMILIES:
            subset = [
                row for row in rows if row["family"] == family and row["split"] == split
            ]
            subset.sort(key=round_sort_key)
            xs = [int(row["round"]) for row in subset]
            ys = [float(row[metric_key]) for row in subset]
            ax.plot(
                xs,
                ys,
                marker=FAMILY_MARKERS[family],
                markersize=4,
                linewidth=1.8,
                color=FAMILY_COLORS[family],
                label=FAMILY_LABELS[family],
            )
        base = baseline_value(rows, split, metric_key)
        if base is not None:
            ax.axhline(
                base,
                color="black",
                linestyle="--",
                linewidth=1.2,
                label="All-data Baseline" if split == splits[0] else None,
            )
        ax.text(
            0.03,
            0.94,
            split.upper(),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
        )
        ax.set_xlabel("Round")
        ax.set_ylabel(ylabel)
        ax.set_xticks(list(ROUNDS))
        ax.set_xlim(-0.5, 20.5)
        ax.grid(True, alpha=0.2, linewidth=0.6)

    handles, labels = axes[0].get_legend_handles_labels()
    ncol = min(5, max(1, len(labels)))
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=ncol,
        frameon=False,
        bbox_to_anchor=(0.5, 1.06),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_name
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Transition1x NEB TS RMSD and barrier learning curves."
    )
    parser.add_argument("--summary_csv", type=Path, default=SUMMARY_CSV)
    parser.add_argument("--output_dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.summary_csv)
    outputs = [
        plot_panel(
            rows,
            metric_key="success_ts_rmsd_mean_A",
            ylabel="TS RMSD Mean (A)",
            output_name="transition1x_neb_ts_rmsd_mean_learning_curve.png",
            output_dir=args.output_dir,
        ),
        plot_panel(
            rows,
            metric_key="success_ts_rmsd_rmse_A",
            ylabel="TS RMSD RMSE (A)",
            output_name="transition1x_neb_ts_rmsd_rmse_learning_curve.png",
            output_dir=args.output_dir,
        ),
        plot_panel(
            rows,
            metric_key="success_forward_barrier_mae_eV",
            ylabel="Forward Barrier MAE (eV)",
            output_name="transition1x_neb_forward_barrier_mae_learning_curve.png",
            output_dir=args.output_dir,
        ),
        plot_panel(
            rows,
            metric_key="success_forward_barrier_rmse_eV",
            ylabel="Forward Barrier RMSE (eV)",
            output_name="transition1x_neb_forward_barrier_rmse_learning_curve.png",
            output_dir=args.output_dir,
        ),
        plot_panel(
            rows,
            metric_key="success_reverse_barrier_mae_eV",
            ylabel="Reverse Barrier MAE (eV)",
            output_name="transition1x_neb_reverse_barrier_mae_learning_curve.png",
            output_dir=args.output_dir,
        ),
        plot_panel(
            rows,
            metric_key="success_reverse_barrier_rmse_eV",
            ylabel="Reverse Barrier RMSE (eV)",
            output_name="transition1x_neb_reverse_barrier_rmse_learning_curve.png",
            output_dir=args.output_dir,
        ),
        plot_panel(
            rows,
            metric_key="success_rate",
            ylabel="NEB SR",
            output_name="transition1x_neb_success_rate_learning_curve.png",
            output_dir=args.output_dir,
        ),
    ]
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
