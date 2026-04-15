from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROUND_SUMMARY_CSV = Path("rets/transition1x_round_model_reaction_eval/transition1x_round_model_reaction_summary.csv")
BASELINE_CSV = Path("rets/transition1x_reaction_energies_scratch_lr1e-2_stage2.csv")
OUTPUT_DIR = Path("rets/transition1x_round_model_reaction_eval/plots")

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
SPLITS = ["val", "test"]
SPLIT_LABELS = {"val": "Validation", "test": "Test"}
METRIC_GROUPS = {
    "mae": (
        "MAE (eV)",
        [("forward", "forward_mae_eV", "Forward Barrier"),
         ("reaction", "reaction_mae_eV", "Reaction Energy"),
         ("reverse", "reverse_mae_eV", "Reverse Barrier")],
    ),
    "rmse": (
        "RMSE (eV)",
        [("forward", "forward_rmse_eV", "Forward Barrier"),
         ("reaction", "reaction_rmse_eV", "Reaction Energy"),
         ("reverse", "reverse_rmse_eV", "Reverse Barrier")],
    ),
    "l4": (
        "L4 (eV)",
        [("forward", "forward_l4_eV", "Forward Barrier"),
         ("reaction", "reaction_l4_eV", "Reaction Energy"),
         ("reverse", "reverse_l4_eV", "Reverse Barrier")],
    ),
}
BASELINE_ERROR_KEYS = {
    "forward": "forward_barrier_error_eV",
    "reaction": "reaction_energy_error_eV",
    "reverse": "reverse_barrier_error_eV",
}


def load_round_rows() -> list[dict[str, str]]:
    with ROUND_SUMMARY_CSV.open() as handle:
        return list(csv.DictReader(handle))


def summarize_errors(values: list[float]) -> tuple[float, float, float]:
    n = len(values)
    mae = sum(abs(v) for v in values) / n
    rmse = (sum(v * v for v in values) / n) ** 0.5
    l4 = (sum(abs(v) ** 4 for v in values) / n) ** 0.25
    return mae, rmse, l4


def load_baseline_metrics() -> dict[str, dict[str, dict[str, float]]]:
    with BASELINE_CSV.open() as handle:
        rows = list(csv.DictReader(handle))

    baseline: dict[str, dict[str, dict[str, float]]] = {}
    for split in SPLITS:
        subset = [row for row in rows if row["split"] == split]
        split_metrics: dict[str, dict[str, float]] = {}
        for quantity, error_key in BASELINE_ERROR_KEYS.items():
            values = [float(row[error_key]) for row in subset]
            mae, rmse, l4 = summarize_errors(values)
            split_metrics[quantity] = {"mae": mae, "rmse": rmse, "l4": l4}
        baseline[split] = split_metrics
    return baseline


def make_plot(metric_name: str, round_rows: list[dict[str, str]], baseline_metrics: dict[str, dict[str, dict[str, float]]]) -> Path:
    ylabel, columns = METRIC_GROUPS[metric_name]

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

    fig, axes = plt.subplots(2, 3, figsize=(9.6, 6.8), dpi=600, sharex=True)

    for row_idx, split in enumerate(SPLITS):
        for col_idx, (quantity, column_key, column_label) in enumerate(columns):
            ax = axes[row_idx, col_idx]
            for family in FAMILIES:
                subset = [r for r in round_rows if r["split"] == split and r["family"] == family]
                subset.sort(key=lambda r: int(r["round"]))
                xs = [int(r["round"]) for r in subset]
                ys = [float(r[column_key]) for r in subset]
                ax.plot(
                    xs,
                    ys,
                    marker="o",
                    markersize=4,
                    linewidth=1.8,
                    color=FAMILY_COLORS[family],
                    label=FAMILY_LABELS[family],
                )

            baseline_value = baseline_metrics[split][quantity][metric_name]
            ax.axhline(
                baseline_value,
                color="black",
                linewidth=1.2,
                linestyle="--",
                label="All-data Baseline" if (row_idx == 0 and col_idx == 0) else None,
            )
            ax.set_title(f"{SPLIT_LABELS[split]}: {column_label}", fontsize=11)
            ax.grid(True, alpha=0.2, linewidth=0.6)
            ax.set_xlim(-0.5, 20.5)
            ax.set_xticks([0, 4, 8, 12, 16, 20])
            if row_idx == 1:
                ax.set_xlabel("Round")
            if col_idx == 0:
                ax.set_ylabel(ylabel)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"transition1x_round_learning_curve_{metric_name}.png"
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    round_rows = load_round_rows()
    baseline_metrics = load_baseline_metrics()
    for metric_name in ("mae", "rmse", "l4"):
        output_path = make_plot(metric_name, round_rows, baseline_metrics)
        print(output_path)


if __name__ == "__main__":
    main()
