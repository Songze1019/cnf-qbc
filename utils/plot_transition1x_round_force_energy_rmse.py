from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt

ROUNDS = (0, 4, 8, 12, 16, 20)
OUTPUT_DIR = Path("rets/transition1x_round_model_reaction_eval/plots")
SUMMARY_CSV = Path("rets/transition1x_round_model_reaction_eval/transition1x_round_force_energy_rmse.csv")
FAMILIES = {
    "fps_qbc": Path("rets/transition1x-fps_qbc-rounds-lr1e-2-stage2"),
    "qbc": Path("rets/transition1x-qbc-rounds-lr1e-2-stage2"),
    "fps": Path("rets/transition1x-fps-rounds-lr1e-2-stage2"),
    "random": Path("rets/transition1x-random-rounds-lr1e-2-stage2"),
}
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
ROW_RE = re.compile(
    r"\|\s*(?P<name>train_Default|valid_Default|Default_Default)\s*"
    r"\|\s*(?P<energy>[0-9.]+)\s*"
    r"\|\s*(?P<force>[0-9.]+)\s*"
    r"\|"
)


def parse_stage_two_metrics(log_path: Path) -> dict[str, tuple[float, float]]:
    text = log_path.read_text(encoding="utf-8")
    marker = "Loaded Stage two model"
    marker_idx = text.rfind(marker)
    if marker_idx < 0:
        raise ValueError(f"Stage two evaluation marker not found in {log_path}")

    stage_two_text = text[marker_idx:]
    metrics: dict[str, tuple[float, float]] = {}
    for match in ROW_RE.finditer(stage_two_text):
        metrics[match.group("name")] = (
            float(match.group("energy")),
            float(match.group("force")),
        )

    required = {"valid_Default", "Default_Default"}
    missing = required - set(metrics)
    if missing:
        raise ValueError(f"Missing final stage-two metrics {sorted(missing)} in {log_path}")
    return metrics


def collect_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for family, root in FAMILIES.items():
        for round_idx in ROUNDS:
            log_path = root / f"round{round_idx}" / "logs" / f"transition1x_round{round_idx}_run-0.log"
            if not log_path.exists():
                raise FileNotFoundError(f"Missing log file: {log_path}")
            metrics = parse_stage_two_metrics(log_path)
            valid_e, valid_f = metrics["valid_Default"]
            test_e, test_f = metrics["Default_Default"]
            rows.append(
                {
                    "family": family,
                    "round": round_idx,
                    "valid_energy_rmse_meV_per_atom": valid_e,
                    "valid_force_rmse_meV_per_A": valid_f,
                    "test_energy_rmse_meV_per_atom": test_e,
                    "test_force_rmse_meV_per_A": test_f,
                    "log_path": str(log_path),
                }
            )
    return rows


def write_csv(rows: list[dict[str, object]]) -> None:
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_rows(rows: list[dict[str, object]]) -> Path:
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

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), dpi=600, sharex=True)
    panels = [
        ("valid", "energy", "Energy RMSE (meV/atom)"),
        ("valid", "force", "Force RMSE (meV/A)"),
        ("test", "energy", "Energy RMSE (meV/atom)"),
        ("test", "force", "Force RMSE (meV/A)"),
    ]

    for ax, (split, quantity, ylabel) in zip(axes.ravel(), panels):
        key = f"{split}_{quantity}_rmse_meV_per_atom" if quantity == "energy" else f"{split}_{quantity}_rmse_meV_per_A"
        for family in FAMILIES:
            subset = [row for row in rows if row["family"] == family]
            subset.sort(key=lambda row: int(row["round"]))
            xs = [int(row["round"]) for row in subset]
            ys = [float(row[key]) for row in subset]
            ax.plot(
                xs,
                ys,
                marker=FAMILY_MARKERS[family],
                markersize=4,
                linewidth=1.8,
                color=FAMILY_COLORS[family],
                label=FAMILY_LABELS[family],
            )
        ax.text(
            0.03,
            0.94,
            f"{split.upper()} {'E' if quantity == 'energy' else 'F'}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
        )
        ax.set_ylabel(ylabel)
        ax.set_xlim(-0.5, 20.5)
        ax.set_xticks(list(ROUNDS))
        ax.grid(True, alpha=0.2, linewidth=0.6)

    for ax in axes[1, :]:
        ax.set_xlabel("Round")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "transition1x_round_energy_force_rmse.png"
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    rows = collect_rows()
    write_csv(rows)
    output_path = plot_rows(rows)
    print(SUMMARY_CSV)
    print(output_path)


if __name__ == "__main__":
    main()
