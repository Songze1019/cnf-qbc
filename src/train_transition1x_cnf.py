from __future__ import annotations

import argparse
import sys
from pathlib import Path
from pprint import pprint

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.trainers.cnf_runner import run_single_xyz_training
else:
    from src.trainers.cnf_runner import run_single_xyz_training


def default_data_path() -> Path:
    return Path("data/transition1x/train.xyz")


def default_out_root() -> Path:
    return Path("src/transition1x/cnf")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a CNF model on a single transition1x xyz file."
    )
    parser.add_argument("--data-path", type=Path, default=default_data_path())
    parser.add_argument("--out-root", type=Path, default=default_out_root())
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit-configs", type=int, default=0)
    parser.add_argument("--limit-train-batches", type=float, default=1.0)
    parser.add_argument("--optimizer", type=str, default="AdamW")
    parser.add_argument("--scheduler-factor", type=float, default=0.98)
    parser.add_argument("--scheduler-patience", type=int, default=20)
    parser.add_argument("--network", type=str, default="PaiNN")
    parser.add_argument("--hidden-channels", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-rbf", type=int, default=64)
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--num-elements", type=int, default=83)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--max-epochs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr-scheduler-monitor", type=str, default="train/loss")
    parser.add_argument("--checkpoint-every-n-epochs", type=int, default=50)
    parser.add_argument("--print-every-n-epochs", type=int, default=20)
    parser.add_argument("--nll-every-n-epochs", type=int, default=0)
    parser.add_argument("--nll-timesteps", type=int, default=200)
    parser.add_argument("--hutchinson-samples", type=int, default=1)
    parser.add_argument("--log-every-n-steps", type=int, default=1)
    parser.add_argument("--pbc", action="store_true", default=False)
    return parser


def make_runner_config(args: argparse.Namespace) -> dict:
    return {
        "data_path": args.data_path,
        "out_root": args.out_root,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "limit_configs": args.limit_configs,
        "limit_train_batches": args.limit_train_batches,
        "optimizer": args.optimizer,
        "scheduler_factor": args.scheduler_factor,
        "scheduler_patience": args.scheduler_patience,
        "network": args.network,
        "hidden_channels": args.hidden_channels,
        "num_layers": args.num_layers,
        "num_rbf": args.num_rbf,
        "cutoff": args.cutoff,
        "num_elements": args.num_elements,
        "pbc": args.pbc,
        "lr": args.lr,
        "max_epochs": args.max_epochs,
        "seed": args.seed,
        "lr_scheduler_monitor": args.lr_scheduler_monitor,
        "checkpoint_every_n_epochs": args.checkpoint_every_n_epochs,
        "print_every_n_epochs": args.print_every_n_epochs,
        "nll_every_n_epochs": args.nll_every_n_epochs,
        "nll_timesteps": args.nll_timesteps,
        "hutchinson_samples": args.hutchinson_samples,
        "log_every_n_steps": args.log_every_n_steps,
    }


def main() -> None:
    args = build_parser().parse_args()
    config = make_runner_config(args)
    print("Training configuration:")
    pprint(config)
    run_dir = run_single_xyz_training(config)
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
