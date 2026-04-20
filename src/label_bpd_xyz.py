import argparse
import json
import re
import types
from pathlib import Path
from typing import cast

import torch
from ase import Atoms
from ase.io import read, write
from tqdm import tqdm

from likelihood.model.normalizing_flow import BaseFlow
from likelihood.model.utils import (
    AtomicData,
    center_of_mass,
    get_data_loader,
    load_from_xyz,
)


torch.set_float32_matmul_precision("high")


def _alias_fields_for_flow(atomic_datas: list[AtomicData]) -> None:
    for data in atomic_datas:
        setattr(data, "atomic_numbers", data.node_attrs)
        setattr(data, "pos", data.positions)


def _patch_hutchinson_divergence(model: BaseFlow) -> None:
    def _patched_hutchinson_divergence(self, v, x, batch, num_graphs, num_samples=1):
        if num_samples < 1:
            raise ValueError("num_samples must be >= 1")

        div = torch.zeros(num_graphs, device=x.device, dtype=x.dtype)
        for s in range(num_samples):
            eps = torch.randn_like(x)
            eps = center_of_mass(eps, batch=batch)

            inner = (v * eps).sum()
            grad = torch.autograd.grad(
                inner,
                x,
                create_graph=False,
                retain_graph=(s < num_samples - 1),
                only_inputs=True,
            )[0]
            node_contrib = (grad * eps).sum(dim=-1)
            div = div + self._graph_sum(
                node_contrib, batch=batch, num_graphs=num_graphs
            )

        return div / float(num_samples)

    model._hutchinson_divergence = types.MethodType(
        _patched_hutchinson_divergence, model
    )


def _compute_bpd(model: BaseFlow, batch, nll_timesteps: int, hutchinson_samples: int):
    batch = batch.to(model.device)
    batch_index = batch.batch if hasattr(batch, "batch") else batch["batch"]

    num_graphs = int(batch_index.max().item()) + 1
    n_atoms = torch.bincount(batch_index, minlength=num_graphs).to(model.device)
    dof = (3 * n_atoms - 3).clamp(min=1).to(torch.get_default_dtype())
    ln2 = torch.log(torch.tensor(2.0, device=model.device))

    with torch.enable_grad():
        _nll, nll_noconst = model.nll(
            atomic_numbers=batch["atomic_numbers"],
            pos=batch["pos"],
            batch=batch_index,
            cell=batch["cell"] if hasattr(batch, "cell") else None,
            n_timesteps=nll_timesteps,
            hutchinson_samples=hutchinson_samples,
            include_prior_constant=None,
        )
    return nll_noconst / (dof * ln2)


def _resolve_latest_run_dir(frac_dir: Path) -> Path:
    subdirs = [p for p in frac_dir.iterdir() if p.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f"No run directories under: {frac_dir}")
    return sorted(subdirs)[-1]


def _resolve_xyz_path(data_dir: Path, frac: str) -> Path:
    if frac == "frac100":
        path = data_dir / "train.xyz"
    else:
        path = data_dir / f"train_{frac}.xyz"
    if not path.exists():
        raise FileNotFoundError(f"XYZ not found for {frac}: {path}")
    return path


def _resolve_ckpt_path(run_dir: Path, ckpt_name: str) -> Path:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    if ckpt_name:
        ckpt_path = ckpt_dir / ckpt_name
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        return ckpt_path

    ckpts = sorted(ckpt_dir.glob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No .ckpt files in: {ckpt_dir}")

    def key(p: Path):
        m = re.search(r"epoch=(\d+)", p.name)
        if m:
            return (0, int(m.group(1)), p.name)
        return (1, 10**12, p.name)

    return sorted(ckpts, key=key)[-1]


def _evaluate_bpd_list(
    ckpt_path: Path,
    xyz_path: Path,
    cutoff: float,
    batch_size: int,
    nll_timesteps: int,
    hutchinson_samples: int,
    device: torch.device,
    progress_desc: str,
) -> list[float]:
    model = BaseFlow.load_from_checkpoint(str(ckpt_path), map_location=device)
    _patch_hutchinson_divergence(model)
    model.to(device)
    model.eval()

    configurations = load_from_xyz(str(xyz_path))
    atomic_datas = [
        AtomicData.from_config(cfg, cutoff=cutoff) for cfg in configurations
    ]
    _alias_fields_for_flow(atomic_datas)
    loader = get_data_loader(atomic_datas, batch_size=batch_size, shuffle=False)

    bpds: list[float] = []
    for batch in tqdm(loader, desc=progress_desc, leave=False):
        bpd = _compute_bpd(
            model=model,
            batch=batch,
            nll_timesteps=nll_timesteps,
            hutchinson_samples=hutchinson_samples,
        )
        bpds.extend(float(v) for v in bpd.detach().float().cpu().tolist())

    return bpds


def _annotate_xyz_with_bpd(
    xyz_in: Path, bpd_values: list[float], xyz_out: Path
) -> None:
    atoms_list = cast(list[Atoms], read(str(xyz_in), index=":"))
    if len(atoms_list) != len(bpd_values):
        raise RuntimeError(
            f"Length mismatch: xyz has {len(atoms_list)} frames, bpd has {len(bpd_values)} values"
        )

    for i, atoms in enumerate(atoms_list):
        atoms.info = dict(atoms.info)
        atoms.info["bpd"] = float(bpd_values[i])

    xyz_out.parent.mkdir(parents=True, exist_ok=True)
    write(str(xyz_out), atoms_list, format="extxyz")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Label fraction-specific xyz files with BPD from corresponding CNF checkpoints."
    )
    ap.add_argument(
        "--cnf-root",
        type=str,
        default="src/rmd17/cnf/aspirin",
        help="Root directory containing frac5/frac10/... subdirectories",
    )
    ap.add_argument(
        "--data-dir",
        type=str,
        default="src/rmd17/data/aspirin_dedup",
        help="Directory containing train_frac*.xyz and train.xyz",
    )
    ap.add_argument(
        "--fractions",
        type=str,
        nargs="+",
        default=["frac5", "frac10", "frac20", "frac40", "frac100"],
    )
    ap.add_argument(
        "--ckpt-name",
        type=str,
        default="last.ckpt",
        help="Checkpoint file under each run's checkpoints/. Set empty string to auto-pick latest epoch ckpt.",
    )
    ap.add_argument("--cutoff", type=float, default=5.0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--nll-timesteps", type=int, default=200)
    ap.add_argument("--hutchinson-samples", type=int, default=1)
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used for Hutchinson noise (matters when hutchinson_samples>0)",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Output directory. Default: <cnf-root>/bpd_labeled",
    )
    args = ap.parse_args()

    cnf_root = Path(args.cnf_root)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir) if args.out_dir else (cnf_root / "bpd_labeled")
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Output dir: {out_dir}")

    summary: dict[str, dict] = {}
    for frac in tqdm(args.fractions, desc="fractions"):
        frac_dir = cnf_root / frac
        if not frac_dir.exists():
            print(f"[WARN] Skip missing fraction directory: {frac_dir}")
            continue

        run_dir = _resolve_latest_run_dir(frac_dir)
        ckpt_path = _resolve_ckpt_path(run_dir, args.ckpt_name)
        xyz_path = _resolve_xyz_path(data_dir, frac)

        print(f"\n[{frac}] run={run_dir.name}")
        print(f"[{frac}] ckpt={ckpt_path}")
        print(f"[{frac}] xyz={xyz_path}")

        bpds = _evaluate_bpd_list(
            ckpt_path=ckpt_path,
            xyz_path=xyz_path,
            cutoff=args.cutoff,
            batch_size=args.batch_size,
            nll_timesteps=args.nll_timesteps,
            hutchinson_samples=args.hutchinson_samples,
            device=device,
            progress_desc=f"{frac} batches",
        )

        out_xyz = out_dir / f"{frac}_with_bpd.xyz"
        _annotate_xyz_with_bpd(xyz_path, bpds, out_xyz)

        t = torch.tensor(bpds)
        stat = {
            "n": int(t.numel()),
            "mean": float(t.mean().item()),
            "std": float(t.std(unbiased=False).item()),
            "min": float(t.min().item()),
            "max": float(t.max().item()),
            "ckpt": str(ckpt_path),
            "xyz": str(xyz_path),
            "out_xyz": str(out_xyz),
        }
        summary[frac] = stat
        print(
            f"[{frac}] wrote={out_xyz} | n={stat['n']} mean={stat['mean']:.6f} std={stat['std']:.6f}"
        )

    payload = {
        "configs": {
            "cnf_root": str(cnf_root),
            "data_dir": str(data_dir),
            "fractions": list(args.fractions),
            "ckpt_name": args.ckpt_name,
            "cutoff": args.cutoff,
            "batch_size": args.batch_size,
            "nll_timesteps": args.nll_timesteps,
            "hutchinson_samples": args.hutchinson_samples,
            "seed": args.seed,
            "device": str(device),
        },
        "summary": summary,
    }
    (out_dir / "bpd_label_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(f"\nWrote summary: {out_dir / 'bpd_label_summary.json'}")


if __name__ == "__main__":
    main()
