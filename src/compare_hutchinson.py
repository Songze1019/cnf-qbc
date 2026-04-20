import argparse
import json
import time
import types
from pathlib import Path

import torch

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


def _compute_bpd(
    model: BaseFlow,
    batch,
    nll_timesteps: int,
    hutchinson_samples: int,
) -> torch.Tensor:
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


def _evaluate_bpd_list(
    ckpt_path: Path,
    batches,
    device: torch.device,
    nll_timesteps: int,
    hutchinson_samples: int,
) -> list[float]:
    model = BaseFlow.load_from_checkpoint(str(ckpt_path), map_location=device)
    _patch_hutchinson_divergence(model)
    model.to(device)
    model.eval()

    vals: list[float] = []
    for batch in batches:
        bpd = _compute_bpd(
            model=model,
            batch=batch,
            nll_timesteps=nll_timesteps,
            hutchinson_samples=hutchinson_samples,
        )
        vals.extend(float(v) for v in bpd.detach().float().cpu().tolist())
    return vals


def _patch_hutchinson_divergence(model: BaseFlow) -> None:
    """Patch divergence estimator to support num_samples > 1 safely.

    Upstream implementation uses retain_graph=False for every sample, which raises
    "Trying to backward through the graph a second time" when num_samples > 1.
    """

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


def _rank_desc(values: list[float]) -> list[int]:
    return sorted(range(len(values)), key=lambda i: (values[i], -i), reverse=True)


def _spearman(rank_a: dict[int, int], rank_b: dict[int, int], n: int) -> float:
    sum_d2 = 0
    for i in range(n):
        d = rank_a[i] - rank_b[i]
        sum_d2 += d * d
    return 1.0 - (6.0 * sum_d2) / (n * (n * n - 1))


def _kendall_tau(values_a: list[float], values_b: list[float]) -> float:
    n = len(values_a)
    concordant = 0
    discordant = 0
    for i in range(n):
        ai = values_a[i]
        bi = values_b[i]
        for j in range(i + 1, n):
            prod = (ai - values_a[j]) * (bi - values_b[j])
            if prod > 0:
                concordant += 1
            elif prod < 0:
                discordant += 1
    denom = n * (n - 1) / 2
    if denom == 0:
        return 0.0
    return (concordant - discordant) / denom


def _compare_pair(
    name_a: str, vals_a: list[float], name_b: str, vals_b: list[float]
) -> dict:
    if len(vals_a) != len(vals_b):
        raise RuntimeError(
            f"Length mismatch: {name_a}={len(vals_a)} vs {name_b}={len(vals_b)}"
        )

    n = len(vals_a)
    order_a = _rank_desc(vals_a)
    order_b = _rank_desc(vals_b)

    same_full_order = order_a == order_b

    rank_a = {idx: r + 1 for r, idx in enumerate(order_a)}
    rank_b = {idx: r + 1 for r, idx in enumerate(order_b)}

    topk = {}
    for k in [5, 10, 20, 30, 50]:
        kk = min(k, n)
        set_a = set(order_a[:kk])
        set_b = set(order_b[:kk])
        inter = len(set_a & set_b)
        union = len(set_a | set_b)
        topk[str(kk)] = {
            "overlap": inter,
            "jaccard": float(inter / union) if union > 0 else 0.0,
        }

    abs_rank_diff = [abs(rank_a[i] - rank_b[i]) for i in range(n)]
    abs_rank_diff_sorted = sorted(abs_rank_diff)

    return {
        "a": name_a,
        "b": name_b,
        "same_full_order": same_full_order,
        "spearman": float(_spearman(rank_a, rank_b, n)),
        "kendall_tau": float(_kendall_tau(vals_a, vals_b)),
        "topk": topk,
        "rank_diff": {
            "mean": float(sum(abs_rank_diff) / n),
            "p50": int(abs_rank_diff_sorted[n // 2]),
            "max": int(max(abs_rank_diff_sorted)),
        },
        "top10_a": order_a[: min(10, n)],
        "top10_b": order_b[: min(10, n)],
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Compare BPD ranking consistency across hutchinson_samples for two checkpoints."
        )
    )
    ap.add_argument(
        "--ckpt-a",
        type=str,
        default="src/rmd17/cnf/aspirin/frac5/20260306-232955/checkpoints/flow-frac5-epoch=199.ckpt",
    )
    ap.add_argument(
        "--ckpt-b",
        type=str,
        default="src/rmd17/cnf/aspirin/frac5/20260306-232955/checkpoints/flow-frac5-epoch=299.ckpt",
    )
    ap.add_argument(
        "--label-a",
        type=str,
        default="epoch199",
    )
    ap.add_argument(
        "--label-b",
        type=str,
        default="epoch299",
    )
    ap.add_argument(
        "--xyz",
        type=str,
        default="src/rmd17/data/aspirin_dedup/train_frac5.xyz",
    )
    ap.add_argument("--cutoff", type=float, default=5.0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--nll-timesteps", type=int, default=200)
    ap.add_argument(
        "--hutchinson",
        type=int,
        nargs="+",
        default=[1, 2, 4, 6],
        help="hutchinson_samples list to compare",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic comparison",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Output directory. Default: sibling folder of ckpt-1999",
    )
    args = ap.parse_args()

    ckpt_a = Path(args.ckpt_a)
    ckpt_b = Path(args.ckpt_b)
    xyz_path = Path(args.xyz)
    for p in [ckpt_a, ckpt_b, xyz_path]:
        if not p.exists():
            raise FileNotFoundError(f"Not found: {p}")

    hutchinson_list = [int(x) for x in args.hutchinson]
    if len(hutchinson_list) < 2:
        raise ValueError("Need at least two hutchinson values to compare.")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    configurations = load_from_xyz(str(xyz_path))
    atomic_datas = [
        AtomicData.from_config(cfg, cutoff=args.cutoff) for cfg in configurations
    ]
    _alias_fields_for_flow(atomic_datas)
    data_loader = get_data_loader(
        atomic_datas, batch_size=args.batch_size, shuffle=False
    )
    batches = list(data_loader)

    print(f"Prepared data: {len(atomic_datas)} configs, {len(batches)} batches")
    print(f"Device: {device}")

    label_a = str(args.label_a)
    label_b = str(args.label_b)

    ckpts = [
        (label_a, ckpt_a),
        (label_b, ckpt_b),
    ]

    bpd_table: dict[str, dict[str, list[float]]] = {}
    for ckpt_tag, ckpt_path in ckpts:
        bpd_table[ckpt_tag] = {}
        for hs in hutchinson_list:
            print(f"Evaluating {ckpt_tag} with hutchinson_samples={hs} ...")
            vals = _evaluate_bpd_list(
                ckpt_path=ckpt_path,
                batches=batches,
                device=device,
                nll_timesteps=args.nll_timesteps,
                hutchinson_samples=hs,
            )
            bpd_table[ckpt_tag][str(hs)] = vals

    comparisons = {}
    for ckpt_tag, _ in ckpts:
        pairs = []
        for i in range(len(hutchinson_list)):
            for j in range(i + 1, len(hutchinson_list)):
                a = str(hutchinson_list[i])
                b = str(hutchinson_list[j])
                pair_result = _compare_pair(
                    name_a=f"h{a}",
                    vals_a=bpd_table[ckpt_tag][a],
                    name_b=f"h{b}",
                    vals_b=bpd_table[ckpt_tag][b],
                )
                pairs.append(pair_result)
        comparisons[ckpt_tag] = pairs

    cross_checkpoint_same_h = []
    for hs in hutchinson_list:
        hs_key = str(hs)
        pair_result = _compare_pair(
            name_a=f"{label_a}_h{hs_key}",
            vals_a=bpd_table[label_a][hs_key],
            name_b=f"{label_b}_h{hs_key}",
            vals_b=bpd_table[label_b][hs_key],
        )
        cross_checkpoint_same_h.append(pair_result)

    for ckpt_tag in [label_a, label_b]:
        print(f"\n=== {ckpt_tag} ranking consistency ===")
        for row in comparisons[ckpt_tag]:
            print(
                f"{row['a']} vs {row['b']} | "
                f"same_full_order={row['same_full_order']} | "
                f"spearman={row['spearman']:.4f} | "
                f"kendall={row['kendall_tau']:.4f} | "
                f"top10_overlap={row['topk']['10']['overlap']}"
            )

    print("\n=== Cross-checkpoint consistency at same hutchinson_samples ===")
    for row in cross_checkpoint_same_h:
        print(
            f"{row['a']} vs {row['b']} | "
            f"same_full_order={row['same_full_order']} | "
            f"spearman={row['spearman']:.4f} | "
            f"kendall={row['kendall_tau']:.4f} | "
            f"top10_overlap={row['topk']['10']['overlap']}"
        )

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else ckpt_a.parent.parent / f"compare_hutchinson_{timestamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "configs": {
            "ckpt_a": str(ckpt_a),
            "ckpt_b": str(ckpt_b),
            "label_a": label_a,
            "label_b": label_b,
            "xyz": str(xyz_path),
            "cutoff": args.cutoff,
            "batch_size": args.batch_size,
            "nll_timesteps": args.nll_timesteps,
            "hutchinson": hutchinson_list,
            "seed": args.seed,
            "device": str(device),
        },
        "bpd": bpd_table,
        "comparisons": comparisons,
        "cross_checkpoint_same_h": cross_checkpoint_same_h,
    }
    (out_dir / "compare_hutchinson.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    torch.save(payload, out_dir / "compare_hutchinson.pt")
    print(f"\nWrote results to: {out_dir}")


if __name__ == "__main__":
    main()
