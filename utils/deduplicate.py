import os

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.io import write
from mace.calculators import MACECalculator
from mace import data as mace_data
from mace.tools import torch_geometric
from e3nn import o3


# ============================================================
# 配置
# ============================================================
ROOT = Path(__file__).parent.resolve()
DEFAULT_NPZ = ROOT / "rmd17" / "npz_data" / "rmd17_aspirin.npz"
DEFAULT_MODEL = "/home/xmcao/huosongze/.mace/model/MACE-OFF23_small.model"
DEFAULT_OUTPUT = ROOT / "data" / "aspirin_dedup.xyz"
DESCRIPTOR_CACHE = ROOT / "data" / "aspirin_mace_descriptors.npz"

# 原子序数 -> 元素符号 映射
ATOMIC_NUMBERS_TO_SYMBOLS = {
    1: "H",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    15: "P",
    16: "S",
    17: "Cl",
}

# 单位转换: kcal/mol -> eV (同样适用于 kcal/mol/A -> eV/A)
KCAL_MOL_TO_EV = 0.04336411530


def parse_args():
    p = argparse.ArgumentParser(
        description="Deduplicate rMD17 aspirin structures via MACE descriptors"
    )
    p.add_argument("--npz", type=str, default=str(DEFAULT_NPZ), help="Input npz file")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL, help="MACE model path")
    p.add_argument(
        "--output", type=str, default=str(DEFAULT_OUTPUT), help="Output XYZ file"
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Euclidean distance threshold in standardized descriptor space. "
        "Structures closer than this are considered duplicates. Default 0.1.",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for descriptor extraction",
    )
    p.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu")
    p.add_argument(
        "--no_cache", action="store_true", help="Ignore cached descriptors, recompute"
    )
    return p.parse_args()


# ============================================================
# Step 0: 将 npz 转换为 ASE Atoms 列表
# ============================================================
def npz_to_atoms(npz_path):
    """
    读取 rMD17 npz 文件，返回 ASE Atoms 列表。
    每个 Atoms 对象包含坐标、能量 (info) 和力 (arrays)。
    """
    print(f"\n{'=' * 60}")
    print(f"Step 0: Loading npz -> ASE Atoms")
    print(f"  File: {npz_path}")
    print(f"{'=' * 60}")

    t0 = time.time()
    data = np.load(npz_path)

    nuclear_charges = data["nuclear_charges"]  # (n_atoms,)
    coords = data["coords"]  # (n_structures, n_atoms, 3)  [Angstrom]
    energies = data["energies"] * KCAL_MOL_TO_EV  # kcal/mol -> eV
    forces = data["forces"] * KCAL_MOL_TO_EV  # kcal/mol/A -> eV/A

    # 居中: 每个结构减去各自的几何中心 (平移不变, 不影响能量/力)
    centers = coords.mean(axis=1, keepdims=True)  # (N, 1, 3)
    coords = coords - centers

    n_structures = len(energies)
    n_atoms = len(nuclear_charges)

    # 构建元素符号列表
    symbols = [ATOMIC_NUMBERS_TO_SYMBOLS[int(z)] for z in nuclear_charges]

    print(f"  Nuclear charges: {nuclear_charges}")
    print(f"  Symbols: {''.join(symbols)}")
    print(f"  Structures: {n_structures}, Atoms per structure: {n_atoms}")
    print(f"  Units: energy [eV], forces [eV/A] (converted from kcal/mol)")

    atoms_list = []
    for i in range(n_structures):
        atoms = Atoms(symbols=symbols, positions=coords[i])
        atoms.info["energy"] = float(energies[i])
        atoms.info["REF_energy"] = float(energies[i])
        atoms.info["config_type"] = "rmd17_aspirin"
        atoms.info["original_index"] = i
        atoms.arrays["forces"] = forces[i].copy()
        atoms.arrays["REF_forces"] = forces[i].copy()
        atoms_list.append(atoms)

    print(f"  Converted {len(atoms_list)} structures in {time.time() - t0:.1f}s")
    return atoms_list


# ============================================================
# Step 1: 提取 MACE 描述符 (batched)
# ============================================================
def extract_descriptors_batched(atoms_list, model_path, device="cuda", batch_size=64):
    """
    用 MACE 模型批量提取 invariant 描述符，对每个结构按原子取平均，
    返回 (N_structures, D) 的 numpy array。
    """
    print(f"\n{'=' * 60}")
    print(f"Step 1: Extracting MACE descriptors")
    print(f"  Structures: {len(atoms_list)}")
    print(f"  Model: {model_path}")
    print(f"  Device: {device}, Batch size: {batch_size}")
    print(f"{'=' * 60}")

    calc = MACECalculator(model_paths=model_path, device=device, enable_cueq=True)
    model = calc.models[0]
    model.eval()

    # 获取 invariant 特征维度信息
    irreps_out = o3.Irreps(str(model.products[0].linear.irreps_out))
    l_max = irreps_out.lmax
    num_inv = irreps_out.dim // (l_max + 1) ** 2
    num_interactions = int(model.num_interactions)
    per_layer_dims = [irreps_out.dim] * num_interactions
    per_layer_dims[-1] = num_inv
    total_inv_dim = sum(per_layer_dims)
    print(
        f"  Descriptor dim per atom: {total_inv_dim} (l_max={l_max}, {num_interactions} layers)"
    )

    # 预处理所有结构为 AtomicData
    print("  Building atomic data objects ...")
    keyspec = mace_data.KeySpecification(
        info_keys=calc.info_keys, arrays_keys=calc.arrays_keys
    )
    atomic_data_list = []
    for atoms in atoms_list:
        config = mace_data.config_from_atoms(
            atoms, key_specification=keyspec, head_name=calc.head
        )
        atomic_data_list.append(
            mace_data.AtomicData.from_config(
                config,
                z_table=calc.z_table,
                cutoff=calc.r_max,
                heads=calc.available_heads,
            )
        )

    # 分批推理
    loader = torch_geometric.dataloader.DataLoader(
        dataset=atomic_data_list,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    struct_descriptors = []
    n_batches = len(loader)
    t_start = time.time()

    for i, batch in enumerate(loader):
        batch = batch.to(device)
        with torch.no_grad():
            output = model(batch.to_dict(), compute_force=False)

        node_feats = output["node_feats"]
        batch_ids = batch.batch

        cur_batch_size = batch_ids.max().item() + 1
        for local_idx in range(cur_batch_size):
            mask = batch_ids == local_idx
            struct_descriptors.append(node_feats[mask].mean(dim=0).cpu())

        n_done = len(struct_descriptors)
        elapsed = time.time() - t_start
        rate = n_done / elapsed if elapsed > 0 else 0
        eta = (len(atoms_list) - n_done) / rate if rate > 0 else 0
        if (i + 1) % max(1, n_batches // 20) == 0 or i == n_batches - 1:
            print(
                f"  [{n_done:>6d}/{len(atoms_list)}] "
                f"{elapsed:.0f}s elapsed, {rate:.1f} struct/s, ETA {eta:.0f}s"
            )

    struct_descriptors = torch.stack(struct_descriptors, dim=0).numpy()
    assert len(struct_descriptors) == len(atoms_list), (
        f"Descriptor count mismatch: {len(struct_descriptors)} != {len(atoms_list)}"
    )
    print(f"  Final descriptor matrix: {struct_descriptors.shape}")
    print(f"  Total time: {time.time() - t_start:.1f}s")
    return struct_descriptors
