from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
from copy import deepcopy

import ase
import torch
import numpy as np
from ase.io import read
from mace.tools import torch_geometric
from torch.nn.functional import pad
from torch_geometric.utils import get_laplacian, scatter, to_dense_adj

from likelihood.model.neighborhood import get_neighborhood


""" Utility functions for training."""
# Gradient clipping
class Queue:
    def __init__(self, max_len=50):
        self.items = []
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def add(self, item):
        self.items.insert(0, item)
        if len(self) > self.max_len:
            self.items.pop()

    def mean(self):
        return np.mean(self.items)

    def std(self):
        return np.std(self.items)
    

def batchwise_l2_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    batch: Optional[torch.Tensor] = None,
    reduce: bool = "mean",
) -> torch.Tensor:
    if batch is None:
        batch = torch.zeros(
            size=(prediction.size(0),), dtype=torch.long, device=prediction.device
        )

    return scatter(
        torch.norm(prediction - target, p=2, dim=-1), index=batch, reduce=reduce
    ).mean(dim=0)


""" Utility functions for dataset. """
Positions = np.ndarray  # [..., 3]
Cell = np.ndarray  # [3,3]
Pbc = tuple  # (3,)

@dataclass
class Configuration:
    atomic_numbers: np.ndarray
    positions: Positions  # Angstrom
    properties: Dict[str, Any]
    cell: Optional[Cell] = None
    pbc: Optional[Pbc] = None

Configurations = List[Configuration]


def config_from_atoms(atoms: ase.Atoms) -> Configuration:
    """Convert ase.Atoms to Configuration"""

    atomic_numbers = np.array(
        [ase.data.atomic_numbers[symbol] for symbol in atoms.symbols]
    )
    pbc = tuple(atoms.get_pbc())
    cell = np.array(atoms.get_cell())
    properties = {}
    for key, value in atoms.info.items():
        properties[key] = value
    # for key, value in atoms.arrays.items():
    #     properties[key] = value # e.g. forces, magmoms

    return Configuration(
        atomic_numbers=atomic_numbers,
        positions=atoms.get_positions(),
        properties=properties,
        pbc=pbc,
        cell=cell,
    )


def load_from_xyz(file_path: str) -> Configurations:
    atoms_list = read(file_path, index=":")
    configurations = [config_from_atoms(atoms) for atoms in atoms_list]
    return configurations


class AtomicData(torch_geometric.data.Data):
    num_graphs: torch.Tensor
    batch: torch.Tensor
    edge_index: torch.Tensor
    node_attrs: torch.Tensor
    edge_vectors: torch.Tensor
    edge_lengths: torch.Tensor
    positions: torch.Tensor
    shifts: torch.Tensor
    unit_shifts: torch.Tensor
    cell: torch.Tensor

    def __init__(
        self,
        edge_index: torch.Tensor,  # [2, n_edges]
        node_attrs: torch.Tensor,  # [n_nodes, ]
        positions: torch.Tensor,  # [n_nodes, 3]
        shifts: torch.Tensor,  # [n_edges, 3],
        unit_shifts: torch.Tensor,  # [n_edges, 3]
        cell: Optional[torch.Tensor],  # [3,3]
        pbc: Optional[torch.Tensor] = None,  # [, 3]
        **extra_data,  # additional properties that aren't hard-coded and therefore not
        # for correct shape, etc
    ):
        # Check shapes
        num_nodes = node_attrs.shape[0]

        assert edge_index.shape[0] == 2 and len(edge_index.shape) == 2
        assert positions.shape == (num_nodes, 3)
        assert shifts.shape[1] == 3
        assert unit_shifts.shape[1] == 3
        assert pbc is None or (pbc.shape[-1] == 3 and pbc.dtype == torch.bool)

        # Aggregate data
        data = {
            "num_nodes": num_nodes,
            "edge_index": edge_index,
            "positions": positions,
            "shifts": shifts,
            "unit_shifts": unit_shifts,
            "cell": cell,
            "node_attrs": node_attrs,
            "pbc": pbc,
        }
        super().__init__(**data, **extra_data)

    @classmethod
    def from_config(
        cls,
        config: Configuration,
        cutoff: float,
        **kwargs,  # pylint: disable=unused-argument
    ) -> "AtomicData":
        edge_index, shifts, unit_shifts, cell, distvecs, dist = get_neighborhood(
            positions=config.positions,
            cutoff=cutoff,
            pbc=deepcopy(config.pbc),
            cell=deepcopy(config.cell),
        )

        cell = (
            torch.tensor(cell, dtype=torch.get_default_dtype())
            if cell is not None
            else torch.tensor(
                3 * [0.0, 0.0, 0.0], dtype=torch.get_default_dtype()
            ).view(3, 3)
        )

        if config.pbc is not None:
            pbc = list(bool(pbc_) for pbc_ in config.pbc)
        else:
            pbc = None
        pbc = (
            torch.tensor(pbc, dtype=torch.bool)
            if pbc is not None
            else torch.tensor([False, False, False], dtype=torch.bool)
        )

        cls_kwargs = dict(
            node_attrs=torch.tensor(config.atomic_numbers, dtype=torch.long),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            positions=torch.tensor(config.positions, dtype=torch.get_default_dtype()),
            shifts=torch.tensor(shifts, dtype=torch.get_default_dtype()),
            unit_shifts=torch.tensor(unit_shifts, dtype=torch.get_default_dtype()),
            cell=cell,
            pbc=pbc,
            edge_vectors=torch.tensor(distvecs, dtype=torch.get_default_dtype()),
            edge_lengths=torch.tensor(dist, dtype=torch.get_default_dtype()),
        )

        # for k, v in config.properties.items():
        #     if k not in cls_kwargs and v is not None:
        #         if len(v.shape) == 1:
        #             # promote to n_atoms x 1
        #             cls_kwargs[k] = torch.tensor(
        #                 v, dtype=torch.get_default_dtype()
        #             ).unsqueeze(-1)
        #         elif len(v.shape) == 2:
        #             cls_kwargs[k] = torch.tensor(v, dtype=torch.get_default_dtype())

        return cls(**cls_kwargs)


def get_data_loader(
    dataset: Sequence[AtomicData],
    batch_size: int,
    shuffle=True,
    drop_last=False,
) -> torch.utils.data.DataLoader:
    return torch_geometric.dataloader.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )


""" Utility functions for flow model. """

def center_pos(pos, batch):
    pos_center = pos - scatter(pos, batch, dim=0, reduce="mean")[batch]
    return pos_center


def linear_schedule(low, high, max_steps, total_steps) -> torch.Tensor:
    schedule = torch.linspace(low, high, steps=max_steps)

    if max_steps < total_steps:
        pad_size = abs(total_steps - max_steps)
        schedule = pad(schedule, pad=(0, pad_size), mode="constant", value=high)

    return schedule


def center_of_mass(x, dim=0, batch=None):
    num_nodes = x.size(0)

    if batch is None:
        batch = torch.zeros(num_nodes, dtype=torch.long, device=x.device)

    x_com = scatter(x, batch, dim=dim, reduce="mean")[batch]
    return x - x_com


def assert_zero_mean(x: torch.Tensor, batch: torch.Tensor, eps=1e-10) -> bool:
    largest_value = x.abs().max().item()
    a = scatter(x, batch, dim=0, reduce="mean") if batch is not None else x.mean(dim=0)
    error = a.abs().max().item()
    rel_error = error / (largest_value + eps)
    assert rel_error < 1e-2, f"Mean is not zero, relative_error {rel_error}"


def unsqueeze_like(x: torch.Tensor, target: torch.Tensor):
    shape = (x.size(0), *([1] * (target.dim() - 1)))
    return x.view(shape)


def find_rigid_alignment(A, B):
    """
    See: https://en.wikipedia.org/wiki/Kabsch_algorithm
    2-D or 3-D registration with known correspondences.
    Registration occurs in the zero centered coordinate system, and then
    must be transported back.
        Args:
        -    A: Torch tensor of shape (N,D) -- Point Cloud to Align (source)
        -    B: Torch tensor of shape (N,D) -- Reference Point Cloud (target)
        Returns:
        -    R: optimal rotation
        -    t: optimal translation
    Test on rotation + translation and on rotation + translation + reflection
        >>> A = torch.tensor([[1., 1.], [2., 2.], [1.5, 3.]], dtype=torch.float)
        >>> R0 = torch.tensor(
            [[np.cos(60), -np.sin(60)], [np.sin(60), np.cos(60)]], dtype=torch.float
        )
        >>> B = (R0.mm(A.T)).T
        >>> t0 = torch.tensor([3., 3.])
        >>> B += t0
        >>> R, t = find_rigid_alignment(A, B)
        >>> A_aligned = (R.mm(A.T)).T + t
        >>> rmsd = torch.sqrt(((A_aligned - B)**2).sum(axis=1).mean())
        >>> rmsd
        tensor(3.7064e-07)
        >>> B *= torch.tensor([-1., 1.])
        >>> R, t = find_rigid_alignment(A, B)
        >>> A_aligned = (R.mm(A.T)).T + t
        >>> rmsd = torch.sqrt(((A_aligned - B)**2).sum(axis=1).mean())
        >>> rmsd
        tensor(3.7064e-07)
    """
    a_mean = A.mean(axis=0)
    b_mean = B.mean(axis=0)
    A_c = A - a_mean
    B_c = B - b_mean
    # Covariance matrix
    H = A_c.T.mm(B_c)
    U, S, V = torch.svd(H)
    # Rotation matrix
    R = V.mm(U.T)
    # Ensure R is a proper rotation matrix
    if torch.det(R) < 0:  # reflection
        V[:, -1] *= -1  # flip the sign of the last column of V
        R = V.mm(U.T)
    # Translation vector
    t = b_mean[None, :] - R.mm(a_mean[None, :].T).T
    t = t.T
    return R, t.squeeze()


def rmsd_align(pos, ref_pos, batch):
    aligned_pos = []
    batch_size = batch.max() + 1
    for i in range(batch_size):
        index = torch.where(batch == i)[0]
        pos_i = pos[index]
        ref_pos_i = ref_pos[index]
        R, t = find_rigid_alignment(pos_i, ref_pos_i)

        pos_i = (R @ pos_i.T).T + t
        aligned_pos.append(pos_i)

    return torch.concat(aligned_pos, dim=0)
