from typing import Optional, Tuple, Union, Sequence

import numpy as np
from matscipy.neighbours import neighbour_list


def get_neighborhood(
    positions: np.ndarray,  # [num_positions, 3]
    cutoff: float,
    pbc: Optional[Tuple[bool, bool, bool]] = None,
    cell: Optional[np.ndarray] = None,  # [3, 3]
    true_self_interaction=False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if pbc is None:
        pbc = (False, False, False)

    if cell is None or cell.any() == np.zeros((3, 3)).any():
        cell = np.identity(3, dtype=float)

    assert len(pbc) == 3 and all(isinstance(i, (bool, np.bool_)) for i in pbc)
    assert cell.shape == (3, 3)

    pbc_x = bool(pbc[0])
    pbc_y = bool(pbc[1])
    pbc_z = bool(pbc[2])
    identity = np.identity(3, dtype=float)
    max_positions = np.max(np.absolute(positions)) + 1
    # Extend cell in non-periodic directions
    # For models with more than 5 layers, the multiplicative constant needs to be increased.
    # temp_cell = np.copy(cell)
    if not pbc_x:
        cell[0, :] = max_positions * 5 * cutoff * identity[0, :]
    if not pbc_y:
        cell[1, :] = max_positions * 5 * cutoff * identity[1, :]
    if not pbc_z:
        cell[2, :] = max_positions * 5 * cutoff * identity[2, :]

    sender, receiver, unit_shifts = neighbour_list(
        quantities="ijS",
        pbc=pbc,
        cell=cell,
        positions=positions,
        cutoff=cutoff,
        # self_interaction=True,  # we want edges from atom to itself in different periodic images
        # use_scaled_positions=False,  # positions are not scaled positions
    )

    if not true_self_interaction:
        # Eliminate self-edges that don't cross periodic boundaries
        true_self_edge = sender == receiver
        true_self_edge &= np.all(unit_shifts == 0, axis=1)
        keep_edge = ~true_self_edge

        # Note: after eliminating self-edges, it can be that no edges remain in this system
        sender = sender[keep_edge]
        receiver = receiver[keep_edge]
        unit_shifts = unit_shifts[keep_edge]

    # Build output
    edge_index = np.stack((sender, receiver))  # [2, n_edges]

    # From the docs: With the shift vector S, the distances D between atoms can be computed from
    # D = positions[j]-positions[i]+S.dot(cell)
    shifts = np.dot(unit_shifts, cell)  # [n_edges, 3]
    distvecs = positions[receiver] - positions[sender] + shifts  # [n_edges, 3]
    dist = np.linalg.norm(distvecs, axis=1)  # [n_edges]

    return edge_index, shifts, unit_shifts, cell, distvecs, dist


def get_batch_neighborhood(
    positions: np.ndarray,  # [num_positions, 3]
    batch: np.ndarray,  # [num_positions]
    cutoff: float,
    pbc: Optional[Union[Tuple[bool, bool, bool], Sequence[Tuple[bool, bool, bool]]]] = None,
    cell: Optional[Union[np.ndarray, Sequence[np.ndarray]]] = None,  # [3, 3] or [B,3,3]
    true_self_interaction: bool = False,
):
    """Batch-safe neighborhood; no edges across graphs.

    Returns:
        edge_index: [2, n_edges]
        shifts: [n_edges, 3]
        unit_shifts: [n_edges, 3]
        cells: [num_graphs, 3, 3]
        distvecs: [n_edges, 3]
        dist: [n_edges]
    """
    if batch is None:
        return get_neighborhood(
            positions=positions,
            cutoff=cutoff,
            pbc=pbc,  # type: ignore[arg-type]
            cell=cell,  # type: ignore[arg-type]
            true_self_interaction=true_self_interaction,
        )

    batch = np.asarray(batch)
    num_graphs = int(batch.max()) + 1 if batch.size > 0 else 0

    edge_indices = []
    shifts_list = []
    unit_shifts_list = []
    distvecs_list = []
    dist_list = []
    cells_list = []

    def select_pbc(graph_idx: int):
        if pbc is None:
            return None
        if isinstance(pbc, (tuple, list)) and len(pbc) == 3 and isinstance(pbc[0], (bool, np.bool_)):
            return pbc
        return pbc[graph_idx]  # type: ignore[index]

    def select_cell(graph_idx: int):
        if cell is None:
            return None
        cell_arr = np.asarray(cell)
        if cell_arr.shape == (3, 3):
            return cell_arr
        return cell_arr[graph_idx]

    for graph_idx in range(num_graphs):
        idx = np.where(batch == graph_idx)[0]
        if idx.size == 0:
            continue

        pos_b = positions[idx]
        edge_index, shifts, unit_shifts, cell_b, distvecs, dist = get_neighborhood(
            positions=pos_b,
            cutoff=cutoff,
            pbc=select_pbc(graph_idx),
            cell=select_cell(graph_idx),
            true_self_interaction=true_self_interaction,
        )

        edge_index = np.stack((idx[edge_index[0]], idx[edge_index[1]]))
        edge_indices.append(edge_index)
        shifts_list.append(shifts)
        unit_shifts_list.append(unit_shifts)
        distvecs_list.append(distvecs)
        dist_list.append(dist)
        cells_list.append(cell_b)

    if len(edge_indices) == 0:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        shifts = np.zeros((0, 3), dtype=float)
        unit_shifts = np.zeros((0, 3), dtype=float)
        distvecs = np.zeros((0, 3), dtype=float)
        dist = np.zeros((0,), dtype=float)
        cells = np.zeros((num_graphs, 3, 3), dtype=float)
    else:
        edge_index = np.concatenate(edge_indices, axis=1)
        shifts = np.concatenate(shifts_list, axis=0)
        unit_shifts = np.concatenate(unit_shifts_list, axis=0)
        distvecs = np.concatenate(distvecs_list, axis=0)
        dist = np.concatenate(dist_list, axis=0)
        cells = np.stack(cells_list, axis=0)

    return edge_index, shifts, unit_shifts, cells, distvecs, dist
