from pathlib import Path
import math
from typing import Any, Dict, Optional, TypeVar

import torch
from torch import Tensor

from likelihood.model.trainer import BaseModel
from likelihood.model.painn import PaiNN
from likelihood.model.neighborhood import get_batch_neighborhood
from likelihood.model.utils import (
    rmsd_align,
    center_of_mass,
    unsqueeze_like,
    batchwise_l2_loss,
)

Config = TypeVar("Config", str, Dict[str, Any])


class BaseFlow(BaseModel):
    """LightningModule for Flow Matching"""

    __prior_types__ = ["gaussian", "harmonic"]
    __interpolation_types__ = ["linear", "gvp", "gvp_w_sigma", "gvp_squared"]

    def __init__(
        self,
        pbc: bool = False,
        # flow matching network args
        network_type: str = "PaiNN",
        hidden_channels: int = 128,
        num_layers: int = 4,
        num_rbf: int = 64,
        cutoff: float = 5.0,
        num_elements: int = 83,
        # flow matching args
        sigma: float = 0.1,
        prior_type: str = "gaussian",
        sample_time_dist: str = "exponential",
        sample_dir: Optional[Path] = None,
        data_path: Optional[Path] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        # setup network
        if network_type == "PaiNN":
            self.network = PaiNN(
                hidden_channels=hidden_channels,
                num_layers=num_layers,
                num_rbf=num_rbf,
                cutoff=cutoff,
                num_elements=num_elements,
            )
        else:
            raise NotImplementedError(f"Network {network_type} not implemented.")

        self.sigma = sigma
        self.cutoff = cutoff
        self.prior_type = prior_type
        self.sample_time_dist = sample_time_dist
        self.pbc = pbc
        self.eps = None
        self.sample_dir = sample_dir

        assert (
            self.prior_type in self.__prior_types__
        ), f"""\nPrior type {prior_type} not available.
            This is the list of implemented prior types {self.__prior_types__}.\n"""

    def sigma_t(self, t):
        return self.sigma * torch.sqrt(t * (1 - t))

    def sigma_dot_t(self, t):
        return self.sigma * 0.5 * (1 - 2 * t) / torch.sqrt(t * (1 - t))

    def _num_graphs_from_batch(self, batch: Optional[Tensor]) -> int:
        if batch is None:
            return 1
        if batch.numel() == 0:
            return 0
        return int(batch.max().item()) + 1

    def _reshape_batched_cell(
        self, cell: Optional[Tensor], batch: Optional[Tensor]
    ) -> Tensor:
        if cell is None:
            raise ValueError("Cell tensor must be provided when pbc=True.")

        num_graphs = self._num_graphs_from_batch(batch)

        if cell.dim() == 3 and cell.shape[-2:] == (3, 3):
            if cell.size(0) == num_graphs:
                return cell
            if cell.size(0) == 1 and num_graphs > 1:
                return cell.expand(num_graphs, -1, -1)
            raise ValueError(
                f"Invalid batched cell shape {tuple(cell.shape)} for {num_graphs} graphs."
            )

        if cell.dim() != 2 or cell.size(-1) != 3:
            raise ValueError(
                f"Invalid cell shape {tuple(cell.shape)}; expected [3,3], [3B,3], or [B,3,3]."
            )

        if cell.size(0) == 3:
            return cell.view(1, 3, 3).expand(num_graphs, -1, -1)

        if cell.size(0) == 3 * num_graphs:
            return cell.contiguous().view(num_graphs, 3, 3)

        raise ValueError(
            f"Invalid flattened cell shape {tuple(cell.shape)} for {num_graphs} graphs."
        )

    def _cell_per_node(
        self, cell_graph: Tensor, batch: Optional[Tensor], n_nodes: int
    ) -> Tensor:
        if batch is None:
            return cell_graph[0].unsqueeze(0).expand(n_nodes, -1, -1)
        return cell_graph[batch]

    def _cart_to_frac(self, x: Tensor, cell_node: Tensor) -> Tensor:
        return torch.linalg.solve(cell_node.transpose(1, 2), x.unsqueeze(-1)).squeeze(
            -1
        )

    def _frac_to_cart(self, s: Tensor, cell_node: Tensor) -> Tensor:
        return torch.einsum("ni,nij->nj", s, cell_node)

    def _wrap_positions_pbc(
        self, x: Tensor, cell_graph: Tensor, batch: Optional[Tensor]
    ) -> Tensor:
        cell_node = self._cell_per_node(cell_graph, batch=batch, n_nodes=x.size(0))
        frac = self._cart_to_frac(x, cell_node)
        frac = torch.remainder(frac, 1.0)
        return self._frac_to_cart(frac, cell_node)

    def _minimum_image_delta(
        self, dx: Tensor, cell_graph: Tensor, batch: Optional[Tensor]
    ) -> Tensor:
        cell_node = self._cell_per_node(cell_graph, batch=batch, n_nodes=dx.size(0))
        frac = self._cart_to_frac(dx, cell_node)
        frac = frac - torch.round(frac)
        return self._frac_to_cart(frac, cell_node)

    def sample_conditional_pt(
        self,
        x0: Tensor,
        x1: Tensor,
        t: Tensor,
        batch: Tensor,
        cell: Optional[Tensor] = None,
    ):
        # Have this here in case sample_conditional_pt
        # is used outside of compute_conditional_vector_field
        # center both x0 and pos (x1: data distribution)
        if not self.pbc:
            x0 = center_of_mass(x0, batch=batch)
            x1 = center_of_mass(x1, batch=batch)
        else:
            cell_graph = self._reshape_batched_cell(cell, batch=batch)
            x0 = self._wrap_positions_pbc(x0, cell_graph=cell_graph, batch=batch)
            x1 = self._wrap_positions_pbc(x1, cell_graph=cell_graph, batch=batch)

        # unsqueeze t and then reshape to number of atoms
        t = t[batch] if batch is not None else t
        t = unsqueeze_like(t, target=x0)

        # linear interpolation between x0 and x1
        # mu_t = self.interpolation_fn(x0, x1, t)
        eps = torch.randn_like(x1)

        # center each around center of mass
        if not self.pbc:
            eps = center_of_mass(eps, batch=batch)
            mu_t = t * x1 + (1 - t) * x0
        else:
            eps = torch.zeros_like(x1)
            delta = self._minimum_image_delta(
                x1 - x0, cell_graph=cell_graph, batch=batch
            )
            mu_t = self._wrap_positions_pbc(
                x0 + t * delta,
                cell_graph=cell_graph,
                batch=batch,
            )

        # no noise at t = 0 or t = 1
        x_t = mu_t  # + self.sigma_t(t) * eps

        return x_t, eps

    def compute_conditional_vector_field(
        self,
        x0: Tensor,
        x1: Tensor,
        t: Tensor,
        batch: Optional[Tensor] = None,
        cell: Optional[Tensor] = None,
    ):
        if batch is None:
            batch = torch.zeros((x1.size(0),), dtype=torch.long, device=self.device)

        if self.pbc:
            cell_graph = self._reshape_batched_cell(cell, batch=batch)
            x_t, _ = self.sample_conditional_pt(x0, x1, t, batch=batch, cell=cell_graph)
            u_t = self._minimum_image_delta(x1 - x0, cell_graph=cell_graph, batch=batch)
            return x_t, u_t

        # sample a gaussian centered around the interpolation of x1, x0
        x_t, eps = self.sample_conditional_pt(x0, x1, t, batch=batch, cell=cell)
        t = unsqueeze_like(t[batch], x1)

        # derivative of interpolate plus derivative of sigma function * noise
        u_t = x1 - x0  # + self.sigma_dot_t(t) * eps

        return x_t, u_t

    def sample_base_dist(
        self,
        size: torch.Size,
        batch: Optional[Tensor] = None,
        cell: Optional[Tensor] = None,
    ) -> Tensor:
        """Sample from prior distribution (Either harmonic or gaussian)"""
        if not self.pbc:
            x0 = torch.randn(size=size, device=self.device)
            x0 = center_of_mass(x0, batch=batch)
        else:
            if batch is None:
                batch = torch.zeros(
                    size=(size[0],), dtype=torch.long, device=self.device
                )
            cell_graph = self._reshape_batched_cell(cell, batch=batch)
            cell_node = self._cell_per_node(cell_graph, batch=batch, n_nodes=size[0])

            frac = torch.rand(size=size, device=self.device, dtype=cell_node.dtype)
            x0 = self._frac_to_cart(frac, cell_node)
            x0 = self._wrap_positions_pbc(x0, cell_graph=cell_graph, batch=batch)

        return x0

    def sample_time(
        self,
        num_samples: int,
        low: float = 1e-4,
        high: float = 0.9999,
        stage: str = "train",
    ):
        """Sample flow-matching time steps for training or validation"""
        if self.sample_time_dist == "uniform" or stage == "val":
            return torch.zeros(size=(num_samples, 1), device=self.device).uniform_(
                low, high
            )
        elif self.sample_time_dist == "logit_norm":
            return torch.sigmoid(torch.randn(size=(num_samples, 1), device=self.device))
        elif self.sample_time_dist == "exponential":
            lam = 3.0
            u = torch.rand(size=(num_samples, 1), device=self.device)
            low_t = torch.tensor(low, device=self.device)
            high_t = torch.tensor(high, device=self.device)
            exp_low = torch.exp(-lam * low_t)
            exp_high = torch.exp(-lam * high_t)
            t_low = -torch.log(exp_low - u * (exp_low - exp_high)) / lam
            t = high_t - (t_low - low_t)
            return t
        else:
            raise NotImplementedError(
                f"Time sampling with {self.sample_time_dist} not implemented"
            )

    def forward(
        self,
        atomic_numbers: Tensor,
        t: Tensor,
        pos: Tensor,
        cell: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
    ):
        # NOTE: Neighbor lists are built on detached coordinates, so the edge *set*
        # is piecewise-constant. Edge vectors/lengths are computed from `pos` (and a
        # constant shift) so `v_t(pos)` stays differentiable w.r.t. `pos`.
        pos_np = pos.detach().cpu().numpy()
        # For non-PBC runs we do not need (and should not trust) `cell` from a
        # batched PyG `Data` object, because it gets concatenated along dim 0.
        if self.pbc:
            cell_graph = self._reshape_batched_cell(cell, batch=batch)
            cell_np = cell_graph.detach().cpu().numpy()
        else:
            cell_np = None
        batch_np = batch.detach().cpu().numpy() if batch is not None else None
        edge_index, shifts, _unit_shifts, _cells, _distvecs, _dist = (
            get_batch_neighborhood(
                positions=pos_np,
                batch=batch_np,
                cutoff=self.cutoff,
                pbc=(self.pbc, self.pbc, self.pbc),
                cell=cell_np,
            )
        )

        edge_index = torch.tensor(edge_index, dtype=torch.long, device=pos.device)
        src, dst = edge_index[0], edge_index[1]
        shifts_t = torch.tensor(shifts, dtype=pos.dtype, device=pos.device)
        edge_vectors = pos[dst] - pos[src] + shifts_t
        edge_lengths = torch.linalg.norm(edge_vectors, dim=-1)
        assert torch.all(edge_lengths < self.cutoff + 1e-6)

        # compute energy and score from network (unique atom indexing)
        t_input = t[batch] if batch is not None else t
        v_t = self.network(
            atomic_numbers=atomic_numbers,
            t=t_input.squeeze(),
            edge_index=edge_index,
            dist=edge_lengths,
            diff=edge_vectors,
            batch=batch,
        )

        if not self.pbc:
            v_t = center_of_mass(v_t, batch=batch)

        return v_t

    def _graph_sum(self, values: Tensor, batch: Tensor, num_graphs: int) -> Tensor:
        out = torch.zeros(num_graphs, device=values.device, dtype=values.dtype)
        return out.index_add(0, batch, values)

    def _gaussian_prior_log_prob(
        self,
        x0: Tensor,
        batch: Tensor,
        include_constant: bool = True,
    ) -> Tensor:
        """Log p(x0) for a centered unit Gaussian per graph.

        We treat each graph's coordinates as living in the zero-COM subspace.
        This reduces the effective dimension by 3 (one translation vector).
        """
        num_graphs = int(batch.max().item()) + 1
        x0 = center_of_mass(x0, batch=batch)

        node_sq = (x0**2).sum(dim=-1)  # [n_nodes]
        sq_per_graph = self._graph_sum(node_sq, batch=batch, num_graphs=num_graphs)

        n_atoms = torch.bincount(batch, minlength=num_graphs).to(x0.device)
        dof = (3 * n_atoms - 3).to(x0.dtype)  # translation removed

        logp = -0.5 * sq_per_graph
        if include_constant:
            logp = logp - 0.5 * dof * math.log(2.0 * math.pi)

        return logp

    def _hutchinson_divergence(
        self,
        v: Tensor,
        x: Tensor,
        batch: Tensor,
        num_graphs: int,
        num_samples: int = 1,
        pbc: bool = False,
    ) -> Tensor:
        """Estimate div(v)(x) per graph via Hutchinson's trace estimator."""
        if num_samples < 1:
            raise ValueError("num_samples must be >= 1")

        div = torch.zeros(num_graphs, device=x.device, dtype=x.dtype)
        for _ in range(num_samples):
            eps = torch.randn_like(x)
            if not pbc:
                eps = center_of_mass(eps, batch=batch)

            inner = (v * eps).sum()
            grad = torch.autograd.grad(
                inner,
                x,
                create_graph=False,
                retain_graph=False,
                only_inputs=True,
            )[0]
            node_contrib = (grad * eps).sum(dim=-1)  # [n_nodes]
            div = div + self._graph_sum(
                node_contrib, batch=batch, num_graphs=num_graphs
            )

        return div / float(num_samples)

    def _uniform_pbc_prior_log_prob(
        self,
        batch: Tensor,
        cell: Tensor,
        include_constant: bool = True,
    ) -> Tensor:
        """Log p(x0) for a uniform prior in the periodic unit cell per graph."""
        num_graphs = int(batch.max().item()) + 1
        if not include_constant:
            return torch.zeros(num_graphs, device=batch.device, dtype=cell.dtype)

        cell_graph = self._reshape_batched_cell(cell, batch=batch)
        volumes = torch.linalg.det(cell_graph).abs().clamp_min(1e-12)
        n_atoms = torch.bincount(batch, minlength=num_graphs).to(cell.dtype)
        return -n_atoms * torch.log(volumes)

    def log_prob(
        self,
        atomic_numbers: Tensor,
        pos: Tensor,
        batch: Optional[Tensor] = None,
        cell: Optional[Tensor] = None,
        n_timesteps: int = 100,
        hutchinson_samples: int = 1,
        include_prior_constant: bool = True,
    ) -> Tensor:
        """Approximate log-likelihood (CNF) of `pos` under the learned flow.

        This treats the trained flow-matching velocity as a deterministic CNF:
            d x / d t = v_theta(x, t)
            d log p(x_t) / d t = - div_x v_theta(x_t, t)

        We integrate the ODE backwards from t=1 -> 0 to obtain x0 and the log-density
        change, then evaluate the Gaussian prior at x0.

        Returns a per-graph `log p(pos)` tensor of shape [num_graphs].
        """
        if batch is None:
            batch = torch.zeros(
                size=(pos.size(0),), dtype=torch.long, device=pos.device
            )

        num_graphs = int(batch.max().item()) + 1
        cell_graph = self._reshape_batched_cell(cell, batch=batch) if self.pbc else None

        # Work on the same manifold as training.
        if self.pbc:
            x = self._wrap_positions_pbc(
                pos, cell_graph=cell_graph, batch=batch
            ).detach()
        else:
            x = center_of_mass(pos, batch=batch).detach()
        logp_change = torch.zeros(num_graphs, device=pos.device, dtype=pos.dtype)

        # Backward integration: t from 1 -> 0.
        t_schedule = torch.linspace(
            1.0, 0.0, steps=n_timesteps + 1, device=pos.device, dtype=pos.dtype
        )

        was_training = self.training
        self.eval()
        with torch.enable_grad():
            for i in range(n_timesteps):
                t_curr = t_schedule[i]
                t_next = t_schedule[i + 1]
                delta_t = t_next - t_curr  # negative

                x = x.detach()
                x.requires_grad_(True)

                t_graph = torch.full(
                    (num_graphs, 1), float(t_curr), device=pos.device, dtype=pos.dtype
                )
                v_t = self(
                    atomic_numbers=atomic_numbers,
                    t=t_graph,
                    pos=x,
                    cell=cell_graph if self.pbc else cell,
                    batch=batch,
                )

                div_v = self._hutchinson_divergence(
                    v=v_t,
                    x=x,
                    batch=batch,
                    num_graphs=num_graphs,
                    num_samples=hutchinson_samples,
                    pbc=self.pbc,
                )
                # We want: log p(x_1) = log p(x_0) - \int_0^1 div v(x_t,t) dt.
                # When integrating *backwards* (t: 1 -> 0), `delta_t` is negative,
                # so accumulating `div_v * delta_t` yields `-\int_0^1 div v dt`.
                logp_change = logp_change + div_v * delta_t

                with torch.no_grad():
                    x = x + delta_t * v_t
                    if self.pbc:
                        x = self._wrap_positions_pbc(
                            x, cell_graph=cell_graph, batch=batch
                        )
                    else:
                        x = center_of_mass(x, batch=batch)

        if was_training:
            self.train()

        x0 = x.detach()
        if self.pbc:
            logp0 = self._uniform_pbc_prior_log_prob(
                batch=batch,
                cell=cell_graph,
                include_constant=include_prior_constant,
            )
        else:
            logp0 = self._gaussian_prior_log_prob(
                x0, batch=batch, include_constant=include_prior_constant
            )
        return logp0 + logp_change

    def nll(
        self,
        atomic_numbers: Tensor,
        pos: Tensor,
        batch: Optional[Tensor] = None,
        cell: Optional[Tensor] = None,
        n_timesteps: int = 100,
        hutchinson_samples: int = 1,
        include_prior_constant: Optional[bool] = True,
    ):
        """Per-graph negative log-likelihood.

        If `include_prior_constant` is a bool, returns a tensor `nll`.
        If `include_prior_constant is None`, computes the CNF likelihood only once and
        returns a tuple `(nll, nll_noconst)`.
        """
        if batch is None:
            raise ValueError("Batch tensor must be provided for nll computation.")

        if include_prior_constant is None:
            logp_noconst = self.log_prob(
                atomic_numbers=atomic_numbers,
                pos=pos,
                batch=batch,
                cell=cell,
                n_timesteps=n_timesteps,
                hutchinson_samples=hutchinson_samples,
                include_prior_constant=False,
            )

            num_graphs = int(batch.max().item()) + 1
            n_atoms = torch.bincount(batch, minlength=num_graphs).to(pos.device)
            if self.pbc:
                cell_graph = self._reshape_batched_cell(cell, batch=batch)
                volumes = torch.linalg.det(cell_graph).abs().clamp_min(1e-12)
                prior_const = -n_atoms.to(pos.dtype) * torch.log(volumes.to(pos.dtype))
            else:
                dof = (3 * n_atoms - 3).to(pos.dtype)
                log2pi = torch.log(
                    torch.tensor(2.0 * math.pi, device=pos.device, dtype=pos.dtype)
                )
                prior_const = -0.5 * dof * log2pi

            logp = logp_noconst + prior_const
            return -logp, -logp_noconst

        return -self.log_prob(
            atomic_numbers=atomic_numbers,
            pos=pos,
            batch=batch,
            cell=cell,
            n_timesteps=n_timesteps,
            hutchinson_samples=hutchinson_samples,
            include_prior_constant=bool(include_prior_constant),
        )

    def generic_step(self, batched_data, batch_idx: int, stage: str):
        z, pos, batch = (
            batched_data["atomic_numbers"],
            batched_data["pos"],
            batched_data["batch"],
        )
        batch_size = batch.max().item() + 1 if batch is not None else 1

        if not self.pbc:
            pos = center_of_mass(pos, batch=batch)
            cell = None  # no cell needed for non-pbc
        else:
            cell = self._reshape_batched_cell(batched_data["cell"], batch=batch)
            pos = self._wrap_positions_pbc(pos, cell_graph=cell, batch=batch)

        # sample base distribution, either from harmonic or gaussian
        # x0 is sampling distribution and not data distribution
        if not self.pbc:
            x0 = self.sample_base_dist(
                pos.shape,
                batch=batch,
            )
        else:
            x0 = self.sample_base_dist(
                pos.shape,
                batch=batch,
                cell=cell,
            )

        # sample time steps equal to number of molecules in a batch
        t = self.sample_time(num_samples=batch_size, stage=stage)

        # TODO: align x0 to x1 (data distribution) with kabsch.
        # NOTE: Don't do this for unique atom indexing.
        # x0 = rmsd_align(pos=x0, ref_pos=pos, batch=batch)

        # sample conditional vector field for positions
        x_t, u_t = self.compute_conditional_vector_field(
            x0=x0,
            x1=pos,
            t=t,
            batch=batch,
            cell=cell,
        )

        # run flow matching network
        v_t = self(
            atomic_numbers=z,
            t=t,
            pos=x_t,
            cell=cell,
            batch=batch,
        )

        # regress against vector field
        loss = batchwise_l2_loss(v_t, u_t, batch=batch, reduce="mean")

        if torch.isnan(loss):
            raise ValueError("Loss is NaN, fix bug")

        # log loss
        self.log_helper(f"{stage}/loss", loss, batch_size=batch_size)

        return loss

    def _compute_delta_t(self, t_schedule: Tensor, t: int) -> Tensor:
        if t + 1 >= t_schedule.size(0):
            return torch.tensor(0.0, device=t_schedule.device)

        t_curr, t_next = t_schedule[t : t + 2]
        return t_next - t_curr

    def sample(
        self,
        atomic_numbers: Tensor,
        batch: Optional[Tensor] = None,
        cell: Optional[Tensor] = None,
        n_timesteps: int = 200,
        sampler_type: str = "ode",
        s_churn: float = 0.0,
        t_min: float = 0.0,
        t_max: float = 1.0,
        out_path: Optional[Path] = None,
    ) -> Tensor:
        """Generate samples with ODE or stochastic flow matching."""
        if out_path is None:
            if self.sample_dir is not None:
                out_path = Path(self.sample_dir) / "sample.xyz"
            else:
                out_path = Path("sample.xyz")
        out_path = Path(out_path)
        out_traj_path = out_path.parent / "traj_" / out_path.name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_traj_path.parent.mkdir(parents=True, exist_ok=True)

        if batch is None:
            batch = torch.zeros(
                size=(atomic_numbers.size(0),), dtype=torch.long, device=self.device
            )

        cell_graph = self._reshape_batched_cell(cell, batch=batch) if self.pbc else None

        t_schedule = torch.linspace(0.0, 1.0, steps=n_timesteps + 1, device=self.device)
        if self.pbc:
            x = self.sample_base_dist(
                (atomic_numbers.size(0), 3),
                batch=batch,
                cell=cell_graph,
            )
            x = self._wrap_positions_pbc(x, cell_graph=cell_graph, batch=batch)
        else:
            x = center_of_mass(
                self.sample_base_dist((atomic_numbers.size(0), 3), batch=batch),
                batch=batch,
            )

        gamma = torch.tensor(s_churn / n_timesteps, device=self.device)

        from ase import Atoms
        from ase.io import write

        if out_traj_path.exists():
            out_traj_path.unlink()

        n = t_schedule.size(0) - 1
        for i in range(n):
            t = t_schedule[i].repeat(x.size(0))
            t = unsqueeze_like(t, x)
            delta_t = self._compute_delta_t(t_schedule, t=i)

            if (
                sampler_type == "stochastic"
                and t_min <= t_schedule[i] < t_max
                and s_churn > 0.0
            ):
                x = x + gamma * torch.randn_like(x)

            v_t = self(
                atomic_numbers=atomic_numbers,
                t=t,
                pos=x,
                cell=cell_graph if self.pbc else None,
                batch=batch,
            )
            x = x + delta_t * v_t
            if self.pbc:
                x = self._wrap_positions_pbc(x, cell_graph=cell_graph, batch=batch)
            else:
                x = center_of_mass(x, batch=batch)

            # save trajectory of first molecule in batch
            idx_0 = batch == 0
            traj_numbers = atomic_numbers[idx_0]
            traj_positions = x[idx_0]
            write(
                str(out_traj_path),
                Atoms(
                    numbers=traj_numbers.cpu().numpy(),
                    positions=traj_positions.cpu().numpy(),
                    cell=(cell_graph[0].detach().cpu().numpy() if self.pbc else None),
                    pbc=(self.pbc, self.pbc, self.pbc),
                ),
                append=True,
            )

        # save final samples
        if out_path.exists():
            out_path.unlink()

        atoms_frames = []
        num_graphs = int(batch.max().item()) + 1
        for graph_idx in range(num_graphs):
            idx = batch == graph_idx
            atoms_frames.append(
                Atoms(
                    numbers=atomic_numbers[idx].cpu().numpy(),
                    positions=x[idx].cpu().numpy(),
                    cell=(
                        cell_graph[graph_idx].detach().cpu().numpy()
                        if self.pbc
                        else None
                    ),
                    pbc=(self.pbc, self.pbc, self.pbc),
                )
            )
        if atoms_frames:
            write(str(out_path), atoms_frames)

        return x


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
