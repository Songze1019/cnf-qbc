import math
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import scatter

from scipy.special import binom


class SinusoidsEmbedding(nn.Module):
    def __init__(self, n_frequencies=10, n_space=3):
        super().__init__()
        self.n_frequencies = n_frequencies
        self.n_space = n_space
        self.frequencies = 2 * math.pi * torch.arange(self.n_frequencies)
        self.dim = self.n_frequencies * 2 * self.n_space

    def forward(self, x):
        emb = x.unsqueeze(-1) * self.frequencies[None, None, :].to(x.device)
        emb = emb.reshape(-1, self.n_frequencies * self.n_space)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class PaiNN(torch.nn.Module):
    def __init__(
        self,
        hidden_channels: int = 512,
        out_channels: int = 8,
        num_layers: int = 6,
        num_rbf: int = 128,
        cutoff: float = 10.0,
        rbf: Dict[str, str] = {"name": "gaussian"},
        envelope: Dict[str, Union[str, float]] = {
            "name": "polynomial",
            "exponent": 5,
        },
        num_elements: int = 83,
        **kwargs,
    ):
        super(PaiNN, self).__init__()

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_rbf = num_rbf
        self.cutoff = cutoff

        self.atom_emb = AtomEmbedding(hidden_channels, num_elements)
        self.t_emb = GaussianFourierProjection(hidden_channels, num_rbf)
        self.embedding = nn.Linear(hidden_channels * 2, hidden_channels)
        self.radial_basis = RadialBasis(
            num_radial=num_rbf,
            cutoff=self.cutoff,
            envelope=envelope,
            rbf=rbf,
        )

        self.message_layers = nn.ModuleList()
        self.update_layers = nn.ModuleList()

        for _ in range(num_layers):
            self.message_layers.append(PaiNNMessage(hidden_channels, num_rbf))
            self.update_layers.append(PaiNNUpdate(hidden_channels))

        self.out_xh = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            ScaledSiLU(),
            nn.Linear(hidden_channels // 2, out_channels),
        )

        self.out_dpos = PaiNNOutput(hidden_channels)

        self.inv_sqrt_2 = 1 / math.sqrt(2.0)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.out_xh[0].weight)
        self.out_xh[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_xh[2].weight)
        self.out_xh[2].bias.data.fill_(0)

    def forward(self, atomic_numbers, t, dist, diff, edge_index, batch=None, **kargs):
        # Compute local atom indices within each molecule (0 to num_atoms_per_mol - 1)
        # if batch is not None:
        #     # Count atoms per molecule and compute local indices
        #     ones = torch.ones_like(batch)
        #     atoms_per_mol = torch.zeros(
        #         batch.max() + 1, dtype=torch.long, device=batch.device
        #     )
        #     atoms_per_mol.scatter_add_(0, batch, ones)
        #     cumsum = torch.cat(
        #         [
        #             torch.zeros(1, dtype=torch.long, device=batch.device),
        #             atoms_per_mol.cumsum(0)[:-1],
        #         ]
        #     )
        #     local_indices = (
        #         torch.arange(len(batch), dtype=torch.long, device=batch.device)
        #         - cumsum[batch]
        #     )
        # else:
        #     local_indices = torch.arange(
        #         len(atomic_numbers), dtype=torch.long, device=atomic_numbers.device
        #     )

        # atom_emb = self.atom_emb(local_indices + 1)
        atom_emb = self.atom_emb(atomic_numbers)
        t_emb = self.t_emb(t)

        xh_t = torch.cat([atom_emb, t_emb], dim=-1)
        x = self.embedding(xh_t)
        vec = torch.zeros(x.size(0), 3, x.size(1)).to(x.device)

        edge_rbf = self.radial_basis(dist)

        for i in range(self.num_layers):
            dx, dvec = self.message_layers[i](
                x,
                vec,
                edge_index,
                edge_rbf,
                diff,
            )
            x = x + dx
            vec = vec + dvec
            x = x * self.inv_sqrt_2

            dx, dvec = self.update_layers[i](x, vec)
            x = x + dx
            vec = vec + dvec
            x = x * self.inv_sqrt_2

        dpos = self.out_dpos(x, vec)

        return dpos


class PaiNNMessage(MessagePassing):
    def __init__(
        self,
        hidden_channels,
        num_rbf,
    ) -> None:
        super(PaiNNMessage, self).__init__(aggr="add", node_dim=0)

        self.hidden_channels = hidden_channels

        self.x_proj = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            ScaledSiLU(),
            nn.Linear(hidden_channels, hidden_channels * 3),
        )
        self.rbf_proj = nn.Linear(num_rbf, hidden_channels * 3)

        self.inv_sqrt_3 = 1 / math.sqrt(3.0)
        self.inv_sqrt_h = 1 / math.sqrt(hidden_channels)
        self.x_layernorm = nn.LayerNorm(hidden_channels)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.x_proj[0].weight)
        self.x_proj[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.x_proj[2].weight)
        self.x_proj[2].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.rbf_proj.weight)
        self.rbf_proj.bias.data.fill_(0)
        self.x_layernorm.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        vec: torch.Tensor,
        edge_index: torch.Tensor,
        edge_rbf: torch.Tensor,
        edge_vector: torch.Tensor,
    ):
        xh = self.x_proj(self.x_layernorm(x))

        # TODO(@abhshkdz): Nans out with AMP here during backprop. Debug / fix.
        rbfh = self.rbf_proj(edge_rbf)

        # propagate_type: (xh: Tensor, vec: Tensor, rbfh_ij: Tensor, r_ij: Tensor)
        dx, dvec = self.propagate(
            edge_index,
            xh=xh,
            vec=vec,
            rbfh_ij=rbfh,
            r_ij=edge_vector,
            size=None,
        )

        return dx, dvec

    def message(self, xh_j, vec_j, rbfh_ij, r_ij):
        x, xh2, xh3 = torch.split(xh_j * rbfh_ij, self.hidden_channels, dim=-1)
        xh2 = xh2 * self.inv_sqrt_3

        vec = vec_j * xh2.unsqueeze(1) + xh3.unsqueeze(1) * r_ij.unsqueeze(2)
        vec = vec * self.inv_sqrt_h

        return x, vec

    def aggregate(
        self,
        features: Tuple[torch.Tensor, torch.Tensor],
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        dim_size: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, vec = features
        x = scatter(x, index, dim=self.node_dim, dim_size=dim_size)
        vec = scatter(vec, index, dim=self.node_dim, dim_size=dim_size)
        return x, vec

    def update(
        self, inputs: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return inputs


class ScalarMessage(MessagePassing):
    def __init__(
        self,
        hidden_channels,
        num_rbf,
    ) -> None:
        super(ScalarMessage, self).__init__(aggr="mean", node_dim=0)

        self.hidden_channels = hidden_channels
        self.inv_sqrt_3 = 1 / math.sqrt(3.0)
        self.inv_sqrt_h = 1 / math.sqrt(hidden_channels)

        self.rbf_proj = nn.Linear(num_rbf, hidden_channels)

        self.x_layernorm = nn.LayerNorm(hidden_channels)

        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_channels * 3, hidden_channels * 3),
            ScaledSiLU(),
            nn.Linear(hidden_channels * 3, hidden_channels * 3),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_channels * 3, hidden_channels * 2),
            ScaledSiLU(),
            nn.Linear(hidden_channels * 2, hidden_channels),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.rbf_proj.weight)
        self.rbf_proj.bias.data.fill_(0)
        self.x_layernorm.reset_parameters()

        nn.init.xavier_uniform_(self.edge_mlp[0].weight)
        self.edge_mlp[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.edge_mlp[2].weight)
        self.edge_mlp[2].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.node_mlp[0].weight)
        self.node_mlp[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.node_mlp[2].weight)
        self.node_mlp[2].bias.data.fill_(0)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_rbf: torch.Tensor,
    ):
        xh = self.x_layernorm(x)

        # TODO(@abhshkdz): Nans out with AMP here during backprop. Debug / fix.
        rbfh = self.rbf_proj(edge_rbf)

        # propagate_type: (xh: Tensor, vec: Tensor, rbfh_ij: Tensor, r_ij: Tensor)
        ds = self.propagate(
            edge_index,
            xh=xh,
            rbfh_ij=rbfh,
            size=None,
        )
        return ds

    def message(self, xh_i, xh_j, rbfh_ij):
        mij = torch.cat([xh_i, xh_j, rbfh_ij], dim=1)
        edge_weight = self.edge_mlp(mij)
        mij = (mij + edge_weight) * self.inv_sqrt_3
        return mij

    def aggregate(
        self,
        features: torch.Tensor,
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        dim_size: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = scatter(
            features, index, dim=self.node_dim, dim_size=dim_size, reduce="mean"
        )
        return x

    def update(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        x = self.node_mlp(inputs)
        return x


class PaiNNUpdate(nn.Module):
    def __init__(self, hidden_channels) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels

        self.vec_proj = nn.Linear(hidden_channels, hidden_channels * 2, bias=False)
        self.xvec_proj = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            ScaledSiLU(),
            nn.Linear(hidden_channels, hidden_channels * 3),
        )

        self.inv_sqrt_2 = 1 / math.sqrt(2.0)
        self.inv_sqrt_h = 1 / math.sqrt(hidden_channels)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.vec_proj.weight)
        nn.init.xavier_uniform_(self.xvec_proj[0].weight)
        self.xvec_proj[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.xvec_proj[2].weight)
        self.xvec_proj[2].bias.data.fill_(0)

    def forward(self, x, vec):
        vec1, vec2 = torch.split(self.vec_proj(vec), self.hidden_channels, dim=-1)
        vec_dot = (vec1 * vec2).sum(dim=1) * self.inv_sqrt_h

        # NOTE: Can't use torch.norm because the gradient is NaN for input = 0.
        # Add an epsilon offset to make sure sqrt is always positive.
        x_vec_h = self.xvec_proj(
            torch.cat([x, torch.sqrt(torch.sum(vec2**2, dim=-2) + 1e-8)], dim=-1)
        )
        xvec1, xvec2, xvec3 = torch.split(x_vec_h, self.hidden_channels, dim=-1)

        dx = xvec1 + xvec2 * vec_dot
        dx = dx * self.inv_sqrt_2

        dvec = xvec3.unsqueeze(1) * vec1

        return dx, dvec


class PaiNNOutput(nn.Module):
    def __init__(self, hidden_channels) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels

        self.output_network = nn.ModuleList(
            [
                GatedEquivariantBlock(
                    hidden_channels,
                    hidden_channels // 2,
                ),
                GatedEquivariantBlock(hidden_channels // 2, 1),
            ]
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in self.output_network:
            layer.reset_parameters()

    def forward(self, x, vec):
        for layer in self.output_network:
            x, vec = layer(x, vec)
        return vec.squeeze()


# Borrowed from TorchMD-Net
class GatedEquivariantBlock(nn.Module):
    """Gated Equivariant Block as defined in Schütt et al. (2021):
    Equivariant message passing for the prediction of tensorial properties and molecular spectra
    """

    def __init__(
        self,
        hidden_channels,
        out_channels,
    ) -> None:
        super(GatedEquivariantBlock, self).__init__()
        self.out_channels = out_channels

        self.vec1_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.vec2_proj = nn.Linear(hidden_channels, out_channels, bias=False)

        self.update_net = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            ScaledSiLU(),
            nn.Linear(hidden_channels, out_channels * 2),
        )

        self.act = ScaledSiLU()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.vec1_proj.weight)
        nn.init.xavier_uniform_(self.vec2_proj.weight)
        nn.init.xavier_uniform_(self.update_net[0].weight)
        self.update_net[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.update_net[2].weight)
        self.update_net[2].bias.data.fill_(0)

    def forward(self, x, v):
        vec1 = torch.norm(self.vec1_proj(v), dim=-2)
        vec2 = self.vec2_proj(v)

        x = torch.cat([x, vec1], dim=-1)
        x, v = torch.split(self.update_net(x), self.out_channels, dim=-1)
        v = v.unsqueeze(1) * vec2

        x = self.act(x)
        return x, v


class AtomEmbedding(torch.nn.Module):
    """
    Initial atom embeddings based on the atom type

    Parameters
    ----------
        emb_size: int
            Atom embeddings size
    """

    def __init__(self, emb_size, num_elements: int) -> None:
        super().__init__()
        self.emb_size = emb_size

        self.embeddings = torch.nn.Embedding(num_elements, emb_size)
        # init by uniform distribution
        torch.nn.init.uniform_(self.embeddings.weight, a=-np.sqrt(3), b=np.sqrt(3))

    def forward(self, Z):
        """
        Returns
        -------
            h: torch.Tensor, shape=(nAtoms, emb_size)
                Atom embeddings.
        """
        h = self.embeddings(Z - 1)  # -1 because Z.min()=1 (==Hydrogen)
        return h


class ScaledSiLU(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale_factor = 1 / 0.6
        self._activation = torch.nn.SiLU()

    def forward(self, x) -> Tensor:
        return self._activation(x) * self.scale_factor


class RadialBasis(torch.nn.Module):
    """

    Parameters
    ----------
    num_radial: int
        Controls maximum frequency.
    cutoff: float
        Cutoff distance in Angstrom.
    rbf: dict = {"name": "gaussian"}
        Basis function and its hyperparameters.
    envelope: dict = {"name": "polynomial", "exponent": 5}
        Envelope function and its hyperparameters.
    """

    def __init__(
        self,
        num_radial: int,
        cutoff: float,
        rbf: Dict[str, str],
        envelope: Dict[str, Union[str, int]],
    ) -> None:
        super().__init__()
        self.inv_cutoff = 1 / cutoff

        env_name = envelope["name"].lower()
        env_hparams = envelope.copy()
        del env_hparams["name"]

        self.envelope: Union[PolynomialEnvelope, ExponentialEnvelope]
        if env_name == "polynomial":
            self.envelope = PolynomialEnvelope(**env_hparams)
        elif env_name == "exponential":
            self.envelope = ExponentialEnvelope(**env_hparams)
        else:
            raise ValueError(f"Unknown envelope function '{env_name}'.")

        rbf_name = rbf["name"].lower()
        rbf_hparams = rbf.copy()
        del rbf_hparams["name"]

        # RBFs get distances scaled to be in [0, 1]
        if rbf_name == "gaussian":
            self.rbf = GaussianSmearing(
                start=0, stop=1, num_gaussians=num_radial, **rbf_hparams
            )
        elif rbf_name == "spherical_bessel":
            self.rbf = SphericalBesselBasis(
                num_radial=num_radial, cutoff=cutoff, **rbf_hparams
            )
        elif rbf_name == "bernstein":
            self.rbf = BernsteinBasis(num_radial=num_radial, **rbf_hparams)
        else:
            raise ValueError(f"Unknown radial basis function '{rbf_name}'.")

    def forward(self, d):
        d_scaled = d * self.inv_cutoff

        env = self.envelope(d_scaled)
        return env[:, None] * self.rbf(d_scaled)


class PolynomialEnvelope(torch.nn.Module):
    """
    Polynomial envelope function that ensures a smooth cutoff.

    Parameters
    ----------
        exponent: int
            Exponent of the envelope function.
    """

    def __init__(self, exponent: int) -> None:
        super().__init__()
        assert exponent > 0
        self.p: float = float(exponent)
        self.a: float = -(self.p + 1) * (self.p + 2) / 2
        self.b: float = self.p * (self.p + 2)
        self.c: float = -self.p * (self.p + 1) / 2

    def forward(self, d_scaled: torch.Tensor) -> torch.Tensor:
        env_val = (
            1
            + self.a * d_scaled**self.p
            + self.b * d_scaled ** (self.p + 1)
            + self.c * d_scaled ** (self.p + 2)
        )
        return torch.where(d_scaled < 1, env_val, torch.zeros_like(d_scaled))


class ExponentialEnvelope(torch.nn.Module):
    """
    Exponential envelope function that ensures a smooth cutoff,
    as proposed in Unke, Chmiela, Gastegger, Schütt, Sauceda, Müller 2021.
    SpookyNet: Learning Force Fields with Electronic Degrees of Freedom
    and Nonlocal Effects
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, d_scaled: torch.Tensor) -> torch.Tensor:
        env_val = torch.exp(-(d_scaled**2) / ((1 - d_scaled) * (1 + d_scaled)))
        return torch.where(d_scaled < 1, env_val, torch.zeros_like(d_scaled))


class SphericalBesselBasis(torch.nn.Module):
    """
    1D spherical Bessel basis

    Parameters
    ----------
    num_radial: int
        Controls maximum frequency.
    cutoff: float
        Cutoff distance in Angstrom.
    """

    def __init__(
        self,
        num_radial: int,
        cutoff: float,
    ) -> None:
        super().__init__()
        self.norm_const = math.sqrt(2 / (cutoff**3))
        # cutoff ** 3 to counteract dividing by d_scaled = d / cutoff

        # Initialize frequencies at canonical positions
        self.frequencies = torch.nn.Parameter(
            data=torch.tensor(np.pi * np.arange(1, num_radial + 1, dtype=np.float32)),
            requires_grad=True,
        )

    def forward(self, d_scaled: torch.Tensor) -> torch.Tensor:
        return (
            self.norm_const
            / d_scaled[:, None]
            * torch.sin(self.frequencies * d_scaled[:, None])
        )  # (num_edges, num_radial)


class BernsteinBasis(torch.nn.Module):
    """
    Bernstein polynomial basis,
    as proposed in Unke, Chmiela, Gastegger, Schütt, Sauceda, Müller 2021.
    SpookyNet: Learning Force Fields with Electronic Degrees of Freedom
    and Nonlocal Effects

    Parameters
    ----------
    num_radial: int
        Controls maximum frequency.
    pregamma_initial: float
        Initial value of exponential coefficient gamma.
        Default: gamma = 0.5 * a_0**-1 = 0.94486,
        inverse softplus -> pregamma = log e**gamma - 1 = 0.45264
    """

    def __init__(
        self,
        num_radial: int,
        pregamma_initial: float = 0.45264,
    ) -> None:
        super().__init__()
        prefactor = binom(num_radial - 1, np.arange(num_radial))
        self.register_buffer(
            "prefactor",
            torch.tensor(prefactor, dtype=torch.float),
            persistent=False,
        )

        self.pregamma = torch.nn.Parameter(
            data=torch.tensor(pregamma_initial, dtype=torch.float),
            requires_grad=True,
        )
        self.softplus = torch.nn.Softplus()

        exp1 = torch.arange(num_radial)
        self.register_buffer("exp1", exp1[None, :], persistent=False)
        exp2 = num_radial - 1 - exp1
        self.register_buffer("exp2", exp2[None, :], persistent=False)

    def forward(self, d_scaled: torch.Tensor) -> torch.Tensor:
        gamma = self.softplus(self.pregamma)  # constrain to positive
        exp_d = torch.exp(-gamma * d_scaled)[:, None]
        return self.prefactor * (exp_d**self.exp1) * ((1 - exp_d) ** self.exp2)


class GaussianSmearing(torch.nn.Module):
    def __init__(
        self,
        start: float = 0.0,
        stop: float = 5.0,
        num_gaussians: int = 50,
    ):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist: Tensor) -> Tensor:
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class ShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x: Tensor) -> Tensor:
        return F.softplus(x) - self.shift


class GaussianFourierProjection(torch.nn.Module):
    """Encoding time steps"""

    def __init__(self, embed_dim, scale=30.0):
        super(GaussianFourierProjection, self).__init__()
        self.W = torch.nn.Parameter(
            torch.randn(embed_dim // 2) * scale, requires_grad=True
        )

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        x_proj = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return x_proj
