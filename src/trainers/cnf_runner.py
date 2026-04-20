from __future__ import annotations

import sys
import time
import types
from pathlib import Path
from typing import Any


def build_run_paths(out_root: Path, timestamp: str) -> tuple[Path, Path]:
    run_dir = out_root / timestamp
    ckpt_dir = run_dir / "checkpoints"
    return run_dir, ckpt_dir


def make_epoch_print_callback(every_n: int = 20):
    from lightning.pytorch.callbacks import Callback

    class EpochPrintCallback(Callback):
        """Print one training summary line every `every_n` epochs."""

        def __init__(self, every_n_epochs: int = 20):
            super().__init__()
            self.every_n_epochs = every_n_epochs

        def on_train_epoch_end(self, trainer, pl_module) -> None:
            epoch = trainer.current_epoch + 1
            if epoch % self.every_n_epochs != 0 and epoch != 1:
                return

            loss = trainer.callback_metrics.get("train/loss", float("nan"))
            lr = trainer.optimizers[0].param_groups[0]["lr"]
            print(
                f"Epoch {epoch}/{trainer.max_epochs}  loss={float(loss):.4f}  lr={lr:.2e}"
            )

    return EpochPrintCallback(every_n_epochs=every_n)


def make_fixed_batch_nll_callback(
    every_n_epochs: int,
    sample_batch,
    nll_timesteps: int = 200,
    hutchinson_samples: int = 1,
):
    from typing import cast

    import torch
    from lightning.pytorch.callbacks import Callback

    _bootstrap_likelihood_package()
    from likelihood.model.normalizing_flow import BaseFlow

    class FixedBatchNllCallback(Callback):
        """Log NLL estimates on one fixed train batch every `every_n_epochs`."""

        def __init__(
            self,
            every_n_epochs: int,
            sample_batch,
            nll_timesteps: int = 200,
            hutchinson_samples: int = 1,
        ):
            super().__init__()
            self.every_n_epochs = every_n_epochs
            self.sample_batch = sample_batch
            self.nll_timesteps = nll_timesteps
            self.hutchinson_samples = hutchinson_samples

        def on_train_epoch_end(self, trainer, pl_module) -> None:
            epoch = trainer.current_epoch + 1
            if epoch % self.every_n_epochs != 0:
                return

            flow = cast(BaseFlow, pl_module)
            flow.eval()
            batch = self.sample_batch.to(flow.device)
            batch_index = batch.batch if hasattr(batch, "batch") else batch["batch"]
            cell = batch["cell"] if hasattr(batch, "cell") else None

            num_graphs = int(batch_index.max().item()) + 1
            n_atoms = torch.bincount(batch_index, minlength=num_graphs).to(flow.device)

            with torch.enable_grad():
                nll, nll_noconst = flow.nll(
                    atomic_numbers=batch["atomic_numbers"],
                    pos=batch["pos"],
                    batch=batch_index,
                    cell=cell,
                    n_timesteps=self.nll_timesteps,
                    hutchinson_samples=self.hutchinson_samples,
                    include_prior_constant=None,
                )

            flow.log(
                "train/nll_est",
                nll.mean(),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=num_graphs,
            )
            flow.log(
                "train/nll_est_noconst",
                nll_noconst.mean(),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=num_graphs,
            )
            flow.log(
                "train/nll_est_per_atom",
                (nll_noconst / n_atoms.clamp(min=1)).mean(),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=num_graphs,
            )

            if flow.pbc:
                if cell is None:
                    raise ValueError("cell is required for PBC NLL logging.")
                cell_graph = flow._reshape_batched_cell(cell, batch=batch_index)
                volumes = torch.linalg.det(cell_graph).abs().clamp_min(1e-12)
                prior_const = -n_atoms.to(nll.dtype) * torch.log(volumes.to(nll.dtype))
                flow.log(
                    "train/nll_prior_const",
                    (-prior_const).mean(),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    batch_size=num_graphs,
                )
            else:
                dof = (3 * n_atoms - 3).clamp(min=1).to(torch.get_default_dtype())
                ln2 = torch.log(torch.tensor(2.0, device=flow.device))
                flow.log(
                    "train/bpd_est",
                    (nll_noconst / (dof * ln2)).mean(),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    batch_size=num_graphs,
                )

            flow.train()

    return FixedBatchNllCallback(
        every_n_epochs=every_n_epochs,
        sample_batch=sample_batch,
        nll_timesteps=nll_timesteps,
        hutchinson_samples=hutchinson_samples,
    )


def _bootstrap_likelihood_package() -> None:
    if "likelihood.model" in sys.modules:
        return

    repo_root = Path(__file__).resolve().parents[2]
    src_root = repo_root / "src"
    model_root = src_root / "model"

    likelihood_pkg = sys.modules.get("likelihood")
    if likelihood_pkg is None:
        likelihood_pkg = types.ModuleType("likelihood")
        likelihood_pkg.__path__ = [str(src_root)]
        sys.modules["likelihood"] = likelihood_pkg

    model_pkg = types.ModuleType("likelihood.model")
    model_pkg.__path__ = [str(model_root)]
    setattr(likelihood_pkg, "model", model_pkg)
    sys.modules["likelihood.model"] = model_pkg


def _load_configurations(data_path: Path, cutoff: float, limit_configs: int):
    _bootstrap_likelihood_package()
    from ase.io import read
    from likelihood.model.utils import AtomicData, config_from_atoms, load_from_xyz

    if limit_configs > 0:
        atoms_list = read(str(data_path), index=f":{limit_configs}")
        configurations = [config_from_atoms(atoms) for atoms in atoms_list]
    else:
        configurations = load_from_xyz(str(data_path))

    atomic_datas = [AtomicData.from_config(cfg, cutoff=cutoff) for cfg in configurations]
    for data in atomic_datas:
        setattr(data, "atomic_numbers", data.node_attrs)
        setattr(data, "pos", data.positions)

    return configurations, atomic_datas


def run_single_xyz_training(config: dict[str, Any]) -> Path:
    import torch
    from lightning.pytorch import Trainer, seed_everything
    from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

    _bootstrap_likelihood_package()
    from likelihood.model.normalizing_flow import BaseFlow
    from mace.tools import torch_geometric as tg

    data_path = Path(config["data_path"])
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    timestamp = config.get("timestamp") or time.strftime("%Y%m%d-%H%M%S")
    out_root = Path(config["out_root"])
    run_dir, ckpt_dir = build_run_paths(out_root=out_root, timestamp=timestamp)
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    torch.set_float32_matmul_precision("high")
    seed_everything(int(config["seed"]), workers=True)

    configurations, atomic_datas = _load_configurations(
        data_path=data_path,
        cutoff=float(config["cutoff"]),
        limit_configs=int(config.get("limit_configs", 0)),
    )
    print(f"Loaded {len(configurations)} configurations from {data_path}")

    train_loader = tg.dataloader.DataLoader(
        dataset=atomic_datas,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        drop_last=True,
        num_workers=int(config.get("num_workers", 0)),
    )

    model = BaseFlow(
        pbc=bool(config["pbc"]),
        network_type=str(config["network"]),
        hidden_channels=int(config["hidden_channels"]),
        num_layers=int(config["num_layers"]),
        num_rbf=int(config["num_rbf"]),
        cutoff=float(config["cutoff"]),
        num_elements=int(config["num_elements"]),
        optimizer_type=str(config["optimizer"]),
        lr=float(config["lr"]),
        lr_scheduler_monitor=str(config["lr_scheduler_monitor"]),
        factor=float(config["scheduler_factor"]),
        patience=int(config["scheduler_patience"]),
        data_path=data_path,
    )

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="flow-transition1x-{epoch:03d}",
            every_n_epochs=int(config.get("checkpoint_every_n_epochs", 50)),
            save_top_k=-1,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        make_epoch_print_callback(every_n=int(config.get("print_every_n_epochs", 20))),
    ]
    nll_every_n_epochs = int(config.get("nll_every_n_epochs", 0))
    if nll_every_n_epochs > 0:
        sample_batch = next(iter(train_loader), None)
        if sample_batch is None:
            raise ValueError("Cannot create NLL monitor callback with an empty train loader.")
        callbacks.append(
            make_fixed_batch_nll_callback(
                every_n_epochs=nll_every_n_epochs,
                sample_batch=sample_batch,
                nll_timesteps=int(config.get("nll_timesteps", 200)),
                hutchinson_samples=int(config.get("hutchinson_samples", 1)),
            )
        )

    trainer = Trainer(
        accelerator=accelerator,
        devices=1,
        max_epochs=int(config["max_epochs"]),
        default_root_dir=str(run_dir),
        callbacks=callbacks,
        log_every_n_steps=int(config.get("log_every_n_steps", 1)),
        limit_train_batches=config.get("limit_train_batches", 1.0),
    )
    trainer.fit(model, train_dataloaders=train_loader)
    return run_dir
