from pathlib import Path
import time
from typing import cast

import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from likelihood.model.normalizing_flow import BaseFlow
from likelihood.model.utils import AtomicData, load_from_xyz
from mace.tools import torch_geometric as tg

torch.set_float32_matmul_precision("high")

DATA_PATH = Path("src/nacl/data/train128.xyz")
BASE_DIR = Path("src/nacl/cnf")

CONFIGS = {
    "batch_size": 16,
    "optimizer": "AdamW",
    "scheduler_factor": 0.98,
    "scheduler_patience": 20,
    "network": "PaiNN",
    "hidden_channels": 128,
    "num_layers": 4,
    "num_rbf": 64,
    "cutoff": 5.0,
    "num_elements": 17,
    "pbc": True,
    "lr": 5e-4,
    "max_epochs": 200,
    "seed": 42,
    "lr_scheduler_monitor": "train/loss",
    "eval_interval": 50,
    "nll_timesteps": 200,
    "hutchinson_samples": 1,
    "sample_timesteps": 200,
}


class SampleAndEvalEveryN(Callback):
    def __init__(
        self,
        every_n_epochs: int,
        sample_batch,
        sample_dir: Path,
        nll_timesteps: int,
        hutchinson_samples: int,
        sample_timesteps: int,
    ):
        self.every_n_epochs = every_n_epochs
        self.sample_batch = sample_batch
        self.sample_dir = sample_dir
        self.nll_timesteps = nll_timesteps
        self.hutchinson_samples = hutchinson_samples
        self.sample_timesteps = sample_timesteps

    def on_train_epoch_end(self, trainer, pl_module):
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

        out_path = self.sample_dir / f"epoch_{epoch:04d}.xyz"
        with torch.no_grad():
            flow.sample(
                atomic_numbers=batch["atomic_numbers"],
                batch=batch_index,
                cell=cell,
                n_timesteps=self.sample_timesteps,
                out_path=out_path,
            )
        flow.train()


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_dir = BASE_DIR / timestamp
    sample_dir = log_dir / "samples"
    ckpt_dir = log_dir / "checkpoints"
    sample_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(CONFIGS["seed"], workers=True)

    configurations = load_from_xyz(str(DATA_PATH))
    print(f"Loaded {len(configurations)} configurations from {DATA_PATH}")

    atomic_datas = [
        AtomicData.from_config(cfg, cutoff=CONFIGS["cutoff"]) for cfg in configurations
    ]
    if CONFIGS["pbc"]:
        for idx, data in enumerate(atomic_datas):
            cell = getattr(data, "cell", None)
            pbc = getattr(data, "pbc", None)

            if cell is None:
                raise ValueError(f"Configuration #{idx} has no cell but pbc=True.")
            if pbc is None:
                raise ValueError(f"Configuration #{idx} has no pbc flag but pbc=True.")
            if cell.numel() != 9:
                raise ValueError(
                    f"Configuration #{idx} has invalid cell shape {tuple(cell.shape)}."
                )
            if pbc.numel() != 3 or not bool(torch.all(pbc)):
                raise ValueError(
                    f"Configuration #{idx} has non-periodic pbc={pbc.tolist()} while pbc=True."
                )

    for data in atomic_datas:
        setattr(data, "atomic_numbers", data.node_attrs)
        setattr(data, "pos", data.positions)

    train_loader = tg.dataloader.DataLoader(
        dataset=atomic_datas,
        batch_size=CONFIGS["batch_size"],
        shuffle=True,
        drop_last=True,
    )
    sample_batch = next(iter(train_loader))

    model = BaseFlow(
        pbc=CONFIGS["pbc"],
        network_type=CONFIGS["network"],
        hidden_channels=CONFIGS["hidden_channels"],
        num_layers=CONFIGS["num_layers"],
        num_rbf=CONFIGS["num_rbf"],
        cutoff=CONFIGS["cutoff"],
        num_elements=CONFIGS["num_elements"],
        optimizer_type=CONFIGS["optimizer"],
        lr=CONFIGS["lr"],
        lr_scheduler_monitor=CONFIGS["lr_scheduler_monitor"],
        sample_dir=sample_dir,
        factor=CONFIGS["scheduler_factor"],
        patience=CONFIGS["scheduler_patience"],
        data_path=DATA_PATH,
    )

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="flow-nacl-{epoch:03d}",
            every_n_epochs=50,
            save_top_k=-1,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        SampleAndEvalEveryN(
            every_n_epochs=CONFIGS["eval_interval"],
            sample_batch=sample_batch,
            sample_dir=sample_dir,
            nll_timesteps=CONFIGS["nll_timesteps"],
            hutchinson_samples=CONFIGS["hutchinson_samples"],
            sample_timesteps=CONFIGS["sample_timesteps"],
        ),
    ]

    trainer = Trainer(
        accelerator=accelerator,
        devices=1,
        max_epochs=CONFIGS["max_epochs"],
        default_root_dir=str(log_dir),
        callbacks=callbacks,
        log_every_n_steps=1,
    )

    trainer.fit(model, train_dataloaders=train_loader)


if __name__ == "__main__":
    main()
