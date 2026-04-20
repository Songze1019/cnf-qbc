from pathlib import Path
from pprint import pprint
from typing import cast
import time

import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from likelihood.model.normalizing_flow import BaseFlow
from likelihood.model.utils import AtomicData
from likelihood.model.utils import load_from_xyz
from mace.tools import torch_geometric as tg

torch.set_float32_matmul_precision("high")


# Define configurations
configs = {
    "pwd": "src/au55-ensemble/",
    "batch_size": 16,
    "optimizer": "AdamW",
    "scheduler_factor": 0.98,
    "scheduler_patience": 20,
    "network": "PaiNN",
    "hidden_channels": 128,
    "num_layers": 4,
    "num_rbf": 64,
    "cutoff": 5.0,
    "num_elements": 83,
    "pbc": True,
    "lr": 1e-3,
    "max_epochs": 1000,
    "seed": 42,
    "lr_scheduler_monitor": "train/loss",
    "sample_interval": 200,
}
pprint("Configurations:")
pprint(configs)

# Setup logging directories
pwd = Path(configs["pwd"])
timestamp = time.strftime("%Y%m%d-%H%M%S")
log_dir = pwd / "logs" / timestamp
sample_dir = log_dir / "samples"
log_dir.mkdir(parents=True, exist_ok=True)
sample_dir.mkdir(parents=True, exist_ok=True)


class CarryOverBatchSampler(torch.utils.data.Sampler[list[int]]):
    def __init__(self, data_source, batch_size: int, shuffle: bool, seed: int):
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self._carry: list[int] = []

    def __iter__(self):
        num_samples = len(self.data_source)
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(num_samples, generator=generator).tolist()
        else:
            indices = list(range(num_samples))
        self.epoch += 1

        if self._carry:
            indices = self._carry + indices
            self._carry = []

        for i in range(0, len(indices), self.batch_size):
            batch = indices[i : i + self.batch_size]
            if len(batch) < self.batch_size:
                self._carry = batch
                break
            yield batch

    def __len__(self) -> int:
        return len(self.data_source) // self.batch_size


# Load and preprocess data
data_path = pwd / "data/test0.xyz"
configurations = load_from_xyz(str(data_path))[:64]
# configurations = [
#     load_from_xyz(str(data_path))[0],
# ] * 512

print(f"Loaded {len(configurations)} configurations from {data_path}")
atomic_datas = [
    AtomicData.from_config(cfg, cutoff=configs["cutoff"]) for cfg in configurations
]
print(f"Converted configurations to AtomicData format.")
print("Check PBC of the first configuration:", getattr(atomic_datas[0], "pbc", None))

# Alias fields to match BaseFlow expectations.
for data in atomic_datas:
    setattr(data, "atomic_numbers", data.node_attrs)
    setattr(data, "pos", data.positions)

seed_everything(configs["seed"], workers=True)
batch_sampler = CarryOverBatchSampler(
    atomic_datas,
    batch_size=configs["batch_size"],
    shuffle=True,
    seed=configs["seed"],
)
train_loader = tg.dataloader.DataLoader(
    dataset=atomic_datas,  # pyright: ignore[reportArgumentType]
    batch_sampler=batch_sampler,
)
sample_batch = next(iter(train_loader))

# Initialize the Normalizing Flow model
model = BaseFlow(
    pbc=configs["pbc"],
    network_type=configs["network"],
    hidden_channels=configs["hidden_channels"],
    num_layers=configs["num_layers"],
    num_rbf=configs["num_rbf"],
    cutoff=configs["cutoff"],
    num_elements=configs["num_elements"],
    optimizer_type=configs["optimizer"],
    lr=configs["lr"],
    lr_scheduler_monitor=configs["lr_scheduler_monitor"],
    sample_dir=sample_dir,
    factor=configs["scheduler_factor"],
    patience=configs["scheduler_patience"],
    data_path=data_path,
)
print("Initialized BaseFlow model:")
print(model)

accelerator = "gpu" if torch.cuda.is_available() else "cpu"
callbacks = [
    ModelCheckpoint(
        dirpath=log_dir / "checkpoints",
        filename="flow-{epoch:03d}",
        every_n_epochs=1000,
        save_top_k=-1,
        save_last=True,
    ),
    LearningRateMonitor(logging_interval="epoch"),
]


class SampleEveryN(Callback):
    def __init__(
        self,
        sample_interval,
        sample_batch,
        sample_dir,
        nll_timesteps: int = 200,
        hutchinson_samples: int = 1,
    ):
        self.sample_interval = sample_interval
        self.sample_batch = sample_batch
        self.sample_dir = sample_dir
        self.nll_timesteps = nll_timesteps
        self.hutchinson_samples = hutchinson_samples

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1

        do_nll = epoch % 200 == 0
        do_sample = self.sample_interval > 0 and epoch % self.sample_interval == 0
        if not (do_nll or do_sample):
            return

        flow = cast(BaseFlow, pl_module)
        flow.eval()
        batch = self.sample_batch.to(flow.device)
        batch_index = batch.batch if hasattr(batch, "batch") else batch["batch"]

        # Compute an approximate NLL on a single (fixed) train batch.
        # This uses CNF-style likelihood estimation with a Hutchinson divergence
        # estimator, so it is more expensive than the flow-matching MSE loss.
        if do_nll:
            num_graphs = int(batch_index.max().item()) + 1
            n_atoms = torch.bincount(batch_index, minlength=num_graphs).to(flow.device)
            dof = (3 * n_atoms - 3).clamp(min=1).to(torch.get_default_dtype())
            ln2 = torch.log(torch.tensor(2.0, device=flow.device))

            with torch.enable_grad():
                nll, nll_noconst = flow.nll(
                    atomic_numbers=batch["atomic_numbers"],
                    pos=batch["pos"],
                    batch=batch_index,
                    cell=batch["cell"] if hasattr(batch, "cell") else None,
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
            flow.log(
                "train/bpd_est",
                (nll_noconst / (dof * ln2)).mean(),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=num_graphs,
            )

        if do_sample:
            out_path = self.sample_dir / f"epoch_{epoch:04d}.xyz"
            with torch.no_grad():
                flow.sample(
                    atomic_numbers=batch["atomic_numbers"],
                    batch=batch_index,
                    out_path=out_path,
                )
        flow.train()


callbacks.append(SampleEveryN(configs["sample_interval"], sample_batch, sample_dir))

trainer = Trainer(
    accelerator=accelerator,
    devices=1,
    max_epochs=configs["max_epochs"],
    default_root_dir=str(log_dir),
    callbacks=callbacks,
    log_every_n_steps=1,
)

trainer.fit(model, train_dataloaders=train_loader)
