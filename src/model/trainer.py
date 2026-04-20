from typing import Dict, Union

import torch
from lightning.pytorch import LightningModule
from loguru import logger as log
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from likelihood.model.utils import Queue


class BaseModel(LightningModule):
    def __init__(
        self,
        # optimizer
        optimizer_type: str = "Adam",
        lr: float = 1e-3,
        factor: float = 0.98,
        patience: int = 10,
        lr_scheduler_monitor: str = "val/loss",
        **kwargs,
    ):
        super().__init__()

        # optimizer
        self.optimizer_type = optimizer_type
        self.lr = lr
        self.opt_betas = (0.95, 0.999)
        self.weight_decay = 0.0
        self.ams_grad = False
        self.grad_norm_max_val = 100.0

        # lr scheduler
        self.factor = factor
        self.patience = patience
        self.lr_scheduler_monitor = lr_scheduler_monitor
        self.lr_scheduler_interval = "epoch"
        self.lr_scheduler_frequency = 1

        # gradient clipping queue
        self.gradnorm_queue = Queue()
        self.gradnorm_queue.add(3000)  # starting value

    def generic_step(self, batch, batch_idx: int, mode: str):
        raise NotImplementedError

    def training_step(self, batch, batch_idx: int):
        return self.generic_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx: int):
        return self.generic_step(batch, batch_idx, "val")

    def configure_optimizers(self) -> Dict[str, Union[Optimizer, LRScheduler]]:
        if self.optimizer_type == "Adam":
            log.info(f"Using Adam optimizer with lr={self.lr}")
            self.optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.lr,
                betas=self.opt_betas,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_type == "AdamW":
            log.info(f"Using AdamW optimizer with lr={self.lr}")
            self.optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.lr,
                betas=self.opt_betas,
                weight_decay=self.weight_decay,
                amsgrad=self.ams_grad,
            )
        else:
            log.info(f"Using SGD optimizer with lr={self.lr}")
            self.optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.lr,
            )

        log.info(
            f"Using ReduceLROnPlateau with factor={self.factor}, patience={self.patience}"
        )
        lr_scheduler_config = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=self.optimizer,
                factor=self.factor,
                patience=self.patience,
            ),
            "monitor": self.lr_scheduler_monitor,
            "interval": self.lr_scheduler_interval,
            "frequency": self.lr_scheduler_frequency,
        }

        return {"optimizer": self.optimizer, "lr_scheduler": lr_scheduler_config}

    def log_helper(self, key: str, value: torch.Tensor, batch_size: int):
        self.log(
            key,
            value,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
            sync_dist=True,
        )

    def on_train_epoch_end(self) -> None:
        if not torch.cuda.is_available():
            return
        # epoch = self.current_epoch + 1
        # device = torch.cuda.current_device()
        # max_mem = torch.cuda.max_memory_allocated(device=device) / (1024**3)
        # if epoch % 20 == 0 or epoch == 1:
        #     log.info(f"CUDA max memory allocated: {max_mem:.1f} GB")
        # torch.cuda.reset_peak_memory_stats(device=device)
        torch.cuda.empty_cache()

    def configure_gradient_clipping(
        self, optimizer, gradient_clip_val, gradient_clip_algorithm
    ):
        """Gradient Clipping as done in the official EDM implementation."""
        # Allow gradient norm to be 150% + 2 * stdev of the recent history.
        max_grad_norm = min(
            1.5 * self.gradnorm_queue.mean() + 2 * self.gradnorm_queue.std(),
            self.grad_norm_max_val,  # do not increase the gradient norm beyond 100
        )
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.parameters(), max_grad_norm, norm_type=2.0
        )

        if float(grad_norm) > max_grad_norm and grad_norm < self.grad_norm_max_val:
            # only update if grad_norm is not too large
            self.gradnorm_queue.add(max_grad_norm)
        else:
            self.gradnorm_queue.add(grad_norm.cpu().item())

        if float(grad_norm) > (10 * max_grad_norm):
            log.info(
                f"Clipped gradient with value {grad_norm:.1f} "
                f"while allowed {max_grad_norm:.1f}"
            )
