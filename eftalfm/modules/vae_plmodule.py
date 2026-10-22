from __future__ import annotations

import torch
from torch import nn
from torch import Tensor

from model_training.lightning import TrainPLModule
from model_training.schedules import get_kl_weight


def _as_log_scalar(value, move_cpu: bool = True):
    if torch.is_tensor(value):
        value = value.detach().float().mean()
        return value.cpu() if move_cpu else value
    return value

class CloudVAEPLModule(TrainPLModule):
    def __init__(self, model:nn.Module, cfg):
        super().__init__(model, cfg)
        self.kl_weight                = cfg.kl_weight
        self.kl_weight_min            = cfg.kl_weight_min
        self.kl_weight_scheduler      = cfg.kl_weight_scheduler
        self.kl_weight_scaling_factor = cfg.kl_weight_scaling_factor
        self.kl_weight_warmup_fraction = getattr(cfg, "kl_weight_warmup_fraction", 0.2)
        self.pos_weight               = cfg.pos_weight if hasattr(cfg, 'pos_weight') else 1.
        self.seq_weight               = cfg.seq_weight if hasattr(cfg, 'seq_weight') else 1.  
        self.charge_weight            = cfg.charge_weight if hasattr(cfg, 'charge_weight') else 1.
        self.spinmp_weight            = cfg.spinmp_weight if hasattr(cfg, 'spinmp_weight') else 1.
        self.ring_weight              = cfg.ring_weight if hasattr(cfg, 'ring_weight') else 1. 
        self.ctx_weight               = cfg.ctx_weight if hasattr(cfg, 'ctx_weight') else 1.
        # initialize wkl to the initial kl weight
        self.wkl = self.kl_weight
        
    @property
    def metric_count_keys(self) -> dict:
        return {}

    def _refresh_wkl(self) -> None:
        """Recompute self.wkl from the schedule using whatever progress signal
        is appropriate for the current training mode (step- vs epoch-limited).

        Uses (global_step, max_steps) under step-limited training, otherwise
        (current_epoch, max_epochs).  Both code paths gracefully degrade to a
        constant max_beta when the horizon is unknown (-1 / None), so the
        scheduler is safe under either setting.
        """
        trainer = getattr(self, "trainer", None)
        if self._is_step_limited_training():
            current = int(getattr(trainer, "global_step", 0) or 0)
            total = int(getattr(trainer, "max_steps", -1) or -1)
        else:
            current = int(getattr(self, "current_epoch", 0) or 0)
            total = int(getattr(trainer, "max_epochs", -1) or -1) if trainer is not None else -1

        self.wkl = get_kl_weight(
            current,
            total,
            min_beta=self.kl_weight_min,
            max_beta=self.kl_weight,
            method=self.kl_weight_scheduler,
            scaling_factor=self.kl_weight_scaling_factor,
            warmup_fraction=self.kl_weight_warmup_fraction,
        )

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self._refresh_wkl()

    def on_train_batch_start(self, batch, batch_idx):
        # Step-based refresh: ensures the schedule advances inside a long epoch
        # (important under step-limited training, where on_train_epoch_start
        # might not fire often enough to drive a meaningful ramp).
        self._refresh_wkl()

    def on_validation_start(self) -> None:
        super().on_validation_start()
        # NOTE: compute_loss() in validation mode hardcodes wkl=0., so the
        # value of self.wkl during validation does not affect the val loss.
        # We deliberately do NOT overwrite self.wkl here so that training
        # resumes with the correctly scheduled weight after validation
        # (especially when Lightning runs mid-epoch validation under
        # step-limited training).
 

    def _has_multiple_val_loaders(self) -> bool:
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return False
        num_val_dataloaders = getattr(trainer, "num_val_dataloaders", 1)
        if num_val_dataloaders is None:
            num_val_dataloaders = 1
        val_dataloaders = getattr(trainer, "val_dataloaders", None)
        if isinstance(val_dataloaders, (list, tuple)):
            num_val_dataloaders = max(int(num_val_dataloaders), len(val_dataloaders))
        return int(num_val_dataloaders) > 1

    def _validation_loader_name(self, dataloader_idx: int) -> str:
        trainer = getattr(self, "trainer", None)
        datamodule = getattr(trainer, "datamodule", None) if trainer is not None else None
        names = getattr(datamodule, "val_loader_names", None)
        if dataloader_idx >= len(names):
            return f"loader{dataloader_idx}"
        return names[dataloader_idx]

    def _validation_step_name(self, dataloader_idx: int) -> str:
        trainer = getattr(self, "trainer", None)
        datamodule = getattr(trainer, "datamodule", None) if trainer is not None else None
        names = getattr(datamodule, "val_loader_names", None)
        dataloader_idx = int(dataloader_idx)
        if names:
            return f"val_{names[dataloader_idx]}" if dataloader_idx < len(names) else f"val_loader{dataloader_idx}"
        if self._has_multiple_val_loaders():
            return f"val_{self._validation_loader_name(dataloader_idx)}"
        return "val"
        
    def compute_loss(self, batch) -> tuple[Tensor, dict, dict]:
        if self.training:
            loss_weights = {
                "wkl" : self.wkl,
                "wpos" : self.pos_weight,
                "wseq" : self.seq_weight, 
                "wcharge" : self.charge_weight,
                "wspinmp" : self.spinmp_weight,
                "wring" : self.ring_weight, 
                "wctx" : self.ctx_weight,
            }
        else:
            loss_weights = {
                "wkl" : 0.0,
                "wpos" : 1.0,
                "wseq" : 1.0, 
                "wcharge" : 1.0,
                "wspinmp" : 1.0,
                "wring" : 1.0, 
                "wctx" : 1.0
            }
        return self.model.compute_loss(batch, **loss_weights)

    def _is_step_limited_training(self) -> bool:
        trainer = getattr(self, "trainer", None)
        return (
            trainer is not None
            and getattr(trainer, "max_steps", -1) > 0
            and getattr(trainer, "max_epochs", -1) < 0
        )

    def _log_scalar_dict(
        self,
        values: dict,
        *,
        step: str,
        on_step: bool,
        on_epoch: bool,
        sync_dist: bool,
        reduce_fx: str = "mean",
        batch_size: int | None = None,
    ) -> None:
        if not values:
            return

        move_cpu = not sync_dist
        log_kwargs = {
            "on_step" : on_step,
            "on_epoch" : on_epoch,
            "reduce_fx" : reduce_fx,
            "sync_dist": sync_dist,
            "add_dataloader_idx": False,
        }
        if batch_size is not None:
            log_kwargs["batch_size"] = batch_size

        self.log_dict(
            {
                f"{step}_{key}": _as_log_scalar(value, move_cpu=move_cpu)
                for key, value in values.items()
            },
            **log_kwargs,
        )

    def general_step(
        self,
        batch,
        step='train',
        on_step:bool=True,
        on_epoch:bool=True,
        sync_dist:bool=False,
        batch_idx:int=0,
    ): 
        loss, loss_dict  = self.compute_loss(batch)
        loss = loss.mean()

        log_kwargs = {
            "step": step,
            "on_step" : on_step,
            "on_epoch" : on_epoch,
            "sync_dist" : sync_dist,
        }
        self._log_scalar_dict({"loss": loss}, **log_kwargs)
        self._log_scalar_dict({key: value for key, value in loss_dict.items()}, **log_kwargs)
        return loss

    def training_step(self, batch, batch_idx, on_step:bool=True, on_epoch:bool=True):
        if self._is_step_limited_training():
            on_step = True
            on_epoch = False
        else:
            on_step = False
            on_epoch = True
        return self.general_step(
            batch,
            step='train',
            on_step=on_step,
            on_epoch=on_epoch,
            sync_dist=self.sync_dist,
            batch_idx=batch_idx,
        )

    def validation_step(
        self,
        batch,
        batch_idx,
        dataloader_idx: int = 0,
        on_step: bool = False,
        on_epoch: bool = True,
    ):
        return self.general_step(
            batch,
            step=self._validation_step_name(dataloader_idx),
            on_step=on_step,
            on_epoch=on_epoch,
            sync_dist=self.sync_dist,
            batch_idx=batch_idx,
        )

    def on_train_epoch_end(self): 
        # Keep Lightning's default behavior (no-op) instead of TrainPLModule override.
        pass

    def on_validation_epoch_end(self):  
        # Keep Lightning's default behavior (no-op) instead of TrainPLModule override.
        pass
    
