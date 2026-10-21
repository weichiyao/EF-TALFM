from __future__ import annotations

import numpy as np 

import torch
from torch import nn
from torch.optim import AdamW, lr_scheduler, Optimizer

from lightning.pytorch import LightningModule

from transformers import (
    get_cosine_schedule_with_warmup, 
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup
)

#Gradient clipping
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
        return np.mean(self.items) if self.items else 0.0

    def std(self):
        return np.std(self.items) if self.items else 0.0
    
    def state_dict(self):
        """Return the state of the queue for saving in a checkpoint."""
        return {"items": self.items}

    def load_state_dict(self, state):
        """Load the state of the queue from a checkpoint."""
        self.items = state["items"]
        

class TrainPLModule(LightningModule):
    def __init__(self, model:nn.Module, cfg):
        """
        model: nn.Module 
            The model to be trained
        lr_scheduler: str
            Can be 'constant', 'cosine', 'polynomical_decay' or 'linear'.
        warmup_steps: int 
            The number of steps for the warmup phase. 
        num_cycles: float
            The number of waves in the cosine schedule (the defaults is to just decrease 
            from the max value to 0 following a half-cosine).
        lr_end: float
            The learning rate to decay to in polynomial_decay.
        power: float
            The decay exponent in polynomial_decay.
        """
        super().__init__()
        self.model              = model
        self.cfg                = cfg
        # Optimizer and LR scheduler
        self.lr                 = cfg.lr 
        self.optimizer          = cfg.optimizer
        self.adam_betas         = cfg.adam_betas
        self.adam_eps           = cfg.adam_eps
        self.weight_decay       = cfg.weight_decay                 
        self.lr_scheduler       = cfg.lr_scheduler
        self.warmup_proportion  = cfg.warmup_proportion
        self.power              = cfg.power  
        self.num_cycles         = cfg.num_cycles   
        self.lr_end             = cfg.lr_end 
        self.lr_interval        = cfg.lr_interval
        self.standparam         = cfg.standparam
        self.log_every_n_steps  = cfg.log_every_n_steps
        self.num_decay_steps    = cfg.num_decay_steps if hasattr(cfg, 'num_decay_steps') else -1 
        self.gradient_clip_adaptive = cfg.gradient_clip_adaptive if hasattr(cfg, 'gradient_clip_adaptive') else False
        self.track_grad_norms   = cfg.track_grad_norms if hasattr(cfg, 'track_grad_norms') else False
        
        if self.gradient_clip_adaptive == 'queue':
            self.gradnorm_queue = Queue()  # Shared gradient norm queue
            self.gradnorm_queue.add(3000)  # Add large value that will be flushed.
        
        # Default value assuming single-GPU or CPU training
        self.sync_dist = False  

    @property
    def log_KeyN_dict(self):
        raise ValueError("Subclasses must define the `log_KeyN_dict` property")
        # return self.model.log_KeyN_dict
    
    def on_train_start(self) -> None:
        # Check Distributed Data Parallel (DDP) or any distributed training strategy is being used
        self.sync_dist = self.trainer.world_size > 1
        # print("\tTraining sync_dist", self.sync_dist)
        
    def on_validation_start(self) -> None:
        # Check Distributed Data Parallel (DDP) or any distributed training strategy is being used
        self.sync_dist = self.trainer.world_size > 1
        # print("\tValidation sync_dist", self.sync_dist)

    def compute_loss(self, batch):
        raise NotImplementedError("Subclasses must implement the `compute_loss` method")
        
    def general_step(self, batch, step='train', on_step:bool=True, on_epoch:bool=True, sync_dist:bool=False):
        # Compute the loss (Reconstruction + KL)
        loss, loss_dict = self.compute_loss(batch)
        # mean over batch dimension if output batch
        loss = loss.mean() 
        if step == 'val' or (((self.global_step + 1) % self.log_every_n_steps == 0) and step == 'train'):
            # log the loss for validation or every log_every_n_steps steps during training
            self.log(
                f'{step}_loss', loss, 
                on_step=on_step, 
                on_epoch=on_epoch, 
                sync_dist=sync_dist,
                reduce_fx='mean'
            )  
            log_loss_dict = {}
            for key, value in loss_dict.items():
                log_loss_dict[f'{step}_{key}'] = value

            self.log_dict(log_loss_dict, 
                    on_step=on_step, 
                    on_epoch=on_epoch, 
                    reduce_fx='sum', 
                    sync_dist=sync_dist
                ) 
        return loss
    
    def general_on_epoch_end(self, step_type:str='train', on_step:bool=True, on_epoch:bool=True):  
        # Calculate and log the average loss at epoch end
        for metric_name, total_n in self.log_KeyN_dict.items():
            if on_step and on_epoch: 
                total_value = self.trainer.logged_metrics.get(f'{step_type}_{metric_name}_epoch', 0.0)
                total_count = self.trainer.logged_metrics.get(f'{step_type}_{total_n}_epoch', 0.0)  
            else:
                total_value = self.trainer.logged_metrics.get(f'{step_type}_{metric_name}', 0.0)
                total_count = self.trainer.logged_metrics.get(f'{step_type}_{total_n}', 0.0)  
                
            # Calculate average
            avg_loss = total_value / (total_count if total_count > 0 else -1.0)  # Avoid division by zeros
            
            # Log average
            self.log(f'{step_type}_Avg_{metric_name}', avg_loss, prog_bar=True, sync_dist=self.sync_dist)


    def training_step(self, batch, batch_idx, on_step:bool=True, on_epoch:bool=True):
        return self.general_step(
            batch, 
            step='train',
            on_step=on_step, 
            on_epoch=on_epoch, 
            sync_dist=self.sync_dist
        )
    
    def on_train_epoch_end(self):
        self.general_on_epoch_end('train', on_step=True, on_epoch=True)
                
    def validation_step(self, batch, batch_idx, on_step:bool=False, on_epoch:bool=True):
        return self.general_step(
            batch, 
            step='val',  
            on_step=on_step, 
            on_epoch=on_epoch, 
            sync_dist=self.sync_dist
        )
    
    def on_validation_epoch_end(self):
        self.general_on_epoch_end('val', on_step=False, on_epoch=True)
    
    def configure_optimizers(self):
        """
        μP version:
        *  uses MuAdam (inherits from torch.optim.Adam, so HF schedulers work)
        *  NEVER scales lr by world-size - μP already makes updates width-invariant
        *  turns weight-decay **off** for LayerNorms & biases
        """
        # Check if `self.trainer` exists to avoid issues during initialization
        # if self.trainer is not None:
        #     # Adjust learning rate based on world size (number of GPUs)
        #     effective_learning_rate = self.lr * self.trainer.world_size
        # else:
        #     # Default to the base learning rate if trainer isn't set yet
        #     effective_learning_rate = self.lr
        if self.optimizer == 'adamw':
            if self.standparam:
                optimizer = AdamW(
                    self.model.parameters(), 
                    lr=self.lr, # effective_learning_rate, 
                    betas=self.adam_betas,
                    weight_decay=self.weight_decay, 
                    eps=self.adam_eps
                )
            else:
                assert hasattr(self.model, 'configure_AdamW'), "Model must implement configure_AdamW when standparam is False"
                optimizer = self.model.configure_AdamW(
                    learning_rate=self.lr,
                    betas=self.adam_betas,
                    weight_decay=self.weight_decay,
                    eps=self.adam_eps
                )
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.lr * param_group.get('lr_scale', 1.0)

        else:
            raise ValueError(f"Optimizer only implemented for 'adamw', received {self.optimizer}.")
        
        # ---------- LR scheduler (unchanged logic) ---------------------------
        if self.lr_scheduler == "none":
            return {"optimizer": optimizer}
        
        self.lr_interval = "epoch" if self.lr_scheduler in {"multistep", "step"} else "step"
        num_training_steps = int(self.trainer.estimated_stepping_batches)
        num_warmup_steps = int(self.warmup_proportion * num_training_steps)  # e.g., 10% of steps for warm-up


        if self.lr_scheduler == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                num_cycles=self.num_cycles
            )
        elif self.lr_scheduler == "cosine_annealing":           # **new option**
            # ── plain cosine annealing to a minimum LR (eta_min) ──
            # • T_max  : total scheduler steps (epochs or batches)
            # • eta_min: your target final LR; fall back to 0 if you don’t pass lr_end
            scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=num_training_steps,
                eta_min=self.lr_end
            )
        elif self.lr_scheduler == 'constant':
            scheduler = get_constant_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=num_warmup_steps
            )
        elif self.lr_scheduler == 'linear':
            scheduler = get_linear_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=num_warmup_steps, 
                num_training_steps=num_training_steps)
        elif self.lr_scheduler == 'polynomial_decay':
            if self.num_decay_steps < 0:
                self.num_decay_steps = num_training_steps
            scheduler = get_polynomial_decay_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=num_warmup_steps, 
                num_training_steps=self.num_decay_steps,
                lr_end=self.lr_end,
                power=self.power
            )
        elif self.lr_scheduler == 'cosine_with_lr_end':
            scheds, milestones = [], []
            if num_warmup_steps > 0 : 
                # 1) warmup 0 -> 1 over warmup_iters
                def warmup_lambda(step):
                    # linearly ramps 0 → 1 over num_warmup_steps
                    return min((step + 1) / float(num_warmup_steps), 1.0)
                scheds.append(lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda))
                milestones.append(num_warmup_steps)
            # 2) cosine from learning_rate -> min_lr over (lr_decay_iters - warmup_iters)
            if self.num_decay_steps < 0:
                self.num_decay_steps = num_training_steps
            scheds.append(
                lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.num_decay_steps - num_warmup_steps,
                    eta_min=self.lr_end))
            milestones.append(self.num_decay_steps)

            # 3) hold constant at min_lr for the rest of training
            #    Lambda returns a fixed multiplicative factor so LR = learning_rate * factor = min_lr
            base_lrs = [g['lr'] for g in optimizer.param_groups]
            lr_lambdas = [(lambda _ , blr=blr: self.lr_end / blr) for blr in base_lrs]
            scheds.append(lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambdas))
            
            scheduler = lr_scheduler.SequentialLR(
                optimizer,
                schedulers=scheds,
                milestones=milestones)
        elif self.lr_scheduler == 'multistep':
            scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50,500,5000], gamma=0.2) 
            #0.001->0.0002->0.00004
            # 1e-3 -> 5e-4 -> 2.5e-4 -> 1.25e-4
        elif self.lr_scheduler == 'step':
            scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
        else:
            raise ValueError(f"Unknown scheduler functions: {self.lr_scheduler}. Only 'cosine', 'constant', 'linear', 'polynomial_decay'.")

        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler,
                                 "interval" : self.lr_interval, 
                                 "frequency": 1}}
    
    def on_save_checkpoint(self, checkpoint):
        
        if self.gradient_clip_adaptive=='queue':
            """Save the queue's state in the checkpoint."""
            checkpoint["gradnorm_queue"] = self.gradnorm_queue.state_dict()

    def on_load_checkpoint(self, checkpoint):
        if self.gradient_clip_adaptive=='queue':
            """Restore the queue's state from the checkpoint."""
            if "gradnorm_queue" in checkpoint: 
                self.gradnorm_queue.load_state_dict(checkpoint["gradnorm_queue"]) 
            else: 
                print("Warning: No gradnorm_queue found in checkpoint.")
                self.gradnorm_queue = Queue()  # Shared gradient norm queue
                self.gradnorm_queue.add(3000)  # Add large value that will be flushed.
             
    def on_before_optimizer_step(self, optimizer):
        """Apply custom gradient clipping logic before the optimizer step.

        It's the ideal place to perform any custom operations before gradients are updated, 
        such as logging or applying custom gradient manipulations like dynamic gradient clipping.

        why not use self.configure_gradient_clipping?
        •	called once per optimizer after backward passes are complete but before optimizer steps.
	    •	Used when you want to define static or predefined clipping behavior for the entire training run.

        Gradient Clipping:
	    •	By default, if you have set gradient_clip_val in the Trainer, 
            Lightning will automatically clip gradients after on_before_optimizer_step with self.clip_gradient
        """
        
        if self.gradient_clip_adaptive == 'queue':
            # Custom adaptive gradient clipping logic
            max_grad_norm = 1.5 * self.gradnorm_queue.mean() + 2 * self.gradnorm_queue.std()

            # Clip gradients using custom logic
            grad_norm = nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=max_grad_norm, norm_type=2.0
            )

            # Update the gradient norm queue with the clipped or actual norm
            if float(grad_norm) > max_grad_norm:
                self.gradnorm_queue.add(float(max_grad_norm))
            else:
                self.gradnorm_queue.add(float(grad_norm))
            
            if self.track_grad_norms:
                # Log the gradient norm for monitoring
                self.log("unclipped grad norm", grad_norm)

                # Optional: Print if clipping occurred
                if float(grad_norm) > max_grad_norm:
                    print(
                        f'Clipped gradient with value {grad_norm:.1f} '
                        f'while allowed {max_grad_norm:.1f}'
                    )
            
    def configure_gradient_clipping(self, optimizer: Optimizer, gradient_clip_val: float | None = None, gradient_clip_algorithm: str | None = None) -> None:
        # return super().configure_gradient_clipping(optimizer, gradient_clip_val, gradient_clip_algorithm)
        if not gradient_clip_val or gradient_clip_val <= 0:
            return
        params = [p for p in self.model.parameters() if p.grad is not None]
        if gradient_clip_algorithm == "norm":
            total_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=float(gradient_clip_val))
            if getattr(self, "track_grad_norms", False) and (self.global_step + 1) % self.log_every_n_steps == 0:
                self.log(
                    "grad_norm/unclipped", 
                     total_norm, 
                     on_step=True, 
                     on_epoch=False,
                     sync_dist=self.sync_dist)
        else:
            torch.nn.utils.clip_grad_value_(params, clip_value=float(gradient_clip_val), force=True)