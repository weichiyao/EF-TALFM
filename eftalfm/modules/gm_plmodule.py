from __future__ import annotations

import torch  
from torch import Tensor  
from eftalfm.modules.train_plmodule import TrainPLModule
from eftalfm.equivariant_free_generative_models.lgm.latent_flow_matching_p import CompletePLatentFlowMatching

 

def reverse_tensor(x):
    return x[torch.arange(x.size(0) - 1, -1, -1)]


class CompletePGenModelModule(TrainPLModule):
    def __init__(self, model:CompletePLatentFlowMatching, cfg):
        super().__init__(model, cfg)
        self.progress_bar = cfg.progress_bar
        self.eval_batch_size = cfg.eval_batch_size
        self.model = model
        self.num_workers = cfg.num_workers 
        self.conditioning_diag_n = getattr(cfg, "gm_conditioning_diag_n", 64)
        
    @property
    def log_KeyN_dict(self): 
        return {}
    
    def pretransform_y(self, y:Tensor):
        return self.model.pretransform_context(y)
    
    def invtransform_y(self, y:Tensor):
        return self.model.invtransform_context(y)
    
    def pretransform_x(self, z:Tensor):
        return self.model.pretransform_latent(z)
    
    def invtransform_x(self, z:Tensor):
        return self.model.invtransform_latent(z)
    
    def path_and_field(self, t:Tensor, x:Tensor, noise:Tensor=None):
        """
        pretransform_x is done inside latent_path_and_field
        """
        return self.model.latent_path_and_field(t, x, noise=noise)

    def sample(self, ns:int, y:Tensor, noise:Tensor=None, seed:int | None=None, return_only_latents:bool=False, **kwargs):
        """
        pretransform_context and invtransform_latent is done inside sample/sample_latents
        """
        if return_only_latents:
            return self.model.sample_latents(ns, y, noise, seed, **kwargs)
        return self.model.sample(ns, y, noise, seed, **kwargs)
        
    def compute_loss(
        self,
        batch,
        compute_logp: bool = False,
        compute_conditioning_diagnostics: bool = False,
    ):
        return self.model.compute_loss(
            batch,
            compute_logp=compute_logp,
            compute_conditioning_diagnostics=compute_conditioning_diagnostics,
            conditioning_diagnostics_n=self.conditioning_diag_n,
        )

    def on_train_start(self) -> None:
        # Check Distributed Data Parallel (DDP) or any distributed training strategy is being used
        self.sync_dist = self.trainer.world_size > 1
    
    def training_step(self, batch, batch_idx, on_step:bool=True, on_epoch:bool=False):
        return self.general_step(
            batch, 
            step='train', 
            on_step=on_step, 
            on_epoch=on_epoch, 
            sync_dist=self.sync_dist
        )

    def validation_step(self, batch, batch_idx, on_step:bool=False, on_epoch:bool=True):
        return self.general_step(
            batch, 
            step='val', 
            on_step=on_step, 
            on_epoch=on_epoch, 
            sync_dist=self.sync_dist
        )
    
    def general_step(self, batch, step='train', prog_bar:bool=True, on_step:bool=True, on_epoch:bool=True, sync_dist:bool=False):
        should_log = (step == 'val') or (((self.global_step + 1) % self.log_every_n_steps == 0) and step == 'train')
        compute_conditioning_diagnostics = (step == 'val') and self.conditioning_diag_n > 0
        # Compute the loss (Reconstruction + KL)
        loss, loss_dict = self.compute_loss(
            batch,
            compute_logp=False,
            compute_conditioning_diagnostics=compute_conditioning_diagnostics,
        )
        # mean over batch dimension if output batch
        loss = loss.mean()

        self.log(
            f'{step}_loss', loss, 
            prog_bar=prog_bar, 
            on_step=on_step, 
            on_epoch=on_epoch,  
            sync_dist=sync_dist
        ) 
        if should_log:
            log_loss_dict = {}
            for key, value in loss_dict.items():
                log_loss_dict[f'{step}_{key}'] = value

            self.log_dict(log_loss_dict, 
                    on_step=on_step, 
                    on_epoch=on_epoch, 
                    reduce_fx='mean', 
                    sync_dist=sync_dist
                ) 

        return loss


    def general_on_epoch_end(self, step_type:str='train', on_step:bool=True, on_epoch:bool=True):  
        if self.trainer.sanity_checking:
            return 
