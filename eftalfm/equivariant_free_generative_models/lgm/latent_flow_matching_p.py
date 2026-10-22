from __future__ import annotations

import torch
from torch import nn
from torch import Tensor
from torchdiffeq import odeint

import numpy as np

 
from eftalfm.equivariant_free_generative_models.vae.vae_models_pcqm4mv2 import CloudVAEModelTransformerP
from eftalfm.equivariant_free_generative_models.lgm.dynamics import CompletePFlowDynamics  
from eftalfm.equivariant_free_generative_models.utils import (
    expand_to_match_size, 
    sum_except_batch, 
    whiten_transform_torch, 
    whiten_inverse_torch, 
    pca_precondition_transform_torch, 
    pca_precondition_inverse_torch

)

from typing import Callable 

_MSE_LOSSES = {"l2", "mse"}
_MAE_LOSSES = {"l1", "mae"}
_SUPPORTED_POINTWISE_LOSSES = _MSE_LOSSES | _MAE_LOSSES | {"smoothl1"}


class HutchinsonNoiseError(ValueError):
    """Raised when an unsupported Hutchinson noise distribution is requested."""


def _unsupported_loss_message(loss_type: str) -> str:
    return (
        f"wrong loss type {loss_type}, only accept "
        "'l2'/'mse', 'l1'/'mae', and 'smoothl1'"
    )


# ----------------------------
# Euler ODE Integrator for reverse path
# ----------------------------
@torch.no_grad()
def euler_integrate_reverse(vf: Callable, y0: Tensor, t_start: float, t_end: float, steps: int = 25) -> Tensor:
    y = y0
    ts = torch.linspace(t_start, t_end, steps + 1, device=y0.device, dtype=y0.dtype)
    for i in range(steps):
        t = ts[i]
        dt = ts[i + 1] - ts[i]
        dy = vf(t, y)
        y = y + dt * dy
    return y

 
class CNFlows(nn.Module):
    """
    Adapated from Continuous-time flow FFJORD [1], with github scripts in 
    https://github.com/vgsatorras/en_flows/
    and 
    https://gist.github.com/francois-rozet/fd6a820e052157f8ac6e2aa39e16c1aa

    Args:
        dynamics (nn.Module): The ODE dynamics function f(t,x).
        trace_method (str): The trace estimation method. One of {'exact', 'hutch'}.

    References:
        [1] FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models,
            Grathwohl et al., 2019, https://arxiv.org/abs/1810.01367
    """
    def __init__(
            self, 
            dynamics:CompletePFlowDynamics, 
            ode_method:str="dopri5",
            ode_rtol:float=1e-4,
            ode_atol:float=1e-4,
            ode_rtol_test:float=1e-7,
            ode_atol_test:float=1e-7,
            trace_method:int='exact', 
            ode_regularization:float=0, 
            hutch_noise:str='gaussian', 
            loss_type:str="l2"
        ):
        super().__init__()
        self.dynamics = dynamics  

        self.hutch_noise = hutch_noise
        self.trace_method = trace_method
        self.ode_regularization = ode_regularization
        
        self.loss_type = loss_type 

        self.set_integration_time() 
        # assert loss_type in {'ot'}
        self.set_odeint(method=ode_method, rtol=ode_rtol, atol=ode_atol, 
                        rtol_test=ode_rtol_test, atol_test=ode_atol_test)
    
    def set_odeint(self, method="dopri5", rtol: float = 1e-4, atol: float = 1e-4, 
                   rtol_test:float=1e-7,
                   atol_test:float=1e-7):
        self.ode_method = method
        self._atol = atol
        self._rtol = rtol
        self._atol_test = atol_test
        self._rtol_test = rtol_test
    
    def set_trace(self, trace):
        assert trace == 'exact' or trace == 'hutch'
        self.odefunc.method = trace
    
    def loss_funct(self, x:Tensor, y:Tensor):
        if self.loss_type in _MSE_LOSSES:
            loss_funct = nn.MSELoss(reduction="none") 
        elif self.loss_type in _MAE_LOSSES:
            loss_funct = nn.L1Loss(reduction="none")
        elif self.loss_type == "smoothl1":
            loss_funct = nn.SmoothL1Loss(reduction="none", beta=0.01)
        else:
            raise ValueError(_unsupported_loss_message(self.loss_type))
        return loss_funct(x, y)
    
    @staticmethod
    def hutch_trace(f, y:Tensor, e=None):
        """Hutchinson's estimator for the Jacobian trace"""
        e_dzdx = torch.autograd.grad(f, y, e, create_graph=True)[0]
        e_dzdx_e = e_dzdx * e
        approx_tr_dzdx = sum_except_batch(e_dzdx_e)
        return approx_tr_dzdx

    @staticmethod
    def only_frobenius(f, y:Tensor, e=None):
        """Hutchinson's estimator for the Jacobian trace"""
        e_dzdx = torch.autograd.grad(f, y, e, create_graph=True)[0]
        frobenius = sum_except_batch(e_dzdx.pow(2))
        return frobenius
    
    @staticmethod
    def hutch_trace_and_frobenius(f, y:Tensor, e=None):
        """Hutchinson's estimator for the Jacobian trace"""
        e_dzdx = torch.autograd.grad(f, y, e, create_graph=True)[0]
        frobenius = sum_except_batch(e_dzdx.pow(2))
        e_dzdx_e = e_dzdx * e
        approx_tr_dzdx = sum_except_batch(e_dzdx_e)
        return approx_tr_dzdx, frobenius

    @staticmethod
    def exact_trace(f, y:Tensor):
        """Exact Jacobian trace
        
        Original:
        ```
        dims = y.size()[1:]
        tr_dzdx = 0.0
        dim_ranges = [range(d) for d in dims]
        for idcs in itertools.product(*dim_ranges):
            batch_idcs = (slice(None),) + idcs
            tr_dzdx += torch.autograd.grad(f[batch_idcs].sum(), y, create_graph=True)[0][batch_idcs]
        return tr_dzdx
        ```
        The following is more efficient with a weight 1e-2 adapted from
        https://gist.github.com/francois-rozet/fd6a820e052157f8ac6e2aa39e16c1aa
        
        """

        # Get the batch size and number of dimensions
        batch_size, num_dims = y.shape
        
        # Create the identity matrix for efficient Jacobian computation
        eye_mat = torch.eye(num_dims, dtype=y.dtype, device=y.device)

        # Expand the identity matrix to match the batch size
        eye_mat = eye_mat.expand(batch_size, num_dims, num_dims).movedim(-1, 0)

        # Compute the Jacobian with respect to x (ignore t)
        jacobian = torch.autograd.grad(f, y, eye_mat, create_graph=True, is_grads_batched=True)[0]

        # Compute the trace of the Jacobian (sum of diagonal elements)
        trace = torch.einsum('i...i', jacobian) 
        return trace * 1e-2

    def before_odeint(self, x:Tensor):
        self.num_evals = 0
        if self.trace_method == 'hutch':

            if self.hutch_noise == 'gaussian':
                # With _eps ~ Normal(0, 1).
                self._eps = torch.randn_like(x)
            elif self.hutch_noise == 'bernoulli':
                # With _eps ~ Rademacher (== Bernoulli on -1 +1 with 50/50 chance).
                self._eps = torch.randint(low=0, high=2, size=x.size()).to(x) * 2 - 1
            else:
                raise HutchinsonNoiseError("Wrong hutchinson noise type")
        
    @property
    def atol(self):
        return self._atol if self.training else self._atol_test

    @property
    def rtol(self):
        return self._rtol if self.training else self._rtol_test
    
    def set_integration_time(self, times=None):
        if times is None:
            times = [0.0, 1.0]
        device = next(iter(self.dynamics.parameters())).device
        self.register_buffer('int_time', torch.tensor(times, dtype=torch.float32, device=device))
        self.register_buffer('inv_int_time', torch.tensor(list(reversed(times)), dtype=torch.float32, device=device))
    
    def path_and_field(self, t:Tensor, x:Tensor, noise:Tensor=None, sigma:float=1e-7):
        """
        Compute the conditional vector field ut(x1|x0) = x1 - x0, see Eq.(15) [1].

        Inputs
        ----------
        x : (bz, k_1,...,k_n, d)
            the target minibatch
        t : (bz, k_1,...,k_n, d_t) or (bz, 1_1,...,1_n, d_t) or (d_t,)
            shape (bz, ) will raise confusion and cause errors 
        noise : same shape as x or (bz, 1_1, ..., 1_n, d)

        Returns
        -------
        ut : conditional vector field ut(x1|x0) 
        pt : phi_t(x1|x0)

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.


        Reverse time, x0 for sample, x1 for noise
        t <- 1 - t 
        """
        
        t = expand_to_match_size(t, x)
        if noise is None:
            # z the source (noise) minibatch
            noise = torch.randn_like(x)  
        # path
        p = (1 - t) * x + (sigma + (1 - sigma) * t) * noise
        # field
        u = (1 - sigma) * noise - x
        return p, u
    
    def compute_loss(self, x: Tensor, noise: Tensor=None):
        """
        x: (bz, d)
        """
        bz = x.size(0)
        t = torch.rand((bz,1), dtype=x.dtype, device=x.device) # (bz, 1)
        
        
        p, u = self.path_and_field(t, x, noise=noise)  # (bz, d) and (bz, d)
        vhat = self.dynamics(t, p)

        return sum_except_batch(self.loss_funct(u, vhat))

    def log_prob(self, x: Tensor) -> Tensor:
        """
        # Forward pass (x -> ODE -> z)
        """
        ldj = x.new_zeros(x.shape[0])
        reg_term = x.new_zeros(x.shape[0])
        
        ldj_act = x.new_zeros(x.shape[0])
        
        dtype = x.dtype 

        # always keep it odeint run in float 
        state = (x.float(), ldj.float(), reg_term.float())
         
        self.before_odeint(x.float())
        statet = odeint(
            self.ode_forward, 
            state, 
            self.int_time.float(),
            method=self.ode_method,
            rtol=self.rtol,
            atol=self.atol
        )
        
        zt, ldjt, reg_termt = statet 
        # cast back to original dtype
        z, ldj, reg_term = zt[-1].to(dtype), ldjt[-1].to(dtype), reg_termt[-1].to(dtype)
        
        return self.prior_likelihood(z) + (ldj_act + ldj*1e2) + reg_term
    
    def prior_likelihood(self, z: Tensor): 
        """
        z: ()
        """
        log_px = sum_except_batch(-0.5 * z.square() - 0.5 * torch.log(z.new_tensor(2 * torch.pi))) / 2
        
        return log_px
             
        
    @torch.no_grad()
    def reverse_fn(self, z:Tensor, **kwargs):
        def wrapper(t:Tensor, x:Tensor):
            """
            t:(,)
            z:(bz,d)
            """
            dx = self.dynamics(t.reshape(-1), x, **kwargs) 
            return dx 
        
        # always keep it odeint run in float 
        dtype = z.dtype 
        self.before_odeint(z.float())
        xt = odeint(
            wrapper, 
            z.float(), 
            self.inv_int_time.float(),
            method=self.ode_method,
            rtol=self.rtol,
            atol=self.atol
        )

        # cast back to the original dtype
        xt = xt.to(dtype)
         
        return xt

    def reverse(self, z, **kwargs):
        xt = self.reverse_fn(z, **kwargs)
        x = xt[-1]
        return x

    def reverse_chain(self, z, **kwargs):
        self.set_integration_time(times=list(np.linspace(0, 1, 50)))
        xt = self.reverse_fn(z, **kwargs)
        # reset
        self.set_integration_time(times=[0.0, 1.0])
        return xt
    
    @torch.no_grad()
    def sample(self, n_samples:int, **kwargs):
        """
        Draw samples from the generative model.
        """
        device = next(self.dynamics.parameters()).device
        dtype  = next(self.dynamics.parameters()).dtype

        z = torch.randn((n_samples, self.dynamics.x_dim), device=device, dtype=dtype)
             
        x = self.reverse_fn(z, **kwargs)
        return x
    
    @torch.no_grad()
    def sample_chain(self, n_samples:int, **kwargs):
        """
        Draw samples from the generative model.
        """
        device = next(self.dynamics.parameters()).device
        z = torch.randn((n_samples, self.dynamics.x_dim), device=device)
             
        x = self.reverse_chain(z, **kwargs)
        return x
    
    def ode_forward(self, t:Tensor, state:tuple[Tensor], **kwargs):
        x, ldj, reg_term = state

        self.num_evals += 1
        with torch.set_grad_enabled(True):
            x.requires_grad_(True)
            t.requires_grad_(True)

            # We always need the dynamics :).
            dx = self.dynamics(t, x, **kwargs)

            if self.ode_regularization > 0:
                # L2-squared norm of (dx)
                dx2 = sum_except_batch(dx.pow(2))

                # If trace is computed exact, frobenius norm is still estimated.
                if self.trace_method == 'exact':
                    ldj = self.exact_trace(dx, x) 
                    frobenius = self.only_frobenius(dx, x, e=self._eps)

                # Combined computation for trace and frobenius estimators.
                elif self.trace_method == 'hutch':
                    ldj, frobenius = self.hutch_trace_and_frobenius(dx, x, e=self._eps)

                reg_term = frobenius + dx2

            else:
                if self.trace_method == 'exact':
                    ldj = self.exact_trace(dx, x) 

                elif self.trace_method == 'hutch':
                    ldj = self.hutch_trace(dx, x, e=self._eps)

                # No regularization terms, set to zero.
                reg_term = torch.zeros_like(ldj)

        return dx, ldj * 1e-2, reg_term

class CompletePLatentFlowMatching(CNFlows):
    def __init__(
            self,
            precomputed_latent:bool=False,
            latent_model:CloudVAEModelTransformerP | None=None,
            latent_stats:dict | None=None, 
            conditioning:bool=False,
            pretransform_context:Callable | None=None,
            invtransform_context:Callable | None=None,  
            t_lower_eps:float=1e-12,
            t_upper_eps:float=1e-12,
            t_norm_scale_cutoff:float=1.0, # by default, no t norm scaling
            dynamics:CompletePFlowDynamics=None, 
            cfg_cond_drop_prob:float=0.0,
            cfg_guidance_scale:float=1.0, 
            ode_method:str="dopri5",  
            ode_rtol:float=1e-4,
            ode_atol:float=1e-4,
            ode_rtol_test:float=1e-7,
            ode_atol_test:float=1e-7,
            ode_steps:int=100,
            trace_method:int='exact', 
            ode_regularization:float=0, 
            hutch_noise:str='gaussian', 
            loss_type:str="l2",
            compile_model:bool=False, 
            **kwargs):
        super().__init__(
            dynamics, 
            ode_method,
            ode_rtol,
            ode_atol,
            ode_rtol_test,
            ode_atol_test,
            trace_method, 
            ode_regularization, 
            hutch_noise, 
            loss_type
        )  
        """
        For strict flow-based generative modeling, each layer's log-det is accounted for. 
        See above Cnflows.

        Flow Matching / OT doesn't rely on the log-det for training. 
        Precompute (mean, std) across the dataset once, and apply it. 
        This is often more stable and consistent at the start.
        """  
        self.t_lower_eps = max(t_lower_eps, self.dynamics.t_lower_eps)
        self.t_upper_eps = max(t_upper_eps, self.dynamics.t_upper_eps)
        print(f"t_lower_eps={self.t_lower_eps:.4f}, t_upper_eps={self.t_upper_eps:.4f}")
        self.ode_steps = ode_steps 
        self.reverse_fn = self.get_reverse_fn(self.ode_method)
        print(f"reverse_fn set to be {self.reverse_fn.__name__}")
 
        if not 0.0 <= cfg_cond_drop_prob <= 1.0:
            raise ValueError(f"cfg_cond_drop_prob must be in [0, 1], got {cfg_cond_drop_prob}.")
        self.cfg_cond_drop_prob = float(cfg_cond_drop_prob)
        self.cfg_guidance_scale = float(cfg_guidance_scale)
        print(
            "classifier-free guidance: "
            f"cond_drop_prob={self.cfg_cond_drop_prob:.4f}, "
            f"guidance_scale={self.cfg_guidance_scale:.4f}"
        )
        self.t_norm_scale_cutoff = t_norm_scale_cutoff
        print(f"t_norm_scale_cutoff={self.t_norm_scale_cutoff:.4f}")
        self.precomputed_latent = precomputed_latent
           
        self.conditioning = conditioning
        self._pretransform_context = pretransform_context
        self._invtransform_context = invtransform_context
        self.latent_stats = latent_stats 
          
        self.instantiate_first_stage(latent_model)
         
        self.context_dim = self.dynamics.y_dim
        self.feature_dim = self.dynamics.x_dim

        self.compile_model = compile_model
        print(f"!!!compile_model: {self.compile_model!s}")
        if self.compile_model:
            self.dynamics = torch.compile(self.dynamics)
            self.latent_model = torch.compile(self.latent_model)
            print("!!!Dynamics model and latent model compiled")
        
    def pretransform_context(self, y: Tensor):
        if self._pretransform_context is not None:
            return self._pretransform_context(y)
        else:
            return y
        
    def invtransform_context(self, y: Tensor):
        if self._invtransform_context is not None:
            return self._invtransform_context(y)
        else:
            return y

    def _cfg_condition_mask(self, y: Tensor) -> Tensor | None:
        if y is None or (not self.training) or self.cfg_cond_drop_prob <= 0:
            return None

        return torch.rand(y.shape[0], device=y.device) >= self.cfg_cond_drop_prob
    
    def _guided_dynamics(
        self,
        t: Tensor,
        x: Tensor,
        y: Tensor = None,
        guidance_scale: float=1.0,
        y_cond_mask: Tensor = None,
    ) -> Tensor:
        if guidance_scale is None:
            guidance_scale = self.cfg_guidance_scale
        guidance_scale = float(guidance_scale)

        if y is None or guidance_scale == 1.0:
            return self.dynamics(t=t, x=x, y=y, y_cond_mask=y_cond_mask)
        if guidance_scale == 0.0:
            return self.dynamics(t=t, x=x, y=None)

        v_uncond = self.dynamics(t=t, x=x, y=None)
        v_cond = self.dynamics(t=t, x=x, y=y, y_cond_mask=y_cond_mask)
        return v_uncond + guidance_scale * (v_cond - v_uncond)
    
    def pretransform_latent(self, z:Tensor):
        """
        z: (b, d)
        """
        logdet = 0. 
        sum_log_std = 0
        if self.latent_stats is not None: 

            if 'evecs_k' in self.latent_stats:
                z = pca_precondition_transform_torch(z, self.latent_stats)
            elif 'evecs' in self.latent_stats:
                z = whiten_transform_torch(z[..., self.latent_stats['active']], self.latent_stats)
            else:
                mean = self.latent_stats["mean"].to(z.device)
                std  = self.latent_stats["std"].to(z.device)
                z = (z - mean) / std

                sum_log_std = torch.sum(torch.log(std))
            logdet = z.new_full((z.shape[0],), -sum_log_std)

        
        return z, logdet
     
    def invtransform_latent(self, z:Tensor):
        """
        z: (b, d)
        """
        if self.latent_stats is not None:
            if 'evecs_k' in self.latent_stats:
                z = pca_precondition_inverse_torch(z, self.latent_stats)
            elif 'evecs' in self.latent_stats:
                b = z.shape[0]
                z_active = whiten_inverse_torch(z, self.latent_stats)
                z_full = torch.zeros((b, self.latent_stats['d']), device=z.device, dtype=z.dtype)
                z_full[..., self.latent_stats['active']] = z_active
                z = z_full
            else:
                z = z * self.latent_stats['std'].to(z.device) + self.latent_stats['mean'].to(z.device)

        
        return z

    def forward(self, z:Tensor, t:Tensor=None, y:Tensor=None, y_cond_mask:Tensor=None) -> Tensor:
        """
        INPUTS:
        =========================
        z : (b, d)
            latent embedding
        y : (1, d)
            context embedding
        t : (b, 1)
            time embedding

        OUTPUTS:
        =========================
        loss : (b, )
            flow matching loss
        """ 
        if t is None:
            t = torch.rand_like(z[..., 0:1])  # (b, 1) 
        p, u = self.path_and_field(t, z)  # (b, d) and (b, d) 
        
        vhat = self.dynamics(t=t, x=p, y=y, y_cond_mask=y_cond_mask)  # (b, d)
        return self.loss_funct(u, vhat).mean(-1) # (b, )

    def _divide_loss_by_time(self, loss: Tensor, denom: Tensor) -> Tensor:
        if self.loss_type in _MSE_LOSSES:
            denom = denom.square()
        elif self.loss_type not in _SUPPORTED_POINTWISE_LOSSES:
            raise ValueError(_unsupported_loss_message(self.loss_type))
        return loss / denom.to(dtype=loss.dtype, device=loss.device)

    def _training_loss_at_t(self, z: Tensor, t: Tensor, y: Tensor = None, y_cond_mask: Tensor = None) -> Tensor:
        loss = self(z, t=t, y=y, y_cond_mask=y_cond_mask)
        if not self.training:
            return loss
        denom = t.reshape(-1).clamp_min(max(float(self.t_norm_scale_cutoff), 1e-12))
        return self._divide_loss_by_time(loss, denom)

    def _bucket_loss_at_t(self, z: Tensor, t: Tensor, y: Tensor = None, y_cond_mask: Tensor = None) -> Tensor:
        return self(z, t=t, y=y, y_cond_mask=y_cond_mask)

    @torch.no_grad()
    def _conditioning_diagnostics(
        self,
        z: Tensor,
        t: Tensor,
        y: Tensor = None,
        max_samples: int = 64,
    ) -> dict:
        # @torch.no_grad() stops autograd from building graphs for the probe
        # forwards (the model params still require grad even though the inputs
        # are detached); self.dynamics.eval() below additionally disables
        # dropout/droppath. Both are needed — eval() alone still tracks grad.
        if max_samples <= 0 or z.shape[0] < 2:
            return {}

        n = min(int(max_samples), z.shape[0])
        z_diag = z[:n].detach()
        t_diag = t[:n].detach()
        y_true = y[:n].detach() if y is not None else None

        perm = torch.roll(torch.arange(n, device=z_diag.device), shifts=1)
        t_shuffled = t_diag[perm]
        t_frozen = torch.full_like(t_diag, 0.5)

        p, u = self.path_and_field(t_diag, z_diag)
        x_in = p
        
        dynamics_training = self.dynamics.training
        self.dynamics.eval()
        try:
            v_true = self.dynamics(t=t_diag, x=x_in, y=y_true)
            v_norm = v_true.norm(dim=-1).mean().clamp_min(1e-12)
            loss_true_mean = self.loss_funct(u, v_true).mean(-1).mean().clamp_min(1e-12)

            def velocity_delta(v_probe: Tensor) -> Tensor:
                return (v_true - v_probe).norm(dim=-1).mean()

            def loss_mean(v_probe: Tensor) -> Tensor:
                return self.loss_funct(u, v_probe).mean(-1).mean()

            def velocity_frac(delta: Tensor) -> Tensor:
                return (delta / v_norm).detach()

            def loss_delta_frac(probe_loss: Tensor) -> Tensor:
                return ((probe_loss - loss_true_mean) / loss_true_mean).detach()

            v_t_shuffled = self.dynamics(t=t_shuffled, x=x_in, y=y_true)
            v_t_frozen = self.dynamics(t=t_frozen, x=x_in, y=y_true)

            dt_shuffled = velocity_delta(v_t_shuffled)
            dt_frozen = velocity_delta(v_t_frozen)
            loss_t_shuffled_mean = loss_mean(v_t_shuffled)
            loss_t_frozen_mean = loss_mean(v_t_frozen)

            diagnostics = {
                "lfm_t_velocity_delta": dt_shuffled.detach(),
                "lfm_t_velocity_delta_frac": velocity_frac(dt_shuffled),
                "lfm_t_frozen_velocity_delta_frac": velocity_frac(dt_frozen),
                "lfm_t_true_loss_probe": loss_true_mean.detach(),
                "lfm_t_shuffled_loss_probe": loss_t_shuffled_mean.detach(),
                "lfm_t_shuffled_loss_delta_frac": loss_delta_frac(loss_t_shuffled_mean),
                "lfm_t_frozen_loss_delta_frac": loss_delta_frac(loss_t_frozen_mean),
            }

            if y_true is not None:
                y_shuffled = y_true[perm]
                v_y_shuffled = self.dynamics(t=t_diag, x=x_in, y=y_shuffled)
                v_uncond = self.dynamics(t=t_diag, x=x_in, y=None)

                dy = velocity_delta(v_y_shuffled)
                duncond = velocity_delta(v_uncond)
                loss_y_shuffled_mean = loss_mean(v_y_shuffled)
                loss_uncond_mean = loss_mean(v_uncond)
                diagnostics.update({
                    "lfm_y_velocity_delta": dy.detach(),
                    "lfm_y_velocity_delta_frac": velocity_frac(dy),
                    "lfm_y_uncond_velocity_delta_frac": velocity_frac(duncond),
                    "lfm_y_true_loss_probe": loss_true_mean.detach(),
                    "lfm_y_shuffled_loss_probe": loss_y_shuffled_mean.detach(),
                    "lfm_y_shuffled_loss_delta_frac": loss_delta_frac(loss_y_shuffled_mean),
                    "lfm_y_uncond_loss_delta_frac": loss_delta_frac(loss_uncond_mean),
                })
            return diagnostics
        finally:
            self.dynamics.train(dynamics_training)

    def compute_loss(
        self, 
        batched_data:dict[(str, Tensor)],
        compute_logp:bool=False,
        compute_conditioning_diagnostics: bool = False,
        conditioning_diagnostics_n: int = 64,
    ) -> tuple[Tensor, dict]:
        if not self.precomputed_latent: 
            z_mu, z_sigma, _ = self.latent_model.encoder(
                x=batched_data['x'],  
                position=batched_data['pos'], 
                num_atoms=batched_data.get('num_atoms', None),
                charge=batched_data.get('charge', None),
                spin_multiplicity=batched_data.get('spin_multiplicity', None),
                number_of_rings=batched_data.get('number_of_rings', None),
                has_fused_rings=batched_data.get('has_fused_rings', None),  
                y=batched_data.get('y', None),
            )  
            # normalize as done during the training of the auto-encoder
            z_mu = self.latent_model.latent_norm(z_mu)                           # (b, d)
        else:
            z_mu    = batched_data['z'] 
            z_sigma = batched_data['z_sigma']
        
        b       = z_mu.shape[0]
        z_mu    = z_mu.view(b, -1)
        z_sigma = z_sigma.view(b, -1)
        # Sample from normal   
        z = z_mu + z_sigma * torch.randn_like(z_sigma)
            
        loss_itemized = {}
        # print(f"after sampling from normal: isnan {str(torch.isnan(z).any())}")
        
        z = z.detach() # (bz, d)
        # Apply transform
        z, _ = self.pretransform_latent(z) 
        
        # print(f"after apply transform: isnan {str(torch.isnan(z).any())}")
        if self.conditioning > 0: 
            y = batched_data['y'].reshape(z.shape[0], -1)
            y = self.pretransform_context(y)  
        else:
            y = None
        y_cond_mask = self._cfg_condition_mask(y)
        
        if not self.training:
            # nt = 3
            loss_ld = torch.zeros((1,), device=z.device, dtype=z.dtype)
            if self.t_lower_eps < 0.01:
                # compute flow matching loss
                t = torch.rand_like(z[..., 0:1])*(0.01-self.t_lower_eps) + self.t_lower_eps  # (b,1) 
                # print("0.01, tmin={:.2f}, tmax={:.2f}".format(t.min().cpu().item(), t.max().cpu().item()))
                loss_ld_t = self._bucket_loss_at_t(z, t=t, y=y, y_cond_mask=y_cond_mask).mean()   # (,)
                # loss_ld += loss_ld_t
                loss_itemized['lfm_loss_0-0.01'] = loss_ld_t.detach()
                # nt += 1
            t = torch.rand_like(z[..., 0:1])*(0.2-0.01) + 0.01  # (b,1) 
            t = t.clamp(1e-8, 1-1e-8)
            # print("0.01-0.2, tmin={:.2f}, tmax={:.2f}".format(t.min().cpu().item(), t.max().cpu().item())) 
            loss_ld_t = self._bucket_loss_at_t(z, t=t, y=y, y_cond_mask=y_cond_mask).mean()       # (,)
            # loss_ld += loss_ld_t
            loss_itemized['lfm_loss_0.01-0.2'] = loss_ld_t.detach()
            
            t = torch.rand_like(z[..., 0:1])*(0.8-0.2) + 0.2  # (b,1)
            t = t.clamp(1e-8, 1-1e-8)
            # print("0.2-0.8, tmin={:.2f}, tmax={:.2f}".format(t.min().cpu().item(), t.max().cpu().item()))
            loss_ld_t = self._bucket_loss_at_t(z, t=t, y=y, y_cond_mask=y_cond_mask).mean()       # (,)
            # loss_ld += loss_ld_t
            loss_itemized['lfm_loss_0.2-0.8'] = loss_ld_t.detach()
            
            t = torch.rand_like(z[..., 0:1])*(0.99-0.8) + 0.8  # (b,1) 
            t = t.clamp(1e-8, 1-1e-8)
            # print("0.8-0.99, tmin={:.2f}, tmax={:.2f}".format(t.min().cpu().item(), t.max().cpu().item()))
            loss_ld_t = self._bucket_loss_at_t(z, t=t, y=y, y_cond_mask=y_cond_mask).mean()       # (,)
            # loss_ld += loss_ld_t
            loss_itemized['lfm_loss_0.8-0.99'] = loss_ld_t.detach()
            
            if self.t_upper_eps < 0.01:
                t = torch.rand_like(z[..., 0:1])*(1-self.t_upper_eps-0.99) + 0.99  # (b,1) 
                t = t.clamp(1e-8, 1-1e-8)
                # print("0.99-, tmin={:.2f}, tmax={:.2f}".format(t.min().cpu().item(), t.max().cpu().item()))
            
                loss_ld_t = self._bucket_loss_at_t(z, t=t, y=y, y_cond_mask=y_cond_mask).mean()       # (,)
                # loss_ld += loss_ld_t
                loss_itemized['lfm_loss_0.99-1'] = loss_ld_t.detach()
                # nt += 1

                del loss_ld_t
        
        t = torch.rand_like(z[..., 0:1])  # (b,1) 
        t = t.clamp(self.t_lower_eps, 1-self.t_upper_eps)
        # print("0.01-0.2, tmin={:.2f}, tmax={:.2f}".format(t.min().cpu().item(), t.max().cpu().item())) 
        loss_ld = self._training_loss_at_t(z, t=t, y=y, y_cond_mask=y_cond_mask)      # (b,)
        assert loss_ld.shape == (b,), f"loss_ld.shape should be (b,), but got {loss_ld.shape}"
            
        if compute_logp:
            logp_sum = self.log_prob(z, y=y)
            loss_itemized.update({'log_likelihood': logp_sum.detach().mean()})
        
        loss_itemized['lfm_loss'] = loss_ld.mean().detach()
        if compute_conditioning_diagnostics:
            loss_itemized.update(
                self._conditioning_diagnostics(
                    z=z,
                    t=t,
                    y=y,
                    max_samples=conditioning_diagnostics_n,
                )
            )

        del z_mu, z_sigma, z 
        torch.cuda.empty_cache()
        return loss_ld.mean(), loss_itemized
    
    def instantiate_first_stage(self, latent_model:CloudVAEModelTransformerP):
        if latent_model is not None:
            latent_model = latent_model.eval() 
            latent_model.train = latent_model.disabled_train
            for param in latent_model.parameters():
                param.requires_grad = False
            self.latent_model = latent_model # register latent_model as a submodule.
            print("\t NOT to train the latent model")
        else:
            print("\t No latent model to instantiate")
            self.latent_model = None

    @torch.no_grad()
    def get_reverse_fn(self, ode_method:str) -> Callable:
        if ode_method == "dopri5":
            return self.reverse_fn_dopri5
        elif ode_method == "euler":
            return self.reverse_fn_euler
        else:
            raise ValueError(f"Received ode_method={ode_method} is not allowed. Only 'dopri5' and 'euler'.")
    
    def _integrate_dopri5(self, z:Tensor, times:Tensor, y=None, guidance_scale=None):
        """
        Default: times=[1., 0.] 
        """
        def field(t:Tensor, x:Tensor):
            t = t.reshape(-1)
            return self._guided_dynamics(
                t=t,
                x=x,
                y=y,
                guidance_scale=guidance_scale, 
            )

        return odeint(
            field, 
            z.reshape(-1, self.dynamics.x_dim), 
            z.new_tensor(times),
            method='dopri5', 
            rtol=self.rtol, 
            atol=self.atol,
        )[-1]

    def _integrate_euler(self, z:Tensor, times:Tensor, y:Tensor=None, guidance_scale=None, steps=25):
        """
        Default: times=[1., 0.], steps=25
        """
        def field(t: Tensor, x: Tensor):
            t_ = t.view(-1, 1).expand(x.shape[0], 1)
            dx = self._guided_dynamics(
                t=t_,
                x=x,
                y=y,
                guidance_scale=guidance_scale
            )
            return dx
        return euler_integrate_reverse(
            field,
            z.reshape(-1, self.dynamics.x_dim),
            t_start=times[0],
            t_end=times[1],
            steps=steps,
        )


    @torch.no_grad()
    def reverse_fn_dopri5(
        self, 
        z:Tensor, 
        t_lower_eps:float,
        t_upper_eps:float,
        y:Tensor=None,
        guidance_scale:float=1.0,
        **kwargs) -> Tensor:
        """
        z: (bz, d)
        y: (bz, d_y)

        """
        self.before_odeint(z)
        return self._integrate_dopri5(
            z, [1-t_upper_eps, t_lower_eps], y=y, guidance_scale=guidance_scale,
        )
        
    
    @torch.no_grad()
    def reverse_fn_euler(
        self,
        z:Tensor,
        t_lower_eps:float,
        t_upper_eps:float,
        y:Tensor=None,
        steps:int=25,
        guidance_scale:float=1.0,
        **kwargs,
    ) -> Tensor: 
        self.before_odeint(z)
        return self._integrate_euler(
            z, [1-t_upper_eps, t_lower_eps], y=y, guidance_scale=guidance_scale, steps=steps,
        )

    @torch.no_grad()
    def sample_latents(
        self, 
        n_samples:int, 
        y:Tensor=None, 
        noise:Tensor=None,
        seed:int|None=None,
        t_lower_eps:float|None=None,
        t_upper_eps:float|None=None,  
        ode_steps:int|None=None,
        guidance_scale:float|None=None,
    ):
        """
        Draw samples from the generative model.
        INPUTS:
        =========================
        context : (d_y, ) or (n_y, d_y)
        noise   : (n_y*n_samples, d_x) 
        z_sigma : (1, d_x)
         
        OUTPUT:
        =========================
        x : (n_y, n_samples, d_x)
        """
        if t_lower_eps is None:
            t_lower_eps = self.t_lower_eps
        if t_upper_eps is None:
            t_upper_eps = self.t_upper_eps  
        if ode_steps is None:
            ode_steps = self.ode_steps
        if guidance_scale is None:
            guidance_scale = self.cfg_guidance_scale
        device = next(self.dynamics.parameters()).device
        dtype  = next(self.dynamics.parameters()).dtype
        if y is None:
            n_y = 1
            y_ = None 
        else:
            y_dim = y.shape[-1]
            n_y = y.shape[0]
            y  = y.unsqueeze(1).expand(n_y, n_samples, y_dim)
            y_ = y.reshape(-1, y_dim)
            # Apply transform
            y_ = self.pretransform_context(y_)
        
        if noise is not None:
            assert noise.dim() == 2 and noise.shape[0] == n_y*n_samples and noise.shape[1] == self.dynamics.x_dim, \
                f"Provided noise is of shape {noise.shape} does not match with the target output with n_y={n_y} and ns={n_samples}"
            noise = noise.to(device=device, dtype=dtype)
        else:
            if seed is not None:
                torch.manual_seed(seed)
            noise = torch.randn((n_y*n_samples, self.dynamics.x_dim), device=device, dtype=dtype)
        
        z0 = self.reverse_fn(
            noise,
            t_lower_eps=t_lower_eps,
            t_upper_eps=t_upper_eps,
            y=y_,
            steps=ode_steps,
            guidance_scale=guidance_scale,
        )
        # Inverse transform
        z0 = self.invtransform_latent(z0) # (n_samples, d) or (n_y*n_samples, d)

        if y is None: 
            # z0: (n_samples, d)
            return (z0, None)
        else:
            # z0: (n_y, n_samples, d)
            z0 = z0.reshape(n_y, n_samples, -1)
            del y_
            torch.cuda.empty_cache()
            return (z0, y)
    
    @torch.no_grad()
    def sample(
        self, 
        n_samples:int | None=None, 
        y:Tensor=None, 
        noise:Tensor=None,
        seed:int | None=None,
        t_lower_eps:float | None=None,
        t_upper_eps:float | None=None, 
        z0:Tensor=None, 
        ode_steps:int=100,
        guidance_scale:float | None=None,
        return_generated_y:bool=False,
        latent_only:bool=False,
    ):
        """
        Draw samples from the generative model.
        INPUTS:
        =========================
        context : (n_y, d_y)
        noise   : (n_y*n_samples, d_x)
        z_sigma : (1, d_x)
         
        OUTPUT:
        =========================
        x : (n_y, n_samples, d_x)

        When ``return_generated_y`` is true, append two property predictions to
        the legacy output tuple: ``generated_y_lat`` from the latent-only
        property path and ``generated_y_mol`` from the generated-molecule
        property path.  Sampling call sites that do not request these values
        retain the legacy output tuple.

        When ``latent_only`` is true, stop before autoregressive molecule
        decoding and return ``(latent, y, generated_y_lat)``.
        """
        if z0 is None: 
            assert n_samples is not None, "n_samples must be provided if z0 is not provided"
            if y is not None:
                assert y.dim() == 2, f"y has to be of dimension 2, received {y.shape}"
                n_y = y.shape[0]
            z0, _ = self.sample_latents(
                n_samples=n_samples, 
                y=y, 
                noise=noise, 
                seed=seed, 
                t_lower_eps=t_lower_eps,
                t_upper_eps=t_upper_eps, 
                ode_steps=ode_steps,
                guidance_scale=guidance_scale,
            )
            if y is None: 
                z0 = z0.unsqueeze(1) # (n_samples, 1, d)
            else:
                d = z0.shape[-1]
                z0 = z0.reshape(-1, 1, d) # (n_y*n_samples, 1, d)
        else:
            assert z0.dim() == 3 
            if y is not None:
                print("Warning: y is provided, but z0 is also provided. y will be ignored.")
        decoder = self.latent_model.decoder.model
        if latent_only:
            latent_embeds_raw = z0 if decoder.use_ada_ln_zero else None
            latent_embeds = (
                decoder.latent_encoder(z0)
                if decoder.latent_encoder is not None
                else z0
            )
            generated_y_lat = decoder._predict_latent_only_property(
                latent_embeds,
                latent_embeds_raw,
            )
            if y is None:
                return z0, None, generated_y_lat
            return (
                z0.reshape(n_y, n_samples, -1),
                y,
                generated_y_lat.reshape(n_y, n_samples, -1),
            )

        decoder_outputs = decoder.generate_cache(z0)
        if len(decoder_outputs) < 6:
            raise ValueError(
                f"Expected decoder.generate_cache to return at least 6 values, "
                f"received {len(decoder_outputs)}."
            )
        (
            generated_pos, 
            generated_seq, 
            generated_charge,
            generated_spinmp,
            generated_number_of_rings,
            generated_has_fused_rings,
            *_,
        ) = decoder_outputs
        generated_y_lat = decoder_outputs[-1]
        generated_y_mol = None
        molecule_property_predictor = getattr(
            decoder, "_predict_molecule_property", None
        )
        if (
            return_generated_y
            and generated_y_lat is not None
            and molecule_property_predictor is not None
        ):
            atom_padding_mask = generated_seq[..., 0].eq(decoder.pad_token_id)
            generated_y_mol = molecule_property_predictor(
                latent_embeds=z0,
                position=generated_pos,
                x=generated_seq,
                padding_mask=atom_padding_mask,
            )
        if y is None: 
            outputs = (
                generated_pos, 
                generated_seq, 
                generated_charge,
                generated_spinmp,
                generated_number_of_rings,
                generated_has_fused_rings, 
                z0, 
                None
            )
            return (
                (*outputs, generated_y_lat, generated_y_mol)
                if return_generated_y
                else outputs
            )
        else:
            generated_pos = generated_pos.reshape(n_y, n_samples, -1, 3)
            generated_seq = generated_seq.reshape(n_y, n_samples, -1, self.latent_model.num_atom_features)
            if generated_charge is not None:
                generated_charge = generated_charge.reshape(n_y, n_samples)
            if generated_spinmp is not None:
                generated_spinmp = generated_spinmp.reshape(n_y, n_samples)
            if generated_number_of_rings is not None:
                generated_number_of_rings = generated_number_of_rings.reshape(n_y, n_samples)
            if generated_has_fused_rings is not None:
                generated_has_fused_rings = generated_has_fused_rings.reshape(n_y, n_samples)
            if generated_y_lat is not None:
                generated_y_lat = generated_y_lat.reshape(n_y, n_samples, -1)
            if generated_y_mol is not None:
                generated_y_mol = generated_y_mol.reshape(n_y, n_samples, -1)
            z0 = z0.reshape(n_y, n_samples, -1)
            outputs = (
                generated_pos, 
                generated_seq, 
                generated_charge,
                generated_spinmp,
                generated_number_of_rings,
                generated_has_fused_rings, 
                z0, 
                y
            )
            return (
                (*outputs, generated_y_lat, generated_y_mol)
                if return_generated_y
                else outputs
            )
    
    @torch.no_grad()
    def log_prob(self, x: Tensor, y:Tensor=None) -> Tensor:
        """Compute log likelihood: Forward pass (x -> ODE -> z)
        y: context embedding
        """
        ldj = x.new_zeros(x.shape[0])
        reg_term = x.new_zeros(x.shape[0])
        
        # 1) Pretransform
        x, ldj_act = self.pretransform_latent(x)
        
        # 2) ODE
        state = (x, ldj, reg_term)
         
        self.before_odeint(x)

        def ode_forward_w_context(t, state):
            return self.ode_forward(t.reshape(-1), state, y=y)

        statet = odeint(
            ode_forward_w_context, 
            state, 
            self.int_time,
            method=self.ode_method,
            rtol=self.rtol,
            atol=self.atol
        )
        
        zt, ldjt, reg_termt = statet
        z, ldj, reg_term = zt[-1], ldjt[-1], reg_termt[-1]
        
        return self.prior_likelihood(z) + (ldj_act + ldj*1e2) + reg_term
    
    def configure_AdamW(self, weight_decay=0.01, learning_rate=1e-4, betas=(0.9, 0.999), eps=1e-8):
        import inspect

        def _is_conditioning_encoder_param(n: str) -> bool:
            return ('t_encoder.' in n) or ('y_encoder.' in n) or ('y_fourier.' in n)

        def _is_adaln_modulation_param(n: str) -> bool:
            return ('.adaLN_modulation_t.' in n) or ('.adaLN_modulation_y.' in n)

        def _is_conditioning_param(n: str) -> bool:
            return _is_conditioning_encoder_param(n) or _is_adaln_modulation_param(n)

        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        
        if (not self.dynamics.standparam):
            emb_weight_params = []
            emb_bias_params = []
            static_ln_params = []
            hidden_ln_params = []
            hidden_weight_params = []
            hidden_bias_params = []
            adaln_mod_weight_params = []
            adaln_mod_bias_params = []
            # Conditioning paths can receive weak gradients (OT-FM is only mildly
            # time-dependent, and low-dim y carries little signal). Under weight
            # decay they collapse toward zero, leaving the velocity field
            # effectively time-/label-blind. Keep them decay-free. AdaLN
            # modulation MLPs still use hidden-layer muP lr scaling below.
            cond_nodecay_params = []
            for n, p in param_dict.items():
                if _is_conditioning_encoder_param(n):
                    cond_nodecay_params.append(p)
                elif _is_adaln_modulation_param(n):
                    if n.endswith('weight'):
                        adaln_mod_weight_params.append(p)
                    elif n.endswith('bias'):
                        adaln_mod_bias_params.append(p)
                elif ('_encoder.' in n) or ('.linear1.' in n) or ('.linear2.' in n) or ('.readout.' in n):
                    if n.endswith('weight'):
                        emb_weight_params.append(p)
                    elif n.endswith('bias'):
                        emb_bias_params.append(p) 
                elif ('.ln_' in n) and ('.ln_f.' not in n): # in the stack
                    hidden_ln_params.append(p)
                elif n.endswith(('c_attn.weight', 'c_fc.weight', 'c_proj.weight')):
                    hidden_weight_params.append(p)
                elif n.endswith(('c_attn.bias', 'c_fc.bias', 'c_proj.bias')):
                    hidden_bias_params.append(p)
                elif ('.ln_' in n) and ('.ln_f.' not in n): # in the stack
                    hidden_ln_params.append(p)
                elif n.endswith(('c_attn.weight', 'c_fc.weight', 'c_proj.weight')):
                    hidden_weight_params.append(p)
                elif n.endswith(('c_attn.bias', 'c_fc.bias', 'c_proj.bias')):
                    hidden_bias_params.append(p)
                elif (".shortcut." in n):
                    if n.endswith('weight'):
                        hidden_weight_params.append(p)
                    elif n.endswith('bias'):
                        hidden_bias_params.append(p)
                elif ('.ln_f.' in n):
                    static_ln_params.append(p)
                else:
                    raise ValueError(f'Unhandled parameter {n}')
            width_lr_scaling = (1 / self.dynamics.width_multiplier)
             
            ### Begin CompleteP code ###
            # for completeP, alpha=1, so effectively 
            depth_lr_scaling = (self.dynamics.depth_multiplier ** (self.dynamics.depth_alpha_exp - 1)) # when alpha=1, depth_lr_scaling == 1
            eps *= (1 / self.dynamics.width_multiplier) * (self.dynamics.depth_multiplier ** (-1 * self.dynamics.depth_alpha_exp))
            optim_groups = [
                {
                    'params': emb_weight_params,
                    'weight_decay': weight_decay,
                    'lr_scale': 1.0,
                },
                {
                    'params': emb_bias_params,
                    'weight_decay': 0.0,
                    'lr_scale': 1.0,
                },
                {
                    'params': cond_nodecay_params,
                    'weight_decay': 0.0,
                    'lr_scale': 1.0,
                },
                {
                    'params': adaln_mod_weight_params,
                    'weight_decay': 0.0,
                    'lr_scale': width_lr_scaling * depth_lr_scaling,
                },
                {
                    'params': adaln_mod_bias_params,
                    'weight_decay': 0.0,
                    'lr_scale': depth_lr_scaling,
                },
                {
                    'params': static_ln_params,
                    'weight_decay': 0.0,
                    'lr_scale': 1.0,
                },
                {
                    'params': hidden_ln_params,
                    'weight_decay': 0.0,
                    'lr_scale': depth_lr_scaling,
                },
                {
                    'params': hidden_weight_params,
                    'weight_decay': weight_decay / width_lr_scaling,
                    'lr_scale': width_lr_scaling * depth_lr_scaling,
                },
                {
                    'params': hidden_bias_params,
                    'weight_decay': 0.0,
                    'lr_scale': depth_lr_scaling,
                }, 
            ]
            ### End CompleteP code ###
        else:
            # Keep the conditioning paths decay-free (see note in the muP branch).
            decay_params = [p for n, p in param_dict.items() if p.dim() >= 2 and not _is_conditioning_param(n)]
            nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2 or _is_conditioning_param(n)]
            optim_groups = [
                {'params': decay_params, 'weight_decay': weight_decay},
                {'params': nodecay_params, 'weight_decay': 0.0}
            ]
            num_decay_params = sum(p.numel() for p in decay_params)
            num_nodecay_params = sum(p.numel() for p in nodecay_params)
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and torch.cuda.is_available()
        extra_args = {'fused': True} if use_fused else {}
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, eps=eps, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

