import torch 
from torch import nn, Tensor
import math 

from typing import Literal
 
class SinusoidalPE(nn.Module):
    def __init__(self, dim: int, min_freq: float = 1.0, max_freq: float = 10_000.0):
        super().__init__()
        assert dim % 2 == 0
        half = dim // 2
        self.dim = dim
        # log-spaced frequencies in [min_freq, max_freq]
        freqs = torch.exp(torch.linspace(math.log(min_freq), math.log(max_freq), half))
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, t: Tensor) -> Tensor:  # (...,)
        """
        INPUTS:
        ===============
        t: (b, 1) or (b,) or (,)
            values in (0,1)
        RETURNS:
        ===============
        emb: (b, 1, d_t) if t is of shape (b, 1)
             (b, d_t) if t is of shape (b,)
             (d_t,) if t is of shape ()
        """
        orig_shape = t.shape      
        u = t.reshape(-1, 1)                                      # (N, 1)
        arg = u * self.freqs[None, :].to(t.device) * (2*math.pi)  # (N, d_t//2)
        emb = torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1)
        return emb.view(*orig_shape, self.dim)                    # (..., d_t)

def logsnr_from_t(t: Tensor, sigma: float, lower_eps:float=1e-12, upper_eps: float=1e-12) -> Tensor:
    """
    input and output are both in (0,1), and same shape
    """
    t = t.clamp(lower_eps, 1-upper_eps)
    return 2*(torch.log1p(-t) - torch.log(t)) - 2*math.log(sigma)

def log_gap_from_t(t: Tensor, lower_eps:float=1e-12, upper_eps: float=1e-12) -> Tensor:
    # \ell(t) = -log(1 - t) = log(1 / (1 - t))
    t = t.clamp(lower_eps, 1 - upper_eps)
    return -torch.log1p(-t)

class GaussianFourierEmbedding(nn.Module):
    """
    Random Fourier features for a scalar input u.
    """
    def __init__(self, dim: int, scale: float = 16.0):
        super().__init__()
        assert dim % 2 == 0, "dim must be even"
        half = dim // 2
        # B ~ N(0, scale^2)
        B = torch.randn(half) * scale
        self.register_buffer("B", B, persistent=False)
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        """
        INPUTS:
        ===============
        t: (b, 1) or (b,) or (,)
            values in (0,1)
        RETURNS:
        ===============
        emb: (b, 1, dim) if t is of shape (b, 1)
             (b, dim) if t is of shape (b,)
             (dim,) if t is of shape ()
        """
        orig_shape = t.shape 
        u = t.reshape(-1, 1)             
        arg = u * self.B[None, :].to(t.device) * 2.0 * math.pi
        emb = torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1)
        return emb.view(*orig_shape, self.dim)

class GaussianFourierFeatures(nn.Module):
    """
    Random Fourier features for a (possibly multi-dimensional) vector input y.

    Maps y of shape (..., in_dim) -> (..., 2 * num_freqs) via a *fixed* random
    projection B ~ N(0, scale^2):

        phi(y) = [sin(2*pi * y @ B), cos(2*pi * y @ B)]

    The projection is registered as a persistent buffer so the exact same random
    features are used at train and inference time (they are saved in the
    state dict). This gives a low-dimensional conditioning signal (e.g. a scalar
    property) a rich, full-rank embedding analogous to the time embedding,
    instead of a single small `Linear(y_dim, d_model)`.
    """
    def __init__(self, in_dim: int, num_freqs: int, scale: float = 1.0):
        super().__init__()
        assert in_dim > 0, "in_dim must be positive"
        assert num_freqs > 0, "num_freqs must be positive"
        # B: (in_dim, num_freqs), fixed random projection (saved with the model)
        B = torch.randn(in_dim, num_freqs) * scale
        self.register_buffer("B", B, persistent=True)
        self.in_dim = in_dim
        self.out_dim = 2 * num_freqs

    def forward(self, y: Tensor) -> Tensor:
        """
        INPUTS:
        ===============
        y: (..., in_dim)
        RETURNS:
        ===============
        emb: (..., 2 * num_freqs)
        """
        arg = (y @ self.B.to(dtype=y.dtype, device=y.device)) * 2.0 * math.pi   # (..., num_freqs)
        return torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1)              # (..., 2*num_freqs)

class TimeEmbedding(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 sigma: float, 
                 gamma : float=0.5,
                 pe_dim: int = 128,
                 method: Literal["logsnr_sinusoidal", "sinusoidal", "fourier"] = "logsnr_sinusoidal",
                 min_freq: float = 1.0,
                 max_freq: float = 10_000.0,
                 fourier_scale: float = 16.0,
                 rescale_logsnr: float = 0.125,
                 add_log_gap: bool = False,
                 lower_eps: float = 1e-12,
                 upper_eps: float = 1e-12,
                 apply_init_weights: bool = False,
                 init_std: float = 0.02,
                 final_norm: bool = False):
        """
        rescale_logsnr: 
            if you want to keep arguments moderate, 
            set rescale_logsnr=1/8 or 1/10 so the largest |u| is ~< 2 before
            multiplying by frequencies. 
            (Empirically stabilizes very high-freq stacks.)
        """
        super().__init__()
        self.sigma = float(sigma)
        self.method = method
        self.add_log_gap = add_log_gap
        self.lower_eps = lower_eps
        self.upper_eps = upper_eps
        self.rescale_logsnr = rescale_logsnr
        self.gamma = gamma
        self.pe_dim = pe_dim

        if method in ["logsnr_sinusoidal", "sinusoidal"]:
            self.base = SinusoidalPE(pe_dim, min_freq=min_freq, max_freq=max_freq) 
        elif method == "fourier":
            self.base = GaussianFourierEmbedding(pe_dim, scale=fourier_scale)
        else:
            raise ValueError(f"TimeEmbedding only accepts 'logsnr_sinusoidal', 't_sinusoidal', 't_fourier'. Received {method}")
        extra = 2 if add_log_gap else 0       # ell and ell_gamma
        self.proj = nn.Sequential(
            nn.Linear(pe_dim + extra, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        # Optional final LayerNorm: keeps the time embedding at ~unit scale
        # regardless of the proj weight magnitude, so weight decay cannot
        # shrink the time signal into irrelevance.
        self.final_norm = nn.LayerNorm(d_model) if final_norm else None

        if apply_init_weights:
            self._reset_parameters(init_std)

    def _reset_parameters(self, init_std:float=0.02):
        for module in self.proj:
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=init_std)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0.0)

    def forward(self, t: Tensor) -> Tensor:
        """
        INPUTS:
        ===============
        t: (b, 1) 
            values in (0,1)

        RETURNS:
        ===============
        t_embeds: (b, 1, d)
        """
        # assert t.dim() == 2, f"t.shape={t.shape}"
        if self.method == "logsnr_sinusoidal":
            u = logsnr_from_t(t, sigma=self.sigma, lower_eps=self.lower_eps, upper_eps=self.upper_eps)  # (b, 1)
            if self.rescale_logsnr is not None:
                u = u * self.rescale_logsnr
        else:
            u = t.clamp(self.lower_eps, 1-self.upper_eps)
        # assert u.dim() == 2, f"t.shape={u.shape}"
        emb = self.base(u)                          # (b, 1, pe_dim) 
        # assert emb.dim() == 3, f"after base embedding, expected of shape (b,1,pe_dim)={t.shape[0],1,self.pe_dim}, but got emb.shape={emb.shape}"
        if self.add_log_gap:
            ell = log_gap_from_t(t, lower_eps=self.lower_eps, upper_eps=self.upper_eps)                           # (b, 1)
            # assert ell.dim() == 2, f"ell.shape={ell.shape}"
            ell_gamma = (1.0 - self.gamma) * ell                            # (b, 1)
            # assert ell_gamma.dim() == 2, f"ell_gamma.shape={ell_gamma.shape}"
            extras = torch.cat([ell, ell_gamma], dim=-1).unsqueeze(-2)      # (b, 1, 2)
            emb = torch.cat([emb, extras], dim=-1)                          # (b, 1, pe_dim+2)
        emb = self.proj(emb)  # (B, 1, d_model)
        if self.final_norm is not None:
            emb = self.final_norm(emb)
        return emb
