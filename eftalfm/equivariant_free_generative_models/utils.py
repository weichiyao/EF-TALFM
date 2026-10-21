from __future__ import annotations
import torch  
from torch import Tensor 
import torch.nn.functional as F
import numpy as np

 
def match_size(x:Tensor, r:Tensor):
    """
    x: (bz,...,d) or (d,)
    r: reference (bz,...,u)

    RETURN
    (bz,...,d)
    """
    x = x.expand(*r.shape[:-1], -1)            
    return x

def expand_to_match_size(x: Tensor, r: Tensor) -> Tensor:
    """
    Expand 'x' so that it has the same shape as 'r' in the first (bz, k1, ..., kn) dimensions,
    and has last dimension d (taken from a).

    r has shape (bz, k1, ..., kn, m).
    x has shape (bz, d) or (d,) 
       note: x of shape (bz,) would give errors

    Returns a new tensor of shape (bz, k1, ..., kn, d).
    """
    # if they are already in the same shape, then assume everything is right
    if x.ndim == r.ndim:
        return x 
    
    # Extract r's shape: (bz, k1, ..., kn, m)
    *start_dims, _ = r.shape  # start_dims = (bz, k1, ..., kn)
    
    # If x: (d,), r: (bz, k1, ..., kn, m) -> x: (bz, 1_1, ..., 1_n, d) 
    if x.ndim == 1:
        x = x.expand(*start_dims, -1)            
        return x

    # Now we expect a to be (bz, d) at this point
    bz = start_dims[0]
    if x.shape[0] != bz:
        raise ValueError(f"Batch dimension of 'x' ({x.shape[0]}) does not match first dim of 'r' ({bz}).")
    
    # Count how many extra dims b has after the batch dimension
    n_extra_dims = len(start_dims) - 1  # (k1, ..., kn)

    # Reshape 'a' to (bz, 1, 1, ..., 1, d) with n_extra_dims of singleton dimensions
    d = x.shape[-1]
    reshape_shape = (bz,) + (1,) * n_extra_dims + (d,)
    x_reshaped = x.view(*reshape_shape)

    # Now expand to match the shape of b in those middle dims
    # final shape -> (bz, k1, ..., kn, d)
    expanded_x = x_reshaped.expand(*start_dims, d)

    return expanded_x

###############################################################################################################

# Defining some useful util functions.
def expm1(x: Tensor) -> Tensor:
    return torch.expm1(x)


def softplus(x: Tensor) -> Tensor:
    return F.softplus(x)


def norm_except_batch(x:Tensor):
    """
    x: (batch_size, n_nodes, d)
    """
    return torch.linalg.vector_norm(x.view(x.size(0), -1), ord=2, dim=-1) 

def clip_noise_schedule(alphas2, clip_value=0.001):
    """
    For a noise schedule given by alpha^2, this clips alpha_t / alpha_t-1. This may help improve stability during
    sampling.
    """
    alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)

    alphas_step = (alphas2[1:] / alphas2[:-1])

    alphas_step = np.clip(alphas_step, a_min=clip_value, a_max=1.)
    alphas2 = np.cumprod(alphas_step, axis=0)

    return alphas2

def sum_except_batch(x:Tensor):
    return x.reshape(x.size(0), -1).sum(dim=-1)

def gaussian_KL(q_mu:Tensor, q_sigma:Tensor, p_mu:Tensor, p_sigma:Tensor, aggregation="sum"):
    """Computes the KL distance between two normal distributions.

    Args:
        q_mu: Mean of distribution q.
        q_sigma: Standard deviation of distribution q.
        p_mu: Mean of distribution p.
        p_sigma: Standard deviation of distribution p.
    Returns:
        The KL distance, summed over all dimensions except the batch dim.
    """
    out = sum_except_batch(
        torch.log(p_sigma / q_sigma)
        + 0.5 * (q_sigma**2 + (q_mu - p_mu)**2) / (p_sigma**2)
        - 0.5
    )
    if aggregation == "mean":
        d = q_mu.shape[-1]
        out /= d

    return out

##############################################################################################################
############ ================================ Latent transform ========================================= #####
##############################################################################################################

def _to_2d_torch(z: Tensor) -> Tensor:
    """
    Accepts shape (N, D) or (N, 1, D), returns (N, D).
    """
    if z.ndim == 3 and z.shape[1] == 1:
        z = z[:, 0, :]
    if z.ndim != 2:
        raise ValueError(f"Expected shape (N, D) or (N, 1, D), got {tuple(z.shape)}")
    return z


@torch.no_grad()
def fit_pca_whitener_torch(
    train_z: Tensor,
    eps: float = 1e-6,
    use_float64: bool = True,
):
    """
    Fit full-covariance PCA whitener on train_z.

    train_z: (N, D) or (N, 1, D)
    returns a dict of torch tensors on the same device as train_z
    """
    z = _to_2d_torch(train_z)
    orig_dtype = z.dtype
    work_dtype = torch.float64 if use_float64 else torch.float32
    z = z.to(work_dtype)
    mean = z.mean(dim=0)                         # (D,)
    zc = z - mean[None, :]                       # (N, D)
    # Sample covariance
    cov = (zc.T @ zc) / max(z.shape[0] - 1, 1)  # (D, D)
    cov = 0.5 * (cov + cov.T)                   # symmetrize for stability
    # eigvals ascending, eigvecs columns
    evals, evecs = torch.linalg.eigh(cov)
    # sort descending
    idx = torch.argsort(evals, descending=True)
    evals = evals[idx]
    evecs = evecs[:, idx]
    evals_clamped = torch.clamp(evals, min=eps)
    inv_sqrt = evals_clamped.rsqrt()
    sqrt = evals_clamped.sqrt()
    return {
        "mean": mean.to(orig_dtype),
        "evecs": evecs.to(orig_dtype),
        "evals": evals.to(orig_dtype),
        "inv_sqrt": inv_sqrt.to(orig_dtype),
        "sqrt": sqrt.to(orig_dtype),
        "eps": eps,
    }


def whiten_transform_torch(
    z: Tensor,
    whitener: dict,
) -> Tensor:
    """
    z: (N, D) or (N, 1, D)
    returns whitened z with same shape as input except squeezed to (N, D)
    """
    z = _to_2d_torch(z)
    mean = whitener["mean"].to(device=z.device, dtype=z.dtype)
    evecs = whitener["evecs"].to(device=z.device, dtype=z.dtype)
    inv_sqrt = whitener["inv_sqrt"].to(device=z.device, dtype=z.dtype)
    zc = z - mean[None, :]
    z_pca = zc @ evecs
    z_white = z_pca * inv_sqrt[None, :]
    return z_white


def whiten_inverse_torch(
    z_white: Tensor,
    whitener: dict,
) -> Tensor:
    """
    z_white: (N, D)
    returns raw latent coordinates (N, D)
    """
    z_white = _to_2d_torch(z_white)
    mean = whitener["mean"].to(device=z_white.device, dtype=z_white.dtype)
    evecs = whitener["evecs"].to(device=z_white.device, dtype=z_white.dtype)
    sqrt = whitener["sqrt"].to(device=z_white.device, dtype=z_white.dtype)
    z_pca = z_white * sqrt[None, :]
    z_raw = z_pca @ evecs.T
    z_raw = z_raw + mean[None, :]
    return z_raw



def pca_precondition_transform_torch(
    z: Tensor,
    precond: dict[str, Tensor],
) -> Tensor:
    """
    z: (N, D) or (N, 1, D)
    returns: (N, k)
    """
    z = _to_2d_torch(z)
    mean = precond["mean"].to(device=z.device, dtype=z.dtype)             # (D,)
    evecs_k = precond["evecs_k"].to(device=z.device, dtype=z.dtype)       # (D, k)
    scale_fwd = precond["scale_fwd"].to(device=z.device, dtype=z.dtype)   # (k,)

    zc = z - mean[None, :]
    z_pca = zc @ evecs_k
    z_pre = z_pca * scale_fwd[None, :]
    return z_pre


def pca_precondition_inverse_torch(
    z_pre: Tensor,
    precond: dict[str, Tensor],
) -> Tensor:
    """
    z_pre: (N, k)
    returns: (N, D)
    """
    z_pre = _to_2d_torch(z_pre)
    mean = precond["mean"].to(device=z_pre.device, dtype=z_pre.dtype)           # (D,)
    evecs_k = precond["evecs_k"].to(device=z_pre.device, dtype=z_pre.dtype)     # (D, k)
    scale_inv = precond["scale_inv"].to(device=z_pre.device, dtype=z_pre.dtype) # (k,)

    z_pca = z_pre * scale_inv[None, :]
    z_raw = z_pca @ evecs_k.T
    z_raw = z_raw + mean[None, :]
    return z_raw

