import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor 
from torch.nn.init import zeros_, normal_
import math

from typing import Callable

def fixed_dimension_init_weights(module, init_std:float=0.02):
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=init_std)
        if module.bias is not None:
            torch.nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=init_std)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()

def init_weights(
    module,
    init_std:float=0.02, 
    standparam:bool=False, 
    width_multiplier:float=1.0):
    if isinstance(module, nn.Linear):
        if standparam:
            normal_(module.weight, mean=0.0, std=init_std)
        else:
            """
                This is the code for the MuP initialization.
                See NanoGPT-mup implementation: 
                https://github.com/EleutherAI/nanoGPT-mup/blob/completep/model.py
                for more details.

                The implementation here is equivalent to the code from Microsoft's mup implementation:
                https://github.com/microsoft/mup/blob/main/examples/Transformer/model.py
                where it initializes the weights with fan-in normal ~ N(0, sqrt(var/d_model))
                
                From NanoGPT-mup implementation:
                    std = init_std / sqrt(d_model / d_base)
                        = (init_std * sqrt(d_base)) / sqrt(d_model)

                From Microsoft's mup implementation:
                    std = sqrt(var / d_model)
                        = sqrt(var) / sqrt(d_model)

                if we set var = init_std**2 * d_base, then the two implementations are equivalent.

                To keep it simple, we use the NanoGPT-mup implementation.
            """
            normal_(module.weight, mean=0.0, std=init_std / math.sqrt(width_multiplier))
            
        if module.bias is not None:
            zeros_(module.bias)

def get_activation_fn(activation: str) -> Callable:
    """Returns the activation function corresponding to `activation`"""
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu 
    elif activation == "silu":
        return F.silu 
    elif activation == "tanh":
        return torch.tanh
    elif activation == "linear":
        return lambda x: x
    elif activation == "approx_gelu":
        return nn.GELU(approximate="tanh")  
    else:
        raise RuntimeError(f"--activation-fn {activation} not supported")

BIG_CONST = 1e10 

def top_k_filter(logits, k, probs=False):
    """
    Masks everything but the k top entries as -infinity (1e10).
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.
    """
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[..., -1].view(*logits.shape[:-1], 1).expand_as(logits)
        if probs:
            return torch.where(logits < batch_mins, torch.ones_like(logits) * 0.0, logits)
        return torch.where(logits < batch_mins, torch.ones_like(logits) * -BIG_CONST, logits)

def sample_from_logits(logits:Tensor, temperature:float=1., top_k:int=50):
    # Apply temperature scaling
    logits = logits / temperature
    # Apply top-k filtering
    filtered_logits = top_k_filter(logits, top_k)
    # Convert logits to probabilities
    probabilities = F.softmax(filtered_logits, dim=-1)
    # Sample from the probability distribution
    sampled_indices = torch.multinomial(probabilities.reshape(-1,probabilities.shape[-1]).contiguous(), num_samples=1)
    sampled_indices = sampled_indices.reshape(logits.shape[:-1])
    return sampled_indices
