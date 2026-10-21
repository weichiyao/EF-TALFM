from __future__ import annotations
import math

import torch
from torch import Tensor 
from torch import nn
import torch.nn.functional as F
from torch.nn import Module 
from torch.nn.init import constant_

from functools import partial


from eftalfm.equivariant_free_generative_models.models.utils import get_activation_fn, init_weights
from eftalfm.equivariant_free_generative_models.models.transformer_p.dropout import DropPath, ClassicDropout



def _check_key_padding_mask(
    key_padding_mask: torch.Tensor, src_len: int, bsz: int
) -> None:
    torch._check_with(
        AssertionError,
        key_padding_mask.shape[0] == bsz,
        lambda: f"Expected key_padded_mask.shape[0] to be {bsz}, but got {key_padding_mask.shape[0]}",
    )
    torch._check_with(
        AssertionError,
        key_padding_mask.shape[1] == src_len,
        lambda: f"Expected key_padded_mask.shape[1] to be {src_len}, but got {key_padding_mask.shape[1]}",
    )


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias:bool=True, eps:float=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.eps = eps

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)

class MuPMLP(nn.Module):
    def __init__(self, 
                 d_model:int,
                 d_out:int | None=None, 
                 dim_feedforward:int | None = None, 
                 bias_front:bool=True, 
                 bias_last:bool=True, 
                 dropout:float=0.1, 
                 droppath_prob:float=0.1, 
                 activation_dropout:float=0.1,
                 activation:str="gelu",
                 norm_layer:nn.Module=None,
                 standparam:bool=False,
                 init_std:float=0.02,
                 width_multiplier:float=1.0,
                 use_conv:bool=False,
                 disable_activation_norm:bool=False
        ):
        super().__init__()
        d_out = d_out or d_model
        dim_feedforward = dim_feedforward or 4 * d_model

        self.use_conv = use_conv
        linear_layer = partial(nn.Conv1d, kernel_size=9, padding=4) if use_conv else nn.Linear
        self.disable_activation_norm = disable_activation_norm
        if not disable_activation_norm:
            self.norm = norm_layer(d_model) if norm_layer is not None and not disable_activation_norm else nn.Identity()

        self.c_fc    = linear_layer(d_model, dim_feedforward, bias=bias_front)
        self.c_proj  = nn.Linear(dim_feedforward, d_out, bias=bias_last)

        self.activation  = get_activation_fn(activation)
        if droppath_prob > 0.0:
            self.dropout_module = DropPath(droppath_prob)
        else:
            self.dropout_module = ClassicDropout(
                dropout, module_name=self.__class__.__name__
            )

        self.activation_dropout_module = ClassicDropout(
            activation_dropout, module_name=self.__class__.__name__
        )
        self.init_std = init_std
        self.width_multiplier = width_multiplier
        self.standparam = standparam

        self.reset_parameters()

    def reset_parameters(self): 
        """
        •	Width and depth are treated on orthogonal axes in the MuP + CompleteP recipe:
	    •	Width → done via parameterization/init and readout scaling (MuReadout).
	    •	Depth → done via forward residual scaling (1/L) and (optionally) optimizer LR per stack.
        """
        
        init_weights(
            self.c_fc,   
            init_std=self.init_std, 
            standparam=self.standparam, 
            width_multiplier=self.width_multiplier)
        init_weights(
            self.c_proj, 
            init_std=self.init_std, 
            standparam=self.standparam, 
            width_multiplier=self.width_multiplier)
         

    def forward(self, x:Tensor) -> Tensor:
        """
        Inputs:
        x : of shape (B, L, in_features)
        Returns
        x : of shape (B, L, out_features)
        """
        if self.use_conv:
            x=x.permute(0,2,1)
        
        x = self.c_fc(x)
        x = self.activation(x)
        x = self.activation_dropout_module(x)
        if not self.disable_activation_norm:
            x = self.norm(x)
        x = self.c_proj(x)
        x = self.dropout_module(x)

        if self.use_conv:
            x=x.permute(0,2,1)
        return x



class MuPSelfAttention(Module):
    def __init__(self, 
                 d_model:int, 
                 nhead:int, 
                 attention_dropout:float=0.1, 
                 dropout:float=0.1, 
                 droppath_prob:float=0.0,
                 bias:bool=True, 
                 attn_mult:float=1.0,
                 standparam:bool=False,
                 init_std:float=0.02,
                 width_multiplier:float=1.0,
                 norm_layer: nn.Module=None):
        super().__init__()
        """
        standparam: bool, whether to use the standard attention logits.
            With MuP (standparam=False), scale the logits as
                scale = 1 / d_head * sqrt(attn_mult).
            Without MuP (standparam=True), scale the logits as
                scale = 1 / sqrt(d_head)
            
        attn_mult: float, the multiplier of the logits.
            If to match with standard attention logits (standparam=True) 
            and if d_head is kept the same across the family of models:
                self.attn_mult  = d_head
            else:
                self.attn_mult  = 1.0

        mup_width_multiplier: float = 1.0
            mup_width_multiplier = width / base_width where base_width is typically 256
        """
        assert d_model % nhead == 0
        self.n_head   = nhead
        self.d_model  = d_model
        self.head_dim = self.d_model // self.n_head
        self.init_std = init_std

        # ---- normalization
        self.ln_q = norm_layer(self.head_dim) if norm_layer is not None else nn.Identity()
        self.ln_k = norm_layer(self.head_dim) if norm_layer is not None else nn.Identity()

        # ---- projections
        self.c_attn = nn.Linear(self.d_model, 3 * self.d_model, bias=bias)
        self.c_proj = nn.Linear(self.d_model, self.d_model, bias=bias)

        # ---- regularization
        self.attention_dropout  = attention_dropout
        if droppath_prob > 0.0:
            self.dropout_module = DropPath(droppath_prob)
        else:
            self.dropout_module = ClassicDropout(
                dropout, module_name=self.__class__.__name__
            )

        # ---- MuP controls 
        self.width_multiplier = width_multiplier
        self.attn_mult = attn_mult
        self.standparam = standparam  # False→1/d_head, True→1/√d_head (legacy)
        
        self.scale = self.register_buffer(
            'scale', 
            torch.tensor(self.width_scale()),
            persistent=True,
        )
        # ---- MuP-friendly init: fan-in normal; **zero-initialize Q block**
        self.reset_parameters()

    def reset_parameters(self):
        """
        •	Width and depth are treated on orthogonal axes in the MuP + CompleteP recipe:
	        •	Width → done via parameterization/init and readout scaling (MuReadout).
	        •	Depth → done via forward residual scaling (1/L) and (optionally) optimizer LR per stack.
        """
        init_weights(self.c_attn, init_std=self.init_std, standparam=self.standparam, width_multiplier=self.width_multiplier)
        init_weights(self.c_proj, init_std=self.init_std, standparam=self.standparam, width_multiplier=self.width_multiplier)
         
        if not self.standparam:
            # zero the first third rows (the Q block) in c_attn.weight
            q_rows = slice(0, self.d_model)  # [0 : d_model)
            constant_(self.c_attn.weight[q_rows, :], 0.)
    
    def width_scale(self):
        """
        Returns the scalar that multiplies QK^T before softmax.

        Modes:
        - MuP   (preferred): 1/d_head * sqrt(attn_mult)
        - Std              : 1/sqrt(d_head) * sqrt(attn_mult)
        """
        if self.standparam:
            return 1.0 / math.sqrt(self.head_dim)  # legacy
        else:
            return (1.0 / self.head_dim) * math.sqrt(self.attn_mult)

    def forward(
        self, 
        x:Tensor, 
        key_padding_mask:Tensor=None, 
        attn_mask:Tensor=None, 
        attn_bias:Tensor=None, 
        is_causal:bool=False,
        kv_cache: dict | None = None,
        use_cache: bool = False
    ) -> tuple[Tensor, dict | None]:
        """
        key_padding_mask:
            If specified, a mask of shape (N,S) indicating which elements within key 
            to ignore for the purpose of attention (i.e. treat as “padding”). 
            For unbatched query, shape should be (S). Binary and float masks are supported. 
            For a binary mask, a True value indicates that the corresponding key value 
            will be ignored for the purpose of attention. 
            For a float mask, it will be directly added to the corresponding key value
        
        attn_mask:
            If specified, a 2D or 3D or 4D mask preventing attention to certain positions. 
            Must be of shape (L, S) or (N⋅num_heads, L, S), where N is the batch size, L is the target sequence length, and S is the source sequence length.    
            A 2D mask will be broadcasted across the batch while a 3D mask allows for a different mask for each entry in the batch. 
            A 4D mask may have shape (N, 1, L, S) or (N, num_heads, L, S); the singleton head form is broadcast without materializing one copy per head.
            Binary and float masks are supported. For a binary mask, a True value indicates that the corresponding position is not allowed to attend. 
            For a float mask, the mask values will be added to the attention weight. If both attn_mask and key_padding_mask are supplied, their types should match.

        is_causal: bool, whether to apply the causal mask.
        """
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.d_model, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, H, T, Dh)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q, k = self.ln_q(q), self.ln_k(k)
        # checked with https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
        # and multi_head_attention_forward from https://github.com/microsoft/mup/blob/main/examples/Transformer/model.py
        # the scaling factor is correct
        if use_cache:
            if kv_cache is not None:
                # ---- KV cache append
                k = torch.cat([kv_cache["k"], k], dim=2)
                v = torch.cat([kv_cache["v"], v], dim=2)
            kv_cache = {"k": k, "v": v}
            is_causal = False # causal handled structurally

        source_length = k.shape[2]
        if use_cache and attn_mask is not None:
            raise ValueError("Cached attention supports key padding masks, not attention masks.")
        if use_cache and attn_bias is not None:
            raise ValueError("Cached attention supports key padding masks, not attention biases.")

        if not use_cache or key_padding_mask is not None:
            key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=None, # F._none_or_dtype(attn_mask),
            other_name="", # "attn_mask",
            target_type=x.dtype)

            attn_mask = F._canonical_mask(
                mask=attn_mask,
                mask_name="attn_mask",
                other_type=None,
                other_name="",
                target_type=x.dtype,
                check_other=False)

            attn_bias = F._canonical_mask(
                mask=attn_bias,
                mask_name="attn_bias",
                other_type=None,
                other_name="",
                target_type=x.dtype,
                check_other=False)
            
            if attn_mask is not None:
                # Normalize to SDPA's broadcastable (B|1, H|1, T, S) form.
                if attn_mask.dim() == 2:
                    correct_2d_size = (T, source_length)
                    if attn_mask.shape != correct_2d_size:
                        raise RuntimeError(
                            f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}."
                        )
                    attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
                elif attn_mask.dim() == 3:
                    correct_3d_size = (B * self.n_head, T, source_length)
                    if attn_mask.shape != correct_3d_size:
                        raise RuntimeError(
                            f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}."
                        ) 
                    attn_mask = attn_mask.view(B, self.n_head, T, source_length)
                elif attn_mask.dim() == 4:
                    correct_batch_and_sequence = (
                        attn_mask.shape[0] == B
                        and attn_mask.shape[2:] == (T, source_length)
                    )
                    if (
                        not correct_batch_and_sequence
                        or attn_mask.shape[1] not in (1, self.n_head)
                    ):
                        raise RuntimeError(
                            "The shape of the 4D attn_mask is "
                            f"{attn_mask.shape}, but should be "
                            f"({B}, 1|{self.n_head}, {T}, {source_length})."
                        )
                else:
                    raise RuntimeError(
                        f"attn_mask's dimension {attn_mask.dim()} is not supported"
                    )
            
            # merge key padding and attention masks
            if key_padding_mask is not None:
                if not torch.jit.is_scripting() and not torch.jit.is_tracing(): # type: ignore
                    _check_key_padding_mask(key_padding_mask, source_length, B)

                key_padding_mask = key_padding_mask.view(B, 1, 1, source_length)
                if attn_mask is None:
                    attn_mask = key_padding_mask
                else:
                    attn_mask = attn_mask + key_padding_mask

            if attn_bias is not None:
                # if not torch.jit.is_scripting() and not torch.jit.is_tracing():
                #     _check_key_padding_mask(attn_bias, T, B)

                if attn_bias.dim() == 3:
                    correct_3d_size = (B, T, source_length)
                    if attn_bias.shape != correct_3d_size:
                        raise RuntimeError(
                            f"The shape of the 3D attn_bias is {attn_bias.shape}, but should be {correct_3d_size}."
                        ) 
                    attn_bias = attn_bias.view(B, 1, T, source_length)
                elif attn_bias.dim() == 4:
                    correct_batch_and_sequence = (
                        attn_bias.shape[0] == B
                        and attn_bias.shape[2:] == (T, source_length)
                    )
                    if (
                        not correct_batch_and_sequence
                        or attn_bias.shape[1] not in (1, self.n_head)
                    ):
                        raise RuntimeError(
                            "The shape of the 4D attn_bias is "
                            f"{attn_bias.shape}, but should be "
                            f"({B}, 1|{self.n_head}, {T}, {source_length})."
                        )
                else:
                    raise RuntimeError(
                        f"attn_bias's dimension {attn_bias.dim()} is not supported"
                    )
                
                if attn_mask is None:
                    attn_mask = attn_bias
                else:
                    attn_mask = attn_mask + attn_bias
            
        # ---- Attention
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
            scale=self.scale,
        )
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        attn_output = self.c_proj(attn_output)
        attn_output = self.dropout_module(attn_output)
        return attn_output, kv_cache
