"""
Full definition of a decoder model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/EleutherAI/nanoGPT-mup/blob/completep/model.py
"""

from __future__ import annotations
from torch import Tensor 
from torch import nn 
from torch.nn import Module 
 
from eftalfm.equivariant_free_generative_models.models.submodel import MuPSelfAttention, MuPMLP, LayerNorm
from eftalfm.equivariant_free_generative_models.models.utils import fixed_dimension_init_weights

class CompletePBlock(Module):
    def __init__(
        self, 
        d_model:int, 
        nhead:int, 
        dim_feedforward:int | None=None,
        dropout:float=0.1,
        attention_dropout:float=0.1, 
        activation_dropout:float=0.1,
        droppath_prob:float=0.0,
        bias:bool=True, 
        activation:str="gelu", 
        disable_activation_norm:bool=False,
        norm_first:bool=True, 
        layer_norm_eps:float=1e-5,
        depth_multiplier:float=1.0, 
        depth_alpha_exp:float=1.0,
        standparam:bool=False,
        attn_mult:float=1.0, 
        init_std:float=0.02,
        width_multiplier:float=1.0):

        super().__init__()
        """
        width_multiplier: float = 1 
            mup_width_multiplier = width / base_width where base_width is typically 256
        depth_multiplier: float = 1.0 
            depth_multiplier = depth / base_depth
        depth_alpha_exp: float = 1.0 
            a float in the range [0.5, 1] that controls 
            how residual branches are scaled as a function of depth. 
            This results in residual connections of the type 
            x = x + depth_multiplier**(-depth_alpha_exp) * branch(x) 
            with LR correction eta *= depth_multiplier**(depth_alpha_exp-1)
        """
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model
        self.norm_first = norm_first
        
        self.attn = MuPSelfAttention(
            d_model, 
            nhead, 
            attention_dropout=attention_dropout, 
            dropout=dropout, 
            droppath_prob=droppath_prob,
            bias=bias, 
            attn_mult=attn_mult,
            standparam=standparam,
            init_std=init_std,
            width_multiplier=width_multiplier)

        self.ln_1 = LayerNorm(d_model, bias=bias, eps=layer_norm_eps) 
        self.ln_2 = LayerNorm(d_model, bias=bias, eps=layer_norm_eps) 
        self.residual_scaling = 1 / (depth_multiplier ** depth_alpha_exp) if depth_alpha_exp > 0.0 else 1.0

        self.mlp = MuPMLP(
            d_model, 
            dim_feedforward=dim_feedforward,
            bias_front=bias,
            bias_last=bias, 
            dropout=dropout, 
            droppath_prob=droppath_prob,
            activation_dropout=activation_dropout,
            activation=activation, 
            disable_activation_norm=disable_activation_norm,
            standparam=standparam, 
            init_std=init_std, 
            width_multiplier=width_multiplier)

    def forward(
        self, 
        x:Tensor, 
        key_padding_mask:Tensor=None, 
        attn_mask:Tensor=None, 
        attn_bias:Tensor=None, 
        is_causal:bool=False,
        kv_cache: dict | None = None,
        use_cache: bool = False
    ) -> Tensor: 
        """
        •	Width and depth are treated on orthogonal axes in the MuP + CompleteP recipe:
	        •	Width → done via parameterization/init and readout scaling (MuReadout).
	        •	Depth → done via forward residual scaling (1/L) and (optionally) optimizer LR per stack.
        """
        if self.norm_first:
            attn_output, kv_cache = self.attn(
                self.ln_1(x), 
                key_padding_mask=key_padding_mask, 
                attn_mask=attn_mask, 
                attn_bias=attn_bias, 
                is_causal=is_causal,
                kv_cache=kv_cache,
                use_cache=use_cache
            )
            x = x + self.residual_scaling * attn_output
            x = x + self.residual_scaling * self.mlp(self.ln_2(x))
        else:
            attn_output, kv_cache = self.attn(
                x, 
                key_padding_mask=key_padding_mask, 
                attn_mask=attn_mask, 
                attn_bias=attn_bias, 
                is_causal=is_causal,
                kv_cache=kv_cache,
                use_cache=use_cache
            )
            x = self.ln_1(x + self.residual_scaling * attn_output)
            x = self.ln_2(x + self.residual_scaling * self.mlp(x))
        
        return x, kv_cache

class CompletePTransformerEncoder(Module):
    def __init__(self, 
                 d_model:int, 
                 n_head:int, 
                 n_layer:int,
                 dim_feedforward:int | None = None, 
                 d_out:int | None = None,
                 dropout:float=0.0, 
                 attention_dropout:float=0.0,
                 activation_dropout:float=0.0,
                 droppath_prob:float=0.0,
                 bias:bool=True, 
                 activation:str="gelu", 
                 disable_activation_norm:bool=False,
                 norm_first:bool=True, 
                 layer_norm_eps:float=1e-5, 
                 width_multiplier:float=1.0, 
                 depth_multiplier:float=1.0, 
                 depth_alpha_exp:float=1.0, 
                 attn_mult:float=1.0, 
                 standparam:bool=False, 
                 init_std:float=0.02,
                 width_input_alpha:float=1.0,
                 width_output_alpha:float=1.0,
                 **kwargs):
        """
        width_input_alpha: float = 1 
            Optional tunable multiplier applied to input embedding forward pass output
        width_output_alpha: float = 1 
            Optional tunable multiplier applied to output unembedding forward pass output
        width_multiplier: float = 1 
            mup_width_multiplier = width / base_width where base_width is typically 256
        depth_multiplier: float = 1.0 
            depth_multiplier = depth / base_depth
        depth_alpha_exp: float = 1.0 
            a float in the range [0.5, 1] that controls 
            how residual branches are scaled as a function of depth. 
            This results in residual connections of the type 
            x = x + depth_multiplier**(-depth_alpha_exp) * branch(x) 
            with LR correction eta *= depth_multiplier**(depth_alpha_exp-1)
        """
        super().__init__()
        self.standparam           = standparam
        self.width_input_alpha    = width_input_alpha
        self.width_output_alpha   = width_output_alpha
        self.width_multiplier     = width_multiplier
        self.depth_multiplier     = depth_multiplier
        self.depth_alpha_exp      = depth_alpha_exp
        self.norm_first           = norm_first
        self.d_out                = d_out

        self.encoder = nn.ModuleList(
            [
                CompletePBlock(
                    d_model=d_model,
                    nhead=n_head,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    droppath_prob=droppath_prob,
                    bias=bias,
                    activation=activation,
                    disable_activation_norm=disable_activation_norm,
                    norm_first=norm_first,
                    layer_norm_eps=layer_norm_eps,
                    width_multiplier=width_multiplier,
                    depth_multiplier=depth_multiplier,
                    depth_alpha_exp=depth_alpha_exp,
                    attn_mult=attn_mult,
                    standparam=standparam,
                    init_std=init_std,
                ) # init_weights has been called in the CompletePBlock
                for _ in range(n_layer)
            ])
        if self.norm_first:
            self.ln_f = LayerNorm(d_model, eps=layer_norm_eps)
        
        self.readout = None
        if d_out is not None:
            self.readout = nn.Linear(d_model, d_out)
            fixed_dimension_init_weights(self.readout, init_std=init_std)

    def forward(
        self, 
        x:Tensor, 
        c:Tensor=None,
        key_padding_mask:Tensor=None, 
        attn_mask:Tensor=None, 
        attn_bias:Tensor=None, 
        is_causal:bool=False,
        kv_caches: dict | None = None,
        use_cache: bool = False,
    ) -> Tensor:
        """
        During generation, model of L layers 
            x.shape == (B, 1, d_model) at every step
            At step 0, 
                input kv_caches = None
                output new_caches[l] = = {"k": (B, H, 1, Dh), "v": (B, H, 1, Dh)}
            At step t, 
                input kv_caches[l] = {"k": (B, H, t, Dh), "v": (B, H, t, Dh)}
                output new_caches[l] = {"k": (B, H, t+1, Dh), "v": (B, H, t+1, Dh)}
        """
        
        if not self.standparam:
            ### Begin muP code ###
            x *= self.width_input_alpha
            ### End muP code ###
        
        new_caches = None
        if use_cache:
            new_caches = []
            if kv_caches is None:
                kv_caches = [None] * len(self.encoder)
            for block, cache in zip(self.encoder, kv_caches):
                x, new_cache = block(
                    x, 
                    key_padding_mask=key_padding_mask, 
                    attn_mask=attn_mask, 
                    attn_bias=attn_bias, 
                    is_causal=is_causal,
                    kv_cache=cache,
                    use_cache=use_cache
                )
                new_caches.append(new_cache)

        else:
            for block in self.encoder:
                x, _ = block(
                    x, 
                    key_padding_mask=key_padding_mask, 
                    attn_mask=attn_mask, 
                    attn_bias=attn_bias, 
                    is_causal=is_causal, 
                )

        if self.norm_first:
            x = self.ln_f(x)
        
        if not self.standparam:
            ### Begin muP code ### 
            # Scaling `x` instead of `logits` allows coord check to log change
            x *= self.width_output_alpha / self.width_multiplier
            ### End muP code ###

        if self.readout is not None:
            x = self.readout(x)

        return x, new_caches
