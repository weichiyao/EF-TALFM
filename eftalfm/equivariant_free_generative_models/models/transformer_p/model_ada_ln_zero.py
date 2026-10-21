
# ===============================================================================
# AdaLNModulation
#===============================================================================
# Adapted from https://github.com/facebookresearch/DiT/blob/main/models.py 

from __future__ import annotations

from torch import nn
from torch import Tensor 

from collections import OrderedDict

from eftalfm.equivariant_free_generative_models.models.submodel import MuPSelfAttention, MuPMLP 
from eftalfm.equivariant_free_generative_models.models.utils import fixed_dimension_init_weights

def modulate(x, shift:Tensor, scale:Tensor):
    return x * (1 + scale) + shift

def approx_gelu(x):
    return nn.GELU(approximate="tanh")(x)


class AdaLNZeroBlock(nn.Module):
    """A transformer block with adaptive layer norm zero (adaLN-Zero) conditioning.
    
    adaLN-Zero: Rreplace the block's normalization behavior with condition-controlled normalization, plus gated residual updates

        Applies modulation specifically around LayerNorm:
        - normalize first
        - then modulate normalized activations with shift/scale
        Adds learned residual gates, e.g.
        - attention branch gate
        - MLP branch gate
        “Zero” means those modulation/gating layers are initialized so the conditioning path starts near zero effect

        Why adaLN-Zero is often stronger for transformers:
        - it modulates each sublayer at the point where transformers are most sensitive: after normalization
        - separate control for attention and MLP is useful
        - zero-init means you do not abruptly disturb pretrained/base behavior at the start of training

    """
    def __init__(self, 
                 d_model:int, 
                 num_heads:int, 
                 dim_feedforward:int|None=None, 
                 bias:bool=True, 
                 c_dim:int|None=None,
                 activation:str="approx_gelu",
                 attn_norm_layer: nn.Module=None,
                 mlp_norm_layer: nn.Module=None,
                 width_multiplier:float=1.0, 
                 depth_multiplier:float=1.0, 
                 depth_alpha_exp:float=1.0,
                 attn_mult:float=1.0, 
                 init_std:float=0.02, 
                 **kwargs):
        super().__init__()
        dim_feedforward = dim_feedforward or 4 * d_model

        self.ln_1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.attn = MuPSelfAttention(
            d_model, 
            nhead=num_heads, 
            dropout=0,
            bias=bias, 
            attn_mult=attn_mult,
            init_std=init_std,
            width_multiplier=width_multiplier,
            norm_layer=attn_norm_layer)

        self.ln_2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.mlp = MuPMLP(
            d_model=d_model,  
            dim_feedforward=dim_feedforward, 
            bias_front=bias, 
            bias_last=bias, 
            dropout=0, 
            activation=activation, 
            init_std=init_std, 
            width_multiplier=width_multiplier,
            norm_layer=mlp_norm_layer)
        
        self.adaLN_modulation = nn.Sequential(
            OrderedDict([
                ("silu", nn.SiLU()),
                ("c_proj", nn.Linear(c_dim, 6 * d_model, bias=True))
            ])
        )
        
        self.residual_scaling = 1 / (depth_multiplier ** depth_alpha_exp) if depth_alpha_exp > 0.0 else 1.0
        self.reset_parameters()

    def reset_parameters(self):
        # Zero-out adaLN modulation layers:
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)


    def forward(
        self, 
        x: Tensor, 
        c: Tensor,
        key_padding_mask: Tensor=None, 
        attn_mask: Tensor=None, 
        attn_bias: Tensor=None, 
        is_causal: bool=False,
        kv_cache: dict | None = None,
        use_cache: bool = False
    ) -> Tensor:
        """
        INPUTS:
        ================================================
        x: (B, L, D)
        c: (B, 1, D) or (B, L, D)
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        attn_output, kv_cache =  self.attn(
            modulate(self.ln_1(x), shift_msa, scale_msa), 
            key_padding_mask=key_padding_mask, 
            attn_mask=attn_mask, 
            attn_bias=attn_bias, 
            is_causal=is_causal,
            kv_cache=kv_cache,
            use_cache=use_cache
        )  
        x = x + self.residual_scaling * gate_msa * attn_output
        x = x + self.residual_scaling * gate_mlp * self.mlp(modulate(self.ln_2(x), shift_mlp, scale_mlp))
        return x, kv_cache # (B, L, D)


class FinalAdaLNZeroLayer(nn.Module):
    """
    The final layer of AdaLNZeroTransformerEncoder.
    """
    def __init__(self, d_model:int, c_dim:int, hidden_dim:int|None=None, init_std:float=0.02):
        super().__init__()
        self.ln_f = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        hidden_dim = d_model if hidden_dim is None else hidden_dim
        self.adaLN_modulation = nn.Sequential(
            OrderedDict([
                ("linear1", nn.Linear(c_dim, hidden_dim, bias=True)),
                ("silu", nn.SiLU()),
                ("linear2", nn.Linear(hidden_dim, 2 * d_model, bias=True))
            ])
        ) 

        fixed_dimension_init_weights(self.adaLN_modulation[0], init_std=init_std)
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)


    def forward(self, x:Tensor, c:Tensor):
        """
        x: (B, L, D)
        c: (B, 1, D) or (B, L, D)
        """
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.ln_f(x), shift, scale)
        return x


class AdaLNZeroTransformerEncoder(nn.Module):
    def __init__(self, 
                 d_model:int, 
                 c_dim:int,
                 n_head:int, 
                 n_layer:int,
                 dim_feedforward:int | None=None, 
                 d_out:int | None=None, 
                 width_multiplier:float=1.0, 
                 depth_multiplier:float=1.0, 
                 depth_alpha_exp:float=1.0, 
                 attn_mult:float=1.0, 
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
        d_out = d_out or d_model
        dim_feedforward = dim_feedforward or 4 * d_model

        self.width_input_alpha    = width_input_alpha
        self.width_output_alpha   = width_output_alpha
        self.width_multiplier     = width_multiplier

        self.encoder = nn.ModuleList(
            [
                AdaLNZeroBlock(
                    d_model=d_model,
                    c_dim=c_dim,
                    dim_feedforward=dim_feedforward,
                    num_heads=n_head,
                    bias=True,
                    activation="approx_gelu",
                    attn_norm_layer=nn.LayerNorm,
                    mlp_norm_layer=None,
                    width_multiplier=width_multiplier,
                    depth_multiplier=depth_multiplier,
                    depth_alpha_exp=depth_alpha_exp,
                    attn_mult=attn_mult,
                    init_std=init_std,
                ) # init_weights has been called in the AdaLNZeroBlock
                for _ in range(n_layer)
            ])
        self.final_layer = FinalAdaLNZeroLayer(d_model, c_dim, init_std=init_std)
        self.readout = nn.Linear(d_model, d_out)

    def reset_parameters(self):
        fixed_dimension_init_weights(self.readout, init_std=self.init_std)

    def forward(
        self, 
        x: Tensor, 
        c: Tensor,
        key_padding_mask: Tensor=None, 
        attn_mask: Tensor=None, 
        attn_bias: Tensor=None, 
        is_causal: bool=False,
        kv_caches: dict | None = None,
        use_cache: bool = False
    ) -> Tensor:
        """
        During generation, model of L layers 
            x.shape == (B, 1, d_model) at every step
            c.shape == (B, 1, D) or (B, L, D)
            At step 0, 
                input kv_caches = None
                output new_caches[l] = = {"k": (B, H, 1, Dh), "v": (B, H, 1, Dh)}
            At step t, 
                input kv_caches[l] = {"k": (B, H, t, Dh), "v": (B, H, t, Dh)}
                output new_caches[l] = {"k": (B, H, t+1, Dh), "v": (B, H, t+1, Dh)}
        """
        assert c.dim() == 3 and c.shape[0] == x.shape[0], "c must be a 3D tensor with the same batch size as x"
        
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
                    c,
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
                    c,
                    key_padding_mask=key_padding_mask, 
                    attn_mask=attn_mask, 
                    attn_bias=attn_bias, 
                    is_causal=is_causal, 
                )
        
        x = self.final_layer(x, c)
        
        ### Begin muP code ### 
        x *= self.width_output_alpha / self.width_multiplier
        ### End muP code ###

        x = self.readout(x) 

        return x, new_caches
