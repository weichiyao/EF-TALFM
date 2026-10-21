from __future__ import annotations
from torch import nn 
from torch import Tensor

from collections import OrderedDict
from eftalfm.equivariant_free_generative_models.models.utils import (
    init_weights,
    get_activation_fn,
    fixed_dimension_init_weights,
)

from eftalfm.equivariant_free_generative_models.models.transformer_p.dropout import ClassicDropout, DropPath

def _broadcast_cond(cond: Tensor, x: Tensor) -> Tensor:
    # cond: (B, D), x: (B, D) or (B, L, D)
    while cond.ndim < x.ndim:
        cond = cond.unsqueeze(1)
    return cond


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1.0 + scale) + shift
 

def _build_adaln_mlp(cond_dim: int, hidden_dim: int, out_dim: int, init_std: float, disable_zero_init_adaln:bool=False) -> nn.Sequential:
    """SiLU MLP cond -> modulation params; final layer zero-init (AdaLN-Zero)."""
    mod = nn.Sequential(
        OrderedDict([
            ("linear1", nn.Linear(cond_dim, hidden_dim, bias=True)),
            ("silu", nn.SiLU()),
            ("linear2", nn.Linear(hidden_dim, out_dim, bias=True)),
        ])
    )
    fixed_dimension_init_weights(mod[0], init_std=init_std)
    if disable_zero_init_adaln: 
        fixed_dimension_init_weights(mod[2], init_std=init_std)
    else: 
        nn.init.zeros_(mod[2].weight)
        nn.init.zeros_(mod[2].bias)
    return mod


class FinalAdaLNZeroLayer(nn.Module):
    """
    cond -> shift, scale for final normalized hidden state.

    Time (`cond_t`) and label (`cond_y`) are injected through *separate*
    zero-initialized modulation MLPs whose outputs are summed, so the label has
    its own dedicated conditioning path rather than being fused with time.
    """
    def __init__(self, cond_dim: int, d_model: int, hidden_dim: int | None = None, init_std: float = 0.02, use_y_cond: bool = False, disable_zero_init_adaln: bool = True):
        super().__init__()
        self.ln_f = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        hidden_dim = d_model if hidden_dim is None else hidden_dim
        self.use_y_cond = use_y_cond
        self.adaLN_modulation_t = _build_adaln_mlp(cond_dim, hidden_dim, 2 * d_model, init_std, disable_zero_init_adaln=disable_zero_init_adaln)
        if use_y_cond:
            self.adaLN_modulation_y = _build_adaln_mlp(cond_dim, hidden_dim, 2 * d_model, init_std, disable_zero_init_adaln=disable_zero_init_adaln)
        else:
            self.adaLN_modulation_y = None

    def forward(self, x: Tensor, cond_t: Tensor, cond_y: Tensor = None):
        mod = self.adaLN_modulation_t(cond_t)
        if self.adaLN_modulation_y is not None and cond_y is not None:
            mod = mod + self.adaLN_modulation_y(cond_y)
        shift, scale = mod.chunk(2, dim=-1)
        shift = _broadcast_cond(shift, x)
        scale = _broadcast_cond(scale, x)
        x = modulate(self.ln_f(x), shift, scale)
        return x


class CompletePAdaLNZeroBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        cond_dim: int,
        dropout: float = 0.0,
        activation_dropout: float = 0.0,
        droppath_prob: float = 0.0, 
        activation: str = "silu",
        width_multiplier: float = 1.0,
        depth_multiplier: float = 1.0,
        depth_alpha_exp: float = 1.0,
        standparam: bool = False,
        init_std: float = 0.02,
        use_y_cond: bool = False,
        disable_zero_init_adaln: bool = False,
    ):
        super().__init__()
        self.init_std = init_std
        self.standparam = standparam
        self.width_multiplier = width_multiplier
        self.use_y_cond = use_y_cond

        self.ln_0 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.c_fc = nn.Linear(d_model, hidden_dim, bias=True)
        self.c_proj = nn.Linear(hidden_dim, d_model, bias=True) 
        self.activation = get_activation_fn(activation)
        if droppath_prob > 0.0:
            self.dropout_module = DropPath(droppath_prob)
        else:
            self.dropout_module = ClassicDropout(
                dropout, module_name=self.__class__.__name__
            )
        self.activation_dropout_module = ClassicDropout(
            activation_dropout, module_name=self.__class__.__name__
        )
        # Separate zero-init modulation paths for time and label; their
        # (shift, scale, gate) outputs are summed in forward.
        self.adaLN_modulation_t = nn.Sequential(
            OrderedDict([
                ("c_fc", nn.Linear(cond_dim, cond_dim)),
                ("silu", nn.SiLU()),
                ("c_proj", nn.Linear(cond_dim, 3 * d_model))
            ])
        )
        if use_y_cond:
            self.adaLN_modulation_y = nn.Sequential(
                OrderedDict([
                    ("c_fc", nn.Linear(cond_dim, cond_dim)),
                    ("silu", nn.SiLU()),
                    ("c_proj", nn.Linear(cond_dim, 3 * d_model))
                ])
            )
        else:
            self.adaLN_modulation_y = None

        self.residual_scaling = (
            1 / (depth_multiplier ** depth_alpha_exp)
            if (depth_alpha_exp > 0 and (not standparam))
            else 1.0
        )
        self.disable_zero_init_adaln = disable_zero_init_adaln
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init_weights(
            self.c_fc,
            init_std=self.init_std,
            standparam=self.standparam,
            width_multiplier=self.width_multiplier,
        )
        init_weights(
            self.c_proj,
            init_std=self.init_std,
            standparam=self.standparam,
            width_multiplier=self.width_multiplier,
        )
        
        for mod in (self.adaLN_modulation_t, self.adaLN_modulation_y):
            if mod is None:
                continue
            fixed_dimension_init_weights(mod[0], init_std=self.init_std)
            if self.disable_zero_init_adaln: 
                fixed_dimension_init_weights(mod[2], init_std=self.init_std)
            else:
                # Zero-out final adaLN modulation layer so each path starts at identity:
                nn.init.zeros_(mod[2].weight)
                nn.init.zeros_(mod[2].bias)

    def forward(self, x: Tensor, cond_t: Tensor, cond_y: Tensor = None) -> Tensor:
        # adaLN modulation:
        # cond -> shift, scale, gate
        # Zero-init final layers so each block starts near identity.
        # Time and label are modulated through separate paths and summed.
        mod = self.adaLN_modulation_t(cond_t)
        if self.adaLN_modulation_y is not None and cond_y is not None:
            mod = mod + self.adaLN_modulation_y(cond_y)
        shift, scale, gate = mod.chunk(3, dim=-1)
        shift = _broadcast_cond(shift, x)
        scale = _broadcast_cond(scale, x)
        gate  = _broadcast_cond(gate, x)
        h = modulate(self.ln_0(x), shift, scale)

        # MLP:
        h = self.c_fc(h)
        h = self.activation(h)
        h = self.activation_dropout_module(h)
        h = self.c_proj(h)
        h = self.dropout_module(h)
        # IMPORTANT: no post-add activation
        x = x + self.residual_scaling * gate * h
        return x


class CompletePAdaLNResNet(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        input_dim: int = 128,
        hidden_dim: int = 512,
        cond_dim: int = 128,
        output_dim: int = 1,
        num_blocks: int = 5,
        activation: str = "silu",
        dropout: float = 0.0,
        activation_dropout: float = 0.0,
        droppath_prob: float = 0.0,
        schedule_droppath: bool = False,
        width_input_alpha: float = 1.0,
        width_output_alpha: float = 1.0,
        width_multiplier: float = 1.0,
        depth_multiplier: float = 1.0,
        depth_alpha_exp: float = 1.0,
        standparam: bool = False,
        init_std: float = 0.02,
        zero_init_readout: bool = False,
        disable_zero_init_adaln: bool = False,
        use_y_cond: bool = False,
    ):
        super().__init__()
        self.use_y_cond = use_y_cond
        self.width_input_alpha = width_input_alpha
        self.width_output_alpha = width_output_alpha
        self.width_multiplier = width_multiplier
        self.depth_multiplier = depth_multiplier
        self.depth_alpha_exp = depth_alpha_exp
        self.standparam = standparam
        self.init_std = init_std
        self.zero_init_readout = zero_init_readout
        self.disable_zero_init_adaln = disable_zero_init_adaln 

        self.in_encoder = nn.Linear(input_dim, d_model)

        blocks = []
        for block_idx in range(num_blocks):
            block_droppath_prob = (
                droppath_prob * block_idx / max(num_blocks - 1, 1)
                if schedule_droppath
                else droppath_prob
            )
            blocks.append(
                CompletePAdaLNZeroBlock(
                    d_model=d_model,
                    hidden_dim=hidden_dim,
                    cond_dim=cond_dim,
                    dropout=dropout,
                    activation_dropout=activation_dropout,
                    droppath_prob=block_droppath_prob,
                    activation=activation,
                    width_multiplier=width_multiplier,
                    depth_multiplier=depth_multiplier,
                    depth_alpha_exp=depth_alpha_exp,
                    standparam=standparam,
                    init_std=init_std,
                    use_y_cond=use_y_cond,
                    disable_zero_init_adaln=disable_zero_init_adaln,
                )
            )
        self.residual_blocks = nn.ModuleList(blocks)

        self.final_layer = FinalAdaLNZeroLayer(
            cond_dim=cond_dim,
            d_model=d_model,
            hidden_dim=cond_dim,
            init_std=init_std,
            use_y_cond=use_y_cond,
            disable_zero_init_adaln=disable_zero_init_adaln
        )
        self.readout = nn.Linear(d_model, output_dim)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        fixed_dimension_init_weights(self.in_encoder, init_std=self.init_std)

        if self.zero_init_readout:
            nn.init.zeros_(self.readout.weight)
            if self.readout.bias is not None:
                nn.init.zeros_(self.readout.bias)
        else:
            fixed_dimension_init_weights(self.readout, init_std=self.init_std)

    def forward(self, x: Tensor, cond_t: Tensor, cond_y: Tensor = None) -> Tensor:
        x = self.in_encoder(x)
        if not self.standparam:
            x = x * self.width_input_alpha

        for block in self.residual_blocks:
            x = block(x, cond_t, cond_y)

        x = self.final_layer(x, cond_t, cond_y)

        if not self.standparam:
            x = x * (self.width_output_alpha / self.width_multiplier)

        x = self.readout(x)
        return x
