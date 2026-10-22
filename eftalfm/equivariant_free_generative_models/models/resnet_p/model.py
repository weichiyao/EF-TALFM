""" 
Modified from https://github.com/EleutherAI/nanoGPT-mup/blob/completep/model.py
"""

from torch import nn 
from torch import Tensor
from eftalfm.equivariant_free_generative_models.models.submodel import LayerNorm
from eftalfm.equivariant_free_generative_models.models.utils import init_weights, get_activation_fn
from eftalfm.equivariant_free_generative_models.models.utils import fixed_dimension_init_weights
from eftalfm.equivariant_free_generative_models.models.dropout import ClassicDropout, DropPath
# Define the ResNet block
class CompletePBasicBlock(nn.Module):
    """
    LayerNorm(dim) normalizes each sample individually 
        across its features.
    Dropout not Standard in Original ResNets: 
        But can be included if overfitting is a concern.
    """
    expansion = 1

    def __init__(
        self, 
        input_dim:int, 
        hidden_dim:int,
        output_dim:int, 
        dropout:float=0.0,
        activation_dropout:float=0.0,
        droppath_prob:float=0.0,
        pre_ln:bool=False,
        disable_activation_ln:bool=False,
        disable_residual_ln:bool=False,
        activation:str="gelu",
        width_multiplier:float=1.0,
        depth_multiplier:float=1.0,
        depth_alpha_exp:float=1.0,
        standparam:bool=False,
        init_std:float=0.02,
    ):
        super().__init__()  
        self.init_std = init_std
        self.standparam = standparam
        self.width_multiplier = width_multiplier 
        

        self.disable_activation_ln = disable_activation_ln
        self.disable_residual_ln = disable_residual_ln
        self.pre_ln = pre_ln
        self.c_fc = nn.Linear(input_dim, hidden_dim, bias=True)
        self.c_proj = nn.Linear(hidden_dim, output_dim, bias=True)
        if pre_ln:
            self.ln_0 = LayerNorm(input_dim)
        if not disable_activation_ln:
            self.ln_1 = LayerNorm(hidden_dim)
        if not disable_residual_ln:
            self.ln_2 = LayerNorm(output_dim) 
        if droppath_prob > 0.0:
            self.dropout_module = DropPath(droppath_prob)
        else:
            self.dropout_module = ClassicDropout(
                dropout, module_name=self.__class__.__name__
            )
        self.activation_dropout_module = ClassicDropout(
            activation_dropout, module_name=self.__class__.__name__
        )
        self.activation = get_activation_fn(activation)
        self.residual_scaling = 1/(depth_multiplier ** depth_alpha_exp) if (depth_alpha_exp > 0 and (not standparam)) else 1.0

        self.shortcut = None
        if input_dim != output_dim:
            self.shortcut = nn.Sequential(
                nn.Linear(input_dim, output_dim, bias=True),
                LayerNorm(output_dim)  
            )
        
        self.reset_parameters()  

    def reset_parameters(self) -> None:
        # muP initialization
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
        if self.shortcut is not None:
            init_weights(
                self.shortcut[0],   
                init_std=self.init_std, 
                standparam=self.standparam, 
                width_multiplier=self.width_multiplier)

    def forward(self, x:Tensor):
        # residual branch
        if self.pre_ln:
            residual = self.c_fc(self.ln_0(x))
        else:
            residual = self.c_fc(x)
        if not self.disable_activation_ln:
            residual = self.ln_1(residual)
        residual = self.activation(residual)
        residual = self.activation_dropout_module(residual)
        residual = self.c_proj(residual)
        if not self.disable_residual_ln:
            # LN normalizes the residual
            residual = self.ln_2(residual)
        # DropPath zeros some samples, scales kept by 1/keep_prob
        # E[residual] = keep_prob · (LN(z)/keep_prob) = LN(z)
        residual = self.dropout_module(residual)
        # If dropout_module first, then ln_2, E[residual] = keep_prob·LN(z) + (1−keep_prob)⋅0 = keep_prob⋅LN(z)
        # With drop_prob=0.1 → keep_prob=0.9, the expected residual contribution is 90% of intended. Stacked across 12 blocks, the cumulative attenuation is:
        # 0.9^12 = 0.282429536481 = 28.24%
        # We're effectively running a network with ~28% of the designed residual signal strength, 
        # -- the effective network depth shrunk dramatically

        ### Begin CompleteP code ###
        # apply depth scaling to the *residual update*, not the shortcut
        residual = self.residual_scaling * residual
        ### End CompleteP code ###

        # shortcut branch
        shortcut = self.shortcut(x) if self.shortcut is not None else x

        out = shortcut + residual
        # Post-add activation is usually not wanted for FM-style vector fields 
        # out = self.activation(out)
        return out

class CompletePResNet(nn.Module):
    def __init__(
        self, 
        d_model:int=128,
        input_dim:int=128,
        hidden_dim:int=128,
        output_dim:int=1, 
        num_blocks:int=5, 
        activation:str="gelu", 
        dropout:float=0.0,
        activation_dropout:float=0.0,
        droppath_prob:float=0.0,
        schedule_droppath:bool=False,
        pre_ln:bool=False,
        disable_activation_ln:bool=False,
        disable_residual_ln:bool=False,
        disable_last_ln:bool=False,
        width_input_alpha:float=1.0,
        width_output_alpha:float=1.0,
        width_multiplier:float=1.0,
        depth_multiplier:float=1.0,
        depth_alpha_exp:float=1.0,
        standparam:bool=False,
        init_std:float=0.02, 
    ):
        super().__init__() 
        self.width_input_alpha = width_input_alpha
        self.width_output_alpha = width_output_alpha
        self.width_multiplier = width_multiplier
        self.depth_multiplier = depth_multiplier
        self.depth_alpha_exp = depth_alpha_exp
        self.standparam = standparam
        self.init_std = init_std
        self.pre_ln = pre_ln
        self.disable_residual_ln = disable_residual_ln
        self.disable_last_ln = disable_last_ln
        blocks = []
        self.in_encoder = nn.Linear(input_dim, d_model)
        for block_idx in range(num_blocks):
            block_droppath_prob = (
                droppath_prob * block_idx / max(num_blocks - 1, 1)
                if schedule_droppath
                else droppath_prob
            )
            blocks.append(
                CompletePBasicBlock(
                    input_dim=d_model,
                    hidden_dim=hidden_dim,
                    output_dim=d_model,
                    dropout=dropout,
                    activation_dropout=activation_dropout,
                    droppath_prob=block_droppath_prob,
                    pre_ln=pre_ln,
                    disable_activation_ln=disable_activation_ln,
                    disable_residual_ln=disable_residual_ln,
                    activation=activation,
                    width_multiplier=width_multiplier,
                    depth_multiplier=depth_multiplier,
                    depth_alpha_exp=depth_alpha_exp,
                    standparam=standparam,
                    init_std=init_std,
                )
            )
        self.residual_blocks = nn.Sequential(*blocks)
        if not self.disable_last_ln:
            self.ln_f = LayerNorm(d_model)
            # which can erase latent magnitude before readout.
        self.readout = nn.Linear(d_model, output_dim)

        self.reset_parameters()
        
    def reset_parameters(self) -> None: 
        fixed_dimension_init_weights(self.in_encoder, init_std=self.init_std)
        fixed_dimension_init_weights(self.readout, init_std=self.init_std)

    def forward(self, x:Tensor) -> Tensor:
        # Input encoder to project the input to the model dimension
        x = self.in_encoder(x)
        if not self.standparam:
            ### Begin muP code ###
            x *= self.width_input_alpha
            ### End muP code ###    
        # ResNet blocks
        x = self.residual_blocks(x)
        # Final layer normalization
        # - flow matching: do not disable so more stable for vector field
        # - predictive: disable so to preserve the magnitude of the latent
        if not self.disable_last_ln:
            x = self.ln_f(x)
        # Readout layer
        if not self.standparam:
            ### Begin muP code ### 
            # Scaling `x` instead of `logits` allows coord check to log change
            x *= self.width_output_alpha / self.width_multiplier
            ### End muP code ###
        
        x = self.readout(x)
        return x
