from __future__ import annotations

import torch 
from torch import nn 
from torch import Tensor
import torch.nn.functional as F


from eftalfm.equivariant_free_generative_models.models.resnet_p.model import CompletePResNet
from eftalfm.equivariant_free_generative_models.utils import match_size,  softplus
from eftalfm.equivariant_free_generative_models.lgm.embedder import TimeEmbedding, GaussianFourierFeatures
from eftalfm.equivariant_free_generative_models.models.utils import fixed_dimension_init_weights
from eftalfm.equivariant_free_generative_models.models.resnet_p.model_adaln import CompletePAdaLNResNet

class CompletePFlowDynamics(nn.Module):
    def __init__(
        self,
        d_model:int,
        x_dim:int,
        y_dim:int,
        t_dim:int=1,
        hidden_dim:int | None=None, 
        x_encoder_type:str="none",
        t_encoder_type:str="logsnr_sinusoidal",
        y_encoder_type:str="none", 
        n_layers:int=3,
        activation:str="gelu",
        freqs:int=32,
        dropout:float=0.0, 
        activation_dropout:float=0.0,
        droppath_prob:float=0.0,
        schedule_droppath:bool=False,
        standparam:bool=False,
        depth_multiplier:float=1.0,
        depth_alpha_exp:float=1.0,
        width_multiplier:float=1.0,
        width_input_alpha:float=1.0,
        width_output_alpha:float=1.0,
        init_std:float=0.02,
        t_lower_eps:float=1e-12,
        t_upper_eps:float=1e-12,
        t_encoder_max_freq:float=10_000.0,
        t_encoder_min_freq:float=1.0,
        t_encoder_final_norm:bool=True,
        y_num_freqs:int=64,
        y_fourier_scale:float=1.0,
        **kwargs
    ):
        super().__init__()
        self.activation = activation
        if activation == "gelu":
            self.activation_layer = nn.GELU()
        elif activation == "elu":
            self.activation_layer = nn.ELU()
        elif activation == "relu":
            self.activation_layer = nn.ReLU()
        elif activation == "tanh":
            self.activation_layer = nn.Tanh()
        elif activation == "softplus":
            self.activation_layer = nn.Softplus()
        elif activation == "sigmoid":
            self.activation_layer = nn.Sigmoid()
        elif activation == "selu":
            self.activation_layer = nn.SELU()
        elif activation == "silu":
            self.activation_layer = nn.SiLU()
        else:
            raise ValueError(f"Only accept activation_layer 'relu', 'elu', 'gelu', 'selu', 'sigmoid', 'silu', 'softplus', 'tanh'. Received {activation}.")
         
        self.t_dim     = t_dim
        self.x_dim     = x_dim 
        self.y_dim     = y_dim 
        self.d_model   = d_model
        
        self.n_layers  = n_layers  
        
        self.dropout = dropout
        self.activation_dropout = activation_dropout
        self.droppath_prob = droppath_prob
        self.schedule_droppath = schedule_droppath 
        self.standparam = standparam
        self.depth_multiplier = depth_multiplier
        self.depth_alpha_exp = depth_alpha_exp
        self.width_multiplier = width_multiplier
        self.width_input_alpha = width_input_alpha
        self.width_output_alpha = width_output_alpha
        self.init_std = init_std
        self.t_lower_eps = t_lower_eps
        self.t_upper_eps = t_upper_eps
        self.self_condition = bool(kwargs.get("self_condition", False))
        self.y_num_freqs = y_num_freqs
        self.y_fourier_scale = y_fourier_scale
        #################################################### 
        self.t_encoder_type = t_encoder_type
        self.t_encoder = TimeEmbedding(
            d_model=self.d_model,
            sigma=1.0, 
            pe_dim=freqs, 
            method=self.t_encoder_type,
            min_freq=t_encoder_min_freq,
            max_freq=t_encoder_max_freq,
            fourier_scale=16,
            lower_eps=t_lower_eps,
            upper_eps=t_upper_eps,
            apply_init_weights=True,
            init_std=init_std,
            final_norm=t_encoder_final_norm)
        self.in_dim  = (3 if self.y_dim > 0 else 2) * self.d_model
        self.output_dim = self.x_dim
        self.hidden_dim = (4*self.d_model) if hidden_dim is None else hidden_dim
        
        #####################################################
        self.x_encoder_type = x_encoder_type
        if self.x_encoder_type == "mlp":
            self.x_encoder = nn.Sequential(
                nn.Linear(self.x_dim*(int(self.self_condition)+1), self.d_model*(int(self.self_condition)+1)*4),
                self.activation_layer, 
                nn.Linear(self.d_model*(int(self.self_condition)+1)*4, self.d_model),
            )
        elif self.x_encoder_type == "linear":
            self.x_encoder = nn.Linear(self.x_dim*(int(self.self_condition)+1), self.d_model)
        else:
            raise ValueError(f"Only accept x_encoder_type 'mlp', 'linear'. Received {x_encoder_type}!")
        ####################################################
        self.y_encoder_type = y_encoder_type
        self.y_fourier = None
        if self.y_dim > 0:
            if self.y_encoder_type == "linear":
                self.y_encoder = nn.Linear(self.y_dim, self.d_model) 
            elif self.y_encoder_type == "mlp":
                self.y_encoder = nn.Sequential(
                    nn.Linear(self.y_dim, self.d_model),
                    nn.SiLU(),
                    nn.Linear(self.d_model, self.d_model)
                )
            elif self.y_encoder_type == "fourier":
                self.y_fourier = GaussianFourierFeatures(
                    in_dim=self.y_dim,
                    num_freqs=self.y_num_freqs,
                    scale=self.y_fourier_scale,
                )
                self.y_encoder = nn.Sequential(
                    nn.Linear(self.y_fourier.out_dim, self.d_model),
                    nn.SiLU(),
                    nn.Linear(self.d_model, self.d_model),
                )
                fixed_dimension_init_weights(self.y_encoder[0], init_std=self.init_std)
                fixed_dimension_init_weights(self.y_encoder[2], init_std=self.init_std)
            else:
                raise ValueError(f"context_encoder_type can only be 'linear', 'mlp', or 'fourier'. Received {y_encoder_type}!")
        else:
            self.y_encoder = None

        self.initialize_model()

    def _encode_context(self, y: Tensor, x: Tensor, y_cond_mask: Tensor = None) -> Tensor | None:
        if self.y_dim <= 0:
            return None

        if y is None:
            return torch.zeros_like(x)

        if self.y_fourier is not None:
            y = self.y_fourier(y)
        context = self.y_encoder(y)
        context = match_size(context, x)
        if y_cond_mask is not None:
            y_cond_mask = match_size(
                y_cond_mask.reshape(-1, 1).to(dtype=torch.bool, device=context.device),
                context,
            )
            context = context.masked_fill(~y_cond_mask, 0.0)
        return context
        
    def initialize_model(self):
        """
        Initialize the dynamics model
        """
        raise NotImplementedError("Subclasses must implement this method to initialize the dynamics model")

    def forward(self, t: Tensor, x: Tensor, y: Tensor = None, y_cond_mask: Tensor = None) -> Tensor:
        """
        x: (bz,d)
        t: (bz,) or (1,)
        y: (bz,d) or (1,d)
        """ 
        x = self.x_encoder(x) 
        t = self.t_encoder(t.reshape(-1))     # (bz, d)
        t = match_size(t, x)                  # (bz, d)
        out = torch.cat((t, x), dim=-1)       # (bz, 2*d)
        y = self._encode_context(y, x, y_cond_mask=y_cond_mask)
        if y is not None:
            out = torch.cat((out, y), dim=-1)    # (bz, 3*d)
        
        return self.model(out)     

class CompletePResNetFlowDynamics(CompletePFlowDynamics):
    def __init__(
        self,
        d_model:int,
        x_dim:int,
        y_dim:int,
        t_dim:int=1,
        hidden_dim:int | None = None, 
        x_encoder_type:str="none",
        t_encoder_type:str="logsnr_sinusoidal",
        y_encoder_type:str="none", 
        n_layers:int=3,
        activation:str="gelu",
        freqs:int=32,
        dropout:float=0.0,  
        activation_dropout:float=0.0,
        droppath_prob:float=0.0,
        schedule_droppath:bool=False,
        pre_ln:bool=False,
        disable_activation_ln:bool=False,
        disable_residual_ln:bool=False,
        standparam:bool=False,
        depth_multiplier:float=1.0,
        depth_alpha_exp:float=1.0,
        width_multiplier:float=1.0,
        width_input_alpha:float=1.0,
        width_output_alpha:float=1.0,
        init_std:float=0.02,
        t_lower_eps:float=1e-12,
        t_upper_eps:float=1e-12,
        t_encoder_max_freq:float=10_000.0,
        t_encoder_min_freq:float=1.0,
        **kwargs
    ):
        self.pre_ln = pre_ln
        self.disable_activation_ln = disable_activation_ln
        self.disable_residual_ln = disable_residual_ln
        super().__init__(
            d_model,
            x_dim,
            y_dim,
            t_dim,
            hidden_dim, 
            x_encoder_type,
            t_encoder_type,
            y_encoder_type, 
            n_layers,
            activation,
            freqs,  
            dropout,  
            activation_dropout,
            droppath_prob,
            schedule_droppath,
            standparam,
            depth_multiplier,
            depth_alpha_exp,
            width_multiplier,
            width_input_alpha,
            width_output_alpha,
            init_std,
            t_lower_eps,
            t_upper_eps,
            t_encoder_max_freq,
            t_encoder_min_freq,
            **kwargs,
        ) 

    def initialize_model(self):
        """
        Initialize the model
        """
        self.model = CompletePResNet(
            d_model=self.d_model,
            input_dim=self.in_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            num_blocks=self.n_layers,
            activation=self.activation,
            dropout=self.dropout,
            activation_dropout=self.activation_dropout,
            droppath_prob=self.droppath_prob,
            schedule_droppath=self.schedule_droppath,
            pre_ln=self.pre_ln,
            disable_activation_ln=self.disable_activation_ln,
            disable_residual_ln=self.disable_residual_ln,
            width_input_alpha=self.width_input_alpha,
            width_output_alpha=self.width_output_alpha,
            width_multiplier=self.width_multiplier,
            depth_multiplier=self.depth_multiplier,
            depth_alpha_exp=self.depth_alpha_exp,  
            standparam=self.standparam,
            init_std=self.init_std,
        )

class CompletePPositiveLinear(torch.nn.Module):
    """Linear layer with weights forced to be positive."""

    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        bias: bool = True,
        init_std:float=0.02,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            torch.empty((out_features, in_features)))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
     
        self.apply(lambda module: fixed_dimension_init_weights(module, init_std=init_std))

    def forward(self, input):
        positive_weight = softplus(self.weight)
        return F.linear(input, positive_weight, self.bias)

class CompletePGammaNetwork(torch.nn.Module):
    """The gamma network models a monotonic increasing function. Construction as in the VDM paper."""
    def __init__(self):
        super().__init__()

        self.l1 = CompletePPositiveLinear(1, 1)
        self.l2 = CompletePPositiveLinear(1, 1024)
        self.l3 = CompletePPositiveLinear(1024, 1)

        self.gamma_0 = torch.nn.Parameter(torch.tensor([-5.]))
        self.gamma_1 = torch.nn.Parameter(torch.tensor([10.]))
        self.show_schedule()

    def show_schedule(self, num_steps=50):
        t = torch.linspace(0, 1, num_steps).view(num_steps, 1)
        gamma = self.forward(t)
        print('Gamma schedule:')
        print(gamma.detach().cpu().numpy().reshape(num_steps))

    def gamma_tilde(self, t:Tensor) -> Tensor:
        l1_t = self.l1(t)
        return l1_t + self.l3(torch.sigmoid(self.l2(l1_t)))

    def forward(self, t:Tensor) -> Tensor:
        zeros, ones = torch.zeros_like(t), torch.ones_like(t)
        # Not super efficient.
        gamma_tilde_0 = self.gamma_tilde(zeros)
        gamma_tilde_1 = self.gamma_tilde(ones)
        gamma_tilde_t = self.gamma_tilde(t)

        # Normalize to [0, 1]
        normalized_gamma = (gamma_tilde_t - gamma_tilde_0) / (
                gamma_tilde_1 - gamma_tilde_0)

        # Rescale to [gamma_0, gamma_1]
        gamma = self.gamma_0 + (self.gamma_1 - self.gamma_0) * normalized_gamma

        return gamma

class CompletePAdaLNResNetFlowDynamics(CompletePFlowDynamics):
    def __init__(
        self,
        d_model: int,
        x_dim: int,
        y_dim: int,
        t_dim: int = 1,
        hidden_dim: int | None = None,
        x_encoder_type: str = "none",
        t_encoder_type: str = "logsnr_sinusoidal",
        y_encoder_type: str = "none",
        n_layers: int = 3,
        activation: str = "gelu",
        freqs: int = 32,
        dropout: float = 0.0,
        activation_dropout: float = 0.0,
        droppath_prob: float = 0.0,
        schedule_droppath: bool = False,
        standparam: bool = False,
        depth_multiplier: float = 1.0,
        depth_alpha_exp: float = 1.0,
        width_multiplier: float = 1.0,
        width_input_alpha: float = 1.0,
        width_output_alpha: float = 1.0,
        init_std: float = 0.02,
        t_lower_eps: float = 1e-12,
        t_upper_eps: float = 1e-12,
        t_encoder_max_freq: float = 10_000.0,
        t_encoder_min_freq: float = 1.0,
        y_num_freqs: int = 64,
        y_fourier_scale: float = 1.0,
        disable_zero_init_adaln: bool = False,
        **kwargs
    ):
        # Set before super().__init__, since initialize_model() runs inside it.
        self.y_num_freqs = y_num_freqs
        self.y_fourier_scale = y_fourier_scale
        self.disable_zero_init_adaln = disable_zero_init_adaln
        super().__init__(
            d_model,
            x_dim,
            y_dim,
            t_dim,
            hidden_dim,
            x_encoder_type,
            t_encoder_type,
            y_encoder_type,
            n_layers,
            activation,
            freqs,
            dropout,
            activation_dropout,
            droppath_prob,
            schedule_droppath,
            standparam,
            depth_multiplier,
            depth_alpha_exp,
            width_multiplier,
            width_input_alpha,
            width_output_alpha,
            init_std,
            t_lower_eps,
            t_upper_eps,
            t_encoder_max_freq,
            t_encoder_min_freq,
            y_num_freqs=y_num_freqs,
            y_fourier_scale=y_fourier_scale,
            **kwargs,
        )

    def initialize_model(self):
        self.cond_dim = self.d_model

        # x path: state only (keep inherited x_encoder).
        # Time is used directly as cond_t (the t_encoder already produces a
        # d_model embedding); each AdaLN path applies its own SiLU-MLP.

        # Label path is selected by y_encoder_type in the parent class.
        self.use_y_cond = self.y_dim > 0

        self.model = CompletePAdaLNResNet(
            d_model=self.d_model,
            input_dim=self.d_model,
            hidden_dim=self.hidden_dim,
            cond_dim=self.cond_dim,
            output_dim=self.output_dim,
            num_blocks=self.n_layers,
            activation=self.activation,
            dropout=self.dropout,
            activation_dropout=self.activation_dropout,
            droppath_prob=self.droppath_prob,
            schedule_droppath=self.schedule_droppath,
            width_input_alpha=self.width_input_alpha,
            width_output_alpha=self.width_output_alpha,
            width_multiplier=self.width_multiplier,
            depth_multiplier=self.depth_multiplier,
            depth_alpha_exp=self.depth_alpha_exp,
            standparam=self.standparam,
            init_std=self.init_std,
            zero_init_readout=False,
            use_y_cond=self.use_y_cond,
            disable_zero_init_adaln=self.disable_zero_init_adaln,
        )

    def _encode_y(self, y: Tensor, x: Tensor, y_cond_mask: Tensor = None) -> Tensor | None:
        """
        Build the label conditioning vector cond_y using the selected y encoder.

        Returns a (bz, d_model) tensor. When y is None (fully unconditional) or
        a sample is dropped via y_cond_mask, that row is set to zeros so the
        label-modulation path produces a consistent "null" signal at both
        train and sample time (matching CFG dropout semantics).
        """
        if not self.use_y_cond:
            return None
        if y is None:
            return torch.zeros_like(x)

        if self.y_fourier is not None:
            y = self.y_fourier(y)
        cond_y = self.y_encoder(y)   # (bz, d_model)
        cond_y = match_size(cond_y, x)
        if y_cond_mask is not None:
            y_cond_mask = match_size(
                y_cond_mask.reshape(-1, 1).to(dtype=torch.bool, device=cond_y.device),
                cond_y,
            )
            cond_y = cond_y.masked_fill(~y_cond_mask, 0.0)
        return cond_y

    def forward(self, t: Tensor, x: Tensor, y: Tensor = None, y_cond_mask: Tensor = None) -> Tensor:
        """
        x: (bz, x_dim)
        t: (bz,) or (1,)
        y: (bz, y_dim) or (1, y_dim)
        """
        xh = self.x_encoder(x)                  # (bz, d_model)

        th = self.t_encoder(t.reshape(-1))      # (bz, d_model)
        cond_t = match_size(th, xh)             # (bz, d_model)

        cond_y = self._encode_y(y, xh, y_cond_mask=y_cond_mask)  # (bz, d_model) or None

        return self.model(xh, cond_t, cond_y)
