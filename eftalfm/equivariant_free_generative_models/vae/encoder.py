from __future__ import annotations

import torch
from torch import nn
from torch import Tensor



 
from equivariant_free_generative_models.models.transformer_p.model import CompletePTransformerEncoder
from eftalfm.equivariant_free_generative_models.models.dropout import ClassicDropout
from equivariant_free_generative_models.vae.layers import ParticleFeature, FinetuneHead


 


class TransformerCompletePEncoder(nn.Module):

    def __init__(
        self,
        latent_dim: int,
        num_atoms: int,
        context_dim: int=0,
        num_encoder_layers: int = 6,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1, 
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        droppath_prob: float = 0.0,
        disable_activation_norm: bool = False,
        max_seq_len: int = 256, 
        use_position_embeddings: bool = True,
        norm_first: bool = False, # set to True 
        activation: str = "relu",  
        width_multiplier: float = 1.0,
        depth_multiplier: float = 1.0,
        depth_alpha_exp: float = 1.0, 
        standparam: bool = False,
        width_disable_attention_scaling: bool = False,
        init_std: float = 0.02,
        width_input_alpha: float = 1.0,
        width_output_alpha: float = 1.0,  
        position_scale: float = 1.0,
        **kwargs,
    ) -> None:
        """
        mode_prob: {2D+3D, 2D, 3D}
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim 
        self.context_dim = context_dim
        self.use_position_embeddings = use_position_embeddings
        self.width_multiplier = width_multiplier
        self.depth_multiplier = depth_multiplier
        self.depth_alpha_exp = depth_alpha_exp
        self.position_scale = position_scale 
        if (not standparam):
            if (not width_disable_attention_scaling):
                ### Begin muP code ###
                attn_mult = 1.0
                ### End muP code ###
            else:
                attn_mult = embedding_dim // num_attention_heads
        else:
            attn_mult = 1.0
        
        print(f"Encoder: width_disable_attention_scaling = {width_disable_attention_scaling}", 
            f"depth_multiplier = {depth_multiplier}", 
            f"depth_alpha_exp = {depth_alpha_exp}", 
            f"attn_mult = {attn_mult}", 
            f"standparam = {standparam}", 
            f"init_std = {init_std}", 
            f"width_multiplier = {width_multiplier}", 
            f"width_input_alpha = {width_input_alpha}", 
            f"width_output_alpha = {width_output_alpha}!!!!!!")
            
        # Positional encoding only need to be present in particle_features but not in attn_bias 
        self.particle_feature = ParticleFeature( 
            hidden_dim=embedding_dim, 
            context_dim=context_dim,
            num_atoms=num_atoms, 
            max_seq_len=max_seq_len,
            use_position_embeddings=use_position_embeddings,
            init_std=init_std
        ) 
        
        self.model = CompletePTransformerEncoder(
            d_model=self.embedding_dim,
            n_head=num_attention_heads,
            n_layer=num_encoder_layers,
            dim_feedforward=ffn_embedding_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            droppath_prob=droppath_prob,
            disable_activation_norm=disable_activation_norm,
            bias=False,
            activation=activation,
            norm_first=norm_first,
            layer_norm_eps=1e-12,
            width_multiplier=width_multiplier,
            depth_multiplier=depth_multiplier,
            depth_alpha_exp=depth_alpha_exp,
            attn_mult=attn_mult,
            standparam=standparam,
            init_std=init_std,
            width_input_alpha=width_input_alpha,
            width_output_alpha=width_output_alpha)

        self.dropout_module = ClassicDropout(
            dropout, module_name=self.__class__.__name__
        )
        self.readout = FinetuneHead(
            input_dim=embedding_dim,
            inner_dim=embedding_dim,
            output_dim=latent_dim*2,
            activation_fn=activation,
            init_std=init_std) 

    def forward(
        self,
        x: Tensor,  
        position: Tensor=None,
        token_padding_mask: Tensor=None,
        num_atoms: Tensor=None,
        charge: Tensor=None,
        spin_multiplicity: Tensor=None,
        number_of_rings: Tensor=None,
        has_fused_rings: Tensor=None,
        y: Tensor=None,
        **kwargs,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """ 
        INPUTS: n_nodes include the eos token 
        ==============================
        - position   (b, n_nodes, 3)  
        - x          (b, n_nodes, 3)
            0 - padding, 1 - eos 
        - padding_mask (b, n_nodes) whether the locations are zero-padded
            False-valid, True-padding
            EOS token is valid, so set as False
        - charge     (b, ) optional
        - spin_multiplicity (b, ) optional
        - number_of_rings (b, ) optional
        - has_fused_rings (b, ) optional 

        RETURNS:
        ==============================
        - z_mean       (b, 1, latent_dim)
        - z_std        (b, 1, latent_dim)
        - padding_mask (b, 1+n_nodes)
        """
        # is_tpu = False
        # compute padding mask. This is needed for multi-head attention
        if position is None:
            raise ValueError(
                "position is required when num_mol_datasets > 0."
            )
        position_scaled = position.clone() / self.position_scale
    
        n_mol, n_max_nodes = x.shape[:2]
        if token_padding_mask is None: 
            if num_atoms is not None:
                token_idx = torch.arange(n_max_nodes, device=x.device).unsqueeze(0) # (1, T)
                token_padding_mask = token_idx >= num_atoms.unsqueeze(1)  # B x T
            else:
                # Fallback for callers that do not pass num_atoms.
                token_padding_mask = x[:, :, 0].eq(0)  # B x T

        padding_mask = torch.zeros(n_mol, n_max_nodes + 1, device=x.device, dtype=torch.bool)
        padding_mask[:, 1:] = token_padding_mask

        if self.context_dim > 0:
            if y is None:
                raise ValueError("y must be provided when encoder context_dim > 0.")
            y = y.reshape(n_mol, -1)
            if y.shape[-1] != self.context_dim:
                raise ValueError(
                    f"Expected y to have context_dim={self.context_dim}, received shape {tuple(y.shape)}."
                )
        
        # encode features 
        x = self.particle_feature(
            x, 
            position_scaled, 
            y=y,
            padding_mask=padding_mask, 
            charge=charge, 
            spin_multiplicity=spin_multiplicity, 
            number_of_rings=number_of_rings, 
            has_fused_rings=has_fused_rings,
        ) # (B, 1+T, d_model)
        
        # mask the padding inputs
        x = x.masked_fill(padding_mask.unsqueeze(-1).to(torch.bool), 0.0)

        # dropout 
        x = self.dropout_module(x)  # (B, 1+T, d_model)

        # transformer encoder
        x, _ = self.model(x, key_padding_mask=padding_mask) # B, 1+T, d_model
        
        # readout 
        x = self.readout(x)   # (B, 1+T, latent_dim*2)
        
        bz = x.shape[0] 
        z = x[:,0,:].reshape(bz, 2, -1)
        z_mean = z[:, :1, :]
        z_std = torch.exp(0.5 * z[:, 1:, :]) # logvar
        
        if torch.any(torch.isnan(z_std)):
            print('Warning: detected nan in z_std, resetting to one.')
            z_std = torch.ones_like(z_std)
    
        return z_mean, z_std, padding_mask
 

class Cloud3DEncoder(nn.Module):
    """
    Implementation for the loss used in masked graph model (MGM) training.
    """

    def __init__(self, args):
        super().__init__()
        self.max_positions = args.max_positions 
         
        
        self.model = TransformerCompletePEncoder(
            latent_dim=args.latent_dim,
            num_atoms=args.num_atoms, 
            context_dim=0, # encoder does not take context as inputs
            num_encoder_layers=args.encoder_n_layers,
            embedding_dim=args.encoder_embed_dim,
            ffn_embedding_dim=args.encoder_dim_feedforward,
            num_attention_heads=args.encoder_n_heads,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout if hasattr(args, "attention_dropout") else 0.0,
            activation_dropout=args.act_dropout if hasattr(args, "act_dropout") else 0.0,
            droppath_prob=args.droppath_prob if hasattr(args, "droppath_prob") else 0.0,
            disable_activation_norm=args.disable_activation_norm if hasattr(args, "disable_activation_norm") else False,
            max_seq_len=self.max_positions, # type: ignore 
            use_position_embeddings=not args.no_token_positional_embeddings,
            norm_first=args.norm_first,
            activation=args.activation, 
            width_multiplier=args.encoder_width_multiplier if hasattr(args, "encoder_width_multiplier") else 1.0,
            depth_multiplier=args.encoder_depth_multiplier if hasattr(args, "encoder_depth_multiplier") else 1.0,
            depth_alpha_exp=args.depth_alpha_exp if hasattr(args, "depth_alpha_exp") else 1.0,
            standparam=args.standparam if hasattr(args, "standparam") else False,
            width_disable_attention_scaling=args.width_disable_attention_scaling if hasattr(args, "width_disable_attention_scaling") else False,
            init_std=args.init_std if hasattr(args, "init_std") else 0.02,
            width_input_alpha=args.width_input_alpha if hasattr(args, "width_input_alpha") else 1.0,
            width_output_alpha=args.width_output_alpha if hasattr(args, "width_output_alpha") else 1.0, 
            position_scale=args.position_scale if hasattr(args, "position_scale") else 1.0,
        ) 
         

    def forward(self, **kwargs) -> tuple[Tensor, Tensor, Tensor]:
        return self.model(**kwargs)

    def max_positions(self) -> int:
        """Maximum output length supported by the encoder."""
        return self.max_positions
