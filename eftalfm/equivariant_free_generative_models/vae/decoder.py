from __future__ import annotations

import torch
from torch import Tensor
from torch import nn
from eftalfm.equivariant_free_generative_models.vae.layers import (
    N_HAS_FUSED_RINGS_TYPES,
    N_NUMBER_OF_RINGS_TYPES, 
    N_SPIN_MULTIPLICITY_TYPES,
    N_TOTAL_CHARGE_TYPES, 
)
 
from eftalfm.equivariant_free_generative_models.models.utils import fixed_dimension_init_weights, get_activation_fn, sample_from_logits

class CloudReadout(nn.Module):
    def __init__(
            self, 
            in_dims:int, 
            hidden_dims:int,
            out_dims:int, 
            activation:str,  
            init_std:float=0.02,
            depth:int=2,
        ):
        super().__init__()
        if depth < 1:
            raise ValueError(f"CloudReadout depth must be >= 1, got {depth}.")
        if depth == 1:
            self.linear1 = nn.Linear(in_dims, out_dims)
            self.readout_hidden_layers = nn.ModuleList()
            self.linear2 = None
        else:
            self.linear1 = nn.Linear(in_dims, hidden_dims)
            self.readout_hidden_layers = nn.ModuleList(
                nn.Linear(hidden_dims, hidden_dims) for _ in range(depth - 2)
            )
            self.linear2 = nn.Linear(hidden_dims, out_dims)
        self.activation_fn = get_activation_fn(activation)  
        self.init_std = init_std
        self.apply(lambda module: fixed_dimension_init_weights(module, init_std=self.init_std))

    def forward_features(self, x: Tensor) -> Tensor:
        """Return the feature immediately before the final output layer."""
        x = self.linear1(x)
        if self.linear2 is None:
            return x
        x = self.activation_fn(x)
        for layer in self.readout_hidden_layers:
            x = self.activation_fn(layer(x))
        return x

    def forward(self, x: Tensor) -> Tensor:
        x = self.forward_features(x)
        if self.linear2 is None:
            return x
        return self.linear2(x)

 
class CloudARDecoder(nn.Module):
    PROPERTY_SOURCE_MOLECULE = 0
    PROPERTY_SOURCE_LATENT = 1
    PROPERTY_SOURCE_JOINT = 2
    PROPERTY_SOURCE_NAMES = ("molecule", "latent", "joint")

    def __init__(
            self,  
            latent_dim:int,
            emb_dim:int,  
            num_atoms:int,                  # number of distinct atom types, including atomic number, chirality, hydradization, etc.
            context_dim:int=0, 
            num_atom_features:int=1,        # number of atom features, if 1 then only decode the atomic number; 
            max_n_nodes:int=1000, 
            n_heads:int=4, 
            depth:int=4,
            dim_feedforward:int | None = None, 
            dropout:float=0.,
            attention_dropout:float=0.0,
            activation_dropout:float=0.0,
            droppath_prob:float=0.0,
            norm_first:bool=True,
            activation:str="gelu", 
            disable_activation_norm:bool=False,
            layer_norm_eps:float=1e-5,
            width_multiplier: float = 1.0,
            depth_multiplier: float = 1.0,
            standparam: bool = False,
            width_disable_attention_scaling: bool = False,
            init_std: float = 0.02,
            width_input_alpha: float = 1.0,
            width_output_alpha: float = 1.0,
            depth_alpha_exp: float = 1.0,
            temperature:float=1,
            top_k:int=1,  
            use_ada_ln_zero:bool=False,
            position_scale: float = 1.0, 
            pos_readout_depth: int = 2, 
            explicit_property_token: bool = False,
            property_molecule_only_prob: float = 0.0,
            property_latent_only_prob: float = 0.0,
            property_eval_mode: str = "sample",  
            **kwargs
        ):
        super().__init__()
        self.latent_dim=latent_dim
        self.emb_dim=emb_dim  
        self.num_atoms=num_atoms                 # number of distinct atom types, including atomic number, chirality, hydradization, etc.
        self.context_dim=context_dim
        self.num_atom_features=num_atom_features        # number of atom features, if 1 then only decode the atomic number; 
        self.max_n_nodes=max_n_nodes 
        self.n_heads=n_heads 
        self.depth=depth
        self.dim_feedforward=dim_feedforward
        self.dropout=dropout
        self.attention_dropout=attention_dropout
        self.activation_dropout=activation_dropout
        self.droppath_prob=droppath_prob
        self.norm_first=norm_first
        self.activation=activation
        self.disable_activation_norm=disable_activation_norm
        self.layer_norm_eps=layer_norm_eps
        self.width_multiplier=width_multiplier
        self.depth_multiplier=depth_multiplier
        self.standparam=standparam
        self.width_disable_attention_scaling=width_disable_attention_scaling
        self.init_std=init_std
        self.width_input_alpha=width_input_alpha
        self.width_output_alpha=width_output_alpha
        self.depth_alpha_exp=depth_alpha_exp
        self.temperature=temperature
        self.top_k=top_k 
        self.use_ada_ln_zero=use_ada_ln_zero 
        self.position_scale=position_scale  
        self.pos_readout_depth=pos_readout_depth 
        self.explicit_property_token = bool(explicit_property_token)
        self.property_molecule_only_prob = float(property_molecule_only_prob)
        self.property_latent_only_prob = float(property_latent_only_prob) 
        if not 0.0 <= self.property_molecule_only_prob <= 1.0:
            raise ValueError(
                "property_molecule_only_prob must be in [0, 1]; received "
                f"{self.property_molecule_only_prob}."
            )
        if not 0.0 <= self.property_latent_only_prob <= 1.0:
            raise ValueError(
                "property_latent_only_prob must be in [0, 1]; received "
                f"{self.property_latent_only_prob}."
            )
        if self.property_molecule_only_prob + self.property_latent_only_prob > 1.0:
            raise ValueError(
                "property_molecule_only_prob + property_latent_only_prob must "
                "be <= 1; received "
                f"{self.property_molecule_only_prob + self.property_latent_only_prob}."
            )
        if property_eval_mode not in {"sample", "molecule", "latent", "joint"}:
            raise ValueError(
                "property_eval_mode must be 'sample', 'molecule', 'latent', or "
                "'joint'; received "
                f"{property_eval_mode!r}."
            )
        self.property_eval_mode = property_eval_mode
        if self.explicit_property_token and self.context_dim <= 0:
            raise ValueError(
                "explicit_property_token requires context_dim > 0."
            )


        if (not self.standparam):
            if (not self.width_disable_attention_scaling):
                ### Begin muP code ###
                self.attn_mult = 1.0
                ### End muP code ###
            else:
                self.attn_mult = self.emb_dim // self.n_heads
        else:
            self.attn_mult = 1.0
        
        print(f"Decoder: width_disable_attention_scaling = {self.width_disable_attention_scaling}", 
            f"depth_multiplier = {self.depth_multiplier}", 
            f"depth_alpha_exp = {self.depth_alpha_exp}", 
            f"attn_mult = {self.attn_mult}", 
            f"standparam = {self.standparam}", 
            f"init_std = {self.init_std}", 
            f"width_multiplier = {self.width_multiplier}", 
            f"width_input_alpha = {self.width_input_alpha}", 
            f"width_output_alpha = {self.width_output_alpha}!!!!!!")

        if self.latent_dim != self.emb_dim:
            self.latent_encoder = nn.Linear(self.latent_dim, self.emb_dim)
        else:
            self.latent_encoder = None

        self.emb_dim_each_x = self.emb_dim // (self.num_atom_features+1)   # even size for each x feature
        self.emb_dim_pos  = self.emb_dim - self.emb_dim_each_x * self.num_atom_features             # leftover goes to the position
 
        assert self.emb_dim_pos + self.emb_dim_each_x * self.num_atom_features == self.emb_dim, \
            f"emb_dim_pos + emb_dim_each_x must be equal to emb_dim; received emb_dim_pos: {self.emb_dim_pos}, emb_dim_each_x: {self.emb_dim_each_x}, emb_dim: {self.emb_dim}"
        
        if self.dim_feedforward is None: 
            self.dim_feedforward = 4*self.emb_dim
              
        self.vocab_size = self.num_atoms
        ## Embedding for Transformer
        self.atomseq_encoder = nn.Embedding(self.vocab_size, self.emb_dim_each_x, padding_idx=0)
        self.mixing          = nn.Linear(self.emb_dim, self.emb_dim)
        # Positional encoding 
        positional_vocab_size = self.max_n_nodes + (2 if self.explicit_property_token else 0)
        self.pte = nn.Embedding(positional_vocab_size, self.emb_dim)
        # Type embedding 
        self.tte_pos = nn.Embedding(1, self.emb_dim_pos)
        self.tte_x   = nn.Embedding(1, self.emb_dim_each_x)  
        self.register_buffer("type_ids", torch.tensor([0], dtype=torch.long), persistent=False)
        
        print(f"Decoder: use_ada_ln_zero = {self.use_ada_ln_zero}")
        if not self.use_ada_ln_zero:
            from eftalfm.equivariant_free_generative_models.models.transformer_p.model import CompletePTransformerEncoder
            self.decoder = CompletePTransformerEncoder(
                d_model=self.emb_dim,
                n_head=self.n_heads,
                n_layer=self.depth,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout,
                attention_dropout=self.attention_dropout,
                activation_dropout=self.activation_dropout,
                droppath_prob=self.droppath_prob,
                disable_activation_norm=self.disable_activation_norm,
                bias=False,
                activation=self.activation,
                norm_first=self.norm_first,
                layer_norm_eps=self.layer_norm_eps,
                depth_multiplier=self.depth_multiplier,
                depth_alpha_exp=self.depth_alpha_exp,
                attn_mult=self.attn_mult,
                standparam=self.standparam,
                init_std=self.init_std,
                width_multiplier=self.width_multiplier,
                width_input_alpha=self.width_input_alpha,
                width_output_alpha=self.width_output_alpha)
        else: 
            from eftalfm.equivariant_free_generative_models.models.transformer_p.model_ada_ln_zero import AdaLNZeroTransformerEncoder
            self.decoder = AdaLNZeroTransformerEncoder(
                d_model=self.emb_dim,
                c_dim=self.latent_dim,
                n_head=self.n_heads,
                n_layer=self.depth,
                dim_feedforward=self.dim_feedforward,
                bias=True,
                activation="approx_gelu",
                layer_norm_eps=self.layer_norm_eps,
                width_multiplier=self.width_multiplier,
                depth_multiplier=self.depth_multiplier,
                depth_alpha_exp=self.depth_alpha_exp,
                attn_mult=self.attn_mult,
                init_std=self.init_std,
                width_input_alpha=self.width_input_alpha,
                width_output_alpha=self.width_output_alpha)
        
        if self.context_dim > 0:
            self.context_decoder = CloudReadout(
                in_dims=self.emb_dim,
                hidden_dims=self.emb_dim,
                out_dims=self.context_dim,    
                activation=self.activation, 
                init_std=self.init_std) 
            if self.explicit_property_token:
                # A learned token identity.  The forward pass places it at
                # sequence index 1 and adds pte(1), so both its physical slot
                # and positional id are fixed across every batch.
                self.property_token = nn.Embedding(1, self.emb_dim)
                self.register_buffer(
                    "property_token_id",
                    torch.tensor([0], dtype=torch.long),
                    persistent=False,
                )
            else:
                self.property_token = None
            print(
                "Decoder: explicit_property_token = "
                f"{self.explicit_property_token}, "
                "property_molecule_only_prob = "
                f"{self.property_molecule_only_prob}, "
                "property_latent_only_prob = "
                f"{self.property_latent_only_prob}, "
                f"property_eval_mode = {self.property_eval_mode}"
            )
        else:
            self.property_token = None
            print("No property required to reconstruct in CloudARDecoder")

         
        self.atompos_encoder = nn.Linear(3, self.emb_dim_pos) 
        ## Readout layer 
        self.atompos_decoder = CloudReadout(
            in_dims=self.emb_dim_pos,
            hidden_dims=self.emb_dim,   
            out_dims=3, 
            activation=self.activation,   
            init_std=self.init_std,
            depth=self.pos_readout_depth)

        self.atomseq_decoder = CloudReadout(
            in_dims=self.emb_dim_each_x,
            hidden_dims=self.emb_dim,
            out_dims=self.vocab_size,    
            activation=self.activation, 
            init_std=self.init_std)
         
        self.charge_decoder = CloudReadout(
            in_dims=self.latent_dim,
            hidden_dims=self.emb_dim,
            out_dims=N_TOTAL_CHARGE_TYPES,    
            activation=self.activation, 
            init_std=self.init_std)
 
        self.spinmp_decoder = CloudReadout(
            in_dims=self.latent_dim,
            hidden_dims=self.emb_dim,
            out_dims=N_SPIN_MULTIPLICITY_TYPES,    
            activation=self.activation, 
            init_std=self.init_std)
 
        self.number_of_rings_decoder = CloudReadout(
            in_dims=self.latent_dim,
            hidden_dims=self.emb_dim,
            out_dims=N_NUMBER_OF_RINGS_TYPES,    
            activation=self.activation, 
            init_std=self.init_std)
             
        self.has_fused_rings_decoder = CloudReadout(
            in_dims=self.latent_dim,
            hidden_dims=self.emb_dim,
            out_dims=N_HAS_FUSED_RINGS_TYPES,    
            activation=self.activation, 
            init_std=self.init_std)


        self.pad_token_id = 0
        self.eos_token_id = 1
        self.reset_parameters()

    def reset_parameters(self):
        fixed_dimension_init_weights(self.tte_pos, init_std=self.init_std)
        fixed_dimension_init_weights(self.tte_x, init_std=self.init_std) 
        
        if self.latent_encoder is not None:
            fixed_dimension_init_weights(self.latent_encoder, init_std=self.init_std) 
        
        fixed_dimension_init_weights(self.atomseq_encoder, init_std=self.init_std)
        fixed_dimension_init_weights(self.mixing, init_std=self.init_std)
        fixed_dimension_init_weights(self.pte, init_std=self.init_std)
        
        if self.context_dim > 0:
            fixed_dimension_init_weights(self.context_decoder, init_std=self.init_std)
            if self.property_token is not None:
                fixed_dimension_init_weights(self.property_token, init_std=self.init_std)
        
        fixed_dimension_init_weights(self.atompos_encoder, init_std=self.init_std) 
        fixed_dimension_init_weights(self.charge_decoder, init_std=self.init_std)
        fixed_dimension_init_weights(self.spinmp_decoder, init_std=self.init_std)
        fixed_dimension_init_weights(self.number_of_rings_decoder, init_std=self.init_std)
        fixed_dimension_init_weights(self.has_fused_rings_decoder, init_std=self.init_std)
        
    def _decode_molecule_logits(self, latent_embeds: Tensor):
        # molecule features 
        pred_charge_logits = self.charge_decoder(latent_embeds).squeeze(1)    # (bz, num_charges)
        pred_spinmp_logits = self.spinmp_decoder(latent_embeds).squeeze(1)   # (bz, num_spin_multiplicities)    
        pred_number_of_rings_logits = self.number_of_rings_decoder(latent_embeds).squeeze(1)   # (bz, num_number_of_rings)
        pred_has_fused_rings_logits = self.has_fused_rings_decoder(latent_embeds).squeeze(1)   # (bz, num_has_fused_rings)
         
        return (
            pred_charge_logits,
            pred_spinmp_logits,
            pred_number_of_rings_logits,
            pred_has_fused_rings_logits
        )

    def _atom_type_embeds(self) -> Tensor:
        return self.tte_x(self.type_ids).view(1, 1, 1, -1)

    def _position_type_embeds(self) -> Tensor:
        return self.tte_pos(self.type_ids).view(1, 1, -1)

    def _embed_atom_features(self, x: Tensor) -> Tensor:
        x_emb = self.atomseq_encoder(x) + self._atom_type_embeds()    # (bz, max_n_nodes, d_x, emb_dim_each_x) + (1, 1, 1, emb_dim_each_x) 
        return x_emb.view(x.shape[0], x.shape[1], -1).contiguous()    # (bz, max_n_nodes, d_x*emb_dim_each_x) 

    def _embed_positions(self, position_scaled: Tensor) -> Tensor:
        return self.atompos_encoder(position_scaled) + self._position_type_embeds()  # (bz, max_n_nodes, emb_dim_pos) + (1, 1, emb_dim_pos)
    
    def _embed_frac_coords(self, frac_coords: Tensor) -> Tensor:
        return self.atomfrac_encoder(frac_coords) + self._position_type_embeds()  # (bz, max_n_nodes, emb_dim_pos) + (1, 1, emb_dim_pos)

    def _property_token_embeds(self, batch_size: int, reference: Tensor) -> Tensor:
        if self.property_token is None:
            raise RuntimeError("Explicit property token is not enabled.")
        token = self.property_token(self.property_token_id).to(
            device=reference.device,
            dtype=reference.dtype,
        )
        return token.unsqueeze(0).expand(batch_size, -1, -1)

    def _build_explicit_property_sequence(
            self,
            latent_embeds: Tensor,
            atom_embeds: Tensor = None,
        ) -> Tensor:
        """Build ``[z, PROPERTY, atoms...]`` with fixed positional ids."""
        reference = atom_embeds if atom_embeds is not None else latent_embeds
        property_embeds = self._property_token_embeds(
            latent_embeds.shape[0],
            reference,
        )
        sequence_parts = [latent_embeds, property_embeds]
        if atom_embeds is not None:
            sequence_parts.append(atom_embeds)
        input_embeds = torch.cat(sequence_parts, dim=1)
        positional_ids = torch.arange(
            input_embeds.shape[1],
            device=input_embeds.device,
        ).unsqueeze(0)
        return input_embeds + self.pte(positional_ids)

    @staticmethod
    def _cast_transformer_input_for_autocast(input_embeds: Tensor) -> Tensor:
        """Keep the assembled residual stream in the active autocast dtype."""
        device_type = input_embeds.device.type
        if torch.is_autocast_enabled(device_type):
            return input_embeds.to(torch.get_autocast_dtype(device_type))
        return input_embeds

    @staticmethod
    def _fold_padding_into_attention_mask(
            attention_mask: Tensor,
            padding_mask: Tensor,
            dtype: torch.dtype,
        ) -> Tensor:
        blocked = attention_mask | padding_mask[:, None, None, :]
        return torch.zeros_like(blocked, dtype=dtype).masked_fill_(
            blocked,
            float("-inf"),
        )

    def _sample_property_sources(
            self,
            batch_size: int,
            device: torch.device,
            requested_mode: str,
        ) -> Tensor:
        mode = requested_mode
        if mode is None:
            mode = "sample" if self.training else self.property_eval_mode
        if mode == "molecule":
            return torch.full(
                (batch_size,),
                self.PROPERTY_SOURCE_MOLECULE,
                dtype=torch.long,
                device=device,
            )
        if mode == "latent":
            return torch.full(
                (batch_size,),
                self.PROPERTY_SOURCE_LATENT,
                dtype=torch.long,
                device=device,
            )
        if mode == "joint":
            return torch.full(
                (batch_size,),
                self.PROPERTY_SOURCE_JOINT,
                dtype=torch.long,
                device=device,
            )
        if mode != "sample":
            raise ValueError(
                "property_attention_mode must be 'sample', 'molecule', "
                f"'latent', or 'joint'; received {mode!r}."
            )
        draws = torch.rand(batch_size, device=device)
        sources = torch.full(
            (batch_size,),
            self.PROPERTY_SOURCE_JOINT,
            dtype=torch.long,
            device=device,
        )
        molecule_cutoff = self.property_molecule_only_prob
        latent_cutoff = molecule_cutoff + self.property_latent_only_prob
        sources.masked_fill_(draws < molecule_cutoff, self.PROPERTY_SOURCE_MOLECULE)
        sources.masked_fill_(
            (draws >= molecule_cutoff) & (draws < latent_cutoff),
            self.PROPERTY_SOURCE_LATENT,
        )
        return sources

    def _property_attention_mask(
            self,
            property_sources: Tensor,
            num_atom_slots: int,
        ) -> Tensor:
        """
        Build per-example attention for [z, PROPERTY, atom_1, ..., atom_N, pad].

        - z attends itself (an all-masked row would yield an invalid softmax),
        - PROPERTY attends itself plus the selected source(s),
        - atom_i attends z, itself, and preceding atoms, never PROPERTY.

        The caller folds padding keys into this mask before attention.
        """
        batch_size = property_sources.shape[0]
        seq_len = num_atom_slots + 2
        mask = torch.ones(
            batch_size,
            seq_len,
            seq_len,
            dtype=torch.bool,
            device=property_sources.device,
        )

        # z has no preceding context, but must retain a valid self-attention key.
        mask[:, 0, 0] = False

        # PROPERTY always retains its learned self token.
        mask[:, 1, 1] = False
        molecule_rows = property_sources == self.PROPERTY_SOURCE_MOLECULE
        latent_rows = property_sources == self.PROPERTY_SOURCE_LATENT
        joint_rows = property_sources == self.PROPERTY_SOURCE_JOINT
        mask[molecule_rows, 1, 2:] = False
        mask[latent_rows, 1, 0] = False
        mask[joint_rows, 1, 0] = False
        mask[joint_rows, 1, 2:] = False

        # Atom hidden state i is teacher-forced and predicts atom i+1, so it
        # may attend its own input token in addition to z and earlier atoms.
        atom_causal_mask = torch.triu(
            torch.ones(
                num_atom_slots,
                num_atom_slots,
                dtype=torch.bool,
                device=property_sources.device,
            ),
            diagonal=1,
        )
        mask[:, 2:, 0] = False
        mask[:, 2:, 2:] = atom_causal_mask
        # mask[:, 2:, 1] remains True: atoms never see PROPERTY.

        # Keep a singleton head dimension.  SDPA broadcasts this mask across
        # heads, avoiding an n_heads-fold mask allocation for every batch.
        return mask[:, None]

    def _predict_latent_only_property(
            self,
            latent_embeds: Tensor,
            latent_embeds_raw: Tensor = None,
        ) -> Tensor:
        """Evaluate the trained latent-only ``[z, PROPERTY]`` path.

        ``latent_embeds`` is the decoder-width latent token after any latent
        projection. The two-token pass is equivalent to latent-only
        teacher-forcing because neither token can attend to atom states.
        """
        if self.context_dim <= 0 or not self.explicit_property_token:
            raise RuntimeError(
                "Latent property prediction requires context_dim > 0 and an "
                "explicit property token."
            )

        batch_size = latent_embeds.shape[0]
        property_input = self._cast_transformer_input_for_autocast(
            self._build_explicit_property_sequence(latent_embeds)
        )
        property_sources = self._sample_property_sources(
            batch_size,
            property_input.device,
            "latent",
        )
        property_hidden = self.one_step(
            input_embeds=property_input,
            latent_embeds=latent_embeds_raw,
            padding_mask=None,
            attention_mask=self._property_attention_mask(
                property_sources,
                num_atom_slots=0,
            ),
        )
        return self.context_decoder(property_hidden[:, 1, :])

    def _predict_molecule_latent_conditioned_property(
            self,
            latent_embeds: Tensor,
            atom_embeds: Tensor,
            atom_padding_mask: Tensor,
            latent_embeds_raw: Tensor = None,
        ) -> Tensor:
        """Predict from atoms whose hidden states are conditioned on encoder z."""
        predicted_property, _summary_features, _readout_features = (
            self._predict_molecule_latent_conditioned_property_with_features(
                latent_embeds,
                atom_embeds,
                atom_padding_mask,
                latent_embeds_raw,
            )
        )
        return predicted_property

    def _predict_molecule_latent_conditioned_property_with_features(
            self,
            latent_embeds: Tensor,
            atom_embeds: Tensor,
            atom_padding_mask: Tensor,
            latent_embeds_raw: Tensor = None,
        ) -> tuple[Tensor, Tensor, Tensor]:
        """Return molecule-path prediction and its ``(s, h)`` features.

        ``s`` is the property-token transformer state. ``h`` is the readout
        feature immediately before the final scalar output layer. The first
        return value is identical to
        :meth:`_predict_molecule_latent_conditioned_property`.
        """
        if self.property_token is None:
            raise RuntimeError("Explicit property token is not enabled.")
        if self.use_ada_ln_zero and latent_embeds_raw is None:
            raise ValueError("latent_embeds_raw is required for AdaLN conditioning.")

        batch_size, num_atom_slots = atom_embeds.shape[:2]
        property_input = self._build_explicit_property_sequence(
            latent_embeds, atom_embeds
        )
        property_padding_mask = torch.cat(
            [atom_padding_mask.new_zeros(batch_size, 2), atom_padding_mask],
            dim=1,
        )
        property_input = property_input.masked_fill(
            property_padding_mask.unsqueeze(-1),
            0.0,
        )
        property_input = self._cast_transformer_input_for_autocast(property_input)

        attention_mask = self._property_attention_mask(
            self._sample_property_sources(
                batch_size, property_input.device, "molecule"
            ),
            num_atom_slots,
        )
        attention_mask = self._fold_padding_into_attention_mask(
            attention_mask,
            property_padding_mask,
            property_input.dtype,
        )

        property_hidden = self.one_step(
            input_embeds=property_input,
            latent_embeds=(latent_embeds_raw if self.use_ada_ln_zero else None),
            padding_mask=None,
            attention_mask=attention_mask,
        )
        summary_features = property_hidden[:, 1, :]
        readout_features = self.context_decoder.forward_features(summary_features)
        if self.context_decoder.linear2 is None:
            predicted_property = readout_features
        else:
            predicted_property = self.context_decoder.linear2(readout_features)
        return predicted_property, summary_features, readout_features

    def _predict_molecule_property(
            self,
            latent_embeds: Tensor,
            position: Tensor,
            x: Tensor,
            padding_mask: Tensor,
            position_padding_mask: Tensor = None,
        ) -> Tensor:
        """Predict from a molecule and its encoder latent.

        ``padding_mask`` may be atom-only ``(B, N)`` or z-prefixed ``(B, N+1)``.
        """
        predicted_property, _summary_features, _readout_features = (
            self._predict_molecule_property_with_features(
                latent_embeds=latent_embeds,
                position=position,
                x=x,
                padding_mask=padding_mask,
                position_padding_mask=position_padding_mask,
            )
        )
        return predicted_property

    def _predict_molecule_property_with_features(
            self,
            latent_embeds: Tensor,
            position: Tensor,
            x: Tensor,
            padding_mask: Tensor,
            position_padding_mask: Tensor = None,
        ) -> tuple[Tensor, Tensor, Tensor]:
        """Return molecule-path prediction and its ``(s, h)`` features."""
        
        latent_embeds_raw = latent_embeds if self.use_ada_ln_zero else None
        if self.latent_encoder is not None:
            latent_embeds = self.latent_encoder(latent_embeds)

        atom_padding_mask = padding_mask[:, -x.shape[1]:].to(
            device=x.device, dtype=torch.bool
        )

        if position_padding_mask is None:
            position_padding_mask = atom_padding_mask.clone()
            position_padding_mask |= x[:, :, 0].eq(self.eos_token_id)
        else:
            position_padding_mask = position_padding_mask.to(
                device=position.device, dtype=torch.bool
            )

        position_embeds = self._embed_positions(position / self.position_scale)
        position_embeds.masked_fill_(
            position_padding_mask.unsqueeze(-1),
            0.0,
        )
        atom_embeds = self.mixing(
            torch.cat([position_embeds, self._embed_atom_features(x)], dim=-1)
        )
        return self._predict_molecule_latent_conditioned_property_with_features(
            latent_embeds,
            atom_embeds,
            atom_padding_mask,
            latent_embeds_raw,
        )

    def forward(
            self, 
            latent_embeds:Tensor, 
            position:Tensor,    
            x:Tensor, 
            padding_mask:Tensor,   
            position_padding_mask: Tensor = None,
            **kwargs
        ):
        """ 
        INPUTS:  (max_n_nodes have the eos token counted)
        - latent_embeds     : (bz, 1, emb_dim) 
        - position          : (b, max_n_nodes, 3)  
        - x                 : (b, max_n_nodes, d_x)
        - charge            : (b,) 
        - spin_multiplicity : (b,)
        - number_of_rings   : (b,)
        - has_fused_rings   : (b,)
        - padding_mask      : (b, 1+max_n_nodes)
            if False, then valid entries, else True, then padding 
            EOS token is valid, so set as False
            +1 for the latent embedding entry
        - position_padding_mask : (b, max_n_nodes)
            if False, then valid entries, else True, then padding 
            EOS token is non-valid, so set as True
        
        RETURNS: (all have the eos token included)
        - pred_positions  : (bz, max_n_nodes, 3)  
        - pred_x          : (bz, max_n_nodes, n_atom_types+1) 
        - pred_charge     : (bz, num_charges)
        - pred_spin       : (bz, num_spin_multiplicities)
        - pred_number_of_rings : (bz, num_number_of_rings)
        - pred_has_fused_rings : (bz, num_has_fused_rings)
        - y                   : (bz, context_dim) optional
        - y_source            : (bz,) optional integer source ids:
                                0=molecule, 1=latent, 2=joint
        """  
         
        bz = x.shape[0]
        
        (
            pred_charge_logits,
            pred_spinmp_logits,
            pred_number_of_rings_logits,
            pred_has_fused_rings_logits, 
        ) = self._decode_molecule_logits(latent_embeds)
        
        pred_y = None
        pred_y_source = None
        
        # Construct input embedding -- the padding part should have the same embeddings
        latent_embeds_raw = latent_embeds.clone() if self.use_ada_ln_zero else None # (bz, 1, latent_dim)
        if self.latent_encoder is not None: 
            latent_embeds = self.latent_encoder(latent_embeds)      # (bz, 1, emb_dim) 
        
        bz, max_n_nodes, d_x = x.shape
        assert self.num_atom_features == d_x, \
            f"self.num_atom_features must be equal to d_x from data; received num_atom_features: {self.num_atom_features}, d_x: {d_x}" 
        
        # Construct atom features embedding
        x_emb = self._embed_atom_features(x)                        #  (bz, max_n_nodes, d_x*emb_dim_each_x) 
        position_scaled = position.clone() / self.position_scale
        pos_emb = self._embed_positions(position_scaled)              # (bz, max_n_nodes, emb_dim_pos)
        
        if position_padding_mask is None:
            # Backward compatibility for molecule-only callers that predate
            # the separate position mask. ``padding_mask`` includes the
            # latent prefix and keeps EOS attention-valid, while positions
            # must exclude both EOS and padded atom rows.
            position_padding_mask = padding_mask[:, -max_n_nodes:].clone()
            position_padding_mask |= x[:, :, 0].eq(self.eos_token_id)

        pos_emb = pos_emb.masked_fill(position_padding_mask.unsqueeze(-1), 0.0)
        # Construct all embedding
        input_embeds = torch.concat([pos_emb, x_emb], dim=-1)             # (bz, max_n_nodes, emb_dim) 
        
        # Mix the atom/position embedding.
        atom_input_embeds = self.mixing(input_embeds)              # (bz, max_n_nodes, emb_dim)

        property_attention_mask = None
        if self.context_dim > 0 and self.explicit_property_token:
            # Explicit, fixed-layout property sequence:
            # [z, PROPERTY, atom_1, ..., atom_N, padding, ..., padding].
            input_embeds = self._build_explicit_property_sequence(
                latent_embeds,
                atom_input_embeds,
            )
            transformer_padding_mask = torch.cat(
                [
                    padding_mask[:, :1],
                    torch.zeros(
                        bz,
                        1,
                        dtype=torch.bool,
                        device=padding_mask.device,
                    ),
                    padding_mask[:, 1:],
                ],
                dim=1,
            )
            input_embeds = input_embeds.masked_fill(
                transformer_padding_mask.unsqueeze(-1),
                0.0,
            )
            pred_y_source = self._sample_property_sources(
                bz,
                input_embeds.device,
                kwargs.get("property_attention_mode"),
            )
            property_attention_mask = self._property_attention_mask(
                pred_y_source,
                max_n_nodes,
            )
        else:
            # Legacy sequence, retained for old checkpoints and context-free
            # models: [z, atom_1, ..., atom_N, padding, ..., padding].
            input_embeds = torch.cat(
                [latent_embeds, atom_input_embeds],
                dim=1,
            )
            positional_ids = torch.arange(
                0,
                1 + max_n_nodes,
                device=input_embeds.device,
            ).unsqueeze(0)
            input_embeds = input_embeds + self.pte(positional_ids)
            transformer_padding_mask = padding_mask
            input_embeds = input_embeds.masked_fill(
                transformer_padding_mask.unsqueeze(-1),
                0.0,
            )

        # Mixed precision does not automatically cast embedding lookups.  Cast
        # the assembled residual stream once so bf16-mixed training does not
        # retain float32 transformer activations after positional addition.
        input_embeds = self._cast_transformer_input_for_autocast(input_embeds)

        attention_padding_mask = transformer_padding_mask
        if property_attention_mask is not None:
            # Fold padding into the compact source mask once instead of asking
            # every transformer layer to canonicalize and merge two masks.
            property_attention_mask = self._fold_padding_into_attention_mask(
                property_attention_mask,
                transformer_padding_mask,
                input_embeds.dtype,
            )
            attention_padding_mask = None
        # Transformer -- Shift so that tokens < n predict n
        hidden_states = self.one_step(
            input_embeds=input_embeds,
            latent_embeds=latent_embeds_raw,                      # (bz, 1, latent_dim)
            padding_mask=attention_padding_mask,
            attention_mask=property_attention_mask,
        )
        
        if self.context_dim > 0:
            if self.explicit_property_token:
                pred_y = self.context_decoder(hidden_states[:, 1, :])
            else:
                pred_y = self.context_decoder(hidden_states[:,-1,:]) # (bz, context_dim)
                pred_y_source = torch.full(
                    (bz,),
                    self.PROPERTY_SOURCE_JOINT,
                    dtype=torch.long,
                    device=hidden_states.device,
                )

        # Readout the x logits, positions
        if self.context_dim > 0 and self.explicit_property_token:
            # z predicts atom_1; atom_i predicts atom_{i+1}.  PROPERTY is not
            # part of the autoregressive reconstruction stream.
            reconstruction_hidden_states = torch.cat(
                [hidden_states[:, :1], hidden_states[:, 2:-1]],
                dim=1,
            )
        else:
            reconstruction_hidden_states = hidden_states[:, :-1]
        hidden_states_pos = reconstruction_hidden_states[...,:self.emb_dim_pos]  # (bz, max_n_nodes, emb_dim_pos)
        hidden_states_seq = reconstruction_hidden_states[...,self.emb_dim_pos:].view(
            bz, max_n_nodes, d_x, -1)                             # (bz, max_n_nodes, d_x, emb_dim_each_x)
         
        pred_seq_logits = self.atomseq_decoder(hidden_states_seq)   # (bz, max_n_nodes, d_x, vocab_size)
        pred_pos        = self.atompos_decoder(hidden_states_pos)   # (bz, max_n_nodes, 3)             
    
          
        del hidden_states, input_embeds, hidden_states_pos, hidden_states_seq
        return (
            pred_pos,  
            pred_seq_logits, 
            pred_charge_logits, 
            pred_spinmp_logits, 
            pred_number_of_rings_logits, 
            pred_has_fused_rings_logits, 
            pred_y,
            pred_y_source,
        )
         
    def construct_node_mask(self, n_nodes:Tensor, max_n_nodes:int, device:torch.device=None): 
        # n_nodes: shape (bz,)
        if device is None:
            device = torch.device("cuda")

        arange = torch.arange(max_n_nodes, device=device).unsqueeze(0)  # shape (1, max_n_nodes)
        n_nodes_expanded = n_nodes.unsqueeze(1)                         # shape (bz, 1)
        node_mask = (arange < n_nodes_expanded).long()                  # shape (bz, max_n_nodes)
        return node_mask

    def generate_masks(self, seq_len:int, is_causal:bool=True, device:torch.device=None):
        if device is None:
            device = torch.device("cuda")
        if is_causal: 
            # Construct the causal attention_mask 
            attention_mask = torch.nn.Transformer.generate_square_subsequent_mask(
                sz=seq_len, device=device, dtype=torch.bool) 
        else: 
            attention_mask = None 
        
        # torch.cuda.empty_cache()
        return attention_mask 

    def one_step(
            self,
            input_embeds: Tensor, 
            latent_embeds: Tensor=None,
            padding_mask: Tensor=None, 
            is_causal: bool=True,
            attention_mask: Tensor=None,
        ):
        """
        input_embeds: (bz, n+1, emb_dim) float
        latent_embeds: (bz, 1, latent_dim) float
        padding_mask: (bz, n+1)  
        node_mask: (bz, n) float
            if 1 valid entries, else 0 padding
        padding_mask: (bz, n+1) bool
            if 0 valid entries, else 1 padding
        """
        # Construct position embedding
        seq_length = input_embeds.shape[1] # which is 1+n_nodes
        if attention_mask is None:
            attention_mask = self.generate_masks(
                seq_length, is_causal, device=input_embeds.device
            )
        elif attention_mask.dim() == 2:
            if attention_mask.shape != (seq_length, seq_length):
                raise ValueError(
                    "2D attention_mask must have shape "
                    f"({seq_length}, {seq_length}); received "
                    f"{tuple(attention_mask.shape)}."
                )
        elif attention_mask.dim() == 3:
            expected = (input_embeds.shape[0] * self.n_heads, seq_length, seq_length)
            if attention_mask.shape != expected:
                raise ValueError(
                    f"3D attention_mask must have shape {expected}; received "
                    f"{tuple(attention_mask.shape)}."
                )
        elif attention_mask.dim() == 4:
            batch_and_sequence_match = (
                attention_mask.shape[0] == input_embeds.shape[0]
                and attention_mask.shape[2:] == (seq_length, seq_length)
            )
            if (
                not batch_and_sequence_match
                or attention_mask.shape[1] not in (1, self.n_heads)
            ):
                raise ValueError(
                    "4D attention_mask must have shape "
                    f"(batch, 1|{self.n_heads}, {seq_length}, {seq_length}); "
                    f"received {tuple(attention_mask.shape)}."
                )
        else:
            raise ValueError(
                "attention_mask must be 2D, 3D, or 4D; received "
                f"{attention_mask.dim()} dimensions."
            )
        # Transformer  
        return self.decoder(
            x=input_embeds, 
            c=latent_embeds,
            attn_mask=attention_mask, 
            key_padding_mask=padding_mask
        )[0]  # (bz, n_nodes+1, emb_dim)

    def one_step_cache(
            self,
            input_embeds: Tensor, 
            latent_embeds: Tensor=None,
            kv_caches: dict|None=None
        ):
        """
        input_embeds: (bz, 1, emb_dim) float
        kv_caches: (bz, H, t, Dh) float
            if None, then no cache
        """
        # Transformer  
        return self.decoder(
            x=input_embeds,  
            c=latent_embeds,
            kv_caches=kv_caches,
            use_cache=True
        )  # (bz, 1, emb_dim)

    @torch.no_grad()
    def generate(self, latent_embeds: Tensor): 
        return self.generate_cache(latent_embeds)  

    
    @torch.inference_mode()
    def generate_cache(self, latent_embeds:Tensor):  
        """
        latent_embeds: (bz, 1, emb_dim) 
        """
        self.eval()
        # predict molecule-level features
        pred_charge_logits = self.charge_decoder(latent_embeds)
        generated_charge = sample_from_logits(
            pred_charge_logits.squeeze(1),
            temperature=self.temperature,
            top_k=self.top_k,
        ) 
        pred_spinmp_logits = self.spinmp_decoder(latent_embeds)
        generated_spinmp = sample_from_logits(
            pred_spinmp_logits.squeeze(1),
            temperature=self.temperature,
            top_k=self.top_k,
        ) 
        pred_number_of_rings_logits = self.number_of_rings_decoder(latent_embeds)
        generated_number_of_rings = sample_from_logits(
            pred_number_of_rings_logits.squeeze(1),
            temperature=self.temperature,
            top_k=self.top_k,
        ) 
        pred_has_fused_rings_logits = self.has_fused_rings_decoder(latent_embeds)
        generated_has_fused_rings = sample_from_logits(
            pred_has_fused_rings_logits.squeeze(1),
            temperature=self.temperature,
            top_k=self.top_k,
        )
        
        
        bz = latent_embeds.shape[0]
        latent_embeds_raw = latent_embeds.clone() if self.use_ada_ln_zero else None
        # map the latent embedding to the transformer input embedding space
        if self.latent_encoder is not None:
            latent_embeds = self.latent_encoder(latent_embeds)


        generated_x = torch.empty((bz, 0, self.num_atom_features)).to(latent_embeds.device).to(torch.int64)
        generated_pos = torch.empty((bz, 0, 3)).to(latent_embeds)

        # tracking which sequences have finished
        finished = torch.zeros((bz, 1), dtype=torch.bool).to(latent_embeds.device)
        
        input_embeds = latent_embeds.clone()           # (bz, 1, latent_dim) 
        
        kv_caches = None
        for i in range(self.max_n_nodes):
            # With the explicit training layout [z, PROPERTY, atoms...], atom
            # positional ids start at 2.  PROPERTY is absent from the generation
            # cache (atoms cannot attend it anyway), but generated atom inputs
            # retain those same learned positional ids.
            positional_id = i
            if self.context_dim > 0 and self.explicit_property_token and i > 0:
                positional_id += 1
            positional_ids = torch.tensor(
                [[positional_id]],
                device=input_embeds.device,
                dtype=torch.long,
            )
            positional_embeds = self.pte(positional_ids)
            input_embeds += positional_embeds

            hidden_states, kv_caches = self.one_step_cache(
                input_embeds,
                latent_embeds=latent_embeds_raw,
                kv_caches=kv_caches,
            )
            hidden_states_pos = hidden_states[..., :self.emb_dim_pos]
            hidden_states_x = hidden_states[..., self.emb_dim_pos:].view(
                bz, 1, self.num_atom_features, -1
            )
            
            pred_pos = self.atompos_decoder(hidden_states_pos)
            pred_x_logits = self.atomseq_decoder(hidden_states_x) 

            pred_x = sample_from_logits(
                pred_x_logits.squeeze(1),
                temperature=self.temperature,
                top_k=self.top_k,
            )
            del pred_x_logits

            pred_x = torch.where(
                finished,
                torch.full_like(pred_x, self.pad_token_id),
                pred_x,
            ).unsqueeze(1)
            just_eos = (pred_x[:, 0, :1] == self.eos_token_id)
            # suppress_position means “should this step have no geometry?”
            suppress_position = finished | just_eos
            pred_pos = torch.where(
                suppress_position.unsqueeze(-1),
                torch.zeros_like(pred_pos),
                pred_pos,
            )

            
            generated_pos = torch.cat((generated_pos, pred_pos), dim=1)
            generated_x = torch.cat((generated_x, pred_x), dim=1)
            
            # update finished sequences: including EOS generated on this step
            finished = finished | just_eos

            if finished.all():
                break
            
            new_pos_embeds = self._embed_positions(pred_pos)            # (bz, 1, emb_dim_pos) 
            new_x_embeds = self._embed_atom_features(pred_x)            # (bz, 1, d_x*emb_dim_each_x)
            input_embeds = torch.concat([new_pos_embeds, new_x_embeds], dim=-1)
            input_embeds = self.mixing(input_embeds)
            input_embeds = input_embeds.masked_fill(finished.unsqueeze(-1), 0.0)

        generated_pos = generated_pos * self.position_scale

        return (
            generated_pos,
            generated_x,
            generated_charge,
            generated_spinmp,
            generated_number_of_rings,
            generated_has_fused_rings
        )
        


class Cloud3DDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__() 
        self.model = CloudARDecoder(
            latent_dim         = cfg.latent_dim,
            emb_dim            = cfg.decoder_embed_dim,  
            num_atoms          = cfg.num_atoms,
            num_atom_features  = cfg.num_atom_features,
            max_n_nodes        = cfg.max_positions, 
            n_heads            = cfg.decoder_n_heads, 
            depth              = cfg.decoder_n_layers, 
            dim_feedforward    = cfg.decoder_dim_feedforward, 
            dropout            = cfg.dropout, 
            attention_dropout  = cfg.decoder_attention_dropout if hasattr(cfg, "decoder_attention_dropout") else 0.0,
            activation_dropout = cfg.decoder_act_dropout if hasattr(cfg, "decoder_act_dropout") else 0.0,
            droppath_prob      = cfg.decoder_droppath_prob if hasattr(cfg, "decoder_droppath_prob") else 0.0,
            disable_activation_norm = cfg.disable_activation_norm if hasattr(cfg, "disable_activation_norm") else False,
            activation         = cfg.activation if hasattr(cfg, "activation") else "gelu",  
            temperature        = cfg.decoder_atomseq_temp if hasattr(cfg, "decoder_atomseq_temp") else 1.,
            top_k              = cfg.decoder_atomseq_topk if hasattr(cfg, "decoder_atomseq_topk") else 1,
            norm_first         = cfg.norm_first if hasattr(cfg, "norm_first") else True,
            layer_norm_eps     = cfg.layer_norm_eps if hasattr(cfg, "layer_norm_eps") else 1e-5,
            width_input_alpha  = cfg.width_input_alpha if hasattr(cfg, "width_input_alpha") else 1.0,
            width_output_alpha = cfg.width_output_alpha if hasattr(cfg, "width_output_alpha") else 1.0,
            width_multiplier   = cfg.decoder_width_multiplier if hasattr(cfg, "decoder_width_multiplier") else 1.0,
            depth_multiplier   = cfg.decoder_depth_multiplier if hasattr(cfg, "decoder_depth_multiplier") else 1.0,
            depth_alpha_exp    = cfg.depth_alpha_exp if hasattr(cfg, "depth_alpha_exp") else 1.0,
            width_disable_attention_scaling = cfg.width_disable_attention_scaling if hasattr(cfg, "width_disable_attention_scaling") else False,
            standparam         = cfg.standparam if hasattr(cfg, "standparam") else False,
            init_std           = cfg.init_std if hasattr(cfg, "init_std") else 0.02, 
            context_dim         = cfg.context_dim if hasattr(cfg, "context_dim") else 0,
            use_ada_ln_zero     = cfg.decoder_use_ada_ln_zero if hasattr(cfg, "decoder_use_ada_ln_zero_corrected") else False,
            position_scale      = cfg.position_scale if hasattr(cfg, "position_scale") else 1.0,
            pos_readout_depth   = cfg.decoder_pos_readout_depth if hasattr(cfg, "decoder_pos_readout_depth") else 2,
            explicit_property_token = getattr(cfg, "decoder_explicit_property_token", False),
            property_molecule_only_prob = getattr(cfg, "decoder_property_molecule_only_prob", 0.0),
            property_latent_only_prob = getattr(cfg, "decoder_property_latent_only_prob", 0.0),
            property_eval_mode = getattr(cfg, "decoder_property_eval_mode", "sample"),
        ) 
    
    def forward(self, latent_embeds: Tensor, **kwargs):
        """ 
        - latent_embeds : (bz, 1, latent_dim)
            latent embedding
        - pos               (b, max_n_nodes+1, 3)
        - x                 (b, max_n_nodes+3) 
        - charge            (b,)
        - spin_multiplicity (b,)

        RETURNS 
        - pred_position      : (bz, max_n_nodes+1, 3)
        - pred_x_logits      : (bz, max_n_nodes+1, n_atom_types+1)
        - pred_charge_logits : (bz, 1, num_charges)
        - pred_spin_logits   : (bz, 1, num_spin_multiplicities)
        """
        
        return self.model(latent_embeds, **kwargs)
         
