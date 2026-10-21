from __future__ import annotations

import math


import torch
from torch import nn 
import torch.nn.functional as F
from torch import BoolTensor, LongTensor, Tensor


from eftalfm.equivariant_free_generative_models.vae.utils import sample_per_molecule_corruption_mask

from eftalfm.data.pcqm4m_v2.compact_atom_features import (
    COMPACT_ATOM_FEATURE_CARDINALITIES,
    COMPACT_ATOM_FIRST_FEATURE_CARDINALITY,
    COMPACT_ATOM_OTHER_FEATURE_CARDINALITY,
    COMPACT_ATOM_INITIAL_SHIFT_WITH_EOS,
)


REC_METRIC_PREFIXES = ("REC_Loss", "REC_MAE")


def _compact_atom_corruption_metadata(
    num_features: int,
    feature_cardinalities = COMPACT_ATOM_FEATURE_CARDINALITIES,
) -> tuple[LongTensor, LongTensor, int]:
    """Build packed-token corruption metadata once during initialization."""
    if num_features < 1:
        raise ValueError(f"num_features must be positive; received {num_features}.")

    cardinalities = tuple(
        int(value) for value in feature_cardinalities[:num_features]
    )
    reserved_cardinalities = (
        COMPACT_ATOM_FIRST_FEATURE_CARDINALITY,
    ) + (COMPACT_ATOM_OTHER_FEATURE_CARDINALITY,) * (num_features - 1)
    invalid_cardinality = any(
        cardinality < 2
        or (feature_idx == 0 and cardinality > reserved_cardinalities[0])
        or (
            0 < feature_idx < num_features - 1
            and cardinality > reserved_cardinalities[feature_idx]
        )
        for feature_idx, cardinality in enumerate(cardinalities)
    )
    if invalid_cardinality:
        raise ValueError(
            "Atom-feature cardinalities must be at least 2; the atomic feature "
            "and every non-final extra feature must fit their token blocks; "
            f"received {cardinalities} with reserved widths "
            f"{reserved_cardinalities}."
        )

    extra_lower_bounds = tuple(
        COMPACT_ATOM_INITIAL_SHIFT_WITH_EOS
        + COMPACT_ATOM_FIRST_FEATURE_CARDINALITY
        + feature_idx * COMPACT_ATOM_OTHER_FEATURE_CARDINALITY
        for feature_idx in range(num_features - 1)
    )
    lower_bounds = torch.tensor(
        (COMPACT_ATOM_INITIAL_SHIFT_WITH_EOS,) + extra_lower_bounds,
        dtype=torch.long,
    )
    replacement_counts_tuple = tuple(value - 1 for value in cardinalities)
    replacement_counts = torch.tensor(replacement_counts_tuple, dtype=torch.long)
    rng_period = math.lcm(*replacement_counts_tuple)
    return lower_bounds, replacement_counts, rng_period


def _sample_compact_atom_feature_replacements(
    atom_features: LongTensor,
    lower_bounds: LongTensor,
    replacement_counts: LongTensor,
    rng_period: int,
) -> LongTensor:
    """Sample a different valid packed token for every selected feature.

    The valid interval for each column is ``[lower_bound, lower_bound + true
    cardinality)`` rather than the entire reserved embedding block. The caller
    supplies cached metadata, so the training hot path performs no range
    construction, floating-point sampling, or dtype validation.
    """
    draws = torch.empty_like(atom_features).random_(rng_period)
    if atom_features.shape[-1] > 1:
        # rng_period is the LCM of all replacement counts, making the modulo
        # result exactly uniform for every column, not merely approximately so.
        draws.remainder_(replacement_counts)

    # A rank in [0, cardinality - 1) is mapped around the original token,
    # producing an exact uniform draw over every other value in the column.
    draws.add_(draws >= atom_features - lower_bounds)
    draws.add_(lower_bounds)
    return draws


def vicreg_var_cov(z: torch.Tensor, eps: float = 1e-6):
    """Minimal VICReg-style regularizer (variance + covariance).

    z: (b, d) ideally roughly standardized.
    Returns: (var_loss, cov_loss) scalars.
    """
    z = z.float()
    b, d = z.shape

    # ----- variance term -----
    # VICReg uses a hinge: penalize dims whose std < 1
    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)          # (d,)
    var_loss = F.relu(1.0 - std).mean()

    # ----- covariance term -----
    if b <= 1:
        cov_loss = torch.zeros((), device=z.device, dtype=z.dtype)
        return var_loss, cov_loss
    # center first, then compute (d x d) covariance
    zc = z - z.mean(dim=0, keepdim=True)                          # (b, d)
    cov = (zc.T @ zc) / (b - 1)                                   # (d, d)

    # off-diagonal squared Frobenius norm
    off_diag = cov - torch.diag(torch.diag(cov))
    cov_loss = (off_diag ** 2).sum() / d                          # scale by d (simple normalization)

    return var_loss, cov_loss


def mu_var_cov_regularizer(z_mu: torch.Tensor, var_floor: float = 0.0):
    """Softly encourages broader, less correlated posterior means across dims.

    mu_var_d relates to active dimensions as it measures whether 
    a latent dimension's posterior mean changes across samples.
    If a dimension is actually carrying information, then different molecules 
    should map to different values of that dimension's z_mu, so across the batch:
    - z_mu[:, d] varies
    - var(z_mu[:, d]) is nontrivial
    
    If a dimension is inactive, then the encoder mostly ignores it and 
    leaves it near the prior mean for every sample, so:
    - z_mu[:, d] ≈ 0 for almost all samples
    - var(z_mu[:, d]) is small

    That is why mu_var_d is a direct proxy for "is this dimension being used 
    to encode sample-dependent information?"
    
    This is a one-sided penalty:
    - no penalty for sufficiently active dimensions
    - positive penalty for under-active dimensions
    That makes it a “variance floor” rather than a target-matching penalty. 
    It pushes dead or weak dimensions up, but it does not push already-active dimensions back down.
    
    With var_floor=0.09, the implied standard-deviation floor is about:
    - sqrt(0.09) = 0.3
    So this regularizer is basically encouraging:
    - std(z_mu[:, d]) >= 0.3 for each dimension, softly.
    """
    z_mu = z_mu.float()
    b, d = z_mu.shape
    zero = torch.zeros((), device=z_mu.device, dtype=z_mu.dtype)

    var_floor = max(float(var_floor), 0.0)
    mu_var_d = z_mu.var(dim=0, unbiased=False)
    var_floor_loss = F.relu(mu_var_d.new_tensor(var_floor) - mu_var_d).mean() if var_floor > 0.0 else zero

    if b <= 1:
        cov_loss = zero
    else:
        zc = z_mu - z_mu.mean(dim=0, keepdim=True)
        cov = (zc.T @ zc) / (b - 1)
        off_diag = cov - torch.diag(torch.diag(cov))
        cov_loss = (off_diag ** 2).sum() / d

    return var_floor_loss, cov_loss


class CloudVAEModelTransformerP(nn.Module):
    def __init__(
        self,
        encoder            : nn.Module,
        decoder            : nn.Module, 
        context_idx        : str="*",
        context_name       : str="y",
        kl_aggregation     : str="sum", 
        pos_loss_type      : str="l2", 
        context_loss_type  : str="l2",
        noise_scale        : float=0.1,
        corruption_ratio   : float=0.0,
        corruption_pos_only: bool=False,
        corruption_type    : str="mask", 
        free_bits_per_dim  : float=0.0,
        sigma_clamp_min    : float=0.0,
        sigma_clamp_max    : float=0.0,
        mu_var_floor       : float=0.09,
        mu_var_floor_weight: float=0.0,
        mu_cov_weight      : float=0.0, 
        standparam         : bool=False,  
        mol_extra_features : list[str] | None = None,
        atom_extra_features : list[str] | None = None,
        position_scale: float = 1.0,  
        property_molecule_only_loss_weight: float = 1.0,
        property_latent_only_loss_weight: float = 1.0,
        property_joint_loss_weight: float = 1.0,  
        **kwargs
    ):
        super().__init__()
        if mol_extra_features is None:
            mol_extra_features = [
                'charge',
                'spinmp',
                'number_of_rings',
                'has_fused_rings',
            ]
        if atom_extra_features is None:
            atom_extra_features = [
                'chiral', 
                'hydrad', 
                'charge', 
                'radical', 
                'aromatic', 
                'is_in_ring', 
                'conjugated', 
                'smallest_ring_size', 
                'num_Hs'
            ]
        self.encoder   = encoder
        self.decoder   = decoder 
        self.context_name = context_name
        encoder_model = getattr(self.encoder, "model", self.encoder)
        decoder_model = getattr(self.decoder, "model", self.decoder)
        encoder_particle_feature = getattr(encoder_model, "particle_feature", None)
        encoder_context_dim = int(getattr(encoder_particle_feature, "context_dim", 0))
        decoder_context_dim = int(getattr(decoder_model, "context_dim", 0))
        if encoder_context_dim not in (0, decoder_context_dim):
            raise ValueError(
                "Encoder context_dim must be zero or match decoder context_dim; "
                f"received encoder={encoder_context_dim}, decoder={decoder_context_dim}."
            )
        self.encoder_context_dim = encoder_context_dim
        model_context_dim = decoder_context_dim
        if context_idx is None or (isinstance(context_idx, str) and context_idx == "*"):
            self.context_idx = None
            self.context_dim = model_context_dim
        else:
            if isinstance(context_idx, str):
                context_idx = [int(idx) for idx in context_idx.split(",") if idx != ""]
            elif isinstance(context_idx, int):
                context_idx = [context_idx]
            elif torch.is_tensor(context_idx):
                context_idx = context_idx.flatten().tolist()
            else:
                context_idx = list(context_idx)
            self.context_idx = torch.tensor(context_idx, dtype=torch.long)
            self.context_dim = len(context_idx)
        if self.context_dim != model_context_dim:
            raise ValueError(
                "Property context selection does not match the model context_dim; "
                f"selected {self.context_dim} values but decoder expects {model_context_dim}."
            )
        if self.context_dim <= 1:
            self.context_metric_suffixes = ()
        elif self.context_idx is None:
            assert self.context_dim == 0, "context_dim must be 0 when context_idx is None"
            # self.context_metric_suffixes = tuple(
            #     f"dim_{dim}" for dim in range(self.context_dim)
            # )
        else:
            self.context_metric_suffixes = tuple(
                f"idx_{int(idx)}" for idx in self.context_idx.tolist()
            )
        print(
            "CloudVAE context dimensions (encoder, decoder): "
            f"({self.encoder_context_dim}, {self.context_dim})"
        )
        self.kl_aggregation = kl_aggregation 
        self.standparam = standparam  
        self.free_bits_per_dim = max(float(free_bits_per_dim), 0.0)
        self.sigma_clamp_min = max(float(sigma_clamp_min), 0.0)
        self.sigma_clamp_max = max(float(sigma_clamp_max), 0.0)
        self.mu_var_floor = max(float(mu_var_floor), 0.0)
        self.mu_var_floor_weight = max(float(mu_var_floor_weight), 0.0)
        self.mu_cov_weight = max(float(mu_cov_weight), 0.0)
        if self.sigma_clamp_max > 0.0 and self.sigma_clamp_max < self.sigma_clamp_min:
            raise ValueError(
                f"sigma_clamp_max ({self.sigma_clamp_max}) must be >= sigma_clamp_min ({self.sigma_clamp_min})"
            )
        self.noise_scale = noise_scale
        self.corruption_ratio = float(corruption_ratio)
        self.corruption_pos_only = bool(corruption_pos_only)
        self.corruption_type = corruption_type
        if self.corruption_type not in ["mask", "noise"]:
            raise ValueError(
                f"corruption_type must be 'mask' or 'noise'; received {self.corruption_type}."
            )
        
        decoder_vocab_size = getattr(decoder_model, "vocab_size", None)
        if decoder_vocab_size is None: 
            raise ValueError(
                "Decoder model must expose `vocab_size` when vae_corruption_ratio > 0 "
                "so the VAE can reserve a decoder-only atom mask token."
            )
        decoder_vocab_size = int(decoder_vocab_size)
        if self.corruption_ratio > 0 and decoder_vocab_size <= 1:
            raise ValueError(
                f"Decoder vocab_size must be > 1 when vae_corruption_ratio > 0; got {decoder_vocab_size}."
            )

        
        if self.corruption_ratio > 0 and self.corruption_type == "mask":
            self.atom_mask_position = nn.Parameter(torch.full((3,), -1.0))
            self.atom_mask_token_id = max(decoder_vocab_size - 1, 0)
         
        print(f"CloundVAE Using free_bits_per_dim: {self.free_bits_per_dim}")
        print(f"CloundVAE Using sigma_clamp_min: {self.sigma_clamp_min}")
        print(f"CloundVAE Using sigma_clamp_max: {self.sigma_clamp_max}")
        print(f"CloundVAE Using mu_var_floor: {self.mu_var_floor}")
        print(f"CloundVAE Using mu_var_floor_weight: {self.mu_var_floor_weight}")
        print(f"CloundVAE Using mu_cov_weight: {self.mu_cov_weight}") 
        print(f"CloundVAE Using noise_scale (corruption): {self.noise_scale}")
        print(f"CloundVAE Using corruption_ratio: {self.corruption_ratio}")
        print(f"CloundVAE Using corruption_pos_only: {self.corruption_pos_only}")
        print(f"CloundVAE Using corruption_type: {self.corruption_type}")
        
        self.loss_fct_mol  = torch.nn.CrossEntropyLoss(reduction='none')  
         
        self.property_source_loss_weights = (
            float(property_molecule_only_loss_weight),
            float(property_latent_only_loss_weight),
            float(property_joint_loss_weight),
        )
        if any(weight < 0.0 for weight in self.property_source_loss_weights):
            raise ValueError(
                "Property source loss weights must be non-negative; received "
                f"{self.property_source_loss_weights}."
            )
        
        print(
            "CloundVAE Using property source loss weights "
            "(molecule, latent, joint): "
            f"{self.property_source_loss_weights}"
        )  
        
        if pos_loss_type.split('-')[0] == "l2":
            self.loss_fct_cts  = torch.nn.MSELoss(reduction='none')
        elif pos_loss_type.split('-')[0] == "smoothl1": 
            beta = float(pos_loss_type.split('-')[1]) if len(pos_loss_type.split('-')) > 1 else 0.5
            self.loss_fct_cts  = torch.nn.SmoothL1Loss(reduction='none', beta=beta)
        else:
            raise ValueError(f"pos_loss_type only accept 'l2', 'smoothl1' and 'smoothl1-0.5'. Received {pos_loss_type}")
            
       
        if pos_loss_type.split('-')[0] == "l2":
            self.position_scale_out = position_scale ** 2
        elif pos_loss_type.split('-')[0] == "smoothl1": 
            self.position_scale_out = position_scale
        self.position_scale = position_scale
        
        
        self.loss_fct_cts_eval = torch.nn.L1Loss(reduction='none')
        self.context_loss_type = context_loss_type
        if self.context_dim > 0:
            if context_loss_type.split('-')[0] == "l2":
                self.loss_fct_context = torch.nn.MSELoss(reduction='none')
            elif context_loss_type.split('-')[0] == "smoothl1":
                beta = float(context_loss_type.split('-')[1]) if len(context_loss_type.split('-')) > 1 else 0.05
                self.loss_fct_context = torch.nn.SmoothL1Loss(reduction='none', beta=beta)
            else:
                raise ValueError(f"context_loss_type only accept 'l2', 'smoothl1' and 'smoothl1-0.5'. Received {context_loss_type}")
          
         
        self.latent_norm = None 
         

        self.mol_extra_features = mol_extra_features
        self.atom_extra_features = atom_extra_features 

        self.num_atom_features = self.decoder.model.num_atom_features
        (
            corruption_lower_bounds,
            corruption_replacement_counts,
            self._corruption_atom_feature_rng_period,
        ) = _compact_atom_corruption_metadata(
            self.num_atom_features, 
        )
        self.atom_feature_cardinalities = tuple(
            (corruption_replacement_counts + 1).tolist()
        )
        self.register_buffer(
            "_corruption_atom_feature_lower_bounds",
            corruption_lower_bounds,
            persistent=False,
        )
        self.register_buffer(
            "_corruption_atom_feature_replacement_counts",
            corruption_replacement_counts,
            persistent=False,
        )
        # self.num_dataset_classes = self.decoder.model.num_dataset_classes
        # print(f"CloundVAE Using num_dataset_classes: {self.num_dataset_classes}")
        print(
            "CloundVAE Using atom_feature_cardinalities: "
            f"{self.atom_feature_cardinalities}"
        )
          
    def disabled_train(self, mode=True):
        """Overwrite model.train with this function to make sure train/eval mode
        does not change anymore."""
        return self

    def _replace_atom_features_(
        self,
        atom_features: LongTensor,
        corruption_mask: BoolTensor,
    ) -> None:
        current = atom_features[..., :self.num_atom_features]
        replacements = _sample_compact_atom_feature_replacements(
            current,
            self._corruption_atom_feature_lower_bounds,
            self._corruption_atom_feature_replacement_counts,
            self._corruption_atom_feature_rng_period,
        )
        torch.where(
            corruption_mask.unsqueeze(-1),
            replacements,
            current,
            out=current,
        )

    def _corrupt_coordinates(
        self,
        coordinates: Tensor,
        corruption_mask: BoolTensor,
        wrap: bool,
    ) -> Tensor:
        coordinate_mask = corruption_mask.unsqueeze(-1)
        if self.corruption_type == "mask":
            mask_position = self.atom_mask_position.to(coordinates).view(1, 1, 3)
            return torch.where(coordinate_mask, mask_position, coordinates)

        noise = torch.randn_like(coordinates).mul_(self.noise_scale)
        noise.mul_(coordinate_mask)
        coordinates.add_(noise)
        if wrap:
            coordinates.remainder_(1.0)
        return coordinates

    def _apply_input_corruption_(
        self,
        batched_data: dict[str, Tensor],
        corruption_mask: BoolTensor,  
    ) -> None:
        if not self.corruption_pos_only:
            atom_features = batched_data['x'][..., :self.num_atom_features]
            if self.corruption_type == "mask":
                atom_features.masked_fill_(
                    corruption_mask.unsqueeze(-1),
                    self.atom_mask_token_id,
                )
            else:
                self._replace_atom_features_(batched_data['x'], corruption_mask)

        molecule_corruption_mask = corruption_mask 
        batched_data['pos'] = self._corrupt_coordinates(
            batched_data['pos'],
            molecule_corruption_mask,
            wrap=False,
        )
         
    def sample_normal(self, mu:Tensor, sigma:Tensor, fix_noise=False) -> Tensor:
        """Samples from a Normal distribution.
        mu   : (bz, 1, d)
        sigma: (bz, 1, d)
        """
        bs = 1 if fix_noise else mu.size(0)
        eps = torch.randn(bs, *mu.shape[1:], dtype=mu.dtype, device=mu.device)
        return mu + sigma * eps

    def compute_reconstruction_loss(
        self,   
        position_padding_mask:Tensor,
        num_atoms:Tensor, 
        x:Tensor,    
        ###### Molecular-specific features ######
        position:Tensor=None, # including the frac_coords 
        charge:Tensor=None,
        spin_multiplicity:Tensor=None,
        number_of_rings:Tensor=None,
        has_fused_rings:Tensor=None,
        ###### Context-specific features ######
        y:Tensor=None,
        ###### Molecular-specific features ######
        pred_position:Tensor=None,  
        pred_x_logits:Tensor=None, 
        pred_charge_logits:Tensor=None,
        pred_spinmp_logits:Tensor=None,
        pred_number_of_rings_logits:Tensor=None,
        pred_has_fused_rings_logits:Tensor=None,
        ###### Context-specific predictions ######
        pred_y:Tensor=None,
        pred_y_source:Tensor=None,
        ###### Loss weights ######
        wpos:float=1.,
        wseq:float=1.,
        wcharge:float=1.,
        wspinmp:float=1.,
        wring:float=1.,
        wctx:float=1.,
        **kwargs
    ):
        """ max_n_nodes have the eos token counted
        GROUND TRUTH:
        - pos                (b, max_n_nodes, 3) 
        - x                  (b, max_n_nodes, d_x)
        - charge             (b,)
        - spinmp             (b,)
        - number_of_rings    (b,)
        - has_fused_rings    (b,) 
        - num_atoms          (b,) 
            includes EOS token
        - padding_mask         (b, max_n_nodes)
            False-valid, True-padding
            EOS token is valid, so set as False
        - position_padding_mask (b, max_n_nodes)
            False-valid, True-padding
            EOS token is non-valid, so set as True
        - y                  (b, context_dim) optional


        PREDICTION:
        - pred_position      (b, max_n_nodes, 3) 
        - pred_x_logits      (b, max_n_nodes, d_x, vocab_size)
        - pred_charge_logits (b, num_charges)
        - pred_spinmp_logits (b, num_spinmp)
        - pred_number_of_rings_logits (b, num_number_of_rings)
        - pred_has_fused_rings_logits (b, num_has_fused_rings) 
        - pred_y             (b, context_dim) optional
        - pred_y_source      (b,) optional; 0=molecule, 1=latent, 2=joint
        """
        loss_itemized = {}
        b, m = x.shape[:2]
        device = x.device 
        zero_loss = pred_x_logits.new_zeros(())

    
        def _zero_graph_loss() -> Tensor:
            return zero_loss

        # Compute molecule-level feature losses 
        charge_loss = self.loss_fct_mol(pred_charge_logits, charge)
        loss_itemized['REC_Loss_mol_charge'] = charge_loss.detach()
         
        spinmp_loss = self.loss_fct_mol(pred_spinmp_logits, spin_multiplicity, "spinmp")
        loss_itemized['REC_Loss_mol_spinmp'] = spinmp_loss.detach()
        
        number_of_rings_loss = self.loss_fct_mol(pred_number_of_rings_logits, number_of_rings)
        loss_itemized['REC_Loss_mol_number_of_rings'] = number_of_rings_loss.detach()
         
        has_fused_rings_loss = self.loss_fct_mol(pred_has_fused_rings_logits, has_fused_rings)
        loss_itemized['REC_Loss_mol_has_fused_rings'] = has_fused_rings_loss.detach()
        
        ##### ===== Context ===== #####
        if self.context_dim > 0:
            pred_y_2d = pred_y.reshape(-1, self.context_dim)
            y_2d = y.reshape(-1, self.context_dim)
            if pred_y_source is None:
                pred_y_source = torch.full(
                    (pred_y_2d.shape[0],),
                    2,
                    dtype=torch.long,
                    device=pred_y_2d.device,
                )
            else:
                pred_y_source = pred_y_source.reshape(-1).to(
                    device=pred_y_2d.device,
                    dtype=torch.long,
                )
            if pred_y_source.shape[0] != pred_y_2d.shape[0]:
                raise ValueError(
                    "pred_y_source must have one entry per property prediction; "
                    f"received {pred_y_source.shape[0]} sources for "
                    f"{pred_y_2d.shape[0]} predictions."
                )
            
            if self.training:
                context_objective_per_dim = self.loss_fct_context(
                    pred_y_2d,
                    y_2d,
                )
            else:
                context_objective_per_dim = self.loss_fct_cts_eval(
                    pred_y_2d,
                    y_2d,
                )
            context_objective_per_sample = context_objective_per_dim.mean(dim=-1)
            source_weights = context_objective_per_sample.new_tensor(
                self.property_source_loss_weights
            )

            loss_itemized['REC_Loss_context'] = context_objective_per_sample.mean().detach()
            source_counts = torch.bincount(pred_y_source, minlength=3)
            safe_source_counts = source_counts.clamp_min(1).to(
                context_objective_per_sample.dtype
            )
            source_active = (source_counts > 0).to(
                context_objective_per_sample.dtype
            )
            source_objective_sums = torch.zeros_like(source_weights).scatter_add_(
                0,
                pred_y_source,
                context_objective_per_sample,
            )
            source_objectives = source_objective_sums / safe_source_counts

            source_objective_per_dim_sums = context_objective_per_dim.new_zeros(
                len(source_weights), self.context_dim
            ).index_add_(0, pred_y_source, context_objective_per_dim)
            source_objectives_per_dim = (
                source_objective_per_dim_sums
                / safe_source_counts[:, None]
            )

            # Sampling probabilities control exposure
            context_loss = (
                source_weights * source_objectives * source_active
            ).sum()
            source_names = ("molecule", "latent", "joint")
            for source_id, source_name in enumerate(source_names):
                loss_itemized[f'REC_Loss_context_{source_name}'] = (
                    source_objectives[source_id].detach()
                )
                loss_itemized[f'REC_Loss_context_{source_name}_count'] = (
                    source_counts[source_id].detach()
                )
            loss_itemized['REC_Loss_context_weighted'] = context_loss.detach()

            # For multi-property context
            if self.context_dim > 1:
                context_objectives_per_dim = context_objective_per_dim.mean(dim=0)
                context_weighted_per_dim = (
                    source_weights[:, None]
                    * source_objectives_per_dim
                    * source_active[:, None]
                ).sum(dim=0)
                for dim, suffix in enumerate(self.context_metric_suffixes):
                    loss_itemized[f'REC_Loss_context_{suffix}'] = (
                        context_objectives_per_dim[dim].detach()
                    )
                    loss_itemized[f'REC_Loss_context_weighted_{suffix}'] = (
                        context_weighted_per_dim[dim].detach()
                    )
                    for source_id, source_name in enumerate(source_names):
                        loss_itemized[
                            f'REC_Loss_context_{source_name}_{suffix}'
                        ] = source_objectives_per_dim[source_id, dim].detach()
        else:
            context_loss = _zero_graph_loss()
        
        #################### ===== Atom types ===== #################
        # Compute the loss for additional atom features (only nonpadded atoms are counted)
        # Compute the loss for atom types (including EOS token)
        padding_mask_x = position_padding_mask.clone()
        padding_mask_x = padding_mask_x.unsqueeze(-1).repeat(1, 1, self.num_atom_features)       # (b, max_n_nodes, d_x)
        batch_idx = torch.arange(b, device=device)
        eos_idx = (num_atoms - 1).clamp(min=0, max=max(m - 1, 0))
        padding_mask_x[batch_idx, eos_idx, 0] = False       # EOS token is counted for the first feature (atomic numbers)
        n_nonpadded_x = num_atoms[:, None].repeat(1, self.num_atom_features)      # (b, d_x)
        n_nonpadded_x[:, 0] = n_nonpadded_x[:, 0].clamp_min(1)
        n_nonpadded_x[:, 1:] = (n_nonpadded_x[:, 1:] - 1).clamp_min(1) # EOS token is only counted for the first feature (atomic numbers)
        
        # Flatten the tokens and compute the sequence loss
        x_loss = self.loss_fct_cat(
            pred_x_logits.view(-1, pred_x_logits.shape[-1]), 
            x.view(-1)
        )                                                    # (b*max_n_nodes*d_x, ) 
        x_loss = x_loss.view(b, -1, self.num_atom_features)  # (b, max_n_nodes, d_x)
        x_loss = x_loss.masked_fill_(padding_mask_x, 0.0)    # (b, max_n_nodes, d_x)
        # Average the loss over the non padded sequences 
        x_loss_per_sample = x_loss.sum(1) / n_nonpadded_x    # (b, d_x)
        
        x_loss = x_loss_per_sample.mean(0)               # (d_x, )


        ##### ===== Positions (Molecule) ===== #####
        n_nonpadded_pos = (num_atoms - 1).clamp_min(1)
        # Compute the position reconstruction metric used for logging.
        if self.training:
            position_loss = self.loss_fct_cts(position / self.position_scale, pred_position).mean(-1) # (b, max_n_nodes, 3) -> (b, max_n_nodes)   
        else:
            position_loss = self.loss_fct_cts_eval(position / self.position_scale, pred_position).mean(-1) # (b, max_n_nodes, 3) -> (b, max_n_nodes)   
        
        # Mask out the padding   
        position_loss = position_loss.masked_fill_(position_padding_mask, 0.0)    # (b, max_n_nodes)
        # Average the loss over the non padded sequences 
        position_loss = position_loss.sum(1) / n_nonpadded_pos                    # (b,)
        # Mean the total loss over batch
        position_loss = position_loss.mean(0)                                     # (,)    
        
        if self.training:
            position_loss_print = position_loss.detach() * self.position_scale_out
        else:
            position_loss_print = position_loss.detach() * self.position_scale

        loss_itemized['REC_Loss_atom_pos'] = position_loss_print.detach()
        loss_itemized['REC_Loss_atom_num_mol'] = (
            x_loss_per_sample.mean(0)[0].detach()
        )
        
        # Sum the overall loss
        reconstruction_x_loss = (
            wseq * x_loss.sum()
            + wpos * position_loss
            + wcharge * charge_loss
            + wspinmp * spinmp_loss
            + wring * (number_of_rings_loss + has_fused_rings_loss) 
        )
        loss = reconstruction_x_loss + wctx * context_loss
        
        
        loss_itemized.update({
            'REC_Loss_atom_num': x_loss[0].detach(),
        })

        for j, feature in enumerate(self.atom_extra_features):
            loss_itemized[f'REC_Loss_atom_{feature}'] = x_loss[j + 1].detach()
        
        return loss, loss_itemized
    
    def get_position_padding_mask(self, num_atoms: Tensor, padding_mask: Tensor) -> Tensor:
        """Return a node-aligned mask where EOS and padded rows are excluded from position decoding."""
        batch_idx = torch.arange(num_atoms.shape[0], device=padding_mask.device)
        eos_idx = (num_atoms - 1).clamp(min=0, max=max(padding_mask.shape[1] - 1, 0))
        position_padding_mask = padding_mask.clone()
        position_padding_mask[batch_idx, eos_idx] = True    # EOS token is excluded from the position loss
        return position_padding_mask
    
    def get_context(self, batched_data: dict[str, Tensor], batch_size: int, device: torch.device) -> Tensor:
        """Select the property/context columns used by the encoder and context reconstruction head."""
        if self.context_dim == 0:
            return None
        if self.context_name not in batched_data:
            raise KeyError(f"Missing context tensor '{self.context_name}' in batched_data.")

        context = batched_data[self.context_name].to(device=device)
        context = context.reshape(batch_size, -1)
        if self.context_idx is not None:
            context_idx = self.context_idx.to(device=device)
            context = context.index_select(dim=1, index=context_idx)
        else:
            if context.shape[1] < self.context_dim:
                raise ValueError(
                    f"Context tensor '{self.context_name}' has {context.shape[1]} columns, "
                    f"but model context_dim is {self.context_dim}."
                )
            context = context[:, :self.context_dim]
        return context
    
    def compute_loss(
            self,
            batched_data:dict[str, Tensor],
            wkl:float=1.,
            wpos:float=1.,
            wseq:float=1., 
            wcharge:float=1.,
            wspinmp:float=1.,
            wring:float=1., 
            wctx:float=1., 
            **kwargs
        ) -> tuple[Tensor, dict]:
        """Computes an estimator for the variational lower bound.
        
        batched_data (max_n_nodes have the eos token counted)
            - position     (b, max_n_nodes, 3)  
            - x            (b, max_n_nodes, 1) 
            - padding_mask (b, max_n_nodes)
                0-valid, 1-padding
        """
        loss_dict = {}
        diagnose_dict = {}
        b, max_n_nodes, _ = batched_data['x'].shape
        device = batched_data['x'].device
       
        num_atoms_with_eos = batched_data.get('num_atoms', None)
        if num_atoms_with_eos is None and 'padding_mask' in batched_data:
            num_atoms_with_eos = (~batched_data['padding_mask'].bool()).sum(dim=-1)
        if num_atoms_with_eos is None:
            num_atoms_with_eos = batched_data['x'][:, :, 0].ne(0).sum(dim=-1)
        if self.corruption_ratio > 0 and self.training:                           # (b, max_n_nodes, 3)
            orig_x   = batched_data['x'].clone()                                   # (b, max_n_nodes, d_x)
            
            orig_pos = batched_data['pos'].clone()       
            # `num_atoms` includes EOS in this dataset, so use `num_atoms - 1`
            # to corrupt only real atoms and never the EOS/padded tail.
            real_num_atoms = torch.clamp(num_atoms_with_eos.to(device=device) - 1, min=0)
            token_idx = torch.arange(max_n_nodes, device=device).unsqueeze(0)  # (1, max_n_nodes)
            valid_mask = token_idx < real_num_atoms.unsqueeze(1)               # (b, max_n_nodes)

            # Corrupt each molecule independently so a single small molecule does
            # not force the entire batch to use the same tiny corruption budget.
            corruption_mask, _ = sample_per_molecule_corruption_mask(
                valid_mask=valid_mask,
                corruption_ratio=self.corruption_ratio,
            )
            self._apply_input_corruption_(
                batched_data,
                corruption_mask,
            )

            invalid_position_mask = ~valid_mask.unsqueeze(-1)
            batched_data['pos'].masked_fill_(invalid_position_mask, 0.0) 
            
        else:
            orig_x   = batched_data['x']  
            orig_pos = batched_data['pos']                                 # (b, max_n_nodes, 3) 
        y = self.get_context(batched_data, b, device)

        # Encoder output.
        z_mu, z_sigma, padding_mask = self.encoder(
            x=orig_x, 
            position=orig_pos, 
            token_padding_mask=batched_data.get('padding_mask', None),
            num_atoms=batched_data('num_atoms', None), 
            charge=batched_data['charge'],
            spin_multiplicity=batched_data['spin_multiplicity'],
            number_of_rings=batched_data['number_of_rings'],
            has_fused_rings=batched_data['has_fused_rings'], 
            y=y if self.encoder_context_dim > 0 else None,
        )
        if num_atoms_with_eos is None:
            num_atoms_with_eos = (~padding_mask[:, 1:]).sum(dim=-1)
        position_padding_mask = self.get_position_padding_mask(num_atoms_with_eos, padding_mask[:, 1:])  # (b, max_n_nodes)
        assert z_mu.dim() == 3, "z_mu and z_sigma must have the same shape"
        assert z_mu.shape[0] == b, "z_mu and z_mu must have the same batch size"
        assert z_mu.shape[1] == 1, "z_mu first dimension must be 1"
        
        z_mu    = z_mu.view(b, -1)
        z_sigma = z_sigma.view(b, -1)
        
        mu_var_floor_loss = torch.zeros((), device=z_mu.device, dtype=z_mu.dtype)
        mu_cov_loss = torch.zeros((), device=z_mu.device, dtype=z_mu.dtype)
         
        if self.sigma_clamp_max > 0.0:
            z_sigma = z_sigma.clamp(max=self.sigma_clamp_max)
        if self.sigma_clamp_min > 0.0:
            z_sigma = z_sigma.clamp(min=self.sigma_clamp_min)

        # KL(q||p) with q=N(z_mu, z_sigma^2), p=N(0, I), preserving existing aggregation mode.
        kl_per_dim_raw = -torch.log(z_sigma + 1e-8) + 0.5 * (z_sigma**2 + z_mu**2) - 0.5
        if self.free_bits_per_dim > 0.0:
            kl_per_dim = torch.clamp(kl_per_dim_raw, min=self.free_bits_per_dim) 
        else:
            kl_per_dim = kl_per_dim_raw 

        if self.kl_aggregation == "mean": 
            loss_kl = kl_per_dim.mean(dim=-1).mean()
        elif self.kl_aggregation == "sum": 
            loss_kl = kl_per_dim.sum(dim=-1).mean()
        else:
            raise ValueError(f"kl_aggregation can only be 'sum' or 'mean'; received {self.kl_aggregation}")

        if self.mu_var_floor_weight > 0.0 or self.mu_cov_weight > 0.0:
            mu_var_floor_loss, mu_cov_loss = mu_var_cov_regularizer(
                z_mu=z_mu,
                var_floor=self.mu_var_floor,
            )
            loss_dict['z_mu var floor Loss'] = mu_var_floor_loss.detach()  # Soft floor on per-dim z_mu variance; high means many dims are underused.
            loss_dict['z_mu cov Loss'] = mu_cov_loss.detach()  # Off-diagonal covariance penalty on z_mu; high means dims are redundant / overly coupled.
            
        
        # Sample from normal 
        z = self.sample_normal(mu=z_mu, sigma=z_sigma)
        loss_dict['KL Loss'] = loss_kl.detach()  # Effective KL after free bits; this is the term that enters the objective.
        
        # Decoder with teacher forcing, using corrupted x and position, force to rely on the latent features instead of the input features
        decoder_kwargs = {
            "latent_embeds": z.view(b, 1, -1),
            "position": batched_data['pos'] if self.num_mol_datasets > 0 else None, # (b, max_n_nodes, 3)
            "x": batched_data['x'][...,:self.num_atom_features], # (b, max_n_nodes, d_x)
            "target_x": orig_x[...,:self.num_atom_features],
            "padding_mask": padding_mask, # (b, 1+n_nodes) EOS token is valid, set as False, and n_nodes includes EOS token; +1 for the latent embedding entry
            "position_padding_mask": position_padding_mask, # (b, n_nodes) EOS token is non-valid, and is set as True, and n_nodes includes EOS token 
        }
        
        decoder_outputs = self.decoder(**decoder_kwargs)
        (
            pred_position,  
            pred_x_logits, 
            pred_charge_logits, 
            pred_spinmp_logits, 
            pred_number_of_rings_logits, 
            pred_has_fused_rings_logits, 
            pred_y,
        ) = decoder_outputs[:10]
        pred_y_source = decoder_outputs[10] if len(decoder_outputs) > 10 else None
        # Compute the reconstruction loss against the original x and position. 
        reconstruction_kwargs = {
            "position": orig_pos,      
            "x": orig_x,   
            "position_padding_mask": position_padding_mask,  # (b, max_n_nodes) EOS token is non-valid, and is set as True, and n_nodes includes EOS token
            "num_atoms": num_atoms_with_eos,
            "charge": batched_data['charge'] ,
            "spin_multiplicity": batched_data['spin_multiplicity'] ,
            "number_of_rings": batched_data['number_of_rings'] ,
            "has_fused_rings": batched_data['has_fused_rings']  ,   
            "y": y,
            "pred_position": pred_position,     
            "pred_x_logits": pred_x_logits,  
            "pred_charge_logits": pred_charge_logits,
            "pred_spinmp_logits": pred_spinmp_logits,
            "pred_number_of_rings_logits": pred_number_of_rings_logits,
            "pred_has_fused_rings_logits": pred_has_fused_rings_logits,        
            "pred_y": pred_y,
            "pred_y_source": pred_y_source,
            "wpos": wpos,
            "wseq": wseq, 
            "wcharge": wcharge,
            "wspinmp": wspinmp,
            "wring": wring,
            "wctx": wctx,
        }
        loss_recon, loss_itemized = self.compute_reconstruction_loss(
            **reconstruction_kwargs
        )
        loss_itemized["REC_Loss_total"] = loss_recon.detach()
        
        # Compute the autoencoder loss mean over batch
        loss = (
            loss_recon
            + wkl * loss_kl
            + self.mu_var_floor_weight * mu_var_floor_loss
            + self.mu_cov_weight * mu_cov_loss
        )
        loss_dict.update(loss_itemized)
        return loss, loss_dict, diagnose_dict
    
    def configure_AdamW(self, weight_decay=0.01, learning_rate=1e-4, betas=(0.9, 0.999), eps=1e-8):
        import inspect
        encoder = getattr(self.encoder, "_orig_mod", self.encoder)
        decoder = getattr(self.decoder, "_orig_mod", self.decoder)
        encoder_model = getattr(encoder, "model", encoder)
        decoder_model = getattr(decoder, "model", decoder)
        assert encoder_model.width_multiplier == decoder_model.width_multiplier, "Encoder and decoder width multiplier must be the same under current implementation"
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        
        if (not self.standparam):
            hidden_weight_suffixes = ('c_attn.weight', 'c_fc.weight', 'c_proj.weight')
            hidden_bias_suffixes = ('c_attn.bias', 'c_fc.bias', 'c_proj.bias')
            bucket_names = (
                'emb_weight',
                'emb_bias',
                'static_ln',
                'hidden_ln',
                'hidden_weight',
                'hidden_bias',
            )
            buckets = {
                scope: {bucket: [] for bucket in bucket_names}
                for scope in ('main', 'flow')
            }

            def add_weight_or_bias(scope, weight_bucket, bias_bucket, name, param):
                if name.endswith('weight'):
                    buckets[scope][weight_bucket].append(param)
                elif name.endswith('bias'):
                    buckets[scope][bias_bucket].append(param)
                else:
                    buckets[scope][bias_bucket].append(param)

            def add_position_flow_param(name, param):
                if ('.res_blocks.' in name) and ('.in_ln.' in name):
                    buckets['flow']['hidden_ln'].append(param)
                elif ('.res_blocks.' in name) and name.endswith('.weight'):
                    buckets['flow']['hidden_weight'].append(param)
                elif ('.res_blocks.' in name) and name.endswith('.bias'):
                    buckets['flow']['hidden_bias'].append(param)
                else:
                    add_weight_or_bias('flow', 'emb_weight', 'emb_bias', name, param)

            for n, p in param_dict.items():
                is_set_flow_param = 'set_flow_head.' in n
                is_position_flow_param = (
                    'atompos_flow_head.' in n
                    or 'atomfrac_flow_head.' in n
                    or 'position_flow_atom_encoder.' in n
                )
                is_formula_decoder_param = 'formula_decoder.' in n
                if is_formula_decoder_param:
                    if '.gru.' in n and '.weight_' in n:
                        buckets['main']['hidden_weight'].append(p)
                    elif '.gru.' in n and '.bias_' in n:
                        buckets['main']['hidden_bias'].append(p)
                    elif '.2.' in n and p.dim() == 1:
                        buckets['main']['static_ln'].append(p)
                    else:
                        add_weight_or_bias('main', 'emb_weight', 'emb_bias', n, p)
                elif is_set_flow_param:
                    if ('.ln_' in n) and ('.ln_f.' not in n):
                        buckets['flow']['hidden_ln'].append(p)
                    elif n.endswith(hidden_weight_suffixes):
                        buckets['flow']['hidden_weight'].append(p)
                    elif n.endswith(hidden_bias_suffixes):
                        buckets['flow']['hidden_bias'].append(p)
                    elif ('.ln_f.' in n):
                        buckets['flow']['static_ln'].append(p)
                    else:
                        add_weight_or_bias('flow', 'emb_weight', 'emb_bias', n, p)
                elif is_position_flow_param:
                    add_position_flow_param(n, p)
                elif n.endswith('pre_atomseq_decoder.weight'):
                    buckets['main']['emb_weight'].append(p)
                elif n.endswith('pre_atomseq_decoder.bias'):
                    buckets['main']['emb_bias'].append(p)
                elif ('_encoder.weight' in n):
                    buckets['main']['emb_weight'].append(p)
                elif ('_encoder.bias' in n):
                    buckets['main']['emb_bias'].append(p)
                elif (n.endswith(('graph_token.weight', 'property_token.weight', 'pte.weight', 'mixing.weight', 'tte_pos.weight', 'tte_x.weight'))):    
                    buckets['main']['emb_weight'].append(p)
                elif (n.endswith('mixing.bias')) or n == 'atom_mask_position':
                    buckets['main']['emb_bias'].append(p)
                elif (
                    (('.linear1.weight') in n)
                    or (('.linear2.weight') in n)
                    or (('.readout_hidden_layers.') in n and n.endswith('.weight'))
                ):
                    buckets['main']['emb_weight'].append(p)
                elif (
                    (('.linear1.bias') in n)
                    or (('.linear2.bias') in n)
                    or (('.readout_hidden_layers.') in n and n.endswith('.bias'))
                ):
                    buckets['main']['emb_bias'].append(p)
                elif ('.readout.') in n:
                    if n.endswith('weight'):
                        buckets['main']['emb_weight'].append(p)
                    elif n.endswith('bias'):
                        buckets['main']['emb_bias'].append(p)
                    else:
                        buckets['main']['emb_bias'].append(p)
                elif ('.ln_' in n) and ('.ln_f.' not in n): # in the stack
                    buckets['main']['hidden_ln'].append(p)
                elif n.endswith(hidden_weight_suffixes):
                    buckets['main']['hidden_weight'].append(p)
                elif n.endswith(hidden_bias_suffixes):
                    buckets['main']['hidden_bias'].append(p)
                elif ('.ln_f.' in n):
                    buckets['main']['static_ln'].append(p)
                else:
                    raise ValueError(f'Unhandled parameter {n}')
            grouped_params = [
                param
                for scope_buckets in buckets.values()
                for params in scope_buckets.values()
                for param in params
            ]
            grouped_param_ids = [id(p) for p in grouped_params]
            if len(grouped_param_ids) != len(set(grouped_param_ids)):
                raise RuntimeError("configure_AdamW assigned at least one parameter to multiple groups.")
            grouped_param_id_set = set(grouped_param_ids)
            missing_param_names = [
                n for n, p in param_dict.items() if id(p) not in grouped_param_id_set
            ]
            if missing_param_names:
                raise RuntimeError(
                    "configure_AdamW did not assign all parameters to optimizer groups: "
                    f"{missing_param_names}"
                )

            main_width_multiplier = float(encoder_model.width_multiplier)
            main_depth_multiplier = float(
                getattr(encoder_model, "depth_multiplier", getattr(decoder_model, "depth_multiplier", 1.0))
            )
            main_depth_alpha_exp = float(
                getattr(encoder_model, "depth_alpha_exp", getattr(decoder_model, "depth_alpha_exp", 1.0))
            )
            flow_heads = [
                head for head in (
                    getattr(decoder_model, "frac_set_flow_head", None),
                    getattr(decoder_model, "mol_set_flow_head", None),
                    getattr(decoder_model, "atomfrac_flow_head", None),
                    getattr(decoder_model, "atompos_flow_head", None),
                    getattr(decoder_model, "formula_frac_set_flow_head", None),
                )
                if head is not None
            ]
            flow_scalings = {
                (
                    float(getattr(head, "width_multiplier", main_width_multiplier)),
                    float(getattr(head, "depth_multiplier", main_depth_multiplier)),
                    float(getattr(head, "depth_alpha_exp", main_depth_alpha_exp)),
                )
                for head in flow_heads
            }
            if len(flow_scalings) > 1:
                raise RuntimeError(
                    "configure_AdamW currently expects all coordinate-flow heads "
                    f"to share CompleteP scaling, got {sorted(flow_scalings)}."
                )
            if flow_scalings:
                flow_width_multiplier, flow_depth_multiplier, flow_depth_alpha_exp = next(iter(flow_scalings))
            else:
                flow_width_multiplier = main_width_multiplier
                flow_depth_multiplier = main_depth_multiplier
                flow_depth_alpha_exp = main_depth_alpha_exp

            def make_completep_groups(
                params,
                width_multiplier,
                depth_multiplier,
                depth_alpha_exp,
            ):
                width_lr_scaling = 1.0 / width_multiplier
                depth_lr_scaling = depth_multiplier ** (depth_alpha_exp - 1.0)
                emb_eps = eps / width_multiplier
                hidden_eps = (eps / width_multiplier) * (depth_multiplier ** (-depth_alpha_exp))
                return [
                    {
                        'params': params['emb_weight'],
                        'weight_decay': weight_decay,
                        'lr_scale': 1.0,
                        'eps': emb_eps,
                    },
                    {
                        'params': params['emb_bias'],
                        'weight_decay': 0.0,
                        'lr_scale': 1.0,
                        'eps': emb_eps,
                    },
                    {
                        'params': params['static_ln'],
                        'weight_decay': 0.0,
                        'lr_scale': 1.0,
                        'eps': emb_eps,
                    },
                    {
                        'params': params['hidden_ln'],
                        'weight_decay': 0.0,
                        'lr_scale': depth_lr_scaling,
                        'eps': hidden_eps,
                    },
                    {
                        'params': params['hidden_weight'],
                        'weight_decay': weight_decay / width_lr_scaling,
                        'lr_scale': width_lr_scaling * depth_lr_scaling,
                        'eps': hidden_eps,
                    },
                    {
                        'params': params['hidden_bias'],
                        'weight_decay': 0.0,
                        'lr_scale': depth_lr_scaling,
                        'eps': hidden_eps,
                    },
                ]
             
            ### Begin CompleteP code ###
            optim_groups = make_completep_groups(
                buckets['main'],
                main_width_multiplier,
                main_depth_multiplier,
                main_depth_alpha_exp,
            )
            if any(buckets['flow'][name] for name in bucket_names):
                optim_groups += make_completep_groups(
                    buckets['flow'],
                    flow_width_multiplier,
                    flow_depth_multiplier,
                    flow_depth_alpha_exp,
                )
            ### End CompleteP code ###
        else:
            decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
            nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
            optim_groups = [
                {'params': decay_params, 'weight_decay': weight_decay},
                {'params': nodecay_params, 'weight_decay': 0.0}
            ]
            num_decay_params = sum(p.numel() for p in decay_params)
            num_nodecay_params = sum(p.numel() for p in nodecay_params)
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and torch.cuda.is_available()
        extra_args = {'fused': True} if use_fused else {}
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, eps=eps, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    def reconstruct(self, batched_data:dict, sample_posterior:bool=False):
        """Computes an estimator for the variational lower bound.
        
        batched_data (max_n_nodes have the eos token counted)
            - position     (b, max_n_nodes, 3)  
            - x            (b, max_n_nodes, 1) 
            - padding_mask (b, max_n_nodes)
                0-valid, 1-padding
        """ 
        b = batched_data['x'].shape[0]
        device = batched_data['x'].device
        
        num_atoms_with_eos = batched_data.get('num_atoms', None)
         
        y = self.get_context(batched_data, b, device)

        # Encoder output.
        z_mu, z_sigma, padding_mask = self.encoder(
            x=batched_data['x'],  
            position=batched_data['pos'] if self.num_mol_datasets > 0 else None,  
            token_padding_mask=batched_data.get('padding_mask', None),
            num_atoms=batched_data.get('num_atoms', None), 
            charge=batched_data['charge'] if self.num_mol_datasets > 0 else None,
            spin_multiplicity=batched_data['spin_multiplicity'] if self.num_mol_datasets > 0 else None,
            number_of_rings=batched_data['number_of_rings'] if self.num_mol_datasets > 0 else None,
            has_fused_rings=batched_data['has_fused_rings'] if self.num_mol_datasets > 0 else None, 
            y=y if self.encoder_context_dim > 0 else None,
        )
        if num_atoms_with_eos is None:
            num_atoms_with_eos = (~padding_mask[:, 1:]).sum(dim=-1)
        assert z_mu.dim() == 3, "z_mu and z_sigma must have the same shape"
        assert z_mu.shape[0] == b, "z_mu and z_mu must have the same batch size"
        assert z_mu.shape[1] == 1, "z_mu first dimension must be 1"
        
        z_mu    = z_mu.view(b, -1)
        z_sigma = z_sigma.view(b, -1) 
 
        if self.sigma_clamp_max > 0.0:
            z_sigma = z_sigma.clamp(max=self.sigma_clamp_max)
        if self.sigma_clamp_min > 0.0:
            z_sigma = z_sigma.clamp(min=self.sigma_clamp_min)


        if sample_posterior:
            # Sample from normal 
            z = self.sample_normal(mu=z_mu, sigma=z_sigma)
        else: 
            z = z_mu
        
        return self.decoder.model.generate_cache(latent_embeds=z.view(b, 1, -1))
        
            
             