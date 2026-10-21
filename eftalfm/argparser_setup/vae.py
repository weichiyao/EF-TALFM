import argparse


# Architecture switches cannot be changed while strictly resuming a checkpoint.
# Runs created before a switch existed must resolve to the legacy architecture.
_VAE_CHECKPOINT_ARCH_DEFAULTS = {
    'decoder_explicit_property_token': False,
    'disable_encoder_context_condition': False,
}


def merge_vae_resume_args(
        saved_args: argparse.Namespace,
        current_args: argparse.Namespace,
    ) -> argparse.Namespace:
    """Merge CLI overrides while preserving checkpoint-shaped VAE options."""
    saved = vars(saved_args).copy()
    current = vars(current_args).copy()
    for name, legacy_default in _VAE_CHECKPOINT_ARCH_DEFAULTS.items():
        current[name] = saved.get(name, legacy_default)
    saved.update(current)
    return argparse.Namespace(**saved)


def add_vae_arguments(parser:argparse.ArgumentParser):
    ## Invariant autoencoder
    parser.add_argument("--vae_noise_scale", type=float, default=0.0, 
                        help="noise to applied on the non-zero positions in vae")
    parser.add_argument("--vae_corruption_ratio", type=float, default=0.0, 
                        help="corruption ratio for the non-zero positions in vae")
    parser.add_argument("--vae_corruption_pos_only", action='store_true', default=False)
    parser.add_argument(
        "--vae_corruption_type",
        type=str,
        choices=("mask", "noise"),
        default="mask",
        help="How selected real atoms are corrupted.",
    ) 
    parser.add_argument("--kl_weight", type=float, default=0.01)
    parser.add_argument("--kl_weight_min", type=float, default=1e-7)
    parser.add_argument("--kl_weight_scaling_factor", type=float, default=5.)
    parser.add_argument("--kl_weight_scheduler", type=str, default="none", help="none | linear | exp_annealing | exp-convex_annealing")
    parser.add_argument("--kl_weight_warmup_fraction", type=float, default=0.2, help="for kl_weight_scheduler='linear', fraction of training over which to ramp min_beta -> max_beta")
    parser.add_argument("--kl_aggregation", type=str, default="sum", help="sum | mean")
    parser.add_argument("--vae_free_bits_per_dim", type=float, default=0.0, help="per-dim free-bits floor in nats for learned-variance VAEs; 0 disables")
    parser.add_argument("--vae_sigma_clamp_min", type=float, default=0.0, help="minimum posterior std clamp for learned-variance VAEs; <=0 disables")
    parser.add_argument("--vae_sigma_clamp_max", type=float, default=0.0, help="maximum posterior std clamp for learned-variance VAEs; <=0 disables")
    parser.add_argument("--vae_mu_var_floor", type=float, default=0.09, help="target floor for per-dim variance of z_mu in learned-variance VAEs; 0.09 corresponds to std 0.3")
    parser.add_argument("--vae_mu_var_floor_weight", type=float, default=0.0, help="weight on soft variance-floor regularizer for z_mu; 0 disables")
    parser.add_argument("--vae_mu_cov_weight", type=float, default=0.0, help="weight on off-diagonal covariance penalty for z_mu; 0 disables")


    parser.add_argument('--position_scale', type=float, default=1.0)
        
    parser.add_argument("--pos_weight", type=float, default=1.)
    parser.add_argument("--seq_weight", type=float, default=1.)
    parser.add_argument("--charge_weight", type=float, default=1.)
    parser.add_argument("--spinmp_weight", type=float, default=1.)
    parser.add_argument("--ring_weight", type=float, default=1.)
    parser.add_argument("--ctx_weight", type=float, default=1.)
    parser.add_argument("--pos_loss_type", type=str, default="l2", help="l2 | smoothl1-0.1 | smoothl1-0.5 | ... ")
    parser.add_argument("--context_loss_type", type=str, default="l2", help="l2 | smoothl1-0.05 | smoothl1-0.1 | ... ")

    parser.add_argument('--latent_dim', type=int, default=512) 
    parser.add_argument('--dropout', type=float, default=0.1)  
    parser.add_argument('--activation', type=str, default="gelu", help="'relu' | 'gelu' | 'tanh' | 'linear'")
    parser.add_argument('--norm_first', action='store_true', default=False)
    parser.add_argument('--layer_norm_eps', type=float, default=1e-5) 
    parser.add_argument("--disable_activation_norm", action='store_true', default=False, help="if true then disable activation normalization")
    parser.add_argument('--width_input_alpha', type=float, default=1.0, help="for width, mu_p tunable")
    parser.add_argument('--width_output_alpha', type=float, default=1.0, help="for width, mu_p tunable")
    parser.add_argument('--width_disable_attention_scaling', action='store_true', default=False, help="for width, default implements mu_p")
    parser.add_argument('--depth_alpha_exp', type=float, default=1.0, help="for depth, default implements complete_p")
         
    ## encoder
    parser.add_argument('--encoder_embed_dim', type=int, default=768) 
    parser.add_argument('--encoder_n_layers', type=int, default=6, help='number of layers for encoder training for latent representation')
    parser.add_argument('--encoder_n_heads', type=int, default=4, help='Encoder') 
    parser.add_argument('--encoder_dim_feedforward', type=int, default=None, help='Encoder')  
    parser.add_argument('--encoder_width_multiplier', type=float, default=1.0)
    parser.add_argument('--encoder_depth_multiplier', type=float, default=1.0)
    parser.add_argument('--no_token_positional_embeddings', action='store_true', default=False)
    parser.add_argument('--attention_dropout', type=float, default=0.1)
    parser.add_argument('--act_dropout', type=float, default=0.0)
    parser.add_argument('--droppath_prob', type=float, default=0.0)
                             
    ## decoder
    parser.add_argument('--decoder_embed_dim', type=int, default=-1)
    parser.add_argument('--decoder_n_heads', type=int, default=4, help='Decoder (TransformerEncoder)') 
    parser.add_argument('--decoder_n_layers', type=int, default=6, help='Decoder (TransformerEncoder)') 
    parser.add_argument('--decoder_dim_feedforward', type=int, default=None, help='Decoder (TransformerEncoder)')  
    parser.add_argument('--decoder_atomseq_temp', type=float, default=1., help='Decoder (TransformerEncoder) sample from logits temperature for atom sequence generation, more diversity when larger than 1')
    parser.add_argument('--decoder_atomseq_topk', type=int, default=1, help='Decoder (TransformerEncoder) sample from logits top_k for atom sequence generation, by default (1) argmax; if 0 then multinomial among all') 
    parser.add_argument('--decoder_pos_readout_depth', type=int, default=2, help='Number of Linear layers in the crystal position readout; 2 preserves the previous setup')
    
    parser.add_argument('--decoder_use_ada_ln_zero', action='store_true', default=False, help="if true then use ada ln zero")
    parser.add_argument('--decoder_width_multiplier', type=float, default=1.0)
    parser.add_argument('--decoder_depth_multiplier', type=float, default=1.0)
    parser.add_argument('--decoder_attention_dropout', type=float, default=0.0)
    parser.add_argument('--decoder_act_dropout', type=float, default=0.0)
    parser.add_argument('--decoder_droppath_prob', type=float, default=0.0)
    
    parser.add_argument(
        '--decoder_explicit_property_token',
        action='store_true',
        default=False,
        help=(
            'replace the legacy last-hidden-state property readout with a learned '
            'token at sequence index 1; during training, property examples are split '
            'among molecule-only, latent-only, and joint attention modes'
        ),
    )
    parser.add_argument(
        '--decoder_property_molecule_only_prob',
        type=float,
        default=0.6,
        help=(
            'per-example probability of the molecule-only property-token pass '
            'during training'
        ),
    )
    parser.add_argument(
        '--decoder_property_latent_only_prob',
        type=float,
        default=0.4,
        help=(
            'per-example probability of the latent-only property-token pass; '
            'one minus this and decoder_property_molecule_only_prob is the '
            'joint-attention probability'
        ),
    )
    parser.add_argument(
        '--decoder_property_eval_mode',
        type=str,
        choices=('sample', 'molecule', 'latent', 'joint'),
        default='sample',
        help=(
            'property-token source mode used outside training; sample preserves '
            'the configured source mixture and enables per-source validation logs'
        ),
    )
    parser.add_argument('--property_molecule_only_loss_weight', type=float, default=1.0)
    parser.add_argument('--property_latent_only_loss_weight', type=float, default=1.0)
    parser.add_argument('--property_joint_loss_weight', type=float, default=1.0)


    return parser
