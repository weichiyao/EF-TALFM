from argparse import Namespace
import os 
import pickle 
import torch 

from collections.abc import Callable

 
from eftalfm.equivariant_free_generative_models.vae.vae_models_pcqm4mv2 import CloudVAEModelTransformerP
from eftalfm.equivariant_free_generative_models.vae.encoder import Cloud3DEncoder
from eftalfm.equivariant_free_generative_models.vae.decoder import Cloud3DDecoder 
from eftalfm.modules.vae_plmodule import CloudVAEPLModule 

from model_training.lightning import load_training_weights 
def get_lgm_model(args:Namespace, precomputed_latent:bool, dm_pretransform_y:Callable | None=None, dm_invtransform_y:Callable | None=None, load_from_checkpoint_filename:str="none"):
    pretrained_vae_checkpoint = args.pretrained_vae_checkpoint if hasattr(args, 'pretrained_vae_checkpoint') else None
    if pretrained_vae_checkpoint is not None:
        pretrained_vae_ckpt_path = os.path.join(args.log_dir, args.pretrained_vae_subfolder, 'vae', pretrained_vae_checkpoint)
        pretrained_vae_args_path = os.path.join(args.log_dir, args.pretrained_vae_subfolder, 'vae', 'args.pickle')
        if not os.path.exists(pretrained_vae_ckpt_path):
            raise ValueError(f"Set to train model with latent representation, but given pretrained_vae_checkpoint path {pretrained_vae_ckpt_path} does not exist!")
        else:
            with open(pretrained_vae_args_path, 'rb') as f:
                vae_args = pickle.load(f)  
            
            vae_model = get_vae_model(vae_args)
            latent_dim = vae_args.latent_dim
            print(f"============= Set to train flow matching model with latent representation with VAE loaded from {pretrained_vae_ckpt_path}. =============")
            vae_module = load_training_weights(
                CloudVAEPLModule(model=vae_model, cfg=vae_args), pretrained_vae_ckpt_path
            )
            del vae_model
            latent_stats = None    
             
            # Get training statistics
            latent_stats_filename = args.gm_latent_stats_filename if hasattr(args, 'gm_latent_stats_filename') else 'latent_stats_scalar.pt'
            latent_stats_path = os.path.join(args.log_dir, args.pretrained_vae_subfolder, 'vae', latent_stats_filename)
            if os.path.exists(latent_stats_path):
                latent_stats = torch.load(latent_stats_path, map_location="cpu", weights_only=True)
                print(f"\tLoaded precomputed latent stats from {latent_stats_path}!")
                if 'active' in latent_stats:
                    if 'd' not in latent_stats:
                        latent_stats['d'] = latent_dim
                        torch.save(latent_stats, latent_stats_path)
                    assert latent_stats['d'] == latent_dim, f"latent_stats['d']={latent_stats['d']} != latent_dim={latent_dim}"
                    latent_dim = len(latent_stats['active'])
                elif 'k' in latent_stats:
                    latent_dim = latent_stats['k']
            else:
                print(f"\tCannot find precomputed latent stats from {latent_stats_path}. Computing...")
                latents_train = torch.load(os.path.join(
                    args.log_dir, args.pretrained_vae_subfolder, 'vae', args.gm_latent_train_file))
                latent_stats_filename_only, _ = os.path.splitext(latent_stats_filename)
                statstype = latent_stats_filename_only.split('_')[-1]
                if statstype == "scalar":
                    mean_ = latents_train['z'].mean() # (,)
                    std_ = latents_train['z'].std(dim=0).mean() # (,)
                    latent_stats = {"mean": mean_, "std": std_}
                    torch.save(latent_stats, latent_stats_path)
                    print(f"\tSaved precomputed latent stats mean={mean_:.4f}, std={std_:.4f} in {latent_stats_path}!")
                elif statstype == "whitener":
                    # here we assume the active dimension is the same as the latent dimension from VAE
                    from eftalfm.equivariant_free_generative_models.utils import fit_pca_whitener_torch
                    latent_stats = fit_pca_whitener_torch(latents_train['z'], eps=1e-6, use_float64=True)
                    latent_stats['active']=torch.arange(latents_train['z'].shape[-1], device=latents_train['z'].device, dtype=torch.long)
                    latent_stats['d']=latent_dim
                    torch.save(latent_stats, latent_stats_path)
                    print(f"\tSaved precomputed latent stats whitener in {latent_stats_path}!")
                elif statstype == "vector":
                    # The flow model flattens each latent sample to (batch, d).
                    # Compute matching per-coordinate statistics so broadcasting
                    # cannot introduce an extra leading dimension.
                    z_train = latents_train['z'].reshape(latents_train['z'].shape[0], -1)
                    mean_ = torch.mean(z_train, dim=0, keepdim=True) # (1,d)
                    std_ = torch.std(z_train, dim=0, keepdim=True)   # (1,d)
                    latent_stats = {"mean": mean_, "std": std_}
                    torch.save(latent_stats, latent_stats_path)
                    print(f"\tSaved precomputed latent VECTOR stats mean of shape {mean_.shape}, std of shape {std_.shape} in {latent_stats_path}!")
                else:
                    raise ValueError(f"statstype={statstype} is not valid! Please use the format 'scalar', 'whitener', or 'vector'")
    else: 
        raise ValueError("Set to train model with latent representation, but pretrained_vae_checkpoint is not given! Please train a VAE model first.")
        
    from eftalfm.equivariant_free_generative_models.lgm.latent_flow_matching_p import CompletePLatentFlowMatching as LatentFlowMatching
    
    if args.gm_dynamics_model=='completep_resnet': 
        from eftalfm.equivariant_free_generative_models.lgm.dynamics import CompletePResNetFlowDynamics as Dynamics
        from eftalfm.modules.gm_plmodule import CompletePGenModelModule as GenModelModule
    elif args.gm_dynamics_model=='completep_adaln_resnet':
        from eftalfm.equivariant_free_generative_models.lgm.dynamics import CompletePAdaLNResNetFlowDynamics as Dynamics
        from eftalfm.modules.gm_plmodule import CompletePGenModelModule as GenModelModule
    else:
        raise ValueError(f"gm_dynamics_model only accept 'resnet' or 'generic'; receive {args.gm_dynamics_model}")
    # Define dynamics model and flows
    dynamics = Dynamics(
        d_model=args.gm_dynamics_d_model if hasattr(args, 'gm_dynamics_d_model') else 128,
        x_dim=latent_dim, 
        y_dim=args.context_dim,
        hidden_dim=args.gm_dynamics_hidden_dim if hasattr(args, 'gm_dynamics_hidden_dim') else 128,
        x_encoder_type=args.gm_dynamics_x_encoder_type if hasattr(args, 'gm_dynamics_x_encoder_type') else "none",
        t_encoder_type=args.gm_dynamics_t_encoder_type if hasattr(args, 'gm_dynamics_t_encoder_type') else "logsnr_sinusoidal",
        y_encoder_type=args.gm_dynamics_y_encoder_type if hasattr(args, 'gm_dynamics_y_encoder_type') else 'linear',
        ###########
        n_layers=args.gm_dynamics_n_layers, 
        activation=args.gm_dynamics_activation,
        freqs=args.gm_dynamics_time_freqs if hasattr(args, 'gm_dynamics_time_freqs') else 32,
        ###########
        dropout=args.gm_dynamics_dropout,
        activation_dropout=args.gm_dynamics_activation_dropout,
        droppath_prob=args.gm_dynamics_droppath_prob,
        schedule_droppath=args.gm_dynamics_schedule_droppath,
        pre_ln=args.gm_dynamics_pre_ln if hasattr(args, 'gm_dynamics_pre_ln') else False,
        disable_activation_ln=args.gm_dynamics_disable_activation_ln if hasattr(args, 'gm_dynamics_disable_activation_ln') else False,
        disable_residual_ln=args.gm_dynamics_disable_residual_ln if hasattr(args, 'gm_dynamics_disable_residual_ln') else False,
        ###########
        width_input_alpha=args.gm_dynamics_width_input_alpha if hasattr(args, 'gm_dynamics_width_input_alpha') else 1.0,
        width_output_alpha=args.gm_dynamics_width_output_alpha if hasattr(args, 'gm_dynamics_width_output_alpha') else 1.0,
        width_multiplier=args.gm_dynamics_width_multiplier if hasattr(args, 'gm_dynamics_width_multiplier') else 1.0,
        depth_multiplier=args.gm_dynamics_depth_multiplier if hasattr(args, 'gm_dynamics_depth_multiplier') else 1.0,
        depth_alpha_exp=args.gm_dynamics_depth_alpha_exp if hasattr(args, 'gm_dynamics_depth_alpha_exp') else 1.0,
        standparam=args.standparam if hasattr(args, 'standparam') else True,
        init_std=args.init_std if hasattr(args, 'init_std') else 1.0,
        ###########
        t_lower_eps=args.gm_t_lower_eps if hasattr(args, 'gm_t_lower_eps') else 1e-12,
        t_upper_eps=args.gm_t_upper_eps if hasattr(args, 'gm_t_upper_eps') else 1e-12,        
        t_encoder_max_freq=args.gm_dynamics_t_encoder_max_freq if hasattr(args, 'gm_dynamics_t_encoder_max_freq') else 10000.0,
        t_encoder_min_freq=args.gm_dynamics_t_encoder_min_freq if hasattr(args, 'gm_dynamics_t_encoder_min_freq') else 1.0,
        t_encoder_final_norm=getattr(args, 'gm_dynamic_t_encoder_final_norm', True), 
        ###########
        y_num_freqs=args.gm_dynamics_y_num_freqs if hasattr(args, 'gm_dynamics_y_num_freqs') else 64,
        y_fourier_scale=args.gm_dynamics_y_fourier_scale if hasattr(args, 'gm_dynamics_y_fourier_scale') else 1.0,
        disable_zero_init_adaln=getattr(args, 'gm_dynamics_disable_adaln_zero_init', False),    
    )     
    model = LatentFlowMatching(
        precomputed_latent=precomputed_latent,
        latent_model=vae_module.model, 
        latent_stats=latent_stats,
        ###########
        conditioning=args.context_dim > 0,  
        pretransform_context=dm_pretransform_y, 
        invtransform_context=dm_invtransform_y,
        ###########
        t_lower_eps=args.gm_t_lower_eps if hasattr(args, 'gm_t_lower_eps') else 1e-12,
        t_upper_eps=args.gm_t_upper_eps if hasattr(args, 'gm_t_upper_eps') else 1e-12,
        t_norm_scale_cutoff=args.gm_t_norm_scale_cutoff if hasattr(args, "gm_t_norm_scale_cutoff") else 1.0,
        ###########
        dynamics=dynamics,        
        cfg_cond_drop_prob=getattr(args, 'gm_cfg_cond_drop_prob', 0.0),
        cfg_guidance_scale=getattr(args, 'gm_cfg_guidance_scale', 1.0),
        ###########
        ode_method=args.gm_ode_method if hasattr(args, 'gm_ode_method') else "euler",
        ode_steps=args.gm_ode_steps if hasattr(args, 'gm_ode_steps') else 100,
        loss_type=args.gm_loss_type if hasattr(args, 'gm_loss_type') else "l2",
        ########### 
        compile_model=args.compile_model if hasattr(args, "compile_model") else False, 
    )
    lightning_module = GenModelModule(model=model, cfg=args)
    
    checkpoint_path = os.path.join(args.log_dir, args.pretrained_vae_subfolder, "flow_matching", args.train, load_from_checkpoint_filename)
    if os.path.exists(checkpoint_path):
        lightning_module = load_training_weights(lightning_module, checkpoint_path)
        print(f"Loaded lightning module from {checkpoint_path}")
    return lightning_module


def _get_vae_context_idx(args:Namespace):
    if getattr(args, "context_dim", 0) == 0:
        return "*"
    return args.context_idx if hasattr(args, "context_idx") else "*"


def get_vae_model(args:Namespace): 
    
    mol_extra_features = []
    atom_extra_features = []  
    mol_extra_features = [
        "charge", 
        "spinmp", 
        "number_of_rings", 
        "has_fused_rings"
    ] 
    atom_extra_features = [
        'chirality',
        'charge',
        'num_Hs',
        'radical',
        'hybrad',
        'conjugated', 
        'is_aromatic',
        'is_in_ring', 
    ]   
    args.num_extra_atom_features = len(atom_extra_features)   
    args.num_atom_features = args.num_extra_atom_features + 1
    
    assert args.num_atoms >= 128+24*(len(atom_extra_features)), f"num_atoms={args.num_atoms} must be greater than 128+{args.preprocess_x_initial_shift}+{args.preprocess_x_offset}*({len(atom_extra_features)})"
    
    # Get encoder and decoder
    encoder = Cloud3DEncoder(args) 
    decoder = Cloud3DDecoder(args)
    
    model = CloudVAEModelTransformerP(
        encoder, 
        decoder,  
        context_idx=_get_vae_context_idx(args),
        context_name=args.context_name if hasattr(args, 'context_name') else "y",
        kl_aggregation=args.kl_aggregation if hasattr(args, "kl_aggregation") else "sum", 
        noise_scale=args.vae_noise_scale if hasattr(args, 'vae_noise_scale') else 0.0,
        corruption_ratio=args.vae_corruption_ratio if hasattr(args, 'vae_corruption_ratio') else 0.0,
        corruption_pos_only=args.vae_corruption_pos_only if hasattr(args, 'vae_corruption_pos_only') else False,
        corruption_type=args.vae_corruption_type if hasattr(args, 'vae_corruption_type') else "mask",
        pos_loss_type=args.pos_loss_type if hasattr(args, 'pos_loss_type') else "l2",
        context_loss_type=args.context_loss_type if hasattr(args, 'context_loss_type') else "l2",
        free_bits_per_dim=args.vae_free_bits_per_dim if hasattr(args, 'vae_free_bits_per_dim') else 0.0,
        sigma_clamp_min=args.vae_sigma_clamp_min if hasattr(args, 'vae_sigma_clamp_min') else 0.0,
        sigma_clamp_max=args.vae_sigma_clamp_max if hasattr(args, 'vae_sigma_clamp_max') else 0.0,
        mu_var_floor=args.vae_mu_var_floor if hasattr(args, 'vae_mu_var_floor') else 0.09,
        mu_var_floor_weight=args.vae_mu_var_floor_weight if hasattr(args, 'vae_mu_var_floor_weight') else 0.0,
        mu_cov_weight=args.vae_mu_cov_weight if hasattr(args, 'vae_mu_cov_weight') else 0.0, 
        noise_reconstruction = args.noise_reconstruction if hasattr(args, 'noise_reconstruction') else False,
        standparam=args.standparam if hasattr(args, 'standparam') else False,
        mol_extra_features=mol_extra_features,
        atom_extra_features=atom_extra_features,
        position_scale=args.position_scale if hasattr(args, 'position_scale') else 1.0,
        property_molecule_only_loss_weight=getattr(
            args, "property_molecule_only_loss_weight", 0.6
        ),
        property_latent_only_loss_weight=getattr(
            args, "property_latent_only_loss_weight", 0.4
        ),
        property_joint_loss_weight=getattr(
            args, "property_joint_loss_weight", 1.0
        ),
    )
    del encoder, decoder 
    return model
