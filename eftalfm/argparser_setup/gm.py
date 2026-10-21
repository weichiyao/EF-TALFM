import argparse

def add_gm_arguments(parser:argparse.ArgumentParser): 
    parser.add_argument("--pretrained_vae_checkpoint", type=str, default="")
    parser.add_argument("--pretrained_vae_subfolder", type=str, default="")
    ##############################################################################################################
    parser.add_argument("--gm_recon_weight", type=float, default=1.0, help="weight for the reconstruction loss in trainable_ae or trainable_decoder mode")
    parser.add_argument("--gm_latent_train_file", type=str, default='./latent_train.pt')
    parser.add_argument("--gm_latent_valid_file", type=str, default='./latent_valid.pt')
    parser.add_argument("--gm_latent_test_file", type=str, default='./latent_test.pt')
    parser.add_argument("--gm_latent_stats_filename", type=str, default='./latent_stats_scalar.pt')
    ## flow matching  
    parser.add_argument("--gm_loss_type", type=str, default="l2", help="'l2' | 'smoothl1'. ")
    parser.add_argument("--gm_t_lower_eps", type=float, default=1e-12)
    parser.add_argument("--gm_t_upper_eps", type=float, default=1e-12)
    parser.add_argument("--gm_t_norm_scale_cutoff", type=float, default=1.0, help="cutoff for the t norm scaling, default is 1.0, which means no t norm scaling")
    parser.add_argument('--gm_normalize_y', action='store_true', default=False)
    parser.add_argument('--gm_normalize_y_mode', type=str, default='rs', help="'rs' for robust scalar, 'ss' for standard scalar") 
    parser.add_argument(
        "--gm_cfg_guidance_scale",
        type=float,
        default=1.0,
        help=(
            "classifier-free guidance scale for conditional latent sampling; "
            "1.0 keeps the original conditional sampler, 0.0 uses the unconditional branch"
        ),
    )
    parser.add_argument(
        "--gm_cfg_cond_drop_prob",
        type=float,
        default=0.0,
        help=(
            "probability of dropping conditioning during latent model training for "
            "classifier-free guidance; 0.0 disables CFG training dropout"
        ),
    )
    parser.add_argument("--gm_ode_method", type=str, default="dopri5", help="'dopri5' | 'euler'. ")
    parser.add_argument("--gm_ode_steps", type=int, default=100, help="number of steps for the ode integrator")
    
    ## dynamics model
    parser.add_argument("--gm_dynamics_width_input_alpha", type=float, default=1.0)
    parser.add_argument("--gm_dynamics_width_output_alpha", type=float, default=1.0)
    parser.add_argument("--gm_dynamics_width_multiplier", type=float, default=1.0)
    parser.add_argument("--gm_dynamics_depth_multiplier", type=float, default=1.0)
    parser.add_argument("--gm_dynamics_depth_alpha_exp", type=float, default=1.0)
    
    parser.add_argument("--gm_dynamics_d_model", type=int, default=8)
    parser.add_argument("--gm_dynamics_activation", type=str, default="gelu")
    parser.add_argument("--gm_dynamics_n_layers", type=int, default=6) 
    
    parser.add_argument("--gm_dynamics_model", type=str, default="resnet", help="'generic', 'resnet'")
    
    parser.add_argument("--gm_dynamics_time_freqs", type=int, default=32)
    parser.add_argument("--gm_dynamics_t_encoder_min_freq", type=float, default=1.0)
    parser.add_argument("--gm_dynamics_t_encoder_max_freq", type=float, default=10000.0)
    
    parser.add_argument("--gm_dynamics_hidden_dim", type=int, default=1024) 
    
    parser.add_argument("--gm_dynamics_dropout", type=float, default=0.)
    parser.add_argument("--gm_dynamics_activation_dropout", type=float, default=0.)
    parser.add_argument("--gm_dynamics_droppath_prob", type=float, default=0.)
    parser.add_argument("--gm_dynamics_schedule_droppath", action="store_true", default=False)
    parser.add_argument("--gm_dynamics_disable_adaln_zero_init", action="store_true", default=False,
                        help="initialize AdaLN modulation output layers normally instead of AdaLN-Zero; only used by AdaLN dynamics")
   
    parser.add_argument("--gm_dynamics_pre_ln", action="store_true", default=False)
    parser.add_argument("--gm_dynamics_disable_activation_ln", action="store_true", default=False)
    parser.add_argument("--gm_dynamics_disable_residual_ln", action="store_true", default=False)
    parser.add_argument("--gm_dynamics_x_encoder_type", type=str, default="none")
    parser.add_argument("--gm_dynamics_y_encoder_type", type=str, default="linear",
                        choices=("none", "linear", "mlp", "fourier"))
    
    parser.add_argument("--gm_dynamics_y_num_freqs", type=int, default=64)
    parser.add_argument("--gm_dynamics_y_fourier_scale", type=float, default=1.0)
    parser.add_argument("--gm_dynamics_t_encoder_type", type=str, default="logsnr_sinusoidal")
    return parser
