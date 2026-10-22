from __future__ import annotations
import argparse
import wandb
import pickle
import os 
import re 

import torch 

from lightning.pytorch import Trainer, seed_everything 
from lightning.pytorch.loggers import WandbLogger 
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from model_training.lightning import (
    get_trainer_inputs,
    fit_trusted_checkpoint as _fit_trusted_checkpoint,
    load_training_weights,
)

from eftalfm.argparser_setup.pcqm4mv2 import add_pcqm4mv2_arguments
from eftalfm.argparser_setup.gm import add_gm_arguments
from eftalfm.argparser_setup.common import add_common_arguments
from eftalfm.argparser_setup.file_parser import ArgumentFileParser
from eftalfm.argparser_setup.vae import add_vae_arguments, merge_vae_resume_args
from eftalfm.data.pcqm4m_v2.data.data_plmodule import PCQM4MPLDataModule
from eftalfm.data.latent_data_plmodule import LatentPLDataModule
from eftalfm.modules.vae_plmodule import CloudVAEPLModule 
from eftalfm.equivariant_free_generative_models.vae.vae_models_pcqm4mv2 import CloudVAEModelTransformerP
from eftalfm.equivariant_free_generative_models.vae.encoder import Cloud3DEncoder
from eftalfm.equivariant_free_generative_models.vae.decoder import Cloud3DDecoder 

os.environ["TOKENIZERS_PARALLELISM"] = "true"

_WANDB_RUN_DIR_RE = re.compile(r"^(?:offline-)?run-[^-]+-([A-Za-z0-9]+)$")


def _print_run_summary(args, label: str = "Run config") -> None:
    summary = {
        "train": args.train, 
        "dataset_name": args.dataset_name,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "max_steps": args.max_steps,
        "max_epochs": args.max_epochs,
        "gm_data_augmentation": getattr(args, "gm_data_augmentation", False),
        "resume_from_checkpoint": args.resume_from_checkpoint,
        "device": args.device,
        "n_devices": args.n_devices,
        "num_workers": args.num_workers,
        "precision": args.precision,
        "progress_bar": args.progress_bar,
        "wandb_project": args.wandb_project,
    }
    print(f"{label}: {summary}")


def _extract_wandb_run_id_from_name(name: str):
    match = _WANDB_RUN_DIR_RE.match(name)
    return match.group(1) if match else None


def _find_existing_wandb_run_id(wandb_dir: str):
    if (wandb_dir is None) or (not os.path.isdir(wandb_dir)):
        return None

    run_id_path = os.path.join(wandb_dir, "wandb_run_id.txt")
    if os.path.isfile(run_id_path):
        try:
            with open(run_id_path, "r", encoding="utf-8") as f:
                run_id = f.read().strip()
            if run_id:
                return run_id
        except OSError:
            pass

    latest_run = os.path.join(wandb_dir, "latest-run")
    if os.path.exists(latest_run):
        latest_run_name = os.path.basename(os.path.realpath(latest_run))
        run_id = _extract_wandb_run_id_from_name(latest_run_name)
        if run_id is not None:
            return run_id

    candidates = []
    try:
        for entry in os.scandir(wandb_dir):
            if not entry.is_dir():
                continue
            run_id = _extract_wandb_run_id_from_name(entry.name)
            if run_id is not None:
                candidates.append((entry.stat().st_mtime, run_id))
    except OSError:
        return None

    if not candidates:
        return None

    candidates.sort(reverse=True)
    return candidates[0][1]


def _persist_wandb_run_id(wandb_dir: str, run_id: str):
    if (wandb_dir is None) or (run_id is None):
        return

    try:
        os.makedirs(wandb_dir, exist_ok=True)
        with open(os.path.join(wandb_dir, "wandb_run_id.txt"), "w", encoding="utf-8") as f:
            f.write(run_id)
            f.write("\n")
    except OSError:
        pass


def _is_rank_zero_process() -> bool:
    for env_name in ("RANK", "SLURM_PROCID"):
        env_value = os.environ.get(env_name)
        if env_value not in (None, ""):
            try:
                return int(env_value) == 0
            except ValueError:
                pass

    rank = getattr(rank_zero_only, "rank", None)
    if rank is not None:
        try:
            return int(rank) == 0
        except (TypeError, ValueError):
            return False

    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank not in (None, ""):
        try:
            return int(local_rank) == 0
        except ValueError:
            pass

    return True


def _coerce_wandb_run_id(run_id):
    if callable(run_id):
        try:
            run_id = run_id()
        except TypeError:
            return None
    if run_id is None:
        return None
    if isinstance(run_id, bytes):
        run_id = run_id.decode("utf-8", errors="ignore")
    if not isinstance(run_id, str):
        run_id = str(run_id)
    run_id = run_id.strip()
    return run_id or None


def _persist_args_pickle(args_path: str, args_namespace: argparse.Namespace):
    if args_path is None:
        return

    try:
        with open(args_path, 'wb') as f:
            pickle.dump(args_namespace, f)
    except OSError:
        pass


def get_lgm_model(args, precomputed_latent, dm_pretransform_y=None, dm_invtransform_y=None, load_from_checkpoint_filename="none"):
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
        t_dim=1,
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
    
    checkpoint_path = os.path.join(args.log_dir, args.pretrained_vae_subfolder, args.train, load_from_checkpoint_filename)
    if os.path.exists(checkpoint_path):
        lightning_module = load_training_weights(lightning_module, checkpoint_path)
        print(f"Loaded lightning module from {checkpoint_path}")
    return lightning_module


def _get_vae_context_idx(args):
    if getattr(args, "context_dim", 0) == 0:
        return "*"
    return args.context_idx if hasattr(args, "context_idx") else "*"


def get_vae_model(args): 
    
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


def create_args(): 
    parser = ArgumentFileParser(
        description='PCQM4Mv2-VAE',
        fromfile_prefix_chars='@',
    ) 
    parser.add_argument("--train", type=str, default="vae", help="'vae' | 'latent'.")
    
    parser = add_common_arguments(parser)
    parser = add_pcqm4mv2_arguments(parser) 
    parser = add_vae_arguments(parser)
    parser = add_gm_arguments(parser)
    
    args = parser.parse_args()
    return args


def _get_checkpoint_dir(args):
    """Resolve EF-TALFM's existing stage-specific checkpoint directory."""
    if args.train.startswith("vae"):
        return os.path.join(args.log_dir, args.train, "vae")
    if args.train.startswith("latent"):
        return os.path.join(args.log_dir, args.pretrained_vae_subfolder, args.train)
    raise ValueError(f"args.train must start with 'vae' or 'latent'; received {args.train!r}")


##########################################################################
def main(args, logger):
    torch.set_float32_matmul_precision('high') 
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    print("Creating data module ...")
    if args.dataset_name[-6:] == 'LATENT':
        dm = LatentPLDataModule(args)
        precomputed_latent = True
        print("Using precomputed latents... ")
    else:
        dm = PCQM4MPLDataModule(args)
        precomputed_latent = False
        print("Computing latents on the fly... ")
    
    print("Creating model module ...") 
    if args.train[:3] == 'vae':
        print("============= Set to train the VAE model. =============")
        model = get_vae_model(args) 
        # Define the lightning module
        lightning_module = CloudVAEPLModule(model=model, cfg=args)
        
        monitor = ['val_loss']
        mode = ["min"]
        filename = ["best-e{epoch}-s{step}-apos{val_REC_Loss_atom_pos:.4f}-anum{val_REC_Loss_atom_num:.4f}-vloss{val_loss:.4f}"]

    elif args.train[:6] == 'latent':
        lightning_module = get_lgm_model(
            args, 
            precomputed_latent=precomputed_latent, 
            dm_pretransform_y=getattr(dm, 'pretransform_y', None), 
            dm_invtransform_y=getattr(dm, 'invtransform_y', None),
        )
        monitor = ['val_lfm_loss']
        mode = ["min"]
        filename = [
            "best-e{epoch}-s{global_step}-vlfm{val_lfm_loss:.4f}", 
        ]
                
    else:
        raise ValueError(f"args.train only accepts 'vae', 'latent'; received {args.train}")
    
    print("Creating trainer ...")
    kwargs_trainer, callbacks = get_trainer_inputs(
        _get_checkpoint_dir(args), args, logger, monitor=monitor, mode=mode, filename=filename
    )
    # Define trainer
    trainer = Trainer(**kwargs_trainer, callbacks=callbacks)
    
    if args.train[:3] == 'vae':
        # Load the checkpoint if exists
        ckpt_path=os.path.join(args.log_dir, args.train, "vae", args.resume_from_checkpoint) if (args.resume_from_checkpoint is not None) else None
        _fit_trusted_checkpoint(trainer, lightning_module, dm, ckpt_path=ckpt_path)
    elif args.train[:6] == 'latent':
        # Load the checkpoint if exists
        ckpt_path=os.path.join(args.log_dir, args.pretrained_vae_subfolder, args.train, args.resume_from_checkpoint) if (args.resume_from_checkpoint is not None) else None
        _fit_trusted_checkpoint(trainer, lightning_module, dm, ckpt_path=ckpt_path)
    else:
        raise ValueError(f"args.train only accepts 'vae', 'latent'; received {args.train}")            # Load the checkpoint if exists
    
    print("Fitting model ...")
    print("DONE.")


if __name__ == "__main__":
    import torch
    if torch.cuda.is_available():
        print("torch CUDA is available")
    else:
        print("torch CUDA is NOT available!!!!!!!")

    args = create_args() 
    _print_run_summary(args, label="Parsed args")
    if args.train[:3] == 'vae':
        # Load the checkpoint if exists
        ckpt_path=os.path.join(args.log_dir, args.train, "vae", args.resume_from_checkpoint) if (args.resume_from_checkpoint is not None and args.resume_from_checkpoint != 'none') else None
        
    elif args.train[:6] == 'latent':
        # Load the checkpoint if exists
        ckpt_path=os.path.join(args.log_dir, args.pretrained_vae_subfolder, args.train, args.resume_from_checkpoint) if (args.resume_from_checkpoint is not None and args.resume_from_checkpoint != 'none') else None
    
    else:
        raise ValueError(f"args.train only accepts 'vae', 'latent'; received {args.train}")            # Load the checkpoint if exists

 
    if (ckpt_path is not None) and os.path.exists(ckpt_path):
        final_folder = None
        if args.train[:3] == 'vae':
            # Load the checkpoint if exists
            args_path=os.path.join(args.log_dir, args.train, "vae", 'args.pickle') 
            final_folder = os.path.join(args.log_dir, args.train, "vae")
            
        elif args.train[:6] == 'latent':
            # Load the checkpoint if exists
            args_path=os.path.join(args.log_dir, args.pretrained_vae_subfolder, args.train, 'args.pickle') 
            final_folder = os.path.join(args.log_dir, args.pretrained_vae_subfolder, args.train)
            
        else:
            raise ValueError(f"args.train only accepts 'vae', 'latent'; received {args.train}")            # Load the checkpoint if exists
        
        print(f"Resuming from checkpoint {ckpt_path}...")
        if os.path.exists(args_path):
            with open(args_path, 'rb') as f:
                args_resumed = pickle.load(f)
            
            args = merge_vae_resume_args(args_resumed, args)
            print("Loaded args.pickle and merged with command-line overrides.")
        else:
            print(f"Could not find args.pickle at {args_path}; resuming with current command-line arguments.")

        if final_folder is not None and (not os.path.exists(final_folder)):
            os.makedirs(final_folder)

        with open(args_path, 'wb') as f:
            pickle.dump(args, f)
        print("Updated args.pickle file")
    else:
        print(f"Could not resume from checkpoint from {ckpt_path}...")
        args.resume_from_checkpoint = None


        if args.train[:3] == 'vae':
            # Check if the log folder exists
            if not os.path.exists(args.log_dir):
                raise ValueError(f"args.train='vae' but args.log_dir={args.log_dir} does not exist!")
            if not os.path.exists(os.path.join(args.log_dir, args.train)):
                os.makedirs(os.path.join(args.log_dir, args.train))
            if not os.path.exists(os.path.join(args.log_dir, args.train, 'vae')):
                os.makedirs(os.path.join(args.log_dir, args.train, 'vae'))
            final_folder = os.path.join(args.log_dir, args.train, 'vae')

        elif args.train[:6] == 'latent':
            # Check if the log folder exists
            if not os.path.exists(args.log_dir):
                raise ValueError(f"args.train='latent' but args.log_dir={args.log_dir} does not exist!")
            # Check if the vae folder exists
            vae_folderpath=os.path.join(args.log_dir, args.pretrained_vae_subfolder)
            if not os.path.exists(vae_folderpath):
                raise ValueError(f"args.train='latent' but vae folder path={vae_folderpath} does not exist!")
            if not os.path.exists(os.path.join(vae_folderpath, args.train)):
                os.makedirs(os.path.join(vae_folderpath, args.train))  
            final_folder=os.path.join(vae_folderpath, args.train) 
                
        else:
            raise ValueError(f"args.train only accepts 'vae', 'latent'; received {args.train}")            # Load the checkpoint if exists
        
        print(f"Folder '{final_folder}' was created, the models will be saved here.")
        
        if not os.path.exists(args.wandb_dir):
            # If the folder does not exist, create it
            os.makedirs(args.wandb_dir)
            print(f"Folder '{args.wandb_dir}' was created, the wandb will be saved here.")
        else:
            print(f"Folder '{args.wandb_dir}' already exists, the wandb will be saved here.")
            
        with open(os.path.join(final_folder, 'args.pickle'), 'wb') as f:
            pickle.dump(args, f)
        
        args_resumed = args

    # Always train/log with the final merged arguments. When resuming, the
    # command-line overrides above should take effect immediately instead of
    # keeping the stale Namespace loaded from args.pickle.
    args_resumed = args
    
    _print_run_summary(args_resumed, label="Effective args")
    seed_everything(args_resumed.seed)

    ## Wandb config
    if args_resumed.no_wandb:
        mode = 'disabled'
    else:
        mode = 'online' if args_resumed.online else 'offline'
    os.makedirs(args_resumed.wandb_dir, exist_ok=True)
    resume_wandb_run_id = getattr(args_resumed, "wandb_run_id", None)
    should_resume_wandb = (
        (not args_resumed.no_wandb)
        and (args_resumed.resume_from_checkpoint not in (None, 'none'))
    )
    if should_resume_wandb and (resume_wandb_run_id is None):
        resume_wandb_run_id = _find_existing_wandb_run_id(args_resumed.wandb_dir)
    if should_resume_wandb:
        if resume_wandb_run_id is not None:
            print(f"Resuming W&B run {resume_wandb_run_id} from '{args_resumed.wandb_dir}'.")
        else:
            print(
                f"Checkpoint resume was requested, but no existing W&B run ID was found in "
                f"'{args_resumed.wandb_dir}'. Starting a new W&B run."
            )
    kwargs = {'entity': args_resumed.wandb_usr, 
              'name': args_resumed.train, 
              'project': args_resumed.wandb_project, 
              'config': args_resumed,
              'settings': wandb.Settings(
                  x_disable_stats=False, 
                  init_timeout=120, 
                  x_stats_sampling_interval=120,
                  x_save_requirements=False
                ),
              'reinit': "finish_previous",
              'mode': mode,
              'dir': args_resumed.wandb_dir,
              'save_dir': args_resumed.wandb_dir}
    if resume_wandb_run_id is not None:
        kwargs['id'] = resume_wandb_run_id
        kwargs['resume'] = 'must'
    wandb_logger = WandbLogger(log_model=False, **kwargs)
    wandb_logger.log_metrics({"debug_wandb_alive": 1.0}, step=0)
    wandb_summary = getattr(wandb_logger.experiment, "summary", None)
    if hasattr(wandb_summary, "update"):
        wandb_summary.update({"debug_wandb_summary_alive": 1})
    if (not args_resumed.no_wandb) and _is_rank_zero_process():
        experiment = wandb_logger.experiment
        run_id = _coerce_wandb_run_id(getattr(experiment, "id", None))
        if run_id is not None:
            args_resumed.wandb_run_id = run_id
            _persist_wandb_run_id(args_resumed.wandb_dir, run_id)
            _persist_args_pickle(os.path.join(final_folder, 'args.pickle'), args_resumed)
    main(args_resumed, wandb_logger)
