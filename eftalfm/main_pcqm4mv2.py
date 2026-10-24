from __future__ import annotations 

import pickle
import os 
import re 

import torch
from argparse import Namespace 

from lightning.pytorch import Trainer, seed_everything  
from model_training.lightning import (
    get_trainer_inputs,
    fit_trusted_checkpoint as _fit_trusted_checkpoint, 
    get_wandb_logger
)
from model_training.args_helper import merge_resume_args

from eftalfm.argparser_setup.pcqm4mv2 import add_pcqm4mv2_arguments
from eftalfm.argparser_setup.gm import add_gm_arguments
from eftalfm.argparser_setup.common import add_common_arguments
from eftalfm.argparser_setup.file_parser import ArgumentFileParser
from eftalfm.argparser_setup.vae import add_vae_arguments 
from eftalfm.data.pcqm4m_v2.data.data_plmodule import PCQM4MPLDataModule
from eftalfm.data.latent_data_plmodule import LatentPLDataModule
from eftalfm.modules.vae_plmodule import CloudVAEPLModule 

from eftalfm.get_model import get_vae_model, get_lgm_model

os.environ["TOKENIZERS_PARALLELISM"] = "true"

_WANDB_RUN_DIR_RE = re.compile(r"^(?:offline-)?run-[^-]+-([A-Za-z0-9]+)$")


def _print_run_summary(args:Namespace, label: str = "Run config") -> None:
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


def _get_checkpoint_dir(args:Namespace):
    """Resolve EF-TALFM's existing stage-specific checkpoint directory."""
    if args.train.startswith("vae"):
        return os.path.join(args.log_dir, args.train, "vae")
    if args.train.startswith("latent"):
        return os.path.join(args.log_dir, args.pretrained_vae_subfolder, "flow_matching", args.train)
    raise ValueError(f"args.train must start with 'vae' or 'latent'; received {args.train!r}")


##########################################################################
def main(args:Namespace, logger):
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
        ckpt_path=os.path.join(args.log_dir, args.pretrained_vae_subfolder, "flow_matching", args.train, args.resume_from_checkpoint) if (args.resume_from_checkpoint is not None) else None
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
        ckpt_path=os.path.join(args.log_dir, args.pretrained_vae_subfolder, "flow_matching", args.train, args.resume_from_checkpoint) if (args.resume_from_checkpoint is not None and args.resume_from_checkpoint != 'none') else None
    
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
            args_path=os.path.join(args.log_dir, args.pretrained_vae_subfolder, "flow_matching", args.train, 'args.pickle') 
            final_folder = os.path.join(args.log_dir, args.pretrained_vae_subfolder, "flow_matching", args.train)
            
        else:
            raise ValueError(f"args.train only accepts 'vae', 'latent'; received {args.train}")            # Load the checkpoint if exists
        
        print(f"Resuming from checkpoint {ckpt_path}...")
        if os.path.exists(args_path):
            with open(args_path, 'rb') as f:
                args_resumed = pickle.load(f)
            
            args = merge_resume_args(args_resumed, args)
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
            if not os.path.exists(os.path.join(vae_folderpath, "flow_matching")):
                os.makedirs(os.path.join(vae_folderpath, "flow_matching"))  
            if not os.path.exists(os.path.join(vae_folderpath, "flow_matching", args.train)):
                os.makedirs(os.path.join(vae_folderpath, "flow_matching", args.train))  
            final_folder=os.path.join(vae_folderpath, "flow_matching", args.train) 
                
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

    wandb_logger = get_wandb_logger(
        wandb_usr=args_resumed.wandb_usr,
        wandb_dir=args_resumed.wandb_dir,
        wandb_expname=args_resumed.train,
        wandb_project=args_resumed.wandb_project,
        wandb_run_id=getattr(args_resumed, "wandb_run_id", None),
        resume_from_checkpoint=args_resumed.resume_from_checkpoint,
        no_wandb=args_resumed.no_wandb,
        wandb_online=args_resumed.online,
        args=args_resumed,
        args_folder=final_folder
    )         
    main(args_resumed, wandb_logger)
