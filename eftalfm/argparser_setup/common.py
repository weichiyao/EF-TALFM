from argparse import ArgumentParser


def _int_or_float(value: str):
    """Preserve integer batch counts while still accepting float fractions."""
    try:
        if all(marker not in value.lower() for marker in (".", "e")):
            return int(value)
        return float(value)
    except ValueError as exc:
        raise ValueError(f"expected int batch count or float fraction, got {value!r}") from exc

def add_common_arguments(parser:ArgumentParser):
    parser.add_argument('--compile_model', action='store_true', default=False, help="whether to compile the model")
    parser.add_argument('--exp_name', type=str, default='debug_0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--property_name', type=str, default='y', help="property name")
    
    parser.add_argument('--data_dir', type=str, default='./temp',
                        help='data directory')
    parser.add_argument('--n_small_train', type=int, default=-1,
                    help='select a small number of instances for training')
    parser.add_argument('--n_small_test', type=int, default=-1,
                        help='select a small number of instances for test')
    parser.add_argument('--n_small_valid', type=int, default=-1,
                        help='select a small number of instances for validation')   

    parser.add_argument('--context_dim', type=int, default=0)
    parser.add_argument('--context_name', type=str, default="y", help="context name")
    parser.add_argument('--context_idx', type=str, default="*", help="either '*', '0' or '0,1,2'")
    parser.add_argument('--context_ws', type=str, default="*", help="'*', '1', or '1,1,1'")
    parser.add_argument('--dtype', type=str, default="float", help="'double' | 'float'.")
    
    parser.add_argument('--max_n_nodes', type=int, default=128)
    
    ## Training 
    parser.add_argument("--standparam", action='store_true', default=False, help="for both width and depth, default implements mu_p and complete_p")
    parser.add_argument('--init_std', type=float, default=0.02)
    parser.add_argument('--track_grad_norms', action='store_true')
    parser.add_argument('--precision', default='32-true', 
                        help="see lighning Trainer. '16-mixed' for FP16.")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--max_epochs', type=int, default=2)
    parser.add_argument('--min_epochs', type=int, default=1) 
    parser.add_argument('--max_steps', type=int, default=-1)
    parser.add_argument('--num_decay_steps', type=int, default=-1)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--dataloader_prefetch_factor', type=int, default=2,
                        help='number of batches each DataLoader worker prefetches')
    parser.add_argument('--disable_persistent_workers', action='store_true', default=False,
                        help='disable persistent DataLoader workers')
    parser.add_argument('--disable_pin_memory', action='store_true', default=False,
                        help='disable DataLoader pinned-memory copies')
    parser.add_argument('--limit_val_batches', type=_int_or_float, default=None)
    parser.add_argument('--limit_train_batches', type=_int_or_float, default=None)
    parser.add_argument('--log_interval', type=int, default=1, 
                        help="check_val_every_n_epoch in Trainer.")
    parser.add_argument('--log_every_n_steps', type=int, default=500, 
                        help="log_every_n_steps in Trainer.")
    parser.add_argument('--early_stopping_patience', type=int, default=10000, 
                        help="patience for early stopping in Trainer.")
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                        help="checkpoint name to be resumed from")
    parser.add_argument('--init_from_checkpoint', type=str, default=None,
                        help="checkpoint path used for weights-only initialization before training")
    parser.add_argument('--init_from_checkpoint_strict', action='store_true', default=False,
                        help="strictly load all weights from init_from_checkpoint")
    parser.add_argument('--log_dir', type=str, default=None, 
                        help="default_root_dir in Trainer.")
    parser.add_argument('--device', type=str, default="gpu", 
                        help="used in Trainer, 'gpu', 'mps', 'cpu', 'auto'.")
    parser.add_argument('--n_devices', type=int, default=1, 
                        help="number of devices per node to be used in Trainer.")
    parser.add_argument('--num_nodes', type=int, default=1, 
                        help="number of nodes to be used in Trainer.")
    parser.add_argument('--use_ddp', action='store_true', default=False, 
                        help="whether to use distributed data parallel strategy in Trainer.")
    parser.add_argument('--progress_bar', action='store_true', default=False, 
                        help="enable_progress_bar in Trainer.")
    parser.add_argument('--save_last_model_only', action='store_true', default=False, 
                        help="only save the last checkpoint in Trainer.")
    parser.add_argument('--gradient_clip_adaptive', default="none", 
                        help=" ema | queue ")
    parser.add_argument('--gradient_clip_val', type=float, default=0,
                        help="default don't clip")
    parser.add_argument('--gradient_clip_algorithm', type=str, default='norm', 
                        help="'norm' (clip gradients' global norm) or 'value' (clip gradients' maximum magnitude).")
    parser.add_argument('--accumulate_grad_batches', type=int, default=1, 
                        help="number of batches to be accumulated for gradient computation.")
    parser.add_argument('--do_test_validation_step', action='store_true', default=False)
    
    parser.add_argument(
        "--step_val_interval",
        type=int,
        default=-1,
        help=(
            "For step-limited training (max_steps > 0 and max_epochs < 0), "
            "validate every N optimizer steps when > 0. This avoids dataset-size "
            "dependent epoch validation."
        ),
    )
    parser.add_argument('--ema', action='store_true', default=False) 
    ### Optimization & lr_scheduler
    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--adam_eps', type=float, default=1e-8,  help="epsilon for Adam optimizer")
    parser.add_argument('--adam_betas', type=eval, default='(0.9,0.999)', help="betas in AdamW optimizer.")
    parser.add_argument('--lr_scheduler', type=str, default='polynomial_decay', help=" 'constant' | 'cosine' | 'polynomical_decay' | 'linear'")
    parser.add_argument('--warmup_proportion', type=float, default=0.1, help="The proportion of the total training steps for warming up.")      
    parser.add_argument('--lr_end', type=float, default=1e-9, help="The learning rate to decay to in polynomial_decay.")
    parser.add_argument('--lr_interval', type=str, default="step", help="The learning rate scheduler update, 'step' or 'epoch'.")
    parser.add_argument('--power', type=float, default=1., help="The decay exponent in polynomial_decay.")
    parser.add_argument('--num_cycles', type=float, default=0.5, help="The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0 following a half-cosine).")
    parser.add_argument('--lr_gamma', type=float, default=0.5, help="gamma in torch.optim.lr_scheduler.StepLR")
    parser.add_argument('--lr_step_size', type=int, default=100, help="step_size (epochs or steps) in torch.optim.lr_scheduler.StepLR or CosineAnealing")
    
    ### Saving
    parser.add_argument('--out_dir', type=str, default='./outputs/', metavar='N',
                        help='folder to output')
            
    ## Wandb
    parser.add_argument('--wandb_usr', type=str, default='weichiy')
    parser.add_argument('--wandb_project', type=str, default='hello_world')
    parser.add_argument('--wandb_dir', type=str, default='./wandb/')
    parser.add_argument('--no_wandb', action='store_true', help='Disable wandb')
    parser.add_argument('--online', action='store_true', help='True = wandb online -- False = wandb offline')
    
    return parser
