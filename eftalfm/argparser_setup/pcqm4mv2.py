import argparse
### PCQM4MV2
def add_pcqm4mv2_arguments(parser:argparse.ArgumentParser):
    parser.add_argument(
        '--dataset_name',
        type=str,
        default='PCQM4M-LSC-V2-3D',
        help=(
            "Dataset mode, including PCQM4M, or LATENT-PCQM4M" 
        ),
    )

    parser.add_argument('--processed_filename', type=str, default='geometric_data_processed.pt', help="processed filename")
    parser.add_argument('--split_filename', type=str, default='split_dict.pt', help="split filename")
    parser.add_argument('--train_subset', type=str, default="train", help="data subset to use for training (e.g. train, valid, test)")
    parser.add_argument('--valid_subset', type=str, default="valid", help="data subset to use for validation (e.g. train, valid, test)")
    parser.add_argument('--skip_invalid_size_inputs_valid_test', action='store_true', 
                        default=False, help="ignore too long or too short lines in valid and test set")
    parser.add_argument('--required_batch_size_multiple', type=int, default=8, 
                        help="batch size will be a multiplier of this value") 
    parser.add_argument("--num_atoms", type=int, default=512*9, help="number of atom types in the graph")
    return parser
