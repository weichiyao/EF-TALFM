from __future__ import annotations
import os 

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
 
import lightning.pytorch as pl

from eftalfm.data.utils import RobustScalerTransform, StandardScalerTransform, MinMaxScalerTransform, get_context

class LatentDataset(Dataset):
    def __init__(self, data_dict:dict[str,Tensor], context_name:str="y", context_idx:str="*"):
        """
        Args:
            z (torch.Tensor): Tensor of shape (n, 1, d)
            y (torch.Tensor): Tensor of shape (n,)
        """
        self.context_name = context_name
 
        # DataLoader workers are forked; indexing CUDA tensors in __getitem__ triggers
        # CUDA init in the child and raises AcceleratorError. Keep all fields on CPU.
        self.data_dict = {
            k: v.detach().cpu() if isinstance(v, Tensor) else v
            for k, v in data_dict.items()
        }
        
        # (Optional) store keys in a list so we know the order of items
        self.keys = list(self.data_dict.keys())

        # Check they all have same length
        self.size = self.data_dict[self.keys[0]].size(0)
        for k in self.keys:
            assert self.data_dict[k].size(0) == self.size, \
                f"All tensors must have same length, but {k} differs."
        
        if context_idx != "*":
            context_idx = torch.tensor([int(idx) for idx in context_idx.split(",")], dtype=torch.long)
            y = data_dict[context_name].reshape(self.size, -1)
            self.data_dict[context_name] = get_context(y, context_idx)
    
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        """
        Return a tuple or a dictionary. 
        - If returning a dict, your DataLoader collate_fn must handle it (default collate 
          can handle dicts if the structure is consistent).
        - If returning a tuple, define the order of items you want.
        """
        # Example: return a dictionary
        item = {}
        for k in self.keys:
            item[k] = self.data_dict[k][idx]
        return item 
    

class LatentPLDataModule(pl.LightningDataModule):
    def __init__(self, cfg, train_file=None, valid_file=None, test_file=None, dummy:bool=False, model_type:str="gm"):
        super().__init__() 

        self.cfg = cfg
        self.dummy = dummy 
        if not self.dummy:
            if model_type == "gm":
                self.train_file  = train_file if train_file is not None else os.path.join(self.cfg.log_dir, self.cfg.pretrained_vae_subfolder, 'vae', self.cfg.gm_latent_train_file)
                self.valid_file  = valid_file if valid_file is not None else os.path.join(self.cfg.log_dir, self.cfg.pretrained_vae_subfolder, 'vae', self.cfg.gm_latent_valid_file)
                self.test_file   = test_file  if test_file is not None  else os.path.join(self.cfg.log_dir, self.cfg.pretrained_vae_subfolder, 'vae', self.cfg.gm_latent_test_file) 
                normalize_y      = self.cfg.gm_normalize_y if hasattr(self.cfg, "gm_normalize_y") else False
                normalize_y_mode = self.cfg.gm_normalize_y_mode if hasattr(self.cfg, "gm_normalize_y_mode") else "rs"
            elif model_type == "pm":
                self.train_file  = train_file if train_file is not None else os.path.join(self.cfg.log_dir, self.cfg.pretrained_vae_subfolder, 'vae', self.cfg.pm_latent_train_file)
                self.valid_file  = valid_file if valid_file is not None else os.path.join(self.cfg.log_dir, self.cfg.pretrained_vae_subfolder, 'vae', self.cfg.pm_latent_valid_file)
                self.test_file   = test_file  if test_file is not None  else os.path.join(self.cfg.log_dir, self.cfg.pretrained_vae_subfolder, 'vae', self.cfg.pm_latent_test_file) 
                normalize_y      = self.cfg.pm_normalize_y if hasattr(self.cfg, "pm_normalize_y") else False
                normalize_y_mode = self.cfg.pm_normalize_y_mode if hasattr(self.cfg, "pm_normalize_y_mode") else "rs"
            else:
                raise ValueError(f"model_type must be 'gm' or 'pm'; received {model_type}")
            
            self.get_pretransform(normalize_y, normalize_y_mode)
    
    def setup(self, stage: str | None = None):
        """
        Load data from files and set up datasets.
        z: (n_train, 1, d)
        y: (n_train,)

        Args:
            stage (str, optional): Stage to set up. Can be 'fit', 'validate', 'test', or 'predict'.
        """
        
        if stage == 'fit' or stage is None:
            # Load training data
            train_data = torch.load(self.train_file, weights_only=True, map_location="cpu")
            self.train_dataset = LatentDataset(train_data, context_name=self.cfg.context_name, context_idx=self.cfg.context_idx)

            # Load validation data 
            val_data = torch.load(self.valid_file, weights_only=True, map_location="cpu") 
            self.val_dataset = LatentDataset(val_data, context_name=self.cfg.context_name, context_idx=self.cfg.context_idx)

        if stage == 'test' or stage is None:
            # Load test data
            test_data = torch.load(self.test_file, weights_only=True, map_location="cpu")  
            self.test_dataset = LatentDataset(test_data, context_name=self.cfg.context_name, context_idx=self.cfg.context_idx)


    def get_pretransform(self, normalize_y:bool=False, normalize_y_mode="rs"):
        train_data = torch.load(self.train_file, weights_only=True, map_location="cpu")
        context_name = self.cfg.context_name if hasattr(self.cfg, "context_name") else "y"
        
        b = train_data['z'].shape[0]
        y_train = train_data[context_name].reshape(b, -1)
        if self.cfg.context_idx != "*":
            context_idx = torch.tensor([int(idx) for idx in self.cfg.context_idx.split(",")], dtype=torch.long)
            y_train = get_context(y_train, context_idx)
    
        self.pretransform_y = None 
        self.invtransform_y = None
        
        if normalize_y:
            if normalize_y_mode == 'rs':
                transform_y_ = RobustScalerTransform()
                transform_y_.fit(y_train)
            elif normalize_y_mode == 'ss':
                transform_y_ = StandardScalerTransform()
                transform_y_.fit(y_train)
            elif normalize_y_mode == 'ms':
                transform_y_ = MinMaxScalerTransform()
                transform_y_.fit(y_train)
            else:
                raise ValueError(f"Only accept normalize_y_mode in 'rs', 'ss', 'ms'. Received {normalize_y_mode}")

            self.pretransform_y = transform_y_.transform
            self.invtransform_y = transform_y_.inverse_transform

    
    def train_dataloader(self):
        """
        Return the training DataLoader.
        """
        return DataLoader(self.train_dataset, 
                          batch_size=self.cfg.batch_size, 
                          shuffle=True, 
                          num_workers=self.cfg.num_workers, 
                          pin_memory=torch.cuda.is_available())


    def val_dataloader(self):
        """
        Return the validation DataLoader.
        """
        return DataLoader(self.val_dataset, 
                          batch_size=self.cfg.batch_size, 
                          shuffle=False, 
                          num_workers=self.cfg.num_workers, 
                          pin_memory=torch.cuda.is_available())


    def test_dataloader(self):
        """
        Return the test DataLoader.
        """
        return DataLoader(self.test_dataset, 
                          batch_size=self.cfg.batch_size, 
                          shuffle=False, 
                          num_workers=self.cfg.num_workers, 
                          pin_memory=torch.cuda.is_available())
