
import numpy as np

import torch
from torch import Tensor, LongTensor
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler


import contextlib
 
def get_context(y: Tensor, context_idx: LongTensor):
        """
        y: (b, d_y)
        """
        if context_idx is not None: 
            context = y.index_select(dim=1, index=context_idx) 
        return context


class RobustScalerTransform():
    def fit(self, x:Tensor): 
        rs = RobustScaler()
        n = x.shape[0]
        rs = rs.fit(x.reshape(n,-1))
        self.center_ = torch.from_numpy(rs.center_)
        self.scale_  = torch.from_numpy(rs.scale_)

    def transform(self, X:Tensor):
        return (X - self.center_.to(X)) / self.scale_.to(X)
    
    def inverse_transform(self, X:Tensor):
        return X * self.scale_.to(X) + self.center_.to(X)


class StandardScalerTransform():

    def fit(self, x:Tensor): 
        ss = StandardScaler()
        n = x.shape[0]
        ss = ss.fit(x.reshape(n,-1))
        
        self.mean_  = torch.from_numpy(ss.mean_)
        self.scale_ = torch.from_numpy(ss.scale_)

    def transform(self, X:Tensor):
        return (X - self.mean_.to(X)) / self.scale_.to(X)
    
    def inverse_transform(self, X:Tensor):
        return X * self.scale_.to(X) + self.mean_.to(X)


class MinMaxScalerTransform():
    def fit(self, x:Tensor): 
        ss = MinMaxScaler(feature_range=(-1, 1))
        n = x.shape[0]
        ss = ss.fit(x.reshape(n,-1))
        self.min_  = torch.from_numpy(ss.min_)
        self.scale_ = torch.from_numpy(ss.scale_)
    def transform(self, X:Tensor):
        return X * self.scale_.to(X) + self.min_.to(X)
    def inverse_transform(self, X:Tensor):
        return (X - self.min_.to(X)) / self.scale_.to(X) 


@contextlib.contextmanager
def numpy_seed(seed, *addl_seeds):
    """Context manager which seeds the NumPy PRNG with the specified seed and
    restores the state afterward"""
    if seed is None:
        yield
        return
    if len(addl_seeds) > 0:
        seed = int(hash((seed, *addl_seeds)) % 1e6)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def get_distances(positions, unique=False, offset=1, do_addition=False):
    """
    positions: (..., n_nodes, 3)
    """
    n_nodes = positions.shape[-2]
    if isinstance(positions, np.ndarray):
        positions = torch.from_numpy(positions)
    # Construct pairwise distance matrix and compute the distance to the origin
    distance_to_origin = torch.norm(positions, dim=-1, keepdim=True)
    pairwise_distances = torch.norm(positions[..., None, :] - positions[..., None, :, :], dim=-1, keepdim=True)
    if do_addition:
        pairwise_daddition = torch.norm(positions[..., None, :] + positions[..., None, :, :], dim=-1, keepdim=True)
    else:
        pairwise_daddition = torch.zeros_like(pairwise_distances)
    
    if unique:
        upper_triangular_indices = torch.triu_indices(n_nodes, n_nodes, offset=offset)
        pairwise_distances = pairwise_distances[..., upper_triangular_indices[0], upper_triangular_indices[1], :]   # (bs, n_nodes*(n_nodes+1)//2, d)  
        pairwise_daddition = pairwise_daddition[..., upper_triangular_indices[0], upper_triangular_indices[1], :]   # (bs, n_nodes*(n_nodes+1)//2, d)  
    return {"distance_to_origin" : distance_to_origin, 
            "pairwise_distances" : pairwise_distances, 
            "pairwise_daddition" : pairwise_daddition}
