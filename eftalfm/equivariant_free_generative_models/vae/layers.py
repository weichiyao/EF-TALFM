 
import torch 
from torch import nn
from torch import Tensor

from eftalfm.equivariant_free_generative_models.models.utils import get_activation_fn, fixed_dimension_init_weights
 
from eftalfm.data.pcqm4m_v2.data.wrapper import allowable_features

N_TOTAL_CHARGE_TYPES = len(allowable_features['possible_total_formal_charge_list'])   
N_SPIN_MULTIPLICITY_TYPES = len(allowable_features['possible_spin_multiplicity_list'])  
N_NUMBER_OF_RINGS_TYPES = len(allowable_features['possible_number_of_rings_list'])      
N_HAS_FUSED_RINGS_TYPES = len(allowable_features['possible_has_fused_rings_list'])       



class ParticleFeature(nn.Module):
    """
    Compute atom features for each atom in the molecule.
    """

    def __init__(
            self,   
            hidden_dim:int, 
            num_atoms:int,          # 128 (# atoms types) + offset * (# extra features) 
            context_dim: int=0, 
            max_seq_len:int=512,
            use_position_embeddings:bool=False,  
            init_std:float=0.02,    
        ):
        super().__init__() 
        self.num_atoms = num_atoms
        self.use_position_embeddings = use_position_embeddings
 
        self.x_encoder   = nn.Embedding(num_atoms,  hidden_dim, padding_idx=0) 
        self.pos_encoder = nn.Linear(3, hidden_dim)
        self.graph_token = nn.Embedding(1, hidden_dim)

        self.context_dim = context_dim
        if self.context_dim > 0:
            self.context_encoder = nn.Linear(self.context_dim, hidden_dim)
        
        
        self.charge_encoder = nn.Embedding(N_TOTAL_CHARGE_TYPES, hidden_dim)
        self.spin_multiplicity_encoder = nn.Embedding(N_SPIN_MULTIPLICITY_TYPES, hidden_dim)
        self.number_of_rings_encoder = nn.Embedding(N_NUMBER_OF_RINGS_TYPES, hidden_dim)
        self.has_fused_rings_encoder = nn.Embedding(N_HAS_FUSED_RINGS_TYPES, hidden_dim)
         
        if self.use_position_embeddings:
            self.pte = nn.Embedding(max_seq_len + 1, hidden_dim, padding_idx=0)
        
        self.apply(lambda module: fixed_dimension_init_weights(module, init_std=init_std))
 
    def forward(
        self, 
        x:Tensor, 
        position:Tensor=None,
        padding_mask:Tensor=None, 
        charge:Tensor=None, 
        spin_multiplicity:Tensor=None, 
        number_of_rings:Tensor=None, 
        has_fused_rings:Tensor=None, 
        y:Tensor=None,
        **kwargs
    ) -> Tensor:
        """
        - padding_mask : (B, n_atoms+1) whether the locations are zero-padded
        - position     : (B, n_node, 3) 
        - x            : (B, n_node, 3)
        - charge            : (B, ) optional
        - spin_multiplicity : (B, ) optional
        - number_of_rings   : (B, ) optional
        - has_fused_rings   : (B, ) optional
        - spacegroup        : (B, ) optional
        - lengths_scaled    : (B, 1, 3) optional
        - angles_radians    : (B, 1, 3) optional
        - frac_coords       : (B, n_node, 3) optional
        - dataset_idx       : (B, ) optional
        - y                 : (B, context_dim) optional
        - padding_mask      : (B, 1+n_node) whether the locations are zero-padded
        
        """ 
        n_graph, n_node = x.size()[:2]
        node_feature = self.x_encoder(x).sum(dim=-2)     # [n_graph, n_node, d_x, n_hidden] -> [n_graph, n_node, n_hidden]
          
        if position is None:
            raise ValueError(
                "position is required when num_mol_datasets > 0."
            )
        pos_emb = self.pos_encoder(position)  # (bz, max_n_nodes, emb_dim)
         
        # node-level features 
        node_feature = node_feature + pos_emb     # [n_graph, n_node, n_hidden] 
        
        if self.use_position_embeddings: 
            # Construct position embedding 
            position_ids = torch.arange(0, n_node, device=x.device).unsqueeze(0) #  (1, n_node) 
            node_feature = node_feature + self.pte(position_ids)

        # cloud-level features  
        node_feature += self.charge_encoder(charge).unsqueeze(1)                       # [n_graph, 1, n_hidden]
        node_feature += self.spin_multiplicity_encoder(spin_multiplicity).unsqueeze(1) # [n_graph, 1, n_hidden]
        node_feature += self.number_of_rings_encoder(number_of_rings).unsqueeze(1)     # [n_graph, 1, n_hidden]
        node_feature += self.has_fused_rings_encoder(has_fused_rings).unsqueeze(1)     # [n_graph, 1, n_hidden]

        if self.context_dim > 0:
            node_feature += self.context_encoder(y).unsqueeze(1)  # [n_graph, 1, n_hidden]
        
        # global features 
        graph_token_feature = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)

        # combine the global and node features
        graph_node_feature = torch.cat([graph_token_feature, node_feature], dim=1)
        
        # mask the padding inputs --  self.use_position_embeddings=True
        graph_node_feature = graph_node_feature.masked_fill(
            padding_mask.unsqueeze(-1).to(torch.bool), 0.0
        ) 

        return graph_node_feature


class FinetuneHead(nn.Module):
    """Head for classification tasks."""

    def __init__(self, input_dim, inner_dim, output_dim, activation_fn, init_std=0.02):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, inner_dim)
        self.activation_fn = get_activation_fn(activation_fn)
        self.linear2 = nn.Linear(inner_dim, output_dim)
        self.init_std = init_std
        self.apply(lambda module: fixed_dimension_init_weights(module, init_std=init_std))

    def forward(self, x, **kwargs):
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.linear2(x)
        return x

