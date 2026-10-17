import torch
from torch_geometric.utils import to_dense_adj, to_dense_batch

import os
import yaml
from dataset import DataInfos
from graphbpe import MolecularGraphTokenizer
from types import SimpleNamespace

def to_dense(x, edge_index, edge_attr, batch, y=None,max_num_nodes=None, edge_pos=None):
    X, node_mask = to_dense_batch(x=x, batch=batch, max_num_nodes=max_num_nodes)
    if max_num_nodes is None:
        max_num_nodes = X.size(1)
    E = to_dense_adj(edge_index=edge_index, batch=batch, edge_attr=edge_attr, max_num_nodes=max_num_nodes)
    E = encode_no_edge(E)

    if edge_pos is not None:
        pos = to_dense_adj(edge_index=edge_index, batch=batch, edge_attr=edge_pos, max_num_nodes=max_num_nodes)
        pos = encode_no_edge(pos)
    else:
        pos = None
    dense_data = PlaceHolder(X=X, E=E, pos=pos, y=y)
    return dense_data, node_mask

def encode_no_edge(E):
    assert len(E.shape) == 4
    if E.shape[-1] == 0:
        return E
    no_edge = torch.sum(E, dim=3) == 0
    first_elt = E[:, :, :, 0]
    first_elt[no_edge] = 1
    E[:, :, :, 0] = first_elt
    diag = torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
    E[diag] = 0
    return E

class PlaceHolder:
    def __init__(self, X, E, y=None, pos=None):
        self.X = X
        self.E = E
        self.y = y
        self.pos = pos

    def type_as(self, x: torch.Tensor):
        """ Changes the device and dtype of X, E, y. """
        self.X = self.X.type_as(x)
        self.E = self.E.type_as(x)
        return self

    def to(self, x: str):
        """ Changes the device and dtype of X, E, y. """
        self.X = self.X.to(x)
        self.E = self.E.to(x)
        if self.y:
            self.y = self.y.to(x)
        if self.pos is not None:
            self.pos = self.pos.to(x)
        return self
    
    def mask(self, node_mask, collapse=False):
        x_mask = node_mask.unsqueeze(-1)          # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)             # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)             # bs, 1, n, 1

        if collapse:
            self.X = torch.argmax(self.X, dim=-1)
            self.E = torch.argmax(self.E, dim=-1)

            self.X[node_mask == 0] = - 1
            self.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = - 1
            if self.pos is not None:
                self.pos = torch.argmax(self.pos, dim=-1)
                self.pos[(e_mask1 * e_mask2).squeeze(-1) == 0] = - 1
        else:
            self.X = self.X * x_mask
            self.E = self.E * e_mask1 * e_mask2
            if self.pos is not None:
                self.pos = self.pos * e_mask1 * e_mask2
            if not torch.allclose(self.E, torch.transpose(self.E, 1, 2)):
                print('E', self.E.shape, 'E.max', self.E.max(), 'E.min', self.E.min(), 'E.mean', self.E.mean(), 'E.std', self.E.std(), 'E.median', self.E.median())
            assert torch.allclose(self.E, torch.transpose(self.E, 1, 2))
        return self

def replace_first_indices(dense_data, node_mask_target, noisy_data):
    """
    Replace elements in dense_data with noisy_data based on node_mask_target.
    
    Args:
        dense_data: PlaceHolder object containing X, E, pos tensors
        node_mask_target: Boolean mask (bs, n) where True indicates positions to replace
        noisy_data: Dictionary containing 'X_t', 'E_t', 'pos_t' tensors
        
    Returns:
        PlaceHolder object with replaced values
    """
    # Clone the original data
    new_X = dense_data.X.clone()
    new_E = dense_data.E.clone()
    new_pos = dense_data.pos.clone() if dense_data.pos is not None else None
    new_y = dense_data.y.clone() if dense_data.y is not None else None
    
    # Create masks for node features and edge features
    x_mask = node_mask_target.unsqueeze(-1)          # bs, n, 1
    e_mask1 = x_mask.unsqueeze(2)                    # bs, n, 1, 1
    e_mask2 = x_mask.unsqueeze(1)                    # bs, 1, n, 1
    edge_mask = e_mask1 * e_mask2                    # bs, n, n, 1
    
    # Apply the masks to replace values
    masked_X_t = noisy_data['X_t'].to(new_X.dtype)
    # Replace only where mask is True, keep original values where mask is False
    new_X = torch.where(x_mask, masked_X_t, new_X)
    
    masked_E_t = noisy_data['E_t'].to(new_E.dtype)
    new_E = torch.where(edge_mask, masked_E_t, new_E)
    
    # if new_pos is not None and 'pos_t' in noisy_data:
    masked_pos_t = noisy_data['pos_t'].to(new_pos.dtype)
    new_pos = torch.where(edge_mask, masked_pos_t, new_pos)
    
    return PlaceHolder(X=new_X, E=new_E, y=new_y, pos=new_pos)

def dict_to_namespace(d):
    return SimpleNamespace(
        **{k: dict_to_namespace(v) if isinstance(v, dict) else v for k, v in d.items()}
    )

def load_config(config_path, data_dir='../data'):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as file:
        cfg_dict = yaml.safe_load(file)
    cfg = dict_to_namespace(cfg_dict)

    vocab_size = cfg.vocab_size
    vocab_ring_len = cfg.vocab_ring_len
    tokenizer_name = cfg.tokenizer_name
    tokenizer = MolecularGraphTokenizer(kekulize=True, name=tokenizer_name, simple_mode=False)
    output_file = f"{data_dir}/tokenizer/vocab{vocab_size}ring{vocab_ring_len}/{tokenizer_name}-token"
    tokenizer.load(output_file)

    data_info = DataInfos(cfg, tokenizer, data_dir)
    return cfg, data_info, tokenizer