import torch
import torch.nn as nn
from torch_scatter import scatter
from torch_geometric.nn import MessagePassing

class Layer(MessagePassing):
    def __init__(self, node_dim=3, edge_dim=3):
        super().__init__(aggr='mean')
        dim = 2 * node_dim + edge_dim
        middle_e = 20
        self.mlp_e = nn.Sequential(
            nn.Linear(dim, middle_e),
            # nn.BatchNorm1d(num_features=middle_e),
            nn.ReLU(),
            nn.Linear(middle_e, edge_dim),
        )
        dim = node_dim + edge_dim + node_dim
        middle_v = 20
        self.mlp_past = nn.Sequential(
            nn.Linear(dim, middle_v),
            nn.ReLU(),
            nn.Linear(middle_v, node_dim),
        )
        self.mlp_future = nn.Sequential(
            nn.Linear(dim, middle_v),
            nn.ReLU(),
            nn.Linear(middle_v, node_dim),
        )
        dim = node_dim + edge_dim
        self.mlp_v = nn.Sequential(
            nn.Linear(dim, middle_v),
            nn.ReLU(),
            nn.Linear(middle_v, node_dim),
        )

    def forward(self, M, H, edge_index, init_M):
        return self.propagate(edge_index, M=M, H=H, init_M=init_M)

    def message(self, M_i, M_j, H, edge_index, init_M):
        past_idx, future_idx = edge_index
        H = self.mlp_e(torch.cat([M_i, M_j, H], dim=1))
        M_i = self.mlp_past(torch.cat([M_i, H, init_M[past_idx]], dim=1))
        M_j = self.mlp_future(torch.cat([M_j, H, init_M[future_idx]], dim=1))
        return M_i, M_j, H, edge_index
    
    def aggregate(self, inputs, index, ptr = None, dim_size = None):
        M_i, M_j, H, edge_index = inputs
        past, future = edge_index
        return self.mlp_v(torch.cat([
            super().aggregate(M_i, future, ptr, dim_size),
            super().aggregate(M_j, past, ptr, dim_size),
        ], dim=1)), H, edge_index
    
class MPN(nn.Module):
    def __init__(self, node_dim=3, edge_dim=3, num_layer=12):
        super().__init__()
        self.layer = Layer(node_dim, edge_dim)
        self.num_layer = num_layer
        middle = 20
        self.mlp = nn.Sequential(
            nn.Linear(edge_dim, middle),
            nn.ReLU(),
            nn.Linear(middle, 2)
        )
        
    def forward(self, M, H, edge_index):
        init_M = M.clone()
        for _ in range(self.num_layer):
            M, H, edge_index = self.layer(M, H, edge_index, init_M)
        return self.mlp(H)
        