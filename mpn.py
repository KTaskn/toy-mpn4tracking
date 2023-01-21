import torch
import torch.nn as nn
from torch_scatter import scatter
from torch_geometric.nn import MessagePassing

class Layer(MessagePassing):
    def __init__(self, node_dim=3, edge_dim=3):
        super().__init__(aggr='add') #  "Max" aggregation.
        dim = 2 * node_dim + edge_dim
        self.mlp_e = nn.Sequential(
            nn.Linear(dim, 20),
            nn.BatchNorm1d(num_features=20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.BatchNorm1d(num_features=20),
            nn.ReLU(),
            nn.Linear(20, edge_dim),
        )
        self.mlp_v = nn.Sequential(
            nn.Linear(2 * node_dim, 20),
            nn.BatchNorm1d(num_features=20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.BatchNorm1d(num_features=20),
            nn.ReLU(),
            nn.Linear(20, node_dim),
        )

    def forward(self, M, H, edge_index):
        return self.propagate(edge_index, M=M, H=H)

    def message(self, M_i, M_j, H, edge_index):
        H = self.mlp_e(torch.cat([M_i, M_j, H], dim=1))
        M = self.mlp_v(torch.cat([M_i, H], dim=1))
        return M, H, edge_index
    
    def aggregate(self, inputs, index, ptr = None, dim_size = None):
        M, H, edge_index = inputs
        return super().aggregate(M, index, ptr, dim_size), H, edge_index
    
class MPN(nn.Module):
    def __init__(self, node_dim=3, edge_dim=3, num_layer=20):
        super().__init__()
        self.num_layer = num_layer
        self.layer = Layer(node_dim, edge_dim)
        self.mlp = nn.Sequential(
            nn.Linear(edge_dim, edge_dim),
            nn.BatchNorm1d(num_features=edge_dim),
            nn.ReLU(),
            nn.Linear(edge_dim, edge_dim),
            nn.BatchNorm1d(num_features=edge_dim),
            nn.ReLU(),
            nn.Linear(edge_dim, 2),
        )
        
    def forward(self, M, H, edge_index):
        for _ in range(self.num_layer):
            M, H, edge_index = self.layer(M, H, edge_index)
        return self.mlp(H)
        