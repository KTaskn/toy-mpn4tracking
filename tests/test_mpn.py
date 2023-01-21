import torch
import torch.nn as nn
from torch_geometric.data import Data
from mpn import Layer, MPN

def generate_data():    
    edge_index = torch.tensor([[0, 1, 1, 2],
                               [1, 0, 2, 1]], dtype=torch.long)
    x = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=torch.float)

    return Data(x=x, edge_index=edge_index)
    

def test_mpn():
    data = generate_data()
    cel = nn.CrossEntropyLoss()
    mpn = MPN(num_layer=1)
    mpn.train()
    
    M = data.x
    H = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ], dtype=torch.float)
    
    labels = torch.tensor([[1], [1], [1], [1]], dtype=torch.float)
    predicts = mpn(M, H, data.edge_index)
    loss = cel(predicts, labels)
    loss.backward()
    


def test_mpnlayer():
    data = generate_data()
    layer = Layer()
    layer.eval()
    M = data.x
    H = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ], dtype=torch.float)
    with torch.no_grad():
        M, H, edge_index = layer(M, H, data.edge_index)
        