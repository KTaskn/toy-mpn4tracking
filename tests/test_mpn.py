import torch
import torch.nn as nn
from torch_geometric.data import Data
from mpn import Layer, MPN
import pandas as pd

def generate_data():    
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
                               [1, 0, 2, 1, 3, 2]], dtype=torch.long)
    x = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]], dtype=torch.float)
    H = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
        [2.0, 2.0, 2.0],
    ], dtype=torch.float)
    return Data(x=x, edge_index=edge_index), H
        

def test_mpn():
    data, H = generate_data()
    cel = nn.CrossEntropyLoss()
    mpn = MPN(num_layer=1)
    mpn.train()
    
    M = data.x    
    labels = torch.ones([H.size(0)], dtype=torch.long)    
    predicts = mpn(M, H, data.edge_index)
    
    assert labels.size(0) == H.size(0)
    assert len(labels.size()) == 1
    assert predicts.size(0) == H.size(0)
    assert predicts.size(1) == 2
    
    loss = cel(predicts, labels)
    loss.backward()


def test_mpnlayer():
    data, H = generate_data()
    layer = Layer()
    layer.eval()
    M = data.x
    with torch.no_grad():
        M, H, edge_index = layer(M, H, data.edge_index)
        