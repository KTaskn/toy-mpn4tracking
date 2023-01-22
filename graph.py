import torch
from torch_geometric.data import Data
from itertools import permutations
from functools import reduce
import pandas as pd
import numpy as np

COLUMNS = ["red", "green", "blue", "x", "y"]
def generate_graph(df):
    l_edge_index = [
        (bf, af)
        for time, bf_grp in df.groupby("time")
        for bf in bf_grp.index
        for af in df[df["time"] > time].index
    ]
    edge_index = torch.tensor(l_edge_index, dtype=torch.long).t()
    x = torch.tensor(df[COLUMNS].to_numpy(), dtype=torch.float)
    return Data(x=x, edge_index=edge_index)

ID_COLUMNS = ["id", "bf"]
def generate_labels(df):
    df = df[ID_COLUMNS].copy()
    df["vidx"] = df.index.tolist()
    df = pd.merge(
        df.pipe(lambda df: df[df["bf"] != "#"])[["bf", "vidx"]].rename(columns={"bf": "id"}),
        df[["id", "vidx"]],
        on="id"
    )
    return torch.tensor(
        df[["vidx_y", "vidx_x"]].to_numpy(),
        dtype=torch.long
    )
    
def generate_train_dataset(df):
    graph = generate_graph(df)
    labels = generate_labels(df)
    
    return graph, torch.tensor([row in labels.tolist() for row in graph.edge_index.T.tolist()], dtype=torch.long)