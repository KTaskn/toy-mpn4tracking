import torch
from torch_geometric.data import Data
from itertools import permutations
from functools import reduce
import pandas as pd
import numpy as np

COLUMNS = ["red", "green", "blue", "x", "y"]
def generate_graph(df):
    # すべてのくみあわせ
    combis = permutations(df.index.tolist(), 2)
    # 同じ時間は除く
    sames = reduce(lambda a, b: a + b, [list(permutations(grp.index.tolist(), 2)) for idx, grp in df.groupby("time")])
    edge_index = torch.tensor(list(set(combis) - set(sames)), dtype=torch.long).t()
    x = torch.tensor(df[COLUMNS].to_numpy(), dtype=torch.float)
    return Data(x=x, edge_index=edge_index)

ID_COLUMNS = ["id", "bf"]
def generate_labels(df):
    df = df[ID_COLUMNS]
    df["vidx"] = df.index.tolist()
    df = pd.merge(
        df.pipe(lambda df: df[df["bf"] != "#"])[["bf", "vidx"]].rename(columns={"bf": "id"}),
        df[["id", "vidx"]],
        on="id"
    )
    return torch.tensor(
        np.vstack((
            df[["vidx_x", "vidx_y"]].to_numpy(),
            df[["vidx_y", "vidx_x"]].to_numpy()
        )),
        dtype=torch.long
    )
    
def generate_train_dataset(df):
    graph = generate_graph(df)
    labels = generate_labels(df)
    
    return graph, torch.tensor([row in labels.tolist() for row in graph.edge_index.T.tolist()], dtype=torch.long)