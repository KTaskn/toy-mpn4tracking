from graph import generate_graph, generate_labels, generate_train_dataset
from torch_geometric.data import Data
import pandas as pd
import torch

def test_generate_graph():
    df = pd.read_csv("./tests/data/data01.csv")
    graph = generate_graph(df)
    assert type(graph) is Data
    assert graph.num_nodes == len(df.index)
    assert graph.num_edges == (df["time"].count() * (df["time"].count() - 1)) - (df["time"].value_counts() *(df["time"].value_counts() - 1)).sum()
    assert graph.is_directed() is False
    
class TestPattern:
    def test_data01(self):
        df = pd.read_csv("./tests/data/data01.csv")
        graph, labels = generate_train_dataset(df)
        assert graph.num_nodes == 2
        assert graph.num_edges == 1 * 2
        assert (labels == torch.tensor([1, 1], dtype=torch.long)).all()


    def test_data02(self):
        df = pd.read_csv("./tests/data/data02.csv")
        graph, labels = generate_train_dataset(df)
        assert graph.num_nodes == 2
        assert graph.num_edges == 2
        assert (labels == torch.tensor([0, 0], dtype=torch.long)).all()
        
    def test_data03(self):
        df = pd.read_csv("./tests/data/data03.csv")
        graph, labels = generate_train_dataset(df)
        assert graph.num_nodes == 3
        assert graph.num_edges == 2 * 2
        assert (labels == torch.tensor([1, 0, 1, 0], dtype=torch.long)).all()
        
    def test_data04(self):
        df = pd.read_csv("./tests/data/data04.csv")
        graph, labels = generate_train_dataset(df)
        assert graph.num_nodes == 4
        assert graph.num_edges == 5 * 2
        assert (labels == torch.tensor([1, 0, 0, 0, 1, 0, 1, 0, 0, 1], dtype=torch.long)).all()
        
        
        
def test_generate_labels():    
    df = pd.read_csv("./tests/data/data01.csv")
    labels = generate_labels(df)
    # 両方向 [[1, 2], [2, 1]]
    assert len(labels) == len(df.pipe(lambda df: df[df["bf"] != "#"]).index) * 2
    bf = labels[:len(labels)//2]
    af = labels[len(labels)//2:]
    af = torch.vstack((af[:, 1], af[:, 0])).T
    assert (bf == af).all()
    
    graph = generate_graph(df)
    [row in labels.tolist() for row in graph.edge_index.T.tolist()]
    
def test_generate_train_dataset():
    df = pd.read_csv("./tests/data/data01.csv")
    graph, labels = generate_train_dataset(df)
    assert labels.size(0) == graph.num_edges
    assert len(labels.size()) == 1
    assert labels.dtype is torch.long
    