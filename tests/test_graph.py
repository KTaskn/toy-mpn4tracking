from graph import generate_graph, generate_labels, generate_train_dataset
from torch_geometric.data import Data
import pandas as pd
import torch

def test_generate_graph():
    df = pd.read_csv("./tests/data/data01.csv")
    graph = generate_graph(df)
    assert type(graph) is Data
    assert graph.num_nodes == len(df.index)
    assert graph.num_edges == ((df["time"].count() * (df["time"].count() - 1)) - (df["time"].value_counts() *(df["time"].value_counts() - 1)).sum()) // 2
    assert graph.is_undirected() is False
    
class TestPattern:
    def test_data01(self):
        df = pd.read_csv("./tests/data/data01.csv")
        graph, labels = generate_train_dataset(df)
        assert graph.num_nodes == 2
        assert graph.num_edges == 1
        assert (labels == torch.tensor([1], dtype=torch.long)).all()


    def test_data02(self):
        df = pd.read_csv("./tests/data/data02.csv")
        graph, labels = generate_train_dataset(df)
        assert graph.num_nodes == 2
        assert graph.num_edges == 1
        assert (labels == torch.tensor([0], dtype=torch.long)).all()
        
    def test_data03(self):
        df = pd.read_csv("./tests/data/data03.csv")
        graph, labels = generate_train_dataset(df)
        assert graph.num_nodes == 3
        assert graph.num_edges == 2
        assert (labels == torch.tensor([1, 0], dtype=torch.long)).all()
        
    def test_data04(self):
        df = pd.read_csv("./tests/data/data04.csv")
        graph, labels = generate_train_dataset(df)
        assert graph.num_nodes == 4
        assert graph.num_edges == 5
        assert (labels == torch.tensor([1, 0, 0, 0, 1], dtype=torch.long)).all()
        
        
        
def test_generate_labels():    
    df = pd.read_csv("./tests/data/data01.csv")
    labels = generate_labels(df)
    # 順方向
    assert len(labels) == len(df.pipe(lambda df: df[df["bf"] != "#"]).index)
    
    graph = generate_graph(df)
    [row in labels.tolist() for row in graph.edge_index.T.tolist()]
    
def test_generate_train_dataset():
    df = pd.read_csv("./tests/data/data01.csv")
    graph, labels = generate_train_dataset(df)
    assert labels.size(0) == graph.num_edges
    assert len(labels.size()) == 1
    assert labels.dtype is torch.long
    