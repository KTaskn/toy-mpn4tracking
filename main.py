import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from mpn import Layer, MPN
from graph import generate_train_dataset
import numpy as np
from tqdm import tqdm
from glob import glob
from sklearn.metrics import f1_score, confusion_matrix

EPOCH = 100000
NUM_LAYER = 3
CUDA = True
if __name__ == "__main__":
    print("loading datasets")
    train_datasets = [generate_train_dataset(pd.read_csv(fname)) for fname in glob("data/train/*.csv")[:1]]
    test_datasets = [generate_train_dataset(pd.read_csv(fname)) for fname in glob("data/test/*.csv")]
    print("done loading datasets")
    mpn = MPN(node_dim=5, edge_dim=5, num_layer=NUM_LAYER)
    mpn = mpn.cuda() if CUDA else mpn
    
    optimizer = optim.Adam(
        mpn.parameters(),
        lr=3e-4,
        weight_decay=1e-4,
        betas=(0.9, 0.999),
    )
        
    pbar = tqdm(range(EPOCH))
    for _ in pbar:        
        mpn.eval()
        for graph, labels in train_datasets:
            with torch.no_grad():
                H = torch.zeros([graph.num_edges, 5], dtype=torch.float)
                
                graph = graph.cuda() if CUDA else graph
                labels = labels.cuda() if CUDA else labels
                H = H.cuda() if CUDA else H
                
                predicts = mpn(graph.x, H, graph.edge_index)
                f1 = f1_score(labels.cpu().numpy(), np.argmax(predicts.cpu().numpy(), axis=1))
                print(confusion_matrix(labels.cpu().numpy(), np.argmax(predicts.cpu().numpy(), axis=1)))

        sum_loss = 0
        mpn.train()
        for graph, labels in train_datasets:
            H = torch.zeros([graph.num_edges, 5], dtype=torch.float)
            weights = torch.tensor([labels.sum(), len(labels) - labels.sum()], dtype=torch.float)
            
            graph = graph.cuda() if CUDA else graph
            labels = labels.cuda() if CUDA else labels
            H = H.cuda() if CUDA else H
            weights = weights.cuda() if CUDA else weights
            
            criterion = nn.CrossEntropyLoss(weight=weights)
            criterion = criterion.cuda() if CUDA else criterion
            
            optimizer.zero_grad()
            
            predicts = mpn(graph.x, H, graph.edge_index)
            loss = criterion(predicts, labels)
                    
            loss.backward()            
            optimizer.step()           
            sum_loss += loss.item() 
        pbar.set_postfix(loss=sum_loss, f1=f1)
            