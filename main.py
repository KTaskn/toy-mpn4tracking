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
from sklearn.utils.class_weight import compute_class_weight

EPOCH = 100000
NUM_LAYER = 12
CUDA = True
EDGE_DIM = 5
if __name__ == "__main__":
    print("loading datasets")
    train_datasets = [generate_train_dataset(pd.read_csv(fname)) for fname in glob("data/train/*.csv")]
    test_datasets = [generate_train_dataset(pd.read_csv(fname)) for fname in glob("data/test/*.csv")]
    print("done loading datasets")
    mpn = MPN(node_dim=5, edge_dim=EDGE_DIM, num_layer=NUM_LAYER)
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
        f1_sum = 0
        for graph, labels in test_datasets:
            with torch.no_grad():
                H = (graph.x[graph.edge_index[0]] - graph.x[graph.edge_index[1]]).pow(2)
                
                graph = graph.cuda() if CUDA else graph
                labels = labels.cuda() if CUDA else labels
                H = H.cuda() if CUDA else H
                
                predicts = mpn(graph.x, H, graph.edge_index)
                f1 = f1_score(labels.cpu().numpy(), np.argmax(predicts.cpu().numpy(), axis=1))
            f1_sum += f1
                # print(confusion_matrix(labels.cpu().numpy(), np.argmax(predicts.cpu().numpy(), axis=1)))
        f1_avg = f1_sum / len(test_datasets)

        sum_loss = 0
        mpn.train()
        for graph, labels in train_datasets:
            H = (graph.x[graph.edge_index[0]] - graph.x[graph.edge_index[1]]).pow(2)
            class_weight = compute_class_weight('balanced', classes=[0, 1], y=labels.numpy())
            weights = torch.tensor(class_weight, dtype=torch.float)
            
            graph = graph.cuda() if CUDA else graph
            labels = labels.cuda() if CUDA else labels
            H = H.cuda() if CUDA else H
            weights = weights.cuda() if CUDA else weights
            
            criterion = nn.CrossEntropyLoss(weight=weights)
            criterion = criterion.cuda() if CUDA else criterion
                        
            predicts = mpn(
                graph.x,
                H,
                graph.edge_index
            )
            loss = criterion(predicts, labels)
                    
            optimizer.zero_grad()
            loss.backward()            
            optimizer.step()   
        sum_loss += loss.item()        
        pbar.set_postfix(loss=sum_loss, f1=f1_avg)
            