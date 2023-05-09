import numpy as np
import dgl
from collections import defaultdict
import torch

def get_env_label(graph,train_mask,env_theta,train_ratio):
    homo_g = dgl.to_homogeneous(graph, ndata=['label'])
    src, dst = homo_g.edges()
    labels = homo_g.ndata['label']
    label_equal = labels[src] == labels[dst]
    graph_dict = defaultdict(set)
    for i, j in zip(src.numpy(), dst.numpy()):
        graph_dict[i].add(j)
    train_set = set()
    for i, flag in zip(range(train_mask.shape[0]), train_mask):
        if (flag):
            train_set.add(i)
    new_label = np.zeros_like(labels)
    for vid in graph_dict.keys():
        nbrs = graph_dict[vid]
        if vid not in train_set:
            continue
        nbrs_train = nbrs & train_set
        equal_num = sum(labels[list(nbrs_train)] != labels[vid]) / (train_ratio * len(nbrs))
        new_label[vid] = 1 if equal_num >= env_theta else -1

    print("Number of anomalous vertices in the training set: ",sum(labels[train_mask]))
    print("Number of vertices with abnormal vertex labels and abnormal environment labels at the same time: ",sum( (labels[train_mask] == torch.tensor(new_label)[train_mask]) & (labels[train_mask] == 1) ))
    print(sum( (labels[train_mask] == torch.tensor(new_label)[train_mask]) & (labels[train_mask] == 1)))
    print("Number of vertices with abnormal environment labels:",sum(new_label == 1))

    return new_label
