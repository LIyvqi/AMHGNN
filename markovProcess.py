import scipy.sparse as sparse
from dgl import backend as F
import numpy as np
import networkx as nx
import dgl
import torch

def matrix_norm(m):
    m = m*1.0/(m.sum(axis=0) + 0.0000001)
    return m

def inflation_oper(m,r):
    m = m ** r
    m = matrix_norm(m)
    return m

def markov_process_one(M,r,theta):
    M = M * M
    M = M.A
    M = matrix_norm(M)
    M = inflation_oper(M,r)
    M[M <= theta] = 0
    M = matrix_norm(M)
    return np.mat(M)


def get_markov_graph(graph,r,min_theta,epoch):
    A = 0
    for etype in graph.etypes:
        A = A + graph.adj(etype=etype).to_dense()

    invD = sparse.diags(F.asnumpy(sum(A)).clip(1) ** -1, dtype=float)
    A = sparse.csr_matrix(A)
    S = A * invD
    S = S.todense()

    mean_degree = np.count_nonzero(S) / S.shape[0]
    theta = (1.0 / mean_degree)

    print("markov process")
    for i in range(epoch):
        S = markov_process_one(S, r, theta)
        tem = S.A
        row, col = np.diag_indices_from(tem)
        tem[row, col] = 0

        ratios = np.count_nonzero(S) / S.shape[0]
        theta = min((1.0 / ratios), min_theta)

        if (i == (epoch-1)):
            src, dst = np.nonzero(S)
            G = nx.Graph()
            edges = [(i, j) for i, j in zip(src, dst)]
            G.add_edges_from(edges)

    add_self_loop = dgl.transforms.AddSelfLoop(new_etypes=True)
    graph = add_self_loop(graph)
    graph = dgl.add_edges(graph, torch.from_numpy(src), torch.from_numpy(dst), etype='self')
    graph = dgl.to_bidirected(graph, True)
    return graph