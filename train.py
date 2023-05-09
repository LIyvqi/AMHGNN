import argparse
import torch
import torch.nn.functional as F
from GetData import GetData
import numpy as np
from models import AMHGNN
from utils import get_mf1_auc_aucpr
import random
import os
from markovProcess import get_markov_graph

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)

def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp

def train(model, G, labels, args, train_mask, val_mask, test_mask,beta):
    print('train/dev/test samples: ', train_mask.sum().item(), val_mask.sum().item(), test_mask.sum().item())
    weight = (1-labels[train_mask]).sum().item() / labels[train_mask].sum().item()
    print('cross entropy weight: ', weight)
    weight = torch.tensor([1.0, weight]).to(device)
    beta = torch.tensor(beta).to(device)

    best_mf1, best_auc, best_auc_pr, test_mf1, test_auc, test_auc_pr,best_val_mif1,test_mif1= 0, 0, 0, 0, 0 ,0, 0,0
    for epoch in np.arange(args.n_epoch) + 1:
        model.train()
        logits,mergin_loss = model(G, "review")
        loss = F.cross_entropy(logits[train_mask], labels[train_mask].to(device), weight) + beta * mergin_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        scheduler.step()

        if epoch % 5 == 0:
            with torch.no_grad():
                model.eval()
                logits,_ = model(G, "review")
                probs = logits.softmax(1).cpu()
                (vmf1, vauc, vauc_pr), (tmf1, tauc, tauc_pr), trec, tpre, train_mf1,v_mif1,t_mif1 = get_mf1_auc_aucpr(labels, probs,train_mask,val_mask,test_mask)
                if (best_mf1 + best_auc + best_auc_pr < vmf1 + vauc + vauc_pr):
                    best_mf1, best_auc, best_auc_pr,best_val_mif1 = vmf1,vauc,vauc_pr,v_mif1
                    test_mf1, test_auc, test_auc_pr,test_mif1 = tmf1,tauc,tauc_pr,t_mif1
            if epoch % 100 == 0:
                print(epoch,'LR: {:.6f} Loss {:.4f} mergin loss {:.4f}, Train MF1 {:.2f} Val: MF1 {:.2f} mif1 {:.2f} AUC {:.2f} AUCpr {:.2f};Test: MF1 {:.2f} AUC {:.2f} AUCpr {:.2f} mif1 {:.2f}'.format(
                                                                                optimizer.param_groups[0]["lr"],
                                                                                loss.item(),
                                                                                mergin_loss.item(),
                                                                                train_mf1 * 100,
                                                                                vmf1 * 100,
                                                                                v_mif1 * 100,
                                                                                vauc * 100,
                                                                                vauc_pr * 100,
                                                                                test_mf1 * 100,
                                                                                test_auc * 100,
                                                                                test_auc_pr * 100,
                                                                                test_mif1 * 100)
                                                                                )
    return test_mf1, test_auc, test_auc_pr,test_mif1


parser = argparse.ArgumentParser(
    description="Training GNN on data"
)

parser.add_argument("--n_epoch", type=int, default=6000)
parser.add_argument("--n_hid", type=int, default=64)
parser.add_argument("--n_inp", type=int, default=32)
parser.add_argument("--clip", type=int, default=1.0)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--max_lr", type=float, default=0.001)

parser.add_argument("--wd", type=float, default=0.0)
parser.add_argument("--dropout", type=float, default=0.0)
parser.add_argument("--m_epoch", type=int, default=50)

parser.add_argument("--zeta", type=float, default=0.0)
parser.add_argument("--env_theta", type=float, default=0.3)
parser.add_argument("--beta", type=float, default=0.5)

# parameters in markov process
parser.add_argument("--r", type=float, default=1.2)
parser.add_argument("--theta", type=float, default=0.01)
parser.add_argument("--cuda", type=int, default=0)

args = parser.parse_args()

device = torch.device("cuda:"+str(args.cuda))
env_theta = args.env_theta
zeta = args.zeta
beta = args.beta

data = GetData('yelp',0.4)
graph, train_mask, val_mask, test_mask = data.split_data()

zeta = torch.tensor(zeta).to(device)

min_theta = args.theta
r = args.r

import pickle
markov_graph_path = './tem_data/' + "markov_graph_" + str(args.r) + "_" + str(args.theta) + "_" + str(args.m_epoch) + ".pkl"
print(markov_graph_path)

if (os.path.exists(markov_graph_path)):
    print(markov_graph_path + " exits")
    with open(markov_graph_path, 'rb') as f:
        graph = pickle.load(f)
else:
    with open(markov_graph_path, 'wb') as f:
        graph = get_markov_graph(graph, r, min_theta, args.m_epoch)
        pickle.dump(graph,f)

print(graph)

from GetEnvirLabel import get_env_label
env_label = get_env_label(graph,train_mask,env_theta=env_theta,train_ratio= 0.4)
env_label = torch.from_numpy(env_label).to(device)


node_dict = {}
edge_dict = {}
for ntype in graph.ntypes:
    node_dict[ntype] = len(node_dict)
for etype in graph.etypes:
    edge_dict[etype] = len(edge_dict)
    graph.edges[etype].data["id"] = (
        torch.ones(graph.number_of_edges(etype), dtype=torch.long)
        * edge_dict[etype]
    )


labels = graph.ndata['label'].clone()
graph = graph.to(device)

set_seed(args.seed)

model = AMHGNN(
    graph,
    node_dict,
    edge_dict,
    n_inp=args.n_inp,
    n_hid=args.n_hid,
    n_out=labels.max().item() + 1,
    n_layers=2,
    n_heads=4,
    train_mask=train_mask.to(device),
    env_label=env_label,
    zeta=zeta,
    dropout=args.dropout,
    use_norm=True,
).to(device)

optimizer = torch.optim.AdamW(model.parameters(),lr = args.max_lr, weight_decay = args.wd)


scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 500, eta_min=0, last_epoch=-1)


print("Training AMHGNN with #param: %d" % (get_n_params(model)))
test_mf1, test_auc, test_auc_pr,test_mif1 = train(model, graph,labels, args, train_mask, val_mask, test_mask, beta)
print("One exp ans Test: MF1 {:.2f} AUC {:.2f} AUCpr {:.2f}  MIF1 {:.2f}".format(test_mf1*100,test_auc*100,test_auc_pr*100,test_mif1*100))
print('--------one exp args----------')
for k in list(vars(args).keys()):
    print('%s: %s' % (k, vars(args)[k]))
