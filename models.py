import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl.function as fn
from dgl.nn.functional import edge_softmax

class ComponentAttention(nn.Module):
    def __init__(self,in_size,hidden_size):
        super(ComponentAttention, self).__init__()
        self.attn_fn = nn.Tanh()
        self.W_f = nn.Sequential(nn.Linear(in_size, hidden_size),
                                 self.attn_fn,
                                 )
        self.W_x = nn.Sequential(nn.Linear(in_size, hidden_size),
                                 self.attn_fn,
                                 )

    def forward(self,z,x,ntype):
        h_z_proj = self.W_f(z)
        x_proj = self.W_x(x).unsqueeze(-1)

        score_logit = torch.bmm(h_z_proj, x_proj)
        soft_score = F.softmax(score_logit, dim=1)
        score = soft_score

        res = z[:, 0, :] * score[:, 0]
        res += z[:, 1, :] * score[:, 1]

        return res,score

class HATT(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        node_dict,
        edge_dict,
        n_heads,
        use_norm=False,
    ):
        super(HATT, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.num_types = len(node_dict)
        self.num_relations = len(edge_dict)
        self.n_heads = n_heads
        self.d_k = out_dim // n_heads
        self.sqrt_dk = math.sqrt(self.d_k)
        self.att = None

        self.k_linears = nn.ModuleList()
        self.q_linears = nn.ModuleList()
        self.use_norm = use_norm

        for t in range(self.num_types):
            self.k_linears.append(nn.Linear(in_dim, out_dim))
            self.q_linears.append(nn.Linear(in_dim, out_dim))

        self.relation_pri = nn.Parameter(
            torch.ones(self.num_relations, self.n_heads)
        )
        self.relation_att = nn.Parameter(
            torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k)
        )

        nn.init.xavier_uniform_(self.relation_att)

    def forward(self, G, h, lowf = True):
        etype_att = {}
        with G.local_scope():
            node_dict, edge_dict = self.node_dict, self.edge_dict
            for srctype, etype, dsttype in G.canonical_etypes:
                sub_graph = G[srctype, etype, dsttype]

                k_linear = self.k_linears[node_dict[srctype]]
                q_linear = self.q_linears[node_dict[dsttype]]

                k = k_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
                q = q_linear(h[dsttype]).view(-1, self.n_heads, self.d_k)

                e_id = self.edge_dict[etype]

                relation_att = self.relation_att[e_id]
                relation_pri = self.relation_pri[e_id]

                k = torch.einsum("bij,ijk->bik", k, relation_att)

                sub_graph.srcdata["k"] = k
                sub_graph.dstdata["q"] = q

                sub_graph.apply_edges(fn.v_dot_u("q", "k", "t"))
                attn_score = (
                    sub_graph.edata.pop("t").sum(-1)
                    * relation_pri
                    / self.sqrt_dk
                )

                if (not lowf):
                    attn_score = (attn_score+0.000001)**(-1)

                attn_score = edge_softmax(sub_graph, attn_score, norm_by="dst")
                etype_att[etype] = attn_score.unsqueeze(-1)

            return etype_att


class HLayer(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        node_dict,
        edge_dict,
        n_heads,
        train_mask,
        env_label,
        zeta,
        dropout=0.2,
        use_norm=False,
    ):
        super(HLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.num_types = len(node_dict)
        self.num_relations = len(edge_dict)
        self.train_mask = train_mask
        self.env_label = env_label
        self.zeta = zeta
        self.n_heads = n_heads
        self.d_k = out_dim // n_heads
        self.sqrt_dk = math.sqrt(self.d_k)
        self.att = None
        self.use_norm = use_norm

        self.lfhatt = HATT(self.in_dim,self.out_dim,self.node_dict,self.edge_dict,self.n_heads,self.use_norm)
        self.comp_att = ComponentAttention(in_size=self.out_dim,hidden_size=self.out_dim)

        self.v_linears = nn.ModuleList()
        self.a_linears = nn.ModuleList()
        self.norms = nn.ModuleList()

        for t in range(self.num_types):
            self.v_linears.append(nn.Linear(in_dim, out_dim))
            self.a_linears.append(nn.Linear(out_dim, out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))

        self.relation_msg = nn.Parameter(
            torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k)
        )

        self.skip = nn.Parameter(torch.ones(self.num_types))
        self.drop = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.relation_msg)

    def forward(self, G, h, out_key = None , mergin = False):
        merge_loss = None
        with G.local_scope():
            lfatt_dict = self.lfhatt(G,h,True)
            hfatt_dict = self.lfhatt(G,h,False)
            node_dict, edge_dict = self.node_dict, self.edge_dict
            for srctype, etype, dsttype in G.canonical_etypes:
                sub_graph = G[srctype, etype, dsttype]
                sub_graph.edata["lt"] = lfatt_dict[etype]
                sub_graph.edata["ht"] = hfatt_dict[etype]

                v_linear = self.v_linears[node_dict[srctype]]
                v = v_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
                e_id = self.edge_dict[etype]

                relation_msg = self.relation_msg[e_id]
                v = torch.einsum("bij,ijk->bik", v, relation_msg)

                sub_graph.srcdata["v_%d" % e_id] = v

            G.multi_update_all(
                {
                    etype: (
                        fn.u_mul_e("v_%d" % e_id, "lt", "m"),
                        fn.sum("m", "lt"),
                    )
                    for etype, e_id in edge_dict.items()
                },
                cross_reducer="mean",
            )

            G.multi_update_all(
                {
                    etype: (
                        fn.u_mul_e("v_%d" % e_id, "ht", "m"),
                        fn.sum("m", "ht"),
                    )
                    for etype, e_id in edge_dict.items()
                },
                cross_reducer="mean",
            )

            new_h = {}
            for ntype in G.ntypes:
                n_id = node_dict[ntype]
                alpha = torch.sigmoid(self.skip[n_id])
                lt = G.nodes[ntype].data["lt"].view(-1, self.out_dim)
                ht = G.nodes[ntype].data["ht"].view(-1, self.out_dim)

                v_linear = self.v_linears[node_dict[ntype]]
                ori_v = v_linear(h[ntype]).view(-1, self.out_dim)

                lt = ori_v + lt
                ht = ori_v - ht

                t,score = self.comp_att(torch.stack([lt,ht],dim=1),ori_v,ntype)

                if (ntype == out_key and mergin):
                    abnormal_mask = (self.env_label == 1) & self.train_mask
                    normal_mask = (self.env_label == -1) & self.train_mask
                    abnormal_loss = (score[abnormal_mask][:,0] - score[abnormal_mask][:,1]) + self.zeta
                    normal_loss = (score[normal_mask][:,1] - score[normal_mask][:,0]) + self.zeta

                    abnormal_loss = torch.clamp(abnormal_loss, -0.0)
                    normal_loss = torch.clamp(normal_loss, -0.0)

                    abnormal_loss = torch.mean(abnormal_loss)
                    normal_loss = torch.mean(normal_loss)

                    tem_loss = normal_loss + abnormal_loss
                    merge_loss = tem_loss

                trans_out = self.drop(self.a_linears[n_id](t))
                trans_out = trans_out * alpha + h[ntype] * (1 - alpha)
                if self.use_norm:
                    new_h[ntype] = self.norms[n_id](trans_out)
                else:
                    new_h[ntype] = trans_out

            return new_h,merge_loss


class AMHGNN(nn.Module):
    def __init__(
        self,
        G,
        node_dict,
        edge_dict,
        n_inp,
        n_hid,
        n_out,
        n_layers,
        n_heads,
        dropout,
        train_mask,
        env_label,
        zeta,
        use_norm=True,
    ):
        super(AMHGNN, self).__init__()
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.gcs = nn.ModuleList()
        self.n_inp = n_inp
        self.n_hid = n_hid
        self.n_out = n_out
        self.n_layers = n_layers
        self.dropout = dropout
        self.train_mask = train_mask
        self.env_label = env_label
        self.zeta = zeta
        self.adapt_ws = nn.ModuleList()
        for t in range(len(node_dict)):
            self.adapt_ws.append(nn.Linear(n_inp, n_hid))

        for _ in range(n_layers):
            self.gcs.append(
                HLayer(
                    n_hid,
                    n_hid,
                    node_dict,
                    edge_dict,
                    n_heads,
                    train_mask = self.train_mask,
                    env_label = self.env_label,
                    zeta = self.zeta,
                    dropout = self.dropout,
                    use_norm=use_norm,
                )
            )

        self.drop = nn.Dropout(self.dropout)
        self.out = nn.Linear(n_hid, n_out)

    def forward(self, G, out_key):
        h = {}
        for ntype in G.ntypes:
            n_id = self.node_dict[ntype]
            emb_n = self.adapt_ws[n_id](G.nodes[ntype].data["feature"])
            emb_n = self.drop(emb_n)
            h[ntype] = F.gelu(emb_n)
        for i in range(self.n_layers - 1):
            h,mergs_loss = self.gcs[i](G, h)

        h,mergs_loss = self.gcs[self.n_layers-1](G, h,out_key,True)

        return self.out(h[out_key]),mergs_loss