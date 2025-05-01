import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.functional import edge_softmax
from dgl.nn.pytorch import GraphConv, GATConv, GINConv
from dgl.nn.pytorch.glob import AvgPooling

########################################################################################################################


def sub_g_rnn_split(hv, pooling, hid_dim, g, max_len, group_arr_list):
    batch_g = []
    batch_g_mask = []

    for i in range(g.batch_size):
        one_g = []
        for j in range(len(group_arr_list[i])):
            sub_g = dgl.node_subgraph(g, group_arr_list[i][j])
            sub_hv = hv[group_arr_list[i][j]]
            sub_g_mean = pooling(sub_g, sub_hv)
            one_g.append(sub_g_mean)

        one_g.extend([torch.zeros(1, hid_dim, device=g.device)] * (max_len - len(group_arr_list[i])))

        batch_g.append(torch.stack(one_g, dim=0).sum(1))


        one_g_mask = torch.cat((torch.ones(len(group_arr_list[i]), hid_dim, device=g.device),
                                torch.zeros(max_len - len(group_arr_list[i]), hid_dim, device=g.device)))

        batch_g_mask.append(one_g_mask)

    rnn_input = torch.stack(batch_g)  # [batch_size, max_len, hid_dim] [256, 15, 256]

    out_put_mask = torch.stack(batch_g_mask)  # [batch_size, max_len, hid_dim] [256, 15, 256]

    out_put_mask_bool = out_put_mask.bool()  # True: has g

    return rnn_input, out_put_mask_bool


def sub_g_rnn_forward(_classself, g, hv, max_len, group_arr_list):
    if not _classself.use_bidirectional_rnn and not _classself.use_multi_attn:
        raise Exception("sub_g_rnn_forward: setting err, use_bidirectional_rnn and use_multi_attn = False")

    rnn_input, out_put_mask_bool = sub_g_rnn_split(
        hv, _classself.pooling, _classself.hid_dim, g, max_len, group_arr_list)

    if _classself.use_bidirectional_rnn:

        rnn_prev_h = torch.zeros(2, g.batch_size, _classself.hid_dim, device=g.device)

        rnn_output, _ = _classself.rnn(rnn_input, rnn_prev_h)

        rnn_output = rnn_output.view(g.batch_size, max_len, 2, _classself.hid_dim)

        rnn_output = (rnn_output[:, :, 0, :] + rnn_output[:, :, 1, :]) / 2

        seq_output = rnn_output

    elif _classself.use_multi_attn:

        query, key, value = rnn_input, rnn_input, rnn_input  # self attn

        attn_output, attn_output_weights = _classself.multihead_attn(query, key, value,
                                                                     key_padding_mask=~out_put_mask_bool[:, :, 0])

        seq_output = attn_output



    masked_output = torch.masked_fill(seq_output, ~out_put_mask_bool, 0)

    masked_output = torch.sum(masked_output, dim=1)

    return _classself.output(torch.cat([masked_output, _classself.pooling(g, hv)], dim=1))


########################################################################################################################
# ATC_GCN 

class ATC_GCN(nn.Module):
    def __init__(self,
                 hid_dim,
                 out_dim,
                 num_layers,
                 num_node_emb_list=None,
                 **kwargs):
        super(ATC_GCN, self).__init__()

        self.num_layers = num_layers
        self.gcns = nn.ModuleList()
        self.hid_dim = hid_dim

        self.use_bidirectional_rnn = "use_bidirectional_rnn" in kwargs and kwargs["use_bidirectional_rnn"]
        self.use_multi_attn = "use_multi_attn" in kwargs and kwargs["use_multi_attn"]

        for _ in range(self.num_layers):
            conv = GraphConv(in_feats=hid_dim, out_feats=hid_dim,
                             activation=F.relu,
                             allow_zero_in_degree=False)

            self.gcns.append(conv)

        self.pooling = AvgPooling()

        emb_dim = hid_dim
        self.node_embeddings = nn.ModuleList()
        for num_emb in num_node_emb_list:
            emb_module = nn.Embedding(num_emb, emb_dim)
            self.node_embeddings.append(emb_module)

        if self.use_bidirectional_rnn:
            self.rnn = nn.RNN(hid_dim, hid_dim, batch_first=True, bidirectional=True)
            self.output = nn.Linear(hid_dim * 2, out_dim)
        elif self.use_multi_attn:
            self.multihead_attn = nn.MultiheadAttention(hid_dim, kwargs["multi_attn_head_num"], batch_first=True)
            self.output = nn.Linear(hid_dim * 2, out_dim)
        else:
            self.output = nn.Linear(hid_dim, out_dim)

    def forward(self, g, categorical_node_feats, _, max_len, group_arr_list):

        with g.local_scope():

            node_embeds = []
            for i, feats in enumerate(categorical_node_feats):
                node_embeds.append(self.node_embeddings[i](feats))
            node_embeds = torch.stack(node_embeds, dim=0).sum(0)
            hv = node_embeds  # (num_nodes, hid_dim)

            for layer in range(self.num_layers):
                hv = self.gcns[layer](g, hv)

            if self.use_bidirectional_rnn or self.use_multi_attn:
                return sub_g_rnn_forward(self, g, hv, max_len, group_arr_list)

            # no sub g rnn
            h_g = self.pooling(g, hv)
            return self.output(h_g)

########################################################################################################################
# ATC_GAT 

class ATC_GAT(ATC_GCN):
    def __init__(self,
                 hid_dim,
                 out_dim,
                 gat_num_heads,
                 num_layers,
                 num_node_emb_list=None,
                 **kwargs):
        super(ATC_GAT, self).__init__(hid_dim, out_dim, num_layers, num_node_emb_list, **kwargs)
        self.gat_num_heads = gat_num_heads

        for _ in range(self.num_layers):
            conv = GATConv(in_feats=hid_dim, out_feats=hid_dim, num_heads=gat_num_heads,
                             activation=F.relu,
                             allow_zero_in_degree=False)

            self.gcns.append(conv)
            
########################################################################################################################
# ATC_GIN

class ATC_GIN(ATC_GCN):
    def __init__(self,
                 hid_dim,
                 out_dim,
                 num_layers,
                 num_node_emb_list=None,
                 **kwargs):
        super(ATC_GIN, self).__init__(hid_dim, out_dim, num_layers, num_node_emb_list, **kwargs)

        for _ in range(self.num_layers):
            lin = nn.Linear(hid_dim, hid_dim)
            conv = GINConv(apply_func=lin,
                             activation=F.relu)

            self.gcns.append(conv)

########################################################################################################################
# ATC_DGCN
class GENConvMLP(nn.Sequential):
    def __init__(self,
                 channels,
                 dropout=0.,
                 bias=True):
        layers = []

        for i in range(1, len(channels)):
            layers.append(nn.Linear(channels[i - 1], channels[i], bias))
            if i < len(channels) - 1:
                layers.append(nn.BatchNorm1d(channels[i], affine=True))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))

        super(GENConvMLP, self).__init__(*layers)


class MessageNorm(nn.Module):
    def __init__(self, learn_scale=True):
        super(MessageNorm, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=learn_scale)

    def forward(self, feats, msg, p=2):
        msg = F.normalize(msg, p=2, dim=-1)
        feats_norm = feats.norm(p=p, dim=-1, keepdim=True)
        return msg * feats_norm * self.scale


class GENConv(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 aggregator='softmax',
                 beta=1.0,
                 use_softmax_aggr=True,
                 learn_beta=True,
                 msg_norm=False,
                 learn_msg_scale=True,
                 mlp_layers=1,
                 eps=1e-7,
                 num_edge_emb_list=None,
                 **kwargs):
        super(GENConv, self).__init__()

        self.aggr = aggregator
        self.eps = eps

        channels = [in_dim]
        for _ in range(mlp_layers - 1):
            channels.append(in_dim * 2)
        channels.append(out_dim)

        self.mlp = GENConvMLP(channels)
        self.msg_norm = MessageNorm(learn_msg_scale) if msg_norm else None

        self.beta = nn.Parameter(torch.Tensor([beta]),
                                 requires_grad=True) if learn_beta and self.aggr == 'softmax' else beta
        self.use_softmax_aggr = use_softmax_aggr

        emb_dim = in_dim
        self.edge_embeddings = nn.ModuleList()
        for num_emb in num_edge_emb_list:
            emb_module = nn.Embedding(num_emb, emb_dim)
            self.edge_embeddings.append(emb_module)

    def forward(self, g, node_feats, categorical_edge_feats):
        with g.local_scope():
            # Node and edge feature size need to match.
            g.ndata['h'] = node_feats

            edge_embeds = []
            for i, feats in enumerate(categorical_edge_feats):
                edge_embeds.append(self.edge_embeddings[i](feats))
            edge_embeds = torch.stack(edge_embeds, dim=0).sum(0)

            # g.edata['h'] = self.edge_encoder(edge_feats)
            g.edata['h'] = edge_embeds

            g.apply_edges(dgl.function.u_add_e('h', 'h', 'm'))

            if self.aggr == 'softmax':
                g.edata['m'] = F.relu(g.edata['m']) + self.eps
                if self.use_softmax_aggr:
                    g.edata['a'] = edge_softmax(g, g.edata['m'] * self.beta)
                    g.update_all(lambda edge: {'x': edge.data['m'] * edge.data['a']}, dgl.function.sum('x', 'm'))
                else:
                    g.update_all(lambda edge: {'x': edge.data['m']}, dgl.function.sum('x', 'm'))
            else:
                raise NotImplementedError(
                    f'Aggregator {self.aggr} is not supported.')

            if self.msg_norm is not None:
                g.ndata['m'] = self.msg_norm(node_feats, g.ndata['m'])

            feats = node_feats + g.ndata['m']

            return self.mlp(feats)


class ATC_DGCN(nn.Module):
    def __init__(self,
                 hid_dim,
                 out_dim,
                 num_layers,
                 dropout=0.,
                 mlp_layers=1,
                 use_bidirectional_rnn=True,
                 num_node_emb_list=None,
                 num_edge_emb_list=None,
                 use_skip_connect=True,
                 use_softmax_aggr=True,
                 use_msgnorm=False,
                 **kwargs):
        super(ATC_DGCN, self).__init__()

        self.use_bidirectional_rnn = use_bidirectional_rnn
        self.use_skip_connect = use_skip_connect

        self.num_layers = num_layers
        self.dropout = dropout
        self.gcns = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.hid_dim = hid_dim

        for _ in range(self.num_layers):
            conv = GENConv(in_dim=hid_dim,
                           out_dim=hid_dim,
                           learn_beta=True,
                           learn_msg_scale=True,
                           msg_norm=use_msgnorm,
                           mlp_layers=mlp_layers,
                           num_edge_emb_list=num_edge_emb_list,
                           use_softmax_aggr=use_softmax_aggr,
                           )

            self.gcns.append(conv)
            self.norms.append(nn.BatchNorm1d(hid_dim, affine=True))

        self.pooling = AvgPooling()

        emb_dim = hid_dim
        self.node_embeddings = nn.ModuleList()
        for num_emb in num_node_emb_list:
            emb_module = nn.Embedding(num_emb, emb_dim)
            self.node_embeddings.append(emb_module)

        self.use_multi_attn = "use_multi_attn" in kwargs and kwargs["use_multi_attn"]

        if self.use_bidirectional_rnn:
            self.rnn = nn.RNN(hid_dim, hid_dim, batch_first=True, bidirectional=True)
            self.output = nn.Linear(hid_dim * 2, out_dim)
        elif self.use_multi_attn:
            self.multihead_attn = nn.MultiheadAttention(hid_dim, kwargs["multi_attn_head_num"], batch_first=True)
            self.output = nn.Linear(hid_dim * 2, out_dim)
        else:
            self.output = nn.Linear(hid_dim, out_dim)

    def forward(self, g, categorical_node_feats, categorical_edge_feats, max_len, group_arr_list):

        with g.local_scope():

            node_embeds = []
            for i, feats in enumerate(categorical_node_feats):
                node_embeds.append(self.node_embeddings[i](feats))
            node_embeds = torch.stack(node_embeds, dim=0).sum(0)
            hv = node_embeds

            he = categorical_edge_feats

            for layer in range(self.num_layers):
                if hv.shape[0] != 1:  # not norm single atom
                    hv1 = self.norms[layer](hv)
                else:
                    hv1 = hv
                hv1 = F.relu(hv1)
                hv1 = F.dropout(hv1, p=self.dropout, training=self.training)
                if self.use_skip_connect:
                    hv = self.gcns[layer](g, hv1, he) + hv
                else:
                    hv = self.gcns[layer](g, hv1, he)

            if self.use_bidirectional_rnn or self.use_multi_attn:
                return sub_g_rnn_forward(self, g, hv, max_len, group_arr_list)

            # no sub g rnn
            h_g = self.pooling(g, hv)
            return self.output(h_g)


########################################################################################################################
# RNN

class BiRNN(nn.Module):
    def __init__(self, n_vocab, hid_dim=256,  out_dim=14, num_layers=1, **kwargs):
        super(BiRNN, self).__init__()
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(n_vocab, hid_dim, padding_idx=n_vocab - 1)
        self.rnn = nn.RNN(hid_dim, hid_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.output = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        batch_size, seq_len = x.shape[0], x.shape[1]
        x = self.embedding(x)
        rnn_prev_h = torch.zeros(2*self.num_layers, batch_size, self.hid_dim, device=x.device)
        rnn_output, _ = self.rnn(x, rnn_prev_h)
        rnn_output = rnn_output.view(batch_size, seq_len, 2, self.hid_dim)
        rnn_output = (rnn_output[:, :, 0, :] + rnn_output[:, :, 1, :]) / 2
        rnn_output = torch.sum(rnn_output, dim=1)
        return self.output(rnn_output)


class TextCNN(nn.Module):
    def __init__(self, n_vocab, hid_dim=256,  out_dim=14, dropout=0.2,
                 num_filters=256, filter_sizes=(2, 4, 6, 8, 10, 16, 24), **kwargs):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(n_vocab, hid_dim, padding_idx=n_vocab - 1)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, num_filters, (k, hid_dim)) for k in filter_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), out_dim)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = self.embedding(x)
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out


#######################################################################################################################
MODEL_MAP = {"GCN": ATC_GCN,
             "GAT": ATC_GAT,
             "GIN": ATC_GIN,
             "DGCN": ATC_DGCN,
             "BiRNN": BiRNN,
             "TextCNN": TextCNN}
