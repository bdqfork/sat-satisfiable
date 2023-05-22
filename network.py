import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import torch
import dgl
import dgl.function as fn


class GCNLayer(nn.Module):

    """Some Information about GNNLayer"""

    def __init__(self, in_dim, etypes, relations, step):
        super(GCNLayer, self).__init__()
        self.etypes = etypes
        self.relations = relations
        self.step = step

        # build LSTM for each relation
        nlayers = []
        for _ in range(len(self.relations)):
            nlayers.append(nn.LSTMCell(in_dim, in_dim, bias=True))
        self.nlayers = nn.ModuleList(nlayers)

        elayers = {}
        # build MLP for each edge
        for etype in etypes:
            elayers[etype] = nn.Sequential(
                nn.Linear(in_dim, in_dim, bias=True),
                nn.ReLU(),
                nn.Linear(in_dim, in_dim, bias=True),
                nn.ReLU(),
                nn.Linear(in_dim, in_dim, bias=True),
                nn.ReLU(),
            )
        self.elayers = nn.ModuleDict(elayers)

    @autocast()
    def forward(self, g, nfeats):
        for _ in range(self.step):
            with g.local_scope():
                for i, ((srctype, dsttype), etypes) in enumerate(self.relations.items()):
                    # aggregate neighboring node information and corresponding edge information
                    etype_dict = {}
                    for etype in etypes:
                        x = self.elayers[etype](nfeats[srctype][1])

                        g.nodes[srctype].data[etype+'_m'] = x.float()

                        etype_dict[etype] = (fn.copy_src(
                            etype+'_m', 'm'), fn.sum('m', 'm'))
                    with autocast(False):
                        g.multi_update_all(etype_dict, 'sum')

                    x = g.nodes[dsttype].data['m']

                    x = x.flatten(1)

                    h_0, c_0 = nfeats[dsttype]
                    h_0 = h_0.flatten(1)
                    c_0 = c_0.flatten(1)

                    # use LSTM to update node information
                    with autocast(False):
                        h, c = self.nlayers[i](x.float(), (h_0, c_0))

                    del h_0
                    del c_0
                    h = h.reshape((h.shape[0], 1, -1))
                    c = c.reshape((c.shape[0], 1, -1))
                    c = F.relu(c)
                    nfeats[dsttype] = (h.float(), c.float())

                    del x
                    del h
                    del c

        return nfeats


class EntityEmbed(nn.Module):
    """Embedding for each node, each type of node corresponds to an embedding matrix"""

    def __init__(self, ntypes, embed_size):
        super(EntityEmbed, self).__init__()
        self.ntypes = ntypes
        self.embedding = nn.Embedding(len(ntypes), embed_size)

    @autocast()
    def forward(self, g):
        nfeats = {}
        for ntype in self.ntypes:
            x = g.nodes[ntype].data['x']
            nfeats[ntype] = (self.embedding(x), self.embedding(x))
            g.nodes[ntype].data.pop('x')
            del x
        return nfeats


class GlobalAttentionPooling(nn.Module):
    def __init__(self, gate_nn, feat_nn=None):
        super(GlobalAttentionPooling, self).__init__()
        self.gate_nn = gate_nn
        self.feat_nn = feat_nn

    def forward(self, feat):
        gate = self.gate_nn(feat)
        assert gate.shape[-1] == 1, "The output of gate_nn should have size 1 at the last axis."
        feat = self.feat_nn(feat) if self.feat_nn else feat

        gate = torch.softmax(gate, 0)

        x = feat * gate

        readout = torch.sum(x, 0)

        return readout


class GNNSATAttention(nn.Module):
    """Message passing relation network"""

    def __init__(self, in_dim, out_dim, ntypes, etypes, relations, step=9, assign=False, use_var=False, attr=False):
        super(GNNSATAttention, self).__init__()
        self.entityEmbed = EntityEmbed(ntypes, in_dim)

        self.gcn = GCNLayer(in_dim, etypes,
                            relations, step)

        # MLP to pool the features of clauses
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim, bias=True),
            nn.ReLU(),
            nn.Linear(in_dim, in_dim, bias=True),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim, bias=True)
        )
        self.pool = GlobalAttentionPooling(nn.Linear(in_dim, 1, bias=True))

        # linear to pool the features of variables
        self.out = nn.Linear(in_dim, out_dim, bias=True)
        self.assign = assign
        self.use_var = use_var
        self.attr = attr

    def graph_attr(self, g, feats, ntype='clause'):
        x = feats.flatten(1)

        g.nodes[ntype].data['x'] = x

        with g.local_scope():
            out = []
            for sub_g in dgl.unbatch(g):
                # take the average of the probability that all clauses can be satisfied
                x = sub_g.nodes[ntype].data['x'].flatten(1)
                with autocast(False):
                    x = self.pool(x)
                x = x.flatten()
                out.append(x)
            out = torch.stack(out, 0)
            return out

    def inference_satisfiability(self, g, feats, ntype='clause'):
        x = feats.flatten(1)

        g.nodes[ntype].data['x'] = x

        with g.local_scope():
            out = []
            for sub_g in dgl.unbatch(g):
                # take the average of the probability that all clauses can be satisfied
                x = sub_g.nodes[ntype].data['x'].flatten(1)
                with autocast(False):
                    x = self.pool(x)
                x = self.mlp(x)
                x = x.flatten()
                out.append(x)
            out = torch.stack(out, 0)
            return out

    def inference_assigns(self, g, feats):
        x = feats.flatten(1)

        x = self.out(x)

        g.nodes['var'].data['x'] = x

        out = []
        for sub_g in dgl.unbatch(g):
            x = sub_g.nodes['var'].data['x'].flatten(1)
            out.append(x)
        return out

    @autocast()
    def forward(self, g):
        # embbeding
        nfeats = self.entityEmbed(g)

        # message passing
        nfeats = self.gcn(g, nfeats)

        # pool and output prediction
        with g.local_scope():
            if self.attr:
                return self.graph_attr(g, nfeats['clause'][1])

            if self.use_var:
                satisfiability = self.inference_satisfiability(
                    g, nfeats['var'][1], 'var')
            else:
                satisfiability = self.inference_satisfiability(
                    g, nfeats['clause'][1])

            if self.assign:
                assigns = self.inference_assigns(g, nfeats['var'][1])

            del nfeats

            if self.assign:
                return satisfiability, assigns
            else:
                return satisfiability


class GNNSAT(nn.Module):
    """Message passing relation network"""

    def __init__(self, in_dim, out_dim, ntypes, etypes, relations, step=9, assign=False, use_var=False, attr=False):
        super(GNNSAT, self).__init__()
        self.entityEmbed = EntityEmbed(ntypes, in_dim)

        self.gcn = GCNLayer(in_dim, etypes,
                            relations, step)

        # MLP to pool the features of clauses
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim, bias=True),
            nn.ReLU(),
            nn.Linear(in_dim, in_dim, bias=True),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim, bias=True)
        )
        # linear to pool the features of variables
        self.out = nn.Linear(in_dim, out_dim, bias=True)
        self.assign = assign
        self.use_var = use_var
        self.attr = attr

    def graph_attr(self, g, feats, ntype='clause'):
        x = feats.flatten(1)

        g.nodes[ntype].data['x'] = x

        out = []
        for sub_g in dgl.unbatch(g):
            # take the average of the probability that all clauses can be satisfied
            x = sub_g.nodes[ntype].data['x'].flatten(1)
            x = x.mean(0).reshape((-1))
            out.append(x)
        out = torch.stack(out, 0)
        return out

    def inference_satisfiability(self, g, feats, ntype='clause'):
        x = feats.flatten(1)

        g.nodes[ntype].data['x'] = x

        out = []
        for sub_g in dgl.unbatch(g):
            # take the average of the probability that all clauses can be satisfied
            x = sub_g.nodes[ntype].data['x'].flatten(1)
            x = x.mean(0).reshape((-1))
            x = self.mlp(x)
            out.append(x)
        out = torch.stack(out, 0)
        return out

    def inference_assigns(self, g, feats):
        x = feats.flatten(1)

        x = self.out(x)

        g.nodes['var'].data['x'] = x

        out = []
        for sub_g in dgl.unbatch(g):
            x = sub_g.nodes['var'].data['x'].flatten(1)
            out.append(x)
        return out

    @autocast()
    def forward(self, g):
        # embbeding
        nfeats = self.entityEmbed(g)

        # message passing
        nfeats = self.gcn(g, nfeats)

        # pool and output prediction
        with g.local_scope():

            if self.attr:
                return self.graph_attr(g, nfeats['clause'][1])

            if self.use_var:
                satisfiability = self.inference_satisfiability(
                    g, nfeats['var'][1], 'var')
            else:
                satisfiability = self.inference_satisfiability(
                    g, nfeats['clause'][1])

            if self.assign:
                assigns = self.inference_assigns(g, nfeats['var'][1])

            del nfeats

            if self.assign:
                return satisfiability, assigns
            else:
                return satisfiability
