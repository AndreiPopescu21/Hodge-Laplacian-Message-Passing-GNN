import torch.nn as nn
import dgl

from .dgn_layer import DGNLayer
from .mlp_readout_layer import MLPReadout

class DGN(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        in_dim = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.type_net = net_params['type_net']
        self.readout = net_params['readout']
        self.graph_norm = net_params['graph_norm']
        self.batch_norm = net_params['batch_norm']
        self.aggregators = net_params['aggregators']
        self.scalers = net_params['scalers']
        self.avg_d = net_params['avg_d']
        self.residual = net_params['residual']
        self.edge_feat = net_params['edge_feat']
        self.posttrans_layers = net_params['posttrans_layers']
        edge_dim = net_params['edge_dim']

        self.layers = [DGNLayer(in_dim=in_dim, out_dim=hidden_dim, dropout=dropout, graph_norm=self.graph_norm,
                                batch_norm=self.batch_norm, residual=self.residual, aggregators=self.aggregators,
                                scalers=self.scalers, avg_d=self.avg_d, posttrans_layers=self.posttrans_layers, type_net=self.type_net).model.to('cuda:0')]

        for i in range(n_layers-1):
            self.layers.append(DGNLayer(in_dim=hidden_dim, out_dim=hidden_dim, dropout=dropout, graph_norm=self.graph_norm,
                                batch_norm=self.batch_norm, residual=self.residual, aggregators=self.aggregators,
                                scalers=self.scalers, avg_d=self.avg_d, posttrans_layers=self.posttrans_layers, type_net=self.type_net).model.to('cuda:0'))

        self.mlp = MLPReadout(hidden_dim, out_dim)

    def forward(self, g, h, e, snorm_n):
        for conv in self.layers:
            h_t = conv(g, h, e, snorm_n)
            h = h_t

        g.ndata['h'] = h
        hg = dgl.sum_nodes(g, 'h')

        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        else:
            hg = dgl.mean_nodes(g, 'h')

        return self.mlp(hg)