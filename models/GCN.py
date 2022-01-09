import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
import dgl

class GCN(nn.Module):
    def __init__(self, net_params):
        super(GCN, self).__init__()

        input_dim = net_params['input_dim']
        hidden_dim = net_params['hidden_dim']
        output_dim = net_params['output_dim']
        n_layers = net_params['n_layers']
        self.readout = net_params['readout']

        self.layers = nn.ModuleList([])
        self.layers.append(GraphConv(input_dim, hidden_dim))

        for i in range(n_layers - 2):
            self.layers.append(GraphConv(hidden_dim, hidden_dim))

        self.mlp_layer = self.mlp = nn.Linear(hidden_dim, output_dim)

    def forward(self, g, h):
        for layer in self.layers:
            h = F.relu(layer(g, h))

        with g.local_scope():
            g.ndata['h'] = h

            if self.readout == 'sum':
                hg = dgl.sum_nodes(g, 'h')
            elif self.readout == 'max':
                hg = dgl.max_nodes(g, 'h')
            elif self.readout == 'mean':
                hg = dgl.mean_nodes(g, 'h')
            else:
                hg = dgl.mean_nodes(g, 'h')
            
            return F.log_softmax(self.mlp_layer(hg), dim=0)