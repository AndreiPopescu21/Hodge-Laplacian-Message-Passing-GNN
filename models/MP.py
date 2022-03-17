import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn

class MPNN_Layer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MPNN_Layer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, g, h, e):
        g.update_all(fn.u_mul_e('node_features', 'edge_features', 'm'),
                    fn.sum('m', 'ft'))
        return self.linear(g.ndata['ft'].float())

class MPNN(nn.Module):
    def __init__(self, net_params):
        super(MPNN, self).__init__()

        input_dim = net_params['input_dim']
        hidden_dim = net_params['hidden_dim']
        output_dim = net_params['output_dim']
        n_layers = net_params['n_layers']
        self.readout = net_params['readout']

        self.layers = nn.ModuleList([])
        self.layers.append(MPNN_Layer(input_dim, hidden_dim))

        for i in range(n_layers - 2):
            self.layers.append(MPNN_Layer(input_dim, hidden_dim))

        self.mlp_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, g, h, e):
        for layer in self.layers:
            h = F.relu(layer(g, h, e))

        with g.local_scope():
            g.ndata['h'] = h

            if self.readout == "sum":
                hg = dgl.sum_nodes(g, 'h')
            elif self.readout == "max":
                hg = dgl.max_nodes(g, 'h')
            elif self.readout == "mean":
                hg = dgl.mean_nodes(g, 'h')
            else:
                hg = dgl.mean_nodes(g, 'h')

        return F.log_softmax(self.mlp_layer(hg), dim=0)