import torch, dgl
import torch.nn as nn
import torch.nn.functional as F
from datasets.ZINC.ZINC_Dataset import ZINC_Dataset
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import dgl.function as fn
from models.DGN.dgn_layer import DGNLayerSimple

class MPNN_Layer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MPNN_Layer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, g, h, e):
        g.update_all(fn.u_mul_e('atom_type', 'e', 'm'),
                    fn.sum('m', 'ft'))
        return self.linear(g.ndata['ft'].float())

class MPNN(nn.Module):
    def __init__(self, net_params):
        super(MPNN, self).__init__()

        # input_dim = net_params['input_dim']
        # hidden_dim = net_params['hidden_dim']
        # output_dim = net_params['output_dim']
        # n_layers = net_params['n_layers']
        # self.readout = net_params['readout']

        input_dim = 2
        hidden_dim = 32
        output_dim = 1
        n_layers = 2
        self.readout = "mean"

        self.embedding = nn.Embedding(28, hidden_dim)
        self.layers = nn.ModuleList([])

        for _ in range(n_layers - 1):
            self.layers.append(MPNN_Layer(input_dim, hidden_dim))

        self.mlp_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, g, h, e):
        # g.edata['e'] = torch.cat([e, g.edata['eig']], axis=1).float()
        g.edata['e'] = g.edata['eig'].float()
        h = self.embedding(h.squeeze().to(torch.long))
        
        for layer in self.layers:
            h = F.relu(layer(g, h, e))

        # with g.local_scope():
        g.ndata['h'] = h
        hg = dgl.sum_nodes(g, 'h')

            # if self.readout == "sum":
            #     hg = dgl.sum_nodes(g, 'h')
            # elif self.readout == "max":
            #     hg = dgl.max_nodes(g, 'h')
            # elif self.readout == "mean":
            #     hg = dgl.mean_nodes(g, 'h')
            # else:
            #     hg = dgl.mean_nodes(g, 'h')

        output = self.mlp_layer(hg)
        return output


def collate_data(samples):
    graphs, labels, _, _ = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)

def get_data(dataset):
    dataset_length = len(dataset)
    num_train = int(dataset_length * 0.8)

    train_sampler = SubsetRandomSampler(torch.arange(num_train))
    test_sampler = SubsetRandomSampler(torch.arange(num_train, dataset_length))

    train_dataloader = GraphDataLoader(
        dataset, sampler=train_sampler, batch_size=128, drop_last=False, collate_fn=collate_data)
    test_dataloader = GraphDataLoader(
        dataset, sampler=test_sampler, batch_size=128, drop_last=False, collate_fn=collate_data)

    return train_dataloader, test_dataloader

def train_model(model, train_dataloader, num_epochs):
    optimizer = torch.optim.Adam(model.parameters())
    loss_func = torch.nn.L1Loss()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batched_graph, labels in train_dataloader:
            labels = labels.view(-1, 1)
            batched_graph.ndata['atom_type'] = batched_graph.ndata['atom_type'].float()
            optimizer.zero_grad()
            logits = model(batched_graph, batched_graph.ndata['atom_type'], batched_graph.edata['bond_type'])
            loss = loss_func(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()
        print('Epoch {}: {}'.format(epoch + 1, epoch_loss))

    return model

if __name__ == '__main__':
    dataset = ZINC_Dataset()

    train_dataloader, test_dataloader = get_data(dataset)

    model = MPNN(None)
    model = train_model(model, train_dataloader, 1000)
