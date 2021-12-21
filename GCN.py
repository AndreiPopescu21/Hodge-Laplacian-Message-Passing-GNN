from Flow_Dataset import Flow_Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dgl.nn import GraphConv
from dgl.dataloading import GraphDataLoader

class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.classify = nn.Linear(hidden_dim, n_classes)

    
    def forward(self, g):
        h = g.in_degrees().view(-1, 1).float()

        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        g.ndata['h'] = h

        hg = dgl.mean_nodes(g, 'h')
        return self.classify(hg)

dataset = Flow_Dataset() # 1500

train_data = dataset[:1500]
test_data = dataset[1500:]

model = GCN(1, 256, 2)

loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
model.train()

