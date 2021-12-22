import dgl
import torch
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv

from datasets.Flow.Flow_Dataset import Flow_Dataset

def collate_data(samples):
    graphs, labels, _ = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)

def get_samples(dataset, ratio, batch_size):
    assert 0 < ratio < 1
    dataset_length = len(dataset)
    num_train = int(dataset_length * ratio)

    train_sampler = SubsetRandomSampler(torch.arange(num_train))
    test_sampler = SubsetRandomSampler(torch.arange(num_train, dataset_length))

    train_dataloader = GraphDataLoader(
        dataset, sampler=train_sampler, batch_size=batch_size, drop_last=False, collate_fn=collate_data)
    test_dataloader = GraphDataLoader(
        dataset, sampler=test_sampler, batch_size=batch_size, drop_last=False, collate_fn=collate_data)

    return train_dataloader, test_dataloader

class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(1, 1024)
        self.conv2 = GraphConv(1024, 1024)
        self.conv3 = GraphConv(1024, 512)
        self.conv4 = GraphConv(512, 512)
        self.fn1 = nn.Linear(512, 512)
        self.fn2 = nn.Linear(512, 256)
        self.fn3 = nn.Linear(256, 2)

    def forward(self, g, h):
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        h = F.relu(self.conv3(g, h))
        h = F.relu(self.conv4(g, h))
        with g.local_scope():
            g.ndata['h'] = h
            hg = dgl.mean_nodes(g, 'h')
            gh = F.relu(self.fn1(hg))
            gh = F.relu(self.fn2(gh))
            return F.log_softmax(self.fn3(gh), dim=0)

def train_model(model, train_dataloader, num_epochs = 5):
    param = torch.optim.Adam(model.parameters())
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batched_graph, labels in train_dataloader:
            param.zero_grad()
            logits = model(batched_graph, batched_graph.ndata['node_features'])
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            param.step()
            epoch_loss += loss.detach().item()
        print('Epoch {}: {}'.format(epoch + 1, epoch_loss))

    return model

def evaluate_model(model, test_dataloader):
    num_correct, num_tests = 0, 0
    for batched_graph, labels in test_dataloader:
        pred = model(batched_graph, batched_graph.ndata['node_features'].float())
        num_correct += (pred.argmax(1) == labels).sum().item()
        num_tests += len(labels)

    return num_correct / num_tests

if __name__ == "__main__":
    dataset = Flow_Dataset()
    train_dataloader, test_dataloader = get_samples(dataset, 0.8, 16)

    model = GCN()
    model = train_model(model, train_dataloader, 10)

    acc = evaluate_model(model, test_dataloader)
    print('Test accuracy:', acc)