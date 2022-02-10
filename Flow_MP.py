import dgl, torch, argparse, json
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F

from datasets.Flow.Flow2_Dataset import Flow_Dataset
from models.MP import MPNN

def get_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="Please give a config.json file!")
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)

    params = config["params"]
    net_params = config["net_params"]

    return params, net_params

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

def train_model(model, train_dataloader, num_epochs = 5):
    param = torch.optim.Adam(model.parameters())
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batched_graph, labels in train_dataloader:
            param.zero_grad()
            logits = model(batched_graph.to('cuda:0'), batched_graph.ndata['node_features'].to('cuda:0'), batched_graph.edata['edge_features'].to('cuda:0'))
            loss = F.cross_entropy(logits, labels.to('cuda:0'))
            loss.backward()
            param.step()
            epoch_loss += loss.detach().item()
        print('Epoch {}: {}'.format(epoch + 1, epoch_loss))

    return model


def evaluate_model(model, test_dataloader):
    num_correct, num_tests = 0, 0
    for batched_graph, labels in test_dataloader:
        pred = model(batched_graph.to('cuda:0'), batched_graph.ndata['node_features'].to('cuda:0'), batched_graph.edata['edge_features'].to('cuda:0'))
        num_correct += (pred.argmax(1) == labels.to('cuda:0')).sum().item()
        num_tests += len(labels)

    return num_correct / num_tests

if __name__ == "__main__":
    params, net_params = get_parameters()
    batch_size = params["batch_size"]
    epochs = params["epochs"]

    dataset = Flow_Dataset()
    train_dataloader, test_dataloader = get_samples(dataset, 0.8, batch_size)

    model = MPNN(net_params)
    model = train_model(model.to('cuda:0'), train_dataloader, epochs)

    acc = evaluate_model(model.to('cuda:0'), test_dataloader)
    print('Test accuracy:', acc)