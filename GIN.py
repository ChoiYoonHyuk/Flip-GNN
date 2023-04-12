import os.path as osp
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.datasets import TUDataset, Planetoid, WikipediaNetwork, Actor, WebKB
from torch_geometric.data import DataLoader
from torch_geometric.nn import GINConv, global_add_pool

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser(description='Dataset')
parser.add_argument('data', type=int, help='data selector')
args = parser.parse_args()
data_id = args.data

if data_id == 0:
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
elif data_id == 1:
    dataset = Planetoid(root='/tmp/Citeseer', name='Citeseer')
elif data_id == 2:
    dataset = Planetoid(root='/tmp/Pubmed', name='Pubmed')
elif data_id == 3:
    dataset = WikipediaNetwork(root='/tmp/Chameleon', name='chameleon')
elif data_id == 4:
    dataset = WikipediaNetwork(root='/tmp/Squirrel', name='squirrel')
elif data_id == 5:
    dataset = Actor(root='/tmp/Actor')
elif data_id == 6:
    dataset = WebKB(root='/tmp/Cornell', name='Cornell')
elif data_id == 7:
    dataset = WebKB(root='/tmp/Texas', name='Texas')
else:
    dataset = WebKB(root='/tmp/Wisconsin', name='Wisconsin')

data = dataset[0].to(device)
num_class = dataset.num_classes

idx = 0
if dataset.root == '/tmp/Chameleon' or dataset.root == '/tmp/Squirrel' or dataset.root == '/tmp/Actor':
    num_class = 5
    labels = dict()
    
    for x in range(len(data.y)):
        label = int(data.y[x])
        
        try:
            labels[label].append(x)
        except KeyError:
            labels[label] = [x]

    train_mask, valid_mask, test_mask = [], [], []
    for c in range(5):
        train_mask.extend(labels[c][0:20])
        cut = int((len(labels[c]) - 20) / 2)
        valid_mask.extend(labels[c][20:20+cut])
        test_mask.extend(labels[c][20+cut:len(labels[c])])
    
    train, valid, test = [], [], []
    for x in range(len(data.y)):
        if x in train_mask:
            train.append(True)
            valid.append(False)
            test.append(False)
        elif x in valid_mask:
            train.append(False)
            valid.append(True)
            test.append(False)
        elif x in test_mask:
            train.append(False)
            valid.append(False)
            test.append(True)
        else:
            train.append(False)
            valid.append(False)
            test.append(True)
    
    data.train_mask, data.val_mask, data.test_mask = torch.tensor(train).to(device), torch.tensor(valid).to(device), torch.tensor(test).to(device)
    data.y = torch.where(data.y > 4, 0, data.y)

if dataset.root == '/tmp/Cornell' or dataset.root == '/tmp/Texas' or dataset.root == '/tmp/Wisconsin':
    num_class = 5
    labels = dict()
    
    for x in range(len(data.y)):
        label = int(data.y[x])
        
        try:
            labels[label].append(x)
        except KeyError:
            labels[label] = [x]

    train_mask, valid_mask, test_mask = [], [], []
    for c in range(5):
        train_mask.extend(labels[c][0:5])
        cut = int((len(labels[c]) - 5) / 2)
        valid_mask.extend(labels[c][5:5+cut])
        test_mask.extend(labels[c][5+cut:len(labels[c])])
    
    train, valid, test = [], [], []
    for x in range(len(data.y)):
        if x in train_mask:
            train.append(True)
            valid.append(False)
            test.append(False)
        elif x in valid_mask:
            train.append(False)
            valid.append(True)
            test.append(False)
        elif x in test_mask:
            train.append(False)
            valid.append(False)
            test.append(True)
        else:
            train.append(False)
            valid.append(False)
            test.append(True)
    
    data.train_mask, data.val_mask, data.test_mask = torch.tensor(train).to(device), torch.tensor(valid).to(device), torch.tensor(test).to(device)
    data.y = torch.where(data.y > 4, 0, data.y)

epoch = 2000
lr = 1e-3


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        num_features = dataset.num_features
        dim = 64

        nn1 = Sequential(Linear(dataset.num_features, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.conv1 = GINConv(Linear(dataset.num_features, dim))
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.conv2 = GINConv(Linear(dim, dim))
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.conv3 = GINConv(Linear(dim, dim))
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINConv(nn4)
        self.conv4 = GINConv(Linear(dim, dim))
        self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINConv(nn5)
        self.conv5 = GINConv(Linear(dim, dim))
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.fc1 = Linear(dim, dim)
        self.fc2 = Linear(dim, dataset.num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        #x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        #x = self.bn2(x)
        '''x = F.relu(self.conv3(x, edge_index))
        #x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        #x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        #x = self.bn5(x)'''
        
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
data.to(device)
data.val_mask = torch.logical_not(torch.logical_or(data.train_mask, data.test_mask))
best_valid, best = 0, 0


for epoch in tqdm(range(1, epoch)):
    model.train()

    optimizer.zero_grad()
    output = model(data.x, data.edge_index)
    loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
    loss.backward()
    
    optimizer.step()
    
    model.eval()

    out = model(data.x, data.edge_index)
    
    _, pred = out.max(dim=1)
    correct = float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    acc = correct / data.test_mask.sum().item()
    valid = float(pred[data.val_mask].eq(data.y[data.val_mask]).sum().item())
    
    if valid > best_valid:
        best_valid = valid
        best = acc
        print(best)
