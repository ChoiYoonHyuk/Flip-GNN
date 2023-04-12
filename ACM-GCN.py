import numpy as np
import random
from tqdm import tqdm
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.datasets import Planetoid, WikipediaNetwork, Actor, WebKB

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

seed = 0
move = 0
lr = 1e-2
epoch = 10000
lc = 20

'''np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)'''

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
        train_mask.extend(labels[c][0:lc])
        cut = int((len(labels[c]) - lc) / 2)
        valid_mask.extend(labels[c][lc:lc+cut])
        test_mask.extend(labels[c][lc+cut:len(labels[c])])
    
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
        train_mask.extend(labels[c][0:lc])
        cut = int((len(labels[c]) - lc) / 2)
        valid_mask.extend(labels[c][lc:lc+cut])
        test_mask.extend(labels[c][lc+cut:len(labels[c])])
    
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


adj = (to_dense_adj(data.edge_index) > 0).squeeze(0)
adj_low = torch.eye(len(data.x)).to(device) + adj
low_norm = 1 / torch.sum(adj_low, dim=1)
adj = torch.eye(len(data.x)).to(device)
adj_low = (adj_low * low_norm.view(-1, 1))
adj_low = torch.matmul(adj_low, adj_low)
adj_high = (torch.eye(len(data.x)).to(device) - adj_low)



class ACM(torch.nn.Module):
    def __init__(self):
        super(ACM, self).__init__()
        self.ite = 0
        self.low = nn.Linear(dataset.num_node_features, 64)
        self.high = nn.Linear(dataset.num_node_features, 64)
        self.diag = nn.Linear(dataset.num_node_features, 64)
        
        '''self.low = GCNConv(dataset.num_node_features, 64)
        self.high = GCNConv(dataset.num_node_features, 64)
        self.diag = GCNConv(dataset.num_node_features, 64)'''
        
        self.lw = nn.Linear(64, 1)
        self.hw = nn.Linear(64, 1)
        self.dw = nn.Linear(64, 1)
        
        self.mix = nn.Linear(3, 3)
        
        self.l2 = nn.Linear(64, num_class)
        self.h2 = nn.Linear(64, num_class)
        self.d2 = nn.Linear(64, num_class)
        
        '''self.l2 = GCNConv(64, num_class)
        self.h2 = GCNConv(64, num_class)
        self.d2 = GCNConv(64, num_class)'''
        
        self.lw2 = nn.Linear(num_class, 1)
        self.hw2 = nn.Linear(num_class, 1)
        self.dw2 = nn.Linear(num_class, 1)
        
        self.mix2 = nn.Linear(3, 3)

    def forward(self, data):
        x = data.x
        opt = 1
        temp = pow(0.99, self.ite)
        
        if opt == 0:
            low = F.dropout(F.relu(self.low(x, adj_low[0], adj_low[1])))
            high = F.dropout(F.relu(self.high(x, adj_high[0], adj_high[1])))
            diag = F.dropout(F.relu(self.diag(x, adj[0], adj[1])))
        else:
            low = torch.matmul(adj_low, F.dropout(F.relu(self.low(x))))
            high = torch.matmul(adj_high, F.dropout(F.relu(self.high(x))))
            diag = torch.matmul(adj, F.dropout(F.relu(self.diag(x))))
        
        al = torch.sigmoid(self.lw(low))
        aw = torch.sigmoid(self.hw(high))
        dw = torch.sigmoid(self.dw(diag))
        cat = torch.cat((al, aw), 1)
        weight = F.softmax(self.mix(torch.cat((cat, dw), 1)))
        
        #x = F.relu(low * weight[:, 0].view(-1, 1) + high * weight[:, 1].view(-1, 1) + diag * weight[:, 2].view(-1, 1))
        x = low * weight[:, 0].view(-1, 1) + high * weight[:, 1].view(-1, 1) + diag * weight[:, 2].view(-1, 1) + low
        
        if opt == 0:
            low = F.dropout(F.relu(self.l2(x, adj_low[0], adj_low[1])))
            high = F.dropout(F.relu(self.h2(x, adj_high[0], adj_high[1])))
            diag = F.dropout(F.relu(self.d2(x, adj[0], adj[1])))
        else:
            low = torch.matmul(adj_low, F.dropout(F.relu(self.l2(x))))
            high = torch.matmul(adj_high, F.dropout(F.relu(self.h2(x))))
            diag = torch.matmul(adj, F.dropout(F.relu(self.d2(x))))
        
        al = torch.sigmoid(self.lw2(low))
        aw = torch.sigmoid(self.hw2(high))
        dw = torch.sigmoid(self.dw2(diag))
        cat = torch.cat((al, aw), 1)
        weight = F.softmax(self.mix2(torch.cat((cat, dw), 1)))
        
        x = low * weight[:, 0].view(-1, 1) + high * weight[:, 1].view(-1, 1) + diag * weight[:, 2].view(-1, 1) + low
        self.ite += 1
        
        return F.log_softmax(x, dim=1)


model = ACM().to(device)
optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

model.train()

early_stop, best_value, best_epoch = 0, 0, 0
best_valid = 0


for epoch in tqdm(range(epoch)):
    optim.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optim.step()
    
    with torch.no_grad():
        model.eval()
        _, pred = model(data).max(dim=1)
        correct = float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
        acc = correct / data.test_mask.sum().item()
        valid = float(pred[data.val_mask].eq(data.y[data.val_mask]).sum().item())
            
        if valid > best_valid:
            best_valid = valid
            best_value, best_epoch = acc, early_stop
            print(best_value)
