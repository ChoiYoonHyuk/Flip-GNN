import numpy as np
import random
import math
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch_geometric.nn import APPNP
from torch_geometric.datasets import Planetoid, WikipediaNetwork, Actor, WebKB

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

seed = 0
move = 0.00
lr = 1e-3
reg = 1.0


#np.random.seed(seed)
#random.seed(seed)
#torch.manual_seed(seed)

parser = argparse.ArgumentParser(description='Dataset')
parser.add_argument('data', type=int, help='data selector')
args = parser.parse_args()
data_id = args.data

if data_id == 0:
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    alpha, beta = .1, .01
    gamma = .1
elif data_id == 1:
    dataset = Planetoid(root='/tmp/Citeseer', name='Citeseer')
    alpha, beta = .1, .001
    gamma = .1
elif data_id == 2:
    dataset = Planetoid(root='/tmp/Pubmed', name='Pubmed')
    alpha, beta = 1, .01
    gamma = .1
elif data_id == 3:
    dataset = WikipediaNetwork(root='/tmp/Chameleon', name='chameleon')
    alpha, beta = 1, .01
    gamma = .3
elif data_id == 4:
    dataset = WikipediaNetwork(root='/tmp/Squirrel', name='squirrel')
    alpha, beta = 1, .01
    gamma = .3
elif data_id == 5:
    dataset = Actor(root='/tmp/Actor')
    alpha, beta = 1e-2, 1e-2
    gamma = .9
elif data_id == 6:
    dataset = WebKB(root='/tmp/Cornell', name='Cornell')
    alpha, beta = 1, .0001
    gamma = .5
elif data_id == 7:
    dataset = WebKB(root='/tmp/Texas', name='Texas')
    alpha, beta = 1, .0001
    gamma = .5
elif data_id == 8:
    dataset = WebKB(root='/tmp/Wisconsin', name='Wisconsin')
    alpha, beta = .01, .001
    gamma = .5

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


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.embed = nn.Linear(dataset.num_node_features + 1, 64)
        self.appnp = APPNP(10, gamma)
        self.pred = nn.Sequential(
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, dataset.num_classes),
        )
        self.pred = nn.Linear(64, dataset.num_classes)
        
        # Identifier of the boundary is flipped or not
        self.k_value = 0
        
        # First flip point (0.5, ..., 0.5, 0)
        #self.t = torch.cat((torch.empty(dataset.num_node_features).fill_(.5).to(device), torch.zeros(1).to(device)))
        self.t = torch.cat((torch.empty(dataset.num_node_features).fill_(.5).to(device), torch.zeros(1).to(device)))
        
    def forward(self, x, edge_index, idx):
        err = 0
        state = model.state_dict()['embed.weight'].T.detach().clone()
        bias = state[data.num_features]
        refine = torch.matmul(self.t, state)
        state[data.num_features] = -2 * refine
        
        if idx == 0 and self.k_value == 1:
            state[data.num_features] = torch.zeros(64).to(device)
            model.state_dict()['embed.weight'].data.copy_(state.T)
            
            self.k_value = 0
            
        elif idx == 1 and self.k_value == 0:
            model.state_dict()['embed.weight'].data.copy_(state.T)
            self.k_value = 1
            
        if idx == 0:
            x = F.dropout(F.relu(self.embed(x)))
        else:
            x = F.dropout(F.relu(-self.embed(x)))
        x = self.appnp(x, edge_index)
        x_out = self.pred(x)
            
        return F.log_softmax(x_out, dim=1), x_out, err, F.softmax(x_out, dim=1)


#data.x = data.x * 1.05 - 0.05

out = 0
tmp = []
best_value = 0
idx = 100
l_u_edges = torch.zeros(len(data.edge_index[0])).to(device)
param = 1.0


model = Net().to(device)
model.train()
optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
epoch, early_stop, best_epoch, idx = 500, 0, 0, 0
best_valid = 0
e_w = 0

def bal(x, idx):
    res = 0
    for i in range(num_class):
        if i != idx:
            if x[i]+x[idx] < 1e-3:
                res += x[i] * (1 - abs(x[i]-x[idx])/1e-3)
            else:
                res += x[i] * (1 - abs(x[i]-x[idx])/(x[i]+x[idx]))
    return res


def dissonance(x):
    dis = 0
    for i in range(len(data.x)):
        for j in range(num_class):
            if sum(x[i]) - x[i][j] < 1e-3:
                dis += (x[i][j] * bal(x[i], j)) / 1e-3
            else:
                dis += (x[i][j] * bal(x[i], j)) / (sum(x[i]) - x[i][j])
    return dis / len(data.x)

tmp2, best_x = [], 0
save = 0
zero, one = torch.zeros(len(data.x), 1).to(device), torch.ones(len(data.x), 1).to(device)
without_bias, with_bias = torch.cat((data.x, zero), 1), torch.cat((1 - data.x, one), 1)


for ee in tqdm(range(epoch)):
    idx = 0
    
    for _ in range(300):    
        model.train()
        
        if idx == 0:
            out, _, err, _ = model(without_bias, data.edge_index, 0)
        else:
            out, _, err, _ = model(with_bias, data.edge_index, 1)
        
        optim.zero_grad()
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask]) * reg #+ err
        loss.backward()
        if idx == 0:
            model.embed.weight.grad = model.embed.weight.grad.clone() * alpha
        else:
            model.embed.weight.grad = -model.embed.weight.grad.clone() * beta
        optim.step()
        
        with torch.no_grad():
            model.eval()
            if idx == 0:
                pred, x, _, xs = model(without_bias, data.edge_index, 0)
            else:
                pred, x, _, xs = model(with_bias, data.edge_index, 1)
            
            _, pred = pred.max(dim=1)
            correct = float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
            acc = correct / data.test_mask.sum().item()
            #if idx == 1:
                #print(acc)
            valid = float(pred[data.val_mask].eq(data.y[data.val_mask]).sum().item())
            
            if valid > best_valid:
                best_valid = valid
                best_value, best_epoch = acc, early_stop
                best_x = x
                save = xs
        idx = 1 - idx
    print(best_value)
