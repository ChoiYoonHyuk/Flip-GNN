import numpy as np
import random
import math
from tqdm import tqdm
from sklearn.metrics import f1_score
import argparse

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.datasets import Planetoid, WikipediaNetwork, Actor, WebKB
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, to_dense_adj, dense_to_sparse

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

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
adj = torch.sum(to_dense_adj(data.edge_index).squeeze(0), dim=0)
num_class = dataset.num_classes
tmp = []

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
        self.gcn_1 = GATv2Conv(dataset.num_node_features, 8, heads=8, dropout=0.6)
        
        self.gcn_2 = GATv2Conv(64, num_class, heads=1, concat=False, dropout=0.6)
        
    def forward(self, data, e_w, idx):
        x, edge_index = data.x, data.edge_index
        
        if idx == 0:
            x, ed = self.gcn_1(x, edge_index, return_attention_weights=True)
            x = F.dropout(F.relu(x))
            x_out, eds = self.gcn_2(x, edge_index, return_attention_weights=True)
            
            err = 0
            
            eidx, att = eds[0], eds[1]
            eid, att = remove_self_loops(eidx, att)
            # deg * rm_e
            degree = adj[eid[0]]
            f_e = torch.mul(att.view(-1), degree)
            #print(max(f_e), torch.mean(f_e), min(f_e))
            
        else:
            x = F.dropout(F.relu(self.gcn_1(x, edge_index, e_w)))
            x_out = self.gcn_2(x, edge_index, e_w)        
            err = 0
            #print(F.log_softmax(x_out, dim=1))
            
        return F.log_softmax(x_out, dim=1), f_e, eid


#data.x = data.x * 1.05 - 0.05

out = 0
tmp = []
best_value = 0
idx_1, idx_2 = 100, 100
l_u_edges = torch.zeros(len(data.edge_index[0])).to(device)
param = 1.0
jj = 0

'''for p in range(len(data.edge_index[0])):
    e_1, e_2 = data.edge_index[0][idx], data.edge_index[1][idx]  
    if data.train_mask[e_1] == False and data.train_mask[e_2] == False:
        l_u_edges[p] = .5
    if data.train_mask[e_1] == False or data.train_mask[e_2] == False:
        l_u_edges[p] = 1.0
        
    l_1, l_2 = data.y[data.edge_index[0][p]], data.y[data.edge_index[1][p]]
    k = random.randrange(0, 100)
    if l_1 == l_2:
        tmp.append(1.0)
    else:
        tmp.append(.0)
    if l_1 == l_2:
        jj += 1
        if k < idx_1:
            tmp.append(1.0)
        else:
            tmp.append(-0.1)
    else:
        if k < idx_2:
            tmp.append(-0.1)
        else:
            tmp.append(1.0)'''
            
edge_weights = torch.tensor(tmp).to(device)

model = Net().to(device)
model.train()
optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
epoch, early_stop, best_epoch, idx = 500, 0, 0, 0
best_valid = 0
e_w = 0

tmp2, best_x = [], 0


        
for _ in tqdm(range(1000)):    
    model.train()
    
    out, _, _ = model(data, 0, 0)

    optim.zero_grad()
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask]) * reg #+ err
    #loss = F.nll_loss(out, data.y)
    loss.backward()
    optim.step()
    
    with torch.no_grad():
        model.eval()
        pred, edge_weight, eid = model(data, 0, 0)
        _, pred = pred.max(dim=1)
        correct = float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
        acc = correct / data.test_mask.sum().item()
        valid = float(pred[data.val_mask].eq(data.y[data.val_mask]).sum().item())
        
        if valid > best_valid:
            best_valid = valid
            best_value, best_epoch = acc, early_stop
            print(best_value)
            
            #edge_weights = torch.tensor(tmp).to(device)
            #n_cut = int(sum(edge_weights))
            #thresh = torch.topk(edge_weight, n_cut)[0][n_cut-1]
            #e_w = torch.where(edge_weight > thresh, torch.ones(1).to(device), torch.zeros(1).to(device))
            #f1 = f1_score(e_w.detach().cpu().numpy(), edge_weights.detach().cpu().numpy())
            #print(torch.sum(torch.logical_and(edge_weights, e_w)))
                
print(best_valid / data.val_mask.sum().item(), best_value)

