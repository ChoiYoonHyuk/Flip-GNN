import numpy as np
import argparse
import random
import math
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid, WikipediaNetwork, Actor, WebKB

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

seed = 0
move = 0.00
lr = 1e-3
reg = 1.0

lc = 20

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
num_class = dataset.num_classes

idx = 0

    
if dataset.root == '/tmp/Chameleon' or dataset.root == '/tmp/Squirrel' or dataset.root == '/tmp/Actor':
    num_class = 5
    lc = 20
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


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.gcn_1 = GCNConv(dataset.num_node_features, 64)
        self.gcn_2 = GCNConv(64, num_class)
        self.t = torch.empty(dataset.num_node_features).fill_(.5).to(device)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        #print(model.state_dict()['gcn_1.bias'])
        
        x = F.dropout(F.relu(self.gcn_1(x, edge_index)))
        #x = F.dropout(torch.sigmoid(self.gcn_1(x, edge_index)))
        x_out = self.gcn_2(x, edge_index)
            
        return F.log_softmax(x_out, dim=1), x_out


#data.x = data.x + 0.01

out = 0
tmp, tmp2 = [], []
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
        tmp2.append(.0)
    else:
        tmp.append(.0)
        tmp2.append(.0)
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
            
#edge_weights = torch.tensor(tmp).to(device)
#edge_weightss = torch.tensor(tmp2).to(device)

adj_matrix = (to_dense_adj(data.edge_index) > 0).squeeze(0)
#hop_2 = torch.matmul(1.0 * adj_matrix, 1.0 * adj_matrix) + adj_matrix > 0
hop_2 = torch.matmul(1.0 * adj_matrix, 1.0 * adj_matrix) > 0
adj_matrix.fill_diagonal_(False)
hop_2.fill_diagonal_(False)

idx = 0
ls = torch.zeros(len(data.x[0])).to(device)
ls1 = torch.zeros(len(data.x[0])).to(device)
ls2 = torch.zeros(len(data.x[0])).to(device)

out = torch.zeros(len(data.x)).to(device)
out2 = torch.zeros(len(data.x)).to(device)
for i in range(len(data.x)):
    if data.train_mask[i]:
        out += adj_matrix[i]
        out2 += hop_2[i]

out = torch.where(torch.where(out > 0, 1, 0) - data.train_mask * 1 > 0, 1, 0)
out2 = torch.where(torch.where(out2 > 0, 1, 0) - out - data.train_mask * 1 > 0, 1, 0)

for i in range(len(data.x)):
    if data.train_mask[i]:
        ls += data.x[i]
    elif out[i]:
        ls1 += data.x[i]
    elif out2[i]:
        ls2 += data.x[i]
    
other = torch.where(torch.ones(len(data.x[0])).to(device) - (ls + ls1 + ls2) > 0, 1, 0)
ls = torch.where(ls > 0, 1, 0)
ls1 = torch.where(torch.where(ls1 > 0, 1, 0) - ls > 0, 1, 0)
ls2 = torch.where(torch.where(ls2 > 0, 1, 0) - ls - ls1 > 0, 1, 0)
zero_sum = torch.zeros(dataset.num_node_features).to(device)

model = Net().to(device)
model.train()
optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
epoch, early_stop, best_epoch, idx = 500, 0, 0, 0
best_valid = 0
e_w = 0
orig_out = []


tmp2, best_x = [], 0
new_mask, new_y = 0, 0


for _ in tqdm(range(1001)):    
    model.train()
    
    out, _ = model(data)

    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    #loss = F.nll_loss(out, data.y)
    optim.zero_grad()
    loss.backward()
    #zero = torch.sum(torch.where(model.gcn_1.weight.grad.detach() == 0, 1, 0), 1)
    #tt = torch.sum(torch.abs(model.gcn_1.weight.grad.detach()), 1)
    #zero_sum[torch.where(zero > 0)] += 1
    #print(torch.mean(tt[ls]), torch.mean(tt[ls1]), torch.mean(tt[ls2]), torch.mean(tt[other]))
    optim.step()
    
    with torch.no_grad():
        model.eval()
        pred, x = model(data)
        _, pred = pred.max(dim=1)
        correct = float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
        acc = correct / data.test_mask.sum().item()
        orig_out.append(float(acc))
        valid = float(pred[data.val_mask].eq(data.y[data.val_mask]).sum().item())
        
        if valid > best_valid:
            best_valid = valid
            best_value, best_epoch = acc, early_stop
            best_x = x
            print(best_value)
            '''tmp, new_y = x.max(dim=1)
            new_mask = torch.where(tmp > 0.3, 1, 0)
            new_mask = new_mask > 0
            #torch.save(model.state_dict(), './param/gcn_' + dataset.name + '.pth')
            
            x1, x2 = pred[data.edge_index[0]], pred[data.edge_index[1]]
            #out = torch.sum(x1*x2, dim=1)
            #thresh = torch.topk(out, int(sum(edge_weights)/param))[0][int(sum(edge_weights)/param) - 1]
            #e_w = torch.where(out > thresh, torch.ones(1).to(device), torch.zeros(1).to(device))
            #mat = torch.sum(torch.logical_not(torch.logical_xor(edge_weights, e_w)))
            e_w = torch.logical_and(x1, x2)
            mat = torch.sum(torch.logical_and(edge_weights, e_w))
            #print(mat / len(data.edge_index[0]))
            
            k, y = F.softmax(best_x, dim=1).max(dim=1)'''
#print(best_valid / data.val_mask.sum().item(), best_value)
#print(ls[torch.where(zero_sum > 500)], ls1[torch.where(zero_sum > 500)], ls2[torch.where(zero_sum > 500)], other[torch.where(zero_sum > 500)])
write_file = './convergence/gcn_' + dataset.root[5:] + '.txt'
w = open(write_file, 'a')
for i in orig_out:
    w.write('%.3f ' % i)
    
'''model = Net().to(device)
optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
best_valid = 0
exc_train_mask = torch.logical_and(new_mask, torch.logical_not(data.train_mask))

for _ in range(500):    
    model.train()
    
    out, _, err = model(data, edge_weights, edge_weightss, 0)

    optim.zero_grad()
    
    loss = F.nll_loss(out[new_mask], new_y[new_mask])
    #loss = F.nll_loss(out[exc_train_mask], new_y[exc_train_mask])
    loss.backward()
    optim.step()
    
    with torch.no_grad():
        model.eval()
        pred, x, _ = model(data, edge_weights, edge_weightss, 0)
        _, pred = pred.max(dim=1)
        correct = float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
        acc = correct / data.test_mask.sum().item()
        valid = float(pred[data.val_mask].eq(data.y[data.val_mask]).sum().item())
        
        if valid > best_valid:
            best_valid = valid
            best_value, best_epoch = acc, early_stop
            best_x = F.softmax(x, dim=1)
            #torch.save(model.state_dict(), './param/gcn_' + dataset.name + '.pth')

print(best_valid / data.val_mask.sum().item(), best_value)'''