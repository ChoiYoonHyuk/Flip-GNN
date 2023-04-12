import numpy as np
import random
from tqdm import tqdm
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.datasets import Planetoid, WikipediaNetwork, Actor, WebKB

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

seed = 0
move = 0
lr = 1e-3
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
if dataset.root == '/tmp/Cora' or dataset.root == '/tmp/Citeseer' or dataset.root == '/tmp/Pubmed':
    num_class = dataset.num_classes
    labels = dict()
    
    for x in range(len(data.y)):
        label = int(data.y[x])
        
        try:
            labels[label].append(x)
        except KeyError:
            labels[label] = [x]

    train_mask, valid_mask, test_mask = [], [], []
    for c in range(num_class):
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

class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(dataset.num_node_features, 64)
        self.fc2 = nn.Linear(64, num_class)

    def forward(self, data):
        x = data.x

        #x = F.dropout(F.relu(self.fc1(x)))
        x = F.dropout(torch.sigmoid(self.fc1(x)))
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)


model = MLP().to(device)
optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

model.train()

early_stop, best_value, best_epoch = 0, 0, 0
best_valid = 0

adj_matrix = (to_dense_adj(data.edge_index) > 0).squeeze(0)
#hop_2 = torch.matmul(1.0 * adj_matrix, 1.0 * adj_matrix) + adj_matrix > 0
hop_2 = torch.matmul(1.0 * adj_matrix, 1.0 * adj_matrix) > 0
adj_matrix.fill_diagonal_(False)
hop_2.fill_diagonal_(False)

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
    
ls = torch.where(ls > 0, 1, 0)
ls1 = torch.where(torch.where(ls1 > 0, 1, 0) - ls > 0, 1, 0)
ls2 = torch.where(torch.where(ls2 > 0, 1, 0) - ls - ls1 > 0, 1, 0)
other = torch.where(torch.ones(len(data.x[0])).to(device) - (ls + ls1 + ls2) > 0, 1, 0)
zero_sum = torch.zeros(dataset.num_node_features).to(device)

for epoch in tqdm(range(epoch)):
    #data.x = -(1.0 - abs(data.x))
    optim.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    #loss = F.nll_loss(out, data.y)
    #optim.zero_grad(set_to_none=True)
    loss.backward()
    zero = torch.sum(torch.where(model.fc1.weight.grad.T.detach() == 0, 1, 0), 1)
    tt = torch.sum(torch.abs(model.fc1.weight.grad.T.detach()), 1)
    zero_sum[torch.where(zero > 0)] += 1
    #print(torch.mean(tt[ls]), torch.mean(tt[ls1]), torch.mean(tt[ls2]), torch.mean(tt[other]))
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
