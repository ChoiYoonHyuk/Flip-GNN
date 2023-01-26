import argparse
import math
import random
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid, WikipediaNetwork, Actor, WebKB


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

seed = 0
move = 0
lr = 1e-3
epoch = 10000
set_bias = True
multi_learning = True


parser = argparse.ArgumentParser(description='Dataset')
parser.add_argument('data', type=int, help='data selector')
args = parser.parse_args()
data_id = args.data

if data_id == 0:
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    alpha, beta = .1, .02
elif data_id == 1:
    dataset = Planetoid(root='/tmp/Citeseer', name='Citeseer')
    alpha, beta = .01, .001
elif data_id == 2:
    dataset = Planetoid(root='/tmp/Pubmed', name='Pubmed')
    alpha, beta = 1, .01
elif data_id == 3:
    dataset = WikipediaNetwork(root='/tmp/Chameleon', name='chameleon')
    alpha, beta = 1, .0001
elif data_id == 4:
    dataset = WikipediaNetwork(root='/tmp/Squirrel', name='squirrel')
    alpha, beta = 1, .01
elif data_id == 5:
    dataset = Actor(root='/tmp/Actor')
    alpha, beta = .0001, .0001
elif data_id == 6:
    dataset = WebKB(root='/tmp/Cornell', name='Cornell')
    alpha, beta = 1, .0001
elif data_id == 7:
    dataset = WebKB(root='/tmp/Texas', name='Texas')
    alpha, beta = 1, .0001
elif data_id == 8:
    dataset = WebKB(root='/tmp/Wisconsin', name='Wisconsin')
    alpha, beta = .01, .001
    

data = dataset[0].to(device)

wrt = dataset.root[5:]

f = open('./convergence/' + wrt + '.txt', 'a')

# Remove outlier class in Chameleon / Squirrel / Actor
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
    lc = 5
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

# Flip GCN
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        if set_bias:
            self.conv1 = GCNConv(dataset.num_node_features + 1, 64)
            self.conv2 = GCNConv(64, dataset.num_classes)
        else:
            self.conv1 = GCNConv(dataset.num_node_features + 1, 64, bias=False)
            self.conv2 = GCNConv(64, dataset.num_classes, bias=False)
        
        # Identifier of the boundary is flipped or not
        self.k_value = 0
        
        # First flip point (0.5, ..., 0.5, 0)
        #self.t = torch.cat((torch.empty(dataset.num_node_features).fill_(.5).to(device), torch.zeros(1).to(device)))
        self.t = torch.cat((torch.empty(dataset.num_node_features).fill_(.0).to(device), torch.zeros(1).to(device)))
        
    def forward(self, x, edge_index, idx):
        state = model.state_dict()['conv1.weight'].detach().clone()
        bias = state[data.num_features]
        
        # If the boundary is flipped and the input is non-flipped (data.x), set bias as 0
        if idx == 0 and self.k_value == 1:
            state[data.num_features] = torch.zeros(64).to(device)
            
            model.state_dict()['conv1.weight'].data.copy_(state)
            self.k_value = 0
            
        # If the boundary is non-flipped and the input is flipped (1 - data.x), adjust bias as below
        elif idx == 1 and self.k_value == 0:
            refine = torch.matmul(self.t, state)
            state[data.num_features] =  - 2 * refine
                        
            model.state_dict()['conv1.weight'].data.copy_(state)
            self.k_value = 1
            
        # First convolution
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.7, training=self.training)
        
        # If input is non-flipped (idx = 0), proceed to second convolution with zero padding
        if idx == 0:
            x = self.conv2(x, edge_index)
        # If the input is flipped (idx = 1), proceed to second convolution with one padding and flipped -x
        else:
            x = self.conv2(-x, edge_index)
            
        return F.log_softmax(x, dim=1)


model = Net().to(device)
optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

# Padding feature w.r.t. each space (original and flipped)
zero, one = torch.zeros(len(data.x), 1).to(device), torch.ones(len(data.x), 1).to(device)
#without_bias, with_bias = torch.cat((data.x, zero), 1), torch.cat((1 - data.x, one), 1)
without_bias, with_bias = torch.cat((data.x, zero), 1), torch.cat((- data.x, one), 1)

idx, start, early_stop = 0, 0, 0
valid, best_valid, best_value, best_epoch, best_test = 0, 0, 0, 0, 0

for epoch in tqdm(range(epoch)):
    optim.zero_grad()
    model.train()
    
    # Based on the hyper-parameter (balance), choose whether to flip or not 
    if idx == 0:
        out = model(without_bias, data.edge_index, 0)
    else:
        out = model(with_bias, data.edge_index, 1)
    
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    # Adjust the gradients
    if idx == 0:
        model.conv1.weight.grad = model.conv1.weight.grad.clone() * alpha
        model.conv2.weight.grad = model.conv2.weight.grad.clone() * alpha
    else:
        model.conv1.weight.grad = model.conv1.weight.grad.clone() * beta
        model.conv2.weight.grad = model.conv2.weight.grad.clone() * beta
        #model.conv1.weight.grad[data.num_features] = model.conv1.weight.grad[data.num_features].clone() * 0
    optim.step()
    
    early_stop += 1
    
    with torch.no_grad():
        model.eval()
        
        # Evaluation
        if idx == 0:
            _, pred = model(without_bias, data.edge_index, 0).max(dim=1)
        else:
            _, pred = model(with_bias, data.edge_index, 1).max(dim=1)
        
        # Retrieve validation score
        correct = float(pred[data.val_mask].eq(data.y[data.val_mask]).sum().item())
        val_acc = correct / data.val_mask.sum().item()
        
        # You can see a test score of flipped points here 
        if idx == 0:
            test_acc = float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()) / data.test_mask.sum().item()
            if test_acc > best_test:
                best_test = test_acc
            #print('Test: {:4f}'.format(test_acc))
            f.write('%.3f\n' % test_acc)
        
        # If attains best valudation score, get test score
        if val_acc > best_valid:
            best_valid = val_acc
            correct = float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
            acc = correct / data.test_mask.sum().item()
            print('Accuracy: {:4f}'.format(acc))
            early_stop = 0
    
    if multi_learning:
        idx = 1 - idx
    
    if early_stop > 2500:
        break
