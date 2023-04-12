import os.path as osp
import numpy as np
import random
import argparse
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, WikipediaNetwork, Actor, WebKB
import torch_geometric.transforms as T
#from torch_geometric.nn import GATConv
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn import inits
from torch_geometric.nn.inits import glorot, zeros
from typing import Union, Tuple, Optional
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, to_dense_adj, dense_to_sparse
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType, OptTensor)
from torch_geometric.nn.conv import MessagePassing
from sklearn.metrics import f1_score

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

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
data = dataset[0].to(device)

num_class = dataset.num_classes
tmp = []

lc = 20

idx = 0
    
if dataset.root == '/tmp/Chameleon' or dataset.root == '/tmp/Squirrel' or dataset.root == '/tmp/Actor':
    num_class = 20
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
    
seed = 10
move = 0.0
epoch = 2000
lr = 1e-2
#data.x = 1.05 * data.x - 0.05

adj = torch.sum(to_dense_adj(data.edge_index).squeeze(0), dim=0)

class Linear(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True,
                 weight_initializer: Optional[str] = None,
                 bias_initializer: Optional[str] = None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer

        if in_channels > 0:
            self.weight = Parameter(torch.Tensor(out_channels, in_channels))
        else:
            self.weight = torch.nn.parameter.UninitializedParameter()
            self._hook = self.register_forward_pre_hook(
                self.initialize_parameters)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def __deepcopy__(self, memo):
        out = Linear(self.in_channels, self.out_channels, self.bias
                     is not None, self.weight_initializer,
                     self.bias_initializer)
        if self.in_channels > 0:
            out.weight = copy.deepcopy(self.weight, memo)
        if self.bias is not None:
            out.bias = copy.deepcopy(self.bias, memo)
        return out

    def reset_parameters(self):
        if self.in_channels > 0:
            if self.weight_initializer == 'glorot':
                inits.glorot(self.weight)
            elif self.weight_initializer == 'uniform':
                bound = 1.0 / math.sqrt(self.weight.size(-1))
                torch.nn.init.uniform_(self.weight.data, -bound, bound)
            elif self.weight_initializer == 'kaiming_uniform':
                inits.kaiming_uniform(self.weight, fan=self.in_channels,
                                      a=math.sqrt(5))
            elif self.weight_initializer is None:
                inits.kaiming_uniform(self.weight, fan=self.in_channels,
                                      a=math.sqrt(5))
            else:
                raise RuntimeError(
                    f"Linear layer weight initializer "
                    f"'{self.weight_initializer}' is not supported")

        if self.in_channels > 0 and self.bias is not None:
            if self.bias_initializer == 'zeros':
                inits.zeros(self.bias)
            elif self.bias_initializer is None:
                inits.uniform(self.in_channels, self.bias)
            else:
                raise RuntimeError(
                    f"Linear layer bias initializer "
                    f"'{self.bias_initializer}' is not supported")


    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight, self.bias)


    @torch.no_grad()
    def initialize_parameters(self, module, input):
        if isinstance(self.weight, torch.nn.parameter.UninitializedParameter):
            self.in_channels = input[0].size(-1)
            self.weight.materialize((self.out_channels, self.in_channels))
            self.reset_parameters()
        module._hook.remove()
        delattr(module, '_hook')


    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, bias={self.bias is not None})')


class GATConv(MessagePassing):
    _alpha: OptTensor

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = 'mean',
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value

        # In case we are operating in bipartite graphs, we apply separate
        # transformations 'lin_src' and 'lin_dst' to source and target nodes:
        '''if isinstance(in_channels, int):
            self.lin_src = Linear(in_channels, heads * out_channels,
                                  bias=False, weight_initializer='glorot')
            self.lin_dst = self.lin_src
        else:
            self.lin_src = Linear(in_channels[0], heads * out_channels, False,
                                  weight_initializer='glorot')
            self.lin_dst = Linear(in_channels[1], heads * out_channels, False,
                                  weight_initializer='glorot')'''
        if isinstance(in_channels, int):
            self.lin_src = Linear(in_channels, out_channels,
                                  bias=False, weight_initializer='glorot')
            self.lin_dst = self.lin_src
        else:
            self.lin_src = Linear(in_channels[0], out_channels, False,
                                  weight_initializer='glorot')
            self.lin_dst = Linear(in_channels[1], out_channels, False,
                                  weight_initializer='glorot')

        # The learnable parameters to compute attention coefficients:
        self.att_src = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = Parameter(torch.Tensor(1, heads, out_channels))

        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False,
                                   weight_initializer='glorot')
            self.att_edge = Parameter(torch.Tensor(1, heads, out_channels))
        else:
            self.lin_edge = None
            self.register_parameter('att_edge', None)

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_src.reset_parameters()
        self.lin_dst.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        glorot(self.att_src)
        glorot(self.att_dst)
        glorot(self.att_edge)
        zeros(self.bias)


    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None,
                return_attention_weights=None):

        H, C = self.heads, self.out_channels

        # We first transform the input node features. If a tuple is passed, we
        # transform source and target node features via separate weights:
        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in 'GATConv'"
            #x_src = x_dst = self.lin_src(x).view(-1, H, C)
            x_src = x_dst = self.lin_src(x).view(-1, 1, C)
        else:  # Tuple of source and target node features:
            x_src, x_dst = x
            assert x_src.dim() == 2, "Static graphs not supported in 'GATConv'"
            #x_src = self.lin_src(x_src).view(-1, H, C)
            x_src = self.lin_src(x_src).view(-1, 1, C)
            if x_dst is not None:
                #x_dst = self.lin_dst(x_dst).view(-1, H, C)
                x_dst = self.lin_dst(x_dst).view(-1, 1, C)

        x = (x_src, x_dst)

        # Next, we compute node-level attention coefficients, both for source
        # and target nodes (if present):
        alpha_src = (x_src * self.att_src).sum(dim=-1)
        alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
        alpha = (alpha_src, alpha_dst)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                # We only want to add self-loops for nodes that appear both as
                # source and target nodes:
                num_nodes = x_src.size(0)
                if x_dst is not None:
                    num_nodes = min(num_nodes, x_dst.size(0))
                num_nodes = min(size) if size is not None else num_nodes
                edge_index, edge_attr = remove_self_loops(
                    edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value=self.fill_value,
                    num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form")

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor, edge_attr: OptTensor)  # noqa
        out = self.propagate(edge_index, x=x, alpha=alpha, edge_attr=edge_attr,
                             size=size)

        alpha = self._alpha
        assert alpha is not None
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias
        
        return_attention_weights = True
        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out


    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        # Given edge-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i

        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            assert self.lin_edge is not None
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            alpha_edge = (edge_attr * self.att_edge).sum(dim=-1)
            alpha = alpha + alpha_edge

        ########################################
        #           Attention layer            #
        ########################################
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha  # Save for later use.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = GATConv(dataset.num_features, 8, heads=8, dropout=0.6)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(64, dataset.num_classes, heads=1, concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        x, ei = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x_out, eid = self.conv2(x, edge_index)
        eidx, att = eid[0], eid[1]
        eid, att = remove_self_loops(eidx, att)
        # deg * rm_e
        degree = adj[eid[0]]
        f_e = torch.mul(att.view(-1), degree)
        #print(max(f_e), torch.mean(f_e), min(f_e))

        return F.log_softmax(x_out, dim=-1), f_e, eid


model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
data.val_mask = torch.logical_not(torch.logical_or(data.train_mask, data.test_mask))

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
edge_weights = torch.ones(len(data.edge_index)).to(device)

def train(data):
    model.train()
    optimizer.zero_grad()
    out, _, _ = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    #loss = F.nll_loss(out, data.y)
    loss.backward()
    optimizer.step()


def test(data, first):
    model.eval()
    out, edge_weight, eid = model(data.x, data.edge_index)
    accs = []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = out[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    
    if first == 1:
        for p in range(len(eid[0])):
            l_1, l_2 = data.y[eid[0][p]], data.y[eid[1][p]]
            
            if l_1 == l_2:
                tmp.append(1.0)
            else:
                tmp.append(0.0)
    #edge_weights = torch.tensor(tmp).to(device)
    '''n_cut = int(sum(edge_weights))
    thresh = torch.topk(edge_weight, n_cut)[0][n_cut-1]
    e_w = torch.where(edge_weight > thresh, torch.ones(1).to(device), torch.zeros(1).to(device))
    f1 = f1_score(e_w.detach().cpu().numpy(), edge_weights.detach().cpu().numpy())
    accs.append(f1)
    logic_acc = torch.sum(torch.logical_and(edge_weights, e_w))
    accs.append(logic_acc)'''
    return accs


best = 0
best_val = 0
orig_out = []
for epoch in tqdm(range(0, 1001)):
    train(data)
    
    #train_acc, val_acc, test_acc, f1, logic_acc = test(data, epoch)
    train_acc, val_acc, test_acc = test(data, epoch)
    orig_out.append(test_acc)
    
    if val_acc > best_val:
        best_val = val_acc
        best = test_acc
        print(test_acc)
        #print('%.3f %d' % (f1, logic_acc))
write_file = './convergence/gat_' + dataset.root[5:] + '.txt'
w = open(write_file, 'a')
for i in orig_out:
    w.write('%.3f ' % i)