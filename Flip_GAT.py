import numpy as np
import argparse
import random
import math
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn import inits
from torch_geometric.nn.inits import glorot, zeros
from typing import Union, Tuple, Optional
from torch_geometric.datasets import Planetoid, WikipediaNetwork, Actor, WebKB
from torch.autograd import Function
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, to_dense_adj, dense_to_sparse
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType, OptTensor)
from torch_geometric.nn.conv import MessagePassing

######################### Read me #########################
# Please refer to Flip_GCN.py for further descrptions
# We refer to attention of pytorch_geometric implementation
# Our main source code starts from line 429
###########################################################

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

seed = 0
move = 0
lr = 1e-3
epoch = 20000
multi_learning = True


parser = argparse.ArgumentParser(description='Dataset')
parser.add_argument('data', type=int, help='data selector')
args = parser.parse_args()
data_id = args.data

if data_id == 0:
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    alpha, beta = .1, .01
elif data_id == 1:
    dataset = Planetoid(root='/tmp/Citeseer', name='Citeseer')
    alpha, beta = .01, .001
elif data_id == 2:
    dataset = Planetoid(root='/tmp/Pubmed', name='Pubmed')
    alpha, beta = 1, .001
elif data_id == 3:
    dataset = WikipediaNetwork(root='/tmp/Chameleon', name='chameleon')
    alpha, beta = 1, .01
elif data_id == 4:
    dataset = WikipediaNetwork(root='/tmp/Squirrel', name='squirrel')
    alpha, beta = 1, 1
elif data_id == 5:
    dataset = Actor(root='/tmp/Actor')
    alpha, beta = .1, .1
elif data_id == 6:
    dataset = WebKB(root='/tmp/Cornell', name='Cornell')
    alpha, beta = .1, .1
elif data_id == 7:
    dataset = WebKB(root='/tmp/Texas', name='Texas')
    alpha, beta = .1, .01
else:
    dataset = WebKB(root='/tmp/Wisconsin', name='Wisconsin')
    alpha, beta = .1, .01



data = dataset[0].to(device)
print(dataset)

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
        if isinstance(in_channels, int):
            self.lin_src = Linear(in_channels, heads * out_channels,
                                  bias=False, weight_initializer='glorot')
            self.lin_dst = self.lin_src
        else:
            self.lin_src = Linear(in_channels[0], heads * out_channels, False,
                                  weight_initializer='glorot')
            self.lin_dst = Linear(in_channels[1], heads * out_channels, False,
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


    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, idx,
                edge_attr: OptTensor = None, size: Size = None,
                return_attention_weights=None):

        H, C = self.heads, self.out_channels

        # We first transform the input node features. If a tuple is passed, we
        # transform source and target node features via separate weights:
        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = x_dst = self.lin_src(x).view(-1, H, C)
        else:  # Tuple of source and target node features:
            x_src, x_dst = x
            assert x_src.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = self.lin_src(x_src).view(-1, H, C)
            if x_dst is not None:
                x_dst = self.lin_dst(x_dst).view(-1, H, C)

        x = (x_src, x_dst)

        #######################################################
        # Negative constant (-1) is applied for flipped space #
        #######################################################
        if idx == 1:
            alpha_src = (x_src * -self.att_src).sum(dim=-1)
            alpha_dst = None if x_dst is None else (x_dst * -self.att_dst).sum(-1)
        else:
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
        out = self.propagate(edge_index, x=x, alpha=alpha, edge_attr=edge_attr, size=size)

        alpha = self._alpha
        assert alpha is not None
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias
        
        return_attention_weights = None
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
        # GAT conv
        self.conv1 = GATConv(dataset.num_features + 1, 8, heads=8, dropout=0.6, bias=False)
        self.conv2 = GATConv(64, dataset.num_classes, heads=1, concat=False, dropout=0.6)
        # Flip discriminator
        self.k_value = 0
        # Flip point
        self.t1 = torch.cat((torch.empty(dataset.num_node_features).fill_(.5).to(device), torch.zeros(1).to(device)))
        # conv1.lin_dst.weight : (64, num_feat + 1), conv1.att_dst : (1, 8, 8)

    def forward(self, x, edge_index, idx):
        # Zero padding in original space
        src = model.state_dict()['conv1.lin_src.weight'].T.detach().clone()
        dst = model.state_dict()['conv1.lin_dst.weight'].T.detach().clone()
        src[data.num_features] = torch.zeros(64).to(device)
        dst[data.num_features] = torch.zeros(64).to(device)
        
        if idx == 0 and self.k_value == 1:
            model.state_dict()['conv1.lin_src.weight'].data.copy_(src.T)
            model.state_dict()['conv1.lin_dst.weight'].data.copy_(dst.T)
            model.state_dict()['conv1.att_src'].data.copy_(-model.state_dict()['conv1.att_src'].detach().clone())
            model.state_dict()['conv1.att_dst'].data.copy_(-model.state_dict()['conv1.att_dst'].detach().clone())
            self.k_value = 0
        
        # Adjust plane for flipped space
        elif idx == 1 and self.k_value == 0:
            src_refine = torch.matmul(self.t1, src)
            dst_refine = torch.matmul(self.t1, dst)
            src[data.num_features] = - 2 * src_refine
            dst[data.num_features] = - 2 * dst_refine
            
            model.state_dict()['conv1.lin_src.weight'].data.copy_(src.T)
            model.state_dict()['conv1.lin_dst.weight'].data.copy_(dst.T)
            model.state_dict()['conv1.att_src'].data.copy_(-model.state_dict()['conv1.att_src'].detach().clone())
            model.state_dict()['conv1.att_dst'].data.copy_(-model.state_dict()['conv1.att_dst'].detach().clone())
            self.k_value = 1
        
        # If idx is 1 (flipped space) 
        # We multiply negative constant to attention vector 
        # Please refer to line 342
        x = F.relu(self.conv1(x, edge_index, 0))
        x = F.dropout(x, p=0.7, training=self.training)
        
        # Then flip hidden features
        # We don't need to adjust attention vector for second convolution
        if idx == 0:
            x = self.conv2(x, edge_index, 0)
        else:
            x = self.conv2(-x, edge_index, 0)            
        
        return F.log_softmax(x, dim=1)
        

# This part is the same as Flip_GCN
model = Net().to(device)

optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
zero, one = torch.zeros(len(data.x), 1).to(device), torch.ones(len(data.x), 1).to(device)
without_bias, with_bias = torch.cat((data.x, zero), 1), torch.cat((1 - data.x, one), 1)

model.train()

idx, early_stop = 0, 0
valid, best_valid, best_value, best_epoch = 0, 0, 0, 0

for epoch in tqdm(range(epoch)):
    optim.zero_grad()
    early_stop += 1

    model.train()
    
    if idx == 0:
        out = model(without_bias, data.edge_index, 0)
    else:
        out = model(with_bias, data.edge_index, 1)
    # Loss function
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    # Adjust gradients
    if idx == 0:
        model.conv1.lin_src.weight.grad = model.conv1.lin_src.weight.grad.clone() * alpha
        model.conv1.lin_dst.weight.grad = model.conv1.lin_dst.weight.grad.clone() * alpha
    else:
        model.conv1.lin_src.weight.grad = model.conv1.lin_src.weight.grad.clone() * beta
        model.conv1.lin_dst.weight.grad = model.conv1.lin_dst.weight.grad.clone() * beta
    optim.step()
        
    with torch.no_grad():
        model.eval()
        
        if idx == 0:
            _, pred = model(without_bias, data.edge_index, 0).max(dim=1)
        else:
            _, pred = model(with_bias, data.edge_index, 1).max(dim=1)
            
        # Validation accuracy
        correct = float(pred[data.val_mask].eq(data.y[data.val_mask]).sum().item())
        val_acc = correct / data.val_mask.sum().item()
        
        # Measure test acc of best validation acc
        if val_acc > best_valid:
            best_valid = val_acc
            correct = float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
            acc = correct / data.test_mask.sum().item()
            print('Accuracy: {:4f}'.format(acc))
            early_stop = 0
    
    if multi_learning:
        idx = 1 - idx
    if early_stop > 10000:
        break
