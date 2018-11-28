'''
Group variables based on BN
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import log2

def descendants(adjfv, adj):
    '''
    Find all descendants for all fauls
    Args:
        adjfv: a |F|×|V| 2D numpy matirx,  the adjacency matrix for fault variables to monitored variables. 
            adjfv[i, j]=1 means there exists edge f_i->v_j.
        adj: a |V|×|V| 2D numpy matirx,  the adjacency matrix for monitored variables to monitored variables. 
            adj[i, j]=1 means there exists edge v_i->v_j. adj[i, j] must be 1 for i==j.
    Returns:
        a matrix has the same shape with adjfv
    '''
    _adjfv = np.array(adjfv)
    fn, nn = _adjfv.shape # fault number, node number
    for i in range(fn):
        open_nodes = set()
        closed_nodes = set()
        for j in range(nn): # init based on adjfv
            if adjfv[i, j] == 1:
                open_nodes.add(j)
        while open_nodes:
            p = open_nodes.pop()
            closed_nodes.add(p)
            for j in range(nn):
                if (adj[p, j]==1) and (j not in closed_nodes):
                    open_nodes.add(j)
        for j in closed_nodes:
            _adjfv[i, j] = 1
    return _adjfv

class cnn_modules(nn.Module):
    '''
    The cnn modules used in the bnn_cnn_diagnoser.
    '''
    def __init__(self, kernel_sizes, feature_maps, input_size):
        '''
        Args:
            kernel_sizes: an int list or tuple whose length = 4. 
                Store the kernel sizes of the four Conv1d layers.
            feature_maps: an int list or tuple whose length = 4.
                Store the output feature numbers of the four Conv1d layers.
            input_size: an int list or tuple with length = 2.
                The first number is the number of input features.
                The second number is the number of time steps.
        ''' 
        assert len(kernel_sizes)==4 and \
               len(feature_maps)==4 and \
               len(input_size)==2
        super(cnn_modules, self).__init__()
        self.kernel_sizes = kernel_sizes
        self.feature_maps = feature_maps
        self.input_size = input_size
        self.padding = [(i-1)//2 for i in kernel_sizes]
        self.pooling = 2 # Pooling size is set as 2
        delta = [(2*p-k+1) for p, k in zip(self.padding, kernel_sizes)]
        self.cnn_out_length = input_size[1]
        for i in delta:
            self.cnn_out_length = (self.cnn_out_length + i)//self.pooling

        self.cnn_sequence = nn.Sequential(                                                                              #           Length (pooling=2)
                            nn.Conv1d(input_size[0], feature_maps[0], kernel_sizes[0], padding=self.padding[0]),        # L1        input_size[1] + delta[0]
                            nn.ReLU(),
                            nn.MaxPool1d(self.pooling),                                                                 # L2        (input_size[1] + delta[0])//2
                            nn.Conv1d(feature_maps[0], feature_maps[1], kernel_sizes[1], padding=self.padding[1]),      # L3        (input_size[1] + delta[0])//2 + delta[1]
                            nn.ReLU(),
                            nn.MaxPool1d(self.pooling),                                                                 # L4        ((input_size[1] + delta[0])//2 + delta[1])//2
                            nn.Conv1d(feature_maps[1], feature_maps[2], kernel_sizes[2], padding=self.padding[2]),      # L5        ((input_size[1] + delta[0])//2 + delta[1])//2 + delata[2]
                            nn.ReLU(),
                            nn.MaxPool1d(self.pooling),                                                                 # L6        (((input_size[1] + delta[0])//2 + delta[1])//2 + delata[2])//2
                            nn.Conv1d(feature_maps[2], feature_maps[3], kernel_sizes[3], padding=self.padding[3]),      # L7        (((input_size[1] + delta[0])//2 + delta[1])//2 + delata[2])//2 + delta[3]
                            nn.ReLU(),
                            nn.MaxPool1d(self.pooling),                                                                 # L8        ((((input_size[1] + delta[0])//2 + delta[1])//2 + delta[2])//2 + delta[3])//2
                        )

    def forward(self, x):
        x = self.cnn_sequence(x)
        return x

    def __repr__(self):
        msg = ''
        for features in self.feature_maps:
            msg = msg + str(features) + ' ->'
        return self.__class__.__name__ + msg

class bnn_cnn_diagnoser(nn.Module):
    '''
    The basic diagnoser constructed by GNN
    '''
    def __init__(self, adjfv, adj, kernel_sizes, feature_maps, fc_numbers, input_size=(5, 128)):
        '''
        Args:
            adjfv: a |F|×|V| 2D numpy matirx,  the adjacency matrix for fault variables to monitored variables. 
                adjfv[i, j]=1 means there exists edge f_i->v_j.
            adj: a |V|×|V| 2D numpy matirx,  the adjacency matrix for monitored variables to monitored variables. 
                adj[i, j]=1 means there exists edge v_i->v_j. adj[i, j] must be 1 for i==j.
            kernel_sizes: an int list or tuple whose length = 4. 
                Store the kernel sizes of the four Conv1d layers.
            feature_maps: an int list or tuple whose length = 4.
                Store the output feature numbers of the four Conv1d layers.
            fc_numbers: an int list or tuple whose length = 2.
                Store the output numbers of the two full connected numbers.
                The output number of the last FC layer should be (fault_number + 1)
            input_size: an int list or tuple with length = 2.
                The first number is the number of input features.
                The second number is the number of time steps.
        '''
        assert len(kernel_sizes)==4 and \
               len(feature_maps)==4 and \
               len(fc_numbers)==2 and \
               len(input_size)==2 and \
               len(adj) == input_size[0]
        super(bnn_cnn_diagnoser, self).__init__()
        self.adj = adj
        self.kernel_sizes = kernel_sizes
        self.feature_maps = feature_maps
        self.fc_numbers = fc_numbers
        self.input_size = input_size

        self.fault_number = len(adjfv)
        self.node_num = len(adj)
        self.module_feature_maps = [2**(int(log2(fm))+1) for fm in feature_maps]
        self.descendants = descendants(adjfv, adj)
        pooling = 2 # Pooling size is set as 2
        delta = [(2*p-k+1) for p, k in zip(self.padding, kernel_sizes)]
        self.cnn_out_length = input_size[1]
        for i in delta:
            self.cnn_out_length = (self.cnn_out_length + i)//pooling
        self.cnn = []
        self.fc = []
        # The CNN layers
        for i in range(self.node_num):
            _cnn = cnn_modules(kernel_sizes, self.module_feature_maps, input_size=(sum(adj[:, i]), input_size[1]))
            self.cnn.append(_cnn)
        # fc layers
        for i in range(self.fault_number+1): # The first one is for normal
            _desc = self.node_num if i==0 else sum(self.descendants[i-1,:])
            _fc = nn.Sequential(
                nn.Linear(self.cnn_out_length*self.module_feature_maps[-1]*_desc, fc_numbers[0]),
                nn.ReLU(),
                nn.BatchNorm1d(fc_numbers[0]),
                nn.Linear(fc_numbers[0], 1),
                )
            self.fc.append(_fc)

    def forward(self, x):
        # x: batch × node_num × time_step
        features = []
        output = []
        for i in range(self.node_num):
            fml_i = torch.nonzero(self.graph[:, i]).view(1,-1)[0]
            x_i = x[:, fml_i, :]
            x_i = self.f[i](x_i)    # x_i: batch × channel × cnn_out
            features.append(x_i)
        for i in range(self.fault_number+1):
            _related = torch.ones(self.node_num) if i==0 else self.descendants[i-1, :]
            _related = torch.nonzero(_related).view(1,-1)[0]
            x = [features[j] for j in _related]
            x = torch.cat(x, 1)
            x = x.view(-1, self.cnn_out_length*self.module_feature_maps[-1]*self.node_num)
            x = self.fc[i](x)
            output.append(x)
        out = torch.cat(output, 1)
        out = nn.functional.softmax(out)
        return out
