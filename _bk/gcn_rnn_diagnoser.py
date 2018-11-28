'''
This file implement the basic GNN proposed in
    [1] F. Scarselli, M. Gori, A. C. Tsoi, M. Hagenbuchner, and G. Monfardini, \
    “The graph neural network model,” IEEE Trans. Neural Networks, vol. 20, no. 1, pp. 61–80, 2009.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.parameter import Parameter


class GCN(nn.Module):
    '''
    A GCN module
    '''
    def __init__(self, in_features, out_features, adj, bias=True):
        '''
        As the parameter names
        '''
        super(GCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.adj = adj
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        batch = input.size(0)
        node = input.size(1)
        weight = self.weight.expand(batch, -1, -1)
        adj = self.adj.expand(batch, -1, -1)
        support = torch.bmm(input, weight)
        output = torch.bmm(adj, support)
        if self.bias is not None:
            bias = self.bias.expand(batch, node, self.out_features)
            return output + bias
        else:
            return output
        
    def reset_parameters(self):
        '''
        reset parameters by xavier_normal_
        '''
        nn.init.xavier_normal_(self.weight.data, gain=0.02) # Implement Xavier Uniform
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0.0)

    def __repr__(self):
            return self.__class__.__name__ + ' (' \
                + str(self.in_features) + ' -> ' \
                + str(self.out_features) + ')'



class gcn_rnn_diagnoser(nn.Module):
    """
    The basic diagnoser constructed by GNN
    """
    def __init__(self, adj, feature_maps, hidden_size, fc_number, node_num=5):
        '''
        adj: graph structure over variables, 5×5. graph[i, j]==1 mean an edge from i to j
             graph should be a pytorch Tensor
        feature_maps: the features in each layer
        hidden_size: the size of the hidden states
        fc_number: the fc number for the output module
        '''
        super(gcn_rnn_diagnoser, self).__init__()
        self.adj = adj.t()
        self.node_num = len(adj)
        self.last_features = hidden_size // self.node_num
        feature_maps = [self.last_features + 1] + feature_maps
        self.feature_maps = feature_maps
        self.hidden_size = self.last_features * self.node_num
        self.fc_number = fc_number
        self.state = None
        self.gcn = []
        for i in range(1, len(feature_maps)):
            gcn = nn.Sequential(
                GCN(feature_maps[i-1], feature_maps[i], self.adj),
                nn.ReLU(),
            )
            self.gcn.append(gcn)
        gcn = nn.Sequential(
                GCN(feature_maps[-1], self.last_features, self.adj),
                nn.Tanh(),
            )
        self.gcn.append(gcn)
        self.output = nn.Sequential(
            nn.Linear(self.node_num + self.hidden_size, fc_number),
            nn.ReLU(),
            nn.BatchNorm1d(fc_number),
            nn.Linear(fc_number, 7),
            nn.Softmax(1),
        )

    def gcn_forward(self, x, state):
        '''
        forward the gcn module
        '''
        # x: batch × nodes
        x = x.view(x.size(0),x.size(1), 1)
        x = torch.cat((x, state), 2)
        x = self.gcn[0](x)
        for i in range(1, len(self.gcn)):
            x = self.gcn[i](x)
        return x    # x is the state now


    def forward(self, x):
        '''
        x: batch × nodes × time =>  time × batch × nodes
        state: batch × hidden_size
        '''
        batch = x.size(0)
        time = x.size(3)
        # batch × nodes × time =>  time × batch × nodes
        x = x.view(-1, self.node_num, time)
        x = x.permute([2, 0, 1])
        state = torch.zeros([batch, self.node_num, self.last_features], requires_grad=True)
        for t in range(time):
            x_t = x[t,:,:]
            state = self.gcn_forward(x_t, state)
        x_end = x[-1,:,:]
        state = state.view(batch, -1)
        x_s = torch.cat((x_end, state), 1)
        out = self.output(x_s)
        return out

    def __repr__(self):
        msg = '('
        for features in self.feature_maps:
            msg = msg + str(features) + ' ->'
        msg = msg + str(self.last_features) + ') ->'
        msg = msg + str(self.fc_number) + ' -> 7'
        return self.__class__.__name__ + msg
