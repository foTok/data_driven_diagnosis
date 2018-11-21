'''
This file implement the basic GNN proposed in
    [1] F. Scarselli, M. Gori, A. C. Tsoi, M. Hagenbuchner, and G. Monfardini, \
    “The graph neural network model,” IEEE Trans. Neural Networks, vol. 20, no. 1, pp. 61–80, 2009.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class gnn_rnn_diagnoser(nn.Module):
    """
    The basic diagnoser constructed by GNN
    """
    def __init__(self, graph, hidden_size, fc_numbers, node_num=5):
        '''
        graph: graph structure over variables, 5×5. graph[i, j]==1 mean an edge from i to j
        hidden_size: hidden_size for the whole system
        fc_numbers: 1×3, the first two are the fc numbers for the transform layers
                         the last one is the fc number for the output layer
        '''
        super(gnn_rnn_diagnoser, self).__init__()
        self.graph = graph
        self.fml_state_num = hidden_size//node_num
        self.fc_numbers = fc_numbers
        self.node_num = node_num
        self.f_input_sizes = []
        self.f = []
        for _ in range(node_num):
            self.f_input_sizes.append((self.fml_state_num + 1)*torch.sum(graph[:, 0]))
        # The RNN layers
        for i in range(node_num):
            f = nn.Sequential(
                nn.Linear(self.f_input_sizes[i], fc_numbers[0]),
                nn.ReLU(),
                nn.Linear(fc_numbers[0], fc_numbers[1]),
                nn.ReLU(),
                nn.Linear(fc_numbers[1], self.fml_state_num),
                nn.ReLU(),
            )
            self.f.append(f)
        # Output layers
        self.output = nn.Sequential(
            nn.Linear((self.fml_state_num+1)*5, fc_numbers[-1]),
            nn.ReLU(),
            nn.BatchNorm1d(fc_numbers[-1]),
            nn.Linear(fc_numbers[-1], 7),
            nn.Softmax(1),
        )

    def forward(self, x):
        '''
        x: batch × nodes × time =>  time × batch × nodes
        state: batch × 5 × fml_state_num
        '''
        # batch × nodes × time =>  time × batch × nodes
        x = x.permute([2, 0, 1])
        batch = x.size(1)
        state = torch.zeros([batch, self.node_num, self.fml_state_num], requires_grad=True)
        for t in range(x.size(0)):
            for i in range(self.node_num):
                fml_i = np.nonzero(self.graph[:, i])[0]
                x_ti = x[t, :, fml_i]
                s_ti = state[:,fml_i,:]
                s_ti = s_ti.view(batch, -1)
                in_i = torch.cat((x_ti, s_ti), 1)
                state[:, i, :] = self.f[i](in_i)
        state = state.view(batch, -1)
        in_all = torch.cat((x[-1,:,:], state),1)
        out = self.output(in_all)
        return out
