'''
This file implement the basic GNN proposed in
    [1] F. Scarselli, M. Gori, A. C. Tsoi, M. Hagenbuchner, and G. Monfardini, \
    “The graph neural network model,” IEEE Trans. Neural Networks, vol. 20, no. 1, pp. 61–80, 2009.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class gnn_lstm_diagnoser(nn.Module):
    """
    The basic diagnoser constructed by GNN
    """
    def __init__(self, graph, hidden_size, fc_numbers, num_layers=2):
        '''
        graph: graph structure over variables, 5×5. graph[i, j]==1 mean an edge from i to j
        hidden_size: hidden_size for the whole system
        fc_numbers: an int
        '''
        super(gnn_lstm_diagnoser, self).__init__()
        self.graph = graph
        self.node_num = len(graph)
        self.fml_state_num = hidden_size//self.node_num
        self.fc_numbers = fc_numbers
        self.num_layers = num_layers
        self.f_input_sizes = []
        self.f = []
        for i in range(self.node_num):
            self.f_input_sizes.append(int(torch.sum(graph[:, i])))
        # The LSTM layers
        for i in range(self.node_num):
            f = nn.LSTM(self.f_input_sizes[i])
            self.f.append(f)
        # Output layers
        self.output = nn.Sequential(
            nn.Linear(self.fml_state_num*self.node_num, fc_numbers),
            nn.ReLU(),
            nn.BatchNorm1d(fc_numbers),
            nn.Linear(fc_numbers, 7),
            nn.Softmax(1),
        )

    def forward(self, x):
        '''
        x: batch × 1 × nodes × time =>  time × batch × nodes
        state: batch × 5 × fml_state_num
        '''
        # batch × 1 × nodes × time =>  time × batch × nodes
        batch = x.size(0)
        time = x.size(3)
        x = x.view(-1, self.node_num, time)
        # time × batch × node
        x = x.permute([2, 0, 1])
        h0 = torch.zeros(self.num_layers, batch, self.fml_state_num)
        c0 = torch.zeros(self.num_layers, batch, self.fml_state_num)
        x_ = []
        for i in range(self.node_num):
            fml_i = torch.nonzero(self.graph[:, i]).view(1,-1)[0]
            x_i = x[:, :, fml_i]
            x_i, (_, _) = self.f[i](x_i, (h0, c0))
            x_.append(x_i[-1, :, :])
        x = torch.cat(x_, 2)
        out = self.output(x)
        return out
