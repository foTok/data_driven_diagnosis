'''
This file implement the basic GNN proposed in
    [1] F. Scarselli, M. Gori, A. C. Tsoi, M. Hagenbuchner, and G. Monfardini, \
    “The graph neural network model,” IEEE Trans. Neural Networks, vol. 20, no. 1, pp. 61–80, 2009.
    Modified in how to handle time sequence.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class gnn_cnn_diagnoser(nn.Module):
    """
    The basic diagnoser constructed by GNN
    """
    def __init__(self, graph, feature_maps, fc_numbers, kernel_size=3, data_size=100):
        '''
        graph: graph structure over variables, 5×5. graph[i, j]==1 mean an edge from i to j
        '''
        super(gnn_cnn_diagnoser, self).__init__()
        self.graph = graph
        self.feature_maps = feature_maps
        self.fc_numbers = fc_numbers
        self.data_size = data_size
        self.node_num = len(graph)
        self.kernel_size = kernel_size
        self.padding = kernel_size//2
        self.delta = 2*self.padding-kernel_size+1
        self.conv_features = feature_maps[1] * ((data_size+2*self.delta)//kernel_size) * self.node_num
        self.f_input_sizes = []
        self.f = []
        for i in range(self.node_num):
            self.f_input_sizes.append(int(torch.sum(graph[:, i])))
        # The RNN layers
        for i in range(self.node_num):
            f = nn.Sequential(
                # 1 × f_input_size[i] × data_size => feature_maps[0] × 1 × (data_size+delta)
                nn.Conv2d(1, feature_maps[0], kernel_size=(self.f_input_sizes[i], kernel_size), padding=(0, self.padding)),
                nn.ReLU(),
                # feature_maps[0] × 1 × (data_size+delta) => feature_maps[1] × 1 × (data_size+2delta)
                nn.Conv2d(feature_maps[0], feature_maps[1], kernel_size=(1, kernel_size), padding=(0, self.padding)),
                nn.ReLU(),
                # feature_maps[1] × 1 × ((data_size+2delta)//kernel_size)
                nn.MaxPool2d(kernel_size=(1, kernel_size))
            )
            self.f.append(f)
        # Output layers
        self.output = nn.Sequential(
            # feature_maps[1] × ((data_size+2delta)//kernel_size) * self.node_num
            nn.Linear(self.conv_features, fc_numbers),
            nn.ReLU(),
            nn.BatchNorm1d(fc_numbers),
            nn.Linear(fc_numbers, 7),
            nn.Softmax(1),
        )

    def forward(self, x):
        # x: batch × 1 × node_num × data_size
        features = []
        for i in range(self.node_num):
            fml_i = torch.nonzero(self.graph[:, i]).view(1,-1)[0]
            # x_i: feature_maps[1] × 1 × ((data_size+2delta)//kernel_size)
            x_i = x[:, :, fml_i, :]
            x_i = self.f[i](x_i)
            features.append(x_i)
        x = torch.cat(features, 1)
        x = x.view(-1, self.conv_features)
        out = self.output(x)
        return out
