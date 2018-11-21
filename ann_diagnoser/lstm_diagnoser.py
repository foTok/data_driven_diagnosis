'''
This file trys to employ Long Short-term Memory network to diagnose BPSK communication system.

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class lstm_diagnoser(nn.Module):
    """
    The basic diagnoser constructed by Classic CNN
    """
    def __init__(self, hidden_size, fc_number, num_layers=2, w=100):
        '''
        kernel_size: an int or a tuple.
        feature_maps: a tuple, the feature numbers of the fisrt and second convolutional layers.
        fc_numbers: an int, the hidden number of full connection layers.
        w: an int, the length of data
        pool: a bool, if conduct pooling
        ''' 
        super(lstm_diagnoser, self).__init__()
        self.w = w
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(5, hidden_size, num_layers)

        self.fc_sequence = nn.Sequential(
                            # feature_maps[2]×size_after_pooling[0]×size_after_pooling[1]=>fc_number
                            nn.Linear(hidden_size, fc_number),
                            nn.ReLU(),
                            nn.BatchNorm1d(fc_number),
                            nn.Linear(fc_number, 7),
                            nn.Softmax(1),
                        )


    def forward(self, x):
        # x: batch × channel × time => time × batch × channel
        # [0, 1, 2] => [2, 0, 1]
        batch = x.size()[0]
        x = x.view(-1, 5, self.w)
        h0 = torch.zeros(self.num_layers, batch, self.hidden_size)
        c0 = torch.zeros(self.num_layers, batch, self.hidden_size)
        x = x.permute([2, 0, 1])
        x, (_, _) = self.lstm(x, (h0, c0))
        # x: time × batch × hidden_size => batch × hidden_size × time
        # [0, 1, 2] => [1, 2, 0]
        x = x.permute([1, 2, 0])
        # x.size() = batch × hidden_size
        x = x[:,:,-1]
        x = self.fc_sequence(x)
        return x
