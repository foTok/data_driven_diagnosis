'''
This file trys to employ Long Short-term Memory network to diagnose BPSK communication system.

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class lstm_diagnoser2(nn.Module):
    """
    The basic diagnoser constructed by LSTM
    """
    def __init__(self, hidden_size, fc_numbers, num_layers=4, input_size=(5, 128)):
        '''
        Args:
            hidden_size: an int, the hidden state number
            fc_numbers: an int list or tuple whose length = 2.
                Store the output numbers of the two full connected numbers.
                The output number of the last FC layer should be (fault_number + 1)
            num_layers: an int, the number of LSTM layers
            input_size: an int list or tuple with length = 2.
                The first number is the number of input features.
                The second number is the number of time steps.
        ''' 
        super(lstm_diagnoser2, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.lstm = nn.LSTM(input_size[0], hidden_size, num_layers)

        self.fc_sequence = nn.Sequential(
                            nn.Linear(hidden_size, fc_numbers[0]),
                            nn.ReLU(),
                            nn.BatchNorm1d(fc_numbers[0]),
                            nn.Linear(fc_numbers[0], fc_numbers[1]),
                            nn.Softmax(1),
                        )


    def forward(self, x):
        # x: batch × channel × time => time × batch × channel
        # [0, 1, 2] => [2, 0, 1]
        batch = x.size()[0]
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

    def features(self, x):
        # x: batch × channel × time => time × batch × channel
        # [0, 1, 2] => [2, 0, 1]
        batch = x.size()[0]
        h0 = torch.zeros(self.num_layers, batch, self.hidden_size)
        c0 = torch.zeros(self.num_layers, batch, self.hidden_size)
        x = x.permute([2, 0, 1])
        x, (_, _) = self.lstm(x, (h0, c0))
        # x: time × batch × hidden_size => batch × hidden_size × time
        # [0, 1, 2] => [1, 2, 0]
        x = x.permute([1, 2, 0])
        # x.size() = batch × hidden_size
        x = x[:,:,-1]
        return x

    def predict(self, x):
        x = self.fc_sequence(x)
        return x
