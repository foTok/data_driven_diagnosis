'''
This file trys to employ Long Short-term Memory network to diagnose.

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class bpsk_student_lstm_diagnoser(nn.Module):
    """
    The basic diagnoser constructed by LSTM
    """
    def __init__(self, num_layers=2, length=128):
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
        super(bpsk_student_lstm_diagnoser, self).__init__()
        self.num_layers = num_layers
        self.lstm = nn.LSTM(4, 2, num_layers)

        self.fc_sequence = nn.Sequential( # TODO
                        )
        self.soft_max = nn.Softmax(1)

    def forward(self, x):
        # x: batch × channel × time => time × batch × channel
        # [0, 1, 2] => [2, 0, 1]
        batch = x.size()[0]
        h0 = torch.zeros(self.num_layers, batch, 2)
        c0 = torch.zeros(self.num_layers, batch, 2)
        x = x.permute([2, 0, 1])
        x, (_, _) = self.lstm(x, (h0, c0))
        # x: time × batch × hidden_size => batch × hidden_size × time
        # [0, 1, 2] => [1, 2, 0]
        x = x.permute([1, 2, 0])
        # x.size() = batch × hidden_size
        state = x[:,:,-1]
        # TODO
        logit = self.fc_sequence(state)
        p = self.soft_max(logit)
        return p, logit, state
