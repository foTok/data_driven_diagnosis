'''
This network is based on the third (start from 1) learned LSTM model.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class mt_student_lstm_diagnoser(nn.Module):
    """
    The basic diagnoser constructed by LSTM
    """
    def __init__(self, num_layers=2, length=64):
        super(mt_student_lstm_diagnoser, self).__init__()
        self.num_layers = num_layers
        self.lstm = nn.LSTM(11, 6, num_layers)
        self.fc = nn.Linear(6, 21)
        self.soft_max = nn.Softmax(1)

    def forward(self, x):
        batch = x.size()[0]
        h0 = torch.zeros(self.num_layers, batch, 6)
        c0 = torch.zeros(self.num_layers, batch, 6)
        # x: batch × channel × time => time × batch × channel
        # [0, 1, 2] => [2, 0, 1]
        x = x.permute([2, 0, 1])
        x, (_, _) = self.lstm(x, (h0, c0))
        # x: time × batch × hidden_size => batch × hidden_size × time
        # [0, 1, 2] => [1, 2, 0]
        x = x.permute([1, 2, 0])
        # x.size() = batch × hidden_size
        state = x[:,:,-1]
        logits = self.fc(state)
        p = self.soft_max(logits)
        return p, logits, state
