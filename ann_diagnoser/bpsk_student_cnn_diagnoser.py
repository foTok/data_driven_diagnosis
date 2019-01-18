import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class bpsk_student_cnn_diagnoser(nn.Module):
    """
    The basic diagnoser constructed by Classic CNN
    """
    def __init__(self, kernel_sizes, length):
        '''
        Args:
            kernel_sizes: an int list or tuple whose length = 2. 
                Store the kernel sizes of the four Conv1d layers.
            fc_number: an int
            input_size: an int list or tuple with length = 2.
                The first number is the number of input features.
                The second number is the number of time steps.
        ''' 
        super(bpsk_student_cnn_diagnoser, self).__init__()
        self.kernel_sizes = kernel_sizes
        self.length = length
        self.padding = [(i-1)//2 for i in kernel_sizes]
        self.pooling = 2 # Pooling size is set as 2
        delta = [(2*p-k+1) for p, k in zip(self.padding, kernel_sizes)]
        self.cnn_out_length = ((length+delta[0])//self.pooling + delta[1])//self.pooling
        self.cnn = nn.Sequential(
                            nn.Conv1d(4, 8, kernel_sizes[0], padding=self.padding[0]), # length+delta[0]
                            nn.ReLU(),
                            nn.MaxPool1d(self.pooling), # (length+delta[0])//self.pooling
                            nn.Conv1d(8, 10, kernel_sizes[1], padding=self.padding[1]), # (length+delta[0])//self.pooling + delta[1]
                            nn.ReLU(),
                            nn.MaxPool1d(self.pooling), # ((length+delta[0])//self.pooling + delta[1])//self.pooling
                        ) # fe9 fe16 fe4 fe58 fe18 fe52 fe27 fe46 fe31 fe32


        self.N   = nn.Linear(self.cnn_out_length*6, 1)
        self.TMA = nn.Linear(self.cnn_out_length*2, 1)
        self.PCR = nn.Linear(self.cnn_out_length*2, 1)
        self.CAR = nn.Linear(self.cnn_out_length*3, 1)
        self.MPL = nn.Linear(self.cnn_out_length*2, 1)
        self.AMP = nn.Linear(self.cnn_out_length*4, 1)
        self.TMB = nn.Linear(self.cnn_out_length*6, 1)

        self.soft_max = nn.Softmax(1)

    def forward(self, x):
        # x: batch × variable × timestep
        # variable: m, p, c, s0, s1
        features = self.cnn(x[:,[1,2,3,4],:])
        # features of faults
        fe_n   = features[:, [0, 1, 2, 3, 4, 5], :]
        fe_tma = features[:, [6, 9], :]
        fe_pcr = features[:, [6, 9], :]
        fe_car = features[:, [3, 5, 6], :]
        fe_mpl = features[:, [0, 1],:]
        fe_amp = features[:, [2, 3, 4, 5], :]
        fe_tmb = features[:, [4, 5, 6 ,7 ,8 ,9], :]
        fe_n   = fe_n.view(-1, self.cnn_out_length*6)
        fe_tma = fe_tma.view(-1, self.cnn_out_length*2)
        fe_pcr = fe_pcr.view(-1, self.cnn_out_length*2)
        fe_car = fe_car.view(-1, self.cnn_out_length*3)
        fe_mpl = fe_mpl.view(-1, self.cnn_out_length*2)
        fe_amp = fe_amp.view(-1, self.cnn_out_length*4)
        fe_tmb = fe_tmb.view(-1, self.cnn_out_length*6)
        logits_n = self.N(fe_n)
        logits_tma = self.TMA(fe_tma)
        logits_pcr = self.PCR(fe_pcr)
        logits_car = self.CAR(fe_car)
        logits_mpl = self.MPL(fe_mpl)
        logits_amp = self.AMP(fe_amp)
        logits_tmb = self.TMB(fe_tmb)
        logits = torch.cat((logits_n, logits_tma, logits_pcr, logits_car, logits_mpl, logits_amp, logits_tmb), 1)
        p = self.soft_max(logits)
        return p, logits, features
