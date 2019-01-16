import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class bpsk_student_cnn_diagnoser(nn.Module):
    """
    The basic diagnoser constructed by Classic CNN
    """
    def __init__(self, kernel_sizes, feature_maps, fc_number, length):
        '''
        Args:
            kernel_sizes: an int list or tuple whose length = 2. 
                Store the kernel sizes of the four Conv1d layers.
            feature_maps: an int list
            fc_number: an int
            input_size: an int list or tuple with length = 2.
                The first number is the number of input features.
                The second number is the number of time steps.
        ''' 
        super(bpsk_student_cnn_diagnoser, self).__init__()
        self.kernel_sizes = kernel_sizes
        self.feature_maps = feature_maps
        self.fc_number = fc_number
        self.length = length
        self.padding = [(i-1)//2 for i in kernel_sizes]
        self.pooling = 2 # Pooling size is set as 2
        delta = [(2*p-k+1) for p, k in zip(self.padding, kernel_sizes)]
        self.cnn_out_length = ((length+delta[0])//self.pooling + delta[1])//self.pooling
        self.cnn0 = nn.Sequential(
                            nn.Conv1d(1, feature_maps[0], kernel_sizes[0], padding=self.padding[0]), # length+delta[0]
                            nn.ReLU(),
                            nn.MaxPool1d(self.pooling), # (length+delta[0])//self.pooling
                            nn.Conv1d(feature_maps[0], 5, kernel_sizes[1], padding=self.padding[1]), # (length+delta[0])//self.pooling + delta[1]
                            nn.ReLU(),
                            nn.MaxPool1d(self.pooling), # ((length+delta[0])//self.pooling + delta[1])//self.pooling
                        ) # fe36 fe54 fe11 fe62 fe23

        self.cnn1 = nn.Sequential(
                            nn.Conv1d(3, feature_maps[1], kernel_sizes[0], padding=self.padding[0]),
                            nn.ReLU(),
                            nn.MaxPool1d(self.pooling),
                            nn.Conv1d(feature_maps[1], 1, kernel_sizes[1], padding=self.padding[1]),
                            nn.ReLU(),
                            nn.MaxPool1d(self.pooling), # ((length+delta[0])//self.pooling + delta[1])//self.pooling
                        ) # fe26

        self.cnn2 = nn.Sequential(
                            nn.Conv1d(2, feature_maps[2], kernel_sizes[0], padding=self.padding[0]),
                            nn.ReLU(),
                            nn.MaxPool1d(self.pooling),
                            nn.Conv1d(feature_maps[2], 3, kernel_sizes[1], padding=self.padding[1]),
                            nn.ReLU(),
                            nn.MaxPool1d(self.pooling), # ((length+delta[0])//self.pooling + delta[1])//self.pooling
                        ) # fe5

        self.cnn3 = nn.Sequential(
                            nn.Conv1d(2, feature_maps[3], kernel_sizes[0], padding=self.padding[0]),
                            nn.ReLU(),
                            nn.MaxPool1d(self.pooling),
                            nn.Conv1d(feature_maps[2], 3, kernel_sizes[1], padding=self.padding[1]),
                            nn.ReLU(),
                            nn.MaxPool1d(self.pooling), # ((length+delta[0])//self.pooling + delta[1])//self.pooling
                        ) # fe8 fe29

        self.N = nn.Sequential(
                            nn.Linear(self.cnn_out_length*4, fc_number),
                            nn.ReLU(),
                            nn.BatchNorm1d(fc_number),
                            nn.Linear(fc_number, 1),
                        )
        self.TMA = nn.Sequential(
                            nn.Linear(self.cnn_out_length*4, fc_number),
                            nn.ReLU(),
                            nn.BatchNorm1d(fc_number),
                            nn.Linear(fc_number, 1),
                        )
        self.PCR = nn.Sequential(
                            nn.Linear(self.cnn_out_length*4, fc_number),
                            nn.ReLU(),
                            nn.BatchNorm1d(fc_number),
                            nn.Linear(fc_number, 1),
                        )
        self.CAR = nn.Sequential(
                            nn.Linear(self.cnn_out_length*3, fc_number),
                            nn.ReLU(),
                            nn.BatchNorm1d(fc_number),
                            nn.Linear(fc_number, 1),
                        )
        self.MPL = nn.Sequential(
                            nn.Linear(self.cnn_out_length*2, fc_number),
                            nn.ReLU(),
                            nn.BatchNorm1d(fc_number),
                            nn.Linear(fc_number, 1),
                        )
        self.AMP = nn.Sequential(
                            nn.Linear(self.cnn_out_length*4, fc_number),
                            nn.ReLU(),
                            nn.BatchNorm1d(fc_number),
                            nn.Linear(fc_number, 1),
                        )
        self.TMB = nn.Sequential(
                            nn.Linear(self.cnn_out_length*4, fc_number),
                            nn.ReLU(),
                            nn.BatchNorm1d(fc_number),
                            nn.Linear(fc_number, 1),
                        )

        self.soft_max = nn.Softmax(1)

    def forward(self, x):
        # x: batch × variable × timestep
        # variable: m, p, c, s0, s1
        x0 = x[:, [4], :]
        x1 = x[:, [1, 3, 4], :]
        x2 = x[:, [3, 4], :]
        x3 = x[:, [1, 4], :]
        # feature group: batch × feature × len
        fg0 = self.cnn0(x0) # fe36 fe54 fe11 fe62 fe23| 0 1 2 3 4
        fg1 = self.cnn1(x1) # fe26 | 5
        fg2 = self.cnn2(x2) # fe5 | 6
        fg3 = self.cnn3(x3) # fe8 fe29 | 7 8
        # all features
        # average features
        features = torch.cat((fg0, fg1, fg2, fg3), 1)
        average_features = torch.mean(features, 2)
        # features of faults
        fe_n   = features[:, [1, 2, 3, 4], :] # fe54 fe11 fe62 fe23
        fe_tma = features[:, [1, 6, 7, 8], :]
        fe_pcr = features[:, [2, 3, 4, 6], :]
        fe_car = features[:, [3, 4, 6], :]
        fe_mpl = features[:, [0, 1],:]
        fe_amp = features[:, [0, 2, 3, 5], :]
        fe_tmb = features[:, [3, 4, 5, 7], :]
        fe_n = fe_n.view(-1, self.cnn_out_length*4)
        fe_tma = fe_tma.view(-1, self.cnn_out_length*4)
        fe_pcr = fe_pcr.view(-1, self.cnn_out_length*4)
        fe_car = fe_car.view(-1, self.cnn_out_length*3)
        fe_mpl = fe_mpl.view(-1, self.cnn_out_length*2)
        fe_amp = fe_amp.view(-1, self.cnn_out_length*4)
        fe_tmb = fe_tmb.view(-1, self.cnn_out_length*4)
        logits_n = self.N(fe_n)
        logits_tma = self.TMA(fe_tma)
        logits_pcr = self.PCR(fe_pcr)
        logits_car = self.CAR(fe_car)
        logits_mpl = self.MPL(fe_mpl)
        logits_amp = self.AMP(fe_amp)
        logits_tmb = self.TMB(fe_tmb)
        logits = torch.cat((logits_n, logits_tma, logits_pcr, logits_car, logits_mpl, logits_amp, logits_tmb), 1)
        p = self.soft_max(logits)
        return p, logits, average_features
