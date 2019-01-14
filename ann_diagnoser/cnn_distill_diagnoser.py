"""
This file defines the basic CNN diagnoser for BPSK communication system
The ANN is composed of a CNN module and a FC(Full Connection) module.

The data organization method is from:
    R. Zhao, R. Yan, Z. Chen, K. Mao, P. Wang, and R. X. Gao, 
    “Deep learning and its applications to machine health monitoring,” 
    Mech. Syst. Signal Process., vol. 115, pp. 213–237, 2019.

The CNN structure is from:
    L. Wen, X. Li, L. Gao, and Y. Zhang, 
    “A New Convolutional Neural Network Based Data-Driven Fault Diagnosis Method,” 
    IEEE Trans. Ind. Electron., vol. 65, no. 7, pp. 1–1, 2017.

Data organization:
    input: batch × feature × length
    output: batch × (fault_number+1)

There are 10 layers in the network.
    01. Conv1d
    02. Maxpool
    03. Conv1d
    04. Maxpool
    05. Conv1d
    06. Maxpool
    07. Conv1d
    08. Maxpool
    09. FC
    10. FC
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class cnn_distill_diagnoser(nn.Module):
    """
    The basic diagnoser constructed by Classic CNN
    """
    def __init__(self, kernel_sizes, feature_maps, fc_numbers, input_size=(5, 128), T=1):
        '''
        Args:
            kernel_sizes: an int list or tuple whose length = 4. 
                Store the kernel sizes of the four Conv1d layers.
            feature_maps: an int list or tuple whose length = 4.
                Store the output feature numbers of the four Conv1d layers.
            fc_numbers: an int list or tuple whose length = 2.
                Store the output numbers of the two full connected numbers.
                The output number of the last FC layer should be (fault_number + 1)
            input_size: an int list or tuple with length = 2.
                The first number is the number of input features.
                The second number is the number of time steps.
        ''' 
        assert len(kernel_sizes)==4 and \
               len(feature_maps)==4 and \
               len(fc_numbers)==2 and \
               len(input_size)==2
        super(cnn_distill_diagnoser, self).__init__()
        self.kernel_sizes = kernel_sizes
        self.feature_maps = feature_maps
        self.fc_number = fc_numbers
        self.input_size = input_size
        self.padding = [(i-1)//2 for i in kernel_sizes]
        self.pooling = 2 # Pooling size is set as 2
        self.T = T
        delta = [(2*p-k+1) for p, k in zip(self.padding, kernel_sizes)]
        self.cnn_out_length = input_size[1]
        for i in delta:
            self.cnn_out_length = (self.cnn_out_length + i)//self.pooling
        
        self.cnn_sequence = nn.Sequential(                                                                              #           Length (pooling=2)
                            nn.Conv1d(input_size[0], feature_maps[0], kernel_sizes[0], padding=self.padding[0]),        # L1        input_size[1] + delta[0]
                            nn.ReLU(),
                            nn.MaxPool1d(self.pooling),                                                                 # L2        (input_size[1] + delta[0])//2
                            nn.Conv1d(feature_maps[0], feature_maps[1], kernel_sizes[1], padding=self.padding[1]),      # L3        (input_size[1] + delta[0])//2 + delta[1]
                            nn.ReLU(),
                            nn.MaxPool1d(self.pooling),                                                                 # L4        ((input_size[1] + delta[0])//2 + delta[1])//2
                            nn.Conv1d(feature_maps[1], feature_maps[2], kernel_sizes[2], padding=self.padding[2]),      # L5        ((input_size[1] + delta[0])//2 + delta[1])//2 + delata[2]
                            nn.ReLU(),
                            nn.MaxPool1d(self.pooling),                                                                 # L6        (((input_size[1] + delta[0])//2 + delta[1])//2 + delata[2])//2
                            nn.Conv1d(feature_maps[2], feature_maps[3], kernel_sizes[3], padding=self.padding[3]),      # L7        (((input_size[1] + delta[0])//2 + delta[1])//2 + delata[2])//2 + delta[3]
                            nn.ReLU(),
                            nn.MaxPool1d(self.pooling),                                                                 # L8        ((((input_size[1] + delta[0])//2 + delta[1])//2 + delta[2])//2 + delta[3])//2
                        )

        self.fc_sequence = nn.Sequential(
                            nn.Linear(self.cnn_out_length*feature_maps[-1], fc_numbers[0]),
                            nn.ReLU(),
                            nn.BatchNorm1d(fc_numbers[0]),
                            nn.Linear(fc_numbers[0], fc_numbers[1]),
                        )
        
        self.soft_max = nn.Softmax(1)

    def set_T(self, T):
        self.T = T

    def forward(self, x):
        fea = self.cnn_sequence(x)
        flatted = fea.view(-1, self.cnn_out_length*self.feature_maps[-1])
        logit = self.fc_sequence(flatted)
        distill = logit / self.T
        p = self.soft_max(distill)
        return p, logit, fea

    def predict(self, fea, T=None):
        flatted = fea.view(-1, self.cnn_out_length*self.feature_maps[-1])
        logit = self.fc_sequence(flatted)
        distill = logit / (self.T if T is None else T)
        p = self.soft_max(distill)
        return p
