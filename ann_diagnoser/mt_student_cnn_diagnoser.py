import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class mt_student_cnn_diagnoser(nn.Module):
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
        super(mt_student_cnn_diagnoser, self).__init__()
        self.kernel_sizes = kernel_sizes
        self.length = length
        self.padding = [(i-1)//2 for i in kernel_sizes]
        self.pooling = 2 # Pooling size is set as 2
        delta = [(2*p-k+1) for p, k in zip(self.padding, kernel_sizes)]
        self.cnn_out_length = ((length+delta[0])//self.pooling + delta[1])//self.pooling
        self.cnn = nn.Sequential(
                            nn.Conv1d(11, 8, kernel_sizes[0], padding=self.padding[0]), # length+delta[0]
                            nn.ReLU(),
                            nn.MaxPool1d(self.pooling), # (length+delta[0])//self.pooling
                            nn.Conv1d(8, 25, kernel_sizes[1], padding=self.padding[1]), # (length+delta[0])//self.pooling + delta[1]
                            nn.ReLU(),
                            nn.MaxPool1d(self.pooling), # ((length+delta[0])//self.pooling + delta[1])//self.pooling
                        ) # [1, 2, 7, 8, 12, 15, 16, 19, 21, 24, 28, 29, 30, 32, 34, 36, 37, 42, 45, 46, 51, 55, 56, 61, 63]


        self.fc   = nn.Linear(self.cnn_out_length*25, 21)

        self.soft_max = nn.Softmax(1)

    def forward(self, x):
        # x: batch × variable × timestep
        features = self.cnn(x)
        flated_features = features.view(-1, self.cnn_out_length*25)
        logits = self.fc(flated_features)
        p = self.soft_max(logits)
        return p, logits, features
