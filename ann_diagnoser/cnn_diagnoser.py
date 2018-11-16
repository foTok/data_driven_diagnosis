"""
This file defines the basic CNN diagnoser for BPSK communication system
The ANN is composed of a CNN module and a FC(Full Connection) module.

The CNN module has 3 CNN layers. We try to keep size of the input data by
setting padding = (kernel_size-1)/2. If kernel_size is an odd number, the size of
the input data will be reserved. But if kernel_size is an even number, the size will
plus one on both heigth and width. The kernel_size for the first two CNN layers 
should be setted as the same but the channel number can be control by featrue_maps
independently. The last CNN module is used to merge all the channel so we set the 
kernel_size as 1 and padding as 0 by default.

The FC module has 3 layers as well. The first layer maps all CNN output into a defined
vector, then the second layer maps them further into fault features. The last SoftMax 
layer is used to predict fault probability distribution based on the fault features.

The data should be:
                    batch × 5 × w(100 by default)
The output should be:
                    batch × 6
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class cnn_diagnoser(nn.Module):
    """
    The basic diagnoser constructed by Classic CNN
    """
    def __init__(self, kernel_size, feature_maps, fc_number, w=100, pool=True):
        '''
        kernel_size: an int or a tuple.
        feature_maps: a tuple, the feature numbers of the fisrt and second convolutional layers.
        fc_numbers: an int, the hidden number of full connection layers.
        w: an int, the length of data
        pool: a bool, if conduct pooling
        ''' 
        super(cnn_diagnoser, self).__init__()
        self.feature_maps = feature_maps
        self.fc_number = fc_number
        self.w = w
        if isinstance(kernel_size, int):
            self.padding = ((kernel_size-1)//2, (kernel_size-1)//2)
            kernel_size = (kernel_size, kernel_size)
        elif isinstance(kernel_size, tuple):
            self.padding = ((kernel_size[0]-1)//2, (kernel_size[1]-1)//2)
        else:
            raise Exception('Error in kernel_size')
        self.kernel_size = kernel_size
        self.delta = (2*self.padding[0]-kernel_size[0]+1, 2*self.padding[1]-kernel_size[1]+1)
        if pool:
            pool_kernel = (1, kernel_size[1])
        else:
            pool_kernel = (1, 1) # means no pooling
        self.size_after_pooling = ((5+2*self.delta[0]-(pool_kernel[0]-1) - 1)//pool_kernel[0] + 1,
                                   (w+2*self.delta[1]-(pool_kernel[1]-1) - 1)//pool_kernel[1] + 1)
        
        self.cnn_sequence = nn.Sequential(
                            # (1, 5, w)=>(feature_maps[0], 5+delta[0], w+delta[1])
                            nn.Conv2d(1, feature_maps[0], kernel_size, padding=self.padding),
                            nn.ReLU(),
                            # (feature_maps[0], 5+delta[0], w+delta[1])
                            # =>(feature_maps[1], 5+2delta[0], w+2delta[1])
                            nn.Conv2d(feature_maps[0], feature_maps[1], kernel_size, padding=self.padding),
                            nn.ReLU(),
                            # Setting kernel_size as 1 and padding as 0 to merge all the features from different channels
                            # (feature_maps[1], 5+2delta[0], w+2delta[1])
                            # => (feature_maps[2], 5+2delta[0], w+2delta[1])
                            nn.Conv2d(feature_maps[1], feature_maps[2], 1),
                            nn.MaxPool2d(pool_kernel),
                        )

        self.fc_sequence = nn.Sequential(
                            # feature_maps[2]×size_after_pooling[0]×size_after_pooling[1]=>fc_number
                            nn.Linear(feature_maps[2]*self.size_after_pooling[0]*self.size_after_pooling[1], fc_number),
                            nn.ReLU(),
                            nn.BatchNorm1d(fc_number),
                            nn.Linear(fc_number, 7),
                            nn.Softmax(1),
                        )

    def forward(self, x):
        x = self.cnn_sequence(x)
        x = x.view(-1, self.feature_maps[2]*self.size_after_pooling[0]*self.size_after_pooling[1])
        x = self.fc_sequence(x)
        return x
