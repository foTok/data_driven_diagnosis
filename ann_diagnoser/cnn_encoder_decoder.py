import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class cnn_encoder_decoder(nn.Module):
    '''
    encoder decoder based on CNN
    '''
    def __init__(self, input_size, feature_sizes, kernel_sizes):
        '''
        Args:
            input_size: an int list or tuple with length=2, the size of input data. (sensor_number, data_length)
            feature_sizes: an int list or tuple with length=2. feature_size[1] is the number of encoded features.
            kernel_sizes: an int list or tuple whose length = 2. 
                Store the kernel sizes of the 2 Conv1d layers.
                ***********Please set kernel sizes as odd numbers.***********
        ''' 
        super(cnn_encoder_decoder, self).__init__()
        self.input_size = input_size
        self.feature_sizes = feature_sizes
        self.kernel_sizes = kernel_sizes
        self.padding = [(i-1)//2 for i in kernel_sizes] # Setting kernel sizes as odd numbers, the padding method will keep the data length unchanged.
        # encoder
        self.encoder_c1 = nn.Conv1d(input_size[0], feature_sizes[0], kernel_sizes[0], padding=self.padding[0]) # length
        self.encoder_a1 = nn.ReLU()
        self.encoder_p1 = nn.MaxPool1d(2, return_indices=True) # (length+delta[0])/2
        self.encoder_c2 = nn.Conv1d(feature_sizes[0], feature_sizes[1], kernel_sizes[1], padding=self.padding[1]) # length/2
        self.encoder_a2 = nn.ReLU()
        self.encoder_p2 = nn.MaxPool1d(2, return_indices=True) # length+delta[0]/4
        # decoder
        self.decoder_up1 = nn.MaxUnpool1d(2) # length/2
        self.decoder_c1 = nn.Conv1d(feature_sizes[1], feature_sizes[0], kernel_sizes[1], padding=self.padding[1]) # length/2
        self.decoder_a1 = nn.PReLU()
        self.decoder_up2 = nn.MaxUnpool1d(2) # length
        self.decoder_c2 = nn.Conv1d(feature_sizes[0], input_size[0], kernel_sizes[0], padding=self.padding[0]) # length
        self.decoder_a2 = nn.PReLU()

    def encode(self, x):
        '''
        x: batch × variable × time_step
        '''
        x = self.encoder_c1(x)
        x = self.encoder_a1(x)
        x, _ = self.encoder_p1(x)
        x = self.encoder_c2(x)
        x = self.encoder_a2(x)
        x, _ = self.encoder_p2(x)
        return x

    def forward(self, x):
        # x: batch × variable × timestep
        # encode
        x = self.encoder_c1(x)
        x = self.encoder_a1(x)
        x, indices1 = self.encoder_p1(x)
        x = self.encoder_c2(x)
        x = self.encoder_a2(x)
        x, indices2 = self.encoder_p2(x)
        # decode
        x = self.decoder_up1(x, indices2)
        x = self.decoder_c1(x)
        x = self.decoder_a1(x)
        x = self.decoder_up2(x, indices1)
        x = self.decoder_c2(x)
        x = self.decoder_a2(x)
        return x
