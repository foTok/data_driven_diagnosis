'''
Train BPSK CNN diagnoser
'''
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
from ann_diagnoser.cnn_encoder_decoder import cnn_encoder_decoder
from ann_diagnoser.utilities import L1
from data_manager2.data_manager import mt_data_manager
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as pl
import numpy as np
import time
import logging

#   settings
snr                 = 20
step_len            = 64
kernel_sizes        = (5, 7)
prefix              = 'mt_cnn_autoencoder_'
#   log
log_path = parentdir + '\\log\\mt\\train\\{}db\\'.format(snr)
if not os.path.isdir(log_path):
    os.makedirs(log_path)
log_name = prefix + 'training_' + time.asctime( time.localtime(time.time())).replace(" ", "_").replace(":", "-")+'.txt'
logfile = log_path + log_name
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename=logfile, level=logging.DEBUG, format=LOG_FORMAT)
#prepare data
data_path = parentdir + '\\tank_systems\\data\\train\\'
mana = mt_data_manager()
mana.load_data(data_path)
mana.add_noise(snr)

model_path = parentdir + '\\ann_diagnoser\\mt\\train\\{}db\\'.format(snr)
name = prefix + str(kernel_sizes)

autoencoder = cnn_encoder_decoder(input_size=[11, step_len], feature_sizes=[64, 25], kernel_sizes=kernel_sizes)
print(autoencoder)
MSE = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.01, weight_decay=8e-3)

#train
epoch = 1000
batch = 1000
train_loss = []
running_loss = 0.0
bg_time = time.time()
for i in range(epoch):
    inputs, _ =  mana.random_h_batch(batch=batch, step_num=step_len, prop=0.2, sample_rate=1.0)
    input_data = torch.from_numpy(inputs)
    _, variable, time_step = input_data.size()
    output_data = input_data.detach()
    optimizer.zero_grad()

    output = autoencoder(input_data)

    mse_loss = MSE(output, output_data)
    l1_loss = L1(autoencoder, reg=5e-4)
    l = mse_loss + l1_loss

    loss_i = l.item()
    running_loss += loss_i
    train_loss.append(loss_i)
    if i % 10 == 9:
        msg = '%d loss: %.5f' %(i + 1, running_loss / 10)
        print(msg)
        logging.info(msg)
        running_loss = 0.0
    l.backward()
    optimizer.step()
ed_time = time.time()
msg = '{}, train time={}'.format(name, ed_time-bg_time)
logging.info(msg)
print(msg)
#save model
torch.save(autoencoder, model_path + name + '.cnn')
msg = 'saved model {} to {}'.format(name, model_path)
logging.info(msg)
print(msg)

#figure
pl.cla()
pl.plot(np.array(train_loss))
pl.title("Training Loss")
pl.xlabel("Epoch")
pl.ylabel("Loss")
pl.savefig(model_path+name +'.svg', format='svg')
