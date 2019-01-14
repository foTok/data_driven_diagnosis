'''
Train MT CNN diagnoser
'''
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
from ann_diagnoser.cnn_diagnoser2 import cnn_diagnoser2
from data_manager2.data_manager import mt_data_manager
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as pl
import numpy as np
import time
import logging

#settings
train_id            = 2
snr                 = 20
times               = 1
sample_rate         = 1.0
step_len            = 64
kernel_sizes        = (8, 4, 4, 4)
feature_maps_vec    = [(8, 16, 32, 64)]
fc_numbers          = (256, 21)
prefix              = 'cnn2'
#   log
log_path = parentdir + '\\log\\mt\\train{}\\{}db\\'.format(train_id, snr)
if not os.path.isdir(log_path):
    os.makedirs(log_path)
log_name = 'CNN_Training_' + time.asctime( time.localtime(time.time())).replace(" ", "_").replace(":", "-")+'.txt'
logfile = log_path + log_name
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename=logfile, level=logging.DEBUG, format=LOG_FORMAT)
#prepare data
data_path = parentdir + '\\tank_systems\\data\\train{}\\'.format(train_id)
mana = mt_data_manager()
mana.load_data(data_path)
mana.add_noise(snr)

for t in range(times):
    model_path = parentdir + '\\ann_diagnoser\\mt\\train{}\\{}db\\{}\\'.format(train_id, snr, t)
    if not os.path.isdir(model_path):
        os.makedirs(model_path)

    for feature_maps in feature_maps_vec:
        model_name = prefix + '{};{};{}'.format(feature_maps, kernel_sizes, fc_numbers)
        para_name  = prefix + 'para{};{};{}'.format(feature_maps, kernel_sizes, fc_numbers)
        fig_name = prefix + 'fig{};{};{}.jpg'.format(feature_maps, kernel_sizes, fc_numbers)

        diagnoser = cnn_diagnoser2(kernel_sizes, feature_maps, fc_numbers, input_size=(11, step_len))
        print(diagnoser)
        loss = nn.CrossEntropyLoss()
        optimizer = optim.Adam(diagnoser.parameters(), lr=0.001, weight_decay=8e-3)

        #train
        epoch = 500
        batch = 1000
        train_loss = []
        running_loss = 0.0
        bg_time = time.time()
        for i in range(epoch):
            inputs, labels = mana.random_h_batch(batch=batch, step_num=64, prop=0.2, sample_rate=sample_rate)
            inputs = torch.from_numpy(inputs)
            labels = torch.from_numpy(labels).long()
            optimizer.zero_grad()
            outputs = diagnoser(inputs)
            l = loss(outputs, labels)

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
        msg = '{}, train time={}'.format(para_name, ed_time-bg_time)
        logging.info(msg)
        print(msg)
        #save model
        torch.save(diagnoser.state_dict(), model_path + para_name)
        torch.save(diagnoser, model_path + model_name)
        msg = 'saved para {} and model {} to {}'.format(para_name, model_name, model_path)
        logging.info(msg)
        print(msg)

        #figure
        pl.cla()
        pl.plot(np.array(train_loss))
        pl.title("Training Loss")
        pl.xlabel("Epoch")
        pl.ylabel("Loss")
        pl.savefig(model_path+fig_name)

print('DONE')
