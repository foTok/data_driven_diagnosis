"""
the main file to conduct the computation
"""
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
from ann_diagnoser.cnn_diagnoser import cnn_diagnoser
from data_manger.bpsk_data_tank import BpskDataTank
from data_manger.utilities import get_file_list
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as pl
import numpy as np
import time
import logging

#settings
logfile = 'CNN_Training_' + time.asctime( time.localtime(time.time())).replace(" ", "_").replace(":", "-")+'.txt'
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename=logfile, level=logging.DEBUG, format=LOG_FORMAT)
snr = 20
train_id = 0
times = 5
epsilon = 0.001
last_limit = 20
data_path = parentdir + '\\bpsk_navigate\\data\\train{}\\'.format(train_id)
prefix = "cnn"
# feature_maps_vec = [(10, 20, 10), (20, 40, 10), (40, 80, 10), (80, 160, 10), \
#                     (10, 20, 20), (20, 40, 20), (40, 80, 20), (80, 160, 20), \
#                     (10, 20, 40), (20, 40, 40), (40, 80, 40), (80, 160, 40)]
feature_maps_vec = [(160, 320, 40), (160, 320, 80), (160, 320, 160), \
                    (320, 600, 40), (320, 640, 80), (320, 640, 160)]
kernel_size = 3
fc_number = 200
#prepare data
mana = BpskDataTank()
step_len=100
list_files = get_file_list(data_path)
for file in list_files:
    mana.read_data(data_path+file, step_len=step_len, snr=snr)

for t in range(times):
    model_path = parentdir + '\\ddd\\ann_model\\train{}\\{}db\\{}\\'.format(train_id, snr, t)
    if not os.path.isdir(model_path):
        os.makedirs(model_path)

    for feature_maps in feature_maps_vec:
        model_name = prefix + '{};{};{}'.format(feature_maps, kernel_size, fc_number)
        para_name  = prefix + 'para{};{};{}'.format(feature_maps, kernel_size, fc_number)
        fig_name = prefix + 'fig{};{};{}.jpg'.format(feature_maps, kernel_size, fc_number)

        diagnoser = cnn_diagnoser(kernel_size, feature_maps, fc_number)
        print(diagnoser)
        loss = nn.CrossEntropyLoss()
        optimizer = optim.Adam(diagnoser.parameters(), lr=0.001, weight_decay=8e-3)

        #train
        epoch = 400
        batch = 1000
        train_loss = []
        running_loss = 0.0
        bg_time = time.time()
        for i in range(epoch):
            inputs, labels, _, _ = mana.random_batch(batch, normal=0.4, single_fault=10, two_fault=0)
            inputs = inputs.view(-1,1,5,step_len)
            labels = torch.sum(labels*torch.Tensor([1,2,3,4,5,6]), 1).long()
            optimizer.zero_grad()
            outputs = diagnoser(inputs)
            l = loss(outputs, labels.long())

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