'''
Train BPSK LSTM diagnoser
'''
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
from ann_diagnoser.lstm_distill_diagnoser import lstm_distill_diagnoser
from ann_diagnoser.utilities import L1
from data_manger.bpsk_data_tank import BpskDataTank
from data_manger.utilities import get_file_list
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as pl
import numpy as np
import time
import logging

#   settings
snr             = 20
times           = 5
hidden_size_vec = [8, 16, 32]
fc_numbers      = (256, 7)
step_len        = 128
prefix          = 'bpsk_lstm_distill_'
#   log
log_path = parentdir + '\\log\\bpsk\\train\\{}db\\'.format(snr)
if not os.path.isdir(log_path):
    os.makedirs(log_path)
log_name = prefix + 'training_' + time.asctime( time.localtime(time.time())).replace(" ", "_").replace(":", "-")+'.txt'
logfile = log_path + log_name
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename=logfile, level=logging.DEBUG, format=LOG_FORMAT)
#prepare data
data_path = parentdir + '\\bpsk_navigate\\data\\train\\'
mana = BpskDataTank()
list_files = get_file_list(data_path)
for file in list_files:
    mana.read_data(data_path+file, step_len=step_len, snr=snr)

for t in range(times):
    model_path = parentdir + '\\ann_diagnoser\\bpsk\\train\\{}db\\{}\\'.format(snr, t)
    if not os.path.isdir(model_path):
        os.makedirs(model_path)

    for hidden_size in hidden_size_vec:
        name = prefix + str(hidden_size)

        diagnoser = lstm_distill_diagnoser(hidden_size, fc_numbers)
        print(diagnoser)
        loss = nn.CrossEntropyLoss()
        optimizer = optim.Adam(diagnoser.parameters(), lr=0.001, weight_decay=8e-3)

        #train
        epoch = 2000
        batch = 1000
        train_loss = []
        running_loss = 0.0
        bg_time = time.time()
        for i in range(epoch):
            inputs, labels, _, _ = mana.random_batch(batch, normal=0.4, single_fault=10, two_fault=0)
            labels = torch.sum(labels*torch.Tensor([1,2,3,4,5,6]), 1).long()
            optimizer.zero_grad()
            _, logits, _ = diagnoser(inputs)
            hard_loss = loss(logits, labels.long())
            l1_loss = L1(diagnoser, reg=5e-4)
            l = hard_loss + l1_loss

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
        torch.save(diagnoser, model_path + name + '.lstm')
        msg = 'saved model {} to {}'.format(name, model_path)
        logging.info(msg)
        print(msg)

        #figure
        pl.cla()
        pl.plot(np.array(train_loss))
        pl.title("Training Loss")
        pl.xlabel("Epoch")
        pl.ylabel("Loss")
        pl.savefig(model_path+name+'.svg', format='svg')
