"""
estimate the diagnoser or feature extractor randomly
"""
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
import time
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as pl
import numpy as np
from data_manger.bpsk_data_tank import BpskDataTank
from data_manger.utilities import get_file_list
from ddd.utilities import single_fault_statistic
from ddd.utilities import acc_fnr_and_fpr

#settings
logfile = 'GNN_RNN_estimation_' + time.asctime( time.localtime(time.time())).replace(" ", "_").replace(":", "-")+'.txt'
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename=logfile, level=logging.DEBUG, format=LOG_FORMAT)
snr = 20
train_id = 0
times = 5
data_path = parentdir + '\\bpsk_navigate\\data\\test\\'
test_batch = 2000
prefix = "gnn_rnn"
hidden_size_vec = [10, 20, 40, 80]
fc_number = [20, 40, 200]

#prepare data
mana = BpskDataTank()
step_len=100
list_files = get_file_list(data_path)
for file in list_files:
    mana.read_data(data_path+file, step_len=step_len, snr=snr)

inputs, labels, _, res = mana.random_batch(test_batch, normal=0.4, single_fault=10, two_fault=0)
inputs = inputs.view(-1,1,5,step_len)
label = labels.detach().numpy()
real_label = np.sum(label*np.array([1,2,3,4,5,6]), 1)

for t in range(times):
    msg = "estimation {}.".format(t)
    logging.info(msg)
    print(msg)
    model_path = parentdir + '\\ddd\\ann_model\\train{}\\{}db\\{}\\'.format(train_id, snr, t)
    if not os.path.isdir(model_path):
        os.makedirs(model_path)
    
    for hidden_size in hidden_size_vec:
        model_name = prefix + '{};{}'.format(hidden_size, fc_number)
        d = torch.load(model_path + model_name)
        d.eval()

        bg = time.clock()         # time start
        outputs = d(inputs)
        ed = time.clock()         # time end
        logging.info('{}, predict time={} for a {} batch'.format(model_name, ed-bg, test_batch))
        print('{}, predict time={} for a {} batch'.format(model_name, ed-bg, test_batch))
        prob = outputs.detach().numpy()
        pre_label = prob.argmax(axis=1)
        acc = np.sum(pre_label == real_label) / len(pre_label)
        logging.info('{}, acc = {}'.format(model_name, acc))
        print('{}, acc = {}'.format(model_name, acc))
