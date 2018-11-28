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
logfile = 'CNN_estimation_' + time.asctime( time.localtime(time.time())).replace(" ", "_").replace(":", "-")+'.txt'
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename=logfile, level=logging.DEBUG, format=LOG_FORMAT)
snr = 20
train_id = 1
times = 5
data_path = parentdir + '\\bpsk_navigate\\data\\test\\'
test_batch = 2000
prefix = "cnn"
kernel_sizes = (8, 4, 4, 4)
feature_maps_vec = [(8, 16, 32, 64), (16, 32, 64, 128), (32, 64, 128, 256), (64, 128, 256, 512)]
fc_numbers = (256, 7)
#prepare data
mana = BpskDataTank()
step_len=128
list_files = get_file_list(data_path)
for file in list_files:
    mana.read_data(data_path+file, step_len=step_len, snr=snr)

inputs, labels, _, res = mana.random_batch(test_batch, normal=0.4, single_fault=10, two_fault=0)
inputs = inputs.view(-1, 5, step_len)
label = labels.detach().numpy()
real_label = np.sum(label*np.array([1,2,3,4,5,6]), 1)

for t in range(times):
    msg = "estimation {}.".format(t)
    logging.info(msg)
    print(msg)
    model_path = parentdir + '\\ddd\\ann_model\\train{}\\{}db\\{}\\'.format(train_id, snr, t)
    if not os.path.isdir(model_path):
        os.makedirs(model_path)
    
    for feature_maps in feature_maps_vec:
        model_name = prefix + '{};{};{}'.format(feature_maps, kernel_sizes, fc_numbers)
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
