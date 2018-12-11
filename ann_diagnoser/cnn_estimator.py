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
from statistics.plot_roc import plotROC
from data_manger.bpsk_data_tank import BpskDataTank
from data_manger.utilities import get_file_list

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
roc_type = 'micro' # 'macro'
kernel_sizes = (8, 4, 4, 4)
feature_maps_vec = [(8, 16, 32, 64), (16, 32, 64, 128), (32, 64, 128, 256), (64, 128, 256, 512)]
fc_numbers = (256, 7)
#prepare data
mana = BpskDataTank()
step_len=128
list_files = get_file_list(data_path)
for file in list_files:
    mana.read_data(data_path+file, step_len=step_len, snr=snr)

for t in range(times):
    msg = "estimation {}.".format(t)
    logging.info(msg)
    print(msg)
    model_path = parentdir + '\\ann_diagnoser\\ann_model\\train{}\\{}db\\{}\\'.format(train_id, snr, t)
    if not os.path.isdir(model_path):
        os.makedirs(model_path)

    inputs, labels, _, res = mana.random_batch(test_batch, normal=1/7, single_fault=10, two_fault=0)
    inputs = inputs.view(-1, 5, step_len)
    label = labels.detach().numpy()
    y_label = np.sum(label*np.array([1,2,3,4,5,6]), 1)
    
    for feature_maps in feature_maps_vec:
        model_name = prefix + '{};{};{}'.format(feature_maps, kernel_sizes, fc_numbers)
        roc_name = 'ROC' + model_name
        d = torch.load(model_path + model_name)
        d.eval()

        bg = time.clock()         # time start
        outputs = d(inputs)
        ed = time.clock()         # time end
        logging.info('{}, predict time={} for a {} batch'.format(model_name, ed-bg, test_batch))
        print('{}, predict time={} for a {} batch'.format(model_name, ed-bg, test_batch))
        y_score = outputs.detach().numpy()

        # statistics
        roc = plotROC()
        roc.analyse(6, y_label, y_score)
        auc = roc.auc(roc_type)
        logging.info('{}, {} auc = {}'.format(model_name, roc_type, auc))
        print('{}, {}, auc = {}'.format(model_name, roc_type, auc))
        roc.plot(roc_type, view=False, file=model_path+roc_name)

print('DONE')
