'''
estimate the diagnoser or feature extractor randomly
'''
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
import time
import logging
import torch
import numpy as np
import matplotlib.pyplot as pl
from statistics.plot_roc import plotROC
from data_manger.bpsk_data_tank import BpskDataTank
from data_manger.utilities import get_file_list

#settings
train_id            = 1
snr                 = 20
times               = 5
step_len            = 128
test_batch          = 2000
kernel_sizes        = (8, 4, 4, 4)
feature_maps_vec    = [(8, 16, 32, 64), (16, 32, 64, 128), (32, 64, 128, 256), (64, 128, 256, 512)]
fc_numbers          = (256, 7)
prefix              = 'cnn'
roc_type            = 'micro' # 'macro'
#   log
log_path = parentdir + '\\log\\bpsk\\train{}\\{}db\\'.format(train_id, snr)
if not os.path.isdir(log_path):
    os.makedirs(log_path)
log_name = 'CNN_Estimation_' + time.asctime( time.localtime(time.time())).replace(" ", "_").replace(":", "-")+'.txt'
logfile = log_path + log_name
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename=logfile, level=logging.DEBUG, format=LOG_FORMAT)
#prepare data
data_path = parentdir + '\\bpsk_navigate\\data\\test\\'
mana = BpskDataTank()
list_files = get_file_list(data_path)
for file in list_files:
    mana.read_data(data_path+file, step_len=step_len, snr=snr)

for t in range(times):
    msg = "estimation {}.".format(t)
    logging.info(msg)
    print(msg)
    model_path = parentdir + '\\ann_diagnoser\\bpsk\\train{}\\{}db\\{}\\'.format(train_id, snr, t)

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
