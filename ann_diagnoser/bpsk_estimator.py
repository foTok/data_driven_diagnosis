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
snr                 = 20
times               = 5
step_len            = 128
test_batch          = 2000
roc_type            = 'micro' # 'macro'
#   log
log_path = parentdir + '\\log\\bpsk\\train\\{}db\\'.format(snr)
if not os.path.isdir(log_path):
    os.makedirs(log_path)
log_name = 'Estimation_' + time.asctime( time.localtime(time.time())).replace(" ", "_").replace(":", "-")+'.txt'
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
    model_path = parentdir + '\\ann_diagnoser\\bpsk\\train\\{}db\\{}\\'.format(snr, t)

    inputs, labels, _, res = mana.random_batch(test_batch, normal=0.4, single_fault=10, two_fault=0)
    inputs = inputs.view(-1, 5, step_len)
    label = labels.detach().numpy()
    y_label = np.sum(label*np.array([1,2,3,4,5,6]), 1)
    
    files = get_file_list(model_path)
    model_names = [f for f in files if f.endswith('.cnn')]
    for name in model_names:
        roc_name = 'ROC' + name
        d = torch.load(model_path + name)
        d.eval()

        bg = time.clock()         # time start
        outputs, _, _ = d(inputs)
        ed = time.clock()         # time end
        logging.info('{}, predict time={} for a {} batch'.format(name, ed-bg, test_batch))
        print('{}, predict time={} for a {} batch'.format(name, ed-bg, test_batch))
        y_score = outputs.detach().numpy()

        # statistics
        roc = plotROC()
        roc.analyse(6, y_label, y_score)
        auc = roc.auc(roc_type)
        logging.info('{}, {} auc = {}'.format(name, roc_type, auc))
        print('{}, {}, auc = {}'.format(name, roc_type, auc))
        roc.plot(roc_type, view=False, file=model_path+roc_name)
