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
from data_manger.bpsk_data_tank import BpskDataTank
from data_manger.utilities import get_file_list
from statistics.plot_roc import plotROC

#   settings
train_id        = 1
snr             = 20
times           = 1
test_batch      = 2000
hidden_size_vec = [32, 64, 128]
fc_number       = (256, 7)
step_len        =100
prefix          = 'lstm'
roc_type        = 'micro' # 'macro'
#   log
log_path = parentdir + '\\log\\bpsk\\train{}\\{}db\\'.format(train_id, snr)
if not os.path.isdir(log_path):
    os.makedirs(log_path)
log_name = 'LSTM_Estimation_' + time.asctime( time.localtime(time.time())).replace(" ", "_").replace(":", "-")+'.txt'
logfile = log_path + log_name
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename=logfile, level=logging.DEBUG, format=LOG_FORMAT)
#prepare data
data_path = parentdir + '\\bpsk_navigate\\data\\test\\'
mana = BpskDataTank()
list_files = get_file_list(data_path)
for file in list_files:
    mana.read_data(data_path+file, step_len=step_len, snr=snr)

inputs, labels, _, res = mana.random_batch(test_batch, normal=1/7, single_fault=10, two_fault=0)
label = labels.detach().numpy()
y_label = np.sum(label*np.array([1,2,3,4,5,6]), 1)

for t in range(times):
    msg = "estimation {}.".format(t)
    logging.info(msg)
    print(msg)
    model_path = parentdir + '\\ann_diagnoser\\bpsk\\train{}\\{}db\\{}\\'.format(train_id, snr, t)
    if not os.path.isdir(model_path):
        os.makedirs(model_path)
    
    for hidden_size in hidden_size_vec:
        model_name = prefix + '{};{}'.format(hidden_size, fc_number)
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
