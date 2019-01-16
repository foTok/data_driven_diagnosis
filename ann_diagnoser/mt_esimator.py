'''
Train MT CNN estimator
'''
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
from data_manager2.data_manager import mt_data_manager
from statistics.plot_roc import plotROC
from data_manger.utilities import get_file_list
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as pl
import numpy as np
import time
import logging

#settings
snr                 = 20
times               = 5
sample_rate         = 1.0
step_len            = 64
test_batch          = 2000
roc_type            = 'micro' # 'macro'
#   log
log_path = parentdir + '\\log\\mt\\train\\{}db\\'.format(snr)
if not os.path.isdir(log_path):
    os.makedirs(log_path)
log_name = 'Estimation_' + time.asctime( time.localtime(time.time())).replace(" ", "_").replace(":", "-")+'.txt'
logfile = log_path + log_name
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename=logfile, level=logging.DEBUG, format=LOG_FORMAT)
#prepare data
data_path = parentdir + '\\tank_systems\\data\\test\\'
mana = mt_data_manager()
mana.load_data(data_path)
mana.add_noise(snr)

for t in range(times):
    msg = "estimation {}.".format(t)
    logging.info(msg)
    print(msg)
    model_path = parentdir + '\\ann_diagnoser\\mt\\train\\{}db\\{}\\'.format(snr, t)

    inputs, y_label = mana.random_h_batch(batch=test_batch, step_num=64, prop=0.2, sample_rate=sample_rate)
    inputs = torch.from_numpy(inputs)
    
    files = get_file_list(model_path)
    model_names = [f for f in files if f.endswith('.cnn')]
    for name in model_names:
        roc_name = 'ROC' + name

        d = torch.load(model_path + name)
        d.eval()

        bg = time.clock()         # time start
        outputs = d(inputs)
        ed = time.clock()         # time end
        logging.info('{}, predict time={} for a {} batch'.format(name, ed-bg, test_batch))
        print('{}, predict time={} for a {} batch'.format(name, ed-bg, test_batch))
        y_score = outputs.detach().numpy()

        # statistics
        roc = plotROC()
        roc.analyse(20, y_label, y_score)
        auc = roc.auc(roc_type)
        logging.info('{}, {} auc = {}'.format(name, roc_type, auc))
        print('{}, {}, auc = {}'.format(name, roc_type, auc))
        roc.plot(roc_type, view=False, file=model_path+roc_name)
