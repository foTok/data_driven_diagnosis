'''
Train MT CNN estimator
'''
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
from ann_diagnoser.cnn_diagnoser import cnn_diagnoser
from data_manager2.data_manager import mt_data_manager
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as pl
import numpy as np
import time
import logging
from statistics.plot_roc import plotROC

#settings
train_id            = 1
snr                 = 20
times               = 5
sample_rate         = 1.0
step_len            = 64
kernel_sizes        = (8, 4, 4, 4)
feature_maps_vec    = [(8, 16, 32, 64), (16, 32, 64, 128), (32, 64, 128, 256), (64, 128, 256, 512)]
fc_numbers          = (256, 21)
test_batch          = 2000
prefix              = 'cnn'
roc_type            = 'micro' # 'macro'
#   log
log_path = parentdir + '\\log\\mt\\train{}\\{}db\\'.format(train_id, snr)
if not os.path.isdir(log_path):
    os.makedirs(log_path)
log_name = 'CNN_Estimation_' + time.asctime( time.localtime(time.time())).replace(" ", "_").replace(":", "-")+'.txt'
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
    model_path = parentdir + '\\ann_diagnoser\\mt\\train{}\\{}db\\{}\\'.format(train_id, snr, t)

    inputs, y_label = mana.random_h_batch(batch=test_batch, step_num=64, prop=0.2, sample_rate=sample_rate)
    inputs = torch.from_numpy(inputs)
    
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
        roc.analyse(20, y_label, y_score)
        auc = roc.auc(roc_type)
        logging.info('{}, {} auc = {}'.format(model_name, roc_type, auc))
        print('{}, {}, auc = {}'.format(model_name, roc_type, auc))
        roc.plot(roc_type, view=False, file=model_path+roc_name)

        # print AUC
        AUC = [roc.auc(i) for i in range(21)]
        msg = 'AUC = {}'.format(str(AUC))
        logging.info(msg)
        print(msg)

print('DONE')
