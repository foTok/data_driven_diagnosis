'''
The script to learn BN based on Greedy Search Improve Naive Bayesian Network
'''

"""
the main file to conduct the computation
"""
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as pl
import numpy as np
import time
import logging
import gc
from graph_model.BN import BN
from graph_model.GSAN import GSAN
from data_manger.bpsk_data_tank import BpskDataTank
from data_manger.utilities import get_file_list
from graph_model.utilities import dis_para
from graph_model.utilities import cat_label_input

#settings
logfile = 'GSAN_Training_' + time.asctime( time.localtime(time.time())).replace(" ", "_").replace(":", "-")+'.txt'
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename=logfile, level=logging.DEBUG, format=LOG_FORMAT)
snr = 20
train_id = 1
times = 5
data_path = parentdir + '\\bpsk_navigate\\data\\train{}\\'.format(train_id)
prefix = 'GSAN'
cpd = ['CPT', 'GAU']
fault = ["tma", "pseudo_rate", "carrier_rate", "carrier_leak", "amplify", "tmb"]
obs = ['m', 'p', 'c', 's0', 's1']
ntypes = ['S', 'D']
dis = [2, 4, 8, 16, 32, 64, 128]
epoch = 50
batch = 400
#prepare data
mana = BpskDataTank()
step_len=128
list_files = get_file_list(data_path)
for file in list_files:
    mana.read_data(data_path+file, step_len=step_len, snr=snr)
mm  = mana.min_max()

msg = 'Log of Training GSAN'
logging.info(msg)
print(msg)

for t in range(times):
    for ntype in ntypes:
        model_path = parentdir + '\\graph_model\\bpsk\\train{}\\{}db\\{}\\'.format(train_id, snr, t)
        if not os.path.isdir(model_path):
            os.makedirs(model_path)
        
        for _type in cpd:
            if _type == 'CPT':
                para = dis
            else:
                para = [0]
            for d in para:
                bins    = [d]*len(obs)
                mins, intervals, bins = dis_para(mm, bins, len(fault), ntype=='D') if d!=0 else [None, None, None]
                para_name  = prefix + 'para, d={}, ptype={}, ntype={}'.format(d, _type, ntype)
                model_name = prefix + 'model, d={}, ptype={}, ntype={}.bn'.format(d, _type, ntype)
                fig_name = prefix + 'fig, d={}, ptype={}, ntype={}.gv'.format(d, _type, ntype)

                learner = GSAN(fault, obs)
                learner.set_ntype(ntype)
                learner.set_type(_type, mins, intervals, bins)
                learner.init_queue()
                #train
                bg_time = time.time()
                for i in range(epoch):
                    inputs, labels, _, _ = mana.random_batch(batch, normal=0.4, single_fault=10, two_fault=0)
                    labels = torch.sum(labels*torch.Tensor([1,2,3,4,5,6]), 1)
                    data = cat_label_input(labels, inputs, ntype=='D')

                    msg = learner.step(data)
                    logging.info(msg)
                    print(msg)

                BN = learner.best_BN()
                ed_time = time.time()
                msg = '{}, train time={}'.format(para_name, ed_time-bg_time)
                logging.info(msg)
                print(msg)

                BN.save(model_path+model_name)
                msg = 'save model {} to {}'.format(model_name, model_path)
                logging.info(msg)
                print(msg)
                BN.graphviz(model_path+fig_name, view=False)
                msg = 'save figure {} to {}'.format(fig_name, model_path)
                logging.info(msg)
                print(msg)
print('DONE!')
