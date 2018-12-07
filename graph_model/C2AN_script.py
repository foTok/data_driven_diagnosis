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
from graph_model.BN import BN
from graph_model.C2AN import C2AN
from data_manger.bpsk_data_tank import BpskDataTank
from data_manger.utilities import get_file_list
from graph_model.utilities import dis_para
from graph_model.utilities import cat_label_input

#settings
logfile = 'C2AN_Training_' + time.asctime( time.localtime(time.time())).replace(" ", "_").replace(":", "-")+'.txt'
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename=logfile, level=logging.DEBUG, format=LOG_FORMAT)
snr = 20
train_id = 0
times = 5
data_path = parentdir + '\\bpsk_navigate\\data\\train{}\\'.format(train_id)
prefix = 'gsin'
fault = ["tma", "pseudo_rate", "carrier_rate", "carrier_leak", "amplify", "tmb"]
obs = ['m', 'p', 'c', 's0', 's1']
#prepare data
mana = BpskDataTank()
step_len=128
list_files = get_file_list(data_path)
for file in list_files:
    mana.read_data(data_path+file, step_len=step_len, snr=snr)

bins    = [5]*len(obs)
mm  = mana.min_max()
mins, intervals, bins = dis_para(mm, bins, len(fault))
batch = 2000
for t in range(times):
    model_path = parentdir + '\\graph_model\\pg_model\\train{}\\{}db\\{}\\'.format(train_id, snr, t)
    if not os.path.isdir(model_path):
        os.makedirs(model_path)

    model_name = prefix + 'CPT'
    fig_name = prefix + 'figCPT.gv'

    inputs, labels, _, _ = mana.random_batch(batch, normal=1.0, single_fault=10, two_fault=0)
    labels = torch.sum(labels*torch.Tensor([1,2,3,4,5,6]), 1)
    data = cat_label_input(labels, inputs)

    learner = C2AN(fault, obs)
    learner.set_batch(data, mins, intervals, bins)
    learner.Build_adj()
    BN = learner.Build_BN()

    BN.save(model_path+model_name)
    msg = 'save {} to {}'.format(model_name, model_path)
    logging.info(msg)
    print(msg)
    BN.graphviz(model_path+fig_name, view=False)
    msg = 'save {} to {}'.format(fig_name, model_path)
    logging.info(msg)
    print(msg)
print('DONE!')
