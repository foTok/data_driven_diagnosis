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
from graph_model.BN import BN
from data_manger.bpsk_data_tank import BpskDataTank
from data_manger.utilities import get_file_list
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as pl
import numpy as np
import time
import logging

#settings
logfile = 'GSIN_Training_' + time.asctime( time.localtime(time.time())).replace(" ", "_").replace(":", "-")+'.txt'
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename=logfile, level=logging.DEBUG, format=LOG_FORMAT)
snr = 20
train_id = 1
times = 5
data_path = parentdir + '\\bpsk_navigate\\data\\train{}\\'.format(train_id)
prefix = 'gsin'
cpd = ['CPT', 'GAU']
fault = [] # TODO
obs = [] # TODO
#prepare data
mana = BpskDataTank()
step_len=128
list_files = get_file_list(data_path)
for file in list_files:
    mana.read_data(data_path+file, step_len=step_len, snr=snr)

for t in range(times):
    model_path = parentdir + '\\ddd\\pg_model\\train{}\\{}db\\{}\\'.format(train_id, snr, t)
    if not os.path.isdir(model_path):
        os.makedirs(model_path)
    for _type in cpd:
        model_name = prefix + '{}'.format(_type)
        fig_name = prefix + 'fig{}.jpg'.format(_type)

        diagnoer = BN(fault, obs)
        #train
        epoch = 2000
        batch = 1000
        train_loss = []
        running_loss = 0.0

        bg_time = time.time()
        for i in range(epoch):
            inputs, labels, _, _ = mana.random_batch(batch, normal=0.4, single_fault=10, two_fault=0)
            # inputs: batch × channel(1) × nodes × time_step
            inputs = inputs.permute([0, 1, 3, 2]) # TODO
