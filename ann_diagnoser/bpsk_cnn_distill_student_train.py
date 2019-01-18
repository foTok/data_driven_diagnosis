'''
Train BPSK CNN diagnoser
'''
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
from ann_diagnoser.bpsk_student_cnn_diagnoser import bpsk_student_cnn_diagnoser
from ann_diagnoser.utilities import L1
from ann_diagnoser.utilities import cross_entropy
from data_manger.bpsk_data_tank import BpskDataTank
from data_manger.utilities import get_file_list
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as pl
import numpy as np
import time
import logging

#   settings
snr                 = 20
times               = 5
step_len            = 128
T                   = 20
kernel_sizes        = (8, 4, 4, 4)
feature_maps        = (8, 16, 32, 64)
fc_numbers          = (256, 7)
feature_maps2       = (8, 4, 4, 4)
kernel_sizes2       = (8, 4)
fc_numbers2         = 256
indexes             = [36, 54, 11, 62, 23, 26, 5, 8, 29]
prefix              = 'bpsk_cnn_student_'
#   log
log_path = parentdir + '\\log\\bpsk\\train\\{}db\\'.format(snr)
if not os.path.isdir(log_path):
    os.makedirs(log_path)
log_name = prefix + 'training_' + time.asctime( time.localtime(time.time())).replace(" ", "_").replace(":", "-")+'.txt'
logfile = log_path + log_name
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename=logfile, level=logging.DEBUG, format=LOG_FORMAT)
#prepare data
data_path = parentdir + '\\bpsk_navigate\\data\\train\\'
mana = BpskDataTank()
list_files = get_file_list(data_path)
for file in list_files:
    mana.read_data(data_path+file, step_len=step_len, snr=snr)

# cumbersome models
cum_models = []
for t in range(times):
    model_path = parentdir + '\\ann_diagnoser\\bpsk\\train\\{}db\\{}\\'
    model_name = prefix + str(feature_maps)
    m = torch.load(model_path + model_name)
    m.eval()
    cum_models.append(m)

# define features
def sparse_features(models, indexes, x):
    m = cum_models[0]
    m.eval()
    _, _, features = m(x)
    features = features[:, indexes, :]
    features = torch.mean(features, 2)
    return features

# define the function to obtain distilled probability
soft_max = torch.nn.Softmax(1)
def distill_T(models, x, T):
    p_list = []
    for m in models:
        _, logits, _ = m(x)
        logits = logits / T
        p = soft_max(logits)
        p = p.view(-1, 7, 1)
        p_list.append(p)
    p = torch.cat(p_list, 2)
    p = torch.mean(p, 2)
    p = p.detach()
    return p

for t in range(times):
    model_path = parentdir + '\\ann_diagnoser\\bpsk\\train{}\\{}db\\{}\\'.format(train_id, snr, t)
    
    model_name = prefix2+ '{};{};{}'.format(feature_maps2, kernel_sizes2, fc_numbers2)
    para_name  = prefix + 'para{};{};{}'.format(feature_maps2, kernel_sizes2, fc_numbers2)
    fig_name = prefix2 + 'fig{};{};{}'.format(feature_maps2, kernel_sizes2, fc_numbers2)

    diagnoser = bpsk_student_cnn_diagnoser(kernel_sizes2, feature_maps2, fc_numbers2, length=step_len)
    print(diagnoser)
    CE = nn.CrossEntropyLoss()
    MSE = nn.MSELoss()
    optimizer = optim.Adam(diagnoser.parameters(), lr=0.001, weight_decay=8e-3)

    #train
    epoch = 1000
    batch = 1000
    train_loss = []
    running_loss = 0.0
    bg_time = time.time()
    for i in range(epoch):
        inputs, labels, _, _ = mana.random_batch(batch, normal=0.4, single_fault=10, two_fault=0)
        labels = torch.sum(labels*torch.Tensor([1,2,3,4,5,6]), 1).long()
        optimizer.zero_grad()

        outputs, logits, features = diagnoser(inputs)
        distilled_outputs = soft_max(logits / T)
        distilled_cumbersome_outputs = distill_T(cum_models, inputs, T)
        learned_features = sparse_features(cum_models, indexes, inputs)

        hard_loss = CE(outputs, labels)
        soft_loss = CE(distilled_outputs, distilled_cumbersome_outputs.long())
        mse_loss = MSE(features, learned_features)
        l1_loss = L1(diagnoser, reg=5e-4)
        l = hard_loss + T**2*soft_loss + mse_loss + l1_loss

        loss_i = l.item()
        running_loss += loss_i
        train_loss.append(loss_i)
        if i % 10 == 9:
            msg = '%d loss: %.5f' %(i + 1, running_loss / 10)
            print(msg)
            logging.info(msg)
            running_loss = 0.0
        l.backward()
        optimizer.step()
    ed_time = time.time()
    msg = '{}, train time={}'.format(para_name, ed_time-bg_time)
    logging.info(msg)
    print(msg)
    #save model
    torch.save(diagnoser, model_path + model_name)
    msg = 'saved model {} to {}'.format(model_name, model_path)
    logging.info(msg)
    print(msg)

    #figure
    pl.cla()
    pl.plot(np.array(train_loss))
    pl.title("Training Loss")
    pl.xlabel("Epoch")
    pl.ylabel("Loss")
    pl.savefig(model_path+fig_name)
