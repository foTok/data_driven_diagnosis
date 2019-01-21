'''
Train MT CNN diagnoser
'''
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
from ann_diagnoser.mt_student_cnn_diagnoser import mt_student_cnn_diagnoser
from ann_diagnoser.utilities import L1
from ann_diagnoser.utilities import cross_entropy
from data_manager2.data_manager import mt_data_manager
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
step_len            = 64
T                   = 20
kernel_sizes        = (8, 4)
indexes             = [1, 2, 7, 8, 12, 15, 16, 19, 21, 24, 28, 29, 30, 32, 34, 36, 37, 42, 45, 46, 51, 55, 56, 61, 63]
prefix              = 'mt_cnn_student_'
#   log
log_path = parentdir + '\\log\\mt\\train\\{}db\\'.format(snr)
if not os.path.isdir(log_path):
    os.makedirs(log_path)
log_name = prefix + 'training_' + time.asctime( time.localtime(time.time())).replace(" ", "_").replace(":", "-")+'.txt'
logfile = log_path + log_name
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename=logfile, level=logging.DEBUG, format=LOG_FORMAT)
#prepare data
data_path = parentdir + '\\tank_systems\\data\\train\\'
mana = mt_data_manager()
mana.load_data(data_path)
mana.add_noise(snr)

# cumbersome models
cum_models = []
for t in range(times):
    model_path = parentdir + '\\ann_diagnoser\\mt\\train\\{}db\\{}\\'.format(snr, t)
    model_name = 'mt_cnn_distill_(8, 16, 32, 64).cnn'
    m = torch.load(model_path + model_name)
    m.eval()
    cum_models.append(m)

# define features
def sparse_features(models, indexes, x):
    m = cum_models[0]
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
        p = p.view(-1, 21, 1)
        p_list.append(p)
    p = torch.cat(p_list, 2)
    p = torch.mean(p, 2)
    p = p.detach()
    return p

for t in range(times):
    model_path = parentdir + '\\ann_diagnoser\\mt\\train\\{}db\\{}\\'.format(snr, t)
    name = prefix + str(kernel_sizes)

    diagnoser = mt_student_cnn_diagnoser(kernel_sizes, step_len)
    print(diagnoser)
    CE = nn.CrossEntropyLoss()
    MSE = nn.MSELoss()
    optimizer = optim.Adam(diagnoser.parameters(), lr=0.01, weight_decay=8e-3)

    #train
    epoch = 1000
    batch = 1000
    train_loss = []
    running_loss = 0.0
    bg_time = time.time()
    for i in range(epoch):
        inputs, labels =  mana.random_h_batch(batch=batch, step_num=step_len, prop=0.2, sample_rate=1.0)
        inputs = torch.from_numpy(inputs)
        labels = torch.from_numpy(labels).long()
        optimizer.zero_grad()

        outputs, logits, features = diagnoser(inputs)
        features = torch.mean(features, 2)
        distilled_outputs = soft_max(logits / T)
        distilled_cumbersome_outputs = distill_T(cum_models, inputs, T)
        learned_features = sparse_features(cum_models, indexes, inputs)

        hard_loss = CE(logits, labels)
        soft_loss = cross_entropy(distilled_outputs, distilled_cumbersome_outputs)
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
    msg = '{}, train time={}'.format(name, ed_time-bg_time)
    logging.info(msg)
    print(msg)
    #save model
    torch.save(diagnoser, model_path + name + '.cnn')
    msg = 'saved model {} to {}'.format(name, model_path)
    logging.info(msg)
    print(msg)

    #figure
    pl.cla()
    pl.plot(np.array(train_loss))
    pl.title("Training Loss")
    pl.xlabel("Epoch")
    pl.ylabel("Loss")
    pl.savefig(model_path+name +'.svg', format='svg')
