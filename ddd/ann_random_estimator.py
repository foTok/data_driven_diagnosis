"""
estimate the diagnoser or feature extractor randomly
"""
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as pl
import numpy as np
from data_manger.bpsk_data_tank import BpskDataTank
from data_manger.utilities import get_file_list
from ann_diagnoser.loss_function import CrossEntropy
from ann_diagnoser.loss_function import MSE
from ddd.utilities import organise_tensor_data
from ddd.utilities import single_fault_statistic
from ddd.utilities import acc_fnr_and_fpr


#data amount
small_data = True
#settings
snr = 20
obj = ["cnn", "dcnn", "dscnn", "rdscnn", "rdsecnn"]
PATH = parentdir
TEST_DATA_PATH = PATH + "\\bpsk_navigate\\data\\test\\"
ANN_PATH = PATH + "\\ddd\\ann_model\\" + ("big_data\\" if not small_data else "small_data\\") + str(snr) + "db\\"
step_len=100
criterion = CrossEntropy
norm = False
dia_name = []
for dia in obj:
    if dia   == "cnn":
        model_name      = "cnn.pkl"
    elif dia == "dcnn":
        model_name      = "dcnn.pkl"
    elif dia == "dscnn":
        model_name      = "dscnn.pkl"
    elif dia == "rdscnn":
        model_name      = "rdscnn.pkl"
        norm = True
    elif dia == "rdsecnn":
        model_name      = "rdsecnn.pkl"
        norm = True
    else:
        print("unkown object!")
        exit(0)
    dia_name.append(model_name)

mana = BpskDataTank()
list_files = get_file_list(TEST_DATA_PATH)
for file in list_files:
    mana.read_data(TEST_DATA_PATH+file, step_len=step_len, snr=snr, norm=norm)
#load diagnoser
diagnoser = []
for name in dia_name:
    d = torch.load(ANN_PATH + name)
    d.eval()
    diagnoser.append(d)
batch       = 10000

inputs, labels, _, res = mana.random_batch(batch, normal=0.4, single_fault=10, two_fault=0)
sen_res = organise_tensor_data(inputs, res)
for k, d in zip(range(len(dia_name)), diagnoser):
    start = time.clock()            # time start
    if obj[k] == "cnn":
        sen_res0 = sen_res[:, :5, :]
        sen_res0 = sen_res0.view(-1,1,5,100)
        outputs = d(sen_res0)
    else:
        outputs = d(sen_res)
    dig_time = time.clock() - start # time end
    prob = outputs.detach().numpy()
    pred = np.round(prob)
    targ = labels.detach().numpy()
    acc_mat = single_fault_statistic(pred, targ)
    acc, fnr, fpr = acc_fnr_and_fpr(pred, targ)
    print(dia_name[k])
    print("time=", dig_time)
    print("acc_mat=", acc_mat)
    print("acc=", acc, "fnr=", fnr, "fpr=", fpr)
