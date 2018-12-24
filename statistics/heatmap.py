'''
Code to plot heatmap
'''
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
import torch
import pickle
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr
from graph_model.BN import BN
from data_manger.bpsk_data_tank import BpskDataTank
from data_manager2.data_manager import mt_data_manager
from data_manger.utilities import get_file_list


def bn_dnn_heat_map(dnn_file, bn_file, x, mean=None):
    '''
    Args:
        dnn_file: the dnn network diagnoser
        bn_file: the BN
    '''
    if isinstance(x, torch.tensor):
        obs = x.detach.numpy()
    elif isinstance(x, np.ndarray):
        obs = torch.Tensor(x)
        x, obs = obs, x
    # ANN
    ann = torch.load(dnn_file)
    if mean is not None:
        ann_f   = ann.features(x, mean)
    else:
        ann_f   = ann.features(x)
    ann_f = ann_f.detach().numpy()
    ann_f = np.exp(-ann_f)
    # BN
    obs = inputs.transpose([0, 2, 1])
    with open(bn_file, 'rb') as f:
        bn = pickle.load(f)
    bn_f = []
    for f_v in range(len(bn.faults)+1):
        eobs = np.pad(obs,((0,0), (1,0)),'constant',constant_values = f_v)
        for kid, parents in bn.adj:
            kid_v = eobs[:, list(kid)]
            parents_v = eobs[:, list(parents)]
            logCost = bn.para.nLogPc(kid, parents, kid_v, parents_v)
            bn_f.append(logCost)
    bn_f = np.array(bn_f)
    ann_f = np.exp(-ann_f)
    data = pearsonr(ann_f, bn_f)
    ax = sns.heatmap(data)

if __name__ == '__main__':
    # BPSK
    test_batch = 2000
    ann = 'filename'
    bn = 'filename'
    ann = parentdir + '\\ann_diagnoser\\bpsk\\train1\\20db\\0\\' + ann
    bn = parentdir + '\\graph_model\\bpsk\\train1\\20db\\0\\' + bn
    data_path = parentdir + '\\bpsk_navigate\\data\\test\\'
    mana = BpskDataTank()
    list_files = get_file_list(data_path)
    for file in list_files:
        mana.read_data(data_path+file, step_len=128, snr=20)
    inputs, _, _, _ = mana.random_batch(test_batch, normal=1/7, single_fault=10, two_fault=0)
    bn_dnn_heat_map(ann, bn, inputs)
    # MT
    test_batch = 700
    ann = 'filename'
    bn = 'filename'
    ann = parentdir + '\\ann_diagnoser\\mt\\train1\\20db\\0\\' + ann
    bn = parentdir + '\\graph_model\\mt\\train1\\20db\\0\\' + bn
    data_path = parentdir + '\\tank_systems\\data\\test\\'
    mana = mt_data_manager()
    mana.load_data(data_path)
    mana.add_noise(20)
    inputs, _ = mana.random_h_batch(batch=test_batch, step_num=64, prop=0.2, sample_rate=1.0)
    bn_dnn_heat_map(ann, bn, inputs)
