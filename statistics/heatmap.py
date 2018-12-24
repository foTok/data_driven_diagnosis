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
import matplotlib.pylab as plt 
from scipy.stats import pearsonr
from graph_model.BN import BN
from data_manger.bpsk_data_tank import BpskDataTank
from data_manager2.data_manager import mt_data_manager
from data_manger.utilities import get_file_list


def bn_dnn_heat_map(dnn_file, bn_file, x, thresh=None, mean=None):
    '''
    Args:
        dnn_file: the dnn network diagnoser
        bn_file: the BN
    '''
    if isinstance(x, torch.Tensor):
        obs = x.detach().numpy()
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
    obs = obs.transpose([0, 2, 1])
    with open(bn_file, 'rb') as f:
        bn = pickle.load(f)
    bn_f = []
    for o in obs:
        bn_fo = []
        for f_v in range(len(bn.fault)+1):
            eobs = np.pad(o,((0,0), (1,0)),'constant',constant_values = f_v)
            for kid, parents in bn.adj:
                kid_v = eobs[:, list(kid)]
                parents_v = eobs[:, list(parents)]
                logCost = bn.para.nLogPc(kid, parents, kid_v, parents_v)
                bn_fo.append(logCost)
        bn_f.append(bn_fo)
    bn_f = np.array(bn_f)
    ann_f = np.exp(-ann_f)
    _, y = ann_f.shape
    _, x = bn_f.shape
    data = np.zeros((x, y))
    for i in range(x):
        for j in range(y):
            data[i, j], _ = pearsonr(bn_f[:, i], ann_f[:, j])
    data = np.abs(data)
    where_are_nan = np.isnan(data)
    data[where_are_nan] = 0
    if thresh is not None:
        less_than_thresh = (data < thresh)
        data[less_than_thresh] = 0
    ax = sns.heatmap(data, vmin=0, vmax=1, cmap="YlGnBu")
    ax.invert_yaxis()
    ax.set_ylabel('BN Conditional Probabilities')
    ax.set_xlabel('DNN Features (exp(-*))')
    plt.show()

if __name__ == '__main__':
    # BPSK
    test_batch = 2000
    ann1 = 'cnn2(8, 16, 32, 64);(8, 4, 4, 4);(256, 7)'
    ann2 = 'lstm232;(256, 7)'
    bn = 'GSANmodel, d=32, ptype=CPT, ntype=S.bn'
    ann1 = parentdir + '\\ann_diagnoser\\bpsk\\train1\\20db\\0\\' + ann1
    ann2 = parentdir + '\\ann_diagnoser\\bpsk\\train1\\20db\\0\\' + ann2
    bn = parentdir + '\\graph_model\\bpsk\\train1\\20db\\0\\' + bn
    data_path = parentdir + '\\bpsk_navigate\\data\\test\\'
    mana = BpskDataTank()
    list_files = get_file_list(data_path)
    for file in list_files:
        mana.read_data(data_path+file, step_len=128, snr=20)
    inputs, _, _, _ = mana.random_batch(test_batch, normal=1/7, single_fault=10, two_fault=0)
    bn_dnn_heat_map(ann1, bn, inputs, mean=True)
    bn_dnn_heat_map(ann2, bn, inputs)
    # MT
    ann1 = 'cnn2(8, 16, 32, 64);(8, 4, 4, 4);(256, 21)'
    ann2 = 'lstm28;(256, 21)'
    bn = 'mt_GSANmodel, d=8, ptype=CPT, ntype=S.bn'
    ann1 = parentdir + '\\ann_diagnoser\\mt\\train1\\20db\\0\\' + ann1
    ann2 = parentdir + '\\ann_diagnoser\\mt\\train1\\20db\\0\\' + ann2
    bn = parentdir + '\\graph_model\\mt\\train1\\20db\\0\\' + bn
    data_path = parentdir + '\\tank_systems\\data\\test\\'
    mana = mt_data_manager()
    mana.load_data(data_path)
    mana.add_noise(20)
    inputs, _ = mana.random_h_batch(batch=test_batch, step_num=64, prop=0.2, sample_rate=1.0)
    bn_dnn_heat_map(ann1, bn, inputs, mean=True)
    bn_dnn_heat_map(ann2, bn, inputs)
