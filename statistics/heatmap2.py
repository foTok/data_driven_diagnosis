'''
This file try to construct heat map between inputs and features,
features and outputs.
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

def heat_map_feature_input(dnnfile, x, mean=None):
    '''
    Args:
        dnnfile: the file path and name of DNN.
        x: inputs, batch × variable × timestep
    Returns:
        heat map
    Mask one variable, and compare the different ouputs.
    '''
    if isinstance(x, np.ndarray):
        x = torch.Tensor(x)
    difference = [] # variable × features
    _, n, _ = x.size()
    ann = torch.load(dnnfile)
    for i in range(n):
        x_clone = x.clone()
        x_clone[:,i,:] = 0
        if mean is not None:
            features1   = ann.features(x, mean)
            features2   = ann.features(x_clone, mean)
        else:
            features1   = ann.features(x)
            features2   = ann.features(x_clone)
        diff = torch.abs(features1 - features2) # batch × features
        diff = torch.mean(diff, dim=0).detach().numpy()
        difference.append(diff)
    difference = np.array(difference)
    where_are_nan = np.isnan(difference)
    difference[where_are_nan] = 0
    threshold = 0.5
    max_ = np.max(difference, axis=0)
    difference = difference / max_
    ax = sns.heatmap(difference, vmin=0, vmax=1, cmap="YlGnBu")
    ax.invert_yaxis()
    ax.set_ylabel('Variables')
    ax.set_xlabel('Features')
    # choose the significant features
    _, fe = difference.shape
    for i in range(fe):
        dvar = difference[:, i]
        svar = [k for k, d in zip(range(n),dvar) if d>threshold]
        print('feature {}: {}'.format(i, svar))
    plt.show()

def heat_map_fault_feature(dnnfile, x, fe, cnn=False):
    '''
    Args:
        dnnfile: the file path and name of DNN.
        x: inputs, batch × variable × timestep
        fe: feature number
    Returns:
        heat map
    Mask one variable, and compare the different ouputs.
    '''
    if isinstance(x, np.ndarray):
        x = torch.Tensor(x)
    difference = [] # features × faults
    ann = torch.load(dnnfile)
    features   = ann.features(x)
    labels = ann.predict(features)
    for i in range(fe):
        features2  = features.clone()
        if cnn:
            features2[:,i,:] = 0
        else:
            features2[:,i] = 0
        labels2 = ann.predict(features2)
        diff = torch.abs(labels - labels2) # batch × faults
        diff = torch.mean(diff, dim=0).detach().numpy()
        difference.append(diff)
    difference = np.array(difference)
    threshold = 0.5
    max_ = np.max(difference, axis=0)
    difference = difference / max_
    ax = sns.heatmap(difference, vmin=0, vmax=1, cmap="YlGnBu")
    ax.invert_yaxis()
    ax.set_ylabel('Features')
    ax.set_xlabel('Faults')
    _, fa = difference.shape
    for i in range(fa):
        dvar = difference[:, i]
        svar = [k for k, d in zip(range(fe),dvar) if d>threshold]
        print('fault {}: {}'.format(i, svar))
    plt.show()


if __name__ == '__main__':
    # BPSK
    test_batch = 2000
    ann1 = 'cnn2(8, 16, 32, 64);(8, 4, 4, 4);(256, 7)'
    ann2 = 'lstm232;(256, 7)'
    ann1 = parentdir + '\\ann_diagnoser\\bpsk\\train1\\20db\\0\\' + ann1
    ann2 = parentdir + '\\ann_diagnoser\\bpsk\\train1\\20db\\0\\' + ann2
    data_path = parentdir + '\\bpsk_navigate\\data\\test\\'
    mana = BpskDataTank()
    list_files = get_file_list(data_path)
    for file in list_files:
        mana.read_data(data_path+file, step_len=128, snr=20)
    inputs, _, _, _ = mana.random_batch(test_batch, normal=1/7, single_fault=10, two_fault=0)
    heat_map_feature_input(ann1, inputs, mean=True)
    heat_map_feature_input(ann2, inputs)
    heat_map_fault_feature(ann1, inputs, 64, cnn=True)
    heat_map_fault_feature(ann2, inputs, 32)

    # MT
    ann1 = 'cnn2(8, 16, 32, 64);(8, 4, 4, 4);(256, 21)'
    ann2 = 'lstm28;(256, 21)'
    ann1 = parentdir + '\\ann_diagnoser\\mt\\train2\\20db\\0\\' + ann1
    ann2 = parentdir + '\\ann_diagnoser\\mt\\train2\\20db\\0\\' + ann2
    data_path = parentdir + '\\tank_systems\\data\\test\\'
    mana = mt_data_manager()
    mana.load_data(data_path)
    mana.add_noise(20)
    inputs, _ = mana.random_h_batch(batch=test_batch, step_num=64, prop=0.2, sample_rate=1.0)
    heat_map_feature_input(ann1, inputs, mean=True)
    heat_map_feature_input(ann2, inputs)
    heat_map_fault_feature(ann1, inputs, 64, cnn=True)
    heat_map_fault_feature(ann2, inputs, 8)

    print('DONE')
