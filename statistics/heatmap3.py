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

def heat_map_feature(dnnfile, x, mean=None):
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
    ann = torch.load(dnnfile)
    if mean is not None:
        features   = ann.features(x, mean)
    else:
        features   = ann.features(x)
    _, fe = features.size()
    features = features.detach().numpy()
    data = np.zeros((fe, fe))
    for i in range(fe):
        for j in range(fe):
            data[i, j], _ = pearsonr(features[:, i], features[:, j])
    ax = sns.heatmap(data, vmin=0, vmax=1, cmap="YlGnBu")
    ax.invert_yaxis()
    ax.set_ylabel('Features')
    ax.set_xlabel('Features')
    plt.show()

def heat_map_cp(bn_file, x):
    '''
    Args:
        dnnfile: the file path and name of DNN.
        x: inputs, batch × variable × timestep
    Returns:
        heat map
    Mask one variable, and compare the different ouputs.
    '''
    if isinstance(x, torch.Tensor):
        x = x.detach().numpy()
    obs = x.transpose([0, 2, 1])
    with open(bn_file, 'rb') as f:
        bn = pickle.load(f)
    batch, _, _ = x.shape
    fml_num = len(bn.obs)+1
    mode_num = len(bn.fault)+1
    data = np.zeros((fml_num, fml_num))
    bn_f = []
    for o in obs:
        bn_fo = []
        for f_v in range(mode_num):
            eobs = np.pad(o,((0,0), (1,0)),'constant',constant_values = f_v)
            for kid, parents in bn.adj:
                kid_v = eobs[:, list(kid)]
                parents_v = eobs[:, list(parents)]
                logCost = bn.para.nLogPc(kid, parents, kid_v, parents_v)
                bn_fo.append(logCost)
        bn_f.append(bn_fo)
    bn_f = np.array(bn_f)
    bn_f2 = np.zeros((batch, fml_num))
    for i in range(mode_num):
        bn_f2 = bn_f2 + bn_f[:, fml_num*i:fml_num*i+fml_num]
    bn_f2 = bn_f2 / mode_num
    for i in range(fml_num):
        for j in range(fml_num):
            data[i, j], _ = pearsonr(bn_f2[:, i], bn_f2[:, j])
    ax = sns.heatmap(data, vmin=0, vmax=1, cmap="YlGnBu")
    ax.invert_yaxis()
    ax.set_ylabel('Conditional Probability')
    ax.set_xlabel('Conditional Probability')
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
    heat_map_feature(ann1, inputs, mean=True)
    heat_map_feature(ann2, inputs)
    heat_map_cp(bn, inputs)

    # MT
    ann1 = 'cnn2(8, 16, 32, 64);(8, 4, 4, 4);(256, 21)'
    ann2 = 'lstm28;(256, 21)'
    bn = 'mt_GSANmodel, d=8, ptype=CPT, ntype=S.bn'
    ann1 = parentdir + '\\ann_diagnoser\\mt\\train2\\20db\\0\\' + ann1
    ann2 = parentdir + '\\ann_diagnoser\\mt\\train2\\20db\\0\\' + ann2
    bn = parentdir + '\\graph_model\\mt\\train1\\20db\\0\\' + bn
    data_path = parentdir + '\\tank_systems\\data\\test\\'
    mana = mt_data_manager()
    mana.load_data(data_path)
    mana.add_noise(20)
    inputs, _ = mana.random_h_batch(batch=test_batch, step_num=64, prop=0.2, sample_rate=1.0)
    heat_map_feature(ann1, inputs, mean=True)
    heat_map_feature(ann2, inputs)
    heat_map_cp(bn, inputs)

    print('DONE')
