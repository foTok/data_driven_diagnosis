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
import graphviz
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt 
from scipy.stats import pearsonr
from data_manger.bpsk_data_tank import BpskDataTank
from data_manager2.data_manager import mt_data_manager
from data_manger.utilities import get_file_list
from feature_selection.utilities import best_index
from feature_selection.utilities import simple_net

def heat_map_feature_input(dnnfile, x, figname=None, isCNN=False, I_r=0.95):
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
    dnn = torch.load(dnnfile)
    for i in range(n):
        x_clone = x.clone()
        x_clone[:,i,:] = 0
        _, _, features1   = dnn(x)
        _, _, features2   = dnn(x_clone)
        if isCNN:
            features1 = torch.mean(features1, dim=2)
            features2 = torch.mean(features2, dim=2)
        diff = torch.abs(features1 - features2) # batch × features
        diff = torch.mean(diff, dim=0).detach().numpy()
        difference.append(diff)
    difference = np.array(difference)
    where_are_nan = np.isnan(difference)
    difference[where_are_nan] = 0
    max_ = np.max(difference, axis=0)
    difference = difference / max_

    fea_num = len(max_)
    I = np.array([np.sqrt(np.sum(difference[:, i]**2)) for i in range(fea_num)])
    Ivf = difference / I
    important_vars = {}
    log = []
    for i in range(fea_num):
        vars = best_index(Ivf[:, i], I_r)
        important_vars[i] = vars
        msg = 'important variables of feature {} = {}'.format(i, vars)
        print(msg)
        log.append(msg)
    logfile = figname + '.txt'
    with open(logfile, 'w') as f:
        f.write('\n'.join(log))

    ax = sns.heatmap(Ivf, vmin=0, vmax=1, cmap="YlGnBu")
    ax.invert_yaxis()
    ax.set_ylabel('Variables')
    ax.set_xlabel('Features')
    plt.title('Importance of Variables to Features')
    if figname is None:
        plt.show()
    else:
        figname = figname if figname.endswith('.svg') else (figname+'.svg')
        plt.savefig(figname, format='svg')
    return important_vars

def heat_map_fault_feature(dnnfile, x, fe, figname=None, isCNN=False, I_r=0.95):
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
    dnn = torch.load(dnnfile)
    labels, _, features1   = dnn(x)
    for i in range(fe):
        features2  = features1.clone()
        if isCNN:
            features2[:,i,:] = 0
        else:
            features2[:,i] = 0
        labels2, _ = dnn.predict(features2)
        diff = torch.abs(labels - labels2) # batch × faults
        diff = torch.mean(diff, dim=0).detach().numpy()
        difference.append(diff)
    difference = np.array(difference)
    max_ = np.max(difference, axis=0)
    difference = difference / max_

    mode_num = len(max_)
    I = np.array([np.sqrt(np.sum(difference[:, i]**2)) for i in range(mode_num)])
    Ivf = difference / I
    important_features = {}
    log = []
    for i in range(mode_num):
        features = best_index(Ivf[:, i], I_r)
        important_features[i] = features
        msg = 'important variables of mode {} = {}'.format(i, features)
        print(msg)
        log.append(msg)
    logfile = figname + '.txt'
    with open(logfile, 'w') as f:
        f.write('\n'.join(log))

    ax = sns.heatmap(Ivf, vmin=0, vmax=1, cmap="YlGnBu")
    ax.invert_yaxis()
    ax.set_ylabel('Features')
    ax.set_xlabel('Modes')
    plt.title('Importance of Features to Modes')
    if figname is None:
        plt.show()
    else:
        figname = figname if figname.endswith('.svg') else (figname+'.svg')
        plt.savefig(figname, format='svg')
    return important_features



if __name__ == '__main__':
    # parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('index', type=int, help='give the index of model')
    parser.add_argument("-s", "--system", type=str, choices=['bpsk', 'mt'], help="choose the system")
    parser.add_argument("-n", "--network", type=str, choices=['cnn', 'lstm'], help="choose the network")
    parser.add_argument("-b", "--batch", type=int, help="set batch size")
    args = parser.parse_args()

    system = ['bpsk', 'mt'] if args.system is None else [args.system]
    network = ['cnn', 'lstm'] if args.network is None else [args.network]
    test_batch = 20000 if args.batch is None else args.batch
    var_list = ['m','p','c','s0','s1']
    mode_list = ['N', 'TMA', 'PCR', 'CAR', 'MPL', 'AMP', 'TMB']
    # BPSK
    if 'bpsk' in system:
        data_path = parentdir + '\\bpsk_navigate\\data\\test\\'
        mana = BpskDataTank()
        list_files = get_file_list(data_path)
        for file in list_files:
            mana.read_data(data_path+file, step_len=128, snr=20)
        inputs, _, _, _ = mana.random_batch(test_batch, normal=1/7, single_fault=10, two_fault=0)
        # CNN
        if 'cnn' in network:
            ann = 'bpsk_cnn_distill_(8, 16, 32, 64).cnn'
            ann = parentdir + '\\ann_diagnoser\\bpsk\\train\\20db\\{}\\'.format(args.index) + ann
            important_vars = heat_map_feature_input(ann, inputs, figname='bpsk\\importance_heat_map_between_varialbe_feature_of_CNN', isCNN=True)
            important_features = heat_map_fault_feature(ann, inputs, 64, figname='bpsk\\importance_heat_map_between_feature_fault_of_CNN', isCNN=True)
            simple_net(important_vars, important_features, 'bpsk\\cnn_simple_net.gv', var_list=var_list, mode_list=mode_list)
        if 'lstm' in network:
            ann = 'bpsk_lstm_distill_8.lstm'
            ann = parentdir + '\\ann_diagnoser\\bpsk\\train\\20db\\{}\\'.format(args.index) + ann
            important_vars = heat_map_feature_input(ann, inputs, figname='bpsk\\importance_heat_map_between_varialbe_feature_of_LSTM')
            important_features = heat_map_fault_feature(ann, inputs, 8, figname='bpsk\\importance_heat_map_between_feature_fault_of_LSTM')
            simple_net(important_vars, important_features, 'bpsk\\lstm_simple_net.gv', var_list=var_list, mode_list=mode_list)

