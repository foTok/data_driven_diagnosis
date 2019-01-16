'''
Implement some algorithms to select important features.
'''
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
from data_manger.bpsk_data_tank import BpskDataTank
from data_manger.utilities import get_file_list
from feature_selection.utilities import sub_data
from feature_selection.utilities import best_index
import numpy as np
import torch

def distance_featue(normal, fault, I_r=0.95, k = 10):
    '''
    Used in [1] H. Khorasgani and G. Biswas, \
    “A combined model-based and data-driven approach for monitoring smart buildings,” \
    in DX2017, 2017.
    Args:
        normal: normal data, np.array, batch × feature(variable) × time_step
        fault: fault data, np.array, batch × feature(variable) × time_step
        k: based on the nearest k values.
    '''
    if not isinstance(normal, np.ndarray):
        normal = normal.detach().numpy()
        fault = fault.detach().numpy()
    def _distance(p1, p2):
        '''
        p1 and p2 are matirxes, feature(variable) × time_step
        '''
        diff = np.abs(p1 - p2)
        dis = np.sqrt(np.sum(diff**2)/diff.size)
        return dis
    batch_n, _, _ = normal.shape
    batch_f, _, _ = fault.shape
    assert k<=batch_n and k<=batch_f
    dis_matrix = np.zeros((batch_n, batch_f))
    for i in range(batch_n):
        for j in range(batch_f):
            dis = _distance(normal[i,:,:], fault[j,:,:])
            dis_matrix[i, j] = dis
    flatted_dis = dis_matrix.reshape(-1)
    sorted_index = np.argsort(flatted_dis)
    mini_pairs = []
    for i in range(k):
        index = sorted_index[i]
        i_th = index // batch_f
        j_th = index %  batch_f
        mini_pairs.append((i_th, j_th))
    dis_vars = None
    for pair in mini_pairs:
        n, f = pair
        dis = np.abs(normal[n,:,:] - fault[f,:,:])
        dis = np.mean(dis, axis=1)
        dis_vars = dis if dis_vars is None else (dis_vars + dis)
    dis = np.sqrt(np.sum(dis_vars**2))
    I_nf = dis_vars / dis
    features = best_index(I_nf, I_r)
    return features


if __name__ == "__main__":
    # bpsk
    train_id = 1
    snr = 20
    step_len = 128
    batch = 2000
    data_path = parentdir + '\\bpsk_navigate\\data\\train{}\\'.format(train_id)
    mana = BpskDataTank()
    list_files = get_file_list(data_path)
    for file in list_files:
        mana.read_data(data_path+file, step_len=step_len, snr=snr)
    inputs, labels, _, _ = mana.random_batch(batch, normal=0.4, single_fault=10, two_fault=0)
    labels = torch.sum(labels*torch.Tensor([1,2,3,4,5,6]), 1).long()
    fault_num = int(torch.max(labels).item())
    # normal data
    normal = sub_data(inputs, labels, 0)
    for i in range(1, fault_num+1):
        fault = sub_data(inputs, labels, i)
        features = distance_featue(normal, fault)
        print('important feautures of fault {} = {}'.format(i, features))
