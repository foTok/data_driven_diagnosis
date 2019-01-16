'''
Convert data to .arff so that they can be used by other tools like Weka.
'''
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
from data_manger.bpsk_data_tank import BpskDataTank
from data_manger.utilities import get_file_list
import numpy as np
import torch

def numpy2arff(input, label, file, var_list, mode_list, rel=None):
    '''
    Args:
        inputs: the original data in numpy.array.
            batch × variable × timestep
        label: batch
        file: save file name
        var_list: the list of str variables
        mode_list: like var_list
        rel: relationship. See manual of Weka.
    '''
    x = input.transpose([0, 2, 1]) # batch × timestep × variable

    rel = 'numpy2arff' if rel is None else rel
    relation = ['@RELATION ' + rel]

    attribute = []
    for v in var_list:
        attribute.append('@ATTRIBUTE '+ v +'	REAL')
    mode = '@ATTRIBUTE mode 	{'
    for m in mode_list:
        prefix = '' if mode.endswith('{') else ','
        mode += (prefix + m)
    mode += '}'
    attribute.append(mode)

    data = ['@DATA']
    for i, l in zip(x, label):
        for j in i:
            entry = ''
            for k in j:
                prefix = '' if entry=='' else ','
                entry += (prefix+str(k))
            entry += ',' + mode_list[int(l)]
            data.append(entry)
    
    model = relation + attribute + data

    with open(file, 'w') as f:
        f.write('\n'.join(model))


if __name__ == "__main__":
    bpsk = True
    mt = False

    if bpsk:
        file_name = 'bpsk\\train.arff'
        var_list = ['m','p','c','s0','s1']
        mode_list = ['N', 'TMA', 'PCR', 'CAR', 'MPL', 'AMP', 'TMB']
        train_id = 1
        snr = 20
        step_len = 128
        batch = 8000
        data_path = parentdir + '\\bpsk_navigate\\data\\train{}\\'.format(train_id)
        # data_path = parentdir + '\\bpsk_navigate\\data\\test\\'
        mana = BpskDataTank()
        list_files = get_file_list(data_path)
        for file in list_files:
            mana.read_data(data_path+file, step_len=step_len, snr=snr)
        inputs, labels, _, _ = mana.random_batch(batch, normal=0.4, single_fault=10, two_fault=0)
        labels = torch.sum(labels*torch.Tensor([1,2,3,4,5,6]), 1).long()
        inputs = inputs.detach().numpy()
        labels = labels.detach().numpy()
        numpy2arff(inputs, labels, file_name, var_list, mode_list)

    if mt:
        pass
