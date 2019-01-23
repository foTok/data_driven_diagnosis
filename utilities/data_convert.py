'''
Convert data to .arff so that they can be used by other tools like Weka.
'''
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
import numpy as np
import torch
import argparse
from data_manger.bpsk_data_tank import BpskDataTank
from data_manager2.data_manager import mt_data_manager
from data_manger.utilities import get_file_list


def numpy2arff(input, label, file, var_list, mode_list, rel=None):
    '''
    Args:
        inputs: the original data in numpy.array.
            batch × variable × timestep
        label: labels in numpy.array
        file: save file name
        var_list: the list of str variables
        mode_list: like var_list
        rel: relationship. See manual of Weka.
    '''
    if len(input.shape) == 3:
        x = input.transpose([0, 2, 1]) # batch × timestep × variable
    else:
        x = input

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
    if len(x.shape)==3:
        for i, l in zip(x, label):
            for j in i:
                entry = ''
                for k in j:
                    prefix = '' if entry=='' else ','
                    entry += (prefix+str(k))
                entry += ',' + mode_list[int(l)]
                data.append(entry)
    else:
        for i, l in zip(x, label):
            entry = ''
            for k in i:
                prefix = '' if entry=='' else ','
                entry += (prefix+str(k))
            entry += ',' + mode_list[int(l)]
            data.append(entry)

    model = relation + attribute + data

    with open(file, 'w') as f:
        f.write('\n'.join(model))

def feature2arff(input, label, model, file, mode_list, rel=None):
    '''
    Args:
        inputs: the original data in numpy.array or torch.tensor.
            batch × variable × timestep
        label: labels in numpy.array
        file: save file name
        var_list: the list of str variables
        mode_list: like var_list
        rel: relationship. See manual of Weka.
    '''
    if isinstance(input, np.ndarray):
        input = torch.tensor(input)
    diagnoser = torch.load(model)
    diagnoser.eval()
    _, _, features = diagnoser(input)
    if len(features.size())==3:
        features = torch.mean(features, 2)
    features = features.detach().numpy()
    _, fea_num = features.shape
    var_list = ['fe'+str(i) for i in range(fea_num)]
    numpy2arff(features, label, file, var_list, mode_list, rel)


if __name__ == "__main__":
    # parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--system", type=str, choices=['bpsk', 'mt'], help="choose the system")
    parser.add_argument("-m", "--model", type=str, help="choose the model file")
    parser.add_argument("-o", "--output", type=str, help="output file name")
    parser.add_argument("-b", "--batch", type=int, help="set batch size")
    parser.add_argument("-p", "--purpose", type=str, choices=['train', 'test', 'test2'], help="purpose")
    args = parser.parse_args()

    snr = 20
    batch = 8000 if args.batch is None else args.batch
    if args.system=='bpsk':
        var_list = ['m','p','c','s0','s1']
        mode_list = ['N', 'TMA', 'PCR', 'CAR', 'MPL', 'AMP', 'TMB']
        step_len = 128
        data_path = parentdir + '\\bpsk_navigate\\data\\{}\\'.format(args.purpose)
        mana = BpskDataTank()
        list_files = get_file_list(data_path)
        for file in list_files:
            mana.read_data(data_path+file, step_len=step_len, snr=snr)
        inputs, labels, _, _ = mana.random_batch(batch, normal=0.4, single_fault=10, two_fault=0)
        labels = torch.sum(labels*torch.Tensor([1,2,3,4,5,6]), 1).long()
        labels = labels.detach().numpy()
        if args.model is None:
            inputs = inputs.detach().numpy()
            numpy2arff(inputs, labels, args.output, var_list, mode_list)
        else:
            feature2arff(inputs, labels, args.model, args.output, mode_list)

    if args.system=='mt':
        step_len = 64
        data_path = parentdir + '\\tank_systems\\data\\{}\\'.format(args.purpose)
        mana = mt_data_manager()
        mana.load_data(data_path)
        mana.add_noise(snr)
        var_list = mana.cfg.variables[0:11]
        mode_list = ['normal'] + mana.cfg.faults
        inputs, labels = mana.random_h_batch(batch=batch, step_num=64, prop=0.2, sample_rate=1.0)
        if args.model is None:
            numpy2arff(inputs, labels, args.output, var_list, mode_list)
        else:
            feature2arff(inputs, labels, args.model, args.output, mode_list)
