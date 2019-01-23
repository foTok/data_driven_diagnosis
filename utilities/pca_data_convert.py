import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
import argparse
import torch
import numpy as np
from feature_selection.PCA_based_feature import PCA_feature_selection
from data_manger.bpsk_data_tank import BpskDataTank
from data_manager2.data_manager import mt_data_manager
from data_manger.utilities import get_file_list
from data_convert import numpy2arff


if __name__ == "__main__":
    # parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--system", type=str, choices=['bpsk', 'mt'], help="choose the system")
    parser.add_argument("-b", "--batch", type=int, help="set batch size")
    args = parser.parse_args()

    snr = 20
    batch = 8000 if args.batch is None else args.batch
    if args.system=='bpsk':
        mode_list = ['N', 'TMA', 'PCR', 'CAR', 'MPL', 'AMP', 'TMB']
        step_len = 128
        pca_selection = PCA_feature_selection(0.95)
        # train
        train_path = parentdir + '\\bpsk_navigate\\data\\train\\'
        mana_train = BpskDataTank()
        list_files = get_file_list(train_path)
        for file in list_files:
            mana_train.read_data(train_path+file, step_len=step_len, snr=snr)
        inputs, labels, _, _ = mana_train.random_batch(batch, normal=0.4, single_fault=10, two_fault=0)
        inputs = inputs.detach().numpy()
        labels = torch.sum(labels*torch.Tensor([1,2,3,4,5,6]), 1).long()
        labels = labels.detach().numpy()
        batch, variable, step = inputs.shape
        inputs = inputs.transpose((0, 2, 1))
        inputs = inputs.reshape((batch*step,variable))
        inputs = pca_selection.learn_from(inputs)
        labels = np.repeat(labels, step)
        _, fe_num = inputs.shape
        var_list = ['fe'+str(i) for i in range(fe_num)]
        numpy2arff(inputs, labels, 'pca_train.arff', var_list, mode_list)
        # test
        test_path = parentdir + '\\bpsk_navigate\\data\\test\\'
        mana_test = BpskDataTank()
        list_files = get_file_list(test_path)
        for file in list_files:
            mana_test.read_data(test_path+file, step_len=step_len, snr=snr)
        inputs, labels, _, _ = mana_test.random_batch(batch, normal=0.4, single_fault=10, two_fault=0)
        inputs = inputs.detach().numpy()
        labels = torch.sum(labels*torch.Tensor([1,2,3,4,5,6]), 1).long()
        labels = labels.detach().numpy()
        batch, variable, step = inputs.shape
        inputs = inputs.transpose((0, 2, 1))
        inputs = inputs.reshape((batch*step,variable))
        labels = np.repeat(labels, step)
        inputs = pca_selection.transform(inputs)
        numpy2arff(inputs, labels, 'pca_test.arff', var_list, mode_list)

    if args.system=='mt':
        step_len = 64
        pca_selection = PCA_feature_selection(0.95)
        # train
        train_path = parentdir + '\\tank_systems\\data\\train\\'
        mana_train = mt_data_manager()
        mana_train.load_data(train_path)
        mana_train.add_noise(snr)
        mode_list = ['normal'] + mana_train.cfg.faults
        inputs, labels = mana_train.random_h_batch(batch=batch, step_num=64, prop=0.2, sample_rate=1.0)
        batch, variable, step = inputs.shape
        inputs = inputs.transpose((0, 2, 1))
        inputs = inputs.reshape((batch*step,variable))
        inputs = pca_selection.learn_from(inputs)
        _, fe_num = inputs.shape
        labels = np.repeat(labels, step)
        var_list = ['fe'+str(i) for i in range(fe_num)]
        numpy2arff(inputs, labels, 'pca_train2.arff', var_list, mode_list)
        # test
        test_path = parentdir + '\\tank_systems\\data\\test2\\'
        mana_test = mt_data_manager()
        mana_test.load_data(test_path)
        mana_test.add_noise(snr)
        inputs, labels = mana_test.random_h_batch(batch=batch, step_num=64, prop=0.2, sample_rate=1.0)
        batch, variable, step = inputs.shape
        inputs = inputs.transpose((0, 2, 1))
        inputs = inputs.reshape((batch*step,variable))
        labels = np.repeat(labels, step)
        inputs = pca_selection.transform(inputs)
        numpy2arff(inputs, labels, 'pca_test2.arff', var_list, mode_list)
