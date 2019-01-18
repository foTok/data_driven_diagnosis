'''
estimate the diagnoser or feature extractor randomly
'''
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
import time
import logging
import torch
import argparse
import numpy as np
import matplotlib.pyplot as pl
from statistics.plot_roc import plotROC
from data_manger.bpsk_data_tank import BpskDataTank
from data_manager2.data_manager import mt_data_manager
from data_manger.utilities import get_file_list

# set the log in log_path
def set_log(log_path):
    if not os.path.isdir(log_path):
        os.makedirs(log_path)
    log_name = 'Estimation_' + time.asctime( time.localtime(time.time())).replace(" ", "_").replace(":", "-")+'.txt'
    logfile = log_path + log_name
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(filename=logfile, level=logging.DEBUG, format=LOG_FORMAT)

# prepare data
# bpsk
def load_bpsk_data(data_path, snr):
    mana = BpskDataTank()
    list_files = get_file_list(data_path)
    for file in list_files:
        mana.read_data(data_path+file, step_len=128, snr=snr)
    return mana

def sample_bpsk_data(mana, batch, npor=0.4): #np, normal portion
    inputs, labels, _, _ = mana.random_batch(batch, normal=npor, single_fault=10, two_fault=0)
    label = labels.detach().numpy()
    y_label = np.sum(label*np.array([1,2,3,4,5,6]), 1)
    return inputs, y_label

# mt
def load_mt_data(data_path, snr):
    mana = mt_data_manager()
    mana.load_data(data_path)
    mana.add_noise(snr)
    return mana

def sample_mt_data(mana, batch, npor=0.2):
    inputs, y_label = mana.random_h_batch(batch=batch, step_num=64, prop=npor, sample_rate=1.0)
    inputs = torch.from_numpy(inputs)
    return inputs, y_label

# general estimation function
def estimate_model(model_path, model_type, inputs, y_label, key=''):
    batch, _, _ = inputs.size()
    files = get_file_list(model_path)
    model_names = []
    if 'cnn' in model_type:
        model_names += [f for f in files if (f.endswith('.cnn') and key in f)]
    if 'lstm' in model_type:
        model_names += [f for f in files if (f.endswith('.lstm') and key in f)]
    for name in model_names:
        roc_name = 'ROC' + name
        d = torch.load(model_path + name)
        d.eval()

        bg = time.clock()         # time start
        outputs, _, _ = d(inputs)
        ed = time.clock()         # time end
        logging.info('{}, predict time={} for a {} batch'.format(name, ed-bg, batch))
        print('{}, predict time={} for a {} batch'.format(name, ed-bg, batch))
        y_score = outputs.detach().numpy()

        # statistics
        roc = plotROC()
        roc.analyse(y_label, y_score)
        auc = roc.auc(roc_type)
        logging.info('{}, {} auc = {}'.format(name, roc_type, auc))
        print('{}, {}, auc = {}'.format(name, roc_type, auc))
        roc.plot(roc_type, view=False, file=model_path+roc_name)

if __name__ == "__main__":
    # input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--system", type=str, choices=['bpsk', 'mt'], help="choose the system")
    parser.add_argument("-n", "--network", type=str, choices=['cnn', 'lstm'], help="choose the network")
    parser.add_argument("-b", "--batch", type=int, help="set batch size")
    parser.add_argument("-k", "--key", type=str, help="key word in model names")
    args = parser.parse_args()

    # settings
    snr         = 20
    times       = 5
    batch       = 2000 if args.batch is None else args.batch
    roc_type    = 'micro' # 'macro'
    system      = ['bpsk', 'mt'] if args.system is None else [args.system]
    network     = ['cnn', 'lstm'] if args.network is None else [args.network]
    key         = '' if args.key is None else args.key
    # log
    log_path = parentdir + '\\log\\bpsk\\train\\{}db\\'.format(snr)
    set_log(log_path)

    if 'bpsk' in system:
        msg = 'Estimate BPSK.'
        logging.info(msg)
        print(msg)
        data_path = parentdir + '\\bpsk_navigate\\data\\test\\'
        mana = load_bpsk_data(data_path, snr)
        for t in range(times):
            msg = 'estimation {}.'.format(t)
            logging.info(msg)
            print(msg)
            model_path = parentdir + '\\ann_diagnoser\\bpsk\\train\\{}db\\{}\\'.format(snr, t)
            inputs, y_label = sample_bpsk_data(mana, batch)
            estimate_model(model_path, network, inputs, y_label, key)

    if 'mt' in system:
        msg = 'Estimate MT.'
        logging.info(msg)
        print(msg)
        data_path = parentdir + '\\tank_systems\\data\\test\\'
        mana = load_mt_data(data_path, snr)
        for t in range(times):
            msg = 'estimation {}.'.format(t)
            logging.info(msg)
            print(msg)
            model_path = parentdir + '\\ann_diagnoser\\mt\\train\\{}db\\{}\\'.format(snr, t)
            inputs, y_label = sample_mt_data(mana, batch)
            estimate_model(model_path, network, inputs, y_label, key)
