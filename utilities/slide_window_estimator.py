'''
Read the result from Weka and use slide window to estimate it again.
'''

import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
import numpy as np
import argparse
from collections import Counter
from statistics.plot import plotROC
from data_manger.utilities import get_file_list


def weka_slide_window_estimator(file_name, step_len, strategy):
    labels = []
    predict = []
    with open(file_name, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            msg = line.split()
            labels.append(int(msg[1][0]))
            predict.append(int(msg[2][0]))
    labels2 = []
    predict2 = []
    num = int(len(labels)/step_len)
    if strategy=='max':
        for i in range(num):
            window_label = labels[step_len*i]
            window_predict = predict[step_len*i:step_len*(i+1)]
            predict_count = Counter(window_predict)
            top_one = predict_count.most_common(1)
            top_one = top_one[0][0]
            labels2.append(window_label)
            predict2.append(top_one)
    else:
        window = int(strategy)
        assert window < step_len
        for i in range(num):
            window_label = labels[step_len*i]
            labels2.append(window_label)
            window_predict = predict[step_len*i:step_len*(i+1)]
            p = 1
            for k in range(step_len - window + 1):
                small_window_predict = window_predict[k:k+window]
                small_window_predict = np.array(small_window_predict)
                if (small_window_predict[0]==small_window_predict).all():
                    p = small_window_predict[0]
                    break
            predict2.append(p)

    labels = np.array(labels2) - 1
    predict = np.array(predict2) - 1

    fault_num = max(labels)
    predict_vector = []
    for p in predict:
        vector = [0]*(fault_num+1)
        vector[p] = 1
        predict_vector.append(vector)
    predict_vector = np.array(predict_vector)
    roc = plotROC()
    roc.analyse(labels, predict_vector)
    auc = roc.auc('micro')
    return auc

def weka_estimator(file_name):
    labels = []
    predict = []
    with open(file_name, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            msg = line.split()
            labels.append(int(msg[1][0]))
            predict.append(int(msg[2][0]))
    labels = np.array(labels) - 1
    predict = np.array(predict) - 1
    fault_num = max(labels)
    predict_vector = []
    for p in predict:
        vector = [0]*(fault_num+1)
        vector[p] = 1
        predict_vector.append(vector)
    predict_vector = np.array(predict_vector)
    roc = plotROC()
    roc.analyse(labels, predict_vector)
    auc = roc.auc('micro')
    return auc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--system", type=str, choices=['bpsk', 'mt'], help="choose the system")
    parser.add_argument("-l", "--length", type=int, help="step length")
    parser.add_argument("-t", "--strategy", type=str, help="strategy")
    parser.add_argument("-i", "--index", type=int, help="index")
    args = parser.parse_args()

    path = parentdir + '\\utilities\\{}\\{}\\'.format(args.system, args.index)
    files = get_file_list(path)
    files = [f for f in files if f.endswith('.txt')]
    
    for f in files:
        if f.startswith('CNN') or f.startswith('LSTM'):
            auc = weka_estimator(path + f)
        else:
            auc = weka_slide_window_estimator(path+f, args.length, args.strategy)
        print('AUC of {} is: {}.'.format(f, auc))
