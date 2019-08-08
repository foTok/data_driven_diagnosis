import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt 
import scipy.stats as stats

def correlationship_analyze(file_name):
    feature = []
    feature_data = []
    label_data = []
    label_data2 = []
    with open(file_name, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            msg = line.split()
            if msg[0]=='@RELATION':
                pass # do nothing
            elif msg[0]=='@ATTRIBUTE':
                if msg[1]!='mode':
                    feature.append(msg[1])
                else:
                    label = msg[2].strip('{}').split(',')
            elif msg[0]=='@DATA':
                pass # do nothing
            else:
                line_data = msg[0].split(',')
                # feature
                for i in range(len(feature)):
                    line_data[i] = float(line_data[i])
                feature_data.append(line_data[:-1])
                # label
                index = label.index(line_data[-1])
                _label = [0.0]*len(label)
                _label[index]=1.0
                label_data.append(_label)
                label_data2.append(float(index))
    spearmanr = np.zeros((len(feature), len(label)))
    spearmanr2 = []
    feature_data = np.array(feature_data)
    label_data = np.array(label_data)
    label_data2 = np.array(label_data2)
    for i in range(len(feature)):
        r = stats.spearmanr(feature_data[:,i], label_data2)
        spearmanr2.append(r[0])
        for j in range(len(label)):
            r = stats.spearmanr(feature_data[:,i], label_data[:,j])
            r = r[0]
            spearmanr[i, j] = r
    return spearmanr, spearmanr2

def correlationship_analyze2(file_name):
    feature = []
    feature_data = {}
    with open(file_name, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            msg = line.split()
            if msg[0]=='@RELATION':
                pass # do nothing
            elif msg[0]=='@ATTRIBUTE':
                if msg[1]!='mode':
                    feature.append(msg[1])
                else:
                    label = msg[2].strip('{}').split(',')
                    for l in label:
                        feature_data[l] = []
            elif msg[0]=='@DATA':
                pass # do nothing
            else:
                line_data = msg[0].split(',')
                # feature
                for i in range(len(feature)):
                    line_data[i] = float(line_data[i])
                _feature = line_data[:-1]
                # label
                _label = line_data[-1]
                feature_data[_label].append(_feature)
    spearmanr = np.zeros((len(feature), len(label)-1))
    normal = label[0]
    for j in range(len(label)-1):
        fault_data = np.array(feature_data[label[j+1]])
        normal_data = np.array(feature_data[normal])
        normal_data = normal_data[:len(fault_data),:]
        a = np.concatenate((fault_data, normal_data))
        b = np.zeros(len(a))
        b[:len(fault_data)] = 1.0
        for i in range(len(feature)):
            r = stats.spearmanr(a[:, i], b)
            # r = stats.pearsonr(a[:, i], b)
            r = r[0]
            spearmanr[i, j] = r
    return spearmanr

if __name__ == "__main__":
    cnn_filename = 'utilities\\bpsk\\0\\cnn_test.arff'
    lstm_filename = 'utilities\\bpsk\\0\\lstm_test.arff'
    encoder_filename = 'utilities\\bpsk\\0\\cnn_encoder_test.arff'
    xlabel = 'Faults'
    ylabel = 'Features'
    # CNN
    cnn_r = correlationship_analyze2(cnn_filename)
    where_are_nan = np.isnan(cnn_r)
    cnn_r[where_are_nan] = 0
    print(cnn_r)
    plt.clf()
    plt.subplot(132)
    ax = sns.heatmap(np.abs(cnn_r), vmin=0, vmax=1, cmap='YlGnBu')
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=10)
    # LSTM
    lstm_r = correlationship_analyze2(lstm_filename)
    where_are_nan = np.isnan(lstm_r)
    lstm_r[where_are_nan] = 0
    plt.subplot(133)
    ax = sns.heatmap(np.abs(lstm_r), vmin=0, vmax=1, cmap='YlGnBu')
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=10)
    print(lstm_r)
    # Autoencoder
    encoder_r = correlationship_analyze2(encoder_filename)
    where_are_nan = np.isnan(encoder_r)
    encoder_r[where_are_nan] = 0
    plt.subplot(131)
    ax = sns.heatmap(np.abs(encoder_r), vmin=0, vmax=1, cmap='YlGnBu')
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=10)
    print(encoder_r)
    plt.show()
