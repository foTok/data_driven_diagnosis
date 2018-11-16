"""
some utilities
"""
import torch
import numpy as np
import matplotlib.pyplot as pl
from graph_model.utilities import vector2number as vec2num

def first_one(X):
    """
    find the first one in x
    """
    for x, i in zip(X, range(len(X))):
        if x == 1:
            return (i + 1)
    return 0

def hist_batch(batch_data):
    """
    scatter the batch in different modes
    """
    lb = np.array([first_one(x) for x in batch_data[:, :6]])
    n = len(batch_data[0,:]) - 6
    for k in range(n):
        sk = batch_data[:, k+6]
        pl.figure(k+1)
        for i in range(6):
            mask = (lb==i)
            sk_i = sk[mask]
            pl.subplot(6,1,i+1)
            pl.hist(sk_i, 30)
    pl.show()

def scatter_batch(batch_data):
    """
    scatter batch
    """
    n = len(batch_data[0, :]) - 6
    plt = batch_data[:, 6:]
    lb = np.array([first_one(x) for x in batch_data[:, :6]])
    x = round(np.sqrt(n))
    y = n//x + (0 if n%9 == 0 else 1)
    pl.figure()
    for i in range(n):
        pl.subplot(x, y, i+1)
        pl.scatter(lb, plt[:, i])
    pl.show()

def organise_data(inputs, labels, res, feature):
    """
    get all features, res0, res1 and res2
    """
    length = len(feature)
    fea_num = len(feature[0,:])
    batch_data = np.zeros((length, 6+fea_num+3))
    #the first 6 colums(0:6) are fault labels
    #["tma", "pseudo_rate", "carrier_rate", "carrier_leak", "amplify", "tmb"]
    fault_labels = labels.detach().numpy()
    batch_data[:,:6] = fault_labels
    #the mid fea_num colums(6:6+fea_num) are features
    feature = feature.detach().numpy()
    batch_data[:, 6:6+fea_num] = feature
    #the last 3 colums(6+fea_num:) are residuals
    res = np.array(res)
    #res12
    res = np.mean(np.abs(res), axis=2)
    batch_data[:,-3:-1] = res[:, :2]
    #res3
    inputs = inputs.detach().numpy()
    # s3 = np.mean(inputs[:, 3], axis=1)
    # s4 = np.mean(inputs[:, 4], axis=1)
    # batch_data[:, -1] = ( s4 - 10 * s3)
    s3 = inputs[:, 3]
    s4 = inputs[:, 4]
    batch_data[:, -1] = np.mean(np.abs( s4 - 10 * s3), axis=1)
    return batch_data

def organise_tensor_data(inputs, res):
    """
    organise inputs and residuals into a tensor to train or predict
    inputs: tensor
    res: list
    """
    if res == []:
        return inputs
    batch, _, step = inputs.shape
    res = np.array(res)
    res = np.array(res)
    res01 = res[:, 0:2, :]
    inputs_np = inputs.detach().numpy()
    s3 = inputs_np[:, 3, :]
    s4 = inputs_np[:, 4, :]
    res2 = s4-10*s3
    res2 = res2.reshape((batch, 1, step))
    res_data = np.concatenate((res01, res2), 1)
    res_data = torch.Tensor(res_data)
    data = torch.cat((inputs, res_data), 1)
    return data

def accuracy(outputs, labels):
    """
    compute the accuracy
    """
    outputs    = outputs.detach().numpy()
    outputs    = np.round(outputs)
    labels     = labels.detach().numpy()
    acc        = ((outputs + labels) == 2)
    length     = len(labels) / len(labels[0,:])
    acc        = np.sum(acc, 0)/length
    return acc

def single_fault_statistic(predictions, labels):
    """
    statistic for single-fault
    predictions and labels: N×F matrix
    """
    _, n = labels.shape
    n    = n + 1                                    #n faults, 1 normal
    acc_mat = np.zeros((n, n))
    for pre, lab in zip(predictions, labels):
        lab = np.where(lab == 1)
        lab = 0 if len(lab[0]) ==0 else (int(lab[0][0]) + 1)
        epred = np.zeros(n)
        if (pre == 1).any():
            epred[1:] = pre
        else:
            epred[0]  = 1
        acc_mat[lab]  = acc_mat[lab] + epred
    acc_sum = np.sum(acc_mat, 1)
    acc_mat = (acc_mat.T / acc_sum).T
    return acc_mat

def acc_fnr_and_fpr(predictions, labels):
    """
    compute the accuracy and false positive rate
    predictions and labels: N×F matrix
    """
    n  = 0                       # negative number number
    p  = 0                       # positive number
    f  = 0                       # fault number
    fn = 0                       # false negative number
    fp = 0                       # false positive number
    co = 0                       # false correct number
    for pre, lab in zip(predictions, labels):
        if (pre == 1).any():
            p = p + 1
        else:
            n = n + 1

        if (lab == 1).any():
            f = f +1
            if (pre == lab).all():
                co = co + 1
            if not (pre == 1).any():
                fn = fn + 1
        else:
            if (pre == 1).any():
                fp = fp + 1
    acc = co / f
    fnr = fn / n
    fpr = fp / p
    return acc, fnr, fpr
