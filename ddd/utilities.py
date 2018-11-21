"""
some utilities
"""
import torch
import numpy as np
import matplotlib.pyplot as pl

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
