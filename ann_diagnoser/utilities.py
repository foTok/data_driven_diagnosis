'''
Some utility functions
'''
import torch
import torch.nn.functional as F

def L1(model, reg=5e-5):
    l1_loss = 0
    for name, param in model.named_parameters():
        if 'bias' not in name:
            l1_loss += F.l1_loss(param, target=torch.zeros_like(param), size_average=False)
    l1_loss *= reg
    return l1_loss

def L1_zero(output, size_average=False):
    return F.l1_loss(output, target=torch.zeros_like(output), size_average=size_average)


def cross_entropy(outputs, targets, average=True):
    ce = - targets * torch.log(outputs)
    ce = torch.sum(ce)
    if average:
        batch = outputs.size(0)
        ce  = ce / batch
    return ce
