"""
data tank to manage data
"""

import os
from collections import defaultdict
import torch
import numpy as np
from torch.autograd import Variable
from utilities import read_data


def parse_filename(filename):
    """
    obtain parameters from the filename
    including fault types and the value of parameters
    """
    filename = os.path.split(filename)
    str_s = filename[-1].split("@")
    list_faults = str_s[0]
    list_faults = list_faults.split(",")

    list_parameters = str_s[1].rstrip(".npy")
    list_parameters = list_parameters.split(",")

    list_parameters2 = []
    for i in list_parameters:
        if i.find("(") != -1:
            i = i.lstrip("(")
            list_parameters2.append(float(i))
        elif i.find(")") != -1:
            i = i.rstrip(")")
            list_parameters2[-1] = [list_parameters2[-1], float(i)]
        else:
            list_parameters2.append(float(i))
    return list_faults, list_parameters2



class BpskDataTank():
    """
    store and manage data
    """
    def __init__(self):
        self.fault_type = ["tma", "pseudo_rate", "carrier_rate", "carrier_leak", "amplify", "tmb"]
        #input:[batch × feature × step_len]
        #list of numpy ndarray
        self.input = []
        #list of list
        self.mode  = []
        #list of list
        self.para  = []
        #list of residuals
        self.res   = []
        #map from mode to data
        self.map   = defaultdict(list)
        self.fe    = None
        self.step  = None

    def set_fault_type(self, fault_type):
        """
        RT
        """
        self.fault_type = fault_type

    def read_data(self, file_name, **kwargs):
        """
        read data and store them
        """
        step_len = 100 if "step_len" not in kwargs else kwargs["step_len"]
        split_point = None if "split_point" not in kwargs else kwargs["split_point"]
        snr = None if "snr" not in kwargs else kwargs["snr"]
        norm = False if "norm" not in kwargs else kwargs["norm"]
        normal, fault, n_res, f_res = read_data(file_name, step_len, split_point, snr, norm)
        list_fault, list_parameters = parse_filename(file_name)

        self.step = step_len
        self.fe   = len(normal[0])

        mode = [0, 0, 0, 0, 0, 0]
        #para = [0, 0, 0, 0, 0, [0, 0]]
        para = [0, 0, 0, 0, 0, 0, 0]
        #normal data
        if norm:
            for i, r in zip(normal, n_res):
                self.map[tuple(mode)].append(len(self.input))
                self.input.append(i)
                self.res.append(r)
                self.mode.append(tuple(mode))
                self.para.append(tuple(para))
        else:
            for i in normal:
                self.map[tuple(mode)].append(len(self.input))
                self.input.append(i)
                self.mode.append(tuple(mode))
                self.para.append(tuple(para))

        #fault data
        #find faults and parameters
        for i, j in zip(list_fault, list_parameters):
            assert i in self.fault_type
            index = self.fault_type.index(i)
            mode[index] = 1
            if isinstance(j, list):
                para[5] = j[0]
                para[6] = j[1]
            else:
                para[index] = j
        if norm:
            for i, r in zip(fault, f_res):
                self.map[tuple(mode)].append(len(self.input))
                self.input.append(i)
                self.res.append(r)
                self.mode.append(tuple(mode))
                self.para.append(tuple(para))
        else:
            for i in fault:
                self.map[tuple(mode)].append(len(self.input))
                self.input.append(i)
                self.mode.append(tuple(mode))
                self.para.append(tuple(para))

    def step_len(self):
        """
        return the step_len
        """
        return self.step

    def feature_num(self):
        """
        return the feature number
        """
        return self.fe

    def info(self):
        """
        print basic infomation
        """
        print("There are {} data".format(len(self.input)))

    def random_batch(self, batch, normal=0, single_fault=10, two_fault=10):
            """
            choose some data and return in torch Tensor
            normal: the proportion for normal
            single_fault: the weight for single fault
            two_fault: the weight for two faults
            """
            #fault number
            fault_num = {}
            for k in range(6):
                mode_vector = [0, 0, 0, 0, 0, 0]
                mode_vector[k] = 1
                mode_vector = tuple(mode_vector)
                len_data = len(self.map[mode_vector])
                fault_num[mode_vector] = single_fault if len_data != 0 else 0
            for k in range(6):
                for j in range(k+1, 6):
                    mode_vector = [0, 0, 0, 0, 0, 0]
                    mode_vector[k] = 1
                    mode_vector[j] = 1
                    mode_vector = tuple(mode_vector)
                    len_data = len(self.map[mode_vector])
                    fault_num[mode_vector] = two_fault if len_data != 0 else 0
            normalization = 0
            for m in fault_num:
                normalization = normalization + fault_num[m]
            if normalization != 0:
                for m in fault_num:
                    fault_num[m] = int(fault_num[m] * batch * (1 - normal)) // normalization

            all_fault = 0
            for m in fault_num:
                all_fault = all_fault + fault_num[m]
            mode_vector = [0, 0, 0, 0, 0, 0]
            mode_vector = tuple(mode_vector)
            fault_num[mode_vector] = int(batch * normal)

            batch = 0
            for m in fault_num:
                batch = batch + fault_num[m]
            #input – input tensor (minibatch x in_channels x iLen)
            #random init
            input_data = torch.randn(batch, self.feature_num(), self.step_len())
            mode = torch.randn(batch, 6)
            para = torch.randn(batch, 7)
            res = []
            #counter
            i = -1
            for m in fault_num:
                len_data = len(self.map[m])
                for _ in range(fault_num[m]):
                    i = i + 1
                    index = int(np.random.random() * len_data)
                    index = self.map[m][index]
                    signal = self.input[index]
                    input_data[i] = torch.from_numpy(signal)
                    mode[i] = torch.Tensor(self.mode[index])
                    para[i] = torch.Tensor(self.para[index])
                    if self.res:
                        res.append(self.res[index])
            return input_data, mode, para, res
