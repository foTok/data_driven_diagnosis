'''
Defines a structure to describe the simulated data file.
'''
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
import pickle

class cfg_item:
    def __init__(self, file_name, fault_type, fault_time, fault_para):
        '''
        Args:
            file_name: string, the data file name.
            fault_time: float, the time a fault occurs.
            fault_para: a nubmer.
        '''
        self.file_name  = file_name
        self.fault_type = fault_type
        self.fault_time = fault_time
        self.fault_para = fault_para


class data_cfg:
    def __init__(self, variables, time_step, faults, fault_paras):
        '''
        Usually, one fault corresponds to one fault parameter. But sometimes, 
        a fault may correspond to several fault parameters.
        Args:
            faults: list of string, fault names.
            fault_paras: list of string, fault parameters names.
        '''
        self.variables  = variables
        self.time_step  = time_step
        self.faults = faults
        self.fault_paras    = fault_paras
        self.files  = []

    def add_file(self, file_name, fault_type='normal', fault_time=-1.0, fault_para_name=None, fault_para_value=None):
        assert fault_type in self.faults or fault_type=='normal'
        fault_para = [0.0]*len(self.fault_paras)
        if isinstance(fault_para_name, str):
            assert fault_para_name in self.fault_paras
            fault_para[self.fault_paras.index(fault_para_name)] = fault_para_value
        elif isinstance(fault_para_name, tuple):
            for para, value in zip(fault_para_name, fault_para_value):
                assert para in self.fault_paras
                fault_para[self.fault_paras.index(para)] = value
        elif fault_para_name is None: #normal
            pass    # nothing to do
        else:
            raise RuntimeError('fault_para_name type error')
        item    = cfg_item(file_name, fault_type, fault_time, fault_para)
        self.files.append(item)

    def save_cfg(self, file):
        _s = pickle.dumps(self)
        with open(file, "wb") as f:
            f.write(_s)
