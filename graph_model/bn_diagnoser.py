'''
This file implement BN(including dynamic BN) based diagnoser
'''
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
import pickle
import numpy as np
from math import exp
from graph_model.BN import BN
from graph_model.DBN import DBN

class bn_diagnoser:
    '''
    bn_diagnoser will load trained network and give the best posteriori result.
    '''
    def __init__(self, file):
        '''
        Args:
            file: a string, the file path and file of the BN or DBN model.
        '''
        self.file   = file  # file path and name
        self.bn = None  # type: BN, DBN

        with open(file, 'rb') as f:
            self.bn = pickle.load(f)

    def _diagnose(self, obs):
        '''
        Args:
            obs: a 2d np.array.
        '''
        fault = self.bn.fault
        logCost = []
        fault = ['normal'] + fault
        for mode in fault:
            _logCost = self.bn.logCost(mode, obs)
            logCost.append(_logCost)
        P = [exp(-c) + 1e-20 for c in logCost]
        sumP = sum(P)
        normP = [p/sumP for p in P]
        fault_id = normP.index(max(normP))
        return fault_id, normP

    def diagnose(self, obs):
        '''
        Args:
            obs: a 3d np.array
                batch × time_steps × node
        '''
        batch = obs.shape[0]
        fault_ids = [0]*batch
        normP = [0]*batch
        for i, o in zip(range(batch),obs):
            fault_ids[i], normP[i] = self._diagnose(o)
        return np.array(fault_ids), np.array(normP)

