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

    def logCost2(self, fault, obs, features):
        '''
        This function returns the *negative log posteriori probability*.
            P(fault|obs) = P(fault, obs)/P(obs)
            P(fault|obs) ~ P(fault, obs)=P(fault)*P(obs|fault)
        Args:
            fault: a string, the fault name.
            obs: a 2d np.array, the values of all monitored variables.
                        time_steps × node
        '''
        logCost = 0
        if isinstance(self.bn, BN):
            features = [0] + [i+1 for i in features]
            f_v = 0 if fault=='normal' else self.bn.adj._fault.index(fault) + 1
            obs = np.pad(obs,((0,0), (1,0)),'constant',constant_values = f_v)
            for kid, parents in self.bn.adj:
                if kid[0] not in features:
                    continue
                kid_v = obs[:, list(kid)]
                parents_v = obs[:, list(parents)]
                logCost += self.bn.para.nLogPc(kid, parents, kid_v, parents_v)
            return logCost
        elif isinstance(self.bn, DBN):
            obs_num = len(self.bn.obs)
            features = [0] + [i+1 for i in features] + [obs_num+i+1 for i in features]
            time_steps, _ = obs.shape
            values = [0]*(time_steps-1)
            f_v = 0 if fault=='normal' else self.bn.adj._fault.index(fault) + 1
            for i in range(time_steps-1):
                datai = [f_v] + list(obs[i, :]) + list(obs[i+1, :])
                values[i] = datai
            values = np.array(values)
            for kid, parents in self.bn.adj:
                if kid[0] not in features:
                    continue
                kid_v = values[:, list(kid)]
                parents_v = values[:, list(parents)]
                logCost += self.bn.para.nLogPc(kid, parents, kid_v, parents_v)
            return logCost
        else:
            raise RuntimeError('Unknown model')

    def _diagnose2(self, obs, features):
        '''
        Args:
            obs: a 2d np.array.
            features: the list of variables
        '''
        fault = self.bn.fault
        logCost = []
        fault = ['normal'] + fault
        for mode in fault:
            _logCost = self.logCost2(mode, obs, features)
            logCost.append(_logCost)
        P = [exp(-c) + 1e-20 for c in logCost]
        sumP = sum(P)
        normP = [p/sumP for p in P]
        fault_id = normP.index(max(normP))
        return fault_id, normP

    def diagnose2(self, obs, features):
        '''
        Args:
            obs: a 3d np.array
                batch × time_steps × node
        '''
        batch = obs.shape[0]
        fault_ids = [0]*batch
        normP = [0]*batch
        for i, o in zip(range(batch),obs):
            fault_ids[i], normP[i] = self._diagnose2(o, features)
        return np.array(fault_ids), np.array(normP)
