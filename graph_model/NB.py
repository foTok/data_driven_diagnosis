'''
Naive Bayesian Network
'''
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
import numpy as np
from graph_model.BN import BN
from graph_model.BN import Bayesian_adj
from graph_model.batch_statistic import CPT_BS
from graph_model.batch_statistic import Gauss_BS
from math import log
from math import exp

class NB:
    '''
    Naive Beyesian Network
    '''
    def __init__(self, fault, obs):
        self.adj    = None
        self.type   = None
        self.ntype  = None
        self.statistic  = None
        self.mins   = None
        self.intervals  = None
        self.bins   = None
        self.delay  = 0
        self.fault  = fault
        self.obs    = obs  
        self.n  = None
        # Naive Init
        adj = Bayesian_adj(self.fault, self.obs)
        adj.naive_init()
        self.adj = adj

    def set_type(self, _type, mins=None, intervals=None, bins=None):
        '''
        Set the parameter type.
        '''
        if _type == 'CPT':
            assert mins is not None and intervals is not None and bins is not None
            self.mins   = np.array(mins)
            self.intervals  = np.array(intervals)
            self.bins   = np.array(bins)
            self.statistic  = CPT_BS(self.mins, self.intervals, self.bins)
        elif _type == 'GAU':
            self.statistic  = Gauss_BS()
        else:
            raise RuntimeError('unknown parameter type')
        self.type   = _type

    def cost(self, adj):
        '''
        compute the likelihood cost
        '''
        cost = 0
        for kid, parents in adj:
            cost_u = self.statistic.fml_cost(kid, parents)
            cost += cost_u
        return cost

    def learn_parameters(self, batch):
        '''
        Args:
            batch: the data
        Returns:
            the cost
        '''
        #self.queue.clear()
        self.statistic.set_batch(batch)
        # Update best
        cost = self.cost(self.adj)
        return cost

    def learned_NB(self):
        '''
        The learned Naive Bayesian network.
        '''
        bBN = BN(self.fault, self.obs)
        bBN.set_type(self.type, self.mins, self.intervals, self.bins)
        bBN.set_adj(self.adj.adj())
        for kid, parents in self.adj:
            if self.type == 'CPT':
                vars = tuple(sorted(list(parents) + list(kid)))
                bBN.add_para(vars, self.statistic.para(vars))
                bBN.add_para(parents, self.statistic.para(parents))
            elif self.type == 'GAU':
                fml = tuple(list(parents) + list(kid))
                bBN.add_para(fml, self.statistic.para(fml))
            else:
                raise RuntimeError('Unknown type')
        return bBN
