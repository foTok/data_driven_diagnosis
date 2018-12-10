'''
defines some components used by Dynamic Bayesian network
'''
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
import numpy as np
import pickle
from math import log
from graph_model.cpt import PT
from graph_model.BN import CPT
from graph_model.BN import Gauss
from graph_model.utilities import Guassian_cost
from graph_model.utilities import discretize
from graphviz import Digraph

class DBN_adj:
    '''
    store Bayesian structure
    '''
    def __init__(self, fault, obs):
        '''
        The first f variables are the faults and the last n variables are observations.
                               0    1   |obs| |obs|+1  2|obs| 2|obs|+1=self._n
                            +------+----------+-----------+
                            |      |          |           |
                            |  0   |     0    |     1     |
                            |      |          |           |
                            +-----------------------------+
                            |      |          |           |
                            |      |          |           |
                            |  0   |     0    |     ?     |
                            |      |          |           |
                            |      |          |           |
                            +-----------------------------+
                            |      |          |           |
                            |      |          |           |
                            |  0   |     0    |     ?     |
                            |      |          |           |
                            |      |          |           |
                            +------+----------+-----------+
        Args:
            fault: a list or tuple of strings, contains the fault names
            obs: a list or tuple of strings, contains the observation variables
        '''
        self._fault = fault
        self._obs = obs
        self._n = 2*len(obs)  + 1   # node number
        self._adj = None

    def adj(self):
        _adj = None
        if self._adj is not None:
            _adj = self._adj.copy()
        return _adj

    def len_var(self):
        return self._n

    def empty_init(self):
        self._adj = np.array([[0]*self._n]*self._n)

    def naive_init(self):
        self._adj = np.array([[0]*self._n]*self._n)
        for i in range(len(self._obs)+1, self._n):
            self._adj[0, i] = 1

    def set_adj(self, adj):
        self._adj = adj

    def available(self, i, j):
        # only edges between observed variables
        if 1<=i<self._n and len(self._obs)+1<=j<self._n and i!=j:
            return True
        return False

    def add_edge(self, i, j):
        assert isinstance(i, int) and isinstance(j, int)
        if not self.available(i,j):
            return False
        if self._adj[i,j]==0 and self._adj[j,i]==0 and not self.has_path(j, i):
            self._adj[i,j]=1
            return True
        return False

    def add_perm(self, i, j):
        assert isinstance(i, int) and isinstance(j, int)
        if not self.available(i, j):
            return False
        if self._adj[i,j]==0 and self._adj[j,i]==0 and not self.has_path(j, i):
            return True
        return False

    def remove_edge(self, i, j):
        assert isinstance(i, int) and isinstance(j, int)
        if not self.available(i,j):
            return False
        if self._adj[i,j]==1:       # When adding edges, self.add_edge makes sure
            self._adj[i,j]=0        # no undirected edge will be added.
            return True
        return False

    def remove_perm(self, i, j):
        assert isinstance(i, int) and isinstance(j, int)
        if not self.available(i,j):
            return False
        if self._adj[i,j]==1:
            return True
        return False

    def reverse_edge(self, i, j):
        assert isinstance(i, int) and isinstance(j, int)
        if not self.available(i,j):
            return False
        if self._adj[i,j]==1 and self._adj[j,i]==0:
            self._adj[i,j]==0
            if self.has_path(i,j):
                self._adj[i,j]=1
                return False
            else:
                self._adj[j,i]==1
                return True
        return False

    def reverse_perm(self, i, j):
        assert isinstance(i, int) and isinstance(j, int)
        if not self.available(i,j):
            return False
        if self._adj[i,j]==1 and self._adj[j,i]==0:
            if self.has_path(i,j):
                return False
            else:
                return True
        return False

    def clone(self):
        copy = DBN_adj(self._fault, self._obs)
        copy._adj = self._adj.copy()
        return copy

    def has_path(self, i, j):
        '''
        Args:
            i,j: int
        Returns:
            If there is a directed path from i to j
        '''
        open = {i}
        while open:
            p = open.pop()
            if self._adj[p,j]==1:
                return True
            kids = np.nonzero(self._adj[p,:])[0]
            for k in kids:
                open.add(k)
        return False

    def __eq__(self, other):
        return (self._adj == other._adj).all()

    def __hash__(self):
        return hash(tuple([tuple(i) for i in self._adj]))

    def __iter__(self):
        self._i = 0
        return self

    def __next__(self):
        if self._i == self._n:
            raise StopIteration
        parents = list(self._adj[:, self._i])
        parents = [i for i, v in enumerate(parents) if v==1]
        kid = [self._i]
        self._i += 1
        return tuple(kid), tuple(parents)

class DBN:
    '''
    Dynamic Bayesian Netowork
    '''
    def __init__(self, fault, obs):
        self.fault  = fault
        self.obs    = obs
        self.adj    = DBN_adj(fault, obs)
        self.para   = None
        self.type   = None
        self.n  = 2*len(obs)+1

    def set_adj(self, adj):
        if isinstance(adj, np.ndarray):
            self.adj.set_adj(adj)
        else:
            self.adj = adj

    def set_type(self, _type, mins=None, intervals=None, bins=None):
        if _type == 'CPT':
            assert mins is not None and intervals is not None and bins is not None
            self.mins   = np.array(mins)
            self.intervals  = np.array(intervals)
            self.bins   = np.array(bins)
            self.para = CPT(mins, intervals, bins)
        elif _type == 'GAU':
            self.para = Gauss()
        else:
            raise RuntimeError('Unknown types.')

    def add_para(self, key, value):
        self.para.add_para(key, value)

    def logCost(self, fault, obs):
        '''
        This function returns the negative log posteriori probability.
            P(fault|obs) = P(fault, obs)/P(obs)
            P(fault|obs) ~ P(fault, obs)=P(fault)*P(obs|fault)
        Args:
            fault: a string, the fault name.
            obs: a 2d np.array matrix; the values of all monitored variables.
                    time_steps × nodes 
        '''
        logCost = 0
        time_steps, _ = obs.shape
        values = [0]*(time_steps-1)
        f_v = 0 if fault=='normal' else self.adj._fault.index(fault)
        for i in range(time_steps-1):
            datai = [f_v] + list(obs[i, :]) + list(obs[i+1, :])
            values[i] = datai
        values = np.array(values)
        for kid, parents in self.adj:
            kid_v = values[:, list(kid)]
            parents_v = values[:, list(parents)]
            logCost += self.para.nLogPc(kid, parents, kid_v, parents_v)
        return logCost

    def save(self, file):
        s = pickle.dumps(self)
        with open(file, "wb") as f:
            f.write(s)

    def graphviz(self, file, view=True):
        '''
        Visulize the DBN using graphviz.
        Args:
            file: file name
        '''
        comment = '{} DBN'.format(self.type)
        dot = Digraph(comment=comment)
        # for fault
        dot.node('node0', 'fault(t)', fillcolor='red', style='filled')
        # for obs in the last time step
        id = 1
        for ot in self.obs:
            dot.node('node{}'.format(id), ot + '(t-1)', fillcolor='yellow', style='filled')
            id += 1
        # for obs in the current time step
        for ot in self.obs:
            dot.node('node{}'.format(id), ot + '(t)', fillcolor='green', style='filled')
            id += 1
        # edges
        adj = self.adj.adj()
        for i in range(self.n):
            for j in range(self.n):
                if adj[i, j] == 1:
                    dot.edge('node{}'.format(i), 'node{}'.format(j))
        dot.render(file, view=view)
