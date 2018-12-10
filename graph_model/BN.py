'''
defines some components used by Bayesian network
'''
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
import numpy as np
import pickle
from math import log
from graph_model.cpt import PT
from graph_model.utilities import sort_v
from graph_model.utilities import Guassian_cost
from graph_model.utilities import Guassian_cost_core
from graph_model.utilities import discretize
from graphviz import Digraph

class Bayesian_adj:
    '''
    store Bayesian structure
    '''
    def __init__(self, fault, obs):
        '''
        The first f variables are the faults and the last n variables are observations.
        Args:
            fault: a list or tuple of strings, contains the fault names
            obs: a list or tuple of strings, contains the observation variables
        '''
        self._fault = fault
        self._obs = obs
        self._n = len(obs)  + 1   # node number
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
        for i in range(1, self._n):
            self._adj[0, i] = 1

    def set_adj(self, adj):
        self._adj = adj

    def available(self, i, j):
        # only edges between observed variables
        if not (1<=i<self._n and 1<=j<self._n and i!=j):
            return False
        return True

    def add_edge(self, i, j):
        assert isinstance(i, int) and isinstance(j, int)
        if not (1<=i<self._n and 1<=j<self._n and i!=j):
            return False
        if self._adj[i,j]==0 and self._adj[j,i]==0 and not self.has_path(j, i):
            self._adj[i,j]=1
            return True
        return False

    def add_perm(self, i, j):
        assert isinstance(i, int) and isinstance(j, int)
        if not (1<=i<self._n and 1<=j<self._n and i!=j):
            return False
        if self._adj[i,j]==0 and self._adj[j,i]==0 and not self.has_path(j, i):
            return True
        return False

    def remove_edge(self, i, j):
        assert isinstance(i, int) and isinstance(j, int)
        if not (1<=i<self._n and 1<=j<self._n and i!=j):
            return False
        if self._adj[i,j]==1:
            self._adj[i,j]=0
            return True
        return False

    def remove_perm(self, i, j):
        assert isinstance(i, int) and isinstance(j, int)
        if not (1<=i<self._n and 1<=j<self._n and i!=j):
            return False
        if self._adj[i,j]==1:
            return True
        return False

    def reverse_edge(self, i, j):
        assert isinstance(i, int) and isinstance(j, int)
        if not (1<=i<self._n and 1<=j<self._n and i!=j):
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
        if not (1<=i<self._n and 1<=j<self._n and i!=j):
            return False
        if self._adj[i,j]==1 and self._adj[j,i]==0:
            if self.has_path(i,j):
                return False
            else:
                return True
        return False

    def clone(self):
        copy = Bayesian_adj(self._fault, self._obs)
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

class Gauss:
    '''
    This class is used to store Gaussian parameters for a Bayesian network.
    Different from the common Gaussian parameters in which the variance is a constance.
    This class assumes that both the mean value and the variance value for a Gassian distribution
    are determined by a set of linear real value parameters.
    '''
    def __init__(self):
        #family tank is a set to store parameters.
        #KEY FML: (parents, kid).
        #parents are several numbers but kid is just only one number.
        #For example, (0,1,2,3,6). {0,1,2,3}-->6
        #VALUE PARA: ((beta, var))
        #beta is a vector stored in a np.array or np.matrix. When there are n parents, 
        #there should be (n+1) variable in beta because there is a constant term.
        #The same for var.
        self.fml_tank   = {}

    def add_para(self, fml, parameters):
        '''
        add parameters into self.fml_tank
        '''
        if fml in self.fml_tank:
            return
        self.fml_tank[fml] = parameters

    def nLogPc(self, kid, parents, kid_v, parents_v):
        '''
        Conditional Probability
        Args:
            kid: unit tuple, (kid_id,)
            parents:tuple, (p0_id,p1_id,...)
            kid_v: 2d np.array
            parents_v: 2d np.array
        Returns:
            P(kid|parents) = P(parents, kid)/P(parents)
            or None
        '''
        fml = tuple(list(parents) + list(kid))
        assert fml in self.fml_tank
        beta, var, _ = self.fml_tank[fml]
        kid_v = kid_v.reshape(-1)
        cost = Guassian_cost_core(parents_v, kid_v, beta, var, norm=False)
        return cost

class CPT:
    '''
    A class to store CPT
    '''
    def __init__(self, mins, intervals, bins):
        self._mins  = np.array(mins)
        self._intv  = np.array(intervals)
        self._bins  = np.array(bins)
        self._pts = {}

    def _discretize(self, vars, values):
        '''
        Args:
            vars: a 1d tuple or list of int
            values: a 1d or 2d np.array
        Returns:
            a 1d or 2d np.array
        '''
        ind = list(vars)
        mins = self._mins[ind]
        intv = self._intv[ind]
        bins = self._bins[ind]
        d_values = discretize(values, mins, intv, bins)
        return d_values

    def add_para(self, vars, pt):
        if vars in self._pts:
            return
        self._pts[vars] = pt

    def nLogPc(self, kid, parents, kid_v, parents_v):
        '''
        Conditional Probability
        Args:
            kid: unit tuple, (kid_id,)
            parents:tuple, (p0_id,p1_id,...)
            kid_v: 2d np.array
            parents_v: 2d np.array
        Returns:
            P(kid|parents) = P(parents, kid)/P(parents)
            or None
        '''
        kid_v   = kid_v if len(kid_v.shape)==2 else kid_v.reshape(1, len(kid_v))
        parents_v   = parents_v if parents == () or len(parents_v.shape)==2 else parents_v.reshape(1, len(parents_v))
        vars, values = sort_v(kid, parents, kid_v, parents_v)
        if vars not in self._pts or (parents not in self._pts and parents!=()):
            return None
        cost = []
        values = self._discretize(vars, values)
        parents_v = None if parents==() else self._discretize(parents, parents_v)
        if parents == ():
            for _values in values:
                Pj = self._pts[vars].p(_values)
                _cost = -log(Pj)
                cost.append(_cost)
        else:
            for _values, _parents_v in zip(values, parents_v):
                Pj = self._pts[vars].p(_values)
                Pp = self._pts[parents].p(_parents_v)
                _cost = -log(Pj/Pp)
                cost.append(_cost)
        cost = np.mean(cost)
        return cost


class BN:
    '''
    Bayesian Netowork
    '''
    def __init__(self, fault, obs):
        self.fault  = fault
        self.obs    = obs
        self.adj    = Bayesian_adj(fault, obs)
        self.para   = None
        self.type   = None
        self.n  = len(obs)+1

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
        This function returns the *negative log posteriori probability*.
            P(fault|obs) = P(fault, obs)/P(obs)
            P(fault|obs) ~ P(fault, obs)=P(fault)*P(obs|fault)
        Args:
            fault: a string, the fault name.
            obs: a 2d np.array, the values of all monitored variables.
                        time_steps × node
        '''
        logCost = 0
        f_v = 0 if fault=='normal' else self.adj._fault.index(fault) + 1
        obs = np.pad(obs,((0,0), (1,0)),'constant',constant_values = f_v)
        for kid, parents in self.adj:
            kid_v = obs[:, list(kid)]
            parents_v = obs[:, list(parents)]
            logCost += self.para.nLogPc(kid, parents, kid_v, parents_v)
        return logCost

    def save(self, file):
        s = pickle.dumps(self)
        with open(file, "wb") as f:
            f.write(s)

    def graphviz(self, file, view=True):
        '''
        Visulize the BN using graphviz.
        Args:
            file: file name
        '''
        comment = '{} BN'.format(self.type)
        dot = Digraph(comment=comment)
        # for fault
        dot.node('node0', 'fault', fillcolor='red', style='filled')
        # for obs
        id = 1
        for o in self.obs:
            dot.node('node{}'.format(id), o, fillcolor='green', style='filled')
            id += 1
        # edges
        adj = self.adj.adj()
        for i in range(self.n):
            for j in range(self.n):
                if adj[i, j] == 1:
                    dot.edge('node{}'.format(i), 'node{}'.format(j))
        dot.render(file, view=view)  
