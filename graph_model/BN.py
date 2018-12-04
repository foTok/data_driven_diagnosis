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
from graph_model.batch_statistic import sort_v
from graph_model.utilities import Guassian_cost
from graph_model.cpt import discretize

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
        self._f = len(fault)
        self._n = len(obs)
        self._v = self._f+self._n
        self._adj = None

    def len_var(self):
        return self._v

    def empty_init(self):
        self._adj = np.array([[0]*self._v]*self._v)

    def naive_init(self):
        self._adj = np.array([[0]*self._v]*self._v)
        for i in range(self._f):
            for j in range(self._f, self._v):
                self._adj[i, j] = 1

    def set_adj(self, adj):
        self._adj = adj

    def available(self, i, j):
        if not (0<=i<self._v and self._f<=j<self._v and i!=j):
            return False
        return True

    def add_edge(self, i, j):
        assert isinstance(i, int) and isinstance(j, int)
        if not (0<=i<self._v and self._f<=j<self._v and i!=j):
            return False
        if i<self._f: # fault->obs
            if self._adj[i,j]==0:
                self._adj[i,j]=1
                return True
        else: # obs->obs
            if self._adj[i,j]==0 and self._adj[j,i]==0 and not self.has_path(j, i):
                self._adj[i,j]=1
                return True
        return False

    def add_perm(self, i, j):
        assert isinstance(i, int) and isinstance(j, int)
        if not (0<=i<self._v and self._f<=j<self._v and i!=j):
            return False
        if i<self._f: # fault->obs
            if self._adj[i,j]==0:
                return True
        else: # obs->obs
            if self._adj[i,j]==0 and self._adj[j,i]==0 and not self.has_path(j, i):
                return True
        return False

    def remove_edge(self, i, j):
        assert isinstance(i, int) and isinstance(j, int)
        if not (0<=i<self._v and self._f<=j<self._v and i!=j):
            return False
        if i<self._f: # fault->obs
            if self._adj[i,j]==1:
                self._adj[i,j]=0
                return True
        else: # obs->obs
            if self._adj[i,j]==1:
                self._adj[i,j]=0
                return True
        return False

    def remove_perm(self, i, j):
        assert isinstance(i, int) and isinstance(j, int)
        if not (0<=i<self._v and self._f<=j<self._v and i!=j):
            return False
        if i<self._f: # fault->obs
            if self._adj[i,j]==1:
                return True
        else: # obs->obs
            if self._adj[i,j]==1:
                return True
        return False

    def reverse_edge(self, i, j):
        assert isinstance(i, int) and isinstance(j, int)
        if not (0<=i<self._v and self._f<=j<self._v and i!=j):
            return False
        if i<self._f:
            return False
        else:
            if self._adj[i,j]==1 and self._adj[j,i]==0:
                self._adj[i,j]==0
                if self.has_path(i,j):
                    self._adj[i,j]=1
                    return False
                else:
                    self._adj[j,i]==1
                    return True
        return False

    def reverse_prem(self, i, j):
        assert isinstance(i, int) and isinstance(j, int)
        if not (0<=i<self._v and self._f<=j<self._v and i!=j):
            return False
        if i<self._f:
            return False
        else:
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
        if self._i == self._v:
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
        cost = Guassian_cost(kid_v, parents_v, beta, var, norm=True)
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
        ind = list(vars)
        mins = self._mins(ind)
        intv = self._intv(ind)
        bins = self._bins(ind)
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
        vars, values = sort_v(kid, parents, kid_v, parents_v)
        if vars not in self._pts or (parents not in self._pts and parents!=()):
            return None
        cost = []
        for _values, _parents_v in zip(values, parents_v):
            _values = self._discretize(vars, values)
            _parents_v = self._discretize(parents, _parents_v)
            Pj = self._pts[vars].p(_values)
            Pp = self._pts[parents].p(_parents_v) if parents!=() else 1
            _cost = -log(Pj/Pp)
            cost.append(_cost)
        cost = np.mean(cost)
        return cost


class BN:
    '''
    Bayesian Netowork
    '''
    def __init__(self, fault, obs):
        self.adj = Bayesian_adj(fault, obs)
        self.para = None
        self.type = None

    def set_adj(self, adj):
        self.adj.set_adj(adj)

    def set_type(self, _type, mins=None, intervals=None, bins=None):
        if _type == 'CPT':
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

    def nLogPc_obs(self, fault, obs):
        '''
        This function returns the negative log posteriori probability.
            P(fault|obs) = P(fault, obs)/P(fault)
        According to chain rule, P(fault) is in the chain to compute P(fault, obs).
        So, to compute P(fault|obs), just ignore P(fault) is the chain.
        Args:
            fault: a string, the fault name.
            obs: the values of all monitored variables.
        '''
        nlogP = 0
        f_index = self.adj._fault.index(fault)
        values = np.array([0]*self.adj._v)
        values[f_index] = 1
        for i, o in zip(range(self.adj._f, self.adj._v), obs):
            values[i] = o
        for kid, parents in self.adj:
            if kid[0] < self.adj._f:
                continue # Ignore fault priori probability
            kid_v = values[list(kid)]
            parents_v = values[list(parents)]
            nlogP += self.para.nLogPc(kid, parents, kid_v, parents_v)
        return nlogP

    def save(self, file):
        s = pickle.dumps(BN)
        with open(file, "wb") as f:
            f.write(s)
