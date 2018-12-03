'''
defines some components used by Bayesian network
'''
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir) 
import numpy as np
from graph_model.cpt import PT
from graph_model.batch_statistic import sort_v

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

    def empty_init(self):
        self._adj = np.array([[0]*self._v]*self._v)

    def naive_init(self):
        self._adj = np.array([[0]*self._v]*self._v)
        for i in range(self._f):
            for j in range(self._f, self._v):
                self._adj[i, j] = 1

    def set_adj(self, adj):
        self._adj = adj

    def add_edge(self, i, j):
        assert isinstance(i, int) and isinstance(j, int)
        assert 0<=i<self._v and self._f<=j<self._v and i!=j
        if i<self._f: # fault->obs
            if self._adj[i,j]==0:
                self._adj[i,j]=1
                return True
        else: # obs->obs
            if self._adj[i,j]==0 and self._adj[j,i]==0 and not self.has_path(j, i):
                self._adj[i,j]=1
                return True
        return False

    def remove_edge(self, i, j):
        assert isinstance(i, int) and isinstance(j, int)
        assert 0<=i<self._v and self._f<=j<self._v and i!=j
        if i<self._f: # fault->obs
            if self._adj[i,j]==1:
                self._adj[i,j]=0
                return True
        else: # obs->obs
            if self._adj[i,j]==1:
                self._adj[i,j]=0
                return True
        return False

    def reverse_edge(self, i, j):
        assert isinstance(i, int) and isinstance(j, int)
        assert 0<=i<self._v and self._f<=j<self._v and i!=j
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
        #Observed or assumed values.
        #This set is used to store the currently observed values or assumed values.
        #KEY ID, real value number.
        #VALUE, NUM, real value number.
        self.obs_ass    = {}

    def add_fml(self, fml, parameters):
        '''
        add parameters into self.fml_tank
        '''
        self.fml_tank[fml] = parameters

    def Pc(self, kid, parents, kid_v, parents_v):
        '''
        Conditional Probability
        Args:
            kid: unit tuple, (kid_id,)
            parents:tuple, (p0_id,p1_id,...)
            kid_v: unit tuple, (kid_id_v,)
            parents_v:tuple, (p0_id_v,p1_id_v,...)
        Returns:
            P(kid|parents) = P(parents, kid)/P(parents)
            or None
        '''
        pass

    def Pj(self, kid, parents, kid_v, parents_v):
        '''
        Joint Probability
        Args:
            kid: unit tuple, (kid_id,)
            parents:tuple, (p0_id,p1_id,...)
            kid_v: unit tuple, (kid_id_v,)
            parents_v:tuple, (p0_id_v,p1_id_v,...)
        Returns:
            P(kid|parents) = P(parents, kid)/P(parents)

            or None
        '''
        pass

class CPT:
    '''
    A class to store CPT
    '''
    def __init__(self):
        self._pts = {}

    def add_pt(self, vars, pt):
        self._pts[vars] = pt

    def Pc(self, kid, parents, kid_v, parents_v):
        '''
        Conditional Probability
        Args:
            kid: unit tuple, (kid_id,)
            parents:tuple, (p0_id,p1_id,...)
            kid_v: unit tuple, (kid_id_v,)
            parents_v:tuple, (p0_id_v,p1_id_v,...)
        Returns:
            P(kid|parents) = P(parents, kid)/P(parents)
            or None
        '''
        vars, values = sort_v(kid, parents, kid_v, parents_v)
        if vars not in self._pts or (parents not in self._pts and parents!=()):
            return None
        Pj = self._pts[vars].p(values)
        Pp = self._pts[parents].p(parents_v) if parents!=() else 1
        return Pj/Pp

    def Pj(self, kid, parents, kid_v, parents_v):
        '''
        Joint Probability
        Args:
            kid: unit tuple, (kid_id,)
            parents:tuple, (p0_id,p1_id,...)
            kid_v: unit tuple, (kid_id_v,)
            parents_v:tuple, (p0_id_v,p1_id_v,...)
        Returns:
            P(kid|parents) = P(parents, kid)/P(parents)

            or None
        '''
        vars, values = sort_v(kid, parents, kid_v, parents_v)
        if vars not in self._pts:
            return None
        return self._pts[vars].p(values)


class BN:
    '''
    Bayesian Netowork
    '''
    def __init__(self, fault, obs):
        self.adj = Bayesian_adj(fault, obs)

    def set_adj(self, adj):
        self.adj.set_adj(adj)

class CPT_BN(BN):
    """
    CPT based Bayesian Network
    """
    def __init__(self, fault, obs):
        super(CPT_BN).__init__(fault, obs)
        self.para = CPT()

    def add_para(self, vars, pt):
        self.para.add_pt(vars, pt)

    def Pc(self, kid, parents, kid_v, parents_v):
        '''
        Conditional Probability
        Args:
            kid: unit tuple, (kid_id,)
            parents:tuple, (p0_id,p1_id,...)
            kid_v: unit tuple, (kid_id_v,)
            parents_v:tuple, (p0_id_v,p1_id_v,...)
        Returns:
            P(kid|parents) = P(parents, kid)/P(parents)
            or None
        '''
        return self.para.Pc(kid, parents, kid_v, parents_v)

    def Pj(self, kid, parents, kid_v, parents_v):
        '''
        Joint Probability
        Args:
            kid: unit tuple, (kid_id,)
            parents:tuple, (p0_id,p1_id,...)
            kid_v: unit tuple, (kid_id_v,)
            parents_v:tuple, (p0_id_v,p1_id_v,...)
        Returns:
            P(kid|parents) = P(parents, kid)/P(parents)
            or None
        '''
        return self.para.Pj(kid, parents, kid_v, parents_v)

    def Pc_obs(self, fault, obs):
        '''
        This function returns the posteriori probability.
            P(fault|obs) = P(fault, obs)/P(fault)
        According to chain rule, P(fault) is in the chain to compute P(fault, obs).
        So, to compute P(fault|obs), just ignore P(fault) is the chain.
        Args:
            fault: a string, the fault name.
            obs: the values of all monitored variables.
        '''
        P = 1
        f_index = self.adj._fault.index(fault)
        values = np.array([0]*self.adj._v)
        values[f_index] = 1
        for i, o in zip(range(self.adj._f, self.adj._v), obs):
            values[i] = o
        for kid, parents in self.adj:
            if kid[0] < self.adj._f:
                continue # Ignore fault priori probability
            kid_v = tuple(values[list(kid)])
            parents_v = tuple(values[list(parents)])
            P *= self.Pc(kid, parents, kid_v, parents_v)
        return P
