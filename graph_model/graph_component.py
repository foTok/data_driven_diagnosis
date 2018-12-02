'''
defines some components used by Bayesian network
'''
import numpy as np

class Bayesian_structure:
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
        self._adj = np.array([[0]*self._v]*self._v)

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
        copy = Bayesian_structure(self._fault, self._obs)
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
        fml = parents + kid
        self._i += 1
        return tuple(fml)

class Bayesian_Gaussian_parameter:
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

    def add_obs_ass(self, id, value):
        '''
        Add a new observation or assumption.
        '''
        self.obs_ass[id] = value

    def clear_obs_ass(self):
        '''
        clear all the observation and assumption.
        '''
        self.obs_ass.clear()

class Bayesian_CPT:
    '''
    A class to store CPT
    '''
    pass
