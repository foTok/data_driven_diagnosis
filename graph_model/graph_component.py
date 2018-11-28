"""
defines some components used by Bayesian network
"""
import numpy as np

class Bayesian_structure:
    """
    store Bayesian structure
    """
    def __init__(self, n=None):
        """
        n is the number of nodes
        """
        #adjacent matrix
        if n is not None:
            self.struct = np.array([[0] * n] * n)
        else:
            self.struct = None
        #for iteration
        self.n = n
        self.i = 0
        self.skip = False
    
    def set_skip(self):
        """
        skip the first 6 nodes
        """
        self.i = 6
        self.skip = True

    def reset_skip(self):
        """
        reset skip
        """
        self.i = 0
        self.skip = False

    def set_n(self, n):
        """
        set node numbers
        """
        self.n = n
        self.struct = np.array([[0] * n] * n)

    def set_struct(self, struct):
        """
        set the struct
        """
        self.struct = struct
        self.n = len(struct)

    def add_edge(self, i, j):
        """
        set connection from node i to node j
        """
        assert i != j
        self.struct[i, j] = 1
        self.struct[j, i] = 0

    def remove_edge(self, i, j):
        """
        remove edge between node i and node j
        """
        assert i != j
        self.struct[i, j] = 0
        self.struct[j, i] = 0

    def reverse_edge(self, i, j):
        """
        reverse edge between node i and node j
        when there is no connection, nothing happens
        """
        assert i != j
        self.struct[i, j], self.struct[j,i] = self.struct[j, i], self.struct[i,j]

    def get_edge(self, i, j):
        """
        get the connection from node i and node j
        """
        assert i != j
        return self.struct[i, j]

    # def is_DAG_DFS(self, i, color):
    #     """
    #     deep first search
    #     return if it is acycle in this search
    #     True: acycle
    #     False:cycle
    #     """
    #     color[i] = 1
    #     for j in range(self.n):
    #         if self.struct[i, j] != 0:
    #             if color[j] == 1:
    #                 return False
    #             elif color[j] == -1:
    #                 continue
    #             else:
    #                 if not self.is_DAG_DFS(j, color):
    #                     return False
    #     color[i] == -1
    #     return True

    # def is_acycle(self):
    #     """
    #     check if the structure is acycle
    #     """
    #     color = [0]*self.n
    #     for i in range(self.n):
    #         if not self.is_DAG_DFS(i, color):
    #             return False
    #     return True

    def clone(self):
        """
        clone it
        """
        copy = Bayesian_structure()
        copy.struct = self.struct.copy()
        copy.n = self.n
        copy.i = self.i
        copy.skip = self.skip
        return copy

    def __eq__(self, other):
        """
        check if it is equal to other
        """
        return (self.struct == other.struct).all()

    def __hash__(self):
        """
        hash function
        """
        return hash(tuple([tuple(i) for i in self.struct]))

    def __iter__(self):
        """
        enumerate all edeges
        """
        return self

    def __next__(self):
        """
        work with __iter__
        """
        if self.i == self.n:
            self.i = 0 if not self.skip else 6
            raise StopIteration
        parents = list(self.struct[:, self.i])
        parents = [i for i, v in enumerate(parents) if v==1]
        kid = [self.i]
        fml = parents + kid
        self.i = self.i + 1
        return tuple(fml)

class Bayesian_Gaussian_parameter:
    """
    This class is used to store Gaussian parameters for a Bayesian network.
    Different from the common Gaussian parameters in which the variance is a constance.
    This class assumes that both the mean value and the variance value for a Gassian distribution
    are determined by a set of linear real value parameters.
    """
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
        """
        add parameters into self.fml_tank
        """
        self.fml_tank[fml] = parameters

    def add_obs_ass(self, id, value):
        """
        Add a new observation or assumption.
        """
        self.obs_ass[id] = value

    def clear_obs_ass(self):
        """
        clear all the observation and assumption.
        """
        self.obs_ass.clear()
