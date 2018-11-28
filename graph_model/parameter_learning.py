"""
define a class to learn Bayesian parameters from batch
"""

import numpy as np

class Parameters_learning:
    """
    The core class
    """
    def __init__(self):
        #batch
        #np.array(). Data for the current batch, batch × (label + residuals + features).
        self.batch                  = None
        #cache pesudo inverse of X
        #{X:var}
        self.p_inverse_cache        = {}                #remember to clear

    def set_batch(self, batch):
        """
        set the current data
        """
        self.batch = batch

    def reset(self):
        """
        reset batch and cache
        """
        self.batch = None
        self.p_inverse_cache.clear()

    def get_p_inverse(self, x):
        """
        get the pesude inverse
        """
        if x in self.p_inverse_cache:
            p_inv = self.p_inverse_cache[x]
        else:
            p_inv = self.p_inverse_from_batch(x)
            self.p_inverse_cache[x] = p_inv             #cache it
        return p_inv

    def p_inverse_from_batch(self, x):
        """
        get p_inverse_from_batch
        x: a tuple
        !!!Please make sure x are listed in increasing order
        """
        N = len(self.batch)
        e = np.ones((N, 1))
        X = np.hstack((e, self.batch[:, x]))
        X = np.mat(X)
        p_inv = (X.T * X).I * X.T
        return p_inv

    def GGM_from_batch(self, fml, cv):
        """
        compute GGM for FML in this batch
        !!!Please make sure parents are listed in increasing order
        """
        x = fml[:-1]  #a tuple
        y = fml[-1]   #an int
        N = len(self.batch)
        e = np.ones((N, 1))
        X = np.hstack((e, self.batch[:, x]))
        X = np.mat(X)
        Y = self.batch[:, y]
        Y.shape = (N, 1)
        p_inv  =self.get_p_inverse(x)
        beta = p_inv * Y
        res = (Y - X*beta)
        if cv:
            var = np.mean(np.multiply(res, res))
        else:
            var = p_inv * np.multiply(res, res)
        #avoid numeric problems
        var = var + 1e-8
        return beta, var
