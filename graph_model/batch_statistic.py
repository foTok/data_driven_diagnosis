"""
TAN learning
"""
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
import numpy as np
from scipy.stats import chi2
from math import log
from math import exp
from graph_model.cpt import PT
from graph_model.utilities import discretize
from graph_model.utilities import sort_v
from graph_model.utilities import num2vec
from graph_model.utilities import Guassian_cost
from graph_model.utilities import Guassian_cost_core


class CPT_BS:
    '''
    Conditional Probability Table based batch statistics
    batch_process: it analyzes data in a batch based on considered groups.
        There are two groups in a family: kid->parents:
            kid ∪ parents and parents
    '''
    def __init__(self, mins, intervals, bins):
        '''
        Each variable is assigned an int id accordingly.
        Args:
            mins: a list or tuple with length=n. The minimal values\
                    for variables.
            intervals: a list or tuple with length=n. The intervals \
                    for variables.
            bins: a list or tuple with lenght=n. The intervel number \
                    for variables.
        E.G.
            The interval for one variable:
                (-inf, min+interval, min+2*interval,...,min+bin*(interval-1),inf)
        '''
        self._mins  = np.array(mins)
        self._intv  = np.array(intervals)
        self._bins  = np.array(bins)
        self._pts   = {}
        self._batch = None
        # Cache
        self._cache_join    = set() # for vars, used in both cost and distance
        # Cache for computing cost or distance
        self._cache_value   = {}  

    def _lps(self):
        for pt in self._pts:
            self._pts[pt].lps()

    def _discretize(self, vars, values):
        ind = list(vars)
        mins = self._mins[ind]
        intv = self._intv[ind]
        bins = self._bins[ind]
        d_values = discretize(values, mins, intv, bins)
        return d_values

    def _pick_value(self, vars, values, subvars):
        '''
        Pick out values of subvars
        Args:
            vars:   all variables
            values: all values
            subvars:    the subvars whose values will be picked out
        Returns:
            a list containing all the picked values.
        '''
        index = [vars.index(i) for i in subvars]
        subvalues = [values[i] for i in index]
        return subvalues

    def _join_statistic(self, jv):
        if jv in self._cache_join or jv==():
            return
        ind = list(jv)
        bins = self._bins[ind]
        if jv not in self._pts:
            self._pts[jv] = PT(bins)
        _batch = self._batch[:, ind]
        for data in _batch:
            d_data = tuple(data)
            self._pts[jv].count(d_data)
        self._cache_join.add(jv)    # cache

    def _batch_process(self, kid, parents, y=None):
        '''
        Args:
            if y is None:
                parents ->  kid
                kid: the tuple of kid variable.
                parents: the tuple of parents variables.
            if y is not None:
                kid(x)  <-  parents(Z)  ->  y
                kid: x in the distance computation
                parents: Z in the distance computation
                y: y in the distance computation
        '''
        key = tuple(list(parents) + list(kid)) if y is None else tuple(list(kid)+list(parents)+list(y))
        if key in self._cache_value:
            return
        if y is None:
            vars = tuple(sorted(list(kid) + list(parents)))
            jvl = [parents, vars]
        else:
            xyZ = tuple(sorted(list(kid) + list(parents) + list(y)))
            xZ  = tuple(sorted(list(kid) + list(parents)))
            yZ  = tuple(sorted(list(parents) + list(y)))
            jvl = [parents, xyZ, xZ, yZ]
        for jv in jvl: # jv must be a tuple and in an ascending order
            self._join_statistic(jv)
        self._lps()

    def _d_chi_square(self, x, y, Z):
        '''
        The distance that obeys Chi square distribution.
        Args:
            x: variable x, an unit tuple (x,)
            y: variable x, an unit tuple (y,)
            Z: their possible parents, a tuple (z0, z1, ...), in ascending order.
        Returns:
            1. The distance defined by Koller.
                M×Sigma{[P(Z)×P(x,y,Z)-P(x,Z)×P(y,Z)]**2/[P(Z)×P(x,Z)×P(y,Z)]}
                M is the length of batch
                Obeys:
                    Chi_square(||x,y,Z||-1)
            2. ||x,y,Z||-1
        '''
        key = tuple(list(x)+list(Z)+list(y))
        if key in self._cache_value:
            return self._cache_value[key]
        self._batch_process(x, Z, y)
        xyZ = tuple(sorted(list(x) + list(Z) + list(y)))
        xZ  = tuple(sorted(list(x) + list(Z)))
        yZ  = tuple(sorted(list(Z) + list(y)))
        bins = self._bins[list(xyZ)]
        n = bins.prod()
        dis = 0
        for i in range(n):
            values  = num2vec(i, bins)
            # P(Z)
            if Z != ():
                Z_v = self._pick_value(xyZ, values, Z)
                PZ  = self._pts[Z].p(Z_v)
            else:
                PZ = 1
            # P(x,Z)
            xZ_v    = self._pick_value(xyZ, values, xZ)
            PxZ = self._pts[xZ].p(xZ_v)
            # P(y,Z)
            yZ_v    = self._pick_value(xyZ, values, yZ)
            PyZ = self._pts[yZ].p(yZ_v)
            # P(x,y,Z)
            PxyZ = self._pts[xyZ].p(values)
            # dis
            dis_i = (PZ*PxyZ-PxZ*PyZ)**2/(PZ*PxZ*PyZ) # due to Laplace smooth, we don't have to add a small number
            dis += dis_i
        dis *= len(self._batch)
        self._cache_value[key] = dis
        return dis, n-1

    def para(self, vars):
        if vars not in self._pts:
            return None
        return self._pts[vars]

    def set_batch(self, batch):
        '''
        Args:
            batch: a 2d np.array
        '''
        _, n = batch.shape  # n is the number of all variables
        vars = [i for i in range(n)]
        batch = self._discretize(vars, batch)
        self._batch = batch
        self._cache_join.clear()
        self._cache_value.clear()

    def fml_cost(self, kid, parents):
        '''
        Args:
            kid: an unit tuple
            parents: a tuple
            cost(kid|parents)=H(kid|parengs)
                             = integrate{P(kid,parents)log(1/P(kid|parents))}
                             = integrate{P(kid,parents)log(P(parents)/P(kid,parents))}
        '''
        fml = tuple(list(parents) + list(kid))
        if fml in self._cache_value:
            return self._cache_value[fml]
        self._batch_process(kid, parents)
        vars = tuple(sorted(list(kid) + list(parents)))
        index = vars.index(kid[0])
        assert vars in self._pts
        bins = self._bins[list(vars)]
        n = bins.prod()
        cost = 0
        for i in range(n):
            vec = num2vec(i, bins)
            Pj = self._pts[vars].p(vec)
            if parents != ():
                vecp = vec[:index] + vec[index+1:]
                Pp = self._pts[parents].p(vecp)
            else:
                Pp = 1
            cost += Pj*log(Pp/Pj)
        self._cache_value[fml] = cost
        return cost

    def independent(self, x, y, Z, alpha=0.95):
        '''
        Test if x and y are independent given Z.
        Args:
            x, y: unit tuples
            Z: tuple
            alpha: confidence, 0.95 by default
        Returns:
            if independent {True, False}
        '''
        chi_statistic, df = self._d_chi_square(x, y, Z)
        threshold = chi2.ppf(alpha, df)
        return chi_statistic <= threshold

    def nLogPc(self, kid, parents, kid_v, parents_v):
        '''
        Negative Log Conditional Probability
        Args:
            kid: unit tuple, (kid_id,)
            parents:tuple, (p0_id,p1_id,...)
            kid_v: np.array
            parents_v: np.array
        Returns:
            P(kid|parents) = P(parents, kid)/P(parents)
            or None
        '''
        vars, values = sort_v(kid, parents, kid_v, parents_v)
        if vars not in self._pts or (parents not in self._pts and parents!=()):
            return None
        values = self._discretize(vars, values)
        parents_v = None if parents==() else self._discretize(parents, parents_v)
        cost = []
        for _values, _parents_v in zip(values, parents_v):
            Pj = self._pts[vars].p(_values)
            Pp = self._pts[parents].p(_parents_v) if parents!=() else 1
            _cost = -log(Pj/Pp)
            cost.append(_cost)
        cost = np.mean(cost)
        return cost


class Gauss_BS:
    '''
    Gaussian CPD based batch statistics
    '''
    def __init__(self):
        #cache for GGM (Guassian Graph Model).{FML:[beta, var, N]}.
        #FML:(p1,p2,...pn, kid), nodes are put in ascending order.
        #beta:[beta0, beta1,...,beta_n].
        #var: [var0, var1,...,var_n].
        #N: real (weight), N is the batch number where the batchs contribute the JPT.
        #Some values in it will be updated in each iteration but the dictory will NOT be cleared.
        self._GGM_cache              = {}
        #batch GGM cache. Should BE cleared and updated in each iteration.
        #NOTICE: This set just store the GGM that computed from batch.
        #the real GGM is merged into self.GGM_cache.
        #It should BE cleared before each iteration.
        self._batch_GGM_cache        = set()                                                            #remember to clear
        #cache pesudo inverse of X
        #{X:var}
        self._p_inverse_cache        = {}                                                               #remember to clear
        self._cost_cache = {}                                                                           #remember to clear
        self._batch = None
    
    def _batch_process(self, kid, parents):
        '''
        Args:
            kid: the tuple of kid variable.
            parents: the tuple of parents variables.
        '''
        fml = tuple(list(parents) + list(kid))
        if fml in self._batch_GGM_cache:
            return
        beta1, var1 = self._GGM_from_batch(kid, parents)
        if fml in self._GGM_cache:
            bvw0 = self._GGM_cache[fml]
            beta0, var0, n = bvw0[0], bvw0[1], bvw0[2]
            w0 = n / (1 + n)
            w1 = 1 / (1 + n)
            beta = w0 * beta0 + w1 * beta1
            var  = w0 * var0 + w1 * var1
            n += 1
            bvw = [beta, var, n]
        else:
            bvw = [beta1, var1, 1]
        # cache
        self._batch_GGM_cache.add(fml)      # updated flag
        self._GGM_cache[fml] = bvw          # cached value

    def _GGM_from_batch(self, kid, parents):
        '''
        Args:
            kid: the tuple of kid variable.
            parents: the tuple of parents variables.
        '''
        x = parents
        y = kid[0]
        N = len(self._batch)
        e = np.ones((N, 1))
        X = np.hstack((e, self._batch[:, x]))
        X = np.mat(X)
        Y = self._batch[:, y]
        Y.shape = (N, 1)
        p_inv  =self._p_inverse(x)
        beta = p_inv * Y
        res = (Y - X*beta)
        var = p_inv * np.multiply(res, res)
        #avoid numeric problems
        var = var + 1e-8
        return beta, var

    def _p_inverse(self, x):
        '''
        get p_inverse
        '''
        if x in self._p_inverse_cache:
            p_inv = self._p_inverse_cache[x]
        else:
            p_inv = self._p_inverse_from_batch(x)
            self._p_inverse_cache[x] = p_inv             #cache it
        return p_inv

    def _p_inverse_from_batch(self, x):
        '''
        get p_inverse_from_batch
        x: a tuple
        !!!Please make sure x are listed in increasing order
        '''
        N = len(self._batch)
        e = np.ones((N, 1))
        X = np.hstack((e, self._batch[:, x]))
        X = np.mat(X)
        p_inv = (X.T * X).I * X.T
        return p_inv

    def para(self, fml):
        if fml not in self._GGM_cache:
            return None
        else:
            return self._GGM_cache[fml]

    def set_batch(self, batch):
        self._batch = batch
        self._batch_GGM_cache.clear()
        self._p_inverse_cache.clear()
        self._cost_cache.clear()

    def fml_cost(self, kid, parents):
        fml = tuple(list(parents) + list(kid))
        if fml in self._cost_cache:
            return self._cost_cache[fml]
        self._batch_process(kid, parents)
        assert fml in self._GGM_cache
        beta, var, _ = self._GGM_cache[fml]
        cost = Guassian_cost(self._batch, fml, beta, var, norm=False)
        self._cost_cache[fml] = cost    # cache
        return cost

    def nLogPc(self, kid, parents, kid_v, parents_v):
        '''
        Negative Log Conditional Probability
        Args:
            kid: unit tuple, (kid_id,)
            parents:tuple, (p0_id,p1_id,...)
            kid_v: np array, Y. np.array([[y0],[y1],[y2]...])
            parents_v: np array, X
        Returns:
            -log(P(kid|parents))
            or None
        '''
        fml = tuple(list(parents) + list(kid))
        assert fml in self._GGM_cache
        beta, var, _ = self._GGM_cache[fml]
        kid_v = kid_v.reshape(-1)
        cost = Guassian_cost(kid_v, parents_v, beta, var, norm=True)
        return cost
