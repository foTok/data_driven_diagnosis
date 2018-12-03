"""
TAN learning
"""
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
import numpy as np
from math import log
from graph_model.cpt import PT
from graph_model.cpt import discretize

def sort_v(kid, parents, kid_v, parents_v):
    '''
    sort varibles and values
    Args:
        kid: unit tuple, (kid_id,)
        parents:tuple, (p0_id,p1_id,...)
        kid_v: unit tuple, (kid_id_v,)
        parents_v:tuple, (p0_id_v,p1_id_v,...)
    '''
    vars = list(kid) + list(parents)
    values = list(kid_v) + list(parents_v)
    data = np.array([vars, values])
    data = data[:,data[0].argsort()] # sorted by variables
    vars = tuple(data[0])
    values = tuple(data[1])
    return vars, values

def num2vec(num, bins):
    vec = []
    for b in bins:
        v = num % b
        num = num // b
        vec.append(v)
    return vec

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
        self._mins = np.array(mins)
        self._intv = np.array(intervals)
        self._bins = np.array(bins)
        self._pts = {}

    def _lps(self):
        for pt in self._pts:
            pt.lps()
        
    def batch_process(self, batch, jvl):
        '''
        Args:
            batch: a N×n matrix. N is the samples number, \
                n is the variable number.
            jvl: a list of tuples. The joint variable list. \
                Indicate which joint variables should be considered.
        '''
        for jv in jvl: # jv must be a tuple and in an ascending order
            ind = list(jv)
            mins = self._mins(ind)
            intv = self._intv(ind)
            bins = self._bins(ind)
            if jv not in self._pts:
                self._pts[jv] = PT(bins)
            for data in batch:
                d_data = discretize(data, mins, intv, bins)
                self._pts[jv].count(d_data)
        self._lps()

    def fml_cost(self, kid, parents):
        '''
            cost(kid|parents)=H(kid|parengs)
                             = integrate{P(kid,parents)log(1/P(kid|parents))}
                             = integrate{P(kid,parents)log(P(parents)/P(kid,parents))}
        '''
        assert kid in self._pts and (parents in self._pts or parents==())
        vars = tuple(sorted(list(kid) + list(parents)))
        index = vars.index(kid[0])
        assert vars in self._pts
        bins = self._bins[vars]
        n = bins.prod()
        cost = 0
        for i in range(n):
            vec = num2vec(i, bins)
            vars_v = tuple(vec)
            Pj = self._pts[vars].p(vars_v)
            if parents != ():
                vecp = vec[:index] + vec[index+1:]
                par_v = tuple(vecp)
                Pp = self._pts[parents].p(par_v)
            else:
                Pp = 1
            cost += Pj*log(Pp/Pj)
        return cost

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


class Gauss_BS:
    '''
    Gaussian CPD based batch statistics
    '''
    def __init__(self):
        pass