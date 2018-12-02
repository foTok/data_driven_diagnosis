'''
A data structure to store Conditional Probability Table.
This file only construct a data structure to store Probability Table.
For CPT:
    P(kid|parents) = P(parents, kid)/P(parents)
'''

def discretize(continuous, mins, intervals, bins):
    '''
    Args:
        continuous: a list or tuple of continuous values
        mins: a list or tuple of minimal values
        intervals: a list or tuple of intervals
        bins: a list or tuple of discretized numbers
    Return:
        A tuple, discretized.
    '''
    vec = []
    for c, m, i, b in zip(continuous, mins, intervals, bins):
        n = (c-m)/i
        n = 0 if n<1 else min(int(n), b-1)
        vec.append(n)
    return tuple(vec)

class PT:
    '''
    Probability Table
    '''
    def __init__(self, bins):
        '''
        Args:
            bins: a or list tuple of variable discretized numbers.
        '''
        self._bins = bins
        self._count = {}
        self._pt = {} # Probability Table
        self._lps = None # Laplace Smooth Factor

    def count(self, vec):
        '''
        We count a vec
        Args:
            vec: a tuple of ints.
        '''
        assert isinstance(vec, tuple)
        self._count[vec] = (self._count[vec]+1) if vec in self._count else 1

    def lps(self):
        '''
        Laplace Smoothed Probability
        '''
        index_num = 1
        for n in self._bins:
            index_num *= n
        miss = index_num-len(self._count)
        count = sum(self._count.values) + miss
        for i in self._count:
            self._pt[i] = self._count[i]/count
        self._lps = 1/count

    def p(self, vec):
        assert isinstance(vec, tuple)
        if vec not in self._pt:
            return self._lps
        else:
            return self._pt[vec]
