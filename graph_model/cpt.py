'''
A data structure to store Conditional Probability Table.
This file only construct a data structure to store Probability Table.
For CPT:
    P(kid|parents) = P(parents, kid)/P(parents)
'''
class PT:
    '''
    Probability Table
    '''
    def __init__(self, bins):
        '''
        Args:
            bins: a or list tuple of variable discretized numbers.
        '''
        self._bins  = bins
        self._count = {}
        self._pt    = {} # Probability Table
        self._lps   = None # Laplace Smooth Factor
        self._in    = 1 # index number
        for n in self._bins:
            self._in *= n

    def count(self, vec, num=1):
        '''
        We count a vec
        Args:
            vec: a tuple of ints.
            num: count number, 1 by default.
        '''
        self._count[vec] = (self._count[vec]+1) if vec in self._count else num

    def lps(self):
        '''
        Laplace Smoothed Probability
        '''
        miss = self._in-len(self._count)
        count = sum(self._count.values()) + miss
        for i in self._count:
            self._pt[i] = self._count[i]/count
        self._lps = 1/count if miss!=0 else None

    def p(self, vec):
        '''
        Args:
            vec: a tuple of ints.
        '''
        if vec not in self._pt:
            return self._lps
        else:
            return self._pt[vec]
