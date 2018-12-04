'''
learning Bayesian model
'''
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
import numpy as np
from graph_model.BN import BN
from graph_model.BN import Bayesian_adj
from graph_model.batch_statistic import CPT_BS
from graph_model.batch_statistic import Gauss_BS
from math import log
from math import exp

class GSIN:
    '''
    Greedy search improved naive Beyesian Network
    '''
    def __init__(self, fault, obs):
        self.queue  = {}
        self.type   = None
        self.statistic  = None
        self.mins   = None
        self.intervals  = None
        self.bins   = None
        self.delay  = 0
        self.fault  = fault
        self.obs    = obs  
        self.n  = len(fault) + len(obs)
        self.vars_r_cost_cache  = {}
        self.graph_r_cost_cache = {}

    def init_queue(self):
        '''
        init the search queue
        '''
        adj = Bayesian_adj(self.fault, self.obs)
        adj.naive_init()
        cost = float('inf')
        self.queue[adj] = cost

    def best_candidate(self):
        '''
        get the best candidate
        '''
        queue = self.queue
        graph, cost = min(queue.items(), key=lambda x:x[1])
        return graph, cost

    def set_type(self, _type, mins=None, intervals=None, bins=None):
        
        if _type == 'CPT':
            assert mins is not None and intervals is not None and bins is not None
            self.mins   = np.array(mins)
            self.intervals  = np.array(intervals)
            self.bins   = np.array(bins)
            self.statistic  = CPT_BS(self.mins, self.intervals, self.bins)
        elif _type == 'GAU':
            self.statistic  = Gauss_BS()
        else:
            raise RuntimeError('unknown parameter type')
        self.type   = _type

    def l_cost(self, adj):
        '''
        compute the likelihood cost
        '''
        cost = 0
        for kid, parents in adj:
            cost_u = self.statistic.fml_cost(kid, parents)
            cost += cost_u
        return cost

    def r_cost(self, adj):
        '''
        compute the regular cost for graph
        '''
        if adj in self.graph_r_cost_cache:
            return self.graph_r_cost_cache[adj]
        #Now we know the regular cost is not computed for the adj
        cost = 0
        for kid, parents in adj:
            vars = tuple(sorted(list(parents) + list(kid)))
            if vars in self.vars_r_cost_cache:
                cost_u = self.vars_r_cost_cache[vars]
            else:#Now we know the fml is not cached
                bins = self.bins[vars]
                cost_u = bins.prod()
                cost_u = log(cost_u)
                self.vars_r_cost_cache[vars] = cost_u
            cost += cost_u
        return cost

    def cost(self, adj):
        '''
        compute the cost of a structure
        '''
        l_cost = self.l_cost(adj)
        r_cost = self.r_cost(adj)
        cost = l_cost + self.delay * r_cost
        return cost

    def step(self, batch, time_step=None):
        '''
        step forward
        '''
        if not self.queue:
            print("Empty queue. Break")
            return
        best, cost = self.best_candidate()
        if time_step is not None:
            print("cost ",time_step, " = ", cost)
        else:
            print(cost)
        self.queue.clear()
        self.statistic.set_batch(batch)
        n = len(batch)
        self.decay = log(n)/(2*n)
        # Update best
        cost = self.cost(best)
        self.queue[best] = cost
        #change randomly
        for i in range(self.n):
            for j in range(self.n):
                if not best.available(i, j):
                    continue
                # add
                if best.add_perm(i, j):
                    add_best = best.clone()
                    add_best.add_edge(i, j)
                    add_cost = self.cost(add_best)
                    self.queue[add_best] = add_cost
                # remove
                if best.remove_prem(i, j):
                    rem_best = best.clone()
                    rem_best.remove_edge(i, j)
                    rem_cost = self.cost(rem_best)
                    self.queue[rem_best] = rem_cost
                # reverse
                if best.reverse_prem(i, j):
                    rev_best = best.clone()
                    rev_best.reverse_edge(i, j)
                    rev_cost = self.cost(rev_best)
                    self.queue[rev_best] = rev_cost

    def best_BN(self):
        '''
        best BN
        '''
        best_adj, _ = self.best_candidate()
        bBN = BN(self.fault, self.obs)
        bBN.set_type(self.type)
        bBN.set_adj(best_adj)
        for kid, parents in best_adj:
            if self.type == 'CPT':
                vars = tuple(sorted(list(parents) + list(kid)))
                bBN.add_para(vars, self.statistic.para(vars))
                bBN.add_para(parents, self.statistic.para(parents))
            elif self.type == 'GAU':
                fml = tuple(list(parents) + list(kid))
                bBN.add_para(fml, self.statistic.para(fml))
            else:
                raise RuntimeError('Unknown type')
        return bBN
