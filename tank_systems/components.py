'''
The file defines models of some components.
tank:
                +-------------+ q_i
                +-----------+ |
                            | |
                            | |
                      +     + +  +
                      |      +   |
                      |      |   |
                      |      v   | +------>
                      |          | | h_o
                      |          | +
                      |          |
                      |          |
                      |          |
            +---------+          +----------+
 q_i0+----> +-------------------------------+  <----+ q_i1

    input: q_i0=0, q_i1=0, q_i=0
    output: h_o
    state: h_o
    parameters: g, gravity constant; leakage, fault parameter for tank;
                A, sectional area of the tank.
    behavior:
            dh_o/dt = (q_i + q_i0 + q_i1 - q_f)/A
            q_f = leakage*sqrt(2*g*h_o)

pip:
            h_i0                                  h_i1
            +--->  +-----------------------+  <----+

            <---+  +-----------------------+  +---->
            q_i0                                  q_i1

      input: h_i0, h_i1
      output: q_i0, q_i1
      parameters: S, sectional area of the pip; stuck, fault parameters for pip.
      behavior:
            q_i0 = (1-stuck)*S*sign(h_i1 - h_i0)*sqrt(2*g*abs(h_i1 - h_i0))
            q_i1 = -q_i0
'''
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
import numpy as np


class tank:
    '''
    The tank model.
    '''
    g   = 9.8 # gravity constant
    def __init__(self, A):
        self.h_o    = 0
        self.A  = A
        self.leakage    = 0

        self.h  = []
        self.f  = []
        self.x  = [] # time axis

    def set_para(self, value, name='leakage'):
        '''
        set parameter values
        '''
        if name == 'leakage':
            self.leakage = value
        else:
            raise RuntimeError('Unkown parameter.')

    def step(self, q_i0=0, q_i1=0, q_i=0, time_step=1.0):
        '''
        accept input and step
        '''
        q_f = self.leakage*np.sqrt(2*tank.g*self.h_o)
        delta = time_step*(q_i + q_i0 + q_i1 - q_f)/self.A
        self.h_o += delta
        # save state and data
        self.h.append(self.h_o)
        self.f.append(self.leakage)
        self.x.append((0 if not self.x else self.x[-1]) + time_step)

    def output(self):
        return self.h_o


class pip:
    '''
    The pip model.
    '''
    g = 9.8
    def __init__(self, S):
        self.q_i0   = 0
        self.q_i1   = 0
        self.S  = S
        self.stuck  = 0

        self.q  = [] # q_i1
        self.f  = []
        self.x  = [] # time axis

    def set_para(self, value, name):
        if name == 'stuck':
            self.stuck = value
        else:
            raise RuntimeError('Unkown parameter.')

    def step(self, h_i0, h_i1, time_step=1.0):
        self.q_i0 = (1-self.stuck)*self.S*(1 if h_i1 > h_i0 else -1)*np.sqrt(2*pip.g*abs(h_i1 - h_i0))
        self.q_i1 = -self.q_i0
        # save data
        self.q.append(self.q_i1)
        self.f.append(self.stuck)
        self.x.append((0 if not self.x else self.x[-1]) + time_step)
        
    def output(self):
        return (self.q_i0, self.q_i1)


class multi_tank:
    '''
    multi-tank systems.
    '''
    def __init__(self, n, A, S):
        self.n  = n # tank number
        self.A  = A
        self.S  = S
        self.tanks  = []
        self.pips   = []

        for _ in range(self.n): # build n tanks
            t = tank(A)
            self.tanks.append(t)

        for _ in range(self.n - 1):
            p = pip(S)
            self.pips.append(p)

    def step(self, q, time_step=1.0):
        # obtain pip outputs
        pip_outs = []
        for i in range(self.n-1):
            h_i0 = self.tanks[i].output()
            h_i1   = self.tanks[i+1].output()
            self.pips[i].step(h_i0, h_i1, time_step)
            q   = self.pips[i].output()
            pip_outs.append(q)
        # step tanks
        for i in range(self.n):
            if i==0:
                pass # TODO
            elif i==(self.n-1):
                pass
            else:
                _, q_i0 = pip_outs[i-1]
                q_i1, _ = pip_outs[i]
                self.tanks[i].step(q_i0, q_i1, 0, time_step)





