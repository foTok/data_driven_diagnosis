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
import matplotlib.pyplot as plt

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
        self.h_o = self.h_o if self.h_o > 0 else 0
        # save state and data
        self.h.append(self.h_o)
        self.f.append(self.leakage)
        self.x.append((0 if not self.x else self.x[-1]) + time_step)

    def output(self):
        return self.h_o

    def trajectory(self, x=True):
        if x:
            return (tuple(self.h), tuple(self.f), tuple(self.x))
        else:
            return (tuple(self.h), tuple(self.f))

    def reset(self):
        self.h_o    = 0
        self.leakage    = 0
        self.h.clear()
        self.f.clear()
        self.x.clear()


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

    def trajectory(self, x=True):
        if x:
            return (tuple(self.q), tuple(self.f), tuple(self.x))
        else:
            return (tuple(self.q), tuple(self.f))

    def reset(self):
        self.q_i0   = 0
        self.q_i1   = 0
        self.stuck  = 0
        self.q.clear()
        self.f.clear()
        self.x.clear()


class multi_tank:
    '''
    multi-tank systems.
    '''
    def __init__(self, n, A, S):
        self.n  = n # tank number
        self.A  = A
        self.S  = S
        self.q  = []    # input
        self.x  = []    # time axis
        self.tanks  = []
        self.pips   = []

        for _ in range(self.n): # build n tanks
            t = tank(A)
            self.tanks.append(t)

        for _ in range(self.n):
            p = pip(S)
            self.pips.append(p)

    def step(self, q, time_step=1.0):
        # obtain pip outputs
        self.q.append(q)
        pip_outs = []
        for i in range(self.n):
            h_i0 = self.tanks[i].output()
            h_i1   = self.tanks[i+1].output() if i!=(self.n-1) else 0
            self.pips[i].step(h_i0, h_i1, time_step)
            _q   = self.pips[i].output()
            pip_outs.append(_q)
        # step tanks
        for i in range(self.n):
            q_i = q if i==0 else 0
            _, q_i0 = pip_outs[i-1] if i!=0 else (0, 0)
            q_i1, _ = pip_outs[i]
            self.tanks[i].step(q_i0, q_i1, q_i, time_step)
        self.x.append((0 if not self.x else self.x[-1]) + time_step)

    def output(self):
        h_t = [0]*self.n        # height of tanks
        q_p = [0]*(self.n)    # q of pips
        for i in range(self.n):
            h_t[i]  = self.tanks[i].output()
        for i in range(self.n):
            _, q_p[i]  = self.pips[i].output()
        return (tuple(h_t), tuple(q_p))

    def trajectory(self, x=True):
        h_t = [0]*self.n        # height of tanks
        q_p = [0]*self.n        # q of pips
        for i in range(self.n):
            h_t[i]  = self.tanks[i].trajectory(x=False)
        for i in range(self.n):
            q_p[i]  = self.pips[i].trajectory(x=False)
        if x:
            return (tuple(h_t), tuple(q_p), tuple(self.x))
        else:
            return (tuple(h_t), tuple(q_p))

    def plot_trajectory(self, trajectory, name, id, view_fault=False):
        '''
        plot the trajectory based on name and id
        '''
        assert isinstance(id, int)
        assert 0<= id < self.n
        if name == 'tank':
            data    = trajectory[0]
            y0label = 'Tank {} Height'.format(id)
            y1label = 'Tank {} Leakage Value'.format(id)
        elif name == 'pip':
            data    = trajectory[1]
            y0label = 'Pip {} Flow'.format(id)
            y1label = 'Pip {} Stuck Value'.format(id)
        else:
            raise RuntimeError('Unknonwn component.')
        data_t  = trajectory[2]
        data_s, data_f = data[id]
        plt.figure()
        plt.plot(data_t, data_s)
        plt.xlabel('Time (s)')
        plt.ylabel(y0label)
        if view_fault:
            plt.figure()
            plt.plot(data_t, data_f)
            plt.xlabel('Time (s)')
            plt.ylabel(y1label)
        plt.show()

    def plot_qi(self):
        plt.figure()
        plt.plot(self.x, self.q)
        plt.xlabel('Time (s)')
        plt.ylabel('Qin')
        plt.show()

    def run(self, q, h0, h1, step_len, time_step=1.0, fault=None):
        '''
        When the height of the first tank achieves h1, stop input; when decrease to h0, start input.
        simulate time: step_len*time_step
        fault: None, no fault. Or (f_name, f_id, f_para, f_time)
        '''
        open    = True
        if fault is not None:
            f_name, f_id, f_para, f_time = fault
            assert isinstance(f_id, int)
            assert 0<= f_id < self.n
            insert_fault    = False
        else:
            insert_fault    = True  # When insert_fault is True, do not need to insert fault again.
        for i in range(step_len):
            open = True if (open and self.tanks[0].output() < h1) or (not open and self.tanks[0].output() < h0) else False
            q_i = q if open else 0.0
            t = i*time_step
            if not insert_fault and t > f_time:
                if f_name == 'tank':
                    component   = self.tanks[f_id]
                    name    = 'leakage'
                elif f_name == 'pip':
                    component   = self.pips[f_id]
                    name    = 'stuck'
                else:
                    raise RuntimeError('Unknown component.')
                component.set_para(f_para, name)
                insert_fault    = True
            self.step(q_i, time_step)
    
    def reset(self):
        self.q.clear()
        self.x.clear()
        for t in self.tanks:
            t.reset()
        for p in self.pips:
            p.reset()

    def np_trajectory(self):
        data = [0]*(2*self.n)
        for i in range(self.n):
            data[i]  = self.tanks[i].trajectory(x=False)[0]
            data[self.n+i]  = self.pips[i].trajectory(x=False)[0]
        return np.array(data)
