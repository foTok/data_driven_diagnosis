'''
Simulate RO system.
'''
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
import numpy as np
import matplotlib.pyplot as plt

class RO:
    # parameters
    I_fp    = 0.1 # N*s^2/m^5
    I_rp    = 2.0 # N*s^2/m^5
    R_fp    = 0.1 # N/m^5
    R_rp    = 0.1 # N/m^5
    C_k     = 565.0 # m^5/N
    R_forward   = 70.0 # N/m^5
    C_tr    = 1.5 # m^5/N
    R_return_l  = 15.0 # N/m^5
    R_return_s  = 8.0 # N/m^5
    R_return_AES    = 5.0 # N/m^5
    C_memb  = 0.6 # m^5/N
    C_brine = 8.0 # m^5/N
    p_fp    = 1.0 # N/m^2
    p_rp    = 160.0 # N/m^2
    # states
    states  = ['q_fp', 'p_tr', 'q_rp', 'p_memb', 'e_Cbrine', 'e_Ck']
    # obs
    obs = {'y1':'q_fp', 'y2':'p_memb', 'y3':'q_fp', 'y4':'e_Cbrine', 'y5':'e_Ck'}

    def __init__(self, step_len=1.0):
        self.step_len   = step_len
        # fault parameters
        self.f_f    = 0
        self.f_r    = 0
        self.f_m    = 0
        # discrete modes
        self.delta1 = None
        self.delta2 = None
        # continous states
        self.q_fp   = 0
        self.p_tr   = 0
        self.q_rp   = 0
        self.d_q_rp = 0 # the delegation for q_rp
        self.p_memb = 0
        self.e_Cbrine   = 0
        self.e_Ck   = 0
        # trajectory
        self.modes  = []
        self.states = []
        self.para_faults    = []
        self.x  = []

    def run(self, t):
        i = 1
        while i*self.step_len < t:
            i += 1
            self.mode_step()
            self.state_step(self.step_len)

    def mode_step(self):
        h0  = 1.200e4
        h1  = 1.204e4
        h2  = 1.215e4
        if self.delta1 is None: # init
            self.delta1, self.delta2 = 1, 0 # init as mode 1
        elif self.delta1==1: # current mode is 1
            if self.e_Ck > h1 and self.e_Ck <= h2:
                self.delta1, self.delta2 = 0, 1 # translate to mode 2
        elif self.delta2==1: # current mode is 2
            if self.e_Ck > h2:
                self.delta1, self.delta2 = 0, 0 # translate to mode 3
        else: # current mode is 3 because delta1, delta2 = 1, 1 is impossible
            if self.e_Ck < h0:
                self.delta1, self.delta2 = 1, 0 # translate to mode 1
        # save
        self.modes.append(1 if self.delta1==1 else 2 if self.delta2==1 else 3)

    def state_step(self, step_len):
        # e_RO20 in Chapter_13
        R_memb  = 0.202*(4.137e11*((self.e_Ck - 12000)/165 + 29))
        # e_RO1
        q_fp    = self.q_fp + step_len* \
                (- RO.R_fp*self.q_fp \
                 - self.p_tr \
                 + RO.p_fp*(1 - self.f_f)) \
                 /RO.I_fp
        # e_RO2
        p_tr    = self.p_tr + step_len* \
                (self.q_fp \
                 + self.delta1*(self.p_memb - self.p_tr)/RO.R_return_l \
                 - self.q_rp \
                 + self.delta2*(self.p_memb - self.p_tr)/RO.R_return_s) \
                 /RO.C_tr
        # e_RO3
        d_q_rp  = self.d_q_rp + step_len* \
                (- RO.R_rp*self.q_rp \
                 - RO.R_forward*self.q_rp \
                 - self.p_memb \
                 + RO.p_rp*(1 - self.f_r))
        q_rp    = (self.delta1 + self.delta2)*d_q_rp/RO.I_rp \
                 + (1 - self.delta1 - self.delta2)*(self.p_tr - self.p_memb)/RO.R_forward
        # e_RO4
        p_memb  = self.p_memb + step_len* \
                (self.q_rp \
                 - self.p_memb/(R_memb*(1+self.f_m)) \
                 - self.delta1*(self.p_memb - self.p_tr)/RO.R_return_l \
                 - self.delta2*(self.p_memb - self.p_tr)/RO.R_return_s \
                 - (1 - self.delta1 - self.delta2)*self.p_memb/RO.R_return_AES) \
                 / RO.C_memb
        # e_RO5
        e_Cbrine    = self.e_Cbrine + step_len*\
                (self.delta1*(self.p_memb - self.p_tr)/RO.R_return_l \
                 + self.delta2*(self.p_memb - self.p_tr)/RO.R_return_s \
                 + (1 - self.delta1 - self.delta2)*self.p_memb/RO.R_return_AES) \
                 / (1.667e-8*RO.C_brine)
        # e_RO6
        e_Ck    = self.e_Ck + step_len* \
                self.q_rp*(6*self.e_Cbrine + 0.1)/(1.667e-8 * RO.C_k)
        
        # save & update
        self.states.append([q_fp, p_tr, q_rp, p_memb, e_Cbrine, e_Ck])
        self.para_faults.append([self.f_f, self.f_r, self.f_m])
        self.x.append((0 if not self.x else self.x[-1]) + step_len)
        self.q_fp, self.p_tr, self.q_rp, self.p_memb, self.e_Cbrine, self.e_Ck = q_fp, p_tr, q_rp, p_memb, e_Cbrine, e_Ck

    def np_modes(self):
        return np.array(self.modes)

    def np_states(self):
        return np.array(self.states)

    def np_obs(self):
        states = np.array(self.states)
        obs = states[:, (1, 3, 0, 4, 6)]
        return obs

    def np_para_faults(self):
        return np.array(self.para_faults)

    def _show(self, name):
        assert isinstance(name, str)
        if name in RO.states:
            i   = RO.states.index(name)
            y   = np.array(self.states)[:, i]
        elif name in RO.obs:
            name    = RO.obs[name]
            i   = RO.states.index(name)
            y   = np.array(self.states)[:, i]
        elif name=='modes':
            y   = np.array(self.modes)
        else:
            raise RuntimeError('Unknown name.')
        x = np.array(self.x)
        plt.figure()
        plt.plot(x, y)
        plt.xlabel('Time(s)')
        plt.ylabel(name)
        plt.show()

    def show(self, name=None):
        if name is not None:
            self._show(name)
        else:
            for name in RO.states:
                self._show(name)

    def reset(self):
        # fault parameters
        self.f_f    = 0
        self.f_r    = 0
        self.f_m    = 0
        # discrete modes
        self.delta1 = 0
        self.delta2 = 0
        # continous states
        self.q_fp   = 0
        self.p_tr   = 0
        self.q_rp   = 0
        self.d_q_rp = 0 # the delegation for q_rp
        self.p_memb = 0
        self.e_Cbrine   = 0
        self.e_Ck   = 0
        # trajectory
        self.modes.clear()
        self.states.clear()
        self.para_faults.clear()
        self.x.clear()
