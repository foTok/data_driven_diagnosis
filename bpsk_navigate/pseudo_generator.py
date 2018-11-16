"""
a pseudo code generator
"""

import fractions
import numpy as np
from math import sin
from math import cos
from math import exp
from math import pi

class Pseudo:
    """
    Generate the pseudo code by shifting
    Faults parameter are !!! Relative Values !!!
    """
    def __init__(self, initial_code=None, code_rate=None, sample_rate=None):
        self.initial_code = [-1, 1, -1, -1, 1, -1, 1] if initial_code is None else initial_code
        self.code_rate = 1023000 if code_rate is None else code_rate
        self.sample_rate = 10230000 if sample_rate is None else sample_rate
        self.code = self.initial_code
        self.initial_time = fractions.Fraction(0)
        self.real_initial_time = fractions.Fraction(0)
        #fault parameters
        self.delta_code_rate = 0
        self.delta_tma = 0
        self.tmb_para = (0, 0)
        self.tma_fault_time = float("Inf")
        self.tmb_fault_time = float("Inf")
        self.code_rate_fault_time = float("Inf")


    def sample(self, time):
        """
        get a sample point from the pseudo generator
        time must increase from 0 to end, if jump to some point directly,
        the code will not be shifted right and the sample value will be wrong.
        """
        #code rate error
        if time < self.code_rate_fault_time:
            time_step = fractions.Fraction(1, self.code_rate)
        else:
            time_step = fractions.Fraction(1, int(self.code_rate * (1 + self.delta_code_rate)))
        #functionality
        if (time - self.initial_time) > time_step:
            self.initial_time = self.initial_time + time_step
            self.real_initial_time = self.initial_time
            self.code = self.code[-1:] + self.code[:-1]
        the_code = self.code[-1]
        #TMA
        if (time >= self.tma_fault_time)\
       and (self.delta_tma != 0)\
       and (self.code[-1] == -1)\
       and (self.code[0] == 1):
            if(time - self.initial_time) <= (time_step * self.delta_tma):
                self.real_initial_time = self.initial_time - time_step
                the_code = 1
            else:
                self.real_initial_time = self.initial_time + time_step * self.delta_tma
        #TMB
        tmb = 1
        if (time >= self.tmb_fault_time)\
       and (self.tmb_para != (0, 0)):
            sigma = self.tmb_para[0]
            f_d = self.tmb_para[1]
            elapse = time - self.real_initial_time
            frq = 2 * pi * f_d
            tmb_error = exp(-sigma * elapse)\
                       *(cos(frq * elapse) + (sigma/frq) * sin(frq * elapse))
            tmb = 1 + tmb_error
        the_code = the_code * tmb
        #return
        return the_code

    def insert_fault_para(self, fault, parameter):
        """
        insert fault
        """
        if fault == "code_rate":
            self.delta_code_rate = fractions.Fraction(parameter)
        elif fault == "tma":
            self.delta_tma = parameter
        elif fault == "tmb":
            self.tmb_para = parameter
        else:
            print("Unknown Fault!")


    def insert_fault_time(self, fault, fault_time):
        """
        insert the fault time
        """
        if fault == "code_rate":
            self.code_rate_fault_time = fault_time
        elif fault == "tma":
            self.tma_fault_time = fault_time
        elif fault == "tmb":
            self.tmb_fault_time = fault_time
        elif fault == "all":
            self.code_rate_fault_time = fault_time
            self.tma_fault_time = fault_time
            self.tmb_fault_time = fault_time
        else:
            print("Unknown Fault!")

    def clear_faults(self):
        """
        clear all the faults
        """
        self.delta_code_rate = 0
        self.delta_tma = 0
        self.tmb_para = (0, 0)
        self.tma_fault_time = float("Inf")
        self.tmb_fault_time = float("Inf")
        self.code_rate_fault_time = float("Inf")

    def re_init(self):
        """
        reset the pseudo generator.
        the initial code, code rate and sample rate are reserved.
        """
        self.clear_faults()
        self.code = self.initial_code
        self.initial_time = fractions.Fraction(0)
        self.real_initial_time = fractions.Fraction(0)

    def generate_signal(self, end_time):
        """
        generate similation signal
        the unit of end_time is s(second)
        Used to generate signal with random input
        """
        length = int(end_time*self.sample_rate)
        sample_step = fractions.Fraction(1, self.sample_rate)
        data = np.zeros([length, 1])    #msg, pseudo, carrier, s0, s1
        #generate input signal randomly
        for i in range(length):
            time = i * sample_step
            p_code = self.sample(time)
            data[i, 0] = p_code
        return data
