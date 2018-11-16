"""
bpsk navigation module
"""
import fractions
from math import sqrt
from math import cos
from math import pi
import numpy as np
from pseudo_generator import Pseudo
from msg_generator import Msg


class Bpsk:
    """
    r(t) = AD(t)C(t)cos(omiga*t+phi)
    Faults parameter are !!! Relative Values !!!
    """
    def __init__(self, amplify=None, initial_code=None, code_rate=None, sample_rate=None):
        #normal parameters
        self.amplify = 10 if amplify is None else amplify
        self.code_rate = 1023000 if code_rate is None else code_rate
        self.phi = 0.5 * pi
        self.sample_rate = self.code_rate*10 if sample_rate is None else sample_rate
        self.pseudo = Pseudo(initial_code=initial_code,\
                             code_rate=self.code_rate, sample_rate=self.sample_rate)
        self.msg = Msg(code_rate=self.code_rate, sample_rate=self.sample_rate)
        #fault parameters
        self.delta_amplify = 0
        self.delta_pseudo_tma = 0
        self.delta_pseudo_tmb = (0, 0)
        self.delta_pseudo_rate = 0
        self.delta_carrier_rate = 0
        self.alpha = 0
        #fault time
        self.amplify_fault_time = float("Inf")
        self.tma_fault_time = float("Inf")
        self.tmb_fault_time = float("Inf")
        self.pseudo_rate_fault_time = float("Inf")
        self.carrier_rate_fault_time = float("Inf")
        self.carrier_leak_fault_time = float("Inf")

    def insert_fault_para(self, fault, parameter):
        """
        insert fault parameters
        """
        if fault == "amplify":
            self.delta_amplify = parameter
        elif fault == "tma":
            self.delta_pseudo_tma = parameter
            self.pseudo.insert_fault_para("tma", parameter)
        elif fault == "tmb":
            self.delta_pseudo_tmb = parameter
            self.pseudo.insert_fault_para("tmb", parameter)
        elif fault == "pseudo_rate":
            self.delta_pseudo_rate = fractions.Fraction(parameter)
            self.pseudo.insert_fault_para("code_rate", parameter)
        elif fault == "carrier_rate":
            self.delta_carrier_rate = parameter
        elif fault == "carrier_leak":
            self.alpha = parameter
        else:
            print("Unknown Fault!")

    def insert_fault_time(self, fault, fault_time):
        """
        set the fault time
        """
        if fault == "amplify":
            self.amplify_fault_time = fault_time
        elif fault == "tma":
            self.tma_fault_time = fault_time
            self.pseudo.insert_fault_time("tma", fault_time)
        elif fault == "tmb":
            self.tmb_fault_time = fault_time
            self.pseudo.insert_fault_para("tmb", fault_time)
        elif fault == "pseudo_rate":
            self.pseudo_rate_fault_time = fault_time
            self.pseudo.insert_fault_para("code_rate", fault_time)
        elif fault == "carrier_rate":
            self.carrier_rate_fault_time = fault_time
        elif fault == "carrier_leak":
            self.carrier_leak_fault_time = fault_time
        elif fault == "all":
            self.amplify_fault_time = fault_time
            self.tma_fault_time = fault_time
            self.tmb_fault_time = fault_time
            self.pseudo_rate_fault_time = fault_time
            self.carrier_rate_fault_time = fault_time
            self.carrier_leak_fault_time = fault_time
            self.pseudo.insert_fault_time("all", fault_time)
        else:
            print("Unknown Fault!")

    def clear_faults(self):
        """
        clear all the faults
        """
        #fault parameters
        self.delta_amplify = 0
        self.delta_pseudo_tma = 0
        self.delta_pseudo_tmb = (0, 0)
        self.delta_pseudo_rate = 0
        self.delta_carrier_rate = 0
        self.alpha = 0
        #fault time
        self.amplify_fault_time = float("Inf")
        self.tma_fault_time = float("Inf")
        self.tmb_fault_time = float("Inf")
        self.pseudo_rate_fault_time = float("Inf")
        self.carrier_rate_fault_time = float("Inf")
        self.carrier_leak_fault_time = float("Inf")
        self.pseudo.clear_faults()

    def re_init(self):
        """
        reset the bpsk signal generator
        """
        self.clear_faults()
        self.pseudo.re_init()

    def modulate(self, msg, code, time):
        """
        modulate one signal point
        """
        amplify = self.amplify if time < self.amplify_fault_time \
            else self.amplify * (1 + self.delta_amplify)
        m_msg = msg * code if time < self.carrier_leak_fault_time \
            else sqrt(1-self.alpha) * msg * code + sqrt(self.alpha)
        carrier_rate = self.code_rate if time < self.carrier_rate_fault_time \
            else self.code_rate * (1 + fractions.Fraction(self.delta_carrier_rate))
        carrier = cos(2 * pi * carrier_rate * time + self.phi)
        sig0 = m_msg * carrier
        sig1 = amplify * sig0

        return carrier, sig0, sig1

    def generate_signal(self, end_time):
        """
        generate similation signal
        the unit of end_time is s(second)
        Used to generate signal with random input
        """
        length = int(end_time*self.sample_rate)
        sample_step = fractions.Fraction(1, self.sample_rate)
        data = np.zeros([length, 5])    #msg, pseudo, carrier, s0, s1
        #generate input signal randomly
        for i in range(length):
            time = i * sample_step
            msg = self.msg.sample(time)
            p_code = self.pseudo.sample(time)
            carrier, sig0, sig1 = self.modulate(msg, p_code, time)
            data[i, 0] = msg
            data[i, 1] = p_code
            data[i, 2] = carrier
            data[i, 3] = sig0
            data[i, 4] = sig1
        return data

    def generate_signal_with_input(self, end_time, input_msg):
        """
        generate similation signal
        the unit of end_time is s(second) with residuals
        Used to gnerate signal with specified input
        """
        length = int(end_time*self.sample_rate)
        sample_step = fractions.Fraction(1, self.sample_rate)
        data = np.zeros([length, 5])    #msg, pseudo, carrier, s0, s1
        #generate input signal randomly
        for i in range(length):
            time = i * sample_step
            msg = input_msg[i]
            p_code = self.pseudo.sample(time)
            carrier, sig0, sig1 = self.modulate(msg, p_code, time)
            data[i, 0] = msg
            data[i, 1] = p_code
            data[i, 2] = carrier
            data[i, 3] = sig0
            data[i, 4] = sig1
        return data
