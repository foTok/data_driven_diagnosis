"""
this module generate msg randomly
"""

import fractions
import numpy as np

class Msg:
    """
    Msg generate msg randomly
    """
    def __init__(self, code_rate=None, sample_rate=None):
        self.code_rate = 1023000 if code_rate is None else code_rate
        self.sample_rate = self.code_rate * 10 if sample_rate is None else sample_rate
        self.initial_time = fractions.Fraction(0)

        self.msg = -1

    def sample(self, time):
        """
        sample a point randomly
        """
        time_step = fractions.Fraction(1, self.code_rate)
        if (time - self.initial_time) > time_step:
            self.msg = 1 if np.random.random() < 0.5 else -1
            self.initial_time = self.initial_time + time_step
        return self.msg
