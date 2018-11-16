"""
generate data
"""
import sys
import numpy as np
from bpsk_generator import Bpsk

def generate_signal(file_name, fault, parameter, time=None, fault_time=None):
    """
    generate signal data and save them in a file
    """
    time = 0.0001 if time is None else time
    fault_time = time / 2 if fault_time is None else fault_time
    bpsk = Bpsk()
    bpsk.insert_fault_para(fault, parameter)
    bpsk.insert_fault_time("all", fault_time)
    data = bpsk.generate_signal(time)
    np.save(file_name, data)

def generate_signal_multi_fault(file_name, fault, parameter, time=None, fault_time=None):
    """
    generate signal data and save them in a file
    """
    time = 0.0001 if time is None else time
    fault_time = time / 2 if fault_time is None else fault_time
    bpsk = Bpsk()
    for f, p in zip(fault, parameter):
        bpsk.insert_fault_para(f, p)
    bpsk.insert_fault_time("all", fault_time)
    data = bpsk.generate_signal(time)
    np.save(file_name, data)


if __name__ == "__main__":
    if len(sys.argv) == 4:
        generate_signal(sys.argv[1], sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 5:
        generate_signal(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    elif len(sys.argv) == 6:
        generate_signal(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    else:
        print("Parameter Error: file_name, fault, parameter, time=None, fault_time=None")
