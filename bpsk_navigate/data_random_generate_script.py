"""
this script is used to generate data
"""
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
import random
import numpy as np
from bpsk_navigate.bpsk_generator import Bpsk

PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
#fault types
FAULT = ["tma", "tmb", "pseudo_rate", "carrier_rate", "carrier_leak", "amplify"]
#fault parameters
PARA_BEGIN = [0.2, (0.8 * 10**6, 7.3 * 10**6), -0.05, -0.05, 0.1, 0.1]
PARA_END = [0.9, (8.8 * 10**6, 13 * 10**6), 0.05, 0.05, 0.5, 1.0]

TIME = 0.0001
FAULT_TIME = TIME / 2
RANDOM = 5
#Single Fault
for fault in FAULT:
    for _ in range(RANDOM):
        if fault != "tmb":
            index = FAULT.index(fault)
            begin = PARA_BEGIN[index]
            end = PARA_END[index]
            while True:
                para = random.uniform(begin, end)
                if abs(para) > 0.01:
                    break
        else:#tmb
            begin1 = PARA_BEGIN[1][0]
            begin2 = PARA_BEGIN[1][1]
            end1 = PARA_END[1][0]
            end2 = PARA_END[1][1]
            sigma = random.uniform(begin1, end1)
            f_d = random.uniform(begin2, end2)
            para = (sigma, f_d)

        bpsk = Bpsk()
        bpsk.insert_fault_para(fault, para)
        bpsk.insert_fault_time("all", FAULT_TIME)
        data = bpsk.generate_signal(TIME)
        file_name = PATH + "\\data\\test\\" +\
            fault + "@" + str(para) + ".npy"
        np.save(file_name, data)


#Two Faults
for f1 in range(len(FAULT)):
    for f2 in range(f1+1, len(FAULT)):
        fault1 = FAULT[f1]
        fault2 = FAULT[f2]
        for _ in range(RANDOM):
            if fault1 != "tmb":
                begin = PARA_BEGIN[f1]
                end = PARA_END[f1]
                while True:
                    para1 = random.uniform(begin, end)
                    if abs(para1) > 0.01:
                        break
            else:#tmb
                begin1 = PARA_BEGIN[1][0]
                begin2 = PARA_BEGIN[1][1]
                end1 = PARA_END[1][0]
                end2 = PARA_END[1][1]
                sigma = random.uniform(begin1, end1)
                f_d = random.uniform(begin2, end2)
                para1 = (sigma, f_d)

            if fault2 != "tmb":
                begin = PARA_BEGIN[f2]
                end = PARA_END[f2]
                para2 = random.uniform(begin, end)
                while True:
                    para2 = random.uniform(begin, end)
                    if abs(para2) > 0.01:
                        break
            else:#tmb
                begin1 = PARA_BEGIN[1][0]
                begin2 = PARA_BEGIN[1][1]
                end1 = PARA_END[1][0]
                end2 = PARA_END[1][1]
                sigma = random.uniform(begin1, end1)
                f_d = random.uniform(begin2, end2)
                para2 = (sigma, f_d)

            bpsk = Bpsk()
            bpsk.insert_fault_para(fault1, para1)
            bpsk.insert_fault_para(fault2, para2)
            bpsk.insert_fault_time("all", FAULT_TIME)
            data = bpsk.generate_signal(TIME)
            file_name = PATH + "\\data\\test\\" +\
                str(fault1+","+fault2+ "@" +str(para1)+","+str(para2)) + ".npy"
            np.save(file_name, data)
