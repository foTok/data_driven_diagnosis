"""
this script is used to generate data
"""
import os
import numpy as np
from bpsk_generator import Bpsk

PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
#fault types
FAULT = ["tma", "tmb", "pseudo_rate", "carrier_rate", "carrier_leak", "amplify"]
#fault parameters
PARA_BEGIN = [0.15, (0.8 * 10**6, 7.3 * 10**6), -0.05, -0.05, 0.1, -1]
PARA_STEP = [0.1, (1 * 10**6, 0.5 * 10**6), 0.01, 0.01, 0.1, 0.1]
PARA_END = [0.9, (8.8 * 10**6, 13 * 10**6), 0.05, 0.05, 0.5, 1]

TIME = 0.0001
FAULT_TIME = TIME / 2
# Single Fault
for fault, para_begin, para_step, para_end in zip(FAULT, PARA_BEGIN, PARA_STEP, PARA_END):
    if isinstance(para_begin, tuple):
        for i in range(int((para_end[0]-para_begin[0])/para_step[0])+1):
            for j in range(int((para_end[1]-para_begin[1])/para_step[1])+1):
                sigma = para_begin[0] + i * para_step[0]
                f_d = para_begin[1] + j * para_step[1]
                parameters = (sigma, f_d)
                bpsk = Bpsk()
                bpsk.insert_fault_para(fault, parameters)
                bpsk.insert_fault_time("all", FAULT_TIME)
                data = bpsk.generate_signal(TIME)
                file_name = PATH + "\\data\\" + str(fault+ "@" +str(parameters)) + ".npy"
                np.save(file_name, data)
                print("save file {}".format(file_name))
    else:
        for i in range(int((para_end-para_begin)/para_step)+1):
            parameters = para_begin + i * para_step
            if parameters == 0:
                continue
            bpsk = Bpsk()
            bpsk.insert_fault_para(fault, parameters)
            bpsk.insert_fault_time("all", FAULT_TIME)
            data = bpsk.generate_signal(TIME)
            file_name = PATH + "\\data\\" + str(fault+ "@" +str(parameters)) + ".npy"
            np.save(file_name, data)
            print("save file {}".format(file_name))

# Two Faults
for f1 in range(len(FAULT)):
    for f2 in range(f1+1, len(FAULT)):
        print("f1={},f2={}".format(f1,f2))
        fault1 = FAULT[f1]
        para_begin1 = PARA_BEGIN[f1]
        para_step1 = PARA_STEP[f1]
        para_end1 = PARA_END[f1]
        fault2 = FAULT[f2]
        para_begin2 = PARA_BEGIN[f2]
        para_step2 = PARA_STEP[f2]
        para_end2 = PARA_END[f2]

        if isinstance(para_begin1, tuple):
            for i in range(int((para_end1[0]-para_begin1[0])/para_step1[0])+1):
                for j in range(int((para_end1[1]-para_begin1[1])/para_step1[1])+1):
                    sigma = para_begin1[0] + i * para_step1[0]
                    f_d = para_begin1[1] + j * para_step1[1]
                    parameters1 = (sigma, f_d)
                    #if fault1 is tmb, tma can never be tmb
                    for k in range(int((para_end2-para_begin2)/para_step2)+1):
                        parameters2 = para_begin2 + k * para_step2
                        if parameters2 == 0:
                            continue
                        bpsk = Bpsk()
                        bpsk.insert_fault_para(fault1, parameters1)
                        bpsk.insert_fault_para(fault2, parameters2)
                        bpsk.insert_fault_time("all", FAULT_TIME)
                        data = bpsk.generate_signal(TIME)
                        file_name = PATH + "\\data\\" +\
                            str(fault1+","+fault2+ "@" +str(parameters1)+","+str(parameters2)) + ".npy"
                        np.save(file_name, data)
                        print("save file {}".format(file_name))
        else:
            for k in range(int((para_end1-para_begin1)/para_step1)+1):
                parameters1 = para_begin1 + k * para_step1
                if parameters1 == 0:
                    continue
                if isinstance(para_begin2, tuple):
                    for i in range(int((para_end2[0]-para_begin2[0])/para_step2[0])+1):
                        for j in range(int((para_end2[1]-para_begin2[1])/para_step2[1])+1):
                            sigma = para_begin2[0] + i * para_step2[0]
                            f_d = para_begin2[1] + j * para_step2[1]
                            parameters2 = (sigma, f_d)
                            bpsk = Bpsk()
                            bpsk.insert_fault_para(fault1, parameters1)
                            bpsk.insert_fault_para(fault2, parameters2)
                            bpsk.insert_fault_time("all", FAULT_TIME)
                            data = bpsk.generate_signal(TIME)
                            file_name = PATH + "\\data\\" +\
                                str(fault1+","+fault2+ "@" +\
                                str(parameters1)+","+str(parameters2)) + ".npy"
                            np.save(file_name, data)
                            print("save file {}".format(file_name))
                else:
                    for i in range(int((para_end2-para_begin2)/para_step2)+1):
                        parameters2 = para_begin2 + i * para_step2
                        if parameters2 == 0:
                            continue
                        bpsk = Bpsk()
                        bpsk.insert_fault_para(fault1, parameters1)
                        bpsk.insert_fault_para(fault2, parameters2)
                        bpsk.insert_fault_time("all", FAULT_TIME)
                        data = bpsk.generate_signal(TIME)
                        file_name = PATH + "\\data\\" +\
                            str(fault1+","+fault2+ "@" +str(parameters1)+","+str(parameters2)) + ".npy"
                        np.save(file_name, data)
                        print("save file {}".format(file_name))
