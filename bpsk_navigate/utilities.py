"""
some utilities
"""
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)

def compose_file_name(faults, paras):
    """
    RT
    """
    if isinstance(faults, str):
        file_name = faults + "@" + str(paras) + ".npy"
    else:#faults and paras are lists
        file_name = faults[0]+","+faults[1]+ "@" +str(paras[0])+","+str(paras[1]) + ".npy"
    return file_name
