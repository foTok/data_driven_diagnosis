import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
import numpy as np
from RO_system.RO_model import RO

ro = RO(1)
ro.run(100)
ro.show('e_Ck')
ro.show('p_memb')
print('DONE')
