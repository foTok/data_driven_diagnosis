"""
plot figures
"""
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
import numpy as np
import matplotlib.pyplot as plt

#accuracy#labels

#xticks
xticks = ["CNN", "DCNN", "DSCNN", "RDSCNN", "RDSECNN"]
#xlabel
xlabel = "Diagnoser"
#ylabel
ylabel = "Time(s)"
#accuracy for single fault
time = np.array([6.55, 1.87, 1.53, 2.23, 2.79])

plt.bar([1,2,3,4,5], time, facecolor = 'grey', edgecolor = 'white', width = 0.5)
plt.xticks([1,2,3,4,5], xticks)
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.show()
