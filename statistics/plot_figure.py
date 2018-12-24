'''
Used to plot figures
'''
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
import numpy as np
from statistics.plot import plot_errorbar

# Training time
mean = np.array([[21.27792764,21.41574135,21.42514238,21.32573242,21.4867485,21.33373322,21.37393718],
                 [249.3361463,250.4340955,266.9112885,536.5230468,2548.621405,0,0],
                 [82.28748069,87.00776148,86.89082656,115.9050936,322.8806002,179.1561019,515.1840709]])

std  = np.array([[0.022905424,0.1114178,0.062868049,0.065350546,0.052710574,0.073122464,0.036965338],
                 [3.119986272,1.775222346,2.622282324,6.923928697,36.031562,0,0],
                 [2.131868414,2.201236151,2.2719665,3.311065214,10.744811,0.907584781,2.805611114]])


conf = 0.95
xlabel = 'Discretization Interval Number'
ylabel = 'Training time (s)'
xticklabel = ('2', '4', '8', '16', '32', '64', '128')
legend = ('NB', 'C2AN', 'GSAN')

plot_errorbar(mean, std, conf, xlabel, ylabel, xticklabel, legend, view=True, file=None, ylimt=(0,3000))


# Diagnosis time
mean = np.array([[13.73477891,13.66759122,13.65221983,13.70984699,13.63556624,13.58258252,13.73225477],
                 [14.67236598,15.08375735,15.17054016,15.21887764,14.63115697,0,0],
                 [15.41853129,14.95583091,14.66830766,14.51600657,14.95149586,14.40705861,14.39137985]])

std  = np.array([[0.146370461,0.062667259,0.063406382,0.129513656,0.115562514,0.061265429,0.099979266],
                 [0.366278149,0.01168368,0.026852399,0.059815607,0.048178583,0,0],
                 [0.229755772,0.506982947,0.254392443,0.190882516,0.46582389,0.287591117,0.356255098]])


conf = 0.95
xlabel = 'Discretization Interval Number'
ylabel = 'Diagnosis time (s)'
xticklabel = ('2', '4', '8', '16', '32', '64', '128')
legend = ('NB', 'C2AN', 'GSAN')

plot_errorbar(mean, std, conf, xlabel, ylabel, xticklabel, legend, view=True, file=None, ylimt=(0,20))



# Micro AUC
mean = np.array([[0.514382585,0.686450272,0.804418912,0.855910952,0.872054014,0.912739388,0.925962449,0.930854082],
                 [0, 0.722880612,0.831039728,0.888515306,0.900422925,0.921304966,0,0],
                 [0.516247891,0.733518776,0.819369932,0.878651905,0.919723061,0.952034898,0.927663129,0.931779932]])

std  = np.array([[0.001723303,0.006898486,0.002663932,0.001259981,0.00102405,0.00184087,0.001369096,0.000630262],
                 [0, 0.009028106,0.001443977,0.0014182,0.002390497,0.001471851,0,0],
                 [0.002932568,0.007401881,0.012393311,0.001870904,0.005152312,0.00063963,0.001396521,0.000709581]])


conf = 0.95
xlabel = 'Discretization Interval Number'
ylabel = 'Micro AUC'
xticklabel = ('GAU', '2', '4', '8', '16', '32', '64', '128')
legend = ('NB', 'C2AN', 'GSAN')

plot_errorbar(mean, std, conf, xlabel, ylabel, xticklabel, legend, view=True, text=False)


# Dynamic
# Training time
mean = np.array([[82.28748069,87.00776148,86.89082656,115.9050936,322.8806002,179.1561019,515.1840709],
                 [407.4190895,390.8743778,466.564575,621.3186441,1302.615357,592.5287191,1600.898716]])

std  = np.array([[2.131868414,2.201236151,2.2719665,3.311065214,10.744811,0.907584781,2.805611114],
                 [18.18767719,12.29737792,13.21480415,16.83679672,41.32192204,1.990070942,3.311731958]])


conf = 0.95
xlabel = 'Discretization Interval Number'
ylabel = 'Training time (s)'
xticklabel = ('2', '4', '8', '16', '32', '64', '128')
legend = ('GSAN', 'DGSAN')

plot_errorbar(mean, std, conf, xlabel, ylabel, xticklabel, legend, ylimt=(0, 1800))


# Diagnosis time
mean = np.array([[15.41853129,14.95583091,14.66830766,14.51600657,14.95149586,14.40705861,14.39137985],
                 [56.148066,56.26076005,54.94242616,54.48370212,54.44581753,53.89585266,54.28705449]])

std  = np.array([[0.229755772,0.506982947,0.254392443,0.190882516,0.46582389,0.287591117,0.356255098],
                 [1.291654969,1.24031524,0.905049283,0.794016454,0.682238408,0.837887222,1.348068551]])


conf = 0.95
xlabel = 'Discretization Interval Number'
ylabel = 'Diagnosis time (s)'
xticklabel = ('2', '4', '8', '16', '32', '64', '128')
legend = ('GSAN','DGSAN')

plot_errorbar(mean, std, conf, xlabel, ylabel, xticklabel, legend,ylimt=(0, 70))



# Micro AUC
mean = np.array([[0.733518776,0.819369932,0.878651905,0.919723061,0.952034898,0.927663129,0.931779932],
                 [0.771551701,0.832690476,0.879842381,0.923895442,0.964262517,0.927934966,0.931537415]])

std  = np.array([[0.007401881,0.012393311,0.001870904,0.005152312,0.00063963,0.001396521,0.000709581],
                 [0.011290886,0.005107924,0.003786913,0.010175057,0.001048433,0.001937305,0.000647753]])


conf = 0.95
xlabel = 'Discretization Interval Number'
ylabel = 'Micro AUC'
xticklabel = ('2', '4', '8', '16', '32', '64', '128')
legend = ('GSAN','DGSAN')

plot_errorbar(mean, std, conf, xlabel, ylabel, xticklabel, legend, view=True, text=False)

# CNN vs LSTM
# Micro AUC
mean = np.array([[0.7924982464515508,0.8719461561171098,0.8659028942448016]])

std  = np.array([[0,0,0]])


conf = 0.95
xlabel = 'Hidden State Number'
ylabel = 'Micro AUC'
xticklabel = ('32', '64', '128')
legend = ('LSTM',)

plot_errorbar(mean, std, conf, xlabel, ylabel, xticklabel, legend, view=True, text=False)



# MT
# Micro AUC
mean = np.array([[0.516981684,0.962278827,0.978736786,0.985832551,0.9833125],
                 [0.68971551,0.991809133,0.999559286,0.999741071,0.998421633],
                 [0.646110357,0.974105153,0.899612959,0.796816939,0.647451224]])

std  = np.array([[0.001044949,0.002867908,0.001700969,0.000869184,0.000365255],
                 [0.002255306,0.00041148,6.2551E-05,9.32143E-05,6.53061E-05],
                 [0.000351786,0.002097092,0.002943367,0.002528061,0.003106122]])


conf = 0.95
xlabel = 'Discretization Interval Number'
ylabel = 'Micro AUC'
xticklabel = ('GAU', '4', '8', '16', '32')
legend = ('NB','GSAN','DGSAN')

plot_errorbar(mean, std, conf, xlabel, ylabel, xticklabel, legend, view=True, text=False)


print('DONE')
