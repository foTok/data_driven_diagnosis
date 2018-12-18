"""
estimate the diagnoser or feature extractor randomly
"""
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
import time
import logging
import numpy as np
from data_manger.bpsk_data_tank import BpskDataTank
from data_manger.utilities import get_file_list
from graph_model.bn_diagnoser import bn_diagnoser
from statistics.plot_roc import plotROC

#settings
logfile = parentdir + '\\log\\bpsk\\'\
        'BN_estimation_' + time.asctime( time.localtime(time.time())).replace(" ", "_").replace(":", "-")+'.txt'
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename=logfile, level=logging.DEBUG, format=LOG_FORMAT)
snr = 20
train_id = 1
times = 5
data_path = parentdir + '\\bpsk_navigate\\data\\test\\'
test_batch = 700
roc_type = 'micro' # 'macro'

# prepare data
mana = BpskDataTank()
step_len=128
list_files = get_file_list(data_path)
for file in list_files:
    mana.read_data(data_path+file, step_len=step_len, snr=snr)

for t in range(times):
    msg = "estimation {}.".format(t)
    logging.info(msg)
    print(msg)
    # prepare model file
    model_path  = parentdir + '\\graph_model\\bpsk\\train{}\\{}db\\{}\\'.format(train_id, snr, t)
    model_files = get_file_list(model_path)
    model_files = [f for f in model_files if f.startswith('GSAN') and f.endswith('.bn')]

    inputs, labels, _, res = mana.random_batch(test_batch, normal=1/7, single_fault=10, two_fault=0)
    inputs = inputs.view(-1, 5, step_len)
    inputs = inputs.permute([0, 2, 1])
    label = labels.detach().numpy()
    y_label = np.sum(label*np.array([1,2,3,4,5,6]), 1)

    for model_name in model_files:
        roc_name = 'ROC' + model_name[:-3]
        diagnoser = bn_diagnoser(model_path + model_name)

        bg = time.clock()         # time start
        _, y_score = diagnoser.diagnose(inputs)
        ed = time.clock()         # time end
        logging.info('{}, predict time={} for a {} batch'.format(model_name, ed-bg, test_batch))
        print('{}, predict time={} for a {} batch'.format(model_name, ed-bg, test_batch))

        # statistics
        roc = plotROC()
        roc.analyse(6, y_label, y_score)
        auc = roc.auc(roc_type)
        logging.info('{}, {} auc = {}'.format(model_name, roc_type, auc))
        print('{}, {}, auc = {}'.format(model_name, roc_type, auc))
        roc.plot(roc_type, view=False, file=model_path+roc_name)

print('DONE')
