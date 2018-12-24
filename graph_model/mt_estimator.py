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
from data_manager2.data_manager import mt_data_manager
from graph_model.bn_diagnoser import bn_diagnoser
from data_manger.utilities import get_file_list
from statistics.plot_roc import plotROC

#settings
train_id    = 1
snr         = 20
times       = 5
test_batch  = 700
step_len    = 64
sample_rate = 1.0
roc_type    = 'micro' # 'macro'
#   log
log_path = parentdir + '\\log\\mt\\train{}\\{}db\\'.format(train_id, snr)
if not os.path.isdir(log_path):
    os.makedirs(log_path)
log_name = 'BN_estimation_' + time.asctime( time.localtime(time.time())).replace(" ", "_").replace(":", "-")+'.txt'
logfile = log_path + log_name
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename=logfile, level=logging.DEBUG, format=LOG_FORMAT)
# prepare data
data_path = parentdir + '\\tank_systems\\data\\test\\'
mana = mt_data_manager()
mana.load_data(data_path)
mana.add_noise(snr)

for t in range(times):
    msg = "estimation {}.".format(t)
    logging.info(msg)
    print(msg)
    # prepare model file
    model_path  = parentdir + '\\graph_model\\mt\\train{}\\{}db\\{}\\'.format(train_id, snr, t)
    model_files = get_file_list(model_path)
    model_files = [f for f in model_files if f.endswith('.bn')]

    inputs, y_label = mana.random_h_batch(batch=test_batch, step_num=step_len, prop=0.2, sample_rate=sample_rate)
    inputs = inputs.transpose([0, 2, 1])

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
        roc.analyse(20, y_label, y_score)
        auc = roc.auc(roc_type)
        logging.info('{}, {} auc = {}'.format(model_name, roc_type, auc))
        print('{}, {}, auc = {}'.format(model_name, roc_type, auc))
        roc.plot(roc_type, view=False, file=model_path+roc_name)

print('DONE')
