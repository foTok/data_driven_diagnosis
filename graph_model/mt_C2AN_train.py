'''
The script to learn BN based on Chi-square Augmented Naive Bayesian Network
'''
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
import time
import logging
from graph_model.BN import BN
from graph_model.C2AN import C2AN
from data_manager2.data_manager import mt_data_manager
from graph_model.utilities import dis_para
from graph_model.utilities import cat_label_input

#   settings
train_id    = 1
snr         = 20
sample_rate = 1.0
step_len    = 64
dis         = [4, 8, 16, 32]
batch       = 20000
times       = 1
prefix      = 'mt_C2AN'
#   log
log_path = parentdir + '\\log\\mt\\train{}\\{}db\\'.format(train_id, snr)
if not os.path.isdir(log_path):
    os.makedirs(log_path)
log_name = 'C2AN_Training_' + time.asctime( time.localtime(time.time())).replace(" ", "_").replace(":", "-")+'.txt'
logfile = log_path + log_name
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename=logfile, level=logging.DEBUG, format=LOG_FORMAT)
#   prepare data
data_path = parentdir + '\\tank_systems\\data\\train{}\\'.format(train_id)
mana = mt_data_manager()
mana.load_data(data_path)
mana.add_noise(snr)
mm  = mana.mm
fault   = mana.cfg.faults
obs = mana.cfg.variables[:10]

msg = 'Log of Training C2AN'
logging.info(msg)
print(msg)

for t in range(times):
    model_path = parentdir + '\\graph_model\\mt\\train{}\\{}db\\{}\\'.format(train_id, snr, t)
    if not os.path.isdir(model_path):
            os.makedirs(model_path)
    for d in dis:
        para_name  = prefix + 'para, d={}'.format(d)
        model_name = prefix + 'model, d={}.bn'.format(d)
        fig_name = prefix + 'fig, d={}.gv'.format(d)

        bins    = [d]*len(obs)
        mins, intervals, bins = dis_para(mm, bins, len(fault))

        inputs, labels = mana.random_h_batch(batch=batch, step_num=step_len, prop=0.2, sample_rate=sample_rate)
        data = cat_label_input(labels, inputs)

        learner = C2AN(fault, obs)
        learner.set_batch(data, mins, intervals, bins)

        bg_time = time.time()
        learner.Build_adj()
        ed_time = time.time()
        msg = '{}, train time={}'.format(para_name, ed_time-bg_time)
        logging.info(msg)
        print(msg)

        BN = learner.Build_BN()
        BN.save(model_path+model_name)
        msg = 'save {} to {}'.format(model_name, model_path)
        logging.info(msg)
        print(msg)
        BN.graphviz(model_path+fig_name, view=False)
        msg = 'save {} to {}'.format(fig_name, model_path)
        logging.info(msg)
        print(msg)
print('DONE!')
