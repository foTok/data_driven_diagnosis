'''
Another data manager
'''
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,parentdir)
import pickle
import numpy as np

class data_manager:
    def __init__(self):
        self.cfg    = None
        self.mm = None
        self.data   = {}
        self.noise_data = {}

    def load_data(self, path):
        '''
        load data in path
        '''
        with open(path+'cfg', 'rb') as f:
            self.cfg = pickle.load(f)
        for file in self.cfg.files:
            filename = file.file_name
            self.data[filename] = np.load(path+filename+'.npy')


class mt_data_manager(data_manager):
    def __init__(self):
        super(mt_data_manager, self).__init__()

    def add_noise(self, snr, begin=400):
        # tank or pipe number
        n = int(len(self.cfg.variables)/2)
        mm = np.array([[float('inf'), -float('inf')]]*(2*n+1))
        # ratio noise
        ratio = 1/(10**(snr/20))
        # height of tanks
        for file in self.data:
            data = [0]*(2*n+1)
            # for input
            qi  = self.data[file][0,:]
            std_qi  = np.std(qi)
            noise_qi = qi + np.random.standard_normal(size=len(qi)) * std_qi * ratio
            mm[0, 0], mm[0, 1] = min(mm[0,0], min(noise_qi[begin:])), max(mm[0,1], np.max(noise_qi[begin:]))
            data[0] = noise_qi
            for i in range(n):
                h   = self.data[file][i+1,:]
                q   = self.data[file][n+i+1,:]
                std_h   = np.std(h)
                std_q   = np.std(q)
                noise_h = h + np.random.standard_normal(size=len(h)) * std_h * ratio
                noise_q = q + np.random.standard_normal(size=len(q)) * std_q * ratio
                mm[i+1, 0], mm[i+1, 1] = min(mm[i+1,0], min(noise_h[begin:])), max(mm[i+1,1], np.max(noise_h[begin:]))
                mm[n+i+1, 0], mm[n+i+1, 1] = min(mm[n+i+1,0], min(noise_q[begin:])), max(mm[n+i+1,1], max(noise_q[begin:]))
                data[i+1] = noise_h
                data[n+i+1]   = noise_q
            self.noise_data[file]   = np.array(data)
        self.mm = mm

    def random_h_batch(self, batch, step_num, prop, sample_rate):
        '''
        return q and height
        '''
        data    = []
        label   = []
        if not self.noise_data:
            raise RuntimeError('You have to add noise by add_noise(self, snr) firstly.')
        n = int(len(self.cfg.variables)/2)  # tank or pip number
        f_n = len(self.cfg.faults)          # fault number
        sample_interval = int(sample_rate / self.cfg.time_step)
        m_batch = [int(batch*prop)] + [int(batch*(1-prop)/f_n)]*f_n
        # modes
        modes = ['normal'] + self.cfg.faults
        # normal
        for n_num, m, l in zip(m_batch, modes, range(len(modes))):
            m_file = [file.file_name for file in self.cfg.files if file.fault_type==m]
            f_time = [file.fault_time for file in self.cfg.files if file.fault_type==m]
            file_samples = [int(n_num/len(m_file))]*(len(m_file)-1)
            file_samples.append(n_num - sum(file_samples))           # sample number choiced from each file
            for file, f_t, i in zip(m_file, f_time, file_samples):
                qh_data    = self.noise_data[file][:n+1,:] # 0 is q and 1:n+1 are heights
                _, len_data = qh_data.shape
                sample_begin    = int(f_t/self.cfg.time_step) if f_t > 0 else 0
                sample_end  = len_data - sample_interval*step_num
                sampled_index   = np.random.choice(range(sample_begin, sample_end), i)
                for index in sampled_index:
                    sampled_data    = qh_data[:, range(index, index+step_num*sample_interval, sample_interval)]
                    data.append(sampled_data)
                    label.append(l)
        # inputs: batch*feature*time_step, labels: batch*label
        return np.array(data).astype(np.float32), np.array(label).astype(np.float32)
