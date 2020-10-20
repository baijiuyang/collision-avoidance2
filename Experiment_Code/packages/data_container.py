from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
import numpy as np

class Data:

    def __init__(self, Hz, trajs=None, info=None, dump=None, header=None):
        self.Hz = Hz
        self.trajs = trajs if trajs else []
        self.dump = dump if dump else {}
        self.info = info if info else {}
        self.header = None
        
    def filter_all(self):
        self.p_subj = []
        for i in range(len(self.trajs)):
            self.p_subj.append(self.get_traj(i))
        self.p_subj = np.array(self.p_subj)
    def add_traj(self, traj):
        self.trajs.append(traj)
        
    def add_header(self, header):
        self.header = header
        
    def add_info(self, info):
        for key, val in info.items():
            if key in self.info:
                self.info[key].append(val)
            else:                
                self.info[key] = val
    
    def add_dump(self, i_traj, err_info):
        self.dump[i_traj] = err_info
        
    def filter(self, i_traj, col, order, cutoff):
        '''
        Filter the data using butterwirth low pass digital foward
        and backward filter.
        '''
        # interpolate and extrapolate (add pads on two sides to prevent boundary effects)
        pad = 3 # seconds
        data = np.array(self.trajs[i_traj])[:, col]
        t = self.info['time_stamp'][i_traj]
        func = interp1d(t, data, axis=0, kind='linear', fill_value='extrapolate')
        indices = np.linspace(-pad, t[-1] + pad, num=len(t) + 2*pad*self.Hz)
        data = func(indices)
        # low pass filter on position
        b, a = butter(order, cutoff/(self.Hz/2.0))
        data = filtfilt(b, a, data, axis=0, padtype=None) # no auto padding
        # remove pads 
        return data[pad * self.Hz + 1 : -pad * self.Hz + 1]
        
    def get_traj(self, i_traj, col=[0, 2], **kwargs):
        # load kwargs
        order = 4 if 'order' not in kwargs else kwargs['order']
        cutoff = 0.6 if 'cutoff' not in kwargs else kwargs['cutoff']
        filtered = True if 'filtered' not in kwargs else kwargs['filtered']
        if filtered:
            return self.filter(i_traj, col, order, cutoff)
        else:
            return np.array(self.trajs[i_traj])[:, col]
            
            
            
  