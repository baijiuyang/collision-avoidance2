'''
This module contains the Data class. It is used to save raw and filtered data
and some recorded and computed trial related information.
'''
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
    
    @staticmethod
    def filter(data, t, Hz, order, cutoff):
        '''
        Filter the data using butterwirth low pass digital foward
        and backward filter.
        '''
        # interpolate and extrapolate (add pads on two sides to prevent boundary effects)
        pad = 3 # seconds
        func = interp1d(t, data, axis=0, kind='linear', fill_value='extrapolate')
        indices = np.linspace(-pad, t[-1] + pad, num=len(t) + 2*pad*Hz)
        data = func(indices)
        # low pass filter on position
        b, a = butter(order, cutoff/(Hz/2.0))
        data = filtfilt(b, a, data, axis=0, padtype=None) # no auto padding
        # remove pads 
        return data[pad * Hz + 1 : -pad * Hz + 1]
        
    def get_traj(self, i_traj, col, **kwargs):
        # load kwargs
        order = 4 if 'order' not in kwargs else kwargs['order']
        cutoff = 0.6 if 'cutoff' not in kwargs else kwargs['cutoff']
        filtered = True if 'filtered' not in kwargs else kwargs['filtered']
        data = np.array(self.trajs[i_traj])[:, col]
        t = self.info['time_stamp'][i_traj]
        if filtered:
            return self.filter(data, t, self.Hz, order, cutoff)
        else:
            return np.array(self.trajs[i_traj])[:, col]
            
            
            
  