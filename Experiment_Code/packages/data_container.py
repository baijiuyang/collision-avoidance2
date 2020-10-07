from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d

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
        
    def filter(self, i_traj, order, cutoff):
        '''
        Filter the data using butterwirth low pass digital foward
        and backward filter.
        '''
        # interpolate and extrapolate (add pads on two sides to prevent boundary effects)
        pad = 3 * self.Hz
        data = self.trajs[i_traj]
        t = list(range(len(data)))
        func = interp1d(t, data, axis=0, kind='linear', fill_value='extrapolate')
        indices = list(range(-pad, len(data) + pad))
        data = func(indices)
        # low pass filter on position
        b, a = butter(order, cutoff/(self.Hz/2.0))
        data = filtfilt(b, a, data, axis=0, padtype=None) # no auto padding
        # remove pads
        data = data[pad:-pad]
        return data
        
    def get_traj(self, i_traj, **kwargs):
        # load kwargs
        order = 4 if 'order' not in kwargs else kwargs['order']
        cutoff = 0.6 if 'cutoff' not in kwargs else kwargs['cutoff']
        filtered = True if 'filtered' not in kwargs else kwargs['filtered']
        if filtered:
            return self.filter(i_traj, order, cutoff)
        else:
            return self.trajs[i_traj]
            
            
            
  