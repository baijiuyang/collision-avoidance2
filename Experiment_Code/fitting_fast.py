import os
import sys
import numpy as np
from scipy.optimize import minimize
import pickle
from packages.helper import ymdhms
from packages.ode_simulator import ODESimulator
from packages import data_container
sys.modules['data_container'] = data_container


fajen_approach = {'name': 'fajen_approach', 'ps': None,
                  'b1': 3.25, 'k1': 7.5, 'c1': 0.4, 'c2': 0.4, 'k2': 1.4}
cohen_avoid2 = {'name': 'cohen_avoid2', 'ps': None,
               'b1': None, 'k1': None, 'c5': None, 'c6': None,
               'b2': None, 'k2': None, 'c7': None, 'c8': None}


i_iter = 0
                
def Bai_movObst1():
    file = os.path.abspath(os.path.join(os.getcwd(),
                                        os.pardir,
                                        'Raw_Data',
                                        'Bai_movObst1_data.pickle'))
    with open(file, 'rb') as f:
        data = pickle.load(f)
        simulator = ODESimulator(data=data)
    return simulator
    
def error(x, simulator, logfile):
    global i_iter
    i_iter += 1
    print(f'iteration {i_iter:03d}, x = ', [f'{a:0.3f}' for a in x])
    cohen_avoid2 = {'name': 'cohen_avoid2',
             'b1': x[0], 'k1': x[1], 'c5': x[2], 'c6': x[3],
             'b2': x[4], 'k2': x[5], 'c7': x[6], 'c8': x[7]}
    simulator.models = [fajen_approach, cohen_avoid2]
    simulator.simulate_all(subj_id=0)
    err = simulator.test('p_dist')
    print(f'error is {err:.6f}')
    with open(logfile, 'a') as file:
        log = [str(x) for x in [i_iter, simulator.models, err]]
        file.write('\t'.join(log) + '\n')
    return err
    
def main():
    simulator = Bai_movObst1()
    logfile = 'fitting_log_' + ymdhms() + '.txt'
    x0 = np.array([3.25, 530, 6, 1.3, 3.25, 530, 6, 1.3])
    res = minimize(error, x0, args=(simulator, logfile), method='nelder-mead',
                    options={'xatol': 1e-6, 'disp': True, 'adaptive': True})
    print(res.x)

if __name__ == "__main__":
    main()
    