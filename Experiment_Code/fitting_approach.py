'''
This module contains the code for model fitting.
'''
import os
import sys
import numpy as np
from scipy import optimize
import pickle
from packages.helper import ymdhms, d_psi, beta, min_dist
from packages.ode_simulator import ODESimulator
from packages import data_container
sys.modules['data_container'] = data_container

approach = {'name': 'fajen_approach', 'ps': None,
                  'b1': 3.25, 'k1': 7.5, 'c1': 0.4, 'c2': 0.4, 'k2': 1.4}

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
    
def error(x, simulator, trials, logfile):
    global i_iter
    i_iter += 1
    print(f'iteration {i_iter:03d}, x = ', [f'{a:0.3f}' for a in x])
    approach = {'name': 'fajen_approach',
             'b1': x[0], 'k1': x[1], 'c1': x[2], 'c2': x[3], 'k2': x[4]}
    simulator.models = [approach]    
    simulator.simulate_all(trials=trials, t_start='obst_out', t_end='end', ps='subj')
    err = simulator.test('p_dist')
    print(f'error is {err:.6f}')
    with open(logfile, 'a') as file:
        log = [str(x) for x in [i_iter, simulator.models, err]]
        file.write('\t'.join(log) + '\n')
    simulator.reset()
    return err
    
def main():
    method = 'dual_annealing'
    subject = 0
    notes = 'Exclude free-walk trials, exclude 180 trials, \
             start simuation when passing order is predictable, \
             use subject preferred speed'
    logfile = 'fitting_log_' + ymdhms() + '.txt'
    bounds = [(0, 30), (0, 70), (0, 10), (0, 10), (0, 10)]
    # x0 = np.array([3.25, 530, 6, 1.3, 3.25, 530, 6, 1.3])
    trials = []
    simulator = Bai_movObst1()
    for i in range(len(simulator.data.trajs)):
        if (simulator.data.info['subj_id'][i] == subject and
            simulator.data.info['obst_speed'][i] != 0 and
            abs(simulator.data.info['obst_angle'][i]) != 180):
            trials.append(i)

    with open(logfile, 'a') as file:
        file.write(f'method: {method}\nbounds: {bounds}\nsubject: {subject}\ntrials: {trials}\nnotes: {notes}\n')            
    if method == 'nelder-mead':
        res = optimize.minimize(error, x0, args=(simulator, trials, logfile), method='nelder-mead',
                        options={'xatol': 1e-6, 'disp': True, 'adaptive': True})
    elif method == 'shgo':
        res = optimize.shgo(error, bounds, args=(simulator, trials, logfile))
    elif method == 'dual_annealing':
        res = optimize.dual_annealing(error, bounds, args=(simulator, trials, logfile),
                                      initial_temp=25000,
                                      visit=2.8)
    elif method == 'differential_evolution':
        res = optimize.differential_evolution(error, bounds, args=(simulator, trials, logfile))
    elif method == 'basinhopping':
        res = optimize.basinhopping(error, bounds, minimizer_kwargs={'args':(simulator, trials, logfile)})
    with open(logfile, 'a') as file:
        file.write(f'The optimal x: {res.x:str}')
    print(res.x)

if __name__ == "__main__":
    main()
    