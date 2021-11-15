'''
This module contains the code for model fitting.
'''
import os
import sys
import time
import argparse
import numpy as np
from scipy import optimize
import pickle
from packages.helper import ymdhms, hms, d_psi, beta, min_dist
from packages.ode_simulator import ODESimulator
from packages import data_container
from parameters import approaches, model_bounds
sys.modules['data_container'] = data_container


i_iter = 0

def create_cmd_args():
    parser = argparse.ArgumentParser(description='Arguments for model training.')
    parser.add_argument('--experiment_name')
    parser.add_argument('--approach_experiment')
    parser.add_argument('--subject')
    parser.add_argument('--t_start')
    parser.add_argument('--t_end')
    parser.add_argument('--ps')
    parser.add_argument('--method')
    parser.add_argument('--approach_model')
    parser.add_argument('--avoid_model')
    parser.add_argument('--training_model')
    parser.add_argument('--trial')
    args = parser.parse_args()
    if not any(args.__dict__.values()):
        return False
    args.training_model_type = args.training_model.split('_')[1][:3]
    return args

def Bai_movObst1(subjects):
    file = os.path.abspath(os.path.join(os.getcwd(),
                                        os.pardir,
                                        'Raw_Data',
                                        'Bai_movObst1_data_30Hz.pickle'))
    with open(file, 'rb') as f:
        data = pickle.load(f)
    sim = ODESimulator(data=data, ref=[0,1])
    trials = []
    for i in range(len(data.trajs)):
        if (data.info['subj_id'][i] in subjects and
            data.info['obst_speed'][i] != 0 and
            abs(data.info['obst_angle'][i]) != 180 and
            i not in data.dump):
            trials.append(i)
    simulator = ODESimulator(data=data)
    print(trials)
    return simulator, trials

def Bai_movObst1b(subjects):
    file = os.path.abspath(os.path.join(os.getcwd(),
                                        os.pardir,
                                        'Raw_Data',
                                        'Bai_movObst1b_data_30Hz.pickle'))
    with open(file, 'rb') as f:
        data = pickle.load(f)
    trials = []
    for i in range(len(data.trajs)):
        if (data.info['subj_id'][i] in subjects and
            data.info['leader_s0'][i] != 0 and
            i not in data.dump):
            trials.append(i)
    simulator = ODESimulator(data=data)
    return simulator, trials

def Fajen_steer1a(subjects):
    file = os.path.abspath(os.path.join(os.getcwd(),
                                        os.pardir,
                                        'Raw_Data',
                                        'Fajen_steer1a_data.pickle'))
    with open(file, 'rb') as f:
        data = pickle.load(f)
    trials = []
    for i in range(len(data.trajs)):
        if (data.info['subj_id'][i] in subjects and
            data.info['ps_trial'][i] and
            i not in data.dump):
            trials.append(i)
    simulator = ODESimulator(data=data)
    return simulator, trials
    
def Cohen_movObst1(subjects):
    file = os.path.abspath(os.path.join(os.getcwd(),
                                        os.pardir,
                                        'Raw_Data',
                                        'Cohen_movObst1_data.pickle'))
    with open(file, 'rb') as f:
        data = pickle.load(f)
    trials = []
    for i in range(len(data.trajs)):
        if (data.info['subj_id'][i] in subjects and
            i not in data.dump):
            trials.append(i)
    simulator = ODESimulator(data=data)
    return simulator, trials

def Cohen_movObst2(subjects):
    file = os.path.abspath(os.path.join(os.getcwd(),
                                        os.pardir,
                                        'Raw_Data',
                                        'Cohen_movObst2_data.pickle'))
    with open(file, 'rb') as f:
        data = pickle.load(f)
    trials = []
    for i in range(len(data.trajs)):
        if ((data.info['subj_id'][i] in subjects) and
            i not in data.dump):
            trials.append(i)
    simulator = ODESimulator(data=data)
    return simulator, trials

def build_simulator(args, subjects):
    if args.experiment_name == 'Bai_movObst1':
        simulator = Bai_movObst1(subjects)
    elif args.experiment_name == 'Bai_movObst1b':
        simulator = Bai_movObst1b(subjects)
    elif args.experiment_name == 'Cohen_movObst1':
        simulator = Cohen_movObst1(subjects)
    elif args.experiment_name == 'Cohen_movObst2':
        simulator = Cohen_movObst2(subjects)
    elif args.experiment_name == 'Fajen_steer1a':
        simulator = Fajen_steer1a(subjects)
    else:
        raise Exception('experiment_name invalid')
    return simulator
    
def error(x, simulator, trials, logfile, args):
    global i_iter
    i_iter += 1
    print(f'iter {i_iter:03d}, subj{args.subject}, {args.training_model}, '
        f'x = ', [f'{a:0.3f}' for a in x])
    tic = time.perf_counter()
    if args.training_model_type == 'app':
        if args.approach_model == 'fajen_approach':
            simulator.models = [{'name': 'fajen_approach',
                 'b1': x[0], 'k1': x[1], 'c1': x[2], 'c2': x[3], 'k2': x[4]}]
        elif args.approach_model == 'fajen_approach2':
            simulator.models = [{'name': 'fajen_approach2',
                 'b1': x[0], 'k1': x[1], 'c1': x[2], 'c2': x[3], 'b2': x[4], 'k2': x[5]}]
        elif args.approach_model == 'acceleration_approach':
            simulator.models = [{'name': 'acceleration_approach',
                 'k': x[0]}]
        elif args.approach_model == 'jerk_approach':
            simulator.models = [{'name': 'jerk_approach',
                 'k': x[0], 'b': x[1]}]
        else:
            print('approach_model invalid')
    elif args.training_model_type == 'avo':
        if args.subject == 'all' or args.subject[0] == '-':
            approach = approaches[args.approach_experiment][args.approach_model][args.method][-1]
        else:
            approach = approaches[args.approach_experiment][args.approach_model][args.method][int(args.subject)]            
        if args.avoid_model == 'cohen_avoid':
            avoid = {'name': 'cohen_avoid',
             'b1': x[0], 'k1': x[1], 'c5': x[2], 'c6': x[3],
             'b2': x[4], 'k2': x[5], 'c7': x[6], 'c8': x[7]}
        elif args.avoid_model == 'cohen_avoid_heading':
            avoid = {'name': 'cohen_avoid_heading',
             'b1': x[0], 'k1': x[1], 'c5': x[2], 'c6': x[3]}
        elif args.avoid_model == 'cohen_avoid2':
            avoid = {'name': 'cohen_avoid2',
             'b1': x[0], 'k1': x[1], 'c5': x[2], 'c6': x[3],
             'b2': x[4], 'k2': x[5], 'c7': x[6], 'c8': x[7]}
        elif args.avoid_model == 'cohen_avoid3':
            avoid = {'name': 'cohen_avoid3',
             'k1': x[0], 'c5': x[1], 'c6': x[2],
             'k2': x[3], 'c7': x[4], 'c8': x[5]}
        elif args.avoid_model == 'cohen_avoid4':
            avoid = {'name': 'cohen_avoid4',
             'k1': x[0], 'c5': x[1], 'c6': x[2],
             'k2': x[3], 'c7': x[4], 'c8': x[5]}
        elif args.avoid_model == 'cohen_avoid4_heading':
            avoid = {'name': 'cohen_avoid4_heading',
             'k1': x[0], 'c5': x[1], 'c6': x[2]}
        elif args.avoid_model == 'cohen_avoid4_thres':
            avoid = {'name': 'cohen_avoid4_thres',
             'k1': x[0], 'c5': x[1], 'c6': x[2],
             'k2': x[3], 'c7': x[4], 'c8': x[5], 'thres': x[6]}
        elif args.avoid_model == 'perpendicular_avoid':
            avoid = {'name': 'perpendicular_avoid',
                'k': x[0], 'c': x[1]}
        elif args.avoid_model == 'perpendicular_avoid2':
            avoid = {'name': 'perpendicular_avoid2',
                'k': x[0], 'c': x[1]}
        else:
            print('avoid_model invalid')
        simulator.models = [approach, avoid]
    else:
        print('training model type is invalid')  
    simulator.simulate_all(trials=trials, t_start=args.t_start, t_end=args.t_end, ps=args.ps)
    err = simulator.test('p_dist')
    if args.training_model_type == 'avo':
        accuracy = simulator.test('order_accuracy')
    else:
        accuracy = None
    print(f'error is {err:.6f}, accuracy is {accuracy}')
    toc = time.perf_counter()
    with open(logfile, 'a') as file:
        log = [str(x) for x in [i_iter, simulator.models, err, 'order_accuracy', accuracy, hms(toc-tic)]]
        try:
            file.write('\t'.join(log) + '\n')
        except:
            print(log)
    simulator.reset()
    return err

def main():
    args = create_cmd_args()
    # Select subjects for fitting
    if args.subject == 'all':
        # Fit model to all subjects
        subjects = range(100)
    elif args.subject[0] == '-':
        # For leave-one-subject-out cross validation
        subjects = [int(x) for x in range(100) if x != -int(args.subject)]
    else:
        # For single subject fitting
        subjects = [int(args.subject)]
    notes = 'na'
    bounds = model_bounds[args.training_model]
    simulator, trials = build_simulator(args, subjects)
    logfile = 'fitting_log_' + ymdhms() + '_' + str(args.subject) + '.txt'
    with open(logfile, 'a') as file:
        file.write(f'experiment_name: {args.experiment_name}\nsubject: {args.subject}\ntrials: {trials}\n'
            f'approach_experiment: {args.approach_experiment}\n'
            f'approach_model: {args.approach_model}, avoid_model: {args.avoid_model}\n'
            f'training_model: {args.training_model}\n'            
            f'method: {args.method}\nbounds: {bounds}\nt_start: {args.t_start}, t_end: {args.t_end}\n'
            f'preferred speed: {args.ps}\nnotes: {notes}\n')              
    if args.method == 'nelder-mead':
        res = optimize.minimize(error, x0, args=(simulator, trials, logfile, args), method='nelder-mead',
                        options={'xatol': 1e-6, 'disp': True, 'adaptive': True})
    elif args.method == 'shgo':
        res = optimize.shgo(error, bounds, args=(simulator, trials, logfile, args))
    elif args.method == 'dual_annealing':
        res = optimize.dual_annealing(error, bounds, args=(simulator, trials, logfile, args),
                                      initial_temp=25000)
    elif args.method == 'differential_evolution':
        res = optimize.differential_evolution(error, bounds, args=(simulator, trials, logfile, args),
                                    updating='immediate', workers=1)
    elif args.method == 'basinhopping':
        res = optimize.basinhopping(error, bounds, minimizer_kwargs={'args':(simulator, trials, logfile, args)})
    with open(logfile, 'a') as file:
        file.write(f'The optimal x: {res.x}')
    print(res.x)

if __name__ == "__main__":
    main()
