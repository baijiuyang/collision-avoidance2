'''
This module contains the code for model fitting.
'''
import os
import sys
import argparse
import numpy as np
from scipy import optimize
import pickle
from packages.helper import ymdhms, d_psi, beta, min_dist
from packages.ode_simulator import ODESimulator
from packages import data_container
sys.modules['data_container'] = data_container

# Define approach models for fitting avoidance models
approaches = {'Bai_movObst1':{'fajen_approach': {}, 'fajen_approach2': {}, 'acceleration_approach': {}}}

approaches['Bai_movObst1']['fajen_approach']['dual_annealing'] = {}
approaches['Bai_movObst1']['fajen_approach']['dual_annealing'][-1] = {'name': 'fajen_approach', 'ps': None, 'b1': 3.25, 'k1': 7.5, 'c1': 0.4, 'c2': 0.4, 'k2': 1.4}
approaches['Bai_movObst1']['fajen_approach']['dual_annealing'][0] = {'name': 'fajen_approach', 'b1': 1.3118996599538175, 'k1': 20.05817176956081, 'c1': 2.533333934342072, 'c2': 0.07555087088494414, 'k2': 0.7413974342994933, 'ps': 1.282479471344188}
approaches['Bai_movObst1']['fajen_approach']['dual_annealing'][1] = {'name': 'fajen_approach', 'b1': 0.8081777207885088, 'k1': 32.80043575704669, 'c1': 8.416453555226326, 'c2': 0.03974207207189774, 'k2': 0.8136444335415792, 'ps': 1.329331601172892}
approaches['Bai_movObst1']['fajen_approach']['dual_annealing'][2] = {'name': 'fajen_approach', 'b1': 0.8802412636720522, 'k1': 18.997479919723922, 'c1': 2.756441843773068, 'c2': 0.0760314647580457, 'k2': 0.9161786337522445, 'ps': 1.2233195327965638}
approaches['Bai_movObst1']['fajen_approach']['dual_annealing'][3] = {'name': 'fajen_approach', 'b1': 1.2314286830727728, 'k1': 24.035668545742503, 'c1': 9.856130794680212, 'c2': 0.02862746960028267, 'k2': 0.2117503203346745, 'ps': 0.9872111178298977}
approaches['Bai_movObst1']['fajen_approach']['dual_annealing'][4] = {'name': 'fajen_approach', 'b1': 0.9183592702325633, 'k1': 12.204919745808612, 'c1': 3.43625676096417, 'c2': 0.11664808708772718, 'k2': 0.8674515064029045, 'ps': 1.4286737076029379}
approaches['Bai_movObst1']['fajen_approach']['dual_annealing'][5] = {'name': 'fajen_approach', 'b1': 1.192778944853645, 'k1': 5.267999799725562, 'c1': 5.096005901694298, 'c2': 0.2242800263696644, 'k2': 0.1765803907428338, 'ps': 1.0751463986115937}
approaches['Bai_movObst1']['fajen_approach']['dual_annealing'][6] = {'name': 'fajen_approach', 'b1': 1.0735811813874558, 'k1': 13.140505400771387, 'c1': 9.98596467077732, 'c2': 0.11649019178319157, 'k2': 0.586444306946999, 'ps': 1.31109726329129}
approaches['Bai_movObst1']['fajen_approach']['dual_annealing'][7] = {'name': 'fajen_approach', 'b1': 1.7471982824310022, 'k1': 5.117879113857131, 'c1': 5.741959801035067, 'c2': 0.21995468988652767, 'k2': 0.5773546487545462, 'ps': 1.156007882055671}
approaches['Bai_movObst1']['fajen_approach']['dual_annealing'][8] = {'name': 'fajen_approach', 'b1': 1.1656785334379702, 'k1': 64.75154006900311, 'c1': 2.39434327185181, 'c2': 0.023455554827526214, 'k2': 0.3991609059701808, 'ps': 1.1563239895464452}
approaches['Bai_movObst1']['fajen_approach']['dual_annealing'][9] = {'name': 'fajen_approach', 'b1': 1.1927778066346024, 'k1': 0.20667325443360973, 'c1': 2.2570241856267543, 'c2': 5.4331038962018186, 'k2': 0.37382088279617776, 'ps': 1.1967108147865295}
approaches['Bai_movObst1']['fajen_approach']['dual_annealing'][10] = {'name': 'fajen_approach', 'b1': 1.1996934237333374, 'k1': 11.292256811441096, 'c1': 1.7756010015861, 'c2': 0.17150790956230327, 'k2': 0.4351205520904613, 'ps': 1.056705515676128}

approaches['Bai_movObst1']['fajen_approach']['differential_evolution'] = {}
approaches['Bai_movObst1']['fajen_approach']['differential_evolution'][-1] = {'name': 'fajen_approach', 'ps': None, 'b1': 3.25, 'k1': 7.5, 'c1': 0.4, 'c2': 0.4, 'k2': 1.4}
approaches['Bai_movObst1']['fajen_approach']['differential_evolution'][0] = {'name': 'fajen_approach', 'b1': 1.322399694018276, 'k1': 14.876666639462048, 'c1': 5.4374557807911295, 'c2': 0.10260590846412238, 'k2': 0.7564633997176441, 'ps': 1.282479471344188}
approaches['Bai_movObst1']['fajen_approach']['differential_evolution'][1] = {'name': 'fajen_approach', 'b1': 0.8179416797170797, 'k1': 16.40077656118177, 'c1': 6.417008356831993, 'c2': 0.07993938648972572, 'k2': 0.8075364662966444, 'ps': 1.329331601172892}
approaches['Bai_movObst1']['fajen_approach']['differential_evolution'][2] = {'name': 'fajen_approach', 'b1': 0.8909408398978989, 'k1': 4.641459981824929, 'c1': 5.516867857632004, 'c2': 0.31100646722160424, 'k2': 0.9262495070122934, 'ps': 1.2233195327965638}
approaches['Bai_movObst1']['fajen_approach']['differential_evolution'][3] = {'name': 'fajen_approach', 'b1': 1.2318155085696687, 'k1': 0.44596523668254273, 'c1': 8.660447058122513, 'c2': 1.5467823698282144, 'k2': 0.2064201810751935, 'ps': 0.9872111178298977}
approaches['Bai_movObst1']['fajen_approach']['differential_evolution'][4] = {'name': 'fajen_approach', 'b1': 0.9187400116180737, 'k1': 12.354994295866103, 'c1': 5.138764981999261, 'c2': 0.1150464883117947, 'k2': 0.8664462743167836, 'ps': 1.4286737076029379}
approaches['Bai_movObst1']['fajen_approach']['differential_evolution'][5] = {'name': 'fajen_approach', 'b1': 1.2035286038289252, 'k1': 11.777197150843216, 'c1': 3.208109885882151, 'c2': 0.10077453224090854, 'k2': 0.17721496598944428, 'ps': 1.0751463986115937}
approaches['Bai_movObst1']['fajen_approach']['differential_evolution'][6] = {'name': 'fajen_approach', 'b1': 1.067814211145924, 'k1': 3.442460688134991, 'c1': 8.719678312497393, 'c2': 0.4467038517464913, 'k2': 0.5873392854183639, 'ps': 1.31109726329129}
approaches['Bai_movObst1']['fajen_approach']['differential_evolution'][7] = {'name': 'fajen_approach', 'b1': 1.684749413674761, 'k1': 3.66442070723263, 'c1': 5.8852457414157255, 'c2': 0.2952550401219027, 'k2': 0.5739458333379727, 'ps': 1.156007882055671}
approaches['Bai_movObst1']['fajen_approach']['differential_evolution'][8] = {'name': 'fajen_approach', 'b1': 1.2054388522573258, 'k1': 3.4363263877000607, 'c1': 1.9210228798200861, 'c2': 0.4370998404967321, 'k2': 0.4014795257504464, 'ps': 1.1563239895464452}
approaches['Bai_movObst1']['fajen_approach']['differential_evolution'][9] = {'name': 'fajen_approach', 'b1': 1.1936413326262518, 'k1': 1.5253260080485234, 'c1': 4.71969731760836, 'c2': 0.7389384145132375, 'k2': 0.37612127377107535, 'ps': 1.1967108147865295}
approaches['Bai_movObst1']['fajen_approach']['differential_evolution'][10] = {'name': 'fajen_approach', 'b1': 1.2398029027292385, 'k1': 14.369326267732422, 'c1': 0.7651801554578793, 'c2': 0.13957426997817068, 'k2': 0.4360750101027109, 'ps': 1.056705515676128}

approaches = {'Fajen_steer1a':{'fajen_approach': {}, 'fajen_approach2': {}, 'acceleration_approach': {}}}

approaches['Fajen_steer1a']['fajen_approach']['differential_evolution'] = {}
approaches['Fajen_steer1a']['fajen_approach']['differential_evolution'][-1] = {'name': 'fajen_approach', 'b1':2.01938384, 'k1':4.90527274, 'c1':2.96094879, 'c2':0.50896525, 'k2':1.61216734}

approaches['Fajen_steer1a']['fajen_approach2']['differential_evolution'] = {}
approaches['Fajen_steer1a']['fajen_approach2']['differential_evolution'][-1] = {'name': 'fajen_approach2', 'b1':2.04992354, 'k1':2.85641543, 'c1':0.54294928, 'c2':0.73857217, 'b2':3.89580222, 'k2':5.04511601}


model_bounds = {}
model_bounds['fajen_approach'] = [(0, 10), (0, 20), (0, 5), (0, 5), (0, 5)]
model_bounds['fajen_approach2'] = [(0, 10), (0, 20), (0, 5), (0, 5), (0, 10), (0, 20)]
model_bounds['cohen_avoid'] = [(0, 50), (0, 800), (0.1, 20), (0.1, 10), (0, 50), (0, 800), (0.1, 10), (0.1, 10)]
model_bounds['cohen_avoid4_thres'] = [(0, 100), (0.1, 10), (1, 30), (0, 100), (0.1, 10), (1, 30), (0, 0.1)]


i_iter = 0

def create_cmd_args():
    parser = argparse.ArgumentParser(description='Arguments for model training.')
    parser.add_argument('--experiment_name')
    parser.add_argument('--approach_experiment')
    parser.add_argument('--subject', type=int)
    parser.add_argument('--t_start')
    parser.add_argument('--t_end')
    parser.add_argument('--ps')
    parser.add_argument('--method')
    parser.add_argument('--approach_model')
    parser.add_argument('--avoid_model')
    parser.add_argument('--training_model')
    args = parser.parse_args()
    if not any(args.__dict__.values()):
        return False
    args.training_model_type = args.training_model.split('_')[1][:3]
    return args
    
def get_input_args():
    args = argparse.Namespace()
    args.experiment_name = input('Type experiment_name (Bai_movObst1):\n')
    arg.approach_experiment = input('Type approach_experiment (Bai_movObst1b):\n')
    args.subject = int(input('Type subject number:\n'))
    args.t_start = input('Type t_start (obst_onset/match_order/obst_out):\n')
    args.t_end = input('Type t_end (obst_out/end):\n') 
    args.ps = input('Type preferred speed (subj/var0):\n') 
    method = {1: 'dual_annealing', 2: 'differential_evolution'}
    args.method = method[int(input('Choose optimizer: 1 -- dual_annealing 2 -- differential_evolution:\n'))]
    args.approach_model = input('Type approach_model (fajen_approach/acceleration_approach/jerk_approach):\n') 
    args.avoid_model = input('Type avoid_model (cohen_avoid):\n')
    args.training_model = input('Type training_model:\n')
    args.training_model_type = args.training_model.split('_')[1][:3]
    return args

def Bai_movObst1(subject):
    file = os.path.abspath(os.path.join(os.getcwd(),
                                        os.pardir,
                                        'Raw_Data',
                                        'Bai_movObst1_data.pickle'))
    with open(file, 'rb') as f:
        data = pickle.load(f)
    trials = []
    for i in range(len(data.trajs)):
        if ((data.info['subj_id'][i] == subject or subject == -1) and
            data.info['obst_speed'][i] != 0 and
            abs(data.info['obst_angle'][i]) != 180):
            trials.append(i)
    simulator = ODESimulator(data=data)
    return simulator, trials

def Bai_movObst1b():
    pass

def Fajen_steer1a(subject):
    file = os.path.abspath(os.path.join(os.getcwd(),
                                        os.pardir,
                                        'Raw_Data',
                                        'Fajen_steer_exp1a_data.pickle'))
    with open(file, 'rb') as f:
        data = pickle.load(f)
    trials = []
    for i in range(len(data.trajs)):
        if ((data.info['subj_id'][i] == subject or subject == -1) and
            data.info['ps_trial'][i] and
            (i not in data.dump)):
            trials.append(i)
    simulator = ODESimulator(data=data)
    return simulator, trials
    
def Cohen_movObst1(subject):
    file = os.path.abspath(os.path.join(os.getcwd(),
                                        os.pardir,
                                        'Raw_Data',
                                        'Cohen_movObst_exp1_data.pickle'))
    with open(file, 'rb') as f:
        data = pickle.load(f)
    trials = []
    for i in range(len(data.trajs)):
        if ((data.info['subj_id'][i] == subject or subject == -1) and
            data.info['obst_speed'][i] and
            data.info['ps_trial'][i] and
            (i not in data.dump)):
            trials.append(i)
    simulator = ODESimulator(data=data)
    return simulator, trials

def build_simulator(experiment_name, subject):
    if experiment_name == 'Bai_movObst1':
        return Bai_movObst1(subject)
    elif experiment_name == 'Bai_movObst1b':
        return Bai_movObst1b(subject)
    elif experiment_name == 'Cohen_movObst1':
        return Cohen_movObst1(subject)
    elif experiment_name == 'Fajen_steer1a':
        return Fajen_steer1a(subject)
    else:
        print('experiment_name invalid')

def error(x, simulator, trials, logfile, args):
    global i_iter
    i_iter += 1
    print(f'iter {i_iter:03d}, subj{args.subject}, {args.training_model}, '
        f'x = ', [f'{a:0.3f}' for a in x])
    if args.training_model_type == 'app':
        if args.approach_model == 'fajen_approach':
            simulator.models = [{'name': 'fajen_approach',
                 'b1': x[0], 'k1': x[1], 'c1': x[2], 'c2': x[3], 'k2': x[4]}]
        elif args.approach_model == 'fajen_approach2':
            simulator.models = [{'name': 'fajen_approach2',
                 'b1': x[0], 'k1': x[1], 'c1': x[2], 'c2': x[3], 'b2': x[4], 'k2': x[5]}]
        elif args.approach_model == 'acceleration_approach':
            print('empty approach model')
            pass
        else:
            print('approach_model invalid')
    elif args.training_model_type == 'avo':
        approach = approaches[args.approach_experiment][args.approach_model][args.method][args.subject]
        if args.avoid_model == 'cohen_avoid':
            avoid = {'name': 'cohen_avoid',
             'b1': x[0], 'k1': x[1], 'c5': x[2], 'c6': x[3],
             'b2': x[4], 'k2': x[5], 'c7': x[6], 'c8': x[7]}
        elif args.avoid_model == 'cohen_avoid4_thres':
            avoid = {'name': 'cohen_avoid4_thres',
             'k1': x[0], 'c5': x[1], 'c6': x[2],
             'k2': x[3], 'c7': x[4], 'c8': x[5], 'thres': x[6]}
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
    with open(logfile, 'a') as file:
        log = [str(x) for x in [i_iter, simulator.models, err, 'order_accuracy', accuracy]]
        file.write('\t'.join(log) + '\n')
    simulator.reset()
    return err

def main():
    args = create_cmd_args()
    if not args:
        args = get_input_args()
    notes = 'Excluded free-walk trials, excluded 180 trials'
    logfile = 'fitting_log_' + str(args.subject) + '_' + ymdhms() + '.txt'
    bounds = model_bounds[args.training_model]
    simulator, trials = build_simulator(args.experiment_name, args.subject)
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
        res = optimize.differential_evolution(error, bounds, args=(simulator, trials, logfile, args))
    elif args.method == 'basinhopping':
        res = optimize.basinhopping(error, bounds, minimizer_kwargs={'args':(simulator, trials, logfile, args)})
    with open(logfile, 'a') as file:
        file.write(f'The optimal x: {res.x}')
    print(res.x)

if __name__ == "__main__":
    main()
