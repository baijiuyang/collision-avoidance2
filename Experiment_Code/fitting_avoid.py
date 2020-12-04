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

# Dual_annealing
approaches = []
approaches.append({'name': 'fajen_approach', 'b1': 1.3118996599538175, 'k1': 20.05817176956081, 'c1': 2.533333934342072, 'c2': 0.07555087088494414, 'k2': 0.7413974342994933, 'ps': 1.282479471344188})
approaches.append({'name': 'fajen_approach', 'b1': 0.8081777207885088, 'k1': 32.80043575704669, 'c1': 8.416453555226326, 'c2': 0.03974207207189774, 'k2': 0.8136444335415792, 'ps': 1.329331601172892})
approaches.append({'name': 'fajen_approach', 'b1': 0.8802412636720522, 'k1': 18.997479919723922, 'c1': 2.756441843773068, 'c2': 0.0760314647580457, 'k2': 0.9161786337522445, 'ps': 1.2233195327965638})
approaches.append({'name': 'fajen_approach', 'b1': 1.2314286830727728, 'k1': 24.035668545742503, 'c1': 9.856130794680212, 'c2': 0.02862746960028267, 'k2': 0.2117503203346745, 'ps': 0.9872111178298977})
approaches.append({'name': 'fajen_approach', 'b1': 0.9183592702325633, 'k1': 12.204919745808612, 'c1': 3.43625676096417, 'c2': 0.11664808708772718, 'k2': 0.8674515064029045, 'ps': 1.4286737076029379})
approaches.append({'name': 'fajen_approach', 'b1': 1.192778944853645, 'k1': 5.267999799725562, 'c1': 5.096005901694298, 'c2': 0.2242800263696644, 'k2': 0.1765803907428338, 'ps': 1.0751463986115937})
approaches.append({'name': 'fajen_approach', 'b1': 1.0735811813874558, 'k1': 13.140505400771387, 'c1': 9.98596467077732, 'c2': 0.11649019178319157, 'k2': 0.586444306946999, 'ps': 1.31109726329129})
approaches.append({'name': 'fajen_approach', 'b1': 1.7471982824310022, 'k1': 5.117879113857131, 'c1': 5.741959801035067, 'c2': 0.21995468988652767, 'k2': 0.5773546487545462, 'ps': 1.156007882055671})
approaches.append({'name': 'fajen_approach', 'b1': 1.1656785334379702, 'k1': 64.75154006900311, 'c1': 2.39434327185181, 'c2': 0.023455554827526214, 'k2': 0.3991609059701808, 'ps': 1.1563239895464452})
approaches.append({'name': 'fajen_approach', 'b1': 1.1927778066346024, 'k1': 0.20667325443360973, 'c1': 2.2570241856267543, 'c2': 5.4331038962018186, 'k2': 0.37382088279617776, 'ps': 1.1967108147865295})
approaches.append({'name': 'fajen_approach', 'b1': 1.1996934237333374, 'k1': 11.292256811441096, 'c1': 1.7756010015861, 'c2': 0.17150790956230327, 'k2': 0.4351205520904613, 'ps': 1.056705515676128})

# Differenttial evolution
# approaches = []
# approaches.append({'name': 'fajen_approach', 'b1': 1.322399694018276, 'k1': 14.876666639462048, 'c1': 5.4374557807911295, 'c2': 0.10260590846412238, 'k2': 0.7564633997176441, 'ps': 1.282479471344188})
# approaches.append({'name': 'fajen_approach', 'b1': 0.8179416797170797, 'k1': 16.40077656118177, 'c1': 6.417008356831993, 'c2': 0.07993938648972572, 'k2': 0.8075364662966444, 'ps': 1.329331601172892})
# approaches.append({'name': 'fajen_approach', 'b1': 0.8909408398978989, 'k1': 4.641459981824929, 'c1': 5.516867857632004, 'c2': 0.31100646722160424, 'k2': 0.9262495070122934, 'ps': 1.2233195327965638})
# approaches.append({'name': 'fajen_approach', 'b1': 1.2318155085696687, 'k1': 0.44596523668254273, 'c1': 8.660447058122513, 'c2': 1.5467823698282144, 'k2': 0.2064201810751935, 'ps': 0.9872111178298977})
# approaches.append({'name': 'fajen_approach', 'b1': 0.9187400116180737, 'k1': 12.354994295866103, 'c1': 5.138764981999261, 'c2': 0.1150464883117947, 'k2': 0.8664462743167836, 'ps': 1.4286737076029379})
# approaches.append({'name': 'fajen_approach', 'b1': 1.2035286038289252, 'k1': 11.777197150843216, 'c1': 3.208109885882151, 'c2': 0.10077453224090854, 'k2': 0.17721496598944428, 'ps': 1.0751463986115937})
# approaches.append({'name': 'fajen_approach', 'b1': 1.067814211145924, 'k1': 3.442460688134991, 'c1': 8.719678312497393, 'c2': 0.4467038517464913, 'k2': 0.5873392854183639, 'ps': 1.31109726329129})
# approaches.append({'name': 'fajen_approach', 'b1': 1.684749413674761, 'k1': 3.66442070723263, 'c1': 5.8852457414157255, 'c2': 0.2952550401219027, 'k2': 0.5739458333379727, 'ps': 1.156007882055671})
# approaches.append({'name': 'fajen_approach', 'b1': 1.2054388522573258, 'k1': 3.4363263877000607, 'c1': 1.9210228798200861, 'c2': 0.4370998404967321, 'k2': 0.4014795257504464, 'ps': 1.1563239895464452})
# approaches.append({'name': 'fajen_approach', 'b1': 1.1936413326262518, 'k1': 1.5253260080485234, 'c1': 4.71969731760836, 'c2': 0.7389384145132375, 'k2': 0.37612127377107535, 'ps': 1.1967108147865295})
# approaches.append({'name': 'fajen_approach', 'b1': 1.2398029027292385, 'k1': 14.369326267732422, 'c1': 0.7651801554578793, 'c2': 0.13957426997817068, 'k2': 0.4360750101027109, 'ps': 1.056705515676128})

avoid = {'name': 'cohen_avoid', 'ps': None,
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
    
def error(x, simulator, trials, logfile):
    global i_iter
    i_iter += 1
    print(f'iteration {i_iter:03d}, x = ', [f'{a:0.3f}' for a in x])
    avoid = {'name': 'cohen_avoid',
             'b1': x[0], 'k1': x[1], 'c5': x[2], 'c6': x[3],
             'b2': x[4], 'k2': x[5], 'c7': x[6], 'c8': x[7]}
    simulator.models = [approach, avoid]    
    simulator.simulate_all(trials=trials, t_start='match_order', t_end='obst_out', ps='subj')
    err = simulator.test('p_dist')
    accuracy = simulator.test('order_accuracy')
    print(f'error is {err:.6f}, accuracy is {accuracy}')
    with open(logfile, 'a') as file:
        log = [str(x) for x in [i_iter, simulator.models, err, 'order_accuracy', accuracy]]
        file.write('\t'.join(log) + '\n')
    simulator.reset()
    return err
    
def main():
    global approach
    method = {1: 'dual_annealing', 2: 'differential_evolution'}
    method = method[int(input('Choose optimizer: 1 -- dual_annealing 2 -- differential_evolution'))]
    subject = int(input('Type subject number: '))
    approach = approaches[subject]
    notes = 'Exclude free-walk trials, exclude 180 trials, \
             start simuation when passing order is predictable, \
             use subject level preferred speed'
    logfile = 'fitting_log_' + ymdhms() + '.txt'
    bounds = [(0, 50), (0, 800), (0.1, 20), (0.1, 10), (0, 50), (0, 800), (0.1, 10), (0.1, 10)]
    x0 = np.array([3.25, 530, 6, 1.3, 3.25, 530, 6, 1.3])
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
        file.write(f'The optimal x: {res.x}')
    print(res.x)

if __name__ == "__main__":
    main()
    