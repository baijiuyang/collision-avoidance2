'''
This module contains the code to generate inputs for the experiments,
which are randomized sequences of conditions.
'''
import random
from itertools import product, accumulate
import numpy as np

# trial list generator
def create_trial_list(IVs, rep, n_freewalk):
    '''
    Args:
        IVs (dict): {'var1': [level1, level2, ...], 'var2': [level1, level2, ...], ...}.
        rep (int): Number of repetitions in each condition.
        n_freewalk (int): Number of freewalk trials.
    
    Return:
        trial_list(list of dicts): Lists of trial configs in the form
            of {'var1': val1, 'var2': val2, ...}. All values are zero for
            freewalk trials.
    '''
    trial_list = []
    # Create test trials
    for combo in product(*IVs.values()):
        one_trial = {var: val for (var, val) in zip(IVs.keys(), combo)}
        trial_list += [one_trial] * rep
    random.shuffle(trial_list)
    N = len(trial_list)
    block_size = N // n_freewalk + 1
    # Add freewalk trials
    i_test = 0
    # Compute the index of freewalk trials
    i_freewalks = np.random.randint(0, block_size - 1, n_freewalk) + \
        np.array(list(accumulate([0] + [block_size] * (n_freewalk - 1))))
    # Insert freewalk trials
    for i in i_freewalks:
        trial_list.insert(i, {var: 0 for var in IVs.keys()})

    return trial_list
    
if __name__ == "__main__":
    # Prepare trial list
    n_subject = 1
    IVs = {'angle': [157.5, 112.5, -157.5, -112.5], 'speed': [0.9, 1.1],
        'dsize': [-0.1, 0, 0.1], 'ipd': [1, 0]}
    rep = 3
    n_freewalk = 10
    practice_trials = create_trial_list({'angle': [157.5, -112.5], 'speed': [1.0],
        'dsize': [0], 'ipd': [0]}, 1, 2)
    n_practice_trials = len(practice_trials)
    for subj in range(n_subject):
        test_trials = create_trial_list(IVs, rep, n_freewalk)
        trials = practice_trials + test_trials
        header = ['i_trial'] + list(IVs.keys())
        with open('Subj' + str(subj).zfill(2) + '.csv', 'a') as file:
            file.write(','.join(header) + '\n')
            for i, trial in enumerate(trials):
                cond = [str(i - n_practice_trials + 1)] + [str(x) for x in list(trial.values())]
                file.write(','.join(cond) + '\n')
    