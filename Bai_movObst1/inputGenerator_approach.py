'''
This module contains the code to generate inputs for the experiments,
which are randomized sequences of conditions.
'''

import random
from itertools import product

# trial list generator
def create_trial_list(conditions, rep, n_freewalk):
    '''
    Args:
        conditions (List[List]): Levels of independent variables.
        rep (int): Number of repetitions in each condition.
        n_freewalk (int): Number of freewalk trials.

    Return:
        trial_list(list of dict): Lists of trial configs in the form
            of {'angle': float, 'speed': float}. The speed is zero in
            freewalk trials.
    '''
    trial_list = []
    # Create test trials
    test_trials = list(product(*conditions)) * rep  # '*' means it is the list argument for the function.
    random.shuffle(test_trials)
    N = len(test_trials)
    block_size = N // n_freewalk + 1
    # Add freewalk trials
    i_test = 0
    for _ in range(n_freewalk):
        i_freewalk = random.randint(0, block_size - 1)
        for j in range(block_size):
            if j == i_freewalk:
                trial_list.append([0] * len(test_trials[0]))
            else:
                trial_list.append(test_trials[i_test])
                i_test += 1
    return trial_list
    
if __name__ == "__main__":
    # Prepare trial list
    n_subject = 20
    s0s = [0.5, 1.4]
    d0s = [4, 8]
    angles = [-20, -15, -10, 10, 15, 20]
    rep = 5
    n_freewalk = 10
    practice_trials = [[0.5, 3, -20], [0.5, 6, 15], [1.4, 3, -10], [1.4, 6, 20]]
    for subj in range(n_subject):
        test_trials = create_trial_list([s0s, d0s, angles], rep, n_freewalk)
        trials = practice_trials + test_trials
        header = ['i_trial', 's0', 'd0', 'angle']
        with open('Approach_subj' + str(subj).zfill(2) + '.csv', 'a') as file:
            file.write(','.join(header) + '\n')
            for i, trial in enumerate(trials):
                cond = [str(x) for x in [i - 3] + list(trial)]
                file.write(','.join(cond) + '\n')
    