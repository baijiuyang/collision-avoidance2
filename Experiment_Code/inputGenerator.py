import random

# trial list generator
def create_trial_list(angles, speeds, rep, n_freewalk):
    '''
    Args:
        angles (list): Angles of interception relative to subject's
            initial heading in degree. Clockwise rotation is positive.
        speeds (list): Speeds of moving obstacle in m/s.
        rep (int): Number of repetitions in each condition.
        n_freewalk (int): Number of freewalk trials.
    
    Return:
        trial_list(list of dict): Lists of trial configs in the form
            of {'angle': float, 'speed': float}. The speed is zero in
            freewalk trials.
    '''
    trial_list = []
    test_trials = []
    # Create test trials
    for angle in angles:
        for speed in speeds:
            test_trials += [{'angle': angle, 'speed': speed}] * rep
    random.shuffle(test_trials)
    
    N = len(angles) * len(speeds) * rep
    block_size = N // n_freewalk + 1
    # Add freewalk trials
    i_test = 0
    
    for _ in range(n_freewalk):
        i_freewalk = random.randint(0, block_size - 1)
        for j in range(block_size):
            if j == i_freewalk:
                trial_list += [{'angle': 0, 'speed': 0}]
            else:
                trial_list.append(test_trials[i_test])
                i_test += 1
    return trial_list
    
if __name__ == "__main__":
    # Prepare trial list
    n_subject = 20
    angles = [180, 157.5, 135, 112.5, 90, -180, -157.5, -135, -112.5, -90]
    speeds = [0.9, 1.0, 1.1, 1.2, 1.3]
    rep = 3
    n_freewalk = 10
    practice_trials = create_trial_list([135, -135], [1.0], 1, 2)
    for subj in range(n_subject):
        test_trials = create_trial_list(angles, speeds, rep, n_freewalk)
        trials = practice_trials + test_trials
        header = ['i_trial', 'angle', 'speed']
        with open('Subj' + str(subj).zfill(2) + '.csv', 'a') as file:
            file.write(','.join(header) + '\n')
            for i, trial in enumerate(trials):
                cond = [str(x) for x in [i - 3, trial['angle'], trial['speed']]]
                file.write(','.join(cond) + '\n')
    