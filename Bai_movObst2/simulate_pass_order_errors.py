'''
simulate_pass_order_errors.py
=============================================================================
Reusable simulation script for the two Cohen experiments:
    Exp 1 -> Cohen_movObst1   (variables: obstacle angle, obstacle speed)
    Exp 2 -> Cohen_movObst2   (variables: obstacle angle, obstacle init. dist)

For each experiment and each avoidance model (cohen_avoid, cohen_avoid2), the
script re-runs the collision-avoidance model over every (subject, trial) using
the all-subject fitted parameters stored in ``parameters.py`` and records, per
trial:

    subj_id, trial_id,
    <experiment variables>,          # obst_angle + (obst_speed | obst_dist)
    approach_model, avoidance_model,
    dist_error,                      # mean model-vs-subject position error (m)
    pred_pass_order,                 # model's passing order   (+1 ahead / -1 behind)
    actual_pass_order,               # subject's passing order (+1 / -1)
    pass_order_correct               # 1 if pred == actual else 0

Two CSV files are written (one per experiment):
    Cohen_movObst1_simulation_pass_order_errors.csv
    Cohen_movObst2_simulation_pass_order_errors.csv

The approach model (fajen_approach) is shared by both experiments and both
avoidance models. The passing-order convention is identical to the simulator's
own ``test('order_accuracy')`` (both quantities live on the same -1/+1 scale),
so ``mean(pass_order_correct)`` reproduces that accuracy.

Usage (must run inside the project's `avoidance` conda environment, from the
project directory so `fitting`/`parameters` import correctly):

    conda activate avoidance
    python simulate_pass_order_errors.py                 # all subjects, both experiments
    python simulate_pass_order_errors.py --subjects 1,2  # quick subset for testing
    python simulate_pass_order_errors.py --exp Cohen_movObst2
=============================================================================
'''
import os
import copy
import csv
import argparse
import numpy as np

from fitting import Cohen_movObst1, Cohen_movObst2
from parameters import approaches, avoidances


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
# Shared approach model (all subjects, key -1), used for BOTH experiments
# and BOTH avoidance models.
APPROACH = approaches['Fajen_steer1a']['fajen_approach']['differential_evolution'][-1]

# Avoidance models to simulate.
AVOID_MODELS = ['cohen_avoid', 'cohen_avoid2']

# Key of the all-subject fit inside avoidances[exp][model]['differential_evolution'].
PARAM_KEY = '-1obst_onset'

# Simulation window (same as the cross-validation / manuscript analysis).
T_START, T_END = 'obst_onset', 'obst_out'

# One entry per experiment: data loader + the experiment-specific variables
# recorded in the output (in addition to obst_angle, which both share).
EXPERIMENTS = {
    'Cohen_movObst1': {'loader': Cohen_movObst1, 'vars': ['obst_angle', 'obst_speed']},
    'Cohen_movObst2': {'loader': Cohen_movObst2, 'vars': ['obst_angle', 'obst_dist']},
}


def _sign_pm1(x):
    '''Return the passing-order sign as an int (+1 / -1; 0 only for exact ties).'''
    return int(np.sign(x))


def run_experiment(exp_name, subjects):
    '''Simulate one experiment for every avoidance model and return result rows.

    Returns
    -------
    rows : list of dict   (one dict per simulated trial per avoidance model)
    exp_vars : list of str  (the experiment-specific variable columns)
    '''
    cfg = EXPERIMENTS[exp_name]
    exp_vars = cfg['vars']
    sim, trials = cfg['loader'](subjects)
    data = sim.data

    rows = []
    for avoid_name in AVOID_MODELS:
        avoid = avoidances[exp_name][avoid_name]['differential_evolution'][PARAM_KEY]

        sim.reset()
        # Deep-copy the model dicts: simulate_all() mutates each model's 'ps'
        # field per trial, and we don't want to clobber the dicts in parameters.py.
        sim.models = [copy.deepcopy(APPROACH), copy.deepcopy(avoid)]
        sim.simulate_all(trials=trials, t_start=T_START, t_end=T_END, ps='trial')

        # Per-trial distance error, aligned index-for-index with sim.i_trials
        # and sim.pass_order_pred.
        errors = sim.test('p_dist', all_errors=True)

        for j, i in enumerate(sim.i_trials):
            pred_order = _sign_pm1(sim.pass_order_pred[j])
            actual_order = _sign_pm1(data.info['pass_order'][i])
            row = {
                'subj_id': data.info['subj_id'][i],
                'trial_id': data.info['trial_id'][i],
                'approach_model': APPROACH['name'],
                'avoidance_model': avoid_name,
                'dist_error': errors[j],
                'pred_pass_order': pred_order,
                'actual_pass_order': actual_order,
                'pass_order_correct': int(pred_order == actual_order),
            }
            for v in exp_vars:
                row[v] = data.info[v][i]
            rows.append(row)

        acc = np.mean([r['pass_order_correct'] for r in rows
                       if r['avoidance_model'] == avoid_name])
        mean_err = np.mean([r['dist_error'] for r in rows
                            if r['avoidance_model'] == avoid_name])
        print(f'  {exp_name} / {avoid_name}: '
              f'{int(len(sim.i_trials))} trials, '
              f'mean dist error {mean_err:.4f} m, order accuracy {acc:.3f}')

    return rows, exp_vars


def write_csv(exp_name, rows, exp_vars):
    cols = (['subj_id', 'trial_id'] + exp_vars +
            ['approach_model', 'avoidance_model', 'dist_error',
             'pred_pass_order', 'actual_pass_order', 'pass_order_correct'])
    out_path = f'{exp_name}_simulation_pass_order_errors.csv'
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)
    print(f'  wrote {len(rows)} rows -> {os.path.abspath(out_path)}')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--exp', choices=list(EXPERIMENTS), default=None,
                   help='Run a single experiment (default: both).')
    p.add_argument('--subjects', default=None,
                   help='Comma-separated subject ids (default: all).')
    return p.parse_args()


def main():
    args = parse_args()
    subjects = (range(100) if args.subjects is None
                else [int(s) for s in args.subjects.split(',')])
    exp_names = [args.exp] if args.exp else list(EXPERIMENTS)

    for exp_name in exp_names:
        print(f'{exp_name}:')
        rows, exp_vars = run_experiment(exp_name, subjects)
        write_csv(exp_name, rows, exp_vars)


if __name__ == '__main__':
    main()
