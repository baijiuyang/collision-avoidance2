# This module contains the defination of ODESimulator class, which was created by Joey Bai to support his dissertation studies.
# please contact baijiuyang@hotmail.com for support. Git: https://github.com/baijiuyang/collision-avoidance.git
from scipy.integrate import solve_ivp
import numpy as np
from numpy.linalg import norm
from numpy import sqrt, gradient
from sklearn.metrics import mean_squared_error, mean_absolute_error
from packages.helper import play_trajs, sp2a, sp2v, psi, beta, d_theta, \
                            d_psi, hms, v2sp, dist, min_sep, av2dsdp
from packages.models import fajen_approach, fajen_approach2, cohen_avoid, cohen_avoid2, \
                            cohen_avoid3, cohen_avoid4, vector_approach, perpendicular_avoid
                            
from collections import defaultdict
import time

class ODESimulator:
    '''
    This class produce simulation of one trial using ode45 method. It takes the model parameters and initial conditions
    and produce the simulated time series of position, speed and acceleration based on specified models.
    Fields:
        Hz (int): The frequency of the simulation.
        var0 (list of floats): Initial conditions.
        models (list of dicts): A list of dictionaries like this: {'name': 'model_name', 'parameter1': float, 'parameter2': float, ...}
        args (dict): Any necessary named argument needed, such as {'w_goal': 0.1, 'w_obst': 0.2}.
        t (list of float): Time stamps of simulation.
        p_pred, p_obst, p_goal (list of lists): Time series of positions in the simulation.
        v_pred (list of lists): Time series of velocities in the simulation.
        phi, dphi, s (list of floats): Time series of headings, derivative of headings and speeds.
        sol (Bunch object, see scipy.integrate.solve_ivp): The solution of ode_func.
        ref (2d vector): The reference vector that defines heading phi.
    '''
    def __init__(self, Hz=None, models=None, args=None, data=None, ref=[0, 1]):
        self.Hz = Hz
        self.var0 = []
        self.models = models
        self.args = args
        self.data = data
        self.ref = ref
        self.reset()
        if data:
            data.filter_all()
            self.use_data()
            
    def use_data(self):
        self.Hz = self.data.Hz
        self.var0 = []
        self.args = {'w_goal': self.data.info['w_goal'], 'w_obst': self.data.info['w_obst']}
        for i in range(len(self.data.trajs)):        
            t0, t1 = self.data.info['stimuli_onset'][i], self.data.info['stimuli_out'][i]
            t_total = (t1 - t0)
            xg0, yg0 = self.data.info['p_goal'][i][t0]
            p_obst = self.data.info['p_obst'][i]
            xo0, yo0 = p_obst[t0]
            vxo0, vyo0 = (p_obst[-1] - p_obst[t0]) / (len(p_obst) - 1 - t0) * self.Hz
            x0, y0 = self.data.p_subj[i][t0]
            vx0, vy0 = (self.data.p_subj[i][t0+1] - self.data.p_subj[i][t0-1]) / 2 * self.Hz
            s0, phi0 = v2sp([vx0, vy0])
            v_pre = (self.data.p_subj[i][t0] - self.data.p_subj[i][t0-2]) / 2 * self.Hz
            v_post = (self.data.p_subj[i][t0+2] - self.data.p_subj[i][t0]) / 2 * self.Hz
            a0 = (v_post - v_pre) / 2 * self.Hz
            _, dphi0 = av2dsdp([vx0, vy0], a0)
            ds0 = (norm(v_post) - norm(v_pre)) / 2 * self.Hz
            self.var0.append([xg0, yg0, xo0, yo0, vxo0, vyo0, x0, y0, vx0, vy0, phi0, s0, dphi0, ds0])
            
    def reset(self):
        self.i_trials = []
        self.p_pred = []
        self.p_subj = []
        self.p_obst = []
        self.p_goal = []
        self.v_pred = []
        self.s_pred = []
        self.phi_pred = []
        self.dphi_pred = []
    
    def simulate(self, var0, i_trial=None, print_time=True):    
        def ode_func(t, var, models, args):
            xg, yg, xo, yo, vxo, vyo, x, y, vx, vy, phi, s, dphi, ds = var
            ref, w_goal, w_obst = self.ref, args['w_goal'], args['w_obst']
            r_g = dist([x, y], [xg, yg])
            r_o = dist([x, y], [xo, yo])
            psi_g = psi([x, y], [xg, yg], ref=ref)
            beta_o = beta([x, y], [xo, yo], [vx, vy])
            d_theta_o = d_theta([x, y], [xo, yo], [vx, vy], [vxo, vyo], w_obst)
            d_psi_o = d_psi([x, y], [xo, yo], [vx, vy], [vxo, vyo])
            
            output = defaultdict(float)
            for model in models:
                # Use preferred speed from data.
                if self.data:
                    model['ps'] = self.data.info['ps_subj'][i_trial]
                if model['name'] == 'fajen_approach':
                    for key, val in fajen_approach(model, phi, dphi, s, psi_g, r_g).items():
                        output[key] += val
                elif model['name'] == 'fajen_approach2':
                    for key, val in fajen_approach2(model, phi, dphi, s, ds, psi_g, r_g).items():
                        output[key] += val
                elif model['name'] == 'vector_approach':
                    for key, val in vector_approach(model, [x, y], [xg, yg], [vx, vy]).items():
                        output[key] += val
                elif model['name'] == 'cohen_avoid':
                    for key, val in cohen_avoid(model, dphi, beta_o, d_psi_o, r_o, s).items():
                        output[key] += val
                elif model['name'] == 'cohen_avoid2':                    
                    for key, val in cohen_avoid2(model, dphi, s, beta_o, d_theta_o, d_psi_o).items():
                        output[key] += val
                elif model['name'] == 'cohen_avoid3':
                    for key, val in cohen_avoid3(model, beta_o, d_theta_o, d_psi_o).items():
                        output[key] += val
                elif model['name'] == 'cohen_avoid4':
                    for key, val in cohen_avoid4(model, beta_o, d_theta_o, d_psi_o).items():
                        output[key] += val
                elif model['name'] == 'perpendicular_avoid':
                    for key, val in perpendicular_avoid(model, beta_o, psi_o, theta_o, d_theta_o, d_psi_o, ref).items():
                        output[key] += val
            if 'ds' in output:
                ddphi, ds = output['ddphi'], output['ds']
                dds = 0
                ax, ay = sp2a(s, ds, phi, dphi, ref=ref)
            elif 'dds' in output:
                ddphi, dds = output['ddphi'], output['dds']
                ax, ay = sp2a(s, ds, phi, dphi, ref=ref)
            elif 'a' in output:
                pass

            dvardt = [0, 0, vxo, vyo, 0, 0, vx, vy, ax, ay, dphi, ds, ddphi, dds]
            return dvardt
        if print_time:
            tic = time.perf_counter()
        t0, t1 = self.data.info['stimuli_onset'][i_trial], self.data.info['stimuli_out'][i_trial]
        t_eval = np.linspace(0, t1 - t0 - 1, t1 - t0) / self.Hz
        self.i_trials.append(i_trial)
        if not self.data:
            self.var0.append(var0)
        sol = solve_ivp(ode_func, [0, t_eval[-1]], var0, t_eval=t_eval, args=[self.models, self.args])
        xg, yg, xo, yo, vxo, vyo, x, y, vx, vy, phi, s, dphi, ds = sol.y
        self.p_pred.append(np.stack((x, y), axis=-1))
        self.p_obst.append(np.stack((xo, yo), axis=-1))
        self.p_goal.append(np.stack((xg, yg), axis=-1))
        self.v_pred.append(np.stack((vx, vy), axis=-1))
        self.phi_pred.append(phi)
        self.s_pred.append(s)        
        self.p_subj.append(self.data.p_subj[i_trial][t0:t1])
        if print_time:
            toc = time.perf_counter()
            print(f'Simulation finished in {hms(toc - tic):s} t_total {t_eval[-1]:f}')
            
    def simulate_all(self, subj_id=None):
        tic = time.perf_counter()
        for i in range(len(self.data.trajs)):
            # Only simulate for one subject
            if subj_id != None and self.data.info['subj_id'][i] != subj_id:
                continue
            # Skip freewalk trials and trials with obst_angle == +-180
            if self.data.info['obst_speed'][i] == 0 or abs(self.data.info['obst_angle'][i]) == 180:
                continue
            self.simulate(self.var0[i], i_trial=i, print_time=False)
        toc = time.perf_counter()
        print(f'Simulation_all finished in {hms(toc - tic):s}')
        
    def play(self, i_trial=0, actual=False, title=None, colors=None, interval=None, save=False):
        '''
        Args:
            interval (float): The pause between two frames in millisecond.
            save (bool): Flag for saving the animation in the current working directory.
        '''
        # Trial index in data instead of simulation results (because some trials are skipped in simulation)
        j_trial = self.i_trials[i_trial]
        w_goal, w_obst = self.args['w_goal'], self.args['w_obst']
        ws = [0.4, w_goal, w_obst] 
        labels = ['pred', 'goal', 'obst']
        if self.data:
            if self.data.info['obst_speed'][j_trial] == 0 or abs(self.data.info['obst_angle'][j_trial]) == 180:
                print('This trial is not simulated')
                return
            w_goal, w_obst = self.data.info['w_goal'], self.data.info['w_obst']
            ws.append(0.4)
            labels.append('subj')
            p_subj = self.data.get_traj(j_trial)            
            # Pad values to p_pred to give it the same length as p_subj
            t0 = self.data.info['stimuli_onset'][j_trial]
            p_pred = np.zeros_like(p_subj)
            p_pred[t0: t0+len(self.p_pred[i_trial])] = self.p_pred[i_trial]            
            # p_pred[:t0], p_pred[t0+len(self.p_pred[i_trial]):] = p_pred[t0], p_pred[t0+len(self.p_pred[i_trial])-1]
            p_pred[:t0], p_pred[t0+len(self.p_pred[i_trial]):] = None, None
            if actual:
                p_goal, p_obst = self.data.info['p_goal'][j_trial], self.data.info['p_obst'][j_trial]
            else:
                # Pad values to simulated trajs to give them the same length as data
                p_goal = np.zeros_like(p_subj)
                p_goal[t0: t0+len(self.p_goal[i_trial])] = self.p_goal[i_trial]
                p_goal[:t0], p_goal[t0+len(self.p_goal[i_trial]):] = p_goal[t0], p_goal[t0+len(self.p_goal[i_trial])-1]
                p_obst = np.zeros_like(p_subj)
                p_obst[t0: t0+len(self.p_obst[i_trial])] = self.p_obst[i_trial]
                p_obst[:t0], p_obst[t0+len(self.p_obst[i_trial]):] = p_obst[t0], p_obst[t0+len(self.p_obst[i_trial])-1]
                
            trajs = [p_pred, p_goal, p_obst, p_subj]
        else:
            trajs = [self.p_pred[i_trial], self.p_goal[i_trial], self.p_obst[i_trial]]

        play_trajs(trajs, ws, self.Hz, ref=self.ref, title=title, labels=labels, colors=colors, interval=interval, save=save)

    def test(self, metric, i_trial=None):
        '''
        Metric has the format of "var_alg". Var can be 'p': position,
        'v': velocity, 'a': acceleration, 's': speed, 'phi': heading, or
        'dca': signed distance of the closest approach. Alg can be
        'dist': average distance, 'MAE': mean absolute error, 'MSE': mean
        squared error, or 'RMSE': root mean squared error.
        '''
        Hz = self.Hz
        var = metric.split('_')[0]
        alg = metric.split('_')[1]

        # Compute variable, iterating through all trials
        if var == 'p':
            preds = self.p_pred
            trues = self.p_subj
        elif var == 's':
            preds = self.s_pred
            trues = []
            for i in range(len(preds)):
                v_subj = gradient(self.p_subj[i], axis=0) * Hz
                trues.append(norm(v_subj, axis=-1))
        elif var == 'phi':
            preds = self.phi_pred
            trues = []
            v_subj = gradient(self.p_subj, axis=0) * Hz            
            for i in range(len(preds)):
                trues.append(v2sp(v_subj[i], ref=self.ref)[1])
        if i_trial:
            trues = trues[i_trial]
            preds = preds[i_trial]
            
        # Compute metric
        vals = []
        for i in range(len(preds)):
            if alg == 'dist':
                vals.append(np.mean(dist(trues[i], preds[i])))
            elif alg == 'MAE':
                vals.append(mean_absolute_error(trues[i], preds[i]))
            elif alg == 'MSE':
                vals.append(mean_squared_error(trues[i], preds[i]))
            elif alg == 'RMSE':
                vals.append(np.sqrt(mean_squared_error(trues[i], preds[i])))
        return np.mean(vals)    