'''
This module contains the ODESimulator class. 
It uses a combination of BDF (first choice) and Euler method (backup choice) to solve odes.
'''
from scipy.integrate import solve_ivp, odeint
import numpy as np
from numpy.linalg import norm
from numpy import sqrt, gradient
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from packages.helper import play_trajs, sp2a, sp2v, psi, beta, d_theta, \
                            d_psi, hms, v2sp, dist, min_sep, av2dsdp, min_dist
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
        self.var0_match = []  # var0 at t0_match
        self.models = models
        self.args = args
        self.data = data
        self.ref = ref        
        self.reset()
        if data:
            self.use_data()
    
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
        self.t0 = []
        self.pass_order_pred = []
    
    def use_data(self):
        self.Hz = self.data.Hz
        self.var0 = []
        self.args = {'w_goal': self.data.info['w_goal'], 'w_obst': self.data.info['w_obst']}
        for i in range(len(self.data.trajs)):
            # print(f'Loading trial {i}')
            if self.data.info['obst_speed'][i]:
                # Compute order passing
                subj = self.data.get_traj(i)
                obst = self.data.info['p_obst'][i]
                # Compute var0 for simulation
                t0, t0_match = self.data.info['stimuli_onset'][i], self.data.info['stimuli_match'][i]
                self.var0.append(self.compute_var0(i, t0))
                self.var0_match.append(self.compute_var0(i, t0_match))             
            else:
                self.var0.append(None)
                self.var0_match.append(None)
        print(f'Loading finished')
        
    def compute_var0(self, i_trial, t0):
        i = i_trial
        xg0, yg0 = self.data.info['p_goal'][i][t0]
        p_obst = self.data.info['p_obst'][i]
        xo0, yo0 = p_obst[t0]
        vxo0, vyo0 = (p_obst[-1] - p_obst[t0]) / (len(p_obst) - 1 - t0) * self.Hz
        x0, y0 = self.data.info['p_subj'][i][t0]
        vx0, vy0 = (self.data.info['p_subj'][i][t0+1] - self.data.info['p_subj'][i][t0-1]) / 2 * self.Hz
        s0, phi0 = v2sp([vx0, vy0])
        v_pre = (self.data.info['p_subj'][i][t0] - self.data.info['p_subj'][i][t0-2]) / 2 * self.Hz
        v_post = (self.data.info['p_subj'][i][t0+2] - self.data.info['p_subj'][i][t0]) / 2 * self.Hz
        a0 = (v_post - v_pre) / 2 * self.Hz
        _, dphi0 = av2dsdp([vx0, vy0], a0)
        ds0 = (norm(v_post) - norm(v_pre)) / 2 * self.Hz
        return xg0, yg0, xo0, yo0, vxo0, vyo0, x0, y0, vx0, vy0, phi0, s0, dphi0, ds0

    @staticmethod
    def odeEuler(f, t, y0, args=None):
        '''Approximate the solution of y'=f(y,t) by Euler's method.

        Parameters
        ----------
        f : function
            Right-hand side of the differential equation y'=f(t,y), y(t_0)=y_0
        y0 : number
            Initial value y(t0)=y0 wher t0 is the entry at index 0 in the array t
        t : array
            1D NumPy array of t values where we approximate y values. Time step
            at each iteration is given by t[n+1] - t[n].

        Returns
        -------
        y : 1D NumPy array
            Approximation y[n] of the solution y(t_n) computed by Euler's method.
        '''
        y = np.zeros((len(t), len(y0)))
        y[0] = y0
        for n in range(0,len(t)-1):
            y[n+1] = y[n] + np.array(f(t[n], y[n], args[0], args[1])) * (t[n+1] - t[n])
        return y

    def simulate(self, var0, t0=None, t1=None, total_time=None, i_trial=None, print_trial=False, print_time=True):
        if print_trial and self.data:
            print(self.data.info['trial_id'][i_trial], i_trial)
        
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
        if total_time:
            t_eval = np.linspace(0, total_time - 1 / self.Hz, total_time * self.Hz)
            self.var0.append(var0)
        else:
            t_eval = np.linspace(0, t1 - t0 - 1, t1 - t0) / self.Hz
            for model in self.models:
                # Use preferred speed from var0.
                model['ps'] = var0[-3]
        
        sol = solve_ivp(ode_func, [0, t_eval[-1]], var0, method='BDF', t_eval=t_eval, args=[self.models, self.args])
        xg, yg, xo, yo, vxo, vyo, x, y, vx, vy, phi, s, dphi, ds = sol.y
        if len(x) != len(t_eval):
            print("simulation ended early, switch to Euler method")
            t_eval2 = t_eval[ : len(t_eval)-len(x)]
            var0 = [xg[-1], yg[-1], xo[-1], yo[-1], vxo[-1], vyo[-1], x[-1], y[-1], vx[-1], vy[-1], phi[-1], s[-1], dphi[-1], ds[-1]]
            y2 = ODESimulator.odeEuler(ode_func, t_eval2, var0, args=[self.models, self.args])
            xg, yg, xo, yo, vxo, vyo, x, y, vx, vy, phi, s, dphi, ds = np.concatenate((sol.y, np.transpose(y2)), axis=1)
        self.p_pred.append(np.stack((x, y), axis=-1))
        self.p_obst.append(np.stack((xo, yo), axis=-1))
        self.p_goal.append(np.stack((xg, yg), axis=-1))
        self.v_pred.append(np.stack((vx, vy), axis=-1))
        self.phi_pred.append(phi)
        self.s_pred.append(s)
        p0, p1, v0, v1 = [x[-2], y[-2]], [xo[-2], yo[-2]], [vx[-2], vy[-2]], [vxo[-2], vyo[-2]]      
        if self.data:
            self.t0.append(t0)
            self.i_trials.append(i_trial)
            self.p_subj.append(self.data.info['p_subj'][i_trial][t0:t0 + len(x)])
            angle = self.data.info['obst_angle'][i_trial]
            self.pass_order_pred.append(np.sign(-angle * beta(p0, p1, v0)))
        if print_time:
            toc = time.perf_counter()
            print(f'Simulation finished in {hms(toc - tic):s} t_total {t_eval[-1]:f}')
    
    def simulate_all(self, trials=None, match_order=False, print_trial=False):
        tic = time.perf_counter()
        n_trials = 0
        if not trials:
            trials = range(len(self.data.trajs))
        for i in trials:
            # Skip freewalk trial and +-180 trials
            if self.data.info['obst_speed'][i] == 0 or abs(self.data.info['obst_angle'][i]) == 180:
                continue
            var0 = self.var0_match[i] if match_order else self.var0[i]
            t0 = self.data.info['stimuli_match'][i] if match_order else self.data.info['stimuli_onset'][i]
            t1 = self.data.info['stimuli_out'][i]
            self.simulate(var0, t0=t0, t1=t1, i_trial=i, print_trial=print_trial, print_time=False)
            n_trials += 1
        toc = time.perf_counter()
        print(f'Simulated {n_trials} trials in {hms(toc - tic):s}')

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
            t0 = self.t0[i_trial]
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
        elif var == 'order':
            preds = self.pass_order_pred
            trues = np.array(self.data.info['pass_order'])[self.i_trials]
        if i_trial:
            trues = trues[i_trial]
            preds = preds[i_trial]
            
        # Compute metric
        if alg == 'accuracy':
            return accuracy_score(trues, preds)
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