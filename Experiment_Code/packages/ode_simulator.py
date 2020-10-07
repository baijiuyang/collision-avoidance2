# This module contains the defination of ODESimulator class, which was created by Joey Bai to support his dissertation studies.
# please contact baijiuyang@hotmail.com for support. Git: https://github.com/baijiuyang/collision-avoidance.git
from scipy.integrate import solve_ivp
import numpy as np
from numpy.linalg import norm
from numpy import sqrt
from math import pi
import matplotlib.pyplot as plt
from packages.helper import play_trajs, rotate, sp2a, v2sp, dist, psi, beta, d_theta, d_psi, hms, v2sp, play_trajs
from packages.models import Fajen_approach, Cohen_avoid2
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
        self.t (list of float): Time stamps of simulation.
        p_subj, p_obst, p_goal (list of lists): Time series of positions in the simulation.
        v_subj, v_obst (list of lists): Time series of velocities in the simulation.
        phi, dphi, s (list of floats): Time series of headings, derivative of headings and speeds.
        sol (Bunch object, see scipy.integrate.solve_ivp): The solution of ode_func.
        ref (2d vector): The reference vector that defines heading phi.
    '''
    def __init__(self, Hz, var0=None, models=None, args=None, ref=[0, 1]):
        self.Hz = Hz
        self.var0 = var0
        self.models = models if models else []
        self.args = args if args else {}
        self.t = None
        self.p_subj = None
        self.p_obst = None
        self.p_goal = None
        self.v_subj = None
        self.v_obst = None
        self.s = None
        self.phi = None
        self.dphi = None        
        self.sol = None
        self.ref = ref
        
    def set_var0(self, var0):
        # var0 = [xg, yg, xo, yo, vxo, vyo, x, y, vx, vy, phi, s, dphi]
        self.var0 = var0
        
    def add_arg(self, key, value):
        ''' 
        key (string): Parameter name.
        value (float): Parameter value.
        '''
        self.args[key] = value
        
    def add_model(self, model):
        '''
        model (dict): {'name': 'model_name', 'parameter1': float, 'parameter2': float, ...}.
        '''
        self.models.append(model)
    
    def simulate(self, t_total):    
        def ode_func(t, var, models, args):
            xg, yg, xo, yo, vxo, vyo, x, y, vx, vy, phi, s, dphi = var
            ref, w_goal, w_obst = self.ref, args['w_goal'], args['w_obst']
            r_g = dist([x, y], [xg, yg])
            psi_g = psi([x, y], [xg, yg], ref=ref)
            beta_o = beta([x, y], [xo, yo], [vx, vy])
            d_theta_o = d_theta([x, y], [xo, yo], [vx, vy], [vxo, vyo], w_obst)
            d_psi_o = d_psi([x, y], [xo, yo], [vx, vy], [vxo, vyo])
            sigmoid = 1 / (1 + np.exp(20 * (np.absolute(beta_o) - 1.3)))

            output = defaultdict(float)
            for model in models:
                if model['name'] == 'Cohen_avoid2':
                    for key, val in Cohen_avoid2(model, dphi, s, beta_o, d_theta_o, d_psi_o, sigmoid).items():
                        output[key] += val
                elif model['name'] == 'Fajen_approach':
                    for key, val in Fajen_approach(model, phi, dphi, s, psi_g, r_g).items():
                        output[key] += val
            if 'ds' in output:
                ds, ddphi = output['ds'], output['ddphi']
                ax, ay = sp2a(s, ds, phi, dphi, ref=ref)
            elif 'dds' in ouput:
                pass
            elif 'a' in output:
                pass
            
            # xg, yg, xo, yo, vxo, vyo, x, y, vx, vy, phi, s, dphi = var
            dvardt = [0, 0, vxo, vyo, 0, 0, vx, vy, ax, ay, dphi, ds, ddphi]
            return dvardt
        tic = time.perf_counter()
        t_eval = np.linspace(0, t_total, t_total * self.Hz)
        self.sol = solve_ivp(ode_func, [0, t_total], self.var0, t_eval=t_eval, args=[self.models, self.args])
        xg, yg, xo, yo, vxo, vyo, x, y, vx, vy, phi, s, dphi = self.sol.y
        self.t = t_eval
        self.p_subj = np.stack((x, y), axis=-1)
        self.p_obst = np.stack((xo, yo), axis=-1)
        self.p_goal = np.stack((xg, yg), axis=-1)
        self.v_subj = np.stack((vx, vy), axis=-1)
        self.v_obst = np.stack((vxo, vyo), axis=-1)
        self.phi = phi
        self.dphi = dphi
        self.s = s
        toc = time.perf_counter()
        print(f'Simulation finished in {hms(tic - toc):s}')
    
    def play(self, colors=None, interval=None, save=False):
        '''
        Args:
            interval (float): The pause between two frames in millisecond.
            save (bool): Flag for saving the animation in the current working directory.
        '''
        ws = [0.4, self.args['w_goal'], self.args['w_obst']] 
        labels = ['subj', 'goal', 'obst']
        trajs = [self.p_subj, self.p_goal, self.p_obst]
        play_trajs(trajs, ws, self.Hz, ref=self.ref, labels=labels, colors=colors, interval=interval, save=save)
            
            
            