# This module contains the defination of ODESimulator class, which was created by Joey Bai to support his dissertation studies.
# please contact baijiuyang@hotmail.com for support. Git: https://github.com/baijiuyang/collision-avoidance.git
from scipy.integrate import solve_ivp
import numpy as np
from numpy.linalg import norm
from numpy import sqrt
from math import pi
import matplotlib.pyplot as plt
from packages.helper import play_trajs, rotate, sp2a, v2sp, dist, psi, beta, d_theta, d_psi

class ODESimulator:
    '''
    This class produce simulation using ode45 method. It takes the model parameters and initial conditions
    and produce the simulated time series of position, speed and acceleration based on specified models.
    Fields:
        models (list): A list of dictionary like this: {'name': 'model_name', 'parameter1': float, 'parameter2': float, ...}
        args (dict): Any necessary named arguments needed, such as {'w_goal': 0.1, 'w_obst': 0.2}
    '''
    def __init__(self, models=None, args=None):
        self.models = models if models else []
        self.args = args if args else {}
        
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
        
    def ode_func(t, var, models, args):
        xg, yg, xo, yo, vxo, vyo, x, y, vx, vy, phi, s, dphi = var
        ref, w_goal, w_obst = args['ref'], args['w_goal'], args['w_obst']
        r_g = dist([x, y], [xg, yg])
        psi_g = psi([x, y], [xg, yg], ref=ref)
        beta_o = beta([x, y], [xo, yo], [vx, vy])
        d_theta_o = d_theta([x, y], [xo, yo], [vx, vy], [vxo, vyo], w_obst)
        d_psi_o = d_psi([x, y], [xo, yo], [vx, vy], [vxo, vyo])
        sigmoid = 1 / (1 + np.exp(20 * (np.absolute(beta_o) - 1.3)))

        def Fajen_approach(args):    
            ps, b1, k1, c1, c2, k2 = args['ps'], args['b1'], args['k1'], args['c1'], args['c2'], args['k2']
            ddphi = -b1 * dphi - k1 * (phi - psi_g) * (np.exp(-c1 * r_g) + c2)
            ds = k2 * (ps - s)
            ax, ay = sp2a(s, ds, phi, dphi, ref=ref)
            output = [ax, ay, ds, ddphi]
            return output
        
        # Known issue: When d_psi_o is zero, it becomes a null model.
        def Cohen_avoid2(args):
            ps, b1, k1, c5, c6, b2, k2, c7, c8 = args['ps'], args['b1'], args['k1'], args['c5'], args['c6'], args['b2'], args['k2'], args['c7'], args['c8']
            ddphi = -b1 * dphi - d_psi_o * k1 * np.exp(-c5 * np.absolute(d_psi_o)) * (1 - np.exp(-c6 * np.maximum(0, d_theta_o))) * sigmoid
            ds = b2 * (ps - s) + d_psi_o * k2 * np.exp(-c7 * np.absolute(d_psi_o)) * (1 - np.exp(-c8 * np.maximum(0, d_theta_o))) * sigmoid
            ax, ay = sp2a(s, ds, phi, dphi, ref=ref)
            output = [ax, ay, ds, ddphi]
            return output
        
        output = np.array([0.0] * 4)
        for model in models:
            if model['name'] == 'Cohen_avoid2':
                output += np.array(Cohen_avoid2(model))
            elif model['name'] == 'Fajen_approach':
                output += np.array(Fajen_approach(model))
        # xg, yg, xo, yo, vxo, vyo, x, y, vx, vy, phi, s, dphi = var
        dvardt = [0, 0, vxo, vyo, 0, 0, vx, vy, output[0], output[1], dphi, output[2], output[3]]
        return dvardt