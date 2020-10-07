# This module contains models of pedestrian locomotion.
import numpy as np
def Fajen_approach(args, phi, dphi, s, psi_g, r_g):    
    ps, b1, k1, c1, c2, k2 = args['ps'], args['b1'], args['k1'], args['c1'], args['c2'], args['k2']
    ddphi = -b1 * dphi - k1 * (phi - psi_g) * (np.exp(-c1 * r_g) + c2)
    ds = k2 * (ps - s)
    return {'ds': ds, 'ddphi': ddphi}

        
# Known issue: When d_psi_o is zero, it becomes a null model.
def Cohen_avoid2(args, dphi, s, beta_o, d_theta_o, d_psi_o, sigmoid):
    ps, b1, k1, c5, c6, b2, k2, c7, c8 = args['ps'], args['b1'], args['k1'], args['c5'], args['c6'], args['b2'], args['k2'], args['c7'], args['c8']
    ddphi = -b1 * dphi - d_psi_o * k1 * np.exp(-c5 * np.absolute(d_psi_o)) * (1 - np.exp(-c6 * np.maximum(0, d_theta_o))) * sigmoid
    ds = b2 * (ps - s) + d_psi_o * k2 * np.exp(-c7 * np.absolute(d_psi_o)) * (1 - np.exp(-c8 * np.maximum(0, d_theta_o))) * sigmoid
    return {'ds': ds, 'ddphi': ddphi}