'''
This module contains the the ODESimulator class.
'''
import numpy as np
from math import pi
from numpy import absolute, exp, sign, array, maximum
from numpy.linalg import norm
from packages.helper import rotate

'''{'name': 'fajen_approach', 'ps': None,
                  'b1': 3.25, 'k1': 7.5, 'c1': 0.4, 'c2': 0.4, 'k2': 1.4}'''
def fajen_approach(args, phi, dphi, s, psi, r):    
    ps, b1, k1, c1, c2, k2 = args['ps'], args['b1'], args['k1'], args['c1'], args['c2'], args['k2']
    ddphi = -b1 * dphi - k1 * (phi - psi) * (exp(-c1 * r) + c2)
    ds = k2 * (ps - s)
    return {'ds': ds, 'ddphi': ddphi}

def fajen_approach2(args, phi, dphi, s, ds, psi, r):
    ps, b1, k1, c1, c2, b2, k2 = args['ps'], args['b1'], args['k1'], args['c1'], args['c2'], args['b2'], args['k2']
    ddphi = -b1 * dphi - k1 * (phi - psi) * (exp(-c1 * r) + c2)
    dds = -b2 * ds + k2 * (ps - s)    
    return {'dds': dds, 'ddphi': ddphi}
    
def cohen_avoid(args, dphi, beta, dpsi, r, s):
    ps, b1, k1, c5, c6, b2, k2, c7, c8 = args['ps'], args['b1'], args['k1'], args['c5'], args['c6'], args['b2'], args['k2'], args['c7'], args['c8']
    step = (sign(pi / 2 - absolute(beta)) + 1) / 2
    ddphi = -b1 * dphi - k1 * dpsi * exp(-c5 * absolute(dpsi) - c6 * r) * step
    ds = -b2 * (s - ps) + sign(beta) * k2 * dpsi * exp(-c7 * absolute(dpsi) - c8 * r) * step
    return {'ds': ds, 'ddphi': ddphi}
    
# Known issue: When dpsi is zero, it becomes a null model.
def cohen_avoid2(args, dphi, s, beta, dtheta, dpsi):
    ps, b1, k1, c5, c6, b2, k2, c7, c8 = args['ps'], args['b1'], args['k1'], args['c5'], args['c6'], args['b2'], args['k2'], args['c7'], args['c8']
    sigmoid = 1 / (1 + exp(20 * (absolute(beta) - 1.3)))
    ddphi = -b1 * dphi - dpsi * k1 * exp(-c5 * absolute(dpsi)) * (1 - exp(-c6 * maximum(0, dtheta))) * sigmoid
    ds = b2 * (ps - s) + sign(beta) * dpsi * k2 * exp(-c7 * absolute(dpsi)) * (1 - exp(-c8 * maximum(0, dtheta))) * sigmoid
    return {'ds': ds, 'ddphi': ddphi}

def cohen_avoid3(args, beta, dtheta, dpsi):
    k1, c5, c6, k2, c7, c8 = args['k1'], args['c5'], args['c6'], args['k2'], args['c7'], args['c8']
    sigmoid = 1 / (1 + exp(20 * (absolute(beta) - 1.3)))
    dds = k2 * sign(beta) * dpsi * exp(-c7 * absolute(dpsi)) * (1 - exp(-c8 * maximum(0, dtheta))) * sigmoid
    ddphi = -k1 * dpsi * exp(-c5 * absolute(dpsi)) * (1 - exp(-c6 * maximum(0, dtheta))) * sigmoid
    return {'dds': dds, 'ddphi': ddphi}

def cohen_avoid4(args, beta, dtheta, dpsi):
    k1, c5, c6, k2, c7, c8 = args['k1'], args['c5'], args['c6'], args['k2'], args['c7'], args['c8']
    sigmoid = 1 / (1 + exp(20 * (absolute(beta) - 1.3)))
    dds = k2 * sign(beta) * sign(dpsi) * exp(-c7 * absolute(dpsi)) * (1 - exp(-c8 * maximum(0, dtheta))) * sigmoid
    ddphi = -k1 * sign(dpsi) * exp(-c5 * absolute(dpsi)) * (1 - exp(-c6 * maximum(0, dtheta))) * sigmoid
    return {'dds': dds, 'ddphi': ddphi}

def cohen_avoid4_thres(args, beta, dtheta, dpsi):
    k1, c5, c6, k2, c7, c8, thres = args['k1'], args['c5'], args['c6'], args['k2'], args['c7'], args['c8'], args['thres']
    dpsi = sign(dpsi) * maximum(abs(dpsi) - thres, 0)
    sigmoid = 1 / (1 + exp(20 * (absolute(beta) - 1.3)))
    dds = k2 * sign(beta) * sign(dpsi) * exp(-c7 * absolute(dpsi)) * (1 - exp(-c8 * maximum(0, dtheta))) * sigmoid
    ddphi = -k1 * sign(dpsi) * exp(-c5 * absolute(dpsi)) * (1 - exp(-c6 * maximum(0, dtheta))) * sigmoid
    return {'dds': dds, 'ddphi': ddphi}

def acceleration_approach(args, p0, p1, v0):
    ps, k = args['ps'], args['k']
    vi = array(p1) - array(p0)
    vp = vi * ps / norm(vi, axis=-1)
    a = k * (vp - array(v0))
    return {'a': a}

def da_approach(args, p0, p1, v0, a0):
    ps, k, b = args['ps'], args['k'], args['b']
    vi = array(p1) - array(p0)
    vp = vi * ps / norm(vi, axis=-1)
    j = k * (vp - array(v0)) - b * array(a0)
    return {'ia': ia, 'da': da}

def perpendicular_avoid(args, beta, psi, theta, dtheta, dpsi, ref):
    k, c = args['k'], args['c']
    alpha = psi - sign(dpsi) * (pi / 2)
    ratio = (maximum(0, dtheta)/theta + c) / (absolute(dpsi) + c) - 1
    sigmoid = 1 / (1 + exp(20 * (absolute(beta) - 1.3)))
    a_mag = k * maximum(0, ratio) * sigmoid # multiply a sigmoid function of beta
    a = rotate([i * a_mag for i in ref], alpha)
    return {'a': a}

def perpendicular_avoid2(args, beta, psi, theta, dtheta, dpsi, ref):
    k, c = args['k'], args['c']
    indicator = absolute(beta) < pi/2
    alpha = psi - sign(dpsi) * (pi / 2)
    ratio = (maximum(0, dtheta)/theta) / (absolute(dpsi) + c)
    a_mag = k * maximum(0, ratio) * indicator
    a = rotate([i * a_mag for i in ref], alpha)
    return {'a': a}

def perpendicular_avoid3():
    pass