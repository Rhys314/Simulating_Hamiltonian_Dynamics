# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 19:18:52 2016
Section 4,5,2, Charged partical in a magnetic field

Test script.
@author: Rhys Poolman
"""
import numpy as np
import Steppers.scovel as scv
import Plotting.TwoDimensions as plt

# inital values and preallocted arrays
Dt = np.float64(0.1)
T = np.float64(100.0)
N = np.int(T/Dt)
p = np.zeros((N, 3), dtype = np.float64)
q = np.zeros((N, 3), dtype = np.float64)
q[0, :] = np.ones(3)
b = np.zeros(3)
b[2] = 1.0
V = lambda x, y, z: -1/np.sqrt(x**2 + y**2 + z**2)
dVdq = lambda q: q/(q[0]**2 + q[1]**2 + q[2]**2)**3/2
M = np.float64(1.0)

# integrate Hamiltonian
stepper = scv._ScovelsMethod(dVdq, M, b, Dt)
for nn in range(N - 1):
    # 2nd Order Scovel method calculations
    q[nn + 1, :], p[nn + 1, :] = stepper.m_integrate(q[nn, :], p[nn, :])
    
#%% Plot
# plot in 2D
plt.plotTrajectory(1, q, [-2.0, 2.0, -2.0, 2.0])#[0.98, 0.995, -0.01, 0.01])
plt.plotPhaseSpace(2, q, p)#, 
                   #[-1.5, 1.5, -1.5, 1.5], [-1.5, 1.5, -1.5, 1.5])
plt.plot(3, q, p)#, 
                   #[-1.5, 1.5, -1.5, 1.5], [-1.5, 1.5, -1.5, 1.5])