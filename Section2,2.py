# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 08:04:04 2015
This is an attempt to recreate the model of a Lennard-Jones potential found in 
section 2.2 on pp. 18.  In particular the calculation of the functions a(t_n) 
and b(t_n) are attemted as well as the energy error.
@author: rpoolman
"""
import Steppers.steppers as step
import numpy as np
import matplotlib.pyplot as plt
import NumericalAlgorithms.derivatives as deriv

# Lennard-Jones potential and derivative
LJ = lambda x: x**-12 - 2.0*x**-6
LJdiff = lambda x: 12*(x**-7 - x**-13)

# constants and arrays
M = 45.0
Dt = np.float64(0.001)
N = np.int(100/Dt)
q = np.zeros(N + 1, dtype = np.float64)
v = np.zeros(N + 1, dtype = np.float64)
t = np.zeros(N + 1, dtype = np.float64)
a = np.zeros(N + 1, dtype = np.float64)
b = np.zeros(N + 1, dtype = np.float64)
e2norm = np.zeros(N + 1, dtype = np.float64)
energyError = np.zeros(N + 1, dtype = np.float64)
phi = np.zeros(N + 1, dtype = np.float64)
dphidq_analytic = np.zeros(N + 1, dtype = np.float64)
dphidq_numeric = np.zeros(N + 1, dtype = np.float64)

# initial conditions
q[0] = 1.9
v[0] = -0.0001
energyConst = 0.5*v[0]  + LJ(q[0])

# calcualte phase space
for ii in range(0, N):
    t[ii + 1] = t[ii] + Dt
    q[ii + 1], v[ii + 1] = step.eulerstep(LJ, q[ii], v[ii], Dt, M)
    dphidq = -deriv.derivative(LJ, q[ii], np.abs(q[ii + 1] - q[ii]), False)[0]
    d2phidq2 = -deriv.derivative(LJdiff, q[ii], np.abs(q[ii + 1] - q[ii]), False)[0]
    a[ii] = np.max([1, d2phidq2])
    E = 0.5*v[ii]**2 + LJ(q[ii])
    b[ii] = 0.5*np.sqrt(dphidq**2 + 2*d2phidq2**2*(E - LJ(q[ii])))
    energyError[ii] = E - energyConst
    
# plot
plt.figure(1)
plt.clf()
ax1 = plt.subplot2grid((3, 2), (0, 0), rowspan = 3)
ax1.plot(q, v)
plt.title('Phase Plot')
plt.xlabel('Generalised Coordainte, $q$')
plt.ylabel('Generalised Velocity, $v(q)$')
plt.axis([0.5, 2.5, -0.5, 0.5])
ax2 = plt.subplot2grid((3, 2), (0, 1))
ax2.plot(t, a)
plt.ylabel('$a(t)$')
plt.axis([0, 100, 0, 6])
ax3 = plt.subplot2grid((3, 2), (1, 1))
ax3.plot(t, b)
plt.ylabel('$b(t)$')
plt.axis([0, 100, 0, 2])
ax4 = plt.subplot2grid((3, 2), (2, 1))
ax4.plot(t, energyError)
ax4.yaxis.get_major_formatter().set_powerlimits((0, 1))
plt.xlabel('Time, $t$')
plt.ylabel('Energy Error')
plt.axis([0, 100, -5e-4, 10e-4])
plt.show()