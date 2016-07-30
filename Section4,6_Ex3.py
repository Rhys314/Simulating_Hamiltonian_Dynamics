# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 07:16:33 2015
Exercise 3. Kepler Problem
Discretize the planar Kepler problem with Hamiltonian
                            H(q, p) = 1/2 p^T.p - 1/||q||
and initial conditions q = (1, 0)^T and p = (0, 1)^T, by the explicit 
Stoermer-Verlet method and the implicit midpoint rule (see section 4.5.2).  
Compare the two methods based on the conservation of energy versus stepsize 
and the number of force field evalations per timestep.  Take the stepsize Dt 
from the interval [0.01, 0.0001] and integrate over a time interval [0, 10].
@author: rpoolman
"""
import Steppers.steppers as step
import numpy as np
import matplotlib.pyplot as plt

# constants and arrays
Dt = 0.0001
T = 10
N = np.int(T/Dt)
t = np.linspace(0, T, N + 1)
q = np.zeros((N + 1, 2))
p = np.zeros((N + 1, 2))
V = lambda qx, qy: 1/np.sqrt(qx**2 + qy**2)

# intial values
q[0, 0] = 1.0
p[0, 1] = 1.0

# calculation with the implicit mid point method
for ii in range(0, N):
    q[ii + 1], p[ii + 1] = step.implicitmidpoint(V, q[ii], p[ii], Dt)
    
# phase plot
plt.figure(1)
plt.clf()
plt.subplots_adjust(wspace = 0.3)
plt.subplot(121)
plt.title('Phase Plot in $x$')
plt.plot(q[:, 0], p[:, 0])
plt.xlabel('Position, $x$')
plt.ylabel('Momentum, $p_x$')
plt.subplot(122)
plt.title('Phase Plot in $y$')
plt.plot(q[:, 1], p[:, 1])
plt.xlabel('Position, $y$')
plt.ylabel('Momentum, $p_y$')

# real plot
plt.figure(2)
plt.clf()
plt.title('Real Space Trajectory')
plt.plot(q[:, 0], q[:, 1])
plt.xlabel('Position, $x$')
plt.ylabel('Position, $y$')
plt.show()

# calculation with the Stoemer-Verlet method
q = np.zeros((N + 1, 2))
p = np.zeros((N + 1, 2))
q[0, 0] = 1.0
p[0, 1] = 1.0
for ii in range(0, N):
    q[ii + 1], p[ii + 1] = step.stoemerstep(V, q[ii], p[ii], Dt)
    
# phase plot
plt.figure(3)
plt.clf()
plt.subplots_adjust(wspace = 0.3)
plt.subplot(121)
plt.title('Phase Plot in $x$')
plt.plot(q[:, 0], p[:, 0])
plt.xlabel('Position, $x$')
plt.ylabel('Momentum, $p_x$')
plt.subplot(122)
plt.title('Phase Plot in $y$')
plt.plot(q[:, 1], p[:, 1])
plt.xlabel('Position, $y$')
plt.ylabel('Momentum, $p_y$')

# real plot
plt.figure(4)
plt.clf()
plt.title('Real Space Trajectory')
plt.plot(q[:, 0], q[:, 1])
plt.xlabel('Position, $x$')
plt.ylabel('Position, $y$')
plt.show()
