# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 17:17:57 2015
Kepler problem and modified equations.
Apply the Stoermer-Verlet to the plannar Kepler problem
                        H(q, p) = 1/2p^Tp - 1/||q||,     q, p belong to R^2.
Use the BCH formular (L + R 5.40) to compute the second-order correction of 
the modified Hamiltonian \tilde{H} for this particular problem.  Verify the 
forth order convergence of the Stoermer-Verlet method with respect to the 
modified Hamiltonian  \tilde{H}_2 numerically.  Take, for example, initial 
conditions q = (1, 0) and p = (0, 1).
@author: rpoolman
"""
import Steppers.steppers as step
import numpy as np
import matplotlib.pyplot as plt

# setups for Kepler problem
V = lambda qx, qy: -1.0/np.sqrt(qx**2.0 + qy**2.0)
Dt = 0.01
T = 10
N = np.int(T/Dt)
q = np.zeros((N, 2))
p = np.zeros((N, 2))
q[0, :] = np.array([1.0, 0.0])
p[0, :] = np.array([0.0, 1.0])

# integrate
#for qRow, pRow in zip(q, p):
for ii in range(len(q) - 1):
    q[ii + 1], p[ii + 1] = step.stoermerstep(V, q[ii], p[ii], Dt)

# plot results
plt.figure(1)
plt.subplots_adjust(hspace = 0.2, wspace = 0.15)
plt.clf()
ax1 = plt.subplot2grid((2,2), (0,0), colspan=2)
plt.title('Real Space Plot of Numerical Solution')
plt.plot(q[:, 0], q[:, 1])
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.axis([-1.5, 1.5, -1.5, 1.5])
ax2 = plt.subplot2grid((2,2), (1,0))
plt.title('Phase Space Plot of Numerical Solution along X')
plt.plot(q[:, 0], p[:, 0])
plt.xlabel('X Coordinate')
plt.ylabel('X Velocity')
plt.axis([-1.5, 1.5, -1.5, 1.5])
ax3 = plt.subplot2grid((2,2), (1,1))
plt.title('Phase Space Plot of Numerical Solution along X')
plt.plot(q[:, 1], p[:, 1])
plt.xlabel('Y Coordinate')
plt.ylabel('Y Velocity')
plt.axis([-1.5, 1.5, -1.5, 1.5])