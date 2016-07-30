# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 13:29:11 2015

Section 5.4, Ex. 6 Kepler Problem and Modified Equations
Apply the Stoermer-Verlet method to the planar Kepler problem
                    H(q, p) = 1/2p^Tp - 1/||q||
Use the BCH formula eq. (5.40) to compute the second-order corrections of the 
modified Hamiltonian \tilde{H} for this particular problem.  Verify the 
forth-order convergence of the Stoermer-Verlet  method with respect to the 
modified Hamiltonian \tilde{H}_2 numerically. Take, for example, initial 
conditions q = (1, 0)^T and p = (1, 0)^T.
@author: Rhys Poolman
"""
import numpy as np
import matplotlib.pylab as plt
import Steppers.steppers as step

# utility functions
def __doNothing(temp):
    return temp
    
# constants and arrays
Dt = 0.01
T = 4*np.pi
N = np.int(T/Dt)
t = np.linspace(0, T, N + 1)
q = np.zeros((N + 1, 2))
p = np.zeros((N + 1, 2))
grad = np.zeros(2)
V = lambda qx, qy: -1/np.sqrt(qx**2 + qy**2)
qmod = np.zeros((N + 1, 2))
pmod = np.zeros((N + 1, 2))
Vmod = lambda qx, qy, px, py, Dt: V(qx, qy) - \
                                  Dt**2/24*(px**2 + py**2)* \
                                  (np.sqrt(qx**2 + qy**2)**3 - \
                                   3*(qx*px + qy*py)/np.sqrt(qx**2 + qy**2)**5)

# intial values
q[0, 0] = 1.0
p[0, 1] = 1.0
qmod[0, 0] = 1.0
pmod[0, 1] = 1.0

# calculation with the Stoemer-Verlet method
for ii in range(0, N):
    # for Hamiltonian H
    q[ii + 1], p[ii + 1] = step.stoemerstep(V, q[ii], p[ii], Dt)
    # for modified Hamiltonian H_mod
    h = Dt/2*np.max(pmod[ii])*np.ones(2)
    temp = qmod[ii] + h
    h = __doNothing(temp) - qmod[ii]
    grad[0] = (Vmod(qmod[ii, 0] + h[0], qmod[ii, 1], pmod[ii, 0], pmod[ii, 1], Dt) \
               - Vmod(qmod[ii, 0] - h[0], qmod[ii, 1], pmod[ii, 0], pmod[ii, 1], Dt))/(2*h[0])
    grad[1] = (Vmod(qmod[ii, 0], qmod[ii, 1] + h[1], pmod[ii, 0], pmod[ii, 1], Dt) \
               - Vmod(qmod[ii, 0], qmod[ii, 1] - h[1], pmod[ii, 0], pmod[ii, 1], Dt))/(2*h[1])
    pnHalf = pmod[ii] - 0.5*Dt*grad
    qmod[ii + 1] = qmod[ii] + Dt*0.5*pnHalf
    pmod[ii + 1] = pnHalf - 0.5*Dt*Dt*grad
    
# phase plot
plt.figure(1)
plt.clf()
# phase plot in x
ax1 = plt.subplot2grid((2, 2), (0, 0))
ax1.set_title('Phase Plot in $x$')
ax1.plot(q[:, 0], p[:, 0], 'b--',
         qmod[:, 0], pmod[:, 0], 'b')
ax1.set_xlabel('Position, $x$')
ax1.set_ylabel('Momentum, $p_x$')
# phase plot in y
ax2 = plt.subplot2grid((2, 2), (0, 1))
ax2.set_title('Phase Plot in $y$')
ax2.plot(q[:, 1], p[:, 1], 'b--',
         qmod[:, 1], pmod[:, 1], 'b')
ax2.set_xlabel('Position, $y$')
ax2.set_ylabel('Momentum, $p_y$')
# real space trajectory
ax3 = plt.subplot2grid((2, 2), (1, 0))
ax3.set_title('Real Space Trajectory')
ax3.plot(q[:, 0], q[:, 1], 'b--',
         qmod[:, 0], qmod[:, 1], 'b')
ax3.set_xlabel('Position, $x$')
ax3.set_ylabel('Position, $y$')
ax3.axis([-1.0, 1.0, -1.0, 1.0])
# real space momentum
ax4 = plt.subplot2grid((2, 2), (1, 1))
ax4.set_title('Real Space Momentum')
ax4.plot(p[:, 0], p[:, 1], 'b--',
         pmod[:, 0], pmod[:, 1], 'b')
ax4.set_xlabel('Position, $x$')
ax4.set_ylabel('Position, $y$')
ax4.axis([-2.0, 2.0, -3.0, 1.0])
