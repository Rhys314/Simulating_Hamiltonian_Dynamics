# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 07:20:57 2016
Section 9.7 Exercise 4 Constrained dynamics and adaptivity

Develop an adaptive method for simulating a constrained Hamiltonian system
such as the spherical pendulum (unit length, unit mass and unit gravitational
constant) subject to soft collisions (inverse power repulsion) with a fixed
body at position q_0.  The equations of motion can be written as

                        dq/dt = M^-1p
                        dp/dt = -grad_q(V(q)) - q*lambda
                        ||q||^2 = 1

where the potential

                        V(q) = z + phi(|q - q_0|),          phi = r^(-alpha)

is the sum of the graviational potential and the distance dependent
interaction potential between the pendulum bob and fixed body.  The Sundman
transformation should be chosen to reduces stepsize in the vicinity of the
collision between the bob and the fixed body.  Test your method and report on
its relative efficiency as a function of the power alpha in the repulsive wall
and the choice of the Sundman transformation function.
@author: rpoolman
"""
import numpy as np
from numpy.linalg import norm
import Steppers.constrained as sc
import Steppers.adaptive as adp
import Steppers.steppers as stp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# functions
def sundmanTrans(q): return norm(q)**3.0/2.0


def g(q): return norm(q)**2 - 1


def G(q): return 2*q


def V_pen(q): return q*norm(q)**-3.0


class V_pencons:
    def __init__(self, stop, alpha):
        self.stop = stop
        self.alpha = alpha

    def grav(self, q): return q*norm(q)**3.0

    def collision(self, q):
        return -alpha*norm(q - self.stop)**-(self.alpha + 1)

    def __call__(self, q): return self.grav(q) + self.collision(q)


class HamiltonianDerivativeWRTp:
    def __init__(self, invM): self.invM = invM

    def __call__(self, q, p): return np.dot(self.invM, p)

# constants
N = 100
T = 2*(np.pi + np.pi/100)
Dt = T/N
useConstraint = True
alphas = np.array([0.001, 0.01, 0.1, 1.0, 10.0])
M = np.diag(np.ones(3))
invM = np.linalg.inv(M)
stop = np.array([0.0, 1.0, 0.0])

# iterate over alpha values
averageEfficiency = np.zeros(5)
q = {}
p = {}
for kk, alpha in enumerate(alphas):
    # set derivative of potential w.r.t spatial coordinate
    if useConstraint:
        dVdq = V_pencons(stop, alpha)
    else:
        dVdq = V_pen

    # initial values
    q['alpha = {}'.format(alpha)] = np.empty((N, 3))
    q['alpha = {}'.format(alpha)][0] = np.array([1.0, 0.0, 0.0],
                                                dtype=np.float64)
    p['alpha = {}'.format(alpha)] = np.empty((N, 3))
    p['alpha = {}'.format(alpha)][0] = np.array([0.0, 1.0, 0.0],
                                                dtype=np.float64)

    # define unconstarined step
    psi = stp.Stoemer_Verlet(dVdq, Dt/2)

    # create stepper with half step
    stoemer = sc.ConstrainedSympleticStep(psi, g, G, Dt/2)

    # define adjoint step (Stoemer-Verlet is symmetric)
    psi_adj = stp.Stoemer_Verlet(dVdq, Dt/2)

    # create adjoint stepper with half step (can't assign value that works by
    # reference, could deep copy but fuck it!)
    stoemer_adj = sc.ConstrainedSympleticStep(psi_adj, g, G, Dt/2)

    # create adaptive stepper and integrate
    stepper = \
        adp.SundmanAdpaptiveStep(stoemer, stoemer_adj, sundmanTrans,
                                 sundmanTrans(q['alpha = {}'.
                                              format(alpha)][0]))

    # calculate trajectory
    for ii in range(N - 1):
        q['alpha = {}'.format(alpha)][ii + 1], \
            p['alpha = {}'.format(alpha)][ii + 1] = \
            stepper(q['alpha = {}'.format(alpha)][ii],
                    p['alpha = {}'.format(alpha)][ii])
        averageEfficiency[kk] = averageEfficiency[kk] + \
            stepper.stepEfficiency/N

# calculate potential
M = np.int(N/10)
X, Y, Z = np.meshgrid(np.linspace(-1.0, 1.0, M),
                      np.linspace(-1.0, 1.0, M),
                      np.linspace(-1.0, 1.0, M))
grad_qV = np.empty((M, M, M, 3))
for ii in range(M):
    for jj in range(M):
        for kk in range(M):
            grad_qV[ii, jj, kk] = dVdq(np.array([X[ii, jj, kk],
                                                 Y[ii, jj, kk],
                                                 Z[ii, jj, kk]]))

# plot
alpha = alphas[4]
plt.figure(1)
plt.clf()
plt.plot(q['alpha = {}'.format(alpha)][:, 0],
         q['alpha = {}'.format(alpha)][:, 1], 'x-')
plt.plot(np.array([stop[0], stop[0]]),
         np.array([stop[1] - 0.1, stop[1] + 0.1]),
         'r')
plt.axis([-1.1, 1.1, -1.1, 1.1])
plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')

fig = plt.figure(2)
plt.clf()
ax = fig.gca(projection='3d')
ax.plot(q['alpha = {}'.format(alpha)][:, 0],
        q['alpha = {}'.format(alpha)][:, 1],
        q['alpha = {}'.format(alpha)][:, 2], 'x-')
ax.quiver(X, Y, Z,
          grad_qV[:, :, :, 0], grad_qV[:, :, :, 1], grad_qV[:, :, :, 2],
          length=0.1)
# cross line 1
plt.plot(np.array([stop[0], stop[0]]),
         np.array([stop[1] - 0.1, stop[1] + 0.1]),
         np.array([stop[2] + 0.1, stop[2] - 0.1]),
         'r')
# cross line 2
plt.plot(np.array([stop[0], stop[0]]),
         np.array([stop[1] - 0.1, stop[1] + 0.1]),
         np.array([stop[2] - 0.1, stop[2] + 0.1]),
         'r')
# horizontal line 1
plt.plot(np.array([stop[0], stop[0]]),
         np.array([stop[1] - 0.1, stop[1] + 0.1]),
         np.array([stop[2] - 0.1, stop[2] - 0.1]),
         'r')
# horizontal line 2
plt.plot(np.array([stop[0], stop[0]]),
         np.array([stop[1] - 0.1, stop[1] + 0.1]),
         np.array([stop[2] + 0.1, stop[2] + 0.1]),
         'r')
# vertical line 1
plt.plot(np.array([stop[0], stop[0]]),
         np.array([stop[1] - 0.1, stop[1] - 0.1]),
         np.array([stop[2] - 0.1, stop[2] + 0.1]),
         'r')
# vertical line 2
plt.plot(np.array([stop[0], stop[0]]),
         np.array([stop[1] + 0.1, stop[1] + 0.1]),
         np.array([stop[2] - 0.1, stop[2] + 0.1]),
         'r')
ax.axis([-1.1, 1.1, -1.1, 1.1])
ax.set_xlabel('x-coordinate')
ax.set_ylabel('y-coordinate')
ax.set_zlabel('z-coordinate')

plt.figure(3)
plt.clf()
plt.semilogx(alphas, averageEfficiency, '-x')
plt.xlabel(r'Value of $\alpha$')
plt.ylabel('Average Number of Force Evaluations per Unit Time')
