# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 07:15:20 2016
Section 12.4 Exercise 3 - Sine-Gordon Equation, Energy-Momentum Conservation

Implement the Euler box scheme for the sine-Gordon equation over a periodic
domain x belonging to (-L/2, L/2], L = 60, and initial data

    u0(x) = 4tan^-1(exp((x - L/4)/sqrt(1 - c^2))) +
            4tan^-1(exp((-x - L/4)/sqrt(1 - c^2)))

and

    v0(x) = -4c/sqrt(1 - c^2)*
            ((exp(x - L/4)/sqrt(1 - c^2))/(1 + exp(2(x - L/4)/sqrt(1 - c^2))) -
            (exp(-x - L/4)/sqrt(1 - c^2))/(1 + exp(2(-x - L/4)/sqrt(1 - c^2))))

with wave speed c = 0.5.

a. Compute the numerical solution over a time-interval [0, 200] using a
stepsize of Dt = 0.01 and a sptail mesh size of Dx = L/3200.

b. Implment the formula Eqn. (L + R 12.27)

    R_i^n+1/2 = (E_i^n+1 - E_i^n)/Dt + (F_i+1/2^n+1/2 - F_i-1/2^n+1/2)/Dx

into your scheme to monitor the residual in the energy conservation law.  You
should reproduce the resutls from L + R Fig 12.3.

c. Find the analog of (L + R 12.27) for the momentum conservation law.
Implement the formula into your scheme and monitor the residual in the discrete
momentum conservation law.
@author: rpoolman
"""
import numpy as np
import Steppers.pde_steppers as pde
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Plotting.ThreeDimensions import plot3D


def initialU(x, L, c):
    """
    A function to calculate the inital u value for a given x.

    Paramteres:
        x - The spatial coordinate.
        L - The domain size.
        c - The wave speed.

    Returns:
        v0 - The inital value of the wave function.
    """
    return 4*np.arctan(np.exp((x - L/4)/np.sqrt(1 - c**2))) + \
        4*np.arctan(np.exp((-x - L/4)/np.sqrt(1 - c**2)))


def initialV(x, L, c):
    """
    A function to calculate the inital u value for a given x.

    Paramteres:
        x - The spatial coordinate.
        L - The domain size.
        c - The wave speed.

    Returns:
        v0 - The inital value of the wave function.
    """
    return -4.0*c/np.sqrt(1 - c**2) * \
        (np.exp((x - L/4.0)/np.sqrt(1 - c**2)) /
         (1 + np.exp(2*(x - L/4.0)/np.sqrt(1 - c**2))) -
         np.exp((-x - L/4.0)/np.sqrt(1 - c**2)) /
         (1 + np.exp(2*(-x - L/4.0)/(np.sqrt(1 - c**2)))))


def update_line(num, x, y, line):
    """
    Update data for animation by producing a series of frames.

    Pameters:
        num - The number of frames.
        x - The independent variable data, consists of a one dimensional array
            of x coordinates.
        y - The dependent variable data, a two dimensional array of y
            coordinates. Must have the same numbero f columsn as the x array
            argument.  The rows are cycled over by the num integer and thus act
            as a time dimension.
        line - The plot of the line to be created

    Returns:
        The output is the line argument once the set data command has been
        called.

    Returns:
        Outputs a line consisting of x and y coordinates.
    """
    line.set_data(x, y[num, :])
    return line,

full = False
reduced = False
L = 60.0
c = 0.5
T = 200.0
Dt = 0.01
if reduced:
    Dt = Dt*10
t = np.arange(0, T, Dt)
Dx = L/3200.0
if reduced:
    Dx = Dx*10
x = np.arange(-L/2.0 - Dx, L/2.0, Dx)
u = np.zeros((len(t), len(x)))
u[0, :] = initialU(x, L, c)
v = np.zeros((len(t), len(x)))
v[0, :] = initialV(x, L, c)
eulerStep = pde.eulerBoxStep(Dt, Dx, lambda x: 1 - np.cos(x),
                             lambda x: np.sin(x), lambda x: x**2/2,
                             lambda x: x)
residual = np.zeros((len(t), len(x)))

# %% calculate
for nn in range(0, len(t) - 1):
    for ii in range(0, len(x)):
        # takes care of cyclic boundary conditions
        iiminus = ii - 1
        if ii == 0:
            iiminus = len(x) - 1
        iiplus = ii + 1
        if ii == len(x) - 1:
            iiplus = 0

        # calculate wave function and velocity
        u_in = np.array([u[nn, iiplus], u[nn, ii], u[nn, iiminus]])
        u[nn + 1, ii], v[nn + 1, ii] = eulerStep(u_in, v[nn, ii])

        # calculate residual
        if nn > 1:
            u_in = np.array(
                [[u[nn - 2, iiplus], u[nn - 2, ii], u[nn - 2, iiminus]],
                 [u[nn - 1, iiplus], u[nn - 1, ii], u[nn - 1, iiminus]],
                 [u[nn, iiplus], u[nn, ii], u[nn, iiminus]]])
            v_in = np.array(
                [[v[nn - 2, iiplus], v[nn - 2, ii], v[nn - 2, iiminus]],
                 [v[nn - 1, iiplus], v[nn - 1, ii], v[nn - 1, iiminus]],
                 [v[nn, iiplus], v[nn, ii], v[nn, iiminus]]])
            residual[nn + 1, ii] = eulerStep.residual(u_in, v_in)

# %% plot
if reduced and not full:
    minxIndex = 108
    maxxIndex = 215
    mintIndex = 800
    maxtIndex = 1200
elif not full:
    minxIndex = 1067
    maxxIndex = 2135
    mintIndex = 8000
    maxtIndex = 12000
else:
    minxIndex = 0
    maxxIndex = len(x) - 1
    mintIndex = 0
    maxtIndex = len(t) - 1


# animate
fig = plt.figure(1)
plt.clf()
# animate spatial distribution
u_an = fig.add_subplot(311)
l, = plt.plot([], [])
plt.xlim(x[minxIndex], x[maxxIndex - 1])
plt.ylim(np.min(u) + 0.1*np.min(u), np.max(u) + 0.1*np.max(u))
line_ani_u = animation.FuncAnimation(fig, update_line,
                                     fargs=(x[minxIndex:maxxIndex],
                                            u[mintIndex:maxtIndex,
                                              minxIndex:maxxIndex], l),
                                     interval=50, blit=True, repeat=500)
# aninmate momentum
v_an = fig.add_subplot(312)
l, = plt.plot([], [])
plt.xlim(x[minxIndex], x[maxxIndex - 1])
plt.ylim(np.min(v) + 0.1*np.min(v), np.max(v) + 0.1*np.max(v))
line_ani_v = animation.FuncAnimation(fig, update_line,
                                     fargs=(x[minxIndex:maxxIndex],
                                            v[mintIndex:maxtIndex,
                                              minxIndex:maxxIndex], l),
                                     interval=50, blit=True, repeat=500)
# aninmate residual
res_an = fig.add_subplot(313)
l, = plt.plot([], [])
plt.xlim(x[minxIndex], x[maxxIndex - 1])
plt.ylim(np.min(residual) + 0.1*np.min(residual),
         np.max(residual) + 0.1*np.max(residual))
line_ani_res = animation.FuncAnimation(fig, update_line,
                                       fargs=(x[minxIndex:maxxIndex],
                                              residual[mintIndex:maxtIndex,
                                              minxIndex:maxxIndex], l),
                                       interval=50, blit=True, repeat=500)

# 3d plot
X, T = np.meshgrid(x, t)
plot3D('Spatial Distribution', X, T, u, 'Wavefunction (u)')
plot3D('Velocity Distribution', X, T, v, 'Velocity (v)')
plot3D('Residual Distribution', X, T, residual, 'Energy Resudual (E)')
