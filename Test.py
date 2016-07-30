# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 07:20:57 2016
Test script.
@author: rpoolman
"""
import numpy as np
from mayavi import mlab


def plot3D(name, X, Y, Z):
    """
    Plots a 3d surface plot of Z using the mayavi mlab.mesh function.

    Parameters
    ----------
    name: string
        The name of the figure.
    X: 2d ndarray
        The x-axis data.
    Y: 2d ndarray
        The y-axis data.
    Z: 2d nd array
        The z-axis data.
    """
    mlab.figure(name)
    mlab.clf()
    plotData = mlab.mesh(X/(np.max(X) - np.min(X)),
                         Y/(np.max(Y) - np.min(Y)),
                         Z/(np.max(Z) - np.min(Z)))
    mlab.outline(plotData)
    mlab.axes(plotData, ranges=[np.min(X), np.max(X),
                                np.min(Y), np.max(Y),
                                np.min(Z), np.max(Z)])


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

reduced = True
L = 60.0
c = 0.5
T = 200.0
Dt = 0.1
t = np.arange(0, T, Dt)
Dx = L/320.0
x = np.arange(-L/2.0 - Dx, L/2.0, Dx)
u = np.zeros((len(t), len(x)))
u[0, :] = initialU(x, L, c)
v = np.zeros((len(t), len(x)))
v[0, :] = initialV(x, L, c)

# skew symmeyric matrices
Kplus = np.zeros((3, 3))
Kminus = np.zeros((3, 3))
Lplus = np.zeros((3, 3))
Lminus = np.zeros((3, 3))
Kplus[1, 0] = 1
Kminus[0, 1] = -1
Lplus[2, 0] = 1
Lminus[0, 2] = -1

# state vector
z = np.zeros((len(t), 3))

# %% calculate


def dplus(zplus, z, Delta):
    """
    Calculates forward difference.

    Parameter
    ---------
    zplus: The next step.
    z: The current step.
    Delta: The step size.

    Returns
    -------
    diff: The difference between the current and net state vector over the step
          size.
    """
    return (zplus - z)/Delta


def dminus(z, zminus, Delta):
    """
    Calculates forward difference.

    Parameter
    ---------
    zplus: The next step.
    z: The current step.
    Delta: The step size.

    Returns
    -------
    diff: The difference between the current and net state vector over the step
          size.
    """
    return (z - zminus)/Delta

for nn in range(0, len(t) - 1):
    for ii in range(0, len(x) - 1):
        
    

for nn in range(0, len(t) - 1):
    for ii in range(0, len(x) - 1):
        # takes care of cyclic boundary conditions
        iiminus = ii - 1
        if ii == 0:
            iiminus = len(x) - 1
        iiplus = ii + 1
        if (ii == len(x) - 1):
            iiplus = 0
        # step
        v[nn + 1, ii] = v[nn, ii] + \
            Dt*((u[nn, iiplus] - 2*u[nn, ii] + u[nn, iiminus])/Dx**2 -
                np.sin(u[nn, ii]))
        u[nn + 1, ii] = u[nn, ii] + Dt*v[nn + 1, ii]

# %% plot
X, T = np.meshgrid(x, t)
plot3D('Spatial Distribution', X, T, u)
plot3D('Velocity Distribution', X, T, v)
