# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 07:18:24 2016

A module to allow for three dimensional trajectories and phases spaces to be
plotted simply.
@author: rpoolman
"""
import numpy as np
from mayavi import mlab


def plot3D(name, X, Y, Z, zlabel):
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
    zlabel: The title that appears on the z-axis.
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
    mlab.xlabel('Space ($x$)')
    mlab.ylabel('Time ($t$)')
    mlab.zlabel(zlabel)
