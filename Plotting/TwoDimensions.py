# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 07:41:25 2016

A module to allow for two dimensional trajectories and phases spaces to be 
plotted simply.
@author: rpoolman
"""
import matplotlib.pyplot as plt

def plotTrajectory(fig, result, axisRange = None):
    """
    A simple 2D line plot of the trajectory of a particle.
    
    Parameters:
        fig - An integer figure number.
        result - A tuple containing both the trajectory in a Nx3 array, 
                 the key is used in the title
        axisRange - A four element one dimensional array that controls axis 
                    size.  Default value is None.
    """
    q = result[1][0]
    plt.figure(fig)
    plt.clf()
    plt.title('Trajectory Plot for {}'.format(result[0]))
    plt.plot(q[:, 0], q[:, 1], '.-')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    if axisRange:
        plt.axis(axisRange)

def plotPhaseSpace(fig, p, q, axisRangeX = None, axisRangeY = None):
    """
    A simple 2D line plot of the phase of a particle in both x and y.
    
    Parameters:
        fig - An integer figure number.
        p - The trajectory data to be plotted in a two dimensional array. 
            The first dimensions is the x-coordinate and the second the
            y-coordinate.
        q - The momentum data to be plotted in a two dimensional array. 
            The first dimensions is the x-coordinate and the second the
            y-coordinate.
        axisRangeX - A four element one dimensional array that controls axis 
                     size of x dimension phase space plot.  Default value is 
                     None.
        axisRangeY - A four element one dimensional array that controls axis 
                     size of y dimension phase space plot.  Default value is 
                     None.
    """
    plt.figure(fig)
    plt.clf()
    plt.suptitle('Phase Diagram')
    # along real space x direction
    plt.subplot(121)
    plt.title('Phase Diagram Along Real Space X Coordinate')
    plt.plot(q[:, 0], p[:, 0], '.-')
    plt.xlabel('Positon, $q_x$')
    plt.ylabel('Momentum, $p_x$')
    if axisRangeX:
        plt.axis(axisRangeX)
    # along real space y direction
    plt.subplot(122)
    plt.title('Phase Diagram Along Real Space Y Coordinate')
    plt.plot(q[:, 1], p[:, 1], '.-')
    plt.xlabel('Positon, $q_y$')
    plt.ylabel('Momentum, $p_y$')
    if axisRangeY:
        plt.axis(axisRangeY)

def plot(fig, q, p, axisRangeX = None, axisRangeY = None):
    """
    A simple 2D line plot of the phase of a particle in both x and y.
    
    Parameters:
        fig - An integer figure number.
        q - The trajectory data to be plotted in a two dimensional array. 
            The first dimensions is the x-coordinate and the second the
            y-coordinate.
        p - The momentum data to be plotted in a two dimensional array. 
            The first dimensions is the x-coordinate and the second the
            y-coordinate.
        axisRangeX - A four element one dimensional array that controls axis 
                     size of x dimension phase space plot.  Default value is 
                     None.
        axisRangeY - A four element one dimensional array that controls axis 
                     size of y dimension phase space plot.  Default value is 
                     None.
    """
    plt.figure(fig)
    plt.subplots_adjust(hspace = 0.2, wspace = 0.15)
    plt.clf()
    # plot trajectory
    plt.subplot2grid((2,2), (0,0), colspan=2)
    plt.title('Real Space Plot of Numerical Solution')
    plt.plot(q[:, 0], q[:, 1])
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    #plt.axis([-1.5, 1.5, -1.5, 1.5])
    # plot phase space of x coordinate
    plt.subplot2grid((2,2), (1,0))
    plt.title('Phase Space Plot of Numerical Solution along X')
    plt.plot(q[:, 0], p[:, 0])
    plt.xlabel('X Coordinate')
    plt.ylabel('X Velocity')
    #plt.axis([-1.5, 1.5, -1.5, 1.5])
    # plot phase space of y coordinate
    plt.subplot2grid((2,2), (1,1))
    plt.title('Phase Space Plot of Numerical Solution along X')
    plt.plot(q[:, 1], p[:, 1])
    plt.xlabel('Y Coordinate')
    plt.ylabel('Y Velocity')
    #plt.axis([-1.5, 1.5, -1.5, 1.5])