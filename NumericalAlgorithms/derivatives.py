# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:09:39 2015

A module that contains various alogirthms relating to the numerical evalatuaion
of mathematical constructs.
@author: rpoolman
"""

import numpy as np

def __doNothing(temp):
    return temp

def derivative(func, x, h, useRidders = False):
    """
    This function calculates the numerical derivative of a function at a point 
    either by Ridders' method of polynomial extrapolation or using the 
    symmertical difference, the flag useRidders indicates which. The value of h 
    is input as an estimated initial stepsize; it need not be small, but rather
    should be an increment in x over which the function changes substantially. 
    An estimate of the error in the derivative is returned.  Translated from 
    NRiC section 5.7, pp. 231.
    
    Parameters
        func: The function which is to have it's derivative calcualted.
        x: The point at which the derivative is to be calcualted.
        h: A first guess at the stepsize for the derivative.
        useRidders: Defaults to False and causes the symmetrical difference 
                    method to be used, if set to true then the symmetrical 
                    difference is used.  If the former is used then no error is
                    returned.
    
    Returns
        The derivative df/dx evaulated at the parameter x and an estimate of 
        the error between numerical derivative and the exact value.  If 
        useRidders is set to False then no error is returned.
    """
    if useRidders:
        # sets maximu size of the tableau
        ntab = 10
        # stesize decreased by con at each iteration
        con = 1.4
        con2 = con*con
        numerical_limits = np.finfo('d')    
        big = numerical_limits.max
        # return when error is SAFE worst than the best so far
        safe = 2.0
        a = np.zeros([ntab, ntab])
        
        if h == 0.0:
            raise ValueError("derivative: Stepsize h must be non-zero.")
                
        hh = h
        a[0, 0] = (func(x + hh) - func(x - hh))/(2.0*hh)
        err = big
        for ii in range(ntab):
            # successive columns in the Neville tableau will go to smaller 
            # stepsizes and higher orders of extrapolation
            hh = hh/con
            #trying a new smaller stepsize
            a[0][ii] = (func(x + hh) - func(x - hh))/(2.0*hh)
            fac = con2
            
            for jj in range(1, ii + 1):
                a[jj, ii] = (a[jj - 1, ii]*fac - a[jj - 1, ii - 1])/(fac - 1.0)
                fac = con2*fac
                errt = np.max([np.abs(a[jj, ii] - a[jj - 1, ii]), 
                               np.abs(a[jj, ii] - a[jj - 1, ii - 1])])
                # the error strategy is to compare each new extrapolation to 
                # one order lower both at the present stepsize and the previous
                # one
                if errt <= err:
                    err = errt
                    dfuncdx = a[jj, ii]
            
            if np.abs(a[ii, ii] - a[ii - 1, ii - 1] >= safe*err):
                break
    else:
        temp = x + h
        h = __doNothing(temp) - x
        dfuncdx = (func(x + h) - func(x - h))/(2.0*h)    
        err = None
        
    return dfuncdx, err

def grad(func, x, h, useRidders = False):
    """
    This function calculates the Cartesian gradiant of a function at a point 
    using the function derivative along each axis.  An estimate of the error 
    in the derivative is returned.
    
    Parameters
        func: The function which is to have it's derivative calcualted.  Must 
              take three arguments.
        x: The Cartesian vector that defines the point at which the derivative 
           is to be calcualted.
        h: A first guess at the stepsize for the derivative.  A Cartiesian 
           vector.
        useRidders: Defaults to False and causes the symmetrical difference 
                    method to be used, if set to true then the symmetrical 
                    difference is used.  If the former is used then no error is
                    returned.
    
    Returns
        The derivative grad f evaulated at the parameter x and an estimate of 
        the error between numerical derivative and the exact value.  If 
        useRidders is set to False then no error is returned.
    """
    # check
    if len(x) != 3 and len(h) != 3:
        raise ValueError('grad: x and h should be Cartesian vectors.')
    
    # arrays
    gradf = np.zeros(3)
    err = None
    if (useRidders):
        err = np.zeros(3)
        # sets maximu size of the tableau
        ntab = 10
        # stesize decreased by con at each iteration
        con = 1.4
        con2 = con*con
        numerical_limits = np.finfo('d')    
        big = numerical_limits.max
        # return when error is SAFE worst than the best so far
        safe = 2.0
        a = np.zeros([ntab, ntab])
        
        if (h[0] == 0.0 or h[1] == 0.0 or h[2] == 0.0):
            raise ValueError("grad: All stepsize h elements must be non-zero.")
                
        hh = h
        
        # x direction
        a[0, 0] = (func(x[0] + hh[0], x[1], x[2]) - func(x[0] - hh[0], x[1], x[2]))/(2.0*hh[0])
        err[0] = big
        for ii in range(ntab):
            # successive columns in the Neville tableau will go to smaller 
            # stepsizes and higher orders of extrapolation
            hh[0] = hh[0]/con
            #trying a new smaller stepsize
            a[0][ii] = (func(x[0] + hh[0], x[1], x[2]) - func(x[0] - hh[0], x[1], x[2]))/(2.0*hh[0])
            fac = con2
            
            for jj in range(1, ii + 1):
                a[jj, ii] = (a[jj - 1, ii]*fac - a[jj - 1, ii - 1])/(fac - 1.0)
                fac = con2*fac
                errt = np.max([np.abs(a[jj, ii] - a[jj - 1, ii]), 
                               np.abs(a[jj, ii] - a[jj - 1, ii - 1])])
                # the error strategy is to compare each new extrapolation to 
                # one order lower both at the present stepsize and the previous
                # one
                if errt <= err[0]:
                    err[0] = errt
                    gradf[0] = a[jj, ii]
            
            if np.abs(a[ii, ii] - a[ii - 1, ii - 1] >= safe*err[0]):
                break
        
        # y direction
        a[0, 0] = (func(x[1], x[1] + hh[1], x[2]) - func(x[0], x[1] - hh[1], x[2]))/(2.0*hh[1])
        err[1] = big
        for ii in range(ntab):
            # successive columns in the Neville tableau will go to smaller 
            # stepsizes and higher orders of extrapolation
            hh[1] = hh[1]/con
            #trying a new smaller stepsize
            a[0][ii] = (func(x[1], x[1] + hh[1], x[2]) - func(x[0], x[1] - hh[1], x[2]))/(2.0*hh[1])
            fac = con2
            
            for jj in range(1, ii + 1):
                a[jj, ii] = (a[jj - 1, ii]*fac - a[jj - 1, ii - 1])/(fac - 1.0)
                fac = con2*fac
                errt = np.max([np.abs(a[jj, ii] - a[jj - 1, ii]), 
                               np.abs(a[jj, ii] - a[jj - 1, ii - 1])])
                # the error strategy is to compare each new extrapolation to 
                # one order lower both at the present stepsize and the previous
                # one
                if errt <= err[1]:
                    err[1] = errt
                    gradf[1] = a[jj, ii]
            
            if np.abs(a[ii, ii] - a[ii - 1, ii - 1] >= safe*err[1]):
                break
        
        # z direction
        a[0, 0] = (func(x[0], x[1], x[2] + hh[2]) - func(x[0], x[1], x[2] - hh[2]))/(2.0*hh[2])
        err[2] = big
        for ii in range(ntab):
            # successive columns in the Neville tableau will go to smaller 
            # stepsizes and higher orders of extrapolation
            hh[2] = hh[2]/con
            #trying a new smaller stepsize
            a[0][ii] = (func(x[0], x[1], x[2] + hh[2]) - func(x[0], x[1], x[2] - hh[2]))/(2.0*hh[2])
            fac = con2
            
            for jj in range(1, ii + 1):
                a[jj, ii] = (a[jj - 1, ii]*fac - a[jj - 1, ii - 1])/(fac - 1.0)
                fac = con2*fac
                errt = np.max([np.abs(a[jj, ii] - a[jj - 1, ii]), 
                               np.abs(a[jj, ii] - a[jj - 1, ii - 1])])
                # the error strategy is to compare each new extrapolation to 
                # one order lower both at the present stepsize and the previous
                # one
                if errt <= err[2]:
                    err[2] = errt
                    gradf[2] = a[jj, ii]
            
            if np.abs(a[ii, ii] - a[ii-1, ii - 1] >= safe*err[2]):
                break
    
        return gradf, err
    else:
        temp = x + h
        h = __doNothing(temp) - x
        gradf = np.nan_to_num((func(x[0] + h[0], x[1], x[2]) - func(x[0] - h[0], x[1], x[2]))/(2*h))
#        gradf[0] = (func(x[0] + h[0], x[1], x[2]) - func(x[0] - h[0], x[1], x[2]))/(2*h[0])
#        gradf[1] = (func(x[0], x[1] + h[1], x[2]) - func(x[0], x[1] - h[1], x[2]))/(2*h[1])
#        gradf[2] = (func(x[0], x[1], x[2] + h[2]) - func(x[0], x[1], x[2]) - h[2])/(2*h[2])
    
        return gradf
    
def grad2D(func, x, h, useRidders = False):
    """
    This function calculates the 2D Cartesian gradiant of a function at a 
    point using the function derivative along each axis.  An estimate of the 
    error in the derivative is returned.
    
    Parameters
        func: The function which is to have it's derivative calcualted.  Must 
              take three arguments.
        x: The 2D Cartesian vector that defines the point at which the 
           derivative is to be calculated.
        h: A first guess at the stepsize for the derivative.  A 2D Cartiesian 
           vector.
        useRidders: Defaults to False and causes the symmetrical difference 
                    method to be used, if set to true then the symmetrical 
                    difference is used.  If the former is used then no error is
                    returned.
    
    Returns
        The derivative grad f evaulated at the parameter x and an estimate of 
        the error between numerical derivative and the exact value.  If 
        useRidders is set to False then no error is returned.
    """
    # check
    if len(x) != 2 and len(h) != 2:
        raise ValueError('grad2D: x and h should be 2D Cartesian vectors.')
    
    # arrays
    gradf = np.zeros(2)
    err = None
    if (useRidders):
        err = np.zeros(2)
        # sets maximu size of the tableau
        ntab = 10
        # stesize decreased by con at each iteration
        con = 1.4
        con2 = con*con
        numerical_limits = np.finfo('d')    
        big = numerical_limits.max
        # return when error is SAFE worst than the best so far
        safe = 2.0
        a = np.zeros([ntab, ntab])
        
        if (h[0] == 0.0 or h[1] == 0.0):
            raise ValueError("grad: All stepsize h elements must be non-zero.")
                
        hh = h
        
        # x direction
        a[0, 0] = (func(x[0] + hh[0], x[1]) - func(x[0] - hh[0], x[1]))/(2.0*hh[0])
        err[0] = big
        for ii in range(ntab):
            # successive columns in the Neville tableau will go to smaller 
            # stepsizes and higher orders of extrapolation
            hh[0] = hh[0]/con
            #trying a new smaller stepsize
            a[0][ii] = (func(x[0] + hh[0], x[1]) - func(x[0] - hh[0], x[1],))/(2.0*hh[0])
            fac = con2
            
            for jj in range(1, ii + 1):
                a[jj, ii] = (a[jj - 1, ii]*fac - a[jj - 1, ii - 1])/(fac - 1.0)
                fac = con2*fac
                errt = np.max([np.abs(a[jj, ii] - a[jj - 1, ii]), 
                               np.abs(a[jj, ii] - a[jj - 1, ii - 1])])
                # the error strategy is to compare each new extrapolation to 
                # one order lower both at the present stepsize and the previous
                # one
                if errt <= err[0]:
                    err[0] = errt
                    gradf[0] = a[jj, ii]
            
            if np.abs(a[ii, ii] - a[ii - 1, ii - 1] >= safe*err[0]):
                break
        
        # y direction
        a[0, 0] = (func(x[1], x[1] + hh[1]) - func(x[0], x[1] - hh[1]))/(2.0*hh[1])
        err[1] = big
        for ii in range(ntab):
            # successive columns in the Neville tableau will go to smaller 
            # stepsizes and higher orders of extrapolation
            hh[1] = hh[1]/con
            #trying a new smaller stepsize
            a[0][ii] = (func(x[1], x[1] + hh[1]) - func(x[0], x[1] - hh[1]))/(2.0*hh[1])
            fac = con2
            
            for jj in range(1, ii + 1):
                a[jj, ii] = (a[jj - 1, ii]*fac - a[jj - 1, ii - 1])/(fac - 1.0)
                fac = con2*fac
                errt = np.max([np.abs(a[jj, ii] - a[jj - 1, ii]), 
                               np.abs(a[jj, ii] - a[jj - 1, ii - 1])])
                # the error strategy is to compare each new extrapolation to 
                # one order lower both at the present stepsize and the previous
                # one
                if errt <= err[1]:
                    err[1] = errt
                    gradf[1] = a[jj, ii]
            
            if np.abs(a[ii, ii] - a[ii - 1, ii - 1] >= safe*err[1]):
                break        
        return gradf, err
    else:
        temp = x + h
        h = __doNothing(temp) - x
        gradf[0] = (func(x[0] + h[0], x[1]) - func(x[0] - h[0], x[1]))/(2*h[0])
        gradf[1] = (func(x[0], x[1] + h[1]) - func(x[0], x[1] - h[1]))/(2*h[1])
        return gradf