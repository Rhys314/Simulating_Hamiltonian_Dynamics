# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:09:39 2015

A module that contains various alogirthms relating to the numerical evalatuaion
of mathematical constructs.
@author: rpoolman
"""

import numpy as np

def derivative(func, x, h):
    """
    This function calcuates the numerical derivative of a function at a point 
    by Ridders' method of polynomial extrapolation.  The value of h is input as
    an estimated initial stepsize; it need not be small, but rather should be 
    an increment in x over which the function changes substantially.  An 
    estimate of the error in the derivative is returned.  Translated from NRiC 
    section 5.7, pp. 231.
    
    Parameters
        func: The function which is to have it's derivative calcualted.
        x: The point at which the derivative is to be calcualted.
        h: A first guess at the stepsize for the derivative
    
    Returns
        The derivative df/dx evaulated a the parameter x and an esitmate of the
        error between numerical derivative and the exact value.
    """
    # sets maximu size of the tableau
    ntab = 10
    # stesize decreased by con at each iteration
    con = 1.4
    con2 = con*con
    numerical_limits = np.finfo()    
    big = numerical_limits.max
    # return when error is SAFE worst than the best so far
    safe = 2.0
    a = np.zeros(ntab, ntab)
    
    if h == 0.0:
        raise ValueError("derivative: Stepsize h must be non-zero."):
            
    hh = h
    a[0, 0] = (func(x + hh) - func(x - hh))/(2.0*hh)
    err = big
    for ii in range(ntab):
        # successive columns in the Neville tableau will go to smaller 
        # stepsizes and higher orders of extrapolation
        hh = hh/con
        #trying a new smaller stepsize
        a[0][i] = (func(x + hh) - func(x - hh))/(2.0*hh)
        fac = con2
        
        for jj in range(1, i + 1):
            a[jj, ii] = (a[jj - 1, ii]*fac - a[jj - 1, ii - 1])/(fac - 1.0)
            fac = con2*fac
    