# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 07:51:03 2015

A module containing a variety of steppers as defined in Simiulating Hamiltonian 
Dynamics.
@author: rpoolman
"""
import numpy as np
import NumericalAlgorithms.derivatives as deriv

def eulerstep(phi, qn, vn, Dt, M):
    """
    This function takes one timestep across a set of equations dq/dt = v, 
    dv/dt = -dphi(q)/dq using the Euler method, where q is a generalised 
    spatial coordinate, v the velocity, t is time and phi is the potential.
    
    Parameters    
        phi: A function defining the potential of the system.
        qn: The initial spatial coordinate at the start of the timestep.
        vn: The initial velocity at the start of the timestep.
        Dt: The length of the time step.
        M: The mass of the particle acting under a force.
    
    Returns
        The function returns the value of the spatial coordinate and velocity 
        after taking a timestep of length Dt using the Euler method.
    """
    qn1 = qn + Dt*vn
    # ensuring h is an exact machine represenable number (NRiC)
    # note using the symmetrized for of the numerical derivative (see NRiC)
    # may want to consider Richardson extrapolation for serious work, again see
    # NRiC pp. 231
#    vn1 = vn - Dt*((phi(qn + h) - phi(qn - h))/(2*h))
    useRidders = True
    if useRidders:
        dphidq, err = deriv.derivative(phi, qn, np.abs(qn1 - qn), useRidders)
    else:
        dphidq = deriv.derivative(phi, qn, np.abs(qn1 - qn), useRidders)
    vn1 = vn - (1/M)*Dt*dphidq
    return qn1, vn1

def eulerAstep(phi, qn, vn, Dt, M):
    """
    This function takes one timestep across a set of equations dq/dt = v, 
    M*dv/dt = -dphi(q)/dq using the Euler A method, where q is a generalised 
    spatial coordinate, v the velocity, t is time and phi is the potential.
    
    Parameters    
        phi: A function defining the potential of the system.
        qn: The initial spatial coordinate at the start of the timestep.
        vn: The initial velocity at the start of the timestep.
        Dt: The length of the time step.
        M: The mass of the particle acting under a force.
    
    Returns
        The function returns the value of the spatial coordinate and velocity 
        after taking a timestep of length Dt using the Euler A method.
    """
    qn1 = qn + Dt*vn
    dphidq, err = deriv.derivative(phi, qn, np.abs(qn1 - qn))
    vn1 = vn - (1/M)*Dt*dphidq
    return qn1, vn1


def eulerBstep(phi, qn, vn, Dt, M):
    """
    This function takes one timestep across a set of equations dq/dt = v,
    M*dv/dt = -dphi(q)/dq using the Euler B method, where q is a generalised
    spatial coordinate, v the velocity, t is time and phi is the potential.

    Parameters
        phi: A function defining the potential of the system.
        qn: The initial spatial coordinate at the start of the timestep.
        vn: The initial velocity at the start of the timestep.
        Dt: The length of the time step.
        M: The mass of the particle acting under a force.

    Returns
        The function returns the value of the spatial coordinate and velocity
        after taking a timestep of length Dt using the Euler B method.
    """
    dphidq, err = deriv.derivative(phi, qn, np.abs(Dt*vn))
    vn1 = vn + (1/M)*Dt*dphidq
    qn1 = qn - Dt*vn1
    return qn1, vn1


def stoemerstep(phi, qn, pn, Dt, M = 1):
    """
    This function takes one timestep across a set of equations dq/dt = v, 
    M*dv/dt = -dphi(q)/dq using the Stoemer-Verlet method, where q is a  
    generalised spatial coordinate, v the velocity, t is time and phi is the 
    potential.
    
    Parameters    
        phi: A function defining the potential of the system.
        qn: The initial spatial coordinate at the start of the timestep.
        vn: The initial velocity at the start of the timestep.
        Dt: The length of the time step.
        M: The mass of the particle acting under a force.  Defaults to unity.
    
    Returns
        The function returns the value of the spatial coordinate and velocity 
        after taking a timestep of length Dt using the Stoemer-Verlet method.
    """
    # use correct gradient
    grad = None
    h = None
    if len(qn) == 3 and len(pn) == 3:
        grad = deriv.grad
        h = np.ones(3)
    elif len(qn) == 2 and len(pn) == 2:
        grad = deriv.grad2D
        h = np.ones(2)
    else:
        raise ValueError('stoemerstep: qn and pn are the wrong length.')
    
    # constants
    useRidders = False
    h = Dt/2*np.max(pn)*h
    
    # Stoermer-Verlet step
    pnHalf = pn - 0.5*Dt*grad(phi, qn, h, useRidders)
    qn1 = qn + Dt*pnHalf/M
    pn1 = pnHalf - 0.5*Dt*grad(phi, qn1, h, useRidders)
    
    return qn1, pn1

def rk4step(phi, qn, vn, Dt, M):
    """
    This function takes one timestep across a set of equations dq/dt = v, 
    M*dv/dt = -dphi(q)/dq using a fourth order Runge-Kutta method, where q is a 
    generalised spatial coordinate, v the velocity, t is time and phi is the 
    potential.
    
    Parameters
        phi: A function defining the potential of the system.
        qn: The initial spatial coordinate at the start of the timestep.
        vn: The initial velocity at the start of the timestep.
        Dt: The length of the time step.
        M: The mass of the particle acting under a force.
    
    Returns
        The function returns the value of the spatial coordinate and velocity 
        after taking a timestep of length Dt using a fourth order Runge-Kutta 
        method.
    """    
    # Z1
    Q1 = qn
    V1 = vn        
    dphidQ1, err = deriv.derivative(phi, Q1, np.abs(Dt*vn))
    # Z2
    Q2 = qn + 0.5*Dt*V1
    V2 = vn - 0.5*Dt*1.0/M*dphidQ1
    dphidQ2, err = deriv.derivative(phi, Q2, np.abs(Dt*vn))
    # Z3
    Q3 = qn + 0.5*Dt*V2
    V3 = vn - 0.5*Dt*1.0/M*dphidQ2
    dphidQ3, err = deriv.derivative(phi, Q3, np.abs(Dt*vn))
    # Z4
    Q4 = qn + Dt*V3
    V4 = vn - Dt*1.0/M*dphidQ3
    dphidQ4, err = deriv.derivative(phi, Q4, np.abs(Dt*vn))
    # final step
    qn1 = qn + Dt/6.0*(V1 + 2.0*V2 + 2.0*V3 + V4)
    vn1 = vn - Dt/(6.0*M)*(dphidQ1 + 2.0*dphidQ2 + 2.0*dphidQ3 + dphidQ4)
    
    return qn1, vn1

def implicitmidpoint(phi, qn, pn, Dt):
    """
    This function takes one timestep across a set of equations dq/dt = v, 
    M*dv/dt = -dphi(q)/dq using the implicit midpoint method.  As the method 
    is implicit we need solve a system of equations to make the step.  This is 
    done with fixed-point or functional iteration.
    
    Parameters
        phi: A function defining the potential of the system.
        qn: The initial Cartesian spatial coordinate at the start of the 
            timestep.
        pn: The initial Cartesian momentum at the start of the timestep.
        Dt: The length of the time step.
    
    Returns
        The function returns the value of the spatial coordinate and velocity 
        after taking a timestep of length Dt using the Euler method.
    """
    # use correct gradient and error function
    grad = None
    error = None
    h = None
    if len(qn) == 3 and len(pn) == 3:
        grad = deriv.grad
        error = lambda zk, zk1: np.sqrt((zk[0, 0] - zk1[0, 0])**2 + \
                                        (zk[0, 1] - zk1[0, 1])**2 + \
                                        (zk[0, 2] - zk1[0, 2])**2) + \
                                np.sqrt((zk[1, 0] - zk1[1, 0])**2 + \
                                        (zk[1, 1] - zk1[1, 1])**2 + \
                                        (zk[1, 2] - zk1[1, 2])**2)
        h = np.ones(3)
    elif len(qn) == 2 and len(pn) == 2:
        grad = deriv.grad2D
        error = lambda zk, zk1: np.sqrt((zk[0, 0] - zk1[0, 0])**2 + \
                                        (zk[0, 1] - zk1[0, 1])**2) + \
                                np.sqrt((zk[1, 0] - zk1[1, 0])**2 + \
                                        (zk[1, 1] - zk1[1, 1])**2)
        h = np.ones(2)
    else:
        raise ValueError('implicitmidpoint: qn and pn are the wrong length.')
    epsilon = 1.0e-9
    
    # calculate the half step
    useRidders = False
    qk = qn
    pk = pn
    h = Dt/2*np.max(np.abs(pk))*h
    e = 1.0
    while e > epsilon:
        #dphidqk, err = grad(phi, qk, h, useRidders)
        dphidqk = grad(phi, qk, h, useRidders)
        qk1 = qn + Dt/2*pk
        qk1 = qn + Dt/2*pk
        pk1 = pn - Dt/2*dphidqk
        e = error(np.array([qk, pk]), np.array([qk1, pk1]))
        qk = qk1
        pk = pk1
    qnHalf = qk
    pnHalf = pk
    #dphidqHalf, err = grad(phi, qnHalf, h, useRidders)
    dphidqHalf = grad(phi, qnHalf, h, useRidders)
    
    # calculate the step
    qn1 = qn + Dt*pnHalf
    pn1 = pn + Dt*dphidqHalf
            
    return qn1, pn1


class Stoemer_Verlet:
    """
    A functor implementation of the Stoemer-Verlet method.  Note this
    implmentation assumes a kinetic energy of the for T = p^T.M^-1.p.
    """
    def __init__(self, dVdq, Dt, M=np.diag(np.ones(3))):
        """
        The constructor for the Stoemer-Verlet functor.

        Parameters:
            dVdq - The first differential with respect to the spatial
                   coordinate.
            Dt - The stepsize this functor will take.
            M - A three dimensional mass matrix.  Default to a unit matrix
        """
        self.dVdq = dVdq
        self.Dt = Dt
        self.invM = np.linalg.inv(M)

    def __call__(self, q, p):
        """
        Returns the value of the spatial and momentum coordinates are a step of
        Dt.
        """
        pHalf = p - 0.5*self.Dt*self.dVdq(q)
        q = q + self.Dt*np.dot(pHalf, self.invM)
        p = pHalf - 0.5*self.Dt*self.dVdq(q)
        return q, p
