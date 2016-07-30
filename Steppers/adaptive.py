# -*- coding: utf-8 -*-
"""
Created on Tue May 17 17:05:06 2016
A module containing various adaptive steppers.
@author: rpoolman
"""
import numpy as np


class SundmanAdpaptiveStep:
    """
    This is a class that takes an integration method as an input then modifies
    the step size after a step of Dt/2, then applies it adjoint to complete the
    step.  The modification to the step is calculated by a Sundman transform.
    """
    def __init__(self, psi, psi_adj, g, gn):
        """
        The constructor that takes the fixed step integration scheme and its
        adjoint.  Note the step size is set by the stepping method functors and
        the average of the two is taken in the constructor.

        Parameters:
            psi - The fixed size stepping method that is used to take the first
                  half step.  Must be a functor with a call that takes only
                  spatial coordinate and momentum coordinate in that order.
            psi_adj - The adoint of psi that takes the second half step.  Must
                      be a functor with a call that takes only spatial
                      coordinate and momentum coordinate in that order.
            g -  A function defining the Sundman transform. N.B. MUST ONLY
                 DEPEND ON THE SPATIAL COORDINATE.
            gn - The inital scaling value for the half step.
        """
        self.Dt = (psi.Dt + psi_adj.Dt)*0.5
        self.psi = psi
        self.psi_adj = psi_adj
        self.g = g
        self.gn = gn
        self.stepEfficiency = 0.0

    def __call__(self, q, p):
        """
        Takes the full step by first applying the the integrator psi for a half
        step jump, correcting the step size using the supplied Sundman
        transform and final uses psi_adj to take the remaining half step with
        the new stepsize.

        Parameters:
            q - The initial spatial coordinate.
            p - The initial momentum coordinate.

        Returns:
            The spatial and momentum coordinate after the step
        """
        self.stepEfficiency = 0.0

        # first half step
        gDt = self.psi.Dt*self.gn
        self.psi.Dt = gDt
        q, p = self.psi(q, p)
        self.stepEfficiency = self.psi.numberForceEvals/self.Dt

        # time step transform
        self.gn = 2*self.g(q) - self.gn

        # second half step
        gDt = self.psi.Dt*self.gn
        self.psi_adj.Dt = gDt
        q, p = self.psi_adj(q, p)
        self.stepEfficiency = \
            0.5*(self.stepEfficiency +
                 self.psi_adj.numberForceEvals/self.Dt)

        return q, p


def stoemerstep(dVdq, g, gn, qn, pn, Dt, invM=np.diag(np.ones(3))):
    """
    This function takes one timestep across a set of equations
    dq/dtau = M^-1 p, dp/dtau = -dphi(q)/dq and dt/dtau = g(q) using the
    Stoemer-Verlet method, where q is a generalised spatial coordinate, p the
    momentum, tau is fictive time, t is the real time and phi is the potential.
    The last equation is a Sundman transform and defines the way step size
    varies.  Note that the LHS of the Sundman transformation is dependent only
    on q and so the stepper is explicit.  Assumes three dimensional coordinate
    system.

    Parameters
        dVdq: A function defining the first derviative of potential of the
              system w.r.t. q.
        g: A function defining the Sundman transform. N.B. MUST ONLY DEPEND ON
           THE SPATIAL COORDINATE.
        gn: The intial scaling value for the first half of the step.
        qn: The initial spatial coordinate at the start of the timestep.
        pn: The initial momentum at the start of the timestep.
        Dt: The length of the time step.
        invM : The inverse of the mass matrix, defaults to  3x3 unit matrix.
        gn: The initial value of the temporal scaling factor.

    Returns
        The function returns the value of the spatial coordinate, momentum and
        scaling parameter gn1 after taking a fictive timestep of approximate
        length Dt using the explicit adative Stoemer-Verlet method.
    """
    # Stoermer-Verlet step
    pnHalf = pn - 0.5*Dt*gn*dVdq(qn)
    qnHalf = qn + 0.5*Dt*gn*np.dot(pnHalf, invM)
    gn1 = 2*g(qnHalf) - gn
    qn1 = qnHalf + 0.5*Dt*gn1*np.dot(pnHalf, invM)
    pn1 = pnHalf - 0.5*Dt*gn1*dVdq(qn1)

    return qn1, pn1, gn1


def stoemerstep_test(dVdq, qn, pn, Dt, invM=np.diag(np.ones(3))):
    """
    This function takes one timestep across a set of equations dq/dt = M^-1 p,
    dv/dt = -dphi(q)/dq using the Stoemer-Verlet method, where q is a
    generalised spatial coordinate, p- the momentum, t is time and phi is the
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
    # Stoermer-Verlet step
    pnHalf = pn - 0.5*Dt*dVdq(qn)
    qn1 = qn + Dt*np.dot(pnHalf, invM)
    pn1 = pnHalf - 0.5*Dt*dVdq(qn)

    return qn1, pn1
