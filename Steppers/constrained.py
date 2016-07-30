# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 17:12:42 2016

A file containing implementations of SHAKE and RATTLE as described in chapter
7 of L + R.
@author: rpoolman
"""
import numpy as np


class SHAKE:
    """
    A class that generates the SHAKE method of numerical integration as
    described in L + R chapter 7 for three dimensions.  IMPRTOANT The velocity
    for this integrator is always one half step out of phase with the position.
    Also note this is a very simple implementation suitable only for one
    constraint.
    """
    def __init__(self, q0, v0, dVdq, g, G, N, T, M=np.diag(np.ones(3))):
        """
        An implementation of the SHAKE method using position-velocity form
        described in section 7.2, pp. 173 of L + R.  Note this is not a
        sympletic method.

        Parameters:
            q0 - The inital positon before the step is taken in three
                 dimensions.
            v0 - The initial velocity before the step is taken in three
                 dimensions.  The SHAKE algorithim startes with the momentum a
                 half time step behind the position.
            dVdq - The first derivative of the potential w.r.t the coordiantes
                   in which the particle resides.
            g - A function that represent the system constraint.
            G - The Jacobian of the constraint function w.r.t the coordinates,
                see L + R chanpter 7.  Must take q as an argument.
            N - The number of steps to be taken.
            T - The total time the integration will occur for.
            M - The point mass, defaults to unit matrix.
        """
        # initalise class
        self.M = M
        self.invM = np.linalg.inv(self.M)
        self.dVdq = dVdq
        self.Dt = T/N
        self.g = g
        self.G = G
        self.N = N
        self.t = np.zeros(self.N, dtype=np.float64)
        self.qn = np.empty((self.N, 3), dtype=np.float64)
        self.qn[0, :] = q0
        self.vn = np.empty((self.N, 3), dtype=np.float64)
        self.vn[0, :] = v0

    def Integrate(self, tol):
        """
        Carries out the integration and stores the calculated trajectory.

        tol - The tolerance to which the SHAKE iteration is calculated.
        """
        for ii in range(0, np.int64(self.N) - 1):
            self._SHAKE(ii, tol)

    def _SHAKE(self, ii, tol):
        """
        Calculates one step using the SHAKE algorithm and is called in the
        Integrate function.

        Parameters:
            ii - The index for the step.
            tol - The tolerance to which the SHAKE iteration is calculated.
        """
        self.t[ii + 1] = ii*self.Dt

        # first unconstrained step
        self.vn[ii + 1, :] = self.vn[ii, :] \
            - self.Dt*np.dot(self.invM, self.dVdq(self.qn[ii, :]))
        q_int = self.qn[ii, :] + self.Dt*self.vn[ii, :]

        # calculate G(q)
        Gqprev = self.G(self.qn[ii, :])

        # loop until tolerance is met
        while np.abs(self.g(q_int)) > tol:
            # calculate delta lambda
            deltaLambda = self.g(q_int) \
                            / np.dot(self.G(q_int),
                                     np.dot(self.invM, Gqprev.T))

            # calculate next coordinate
            q_int = q_int - np.dot(np.dot(self.invM, Gqprev.T),
                                   deltaLambda)

        # set the coordinate at the next time step
        self.qn[ii + 1, :] = q_int


def sympleticstep(dVdq, g, G, dHdp, qn, pn, Dt, tol=1e-5,
                  invM=np.diag(np.ones(3))):
    """
    This function takes one timestep across the set of equations
    dq/dt = grad(H(q, p)), dp/dt = -grad(H(q, p)) - G(q)^T.l and 0 = g(q) using
    a symplectic constrained integrator.  The spaitial coordinate is q, the
    momentum is given by p, the Hamiltonian is given by H(q, p) and the
    constraint by g(q), where the first derivative is w.r.t q is G(q) and l are
    the multipliers The unconstrained step is taken by the Stoermer-Verlet
    method before the result is projected on to both configuration and Tangent
    manifolds.

    Parameters
        dVdq: A function defining the first derviative of potential of the
              system w.r.t. q.
        g: A function defining the constraint.
        G: A function defining the first derivate of the constraint function
           with respect to q.
        dHdp: A function defining the first derivative of the Hamiltonian
              w.r.t. momentum.
        qn: The initial spatial coordinate at the start of the timestep.
        pn: The initial momentum at the start of the timestep.
        Dt: The length of the time step.
        invM: The inverse of the mass matrix, defaults to  3x3 unit matrix.

    Returns
        The function returns the value of the spatial and momentum coordinates
        after taking a timestep of length Dt using the constrained
        Stoemer-Verlet method.
    """
    pn_bar = pn

    # calculate G(q)
    Gqprev = G(qn)

    # Stoemer-Verlet unconstrained step
    pnHalf_bar = pn_bar - 0.5*Dt*dVdq(qn)
    qn = qn + Dt*np.dot(pnHalf_bar, invM)
    pn_bar = pnHalf_bar - 0.5*Dt*dVdq(qn)

    # loop until tolerance is met for spatial coordinate
    while np.abs(g(qn)) > tol:
        # calculate lagrangian multiplier offset
        deltaLambda = g(qn)/np.dot(G(qn), np.dot(invM, Gqprev.T))

        # project unconstrained step on to configuration manifold
        qn = qn - np.dot(np.dot(invM, Gqprev.T), deltaLambda)
        pn_bar = pn_bar - 0.5*np.dot(np.dot(invM, Gqprev.T), deltaLambda)

    # project velocity on to tangent manifold
    G_qn = G(qn)
    while (np.abs(np.dot(G_qn, dHdp(qn, pn_bar))) > tol).all():
        # calculate lagrangian multiplier offset
        deltaLambda = np.dot(G_qn, pn_bar)

        # project unconstrained momentum on to tangent manifold
        pn_bar = pn_bar - 0.5*Dt*np.dot(G_qn, deltaLambda)

    return qn, pn_bar


class ConstrainedSympleticStep:
    """
    A functor that will take a constrained step based on the supplied sympletic
    method and constraint. Note this assumes that the Hamiltonian is seperable
    and of the form H = p^.TM.p/2 + V(q).
    """
    def __init__(self, psi, g, G, Dt, tol=1e-5, invM=np.diag(np.ones(3))):
        """
        The constructor for the stepper that takes all the functions and
        constant.  Note that the hidden constraint on the tangent manifold
        requires dH/dp and d^2H/dp^2.  As we assume that H = p^T.M.p/2 + V(q)
        we also assume that dH/dp = M^-1.p and d^2/dp^2 = M^-1

        Parameters
            psi: A functor that return spatial and momentum coordinates, the
                 call method of which is __call__(q, p).
            g: A function defining the constraint.
            G: A function defining the first derivate of the constraint
               function with respect to q.
            Dt: The length of the time step.
            invM: The inverse of the mass matrix, defaults to  3x3 unit matrix.
        """
        self.psi = psi
        self.g = g
        self.G = G
        self.Dt = Dt
        self.tol = tol
        self.invM = invM
        self.numberForceEvals = 0

    def __call__(self, q, p):
        """
        Takes one timestep across the set of equations dq/dt = grad(H(q, p)),
        dp/dt = -grad(H(q, p)) - G(q)^T.l and 0 = g(q) using a symplectic
        constrained integrator.  The spaitial coordinate is q, the momentum is
        given by p, the Hamiltonian is given by H(q, p) and the constraint by
        g(q), where the first derivative is w.r.t q is G(q) and l are the
        multipliers The unconstrained step is taken by the Stoermer-Verlet
        method before the result is projected on to both configuration and
        tangent manifolds.

        Parameters
            q: The initial spatial coordinate at the start of the timestep.
            p: The initial momentum at the start of the timestep.

        Returns
            The function returns the value of the spatial and momentum
            coordinates after taking a timestep of length Dt using the
            constrained Stoemer-Verlet method.
        """
        pn_bar = p

        # calculate G(q)
        Gqprev = self.G(q)

        # unconstrained step
        q, pn_bar = self.psi(q, pn_bar)

        # loop until tolerance is met for spatial coordinate
        self.numberForceEvals = 0
        N = 0
        while np.abs(self.g(q)) > self.tol and N < 1000:
            # calculate lagrangian multiplier offset
            deltaLambda = self.g(q)/np.dot(self.G(q),
                                           np.dot(self.invM, Gqprev.T))

            # project unconstrained step on to configuration manifold
            q = q - np.dot(np.dot(self.invM, Gqprev.T), deltaLambda)
            pn_bar = pn_bar - 0.5*np.dot(np.dot(self.invM, Gqprev.T),
                                         deltaLambda)
            N = N + 1
        self.numberForceEvals = N

        # project velocity on to tangent manifold
        G_qn = self.G(q)
        N = 0
        while (np.abs(np.dot(G_qn, np.dot(self.invM, pn_bar))) >
               self.tol).all() and N < 1000:
            # calculate lagrangian multiplier offset
            deltaLambda = np.dot(G_qn, pn_bar)

            # project unconstrained momentum on to tangent manifold
            pn_bar = pn_bar - 0.5*self.Dt*np.dot(G_qn, deltaLambda)
            N = N + 1
        self.numberForceEvals = self.numberForceEvals + N

        return q, pn_bar
