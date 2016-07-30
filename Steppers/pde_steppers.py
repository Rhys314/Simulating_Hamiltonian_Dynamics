# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 07:27:19 2016
A module containing various steppers for take a time and space step to solve
partial differential equations.
@author: rpoolman
"""


class eulerBoxStep:
    """
    A functor that calculates a step for a PDE of the form
    u_tt = dsigma(du/dx)/dx - f_prime(u) using a EulerB-like scheme, combined
    with the box method.
    """
    def __init__(self, Dt, Dx, f, fPrime, sigma, sigmaPrime):
        """
        Parameters
        ----------
            Dt: double
                Temporal step size.
            Dx: double
                Spatial step size.
            f: function
                The u dependent term in the Hamiltonian.
            fPrime: function
                The first spatial derivative of f.
            sigma: function
                The du/dx dependent term in the Hamiltonian.
            sigmaPrime: function
                The first spatial derivative of sigma.
        """
        self.Dt = Dt
        self.Dx = Dx
        self.f = f
        self.fPrime = fPrime
        self.sigma = sigma
        self.sigmaPrime = sigmaPrime

    def __call__(self, u, v):
        """
        Takes one step using the Euler box method.

        Parameters
        ----------
            u: 3 double array
                Contains u^n_i-1, u^n_i and u^n_i+1.
            v: double
                A scalar containing v^n_i.

        Returns
        -------
            z: 2 double array
                The first element is the distribution and the second element
                momentum.
        """
        # calculate time step
        v_next = v + \
            self.Dt*((u[2] - 2*u[1] + u[0])/self.Dx**2 - self.fPrime(u[1]))
        u_next = u[1] + self.Dt*v_next

        return u_next, v_next

    def residual(self, u, v):
        """
        Calculates the residual.

        Parameter
        ---------
        u: 3x3 double array
            An array of spatial distribution elements centred on u_i^n, with
            the space coordinate along the rows and the time step down the
            columns.

        v: 3x3 double array
            An array of velocity distribution elements centred on v_i^n, with
            the space coordinate along the rows and the time step down the
            columns.

        Return
        ------
        residual: double
            The sum of the energy and flux differnces over a step.
        """
        return (self.E(u[2, :], v[2, :]) -
                self.E(u[1, :], v[1, :]))/self.Dt + \
               (self.F(u[1:, 1:], v[1:, 1]) -
                self.F(u[1:, 0:2], v[1:, 0]))/self.Dx

    def E(self, u, v):
        """
        Calculates the total energy of the system.

        Parameters
        ----------
        u: 3 double array
            Containing the previous, current and next value of the sptial
            distribution.
        v: 3 double array
            Containing the previous, current and next value of the velocity
            distribution.

        Returns
        -------
        energy: double
            The total energy of the system.
        """
        return 0.5*v[1] + self.sigma((u[1] - u[0])/self.Dx) + self.f(u[1])

    def F(self, u, v):
        """
        Calculates the energy flux in the system.

        Parameters
        ----------
        u: 2x2 double array
            Part of the spatial distribution, with the space coordinate along
            the rows and the time step down the columns.  The first column is
            at space step i and the first row is at time step n.
        v: 1x2 double array
           Part of the velocity distribution, from the ith space step.  The
           first element is from the nth time.

        Returns
        -------
        flux: double
            The flux.
        """
        return -0.5*(v[0]*self.sigmaPrime((u[0, 1] - u[0, 0])/self.Dx) +
                     v[1]*self.sigmaPrime((u[1, 1] - u[1, 0])/self.Dx))
