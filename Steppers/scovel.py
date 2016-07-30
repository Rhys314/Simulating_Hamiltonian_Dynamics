# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 16:57:05 2016

A file that contains steppers based on Scovel's method at various orders.  
Details of the mathematical background an some example usages can be found in 
Chapters 4 and 6 of Simulating Hamiltonian Dynamics.
@author: rpoolman
"""
import numpy as np
import NumericalAlgorithms.derivatives as deriv

class _ScovelsMethod:
    """
    The second order Scovel method that may be concatenated to create higher 
    order methods.  This is a base class and should not generally be used.  
    Instead use the Composition class and set it two the order you wish to use.
    The details of this method are described in L + R section 4.5.2, pp. 94.
    """    
    def __init__(self, func, M, b, Dt = None, useNumericalDerivatives = False):
        """
        Parameters
            func - Either the potential function or its first spatial 
                   derivative, depending on whether useNumericalDerivatices is
                   False or True.  Must take a one-dimensional three element 
                   vector represent the spatial coordinates at which at which 
                   the function will evaluate.
            M - The particle mass.
            b - A three element vector used to define the force acting on the 
                system.
            Dt - The timestep size, which defaults to none.  This allows  
                 adpative methods to applied.  If a value is provided then 
                 step size is fixed.
            useNumericalDerivatives: If True the first spatial derivative of 
                                     the potential function is calculated by a 
                                     finite difference method.  If False then 
                                     func must be an analytically derived 
                                     first derivative of the potential 
                                     function.  Defaults to False.
        """
        self.m_func = func
        self.m_M = M
        
        if np.shape(b) != (3,):
            raise ValueError("_ScovelsMethod.__init__: Vector b must be a three-vector.")
        else:
            self.m_B = np.zeros((3,3))
            self.m_B[1, 0] = b[2]
            self.m_B[2, 0] = -b[1]
            self.m_B[0, 1] = -b[2]
            self.m_B[2, 1] = b[0]
            self.m_B[0, 2] = b[1]
            self.m_B[1, 2] = -b[0]
        
        self.b_norm = np.sqrt(b[0]**2 + b[1]**2 + b[2]**2)
        if Dt is None:
            self.m_F = self._F
            self.m_expBt = self._expBt
        else:
            self.m_Dt = Dt
            self.m_F = 1.0/self.m_M*(Dt*np.diag(np.ones(3)) + \
                       (1.0 - np.cos(self.b_norm*Dt))/self.b_norm**2.0*self.m_B - \
                       (np.sin(self.b_norm*Dt) - self.b_norm*Dt)/self.b_norm**3.0*np.dot(self.m_B.T, self.m_B))
            self.m_expBt = np.diag(np.ones(3)) + \
                           np.sin(self.b_norm*self.m_Dt)/self.b_norm*self.m_B - \
                           2*(np.sin(self.b_norm*self.m_Dt/2)/self.b_norm)**2*np.dot(self.m_B.T, self.m_B)
        
        if Dt is None and useNumericalDerivatives:
            self.m_step = self._numericScovelAdaptive
        elif Dt is None and not useNumericalDerivatives:
            self.m_step = self._analyticScovelAdaptive
        elif Dt is not None and useNumericalDerivatives:
            self.m_step = self._numericScovel
        elif Dt is not None and not useNumericalDerivatives:
            self.m_step = self._analyticScovel
    
    def _expBt(self, t):
        """
        Rodrigues' formula, used to calculate the expontential of a 3x3 matrix.
        
        Parameters:
            t - The time at which the solution is to be calculated.
        
        Returns:
            A 3x3 matrix exp(Bt) where t is the time and B the matrix.
        """
        return np.diag(np.ones(3)) + \
               np.sin(self.b_norm*t)/self.b_norm*self.m_B - \
               2*(np.sin(self.b_norm*t/2)/self.b_norm)**2*np.dot(self.m_B.T, self.m_B)
    
    def _F(self, t):
        """
        Matrix equation that forms part of the linear system solution described 
        in L + R section 4.5.2, p. 95.
        
        Parameters:
            t - The time at which the solution is to be calculated.
        
        Returns:
            A 3x3 matrix that is used to calculate the coordinate at the next 
            time step.
        """
        return 1.0/self.m_M*(t*np.diag(np.ones(3)) + \
               (1.0 - np.cos(self.b_norm*t))/self.b_norm**2.0*self.m_B - \
               (np.sin(self.b_norm*t) - self.b_norm*t)/self.b_norm**3.0*np.dot(self.m_B.T, self.m_B))
    
    def _numericScovelAdaptive(self, Dt, qn, pn):
        """
        A Scovel method time step with sptial derivatives of the potenial 
        calculated with finite differences.
        
        Parameters:
            Dt - The time step size.
            qn - The coordinate in three dimensions at the start of the step.
            pn - The momentum in three dimensions at the start of the step.
        
        Returns:
            The coordinate and momentum after the step.            
        """
        pHalf = pn - Dt/2*deriv.grad(self.m_func, qn, Dt*pn)
        qn1 = qn + np.dot(self.m_F(Dt), pHalf)
        pn1 = np.dot(self.m_expBt(Dt), pHalf) - \
              Dt/2*deriv.grad(self.m_func, qn1, Dt*pn)
        return qn1, pn1
    
    def _analyticScovelAdaptive(self, Dt, qn, pn):
        """
        A Scovel method time step  with sptial derivatives of the potenial 
        calculated from a pre-derived analytical function.  This offers 
        superior performance
        
        Parameters:
            Dt - The time step size.
            qn - The coordinate in three dimensions at the start of the step.
            pn - The momentum in three dimensions at the start of the step.
        
        Returns:
            The coordinate and momentum after the step.            
        """
        pHalf = pn - Dt/2*self.m_func(qn)
        qn1 = qn + np.dot(self.m_F(Dt), pHalf)
        pn1 = np.dot(self.m_expBt(Dt), pHalf) - Dt/2*self.m_func(qn1)        
        return qn1, pn1
    
    def _numericScovel(self, qn, pn):
        """
        A Scovel method time step with sptial derivatives of the potenial 
        calculated with finite differences.
        
        Parameters:
            qn - The coordinate in three dimensions at the start of the step.
            pn - The momentum in three dimensions at the start of the step.
        
        Returns:
            The coordinate and momentum after the step.            
        """
        pHalf = pn - self.m_Dt/2*deriv.grad(self.m_func, qn, self.m_Dt*pn)
        qn1 = qn + np.dot(self.m_F, pHalf)
        pn1 = np.dot(self.m_expBt, pHalf) - \
              self.m_Dt/2*deriv.grad(self.m_func, qn1, self.m_Dt*pn)        
        return qn1, pn1
    
    def _analyticScovel(self, qn, pn):
        """
        A Scovel method time step  with sptial derivatives of the potenial 
        calculated from a pre-derived analytical function.  This offers 
        superior performance.
        
        Parameters:
            qn - The coordinate in three dimensions at the start of the step.
            pn - The momentum in three dimensions at the start of the step.
        
        Returns:
            The coordinate and momentum after the step.            
        """
        pHalf = pn - self.m_Dt/2*self.m_func(qn)
        qn1 = qn + np.dot(self.m_F, pHalf)
        pn1 = np.dot(self.m_expBt, pHalf) - self.m_Dt/2*self.m_func(qn1)        
        return qn1, pn1
    
    def workPerUnitTime(self):
        """
        The work per unit time is defined as W = N*S/T, where N is the number 
        of time steps, S is the number of stages and T is the period.  See 
        L + R, section 6.6, pp. 165 for details.  I have remove the T 
        dependency with N = T/Dt leading to S/Dt.  For the Scovel method there 
        is only one stage so W = 1/Dt
        
        Return:
            A value for the work per unit time as calculated by S/Dt.
        """
        return 1/self.m_Dt

class Composition(_ScovelsMethod):
    """
    A dervied class to create compositions of Scovels 2nd order method by 
    repeated application across a weighted time step.  The scheme is described 
    in L + R, section 6.2.2 on pp. 147.
    """
    def __init__(self, func, M, b, w, Dt = None, 
                 useNumericalDerivatives = False):
        """
        Parameters:
            func - Either the potential function or its first spatial 
                   derivative, depending on whether useNumericalDerivatices is
                   False or True.  Must take a one-dimensional three element 
                   vector represent the spatial coordinates at which at which 
                   the function will evaluate.
            M - The mass of the particle to be simulated.
            b - A three element vector used to define the force acting on the 
                system.
            w - The first half of array of weights to be applied to the 
                time-step size.  Note the final array must be symmeteric and 
                have an odd number of elements.  The second half is the mirror 
                of the first and the central element is the sum of hte first 
                half.
            Dt - The timestep size, which defaults to none.  This allows  
                 adpative methods to applied.  If a value is provided then 
                 step size is fixed.
            useNumericalDerivatives - If True the first spatial derivative of 
                                      the potential function is calculated by a 
                                      finite difference method.  If False then 
                                      func must be an analytically derived 
                                      first derivative of the potential 
                                      function.  Defaults to False.
        """
        # initialse the child class
        self.m_w = np.zeros(len(w)*2 + 1)
        self.m_w[0 : np.int(len(w))] = w
        self.m_w[np.int(len(w)) + 1:] = w[::-1]
        self.m_w[np.int(len(w))] = 1 - 2*np.sum(w)
        self.m_Dt = Dt
        self.m_stages = len(self.m_w)
        
        # initailise the base class
        super(Composition, self).__init__(func, M, b, None, 
                                          useNumericalDerivatives)
        
    def step(self, qn, pn):
        qn1 = qn
        pn1 = pn
        for w in self.m_w:
            qn1, pn1 = self.m_step(w*self.m_Dt, qn1, pn1)
        return qn1, pn1
    
    def workPerUnitTime(self):
        """
        The work per unit time is defined as W = N*S/T, where N is the number 
        of time steps, S is the number of stages and T is the period.  See 
        L + R, section 6.6, pp. 165 for details.  I have remove the T 
        dependency with N = T/Dt leading to S/Dt.
        
        Return:
            A value for the work per unit time as calculated by S/Dt.
        """
        return self.m_stages/self.m_Dt

class Processing(Composition):
    """
    A derived class to create a post-processed composition method as described 
    in L + R section 6.2.3, pp. 148.  The idea here is that a composition 
    method is applied to z after it has undergone a transformation.  When the 
    output is required the composition method is then applied to the results.  
    This allows a higher order method to be used at a smaller computational 
    price.  This class Sovel's method is the transform.
    """
    def __init__(self, func, M, b, w, c, Dt, 
                 useNumericalDerivatives = False):
        """
        Parameters:
            func - Either the potential function or its first spatial 
                   derivative, depending on whether useNumericalDerivatices is
                   False or True.  Must take a one-dimensional three element 
                   vector represent the spatial coordinates at which at which 
                   the function will evaluate.
            M - The mass of the particle to be simulated.
            b - A three element vector used to define the force acting on the 
                system.
            w - The first half of array of weights to be applied to the 
                time-step size.  Note the final array must be symmeteric and 
                have an odd number of elements.  The second half is the mirror 
                of the first and the central element is the sum of hte first 
                half.
            c - The coefficients applied to the coordinate transform.
            Dt - The timestep size, which defaults to none.  This allows  
                 adpative methods to applied.  If a value is provided then 
                 step size is fixed.
            useNumericalDerivatives - If True the first spatial derivative of 
                                      the potential function is calculated by a 
                                      finite difference method.  If False then 
                                      func must be an analytically derived 
                                      first derivative of the potential 
                                      function.  Defaults to False.
        """
        # initialise the base class
        super().__init__(func, M, b, w, None, useNumericalDerivatives)
        
        # need to store step size
        self.m_Dt = Dt
        
        # initailise the transform coefficients
        self.m_c = np.zeros(len(c) + 1)
        self.m_c[1:len(c) + 1] = c
        self.m_c[0] = -np.sum(c)
        
    def applyTransform(self, qn, pn):
        """
        Applies the transform to the coordinates q and momentum p, after which 
        they may be integrated.
        
        Parameters:
            qn - The coordinate to which the transform is applied.
            pn - The momentum o which the transform is applied.
        
        Returns:
            The transformed coordinates and momentum to which integration may 
            now occur.
        """        
        qn_hat = qn
        pn_hat = pn
        for c in self.m_c:
            qn_hat, pn_hat = self.m_step(c*self.m_Dt, qn_hat, pn_hat)
        for c in self.m_c:
            qn_hat, pn_hat = self.m_step(-c*self.m_Dt, qn_hat, pn_hat)
        return qn_hat, pn_hat
        
    def applyInverseTransform(self, qn_hat, pn_hat):
        """
        Applies the inverse transform to the coordinates q and momentum p, 
        after which they are suitable for further analysis.
        
        Parameters:
            qn_hat - The transformed coordinate to which the inverse transform 
                     is applied.
            pn_hat - The transformed momentum o which the inverse transform 
                     is applied.
        
        Returns:
            The coordinates and momentum in the original frame which can now 
            be further analysed.
        """        
        qn = qn_hat
        pn = pn_hat
        for c in self.m_c[::-1]:
            qn_hat, pn_hat = self.m_step(c*self.m_Dt, qn_hat, pn_hat)        
        for c in self.m_c[::-1]:
            qn_hat, pn_hat = self.m_step(-c*self.m_Dt, qn_hat, pn_hat)
        return qn, pn
    
    def integrate(self, qn0, pn0, T):
        """
        A function to calcualte the trajectory of a particle up to the time 
        t = T and return the position and mometum of that particle at the first
        value of t > T.
        
        Parameters:
            qn0 - The initial coordinates
            qn1 - The initial momentum.
            T - The time up to which the integration occurs.
            
        Returns:
            The coordinates and momentum of the particle at time T.
        """
        # transform qn0 and pn0 to qn0_hat and pn0_hat
        qn_hat, pn_hat = self.applyTransform(qn0, pn0)
        
        # integrate up to time t = T
        N = np.int(np.ceil(T/self.m_Dt))
        for ii in range(N):
            qn_hat, pn_hat = self.step(qn_hat, pn_hat)
        
        # inverse transform the results back to oringal frame of reference
        return self.applyInverseTransform(qn_hat, pn_hat)