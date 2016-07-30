# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 17:09:26 2016

@author: rpoolman
"""
####################################################################
# nl_wave_eqn.py
#
# Template for the PHY 209 non-linear wave equation problem.
#
# F.Pretorius
####################################################################

import numpy as np

####################################################################
# cn_wave class:
####################################################################
class cn_wave:
    '''This class describes and provides methods to evolve a 2D non-linear 
    wave-equation:

    - d^2 phi(x,t)/dt^2 + d^2 phi(x,t)/dx^2 = F(phi)

    on a periodic boundary, discretized using a two time level,
    iterative Crank-Nicholson-type scheme. The equations are converted
    to first-order-in-time via

      d phi(x,t)/dt - pi(xi,t) = 0 
      d pi(xi,t)/dt - d^2 phi(x,t)/dx^2 + F(phi) = 0

    Public Methods
    ==============

    __init__(self,F,dF,args,xmin,xmax,Nx)

       F(phi,args) = is a user-specified function returning F(phi) 
       dF(phi,args) = is a user-specified function returning dF/dphi(phi)
       xmin,xmax define the spatial integration domain
       Nx = is the number of points, so dx=(xmax-xmin)/(N-1)

       All initial data set to zero, t set to 0.

    set_phi_n(self,phi_init,args)
    set_pi_n(self,pi_init,args)

       phi_init(x,args), pi_init(x,args) are user specified functions 
       that set the initial data phi(x,t=0) and pi=d(phi)/dt(x,t=0)
       args is a list of parameters to pass to the functions.

    set_CFL(self,CFL)

       sets the CFL factor, so dt=CFL*dx

    set_iter(self,tol,max_iter)

       sets the parameters the interative solution method ... will iterative
       until a norm of the residual is below tol, up to a maximum of max_iter 
       iterations

    set_KO_filter(eps)

       sets  Kreiss-Oliger dissipation parameter eps (<1) 

    step(self)

       take 1 time step of size dt. The equations are solved iteratively,
       and the iterativion proceeds until the residual is below tol,
       unless max_iter is exceeded. After the time step, time levels
       np1 and n are swapped, so level n becomes the current (latest)
       time, and np1 holds the past time level data.
       Returns [iter,tol], the number of iterations required, and pre-last step
       residual

    Qeury methods ... returns related data:

    phi_n()
    pi_n()
    x()
    t()
    dt()

    Only the above 5 have been implemented ... write
    those below as needed.

    phi_np1()
    pi_np1()
    pi_res()
    phi_res()
    F_n()
    F_np1()
    dF_n()
    dF_np1()
    [xmin,xmax,dx]=grid()
    
    '''
    ####################################################################
    # public methods
    ####################################################################

    def __init__(self,F,dF,args,xmin,xmax,Nx):
       self._F=F
       self._dF=dF
       self._args=args
       self._xmin=xmin
       self._xmax=xmax
       if (Nx<=1):
          print("Error ... Nx must be greater than 1")
          raise("invalid arguments")
       self._N=Nx
       self._dx=(xmax-xmin)/(Nx-1)
       self._phi_n=np.zeros(Nx)
       self._phi_np1=np.zeros(Nx)
       self._pi_n=np.zeros(Nx)
       self._pi_np1=np.zeros(Nx)
       self._phi_res=np.zeros(Nx)
       self._pi_res=np.zeros(Nx)
       self._F_n=np.zeros(Nx)
       self._F_np1=np.zeros(Nx)
       self._dF_n=np.zeros(Nx)
       self._dF_np1=np.zeros(Nx)

       self._t=0
       self._tol=1.0e-4
       self._max_iter=50
       self._dt=0.5*self._dx
       self._eps=0

    def set_phi_n(self,phi_init,args):
       self._phi_n=phi_init(self.x(),args)
       # enforce periodicity
       self._phi_n[self._N-1]=self._phi_n[0]

    def set_pi_n(self,pi_init,args):
       self._pi_n=pi_init(self.x(),args)
       # enforce periodicity
       self._pi_n[self._N-1]=self._pi_n[0]

    def set_CFL(self,CFL):
       self._dt=CFL*self._dx

    def set_iter(self,tol,max_iter):
       self._tol=tol
       self._max_iter=max_iter

    def set_KO_filter(self,eps):
       if (eps<0 or eps>1):
          raise 'set_KO_filter: error ... eps out of range'
       self._eps=eps

    def step(self):
       N=self._N
       dx=self._dx
       dt=self._dt
       CFL=dt/dx
       #filter if desired
       if (self._eps>=0):
          self._phi_n=KO_filter(self._phi_n,self._eps)
          self._pi_n=KO_filter(self._pi_n,self._eps)
       #initial guess ... copy N to N+1
       self._phi_np1=self._phi_n[:]
       self._pi_np1=self._pi_n[:]
       iter=0
       tol=self._tol+1.0
       while (iter<self._max_iter and tol>self._tol):
          self._comp_all_F_dF()
          self._comp_residual()
          tol=np.max(np.abs(self._phi_res))+np.max(np.abs(self._pi_res))
          # elements of the inverse-Jacobian, iJ_11=iJ_22, iJ_12 & iJ_21
          dFh=self._dF_np1
          T0=(2.0+dFh*dx**2)
          idet=2.0/(4.0+CFL**2*T0)
          iJ_11=2*dt*idet
          iJ_22=iJ_11
          iJ_12=dt**2*idet
          iJ_21=-T0*CFL**2*idet
          # 1 Newton iteration : compute next guess
          self._phi_np1=self._phi_np1 - (iJ_11*self._phi_res + iJ_12*self._pi_res)
          self._pi_np1=self._pi_np1   - (iJ_21*self._phi_res + iJ_22*self._pi_res)
          iter=iter+1
       self._t+=dt
       if (tol>self._tol):
          print("WARNING: failed to solve equations to within", self._tol, " in ",self._max_iter," iterations")
          print("         t=",self._t," tol=", tol)
       # swap time levels for phi & pi
       tmp=self._phi_np1; self._phi_np1=self._phi_n; self._phi_n=tmp
       tmp=self._pi_np1; self._pi_np1=self._pi_n; self._pi_n=tmp
       return [iter,tol]

    def phi_n(self):
       return self._phi_n[:]

    def pi_n(self):
       return self._pi_n[:]

    def t(self):
       return self._t

    def dt(self):
       return self._dt

    def x(self):
       return np.arange(self._xmin,self._xmax+self._dx,self._dx)

    ####################################################################
    # private methods
    ####################################################################

    def _comp_all_F_dF(self):
       for i in range(self._N):
          self._F_n[i]=self._F(self._phi_n[i],self._args)
          self._F_np1[i]=self._F(self._phi_np1[i],self._args)
          self._dF_n[i]=self._dF(self._phi_n[i],self._args)
          self._dF_np1[i]=self._dF(self._phi_np1[i],self._args)

    def _comp_residual(self):
      # Note: assumes F_n, F_np1, dF_n, dF_np1 valid
      dx=self._dx
      dt=self._dt
      phi_np1=self._phi_np1; phi_n=self._phi_n
      pi_np1=self._pi_np1; pi_n=self._pi_n
      F_np1=self._F_np1; F_n=self._F_n

      phi_xx_n=d2f_dx2_per(phi_n,dx)
      phi_xx_np1=d2f_dx2_per(phi_np1,dx)

      # d phi(x,t)/dt - pi(xi,t) = 0 
      self._phi_res = (phi_np1-phi_n)/dt - 0.5*(pi_np1+pi_n)

      # d pi(xi,t)/dt - d^2 phi(x,t)/dx^2 + F(phi) = 0
      self._pi_res = (pi_np1-pi_n)/dt - 0.5*(phi_xx_np1+phi_xx_n) + 0.5*(F_np1+F_n)

####################################################################
# a few utility functions
####################################################################
def d2f_dx2_per(f,dx):
   '''returns d^2f/dx^2, assuming f is periodic : f[0]=f[N-1]'''
   N=np.size(f)
   d2f=np.zeros(N)
   d2f[0]=(f[1]-2*f[0]+f[N-2])/dx**2
   for i in range(1,N-1):
      d2f[i]=(f[i+1]-2*f[i]+f[i-1])/dx**2
   d2f[N-1]=d2f[0]
   return d2f

def KO_filter(f,eps):
   ''' returns a Kreiss-Oliger filtered version of f, assumed to be periodic : : f[0]=f[N-1]'''
   N=np.size(f)
   KO_f=f[:]
   for i in range(0,N):
      im1=i-1
      im2=i-2
      ip1=i+1
      ip2=i+2
      if (im1<0): im1+=(N-1)
      if (im2<0): im2+=(N-1)
      if (ip1>(N-1)): ip1-=(N-1)
      if (ip2>(N-1)): ip2-=(N-1)
      KO_f[i]-=eps/16.0*(f[im2]-4*f[im1]+16*f[i]-4*f[ip1]+f[ip2])
   return KO_f

def gaussian(x,args):
   '''returns A*exp(-(x-x0)^2/sigma^2), were args=[A,x0,sigma]'''
   A=args[0]
   x0=args[1]
   sigma=args[2]
   return A*np.exp(-(x-x0)**2/sigma**2)

def d_gaussian(x,args):
   '''returns d/dx[A*exp(-(x-x0)^2/sigma^2)], were args=[A,x0,sigma]'''
   A=args[0]
   x0=args[1]
   sigma=args[2]
   return (-2*(x-x0)/sigma**2)*A*np.exp(-(x-x0)**2/sigma**2)