# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 07:20:57 2016
Test script.
@author: rpoolman
"""
####################################################################
# wave_example.py
#
# Template for the PHY 209 soliton lab showing how the cn_wave class 
# in nl_wave_eqn.py could be used.
#
# F.Pretorius
####################################################################

import Steppers.nl_wave_eqn as wv
import numpy as np
#import gobject
import matplotlib
matplotlib.use('GTKAgg')
import matplotlib.pyplot as plt

####################################################################
# right hand side for a massive wave equation:
# F(phi)=m^2 phi
# dF(phi)= d(F)/dphi = m^2
####################################################################
def F_massive(phi,m):
   return m**2*phi

def dF_massive(phi,m):
   return m**2

# grid 
xmin=-1.0
xmax=1.0
Nx=65
# massless wave
m0=0.0
# massive wave
m1=10.0

# gaussian ID for phi (note, a gaussian is not
# periodic, but if we make it narrow it shouldn't matter)
A_phi=1.0
x0=0.0
sigma=0.25

m0_wave=wv.cn_wave(F_massive,dF_massive,m0,xmin,xmax,Nx)
m1_wave=wv.cn_wave(F_massive,dF_massive,m1,xmin,xmax,Nx)

m0_wave.set_phi_n(wv.gaussian,[A_phi,x0,sigma])
m1_wave.set_phi_n(wv.gaussian,[A_phi,x0,sigma])

# Make the pulse right moving (exact for the massless case only)
# comment out for time-symmetric ID (i.e. keep pi 0)
A_pi=-A_phi
m0_wave.set_pi_n(wv.d_gaussian,[A_pi,x0,sigma])
m1_wave.set_pi_n(wv.d_gaussian,[A_pi,x0,sigma])

# Evolve them 1 "light crossing" time, [xmax-mxin]/c, c=1.
tl=xmax-xmin
t=m0_wave.t()

#use the gobject back end to animate
fig = plt.figure()
ax = fig.add_subplot(111)

phi_m0, = ax.plot(m0_wave.x(),m0_wave.phi_n(),label="massless")
phi_m1, = ax.plot(m0_wave.x(),m0_wave.phi_n(),label="massive")
ax.set_ylim(-A_phi,A_phi)
ax.legend()

def animate():
   [iter0,tol0]=m0_wave.step()
   [iter1,tol1]=m1_wave.step()
   t=m0_wave.t()
   print("t=", t," [",iter0,iter1,"] steps required for m=[0 1]; tol=[",tol0,tol1,"]")
   phi_m0.set_ydata(m0_wave.phi_n())
   phi_m1.set_ydata(m1_wave.phi_n())
   fig.canvas.draw() 
   if (t<tl):
      return True 
   else:
      return False

#gobject.timeout_add(0, animate)
plt.show()