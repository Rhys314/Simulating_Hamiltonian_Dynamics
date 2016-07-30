# -*- coding: utf-8 -*-
"""
Created on Fri Nov 06 15:41:54 2015
Graphs the potnetial of Kepler problem with one-degree-of-freedom as requested 
at the end of excercise 5b in section 3.7 of Leimkuhler and Reich.
@author: rpoolman
"""
import numpy as np
import matplotlib.pyplot as plt

N = 1001
N_angMmtm = 11
psi = lambda m3, r: m3**2.0/(2.0*r**2.0) - 1.0/r
r = np.linspace(1.0, 10.0, N)
angMmtm = np.linspace(0.0, 1.0, N_angMmtm)
U = []
traces = []
legendEntries = []

# calculate potential
for m3 in angMmtm:
    U.append(psi(m3, r))
   
# plot potential
plt.figure(1)
plt.clf()
for ii in range(0, N_angMmtm):
    legendEntries.append('$m_3 = $' + np.str(angMmtm[ii]))
    traces.append(plt.plot(r, U[ii], label = legendEntries))
plt.legend(traces, legendEntries, 'upper right')
plt.xlabel('Radial Coordiante, r')
plt.ylabel('Potential Energy, $\psi = m_3^2/2r^2 - 1/r$')
plt.show()

# plot potential starting point angainst angular momentum
plt.figure(2)
plt.clf()
plt.plot(angMmtm, np.array(U)[:, 0])
plt.xlabel('Angular Momentum, $m_3$')
plt.ylabel('Potential Energy near Origin, $\psi(1.0)$')
plt.show()
plt.show()