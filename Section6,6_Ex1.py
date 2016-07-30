# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 19:18:52 2016
Section 6.6, Ex. 1 Numerical Comparison for Arenstorf Orbit

Implement Scovel's second order splitting method (compare Section 4.5.2) for 
the restricted three body problem of Section 6.5.1.  Using Scovel's method, 
also implement the higher-order composition methods as described in Sections 
6.2.2 - 6.2.3.  Reproduce the data show in Fig 6.2.  Note that the work per 
unit time interval is defined as the number of timesteps taken to approximate 
the solution over one period T multiplied by the number of the method and 
divided by the period T.  The accuracy is defined in terms of absolute error 
in the computed position after one period, i.e.
                        e(T) = max(x(T) - 0.994, y(T)).
@author: Rhys Poolman
"""
import numpy as np
import scipy as sp
import Steppers.scovel as stp
import multiprocessing as mp
import pickle as pkl
import matplotlib.pyplot as plt
import itertools as its

def Integrate(stepper, key, q0, p0, T, output):
    results = {}
    Dt = stepper[key].m_Dt
    N = np.int(T/Dt)
    q = np.zeros((N, 3))
    p = np.zeros((N, 3))
    q[0, :] = q0
    p[0, :] = p0
    stp = stepper[key]
    if key[:6] == 'Scovel':
        for nn in range(N - 1):
            q[nn + 1, :], p[nn + 1, :] = stp.m_step(q[nn, :], p[nn, :])
    elif key[:12] == 'Post-Process':
        stp.integrate(q0, p0, Dt*(N - 1))
    else:
        for nn in range(N - 1):
            q[nn + 1, :], p[nn + 1, :] = stp.step(q[nn, :], p[nn, :])
    results[Dt] = [q, p]
    
    outdict = {}
    outdict[key] = results
    output.put(outdict)

Dts = np.array([0.001, 0.00066, 0.00033, 0.0001, 0.000066, 0.000033, 0.00001],
               dtype = np.float64()) 

# inital values and preallocted arrays
T = np.float64(17.06521656015796255)
#N = np.int(T/Dt)
mu1 = np.float64(0.012277471)
mu2 = 1.0 - mu1
b = np.zeros(3)
b[2] = -2.0
def r1(x, y):
    return np.sqrt((x - mu2)**2 + y**2)
def r2(x, y):
    return np.sqrt((x + mu1)**2 + y**2)
def V(x, y, z):
    return -(x**2 + y**2)/2 - mu1/r1(x, y) - mu2/r2(x, y)
def dVdq(q):
    return - q - \
           mu1/r1(q[0], q[1])**3*np.array([mu2 - q[0], 
                                           -q[1],
                                           0.0]) + \
           mu2/r2(q[0], q[1])**3*np.array([mu1 + q[0], 
                                           q[1],
                                           0.0])

# integrate Hamiltonian
stepper = {}
for Dt in Dts:
    key = 'Scovel, Dt = {}'.format(Dt)
    stepper[key] = stp._ScovelsMethod(dVdq, 1.0, b, Dt)
    key = '4th order, s = 5, Dt = {}'.format(Dt)
    w = np.array([0.28, 0.62546642846767004501])
    stepper[key] = stp.Composition(dVdq, 1.0, b, w, Dt)
    key = '6th order, s = 7, Dt = {}'.format(Dt)
    w = np.array([0.78451361047755726382, 0.23557321335935813368, \
                  -1.17767998417887100695])
    stepper[key] = stp.Composition(dVdq, 1.0, b, w, Dt)
    key = '6th order, s = 9, Dt = {}'.format(Dt)
    w = np.array([0.39216144400731413928, 0.33259913678935943860, \
                  -0.70624617255763935981, 0.08221359629355080023])
    stepper[key] = stp.Composition(dVdq, 1.0, b, w, Dt)
    key = '8th order, s = 15, Dt = {}'.format(Dt)
    w = np.array([0.74167036435061295345, -0.40910082580003159400,
                  0.19075471029623837995, -0.57386247111608226666,
                  0.29906418130365592384, 0.33462491824529818378,
                  0.31529309239676659663])
    stepper[key] = stp.Composition(dVdq, 1.0, b, w, Dt)
    key = 'Post-Process, Dt = {}'.format(Dt)
    w = np.array([0.513910778424374, 0.364193022833858,
                  -0.867423280969274])
    c = np.array([-0.461165940466494, -0.074332422810238, 
                  0.384998538774070, 0.375012038697862])
    q0 = np.array([0.994, 0.0, 0.0])
    p0 = np.array([0.0, -2.001585106379082, 0.0])
    stepper[key] = stp.Processing(dVdq, 1.0, b , w, c, Dt)

recalculate = False
if recalculate:   
        
    # note the s refers to the number of stages see L + R, section 6.2, pp. 142
    outputQueue = mp.Queue()
    results = {}
    
    if __name__ == '__main__':
        keys = stepper.keys()
        procs = []
        for key in keys:
            procs.append( mp.Process(target = Integrate, 
                                     args = (stepper, key, q0, p0, T, outputQueue)) )
            procs[-1].start()
        
        for ii in range(len(procs)):
            results.update(outputQueue.get())
        
        for p in procs:
            p.join()
            print(p.exitcode)
            
    # save data to file
    with open(r'Section6,6_Ex1.pickle', 'wb') as file:
        pkl.dump(results, file)
else:
    with open(r'Section6,6_Ex1_Large.pickle', 'rb') as file:
        results = pkl.load(file)
    
    errors = {}
    errors['Scovel'] = np.zeros((len(Dts), 2))
    errors['4th order, s = 5'] = np.zeros((len(Dts), 2))
    errors['6th order, s = 7'] = np.zeros((len(Dts), 2))
    errors['6th order, s = 9'] = np.zeros((len(Dts), 2))
    errors['8th order, s = 15'] = np.zeros((len(Dts), 2))
    errors['Post-Process'] = np.zeros((len(Dts), 2))
    for key in results.keys():
        splitKey = key.rpartition(', Dt = ')

        Dt = np.float64(splitKey[2])        
        ii = np.where(Dts == Dt)[0][0]
        
        posError = sp.linalg.norm(results[key][Dt][0][0, :] - \
                                  results[key][Dt][0][-1, :])        
        work = stepper[key].workPerUnitTime()
        
        errors[splitKey[0]][ii, 0] = work
        errors[splitKey[0]][ii, 1] = posError

# plot
fig = plt.figure(1)
plt.clf()
plt.title('accuracy-work diagram')
styles = its.cycle(['xb-', '+g-', '*r-', '^c-', 'om-', 'Dy-'])
lines = []
labels = []
for error in errors.items():
    style = next(styles)
    line, = plt.loglog(error[1][:, 0], error[1][:, 1], style)
    lines.append(line)
    labels.append(error[0])
plt.xlabel('number of force evaluations per unit time interval')
plt.ylabel('positional error after one period')
fig.legend(lines, labels)