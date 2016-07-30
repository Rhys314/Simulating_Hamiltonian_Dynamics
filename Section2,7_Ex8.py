# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 07:23:29 2015

The solution to exercise 8 in section 2.7 of Simulating Hamiltonia Dynamics by 
B. Leimkuhler and S Reich.

Ex. 8 Computer project with one-step methods
In this excerise you will write a small computer program to test a numerical 
method for solving dq/dt = v, dv/dt = -dphi(q)/dq, q(0) = q^0, v(0) = v^0.  
Refer to the preface for a discussion of computer software.

    a. Write a module eulerstep that takes as input:
        - an arbitrary function phi : R -> R,
        - real scalars q^n, v^n, Dt,
       and computes the results of taking a single step with Euler's method
       applied to the differential equation.
    
    b. Write a computer program stepper which takes as inputs:
        - an arbitrary function phi : R -> R,
        - real scalars q^0, v^0, Dt,
        - integer N,
        - the name of a module (such as eulerstep) which implements a one-step
          method for the differential equation.
       Then solve the system dq/dt = v, dv/dt = -dphi(q)/dq by taking N steps  
       with Euler's method starting from q^0, v^0.  The program should produce  
       as output a pair of (N + 1) one-dimensional arrays Q, V consisting of 
       the beginning and ending positions and velocities and all intermediate 
       steps.

    c. Write modules eulerAstep, eulerBstep, stoemerstep and rk4step with 
       similar inputs and outputs to the module eulerstep but implementing a 
       single timestep of Euler-A, Euler-B, Stoemer-Verlet and forth-order 
       Runge-Kutta methods of the text.

    d. Experiment with the various methods using the stepper routine. Examine
       the energy conservation of the various methods, when applied to a Morse 
       oscillator with unit coefficients, phi(q) = (1 - exp(-q))^2.
    
@author: rpoolman
"""
import Steppers.steppers as step
import numpy as np
import matplotlib.pyplot as plt

# potential functions
LJ = lambda x: np.power(1/x, 12.0) - 2.0*np.power(1/x, 6.0) # Lennard-Jones potential
LJdiff = lambda x: -12*(np.power(1/x, 11.0) - np.power(1/x, 5.0))
Morse = lambda x: np.power((1 - np.exp(-x)), 2) # Morse potential
Oscillator = lambda x: x*x # harmonic oscillator
Dt = 0.0001
N = np.int(2*np.pi/Dt)
q = np.zeros(N + 1, dtype = np.float64)
v = np.zeros(N + 1, dtype = np.float64)
t = np.zeros(N + 1, dtype = np.float64)
M = 1.0


runSubExcerise = 'a'

if runSubExcerise == 'a':
    # a. computes one timestep usingthe Euler method for an arbitrary function
    qn = 2.5
    vn = 0.01
    qn1, vn1 = step.eulerstep(LJ, qn, vn, Dt)
    print(qn1, vn1)
elif runSubExcerise == 'b':
    # b. computes the trajectories of a system dq/dt = v, dv/dt = -dphi(q)/dq
    #    with arbitrary phi(q) using Eulers method
    q[0] = 2.0
    v[0] = 0.001
    for ii in range(N):
        q[ii + 1], v[ii + 1] = step.eulerstep(LJ, q[ii], v[ii], Dt)
        t[ii + 1] = Dt*(ii + 1)
    plt.figure(1)
    plt.clf()
    plt.plot(q, v)
elif runSubExcerise == 'c':
    # c. computes the trajectories of a system dq/dt = v, dv/dt = -phi(q)/dq
    # with arbitrary phi(q) using either Eulere's A method, Euler's B methods, 
    # the Stoemer-Verlet method or a fourth order Runge-Kutta method.
    qExact = np.zeros(N, dtype = np.float64)
    vExact = np.zeros(N, dtype = np.float64)
    useMethod = 'rungeKutta4'
    if useMethod == 'eulerA':
        stepper = step.eulerAstep
    if useMethod == 'eulerB':
        stepper = step.eulerBstep
    if useMethod == 'stoemerVerlet':
        stepper = step.stoemerstep
    if useMethod == 'rungeKutta4':
        stepper = step.rk4step
    q[0] = 1.0
    v[0] = 0.001
    for ii in range(N):
        # numerical approximation
        q[ii + 1], v[ii + 1] = stepper(Oscillator, q[ii], v[ii], Dt, M)
        # exact method check
        qExact[ii] = np.sin(t[ii])
        vExact[ii] = -np.cos(t[ii])
        # indepedent variable
        t[ii + 1] = Dt*(ii + 1)
    plt.figure(1)
    plt.clf()
    plt.plot(q, v, 'b.', qExact, vExact, 'g')
    #plt.axis([-4.0, 4.0,- 4.0, 4.0])

elif runSubExcerise == 'd':
    # arrays
    qEA = np.zeros(N + 1, dtype = np.float64)
    vEA = np.zeros(N + 1, dtype = np.float64)
    qEB = np.zeros(N + 1, dtype = np.float64)
    vEB = np.zeros(N + 1, dtype = np.float64)
    qSV = np.zeros(N + 1, dtype = np.float64)
    vSV = np.zeros(N + 1, dtype = np.float64)
    qRK4 = np.zeros(N + 1, dtype = np.float64)
    vRK4 = np.zeros(N + 1, dtype = np.float64)
    # steppers
    eulerA = step.eulerAstep
    eulerB = step.eulerBstep
    stoemerVerlet = step.stoemerstep
    rungeKutta4 = step.rk4step
    # inital values
    qEA[0] = 1.0
    vEA[0] = 0.001
    qEB[0] = 1.0
    vEB[0] = 0.001
    qSV[0] = 1.0
    vSV[0] = 0.001
    qRK4[0] = 1.0
    vRK4[0] = 0.001
    # solution
    for ii in range(N):
        # numerical approximation
        qEA[ii + 1], vEA[ii + 1] = eulerA(Morse, qEA[ii], vEA[ii], Dt, M)
        qEB[ii + 1], vEB[ii + 1] = eulerB(Morse, qEB[ii], vEB[ii], Dt, M)
        qSV[ii + 1], vSV[ii + 1] = stoemerVerlet(Morse, qSV[ii], vSV[ii], Dt, M)
        qRK4[ii + 1], vRK4[ii + 1] = rungeKutta4(Morse, qRK4[ii], vRK4[ii], Dt, M)
        # indepedent variable
        t[ii + 1] = Dt*(ii + 1)
            
    plt.figure(1)
    plt.clf()
    plt.plot(qEA, vEA, qEB, vEB, qSV, vSV, qRK4, vRK4)
else:
    print "Only four excerises labeled 'a' to 'd'."        
