# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 07:26:05 2016
Tests.
@author: rpoolman
"""
import numpy as np

N = 321
test = np.array([np.float64(x) for x in range(N)])

for ii in range(N):
    iiprev = ii - 1
    if ii == 0:
        iiprev = N - 1
    iinext = ii + 1
    if ii == N - 1:
        iinext = 0
    print([test[iiprev], test[ii], test[iinext]])
