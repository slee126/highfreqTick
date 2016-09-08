# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 22:26:08 2015

@author: peeay
"""
#change the test basis
import numpy as np

bookES = np.load('bookES.npy')
bookSPY = np.load('bookSPY.npy')

ind1 = bookES[1, :].ravel().nonzero()
ind1 = ind1[0][:]
ind2 = bookES[5, :].ravel().nonzero()
ind2 = ind2[0][:]
ind5 = np.union1d(ind1, ind2)

ind3 = bookSPY[1, :].ravel().nonzero()
ind3 = ind3[0][:]
ind4 = bookSPY[5, :].ravel().nonzero()
ind4 = ind4[0][:]
ind6 = np.union1d(ind3, ind4)

ind = np.union1d(ind5, ind6)
midES = .5*bookES[0, ind] + .5*bookES[4, ind]
midSPY = .5*bookSPY[0, ind] + .5*bookSPY[4, ind]

basis = np.divide(midES, midSPY)
basis_avg = np.average(basis)
basis_med = np.average(basis)
basis_med = 9.972461330431052*100

np.save('basis.npy', basis_med)
