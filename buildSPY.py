# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
from pandas.tseries.offsets import *

def uniq_(mat):
    counter = 0

    uniqA = np.zeros(mat.shape)
    for i in range(mat.shape[1]-1):
        if mat[0, i+1] != mat[0,i]:
            uniqA[:, counter] = mat[:, i].transpose()
            counter +=1
    return uniqA[:, 0:counter]
    
year = '2013'
month = '02'
day = '14'

# Trading timSPY
startTime = pd.to_datetime(month + '/' + day + '/' + year + ' 09:30:00.000')
#startTime = startTime.tz_localize('US/Eastern')
stopTime = pd.to_datetime(month + '/' + day + '/' + year + ' 16:00:00.000')
#stopTime = stopTime.tz_localize('US/Eastern')

# Import the data
#itchData = pd.read_pickle('/Users/seunglee/Github/ResponseData/spyItch_'+year+month+day+'.pkl')
itchData = pd.read_csv('/Users/seunglee/Dropbox/Github/ResponseData/SPY_Quotes_20130214.txt')
# Import the data
#@itchData = pd.read_pickle('C:\Users\seunglee\Dropbox\shareUbuntu\spyItch_'+year+month+day+'.pkl')
#itchData = pd.read_pickle('C:\Users\peeay\Dropbox\shareUbuntu\spyItch_'+year+month+day+'.pkl')
#itchData = pd.read_csv('C:\Users\peeay\Dropbox\shareUbuntu\SPY_Quotes_20130214.txt')
itchData.index = pd.to_datetime(itchData.Time)

# Subset the data
itchSub = itchData.loc[(itchData.index >= startTime) & (itchData.index <= stopTime)]

nMS = np.array(itchSub.index-startTime,dtype='timedelta64[ms]').astype('double')
nUpdate = np.array(itchSub.UpdateType).astype('double')
nPrice = np.array(itchSub.Price).astype('double')
nSize = np.array(itchSub.Size).astype('double')
nLevel = np.array(itchSub.Level).astype('double')
nEntry = np.array(itchSub.EntryType).astype('double')

nMat = np.matrix([nMS, nUpdate, nEntry, nPrice, nSize, nLevel])

temp = ~(((nMat[5, :] > 1) & (nMat[2, :] != 2)))
a = nMat[:, temp.view(np.ndarray).ravel()==1]

uniqA = uniq_(a)
bookBids = uniqA[:, uniqA[2, :] == 0]

bookOffers = uniqA[:, uniqA[2, :] == 1]

temp = ((nMat[2, :] == 2) & (nMat[5, :] == 2))
SPY_tradeB = nMat[:, temp.view(np.ndarray).ravel()==1]
SPY_tradeB = uniq_(SPY_tradeB)

temp = ((nMat[2, :] == 2) & (nMat[5, :] == 1))
SPY_tradeO = nMat[:, temp.view(np.ndarray).ravel()==1]
SPY_tradeO = uniq_(SPY_tradeO)

bookSPY = np.zeros(shape=(6, 60**2*1000*6.5))

ind_ = bookBids[0, :].astype('int') - 1
bookSPY[0, ind_] = bookBids[3, :]
bookSPY[1, ind_] = bookBids[4, :]

ind_ = bookOffers[0, :].astype('int') - 1
bookSPY[4, ind_] = bookOffers[3, :]
bookSPY[5, ind_] = bookOffers[4, :]

ind_ = SPY_tradeB[0, :].astype('int') - 1
bookSPY[2, ind_] = SPY_tradeB[3, :]
bookSPY[3, ind_] = SPY_tradeB[4, :]
bookSPY[0, ind_] = SPY_tradeB[3, :]
bookSPY[1, ind_] = SPY_tradeB[4, :]
bookSPY[4, ind_] = SPY_tradeB[3, :] + .01

ind_ = SPY_tradeO[0, :].astype('int') - 1
bookSPY[2, ind_] = SPY_tradeO[3, :]
bookSPY[3, ind_] = SPY_tradeO[4, :]
bookSPY[4, ind_] = SPY_tradeO[3, :]
bookSPY[5, ind_] = SPY_tradeO[4, :]
bookSPY[0, ind_] = SPY_tradeO[3, :] -.01

bookSPY[0, (bookSPY[0, :] == 0) & (bookSPY[4, :] > 0)] = bookSPY[4, (bookSPY[0, :] == 0) & (bookSPY[4, :] > 0)] - .01
bookSPY[4, (bookSPY[0, :] > 0) & (bookSPY[4, :] == 0)] = bookSPY[0, (bookSPY[0, :] > 0) & (bookSPY[4, :] == 0)] + .01

bid = 151.69
ask = 151.70
for i in range(0, int(60**2*1000*6.5)):
    if(bookSPY[0, i] == 0):
        bookSPY[0, i] = bid
    else:
        bid = bookSPY[0, i]
        
    if(bookSPY[4, i] == 0):
        bookSPY[4, i] = ask
    else:
        ask = bookSPY[4, i]
        
np.save('bookSPY.npy', bookSPY)
bookSPY=bookSPY.transpose()
np.savetxt('pythonspy.csv', bookSPY[0:200000, :], fmt = '%.2f', delimiter = ',')

        
        















