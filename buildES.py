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

# Trading times
startTime = pd.to_datetime(month + '/' + day + '/' + year + ' 09:30:00.000')
startTime = startTime.tz_localize('US/Eastern')
stopTime = pd.to_datetime(month + '/' + day + '/' + year + ' 16:00:00.000')
stopTime = stopTime.tz_localize('US/Eastern')

# Import the data
cmeData = pd.read_pickle('/Users/seunglee/Github/ResponseData/es_'+year+month+day+'.pkl')
#cmeData = pd.read_pickle('C:\Users\seunglee\Dropbox\shareUbuntu\es_'+year+month+day+'.pkl')

# Subset the data
cmeSub = cmeData.loc[(cmeData.index >= startTime) & (cmeData.index <= stopTime)]

nMS = np.array(cmeSub.index-startTime,dtype='timedelta64[ms]').astype('double')
nUpdate = np.array(cmeSub.UpdateType).astype('double')
nPrice = np.array(cmeSub.Price).astype('double')
nSize = np.array(cmeSub.Size).astype('double')
nLevel = np.array(cmeSub.Level).astype('double')
nEntry = np.array(cmeSub.EntryType).astype('double')

nMat = np.matrix([nMS, nUpdate, nEntry, nPrice, nSize, nLevel])
temp = ~(((nMat[5, :] > 1) & (nMat[2, :] != 2)) | (nMat[1, :] ==2))
a = nMat[:, temp.view(np.ndarray).ravel()==1]

uniqA = uniq_(a)
bookBids = uniqA[:, uniqA[2, :] == 0] 
bookOffers = uniqA[:, uniqA[2, :] == 1]

temp = ((nMat[2, :] == 2) & (nMat[5, :] == 2))
ES_tradeB = nMat[:, temp.view(np.ndarray).ravel()==1]
ES_tradeB = uniq_(ES_tradeB)

temp = ((nMat[2, :] == 2) & (nMat[5, :] == 1))
ES_tradeO = nMat[:, temp.view(np.ndarray).ravel()==1]
ES_tradeO = uniq_(ES_tradeO)


bookES = np.zeros(shape=(6, 60**2*1000*6.5))

ind_ = bookBids[0, :].astype('int') - 1
bookES[0, ind_] = bookBids[3, :] 
bookES[1, ind_] = bookBids[4, :]

ind_ = bookOffers[0, :].astype('int') -1
bookES[4, ind_] = bookOffers[3, :]
bookES[5, ind_] = bookOffers[4, :]

ind_ = ES_tradeB[0, :].astype('int') - 1
bookES[2, ind_] = ES_tradeB[3, :]
bookES[3, ind_] = ES_tradeB[4, :]
bookES[0, ind_] = ES_tradeB[3, :]
bookES[1, ind_] = ES_tradeB[4, :]
bookES[4, ind_] = ES_tradeB[3, :] + 25

ind_ = ES_tradeO[0, :].astype('int') - 1
bookES[2, ind_] = ES_tradeO[3, :]
bookES[3, ind_] = ES_tradeO[4, :]
bookES[4, ind_] = ES_tradeO[3, :]
bookES[5, ind_] = ES_tradeO[4, :]
bookES[0, ind_] = ES_tradeO[3, :] - 25

bookES[0, (bookES[0, :] == 0) & (bookES[4, :] > 0)] = bookES[4, (bookES[0, :] == 0) & (bookES[4, :] > 0)] - 25
bookES[4, (bookES[0, :] > 0) & (bookES[4, :] == 0)] = bookES[0, (bookES[0, :] > 0) & (bookES[4, :] == 0)] + 25

bid = 15150
ask = 151275
for i in range(0, int(60**2*1000*6.5)):
    if(bookES[0, i] == 0):
        bookES[0, i] = bid
    else:
        bid = bookES[0, i]
        
    if(bookES[4, i] == 0):
        bookES[4, i] = ask
    else:
        ask = bookES[4, i]
        
np.save('bookES.npy', bookES)
bookES=bookES.transpose()
np.savetxt('pythones.csv', bookES[0:200000, :], fmt = '%.2f', delimiter = ',')















