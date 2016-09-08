# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 21:36:13 2115

@author: seung
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from IPython.display import display
start = time.time()
year = '2014'
month = '03'
day = '05'

# Time window and event counter storage
nWindow = 30
Window_int = np.arange(-nWindow, nWindow+1)
cmeFName = '../../Data/CME/ESH4_Trades_2014_' + month + '_' + day + '.txt.gz'

# Matrix to insert ES and SPY trades
tot_time_slots = int(60*60*1000*6.5+nWindow)
comb_All = np.zeros(tot_time_slots*2).reshape(tot_time_slots, 2)

# Trading times
startTime = pd.to_datetime(month + '/' + day + '/' + year + ' 09:30:01.000')
startTime = startTime.tz_localize('US/Eastern')
stopTime = pd.to_datetime(month + '/' + day + '/' + year + ' 16:00:00.000')
stopTime = stopTime.tz_localize('US/Eastern')
offset1 = 9.5*60*60*1000
            
##################################################################
# CME data
#################################################################         

cmeData = pd.read_csv(cmeFName, compression='gzip', index_col=0)
cmeData.index = pd.to_datetime(cmeData.index, utc=True)
cmeData.index = cmeData.index.tz_convert('US/Eastern')

# Keep only equities markets trading hours
cmeSub = cmeData.loc[(cmeData.index >= startTime) & (cmeData.index <= stopTime)]

# Remove condition codes
cmeSubNoCond = cmeSub.loc[np.isnan(cmeSub.Condition)]

# Pull last prices of the millisecond
cmeUniqueTimes = cmeSubNoCond.index.unique()
cmePrices = cmeSubNoCond.Price.asof(cmeUniqueTimes)

# Compute price diffs
cmeRets = cmePrices.diff()[1:]
time_Ind_ES =cmeUniqueTimes.hour*60*60*1000 + cmeUniqueTimes.minute*60*1000 + \
    cmeUniqueTimes.second*1000 + cmeUniqueTimes.microsecond/1000 - offset1
time_Ind_ES = time_Ind_ES[1:]
time_Ind_ES = time_Ind_ES.astype(int)
info_ES = np.matrix([time_Ind_ES, cmeRets]).T


##################################################################
# ITCH data
##################################################################

# Get the data
itchFName = '../../Data/ITCH/SPY_Trades_' + year + month + day + '.txt.gz'
itchData = pd.read_csv(itchFName, compression='gzip', index_col=0)
itchData.index = pd.to_datetime(itchData.index, utc=False)
itchData.index = itchData.index.tz_localize('US/Eastern')
itchData = itchData.sort()

# Keep only equities markets trading hours
itchSub = itchData.loc[(itchData.index >= startTime) & (itchData.index <= stopTime)]

# Pull last prices of the millisecond
itchUniqueTimes = itchSub.index.unique()
itchPrices = itchSub.Price.asof(itchUniqueTimes)

# Compute the price diffs
itchRets = itchPrices.diff()[1:]
time_Ind_SPY =itchUniqueTimes.hour*60*60*1000 + itchUniqueTimes.minute*60*1000 + \
    itchUniqueTimes.second*1000 + itchUniqueTimes.microsecond/1000 - offset1
time_Ind_SPY = time_Ind_SPY[1:]
time_Ind_SPY = time_Ind_SPY.astype(int)
info_SPY = np.matrix([time_Ind_SPY, itchRets]).T

# Insert into comb_All
comb_All[time_Ind_ES, 0] = cmeRets
comb_All[time_Ind_SPY,1] = itchRets

# ES upticks
cmePosTimes = np.where(comb_All[:, 0] > 0)
cmePosTimes = cmePosTimes[0]
N1 = cmePosTimes.size
#ES downticks
cmeNegTimes = np.where(comb_All[:, 0] < 0)
cmeNegTimes = cmeNegTimes[0]
N2 = cmeNegTimes.size

###############################################################################
# Price Responses
###############################################################################
pr_mat_add1 = np.tile(Window_int, (N1, 1))
temp1 = np.tile(cmePosTimes, (2*nWindow + 1, 1)).T
pr_mat_add_u = pr_mat_add1 + temp1
pr_mat_val_u = comb_All[pr_mat_add_u.reshape(N1*(2*nWindow + 1), 1), 1].reshape(N1, 2*nWindow + 1)

pr_mat_add2 = np.tile(Window_int, (N2, 1))
temp2 = np.tile(cmeNegTimes, (2*nWindow + 1, 1)).T
pr_mat_add_d = pr_mat_add2 + temp2
pr_mat_val_d = comb_All[pr_mat_add_d.reshape(N2*(2*nWindow + 1), 1), 1].reshape(N2, 2*nWindow + 1)

pr_response_up = pr_mat_val_u.sum(0)
pr_response_down = pr_mat_val_d.sum(0)
pr_response_total = pr_response_up - pr_response_down


#% Sigma Calculation
ES_jumps_u = np.sign(comb_All[pr_mat_add_u.reshape(N1*(2*nWindow + 1), 1), 0]);
N_u_sigma = ES_jumps_u.sum(0);    
ES_jumps_d = np.sign(comb_All[pr_mat_add_d.reshape(N2*(2*nWindow + 1), 1), 0]);
N_d_sigma = -1*ES_jumps_d.sum(0)
bin_count_u = (pr_mat_val_u != 0).sum(0)
bin_count_d = (pr_mat_val_d != 0).sum(0)
sigma_u =  np.sqrt(bin_count_u)*.01/N_u_sigma
sigma_d =  np.sqrt(bin_count_d)*.01/N_d_sigma

# normalized price response
res_up_n = pr_response_up/N_u_sigma
res_down_n = pr_response_down/N_d_sigma
res_tot_n = res_up_n - res_down_n

# Plot (normalized)
fig1 = plt.figure(figsize=(12,8))
plt.plot(Window_int, res_tot_n, marker='o', alpha=0.5)
fig1.suptitle('Price Response ' + month + '/' + day + '/' + year)
plt.axvline(c='black')
plt.axvline(x=4,c='red')


#############################################################
# Entropy SPY|ES
#############################################################

#SPY upticks
itchPosTimes = np.where(comb_All[:, 1] > 0)
itchPosTimes = itchPosTimes[0]
Nitch1 = cmePosTimes.size
#SPY downticks
itchNegTimes = np.where(comb_All[:, 1] < 0)
itchNegTimes = itchNegTimes[0]
Nitch2 = cmeNegTimes.size

cutoff = 20
z_es = np.arange(0, tot_time_slots - nWindow )
z_spy = np.arange(0, tot_time_slots - nWindow)
z_es = np.delete(z_es,np.union1d(cmePosTimes, cmeNegTimes))
z_es = np.delete(z_es,np.union1d(cmePosTimes, cmeNegTimes))

pr_ES = np.zeros((3))
pr_SPY = np.zeros((3))

tot_time_slots = float(tot_time_slots)
pr_ES[0] = cmePosTimes.size/tot_time_slots
pr_ES[1]= cmeNegTimes.size/tot_time_slots
pr_ES[2] = 1 - np.sum(pr_ES)

pr_SPY[0] = itchPosTimes.size/tot_time_slots
pr_SPY[1] = itchNegTimes.size/tot_time_slots
pr_SPY[2] = 1 - np.sum(pr_SPY)

Hx = -sum(pr_ES*np.log2(pr_ES))
Hy = -sum(pr_SPY*np.log2(pr_SPY))


N = tot_time_slots - 20
ind1_ = cmePosTimes[np.where(cmePosTimes < N)]
ind2_ = cmeNegTimes[np.where(cmeNegTimes < N)]

I_ = np.zeros((3, 3, 20))
c_pr = np.zeros((3, 3, 20))
j_pr = np.zeros((3, 3, 20))
for i in range(0, 20):
    match_u = ind1_ + i
    match_d = ind2_ + i
    match_0 = np.arange(0, tot_time_slots - nWindow)
    match_0 = np.delete(match_0, np.union1d(match_u, match_d))
    match_0 = match_0.astype(int)
    
    #Sum of Indicator Function 
    I_[0, 0, i]= np.where(comb_All[match_u, 1] > 0)[0].size
    I_[0, 1, i] = np.where(comb_All[match_u, 1] < 0)[0].size
    I_[0, 2, i] = match_u.size - sum(I_[0, :, i]);
    
    I_[1, 0, i]= np.where(comb_All[match_d, 1] > 0)[0].size
    I_[1, 1, i] = np.where(comb_All[match_d, 1] < 0)[0].size
    I_[1, 2, i] = match_d.size - sum(I_[1, :, i]);
    
    I_[2, 0, i]= np.where(comb_All[match_0, 1] > 0)[0].size
    I_[2, 1, i] = np.where(comb_All[match_0, 1] < 0)[0].size
    I_[2, 2, i] = match_0.size - sum(I_[2, :, i]);
    
# %conditional probability
c_pr[0, :, :] = I_[0, :, :]/match_u.size
c_pr[1, :, :] = I_[1, :, :]/match_d.size
c_pr[2, :, :] = I_[2, :, :]/match_0.size
#joint probability
j_pr[0, :, :] = c_pr[0, :, :]*pr_ES[0]
j_pr[1, :, :] = c_pr[1, :, :]*pr_ES[1]
j_pr[2, :, :] = c_pr[2, :, :]*pr_ES[2]

# %H(Y|X)
Hy_c_x = -sum(sum(j_pr[:, :, :]*np.log2(c_pr[:, :, :])))

# %Mutual Information I(Y, X) = H(Y) - H(Y|X)
Iy_x = Hy - Hy_c_x;

#Relative Entropy
#D(p|q) where p = unconditional q = conditional
D = np.zeros((3, 20));
for i in range(0, 20):
    D[0, i] = sum(pr_SPY*np.log2(np.divide(pr_SPY,c_pr[0, :, i])))
    D[1, i] = sum(pr_SPY*np.log2(np.divide(pr_SPY,c_pr[1, :, i])))
    D[2, i] = sum(pr_SPY*np.log2(np.divide(pr_SPY,c_pr[2, :, i])))


#############################################################
# Entropy ES|SPY
#############################################################


N = tot_time_slots - 20
ind3_ = itchPosTimes[np.where(itchPosTimes < N)]
ind4_ = itchNegTimes[np.where(itchNegTimes < N)]

I_1 = np.zeros((3, 3, 20))
c_pr1 = np.zeros((3, 3, 20))
j_pr1 = np.zeros((3, 3, 20))
for i in range(0, 20):
    match_u = ind1_ + i
    match_d = ind2_ + i
    match_0 = np.arange(0, tot_time_slots - nWindow)
    match_0 = np.delete(match_0, np.union1d(match_u, match_d))
    match_0 = match_0.astype(int)
    
    #Sum of Indicator Function 
    I_1[0, 0, i]= np.where(comb_All[match_u, 0] > 0)[0].size
    I_1[0, 1, i] = np.where(comb_All[match_u, 0] < 0)[0].size
    I_1[0, 2, i] = match_u.size - sum(I_1[0, :, i]);
    
    I_1[1, 0, i]= np.where(comb_All[match_d, 0] > 0)[0].size
    I_1[1, 1, i] = np.where(comb_All[match_d, 0] < 0)[0].size
    I_1[1, 2, i] = match_d.size - sum(I_1[1, :, i]);
    
    I_1[2, 0, i]= np.where(comb_All[match_0, 0] > 0)[0].size
    I_1[2, 1, i] = np.where(comb_All[match_0, 0] < 0)[0].size
    I_1[2, 2, i] = match_0.size - sum(I_1[2, :, i]);
    
# %conditional probability
c_pr1[0, :, :] = I_1[0, :, :]/match_u.size
c_pr1[1, :, :] = I_1[1, :, :]/match_d.size
c_pr1[2, :, :] = I_1[2, :, :]/match_0.size
#joint probability
j_pr1[0, :, :] = c_pr1[0, :, :]*pr_SPY[0]
j_pr1[1, :, :] = c_pr1[1, :, :]*pr_SPY[1]
j_pr1[2, :, :] = c_pr1[2, :, :]*pr_SPY[2]

# %H(Y|X)
Hx_c_y = -sum(sum(j_pr[:, :, :]*np.log2(c_pr[:, :, :])))

# %Mutual Information I(Y, X) = H(Y) - H(Y|X)
Ix_y = Hx - Hx_c_y;

#Relative Entropy
#D(p|q) where p = unconditional q = conditional
D1 = np.zeros((3, 20));
for i in range(0, 20):
    D1[0, i] = sum(pr_ES*np.log2(np.divide(pr_ES,c_pr1[0, :, i])))
    D1[1, i] = sum(pr_ES*np.log2(np.divide(pr_ES,c_pr1[1, :, i])))
    D1[2, i] = sum(pr_ES*np.log2(np.divide(pr_ES,c_pr1[2, :, i])))




#figures------------------------------------------------------
# Plot (normalized)

fig2 = plt.figure(figsize=(12,8))
ax = plt.subplot(2, 1, 1)
ax.plot(Ix_y)
ax.set_title('Mutual Information of given ES')
ax.axvline(x=4,c='red')

ax = plt.subplot(2, 1, 2)
ax.plot(Iy_x)
ax.set_title('Mutual Information of given SPY')
ax.axvline(x=4,c='red')


fig3 = plt.figure(figsize=(12,8))
fig3.suptitle('Relative Entropy conditional on ES')
ax = plt.subplot(3, 1, 1)
ax.plot(D[0, :], c='red')
ax.set_title('relative entropy conditional up')
ax.axvline(x=4,c='red')

ax = plt.subplot(3, 1, 2)
ax.plot(D[1, :], c='red')
ax.set_title('relative entropy conditional down')
ax.axvline(x=4,c='red')

ax = plt.subplot(3, 1, 3)
ax.plot(D[2, :], c='red')
ax.set_title('relative entropy conditional unch')
ax.axvline(x=4,c='red')


fig4 = plt.figure(figsize=(12,8))
fig4.suptitle('Relative Entropy conditional on SPY')
ax = plt.subplot(3, 1, 1)
ax.plot(D1[0, :], c='red')
ax.set_title('relative entropy conditional up')
ax.axvline(x=4,c='red')

ax = plt.subplot(3, 1, 2)
ax.plot(D1[1, :], c='red')
ax.set_title('relative entropy conditional down')
ax.axvline(x=4,c='red')

ax = plt.subplot(3, 1, 3)
ax.plot(D1[2, :], c='red')
ax.set_title('relative entropy conditional unch')
ax.axvline(x=4,c='red')


plt.show()
print time.time() - start





