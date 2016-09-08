# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 20:38:58 2015

@author: seunglee
"""

#change basis testing

import numpy as np
import matplotlib.pyplot as plt

nWindow = 30
Window_int = np.arange(-nWindow, nWindow+1)

#these should be turned into functions
bookES = np.load('bookES.npy')
bookSPY = np.load('bookSPY.npy')
basis_med = np.load('basis.npy')

#############################################
bookSPY = bookSPY.transpose()
bookES = bookES.transpose()
bookES[:, 0] = bookES[:, 0]/basis_med
bookES[:, 2] = bookES[:, 2]/basis_med
bookES[:, 4] = bookES[:, 4]/basis_med

bookES[:, 1] = .01*bookES[:, 1]*basis_med/2
bookES[:, 3] = .01*bookES[:, 3]*basis_med/2*100
bookES[:, 5] = .01*bookES[:, 5]*basis_med/2

temp = np.arange(60**2*6.5*1000).astype('int')
bookES = np.column_stack((bookES, temp))
bookSPY = np.column_stack((bookSPY, temp))
###############################################


#There are 8 cases: es/spy --> spy/es bid1>bid2, bid2>bid1, off1>off2, off2>off1

#case1: where es --> spy both trade on bid, bidES > bidSPY (NO Response GOOD)
subData = bookES[(bookES[:, 2] == bookES[:, 0])  & (bookES[:, 3] > 1), :]
ind1 = subData[:, 6].astype('int')
ind1 = ind1 - 1
N1 = ind1.size
ind1_vol = subData[:,3].astype('int')

sec2_jumps1 = (bookSPY[:, 2] == bookSPY[:, 0]) & (bookES[:, 0] > bookSPY[:, 0])
sec2_vol1 = np.multiply(((bookSPY[:, 2] == bookSPY[:, 0]) & (bookES[:, 0] > bookSPY[:, 0])).astype('double'), bookSPY[:, 3]).astype('double')

pr_mat_add1 = np.tile(Window_int, (N1, 1)) + 2
temp1 = np.tile(ind1, (2*nWindow + 1, 1)).T
pr_mat_add = pr_mat_add1 + temp1
pr_mat_val1 = sec2_jumps1[pr_mat_add.reshape(N1*(2*nWindow + 1))].reshape(N1, 2*nWindow + 1)


vol1_mat = np.tile(ind1_vol, (2*nWindow + 1, 1)).T
pr_mat_val_vol1 = sec2_vol1[pr_mat_add.reshape(N1*(2*nWindow + 1))].reshape(N1, 2*nWindow + 1)
pr_mat_val_vol1[pr_mat_val_vol1 > vol1_mat] = vol1_mat[pr_mat_val_vol1 > vol1_mat]

res1 = pr_mat_val1.sum(0).astype('double')/N1
test = pr_mat_val_vol1.sum(0)
res1_vol = np.divide(pr_mat_val_vol1, vol1_mat).sum(0)
#######################################################################

#case2: where es --> spy both trade on bid, bidES < bidSPY (Response GOOD)
sec2_jumps2 = (bookSPY[:, 2] == bookSPY[:, 0]) & (bookES[:, 0] < bookSPY[:, 0])
sec2_vol2 = np.multiply(((bookSPY[:, 2] == bookSPY[:, 0]) & (bookES[:, 0] < bookSPY[:, 0])).astype('double'), bookSPY[:, 3]).astype('double')


pr_mat_val2 = sec2_jumps2[pr_mat_add.reshape(N1*(2*nWindow + 1))].reshape(N1, 2*nWindow + 1)
pr_mat_val_vol2 = sec2_vol2[pr_mat_add.reshape(N1*(2*nWindow + 1))].reshape(N1, 2*nWindow + 1)
pr_mat_val_vol2[pr_mat_val_vol2 > vol1_mat] = vol1_mat[pr_mat_val_vol2 > vol1_mat]

res2 = pr_mat_val2.sum(0).astype('double')/N1
test2 = pr_mat_val_vol2.sum(0)
res2_vol = np.divide(pr_mat_val_vol2, vol1_mat).sum(0)
#######################################################################


fig1 = plt.figure(figsize=(12,8))
plt.plot(Window_int, res1, color = 'black')
plt.plot(Window_int, res2, color='red')
plt.title('ES --> SPY (bid) (red higher good)')

fig2 = plt.figure(figsize=(12,8))
plt.plot(Window_int, res1_vol, color = 'black')
plt.plot(Window_int, res2_vol, color='red')
plt.title('Volume ES --> SPY (bid) (red higher good)')


#case3: where es --> spy both trade on ask, askES > askSPY (Response GOOD)
subData = bookES[(bookES[:, 2] == bookES[:, 4])  & (bookES[:, 3] > 1), :]
ind3 = subData[:, 6].astype('int')
ind3 = ind3 - 1
N3 = ind3.size
ind3_vol = subData[:,3].astype('int')

sec2_jumps3 = (bookSPY[:, 2] == bookSPY[:, 4]) & (bookES[:, 4] > bookSPY[:, 4])
sec2_vol3 = np.multiply(((bookSPY[:, 2] == bookSPY[:, 4]) & (bookES[:, 4] > bookSPY[:, 4])).astype('double'), bookSPY[:, 3]).astype('double')

pr_mat_add3 = np.tile(Window_int, (N3, 1)) + 2
temp3 = np.tile(ind3, (2*nWindow + 1, 1)).T
pr_mat_add = pr_mat_add3 + temp3
pr_mat_val3 = sec2_jumps3[pr_mat_add.reshape(N3*(2*nWindow + 1))].reshape(N3, 2*nWindow + 1)


vol3_mat = np.tile(ind3_vol, (2*nWindow + 1, 1)).T
pr_mat_val_vol3 = sec2_vol3[pr_mat_add.reshape(N3*(2*nWindow + 1))].reshape(N3, 2*nWindow + 1)
pr_mat_val_vol3[pr_mat_val_vol3 > vol3_mat] = vol3_mat[pr_mat_val_vol3 > vol3_mat]

res3 = pr_mat_val3.sum(0).astype('double')/N3
test = pr_mat_val_vol3.sum(0)
res3_vol = np.divide(pr_mat_val_vol3, vol3_mat).sum(0)
#######################################################################

#case4: where es --> spy both trade on offer, bidES < bidSPY (No Response GOOD)
sec2_jumps4 = (bookSPY[:, 2] == bookSPY[:, 4]) & (bookES[:, 4] < bookSPY[:, 4])
sec2_vol4 = np.multiply(((bookSPY[:, 2] == bookSPY[:, 4]) & (bookES[:, 4] < bookSPY[:, 4])).astype('double'), bookSPY[:, 3]).astype('double')

pr_mat_val4 = sec2_jumps4[pr_mat_add.reshape(N3*(2*nWindow + 1))].reshape(N3, 2*nWindow + 1)
pr_mat_val_vol4 = sec2_vol4[pr_mat_add.reshape(N3*(2*nWindow + 1))].reshape(N3, 2*nWindow + 1)
pr_mat_val_vol4[pr_mat_val_vol4 > vol3_mat] = vol3_mat[pr_mat_val_vol4 > vol3_mat]

res4 = pr_mat_val4.sum(0).astype('double')/N3
test2 = pr_mat_val_vol4.sum(0)
res4_vol = np.divide(pr_mat_val_vol4, vol3_mat).sum(0)
#######################################################################


fig3 = plt.figure(figsize=(12,8))
plt.plot(Window_int, res4, color = 'black')
plt.plot(Window_int, res3, color='red')
plt.title('ES --> SPY (ask) (red higher good)')

fig4 = plt.figure(figsize=(12,8))
plt.plot(Window_int, res4_vol, color = 'black')
plt.plot(Window_int, res3_vol, color='red')
plt.title('Volume ES --> SPY (ask) (red higher good)')


#case5: where spy --> es both trade on bid, bidSPY > bidES (NO Response GOOD)
subData = bookSPY[(bookSPY[:, 2] == bookSPY[:, 0])  & (bookSPY[:, 3] > 1), :]
ind5 = subData[:, 6].astype('int')
ind5 = ind5 - 1
N5 = ind5.size
ind5_vol = subData[:,3].astype('int')

sec2_jumps5 = (bookES[:, 2] == bookES[:, 0]) & (bookSPY[:, 0] > bookES[:, 0])
sec2_vol5 = np.multiply(((bookES[:, 2] == bookES[:, 0]) & (bookSPY[:, 0] > bookES[:, 0])).astype('double'), bookES[:, 3]).astype('double')

pr_mat_add5 = np.tile(Window_int, (N5, 1)) + 2
temp5 = np.tile(ind5, (2*nWindow + 1, 1)).T
pr_mat_add = pr_mat_add5 + temp5
pr_mat_val5 = sec2_jumps5[pr_mat_add.reshape(N5*(2*nWindow + 1))].reshape(N5, 2*nWindow + 1)


vol5_mat = np.tile(ind5_vol, (2*nWindow + 1, 1)).T
pr_mat_val_vol5 = sec2_vol5[pr_mat_add.reshape(N5*(2*nWindow + 1))].reshape(N5, 2*nWindow + 1)
pr_mat_val_vol5[pr_mat_val_vol5 > vol5_mat] = vol5_mat[pr_mat_val_vol5 > vol5_mat]

res5 = pr_mat_val5.sum(0).astype('double')/N5
test = pr_mat_val_vol5.sum(0)
res5_vol = np.divide(pr_mat_val_vol5, vol5_mat).sum(0)
#######################################################################

#case6: where es --> spy both trade on bid, bidSPY < bidES (Response GOOD)
sec2_jumps6 = (bookES[:, 2] == bookES[:, 0]) & (bookSPY[:, 0] < bookES[:, 0])
sec2_vol6 = np.multiply(((bookES[:, 2] == bookES[:, 0]) & (bookSPY[:, 0] < bookES[:, 0])).astype('double'), bookES[:, 3]).astype('double')


pr_mat_val6 = sec2_jumps6[pr_mat_add.reshape(N5*(2*nWindow + 1))].reshape(N5, 2*nWindow + 1)
pr_mat_val_vol6 = sec2_vol6[pr_mat_add.reshape(N5*(2*nWindow + 1))].reshape(N5, 2*nWindow + 1)
pr_mat_val_vol6[pr_mat_val_vol6 > vol5_mat] = vol5_mat[pr_mat_val_vol6 > vol5_mat]

res6 = pr_mat_val6.sum(0).astype('double')/N5
test = pr_mat_val_vol6.sum(0)
res6_vol = np.divide(pr_mat_val_vol6, vol5_mat).sum(0)
#######################################################################


fig5 = plt.figure(figsize=(12,8))
plt.plot(Window_int, res5, color = 'black')
plt.plot(Window_int, res6, color='red')
plt.title('ES --> SPY (bid) (red higher good)')

fig6 = plt.figure(figsize=(12,8))
plt.plot(Window_int, res5_vol, color = 'black')
plt.plot(Window_int, res6_vol, color='red')
plt.title('Volume ES --> SPY (bid) (red higher good)')



#case7: where spy --> es both trade on ask, askspy > askes (Response GOOD)
subData = bookSPY[(bookSPY[:, 2] == bookSPY[:, 4])  & (bookSPY[:, 3] > 1), :]
ind7 = subData[:, 6].astype('int')
ind7 = ind7 - 1
N7 = ind7.size
ind7_vol = subData[:,3].astype('int')

sec2_jumps7 = (bookES[:, 2] == bookES[:, 4]) & (bookSPY[:, 4] > bookES[:, 4])
sec2_vol7 = np.multiply(((bookES[:, 2] == bookES[:, 4]) & (bookSPY[:, 4] > bookES[:, 4])).astype('double'), bookES[:, 3]).astype('double')

pr_mat_add7 = np.tile(Window_int, (N7, 1)) + 2
temp7 = np.tile(ind7, (2*nWindow + 1, 1)).T
pr_mat_add = pr_mat_add7 + temp7
pr_mat_val7 = sec2_jumps7[pr_mat_add.reshape(N7*(2*nWindow + 1))].reshape(N7, 2*nWindow + 1)


vol7_mat = np.tile(ind7_vol, (2*nWindow + 1, 1)).T
pr_mat_val_vol7 = sec2_vol7[pr_mat_add.reshape(N7*(2*nWindow + 1))].reshape(N7, 2*nWindow + 1)
pr_mat_val_vol7[pr_mat_val_vol7 > vol7_mat] = vol7_mat[pr_mat_val_vol7 > vol7_mat]

res7 = pr_mat_val7.sum(0).astype('double')/N7
test = pr_mat_val_vol7.sum(0)
res7_vol = np.divide(pr_mat_val_vol7, vol7_mat).sum(0)
#######################################################################

#case8: where spy --> es both trade on offer, askES < askSPY (No Response GOOD)
sec2_jumps8 = (bookES[:, 2] == bookES[:, 4]) & (bookSPY[:, 4] < bookES[:, 4])
sec2_vol8 = np.multiply(((bookES[:, 2] == bookES[:, 4]) & (bookSPY[:, 4] < bookES[:, 4])).astype('double'), bookES[:, 3]).astype('double')

pr_mat_val8 = sec2_jumps8[pr_mat_add.reshape(N7*(2*nWindow + 1))].reshape(N7, 2*nWindow + 1)
pr_mat_val_vol8 = sec2_vol8[pr_mat_add.reshape(N7*(2*nWindow + 1))].reshape(N7, 2*nWindow + 1)
pr_mat_val_vol8[pr_mat_val_vol8 > vol7_mat] = vol7_mat[pr_mat_val_vol8 > vol7_mat]

res8 = pr_mat_val8.sum(0).astype('double')/N7
test = pr_mat_val_vol8.sum(0)
res8_vol = np.divide(pr_mat_val_vol8, vol7_mat).sum(0)
#######################################################################


fig7 = plt.figure(figsize=(12,8))
plt.plot(Window_int, res7, color = 'red')
plt.plot(Window_int, res8, color='black')
plt.title('ES --> SPY (ask) (red higher good)')

fig8 = plt.figure(figsize=(12,8))
plt.plot(Window_int, res7_vol, color = 'red')
plt.plot(Window_int, res8_vol, color='black')
plt.title('Volume SPY --> ES (ask) (red higher good)')




