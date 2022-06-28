from astropy.io import fits
import numpy as np
import scipy.spatial
from sklearn.neighbors import KDTree, BallTree
from scipy.stats import poisson, erlang
from scipy import interpolate
from os import urandom
import struct
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from astropy.coordinates import SkyCoord
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['text.color'] = 'black'
matplotlib.rcParams['grid.color'] = 'grey'
matplotlib.rcParams['grid.linestyle'] = '--'
matplotlib.rcParams['grid.linewidth'] = 0.4
matplotlib.rcParams['grid.alpha'] = 0.5
fig = plt.figure()
from matplotlib.ticker import AutoMinorLocator, LogLocator
cosmo = FlatLambdaCDM(H0=67.74, Om0=0.3089)
h = 0.6774
baseDir = '/oak/stanford/orgs/kipac/users/ycwang19/VPF/SZ/redMapper'

################################

def CDFVolkNN(vol): # CDF
    CDF = []
    N = vol.shape[0]
    l = vol.shape[1]
    gof = ((np.arange(0, N) + 1) / N)
    for c in range(l):
        ind = np.argsort(vol[:, c])
        sVol= vol[ind, c]
        # return array of interpolating functions
        CDF.append(interpolate.interp1d(sVol, gof, kind = 'linear', \
                                        bounds_error=False)) # x = sVol, y = gof
    return CDF

Dis = np.loadtxt(baseDir+'/Dis_RM.txt') 
Disi = np.loadtxt(baseDir+'/Disi_RM.txt')
print(Dis.shape)
print(Disi.shape)
ks = np.asarray([1, 5, 10, 50, 2, 3, 4, 8, 5, 9]) # All k orders

D1 = Dis[:, 0]
D2 = Dis[:, 4]
D3 = Dis[:, 5]
D4 = Dis[:, 6]
D5 = Dis[:, 8]

I1 = Disi[:, 0].astype(np.int64)
I2 = Disi[:, 4].astype(np.int64)
I3 = Disi[:, 5].astype(np.int64)
I4 = Disi[:, 6].astype(np.int64)
I5 = Disi[:, 8].astype(np.int64)

lenz = 1000
for i in range(lenz): #looping over each individual redMaPPer cluster
    print(i)
    '''
    id1 = np.where(I1 == i)[0]
    id2 = np.where(I2 == i)[0]
    id3 = np.where(I3 == i)[0]
    id4 = np.where(I4 == i)[0]
    
    d1 = D1
    d2 = D2
    d3 = D3
    d4 = D4
    d5 = D5
    
    d1[id1] = D2[id1]
    d2[id2] = D3[id2]
    d3[id3] = D4[id3]
    d4[id4] = D5[id4]
    
    dj = np.vstack((d1, d2, d3, d4)).T
    np.savetxt(baseDir+'/jack_loo/dj_{}.txt'.format(i), dj)
    '''

    id1 = np.where(I1 != i)[0]
    id2 = np.where(I2 != i)[0]
    id3 = np.where(I3 != i)[0]
    id4 = np.where(I4 != i)[0]
    
    d1 = D1
    d2 = D2
    d3 = D3
    d4 = D4

    np.savetxt(baseDir+'/jack_loo/dj_1_{}.txt'.format(i), d1[id1])
    np.savetxt(baseDir+'/jack_loo/dj_2_{}.txt'.format(i), d2[id2])
    np.savetxt(baseDir+'/jack_loo/dj_3_{}.txt'.format(i), d3[id3])
    np.savetxt(baseDir+'/jack_loo/dj_4_{}.txt'.format(i), d4[id4])



