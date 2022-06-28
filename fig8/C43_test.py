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

def VolumekNN(xin, xout, k=1, periodic = 0):
    if isinstance(k, int): k = [k] # 
    dim = xin.shape[1] # dimension of every row
    #Ntot = xin.shape[0] # dimension of entries (length of column)
    xtree = scipy.spatial.cKDTree(xin, boxsize=periodic)
    #print('k = ', k)
    dis, disi = xtree.query(xout, k=k, n_jobs=8) # dis is the distance to the kth nearest neighbour, disi is the id of that neighbour
    vol = np.empty_like(dis) # same shape as distance including all k values
    Cr = [2, np.pi, 4 * np.pi / 3, np.pi**2, 8*np.pi**2/15][dim - 1]  # Volume prefactor for 1,2, 3D
    for c, k in enumerate(np.nditer(np.array(k))):
        #print('c, dim, dis = ', c, dim, dis[:, c]**dim / k)
        vol[:, c] = Cr * dis[:, c]**dim / k # the overdense average volume per point in sphere
        #print('vol = ', vol[:, c])
    return vol

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


################################

bine = np.logspace(np.log10(30), np.log10(220), 51)
binw = bine[1:] - bine[:-1]
binc = (bine[1:] + bine[:-1]) / 2

vol_h = np.loadtxt(baseDir+'/Dis_h_3D_Y500.txt') # Reads in r kNN data
CDFs_h = CDFVolkNN(vol_h)
CDF_1h = interpolate.interp1d(binc, CDFs_h[0](binc), kind='linear', bounds_error=False, fill_value=(0.,1.))
CDF_2h = interpolate.interp1d(binc, CDFs_h[4](binc), kind='linear', bounds_error=False, fill_value=(0.,1.))
CDF_3h = interpolate.interp1d(binc, CDFs_h[5](binc), kind='linear', bounds_error=False, fill_value=(0.,1.))
CDF_4h = interpolate.interp1d(binc, CDFs_h[6](binc), kind='linear', bounds_error=False, fill_value=(0.,1.))

Yd1 = CDF_1h(binc)
Yd2 = CDF_2h(binc)
Yd3 = CDF_3h(binc)
Yd4 = CDF_4h(binc)

print(Yd1)
print(Yd2)
print(Yd3)
print(Yd4)

def CDF_2nn(CDF_1NN):
    c2 = CDF_1NN + (1-CDF_1NN) * np.log(1-CDF_1NN)
    return c2

def CDF_3NN(CDF_1NN, CDF_2NN):
    c3 = CDF_2NN + ( (1-CDF_1NN)*np.log(1-CDF_1NN) + (CDF_1NN - CDF_2NN) - 
                    1/2*(CDF_1NN - CDF_2NN)**2/(1-CDF_1NN) )
    return c3

def CDF_4NN(CDF_1NN, CDF_2NN, CDF_3NN):
    c4 = CDF_3NN + (CDF_1NN - CDF_2NN)/(1 - CDF_1NN) * ( (1-CDF_1NN)*np.log(1-CDF_1NN) + (CDF_1NN - CDF_2NN)
                                                        - 1/6 * (CDF_1NN-CDF_2NN)**2/(1-CDF_1NN))
    return c4

C2 = CDF_2nn(Yd1)
C3 = CDF_3NN(Yd1, Yd2)
C41 = CDF_4NN(Yd1, Yd2, C3)
C42 = CDF_4NN(Yd1, Yd2, Yd3)
print(C3)
print(C41)
print(C42)

def get_pCDF(cdf):
    id_rise = np.where(cdf <= 0.5)[0]
    id_drop = np.where(cdf > 0.5)[0]
    pcdf = np.concatenate((cdf[id_rise], 1-cdf[id_drop]))
    return pcdf

pCDF_1 = get_pCDF(Yd1)
pCDF_2 = get_pCDF(Yd2)
pCDF_3 = get_pCDF(Yd3)
pCDF_4 = get_pCDF(Yd4)
pCDF_C2 = get_pCDF(C2)
pCDF_C3 = get_pCDF(C3)
pCDF_C41 = get_pCDF(C41)
pCDF_C42 = get_pCDF(C42)


#3NN
bine = np.logspace(np.log10(40), np.log10(220), 26)
binw = bine[1:] - bine[:-1]
binc3 = (bine[1:] + bine[:-1]) / 2
#4NN
bine = np.logspace(np.log10(55), np.log10(220), 26)
binw = bine[1:] - bine[:-1]
binc4 = (bine[1:] + bine[:-1]) / 2

CDF34_mean = np.loadtxt('/oak/stanford/orgs/kipac/users/ycwang19/VPF/SZ/redMapper/C43/CDF34_mean.txt')
c3 = CDF34_mean[:25]
c4 = CDF34_mean[25:]
p3 = get_pCDF(c3)
p4 = get_pCDF(c4)
'''
CDF_RM = np.loadtxt('/oak/stanford/orgs/kipac/users/ycwang19/VPF/SZ/redMapper/C43/CDF_RM_C43.txt')
c3 = CDF_RM[:25]
c4 = CDF_RM[25:]
p3 = get_pCDF(c3)
p4 = get_pCDF(c4)
'''

def cdfVolkNN(vol): # CDF
    N = len(vol)
    gof = ((np.arange(0, N) + 1) / N)
    ind = np.argsort(vol)
    sVol= vol[ind]
    cdf = interpolate.interp1d(sVol, gof, kind = 'linear', bounds_error=False)
    return cdf

################################
jack = 1000
for i in range(jack):
    print(i)
    dj1 = np.loadtxt(baseDir+'/jack_loo/dj_1_{}.txt'.format(i))
    dj2 = np.loadtxt(baseDir+'/jack_loo/dj_2_{}.txt'.format(i))
    dj3 = np.loadtxt(baseDir+'/jack_loo/dj_3_{}.txt'.format(i))
    dj4 = np.loadtxt(baseDir+'/jack_loo/dj_4_{}.txt'.format(i))
    cdf_j1 = cdfVolkNN(dj1)
    cdf_j2 = cdfVolkNN(dj2)
    cdf_j3 = cdfVolkNN(dj3)
    cdf_j4 = cdfVolkNN(dj4)

    cdf1 = cdf_j1(binc)
    cdf2 = cdf_j2(binc)
    cdf3 = cdf_j3(binc)
    cdf4 = cdf_j4(binc)
    
    np.savetxt(baseDir+'/jack_loo/jack_{}_1NN.txt'.format(i), cdf1)
    np.savetxt(baseDir+'/jack_loo/jack_{}_2NN.txt'.format(i), cdf2)
    np.savetxt(baseDir+'/jack_loo/jack_{}_3NN.txt'.format(i), cdf3)
    np.savetxt(baseDir+'/jack_loo/jack_{}_4NN.txt'.format(i), cdf4)
    
    if i == 0:
        CDF1 = [cdf1]
        CDF2 = [cdf2]
        CDF3 = [cdf3]
        CDF4 = [cdf4]
    else:
        CDF1 = np.concatenate((CDF1, [cdf1]), axis = 0)
        CDF2 = np.concatenate((CDF2, [cdf2]), axis = 0)
        CDF3 = np.concatenate((CDF3, [cdf3]), axis = 0)
        CDF4 = np.concatenate((CDF4, [cdf4]), axis = 0)


std1 = []
std2 = []
std3 = []
std4 = []
for j in range(len(binc)):
    std1.append(np.sqrt(jack-1) * np.std(CDF1[:, j]))
    std2.append(np.sqrt(jack-1) * np.std(CDF2[:, j]))
    std3.append(np.sqrt(jack-1) * np.std(CDF3[:, j]))
    std4.append(np.sqrt(jack-1) * np.std(CDF4[:, j]))

std1 = np.asarray(std1)
std2 = np.asarray(std2)
std3 = np.asarray(std3)
std4 = np.asarray(std4)
np.savetxt(baseDir+'/jack_std1_{}.txt'.format(jack), std1)
np.savetxt(baseDir+'/jack_std2_{}.txt'.format(jack), std2)
np.savetxt(baseDir+'/jack_std3_{}.txt'.format(jack), std3)
np.savetxt(baseDir+'/jack_std4_{}.txt'.format(jack), std4)
