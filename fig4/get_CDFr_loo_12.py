from astropy.io import fits
import numpy as np
#import matplotlib.pyplot as plt
#from mpl_toolkits.axes_grid1 import make_axes_locatable
#import matplotlib
import scipy.spatial
from sklearn.neighbors import KDTree, BallTree
from scipy.stats import poisson, erlang
from scipy import interpolate
from os import urandom
import struct
'''
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['text.color'] = 'black'
matplotlib.rcParams['grid.color'] = 'grey'
matplotlib.rcParams['grid.linestyle'] = '--'
matplotlib.rcParams['grid.linewidth'] = 0.4
matplotlib.rcParams['grid.alpha'] = 0.5
fig = plt.figure()
'''
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from astropy.coordinates import SkyCoord
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

#30 140
bine = np.logspace(np.log10(35), np.log10(135), 26)
binw = bine[1:] - bine[:-1]
binc1 = (bine[1:] + bine[:-1]) / 2
#50 160
bine = np.logspace(np.log10(55), np.log10(155), 26)
binw = bine[1:] - bine[:-1]
binc2 = (bine[1:] + bine[:-1]) / 2

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


vol_h = np.loadtxt(baseDir+'/Dis_h_3D_Y500.txt') # Reads in r kNN data
#vol_h = np.loadtxt(baseDir+'/Vol_h_3D_Y500c.txt') # Reads in V kNN data
CDFs_h = CDFVolkNN(vol_h)
dummyvpf_h = 1.-CDFs_h[0](binc1)
dummyvpf_h[np.isnan(dummyvpf_h)] = 0
VPF_h = interpolate.interp1d(binc1, dummyvpf_h, kind='linear', bounds_error=False, fill_value=(1.,0.))
CDF_1h = interpolate.interp1d(binc1, CDFs_h[0](binc1), kind='linear', bounds_error=False, fill_value=(0.,1.))
CDF_2h = interpolate.interp1d(binc2, CDFs_h[4](binc2), kind='linear', bounds_error=False, fill_value=(0.,1.))
CDF_3h = interpolate.interp1d(binc2, CDFs_h[5](binc2), kind='linear', bounds_error=False, fill_value=(0.,1.))
Yd1 = CDF_1h(binc1)
Yd2 = CDF_2h(binc2)

vol_g = np.load(baseDir+'/Poi/Dis_g_Y500_0.npy') # Reads in r kNN data
CDFs_g = CDFVolkNN(vol_g)
dummyvpf_g = 1.-CDFs_g[0](binc1)
dummyvpf_g[np.isnan(dummyvpf_g)] = 0
VPF_g = interpolate.interp1d(binc1, dummyvpf_g, kind='linear', bounds_error=False, fill_value=(1., 0.))
CDF_1g = interpolate.interp1d(binc1, CDFs_g[0](binc1), kind='linear', bounds_error=False, fill_value=(0.,1.))
CDF_2g = interpolate.interp1d(binc2, CDFs_g[4](binc2), kind='linear', bounds_error=False, fill_value=(0.,1.))
CDF_3g = interpolate.interp1d(binc2, CDFs_g[5](binc2), kind='linear', bounds_error=False, fill_value=(0.,1.))

Yp1 = [CDF_1g(binc1)]
Yp2 = [CDF_2g(binc2)]
print(Yp1)
print(Yp2)
len_poi = 2000
for i in range(1, len_poi):
    print('i = ', i)
    vol_g = np.load(baseDir+'/Poi/Dis_g_Y500_{}.npy'.format(i))
    CDFs_g = CDFVolkNN(vol_g)
    dummyvpf_g = 1.-CDFs_g[0](binc1)
    dummyvpf_g[np.isnan(dummyvpf_g)] = 0
    VPF_g = interpolate.interp1d(binc1, dummyvpf_g, kind='linear', bounds_error=False, fill_value=(1., 0.))
    CDF_1g = interpolate.interp1d(binc1, CDFs_g[0](binc1), kind='linear', bounds_error=False, fill_value=(0.,1.))
    CDF_2g = interpolate.interp1d(binc2, CDFs_g[4](binc2), kind='linear', bounds_error=False, fill_value=(0.,1.))
    CDF_3g = interpolate.interp1d(binc2, CDFs_g[5](binc2), kind='linear', bounds_error=False, fill_value=(0.,1.))
    yp1 = CDF_1g(binc1)
    yp2 = CDF_2g(binc2)
    Yp1 = np.concatenate((Yp1, [yp1]), axis = 0)
    Yp2 = np.concatenate((Yp2, [yp2]), axis = 0)
    
Mp1 = []
Mp2 = []
Low1 = []
Low2 = []
High1 = []
High2 = []
for i in range(len(binc1)):
    y1 = Yp1[:, i]
    idy1 = np.where(np.isfinite(y1) == 1)[0]
    m1 = np.average(y1[idy1])
    std1 = np.std(y1[idy1])
    Mp1.append(m1)
    Low1.append(m1-std1)
    High1.append(m1+std1)
    '''
    Mp1.append(np.percentile(y1[idy1], 50))
    Low1.append(np.percentile(y1[idy1], 15.85))
    High1.append(np.percentile(y1[idy1], 84.15))
    '''

for i in range(len(binc2)):   
    y2 = Yp2[:, i]
    idy2 = np.where(np.isfinite(y2) == 1)[0]
    m2 = np.average(y2[idy2])
    std2 = np.std(y2[idy2])
    Mp2.append(m2)
    Low2.append(m2-std2)
    High2.append(m2+std2)
    '''
    Mp2.append(np.percentile(y2[idy2], 50))
    Low2.append(np.percentile(y2[idy2], 15.85))
    High2.append(np.percentile(y2[idy2], 84.15))
    '''

Mp1 = np.asarray(Mp1)
Low1 = np.asarray(Low1)
High1 = np.asarray(High1)
id_fin1 = np.where(np.isfinite(Mp1) == 1)[0]
print('Fin 1 = ', len(id_fin1), len(Mp1))
Mp2 = np.asarray(Mp2)
Low2 = np.asarray(Low2)
High2 = np.asarray(High2)
id_fin2 = np.where(np.isfinite(Mp2) == 1)[0]
print('Fin 2 = ', len(id_fin2), len(Mp2))

##########################################
# Saves CDF results in the format of peaked CDFs

x_up1 = binc1[id_fin1]
Med1 = Mp1[id_fin1] / Mp1[id_fin1]
Low1 = Low1[id_fin1] / Mp1[id_fin1]
High1 = High1[id_fin1] / Mp1[id_fin1]
R1 = Yd1[id_fin1] / Mp1[id_fin1]
np.savetxt(baseDir+'/CDF_12/Rand_med_1.txt', Mp1[id_fin1])
np.savetxt(baseDir+'/CDF_12/x_1.txt', x_up1)
np.savetxt(baseDir+'/CDF_12/Med1.txt', Med1)
np.savetxt(baseDir+'/CDF_12/Low1.txt', Low1)
np.savetxt(baseDir+'/CDF_12/High1.txt', High1)
np.savetxt(baseDir+'/CDF_12/R1.txt', R1)

x_up2 = binc2[id_fin2]
Med2 = Mp2[id_fin2] / Mp2[id_fin2]
Low2 = Low2[id_fin2] / Mp2[id_fin2]
High2 = High2[id_fin2] / Mp2[id_fin2]
R2 = Yd2[id_fin2] / Mp2[id_fin2]
np.savetxt(baseDir+'/CDF_12/Rand_med_2.txt', Mp2[id_fin2])
np.savetxt(baseDir+'/CDF_12/x_2.txt', x_up2)
np.savetxt(baseDir+'/CDF_12/Med2.txt', Med2)
np.savetxt(baseDir+'/CDF_12/Low2.txt', Low2)
np.savetxt(baseDir+'/CDF_12/High2.txt', High2)
np.savetxt(baseDir+'/CDF_12/R2.txt', R2)


