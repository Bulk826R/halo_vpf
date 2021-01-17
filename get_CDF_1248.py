from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
import scipy.spatial
from sklearn.neighbors import KDTree, BallTree
from scipy.stats import poisson, erlang
from scipy import interpolate
from os import urandom
import struct
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['text.color'] = 'black'
matplotlib.rcParams['grid.color'] = 'grey'
matplotlib.rcParams['grid.linestyle'] = '--'
matplotlib.rcParams['grid.linewidth'] = 0.4
matplotlib.rcParams['grid.alpha'] = 0.5
fig = plt.figure()

from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from astropy.coordinates import SkyCoord
cosmo = FlatLambdaCDM(H0=67.74, Om0=0.3089)
h = 0.6774
baseDir = '/oak/stanford/orgs/kipac/users/ycwang19/VPF/SZ'

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

bine = np.logspace(1.5, 3.0, 51)
binw = bine[1:] - bine[:-1]
binc1 = (bine[1:] + bine[:-1]) / 2

bine = np.logspace(1.8, 3.05, 51)
binw = bine[1:] - bine[:-1]
binc2 = (bine[1:] + bine[:-1]) / 2

bine = np.logspace(1.95, 3.1, 51)
binw = bine[1:] - bine[:-1]
binc4 = (bine[1:] + bine[:-1]) / 2

bine = np.logspace(2.05, 3.15, 51)
binw = bine[1:] - bine[:-1]
binc8 = (bine[1:] + bine[:-1]) / 2

vol_h = np.loadtxt(baseDir+'/Dis_h_3D_Y500.txt') # Reads in r kNN data
#vol_h = np.loadtxt(baseDir+'/Vol_h_3D_Y500c.txt') # Reads in V kNN data
CDFs_h = CDFVolkNN(vol_h)
dummyvpf_h = 1.-CDFs_h[0](binc1)
dummyvpf_h[np.isnan(dummyvpf_h)] = 0
VPF_h = interpolate.interp1d(binc1, dummyvpf_h, kind='linear', bounds_error=False, fill_value=(1.,0.))
CDF_1h = interpolate.interp1d(binc1, CDFs_h[0](binc1), kind='linear', bounds_error=False, fill_value=(0.,1.))
CDF_5h = interpolate.interp1d(binc4, CDFs_h[1](binc4), kind='linear', bounds_error=False, fill_value=(0.,1.))
CDF_10h = interpolate.interp1d(binc8, CDFs_h[2](binc8), kind='linear', bounds_error=False, fill_value=(0.,1.))
CDF_50h = interpolate.interp1d(binc8, CDFs_h[3](binc8), kind='linear', bounds_error=False, fill_value=(0.,1.))
CDF_2h = interpolate.interp1d(binc2, CDFs_h[4](binc2), kind='linear', bounds_error=False, fill_value=(0.,1.))
CDF_3h = interpolate.interp1d(binc2, CDFs_h[5](binc2), kind='linear', bounds_error=False, fill_value=(0.,1.))
CDF_4h = interpolate.interp1d(binc4, CDFs_h[6](binc4), kind='linear', bounds_error=False, fill_value=(0.,1.))
CDF_8h = interpolate.interp1d(binc8, CDFs_h[7](binc8), kind='linear', bounds_error=False, fill_value=(0.,1.))
CDF_5h = interpolate.interp1d(binc4, CDFs_h[8](binc4), kind='linear', bounds_error=False, fill_value=(0.,1.))
CDF_9h = interpolate.interp1d(binc8, CDFs_h[9](binc8), kind='linear', bounds_error=False, fill_value=(0.,1.))
'''
# Section for finding the regions of different kNN CDFs within the range of [0.001 and 0.999]

cdf1 = CDF_1h(binc1)
id1 = np.where((cdf1 >= 1e-3) & (cdf1 <= 1-1e-3))[0]
print('1NN range = ' ,np.log10(np.min(binc1[id1])), np.log10(np.max(binc1[id1])))

cdf2 = CDF_2h(binc1)
id2 = np.where((cdf2 >= 1e-3) & (cdf2 <= 1-1e-3))[0]
print('2NN range = ' ,np.log10(np.min(binc1[id2])), np.log10(np.max(binc1[id2])))

cdf3 = CDF_3h(binc1)
id3 = np.where((cdf3 >= 1e-3) & (cdf3 <= 1-1e-3))[0]
print('3NN range = ' ,np.log10(np.min(binc1[id3])), np.log10(np.max(binc1[id3])))

cdf4 = CDF_4h(binc1)
id4 = np.where((cdf4 >= 1e-3) & (cdf4 <= 1-1e-3))[0]
print('4NN range = ' ,np.log10(np.min(binc1[id4])), np.log10(np.max(binc1[id4])))

cdf5 = CDF_5h(binc1)
id5 = np.where((cdf5 >= 1e-3) & (cdf5 <= 1-1e-3))[0]
print('5NN range = ' ,np.log10(np.min(binc1[id5])), np.log10(np.max(binc1[id5])))

cdf8 = CDF_8h(binc1)
id8 = np.where((cdf8 >= 1e-3) & (cdf8 <= 1-1e-3))[0]
print('8NN range = ' ,np.log10(np.min(binc1[id8])), np.log10(np.max(binc1[id8])))

cdf9 = CDF_9h(binc1)
id9 = np.where((cdf9 >= 1e-3) & (cdf9 <= 1-1e-3))[0]
print('9NN range = ' ,np.log10(np.min(binc1[id9])), np.log10(np.max(binc1[id9])))
'''
Yd1 = CDF_1h(binc1)
Yd2 = CDF_2h(binc2)
Yd4 = CDF_4h(binc4)
Yd8 = CDF_8h(binc8)
#print('Yd = ', Yd)

vol_g = np.load(baseDir+'/Poi/Dis_g_Y500_0.npy') # Reads in r kNN data
#vol_g = np.load(baseDir+'/Poi/Vol_g_Y500c_0.npy') # Reads in V kNN data
CDFs_g = CDFVolkNN(vol_g)
dummyvpf_g = 1.-CDFs_g[0](binc1)
dummyvpf_g[np.isnan(dummyvpf_g)] = 0
VPF_g = interpolate.interp1d(binc1, dummyvpf_g, kind='linear', bounds_error=False, fill_value=(1., 0.))
CDF_1g = interpolate.interp1d(binc1, CDFs_g[0](binc1), kind='linear', bounds_error=False, fill_value=(0.,1.))
CDF_5g = interpolate.interp1d(binc4, CDFs_g[1](binc4), kind='linear', bounds_error=False, fill_value=(0.,1.))
CDF_10g = interpolate.interp1d(binc8, CDFs_g[2](binc8), kind='linear', bounds_error=False, fill_value=(0.,1.))
CDF_50g = interpolate.interp1d(binc8, CDFs_g[3](binc8), kind='linear', bounds_error=False, fill_value=(0.,1.))
CDF_2g = interpolate.interp1d(binc2, CDFs_g[4](binc2), kind='linear', bounds_error=False, fill_value=(0.,1.))
CDF_3g = interpolate.interp1d(binc2, CDFs_g[5](binc2), kind='linear', bounds_error=False, fill_value=(0.,1.))
CDF_4g = interpolate.interp1d(binc4, CDFs_g[6](binc4), kind='linear', bounds_error=False, fill_value=(0.,1.))
CDF_8g = interpolate.interp1d(binc8, CDFs_g[7](binc8), kind='linear', bounds_error=False, fill_value=(0.,1.))
CDF_5g = interpolate.interp1d(binc4, CDFs_g[8](binc4), kind='linear', bounds_error=False, fill_value=(0.,1.))
CDF_9g = interpolate.interp1d(binc8, CDFs_g[9](binc8), kind='linear', bounds_error=False, fill_value=(0.,1.))


Yp1 = [CDF_1g(binc1)]
Yp2 = [CDF_2g(binc2)]
Yp4 = [CDF_4g(binc4)]
Yp8 = [CDF_8g(binc8)]
print(Yp1)
print(Yp2)
print(Yp4)
print(Yp8)
len_poi = 2000
for i in range(1, len_poi):
    print('i = ', i)
    vol_g = np.load(baseDir+'/Poi/Dis_g_Y500_{}.npy'.format(i))
    #vol_g = np.load(baseDir+'/Poi/Vol_g_Y500c_{}.npy'.format(i))
    CDFs_g = CDFVolkNN(vol_g)
    dummyvpf_g = 1.-CDFs_g[0](binc1)
    dummyvpf_g[np.isnan(dummyvpf_g)] = 0
    VPF_g = interpolate.interp1d(binc1, dummyvpf_g, kind='linear', bounds_error=False, fill_value=(1., 0.))
    CDF_1g = interpolate.interp1d(binc1, CDFs_g[0](binc1), kind='linear', bounds_error=False, fill_value=(0.,1.))
    CDF_5g = interpolate.interp1d(binc4, CDFs_g[1](binc4), kind='linear', bounds_error=False, fill_value=(0.,1.))
    CDF_10g = interpolate.interp1d(binc8, CDFs_g[2](binc8), kind='linear', bounds_error=False, fill_value=(0.,1.))
    CDF_50g = interpolate.interp1d(binc8, CDFs_g[3](binc8), kind='linear', bounds_error=False, fill_value=(0.,1.))
    CDF_2g = interpolate.interp1d(binc2, CDFs_g[4](binc2), kind='linear', bounds_error=False, fill_value=(0.,1.))
    CDF_3g = interpolate.interp1d(binc2, CDFs_g[5](binc2), kind='linear', bounds_error=False, fill_value=(0.,1.))
    CDF_4g = interpolate.interp1d(binc4, CDFs_g[6](binc4), kind='linear', bounds_error=False, fill_value=(0.,1.))
    CDF_8g = interpolate.interp1d(binc8, CDFs_g[7](binc8), kind='linear', bounds_error=False, fill_value=(0.,1.))
    CDF_5g = interpolate.interp1d(binc4, CDFs_g[8](binc4), kind='linear', bounds_error=False, fill_value=(0.,1.))
    CDF_9g = interpolate.interp1d(binc8, CDFs_g[9](binc8), kind='linear', bounds_error=False, fill_value=(0.,1.))

    yp1 = CDF_1g(binc1)
    yp2 = CDF_2g(binc2)
    yp4 = CDF_4g(binc4)
    yp8 = CDF_8g(binc8)
    Yp1 = np.concatenate((Yp1, [yp1]), axis = 0)
    Yp2 = np.concatenate((Yp2, [yp2]), axis = 0)
    Yp4 = np.concatenate((Yp4, [yp4]), axis = 0)
    Yp8 = np.concatenate((Yp8, [yp8]), axis = 0)

Mp1 = []
Mp2 = []
Mp4 = []
Mp8 = []
STD1 = []
STD2 = []
STD4 = []
STD8 = []
for i in range(len(binc1)):
    y1 = Yp1[:, i]
    idy1 = np.where(np.isfinite(y1) == 1)[0]
    Mp1.append(np.mean(y1[idy1]))
    STD1.append(np.std(y1[idy1]))

for i in range(len(binc2)):   
    y2 = Yp2[:, i]
    idy2 = np.where(np.isfinite(y2) == 1)[0]
    Mp2.append(np.mean(y2[idy2]))
    STD2.append(np.std(y2[idy2]))

for i in range(len(binc4)):
    y4 = Yp4[:, i]
    idy4 = np.where(np.isfinite(y4) == 1)[0]
    Mp4.append(np.mean(y4[idy4]))
    STD4.append(np.std(y4[idy4]))
    
for i in range(len(binc8)):
    y8 = Yp8[:, i]
    idy8 = np.where(np.isfinite(y8) == 1)[0]
    Mp8.append(np.mean(y8[idy8]))
    STD8.append(np.std(y8[idy8]))

Mp1 = np.asarray(Mp1)
STD1 = np.asarray(STD1)
id_fin1 = np.where(np.isfinite(Mp1) == 1)[0]
print('Fin 1 = ', len(id_fin1), len(Mp1))
Mp2 = np.asarray(Mp2)
STD2 = np.asarray(STD2)
id_fin2 = np.where(np.isfinite(Mp2) == 1)[0]
print('Fin 2 = ', len(id_fin2), len(Mp2))
Mp4 = np.asarray(Mp4)
STD4 = np.asarray(STD4)
id_fin4 = np.where(np.isfinite(Mp4) == 1)[0]
print('Fin 4 = ', len(id_fin4), len(Mp4))
Mp8 = np.asarray(Mp8)
STD8 = np.asarray(STD8)
id_fin8 = np.where(np.isfinite(Mp8) == 1)[0]
print('Fin 8 = ', len(id_fin8), len(Mp8))

##########################################
# Saves CDF results in the format of peaked CDFs

IDh_rise = np.where(Yd1[id_fin1] <= 0.5)[0]
IDh_drop = np.where(Yd1[id_fin1] > 0.5)[0]
x_up1 = binc1[id_fin1]
y_up1 = np.concatenate((Yd1[id_fin1][IDh_rise], 1-Yd1[id_fin1][IDh_drop]))
np.savetxt(baseDir+'/CDF/x_up_1NN.txt', x_up1)
np.savetxt(baseDir+'/CDF/y_up_1NN.txt', y_up1)
IDg_rise = np.where(Mp1[id_fin1] <= 0.5)[0]
IDg_drop = np.where(Mp1[id_fin1] > 0.5)[0]
X_up1 = binc1[id_fin1]
Y_up1 = np.concatenate((Mp1[id_fin1][IDg_rise], 1-Mp1[id_fin1][IDg_drop]))
S_up1 = STD1[id_fin1]
np.savetxt(baseDir+'/CDF/X_up_1NN.txt', X_up1)
np.savetxt(baseDir+'/CDF/Y_up_1NN.txt', Y_up1)
np.savetxt(baseDir+'/CDF/S_up_1NN.txt', S_up1)
rD1 = np.abs(Yd1 - Mp1) / STD1
x_down1 = binc1[id_fin1]
y_down1 = rD1[id_fin1]
np.savetxt(baseDir+'/CDF/x_down_1NN.txt', x_down1)
np.savetxt(baseDir+'/CDF/y_down_1NN.txt', y_down1)

IDh_rise = np.where(Yd2[id_fin2] <= 0.5)[0]
IDh_drop = np.where(Yd2[id_fin2] > 0.5)[0]
x_up2 = binc2[id_fin2]
y_up2 = np.concatenate((Yd2[id_fin2][IDh_rise], 1-Yd2[id_fin2][IDh_drop]))
np.savetxt(baseDir+'/CDF/x_up_2NN.txt', x_up2)
np.savetxt(baseDir+'/CDF/y_up_2NN.txt', y_up2)
IDg_rise = np.where(Mp2[id_fin2] <= 0.5)[0]
IDg_drop = np.where(Mp2[id_fin2] > 0.5)[0]
X_up2 = binc2[id_fin2]
Y_up2 = np.concatenate((Mp2[id_fin2][IDg_rise], 1-Mp2[id_fin2][IDg_drop]))
S_up2 = STD2[id_fin2]
np.savetxt(baseDir+'/CDF/X_up_2NN.txt', X_up2)
np.savetxt(baseDir+'/CDF/Y_up_2NN.txt', Y_up2)
np.savetxt(baseDir+'/CDF/S_up_2NN.txt', S_up2)
rD2 = np.abs(Yd2 - Mp2) / STD2
x_down2 = binc2[id_fin2]
y_down2 = rD2[id_fin2]
np.savetxt(baseDir+'/CDF/x_down_2NN.txt', x_down2)
np.savetxt(baseDir+'/CDF/y_down_2NN.txt', y_down2)

IDh_rise = np.where(Yd4[id_fin4] <= 0.5)[0]
IDh_drop = np.where(Yd4[id_fin4] > 0.5)[0]
x_up4 = binc4[id_fin4]
y_up4 = np.concatenate((Yd4[id_fin4][IDh_rise], 1-Yd4[id_fin4][IDh_drop]))
np.savetxt(baseDir+'/CDF/x_up_4NN.txt', x_up4)
np.savetxt(baseDir+'/CDF/y_up_4NN.txt', y_up4)
IDg_rise = np.where(Mp4[id_fin4] <= 0.5)[0]
IDg_drop = np.where(Mp4[id_fin4] > 0.5)[0]
X_up4 = binc4[id_fin4]
Y_up4 = np.concatenate((Mp4[id_fin4][IDg_rise], 1-Mp4[id_fin4][IDg_drop]))
S_up4 = STD4[id_fin4]
np.savetxt(baseDir+'/CDF/X_up_4NN.txt', X_up4)
np.savetxt(baseDir+'/CDF/Y_up_4NN.txt', Y_up4)
np.savetxt(baseDir+'/CDF/S_up_4NN.txt', S_up4)
rD4 = np.abs(Yd4 - Mp4) / STD4
x_down4 = binc4[id_fin4]
y_down4 = rD4[id_fin4]
np.savetxt(baseDir+'/CDF/x_down_4NN.txt', x_down4)
np.savetxt(baseDir+'/CDF/y_down_4NN.txt', y_down4)

IDh_rise = np.where(Yd8[id_fin8] <= 0.5)[0]
IDh_drop = np.where(Yd8[id_fin8] > 0.5)[0]
x_up8 = binc8[id_fin8]
y_up8 = np.concatenate((Yd8[id_fin8][IDh_rise], 1-Yd8[id_fin8][IDh_drop]))
np.savetxt(baseDir+'/CDF/x_up_8NN.txt', x_up8)
np.savetxt(baseDir+'/CDF/y_up_8NN.txt', y_up8)
IDg_rise = np.where(Mp8[id_fin8] <= 0.5)[0]
IDg_drop = np.where(Mp8[id_fin8] > 0.5)[0]
X_up8 = binc8[id_fin8]
Y_up8 = np.concatenate((Mp8[id_fin8][IDg_rise], 1-Mp8[id_fin8][IDg_drop]))
S_up8 = STD8[id_fin8]
np.savetxt(baseDir+'/CDF/X_up_8NN.txt', X_up8)
np.savetxt(baseDir+'/CDF/Y_up_8NN.txt', Y_up8)
np.savetxt(baseDir+'/CDF/S_up_8NN.txt', S_up8)
rD8 = np.abs(Yd8 - Mp8) / STD8
x_down8 = binc8[id_fin8]
y_down8 = rD8[id_fin8]
np.savetxt(baseDir+'/CDF/x_down_8NN.txt', x_down8)
np.savetxt(baseDir+'/CDF/y_down_8NN.txt', y_down8)

