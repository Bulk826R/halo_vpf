from astropy.io import fits
import numpy as np
#import matplotlib.pyplot as plt
#from mpl_toolkits.axes_grid1 import make_axes_locatable
#import matplotlib
import scipy.spatial
from sklearn.neighbors import KDTree, BallTree
from scipy import interpolate
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

####################################
#30 140
bine = np.logspace(np.log10(35), np.log10(135), 26)
binw = bine[1:] - bine[:-1]
binc1 = (bine[1:] + bine[:-1]) / 2
#50 160
bine = np.logspace(np.log10(55), np.log10(155), 26)
binw = bine[1:] - bine[:-1]
binc2 = (bine[1:] + bine[:-1]) / 2

vol_h = np.loadtxt(baseDir+'/Dis_h_3D_Y500.txt') # Reads in redMaPPers kNN data
CDFs_h = CDFVolkNN(vol_h)
CDF_1h = interpolate.interp1d(binc1, CDFs_h[0](binc1), kind='linear', bounds_error=False, fill_value=(0.,1.))
CDF_2h = interpolate.interp1d(binc2, CDFs_h[4](binc2), kind='linear', bounds_error=False, fill_value=(0.,1.))
Yd1 = CDF_1h(binc1)
Yd2 = CDF_2h(binc2)
print(Yd1)
print(Yd2)

vol_g = np.load(baseDir+'/Poi/Dis_g_Y500_0.npy') # Reads in random data
CDFs_g = CDFVolkNN(vol_g)
CDF_1g = interpolate.interp1d(binc1, CDFs_g[0](binc1), kind='linear', bounds_error=False, fill_value=(0.,1.))
CDF_2g = interpolate.interp1d(binc2, CDFs_g[4](binc2), kind='linear', bounds_error=False, fill_value=(0.,1.))
Yp1 = [CDF_1g(binc1)]
Yp2 = [CDF_2g(binc2)]
print(Yp1)
print(Yp2)
len_poi = 2000
for i in range(1, len_poi):
    print('i = ', i)
    vol_g = np.load(baseDir+'/Poi/Dis_g_Y500_{}.npy'.format(i))
    CDFs_g = CDFVolkNN(vol_g)
    CDF_1g = interpolate.interp1d(binc1, CDFs_g[0](binc1), kind='linear', bounds_error=False, fill_value=(0.,1.))
    CDF_2g = interpolate.interp1d(binc2, CDFs_g[4](binc2), kind='linear', bounds_error=False, fill_value=(0.,1.))
    yp1 = CDF_1g(binc1)
    yp2 = CDF_2g(binc2)
    Yp1 = np.concatenate((Yp1, [yp1]), axis = 0)
    Yp2 = np.concatenate((Yp2, [yp2]), axis = 0)

Mp1 = []
Mp2 = []

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


Mp1 = np.asarray(Mp1)
STD1 = np.asarray(STD1)
id_fin1 = np.where(np.isfinite(Mp1) == 1)[0]
print('Fin 1 = ', len(id_fin1), len(Mp1))
Mp2 = np.asarray(Mp2)
STD2 = np.asarray(STD2)
id_fin2 = np.where(np.isfinite(Mp2) == 1)[0]
print('Fin 2 = ', len(id_fin2), len(Mp2))
##########################################
# Saves CDF results in the format of peaked CDFs

np.savetxt(baseDir+'/CDF_12/CDF_red_1NN.txt', Yd1)
np.savetxt(baseDir+'/CDF_12/CDF_red_2NN.txt', Yd2)

IDh_rise = np.where(Yd1[id_fin1] <= 0.5)[0]
IDh_drop = np.where(Yd1[id_fin1] > 0.5)[0]
x_up1 = binc1[id_fin1]
print(Yd2[id_fin2], Yd2)
y_up1 = np.concatenate((Yd1[id_fin1][IDh_rise], 1-Yd1[id_fin1][IDh_drop]))
np.savetxt(baseDir+'/CDF_12/x_up_1NN.txt', x_up1)
np.savetxt(baseDir+'/CDF_12/y_up_1NN.txt', y_up1)
print(len(x_up1), len(y_up1))

IDh_rise = np.where(Yd2[id_fin2] <= 0.5)[0]
IDh_drop = np.where(Yd2[id_fin2] > 0.5)[0]
x_up2 = binc2[id_fin2]
y_up2 = np.concatenate((Yd2[id_fin2][IDh_rise], 1-Yd2[id_fin2][IDh_drop]))
np.savetxt(baseDir+'/CDF_12/x_up_2NN.txt', x_up2)
np.savetxt(baseDir+'/CDF_12/y_up_2NN.txt', y_up2)

################

IDg_rise = np.where(Mp1[id_fin1] <= 0.5)[0]
IDg_drop = np.where(Mp1[id_fin1] > 0.5)[0]
X_up1 = binc1[id_fin1]
Y_up1 = np.concatenate((Mp1[id_fin1][IDg_rise], 1-Mp1[id_fin1][IDg_drop]))
S_up1 = STD1[id_fin1]
np.savetxt(baseDir+'/CDF_12/X_up_1NN.txt', X_up1)
np.savetxt(baseDir+'/CDF_12/Y_up_1NN.txt', Y_up1)
np.savetxt(baseDir+'/CDF_12/S_up_1NN.txt', S_up1)
rD1 = np.abs(Yd1 - Mp1) / STD1
x_down1 = binc1[id_fin1]
y_down1 = rD1[id_fin1]
np.savetxt(baseDir+'/CDF_12/x_down_1NN.txt', x_down1)
np.savetxt(baseDir+'/CDF_12/y_down_1NN.txt', y_down1)

IDg_rise = np.where(Mp2[id_fin2] <= 0.5)[0]
IDg_drop = np.where(Mp2[id_fin2] > 0.5)[0]
X_up2 = binc2[id_fin2]
Y_up2 = np.concatenate((Mp2[id_fin2][IDg_rise], 1-Mp2[id_fin2][IDg_drop]))
S_up2 = STD2[id_fin2]
np.savetxt(baseDir+'/CDF_12/X_up_2NN.txt', X_up2)
np.savetxt(baseDir+'/CDF_12/Y_up_2NN.txt', Y_up2)
np.savetxt(baseDir+'/CDF_12/S_up_2NN.txt', S_up2)
rD2 = np.abs(Yd2 - Mp2) / STD2
x_down2 = binc2[id_fin2]
y_down2 = rD2[id_fin2]
np.savetxt(baseDir+'/CDF_12/x_down_2NN.txt', x_down2)
np.savetxt(baseDir+'/CDF_12/y_down_2NN.txt', y_down2)



