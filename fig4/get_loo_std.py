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
#matplotlib.rcParams['grid.color'] = 'grey'
#matplotlib.rcParams['grid.linestyle'] = '--'
#matplotlib.rcParams['grid.linewidth'] = 0.4
#matplotlib.rcParams['grid.alpha'] = 0.5
fig = plt.figure()
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from astropy.coordinates import SkyCoord
from matplotlib.ticker import AutoMinorLocator, LogLocator
cosmo = FlatLambdaCDM(H0=67.74, Om0=0.3089)
h = 0.6774
baseDir = '/oak/stanford/orgs/kipac/users/ycwang19/VPF/SZ/redMapper'


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

def cdfVolkNN(vol): # CDF
    N = len(vol)
    gof = ((np.arange(0, N) + 1) / N)
    ind = np.argsort(vol)
    sVol= vol[ind]
    cdf = interpolate.interp1d(sVol, gof, kind = 'linear', bounds_error=False)
    return cdf

lenz = 1000
for i in range(lenz):
    print(i)
    dj1 = np.loadtxt(baseDir+'/jack_loo/dj_1_{}.txt'.format(i))
    dj2 = np.loadtxt(baseDir+'/jack_loo/dj_2_{}.txt'.format(i))
    cdf_j1 = cdfVolkNN(dj1)
    cdf_j2 = cdfVolkNN(dj2)
    cdf1 = cdf_j1(binc1)
    cdf2 = cdf_j2(binc2)

    if i == 0:
        CDF1 = [cdf1]
        CDF2 = [cdf2]
    else:
        CDF1 = np.concatenate((CDF1, [cdf1]), axis = 0)
        CDF2 = np.concatenate((CDF2, [cdf2]), axis = 0)
        
std1 = []
std2 = []
for j in range(len(binc1)):
    std1.append(np.sqrt(lenz-1) * np.std(CDF1[:, j]))
    std2.append(np.sqrt(lenz-1) * np.std(CDF2[:, j]))
    
std1 = np.asarray(std1)
std2 = np.asarray(std2)
np.savetxt(baseDir+'/jack_12/std1_loo.txt', std1)
np.savetxt(baseDir+'/jack_12/std2_loo.txt', std2)