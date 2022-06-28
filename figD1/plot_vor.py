import kmeans_radec
from kmeans_radec import KMeans, kmeans_sample
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
from Corrfunc.theory.DD import DD
from Corrfunc.io import read_catalog
from Corrfunc.utils import convert_3d_counts_to_cf
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
from matplotlib.ticker import AutoMinorLocator, LogLocator
from astropy.cosmology import FlatLambdaCDM
import healpy as hp
cosmo = FlatLambdaCDM(H0=67.74, Om0=0.3089)
h = 0.6774
baseDir = '/oak/stanford/orgs/kipac/users/ycwang19/VPF/SZ/redMapper'

######

zmask = fits.open(baseDir+'/redmapper_dr8_public_v6.3_zmask.fits')[1].data 
hpix = zmask['HPIX']
zMax = zmask['ZMAX']
f_good = zmask['FRACGOOD']
mask = np.where((f_good > 0.5) & (zMax > 0.3))[0]
Hpix = hpix[mask] # NSIDE 2048 heal pixels that are good area
RA, DEC = hp.pix2ang(2048, Hpix, lonlat = True)
print(RA, DEC )

######

vol_h = np.loadtxt(baseDir+'/Dis_r2D_z0p2.txt') # Reads in r kNN data
cdf1 = vol_h[:, 0]
cdf2 = vol_h[:, 4]
cdf3 = vol_h[:, 5]

rbar = 76.864

######
#1NN
ax1 = fig.add_subplot(1, 1, 1, projection = 'hammer') #Hammer in radians
cax = fig.add_axes([0.92, 0.25, 0.02, 0.5])
cax.tick_params(labelsize = 10, direction = 'in')

im1 = ax1.scatter(RA/180*np.pi-np.pi, DEC/180*np.pi, 
                  c = cdf1, marker = 'o', alpha = 0.8, s = 0.8, lw = 0., 
                  rasterized = True, cmap = 'PuOr', label = r'$\mathrm{1NN}$')

'''
im1 = ax1.scatter(RA/180*np.pi-np.pi, DEC/180*np.pi, 
                  c = cdf2, marker = 'o', alpha = 0.8, s = 0.8, lw = 0., 
                  rasterized = True, cmap = 'PuOr', label = r'$\mathrm{2NN}$')

im1 = ax1.scatter(RA/180*np.pi-np.pi, DEC/180*np.pi, 
                  c = cdf2-cdf1, marker = 'o', alpha = 0.8, s = 0.8, lw = 0., 
                  rasterized = True, cmap = 'PuOr', label = r'$\mathrm{2NN-1NN}$')

im1 = ax1.scatter(RA/180*np.pi-np.pi, DEC/180*np.pi, 
                  c = cdf3-cdf2, marker = 'o', alpha = 0.8, s = 0.8, lw = 0., 
                  rasterized = True, cmap = 'PuOr', label = r'$\mathrm{3NN-2NN}$')
'''
cbar = fig.colorbar(im1, cax=cax, orientation='vertical')
cbar.set_label(r'$r\,$[$\mathrm{Mpc}$]', fontsize = 12, rotation = 270, labelpad = 16)

ax1.grid('True')
ax1.tick_params(labelsize = 10)
ax1.legend(loc = 'lower center', fontsize = 12)



fig.savefig(baseDir+'/../../figures/vor_1.png', dpi = 400, bbox_inches = 'tight')
#plt.show()
