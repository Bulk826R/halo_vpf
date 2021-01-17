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
import matplotlib.colors as colors
from matplotlib.colors import Normalize, LogNorm
fig = plt.figure()

from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from astropy.coordinates import SkyCoord
cosmo = FlatLambdaCDM(H0=67.74, Om0=0.3089)
h = 0.6774

cov = np.loadtxt('/home/bulk826/Desktop/Stanford/Research3/figures/SZ/cdf_cov_2000.txt')

corr = np.zeros_like(cov)
for i in range(len(cov)):
    for j in range(len(cov[0])):
        corr[i, j] += cov[i, j] / np.sqrt(cov[i, i] * cov[j, j])

###########################################

fig.set_size_inches(8, 8)

ax1 = fig.add_subplot(1, 1, 1)
divider = make_axes_locatable(ax1)
cax = divider.append_axes('right', size='5%', pad=0.08)
cax.tick_params(labelsize = 18, direction = 'in')

print(np.max(corr[np.where(corr<1)]), np.min(corr))
'''
im1 = ax1.imshow(np.abs(cov.T), cmap = 'inferno', norm = colors.LogNorm(1e-9, np.max(cov)), 
                 interpolation = 'nearest', aspect='auto')
'''
im1 = ax1.imshow(corr.T, cmap = 'seismic', norm = colors.Normalize(-1, 1), 
                interpolation = 'nearest', aspect='auto')

xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()

print(cov.T/np.max(cov))
ax1.tick_params(length = 0)
ax1.set_xticklabels([])
ax1.set_yticklabels([])

xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()
width = xmax - xmin
height = ymax - ymin
ax1.text(xmin + 128*width, ymax + 4*height, r'kNN CDFs', 
         horizontalalignment = 'left', verticalalignment = 'top', 
         fontsize = 22, color = 'black')

cbar = fig.colorbar(im1, cax=cax, orientation='vertical')
cbar.set_label(r'Correlation Matrix', fontsize = 24, rotation = 270, labelpad = 32)
ax1.tick_params(labelsize = 18)

fig.savefig('/home/bulk826/Desktop/Stanford/Research3/figures/SZ/Cov_CDF.png', dpi = 400, bbox_inches = 'tight')