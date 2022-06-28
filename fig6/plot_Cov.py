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

baseDir = '/home/bulk826/Desktop/Stanford/Research3/data/redMapper'
cov = np.loadtxt(baseDir+'/cdf_cov_12_2000.txt')
corr1 = np.zeros_like(cov)
for i in range(len(cov)):
    for j in range(len(cov[0])):
        corr1[i, j] += cov[i, j] / np.sqrt(cov[i, i] * cov[j, j])

cov = np.loadtxt(baseDir+'/xi_cov_2000.txt')
corr2 = np.zeros_like(cov)
for i in range(len(cov)):
    for j in range(len(cov[0])):
        corr2[i, j] += cov[i, j] / np.sqrt(cov[i, i] * cov[j, j])

#####################################

fig.set_size_inches(16, 6.4)
plt.subplots_adjust(wspace = 0.48, hspace = 0.2)

ax1 = fig.add_subplot(1, 2, 1)
divider = make_axes_locatable(ax1)
cax = divider.append_axes('right', size='5%', pad=0.08)
cax.tick_params(labelsize = 18, direction = 'in')

im1 = ax1.imshow(corr1.T, cmap = 'seismic', norm = colors.Normalize(-1, 1), 
                interpolation = 'nearest', aspect='auto')

xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()
ax1.tick_params(length = 0)
ax1.set_xticks([12, 38])
ax1.set_yticks([12, 38])
ax1.set_xticklabels([r'$\mathrm{1NN}$', r'$\mathrm{2NN}$'])
ax1.set_yticklabels([r'$\mathrm{1NN}$', r'$\mathrm{2NN}$'])
ax1.tick_params(labelsize = 22, pad = 8)
cbar = fig.colorbar(im1, cax=cax, orientation='vertical')
cbar.set_label(r'$\mathcal{R}$', fontsize = 24, rotation = 270, labelpad = 32)


ax1 = fig.add_subplot(1, 2, 2)
divider = make_axes_locatable(ax1)
cax = divider.append_axes('right', size='5%', pad=0.08)
cax.tick_params(labelsize = 18, direction = 'in')
im1 = ax1.imshow(corr2.T, cmap = 'seismic', norm = colors.Normalize(-1, 1), 
                interpolation = 'nearest', aspect='auto')

ax1.tick_params(length = 0)
ax1.set_xticks([25, ])
ax1.set_yticks([25, ])
ax1.set_xticklabels([r'$\xi(r)$'])
ax1.set_yticklabels([r'$\xi(r)$'])
ax1.tick_params(labelsize = 22, pad = 8)
cbar = fig.colorbar(im1, cax=cax, orientation='vertical')
cbar.set_label(r'$\mathcal{R}$', fontsize = 24, rotation = 270, labelpad = 32)

fig.savefig('/home/bulk826/Desktop/Stanford/Research3/figures/paper/draft/Cov_CDF_xi.pdf', dpi = 400, bbox_inches = 'tight')