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
baseDir1 = '/oak/stanford/orgs/kipac/users/ycwang19/VPF/SZ/redMapper'
baseDir2 = '/oak/stanford/orgs/kipac/users/arkab/updated_data'

data1 = np.loadtxt('/oak/stanford/orgs/kipac/users/ycwang19/VPF/SZ/cf_qui/cf_rand.txt')
data2 = np.loadtxt('/oak/stanford/orgs/kipac/users/ycwang19/VPF/SZ/cf_qui/cf_qui.txt')

random_s1 = data1[:, 2]
random_cdf1 = data1[:, 1]
random_med1 = random_cdf1
random_low1 = random_cdf1 - random_s1
random_high1 = random_cdf1 + random_s1

data_s1 = data2[:, 2]
data_cdf1 = data2[:, 3]
binc = data2[:, 0]
data_med1 = data_cdf1 
data_low1 = data_cdf1 - data_s1
data_high1 = data_cdf1 + data_s1

dCF_data_med = (data_med1 - random_med1) / random_s1
dCF_data_low = (data_low1 - random_med1) / random_s1
dCF_data_high = (data_high1 - random_med1) / random_s1

len_bins = 51
bins = np.logspace(np.log10(35), np.log10(155), len_bins)

##################

fig.set_size_inches(8, 14)
plt.subplots_adjust(wspace = 0.24, hspace = 0.36)

ax1 = fig.add_subplot(2, 1, 1)
#!!!!!!!!!!!!!!!!!
ax1.plot(binc, data_med1, color = 'crimson', lw = 1.6, alpha = 0.8, linestyle = '-.', label = r'Clusters')
ax1.fill_between(binc, data_low1, data_high1, alpha = 0.3, color = 'crimson')

ax1.plot(binc, random_med1, lw = 1.6, alpha = 0.8, color = 'royalblue', 
         dashes = (5, 2.4), label = r'Randoms')
ax1.fill_between(binc, random_low1, random_high1, alpha = 0.3, color = 'deepskyblue')
#!!!!!!!!!!!!!!!!!
ax1.set_xlabel(r'$r\,$[$\mathrm{Mpc}$]', fontsize = 24, labelpad = 8)
ax1.set_ylabel(r'$\xi(r)$', fontsize = 24)
ax1.legend(loc = 'upper right', fontsize = 16, frameon = False, borderpad = 0.5)

ax1.set_xscale('log')
ax1.set_xticks(np.logspace(2, 2, 1))
minorLocator = AutoMinorLocator()
loc1 = LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0), numdecs=10, numticks = 10)
ax1.xaxis.set_minor_locator(loc1)
ax1.tick_params(which='major', labelsize = 18, width = 1., length = 6, direction='in', pad = 4, bottom = True, top = False, left = True, right = True)
ax1.tick_params(which='minor', labelsize = 18, width = 1., length = 3, direction='in', bottom = True, top = False, left = True, right = True)

ax12= ax1.twiny()
ax12.set_xlabel(r'$\log_{10} V\ $[$\mathrm{Mpc^{3}}$]', fontsize = 24, labelpad = 8)
xmin, xmax = ax1.get_xlim()
ax12.set_xlim([xmin, xmax])
ax12.set_xscale('log')
ax12.tick_params(labelsize = 16)
v0 = np.logspace(4, 8, 5)
r0 = (v0 * 3 / (4*np.pi))**(1/3)
ax12.set_xticks(r0)
ax12.set_xticklabels(np.log10(v0))
ax12.tick_params(which='major', labelsize = 18, width = 1., length = 6, direction='in', pad = 4, bottom = False, top = True, left = True, right = True)
ax12.tick_params(which='minor', length=0.)

######

ax2 = fig.add_subplot(2, 1, 2)

ax2.plot(binc, dCF_data_med, lw = 1.6, color = 'crimson', alpha = 0.8, linestyle = '-.')
ax2.fill_between(binc, dCF_data_low, dCF_data_high, alpha = 0.3, color = 'crimson')

V1 = np.min(binc)
V2 = np.max(binc)
x = np.linspace(V1, V2, 10)
y = np.ones_like(x)
ax2.fill_between(x, -1*y, 1*y, color = 'deepskyblue', lw = 0., alpha = 0.2)
ax2.axhline(y = 0, lw = 1.2, color = 'royalblue', alpha = 0.8, dashes = (5, 2.4))

ax2.set_xlabel(r'$r\,$[$\mathrm{Mpc}$]', fontsize = 24)
ax2.set_ylabel(r'$\delta\,\xi(r) / \sigma_{\xi}(r)$', fontsize = 24)
ax2.legend(loc = 'upper right', fontsize = 16, frameon = False, borderpad = 0.5)

ax2.set_xscale('log')
ax2.set_xticks(np.logspace(2, 2, 1))
minorLocator = AutoMinorLocator()
loc1 = LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0), numdecs=10, numticks = 10)
ax2.xaxis.set_minor_locator(loc1)
ax2.tick_params(which='major', labelsize = 18, width = 1., length = 6, direction='in', pad = 4, bottom = True, top = False, left = True, right = True)
ax2.tick_params(which='minor', labelsize = 18, width = 1., length = 3, direction='in', bottom = True, top = False, left = True, right = True)

ax22= ax2.twiny()
ax22.set_xlabel(r'$\log_{10} V\ $[$\mathrm{Mpc^{3}}$]', fontsize = 24, labelpad = 8)
xmin, xmax = ax2.get_xlim()
ax22.set_xlim([xmin, xmax])
ax22.set_xscale('log')
ax22.tick_params(which='minor', length=0.)
v0 = np.logspace(4, 8, 5)
r0 = (v0 * 3 / (4*np.pi))**(1/3)
ax22.set_xticks(r0)
ax22.set_xticklabels(np.log10(v0))
ax22.tick_params(which='major', labelsize = 18, width = 1., length = 6, direction='in', pad = 4, bottom = False, top = True, left = True, right = True)
ax22.tick_params(which='minor', length=0.)

fig.savefig(baseDir1+'/../../figures/xi_loo_D12.pdf', dpi = 400, bbox_inches = 'tight')
plt.show()