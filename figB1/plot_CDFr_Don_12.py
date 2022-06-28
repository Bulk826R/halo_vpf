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

data_qui = np.loadtxt('/oak/stanford/orgs/kipac/users/ycwang19/VPF/SZ/knn_qui/knn_qui.txt')
data_rand = np.loadtxt('/oak/stanford/orgs/kipac/users/ycwang19/VPF/SZ/knn_qui/knn_rand.txt')
data_loo = np.loadtxt('/oak/stanford/orgs/kipac/users/ycwang19/VPF/SZ/knn_loo/knn_loo.txt')

'''
data_s1 = np.loadtxt(baseDir2+'/1nn_data.txt')
jack_s1 = np.loadtxt(baseDir2+'/1nn_jackknife.txt')
random_s1 = np.loadtxt(baseDir2+'/1nn_random.txt')
data_cdf1 = np.loadtxt(baseDir2+'/means/1nn_data.txt')
jack_cdf1 = np.loadtxt(baseDir2+'/means/1nn_jackknife.txt')
random_cdf1 = np.loadtxt(baseDir2+'/means/1nn_random.txt')
'''
data_cdf1 = data_qui[:, 4] # Box no. 528 realization, CDF measurement correlated, don't use mean
jack_cdf1 = data_loo[:, 0]
random_cdf1 = data_rand[:, 0]
data_s1 = data_qui[:, 2]
jack_s1 = data_loo[:, 2]
random_s1 = data_rand[:, 2]
data_med1 = data_cdf1 / random_cdf1
data_low1 = (data_cdf1 - data_s1) / random_cdf1
data_high1 = (data_cdf1 + data_s1) / random_cdf1
jack_low1 = (data_cdf1 - jack_s1) / random_cdf1
jack_high1 = (data_cdf1 + jack_s1) / random_cdf1
random_med1 = np.ones_like(random_cdf1)
random_low1 = 1 - random_s1/random_cdf1
random_high1 = 1 + random_s1/random_cdf1

'''
data_s2 = np.loadtxt(baseDir2+'/2nn_data.txt')
jack_s2 = np.loadtxt(baseDir2+'/2nn_jackknife.txt')
random_s2 = np.loadtxt(baseDir2+'/2nn_random.txt')
data_cdf2 = np.loadtxt(baseDir2+'/means/2nn_data.txt')
jack_cdf2 = np.loadtxt(baseDir2+'/means/2nn_jackknife.txt')
random_cdf2 = np.loadtxt(baseDir2+'/means/2nn_random.txt')
'''
data_cdf2 = data_qui[:, 5] # Box no. 528 realization, CDF measurement correlated, don't use mean
jack_cdf2 = data_loo[:, 1]
random_cdf2 = data_rand[:, 1]
data_s2 = data_qui[:, 3]
jack_s2 = data_loo[:, 3]
random_s2 = data_rand[:, 3]

data_med2 = data_cdf2 / random_cdf2
data_low2 = (data_cdf2 - data_s2) / random_cdf2
data_high2 = (data_cdf2 + data_s2) / random_cdf2
jack_low2 = (data_cdf2 - jack_s2) / random_cdf2
jack_high2 = (data_cdf2 + jack_s2) / random_cdf2
random_med2 = np.ones_like(random_cdf2)
random_low2 = 1 - random_s2/random_cdf2
random_high2 = 1 + random_s2/random_cdf2

x_1 = np.logspace(np.log10(35), np.log10(135), 26)
x_2 = np.logspace(np.log10(55), np.log10(155), 26)
x_1 = (x_1[1:] + x_1[:-1]) / 2
x_2 = (x_2[1:] + x_2[:-1]) / 2

##################################

fig.set_size_inches(16, 6.8)
plt.subplots_adjust(wspace = 0.24, hspace = 0.2)


ax1 = fig.add_subplot(1, 2, 1)

ax1.plot(x_1, data_med1, color = 'blueviolet', linewidth = 1.6, alpha = 0.8, label = r'Clusters')
ax1.errorbar(x_1, data_med1, yerr = [data_med1-data_low1, data_high1-data_med1], color = 'blueviolet',
             lw = 1.2, fmt = 'x', alpha = 0.8, capsize = 4.2, markersize = 5, label = 'Sim-to-sim')
ax1.fill_between(x_1, jack_low1, jack_high1,  alpha = 0.3, color = 'blueviolet', lw = 0., label = r'Jackknife')
ax1.fill_between(x_1, random_low1, random_high1, alpha = 0.2, color = 'black', lw = 0.)
ax1.plot(x_1, random_low1, linewidth = 0.8, dashes = (5, 2.4), alpha = 0.8, color = 'black')
ax1.plot(x_1, random_high1, linewidth = 0.8, dashes = (5, 2.4), alpha = 0.8, color = 'black')
ax1.plot(x_1, random_med1, lw = 1.6, alpha = 0.8, color = 'black', linestyle = '-.', label = r'Randoms')

ax1.set_xlabel(r'$r\,$[$\mathrm{Mpc}$]', fontsize = 24, labelpad = 8)
ax1.set_ylabel(r'$\mathrm{1NN/\langle 1NN_{rand} \rangle}$', fontsize = 24, labelpad = 8)
#ax1.legend(loc = 'lower right', fontsize = 16, frameon = False, borderpad = 0.5)
ax1.legend(bbox_to_anchor=(0.05, 0.42, 0.99, .12), loc='lower left', 
           fontsize = 16, ncol = 1, frameon = False, borderpad = 0.5)

ax1.set_xscale('log')
ax1.set_xticks([30, 40, 50, 60, 80, 100, 120, 140])
ax1.set_xticklabels([r'30', r'40', r'50', r'60', r'80', r'100', r'120', r'140'], fontsize = 18)
ax1.tick_params(which='major', labelsize = 18, width = 1., length = 4, direction='in', pad = 4, bottom = True, top = True, left = True, right = True)
minorLocator = AutoMinorLocator()
ax1.yaxis.set_minor_locator(minorLocator)
ax1.tick_params(which='minor', labelsize = 18, width = 1., length = 3, direction='in', bottom = False, top = False, left = True, right = True)


ax2 = fig.add_subplot(1, 2, 2)

ax2.plot(x_2, data_med2, color = 'orangered', linewidth = 1.6, alpha = 0.8, label = r'Clusters')
ax2.errorbar(x_2, data_med2, yerr = [data_med2-data_low2, data_high2-data_med2], color = 'orangered',
             lw = 1.2, fmt = 'x', alpha = 0.8, capsize = 4.2, markersize = 5, label = 'Sim-to-sim')
ax2.fill_between(x_2, jack_low2, jack_high2,  alpha = 0.3, color = 'orangered', lw = 0., label = r'Jackknife')
ax2.fill_between(x_2, random_low2, random_high2, alpha = 0.2, color = 'black', lw = 0.)
ax2.plot(x_2, random_low2, linewidth = 0.8, dashes = (5, 2.4), alpha = 0.8, color = 'black')
ax2.plot(x_2, random_high2, linewidth = 0.8, dashes = (5, 2.4), alpha = 0.8, color = 'black')
ax2.plot(x_2, random_med2, lw = 1.6, alpha = 0.8, color = 'black', linestyle = '-.', label = r'Randoms')

ax2.set_xlabel(r'$r\,$[$\mathrm{Mpc}$]', fontsize = 24, labelpad = 8)
ax2.set_ylabel(r'$\mathrm{2NN/\langle 2NN_{rand} \rangle}$', fontsize = 24, labelpad = 8)
ax2.legend(loc = 'upper right', fontsize = 16, frameon = False, borderpad = 0.5)

ax2.set_xscale('log')
ax2.set_xticks([50, 60, 70, 80, 90, 100, 120, 140, 160])
ax2.set_xticklabels([r'50', r'60', r'70', r'80', r'90', r'100', r'120', r'140', r'160'], fontsize = 18)
ax2.tick_params(which='major', labelsize = 18, width = 1., length = 4, direction='in', pad = 4, bottom = True, top = True, left = True, right = True)
minorLocator = AutoMinorLocator()
ax2.yaxis.set_minor_locator(minorLocator)
ax2.tick_params(which='minor', labelsize = 18, width = 1., length = 3, direction='in', bottom = True, top = True, left = True, right = True)


fig.savefig(baseDir1+'/../../figures/CDFr_Don_12.pdf', dpi = 400, bbox_inches = 'tight')
plt.show()
