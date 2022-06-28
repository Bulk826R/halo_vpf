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
from matplotlib.ticker import AutoMinorLocator, LogLocator
cosmo = FlatLambdaCDM(H0=67.74, Om0=0.3089)
h = 0.6774
baseDir = '/oak/stanford/orgs/kipac/users/ycwang19/VPF/SZ/redMapper'

###################

x_up_1 = np.loadtxt(baseDir+'/CDF_12/x_up_1NN.txt')
y_up_1 = np.loadtxt(baseDir+'/CDF_12/y_up_1NN.txt')
X_up_1 = np.loadtxt(baseDir+'/CDF_12/X_up_1NN.txt')
Y_up_1 = np.loadtxt(baseDir+'/CDF_12/Y_up_1NN.txt')
S_up_1 = np.loadtxt(baseDir+'/CDF_12/S_up_1NN.txt')
x_down_1 = np.loadtxt(baseDir+'/CDF_12/x_down_1NN.txt')
y_down_1 = np.loadtxt(baseDir+'/CDF_12/y_down_1NN.txt')

x_up_2 = np.loadtxt(baseDir+'/CDF_12/x_up_2NN.txt')
y_up_2 = np.loadtxt(baseDir+'/CDF_12/y_up_2NN.txt')
X_up_2 = np.loadtxt(baseDir+'/CDF_12/X_up_2NN.txt')
Y_up_2 = np.loadtxt(baseDir+'/CDF_12/Y_up_2NN.txt')
S_up_2 = np.loadtxt(baseDir+'/CDF_12/S_up_2NN.txt')
x_down_2 = np.loadtxt(baseDir+'/CDF_12/x_down_2NN.txt')
y_down_2 = np.loadtxt(baseDir+'/CDF_12/y_down_2NN.txt')

fig.set_size_inches(20, 6.4)
plt.subplots_adjust(wspace = 0.2, hspace = 0.24)

ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(x_up_1, y_up_1, color = 'blueviolet', linewidth = 1.6, 
         alpha = 0.8, label = r'$\mathrm{1NN}$')
ax1.fill_between(X_up_1, Y_up_1-1*S_up_1, Y_up_1+1*S_up_1, alpha = 0.3, color = 'blueviolet', lw = 0.)
ax1.fill_between(X_up_1, Y_up_1-2*S_up_1, Y_up_1+2*S_up_1, alpha = 0.2, color = 'blueviolet', lw = 0.)
ax1.fill_between(X_up_1, Y_up_1-3*S_up_1, Y_up_1+3*S_up_1, alpha = 0.1, color = 'blueviolet', lw = 0.)

ax1.plot(x_up_2, y_up_2, color = 'orangered', linewidth = 1.6, 
         alpha = 0.8, label = r'$\mathrm{2NN}$')
ax1.fill_between(X_up_2, Y_up_2-1*S_up_2, Y_up_2+1*S_up_2, alpha = 0.3, color = 'orangered', lw = 0.)
ax1.fill_between(X_up_2, Y_up_2-2*S_up_2, Y_up_2+2*S_up_2, alpha = 0.2, color = 'orangered', lw = 0.)
ax1.fill_between(X_up_2, Y_up_2-3*S_up_2, Y_up_2+3*S_up_2, alpha = 0.1, color = 'orangered', lw = 0.)

ax1.grid(color = 'k', linestyle = '--', linewidth = 1.2, alpha = 0.2, which = 'both', axis = 'both')
ax1.grid(color = 'k', linestyle = '-', linewidth = 1.2, alpha = 0.2, which = 'major', axis = 'both')
ax1.set_xlabel(r'$r\,$[$\mathrm{Mpc}$]', fontsize = 24, labelpad = 8)
ax1.set_ylabel(r'$\mathrm{pCDF}$', fontsize = 24, labelpad = 8)
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_ylim([1e-2, 0.7])
ax1.set_xticks(np.logspace(2, 2, 1))
ax1.set_yticks(np.logspace(-2, -1, 2))
minorLocator = AutoMinorLocator()
loc1 = LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0), numdecs=10, numticks = 10)
ax1.xaxis.set_minor_locator(loc1)
minorLocator = AutoMinorLocator()
loc2 = LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0), numdecs=10, numticks = 10)
ax1.yaxis.set_minor_locator(loc2)
ax1.tick_params(which='major', labelsize = 18, width = 1., length = 6, direction='in', pad = 4, bottom = True, top = False, left = True, right = True)
ax1.tick_params(which='minor', labelsize = 18, width = 1., length = 3, direction='in', bottom = True, top = False, left = True, right = True)
ax1.legend(loc = 'upper left', fontsize = 16)

ax12= ax1.twiny()
ax12.set_xlabel(r'$\log_{10} V\ $[$\mathrm{Mpc^{3}}$]', fontsize = 24, labelpad = 8)
xmin, xmax = ax1.get_xlim()
ax12.set_xlim([xmin, xmax])
ax12.set_xscale('log')
v0 = np.logspace(4, 8, 5)
r0 = (v0 * 3 / (4*np.pi))**(1/3)
ax12.set_xticks(r0)
ax12.set_xticklabels(np.log10(v0))
ax12.tick_params(which='major', labelsize = 18, width = 1., length = 6, direction='in', pad = 4, bottom = False, top = True, left = True, right = True)
ax12.tick_params(which='minor', length=0.)

###################

x_up_1 = np.loadtxt(baseDir+'/CIC_12/x_up_1NN.txt')
y_up_1 = np.loadtxt(baseDir+'/CIC_12/y_up_1NN.txt')
X_up_1 = np.loadtxt(baseDir+'/CIC_12/X_up_1NN.txt')
Y_up_1 = np.loadtxt(baseDir+'/CIC_12/Y_up_1NN.txt')
S_up_1 = np.loadtxt(baseDir+'/CIC_12/S_up_1NN.txt')
x_down_1 = np.loadtxt(baseDir+'/CIC_12/x_down_1NN.txt')
y_down_1 = np.loadtxt(baseDir+'/CIC_12/y_down_1NN.txt')

x_up_2 = np.loadtxt(baseDir+'/CIC_12/x_up_2NN.txt')
y_up_2 = np.loadtxt(baseDir+'/CIC_12/y_up_2NN.txt')
X_up_2 = np.loadtxt(baseDir+'/CIC_12/X_up_2NN.txt')
Y_up_2 = np.loadtxt(baseDir+'/CIC_12/Y_up_2NN.txt')
S_up_2 = np.loadtxt(baseDir+'/CIC_12/S_up_2NN.txt')
x_down_2 = np.loadtxt(baseDir+'/CIC_12/x_down_2NN.txt')
y_down_2 = np.loadtxt(baseDir+'/CIC_12/y_down_2NN.txt')

ax1 = fig.add_subplot(1, 2, 2)
ax1.plot(x_up_1, y_up_1, color = 'blueviolet', linewidth = 1.6, 
         alpha = 0.8, label = r'$k=1$') #\mathrm{1NN}
ax1.fill_between(X_up_1, Y_up_1-1*S_up_1, Y_up_1+1*S_up_1, alpha = 0.3, color = 'blueviolet', lw = 0.)
ax1.fill_between(X_up_1, Y_up_1-2*S_up_1, Y_up_1+2*S_up_1, alpha = 0.2, color = 'blueviolet', lw = 0.)
ax1.fill_between(X_up_1, Y_up_1-3*S_up_1, Y_up_1+3*S_up_1, alpha = 0.1, color = 'blueviolet', lw = 0.)

ii = 0
ax1.plot(x_up_2[ii:], y_up_2[ii:], color = 'orangered', linewidth = 1.6, 
         alpha = 0.8, label = r'$k=2$')
ax1.fill_between(X_up_2[ii:], Y_up_2[ii:]-1*S_up_2[ii:], Y_up_2[ii:]+1*S_up_2[ii:], alpha = 0.3, color = 'orangered', lw = 0.)
ax1.fill_between(X_up_2[ii:], Y_up_2[ii:]-2*S_up_2[ii:], Y_up_2[ii:]+2*S_up_2[ii:], alpha = 0.2, color = 'orangered', lw = 0.)
ax1.fill_between(X_up_2[ii:], Y_up_2[ii:]-3*S_up_2[ii:], Y_up_2[ii:]+3*S_up_2[ii:], alpha = 0.1, color = 'orangered', lw = 0.)

ax1.set_xlabel(r'$r\,$[$\mathrm{Mpc}$]', fontsize = 24)
ax1.set_ylabel(r'$P_{k|V}$', fontsize = 24)
#ax1.set_ylabel(r'$\mathrm{CIC}$', fontsize = 24)
ax1.legend(loc = 'upper left', fontsize = 16)
ax1.grid(color='k', linestyle='--', linewidth = 1.2, alpha=.2, which='both', axis='both')
ax1.grid(color='k', linestyle='-', linewidth = 1.2, alpha=.2, which='major', axis='both')

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_ylim([3e-2, 0.5])
ax1.set_xticks(np.logspace(2, 2, 1))
ax1.set_yticks(np.logspace(-1, -1, 1))
minorLocator = AutoMinorLocator()
loc1 = LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0), numdecs=10, numticks = 10)
ax1.xaxis.set_minor_locator(loc1)
minorLocator = AutoMinorLocator()
loc2 = LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0), numdecs=10, numticks = 10)
ax1.yaxis.set_minor_locator(loc2)
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

fig.savefig(baseDir+'/../../figures/CDF_CIC_12.pdf', dpi = 400, bbox_inches = 'tight')
plt.show()
