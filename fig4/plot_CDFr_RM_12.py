
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

x_1 = np.loadtxt(baseDir+'/CDF_12/x_1.txt')
Med1 = np.loadtxt(baseDir+'/CDF_12/Med1.txt')
Low1 = np.loadtxt(baseDir+'/CDF_12/Low1.txt')
High1 = np.loadtxt(baseDir+'/CDF_12/High1.txt')
R1 = np.loadtxt(baseDir+'/CDF_12/R1.txt')

x_2 = np.loadtxt(baseDir+'/CDF_12/x_2.txt')
Med2 = np.loadtxt(baseDir+'/CDF_12/Med2.txt')
Low2 = np.loadtxt(baseDir+'/CDF_12/Low2.txt')
High2 = np.loadtxt(baseDir+'/CDF_12/High2.txt')
R2 = np.loadtxt(baseDir+'/CDF_12/R2.txt')


##################################
i = 0
res1 = np.loadtxt(baseDir+'/jack_12/res_jack_1_{}.txt'.format(i))
res2 = np.loadtxt(baseDir+'/jack_12/res_jack_2_{}.txt'.format(i))
Res1 = [res1[:, 1]]
Res2 = [res2[:, 1]]
Binc1 = res1[:, 0]
Binc2 = res2[:, 0]

jack = 200
for i in range(1, jack):
    res1 = np.loadtxt(baseDir+'/jack_12/res_jack_1_{}.txt'.format(i))
    res2 = np.loadtxt(baseDir+'/jack_12/res_jack_2_{}.txt'.format(i))
    Res1 = np.concatenate((Res1, [res1[:, 1]]), axis = 0)
    Res2 = np.concatenate((Res2, [res2[:, 1]]), axis = 0)

##################################

fig.set_size_inches(16, 6.8)
plt.subplots_adjust(wspace = 0.24, hspace = 0.2)

ax1 = fig.add_subplot(1, 2, 1)

ax1.plot(x_1, R1, color = 'blueviolet', linewidth = 1.6, alpha = 0.8, label = r'redMaPPer')
ax1.fill_between(x_1, Low1, High1, alpha = 0.2, color = 'black', lw = 0.)
ax1.plot(x_1, Low1, linewidth = 0.8, dashes = (5, 2.4), alpha = 0.8, color = 'black')
ax1.plot(x_1, High1, linewidth = 0.8, dashes = (5, 2.4), alpha = 0.8, color = 'black')
ax1.plot(x_1, np.ones_like(x_1), lw = 1.6, alpha = 0.8, color = 'black', linestyle = '-.', label = r'randoms (2000)')

Low1 = []
High1 = []
std1 = []
for j in range(len(Binc1)):
    Low1.append(np.percentile(Res1[:, j], 2.3))
    High1.append(np.percentile(Res1[:, j], 97.7))
    std1.append(np.sqrt(jack-1) * np.std(Res1[:, j]))

Low1 = np.asarray(Low1)
High1 = np.asarray(High1)
std1 = np.asarray(std1)
ax1.fill_between(Binc1, R1-std1, R1+std1,  alpha = 0.3, color = 'blueviolet', lw = 0.)

ax1.set_xlabel(r'$r\,$[$\mathrm{Mpc}$]', fontsize = 24, labelpad = 8)
ax1.set_ylabel(r'$\mathrm{1NN/\langle 1NN_{rand} \rangle}$', fontsize = 24, labelpad = 8)
ax1.legend(loc = 'upper right', fontsize = 16, frameon = False, borderpad = 0.5)
#ax1.grid(color='k', linestyle='--', linewidth = 1.2, alpha=.2, which='both', axis='both')
#ax1.grid(color='k', linestyle='-', linewidth = 1.2, alpha=.2, which='major', axis='both')
ax1.set_ylim([0.845, 1.05])
ax1.set_xscale('log')
ax1.set_xticks([30, 40, 50, 60, 80, 100, 120, 140])
ax1.set_xticklabels([r'30', r'40', r'50', r'60', r'80', r'100', r'120', r'140'], fontsize = 18)
ax1.tick_params(which='major', labelsize = 18, width = 1., length = 4, direction='in', pad = 4, bottom = True, top = True, left = True, right = True)
minorLocator = AutoMinorLocator()
ax1.yaxis.set_minor_locator(minorLocator)
ax1.tick_params(which='minor', labelsize = 18, width = 1., length = 3, direction='in', bottom = False, top = False, left = True, right = True)


ax2 = fig.add_subplot(1, 2, 2)

ax2.plot(x_2, R2, color = 'orangered', linewidth = 1.6, alpha = 0.8, label = r'redMaPPer')
ax2.fill_between(x_2, Low2, High2, alpha = 0.2, color = 'black', lw = 0.)
ax2.plot(x_2, Low2, linewidth = 0.8, dashes = (5, 2.4), alpha = 0.8, color = 'black')
ax2.plot(x_2, High2, linewidth = 0.8, dashes = (5, 2.4), alpha = 0.8, color = 'black')
ax2.plot(x_2, np.ones_like(x_2), lw = 1.6, alpha = 0.8, color = 'black', linestyle = '-.', label = r'randoms (2000)')
Low2 = []
High2 = []
std2 = []
for j in range(len(Binc2)):
    Low2.append(np.percentile(Res2[:, j], 15.85))
    High2.append(np.percentile(Res2[:, j], 84.15))
    std2.append(np.sqrt(jack-1) * np.std(Res2[:, j]))

Low2 = np.asarray(Low2)
High2 = np.asarray(High2)
std2 = np.asarray(std2)
#ax2.fill_between(Binc2, Low2, High2,  alpha = 0.2, color = 'darkslateblue', lw = 0.)
ax2.fill_between(Binc2, R2-std2, R2+std2,  alpha = 0.3, color = 'orangered', lw = 0.)

ax2.set_xlabel(r'$r\,$[$\mathrm{Mpc}$]', fontsize = 24, labelpad = 8)
ax2.set_ylabel(r'$\mathrm{2NN/\langle 2NN_{rand} \rangle}$', fontsize = 24, labelpad = 8)
ax2.legend(loc = 'upper right', fontsize = 16, frameon = False, borderpad = 0.5)
#ax2.grid(color='k', linestyle='--', linewidth = 1.2, alpha=.2, which='both', axis='both')
#ax2.grid(color='k', linestyle='-', linewidth = 1.2, alpha=.2, which='major', axis='both')

ax2.set_xscale('log')
ax2.set_xticks([50, 60, 70, 80, 90, 100, 120, 140, 160])
ax2.set_xticklabels([r'50', r'60', r'70', r'80', r'90', r'100', r'120', r'140', r'160'], fontsize = 18)
ax2.tick_params(which='major', labelsize = 18, width = 1., length = 4, direction='in', pad = 4, bottom = True, top = True, left = True, right = True)
minorLocator = AutoMinorLocator()
ax2.yaxis.set_minor_locator(minorLocator)
ax2.tick_params(which='minor', labelsize = 18, width = 1., length = 3, direction='in', bottom = True, top = True, left = True, right = True)



fig.savefig(baseDir+'/../../figures/CDFr_RM_12.pdf', dpi = 400, bbox_inches = 'tight')
plt.show()
