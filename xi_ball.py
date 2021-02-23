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
cosmo = FlatLambdaCDM(H0=67.74, Om0=0.3089)
h = 0.6774
baseDir = '/oak/stanford/orgs/kipac/users/ycwang19/VPF/SZ'

bins = np.logspace(-4, np.log10(np.pi), 30)
n = 318
m = 393216

DD_sz = np.loadtxt(baseDir+'/haver/Xi_sz_DD.txt')
DR_sz = np.loadtxt(baseDir+'/haver/Xi_sz_DR.txt')
RR = np.loadtxt(baseDir+'/haver/Xi_sz_RR.txt')
DD_sz = DD_sz[1:] - DD_sz[:-1]
DR_sz = DR_sz[1:] - DR_sz[:-1]
RR = RR[1:] - RR[:-1]

cf_sz = ((m/n)**2 * DD_sz - 2*(m/n) * DR_sz + RR) / RR

DD_poi = np.load(baseDir+'/haver/Xi_poi_DD_0.npy')
DR_poi = np.load(baseDir+'/haver/Xi_poi_DR_0.npy')
DD_poi = DD_poi[1:] - DD_poi[:-1]
DR_poi = DR_poi[1:] - DR_poi[:-1]
cf_poi = ((m/n)**2 * DD_poi - 2*(m/n) * DR_poi + RR) / RR
CF_poi = [cf_poi]
len_poi = 2000
for i in range(1, len_poi):
    DD_poi = np.load(baseDir+'/haver/Xi_poi_DD_{}.npy'.format(i))
    DR_poi = np.load(baseDir+'/haver/Xi_poi_DR_{}.npy'.format(i))
    DD_poi = DD_poi[1:] - DD_poi[:-1]
    DR_poi = DR_poi[1:] - DR_poi[:-1]
    cf_poi = ((m/n)**2 * DD_poi - 2*(m/n) * DR_poi + RR) / RR
    CF_poi = np.concatenate((CF_poi, [cf_poi]), axis = 0)
    
CF_med = []
CF_low = []
CF_high = []
CF_mean = []
CF_std = []
for i in range(len(CF_poi[0])):
    #id_fin = np.where(np.isfinite(cf_poi[:, i]) == 1)[0]
    med = np.percentile(CF_poi[:, i], 50)
    low = np.percentile(CF_poi[:, i], 16)
    high = np.percentile(CF_poi[:, i], 84)
    mean = np.mean(CF_poi[:, i])
    std = np.std(CF_poi[:, i])
    CF_med.append(med)
    CF_low.append(low)
    CF_high.append(high)
    CF_mean.append(mean)
    CF_std.append(std)

CF_med = np.asarray(CF_med)
CF_low = np.asarray(CF_low)
CF_high = np.asarray(CF_high)
CF_mean = np.asarray(CF_mean)
CF_std = np.asarray(CF_std)

id_poi = np.where(np.isfinite(CF_mean) == 1)[0]
CF_med = CF_med[id_poi]
CF_low = CF_low[id_poi]
CF_high = CF_high[id_poi]
CF_mean = CF_mean[id_poi]
CF_std = CF_std[id_poi]

id_cf = np.where(np.isfinite(cf_sz) == 1)[0]
dCF = (cf_sz[id_poi] - CF_mean)/CF_std

##################

fig.set_size_inches(8, 14)
plt.subplots_adjust(wspace = 0.24, hspace = 0.42)

ax1 = fig.add_subplot(2, 1, 1)
#!!!!!!!!!!!!!!!!!

#ax1.plot(bins[:-1], DD_sz)
'''
ax1.plot(bins[:-1][id_cf][10:], cf_sz[id_cf][10:], lw = 2., color = 'royalblue', alpha = 0.8,
         label = r'Planck SZ ($Y_{5R_{500}}\geqslant 3\times 10^{-3}\,\mathrm{arcmin^{2}}$)')
'''
ax1.scatter(bins[:-1][id_cf][10:], cf_sz[id_cf][10:], color = 'royalblue', s = 82, lw = 0.8, 
            facecolors = 'None', marker = 'D', rasterized = True, alpha = 0.8, 
            label = r'Planck SZ ($Y_{5R_{500}}\geqslant 3\times 10^{-3}\,\mathrm{arcmin^{2}}$)')

#!!!!!!!!!!!!!!!!!
ax1.plot(bins[:-1][id_poi][10:], CF_mean[10:], lw = 1.6, alpha = 0.8, color = 'crimson', 
         dashes = (5, 2.4), label = r'Poisson randoms ('+str(len_poi)+') median')
ax1.fill_between(bins[:-1][id_poi][10:], CF_mean[10:]-CF_std[10:], CF_mean[10:]+CF_std[10:], alpha = 0.3, 
                 color = 'tomato')

ax1.set_xlabel(r'$\theta\,$[$\mathrm{rad}$]', fontsize = 24, labelpad = 8)
ax1.set_ylabel(r'$w(\theta)$', fontsize = 24)
ax1.legend(loc = 'upper right', fontsize = 16, frameon = False, borderpad = 0.5)

ax1.set_xscale('log')
ax1.set_xticks(np.logspace(-2, 0, 3))
minorLocator = AutoMinorLocator()
loc1 = LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0), numdecs=10, numticks = 10)
ax1.xaxis.set_minor_locator(loc1)
ax1.tick_params(which='major', labelsize = 18, width = 1., length = 6, direction='in', pad = 4, bottom = True, top = False, left = True, right = True)
ax1.tick_params(which='minor', labelsize = 18, width = 1., length = 3, direction='in', bottom = True, top = False, left = True, right = True)

'''
ax12= ax1.twiny()
ax12.set_xlabel(r'$\log_{10} V\,$[$\mathrm{Mpc^{3}}$]', fontsize = 24, labelpad = 8)
xmin, xmax = ax1.get_xlim()
ax12.set_xlim([xmin, xmax])
ax12.set_xscale('log')
ax12.tick_params(labelsize = 16)
v0 = np.logspace(3, 10, 8)
r0 = (v0 * 3 / (4*np.pi))**(1/3)
ax12.set_xticks(r0)
ax12.set_xticklabels(np.log10(v0))
ax12.tick_params(which='major', labelsize = 18, width = 1., length = 6, direction='in', pad = 4, bottom = False, top = True, left = True, right = True)
ax12.tick_params(which='minor', length=0.)
'''
######

ax2 = fig.add_subplot(2, 1, 2)


#ax2.plot(bins[:-1][id_poi][10:], dCF[10:], lw = 2., color = 'black', alpha = 0.5, label = r'Planck SZ')
ax2.scatter(bins[:-1][id_poi][10:], dCF[10:], lw = 0.8, color = 'black', alpha = 0.5, facecolors = 'None',
            marker = 'D', s = 82, rasterized = True, label = r'Planck SZ')

V1 = np.min(bins[:-1][id_poi][10:])
V2 = np.max(bins[:-1][id_poi][10:])

x = np.linspace(V1, V2, 10)
y = np.ones_like(x)
ax2.fill_between(x, -2*y, 2*y, color = 'dimgray', lw = 0., alpha = 0.2, 
                 label = r'Poisson sample $2\,\sigma$')

ax2.set_xlabel(r'$\theta\,$[$\mathrm{rad}$]', fontsize = 24)
ax2.set_ylabel(r'$\delta\,w(\theta) / \sigma_{w}(\theta)$', fontsize = 24)
ax2.legend(loc = 'upper right', fontsize = 16, frameon = False, borderpad = 0.5)

ax2.set_xscale('log')
ax2.set_xticks(np.logspace(-2, 0, 3))
minorLocator = AutoMinorLocator()
loc1 = LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0), numdecs=10, numticks = 10)
ax2.xaxis.set_minor_locator(loc1)
ax2.tick_params(which='major', labelsize = 18, width = 1., length = 6, direction='in', pad = 4, bottom = True, top = False, left = True, right = True)
ax2.tick_params(which='minor', labelsize = 18, width = 1., length = 3, direction='in', bottom = True, top = False, left = True, right = True)

'''
ax22= ax2.twiny()
ax22.set_xlabel(r'$\log_{10} V\,$[$\mathrm{Mpc^{3}}$]', fontsize = 24, labelpad = 8)
xmin, xmax = ax2.get_xlim()
ax22.set_xlim([xmin, xmax])
ax22.set_xscale('log')
ax22.tick_params(which='minor', length=0.)
v0 = np.logspace(3, 10, 8)
r0 = (v0 * 3 / (4*np.pi))**(1/3)
ax22.set_xticks(r0)
ax22.set_xticklabels(np.log10(v0))
ax22.tick_params(which='major', labelsize = 18, width = 1., length = 6, direction='in', pad = 4, bottom = False, top = True, left = True, right = True)
ax22.tick_params(which='minor', length=0.)
'''
#!!!!!!!!!!!!!!!!!!
fig.savefig(baseDir+'/../figures/xi_haver.pdf', dpi = 200, bbox_inches = 'tight')
#!!!!!!!!!!!!!!!!!!
plt.show()
