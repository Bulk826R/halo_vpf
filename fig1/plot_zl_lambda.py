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
from matplotlib.ticker import AutoMinorLocator, LogLocator
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
baseDir = '/home/bulk826/Desktop/Stanford/Research3'

l1 = cosmo.comoving_distance(0.1)
l3 = cosmo.comoving_distance(0.3)
print(l1, l3)

################################
# Import Planck SZ DR2 data 
header_list = fits.open(baseDir+'/data/redMapper/redmapper_dr8_public_v6.3_catalog.fits')[1].data
hdul = fits.open(baseDir+'/data/redMapper/redmapper_dr8_public_v6.3_catalog.fits')
hdr = hdul[1].columns
print(hdr)
#hdr variable for checking column names

data = fits.open(baseDir+'/data/redMapper/redmapper_dr8_public_v6.3_catalog.fits')[1].data
ZL = data['Z_LAMBDA']
zmin = 0.1
zmax = 0.3
idz = np.where((ZL >= zmin) & (ZL <= zmax))[0]

print(len(idz))

RA = data['RA'][idz]
DEC = data['DEC'][idz]
Z_SPEC = data['Z_SPEC'][idz]
Rich = data['LAMBDA'][idz]
pz_bins = data['PZBINS'][idz]
PZ = data['PZ'][idz]
ZL = ZL[idz]
Zerr = data['Z_LAMBDA_ERR'][idz]
Lerr = data['LAMBDA_ERR'][idz]

lenz = 1000
idr = np.argsort(Rich)[::-1]
z_l = ZL[idr][0:lenz]
z_err = Zerr[idr][0:lenz]
r_err = Lerr[idr][0:lenz]
z_s = Z_SPEC[idr][0:lenz]
rich = Rich[idr][0:lenz]
RA_l = RA[idr][0:lenz]
DEC_l = DEC[idr][0:lenz]
ids = np.where(z_s >= 0.)[0]
print(np.min(rich), len(ids))
Nbins = np.linspace(0.1, 0.3, 21)
yz, xz = np.histogram(z_l, bins = Nbins)

#print(RA_l, DEC_l)

##################################
#randoms
header_list = fits.open(baseDir+'/data/redMapper/redmapper_dr8_public_v6.3_randoms.fits')[1].data
hdul = fits.open(baseDir+'/data/redMapper/redmapper_dr8_public_v6.3_randoms.fits')
hdr = hdul[1].columns
#print(hdr)
rand = fits.open(baseDir+'/data/redMapper/redmapper_dr8_public_v6.3_randoms.fits')[1].data

Zr = rand['Z']
idz = np.where((Zr >= zmin) & (Zr <= zmax))[0]

RA_r = rand['RA'][idz]
DEC_r = rand['DEC'][idz]
Z = rand['Z'][idz]
Rich_r = rand['LAMBDA'][idz]
Weight_r = rand['WEIGHT'][idz]

idr = np.where(Rich_r >= np.min(rich))[0]
z_r = Z[idr]
rich_r = Rich_r[idr]
#print(np.min(rich_r), len(rich_r))
yz1, xz1 = np.histogram(Z, bins = Nbins)
print('idr = ', len(idr))
np.random.seed(12)
ii = 10
id_r = np.random.permutation(np.arange(len(z_r)))[ii*lenz:(ii+1)*lenz]
#print(id_r)


##################################

fig.set_size_inches(10, 10)

ax1 = fig.add_subplot(1, 1, 1)
ax1.errorbar(z_l, rich, xerr = z_err, yerr = r_err, color = 'royalblue', 
             fmt = 'D', alpha = 0.6, markersize = 3.2, capsize = 2.4, lw = 0.8)

ax1.set_xlabel(r'photo-$z_{\lambda}$', fontsize = 24, labelpad = 8)
ax1.set_ylabel(r'richness $\lambda$', fontsize = 24, labelpad = 8)
ax1.tick_params(labelsize = 18)
#ax1.axvline(x = 0.33, dashes = (5, 2.4), lw = 1.2, color = 'black')

xmin, xmax = ax1.get_xlim()
ymin, ymax = ax1.get_ylim()
width = xmax - xmin
height = ymax - ymin
ax1.text(xmin + width*0.02, ymax - height * 0.04, 
         r'1000 richest $0.1 \leqslant z_{\lambda} \leqslant 0.3$', 
         horizontalalignment = 'left', verticalalignment = 'top', 
         fontsize = 18, color = 'black', alpha = 0.8)

######

divider = make_axes_locatable(ax1)
axHistx = divider.append_axes("top", 2, pad=0.2, sharex=ax1)
axHisty = divider.append_axes("right", 2, pad=0.2, sharey=ax1)

axHistx.xaxis.set_tick_params(labelbottom=False)
axHisty.yaxis.set_tick_params(labelleft=False)
axHistx.set_ylabel('$N(z_{\lambda})$', fontsize = 24)
axHisty.set_xlabel('$N(\lambda)$', fontsize = 24)

Zbins = np.linspace(0, 0.6, 31)
axHistx.hist(z_l, bins = Nbins, color = 'dimgray', edgecolor = 'black', 
         alpha = 0.5, linewidth = 1.2)

axHisty.hist(rich, bins = 25, color = 'orangered', edgecolor = 'black', 
         alpha = 0.5, linewidth = 1.2, orientation='horizontal')

axHistx.legend(loc = 'upper right', fontsize = 16, frameon = False)
plt.setp(axHistx.get_xticklabels(), visible = False)
plt.setp(axHisty.get_yticklabels(), visible = False)

minorLocator = AutoMinorLocator()
axHistx.xaxis.set_minor_locator(minorLocator)
minorLocator = AutoMinorLocator()
axHistx.yaxis.set_minor_locator(minorLocator)
axHistx.tick_params(which='major', labelsize = 18, width = 1., length = 6, direction='in', pad = 4, bottom = True, top = True, left = True, right = True)
axHistx.tick_params(which='minor', labelsize = 18, width = 1., length = 3, direction='in', bottom = True, top = True, left = True, right = True)


minorLocator = AutoMinorLocator()
axHisty.xaxis.set_minor_locator(minorLocator)
minorLocator = AutoMinorLocator()
axHisty.yaxis.set_minor_locator(minorLocator)
axHisty.tick_params(which='major', labelsize = 18, width = 1., length = 6, direction='in', pad = 4, bottom = True, top = True, left = True, right = True)
axHisty.tick_params(which='minor', labelsize = 18, width = 1., length = 3, direction='in', bottom = True, top = True, left = True, right = True)



fig.savefig(baseDir+'/figures/paper/draft/zl_lambda.pdf', dpi = 400, bbox_inches = 'tight')
plt.show()

