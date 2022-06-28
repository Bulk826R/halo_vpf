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
cosmo = FlatLambdaCDM(H0=67.74, Om0=0.3089)
h = 0.6774
baseDir = '/home/bulk826/Desktop/Stanford/Research3'

################################
# Import Planck SZ DR2 data 
header_list = fits.open(baseDir+'/data/redMapper/redmapper_dr8_public_v6.3_catalog.fits')[1].data
hdul = fits.open(baseDir+'/data/redMapper/redmapper_dr8_public_v6.3_catalog.fits')
hdr = hdul[1].columns
print(hdr)
#hdr variable for checking column names

data = fits.open(baseDir+'/data/redMapper/redmapper_dr8_public_v6.3_catalog.fits')[1].data
ZL = data['Z_LAMBDA']
ra = data['RA']

PZ = data['PZ']
PZ_bins = data['PZBINS']

print(PZ[0])
print(PZ_bins[0])

zmin = 0.1
zmax = 0.3
idz = np.where((ZL >= zmin) & (ZL <= zmax))[0]
#idz = np.where((ZL >= zmin) & (ZL <= zmax) & (ra > 90) & (ra < 270))[0]

RA = data['RA'][idz]
DEC = data['DEC'][idz]
Z_SPEC = data['Z_SPEC'][idz]
Rich = data['LAMBDA'][idz]
pz_bins = data['PZBINS'][idz]
PZ = data['PZ'][idz]
ZL = ZL[idz]

lenz = 1000
idr = np.argsort(Rich)[::-1]
z_l = ZL[idr][0:lenz]
z_s = Z_SPEC[idr][0:lenz]
rich = Rich[idr][0:lenz]
ra = RA[idr][0:lenz] / 180 * np.pi - np.pi
dec = DEC[idr][0:lenz] / 180 * np.pi
ids = np.where(z_s >= 0.)[0]
print(np.min(rich), len(ids))
Nbins = np.linspace(0, 0.3, 31)
yz, xz = np.histogram(z_l, bins = Nbins)

print(np.mean(z_l))

##################################
#randoms
header_list = fits.open(baseDir+'/data/redMapper/redmapper_dr8_public_v6.3_randoms.fits')[1].data
hdul = fits.open(baseDir+'/data/redMapper/redmapper_dr8_public_v6.3_randoms.fits')
hdr = hdul[1].columns
print(hdr)
rand = fits.open(baseDir+'/data/redMapper/redmapper_dr8_public_v6.3_randoms.fits')[1].data

Zr = rand['Z']
ra_r = rand['RA']
idz = np.where((Zr >= zmin) & (Zr <= zmax))[0]
#idz = np.where((Zr >= zmin) & (Zr <= zmax)  & (ra_r > 90) & (ra_r < 270))[0]

RA_r = rand['RA'][idz]
DEC_r = rand['DEC'][idz]
Z = rand['Z'][idz]
Rich_r = rand['LAMBDA'][idz]
Weight_r = rand['WEIGHT'][idz]
print(Weight_r)

idr = np.where(Rich_r >= np.min(rich))[0]
z_r = Z[idr]
rich_r = Rich_r[idr]
ra_r = RA_r[idr] / 180 * np.pi - np.pi
dec_r = DEC_r[idr] / 180 * np.pi
print(np.min(rich_r), len(rich_r))
yz1, xz1 = np.histogram(Z, bins = Nbins)

##################################

#RA DEC
fig.set_size_inches(8, 8)

print(np.min(dec))
ax1 = fig.add_subplot(2, 1, 1, projection = 'hammer') #Hammer in radians
ax1.scatter(ra, dec, color = 'crimson', marker = 'x', alpha = 0.8, s = 6, lw = 0.8, label = r'redMaPPer')
ax1.grid('True')
ax1.legend(loc = 'upper right', fontsize = 9)
ax1.tick_params(labelsize = 10)
#ax1.text(0., -15/180*np.pi, r'$0 < z < 0.1$', fontsize = 8)

ax2 = fig.add_subplot(2, 1, 2, projection = 'hammer') #Mollweide in radians
ax2.scatter(ra_r[lenz:2*lenz], dec_r[lenz:2*lenz], color = 'royalblue', 
            marker = '^', alpha = 0.72, s = 6, lw = 0.8, label = r'Randoms (1000)')

print(len(ra_r[lenz:2*lenz]))
ax2.grid('True')
ax2.legend(loc = 'upper right', fontsize = 9)
ax2.tick_params(labelsize = 10)

fig.savefig(baseDir+'/figures/paper/draft/RM_RA_DEC.pdf', dpi = 400, bbox_inches = 'tight')
plt.show()