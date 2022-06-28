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
baseDir = '/home/bulk826/Desktop/Stanford/Research3/data'

################################
# Import redMaPPer data 
header_list = fits.open(baseDir+'/redMapper/redmapper_dr8_public_v6.3_catalog.fits')[1].data
hdul = fits.open(baseDir+'/redMapper/redmapper_dr8_public_v6.3_catalog.fits')
hdr = hdul[1].columns
#print(hdr)
#hdr variable for checking column names

data = fits.open(baseDir+'/redMapper/redmapper_dr8_public_v6.3_catalog.fits')[1].data
ZL = data['Z_LAMBDA']
ra = data['RA']
zmin = 0.1
zmax = 0.3
#idz = np.where((ZL >= zmin) & (ZL <= zmax) & (ra > 90) & (ra < 270))[0]
idz = np.where((ZL >= zmin) & (ZL <= zmax))[0]

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
ids = np.where(z_s >= 0.)[0]
RA_l = RA[idr][0:lenz]
DEC_l = DEC[idr][0:lenz]
R_l = cosmo.comoving_distance(z_l).value

def get_xyz(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.vstack((x, y, z)).T

pos_3D = get_xyz(R_l, (90-DEC_l)/180*np.pi, RA_l/180*np.pi) # 3D position for the selected Planck SZ catalog
l2 = cosmo.comoving_distance(zmax).value # radius of the two light cones
print('Cone radius (Mpc) = ', l2)

boxsize = l2 # half box size
#######################

X = np.vstack((RA_l, DEC_l)).T
ncen = 200
'''
km = kmeans_sample(X, ncen, maxiter=100, tol=1.0e-5)

# the centers found by the algorithm
print("found centers:",km.centers)

# did we converge?
print("converged?",km.converged)

# labels are the index to the nearest center for each point in X
print("labels size:",km.labels.size)

# how many in each cluster? Should be fairly uniform
print("cluster sizes:", np.bincount(km.labels))

print(np.mean(np.bincount(km.labels)))


# the distance to each center [Npoints, Ncen]
print("shape of distances:",km.distances.shape)
'''
np.random.seed(16)
IDk = np.arange(len(X))
np.random.shuffle(IDk)
idk = IDk[0:ncen]

cen_guess=np.zeros( (ncen, 2) )
ra_guess = X[idk, 0] #np.random.rand(ncen) * 360
dec_guess = X[idk, 1] #np.random.rand(ncen) * 180 - 90
cen_guess[:,0] = ra_guess
cen_guess[:,1] = dec_guess

# Fast version
# if you have numba installed, you can run a faster
# version
km=KMeans(cen_guess)
km.run(X, maxiter=1000)

# the centers found by the algorithm
print("found centers:",km.centers)

# did we converge?
print("converged?",km.converged)

# labels are the index to the nearest center for each point in X
print("labels size:",km.labels.size)
print(np.min(km.labels), np.max(km.labels))
# how many in each cluster? Should be fairly uniform
print("cluster sizes:", np.bincount(km.labels))

print(np.mean(np.bincount(km.labels)))
#print(km.distances)

# the distance to each center [Npoints, Ncen]
print("shape of distances:",km.distances.shape)

#fig.savefig(baseDir+'/../figures/paper/draft/kmeans_test.png', dpi = 200, bbox_inches = 'tight')

##################################

#RA DEC
fig.set_size_inches(8, 8)

ax1 = fig.add_subplot(1, 1, 1, projection = 'hammer') #Hammer in radians

for i in range(ncen):
    r = np.random.rand()
    g = np.random.rand()
    b = np.random.rand()
    rgb = (r, g, b)
    idkm = np.where(km.labels == i)[0]
    ax1.scatter(X[idkm, 0]/180*np.pi-np.pi, X[idkm,1]/180*np.pi, color = rgb, 
                marker = 'o', alpha = 0.8, s = 6.4, lw = 0., rasterized = True)

#ax1.scatter(ra_guess/180*np.pi-np.pi, dec_guess/180*np.pi, color = 'crimson', marker = '^', s = 6, label = r'final center')
ax1.grid('True')
#ax1.legend(loc = 'upper right', fontsize = 9)
ax1.tick_params(labelsize = 10)
#ax1.text(0., -15/180*np.pi, r'$0 < z < 0.1$', fontsize = 8)

###############################
# query points
'''
h_Nl = 100 # 100 query points for every dimension
datatype_h = type(pos_3D[0])
dx = boxsize/(h_Nl)
xh = yh = zh = np.linspace((-h_Nl)*dx, (h_Nl)*dx, h_Nl, dtype = datatype_h)
xh, yh, zh  = np.meshgrid(xh, yh, zh)
xh = xh.flatten()
yh = yh.flatten()
zh = zh.flatten()
xhg = np.ones((h_Nl**3, 3))
xhg[:, 0] = xh
xhg[:, 1] = yh
xhg[:, 2] = zh
# zmask
zmask = fits.open(baseDir+'/redMapper/redmapper_dr8_public_v6.3_zmask.fits')[1].data 
hpix = zmask['HPIX']
zMax = zmask['ZMAX']
f_good = zmask['FRACGOOD']
mask = np.where((f_good > 0.5) & (zMax > 0.3))[0]
Hpix = hpix[mask] # NSIDE 2048 heal pixels that are good area

NSIDE = 2048
vp = hp.vec2pix(NSIDE, xhg[:, 0], xhg[:, 1], xhg[:, 2])
ID_in = []
for i in range(len(vp)):
    if vp[i] in Hpix:
        ID_in.append(i)
    else:
        continue

ID_in = np.asarray(ID_in)
xhg_in = xhg[ID_in]
r_in = np.sqrt(xhg_in[:, 0]**2 + xhg_in[:, 1]**2 + xhg_in[:, 2]**2)
rmin = cosmo.comoving_distance(zmin).value
rmax = cosmo.comoving_distance(zmax).value
id_in = np.where((r_in >= rmin) & (r_in <= rmax))[0]
xhg_in = xhg_in[id_in]
'''

xhg_in = np.loadtxt(baseDir+'/redMapper/xhg_in.txt')

#Jackknife module

def get_ra_dec(x, y, z):
    phi = np.arctan2(y, x)
    theta = np.arctan2(np.sqrt(x**2 + y**2), z)
    #print(np.min(theta), np.max(theta))
    rra = phi/np.pi*180
    idp = np.where(phi < 0)[0]
    rra[idp] += 360
    ddec = -theta/np.pi*180 + 90

    return rra, ddec

RA_in, DEC_in = get_ra_dec(xhg_in[:, 0], xhg_in[:, 1], xhg_in[:, 2])

RA_DEC = np.vstack((RA_in, DEC_in)).T
ncen = 200
IDk = np.arange(len(RA_DEC))
np.random.shuffle(IDk)
idk = IDk[0:ncen]

cen_guess=np.zeros( (ncen, 2) )
ra_guess = RA_DEC[idk, 0] #np.random.rand(ncen) * 360
dec_guess = RA_DEC[idk, 1] #np.random.rand(ncen) * 180 - 90
cen_guess[:,0] = ra_guess
cen_guess[:,1] = dec_guess

# Fast version
# if you have numba installed, you can run a faster
# version
km=KMeans(cen_guess)
km.run(RA_DEC, maxiter=1000)
print(km.labels)

res = np.vstack((RA_in, DEC_in, km.labels)).T
np.savetxt(baseDir+'/../figures/paper/draft/kmeans_query.txt', res)

#res = np.loadtxt(baseDir+'/../figures/paper/draft/kmeans_query.txt')
RA_DEC = res[:, 0:2]
kml = res[:, 2]

'''
ax2 = fig.add_subplot(2, 1, 2, projection = 'hammer') #Hammer in radians
np.random.seed(28)
for i in range(ncen):
    r = np.random.rand()
    g = np.random.rand()
    b = np.random.rand()
    rgb = (r, g, b)
    #idkm = np.where(km.labels == i)[0]
    idkm = np.where(kml == i)[0]
    
    ax2.scatter(RA_DEC[idkm, 0]/180*np.pi-np.pi, RA_DEC[idkm,1]/180*np.pi, color = rgb, 
                marker = 'o', alpha = 0.82, s = 0.24, lw = 0., rasterized = True)

#ax2.scatter(ra_guess/180*np.pi-np.pi, dec_guess/180*np.pi, color = 'crimson', marker = '^', s = 6, label = r'final center')
ax2.grid('True')
#ax2.legend(loc = 'upper right', fontsize = 9)
ax2.tick_params(labelsize = 10)
#ax2.text(0., -15/180*np.pi, r'$0 < z < 0.1$', fontsize = 8)
'''

fig.savefig(baseDir+'/../figures/paper/draft/kmeans_plot.pdf', dpi = 400, bbox_inches = 'tight')
plt.show()
