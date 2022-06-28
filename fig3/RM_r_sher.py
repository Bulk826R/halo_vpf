from astropy.io import fits
import numpy as np
import scipy.spatial
from sklearn.neighbors import KDTree, BallTree
from scipy.stats import poisson, erlang
from scipy import interpolate
from os import urandom
import struct
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from astropy.coordinates import SkyCoord
import healpy as hp
from multiprocessing import Pool
cosmo = FlatLambdaCDM(H0=67.74, Om0=0.3089)
h = 0.6774
baseDir = '/oak/stanford/orgs/kipac/users/ycwang19/VPF/SZ'

################################
# Import redMaPPer data 
header_list = fits.open(baseDir+'/redMapper/redmapper_dr8_public_v6.3_catalog.fits')[1].data
hdul = fits.open(baseDir+'/redMapper/redmapper_dr8_public_v6.3_catalog.fits')
hdr = hdul[1].columns
#print(hdr)
#hdr variable for checking column names

data = fits.open(baseDir+'/redMapper/redmapper_dr8_public_v6.3_catalog.fits')[1].data
ZL = data['Z_LAMBDA']
zmin = 0.1
zmax = 0.3
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
ks = np.asarray([1, 5, 10, 50, 2, 3, 4, 8, 5, 9]) # All k orders
c_ks = ['springgreen', 'deepskyblue', 'tomato', 'slateblue']
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

print(len(xhg), len(xhg_in), len(pos_3D))
'''

xhg_in = np.loadtxt(baseDir+'/redMapper/xhg_in.txt')

# Although called VolumekNN, it is really RadiuskNN
def VolumekNN(xin, xout, k=1, periodic = 0):
    if isinstance(k, int): k = [k] # 
    xtree = scipy.spatial.cKDTree(xin, boxsize=periodic)
    dis, disi = xtree.query(xout, k=k, n_jobs=8) # dis is the distance to the kth nearest neighbour, disi is the id of that neighbour
    return dis # distances to nearest neighbors

vol_h = VolumekNN(pos_3D, xhg_in, k=ks) # distance to kNN for Planck SZ
np.savetxt(baseDir+'/redMapper/Dis_h_3D_Y500.txt', vol_h) # Y500 > 3e-3

##################################
#randoms
header_list = fits.open(baseDir+'/redMapper/redmapper_dr8_public_v6.3_randoms.fits')[1].data
hdul = fits.open(baseDir+'/redMapper/redmapper_dr8_public_v6.3_randoms.fits')
hdr = hdul[1].columns
#print(hdr)
rand = fits.open(baseDir+'/redMapper/redmapper_dr8_public_v6.3_randoms.fits')[1].data

Zr = rand['Z']
idz = np.where((Zr >= zmin) & (Zr <= zmax))[0]

RAr = rand['RA'][idz]
DECr = rand['DEC'][idz]
Z = rand['Z'][idz]
Rich_r = rand['LAMBDA'][idz]
Weight_r = rand['WEIGHT'][idz]

idr = np.where(Rich_r >= np.min(rich))[0]
z_r = Z[idr]
rich_r = Rich_r[idr]
RA_r = RAr[idr]
DEC_r = DECr[idr]
R_r = cosmo.comoving_distance(z_r).value

np.random.seed(12)
len_poi = 5000 # Number of Poisson realizations
'''
for j in range(len_poi):
    
    id_r = np.random.permutation(np.arange(len(z_r)))[:lenz]
    R_poi = R_r[id_r]
    Theta_poi = (90-DEC_r[id_r])/180*np.pi
    Phi_poi = RA_r[id_r]/180*np.pi
    poi_3D = get_xyz(R_poi, Theta_poi, Phi_poi)

    vol_g = VolumekNN(poi_3D, xhg_in, k=ks)
    #print(vol_g, j)
    np.save(baseDir+'/redMapper/Poi/Dis_g_Y500_{}.npy'.format(j), vol_g) # Y500 > 3e-3
'''
##################

def get_poi(j):
    id_r = np.random.permutation(np.arange(len(z_r)))[:lenz]
    R_poi = R_r[id_r]
    Theta_poi = (90-DEC_r[id_r])/180*np.pi
    Phi_poi = RA_r[id_r]/180*np.pi
    poi_3D = get_xyz(R_poi, Theta_poi, Phi_poi)

    vol_g = VolumekNN(poi_3D, xhg_in, k=ks)
    #print(vol_g, j)
    np.save(baseDir+'/redMapper/Poi/Dis_g_Y500_{}.npy'.format(j), vol_g) 
    return
    
ID_poi = np.arange(len_poi)
if __name__ == '__main__':
    p = Pool(16)
    p.map(get_poi, ID_poi)
