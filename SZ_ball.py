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
cosmo = FlatLambdaCDM(H0=67.74, Om0=0.3089)
h = 0.6774
baseDir = '/oak/stanford/orgs/kipac/users/ycwang19/VPF/SZ'

################################
# Import Planck SZ DR2 data 
header_list = fits.open(baseDir+'/data/HFI_PCCS_SZ-union_R2.08.fits')[1].data
hdul = fits.open(baseDir+'/data/HFI_PCCS_SZ-union_R2.08.fits')
hdr = hdul[1].columns
print(hdr)
#hdr variable for checking column names

data = fits.open(baseDir+'/data/HFI_PCCS_SZ-union_R2.08.fits')[1].data
IR_FLAG = data['IR_FLAG'] # IR contamination flag
Q_neu = data['Q_NEURAL'] # Quality factor determined by neural net
FLAG = np.where((Q_neu > 0.4) & (IR_FLAG == 0))[0] # Good neural network confidence, not in regions of heavy IR contamination

# All quantities loaded below account for quality factor and removal of IR contaminated clusters
RA = data['RA'][FLAG]
DEC = data['DEC'][FLAG]
Z = data['REDSHIFT'][FLAG] # raw redshift
rID = data['REDSHIFT_ID'][FLAG]
vID = data['VALIDATION'][FLAG]
SNR = data['SNR'][FLAG] # Signal to noise ratio
Y_5R500 = data['Y5R500'][FLAG] * 1e-3 #arcmin^2
Y_5R500_err = data['Y5R500_ERR'][FLAG] * 1e-3 #arcmin^2
Msz = data['MSZ'][FLAG] # in 1e14 Msun
Msz_err_high = data['MSZ_ERR_UP'][FLAG] # in 1e14 Msun
Msz_err_low = data['MSZ_ERR_LOW'][FLAG] # in 1e14 Msun
Gal_B = data['GLAT'][FLAG]
Gal_L = data['GLON'][FLAG]
Msz *= 1e14
Msz_err_high *= 1e14
Msz_err_low *= 1e14

# Spectroscopic redshift clusters
ID_spec = np.where((vID == 10) | (vID == 11) | (vID == 14))[0] 
# Photometric redshift clusters
ID_photo = np.where((vID == 12) | (vID == 13) | (vID == 15) | (vID == 16) | (vID == 21) | (vID == 22) | (vID == 23) | (vID == 24))[0]
# Spec+Photo-z overlap clusters
ID_mix = np.where((vID == 20) | (vID == 25) | (vID == 30))[0]
# No redshift confirmation
ID_no = np.where(vID == -1)[0]
# Redshifts that are within [0, 0.6] and non-empty, enforces SNR
ID_z = np.where((vID != -1) & (Z >= 0.) & (Z < 0.6) & (SNR >= 4.5))[0]
ID_z2 = np.where((vID != -1) & (Z >= 0.) & (Z < 0.6))[0]
print('ID_spec = ', len(ID_spec))
print('ID_photo = ', len(ID_photo))
print('ID_z = ', len(ID_z))
print('ID_z2 = ', len(ID_z2)) # these two are equal, Q>0.4 guarantees SNR >= 4.5

#############################

z = Z[ID_z]
vID_z = vID[ID_z]
s = 0.03 * np.ones_like(z) # photometric redshift error (standard deviation for Gaussian)

#!!!!!!!!!!!!!!!!!!!!!!!!!!
np.random.seed(12)
#!!!!!!!!!!!!!!!!!!!!!!!!!!

zp = np.random.normal(z, s, len(z)) # Resample redshifts to account for photo-z error
#Replace spectroscopic redshift with original values (thus photo-z errors only applies to non-spec-z redshifts)
ID_spec_z = np.where((vID_z == 10) | (vID_z == 11) | (vID_z == 14))[0]
zp[ID_spec_z] = z[ID_spec_z] 
zp_err = np.zeros_like(zp)
ID_photo_z = np.where((vID_z != 10) & (vID_z != 11) & (vID_z != 14))[0]
zp_err[ID_photo_z] = s[ID_photo_z] # redshift errors for photo-z clusters set to 0.03, 0. for spec-z clusters

R_z = cosmo.comoving_distance(zp).value # Convert redshift to comoving distance
Y500_z = Y_5R500[ID_z]
Y500_errz = Y_5R500_err[ID_z]
Msz_z = Msz[ID_z]
Msz_zlow = Msz_err_low[ID_z]
Msz_zhigh = Msz_err_high[ID_z]
Z_z = zp
Z_errz = zp_err
'''
DEC_z = DEC[ID_z]
RA_z = RA[ID_z]
c_icrs = SkyCoord(ra=RA_z*u.degree, dec=DEC_z*u.degree, frame='icrs')
gal_cord = c_icrs.galactic 
gal_L = gal_cord.l.degree 
gal_B = gal_cord.b.degree 
print('L, B = ', gal_L[0]-180, gal_B[0])
#print(np.min(RA_z), np.max(RA_z))
'''
Gal_Bz = Gal_B[ID_z] #Galatic latitude
Gal_Lz = Gal_L[ID_z] #Galactic longitude
gal_L = Gal_Lz #Longitude (0, 360) deg
gal_B = Gal_Bz #Latitude  (-90, 90) deg
#print('GL, GB = ', Gal_Lz[0]-180, Gal_Bz[0])

Dec_mask = np.where((gal_B/180*np.pi <= -np.pi/6) | (gal_B/180*np.pi >= np.pi/6))[0] # galactic angular mask
R_z = R_z[Dec_mask]
gal_L = gal_L[Dec_mask]
gal_B = gal_B[Dec_mask]
Z_z = Z_z[Dec_mask]
Z_errz = Z_errz[Dec_mask]
Y500_z = Y500_z[Dec_mask]
Y500_errz = Y500_errz[Dec_mask]
Msz_z = Msz_z[Dec_mask]
Msz_zlow = Msz_zlow[Dec_mask]
Msz_zhigh = Msz_zhigh[Dec_mask]

print('len after Dec mask = ', len(Msz_z))

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Applying Y500 cut, and also redshift cut again, since applying Gaussian 
# errors in line 73 can bring redshifts out of the [0, 0.6] range (10 clusters exit the range in total)
idz = np.where((Z_z <= 0.6) & (Z_z > 0.) & (Y500_z >= 3e-3))[0]  
#idz = np.where((Z_z <= 0.6) & (Z_z > 0.) & (Y500_z < 3e-3))[0]
#idz = np.where((Z_z <= 0.6) & (Z_z > 0.))[0]

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Y500 = Y500_z[idz]
Y500_err = Y500_errz[idz]
msz = Msz_z[idz]
msz_low = Msz_zlow[idz]
msz_high = Msz_zhigh[idz]
Z = Z_z[idz]
Zerr = Z_errz[idz]

Nbins = np.linspace(0, 0.6, 31)
yz, xz = np.histogram(Z, bins = Nbins)

def get_xyz(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.vstack((x, y, z)).T

pos_3D = get_xyz(R_z, (90-gal_B)/180*np.pi, gal_L/180*np.pi)[idz] # 3D position for the selected Planck SZ catalog
r_3D = np.sqrt(pos_3D[:, 0]**2 + pos_3D[:, 1]**2 + pos_3D[:, 2]**2) # 3D distance to origin
l2 = np.max(r_3D) # radius of the two light cones
print('Cone radius = ', l2)

boxsize = l2 # half box size
ks = np.asarray([1, 5, 10, 50, 2, 3, 4, 8, 5, 9]) # All k orders
c_ks = ['springgreen', 'deepskyblue', 'tomato', 'slateblue']

zq = 0.2
r_q = cosmo.comoving_distance(zq).value 
NSIDE = 256
NPIX = hp.nside2npix(NSIDE)
m = np.arange(NPIX)
theta_q, phi_q = hp.pix2ang(nside = NSIDE, ipix = m)
idq = np.where((theta_q <= np.pi/3) | (theta_q >= 2*np.pi/3))[0]
Theta_q = theta_q[idq]
Phi_q = phi_q[idq]
R_q = r_q * np.ones_like(idq)
xhg_in = get_xyz(R_q, Theta_q, Phi_q)

# Although called VolumekNN, it is really RadiuskNN
def VolumekNN(xin, xout, k=1, periodic = 0):
    xtree = BallTree(xin, metric = 'haversine')  
    dis, disi = xtree.query(xout, k=k)    # dis is the distance to the kth nearest neighbour, disi is the id of that neighbour
    r = np.logspace(-4, np.log10(np.pi), 30)
    xi_ball = xtree.two_point_correlation(xout, r)
    Xi_ball = xtree.two_point_correlation(xin, r)
    #print(np.log10(np.min(dis)))
    return dis, Xi_ball, xi_ball 

X = np.vstack((gal_L[idz]/180*np.pi, (90-gal_B[idz])/180*np.pi)).T
Xq = np.vstack((Phi_q, Theta_q)).T

vol_h, Xi_sz, xi_sz = VolumekNN(X, Xq, k=10) # distance to kNN for Planck SZ
np.savetxt(baseDir+'/haver/Dis_h_3D_Y500.txt', vol_h) # Y500 > 3e-3
np.savetxt(baseDir+'/haver/Xi_sz_DD.txt', Xi_sz) # Y500 > 3e-3
np.savetxt(baseDir+'/haver/Xi_sz_DR.txt', xi_sz) # Y500 > 3e-3

vol_r, Xi_rr, xi_rr = VolumekNN(Xq, Xq, k=10)
np.savetxt(baseDir+'/haver/Xi_sz_RR.txt', Xi_rr)

len_poi = 200 # Number of Poisson realizations
for j in range(len_poi):
    #!!!!!!!!!!!!!!!!
    np.random.seed(j**2+13) #fixing random seeds for each Poisson realization
    #!!!!!!!!!!!!!!!!
    r = np.random.rand(yz[0]) * (xz[1] - xz[0])
    Z_poi = r + xz[0]
    for i in range(1, len(xz)-1):
        r1 = np.random.rand(yz[i]) * (xz[i+1] - xz[i])
        r2 = r1 + xz[i]
        Z_poi = np.concatenate((Z_poi, r2)) # Z sequentially increases with bin number, need to reshuffle

    half_3D = int(np.round(len(pos_3D)/2, decimals = 0))
    h_3D = len(pos_3D) - half_3D
    R_poi = cosmo.comoving_distance(Z_poi).value
    np.random.shuffle(R_poi) # Reshuffles R
    #print('max R_poi', np.max(R_poi))

    #Sample should be even in cos(theta) and phi, dOmega = sin(theta)dtheta dphi, dcos(theta) = sin(theta)dtheta
    cos_theta_poi1 = np.random.rand(half_3D) * 1/2 + 1/2 # uniform random from 1/2 to 1
    cos_theta_poi2 = -(np.random.rand(half_3D) * 1/2 + 1/2) # uniform random from -1 to -1/2
    Theta_poi1 = np.arccos(cos_theta_poi1) # into radians
    Theta_poi2 = np.arccos(cos_theta_poi2) # into radians
    Theta_poi = np.concatenate((Theta_poi1, Theta_poi2)) # Radians (0, 3/pi) | (2/3pi, pi)
    Phi_poi = np.random.rand(len(Theta_poi)) * 2 * np.pi # Radians (0, 2pi)
    
    X = np.vstack((Phi_poi, Theta_poi)).T

    vol_g, Xi_poi, xi_poi = VolumekNN(X, Xq, k=10) # distance to kNN for Planck SZ
    np.save(baseDir+'/haver/Poi/Dis_g_Y500_{}.npy'.format(j), vol_g) # Y500 > 3e-3
    np.save(baseDir+'/haver/Poi/Xi_poi_DD_{}.npy'.format(j), Xi_poi) # Y500 > 3e-3
    np.save(baseDir+'/haver/Poi/Xi_poi_DR_{}.npy'.format(j), xi_poi) # Y500 > 3e-3

##################
