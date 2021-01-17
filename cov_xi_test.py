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
from astropy import units as u
from astropy.coordinates import SkyCoord
cosmo = FlatLambdaCDM(H0=67.74, Om0=0.3089)
h = 0.6774
baseDir = '/home/bulk826/Desktop/Stanford/Research2/mc3-master'

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
#######################

nthreads = 4
N = len(pos_3D)
X = pos_3D[:, 0] + boxsize # [0, 2*boxsize]
Y = pos_3D[:, 1] + boxsize # [0, 2*boxsize]
Z = pos_3D[:, 2] + boxsize # [0, 2*boxsize]

##################

t = 10
r = np.random.rand(t * yz[0]) * (xz[1] - xz[0])
Z_poi = r + xz[0]
for i in range(1, len(xz)-1):
    r1 = np.random.rand(t * yz[i]) * (xz[i+1] - xz[i])
    r2 = r1 + xz[i]
    Z_poi = np.concatenate((Z_poi, r2))
        
np.random.shuffle(Z_poi)
R_poi = cosmo.comoving_distance(Z_poi).value
half_3D = int(t * np.round(len(pos_3D)/2, decimals = 0))
h_3D = t * len(pos_3D) - half_3D
Theta_poi1 = np.random.rand(half_3D) * np.pi/3
Theta_poi2 = np.random.rand(h_3D) * np.pi/3 + 2/3*np.pi
Theta_poi = np.concatenate((Theta_poi1, Theta_poi2))
Phi_poi = np.random.rand(t * len(pos_3D)) * 2 * np.pi
R_3D = get_xyz(R_poi, Theta_poi, Phi_poi)

rand_X = R_3D[:, 0] + boxsize
rand_Y = R_3D[:, 1] + boxsize
rand_Z = R_3D[:, 2] + boxsize
rand_N = len(rand_X)

nbins = 50
#bins = np.logspace(1.2, 3.2, nbins + 1) # note that +1 to nbins
bins = np.logspace(0.8, 3.2, nbins + 1) 
autocorr=1
DD_counts = DD(autocorr, nthreads, bins, X, Y, Z,
               periodic=False, verbose=True)

autocorr=0
DR_counts = DD(autocorr, nthreads, bins, X, Y, Z,
               X2=rand_X, Y2=rand_Y, Z2=rand_Z,
               periodic=False, verbose=True)

autocorr=1
RR_counts = DD(autocorr, nthreads, bins, rand_X, rand_Y, rand_Z,
               periodic=False, verbose=True)

cf_sz = convert_3d_counts_to_cf(N, N, rand_N, rand_N,
                             DD_counts, DR_counts,
                             DR_counts, RR_counts)

#################################################################

r = np.random.rand(1 * yz[0]) * (xz[1] - xz[0])
Z_poi = r + xz[0]
for i in range(1, len(xz)-1):
    r1 = np.random.rand(1 * yz[i]) * (xz[i+1] - xz[i])
    r2 = r1 + xz[i]
    Z_poi = np.concatenate((Z_poi, r2))

np.random.shuffle(Z_poi)
R_poi = cosmo.comoving_distance(Z_poi).value
half_3D = int(1 * np.round(len(pos_3D)/2, decimals = 0))
h_3D = 1 * len(pos_3D) - half_3D
Theta_poi1 = np.random.rand(half_3D) * np.pi/3
Theta_poi2 = np.random.rand(h_3D) * np.pi/3 + 2/3*np.pi
Theta_poi = np.concatenate((Theta_poi1, Theta_poi2))
Phi_poi = np.random.rand(1 * len(pos_3D)) * 2 * np.pi
poi_3D = get_xyz(R_poi, Theta_poi, Phi_poi)

t = 10
r = np.random.rand(t * yz[0]) * (xz[1] - xz[0])
Z_poi = r + xz[0]
for i in range(1, len(xz)-1):
    r1 = np.random.rand(t * yz[i]) * (xz[i+1] - xz[i])
    r2 = r1 + xz[i]
    Z_poi = np.concatenate((Z_poi, r2))
        
np.random.shuffle(Z_poi)
R_poi = cosmo.comoving_distance(Z_poi).value
half_3D = int(t * np.round(len(pos_3D)/2, decimals = 0))
h_3D = t * len(pos_3D) - half_3D
Theta_poi1 = np.random.rand(half_3D) * np.pi/3
Theta_poi2 = np.random.rand(h_3D) * np.pi/3 + 2/3*np.pi
Theta_poi = np.concatenate((Theta_poi1, Theta_poi2))
Phi_poi = np.random.rand(t * len(pos_3D)) * 2 * np.pi
po_3D = get_xyz(R_poi, Theta_poi, Phi_poi)

rand_X = po_3D[:, 0] + boxsize
rand_Y = po_3D[:, 1] + boxsize
rand_Z = po_3D[:, 2] + boxsize
rand_N = len(rand_X)

X = poi_3D[:, 0] + boxsize # [0, 2*boxsize]
Y = poi_3D[:, 1] + boxsize # [0, 2*boxsize]
Z = poi_3D[:, 2] + boxsize # [0, 2*boxsize]

autocorr=1
DD_counts = DD(autocorr, nthreads, bins, X, Y, Z,
               periodic=False, verbose=True)

autocorr=0
DR_counts = DD(autocorr, nthreads, bins, X, Y, Z,
               X2=rand_X, Y2=rand_Y, Z2=rand_Z,
               periodic=False, verbose=True)

autocorr=1
RR_counts = DD(autocorr, nthreads, bins, rand_X, rand_Y, rand_Z,
               periodic=False, verbose=True)

cf_poi = [convert_3d_counts_to_cf(N, N, rand_N, rand_N,
                             DD_counts, DR_counts,
                             DR_counts, RR_counts)]
len_poi = 1000
for i in range(1, len_poi):
    #np.random.seed(i**2+13)
    r = np.random.rand(1 * yz[0]) * (xz[1] - xz[0])
    Z_poi = r + xz[0]
    for i in range(1, len(xz)-1):
        r1 = np.random.rand(1 * yz[i]) * (xz[i+1] - xz[i])
        r2 = r1 + xz[i]
        Z_poi = np.concatenate((Z_poi, r2))
        
    np.random.shuffle(Z_poi)
    R_poi = cosmo.comoving_distance(Z_poi).value
    half_3D = int(1 * np.round(len(pos_3D)/2, decimals = 0))
    h_3D = 1 * len(pos_3D) - half_3D
    Theta_poi1 = np.random.rand(half_3D) * np.pi/3
    Theta_poi2 = np.random.rand(h_3D) * np.pi/3 + 2/3*np.pi
    Theta_poi = np.concatenate((Theta_poi1, Theta_poi2))
    Phi_poi = np.random.rand(1 * len(pos_3D)) * 2 * np.pi
    poi_3D = get_xyz(R_poi, Theta_poi, Phi_poi)
    
    N = len(poi_3D)
    X = poi_3D[:, 0] + boxsize
    Y = poi_3D[:, 1] + boxsize
    Z = poi_3D[:, 2] + boxsize
    
    t = 10    
    r = np.random.rand(t * yz[0]) * (xz[1] - xz[0])
    Z_poi = r + xz[0]
    for i in range(1, len(xz)-1):
        r1 = np.random.rand(t * yz[i]) * (xz[i+1] - xz[i])
        r2 = r1 + xz[i]
        Z_poi = np.concatenate((Z_poi, r2))
        
    np.random.shuffle(Z_poi)
    R_poi = cosmo.comoving_distance(Z_poi).value
    half_3D = int(t * np.round(len(poi_3D)/2, decimals = 0))
    h_3D = t * len(poi_3D) - half_3D
    Theta_poi1 = np.random.rand(half_3D) * np.pi/3
    Theta_poi2 = np.random.rand(h_3D) * np.pi/3 + 2/3*np.pi
    Theta_poi = np.concatenate((Theta_poi1, Theta_poi2))
    Phi_poi = np.random.rand(t * len(poi_3D)) * 2 * np.pi
    po_3D = get_xyz(R_poi, Theta_poi, Phi_poi)

    rand_X = po_3D[:, 0] + boxsize
    rand_Y = po_3D[:, 1] + boxsize
    rand_Z = po_3D[:, 2] + boxsize
    rand_N = len(rand_X)
    
    #nbins = 21
    #bins = np.logspace(1, 3, nbins + 1) # note that +1 to nbins
    autocorr=1
    DD_counts = DD(autocorr, nthreads, bins, X, Y, Z,
                   periodic=False, verbose=True)

    autocorr=0
    DR_counts = DD(autocorr, nthreads, bins, X, Y, Z,
                   X2=rand_X, Y2=rand_Y, Z2=rand_Z,
                   periodic=False, verbose=True)

    autocorr=1
    RR_counts = DD(autocorr, nthreads, bins, rand_X, rand_Y, rand_Z,
                   periodic=False, verbose=True)

    cf = convert_3d_counts_to_cf(N, N, rand_N, rand_N,
                                 DD_counts, DR_counts,
                                 DR_counts, RR_counts)
    cf_poi = np.concatenate((cf_poi, [cf]), axis = 0)

print('cf = ', cf_poi)
cf_poi = cf_poi

CF_med = []
CF_low = []
CF_high = []
CF_mean = []
CF_std = []
for i in range(len(cf_poi[0])):
    #id_fin = np.where(np.isfinite(cf_poi[:, i]) == 1)[0]
    med = np.percentile(cf_poi[:, i], 50)
    low = np.percentile(cf_poi[:, i], 16)
    high = np.percentile(cf_poi[:, i], 84)
    mean = np.mean(cf_poi[:, i])
    std = np.std(cf_poi[:, i])
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

dCF = (cf_sz[id_poi] - CF_mean)/CF_std
id_cf = np.where(np.isfinite(cf_sz) == 1)[0]

##################
#Module for calculating covariance matrix

def update_covmat(mean_data, data_i):
    mat = np.zeros((mean_data.shape[0],mean_data.shape[0]))
    #print (mat.shape, mean_data.shape, data_i.shape)
    for i in range(mean_data.shape[0]):
        for j in range(mean_data.shape[0]):
            mat[i,j] = (data_i[i] - mean_data[i])*(data_i[j]-mean_data[j])
    return mat

cov_mat = np.zeros((len(bins[:-1][id_poi]),len(bins[:-1][id_poi])))
for i in range(len_poi):
    cov_mat += update_covmat(CF_mean, cf_poi[i,:])

cov_mat /= len_poi
diag = np.zeros(len(bins[:-1][id_poi]))
np.savetxt('/home/bulk826/Desktop/Stanford/Research3/figures/SZ/xi_cov_{}.txt'.format(len_poi), cov_mat)
for i in range(len(bins[:-1][id_poi])):
    diag[i] = np.sqrt(cov_mat[i,i])
    
inv_covmat = (len_poi - len(bins[:-1][id_poi]) -2)/(len_poi - 1) * np.linalg.inv(cov_mat)
print('inv_cov = ', inv_covmat)
T1 = np.dot(inv_covmat, (cf_sz[id_cf] - CF_mean))
T2 = np.dot(cf_sz[id_cf] - CF_mean, T1)
data_chi2 = T2
#print (T1.shape)

rand_chi2 = np.zeros(len_poi)
for i in range(len_poi):
    T1 = np.dot(inv_covmat, (cf_poi[i] - CF_mean))
    T2 = np.dot(cf_poi[i]-CF_mean, T1)
    rand_chi2[i] = T2
    

print('Data chi = ', data_chi2)
print('Rand chi = ', rand_chi2)

mask = np.where(rand_chi2>data_chi2)
p = len(rand_chi2[mask])/len(rand_chi2)
print(r'$p$-value = ', p)


fig.set_size_inches(8, 6)

ax1 = fig.add_subplot(1, 1, 1)
Nbins = 50
ax1.hist(rand_chi2, bins = Nbins, color = 'darkturquoise', edgecolor = 'black', 
         alpha = 0.4, linewidth = 1.2,
         label = r'Poisson #:{}'.format(len_poi))

ax1.axvline(x = data_chi2, lw = 2.4, color = 'crimson', alpha = 0.8, dashes = (5, 2.4), label = r'Planck SZ')

ax1.set_xlabel(r'$\chi^{2}$', fontsize = 24)
ax1.set_ylabel(r'$\#\ \mathrm{of}\ \xi(r)$', fontsize = 24)
ax1.legend(loc = 'upper right', fontsize = 16)

minorLocator = AutoMinorLocator()
ax1.xaxis.set_minor_locator(minorLocator)
minorLocator = AutoMinorLocator()
ax1.yaxis.set_minor_locator(minorLocator)
ax1.tick_params(which='major', labelsize = 18, width = 1., length = 6, direction='in', pad = 4, bottom = True, top = True, left = True, right = True)
ax1.tick_params(which='minor', labelsize = 18, width = 1., length = 3, direction='in', bottom = True, top = True, left = True, right = True)


xmin, xmax = ax1.get_xlim()
ymin, ymax = ax1.get_ylim()
width = xmax - xmin
height = ymax - ymin
ax1.text(xmax - width * 0.05, ymax - height * 0.2, r'$p = $'+'{:.3f}'.format(p), 
         horizontalalignment = 'right', verticalalignment = 'top', fontsize = 16, color = 'black', alpha = 0.8)


#fig.savefig('/home/bulk826/Desktop/Stanford/Research3/figures/SZ/cov_xi_{}.png'.format(len_poi), dpi = 200, bbox_inches = 'tight')
fig.savefig('/home/bulk826/Desktop/Stanford/Research3/figures/paper/cov_xi_{}.png'.format(len_poi), dpi = 400, bbox_inches = 'tight')
plt.show()