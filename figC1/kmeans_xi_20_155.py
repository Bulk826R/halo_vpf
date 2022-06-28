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
import kmeans_radec
from kmeans_radec import KMeans, kmeans_sample
cosmo = FlatLambdaCDM(H0=67.74, Om0=0.3089)
h = 0.6774
baseDir = '/home/bulk826/Desktop/Stanford/Research3/data'
np.random.seed(24)
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

nthreads = 4
N = len(pos_3D)
X = pos_3D[:, 0] + boxsize # [0, 2*boxsize]
Y = pos_3D[:, 1] + boxsize # [0, 2*boxsize]
Z = pos_3D[:, 2] + boxsize # [0, 2*boxsize]

print(X[:5], Y[:5], Z[:5])

##################################
#randoms
header_list = fits.open(baseDir+'/redMapper/redmapper_dr8_public_v6.3_randoms.fits')[1].data
hdul = fits.open(baseDir+'/redMapper/redmapper_dr8_public_v6.3_randoms.fits')
hdr = hdul[1].columns
#print(hdr)
rand = fits.open(baseDir+'/redMapper/redmapper_dr8_public_v6.3_randoms.fits')[1].data

Zr = rand['Z']
ra_r = rand['RA']
#idz = np.where((Zr >= zmin) & (Zr <= zmax)  & (ra_r > 90) & (ra_r < 270))[0]
idz = np.where((Zr >= zmin) & (Zr <= zmax))[0]

RAr = rand['RA'][idz]
DECr = rand['DEC'][idz]
Zr = rand['Z'][idz]
Rich_r = rand['LAMBDA'][idz]
Weight_r = rand['WEIGHT'][idz]

idr = np.where(Rich_r >= np.min(rich))[0]
z_r = Zr[idr]
rich_r = Rich_r[idr]
RA_r = RAr[idr]
DEC_r = DECr[idr]
R_r = cosmo.comoving_distance(z_r).value
Weight_r = Weight_r[idr]


t = 100 # number density ratio of R to D
id_RR = np.random.choice(len(z_r), t*lenz, replace = False)
R_poi = R_r[id_RR]
W_r = Weight_r[id_RR]
Theta_poi = (90-DEC_r[id_RR])/180*np.pi
Phi_poi = RA_r[id_RR]/180*np.pi
R_3D = get_xyz(R_poi, Theta_poi, Phi_poi)
print('R_3D = ', R_3D)

rand_X = R_3D[:, 0] + boxsize
rand_Y = R_3D[:, 1] + boxsize
rand_Z = R_3D[:, 2] + boxsize
rand_N = len(rand_X)

#####################################################
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
len_bins = 51

#bins = np.logspace(1., 2.3, 14)
bins = np.logspace(np.log10(20), np.log10(155), len_bins)
binc = np.sqrt(bins[:-1] * bins[1:])

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#####################################################

#bins = np.linspace(40, 200, 20)
autocorr=1
DD_counts = DD(autocorr, nthreads, bins, X, Y, Z, periodic=False, verbose=True)


autocorr=0
DR_counts = DD(autocorr, nthreads, bins, X, Y, Z,
               X2=rand_X, Y2=rand_Y, Z2=rand_Z,
               periodic=False, verbose=True, weights2 = W_r)

autocorr=1
RR_counts = DD(autocorr, nthreads, bins, rand_X, rand_Y, rand_Z,
               periodic=False, verbose=True, weights1 = W_r)

cf_sz = convert_3d_counts_to_cf(N, N, rand_N, rand_N,
                             DD_counts, DR_counts,
                             DR_counts, RR_counts)

################################################################
#Jackknife module

RA_DEC = np.vstack((RA_l, DEC_l)).T
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

idkm = np.where(km.labels != 0)[0]

autocorr=1
DD_counts = DD(autocorr, nthreads, bins, X[idkm], Y[idkm], Z[idkm], periodic=False, verbose=True)

autocorr=0
DR_counts = DD(autocorr, nthreads, bins, X[idkm], Y[idkm], Z[idkm],
               X2=rand_X, Y2=rand_Y, Z2=rand_Z,
               periodic=False, verbose=True, weights2 = W_r)

autocorr=1
RR_counts = DD(autocorr, nthreads, bins, rand_X, rand_Y, rand_Z,
               periodic=False, verbose=True, weights1 = W_r)

cf_jack = [convert_3d_counts_to_cf(len(idkm), len(idkm), rand_N, rand_N,
                                   DD_counts, DR_counts,
                                   DR_counts, RR_counts)]

for i in range(1, ncen):
    idkm = np.where(km.labels != i)[0]
    xx = X[idkm]
    yy = Y[idkm]
    zz = Z[idkm]
    
    autocorr=1
    DD_counts = DD(autocorr, nthreads, bins, xx, yy, zz, periodic=False, verbose=True)

    autocorr=0
    DR_counts = DD(autocorr, nthreads, bins, xx, yy, zz,
                   X2=rand_X, Y2=rand_Y, Z2=rand_Z,
                   periodic=False, verbose=True, weights2 = W_r)

    autocorr=1
    RR_counts = DD(autocorr, nthreads, bins, rand_X, rand_Y, rand_Z,
                   periodic=False, verbose=True, weights1 = W_r)

    cf_j = convert_3d_counts_to_cf(len(idkm), len(idkm), rand_N, rand_N,
                                   DD_counts, DR_counts,
                                   DR_counts, RR_counts)
    
    cf_jack = np.concatenate((cf_jack, [cf_j]), axis = 0)
    
CJ_med = []
CJ_low = []
CJ_high = []
CJ_mean = []
CJ_std = []
CJ_var = []
for i in range(len(cf_jack[0])):
    #id_fin = np.where(np.isfinite(cf_poi[:, i]) == 1)[0]
    med = np.percentile(cf_jack[:, i], 50)
    low = np.percentile(cf_jack[:, i], 2.3)
    high = np.percentile(cf_jack[:, i], 97.7)
    mean = np.mean(cf_jack[:, i])
    std = np.std(cf_jack[:, i])
    var = np.sum((cf_jack[:, i] - cf_sz[i])**2) * (ncen - 1)/ncen
    CJ_med.append(med)
    CJ_low.append(low)
    CJ_high.append(high)
    CJ_mean.append(mean)
    CJ_std.append(std)
    CJ_var.append(var)

CJ_med = np.asarray(CJ_med)
CJ_low = np.asarray(CJ_low)
CJ_high = np.asarray(CJ_high)
CJ_mean = np.asarray(CJ_mean)
CJ_std = np.asarray(CJ_std)
CJ_var = np.asarray(np.sqrt(CJ_var))

id_jack = np.where(np.isfinite(CJ_mean) == 1)[0]
CJ_med = CJ_med[id_jack]
CJ_low = CJ_low[id_jack]
CJ_high = CJ_high[id_jack]
CJ_mean = CJ_mean[id_jack]
CJ_std = CJ_std[id_jack]
CJ_var = CJ_var[id_jack]    

#################################################################
id_r = np.random.choice(len(z_r), lenz, replace = False)
R_poi = R_r[id_r]
W_poi = Weight_r[id_r]
Theta_poi = (90-DEC_r[id_r])/180*np.pi
Phi_poi = RA_r[id_r]/180*np.pi
poi_3D = get_xyz(R_poi, Theta_poi, Phi_poi)

id_RR = np.random.choice(len(z_r), t*lenz, replace = False)
R_poi = R_r[id_RR]
W_r = Weight_r[id_RR]
Theta_poi = (90-DEC_r[id_RR])/180*np.pi
Phi_poi = RA_r[id_RR]/180*np.pi
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
               periodic=False, verbose=True, weights1 = W_poi)

autocorr=0
DR_counts = DD(autocorr, nthreads, bins, X, Y, Z,
               X2=rand_X, Y2=rand_Y, Z2=rand_Z,
               periodic=False, verbose=True, weights1 = W_poi, weights2 = W_r)

autocorr=1
RR_counts = DD(autocorr, nthreads, bins, rand_X, rand_Y, rand_Z,
               periodic=False, verbose=True, weights1 = W_r)

cf_poi = [convert_3d_counts_to_cf(N, N, rand_N, rand_N,
                             DD_counts, DR_counts,
                             DR_counts, RR_counts)]

len_poi = 2000 # Number of Poisson realizations
for j in range(1, len_poi):
    
    id_r = np.random.choice(len(z_r), lenz, replace = False)
    R_poi = R_r[id_r]
    W_poi = Weight_r[id_r]
    Theta_poi = (90-DEC_r[id_r])/180*np.pi
    Phi_poi = RA_r[id_r]/180*np.pi
    poi_3D = get_xyz(R_poi, Theta_poi, Phi_poi)
 
    N = len(poi_3D)
    X = poi_3D[:, 0] + boxsize
    Y = poi_3D[:, 1] + boxsize
    Z = poi_3D[:, 2] + boxsize
    
    id_RR = np.random.choice(len(z_r), t*lenz, replace = False)
    R_poi = R_r[id_RR]
    W_r = Weight_r[id_RR]
    Theta_poi = (90-DEC_r[id_RR])/180*np.pi
    Phi_poi = RA_r[id_RR]/180*np.pi
    po_3D = get_xyz(R_poi, Theta_poi, Phi_poi)

    rand_X = po_3D[:, 0] + boxsize
    rand_Y = po_3D[:, 1] + boxsize
    rand_Z = po_3D[:, 2] + boxsize
    rand_N = len(rand_X)
    
    #nbins = 21
    #bins = np.logspace(1, 3, nbins + 1) # note that +1 to nbins
    autocorr=1
    DD_counts = DD(autocorr, nthreads, bins, X, Y, Z,
                   periodic=False, verbose=True, weights1 = W_poi)

    autocorr=0
    DR_counts = DD(autocorr, nthreads, bins, X, Y, Z,
                   X2=rand_X, Y2=rand_Y, Z2=rand_Z,
                   periodic=False, verbose=True, weights1 = W_poi, weights2 = W_r)

    autocorr=1
    RR_counts = DD(autocorr, nthreads, bins, rand_X, rand_Y, rand_Z,
                   periodic=False, verbose=True, weights1 = W_r)

    cf = convert_3d_counts_to_cf(N, N, rand_N, rand_N,
                                 DD_counts, DR_counts,
                                 DR_counts, RR_counts)
    cf_poi = np.concatenate((cf_poi, [cf]), axis = 0)

#print('cf = ', cf_poi)

CF_med = []
CF_low = []
CF_high = []
CF_mean = []
CF_std = []
for i in range(len(cf_poi[0])):
    #id_fin = np.where(np.isfinite(cf_poi[:, i]) == 1)[0]
    med = np.percentile(cf_poi[:, i], 50)
    low = np.percentile(cf_poi[:, i], 15.85)
    high = np.percentile(cf_poi[:, i], 84.15)
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

dCF_low = (CJ_med-CJ_var - CF_mean)/CF_std
dCF_high = (CJ_med+CJ_var - CF_mean)/CF_std

##################
#Module for calculating covariance matrix

def update_covmat(mean_data, data_i):
    mat = np.zeros((mean_data.shape[0],mean_data.shape[0]))
    #print (mat.shape, mean_data.shape, data_i.shape)
    for i in range(mean_data.shape[0]):
        for j in range(mean_data.shape[0]):
            mat[i,j] = (data_i[i] - mean_data[i])*(data_i[j]-mean_data[j])
    return mat

#cov_mat = np.zeros((len(bins[:-1][id_poi]),len(bins[:-1][id_poi])))
cov_mat = np.zeros((len(bins[:-1]),len(bins[:-1])))
for i in range(len_poi):
    cov_mat += update_covmat(CF_mean, cf_poi[i,:])

cov_mat /= len_poi
diag = np.zeros(len(bins[:-1]))
for i in range(len(bins[:-1])):
    diag[i] = np.sqrt(cov_mat[i,i])
    
inv_covmat = (len_poi - len(bins[:-1]) -2)/(len_poi - 1) * np.linalg.inv(cov_mat)
#print('inv_cov = ', inv_covmat)
T1 = np.dot(inv_covmat, (cf_sz[id_cf] - CF_mean))
T2 = np.dot(cf_sz[id_cf] - CF_mean, T1)
data_chi2 = T2
rand_chi2 = np.zeros(len_poi)
for i in range(len_poi):
    T1 = np.dot(inv_covmat, (cf_poi[i] - CF_mean))
    T2 = np.dot(cf_poi[i]-CF_mean, T1)
    rand_chi2[i] = T2
    
print('Data chi = ', data_chi2)
#print('Rand chi = ', rand_chi2)

mask = np.where(rand_chi2>data_chi2)
p = len(rand_chi2[mask])/len(rand_chi2)
#print(r'$p$-value = ', p)
jack_chi2 = []
for i in range(len(cf_jack)):
    #t1 = np.dot(inv_covmat, ((cf_jack[i, id_cf] - cf_sz[id_cf]) * np.sqrt(ncen - 1) + cf_sz[id_cf] - CF_mean))
    #t2 = np.dot(((cf_jack[i, id_cf] - cf_sz[id_cf]) * np.sqrt(ncen - 1) + cf_sz[id_cf] - CF_mean), t1)
    t1 = np.dot(inv_covmat, ((cf_jack[i] - CJ_med) * np.sqrt(ncen - 1) + CJ_med - CF_med))
    t2 = np.dot(((cf_jack[i] - CJ_med) * np.sqrt(ncen - 1) + CJ_med - CF_med), t1)
    jack_chi2.append(t2)
    
print((cf_jack[100, id_cf] - cf_sz[id_cf])*(ncen-1))
print((cf_sz[id_cf] - CF_mean)*2*np.sqrt(ncen-1))
jack_chi2 = np.asarray(jack_chi2)

##################

fig.set_size_inches(8, 14)
plt.subplots_adjust(wspace = 0.24, hspace = 0.24)

ax1 = fig.add_subplot(2, 1, 1)
#!!!!!!!!!!!!!!!!!
ax1.scatter(binc[id_cf], cf_sz[id_cf], color = 'crimson', s = 82, lw = 0.8, 
            facecolors = 'None', marker = 'D', rasterized = True, alpha = 0.8, 
            label = r'redMaPPer')
#!!!!!!!!!!!!!!!!!
#############
#Plot Jackknife
ax1.plot(binc[id_jack], CJ_mean, lw = 1.6, alpha = 0.8, color = 'crimson', 
         linestyle = '-.')
ax1.fill_between(binc[id_jack], CJ_mean-CJ_var, CJ_mean+CJ_var, alpha = 0.3, 
                 color = 'crimson', label = r'Jackknife error')
#############

ax1.plot(binc[id_poi], CF_med, lw = 1.6, alpha = 0.8, color = 'royalblue', 
         dashes = (5, 2.4), label = r'Randoms median')
ax1.fill_between(binc[id_poi], CF_low, CF_high, alpha = 0.3, 
                 color = 'deepskyblue', label = r'Randoms $1\,\sigma$')

ax1.set_xlabel(r'$r\,$[$\mathrm{Mpc}$]', fontsize = 24, labelpad = 8)
ax1.set_ylabel(r'$\xi(r)$', fontsize = 24)
ax1.legend(loc = 'upper right', fontsize = 16, frameon = False, borderpad = 0.5)
ax1.set_xscale('log')
ax1.set_xticks(np.logspace(2, 2, 1))
minorLocator = AutoMinorLocator()
loc1 = LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0), numdecs=10, numticks = 10)
ax1.xaxis.set_minor_locator(loc1)
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

ax1 = fig.add_subplot(2, 1, 2)
Nbins = 14
ax1.hist(rand_chi2, bins = Nbins, color = 'royalblue', edgecolor = 'black', 
         alpha = 0.4, linewidth = 1.2, density=True,
         label = r'Randoms ({})'.format(len_poi))

print(jack_chi2)
ax1.axvline(x = data_chi2, lw = 2.4, color = 'crimson', alpha = 0.8, 
            dashes = (5, 2.4), label = r'redMaPPer')

ax1.set_xlabel(r'$\chi^{2}$', fontsize = 24)
ax1.set_ylabel(r'$\mathrm{PDF}(\xi)$', fontsize = 24)
ax1.legend(loc = 'upper center', fontsize = 16, frameon = False, borderpad = 0.5)

minorLocator = AutoMinorLocator()
ax1.xaxis.set_minor_locator(minorLocator)
minorLocator = AutoMinorLocator()
ax1.yaxis.set_minor_locator(minorLocator)
ax1.tick_params(which='major', labelsize = 18, width = 1., length = 6, direction='in', pad = 4, bottom = True, top = True, left = True, right = True)
ax1.tick_params(which='minor', labelsize = 18, width = 1., length = 3, direction='in', bottom = True, top = True, left = True, right = True)
#ax1.set_xlim([20, 320])

xmin, xmax = ax1.get_xlim()
ymin, ymax = ax1.get_ylim()
width = xmax - xmin
height = ymax - ymin

######

fig.savefig('/home/bulk826/Desktop/Stanford/Research3/figures/paper/draft/kmean_xi_20_155.pdf', dpi = 400, bbox_inches = 'tight')
plt.show()