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
from matplotlib.ticker import AutoMinorLocator, LogLocator
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from astropy.coordinates import SkyCoord
cosmo = FlatLambdaCDM(H0=67.74, Om0=0.3089)
h = 0.6774
baseDir = '/oak/stanford/orgs/kipac/users/ycwang19/VPF/SZ'

################################

def VolumekNN(xin, xout, k=1, periodic = 0):
    if isinstance(k, int): k = [k] # 
    dim = xin.shape[1] # dimension of every row
    #Ntot = xin.shape[0] # dimension of entries (length of column)
    xtree = scipy.spatial.cKDTree(xin, boxsize=periodic)
    #print('k = ', k)
    dis, disi = xtree.query(xout, k=k, n_jobs=8) # dis is the distance to the kth nearest neighbour, disi is the id of that neighbour
    vol = np.empty_like(dis) # same shape as distance including all k values
    Cr = [2, np.pi, 4 * np.pi / 3, np.pi**2, 8*np.pi**2/15][dim - 1]  # Volume prefactor for 1,2, 3D
    for c, k in enumerate(np.nditer(np.array(k))):
        #print('c, dim, dis = ', c, dim, dis[:, c]**dim / k)
        vol[:, c] = Cr * dis[:, c]**dim / k # the overdense average volume per point in sphere
        #print('vol = ', vol[:, c])
    return vol

def CDFVolkNN(vol): # CDF
    CDF = []
    N = vol.shape[0]
    l = vol.shape[1]
    gof = ((np.arange(0, N) + 1) / N)
    for c in range(l):
        ind = np.argsort(vol[:, c])
        sVol= vol[ind, c]
        # return array of interpolating functions
        CDF.append(interpolate.interp1d(sVol, gof, kind = 'linear', \
                                        bounds_error=False)) # x = sVol, y = gof
    return CDF

def CICkNN(xin, xout, kmax, vol, periodic = 0):
    CDF = []
    dim = xin.shape[1]
    Ntot = xin.shape[0]
    kneed = np.arange(1, kmax+2)
    #print("Calculate these neighbors:", kneed)
    vol = VolumekNN(xin, xout, kneed, periodic = periodic)
    
    for i in range(1, kmax+1):
        vol[:, i] = vol[:, i] * (i+1)  # need volume not specific volume
    CDFs = CDFVolkNN(vol)
    binc = np.logspace(np.log10(np.min(vol)), np.log10(np.max(vol)), 5000)
    CIC = []
    for kst in range(0, kmax):
        vols = binc # CDFs[kst+1].x
        tCIC = CDFs[kst](vols)-CDFs[kst+1](vols)
        tCIC[np.isnan(tCIC)] = 0.
        CIC.append(interpolate.interp1d(vols,tCIC,  kind='linear', \
                                        bounds_error=False, fill_value=(0.,0.)))
    
    # Add the void probability function as the zero counts in cell statistic
    dummyvpf = 1. - CDFs[0](binc)
    dummyvpf[np.isnan(dummyvpf)] = 0
    VPF = interpolate.interp1d(binc, dummyvpf, kind='linear', bounds_error=False, fill_value=(1.,0.))
    CIC = np.insert(CIC, 0, VPF)
    return vol, CDFs, CIC

################################

bine = np.logspace(1.5, 3.0, 51)
binw = bine[1:] - bine[:-1]
binc1 = (bine[1:] + bine[:-1]) / 2

bine = np.logspace(1.8, 3.05, 51)
binw = bine[1:] - bine[:-1]
binc2 = (bine[1:] + bine[:-1]) / 2

bine = np.logspace(1.95, 3.1, 51)
binw = bine[1:] - bine[:-1]
binc4 = (bine[1:] + bine[:-1]) / 2

bine = np.logspace(2.05, 3.15, 51)
binw = bine[1:] - bine[:-1]
binc8 = (bine[1:] + bine[:-1]) / 2

vol_h = np.loadtxt(baseDir+'/Dis_h_3D_Y500.txt')
#vol_h = np.loadtxt(baseDir+'/Vol_h_3D_Y500c.txt')
CDFs_h = CDFVolkNN(vol_h)
dummyvpf_h = 1.-CDFs_h[0](binc1)
dummyvpf_h[np.isnan(dummyvpf_h)] = 0
VPF_h = interpolate.interp1d(binc1, dummyvpf_h, kind='linear', bounds_error=False, fill_value=(1.,0.))
CDF_1h = interpolate.interp1d(binc1, CDFs_h[0](binc1), kind='linear', bounds_error=False, fill_value=(0.,1.))
CDF_5h = interpolate.interp1d(binc4, CDFs_h[1](binc4), kind='linear', bounds_error=False, fill_value=(0.,1.))
CDF_10h = interpolate.interp1d(binc8, CDFs_h[2](binc8), kind='linear', bounds_error=False, fill_value=(0.,1.))
CDF_50h = interpolate.interp1d(binc8, CDFs_h[3](binc8), kind='linear', bounds_error=False, fill_value=(0.,1.))
CDF_2h = interpolate.interp1d(binc2, CDFs_h[4](binc2), kind='linear', bounds_error=False, fill_value=(0.,1.))
CDF_3h = interpolate.interp1d(binc2, CDFs_h[5](binc2), kind='linear', bounds_error=False, fill_value=(0.,1.))
CDF_4h = interpolate.interp1d(binc4, CDFs_h[6](binc4), kind='linear', bounds_error=False, fill_value=(0.,1.))
CDF_8h = interpolate.interp1d(binc8, CDFs_h[7](binc8), kind='linear', bounds_error=False, fill_value=(0.,1.))
CDF_5h = interpolate.interp1d(binc4, CDFs_h[8](binc4), kind='linear', bounds_error=False, fill_value=(0.,1.))
CDF_9h = interpolate.interp1d(binc8, CDFs_h[9](binc8), kind='linear', bounds_error=False, fill_value=(0.,1.))

Yd1 = CDF_1h(binc1)
Yd2 = CDF_2h(binc2)
Yd4 = CDF_4h(binc4)
Yd8 = CDF_8h(binc8)
#print('Yd = ', Yd)

vol_g = np.load(baseDir+'/Poi/Dis_g_Y500_0.npy')
#vol_g = np.load(baseDir+'/Poi/Vol_g_Y500c_0.npy')
CDFs_g = CDFVolkNN(vol_g)
dummyvpf_g = 1.-CDFs_g[0](binc1)
dummyvpf_g[np.isnan(dummyvpf_g)] = 0
VPF_g = interpolate.interp1d(binc1, dummyvpf_g, kind='linear', bounds_error=False, fill_value=(1., 0.))
CDF_1g = interpolate.interp1d(binc1, CDFs_g[0](binc1), kind='linear', bounds_error=False, fill_value=(0.,1.))
CDF_5g = interpolate.interp1d(binc4, CDFs_g[1](binc4), kind='linear', bounds_error=False, fill_value=(0.,1.))
CDF_10g = interpolate.interp1d(binc8, CDFs_g[2](binc8), kind='linear', bounds_error=False, fill_value=(0.,1.))
CDF_50g = interpolate.interp1d(binc8, CDFs_g[3](binc8), kind='linear', bounds_error=False, fill_value=(0.,1.))
CDF_2g = interpolate.interp1d(binc2, CDFs_g[4](binc2), kind='linear', bounds_error=False, fill_value=(0.,1.))
CDF_3g = interpolate.interp1d(binc2, CDFs_g[5](binc2), kind='linear', bounds_error=False, fill_value=(0.,1.))
CDF_4g = interpolate.interp1d(binc4, CDFs_g[6](binc4), kind='linear', bounds_error=False, fill_value=(0.,1.))
CDF_8g = interpolate.interp1d(binc8, CDFs_g[7](binc8), kind='linear', bounds_error=False, fill_value=(0.,1.))
CDF_5g = interpolate.interp1d(binc4, CDFs_g[8](binc4), kind='linear', bounds_error=False, fill_value=(0.,1.))
CDF_9g = interpolate.interp1d(binc8, CDFs_g[9](binc8), kind='linear', bounds_error=False, fill_value=(0.,1.))


Yp1 = [CDF_1g(binc1)]
Yp2 = [CDF_2g(binc2)]
Yp4 = [CDF_4g(binc4)]
Yp8 = [CDF_8g(binc8)]
print(Yp1)
print(Yp2)
print(Yp4)
print(Yp8)
len_poi = 2000
for i in range(1, len_poi):
    print('i = ', i)
    vol_g = np.load(baseDir+'/Poi/Dis_g_Y500_{}.npy'.format(i))
    #vol_g = np.load(baseDir+'/Poi/Vol_g_Y500c_{}.npy'.format(i))
    CDFs_g = CDFVolkNN(vol_g)
    dummyvpf_g = 1.-CDFs_g[0](binc1)
    dummyvpf_g[np.isnan(dummyvpf_g)] = 0
    VPF_g = interpolate.interp1d(binc1, dummyvpf_g, kind='linear', bounds_error=False, fill_value=(1., 0.))
    CDF_1g = interpolate.interp1d(binc1, CDFs_g[0](binc1), kind='linear', bounds_error=False, fill_value=(0.,1.))
    CDF_5g = interpolate.interp1d(binc4, CDFs_g[1](binc4), kind='linear', bounds_error=False, fill_value=(0.,1.))
    CDF_10g = interpolate.interp1d(binc8, CDFs_g[2](binc8), kind='linear', bounds_error=False, fill_value=(0.,1.))
    CDF_50g = interpolate.interp1d(binc8, CDFs_g[3](binc8), kind='linear', bounds_error=False, fill_value=(0.,1.))
    CDF_2g = interpolate.interp1d(binc2, CDFs_g[4](binc2), kind='linear', bounds_error=False, fill_value=(0.,1.))
    CDF_3g = interpolate.interp1d(binc2, CDFs_g[5](binc2), kind='linear', bounds_error=False, fill_value=(0.,1.))
    CDF_4g = interpolate.interp1d(binc4, CDFs_g[6](binc4), kind='linear', bounds_error=False, fill_value=(0.,1.))
    CDF_8g = interpolate.interp1d(binc8, CDFs_g[7](binc8), kind='linear', bounds_error=False, fill_value=(0.,1.))
    CDF_5g = interpolate.interp1d(binc4, CDFs_g[8](binc4), kind='linear', bounds_error=False, fill_value=(0.,1.))
    CDF_9g = interpolate.interp1d(binc8, CDFs_g[9](binc8), kind='linear', bounds_error=False, fill_value=(0.,1.))

    yp1 = CDF_1g(binc1)
    yp2 = CDF_2g(binc2)
    yp4 = CDF_4g(binc4)
    yp8 = CDF_8g(binc8)
    Yp1 = np.concatenate((Yp1, [yp1]), axis = 0)
    Yp2 = np.concatenate((Yp2, [yp2]), axis = 0)
    Yp4 = np.concatenate((Yp4, [yp4]), axis = 0)
    Yp8 = np.concatenate((Yp8, [yp8]), axis = 0)

Mp1 = []
Mp2 = []
Mp4 = []
Mp8 = []
STD1 = []
STD2 = []
STD4 = []
STD8 = []
for i in range(len(binc1)):
    y1 = Yp1[:, i]
    idy1 = np.where(np.isfinite(y1) == 1)[0]
    Mp1.append(np.mean(y1[idy1]))
    STD1.append(np.std(y1[idy1]))

for i in range(len(binc2)):   
    y2 = Yp2[:, i]
    idy2 = np.where(np.isfinite(y2) == 1)[0]
    Mp2.append(np.mean(y2[idy2]))
    STD2.append(np.std(y2[idy2]))

for i in range(len(binc4)):
    y4 = Yp4[:, i]
    idy4 = np.where(np.isfinite(y4) == 1)[0]
    Mp4.append(np.mean(y4[idy4]))
    STD4.append(np.std(y4[idy4]))
    
for i in range(len(binc8)):
    y8 = Yp8[:, i]
    idy8 = np.where(np.isfinite(y8) == 1)[0]
    Mp8.append(np.mean(y8[idy8]))
    STD8.append(np.std(y8[idy8]))

Mp1 = np.asarray(Mp1)
STD1 = np.asarray(STD1)
id_fin1 = np.where(np.isfinite(Mp1) == 1)[0]
print('Fin 1 = ', len(id_fin1), len(Mp1))
Mp2 = np.asarray(Mp2)
STD2 = np.asarray(STD2)
id_fin2 = np.where(np.isfinite(Mp2) == 1)[0]
print('Fin 2 = ', len(id_fin2), len(Mp2))
Mp4 = np.asarray(Mp4)
STD4 = np.asarray(STD4)
id_fin4 = np.where(np.isfinite(Mp4) == 1)[0]
print('Fin 4 = ', len(id_fin4), len(Mp4))
Mp8 = np.asarray(Mp8)
STD8 = np.asarray(STD8)
id_fin8 = np.where(np.isfinite(Mp8) == 1)[0]
print('Fin 8 = ', len(id_fin8), len(Mp8))

#################################

def update_covmat(mean_data, data_i):
    mat = np.zeros((mean_data.shape[0],mean_data.shape[0]))
    #print (mat.shape, mean_data.shape, data_i.shape)
    for i in range(mean_data.shape[0]):
        for j in range(mean_data.shape[0]):
            if np.isfinite(data_i[i]) == 1 and np.isfinite(data_i[j]) == 1:
                mat[i,j] = (data_i[i] - mean_data[i])*(data_i[j]-mean_data[j])
            else:
                continue
            
    return mat

bins = np.concatenate((binc1[id_fin1], binc2[id_fin2], binc4[id_fin4], binc8[id_fin8]))
CDF_mean = np.concatenate((Mp1[id_fin1], Mp2[id_fin2], Mp4[id_fin4], Mp8[id_fin8]))
CDF_SZ = np.concatenate((Yd1[id_fin1], Yd2[id_fin2], Yd4[id_fin4], Yd8[id_fin8]))
CDF_poi = np.concatenate((Yp1[:, id_fin1], Yp2[:, id_fin2], Yp4[:, id_fin4], Yp8[:, id_fin8]), axis = 1)
cov_mat = np.zeros((len(bins), len(bins)))
#print(CDF_mean, CDF_SZ, CDF_poi)
print('Len bins = ', len(bins))
for i in range(len_poi):
    cov_mat += update_covmat(CDF_mean, CDF_poi[i,:])

print('cov_mat = ', cov_mat)
cov_mat /= len_poi
diag = np.zeros(len(bins))
np.savetxt('/oak/stanford/orgs/kipac/users/ycwang19/VPF/SZ/cdf_cov_{}.txt'.format(len_poi), cov_mat)

for i in range(len(bins)):
    diag[i] = np.sqrt(cov_mat[i,i])
    
inv_covmat = (len_poi - len(bins) -2)/(len_poi - 1) * np.linalg.inv(cov_mat) # Hartlap factor for the inverse
print('inv_cov = ', inv_covmat)
T1 = np.dot(inv_covmat, (CDF_SZ - CDF_mean))
T2 = np.dot(CDF_SZ - CDF_mean, T1)
data_chi2 = T2
#print (T1.shape)

rand_chi2 = np.zeros(len_poi)
for i in range(len_poi):
    T1 = np.dot(inv_covmat, (CDF_poi[i] - CDF_mean))
    T2 = np.dot(CDF_poi[i]-CDF_mean, T1)
    rand_chi2[i] = T2
    
print('Data chi = ', data_chi2)
print('Rand chi = ', rand_chi2)
mask = np.where(rand_chi2>data_chi2)
p = len(rand_chi2[mask])/len(rand_chi2)
print(r'$p$-value = ', p)

#################################################

fig.set_size_inches(8, 6)

ax1 = fig.add_subplot(1, 1, 1)
Nbins = 50
ax1.hist(rand_chi2, bins = Nbins, color = 'royalblue', edgecolor = 'black', 
         alpha = 0.4, linewidth = 1.2,
         label = r'Poisson #:{}'.format(len_poi))

ax1.axvline(x = data_chi2, lw = 2.4, color = 'crimson', alpha = 0.8, dashes = (5, 2.4), label = r'Planck SZ')

ax1.set_xlabel(r'$\chi^{2}$', fontsize = 24)
ax1.set_ylabel(r'$\#\ \mathrm{of}\ \mathrm{CDFs}$', fontsize = 24)
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
ax1.text(xmax - width * 0.04, ymax - height * 0.2, r'$p = $'+'{:.3f}'.format(p), 
         horizontalalignment = 'right', verticalalignment = 'top', fontsize = 16, color = 'black', alpha = 0.8)


fig.savefig('/oak/stanford/orgs/kipac/users/ycwang19/VPF/figures/cov_cdf_{}.png'.format(len_poi), dpi = 400, bbox_inches = 'tight')
plt.show()
