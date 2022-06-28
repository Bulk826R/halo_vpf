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
baseDir = '/oak/stanford/orgs/kipac/users/ycwang19/VPF/SZ/redMapper'

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


################################

#30 140
bine = np.logspace(np.log10(35), np.log10(135), 26)
binw = bine[1:] - bine[:-1]
binc1 = (bine[1:] + bine[:-1]) / 2
#50 160
bine = np.logspace(np.log10(55), np.log10(155), 26)
binw = bine[1:] - bine[:-1]
binc2 = (bine[1:] + bine[:-1]) / 2

vol_h = np.loadtxt(baseDir+'/Dis_h_3D_Y500.txt') # Reads in r kNN data
CDFs_h = CDFVolkNN(vol_h)
CDF_1h = interpolate.interp1d(binc1, CDFs_h[0](binc1), kind='linear', bounds_error=False, fill_value=(0.,1.))
CDF_2h = interpolate.interp1d(binc2, CDFs_h[4](binc2), kind='linear', bounds_error=False, fill_value=(0.,1.))
Yd1 = CDF_1h(binc1)
Yd2 = CDF_2h(binc2)

vol_g = np.load(baseDir+'/Poi/Dis_g_Y500_0.npy') # Reads in r kNN data
CDFs_g = CDFVolkNN(vol_g)
CDF_1g = interpolate.interp1d(binc1, CDFs_g[0](binc1), kind='linear', bounds_error=False, fill_value=(0.,1.))
CDF_2g = interpolate.interp1d(binc2, CDFs_g[4](binc2), kind='linear', bounds_error=False, fill_value=(0.,1.))

Yp1 = [CDF_1g(binc1)]
Yp2 = [CDF_2g(binc2)]
print(Yp1)
print(Yp2)
len_poi = 2000
for i in range(1, len_poi):
    print('i = ', i)
    vol_g = np.load(baseDir+'/Poi/Dis_g_Y500_{}.npy'.format(i))
    CDFs_g = CDFVolkNN(vol_g)
    dummyvpf_g = 1.-CDFs_g[0](binc1)
    dummyvpf_g[np.isnan(dummyvpf_g)] = 0
    VPF_g = interpolate.interp1d(binc1, dummyvpf_g, kind='linear', bounds_error=False, fill_value=(1., 0.))
    CDF_1g = interpolate.interp1d(binc1, CDFs_g[0](binc1), kind='linear', bounds_error=False, fill_value=(0.,1.))
    CDF_2g = interpolate.interp1d(binc2, CDFs_g[4](binc2), kind='linear', bounds_error=False, fill_value=(0.,1.))
    yp1 = CDF_1g(binc1)
    yp2 = CDF_2g(binc2)
    Yp1 = np.concatenate((Yp1, [yp1]), axis = 0)
    Yp2 = np.concatenate((Yp2, [yp2]), axis = 0)

Mp1 = []
Mp2 = []
STD1 = []
STD2 = []
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

Mp1 = np.asarray(Mp1)
STD1 = np.asarray(STD1)
id_fin1 = np.where(np.isfinite(Mp1) == 1)[0]
print('Fin 1 = ', len(id_fin1), len(Mp1))
Mp2 = np.asarray(Mp2)
STD2 = np.asarray(STD2)
id_fin2 = np.where(np.isfinite(Mp2) == 1)[0]
print('Fin 2 = ', len(id_fin2), len(Mp2))

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

bins = np.concatenate((binc1[id_fin1], binc2[id_fin2]))
CDF_mean = np.concatenate((Mp1[id_fin1], Mp2[id_fin2]))
CDF_SZ = np.concatenate((Yd1[id_fin1], Yd2[id_fin2]))
CDF_poi = np.concatenate((Yp1[:, id_fin1], Yp2[:, id_fin2]), axis = 1)
cov_mat = np.zeros((len(bins), len(bins)))
print('Len bins = ', len(bins))
for i in range(len_poi):
    idf = np.where(np.isfinite(CDF_poi[i])==1)[0]
    cov_mat += update_covmat(CDF_mean[idf], CDF_poi[i,idf])
    
print('cov_mat = ', cov_mat)
cov_mat /= len_poi
diag = np.zeros(len(bins))
np.savetxt('/oak/stanford/orgs/kipac/users/ycwang19/VPF/SZ/redMapper/cdf_cov_12_{}.txt'.format(len_poi), cov_mat)

w, v = np.linalg.eig(cov_mat)
#print('C inverse Eigenvalues, Eigenvectors = ', w, v)
print('Cov eigen vals = ', w)
print('Max, min eig vals of Cov = ', np.max(w), np.min(w))

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

np.savetxt('/oak/stanford/orgs/kipac/users/ycwang19/VPF/SZ/redMapper/cdf_cov_{}_12.txt'.format(len_poi), cov_mat)
np.savetxt('/oak/stanford/orgs/kipac/users/ycwang19/VPF/SZ/redMapper/CDF_SZ_12.txt', CDF_SZ)
np.savetxt('/oak/stanford/orgs/kipac/users/ycwang19/VPF/SZ/redMapper/CDF_mean_12.txt', CDF_mean)
np.savetxt(baseDir+'/rand_chi2_{}_12.txt'.format(len_poi), rand_chi2)

#################################################

fig.set_size_inches(8, 6)

ax1 = fig.add_subplot(1, 1, 1)
Nbins = 50
ax1.hist(rand_chi2, bins = Nbins, color = 'royalblue', edgecolor = 'black', 
         alpha = 0.4, linewidth = 1.2, density=True,
         label = r'Randoms ({})'.format(len_poi))

ax1.axvline(x = data_chi2, lw = 2.4, color = 'crimson', alpha = 0.8, dashes = (5, 2.4), label = r'redMaPPer')

vol_h = np.loadtxt(baseDir+'/Dis_jack_{}.txt'.format(0)) # Reads in r kNN data
CDFs_h = CDFVolkNN(vol_h)
CDF_1h = interpolate.interp1d(binc1, CDFs_h[0](binc1), kind='linear', bounds_error=False, fill_value=(0.,1.))
CDF_2h = interpolate.interp1d(binc2, CDFs_h[4](binc2), kind='linear', bounds_error=False, fill_value=(0.,1.))

Yj1 = CDF_1h(binc1)
Yj2 = CDF_2h(binc2)
CDF_jack = [np.concatenate((Yj1[id_fin1], Yj2[id_fin2]))]
jack = 200
for i in range(jack):
    vol_h = np.loadtxt(baseDir+'/Dis_jack_{}.txt'.format(i)) # Reads in r kNN data

    CDFs_h = CDFVolkNN(vol_h)
    dummyvpf_h = 1.-CDFs_h[0](binc1)
    dummyvpf_h[np.isnan(dummyvpf_h)] = 0
    VPF_h = interpolate.interp1d(binc1, dummyvpf_h, kind='linear', bounds_error=False, fill_value=(1.,0.))
    CDF_1h = interpolate.interp1d(binc1, CDFs_h[0](binc1), kind='linear', bounds_error=False, fill_value=(0.,1.))
    CDF_2h = interpolate.interp1d(binc2, CDFs_h[4](binc2), kind='linear', bounds_error=False, fill_value=(0.,1.))
    CDF_3h = interpolate.interp1d(binc2, CDFs_h[5](binc2), kind='linear', bounds_error=False, fill_value=(0.,1.))
    
    yj1 = CDF_1h(binc1)
    yj2 = CDF_2h(binc2)
    cdf_jack = np.concatenate((yj1[id_fin1], yj2[id_fin2]))
    CDF_jack = np.concatenate((CDF_jack, [cdf_jack]), axis = 0)   

jack_chi2 = []
for i in range(jack):
    t1 = np.dot(inv_covmat, (CDF_jack[i] - CDF_mean))
    t2 = np.dot(CDF_jack[i] - CDF_mean, t1)
    jack_chi2.append(t2)
    
jack_chi2 = np.asarray(jack_chi2)
np.savetxt(baseDir+'/jack_chi2_{}_12.txt'.format(len_poi), jack_chi2)
ax1.hist(jack_chi2, bins = 8, color = 'crimson', edgecolor = 'black', 
         alpha = 0.4, linewidth = 1.2, density=True,
         label = r'Jackknife ({})'.format(jack))

ax1.set_xlabel(r'$\chi^{2}$', fontsize = 24)
ax1.set_ylabel(r'$\#\ \mathrm{of}\ \mathrm{CDFs}$', fontsize = 24)
ax1.legend(loc = 'upper left', fontsize = 16, frameon = False, borderpad = 0.5)

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


fig.savefig('/oak/stanford/orgs/kipac/users/ycwang19/VPF/figures/Chi2_RM_12_{}_50bins_35.png'.format(len_poi), dpi = 400, bbox_inches = 'tight')
plt.show()
