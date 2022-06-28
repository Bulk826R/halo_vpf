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
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['text.color'] = 'black'
matplotlib.rcParams['grid.color'] = 'grey'
matplotlib.rcParams['grid.linestyle'] = '--'
matplotlib.rcParams['grid.linewidth'] = 0.4
matplotlib.rcParams['grid.alpha'] = 0.5
fig = plt.figure()
from matplotlib.ticker import AutoMinorLocator, LogLocator
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

bine = np.logspace(np.log10(30), np.log10(220), 51)
binw = bine[1:] - bine[:-1]
binc = (bine[1:] + bine[:-1]) / 2

#3NN
#bine = np.logspace(np.log10(40), np.log10(220), 26)
bine = np.logspace(np.log10(50), np.log10(160), 26)
binw = bine[1:] - bine[:-1]
binc3 = (bine[1:] + bine[:-1]) / 2
#4NN
#bine = np.logspace(np.log10(55), np.log10(220), 26)
bine = np.logspace(np.log10(60), np.log10(160), 26)
binw = bine[1:] - bine[:-1]
binc4 = (bine[1:] + bine[:-1]) / 2

vol_h = np.loadtxt(baseDir+'/Dis_h_3D_Y500.txt') # Reads in r kNN data
CDFs_h = CDFVolkNN(vol_h)
CDF_1h = interpolate.interp1d(binc, CDFs_h[0](binc), kind='linear', bounds_error=False, fill_value=(0.,1.))
CDF_2h = interpolate.interp1d(binc, CDFs_h[4](binc), kind='linear', bounds_error=False, fill_value=(0.,1.))
CDF_3h = interpolate.interp1d(binc, CDFs_h[5](binc), kind='linear', bounds_error=False, fill_value=(0.,1.))
CDF_4h = interpolate.interp1d(binc, CDFs_h[6](binc), kind='linear', bounds_error=False, fill_value=(0.,1.))

Yd1 = CDF_1h(binc)
Yd2 = CDF_2h(binc)
Yd3 = CDF_3h(binc)
Yd4 = CDF_4h(binc)

print(Yd1)
print(Yd2)
print(Yd3)
print(Yd4)

def CDF_2nn(CDF_1NN):
    c2 = CDF_1NN + (1-CDF_1NN) * np.log(1-CDF_1NN)
    return c2

def CDF_3NN(CDF_1NN, CDF_2NN):
    c3 = CDF_2NN + ( (1-CDF_1NN)*np.log(1-CDF_1NN) + (CDF_1NN - CDF_2NN) - 
                    1/2*(CDF_1NN - CDF_2NN)**2/(1-CDF_1NN) )
    return c3

def CDF_4NN(CDF_1NN, CDF_2NN, CDF_3NN):
    c4 = CDF_3NN + (CDF_1NN - CDF_2NN)/(1 - CDF_1NN) * ( (1-CDF_1NN)*np.log(1-CDF_1NN) + (CDF_1NN - CDF_2NN)
                                                        - 1/6 * (CDF_1NN-CDF_2NN)**2/(1-CDF_1NN))
    return c4

C2 = CDF_2nn(Yd1)
C3 = CDF_3NN(Yd1, Yd2)
C41 = CDF_4NN(Yd1, Yd2, C3)
C42 = CDF_4NN(Yd1, Yd2, Yd3)
print(C3)
print(C41)
print(C42)

def get_pCDF(cdf):
    id_rise = np.where(cdf <= 0.5)[0]
    id_drop = np.where(cdf > 0.5)[0]
    pcdf = np.concatenate((cdf[id_rise], 1-cdf[id_drop]))
    return pcdf

pCDF_1 = get_pCDF(Yd1)
pCDF_2 = get_pCDF(Yd2)
pCDF_3 = get_pCDF(Yd3)
pCDF_4 = get_pCDF(Yd4)
pCDF_C2 = get_pCDF(C2)
pCDF_C3 = get_pCDF(C3)
pCDF_C41 = get_pCDF(C41)
pCDF_C42 = get_pCDF(C42)

jack = 1000
std1 = np.loadtxt(baseDir+'/jack_std1_{}.txt'.format(jack))
std2 = np.loadtxt(baseDir+'/jack_std2_{}.txt'.format(jack))
std3 = np.loadtxt(baseDir+'/jack_std3_{}.txt'.format(jack))
std4 = np.loadtxt(baseDir+'/jack_std4_{}.txt'.format(jack))
'''
s2 = np.loadtxt('/oak/stanford/orgs/kipac/users/ycwang19/VPF/SZ/redMapper/C43/cdf_s2.txt')
s3 = np.loadtxt('/oak/stanford/orgs/kipac/users/ycwang19/VPF/SZ/redMapper/C43/cdf_s3.txt')
s4 = np.loadtxt('/oak/stanford/orgs/kipac/users/ycwang19/VPF/SZ/redMapper/C43/cdf_s4.txt')
'''
###################################################

inv_covmat_2 = np.loadtxt('/oak/stanford/orgs/kipac/users/ycwang19/VPF/SZ/redMapper/C43/inv_cov_2_.txt')
inv_covmat_34 = np.loadtxt('/oak/stanford/orgs/kipac/users/ycwang19/VPF/SZ/redMapper/C43/inv_cov_34_.txt')

#CDF_RM = np.loadtxt('/oak/stanford/orgs/kipac/users/ycwang19/VPF/SZ/redMapper/C43/CDF_RM_C43.txt')
#Yd2 = np.loadtxt('/oak/stanford/orgs/kipac/users/ycwang19/VPF/SZ/redMapper/C43/CDF_RM_2.txt')
CG = np.loadtxt('/oak/stanford/orgs/kipac/users/ycwang19/VPF/SZ/redMapper/C43/CDF_RM_p43.txt')
#C2 = np.loadtxt('/oak/stanford/orgs/kipac/users/ycwang19/VPF/SZ/redMapper/C43/CDF_RM_p2.txt')
#CDF2_mean = np.loadtxt('/oak/stanford/orgs/kipac/users/ycwang19/VPF/SZ/redMapper/C43/CDF2_mean.txt')
CDF34_mean = np.loadtxt('/oak/stanford/orgs/kipac/users/ycwang19/VPF/SZ/redMapper/C43/CDF34_mean.txt')


T1 = np.dot(inv_covmat_34, (CG - CDF34_mean))
T2 = np.dot((CG - CDF34_mean), T1)
data_chi_34 = T2

rand_chi_34 = np.zeros(jack)
for i in range(jack):
    print(i)
    cdf3 = np.loadtxt(baseDir+'/jack_loo/jack_{}_3NN.txt'.format(i))
    cdf4 = np.loadtxt(baseDir+'/jack_loo/jack_{}_4NN.txt'.format(i))
    c3h = interpolate.interp1d(binc, cdf3, kind='linear', bounds_error=False, fill_value=(0.,1.))
    c4h = interpolate.interp1d(binc, cdf4, kind='linear', bounds_error=False, fill_value=(0.,1.))
    cg3 = c3h(binc3)
    cg4 = c4h(binc4)    
    cdf = np.concatenate((cg3, cg4))
    
    T1 = np.dot(inv_covmat_34, np.sqrt(jack - 1)*(cdf - CDF34_mean))
    T2 = np.dot(np.sqrt(jack - 1)*(cdf - CDF34_mean), T1)
    #T1 = np.dot(inv_covmat_34, np.sqrt(jack - 1)*(cdf - CDF_RM))
    #T2 = np.dot(np.sqrt(jack - 1)*(cdf - CDF_RM), T1)
    rand_chi_34[i] = T2
    
print('Data chi = ', data_chi_34)
print('Rand chi = ', rand_chi_34)
mask = np.where(rand_chi_34>data_chi_34)
p = len(rand_chi_34[mask])/len(rand_chi_34)
print(r'$p$-value = ', p)

###################################################
fig.set_size_inches(8, 14)
plt.subplots_adjust(wspace = 0.12, hspace = 0.24)

ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(binc, pCDF_1, lw = 1., color = 'blueviolet', alpha = 0.8, label = r'$\mathrm{1NN}$')
ax1.plot(binc, pCDF_2, lw = 1., color = 'orangered', alpha = 0.8, label = r'$\mathrm{2NN}$')
ax1.plot(binc, pCDF_3, lw = 1., color = 'darkturquoise', alpha = 0.8, label = r'$\mathrm{3NN}$')
ax1.plot(binc, pCDF_4, lw = 1., color = 'seagreen', alpha = 0.8, label = r'$\mathrm{4NN}$')

ax1.fill_between(binc, pCDF_1-std1, pCDF_1+std1, alpha = 0.2, color = 'blueviolet', lw = 0.)
ax1.fill_between(binc, pCDF_2-std2, pCDF_2+std2, alpha = 0.2, color = 'orangered', lw = 0.)
ax1.fill_between(binc, pCDF_3-std3, pCDF_3+std3, alpha = 0.2, color = 'darkturquoise', lw = 0.)
ax1.fill_between(binc, pCDF_4-std4, pCDF_4+std4, alpha = 0.2, color = 'seagreen', lw = 0.)

ax1.plot(binc, pCDF_C3, lw = 1.6, color = 'darkturquoise', linestyle = ':', alpha = 0.8, label = r'$\mathrm{3NN(1,2)}$')
ax1.plot(binc, pCDF_C41, lw = 1.6, color = 'seagreen', linestyle = ':', alpha = 0.8, label = r'$\mathrm{4NN(1,2)}$')
#Errors not Jackknife due to Equation 12 and 13 propagation
#ax1.errorbar(binc, pCDF_C3, yerr = s3, fmt = '.', capsize = 1.2, markersize = 1, lw = 0.8, alpha = 0.8, color = 'darkturquoise' )
#ax1.errorbar(binc, pCDF_C41, yerr = s4, fmt = '.', capsize = 1.2, markersize = 1, lw = 0.8, alpha = 0.8, color = 'seagreen' )


ax1.set_xlabel(r'$r\,$[$\mathrm{Mpc}$]', fontsize = 24, labelpad = 8)
ax1.set_ylabel(r'$\mathrm{pCDF}$', fontsize = 24, labelpad = 8)
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.tick_params(which='major', labelsize = 18, width = 1., length = 6, direction='in', pad = 4, bottom = True, top = True, left = True, right = True)
ax1.tick_params(which='minor', labelsize = 18, width = 1., length = 3, direction='in', bottom = True, top = True, left = True, right = True)
ax1.legend(bbox_to_anchor=(0.0, 1.1, 0.99, .102), loc='upper center', 
           fontsize = 16, ncol = 3, frameon = True, borderpad = 0.5)
ax1.set_ylim([1e-3, 1])
ax1.grid(color='k', linestyle='--', linewidth = 1.2, alpha=.2, which='both', axis='both')
ax1.grid(color='k', linestyle='-', linewidth = 1.2, alpha=.2, which='major', axis='both')
ax1.set_xlim([35, 200])


ax1 = fig.add_subplot(2, 1, 2)
Nbins = 25
ax1.hist(rand_chi_34, bins = Nbins, color = 'royalblue', edgecolor = 'black', 
         alpha = 0.4, linewidth = 1.2, density=True,
         label = r'Measurement Jackknife ({})'.format(jack))

ax1.axvline(x = data_chi_34, lw = 2.4, color = 'crimson', alpha = 0.8, dashes = (5, 2.4), label = r'Gaussian field prediction')

ax1.set_xlabel(r'$\chi^{2}$', fontsize = 24)
ax1.set_ylabel(r'$\mathrm{PDF(3NN \oplus 4NN)}$', fontsize = 24)
ax1.legend(loc = 'upper center', fontsize = 16, frameon = False, borderpad = 0.5)
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
ax1.text(xmin + width * 0.28, ymin + height * 0.12, r'$\mathrm{3NN\ CDF} \in [50\,\mathrm{Mpc}, 160\,\mathrm{Mpc}]$', 
         horizontalalignment = 'left', verticalalignment = 'bottom', fontsize = 18, color = 'black', alpha = 0.8)
ax1.text(xmin + width * 0.28, ymin + height * 0.04, r'$\mathrm{4NN\ CDF} \in [60\,\mathrm{Mpc}, 160\,\mathrm{Mpc}]$', 
         horizontalalignment = 'left', verticalalignment = 'bottom', fontsize = 18, color = 'black', alpha = 0.8)


fig.savefig('/oak/stanford/orgs/kipac/users/ycwang19/VPF/figures/C43.pdf', dpi = 400, bbox_inches = 'tight')
plt.show()
