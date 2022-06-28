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
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=67.74, Om0=0.3089)
h = 0.6774
baseDir = '/oak/stanford/orgs/kipac/users/ycwang19/VPF/SZ/redMapper'
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
bine = np.logspace(np.log10(60), np.log10(160), 26)
binw = bine[1:] - bine[:-1]
binc4 = (bine[1:] + bine[:-1]) / 2


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

vol_h = np.loadtxt(baseDir+'/Dis_h_3D_Y500.txt') # Reads in r kNN data
CDFs_h = CDFVolkNN(vol_h)

################################

Yd1 = np.loadtxt(baseDir+'/C43/CDF_1NN.txt')
Yd2 = np.loadtxt(baseDir+'/C43/CDF_2NN.txt') 
Yd3 = CDFs_h[5](binc3)
Yd4 = CDFs_h[6](binc4)

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
C4 = CDF_4NN(Yd1, Yd2, C3)

C3h = interpolate.interp1d(binc, C3, kind='linear', bounds_error=False, fill_value=(0.,1.))
C4h = interpolate.interp1d(binc, C4, kind='linear', bounds_error=False, fill_value=(0.,1.))
CG3 = C3h(binc3)
CG4 = C4h(binc4)
CG = np.concatenate((CG3, CG4))

################################

CDF_2 = []
CDF_34 = []

jack = 1000
for i in range(jack):
    print(i)
    cdf1 = np.loadtxt(baseDir+'/jack_loo/jack_{}_1NN.txt'.format(i))
    cdf2 = np.loadtxt(baseDir+'/jack_loo/jack_{}_2NN.txt'.format(i))
    cdf3 = np.loadtxt(baseDir+'/jack_loo/jack_{}_3NN.txt'.format(i))
    cdf4 = np.loadtxt(baseDir+'/jack_loo/jack_{}_4NN.txt'.format(i))
    c3h = interpolate.interp1d(binc, cdf3, kind='linear', bounds_error=False, fill_value=(0.,1.))
    c4h = interpolate.interp1d(binc, cdf4, kind='linear', bounds_error=False, fill_value=(0.,1.))

    cg3 = c3h(binc3)
    cg4 = c4h(binc4)

    if i == 0:
        CDF_2 = [cdf2]
        CDF_34 = np.array([np.concatenate((cg3, cg4))])
    else:
        CDF_2 = np.concatenate((CDF_2, [cdf2]), axis = 0)
        CDF_34 = np.concatenate((CDF_34, [np.concatenate((cg3, cg4))]), axis = 0)

CDF2_mean = np.average(CDF_2, axis = 0)
CDF34_mean = np.average(CDF_34, axis = 0)

################################

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

cov_mat_2 = np.zeros((len(binc), len(binc)))
for i in range(jack):
    #idf = np.where(np.isfinite(CDF_poi[i])==1)[0]
    cov_mat_2 += update_covmat(CDF2_mean, CDF_2[i])

cov_mat_2 = cov_mat_2 * (jack - 1)/jack #

bins = np.concatenate((binc3, binc4))
CDF_RM = np.concatenate((Yd3, Yd4))
cov_mat_34 = np.zeros((len(bins), len(bins)))
for i in range(jack):
    #idf = np.where(np.isfinite(CDF_poi[i])==1)[0]
    cov_mat_34 += update_covmat(CDF34_mean, CDF_34[i])

cov_mat_34 = cov_mat_34 * (jack - 1)/jack #
    
print('cov_mat_2 = ', cov_mat_2)
print('cov_mat_34 = ', cov_mat_34)
np.savetxt('/oak/stanford/orgs/kipac/users/ycwang19/VPF/SZ/redMapper/C43/cov_2.txt', cov_mat_2)
np.savetxt('/oak/stanford/orgs/kipac/users/ycwang19/VPF/SZ/redMapper/C43/cov_34.txt', cov_mat_34)
np.savetxt('/oak/stanford/orgs/kipac/users/ycwang19/VPF/SZ/redMapper/C43/CDF_RM_C43.txt', CDF_RM)
np.savetxt('/oak/stanford/orgs/kipac/users/ycwang19/VPF/SZ/redMapper/C43/CDF_RM_2.txt', Yd2)
np.savetxt('/oak/stanford/orgs/kipac/users/ycwang19/VPF/SZ/redMapper/C43/CDF_RM_p43.txt', CG)
np.savetxt('/oak/stanford/orgs/kipac/users/ycwang19/VPF/SZ/redMapper/C43/CDF_RM_p2.txt', C2)
np.savetxt('/oak/stanford/orgs/kipac/users/ycwang19/VPF/SZ/redMapper/C43/CDF2_mean.txt', CDF2_mean)
np.savetxt('/oak/stanford/orgs/kipac/users/ycwang19/VPF/SZ/redMapper/C43/CDF34_mean.txt', CDF34_mean)

#############################################
inv_covmat_2 = (jack - len(binc) -2)/(jack - 1) * np.linalg.inv(cov_mat_2) # Hartlap factor for the inverse
inv_covmat_34 = (jack - len(bins) -2)/(jack - 1) * np.linalg.inv(cov_mat_34) # Hartlap factor for the inverse
print('inv_cov_2 = ', inv_covmat_2)
print('inv_cov_34 = ', inv_covmat_34)
np.savetxt('/oak/stanford/orgs/kipac/users/ycwang19/VPF/SZ/redMapper/C43/inv_cov_2_.txt', inv_covmat_2)
np.savetxt('/oak/stanford/orgs/kipac/users/ycwang19/VPF/SZ/redMapper/C43/inv_cov_34_.txt', inv_covmat_34)
