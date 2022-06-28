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

################################

CDF_34 = []
jack = 526
for i in range(jack):
    print(i)
    data = np.loadtxt('/oak/stanford/orgs/kipac/users/ycwang19/VPF/SZ/knn_data/data_{}.txt'.format(i))
    cdf1 = data[:, 1]
    cdf2 = data[:, 2]
    cdf3 = data[:, 3]
    cdf4 = data[:, 4]
    c3h = interpolate.interp1d(binc, cdf3, kind='linear', bounds_error=False, fill_value=(0.,1.))
    c4h = interpolate.interp1d(binc, cdf4, kind='linear', bounds_error=False, fill_value=(0.,1.))
    cg3 = c3h(binc3)
    cg4 = c4h(binc4)

    if i == 0:
        CDF_34 = np.array([np.concatenate((cg3, cg4))])
    else:
        CDF_34 = np.concatenate((CDF_34, [np.concatenate((cg3, cg4))]), axis = 0)

CDF34_mean = np.average(CDF_34, axis = 0)
np.savetxt('/oak/stanford/orgs/kipac/users/ycwang19/VPF/SZ/knn_qui/CDF34_mean.txt', CDF34_mean)

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

bins = np.concatenate((binc3, binc4))
cov_mat_34 = np.zeros((len(bins), len(bins)))
for i in range(jack):
    cov_mat_34 += update_covmat(CDF34_mean, CDF_34[i])

cov_mat_34 = cov_mat_34/jack # * (jack - 1)
print('cov_mat_34 = ', cov_mat_34)
inv_covmat_34 = (jack - len(bins) -2)/(jack - 1) * np.linalg.inv(cov_mat_34) # Hartlap factor for the inverse
print('inv_cov_34 = ', inv_covmat_34)
np.savetxt('/oak/stanford/orgs/kipac/users/ycwang19/VPF/SZ/knn_qui/inv_cov_34_.txt', inv_covmat_34)
