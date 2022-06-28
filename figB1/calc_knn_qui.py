import numpy as np

Nhalos = 526
for i in range(Nhalos):
    fname = '/oak/stanford/orgs/kipac/users/ycwang19/VPF/SZ/knn_qui/data_'+str(i)+'.txt'
    data = np.loadtxt(fname)
    cdf1 = data[:, 1]
    cdf2 = data[:, 2]
    
    if i == 0:
        CDF1 = [cdf1]
        CDF2 = [cdf2]
    else:
        CDF1 = np.concatenate((CDF1, [cdf1]), axis = 0)
        CDF2 = np.concatenate((CDF2, [cdf2]), axis = 0)
        
CDF1_mean = np.average(CDF1, axis = 0)
CDF2_mean = np.average(CDF2, axis = 0)
CDF1_std = np.std(CDF1, axis = 0)
CDF2_std = np.std(CDF2, axis = 0)        

fname = '/oak/stanford/orgs/kipac/users/ycwang19/VPF/SZ/knn_qui/data_'+str(528)+'.txt'
data = np.loadtxt(fname)
cdf01 = data[:, 1]
cdf02 = data[:, 2]

res = np.vstack((CDF1_mean, CDF2_mean, CDF1_std, CDF2_std, cdf01, cdf02)).T
np.savetxt('/oak/stanford/orgs/kipac/users/ycwang19/VPF/SZ/knn_qui/knn_qui.txt', res)