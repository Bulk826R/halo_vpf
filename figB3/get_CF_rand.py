import numpy as np

len_bins = 51
bins = np.logspace(np.log10(35), np.log10(155), len_bins)
binc = np.sqrt(bins[:-1] * bins[1:])
Nhalos = 526

for sim_number in range (Nhalos):
    fname = '/oak/stanford/orgs/kipac/users/ycwang19/VPF/SZ/cf_qui/rand_'+str(sim_number)+'.txt'
    res = np.loadtxt(fname)
    cf_sz = res[:, 1]
    if sim_number == 0:
        CF_sz = [cf_sz]
    else:
        CF_sz = np.concatenate((CF_sz, [cf_sz]), axis = 0)
        
CF_mean = np.average(CF_sz, axis = 0)
CF_std = np.std(CF_sz, axis = 0) 

Res = np.vstack((binc, CF_mean, CF_std)).T
np.savetxt('/oak/stanford/orgs/kipac/users/ycwang19/VPF/SZ/cf_qui/cf_rand.txt', Res)