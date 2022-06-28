from os import read, urandom
import numpy as np
import sys, os
import scipy.spatial
from scipy import interpolate
from scipy.stats import poisson, erlang
import struct
import pandas as pd
import readfof
from Corrfunc.theory.DD import DD
from Corrfunc.utils import convert_3d_counts_to_cf

r = np.logspace(np.log10(10),np.log10(200),14)
#
len_bins = 51
bins = np.logspace(np.log10(35), np.log10(155), len_bins)
binc = np.sqrt(bins[:-1] * bins[1:])
Nhalos = 526
Boxsize = 1000.
nthreads = 4
sim_type = 'fiducial'

nthreads = 8


for sim_number in range (1000):
    path = ('/oak/stanford/orgs/kipac/users/arkab/quijote_scratch/quijote_data/'+sim_type
                +'/sim_'+str(sim_number)+'/')
    FoF = readfof.FoF_catalog(path, int(3), long_ids=False,
                                 swap=False, SFR=False, read_IDs=False)
    pos = FoF.GroupPos/1e3            #Halo positions in Mpc/h
    mass  = FoF.GroupMass*1e10
    idx = np.argsort(mass)
    idx = idx[::-1]
    mass = mass[idx]
    pos = pos[idx]
    pos = pos[:Nhalos]
    mass = mass[:Nhalos]

    X = pos[:, 0]
    Y = pos[:, 1]
    Z = pos[:, 2]
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
    Z = Z.astype(np.float64)
    
    N = Nhalos
    rand_N = 100000 # Same as the randoms used in kmean_xi, 100x1000
    rand_X = np.random.rand(rand_N) * Boxsize
    rand_Y = np.random.rand(rand_N) * Boxsize
    rand_Z = np.random.rand(rand_N) * Boxsize
    #print(X.dtype, rand_X.dtype)
    autocorr=1
    DD_counts = DD(autocorr, nthreads, bins, X, Y, Z, periodic=False, verbose=True)

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
    '''
    results = xi(Boxsize, nthreads, bins,
                pos[:,0], pos[:,1], pos[:,2],
                output_ravg=True)
    binc = results['ravg']
    CF = results['xi']
    '''
    res = np.vstack((binc, cf_sz)).T
    
    fname = '/oak/stanford/orgs/kipac/users/ycwang19/VPF/SZ/cf_qui/data_'+str(sim_number)+'.txt'
    np.savetxt(fname, res)
    

