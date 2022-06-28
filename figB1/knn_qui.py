from os import read, urandom
import numpy as np
import sys, os
import scipy.spatial
from scipy import interpolate
from scipy.stats import poisson, erlang
import struct
import pandas as pd
import readfof

def create_bins(kNN):
    bins = np.zeros((25,len(kNN)))

    bine = np.logspace(np.log10(35), np.log10(135), 26)
    binw = bine[1:] - bine[:-1]
    bins[:, 0] = (bine[1:] + bine[:-1]) / 2

    bine = np.logspace(np.log10(55), np.log10(155), 26)
    binw = bine[1:] - bine[:-1]
    bins[:, 1] = (bine[1:] + bine[:-1]) / 2

    bins[:, 2] = np.logspace(np.log10(80), np.log10(190), 25)
    bins[:, 3] = np.logspace(np.log10(120), np.log10(240), 25)
    
    return bins

def cdf_vol_knn(vol):
    '''
    Computes an interpolating function to evaluate CDF 
    at a given radius.
    
    Parameters
    ----------
    
    vol: float[:,:]
        List of nearest neighbor distances for each kNN.
        vol.shape[1] should be # of kNN
    
    Returns
    -------
    
    cdf: scipy interpolating function for each kNN
    '''
    
    cdf = []
    n = vol.shape[0]
    l = vol.shape[1]
    gof = ((np.arange(0, n) + 1) / (n*1.0))
    for c in range(l):
        ind = np.argsort(vol[:, c])
        s_vol= vol[ind, c]
        cdf.append(interpolate.interp1d(s_vol, gof, kind='linear', 
                                        bounds_error=False))
    return cdf

def compute_cdf(pos, kNN, random_pos, boxsize, bins):
    '''
    Computes the CDF of nn distances of 
    data points from a set of space-filling
    randoms.
    
    Currently set for periodic boundary 
    conditions
    
    Parameters
    ----------
    
    pos: float[:,:]
        Positions of particles (data)
    kNN: int list
        List of k nearest neighbor distances
        that need to be computed
    nrandoms: int
        Number of randoms to be used 
        for the calculation
    boxsize: float
        Size of the simulation box
    bins: float
        Bin centers for kNN (assumed same for now)
        
    Returns
    -------
    
    data: float[:,:]
        kNN CDFs at the requested bin centers
    '''
    
    #periodic = boxsize
    periodic = 0
    xtree = scipy.spatial.cKDTree(pos, boxsize=periodic)
    vol, disi = xtree.query(random_pos, k=kNN, n_jobs=-1)
    
    #bine = np.logspace(-0.5, 1.9, 5000)
    bine = np.linspace(10, 300, 5000)
    binc = (bine[1:] + bine[:-1]) / 2

    #Now get the CDF
    data = (np.zeros((bins.shape[0], 
                      len(kNN))))
    cdfs = cdf_vol_knn(vol)
    for i in range(len(kNN)):
        dummycdf = cdfs[i](binc)
        dummycdf[np.isnan(dummycdf)] = 1.0
        cdf_interp = interpolate.interp1d(binc, dummycdf, 
                                          kind='linear', 
                                          bounds_error=False, 
                                          fill_value=(0., 0.))
        data[:, i] = cdf_interp(bins[:, i])
    return data

Nhalos = 526#1000
nrandoms = 100*100**3
kNN = [1,2,4,8]
#bins = np.logspace(0.4,1.7,60)
#bins = np.linspace(6,30,10)
#bins = np.linspace(12,32,20)
#bins = np.logspace(1.0,2.1,40)
bins = create_bins(kNN)
boxsize = 1000.
sim_type = 'fiducial'

for sim_number in range (1000): # We only need one sim for the jackknife calculation
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
    
    pos = (pos+boxsize)%boxsize
    #Set up the measurement grid here, then we will subdivide
    x = np.linspace(0.,boxsize,200,endpoint=False)
    y = np.linspace(0.,boxsize,200,endpoint=False)
    z = np.linspace(0.,boxsize,200,endpoint=False)
    xv, yv, zv = np.meshgrid(x,y,z)
   
    xv = np.reshape(xv,200**3)
    yv = np.reshape(yv,200**3)
    zv = np.reshape(zv,200**3)
    random_pos = np.column_stack((xv,yv,zv))
    
    data = np.zeros((len(bins),len(kNN)+1))
    data[:, 0] = bins[:, 0]
    data[:,1:] = compute_cdf(pos, kNN, random_pos, boxsize, bins)
    fname = '/oak/stanford/orgs/kipac/users/ycwang19/VPF/SZ/knn_qui/data_'+str(sim_number)+'.txt'
    np.savetxt(fname, data)