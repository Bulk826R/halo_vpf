import numpy as np
import scipy.spatial
from sklearn.neighbors import KDTree, BallTree
import matplotlib.pylab as plt
from scipy.stats import poisson, erlang
from scipy import interpolate
from os import urandom
import struct
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle
import matplotlib
matplotlib.use('agg')
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['text.color'] = 'black'
matplotlib.rcParams["grid.color"] = 'grey'
matplotlib.rcParams["grid.linestyle"] = '--'
plt.rc("text", usetex=False)
fig = plt.figure()

def VolumekNN(xin, xout, k=1, periodic=0):
    if isinstance(k, int): k = [k] # 
    dim = xin.shape[1]
    Ntot = xin.shape[0]
    xtree = scipy.spatial.cKDTree(xin, boxsize=periodic)
    dis, disi = xtree.query(xout, k=k, n_jobs=-1)
    vol = np.empty_like(dis) # same shape as distance including all k values
    Cr = Ntot * [2, np.pi, 4 * np.pi / 3, np.pi**2, 8*np.pi**2/15][dim - 1]  # Volume prefactor for 1,2, 3D
    for c, k in enumerate(np.nditer(np.array(k))):
        vol[:,c] = Cr * dis[:,c]**dim/k
    return vol

Npoints = 4096
uni_Nl = 600
periodic = 1
dim = 2
dummy = np.random.seed(seed=11)
ks = (1, 2, 10, 50)
x = np.random.rand(Npoints,dim)
if 1 :
    for i, xv in np.ndenumerate(x):
        x[i] = np.abs((struct.unpack('i',urandom(4) ))[0])/2**31
#x = np.sin(x*np.pi)
#xtree = scipy.spatial.cKDTree(x,boxsize=1)
#xtree = KDTree(x)

#        for Nsample in Nsamples:
datatype = type(x[1])
dx = 1/(uni_Nl)
xg = yg = np.linspace(0.*dx, (uni_Nl)*dx, uni_Nl,dtype=datatype)
xg, yg  = np.meshgrid(xg,yg)
xg = xg.flatten()
yg = yg.flatten()
xxg = np.ones((uni_Nl**2, 2))
xxg[:,0] = xg
xxg[:,1] = yg

Ngp = uni_Nl**2
vol = VolumekNN(x, xxg,k=ks, periodic=periodic)
rho = 1/vol
print (vol.shape[0])

extent = 0.,1-dx, 0, 1-dx
import mpl_toolkits.axes_grid1 as axes_grid1
fig = plt.figure(frameon=False)
grid = axes_grid1.AxesGrid(
    fig, 111, nrows_ncols=(2, len(ks)), axes_pad = 0., cbar_location = "right",
    cbar_mode="edge", cbar_size="15%", cbar_pad="5%",)
bbox = dict(boxstyle="round", fc="0.8", alpha=.7)

for kp in range(len(ks)):
    im0 = grid[kp].imshow(np.log10(vol[:,kp].reshape((uni_Nl,uni_Nl))),\
                          aspect='equal',extent=extent,\
                          origin='lower', cmap='coolwarm',vmin=-1,vmax=1)
    im1 = grid[kp+4].imshow(np.log10(rho[:,kp].reshape((uni_Nl,uni_Nl))),\
                          aspect='equal',extent=extent,\
                          origin='lower', cmap='coolwarm',vmin=-1.,vmax=1)
    grid[kp].set_xlim((0,1))
    grid[kp].set_ylim((0,1))
    grid[kp+4].set_xlim((0,1))
    grid[kp+4].set_ylim((0,1))


    if kp != 0:
        dummy = grid[kp+4].set_xticks([])
    dummy = grid[kp].plot(x[:,0],x[:,1],'k.',ms=1.);
    dummy = grid[kp+4].plot(x[:,0],x[:,1],'k.',ms=1.);

    dummy = grid[kp].annotate('k = '+str(ks[kp]), xy=(.03,.04),bbox=bbox);
    dummy = grid[kp+4].annotate('k = '+str(ks[kp]), xy=(.03,.04),bbox=bbox);
    
    boxsl = np.sqrt(ks[kp]/Npoints/np.pi)
    rect = Circle((1-boxsl,1-boxsl),boxsl)
    pc = PatchCollection([rect], facecolor="white", alpha=.95,
                             edgecolor="black")
    pc2 = PatchCollection([rect], facecolor="white", alpha=.95,
                             edgecolor="black")
    
    grid[kp].add_collection(pc)
    grid[kp+4].add_collection(pc2)




grid[0].annotate('$\log V_{kNN}$', xy=(.03,.91),color='black',bbox=bbox);
grid[4].annotate('$\log \delta_{kNN}$', xy=(.03,.91),color='black',bbox=bbox);

grid.cbar_axes[0].colorbar(im0);
grid.cbar_axes[1].colorbar(im1);

fig.savefig('/home/bulk826/Desktop/Stanford/Research3/figures/kNN_test.png', dpi = 200, bbox_inches = 'tight')
plt.show()