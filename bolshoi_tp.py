import sys
import os
import halotools
import halotools.mock_observables as mo
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.spatial
import Corrfunc
from Corrfunc.theory.xi import xi
from Corrfunc.theory.DD import DD
from Corrfunc.utils import convert_3d_counts_to_cf
import h5py
import os
from colossus.cosmology import cosmology
from colossus.lss import peaks
from colossus.lss import bias
from scipy import interpolate
from scipy import integrate

#os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
#nthreads = 4
# plt.style.use(style.notebook)
import matplotlib
matplotlib.use('pdf')
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['text.color'] = 'black'
matplotlib.rcParams["grid.color"] = 'grey'
matplotlib.rcParams["grid.linestyle"] = '--'
plt.rc("text", usetex=False)
fig = plt.figure()
fig.set_size_inches(8, 6)


#Download bolshoi z=0 halo table and particle table
from halotools.sim_manager import DownloadManager
dman = DownloadManager()

#dman.download_processed_halo_table('bolshoi', halo_finder='rockstar', redshift=0.0)
#dman.download_ptcl_table('bolshoi', redshift=0.0)


from halotools.sim_manager import CachedHaloCatalog 
bolshoi_z0 = CachedHaloCatalog(simname = 'bolshoi', halo_finder = 'rockstar', version_name = 'halotools_v0p4', redshift = -0.0003) 
halos = bolshoi_z0.halo_table
particles = bolshoi_z0.ptcl_table
Nhalosin = len(bolshoi_z0.halo_table)
boxsize = 250
print("Number of Halos: ", len(halos))


msel = np.sort(halos["halo_mvir"])
halos.sort("halo_mvir")
mthresh = np.array((5e10, 1e11, 1e12, 1e13, 1e14))
ithresh = np.arange(5)
Nhthresh = np.arange(5)
count = 0
for i in mthresh:
    ithresh[count] = (np.where(msel >= i))[0][0]
    Nhthresh[count] = Nhalosin - ithresh[count] + 1
    print(
        "Greater than %8.2e Msun has %7g halos"
        % (msel[ithresh[count]], Nhthresh[count])
    )
    count = count + 1


fig.set_size_inches(8, 6)

'''
xMh = Nhalosin - np.arange(Nhalosin)
plt.loglog(np.sort(halos["halo_m200c"]), xMh, label=r"$M_{200,\mathrm{crit}}$")
plt.semilogy(np.sort(halos["halo_mvir"]), xMh, label=r"$M_{\mathrm{vir}}$")
plt.semilogy(np.sort(halos["halo_mpeak"]), xMh, label=r"$M_{\mathrm{peak}}$")

for c, i in enumerate(ithresh):
    plt.plot(
        halos["halo_mvir"][i],
        xMh[i],
        "+",
        markersize=18,
        label=r"#" + str(Nhthresh[c]),
    )



plt.xlabel("Mass in $M_{\odot}$")
plt.ylabel("# of halos with M > ")
plt.xlim((3e10, 1e15))
plt.ylim((1, 2e6))

plt.legend(loc = 'lower left', fontsize = 16)
'''


nbins = 30
bins = np.logspace(
    -1, 1.5, nbins + 1
)  # Note the + 1 to nbins, If you reduce the max to 1.3 (smaller max radius the calculation is much faster)

seed = 42
np.random.seed(seed)
N = 1000000
X = np.random.uniform(0, boxsize, N)
Y = np.random.uniform(0, boxsize, N)
Z = np.random.uniform(0, boxsize, N)
weights = np.ones_like(X)
nthreads = 8
results = xi(boxsize, nthreads, bins[3::4], X, Y, Z, output_ravg=True)
print("                         ravg                              xi")
for r in results:
    print(
        "{0:10.6f} {1:10.6f} {2:10.6f} {3:10d} {4:10.6f} {5:10.4f}".format(
            r["rmin"], r["rmax"], r["ravg"], r["npairs"], r["weightavg"], r["xi"]
        )
    )

#All particles
dm_xi = xi(
    boxsize,
    nthreads,
    bins,
    particles["x"],
    particles["y"],
    particles["z"],
    output_ravg=True,
    verbose=True,
)

#Downsampling
sel = np.random.randint(0, len(particles), 100000)
dm_xi_ds5 = xi(
    boxsize,
    nthreads,
    bins,
    particles["x"][sel],
    particles["y"][sel],
    particles["z"][sel],
    output_ravg=True,
    verbose=True,
)
sel = np.random.randint(0, len(particles), 10000)
dm_xi_ds4 = xi(
    boxsize,
    nthreads,
    bins[::2],
    particles["x"][sel],
    particles["y"][sel],
    particles["z"][sel],
    output_ravg=True,
    verbose=False,
)
sel = np.random.randint(0, len(particles), 1000)
dm_xi_ds3 = xi(
    boxsize,
    nthreads,
    bins[5::3],
    particles["x"][sel],
    particles["y"][sel],
    particles["z"][sel],
    output_ravg=True,
    verbose=False,
)


cosmology.setCosmology("bolshoi")
cosmo = cosmology.getCurrent()

Nk = 1000  # number of wavenumber bins
z = 0.0
kw = np.logspace(-1.5, 3, Nk)
R = 2 * np.pi / kw
Pk = cosmo.matterPowerSpectrum(kw, z, model="eisenstein98_zb")
xik = cosmo.correlationFunction(R)
M = peaks.lagrangianM(R)
nu = peaks.peakHeight(M, z)
Mstar = peaks.nonLinearMass(z)
Rstar = peaks.lagrangianR(Mstar)
kstar = 2 * np.pi / Rstar
print("Mstar/1e12, kstar, Rstar:", Mstar / 1e12, kstar, Rstar)
b = bias.haloBiasFromNu(nu, model="sheth01")

from nbodykit.lab import *
from nbodykit import setup_logging, style

pcat = ArrayCatalog(
    {"Position": np.stack([particles["x"], particles["y"], particles["z"]], axis=1)}
)
Pkp = FFTPower(pcat, mode="1d", Nmesh=256, BoxSize=250).power

pcat = ArrayCatalog(
    {
        "Position": np.stack(
            [halos["halo_x"], halos["halo_y"], halos["halo_z"]], axis=1
        )[ithresh[0] : ithresh[1], :]
    }
)
Pkh1 = FFTPower(pcat, mode="1d", Nmesh=256, BoxSize=250).power
pcat = ArrayCatalog(
    {
        "Position": np.stack(
            [halos["halo_x"], halos["halo_y"], halos["halo_z"]], axis=1
        )[ithresh[0] :, :]
    }
)
Pkh1b = FFTPower(pcat, mode="1d", Nmesh=256, BoxSize=250).power
pcat = ArrayCatalog(
    {
        "Position": np.stack(
            [halos["halo_x"], halos["halo_y"], halos["halo_z"]], axis=1
        )[ithresh[1] : ithresh[2], :]
    }
)

Pkh2 = FFTPower(pcat, mode="1d", Nmesh=256, BoxSize=250).power
pcat = ArrayCatalog(
    {
        "Position": np.stack(
            [halos["halo_x"], halos["halo_y"], halos["halo_z"]], axis=1
        )[ithresh[2] : ithresh[3], :]
    }
)


Pkh3 = FFTPower(pcat, mode="1d", Nmesh=256, BoxSize=250).power
pcat = ArrayCatalog(
    {
        "Position": np.stack(
            [halos["halo_x"], halos["halo_y"], halos["halo_z"]], axis=1
        )[ithresh[3] : ithresh[4], :]
    }
)
Pkh4 = FFTPower(pcat, mode="1d", Nmesh=256, BoxSize=250).power
pcat = ArrayCatalog(
    {
        "Position": np.stack(
            [halos["halo_x"], halos["halo_y"], halos["halo_z"]], axis=1
        )[ithresh[4] :, :]
    }
)
Pkh5 = FFTPower(pcat, mode="1d", Nmesh=256, BoxSize=250).power

m14 = np.stack([halos["halo_x"], halos["halo_y"], halos["halo_z"]], axis=1)
print('m = ', m14)
print('len m', len(m14))
print('len m14 = ', len(m14[ithresh[4] :, :]))

rand_x = np.random.rand(len(m14[ithresh[4] :, :])) * 250
rand_y = np.random.rand(len(m14[ithresh[4] :, :])) * 250
rand_z = np.random.rand(len(m14[ithresh[4] :, :])) * 250

pcat = ArrayCatalog(
    {
        "Position": np.stack(
            [rand_x, rand_y, rand_z], axis=1)
    }
)

Pkh_rand = FFTPower(pcat, mode="1d", Nmesh=256, BoxSize=250).power

##########################################
# print the shot noise subtracted P(k)
plt.loglog(Pkp["k"], Pkp["power"].real - Pkp.attrs["shotnoise"], label="DM particles")
plt.loglog(
    Pkh1["k"],
    Pkh1["power"].real - Pkh1.attrs["shotnoise"],
    label=r"all above $5\cdot 10^{10}$",
)
plt.loglog(
    Pkh1b["k"],
    Pkh1b["power"].real - Pkh1b.attrs["shotnoise"],
    label="$\cdot 10^{10} < M < 10^{11}$",
)
plt.loglog(
    Pkh2["k"],
    Pkh2["power"].real - Pkh2.attrs["shotnoise"],
    label="$ 10^{11} < M < 10^{12}$",
)
plt.loglog(
    Pkh3["k"],
    Pkh3["power"].real - Pkh3.attrs["shotnoise"],
    label="$10^{12} < M < 10^{13}$",
)
plt.loglog(
    Pkh4["k"],
    Pkh4["power"].real - Pkh4.attrs["shotnoise"],
    label="$10^{13} < M < 10^{14}$",
)
plt.loglog(
    Pkh5["k"], Pkh5["power"].real - Pkh5.attrs["shotnoise"], label="$ 10^{14} > M$"
)
#
''' 
plt.loglog(
    Pkh_rand["k"], Pkh_rand["power"].real - Pkh_rand.attrs["shotnoise"], label="Poisson",
    linestyle='-.', color = 'black'
)
'''
#

plt.plot(kw, Pk, label="linear theory")
# format the axes
plt.legend(loc = 'lower left')
plt.xlabel(r"$k$ [$h \ \mathrm{Mpc}^{-1}$]")
plt.ylabel(r"$P(k)$ [$h^{-3}\mathrm{Mpc}^3$]")
plt.xlim(0.01, 3.6)
plt.ylim(10, 4e5)
fig.savefig('/home/bulk826/Desktop/Stanford/Research3/figures/bolshoi_tp.png', dpi = 400, bbox_inches = 'tight')
plt.show()