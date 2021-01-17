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
from matplotlib.ticker import AutoMinorLocator, LogLocator
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['text.color'] = 'black'
matplotlib.rcParams['grid.color'] = 'grey'
matplotlib.rcParams['grid.linestyle'] = '--'
matplotlib.rcParams['grid.linewidth'] = 0.4
matplotlib.rcParams['grid.alpha'] = 0.5
fig = plt.figure()

from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from astropy.coordinates import SkyCoord
cosmo = FlatLambdaCDM(H0=67.74, Om0=0.3089)
h = 0.6774
baseDir = '/oak/stanford/orgs/kipac/users/ycwang19/VPF/SZ'

################################
# Import Planck SZ DR2 data 
header_list = fits.open(baseDir+'/data/HFI_PCCS_SZ-union_R2.08.fits')[1].data
hdul = fits.open(baseDir+'/data/HFI_PCCS_SZ-union_R2.08.fits')
hdr = hdul[1].columns
print(hdr)
#hdr variable for checking column names

data = fits.open(baseDir+'/data/HFI_PCCS_SZ-union_R2.08.fits')[1].data
IR_FLAG = data['IR_FLAG'] # IR contamination flag
Q_neu = data['Q_NEURAL'] # Quality factor determined by neural net
FLAG = np.where((Q_neu > 0.4) & (IR_FLAG == 0))[0] # Good neural network confidence, not in regions of heavy IR contamination

# All quantities loaded below account for quality factor and removal of IR contaminated clusters
RA = data['RA'][FLAG]
DEC = data['DEC'][FLAG]
Z = data['REDSHIFT'][FLAG] # raw redshift
rID = data['REDSHIFT_ID'][FLAG]
vID = data['VALIDATION'][FLAG]
SNR = data['SNR'][FLAG] # Signal to noise ratio
Y_5R500 = data['Y5R500'][FLAG] * 1e-3 #arcmin^2
Y_5R500_err = data['Y5R500_ERR'][FLAG] * 1e-3 #arcmin^2
Msz = data['MSZ'][FLAG] # in 1e14 Msun
Msz_err_high = data['MSZ_ERR_UP'][FLAG] # in 1e14 Msun
Msz_err_low = data['MSZ_ERR_LOW'][FLAG] # in 1e14 Msun
Gal_B = data['GLAT'][FLAG]
Gal_L = data['GLON'][FLAG]
Msz *= 1e14
Msz_err_high *= 1e14
Msz_err_low *= 1e14

# Spectroscopic redshift clusters
ID_spec = np.where((vID == 10) | (vID == 11) | (vID == 14))[0] 
# Photometric redshift clusters
ID_photo = np.where((vID == 12) | (vID == 13) | (vID == 15) | (vID == 16) | (vID == 21) | (vID == 22) | (vID == 23) | (vID == 24))[0]
# Spec+Photo-z overlap clusters
ID_mix = np.where((vID == 20) | (vID == 25) | (vID == 30))[0]
# No redshift confirmation
ID_no = np.where(vID == -1)[0]
# Redshifts that are within [0, 0.6] and non-empty, enforces SNR
ID_z = np.where((vID != -1) & (Z >= 0.) & (Z < 0.6) & (SNR >= 4.5))[0]
ID_z2 = np.where((vID != -1) & (Z >= 0.) & (Z < 0.6))[0]
print('ID_spec = ', len(ID_spec))
print('ID_photo = ', len(ID_photo))
print('ID_z = ', len(ID_z))
print('ID_z2 = ', len(ID_z2)) # these two are equal, Q>0.4 guarantees SNR >= 4.5

#############################

z = Z[ID_z]
vID_z = vID[ID_z]
s = 0.03 * np.ones_like(z) # photometric redshift error (standard deviation for Gaussian)

#!!!!!!!!!!!!!!!!!!!!!!!!!!
np.random.seed(12)
#!!!!!!!!!!!!!!!!!!!!!!!!!!

zp = np.random.normal(z, s, len(z)) # Resample redshifts to account for photo-z error
#Replace spectroscopic redshift with original values (thus photo-z errors only applies to non-spec-z redshifts)
ID_spec_z = np.where((vID_z == 10) | (vID_z == 11) | (vID_z == 14))[0]
zp[ID_spec_z] = z[ID_spec_z] 
zp_err = np.zeros_like(zp)
ID_photo_z = np.where((vID_z != 10) & (vID_z != 11) & (vID_z != 14))[0]
zp_err[ID_photo_z] = s[ID_photo_z] # redshift errors for photo-z clusters set to 0.03, 0. for spec-z clusters

R_z = cosmo.comoving_distance(zp).value # Convert redshift to comoving distance
Y500_z = Y_5R500[ID_z]
Y500_errz = Y_5R500_err[ID_z]
Msz_z = Msz[ID_z]
Msz_zlow = Msz_err_low[ID_z]
Msz_zhigh = Msz_err_high[ID_z]
Z_z = zp
Z_errz = zp_err
'''
DEC_z = DEC[ID_z]
RA_z = RA[ID_z]
c_icrs = SkyCoord(ra=RA_z*u.degree, dec=DEC_z*u.degree, frame='icrs')
gal_cord = c_icrs.galactic 
gal_L = gal_cord.l.degree 
gal_B = gal_cord.b.degree 
print('L, B = ', gal_L[0]-180, gal_B[0])
#print(np.min(RA_z), np.max(RA_z))
'''
Gal_Bz = Gal_B[ID_z] #Galatic latitude
Gal_Lz = Gal_L[ID_z] #Galactic longitude
gal_L = Gal_Lz #Longitude (0, 360) deg
gal_B = Gal_Bz #Latitude  (-90, 90) deg
#print('GL, GB = ', Gal_Lz[0]-180, Gal_Bz[0])

Dec_mask = np.where((gal_B/180*np.pi <= -np.pi/6) | (gal_B/180*np.pi >= np.pi/6))[0] # galactic angular mask
R_z = R_z[Dec_mask]
gal_L = gal_L[Dec_mask]
gal_B = gal_B[Dec_mask]
Z_z = Z_z[Dec_mask]
Z_errz = Z_errz[Dec_mask]
Y500_z = Y500_z[Dec_mask]
Y500_errz = Y500_errz[Dec_mask]
Msz_z = Msz_z[Dec_mask]
Msz_zlow = Msz_zlow[Dec_mask]
Msz_zhigh = Msz_zhigh[Dec_mask]

print('len after Dec mask = ', len(Msz_z))

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

idz1 = np.where((Z_z <= 0.6) & (Z_z > 0.) & (Y500_z >= 3e-3))[0]
idz2 = np.where((Z_z <= 0.6) & (Z_z > 0.) & (Y500_z < 3e-3))[0]
idz = np.where((Z_z <= 0.6) & (Z_z > 0.))[0]
print('Len idz = ', len(idz))
#print(np.where(Z_z == 0)[0])

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Y5001 = Y500_z[idz1]
Y500_err1 = Y500_errz[idz1]
msz1 = Msz_z[idz1]
msz_low1 = Msz_zlow[idz1]
msz_high1 = Msz_zhigh[idz1]
Z1 = Z_z[idz1]
Zerr1 = Z_errz[idz1]

Y5002 = Y500_z[idz2]
Y500_err2 = Y500_errz[idz2]
msz2 = Msz_z[idz2]
msz_low2 = Msz_zlow[idz2]
msz_high2 = Msz_zhigh[idz2]
Z2 = Z_z[idz2]
Zerr2 = Z_errz[idz2]

Y500 = Y500_z[idz]
Y500_err = Y500_errz[idz]
msz = Msz_z[idz]
msz_low = Msz_zlow[idz]
msz_high = Msz_zhigh[idz]
Z = Z_z[idz]
Zerr = Z_errz[idz]

####################

fig.set_size_inches(10, 10)

ax2 = fig.add_subplot(1, 1, 1)
ax2.errorbar(Z1, msz1, xerr = Zerr1, yerr = [msz_low1, msz_high1], fmt = 'x',
            color = 'royalblue',  lw = 1.6, capsize = 4.2, capthick = 1.2, 
            alpha = 0.6, label = r'$Y_{5R_{500}}\geqslant 3\times 10^{-3}\,\mathrm{arcmin^2}$'+'$\#${}'.format(len(Z1)))

ax2.errorbar(Z2, msz2, xerr = Zerr2, yerr = [msz_low2, msz_high2], fmt = '^', 
            color = 'crimson',  lw = 1.6, capsize = 4.2, capthick = 1.2, 
            alpha = 0.6, label = r'$Y_{5R_{500}} < 3\times 10^{-3}\,\mathrm{arcmin^2}$'+'$\#${}'.format(len(Z2)))

ax2.set_xlabel(r'$z$', fontsize = 24)
ax2.set_ylabel(r'$M_{\mathrm{SZ}}\,$[$\mathrm{M_{\odot}}$]', fontsize = 24)
ax2.set_xlim([-0.04, 0.64])
ax2.set_yscale('log')
ax2.legend(loc = 'lower right', fontsize = 16, frameon = False, borderpad = 0.6)

minorLocator = AutoMinorLocator()
ax2.xaxis.set_minor_locator(minorLocator)
ax2.set_yscale('log')
ax2.set_yticks(np.logspace(14, 15, 2))
minorLocator = AutoMinorLocator()
loc2 = LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0), numdecs=10, numticks = 10)
ax2.yaxis.set_minor_locator(loc2)
ax2.tick_params(which='major', labelsize = 18, width = 1., length = 6, direction='in', pad = 4, bottom = True, top = True, left = True, right = True)
ax2.tick_params(which='minor', labelsize = 18, width = 1., length = 3, direction='in', bottom = True, top = True, left = True, right = True)


######

divider = make_axes_locatable(ax2)
axHistx = divider.append_axes("top", 2, pad=0.2, sharex=ax2)
axHisty = divider.append_axes("right", 2, pad=0.2, sharey=ax2)

axHistx.xaxis.set_tick_params(labelbottom=False)
axHisty.yaxis.set_tick_params(labelleft=False)
axHistx.set_ylabel('$N(z)$', fontsize = 24)
axHisty.set_xlabel('$N(M_{\mathrm{SZ}})$', fontsize = 24)

Zbins = np.linspace(0, 0.6, 31)
axHistx.hist(Z1, bins = Zbins, color = 'deepskyblue', edgecolor = 'black', 
         alpha = 0.4, linewidth = 1.2)
axHistx.hist(Z2, bins = Zbins, color = 'crimson', edgecolor = 'black', 
         alpha = 0.4, linewidth = 1.2)

y, x = np.histogram(Z, bins = Zbins)
axHistx.step(x[:-1], y, lw = 1.2, dashes = (3.2, 0.5, 1.2, 0.5), alpha = 0.6, 
             color = 'black', label = r'Planck SZ (all)')
print(len(y), len(x))

Bins = np.logspace(np.min(np.log10(msz)), np.max(np.log10(msz)), 26)

axHisty.hist(msz1, bins = Bins, color = 'deepskyblue', edgecolor = 'black', 
         alpha = 0.4, linewidth = 1.2, orientation='horizontal')
axHisty.hist(msz2, bins = Bins, color = 'crimson', edgecolor = 'black', 
         alpha = 0.4, linewidth = 1.2, orientation='horizontal')

y, x = np.histogram(msz, bins = Bins)
axHisty.step(y, x[:-1], lw = 1.2, dashes = (3.2, 0.5, 1.2, 0.5), alpha = 0.6, 
             color = 'black', label = r'Planck SZ (all)')


axHistx.legend(loc = 'upper right', fontsize = 16, frameon = False)
plt.setp(axHistx.get_xticklabels(), visible = False)
plt.setp(axHisty.get_yticklabels(), visible = False)


axHisty.set_yscale('log')
axHisty.set_yticks(np.logspace(14, 15, 2))
minorLocator = AutoMinorLocator()
locHisty = LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0), numdecs=10, numticks = 10)
axHisty.yaxis.set_minor_locator(loc2)
axHisty.tick_params(which='major', labelsize = 18, width = 1., length = 6, direction='in', pad = 4, bottom = True, top = True, left = True, right = True)
axHisty.tick_params(which='minor', labelsize = 18, width = 1., length = 3, direction='in', bottom = True, top = True, left = True, right = True)

minorLocator = AutoMinorLocator()
axHistx.xaxis.set_minor_locator(minorLocator)
minorLocator = AutoMinorLocator()
axHistx.yaxis.set_minor_locator(minorLocator)
axHistx.tick_params(which='major', labelsize = 18, width = 1., length = 6, direction='in', pad = 4, bottom = True, top = True, left = True, right = True)
axHistx.tick_params(which='minor', labelsize = 18, width = 1., length = 3, direction='in', bottom = True, top = True, left = True, right = True)


fig.savefig('/oak/stanford/orgs/kipac/users/ycwang19/VPF/figures/SZ_dist.pdf', dpi = 200, bbox_inches = 'tight')
plt.show()

