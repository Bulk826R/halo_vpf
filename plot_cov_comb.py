from __future__ import unicode_literals
import numpy as np
import sys, os
import matplotlib.pylab as plt
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
import h5py as h5
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import AutoMinorLocator
from matplotlib.colors import Normalize, LogNorm
import matplotlib.cm as cm
import matplotlib.colors as colors
import pynbody
import pynbody.plot.sph as sph
import glob
from scipy.ndimage import gaussian_filter
import matplotlib.image as mpimg
fig = plt.figure()

im1 = mpimg.imread('/home/bulk826/Desktop/Stanford/Research3/figures/SZ/Cov_CDF.png')
im2 = mpimg.imread('/home/bulk826/Desktop/Stanford/Research3/figures/SZ/Cov_xi.png')

fig.set_size_inches(16, 7.2)
plt.subplots_adjust(wspace = 0.02, hspace = 0.)

ax1 = fig.add_subplot(1, 2, 1)
ax1.imshow(im1, aspect='auto')
ax1.axis('off')

ax2 = fig.add_subplot(1, 2, 2)
ax2.imshow(im2, aspect='auto')
ax2.axis('off')

fig.savefig('/home/bulk826/Desktop/Stanford/Research3/figures/paper/cov.pdf', dpi = 400, bbox_inches = 'tight')
plt.show()


