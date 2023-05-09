################################################
# By Silja L. Christensen
# May 2023

# This script plots two samples from the SGP prior. 

# Requires installation of CUQIpy: This code runs on a pre-released version of CUQIpy. Install via command:
# pip install git+https://github.com/CUQI-DTU/CUQIpy.git@sprint16_add_JointModel
################################################

#%%
import numpy as np
import scipy as sp
import sys
import os
import matplotlib.pyplot as plt

import funs

sys.path.append('../CUQIpy/') # Delete this line if cuqipy is installed as described above.

import cuqi

# %load_ext autoreload
# %autoreload 2

#%%
#=======================================================================
# Parameters
#=========================================================================
np.random.seed(1000)

# Data 
realdata = False
flat_order = "F"
rot_k = 0

# Reconstruction domain
N = 500          # reconstruction of NxN pixels
domain = 55

# SGP prior
steel = 2e-2*7.9
air = 0
PUrubber = 5.1e-2*0.94
PEfoam = 5.1e-2*0.15
concrete = 4.56e-2*2.3
mu_vals = np.array([steel, PEfoam, PUrubber, concrete, air])
prec_vals = 4*np.array([50000, 50000, 50000, 50000, 50000])
prec_MRF = 10000
# masks
if realdata == False:
    # Synthetic
    maskid = np.array([5,1,2,3,4,5])
    piperadii = np.array([9,11,16,17.5,23])
    maskradii = np.array([[0,piperadii[0]-0.5],
                            [piperadii[0]+0.5, piperadii[1]-0.5],
                            [piperadii[1]+0.5, piperadii[2]-0.5],
                            [piperadii[2]+0.5, piperadii[3]-0.5],
                            [piperadii[3]+0.5, piperadii[4]-0.5],
                            [piperadii[4]+0.5, 50]])
    maskcenterinner = np.zeros((6,2))
    maskcenterouter = np.zeros((6,2))
else:
    maskid = np.array([5,1,2,3,4,5])
    maskradii = np.array([[0,8.7],
                            [9.7,10.3],
                            [11.3,15],
                            [15.5,17.1],
                            [18.1,22],
                            [23,50]])
    maskcenterinner = np.array([[0,0],
                            [0.6,0.8],
                            [0.6,0.8],
                            [0,0.4],
                            [0,0.4],
                            [0,0.4]])
    maskcenterouter= np.array([[0.6,0.8],
                            [0.6,0.8],
                            [0,0.4],
                            [0,0.4],
                            [0,0.4],
                            [0,0.4]])

# Filepaths

path = '../../../../../../work3/swech/priorsamples/'
os.makedirs(path, exist_ok=True)

# =================================================================
# Prior
# =================================================================
# Structural Gaussian Prior on z
mask, IID_mean, _, IID_sqrtprec, D1, D2 = funs.SGP(N, domain, maskradii, maskcenterinner, maskcenterouter,maskid, mu_vals, prec_vals, bndcond = 'zero', flat_order = flat_order, rot_k = rot_k)
GMRF_mean = np.zeros(N**2)
GMRF_sqrtprec = np.sqrt(prec_MRF)*sp.sparse.vstack([D2, D1])
z = funs.myJointGaussianSqrtPrec(means = [IID_mean, GMRF_mean], sqrtprecs = [IID_sqrtprec, GMRF_sqrtprec], geometry=cuqi.geometry.Image2D(im_shape = (N,N), order = "F"), name="z")

z_s = z.sample(2).samples

# Plot
cmin_im = -0.08
cmax_im = 0.2

colmap = "gray"
font_size = 14

fig, axs = plt.subplots(nrows = 1, ncols = 4, figsize = (18, 4), gridspec_kw=dict(wspace=0.3))

x_ticks = np.array([0,100,200,300,400,500])
y_ticks = np.array([0,100,200,300,400,500])
cs0 = axs[0].imshow(np.rot90(z_s[:,0].reshape((N,N), order = flat_order),k=rot_k), extent=[0, N, N, 0], aspect='equal', cmap=colmap, vmin = cmin_im, vmax = cmax_im)
cs1 = axs[2].imshow(np.rot90(z_s[:,1].reshape((N,N), order = flat_order),k=rot_k), extent=[0, N, N, 0], aspect='equal', cmap=colmap, vmin = cmin_im, vmax = cmax_im)

axs[0].tick_params(axis='both', which='both', labelsize=font_size)
axs[0].set_xticks(x_ticks)
axs[0].set_yticks(y_ticks)
axs[2].tick_params(axis='both', which='both', labelsize=font_size)
axs[2].set_xticks(x_ticks)
axs[2].set_yticks(y_ticks)

xlim = np.array([100, 150])
ylim = np.array([200, 150])
x_ticks = np.linspace(xlim[0], xlim[1], 6, endpoint = True)
y_ticks = np.linspace(ylim[1], ylim[0], 6, endpoint = True)
cs2 = axs[1].imshow(np.rot90(z_s[:,0].reshape((N,N), order = flat_order),k=rot_k), extent=[0, N, N, 0], aspect='equal', cmap=colmap, vmin = cmin_im, vmax = cmax_im)
cs3 = axs[3].imshow(np.rot90(z_s[:,1].reshape((N,N), order = flat_order),k=rot_k), extent=[0, N, N, 0], aspect='equal', cmap=colmap, vmin = cmin_im, vmax = cmax_im)
axs[1].set_xlim(xlim[0],xlim[1])
axs[1].set_ylim(ylim[0],ylim[1])
axs[3].set_xlim(xlim[0],xlim[1])
axs[3].set_ylim(ylim[0],ylim[1])
axs[0].plot(np.array([xlim[0],xlim[1],xlim[1],xlim[0],xlim[0]]), np.array([ylim[0],ylim[0],ylim[1],ylim[1],ylim[0]]), '-r', linewidth = 1.5)
axs[2].plot(np.array([xlim[0],xlim[1],xlim[1],xlim[0],xlim[0]]), np.array([ylim[0],ylim[0],ylim[1],ylim[1],ylim[0]]), '-r', linewidth = 1.5)
axs[1].tick_params(axis='both', which='both', labelsize=font_size)#, colors='red')
axs[1].set_xticks(x_ticks)
axs[1].set_yticks(y_ticks)
axs[3].tick_params(axis='both', which='both', labelsize=font_size)#, colors='red')
axs[3].set_xticks(x_ticks)
axs[3].set_yticks(y_ticks)
axs[1].spines[['bottom', 'top', 'left', 'right']].set_color('red')
axs[1].spines[['bottom', 'top', 'left', 'right']].set_lw(1.5)
axs[3].spines[['bottom', 'top', 'left', 'right']].set_color('red')
axs[3].spines[['bottom', 'top', 'left', 'right']].set_lw(1.5)

axs[0].annotate('', xy=(axs[1].get_position().x0, axs[1].get_position().y0), xytext=(xlim[1], ylim[0]), xycoords='figure fraction', textcoords='data', arrowprops = dict([('arrowstyle','-'), ('color','red'), ('linewidth',1.5)]))
axs[0].annotate('', xy=(axs[1].get_position().x0, 1-axs[1].get_position().y0), xytext=(xlim[1], ylim[1]), xycoords='figure fraction', textcoords='data', arrowprops = dict([('arrowstyle','-'), ('color','red'), ('linewidth',1.5)]))

axs[2].annotate('', xy=(axs[3].get_position().x0, axs[3].get_position().y0), xytext=(xlim[1], ylim[0]), xycoords='figure fraction', textcoords='data', arrowprops = dict([('arrowstyle','-'), ('color','red'), ('linewidth',1.5)]))
axs[2].annotate('', xy=(axs[3].get_position().x0, 1-axs[3].get_position().y0), xytext=(xlim[1], ylim[1]), xycoords='figure fraction', textcoords='data', arrowprops = dict([('arrowstyle','-'), ('color','red'), ('linewidth',1.5)]))


cax = fig.add_axes([axs[3].get_position().x1+0.01,axs[3].get_position().y0,0.02,axs[3].get_position().height])
cbar = plt.colorbar(cs3, cax=cax) 
cbar.ax.tick_params(labelsize=font_size) 

filename = "prior_samples"
plt.savefig(path + filename + '.png', transparent=False)
plt.savefig(path + filename + '.eps', format='eps')
plt.close()





