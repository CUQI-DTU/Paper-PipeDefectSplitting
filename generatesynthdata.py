################################################
# By Silja L. Christensen
# August 2022 - May 2023

# This script generates and saves synthetic subsea pipe CT data used for numerical experiments
################################################

import numpy as np
import scipy.io as spio
import os
import astra
import matplotlib.pyplot as plt

import funs
import plotfuns

#%%=======================================================================
# Parameters
#=========================================================================
# CT problem
phantomname = 'DeepSeaOilPipe8'   # choose phantom
rnl = 0.05                      # relative noise level
n = 1024                        # phantom dimension
ag = "full"                    # Problem geometry

# reconstruction parameters
N = 500
niter = 400
domain_recon      = 55              # physical size of object

# Filepaths
path = '../data/SyntheticData/{}_rnl{:d}_geom{}/'.format(phantomname, int(rnl*100), ag)
os.makedirs(path, exist_ok=True)

#%%=======================================================================
# Define geometry
#=========================================================================
# Aqusition geometry
p, theta, stc, ctd, shift, vectors, dl, dlA = funs.geom_Data20180911(ag)
q = len(theta)
domain      = 55              # physical size of object

#%%=======================================================================
# Create/load sinogram
#=========================================================================
# create phantom
phantom, _, _, _, _ = getattr(funs, phantomname)(n,True)

# geometries
vol_geom = astra.create_vol_geom(n,n,-domain/2,domain/2,-domain/2,domain/2)
proj_geom = astra.create_proj_geom('fanflat_vec', p, vectors)
proj_id0 = astra.create_projector('cuda', proj_geom, vol_geom) # line_fanflat

# create sinogram
_ , sino_astra = astra.create_sino(phantom, proj_id0)
# clean up
b_true = sino_astra.flatten()
m = len(b_true)  # dimension of data

# add noise
e0 = np.random.normal(0, 1, np.shape(b_true))
noise_std = rnl*np.linalg.norm(sino_astra)/np.linalg.norm(e0)
lambd = 1/noise_std**2
noise = noise_std*e0
b_data = b_true + noise

# underlying true in same dimension as reconstruction
x_true, radii, defectmask, vertices, centers = getattr(funs, phantomname)(N,True)
x_truef = x_true.flatten(order='F')
xt_norm = np.linalg.norm(x_truef)


#%%=======================================================================
# Astra SART reconstruction
#=========================================================================

# Starting point
#geometries
vol_geom = astra.create_vol_geom(N,N,-domain_recon/2,domain_recon/2,-domain_recon/2,domain_recon/2)
proj_geom = astra.create_proj_geom('fanflat_vec', p, vectors)
# data objects
sino_id     = astra.data2d.create('-sino', proj_geom, b_data.reshape((q,p), order = "C"))   # sino object
rec_id      = astra.data2d.create('-vol', vol_geom)           # recon object
# Set up the parameters for a reconstruction algorithm using the GPU
cfg = astra.astra_dict('SART_CUDA') # SIRT_CUDA, SART_CUDA, EM_CUDA, FBP_CUDA (see the FBP sample)
cfg['ReconstructionDataId'] = rec_id
cfg['ProjectionDataId'] = sino_id
cfg['option']={}
# Create the algorithm object from the configuration structure
alg_id = astra.algorithm.create(cfg)
astra.algorithm.run(alg_id, niter)
# retrieve reconstruction and save it
rec = astra.data2d.get(rec_id)
x_recon = rec.flatten(order='F')
# clean up
astra.algorithm.delete(alg_id)
astra.data2d.delete(rec_id)
astra.data2d.delete(sino_id)

#%%=======================================================================
# Save data
#=========================================================================

mdict={'phantomname': phantomname,
        'ag': ag,
        'rnl': rnl,
        'lambd': lambd,
        'n': n,
        'phantom': phantom, 
        'b_data': b_data,
        'sino_astra': sino_astra,
        'b_true': b_true,
        'noise': noise,
        'p': p,
        'q': q,
        'path': path}

recon_dict = {'N': N,
                'x_recon': x_recon, 
                'x_true': x_true, 
                'x_truef': x_truef, 
                'radii': radii, 
                'defectmask': defectmask, 
                'vertices': vertices}

spio.savemat('{}recon_N{:d}.mat'.format(path, int(N)), recon_dict)
spio.savemat(path + 'data.mat', mdict)


#%%=======================================================================
# Plot data
#=========================================================================

cmin = -0.08
cmax = 0.2

plotfuns.image2D(x_recon, int(np.sqrt(len(x_recon.flatten()))), domain, 'SIRT', path, 'sirt', cmin=cmin, cmax=cmax)
plotfuns.image2D(phantom.flatten(order='F'), int(np.sqrt(len(phantom.flatten()))), domain, 'Phantom', path, 'phantom', cmin=cmin, cmax=cmax)
plotfuns.sino(b_data, p, q, path, 'sino_noisy', 'Sinogram with {} % noise'.format(rnl*100), cmin = None, cmax = None)

fig,ax = plt.subplots(nrows = 1, ncols = 1)
ax.imshow(x_true.flatten(order="F").reshape((N,N), order = "C"))
for i in range(centers.shape[1]):
        corner_x1 = centers[0,i]-25
        corner_x2 = centers[0,i]+25
        corner_y1 = centers[1,i]-25
        corner_y2 = centers[1,i]+25
        ax.plot(np.array([corner_y2, corner_y2, corner_y1, corner_y1, corner_y2]), 
                np.array([corner_x1, corner_x2, corner_x2, corner_x1, corner_x1]),
                'r-')
plt.savefig(path + 'phantom_with_zoommarks.png')

fig,ax = plt.subplots(nrows = 1, ncols = 1)
size = 13
defectno = 13
test = np.vstack([x_true.flatten(order = "F"), x_true.flatten(order = "F"), x_true.flatten(order = "F")]).T
print(test.shape)
defectidx = np.array([int(np.round(centers[1,defectno]-size)),int(np.round(centers[1,defectno]+size)), int(np.round(centers[0,defectno]-size)),int(np.round(centers[0,defectno]+size))])
ax.imshow(test.reshape((N,N,3), order = "F")[defectidx[0]:defectidx[1], defectidx[2]:defectidx[3], :].reshape(((defectidx[1]-defectidx[0])*(defectidx[3]-defectidx[2]),3), order  = "F")[:,0].reshape(((defectidx[1]-defectidx[0]),(defectidx[3]-defectidx[2])), order = "F"))
plt.savefig(path + 'defectzoom.png')