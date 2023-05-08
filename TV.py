################################################
# By Silja L. Christensen
# August 2022 - May 2023

# This script performs TV or CGLS reconstructions of CT data from subsea pipes

# Requires installation of CIL - Core Imaging Library: https://tomographicimaging.github.io/CIL/nightly/index.html
################################################

#%%
import numpy as np
import scipy.io as spio

import funs
import plotfuns

from cil.framework import AcquisitionGeometry, ImageGeometry, AcquisitionData
from cil.plugins.astra.operators import ProjectionOperator
from cil.optimisation.algorithms import FISTA, CGLS
from cil.optimisation.functions import LeastSquares
from cil.plugins.ccpi_regularisation.functions import FGP_TV
from cil.utilities.display import show2D


#%%
#=======================================================================
# Parameters
#=========================================================================

# Data 
realdata = False
phantomname = 'DeepSeaOilPipe8'   # choose phantom
ag = "limited90" 
if ag == "limited90":
    start_angle = -np.pi/2-2*15/180*np.pi
elif ag == "sparseangles20percent":
    start_angle = 4/180*np.pi
else:
    start_angle = 0
data_std = 0.05

rnl = 0.02                      # relative noise level
if realdata == False:
    datapath = '../../FORCE/data/SyntheticData/{}_rnl{:d}_geom{}/'.format(phantomname, int(rnl*100), ag)
else:
    datapath = '../../FORCE/data/Data_20180911/'

datafile = 'sinoN8.dat'

alpha = 0.4
cmap = 'gray'
device = 'gpu'

if realdata == True:
    output_str = "real_{}".format(ag)
else:
    output_str = "synth_{}".format(ag)


#%%=======================================================================
# Define geometry
#=========================================================================
# Aqusition geometry
det_count, theta, stc, ctd, shift, vectors, dl, dlA = funs.geom_Data20180911(ag)
angle_count = len(theta)

beam_type="fanflat_vec"
proj_type='cuda'
angles=theta
beamshift_x=shift
source_y=-stc
detector_y=ctd
det_length=dlA
domain=(55, 55)

#%%=======================================================================
# Create/load sinogram
#=========================================================================
N = 500          # reconstruction of NxN pixels
if realdata == False:
    data = spio.loadmat(datapath + 'data.mat')
    lambd = data['lambd'][0][0]
    phantom = data['phantom']
    sino_astra = data['sino_astra']
    b_true = data['b_true'][0]
    b_data = data['b_data'][0]
    noise = data['noise'][0]

    # underlying true in same dimension as reconstruction
    x_true, radii = getattr(funs, phantomname)(N,False)
    x_truef = x_true.flatten(order='F')
    xt_norm = np.linalg.norm(x_truef)

    u_true, radii, defectmask, vertices, _= getattr(funs, phantomname)(N,True)
    u_truef = u_true.flatten(order='F')
    ut_norm = np.linalg.norm(u_truef)

else:
    # load data and change to correct data structure
    sino = np.loadtxt(datapath + datafile, delimiter=';')
    sino = sino.astype('float32')
    sino_astra = np.rot90(sino, k = 1)
    sino_astra = sino_astra[:-8, :]
    if ag == 'sparseangles50percent':
        sino_astra  = sino_astra[::4, :]
    elif ag == 'sparseangles20percent':
        sino_astra  = sino_astra[::10, :]
    elif ag == 'sparseangles':
        sino_astra  = sino_astra[::20, :]
    elif ag == 'full':
        sino_astra  = sino_astra[::2, :]
    elif ag == 'limited90':
        sino_astra  = sino_astra[::2, :][15:105,:]
    b_data = sino_astra.flatten()
    m = len(b_data)

    # compute noise parameters
    noise_std = data_std
    lambd = 1/noise_std**2

#%% =================================================================
# Reconstruction model
# =================================================================
cil_ag = AcquisitionGeometry.create_Cone2D(source_position = [beamshift_x,source_y], 
                                        detector_position = [beamshift_x,detector_y], 
                                        detector_direction_x=[1, 0], 
                                        rotation_axis_position=[0, 0])
cil_ag.set_angles(angles=angles, initial_angle = start_angle, angle_unit='radian')
cil_ag.set_panel(num_pixels=det_count, pixel_size=dl)

cil_ig = ImageGeometry(voxel_num_x=N, 
                   voxel_num_y=N, 
                   voxel_size_x=domain[0]/N, 
                   voxel_size_y=domain[0]/N)

A_cil = ProjectionOperator(cil_ig, cil_ag)

#%% setup data
b_data = b_data.reshape(sino_astra.shape).flatten(order="F")
data = AcquisitionData(array=np.rot90(b_data.astype(np.float32).reshape(det_count, angle_count)), geometry=cil_ag)

#%% TV reconstruction
F = LeastSquares(A_cil, data)
TV = FGP_TV(alpha=alpha, device='gpu')
initial = cil_ig.allocate(0.0)
my_TV = FISTA(f=F, 
                g=TV,
                initial=initial, 
                max_iteration=2000, 
                update_objective_interval=1)
my_TV.run(2000, verbose=True)
recon_TV = my_TV.solution
plotfuns.image2D(np.rot90(recon_TV.as_array(), k = 3).flatten(order="C"), N, domain[0], "../output/", "TV_{}_alpha{}".format(output_str, alpha), cmin=-0.08, cmax=0.2, colmap = cmap, axis_visible = False)

#%% CGLS recon
# initial = cil_ig.allocate(0.0)
# my_CGLS = CGLS(operator=A_cil, 
#                 data = data,
#                 x_init=initial, 
#                 max_iteration=1000, 
#                 update_objective_interval=1)
# my_CGLS.run(7, verbose=True)
# recon_CGLS = my_CGLS.solution
# show2D(recon_CGLS, 
#        cmap=cmap, 
#        origin='upper-left')
# plt.savefig('../output/CGLS_CIL.png')

