################################################
# By Silja L. Christensen
# August 2022 - May 2023

# This script runs Bayesian reconstruction of CT of subsea pipes with built-in defect defection

# Requires installation of CUQIpy: This code runs on a pre-released version of CUQIpy. Install via command:
# pip install git+https://github.com/CUQI-DTU/CUQIpy.git@sprint16_add_JointModel
# Requires real data downloaded from 10.5281/zenodo.6817690 or synthetic data generated with the script generatesynthdata.py

# Processed results are saved and can be plotted with plotting.py
################################################

#%%
import numpy as np
import scipy as sp
import scipy.io as spio
import random
import os
import time
import sys
from PIL import Image

import funs
import postprocessfuns as pp

sys.path.append('../../CUQIpy/') # Delete this line if cuqipy is installed as described above.

import cuqi

# %load_ext autoreload
# %autoreload 2

#%%
#=======================================================================
# Parameters
#=========================================================================
# Data 
realdata = False
flat_order = "F"
rot_k = 0
phantomname = 'DeepSeaOilPipe8'   # choose phantom
ag ="sparseangles20percent" 
data_std = 0.05
rnl = 0.02                      # relative noise level
if realdata == False:
    datapath = '../../FORCE/data/SyntheticData/{}_rnl{:d}_geom{}/'.format(phantomname, int(rnl*100), ag)
else:
    datapath = '../../FORCE/data/Data_20180911/'
datafile = 'sinoN8.dat'

# Reconstruction domain
N = 500          # reconstruction of NxN pixels

# Samplers
n_b = int(2000)          # burn-in 
n_s = int(8000)             # number of saved samples
n_t = n_s+n_b               # total number of saved samples

# Gamma MRF prior
omega0 = 4
s_bc = 1e-5
s_init = 1e-5
s_minval = 1e-7 # not passed to myIGConjugate. Please change myIGConjugate sampler manually in funs

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
if realdata == True:
    path1 = '../../../../../../work3/swech/' + ag + '/nb{}_ns{}_real_'.format(n_b, n_s)
else:
    path1 = '../../../../../../work3/swech/' + ag + '/nb{}_ns{}_'.format(n_b, n_s)

path2 = 'rnl{:d}_precpipe{:d}_precout{:d}_precMRF{:d}_omega0{}_sbc{}_sinit{}_sminval{}'.format(int(rnl*100), int(prec_vals[3]), int(prec_vals[4]), int(prec_MRF), omega0, s_bc, s_init, s_minval)

path = path1 + path2 + "/"
os.makedirs(path, exist_ok=True)

#%%=======================================================================
# Create/load sinogram
#=========================================================================

# Aqusition geometry
det_count, theta, stc, ctd, shift, vectors, dl, dlA = funs.geom_Data20180911(ag)
angle_count = len(theta)
domain=(55, 55)

# Load data
if realdata == False:
    data = spio.loadmat(datapath + 'data.mat')
    lambd = data['lambd'][0][0]
    phantom = data['phantom']
    sino_astra = data['sino_astra']
    b_data = data['b_data'][0]

    # underlying true in same dimension as reconstruction
    z_true, radii = getattr(funs, phantomname)(N,False)
    z_truef = z_true.flatten(order=flat_order)
    zt_norm = np.linalg.norm(z_truef)

    u_true, radii, defectmask, vertices, centers = getattr(funs, phantomname)(N,True)

    u_truef = u_true.flatten(order="F")
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

    # compute noise parameters
    noise_std = data_std
    lambd = 1/noise_std**2

#%% =================================================================
# Reconstruction model
# ===================================================================
beam_type="fanflat_vec"
proj_type='cuda'
angles=theta
beamshift_x=shift
source_y=-stc
detector_y=ctd
det_length=dlA
domain=(55, 55)

# Astra model
A = funs.myShiftedFanBeam2DModel(beam_type=beam_type, 
                    proj_type=proj_type,
                    im_size=(N, N), 
                    det_count=det_count, 
                    angles=angles, 
                    beamshift_x=beamshift_x, 
                    source_y=source_y, 
                    detector_y=detector_y, 
                    det_length=det_length, 
                    domain=domain)
    
# =================================================================
# Observed data
# =================================================================
b_data = cuqi.samples.CUQIarray(b_data.reshape(sino_astra.shape).flatten(order="F"), geometry = cuqi.geometry.Image2D((angle_count, det_count), order = "F"))
m = det_count*angle_count

# =================================================================
# Prior
# =================================================================
# Structural Gaussian Prior on z
mask, IID_mean, _, IID_sqrtprec, D1, D2 = funs.SGP(N, domain[0], maskradii, maskcenterinner, maskcenterouter,maskid, mu_vals, prec_vals, bndcond = 'zero', flat_order = flat_order, rot_k = rot_k)
GMRF_mean = np.zeros(N**2)
GMRF_sqrtprec = np.sqrt(prec_MRF)*sp.sparse.vstack([D2, D1])
z = funs.myJointGaussianSqrtPrec(means = [IID_mean, GMRF_mean], sqrtprecs = [IID_sqrtprec, GMRF_sqrtprec], geometry=A.domain_geometry, name="z")
z0 = z.sample()
z.init_point = z0
A_z = A(z) # model that takes z as input parameter

# Prior on d
def d_sqrtprec(s_):
    return np.sqrt(1/s_)
d = cuqi.distribution.Gaussian(mean = np.zeros(N**2), sqrtprec = lambda s: d_sqrtprec(s), geometry=A.domain_geometry, name = "d")
# CGLS reconstruction without prior used for init point
def A_flag(x, flag):
    if flag == 1:
        out = A.forward(x)
    elif flag == 2:
        out = A.adjoint(x)             
    return out     
ML, _ = cuqi.solver.CGLS(A_flag, b_data, np.zeros(N**2), 10).solve()
d.init_point = ML - z0
A_d = A(d)

# Hyperprior on s
def s_scale(w_):
    # Weights computed from auxilirary variables
    W = np.reshape(w_, (N+1, N+1), order = flat_order)
    w1 = W[:-1,:-1].flatten(order=flat_order) # w_(i,j)
    w2 = W[1:,:-1].flatten(order=flat_order) # w_(i+1,j)
    w3 = W[:-1,1:].flatten(order=flat_order) # w_(i,j+1)
    w4 = W[1:,1:].flatten(order=flat_order) # w_(i+1,j+1)
    omega1 = (w1+w2+w3+w4)/4
    return omega0*omega1
s = cuqi.distribution.InverseGamma(shape=omega0, location = 0, scale=lambda w: s_scale(w), geometry=A.domain_geometry, name="s")
s.init_point = s_init*np.ones(N**2)

# Hyperprior on w
def w_rate(s_):
    # Weights computed from s
    # add row and column to S. Defines dirichlet boundary condition.
    S = s_bc*np.ones((N+2, N+2))
    S[1:-1,1:-1] = np.reshape(s_, (N, N), order = flat_order)
    s1 = S[1:,1:].flatten(order=flat_order) # s_(i,j)
    s2 = S[:-1,1:].flatten(order=flat_order) # s_(i-1,j)
    s3 = S[1:,:-1].flatten(order=flat_order) # s_(i,j-1)
    s4 = S[:-1,:-1].flatten(order=flat_order) # s_(i-1,j-1)
    omega2 = (1/s1 + 1/s2 + 1/s3 + 1/s4)/4
    return (omega0*omega2)
w = cuqi.distribution.Gamma(shape=omega0, rate=lambda s: w_rate(s), geometry=cuqi.geometry.Image2D((N+1,N+1), order = A.domain_geometry.order), name="w")
w.init_point = s_init*np.ones((N+1)**2)

#=======================================================================
# Sample posterior
#=========================================================================
# Setup joint linear model
A_joint = A_z + A_d 

# Setup likelihood, joint distribution and sampler
b  = cuqi.distribution.Gaussian(mean = A_joint, sqrtprec = np.sqrt(lambd), geometry=A.range_geometry, name = "b")
P = cuqi.distribution.JointDistribution(b, z, d, s, w)
sampler = cuqi.sampler.Gibbs(P(b=b_data), {'z': cuqi.sampler.Linear_RTO, 'd': cuqi.sampler.Linear_RTO, 's': funs.myIGConjugate, 'w': funs.myGammaSampler})
    
#%% Sample
np.random.seed(1000)
print('\n***MCMC***\n')
f = open(path + "log.txt", "w")
f.write('\n***MCMC***\n')
f.close()

st = time.time()

samples = sampler.sample(n_t,0)

print('\nElapsed time:', time.time()-st, '\n') 
f = open(path + "log.txt", "a")
f.write('\nElapsed time: {:f} \n'.format(time.time()-st)) 
f.close()

#%%=======================================================================
#Post processing
#=========================================================================

print('Post processing...')
f = open(path + "log.txt", "a")
f.write('Post processing...')
f.close()

z_s = samples['z'].samples
d_s = samples['d'].samples
s_s = samples['s'].samples
w_s = samples['w'].samples

#%%
# Chains for visualization
if realdata == False:
    #Boundary
    pixel1_d_polar = np.array([23, np.pi/4])
    pixel1_d = np.array([pixel1_d_polar[0]*np.cos(pixel1_d_polar[1]), pixel1_d_polar[0]*np.sin(pixel1_d_polar[1])])
    pixel1_xy = pixel1_d*N/domain[0]+N/2
    chainno1 = int(pixel1_xy[0])+(N-int(pixel1_xy[1]))*N
    chainno1_w = int(pixel1_xy[0])+(N+1-int(pixel1_xy[1]))*(N+1)
    #Concrete
    pixel2_d_polar = np.array([20.25, np.pi/4])
    pixel2_d = np.array([pixel2_d_polar[0]*np.cos(pixel2_d_polar[1]), pixel2_d_polar[0]*np.sin(pixel2_d_polar[1])])
    pixel2_xy = pixel2_d*N/domain[0]+N/2
    chainno2 = int(pixel2_xy[0])+(N-int(pixel2_xy[1]))*N
    chainno2_w = int(pixel2_xy[0])+(N+1-int(pixel2_xy[1]))*(N+1)
    # Positive defect
    no = 14-1
    chainno3 = int(centers[0,no])*N+int(centers[1,no]) # 116933
    chainno3_w = int(centers[0,no])*(N+1)+int(centers[1,no])
    # Negative defect
    no = 5-1
    chainno4 = int(centers[0,no])*N+int(centers[1,no])
    chainno4_w = int(centers[0,no])*(N+1)+int(centers[1,no])
    
else:
    #Boundary
    pixel1_xy = np.array([400, 315])
    chainno1 = int(pixel1_xy[0])+(N-int(pixel1_xy[1]))*N
    chainno1_w = int(pixel1_xy[0])+(N+1-int(pixel1_xy[1]))*(N+1)
    # Concrete
    pixel2_xy = np.array([430, 310])
    chainno2 = int(pixel2_xy[0])+(N-int(pixel2_xy[1]))*N
    chainno2_w = int(pixel2_xy[0])+(N+1-int(pixel2_xy[1]))*(N+1)
    # Positive defect
    pixel3_xy = np.array([415, 327])
    chainno3 = int(pixel3_xy[0])+(N-int(pixel3_xy[1]))*N
    chainno3_w = int(pixel3_xy[0])+(N+1-int(pixel3_xy[1]))*(N+1)
    # Negative defect
    pixel4_xy = np.array([405, 325])
    chainno4 = int(pixel4_xy[0])+(N-int(pixel4_xy[1]))*N
    chainno4_w = int(pixel4_xy[0])+(N+1-int(pixel4_xy[1]))*(N+1)

chainno = np.array([chainno1, chainno2, chainno3, chainno4])
chainno_w = np.array([chainno1_w, chainno2_w, chainno3_w, chainno4_w])
#%%
# Chains for visualization, and random chains for iact
chainno_iact = random.sample(range(1, N**2+1), 100)
tau_max = np.ceil(1) # thinning
# Quantiles
quant = np.array([0.025, 0.25, 0.5, 0.75, 0.975])

# z post processing
z_chains = z_s[chainno, :]      # pick out chains for visualization
z_tau_list, _ = pp.iact(z_s[chainno_iact, :].T)  # Autocorrelation time
z_thin = pp.burnthin(z_s, n_b, tau_max)
size = 13
if realdata == False:
    defectno = 13
    defectidx = np.array([int(np.round(centers[1,defectno]-size)),int(np.round(centers[1,defectno]+size)), int(np.round(centers[0,defectno]-size)),int(np.round(centers[0,defectno]+size))])
else:
    defectidx = np.array([int(np.round(410-size)),int(np.round(410+size)), int(np.round(325-size)),int(np.round(325+size))])
z_ess = cuqi.samples.Samples(z_thin.reshape((N,N,n_s), order = "F")[defectidx[0]:defectidx[1], defectidx[2]:defectidx[3], :].reshape(((defectidx[1]-defectidx[0])*(defectidx[3]-defectidx[2]),n_s), order = "F")).compute_ess()
z_chains_thin = pp.burnthin(z_chains, n_b, tau_max)
z_mean, z_std, z_q = pp.statistics(z_thin, quant)


# d post processing
d_chains = d_s[chainno, :]      # pick out chains for visualization
d_tau_list, _ = pp.iact(d_s[chainno_iact, :].T)  # Autocorrelation time
d_thin = pp.burnthin(d_s, n_b, tau_max)
d_ess = cuqi.samples.Samples(d_thin.reshape((N,N,n_s), order = "F")[defectidx[0]:defectidx[1], defectidx[2]:defectidx[3], :].reshape(((defectidx[1]-defectidx[0])*(defectidx[3]-defectidx[2]),n_s),order = "F")).compute_ess()
d_chains_thin = pp.burnthin(d_chains, n_b, tau_max)
d_mean, d_std, d_q = pp.statistics(d_thin, quant)

# s post processing
s_chains = s_s[chainno, :]      # pick out chains for visualization
s_tau_list, _ = pp.iact(s_s[chainno_iact, :].T)  # Autocorrelation time
s_thin = pp.burnthin(s_s, n_b, tau_max)
s_ess = cuqi.samples.Samples(s_thin.reshape((N,N,n_s), order = "F")[defectidx[0]:defectidx[1], defectidx[2]:defectidx[3], :].reshape(((defectidx[1]-defectidx[0])*(defectidx[3]-defectidx[2]),n_s),order = "F")).compute_ess()
s_chains_thin = pp.burnthin(s_chains, n_b, tau_max)
s_mean, s_std, s_q = pp.statistics(s_thin, quant)

# w post processing
w_chains = w_s[chainno_w, :]      # pick out chains for visualization
w_tau_list, _ = pp.iact(w_s[chainno_iact, :].T)  # Autocorrelation time
w_thin = pp.burnthin(w_s, n_b, tau_max)
w_ess = cuqi.samples.Samples(w_thin.reshape((N+1,N+1,n_s), order = "F")[defectidx[0]+1:defectidx[1]+1, defectidx[2]+1:defectidx[3]+1, :].reshape(((defectidx[1]-defectidx[0])*(defectidx[3]-defectidx[2]),n_s),order = "F")).compute_ess()
w_chains_thin = pp.burnthin(w_chains, n_b, tau_max)
w_mean, w_std, w_q = pp.statistics(w_thin, quant)

# Reconstruction postprocessing
# Compute reconstructed image
recon = z_s+d_s
recon_thin = z_thin + d_thin
# Relative error
if realdata == False:
    recon_e = pp.relative_error(recon, u_truef, ut_norm, n_t)
    recon_e_thin = pp.burnthin(recon_e, n_b, tau_max)
# Posterior realizations
npost = 6
rseed = 0
post_realiz, post_idx = pp.posterior_realizations(recon_thin, npost, rseed)
post_d_realiz, post_idx = pp.posterior_realizations(d_thin, npost, rseed)
# Statistics
recon_mean, recon_std, recon_q = pp.statistics(recon_thin, quant)

# Std on different scales
factor = np.array([2,4,8,16,32])
d_std_resampled = []
for j in range(len(factor)):
    # Resample each image in posterior sample
    resampled = np.zeros(((N//factor[j])**2,d_thin.shape[1]))
    for i in range(d_thin.shape[1]):
        resampled[:,i] = np.array(Image.fromarray(d_thin[:,i].reshape(N,N)).resize((N//factor[j],N//factor[j]), resample=Image.ANTIALIAS)).flatten()
    
    # Compute std of resampled samples
    d_std_resampled.append(np.std(resampled, axis=1))



#%%=======================================================================
# Saving
#=========================================================================

mdict={'z_mean': [], 
        'z_std': [],
        'z_q': [], 
        'd_mean': [], 
        'd_std': [],
        'd_std_resampled': [],
        'factor': [],
        'd_q': [], 
        's_mean': [], 
        's_std': [],
        's_q': [], 
        'w_mean': [], 
        'w_std': [],
        'w_q': [], 
        'recon_mean': [],
        'recon_std': [],
        'quant': quant,
        'post_realiz': [],
        'post_d_realiz': [],
        'post_idx': [],
        'npost': [],
        'z_chains': [],
        'z_chains_thin': [],
        'd_chains': [],
        'd_chains_thin': [],
        's_chains': [],
        's_chains_thin': [],
        'w_chains': [],
        'w_chains_thin': [],
        'chainno': chainno,
        'chainno_w': chainno_w,
        'N': N,
        'n_s': n_s,
        'n_b': n_b, 
        'z0': z0,
        'ML': ML,
        'z_tau_list': [],
        'd_tau_list': [],
        's_tau_list': [],
        'w_tau_list': [],
        'z_ess': [],
        'd_ess': [],
        's_ess': [],
        'w_ess': [],
        'chainno_iact': chainno_iact,
        'tau_max': tau_max,
        'phantom': [], 
        'z_true': [],
        'u_true': [],
        'recon_e': [],
        'recon_e_thin': [],
        'b_data': b_data,
        'det_count': det_count,
        'angle_count': angle_count,
        'rnl': rnl,
        'mu_vals': mu_vals,
        'prec_vals': prec_vals,
        'prec_MRF': prec_MRF,
        'defectmask': [],
        'vertices': [],
        'centers': [],
        'maskradii': maskradii,
        'maskcenterinner': maskcenterinner,
        'maskcenterouter': maskcenterouter,
        'mask': mask}

if realdata == False:
    mdict["phantom"].append(phantom)
    mdict["defectmask"].append(defectmask)
    mdict["vertices"].append(vertices)
    mdict["centers"].append(centers)
    mdict["z_true"].append(z_true)
    mdict["u_true"].append(u_true)
    mdict["recon_e"].append(recon_e)
    mdict["recon_e_thin"].append(recon_e_thin)


mdict['z_mean'].append(z_mean)
mdict['z_std'].append(z_std)
mdict['z_q'].append(z_q)
mdict['z_chains'].append(z_chains)
mdict['z_chains_thin'].append(z_chains_thin)
mdict['z_tau_list'].append(z_tau_list)
mdict['z_ess'].append(z_ess)

mdict['d_mean'].append(d_mean)
mdict['d_std'].append(d_std)
mdict['d_std_resampled'].append(d_std_resampled)
mdict['factor'].append(factor)
mdict['d_q'].append(d_q)
mdict['d_chains'].append(d_chains)
mdict['d_chains_thin'].append(d_chains_thin)
mdict['d_tau_list'].append(d_tau_list)
mdict['d_ess'].append(d_ess)
mdict['s_mean'].append(s_mean)
mdict['s_std'].append(s_std)
mdict['s_q'].append(s_q)
mdict['s_chains'].append(s_chains)
mdict['s_chains_thin'].append(s_chains_thin)
mdict['s_tau_list'].append(s_tau_list)
mdict['s_ess'].append(s_ess)
mdict['recon_mean'].append(recon_mean)
mdict['recon_std'].append(recon_std)
mdict['post_realiz'].append(post_realiz)
mdict['post_d_realiz'].append(post_d_realiz)
mdict['post_idx'].append(post_idx)
mdict['npost'].append(npost)
mdict['w_mean'].append(w_mean)
mdict['w_std'].append(w_std)
mdict['w_q'].append(w_q)
mdict['w_chains'].append(w_chains)
mdict['w_chains_thin'].append(w_chains_thin)
mdict['w_tau_list'].append(w_tau_list)
mdict['w_ess'].append(w_ess)

print('Saving...')
f = open(path + "log.txt", "a")
f.write('Saving...')
f.close()

spio.savemat(path + 'pp_results.mat', mdict)


