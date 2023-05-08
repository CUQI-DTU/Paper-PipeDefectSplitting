################################################
# By Silja L. Christensen
# August 2022 - May 2023

# This script plots results from main script and saves them
################################################

#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio

import plotfuns as plotfuns

#%% Load data
realdata = False
only_zoom = False
phantomname = 'DeepSeaOilPipe8'
path = "../Experiments/Splitting/GammaMRFPrior/cuqipy_github/ASTRA/sparseangles20percent/nb2000_ns8000_rnl2_precpipe200000_precout200000_precMRF10000_omega04_sbc1e-05_sinit1e-05_sminval1e-07/"
mat = spio.loadmat(path + 'pp_results.mat')

flat_order = "C"
flat_order_sino = "F"
rot_k = 0 

z0 = mat['z0'][0] 
z_mean = mat['z_mean'][0] 
z_std = mat['z_std'][0] 
z_ess = mat['z_ess'][0]
z_chains = mat['z_chains'][0]
z_chains_thin = mat['z_chains_thin'][0]
post_realiz = mat['post_realiz'][0]

d_mean = mat['d_mean'][0] 
d_std = mat['d_std'][0] 
d_std_resampled = mat['d_std_resampled'][0]
d_ess = mat['d_ess'][0]
d_chains = mat['d_chains'][0]
d_chains_thin = mat['d_chains_thin'][0]
post_d_realiz = mat['post_d_realiz'][0]
post_idx = mat['post_idx'][0]

s_mean = mat['s_mean'][0] 
s_std = mat['s_std'][0] 
s_ess = mat['s_ess'][0]
s_chains = mat['s_chains'][0]
s_chains_thin = mat['s_chains_thin'][0]

w_mean = mat['w_mean'][0] 
w_std = mat['w_std'][0] 
w_ess = mat['w_ess'][0]
w_chains = mat['w_chains'][0]
w_chains_thin = mat['w_chains_thin'][0]

recon_mean = mat['recon_mean'][0] 
recon_std = mat['recon_std'][0] 

chainno = mat['chainno'][0]
tau_max = mat['tau_max'][0][0]

N = mat['N'][0][0]
domain = (55,55)

b_data = mat['b_data'][0]
det_count = mat['det_count'][0][0]
angle_count = mat['angle_count'][0][0]

if realdata == False:
    phantom = mat['phantom'][0]
    vertices = mat['vertices'][0]
    centers = mat['centers'][0]
else:
    vertices = None
    centers = None

try:
    CGLS = mat['ML'][0]
    plotfuns.image2D(CGLS, int(np.sqrt(len(CGLS))), domain[0], path, 'CGLS', flat_order = flat_order, rot_k = rot_k, font_size = 15, axis_visible = False)
except:
    print("No CGLS")

#%%=======================================================================
# Plotting
#=========================================================================

print('Plotting...')
f = open(path + "log.txt", "a")
f.write('Plotting..')
f.close()

cmin_im = -0.08
cmax_im = 0.2

if realdata == True:
    cabsmax = 0.3
    cmin_std = 1e-3
    cmax_std = 5e-2
    ess_log = True
else:
    cabsmax = max(abs(min(d_mean)), abs(max(d_mean)))
    cmin_std = 1e-3
    cmax_std = 5e-2
    ess_log = True

ess_fontsize = 26
x_ticks = np.array([0,100,200,300,400,500])
y_ticks = np.array([0,100,200,300,400,500])

if realdata == False:
    plotfuns.image2D_zoom_synthdefects(d_mean, N, realdata, vertices, centers, path, 'd_mean_zoom_all', cmin=-cabsmax, cmax=cabsmax, xlim = None, ylim = None, colmap = "gray", flat_order = flat_order, rot_k = rot_k)
    plotfuns.image2D_zoom_synthdefects(d_std, N, realdata, vertices, centers, path, 'd_std_zoom_all', log = True, cmin = cmin_std, cmax=cmax_std, xlim = None, ylim = None, colmap = "gray", flat_order = flat_order, rot_k = rot_k)
    plotfuns.image2D_zoom_synthdefects(s_mean, N, realdata, vertices, centers, path, 's_mean_zoom_all', cmin=None, cmax=None, xlim = None, ylim = None, colmap = "gray", log = True, flat_order = flat_order, rot_k = rot_k)
    plotfuns.image2D_zoom_synthdefects(w_mean, N+1, realdata, vertices, centers, path, 'w_mean_zoom_all', cmin=None, cmax=None, xlim = None, ylim = None, colmap = "gray", log = True, flat_order = flat_order, rot_k = rot_k)

plotfuns.image2D_zoom_manuscript(d_mean, d_std, N, realdata, vertices, centers, path, 'd_stats_zoom', cmin_mean=-cabsmax, cmax_mean=cabsmax, cmin_std=cmin_std, cmax_std=cmax_std, colmap = "gray", flat_order = flat_order, rot_k = rot_k)

# Plot with chain pixels marked
#plotfuns.markchainpixels(recon_mean, recon_mean, chainno, N, realdata, cmin_im, cmax_im, path, flat_order = flat_order, rot_k = rot_k)

# Plot z
plotfuns.image2D(z_ess, int(np.sqrt(len(z_ess))), domain[0], path, 'ess_z', flat_order = flat_order, rot_k = rot_k, font_size = ess_fontsize, axis_visible = False, figsize = (5.7,4))
plotfuns.image2D(z_mean, N, domain[0], path, 'z_mean', cmin=cmin_im, cmax=cmax_im, flat_order = flat_order, rot_k = rot_k, x_ticks = x_ticks, y_ticks = y_ticks)
plotfuns.image2D(z0, N, domain[0], path, 'z0', cmin=cmin_im, cmax=cmax_im, flat_order = flat_order, rot_k = rot_k)
plotfuns.image2D(z_std, N, domain[0], path, 'z_std', cmin=None, cmax=None, flat_order = flat_order, rot_k = rot_k, x_ticks = x_ticks, y_ticks = y_ticks)
plotfuns.xchains(z_chains, z_chains_thin, chainno, tau_max, 'z_chains', path)

# Plot d
plotfuns.image2D(d_ess, int(np.sqrt(len(d_ess))), domain[0], path, 'ess_d', log = ess_log, flat_order = flat_order, rot_k = rot_k, font_size = ess_fontsize, axis_visible = False, figsize = (5.7,4))
plotfuns.image2D(d_mean, N, domain[0], path, 'd_mean', cmin=-cabsmax, cmax=cabsmax, colmap = "gray", flat_order = flat_order, rot_k = rot_k, x_ticks = x_ticks, y_ticks = y_ticks)
plotfuns.image2D(d_std, N, domain[0], path, 'd_std_log', cmin=cmin_std, cmax=cmax_std, log = True, flat_order = flat_order, rot_k = rot_k, x_ticks = x_ticks, y_ticks = y_ticks)
plotfuns.xchains(d_chains, d_chains_thin, chainno, tau_max, 'd_chains', path)    

# Plot s
plotfuns.image2D(s_ess, int(np.sqrt(len(s_ess))), domain[0], path, 'ess_s', log = ess_log, flat_order = flat_order, rot_k = rot_k, font_size = ess_fontsize, axis_visible = False, figsize = (5.7,4))
plotfuns.image2D(s_mean, N, domain[0], path, 's_mean', cmin=None, cmax=None, log = True, flat_order = flat_order, rot_k = rot_k, x_ticks = x_ticks, y_ticks = y_ticks)
plotfuns.image2D(s_std, N, domain[0], path, 's_std', cmin=None, cmax=None, log = True, flat_order = flat_order, rot_k = rot_k, x_ticks = x_ticks, y_ticks = y_ticks)
plotfuns.xchains(s_chains, s_chains_thin, chainno, tau_max, 's_chains', path, log = True)

# Plot w
plotfuns.image2D(w_ess, int(np.sqrt(len(w_ess))), domain[0], path, 'ess_w',log = ess_log,  flat_order = flat_order, rot_k = rot_k, font_size = ess_fontsize, axis_visible = False, figsize = (5.7,4))
plotfuns.image2D(w_mean, N+1, domain[0], path, 'w_mean', cmin=None, cmax=None, log = True, flat_order = flat_order, rot_k = rot_k, x_ticks = x_ticks, y_ticks = y_ticks)
plotfuns.image2D(w_std, N+1, domain[0], path, 'w_std', cmin=None, cmax=None, log = True, flat_order = flat_order, rot_k = rot_k, x_ticks = x_ticks, y_ticks = y_ticks)
plotfuns.xchains(w_chains, w_chains_thin, chainno, tau_max, 'w_chains', path, log = True)

# Plot reconstruction
plotfuns.image2D(recon_mean, N, domain[0], path, 'recon_mean', cmin=cmin_im, cmax=cmax_im, flat_order = flat_order, rot_k = rot_k, font_size = 12, font_size_axis = 12)
plotfuns.image2D(recon_std, N, domain[0], path, 'recon_std', cmin=None, cmax=None, flat_order = flat_order, rot_k = rot_k)
size = 13
if realdata == False:
    defectno = 13
    defectidx = np.array([int(np.round(centers[1,defectno]-size)),int(np.round(centers[1,defectno]+size)), int(np.round(centers[0,defectno]-size)),int(np.round(centers[0,defectno]+size))])
else:
    defectidx = np.array([int(np.round(410-size)),int(np.round(410+size)), int(np.round(325-size)),int(np.round(325+size))])
plotfuns.image2D(recon_mean.reshape((N,N), order = "F")[defectidx[0]:defectidx[1], defectidx[2]:defectidx[3]].flatten(order = "F"), defectidx[1]-defectidx[0], domain[0], path, 'ess_recon', cmin=cmin_im, cmax=cmax_im, flat_order = flat_order, rot_k = rot_k, font_size = ess_fontsize, axis_visible = False, figsize = (5.7,4))

# Plot posterior realizations
for i in range(6):
    plotfuns.image2D(post_realiz[:,i], N, domain[0], path, 'posterior_realiz{}'.format(post_idx[i]), cmin=cmin_im, cmax=cmax_im, flat_order = flat_order, rot_k = rot_k)
    plotfuns.image2D(post_d_realiz[:,i], N, domain[0], path, 'posterior_d_realiz{}'.format(post_idx[i]), flat_order = flat_order, rot_k = rot_k)

# Plot sinogram, phantom and error
if realdata == False:
    plotfuns.image2D(phantom.flatten(order='F'), int(np.sqrt(len(phantom.flatten()))), domain[0], path, 'phantom', cmin=cmin_im, cmax=cmax_im)
plotfuns.sino(b_data, angle_count, det_count, path, 'sino_noisy', cmin = None, cmax = None, flat_order=flat_order_sino)

#%% Phantom with defect numbers
if realdata == False:
    fig = plt.figure(figsize=(5,4))
    fig.subplots_adjust(wspace=.5)

    ax = plt.subplot(111)
    domain = 55
    colmap = "gray"
    cs = ax.imshow(phantom.flatten(order = "F").reshape(1024,1024), extent=[0, 1024, 1024, 0], aspect='equal', cmap=colmap, vmin = cmin_im, vmax = cmax_im)
    col = "red"
    font_size = 12
    g = 30
    x_pos = np.array([g, g, g, g, 0, 0, 0, 0, -10, g, g, g, 58, 20, g, 38])/500*1024
    y_pos = np.array([0, 0, 10, 16, 40, 40, 40, 40, 20, g, g, g, 10, 33, 35, 28])/500*1024
    for i in range(16):
        plt.annotate("{}.".format(i+1), xy = (centers[1,i]/500*1024-x_pos[i], centers[0,i]/500*1024+y_pos[i]), color = col, fontsize = font_size)
    cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.03,ax.get_position().height])
    cbar = plt.colorbar(cs, cax=cax)
    cbar.ax.tick_params(labelsize=font_size) 
    ax.tick_params(axis='both', which='both', labelsize=font_size)
    plt.savefig(path + "phantom_annotated" + '.png', transparent=True)
    plt.savefig(path + "phantom_annotated" + '.eps', format='eps')
else:
    fig = plt.figure(figsize=(5,4))
    fig.subplots_adjust(wspace=.5)
    ax = plt.subplot(111)
    domain = 55
    colmap = "gray"
    cs = ax.imshow(recon_mean.reshape(N,N), extent=[0, N, N, 0], aspect='equal', cmap=colmap, vmin = cmin_im, vmax = cmax_im)
    col = "red"
    font_size = 12
    plt.annotate("1.", xy = (95/500*N, 120/500*N), color = col, fontsize = font_size)
    plt.annotate("2.", xy = (240/500*N, 115/500*N), color = col, fontsize = font_size)
    plt.annotate("3.", xy = (280/500*N, 425/500*N), color = col, fontsize = font_size)
    plt.annotate("4.", xy = (380/500*N, 340/500*N), color = col, fontsize = font_size)
    cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.03,ax.get_position().height])
    cbar = plt.colorbar(cs, cax=cax)
    cbar.ax.tick_params(labelsize=font_size) 
    ax.tick_params(axis='both', which='both', labelsize=font_size)
    plt.savefig(path + "recon_annotated" + '.png', transparent=True)
    plt.savefig(path + "recon_annotated" + '.eps', format='eps')