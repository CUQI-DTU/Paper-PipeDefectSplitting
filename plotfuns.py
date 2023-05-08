################################################
# By Silja L. Christensen
# August 2022 - May 2023

# This file contains plot function used by "plotting.py"
################################################

#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

# =================================================================
# Plot domain
# =================================================================

def image2D(imagevec, N, domain, path, filename, cmin=None, cmax=None, xlim = None, ylim = None, colmap = "gray", log = False, flat_order = "C", rot_k = 0, font_size = 16, font_size_axis = 13, axis_visible = True, x_ticks = None, y_ticks = None, figsize = (5.7,4)):
    # fontsize 20 for full, synth
    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(wspace=.5)

    ax = plt.subplot(111)
    if log == False:
        cs = ax.imshow(np.rot90(imagevec.reshape((N,N), order = flat_order),k=rot_k), extent=[0, N, N, 0], aspect='equal', cmap=colmap, vmin = cmin, vmax = cmax)
    else:
        cs = ax.imshow(np.rot90(imagevec.reshape((N,N), order = flat_order),k=rot_k), extent=[0, N, N, 0], aspect='equal', cmap=colmap, vmin = cmin, vmax = cmax, norm="log")
    cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.03,ax.get_position().height])
    cbar = plt.colorbar(cs, cax=cax) 
    if xlim is not None:
        ax.set_xlim(xlim[0],xlim[1])
        ax.set_ylim(ylim[0],ylim[1])
    if axis_visible == False:
        ax.tick_params(axis='both', which='both', length=0)
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
    else:
        ax.tick_params(axis='both', which='both', labelsize=font_size_axis)
    if x_ticks is not None:
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
    cbar.ax.tick_params(labelsize=font_size) 

    plt.savefig(path + filename + '.png', transparent=False)
    plt.savefig(path + filename + '.eps', format='eps')
    plt.close()

def image2D_zoom_synthdefects(imagevec, N, realdata, vertices, centers, path, filename, cmin=None, cmax=None, xlim = None, ylim = None, colmap = "gray", log = False, flat_order = "C", rot_k = 0):
    if realdata == False:
        xlim = centers[1,:]
        ylim = centers[0,:]
    else:
        xlim = np.array([85, 220, 280, 255, 360, 135])/500*N
        ylim = np.array([165, 155, 140, 465, 375, 255])/500*N
    xlim = xlim.astype(int)
    ylim = ylim.astype(int)
        
    if realdata == False:
        fig, axs = plt.subplots(nrows = 4, ncols = 4, figsize = (18, 16))
    else:
        fig, axs = plt.subplots(nrows = 2, ncols = 3, figsize = (12, 8))
    axs = axs.flatten()
    for i in range(len(xlim)):
        if log == False:
            cs = axs[i].imshow(np.rot90(imagevec.reshape((N,N), order = flat_order),k=rot_k), extent=[0, N, N, 0], vmin = cmin, vmax = cmax, interpolation = "none", cmap = colmap)
        else:
            cs = axs[i].imshow(np.rot90(imagevec.reshape((N,N), order = flat_order),k=rot_k), extent=[0, N, N, 0], vmin = cmin, vmax = cmax, interpolation = "none", cmap = colmap, norm="log")
        axs[i].set_xlim(xlim[i]-25/500*N,xlim[i]+25/500*N)
        axs[i].set_ylim(ylim[i]+25/500*N,ylim[i]-25/500*N)
        if realdata == False and i < 12:
            axs[i].plot(np.hstack([vertices[i][:,0], vertices[i][0,0]]), np.hstack([vertices[i][:,1], vertices[i][0,1]]), '--', color = "red")
        elif realdata == False and i == 15:
            # ang = np.array([3*np.pi/4+np.pi/9, 3*np.pi/4+np.pi/9, 3*np.pi/4, 3*np.pi/4, 3*np.pi/4-np.pi/9])
            # dist = 20.25/domain*N
            ang = np.pi/4-np.pi/9-60/180*np.pi
            dist = 20.25/55*N
            circle = plt.Circle( (N/2 + dist*np.cos(ang), N/2 - dist*np.sin(ang)),
                                      0.3/55*N,
                                      fill = False, ls = '--', edgecolor = "red")
            axs[i].add_artist(circle)
        elif realdata == False and i == 14:
            ang = np.pi/4-60/180*np.pi
            dist = 20.25/55*N
            circle1 = plt.Circle( (N/2 + dist*np.cos(ang), N/2 - dist*np.sin(ang)),
                                      0.3/55*N,
                                      fill = False, ls = '--', edgecolor = "red")
            circle2 = plt.Circle( (N/2 + dist*np.cos(ang), N/2 - dist*np.sin(ang)),
                                      1/55*N,
                                      fill = False, ls = '--', edgecolor = "red")
            axs[i].add_artist(circle1)
            axs[i].add_artist(circle2)
        elif realdata == False and i == 13:
            ang = np.pi/4+np.pi/9-60/180*np.pi
            dist = 20.25/55*N
            circle1 = plt.Circle( (N/2 + dist*np.cos(ang), N/2 - dist*np.sin(ang)),
                                      0.3/55*N,
                                      fill = False, ls = '--', edgecolor = "red")
            circle2 = plt.Circle( (N/2 + dist*np.cos(ang), N/2 - dist*np.sin(ang)),
                                      1/55*N,
                                      fill = False, ls = '--', edgecolor = "red")
            axs[i].add_artist(circle1)
            axs[i].add_artist(circle2)
        elif realdata == False and i == 12:
            c_cross = np.array([N/2, N/2-20.25/55*N])
            a = (2/np.sqrt(2))/55*N
            b = (0.2/np.sqrt(2))/55*N
            axs[i].plot(c_cross[0]+np.array([a, b, a, a-b, 0, -a+b, -a, -b, -a, -a+b, 0, a-b, a]), c_cross[1]+np.array([a-b, 0, -a+b, -a, -b, -a, -a+b, 0, a-b, a, b, a, a-b]), '--', color = "red")

        axs[i].set_title("Defect no. {}".format(i+1))
    if realdata == False:
        cax_left = axs[11].get_position().x1+0.02
        cax_bottom = axs[11].get_position().y0
        cax_height = axs[5].get_position().y1 - axs[11].get_position().y0
    else:
        cax_left = axs[5].get_position().x1+0.02
        cax_bottom = axs[5].get_position().y0
        cax_height = axs[2].get_position().y1 - axs[5].get_position().y0
    cax_width = 0.015
    cax = fig.add_axes([cax_left,cax_bottom,cax_width,cax_height])
    cbar = plt.colorbar(cs, cax=cax)

    plt.savefig(path + filename + '.png', transparent=False)
    plt.savefig(path + filename + '.eps', format='eps')
    plt.close()

def image2D_zoom_manuscript(meanvec, stdvec, N, realdata, vertices, centers, path, filename, cmin_mean=None, cmax_mean=None, cmin_std=None, cmax_std=None, colmap = "gray", flat_order = "C", rot_k = 0, axis_visible = False):
    font_size_cbar = 14
    font_size_title = 14
    font_size_axis = 11
    if realdata == False:
        #size = np.array([50, 50, 100, 50])
        #xlim = np.array([centers[1,3], centers[1,12], centers[1,13]+(centers[1,14]-centers[1,13])/2, centers[1,15]])
        #ylim = np.array([centers[0,3], centers[0,12], centers[0,13]+(centers[0,14]-centers[0,13])/2, centers[0,15]])
        size = np.array([50, 50, 50, 50, 50])
        xlim = np.array([centers[1,3], centers[1,12], centers[1,13], centers[1,14], centers[1,15]])
        ylim = np.array([centers[0,3], centers[0,12], centers[0,13], centers[0,14], centers[0,15]])
    else:
        xlim = np.array([130, 270, 305, 410])/500*N
        ylim = np.array([110, 105, 415, 325])/500*N
        size = np.array([80, 50, 50, 50])/500*N
    xlim = xlim.astype(int)
    ylim = ylim.astype(int)
        
    if realdata == False:
        fig, axs = plt.subplots(nrows = 2, ncols = 5, figsize = (11, 4.5), gridspec_kw=dict(wspace=0.05, hspace=0.05))
    else:
        fig, axs = plt.subplots(nrows = 2, ncols = 4, figsize = (11, 5.5), gridspec_kw=dict(wspace=0.05, hspace=0.02))
    #axs = axs.flatten()
    for i in range(len(xlim)):
        cs_mean = axs[0,i].imshow(np.rot90(meanvec.reshape((N,N), order = flat_order),k=rot_k), extent=[0, N, N, 0], vmin = cmin_mean, vmax = cmax_mean, interpolation = "none", cmap = colmap)
        cs_std = axs[1,i].imshow(np.rot90(stdvec.reshape((N,N), order = flat_order),k=rot_k), extent=[0, N, N, 0], vmin = cmin_std, vmax = cmax_std, interpolation = "none", cmap = colmap, norm = "log")        
        axs[0,i].set_xlim(xlim[i]-size[i]/2/500*N,xlim[i]+size[i]/2/500*N)
        axs[1,i].set_xlim(xlim[i]-size[i]/2/500*N,xlim[i]+size[i]/2/500*N)
        axs[0,i].set_ylim(ylim[i]+size[i]/2/500*N,ylim[i]-size[i]/2/500*N)
        axs[1,i].set_ylim(ylim[i]+size[i]/2/500*N,ylim[i]-size[i]/2/500*N)

        if axis_visible == False:
            axs[0,i].tick_params(axis='both', which='both', length=0)
            axs[1,i].tick_params(axis='both', which='both', length=0)
            plt.setp(axs[0,i].get_xticklabels(), visible=False)
            plt.setp(axs[0,i].get_yticklabels(), visible=False)
            plt.setp(axs[1,i].get_xticklabels(), visible=False)
            plt.setp(axs[1,i].get_yticklabels(), visible=False)
        else:
            axs[0,i].tick_params(axis='both', which='both', labelsize=font_size_axis)
            axs[1,i].tick_params(axis='both', which='both', labelsize=font_size_axis)

        if realdata == False: #and i == 0:
            axs[0,i].plot(np.hstack([vertices[3][:,0], vertices[3][0,0]]), np.hstack([vertices[3][:,1], vertices[3][0,1]]), '--', color = "red")
            axs[1,i].plot(np.hstack([vertices[3][:,0], vertices[3][0,0]]), np.hstack([vertices[3][:,1], vertices[3][0,1]]), '--', color = "red")

            ang = np.pi/4-np.pi/9-60/180*np.pi
            dist = 20.25/55*N
            for j in range(2):
                circle = plt.Circle( (N/2 + dist*np.cos(ang), N/2 - dist*np.sin(ang)),
                                        0.3/55*N,
                                        fill = False, ls = '--', edgecolor = "red")
                axs[j,i].add_artist(circle)

            ang = np.pi/4-60/180*np.pi
            dist = 20.25/55*N
            for j in range(2):
                circle1 = plt.Circle( (N/2 + dist*np.cos(ang), N/2 - dist*np.sin(ang)),
                                        0.3/55*N,
                                        fill = False, ls = '--', edgecolor = "red")
                circle2 = plt.Circle( (N/2 + dist*np.cos(ang), N/2 - dist*np.sin(ang)),
                                        1/55*N,
                                        fill = False, ls = '--', edgecolor = "red")
                axs[j,i].add_artist(circle1)
                axs[j,i].add_artist(circle2)

            ang = np.pi/4+np.pi/9-60/180*np.pi
            dist = 20.25/55*N
            for j in range(2):
                circle1 = plt.Circle( (N/2 + dist*np.cos(ang), N/2 - dist*np.sin(ang)),
                                        0.3/55*N,
                                        fill = False, ls = '--', edgecolor = "red")
                circle2 = plt.Circle( (N/2 + dist*np.cos(ang), N/2 - dist*np.sin(ang)),
                                        1/55*N,
                                        fill = False, ls = '--', edgecolor = "red")
                axs[j,i].add_artist(circle1)
                axs[j,i].add_artist(circle2)

            c_cross = np.array([N/2, N/2-20.25/55*N])
            a = (2/np.sqrt(2))/55*N
            b = (0.2/np.sqrt(2))/55*N
            for j in range(2):
                axs[j,i].plot(c_cross[0]+np.array([a, b, a, a-b, 0, -a+b, -a, -b, -a, -a+b, 0, a-b, a]), c_cross[1]+np.array([a-b, 0, -a+b, -a, -b, -a, -a+b, 0, a-b, a, b, a, a-b]), '--', color = "red")
        
        if realdata == True:
            axs[0,i].set_title("Defect no. {}".format(i+1), fontsize = font_size_title, weight = "bold")
        else:
            axs[0,0].set_title("Defect no. 4", fontsize = font_size_title, weight = "bold")
            axs[0,1].set_title("Defect no. 13", fontsize = font_size_title, weight = "bold")
            axs[0,2].set_title("Defect no. 14", fontsize = font_size_title, weight = "bold")
            axs[0,3].set_title("Defect no. 15", fontsize = font_size_title, weight = "bold")
            axs[0,4].set_title("Defect no. 16", fontsize = font_size_title, weight = "bold")
        axs[0,0].set_ylabel("Mean", fontsize = font_size_title, weight = "bold")
        axs[1,0].set_ylabel("Std", fontsize = font_size_title, weight = "bold")

    if realdata == False:
        axno = 4
    else:
        axno = 3
    cax_left = axs[0,axno].get_position().x1+0.01
    cax_bottom = axs[0,axno].get_position().y0
    cax_height = axs[0,axno].get_position().y1 - axs[0,axno].get_position().y0
    cax_width = 0.015
    cax = fig.add_axes([cax_left,cax_bottom,cax_width,cax_height])
    cbar = plt.colorbar(cs_mean, cax=cax)
    cbar.ax.tick_params(labelsize=font_size_cbar)

    cax_left = axs[1,axno].get_position().x1+0.01
    cax_bottom = axs[1,axno].get_position().y0
    cax_height = axs[1,axno].get_position().y1 - axs[1,axno].get_position().y0
    cax_width = 0.015
    cax = fig.add_axes([cax_left,cax_bottom,cax_width,cax_height])
    cbar = plt.colorbar(cs_std, cax=cax)
    cbar.ax.tick_params(labelsize=font_size_cbar) 

    plt.savefig(path + filename + '.png', transparent=False)
    plt.savefig(path + filename + '.eps', format='eps')
    plt.close()


# =================================================================
# Plot the sinogram
# =================================================================
def sino(b_data, p, q, path, filename, cmin = None, cmax = None, colmap = "gray", flat_order = "F"):
    fig = plt.figure(figsize=(4,5))
    fig.subplots_adjust(wspace=.5)

    font_size = 12
    ax2 = plt.subplot(111)
    cs = ax2.imshow(b_data.reshape(p,q, order = flat_order), cmap=colmap, vmin = cmin, vmax = cmax, aspect=2)
    cax = fig.add_axes([ax2.get_position().x1+0.02,ax2.get_position().y0,0.03,ax2.get_position().height])
    cbar = plt.colorbar(cs, cax=cax) 
    cbar.ax.tick_params(labelsize=font_size) 
    ax2.set_ylabel('Projection angle [deg]', fontsize = font_size)
    ax2.set_xlabel('Detector pixel', fontsize = font_size)
    ax2.set_yticks(np.linspace(0,p,7, endpoint=True))
    ax2.tick_params(axis='both', which='both', labelsize=font_size)
    fig.subplots_adjust(wspace=.5)
    plt.savefig(path + filename + '.png', transparent=False)
    plt.savefig(path + filename + '.eps', format='eps')
    plt.close()

# =================================================================
# Plot x chains
# =================================================================
def xchains(x_chains, x_chains_thin, chainno, tau_max, filename, path, log = False, hist = 1):

    nochains = len(chainno)
    cmap = plt.get_cmap("tab10")

    fig, axs = plt.subplots(nrows=nochains, ncols=2, figsize=(10,int(2*nochains)))
    if nochains == 1:
        axs = np.expand_dims(axs, axis=0)

    for i in range(nochains):
        if log == False:
            axs[i,0].plot(x_chains[i,:], color = cmap(i))
            axs[i,1].plot(x_chains_thin[i,:], color = cmap(i))
        else:
            axs[i,0].semilogy(x_chains[i,:], color = cmap(i))
            axs[i,1].semilogy(x_chains_thin[i,:], color = cmap(i))

    axs[0,0].set_title('Pixel chain')
    axs[0,1].set_title('Pixel chain after burnin and thin with tau = %i' %(int(tau_max)))

    plt.tight_layout()
    plt.savefig(path + filename + '.png')
    plt.savefig(path + filename + '.eps', format='eps')
    plt.close()

    if hist == 1:
        fig, axs = plt.subplots(nrows=int(np.ceil(nochains/2)), ncols=2, figsize=(10,int(1*nochains+1)))
        axs = axs.flatten()
        for i in range(nochains):
            axs[i].hist(x_chains_thin[i,:], bins = 30, color = cmap(i), density = True)

        plt.tight_layout()
        plt.savefig(path + filename + 'hist.png', transparent=False)
        plt.savefig(path + filename + 'hist.eps', format='eps')
        plt.close()

# =================================================================
# Mark chain pixels in image
# =================================================================
def markchainpixels(x_mean, x_std, chainno, N, realdata, cmin, cmax, path, colmap = "gray", flat_order = "C", rot_k = 0):

    # compute pixel positions
    tmp = np.zeros(N**2)
    pixel_x = np.zeros(len(chainno))
    pixel_y = np.zeros(len(chainno))
    for i in range(len(chainno)):
        tmp[chainno[i]] = chainno[i]
    tmp = np.rot90(tmp.reshape((N,N), order = flat_order), k = rot_k)
    for i in range(len(chainno)):
        pixel = np.where(tmp == chainno[i])
        pixel_x[i] = pixel[1]
        pixel_y[i] = pixel[0]

    fig = plt.figure(figsize=(5,4))
    ax2 = plt.subplot(111)
    cs = ax2.imshow(np.rot90(x_mean.reshape((N,N), order = flat_order),k=rot_k), aspect='equal', cmap=colmap, vmin = cmin, vmax = cmax)
    cax = fig.add_axes([ax2.get_position().x1+0.01,ax2.get_position().y0,0.05,ax2.get_position().height])
    cbar = plt.colorbar(cs, cax=cax) 
    for i in range(len(chainno)):
        ax2.plot(pixel_x[i], pixel_y[i], 'o')
    ax2.tick_params(axis='both', which='both', length=0)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax2.get_yticklabels(), visible=False)
    if realdata == True:
        ax2.set_xlim(0.25, 0.75)  
        ax2.set_ylim(0.25, 0.75) 
    plt.savefig(path + 'posterior_mean_marked_pixels.png')
    plt.savefig(path + 'posterior_mean_marked_pixels' + '.eps', format='eps')

    fig = plt.figure(figsize=(5,4))
    ax3 = plt.subplot(111)
    cs = ax3.imshow(np.rot90(x_std.reshape((N,N), order = flat_order), k = rot_k), aspect='equal', cmap=colmap)
    cax = fig.add_axes([ax3.get_position().x1+0.01,ax3.get_position().y0,0.05,ax3.get_position().height])
    cbar = plt.colorbar(cs, cax=cax) 
    for i in range(len(chainno)):
        ax3.plot(pixel_x[i], pixel_y[i], 'o')
    ax3.tick_params(axis='both', which='both', length=0)
    plt.setp(ax3.get_xticklabels(), visible=False)
    plt.setp(ax3.get_yticklabels(), visible=False)
    if realdata == True:
        ax3.set_xlim(0.25, 0.75)  
        ax3.set_ylim(0.25, 0.75) 
    plt.savefig(path + 'posterior_std_marked_pixels.png')
    plt.close()
