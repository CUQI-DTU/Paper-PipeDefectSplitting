################################################
# By Silja L. Christensen
# August 2022 - May 2023

# This script contains functions used for postprocessing in main script.
# The functions are originally from: https://github.com/CUQI-DTU/Paper-SGP
################################################

import numpy as np
import random

def relative_error(x_s, x_truef, xt_norm, n_t):
    x_e = np.zeros(n_t)
    for s in range(n_t):
        x_e[s] = np.linalg.norm(x_s[:, s]-x_truef)/xt_norm
    return x_e

def statistics(X, quant):
    # This function performs statictics on the posterior.
    # Computes mean, standard deviation, and quantiles specified by quant.

    x_mean = np.mean(X, axis=1)
    x_std = np.std(X, axis=1)
    x_q = np.zeros((np.size(X, 0), np.size(quant)))
    for i, q in enumerate(quant):
        x_q[:, i] = np.quantile(X, q, axis=1)

    return x_mean, x_std, x_q

def posterior_realizations(X, npost, rseed):
    # This function picks out posterior realizations
    
    if rseed == 'none':
        post_idx = range(npost)
    else:
        niter = np.size(X, 1)
        random.seed(rseed)
        post_idx = random.sample(range(0, niter), npost)
    post_realiz = X[:, post_idx]

    return post_realiz, post_idx

def burnthin(X, n_b, tau_max):
    # This function removes burnin and thin in chains according to autocorrelation time. 
    # Can be used on any variable X, where the chains are along the rows.
    if X.ndim == 1:
        X_thin = X[n_b::int(tau_max)]
    elif X.ndim == 2:
        X_thin = X[:, n_b::int(tau_max)]

    return X_thin

def iact(dati):
    # Implementation from Computational Uncertainty Quantification for Inverse Problems by John Bardsley
        
    if len(np.shape(dati)) == 1:
        dati =  dati[:, np.newaxis]

    mx, nx  = np.shape(dati)
    
    
    tau = np.zeros(nx)
    m   = np.zeros(nx)
    
    x       = np.fft.fft(dati, axis=0)
    xr      = np.real(x)
    xi      = np.imag(x)
    xr      = xr**2 + xi**2
    xr[0,:] = 0
    xr      = np.real(np.fft.fft(xr, axis=0))
    var     = xr[0,:] / len(dati) / (len(dati)-1)
    
    for j in range(nx):
        if var[j] == 0:
            continue
        
        xr[:,j] = xr[:,j]/xr[0,j]
        summ    = -1/3
        
        for i in range(len(dati)):
            summ = summ + xr[i,j] - 1/6
            if summ < 0:
                tau[j]  = 2*(summ + (i-1)/6)
                m[j]    = i
                break
                
    return tau, m