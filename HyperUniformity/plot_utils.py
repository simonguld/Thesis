# Author: Simon Guldager Andersen
# Date (latest update): 

### SETUP ------------------------------------------------------------------------------------

## Imports:

import numpy as np
import matplotlib.pyplot as plt

plt.style.use('sg_article')

from utils import *


### FUNCTIONS ----------------------------------------------------------------------------------

def plot_structure_factor(kbins, smeans, sstds, k = None, sf_estimated = None):
    """
    Plot structure factor
    """

    k_begin_lines_idx = 3
    kmin, kmax = np.nanmin(kbins), np.nanmax(kbins)
    sf_min, sf_max = np.nanmin(smeans), np.nanmax(smeans)
    x = np.linspace(kmin, kbins[k_begin_lines_idx], 10)

    fig, ax = plt.subplots()
    
    if k and sf_estimated:
        ax.scatter(np.linalg.norm(k, axis = 1), sf_estimated, label='Structure factor', s=2.5, alpha=0.3)

    ax.hlines(1, x[0], kmax, label=r'Possion', linestyles='dashed', colors='k')
    ax.plot(x, smeans[k_begin_lines_idx] * x**0.1 / x[-1]**0.1, label=r'$k^{0.1}$')
    ax.plot(x, smeans[k_begin_lines_idx] * x**0.2 / x[-1]**0.2, label=r'$k^{0.2}$')
    ax.plot(x, smeans[k_begin_lines_idx] * x**0.3 /x[-1]**0.3, label=r'$k^{0.3}$')
    ax.errorbar(kbins, smeans, yerr = sstds, fmt = 's-', label = 'Binned means', alpha = .8, color = 'red', ecolor = 'black', markersize = 5)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks(np.logspace(np.log10(kmin), np.log10(kmax), 5), np.round(np.logspace(np.log10(kmin), np.log10(kmax), 5),2))   
    
    ax.set_ylim([sf_min/2, sf_max + 3])
    ax.set_xlim([kmin - 0.01, kmax + 0.1])
    ax.legend(ncol=3, fontsize = 14)
    ax.set_xlabel(r'$|k|$')
    ax.set_ylabel(r'$S(k)$')
    ax.set_title(r'Scaling of structure factor with $k$')
    fig.tight_layout()
    return fig, ax

def plot_pair_corr_function(rad_arr, pcf_arr, act_idx = None, frame = None):
    """
    Plot pair correlation function
    """

    if act_idx is None:
        act_idx = 0

    try:
        r = rad_arr
        pcf_av = pcf_arr[:, :, act_idx, 0]
        pcf_std = pcf_arr[:, :, act_idx, 1]
    except:
        raise ValueError("No pair correlation function data provided")

    if frame is None:
        pcf_av, pcf_std = calc_weighted_mean(pcf_av, pcf_std, axis = 0)

    title = "Time av. pair correlation function" if frame is None else "Pair correlation function, frame = {}".format(frame)

    fig, ax = plt.subplots()
    ax.errorbar(r, pcf_av, yerr = pcf_std, fmt = '.', markersize = 4, alpha = 0.5)
    ax.set_xlabel(rf"$r$ (radius of observation window)")
    ax.set_ylabel(rf"$g(r)$")
    ax.set_title(title)
    return fig, ax

def sfac_plotter(act_list, kbins, sfac_av, Npoints_to_fit=8, act_idx_bounds=None, ):
    ncols = 4
    nrows = int(np.ceil(len(act_list[act_idx_bounds[0]:act_idx_bounds[1]]) / ncols))
    height = nrows * 4
    fig, ax = plt.subplots(nrows = nrows, ncols = ncols, figsize=(20,height))
    ax = ax.flatten()  

    k_begin_lines_idx = Npoints_to_fit - 1

    s_av = sfac_av[:, :, 0]
    s_std = sfac_av[:, :, 1]
    kmin, kmax = np.nanmin(kbins), np.nanmax(kbins)
    x = kbins[:Npoints_to_fit]

    for i, act in enumerate(act_list[act_idx_bounds[0]: act_idx_bounds[1]]):
        act_idx = act_list.index(act)
        sf_min, sf_max = np.nanmin(s_av[:, act_idx]), np.nanmax(s_av[:, act_idx])
         

        if i == 0:
            ax[i].hlines(1, x[0], kmax, label=r'Possion', linestyles='dashed', colors='k')
            ax[i].errorbar(kbins, s_av[:, act_idx], yerr = s_std[:, act_idx], fmt = 's', \
                        alpha = .6, color = 'blue', ecolor = 'black', markersize = 5, label = 'Binned mean')
            ax[i].plot(x, s_av[k_begin_lines_idx, act_idx] * x**0.1 / x[-1]**0.1, '--', label=r'$k^{0.1}$',alpha=0.5,)
            ax[i].plot(x, s_av[k_begin_lines_idx, act_idx] * x**0.2 / x[-1]**0.2, '--', label=r'$k^{0.2}$',alpha=0.5,)
            ax[i].plot(x, s_av[k_begin_lines_idx, act_idx] * x**0.3 /x[-1]**0.3, '--', label=r'$k^{0.3}$', alpha=0.5,)
            
        else:
            ax[i].errorbar(kbins, s_av[:, act_idx], yerr = s_std[:, act_idx], fmt = 's', \
                        alpha = .6, color = 'blue', ecolor = 'black', markersize = 5,)
            ax[i].hlines(1, x[0], kmax, linestyles='dashed', colors='k')
            ax[i].plot(x, s_av[k_begin_lines_idx, act_idx] * x**0.1 / x[-1]**0.1, '--', alpha=0.5)
            ax[i].plot(x, s_av[k_begin_lines_idx, act_idx] * x**0.2 / x[-1]**0.2, '--', alpha=0.5)
            ax[i].plot(x, s_av[k_begin_lines_idx, act_idx] * x**0.3 /x[-1]**0.3, '--', alpha=0.5)
            

        ax[i].text(0.65, 0.2, rf'$\zeta$ = {act}', transform=ax[i].transAxes, fontsize=14, verticalalignment='top')

        ax[i].set_xscale('log')
        ax[i].set_yscale('log')
        ax[i].set_xticks([kmin, 0.1, kmax], [np.round(kmin,2), 0.1, np.round(kmax,1)])
        ax[i].set_yticks([0.3, 0.4, 0.6, 1], [0.3, 0.4, 0.6, 1])

    fig.suptitle('Time av. structure factor different activities', y=1.05)
    fig.supxlabel(r'$|k|$')
    fig.supylabel(r'$S(k)$', x = 0)
    fig.legend(ncol=6, fontsize = 14, bbox_to_anchor=(0.8, 1.01))
    fig.tight_layout()
    return fig, ax