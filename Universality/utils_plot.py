# Author: Simon Guldager Andersen


## Imports:
import os
import pickle as pkl
import warnings
import time
import glob

from functools import wraps
from multiprocessing.pool import Pool as Pool

import numpy as np
from scipy.stats import moment
import matplotlib.pyplot as plt

# Plotting functions -------------------------------------------------------------------

def plot_moments(moment_dict, act_dict, L_list, moment_label, plot_binder=False, savepath=None):
    """Plot moments of CID vs activity for different system sizes."""
    
    fig, ax = plt.subplots(ncols=4, figsize=(16, 4))
    marker_shape = ['d', '*', '^', 'v', 'D', 'P-', 'X-', 'h-', 'd-', 'p-', 'H-', '8-', '1-', '2-']
    xlim = (0.018, 0.035)
    ylims = [(0, None), (0, 1), (0, None)]

    for i, LX in enumerate(L_list):
        act_list = act_dict[LX]
        moments = moment_dict[LX]
        binder = 1 - moments[3,:] / (3 * moments[1,:]**2)
        ax[0].plot(act_list, moments[0, :], label=f'L={LX}', marker=marker_shape[i % len(marker_shape)], alpha=0.7)
        ax[1].plot(act_list, moments[1, :], label=f'L={LX}', marker=marker_shape[i % len(marker_shape)], alpha=0.7)
        if plot_binder:
            ax[2].plot(act_list, binder, label=f'L={LX}', marker=marker_shape[i % len(marker_shape)], alpha=0.7)
        else:
            ax[2].plot(act_list, moments[2, :], label=f'L={LX}', marker=marker_shape[i % len(marker_shape)], alpha=0.7)
        ax[3].plot(act_list, moments[3, :], label=f'L={LX}', marker=marker_shape[i % len(marker_shape)], alpha=0.7)

    ax[0].set_ylabel(rf'Mean({moment_label})')
    ax[1].set_ylabel(rf'Var({moment_label})')
    ax[2].set_ylabel(rf' Binder Cumulant ({moment_label})' if plot_binder else rf'Skew({moment_label})')
    ax[3].set_ylabel(rf'Kurt({moment_label})')

    for a in ax:
        a.set_xlabel(r'Activity ($\tilde{\zeta}$)')
        a.legend(handlelength=1)
        a.set_xlim(xlim)
        ylims = a.get_ylim()
        a.vlines(0.022, ylims[0], ylims[1], colors='k', linestyles='dashed', lw=1, alpha=0.8,zorder=-5)
        a.set_ylim(ylims)
    if savepath:
        fig.savefig(savepath, bbox_inches='tight', dpi=620, pad_inches=.05)
    return fig, ax