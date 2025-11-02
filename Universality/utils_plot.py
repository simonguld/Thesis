# Author: Simon Guldager Andersen

## Imports:
import numpy as np
import matplotlib.pyplot as plt
from sympy import div

# CID plotting functions -------------------------------------------------------------------

def plot_moments(moment_dict, act_dict, L_list, moment_label, act_critical=None, xlims=None, plot_binder=False, savepath=None):
    """Plot moments of CID vs activity for different system sizes."""
    
    fig, ax = plt.subplots(ncols=4, figsize=(16, 4))
    marker_shape = ['d', '*', '^', 'v', 'D', 'P-', 'X-', 'h-', 'd-', 'p-', 'H-', '8-', '1-', '2-']

    for i, LX in enumerate(L_list):
        act_list = act_dict[LX]
        moments = moment_dict[LX]
        binder = 1 - moments[3,:] / (3 * moments[1,:]**2)

        ax[0].plot(act_list, moments[0, :], label=f'L={LX}', marker=marker_shape[i % len(marker_shape)], alpha=0.7)
        ax[1].plot(act_list, moments[1, :], label=f'L={LX}', marker=marker_shape[i % len(marker_shape)], alpha=0.7)
        ax[3].plot(act_list, moments[3, :], label=f'L={LX}', marker=marker_shape[i % len(marker_shape)], alpha=0.7)

        if plot_binder:
            ax[2].plot(act_list, binder, label=f'L={LX}', marker=marker_shape[i % len(marker_shape)], alpha=0.7)
        else:
            ax[2].plot(act_list, moments[2, :], label=f'L={LX}', marker=marker_shape[i % len(marker_shape)], alpha=0.7)
        
    ax[0].set_ylabel(rf'Mean({moment_label})')
    ax[1].set_ylabel(rf'Var({moment_label})')
    ax[2].set_ylabel(rf' Binder Cumulant ({moment_label})' if plot_binder else rf'Skew({moment_label})')
    ax[3].set_ylabel(rf'Kurt({moment_label})')

    for axx in ax:
        axx.set(xlabel=r'Activity ($\tilde{\zeta}$)')
        xlim = axx.get_xlim() if xlims is None else xlims
        ylim = axx.get_ylim()
        axx.set_xlim(xlim)
        axx.set_ylim(ylim)
        axx.hlines(0, xlim[0], xlim[1], colors='black', linestyles='solid', lw=1, alpha=0.8,zorder=-5)
        if act_critical is not None:
            axx.vlines(act_critical, ylim[0], ylim[1], color='k', linestyle='--', zorder=-5, lw=1)
    if savepath:
        fig.savefig(savepath, bbox_inches='tight', dpi=620, pad_inches=.05)
        print(f"Figure saved to: {savepath}")
    return fig, ax

def plot_cid_and_derivative(L_list, act_dict, cid_time_av_dict, \
                            dcid_dict, act_critical=None, xlims=None, plot_abs=False, shift_act=False, savepath=None):
    """
    Plot CID and its derivative vs activity for different system sizes.

    Parameters:
    ----------
    L_list : list
        List of system sizes (e.g., [16, 32, 64, 128]).
    act_dict : dict
        Dictionary mapping system size to activity list.
    cid_time_av_dict : dict
        Dictionary mapping system size to CID time-averaged data.
        Each value should be an array with shape (n, 2): [mean, std].
    dcid_dict : dict
        Dictionary mapping system size to derivative data.
        Each value should be an array with shape (n-1, 2): [mean, std].
    xlims : list, default [0, None]
        x-axis limits for the plots.
    savepath : str, default None
        Path where the figure is saved if not None

    Returns:
    -------
    fig, ax0 : matplotlib.figure.Figure, np.ndarray of Axes
        The created figure and axes array.
    """


    fig, ax0 = plt.subplots(ncols=2, figsize=(10, 4))
    marker_shape = ['d-', 'o-', '^-', 'v-', 'D-', 'P-', 'X-', 'h-', 
                    'd-', 'p-', 'H-', '8-', '1-', '2-']

    ax_cid, ax_dcid = ax0

    for i, LX in enumerate(L_list):
        act_list = act_dict[LX]

        cid_time_av = cid_time_av_dict[LX].copy()
        deriv_cid = dcid_dict[LX].copy()
        Nderiv = deriv_cid.shape[0]
        act_diff_tot = act_list[:-1] + np.diff(act_list) / 2 if shift_act else act_list[:Nderiv]
        if plot_abs:
            deriv_cid[:, 0] = np.abs(deriv_cid[:, 0])

        ax_cid.errorbar(act_list,cid_time_av[:, 0],yerr=cid_time_av[:, 1],
            fmt=marker_shape[i % len(marker_shape)],
            lw=1,label=f'L={LX}',
            elinewidth=1.5, alpha=0.6)
        ax_dcid.errorbar(act_diff_tot,deriv_cid[:, 0],yerr=deriv_cid[:, 1],
            fmt=marker_shape[i % len(marker_shape)],
            lw=1, label=f'L={LX}',
            elinewidth=1.5, alpha=0.6)

    for ax in ax0:
        ax.set(xlabel=r'Activity ($\tilde{\zeta}$)')
        xlim = ax.get_xlim() if xlims is None else xlims
        ylim = ax.get_ylim()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.hlines(0, xlim[0], xlim[1], colors='black', linestyles='solid', lw=1, alpha=0.8,zorder=-5)
        if act_critical is not None:
            ax.vlines(act_critical, ylim[0], ylim[1], color='k', linestyle='--', zorder=-5, lw=1)

    if plot_abs:
        dy_label = r'$\vert d\textrm{CID}/d \tilde{\zeta} \vert$'
    else:
        dy_label = r'$d\textrm{CID}/d\tilde{\zeta}$'
        
    ax_cid.set_ylabel(r'$\textrm{CID}$')
    ax_dcid.set_ylabel(dy_label)
    ax_dcid.legend()

    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, bbox_inches='tight', dpi=620, pad_inches=0.05)
        print(f"Figure saved to: {savepath}")
    return fig, ax0

def plot_div_and_derivative(L_list, act_dict, frac_time_av_dict, \
                        dfrac_dict, act_critical=None, xlims=None, plot_abs=False, shift_act=False, savepath=None):
    """
    Plot Divergence and its derivative vs activity for different system sizes
    
    Parameters:
    ----------
    L_list : list
        List of system sizes (e.g., [16, 32, 64, 128]).
    act_dict : dict
        Dictionary mapping system size to activity list.
    frac_time_av_dict : dict
        Dictionary mapping system size to CID/CID_shuffle time-averaged data.
        Each value should be an array with shape (n, 2): [mean, std].
    dfrac_dict : dict
        Dictionary mapping system size to derivative data.
        Each value should be an array with shape (n-1, 2): [mean, std].
    savepath : str, default None    
        Path where the figure is saved if not None

    Returns:
    --------
    fig, ax0 : matplotlib.figure.Figure, np.ndarray of Axes
        The created figure and axes array.
    """
    fig, ax0 = plt.subplots(ncols=2, figsize=(10, 4))
    marker_shape = ['d-', 'o-', '^-', 'v-', 'D-', 'P-', 'X-', 'h-', 
                    'd-', 'p-', 'H-', '8-', '1-', '2-']
    ax_div, ax_ddiv = ax0

    for i, LX in enumerate(L_list):
        act_list = act_dict[LX]

        div_time_av = frac_time_av_dict[LX].copy()
        deriv_div = dfrac_dict[LX].copy()
        Nderiv = deriv_div.shape[0]
        act_diff_tot = act_list[:-1] + np.diff(act_list) / 2 if shift_act else act_list[:Nderiv]

        # Convert CID/CID_shuffle to Divergence
        div_time_av[:, 0] = 1 - div_time_av[:, 0]
        deriv_div[:, 0] *= -1
        if plot_abs:
            deriv_div[:, 0] = np.abs(deriv_div[:, 0])

        ax_div.errorbar(act_list,div_time_av[:, 0],yerr=div_time_av[:, 1],
            fmt=marker_shape[i % len(marker_shape)],
            lw=1,label=f'L={LX}',
            elinewidth=1.5, alpha=0.6)
        ax_ddiv.errorbar(act_diff_tot,deriv_div[:, 0],yerr=deriv_div[:, 1],
            fmt=marker_shape[i % len(marker_shape)],
            lw=1, label=f'L={LX}',
            elinewidth=1.5, alpha=0.6)
        
    for ax in ax0:
        ax.set(xlabel=r'Activity ($\tilde{\zeta}$)')
        xlim = ax.get_xlim() if xlims is None else xlims
        ylim = ax.get_ylim()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.hlines(0, xlim[0], xlim[1], colors='black', linestyles='solid', lw=1, alpha=0.8,zorder=-5)
        if act_critical is not None:
            ax.vlines(act_critical, ylim[0], ylim[1], color='k', linestyle='--', zorder=-5, lw=1)
    if plot_abs:
        dy_label = r'$\vert d\mathcal{D}/d \tilde{\zeta} \vert$'
    else:
        dy_label = r'$d\mathcal{D}/d\tilde{\zeta}$'
    ax_div.set_ylabel(r'$\mathcal{D}$')
    ax_ddiv.set_ylabel(dy_label)
    ax_ddiv.legend()
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, bbox_inches='tight', dpi=620, pad_inches=0.05)
        print(f"Figure saved to: {savepath}")
    return fig, ax0

def plot_cid_fluctuations(L_list, act_dict, cid_time_av_dict, cid_var_dict, dcid_dict, \
                            act_critical=None, xlims=None, plot_abs=False, shift_act=False, savepath=None):
    """
    Plot CID variance and derivative vs activity for different system sizes.

    Parameters:
    ----------
    L_list : list
        List of system sizes (e.g., [16, 32, 64, 128]).
    act_dict : dict
        Dictionary mapping system size to activity list.
    cid_time_av_dict : dict
        Dictionary mapping system size to CID time-averaged data.
        Each value should be an array with shape (n, 2): [mean, std].
    cid_var_dict : dict
        Dictionary mapping system size to CID variance data.
        Each value should be an array with shape (n, 2): [mean, sem].
    dcid_dict : dict
        Dictionary mapping system size to derivative data.
        Each value should be an array with shape (n-1, 2): [mean, sem].
    xlims : list, default [0, None]
        x-axis limits for the plots.
    plot_abs : bool, default False
        Whether to plot the absolute value of the derivative.
    savepath : str, default None
        Path where the figure is saved if not None

    Returns:
    -------
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
        The created figure and axes.
    """

    ncols = len(L_list)
    w = 4 * ncols
    #w += 1 if ncols < 3 else 0
    fig, ax0 = plt.subplots(ncols=ncols, figsize=(w,4))
    marker_shape = ['d-', 'o-', '^-', 'v-', 'D-', 'P-', 'X-', 'h-', 
                    'd-', 'p-', 'H-', '8-', '1-', '2-']
    
    if plot_abs:
        dylabel_per = r'$\vert \frac{d\textrm{CID}}{d\tilde{\zeta}}\vert/\vert\textrm{CID} \vert$'
        dylabel = r'$\vert d\textrm{CID}/d\tilde{\zeta}\vert$'
    else:
        dylabel_per = r'$\frac{d \textrm{CID}}{d\tilde{\zeta}}/\textrm{CID}$'
        dylabel = r'$\frac{d \textrm{CID}}{d\tilde{\zeta}}$'

    for i, LX in enumerate(L_list):
        ax = ax0 if ncols == 1 else ax0[i]

        act_list = act_dict[LX]
        
        cid_tav = cid_time_av_dict[LX].copy()
        cid_var = cid_var_dict[LX].copy()
        deriv_cid = dcid_dict[LX].copy()
        Nderiv = deriv_cid.shape[0]

        act_diff_tot = act_list[:-1] + np.diff(act_list) / 2 if shift_act else act_list[:Nderiv]
        if plot_abs:
            deriv_cid = np.abs(deriv_cid)

        normalizer = 0.5 * (cid_tav[:Nderiv, 0] + cid_tav[-Nderiv:, 0])
        deriv_cid_per = deriv_cid / normalizer[:, None]

        # normalize by magnitude
        cid_ndims = cid_var.ndim
        cid_var /= np.nanmax(cid_var[:, 0]) if cid_ndims == 2 else np.nanmax(cid_var)
        deriv_cid /= np.nanmax(np.abs(deriv_cid))
        deriv_cid_per /= np.nanmax(np.abs(deriv_cid_per))

        if cid_ndims == 2:
            ax.errorbar(act_list,cid_var[:, 0],yerr=cid_var[:, 1],
                fmt=marker_shape[0],
                lw=1,label=r'$\mathrm{Var} (CID)$',
                elinewidth=1.5, alpha=0.6)
        else:
            ax.plot(act_list,cid_var,
                marker=marker_shape[0][0],
                lw=1,label=r'$\mathrm{Var} (CID)$',
                alpha=0.6)
        ax.errorbar(act_diff_tot,deriv_cid[:, 0],yerr=deriv_cid[:, 1],
            fmt=marker_shape[1],
            lw=1, label=dylabel,
            elinewidth=1.5, alpha=0.6)
        ax.errorbar(act_diff_tot,deriv_cid_per[:, 0],yerr=deriv_cid_per[:, 1],
            fmt=marker_shape[2],
            lw=1, label=dylabel_per,
            elinewidth=1.5, alpha=0.6)  
        ax.set_title(f'L={LX}')

    ax = [ax0] if ncols == 1 else ax0
    for axx in ax:
        axx.set(xlabel=r'Activity ($\tilde{\zeta}$)')
        xlim = axx.get_xlim() if xlims is None else xlims
        ylim = axx.get_ylim()
        axx.set_xlim(xlim)
        axx.set_ylim(axx.get_ylim())
        axx.hlines(0, xlim[0], xlim[1], colors='black', linestyles='solid', lw=1, alpha=0.8,zorder=-5)
        if act_critical is not None:
            axx.vlines(act_critical, ylim[0], ylim[1], color='k', linestyle='--', zorder=-5, lw=1)
        axx.legend()

    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, bbox_inches='tight', dpi=620, pad_inches=0.05)
        print(f"Figure saved to: {savepath}")
    return fig, ax

def plot_div_fluctuations(L_list, act_dict, frac_time_av_dict, dfrac_dict, div_var_dict, act_critical=None,
                        xlims=None, plot_div_per=False, plot_abs=False, shift_act=False, savepath=None):
    """
    Plot divergence variance and derivative vs activity for different system sizes. 

    Parameters:
    ----------
    L_list : list
        List of system sizes (e.g., [16, 32, 64, 128]).
    act_dict : dict
        Dictionary mapping system size to activity list.  
    frac_time_av_dict : dict
        Dictionary mapping system size to CID/CID_shuffle time-averaged data.      
    dfrac_dict : dict
        Dictionary mapping system size to derivative data of frac.
        Each value should be an array with shape (n-1, 2): [mean, std].
    div_var_dict : dict
        Dictionary mapping system size to divergence variance data (= varFrac).
        Each value should be an array with shape (n, 2): [mean, sem].
    xlims : list, default [0, None]
        x-axis limits for the plots.
    plot_abs : bool, default False  
        Whether to plot the absolute value of the derivative.
    savepath : str, default None
        Path where the figure is saved if not None  

    Returns:
    -------
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
        The created figure and axes.
    """
    ncols = len(L_list)
    w = 4 * ncols
    #w += 1 if ncols < 3 else 0
    fig, ax0 = plt.subplots(ncols=ncols, figsize=(w,4))
    marker_shape = ['d-', 'o-', '^-', 'v-', 'D-', 'P-', 'X-', 'h-', 
                    'd-', 'p-', 'H-', '8-', '1-', '2-']
    if plot_abs:
        dylabel_per = r'$\vert \frac{d\mathrm{Div}}{d\tilde{\zeta}}\vert/\vert\mathrm{Div} \vert$'
        dylabel = r'$\vert d\mathrm{Div}/d\tilde{\zeta}\vert$'
    else:
        dylabel_per = r'$\frac{d \mathrm{Div}}{d\tilde{\zeta}}/\mathrm{Div}$'
        dylabel = r'$\frac{d\mathrm{Div}}{d\tilde{\zeta}}$'

    for i, LX in enumerate(L_list):
        ax = ax0 if ncols == 1 else ax0[i]

        act_list = act_dict[LX]

        div_var = div_var_dict[LX].copy()
        deriv_div = dfrac_dict[LX].copy()
        div_tav = frac_time_av_dict[LX].copy()
        Nderiv = deriv_div.shape[0]

        act_diff_tot = act_list[:-1] + np.diff(act_list) / 2 if shift_act else act_list[:Nderiv]

        # Convert CID/CID_shuffle to Divergence
        div_tav[:, 0] = 1 - div_tav[:, 0]
        deriv_div[:, 0] *= -1
        if plot_abs:
            deriv_div = np.abs(deriv_div)

        normalizer = 0.5 * (div_tav[:Nderiv, 0] + div_tav[-Nderiv:, 0])
        deriv_div_per = deriv_div / normalizer[:, None]

        # normalize by magnitude
        div_ndims = div_var.ndim
        div_var /= np.nanmax(div_var[:, 0]) if div_ndims == 2 else np.nanmax(div_var)
        deriv_div /= np.nanmax(np.abs(deriv_div))
        deriv_div_per /= np.nanmax(np.abs(deriv_div_per))

        if div_ndims == 2:
            ax.errorbar(act_list,div_var[:, 0],yerr=div_var[:, 1],
                fmt=marker_shape[0],
                lw=1,label=r'$\mathrm{Var} (\mathrm{Div})$',
                elinewidth=1.5, alpha=0.6)
        else:
            ax.plot(act_list,div_var,
                marker=marker_shape[0][0],
                lw=1,label=r'$\mathrm{Var} (\mathrm{Div})$',
                alpha=0.6)

        ax.errorbar(act_diff_tot,deriv_div[:, 0],yerr=deriv_div[:, 1],
            fmt=marker_shape[1],
            lw=1, label=dylabel,
            elinewidth=1.5, alpha=0.6)
        if plot_div_per:
            ax.errorbar(act_diff_tot,deriv_div_per[:, 0],yerr=deriv_div_per[:, 1],
                fmt=marker_shape[2],
                lw=1, label=dylabel_per,
                elinewidth=1.5, alpha=0.6)  
        ax.set_title(f'L={LX}')

    ax = [ax0] if ncols == 1 else ax0
    for axx in ax:
        axx.set(xlabel=r'Activity ($\tilde{\zeta}$)')
        xlim = axx.get_xlim() if xlims is None else xlims
        ylim = axx.get_ylim()
        axx.set_xlim(xlim)
        axx.set_ylim(ylim)
        axx.hlines(0, xlim[0], xlim[1], colors='black', linestyles='solid', lw=1, alpha=0.8,zorder=-5)
        if act_critical is not None:
            axx.vlines(act_critical, ylim[0], ylim[1], color='k', linestyle='--', zorder=-5, lw=1)
        axx.legend()
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, bbox_inches='tight', dpi=620, pad_inches=0.05)
        print(f"Figure saved to: {savepath}")
    return fig, ax
