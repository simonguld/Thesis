# Author: Simon Guldager Andersen
# Date (latest update): 

### SETUP ------------------------------------------------------------------------------------

## Imports:

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from itertools import product


plt.style.use('sg_article')

from utils import *


### FUNCTIONS ----------------------------------------------------------------------------------

def frame_plotter(frame, plot_director = True,  ms = 4, engine = plt):
    """Plot a frame. If plot_director is False, the velocity field is plotted instead.
    
    Example usage:
    
    plotter = lambda frame, engine: frame_plotter(frame, plot_director = False, engine = engine)
    animate(ar, plotter, rng=[0, Nframes], inter=1000, show=False, save=True);
    
    """
    
    # set plotting params 
    plt.tick_params(
    axis='both',          # changes apply to both axes
    which='both',      # both major and minor ticks are affected
    bottom=False,  
    left=False,
    right=False,       
    top=False)#,      
    #labelbottom=False) 
   

    ms = 4
    alpha = 1
    avg = 1
    ms_dir, lw_dir = 1.2, 1.2

    if plot_director:
        mp.nematic.plot.director(frame, ms=ms_dir, lw=lw_dir, avg=avg, alpha=.9, engine=engine)
        mp.nematic.plot.defects(frame, ms = ms, alpha=alpha, engine = engine) 
    else:
        mp.nematic.plot.velocity(frame, engine)
        mp.nematic.plot.defects(frame, ms = ms, engine = engine) 
    return

def animate(oa, fn, rng=[], inter=500, boundaries=None, show=True, save = False, save_path = None):
    """Show a frame-by-frame animation.

    Parameters:
    oa -- the output archive
    fn -- the plot function (argument: frame, plot engine)
    rng -- range of the frames to be ploted
    interval -- time between frames (ms)
    """
    # set range
    if len(rng)==0:
        rng = [ 1, oa.numframes+1 ]
    # create the figure
    fig = plt.figure(figsize = (3.,3.))

    # the local animation function
    def animate_fn(i):
        # we want a fresh figure everytime
        fig.clf()
        # add subplot, aka axis
        #ax = fig.add_subplot(111)
        # load the frame
        f = get_frame_number(i, oa._path, oa.ninfo)

       # plt.title(f'Frame {f}')
        fig.text(0.2, 1.02, '-1/2', fontsize=14, verticalalignment='bottom', color='blue', fontweight='bold')
        fig.text(0.8, 1.02, '+1/2',fontsize=14, verticalalignment='bottom', color='green', fontweight='bold');
        
        if boundaries:
            plt.ylim(boundaries[0],boundaries[1])
            plt.xlim(boundaries[0],boundaries[1])
    
        xticks=range(boundaries[0] + 20, boundaries[1], 20)
        plt.xticks(xticks)
        plt.yticks(xticks)
        plt.axis('off')
    
        frame = oa._read_frame(f)
        # call the global function
        fn(frame, plt)

    anim = FuncAnimation(fig, animate_fn,
                             frames=np.arange(rng[0], rng[1]),
                             interval=inter, blit=False)
    if save:
        LX = oa.LX
        act = oa.zeta
        save_to = save_path if save_path else f'anim_L{LX}_zeta{act}.mp4'
        fps = 1000 / inter
        anim.save(save_to, dpi=420, fps=fps)

    if show==True:
      plt.show()
      return

    return anim

def plot_frames(ar, archive_path, frame_idx_bounds = [], save = False, save_path = None):
    """

    Parameters:
    -----------
    ar: massPy.archive object
        massPy archive object containing the data
    archive_path: str
        path to the archive
    frame_idx_bounds: list 
        list of two integers specifying the bounds of the frames to plot (last frame is not included). If empty, all frames are plotted
    save: bool
        whether to save the figure or not
    """


    if len(frame_idx_bounds) == 0:
        Nframes = len([i for i in os.listdir(archive_path) if i.startswith('frame')])
    else:
        Nframes = frame_idx_bounds[1] - frame_idx_bounds[0]

    nrows = int(np.ceil(Nframes/3))
    fig_height = 6 * nrows
    fig, ax = plt.subplots(nrows=2,ncols=3,figsize=(18, fig_height))
    ax = ax.flatten()
    for i in range(Nframes):
        

        f = get_frame_number(i, archive_path, ar.ninfo)
        frame = ar._read_frame(f)
        

        # Get actual frame number
        frame_num = ar.nstart + f * ar.ninfo
        ax[i].set(title = f'Frame = {frame_num:.1e}')

        LX = ar.LX
        LY = ar.LY 
        Qxx_dat = frame.QQxx.reshape(LX, LY)
        Qyx_dat = frame.QQyx.reshape(LX, LY)
        
        defects = mp.nematic.nematicPy.get_defects(Qxx_dat, Qyx_dat, LX, LY)

        ms = 4
        alpha = 1

        ax[i].xaxis.set_ticks_position('none')
        ax[i].yaxis.set_ticks_position('none')

        mp.nematic.plot.defects(frame, ax[i], ms = ms, alpha=alpha)
        mp.nematic.plot.director(frame, ax[i], ms = .5, lw=.7)


    fig.text(0.35, 0.99, '-1/2', fontsize=14, verticalalignment='bottom', color='blue', fontweight='bold')
    fig.text(0.65, 0.99, '+1/2',fontsize=14, verticalalignment='bottom', color='green', fontweight='bold');

    if save:
        save_to = save_path if save_path is not None else 'frames.png'
        fig.savefig(save_to, dpi = 420, bbox_inches='tight')

    return fig, ax

def plot_structure_factor(kbins, smeans, sstds, k = None, plot_scaling=False, LX = None, sf_estimated = None):
    """
    Plot structure factor
    """

    k_begin_lines_idx = 10
    kmin, kmax = np.nanmin(kbins), np.nanmax(kbins)
    #sf_min, sf_max = np.nanmin(smeans), np.nanmax(smeans)
    x = np.linspace(kmin, kbins[k_begin_lines_idx], 10)

    fig, ax = plt.subplots()
    
    if k and sf_estimated:
        ax.scatter(np.linalg.norm(k, axis = 1), sf_estimated, label='Structure factor', s=2.5, alpha=0.3)

    ax.hlines(1, x[0], kmax, label=r'Possion', linestyles='dashed', colors='k')
    ax.errorbar(kbins, smeans, yerr = sstds, fmt = 's-', label = 'Binned means', alpha = .8, color = 'red', ecolor = 'black', markersize = 5)

    if plot_scaling:
        ax.plot(x, smeans[k_begin_lines_idx] * x**0.1 / x[-1]**0.1, label=r'$k^{0.1}$')
        ax.plot(x, smeans[k_begin_lines_idx] * x**0.2 / x[-1]**0.2, label=r'$k^{0.2}$')
        ax.plot(x, smeans[k_begin_lines_idx] * x**0.3 /x[-1]**0.3, label=r'$k^{0.3}$')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(ncol=3, fontsize = 14)
    ax.set_xlabel(r'Norm of wavenumber ($k$)')
    ax.set_ylabel(r'Time av. structure factor')

    if LX == 2048:
            ax.set_xticks([kmin, 0.01, 0.1, kmax], [np.round(kmin,3), 0.01, 0.1, np.round(kmax,1)])
            ax.set_yticks([0.3, 0.4, 0.6, 1, 5], [0.3, 0.4, 0.6, 1, 5])
    else:
        ax.set_xticks([kmin, 0.1, kmax], [np.round(kmin,3), 0.1, np.round(kmax,1)])
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

def vorticity_abs(frame, engine=plt, vmin=None, vmax=None, abs=False):
    """ Plot the vorticity """
    # vorticity computed with 5-point stencil
    w = mp.base_modules.flow.vorticity(frame)
    w = np.abs(w) if abs else w
    wmax = np.max(np.abs(w))
    # plot using engine
    vmin = None if vmin is None else vmin * wmax
    vmax = None if vmax is None else vmax * wmax

    im = engine.imshow(w.T, cmap='turbo', interpolation='lanczos', origin='lower', vmin= vmin, vmax= vmax)
    # add colorbar
    divider = make_axes_locatable(engine)
    cax = divider.append_axes("right", size="5%", pad="2%")
    plt.colorbar(im, cax)

def director(frame, engine=plt, scale=False, avg=4, ms = 1, alpha=1, lw=.5):
    """ Plot the director field """
    # get nematic field tensor
    Qxx = frame.QQxx.reshape(frame.LX, frame.LY)
    Qyx = frame.QQyx.reshape(frame.LX, frame.LY)
    # get order S and director (nx, ny)
    S, nx, ny = mp.nematic.nematicPy.get_director(Qxx, Qyx)
    # plot using engine
    x, y = [], []
    for i, j in product(np.arange(frame.LX, step=avg), np.arange(frame.LY, step=avg)):
        f = avg*(S[i,j] if scale else 1.)
        x.append(i - f*nx[i,j]/2.)
        x.append(i + f*nx[i,j]/2.)
        x.append(None)
        y.append(j - f*ny[i,j]/2.)
        y.append(j + f*ny[i,j]/2.)
        y.append(None)
    engine.plot(x, y, color='k', linestyle='-', markersize=1, linewidth=lw, alpha=alpha)

def order(frame, engine=plt, vmin=None):
    """ Plot the order """
    # get order S
    S, _, _ = mp.nematic.nematicPy.get_director(frame.QQxx, frame.QQyx)
    S = S.reshape(frame.LX, frame.LY)
    # plot using engine
    Smax = np.max(np.abs(S))
    # plot using engine
    vmin = np.min(Smax) if vmin is None else vmin
    im = engine.imshow(S.T, cmap='turbo', interpolation='lanczos', origin='lower', alpha=.85, clim=(vmin, min(1,Smax)))
    # add colorbar
    divider = make_axes_locatable(engine)
    cax = divider.append_axes("right", size="5%", pad="2%")
    cbar = plt.colorbar(im, cax)