# Author: Simon Guldager & Patrizio Cugia di Sant'Orsola
# Date (latest update): 

### SETUP ------------------------------------------------------------------------------------

## Imports:

import os
import sys
import pickle
import glob
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree

from iminuit import Minuit

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import markers, rcParams
from cycler import cycler
import matplotlib.ticker

from utils import *

## Change directory to current one
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
sys.path.append(os.getcwd())

## Set plotting style and print options
sns.set_theme()
sns.set_style("darkgrid")
sns.set_context("paper") #Possible are paper, notebook, talk and poster


d = {'lines.linewidth': 2, 'axes.titlesize': 18, 'axes.labelsize': 18, 'xtick.labelsize': 12, 'axes.labelweight': 'bold', 'ytick.labelsize': 12,\
     'legend.fontsize': 15, 'font.family': 'serif', 'font.weight': 'bold', 'figure.titlesize': 20,'figure.titleweight': 'bold',\
          'figure.labelsize': 18,'figure.labelweight': 'bold', 'figure.figsize': (9,6), }
d_colors = {'axes.prop_cycle': cycler(color = ['teal', 'navy', 'coral', 'plum', 'purple', 'olivedrab',\
         'black', 'red', 'cyan', 'yellow', 'khaki','lightblue'])}
rcParams.update(d)
rcParams.update(d_colors)
np.set_printoptions(precision = 5, suppress=1e-10)

### FUNCTIONS --------------------------------------------------------------------------------

def calc_density_fluctuations(points_arr, window_sizes, boundaries=None, N_center_points=None, Ndof=1, dist_to_boundaries=None, normalize=False, verbose=False):
    """
    Calculates the density fluctuations for a set of points in a 2D plane for different window sizes.
    For each window_size (i.e. radius), the density fluctuations are calculated by choosing N_center_points random points
    inside a region determined by dist_to_boundaries and calculating the number of points within a circle of radius R for each
    of these points, from which the number and density variance can be calculated.

    Parameters:
    -----------
    points_arr : (numpy array) - Array of points in 2D plane
    window_sizes : (numpy array or list) - Array of window sizes (i.e. radii) for which to calculate density fluctuations
    boundaries : (list of lists) - List of tuples with the format [[x_min, x_max], [y_min, y_max]]. If None, no boundaries are used.
    N_center_points : (int) - Number of center points to use for each window size. If None, all points are used.
    Ndof : (int) - Number of degrees of freedom to use for variance calculation
    dist_to_boundaries : (float) - Maximum distance to the boundaries. Centers will be chosen within this region.
    normalize : (bool) - If True, the density fluctuations are normalized by the square of the average density of the system.

    Returns:
    --------
    var_counts : (numpy array) - Array containing the number variance for each window size
    var_densities : (numpy array) - Array containing the density variance for each window size
    """

    # If dist_to_boundaries is not given, use the maxium window size
    dist_to_boundaries = window_sizes[-1] if dist_to_boundaries is None else dist_to_boundaries

    if boundaries is None:
        xmin, xmax = np.min(points_arr[:, 0]), np.max(points_arr[:, 0])
        ymin, ymax = np.min(points_arr[:, 1]), np.max(points_arr[:, 1])
    else:
        xmin, xmax = boundaries[0]
        ymin, ymax = boundaries[1]

    center_mask_x = (points_arr[:, 0] - dist_to_boundaries >= xmin) & (points_arr[:, 0] + dist_to_boundaries <= xmax)
    center_mask_y = (points_arr[:, 1] - dist_to_boundaries >= ymin) & (points_arr[:, 1] + dist_to_boundaries <= ymax)
    center_mask = center_mask_x & center_mask_y


    # Construct mask for points within boundaries
    center_mask_x = (points_arr[:, 0] - dist_to_boundaries >= np.min(points_arr[:, 0])) & (points_arr[:, 0] + dist_to_boundaries <= np.max(points_arr[:, 0]))
    center_mask_y = (points_arr[:, 1] - dist_to_boundaries >= np.min(points_arr[:, 1])) & (points_arr[:, 1] + dist_to_boundaries <= np.max(points_arr[:, 1]))
    center_mask = center_mask_x & center_mask_y
    

    # If N is not given, use all points within boundaries
    Npoints = len(points_arr)
    Npoints_within_boundaries = center_mask.sum()
    N_center_points = Npoints_within_boundaries if N_center_points is None else N_center_points

    if verbose:
        print("Number of points within boundaries: ", Npoints_within_boundaries)
        print("Number of points to use: ", N_center_points)

    if N_center_points > Npoints_within_boundaries:
        if verbose:
            print(f"Warning: N_center_points is larger than the number of points within the boundaries.\
                Using all {Npoints_within_boundaries} points within boundaries instead.")
        N_center_points = Npoints_within_boundaries

    # If N_center_points is equal to Npoints_within_boundaries, use all points within boundaries
    use_all_center_points = (N_center_points == Npoints_within_boundaries)
    if use_all_center_points:
        center_points = points_arr[center_mask]


    # Initialize KDTree
    tree = KDTree(points_arr)

    # Initialize density array, density variance array, and counts variance arrays
    var_counts = np.empty_like(window_sizes, dtype=float)
    var_densities = np.empty_like(var_counts)
    av_counts = np.empty_like(var_counts)

    for i, radius in enumerate(window_sizes):
        if use_all_center_points:
            pass
        else:
            indices = np.random.choice(np.arange(Npoints)[center_mask], N_center_points, replace=False)
            center_points = points_arr[indices]

        N = center_points.shape[0]
        center_points += 0 #0.02 * (xmax-xmin) * np.random.rand(N, 2)

        # Calculate no. of points within circle for each point
        counts = tree.query_radius(center_points, r=radius, count_only=True)

        # Calculate average counts
        av_counts[i] = np.mean(counts)

        # Calculate number and density variance
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            var_counts[i] = np.var(counts, ddof=Ndof)
            densities = counts / (np.pi * radius**2)
            var_densities[i] = np.var(densities, ddof=Ndof)

    if normalize:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            av_densities = np.nanmean(av_counts / (np.pi * window_sizes**2))
            var_densities /= av_densities**2

    return var_counts, var_densities, av_counts

def fit_func(x, alpha, beta):
    return beta * (2 - alpha) + (2 - alpha) * x

def power_func(x, b): #, c, d, e):
    return  b * x ** 2

### TODO:

# Test all relevant Ns (100-750) for many exp across different Rmax (0.05-0.1). Plot chi2 vs. Rmax. Find optimal Rmax.
# Do combos of N and Nexp to get same statistics
### .... maybe the goal is actually to find the alpha and its uncertainty so that we can subtract it later or add it to uncertainty



def main():
    run_simulation = True

    # create mock data
    N_list = np.arange(100,3500,400) #np.arange(100,900,100) #15_000
    Ntot = 80_000 # 300_000 #100_000
    Nexp_list = (Ntot / N_list).astype('int')
    N_center_points_fraction = 1
    Nwindows = 30


    x_boundaries = [0, 1]
    y_boundaries = [0, 1] 
    boundaries = [x_boundaries, y_boundaries]   

    R_boundary = 0.1 * x_boundaries[-1]
    Rmin = 0.005 * x_boundaries[-1]
    Rmax_list = np.round(np.arange(0.1 * x_boundaries[-1], 0.2 * x_boundaries[-1], 0.1),3)
    normalize = False

    param_guess_lin = np.array([0.1, 3])
    param_guess_power = np.array([2400])

    fitted_params_arr = np.zeros((len(N_list), len(Rmax_list), 2))
    stats_arr = np.zeros((len(N_list), len(Rmax_list), 4))

    if run_simulation:
        for k, N in enumerate(N_list):
            print(f"\nBeginning analysis for N =  {N}")
            for j, Rmax in enumerate(Rmax_list):
                print(f"Beginning analysis for Rmax =  {Rmax}")
                t0 = time.time()
              #  R = np.logspace(np.log10(Rmin), np.log10(Rmax), Nwindows)  
                R = np.linspace(Rmin, Rmax, Nwindows)
    
                R_boundary = R[-1]
                N_center_points = int(N_center_points_fraction * (1 - 2 * R_boundary) ** 2 * N)

                # calculate density fluctuations N times for statistics
                counts_var = np.empty((Nexp_list[k], len(R)), dtype=float)

                for i in range(Nexp_list[k]):
                    field = np.random.uniform(x_boundaries[0], x_boundaries[1], (N, 2))
                    counts_var[i], _, _ = calc_density_fluctuations(field, R, boundaries = boundaries, \
                                        N_center_points = N_center_points, dist_to_boundaries=R_boundary, normalize=normalize)
                

                count_var_av = np.mean(counts_var, axis=0)
                count_var_std = np.std(counts_var, axis=0, ddof=1) / np.sqrt(Nexp_list[k])
                
                count_var_av_log = np.log(count_var_av)
                count_var_std_log = count_var_std / count_var_av

                print("Relative error: ", count_var_std / count_var_av)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=np.VisibleDeprecationWarning)
                    fit = do_chi2_fit(fit_func, np.log(R), count_var_av_log, count_var_std_log, param_guess_lin, verbose = True)
                    power_fit = do_chi2_fit(power_func, R, count_var_av, count_var_std, param_guess_power, verbose = False)

                Ndof, chi2, prop = get_statistics_from_fit(fit, len(R), subtract_1dof_for_binning = False)
                _, _, prop_power = get_statistics_from_fit(power_fit, len(R), subtract_1dof_for_binning = False)

                stats_arr[k, j] = Ndof, chi2, prop, prop_power
                fitted_params_arr[k, j] = fit.values['alpha'], fit.errors['alpha']

                fig, ax = plt.subplots()
                ax.errorbar(R, count_var_av, yerr=count_var_std, fmt = 'o', color = 'black', alpha = 0.5, elinewidth = 1, capsize = 2, capthick = 1, markersize = 4)
                ax.plot(R, np.exp(fit_func(np.log(R), *fit.values)), '-', color = 'red', label = rf'Fit: $y = 2 \beta + 2 x$')
                ax.plot(R, power_func(R, *power_fit.values), '-', color = 'blue', label = rf'Fit: $y = \beta x^2$')
                ax.legend()
               # ax.set_xscale('log')
               # ax.set_yscale('log')
                plt.show()
                
                t2 = time.time()
                print("Time elapsed: ", np.round(t2-t0,2))

            print(f"power pvals", stats_arr[k, :, 3])
            print(f"linear pvals", stats_arr[k, :, 2])
            print(f'alpha std', fitted_params_arr[k, :, 1].mean(), "\u00B1", fitted_params_arr[k, :, 1].std(ddof=1))

        # save data
   #     np.save(f"data/fitted_params_arr_nfrac{N_center_points_fraction}_ntot{Ntot}.npy", fitted_params_arr)
    #    np.save(f"data/stats_arr_nfrac{N_center_points_fraction}_ntot{Ntot}.npy", stats_arr)
    else:
        #load
        fitted_params_arr = np.load(f"data/fitted_params_arr_nfrac{N_center_points_fraction}_ntot{Ntot}.npy")
        stats_arr = np.load(f"data/stats_arr_nfrac{N_center_points_fraction}_ntot{Ntot}.npy")


    print("alpha errors: ", fitted_params_arr[:, :, 1].mean(axis=1), " +/-", fitted_params_arr[:, :, 1].std(axis=1, ddof=1))

    # plot results
    fig, ax = plt.subplots(ncols = 3, nrows = 3, figsize = (12, 9))
    #fig2, ax2 = plt.subplots(ncols = 3, nrows = int(np.ceil(len(N_list) / 3)), figsize = (12, 10))
    fig2, ax2 = plt.subplots(ncols = 3, nrows = 3, figsize = (12, 9))
    ax = ax.flatten()
    ax2 = ax2.flatten()

    for i, N in enumerate(N_list):
        # plot pval of power fit (as o) and linear fit as (s)
        # also mark the 0.05 and 0.01 lines

        if i == 0:
            kwargs1 = {'label': rf'Fit 1: $y = 2 \beta + 2 x$ '}
            kwargs2 = {'label': rf'Fit 2: $y = (2 - \alpha) \beta + (2 - \alpha) x$'}
            kwargs3 = {'label': rf'$p = 0.05$'}
            kwargs4 = {'label': rf'$p = 0.01$'}
        else:
            kwargs1 = {}
            kwargs2 = {}
            kwargs3 = {}
            kwargs4 = {}

        ax2[i].plot(Rmax_list, stats_arr[i, :, 3], 's-', alpha = 0.34, **kwargs1, markersize = 4)
        ax2[i].plot(Rmax_list, stats_arr[i, :, 2], 'o-', alpha = 0.34, **kwargs2, markersize = 4)
       # ax2[i].plot(Rmax_list, np.ones_like(Rmax_list) * 0.05, '--', color = 'black', lw = 1.5,)
       # ax2[i].plot(Rmax_list, np.ones_like(Rmax_list) * 0.01, '--', color = 'black',lw = 1.5,)

        ax2[i].axhline(y = 0.05, color='black', linestyle='--', alpha=0.5, lw = 1.5)
        ax2[i].axhline(y = 0.01, color='black', linestyle='--', alpha=0.5, lw = 1.5)
        ax2[i].set_ylim([0.001,3])
        
        ax2[i].set_yscale('log')
        ax2[i].set_yticks([0.001, 0.01, 0.05, 0.1, 1], labels = [0.001, 0.01, 0.05, 0.1, 1])
    

        ax2[i].text(0.01, 0.9, f'N = {N}, Nexp = {Nexp_list[i]}', transform=ax2[i].transAxes, fontsize=10, verticalalignment='bottom')
        if i % 3 != 0:
            ax2[i].yaxis.set_ticklabels([])
        

    ax2[-1].axis('off')

    fig2.legend(loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=4, fontsize=12) #, fancybox=True, shadow=True)
    fig2.suptitle(r"Chi2 p-value of each fit vs. $R_{max}$")
    fig2.supxlabel(r"$R_{max}$")
    fig2.supylabel(r"p-value")
  #  fig2.tight_layout()


    # plot parameters for different N against R max.
    # on the other y-axis, plot the p-value of the fit

    colors = ['teal', 'navy', 'coral', 'plum', 'purple', 'olivedrab',\
         'black', 'red', 'cyan', 'yellow', 'khaki','lightblue']
    ylim = [np.min(fitted_params_arr[:, :, 0]) - 0.2 * np.abs(np.min(fitted_params_arr[:, :, 0])), np.max(fitted_params_arr[:, :, 0]) * 1.2]
    yticks = np.round(np.linspace(ylim[0], ylim[1], 5),2)
    ylim = [-0.05, 0.15]
    yticks = [-0.05, 0, 0.05, 0.10, 0.15]
    xticks = [0.05, 0.075, 0.1, 0.125, 0.15]

    for i, N in enumerate(N_list):
        # find points where p-value is below 0.05
        cutoff = 0.01
        p_value_mask = stats_arr[i, :, 2] < cutoff
        p_value_mask_power = stats_arr[i, :, 3] < cutoff

        p_mask1 = p_value_mask & p_value_mask_power
        p_mask2 = p_value_mask & ~p_value_mask_power
        p_mask3 = ~p_value_mask & p_value_mask_power
        p_mask4 = ~p_value_mask & ~p_value_mask_power


        # plot the points where p-value is below 0.05 in red, and the others in black
        kwargs = {'alpha': 0.8, 'elinewidth': 1, 'capsize': 2, 'capthick': 1, 'markersize': 7} #, 'ecolor': 'black',

        if i == 0:
            kwargs1 = {'label': f"p < {cutoff} "}
            kwargs2 = {'label': f"p > {cutoff} "}
        else:
            kwargs1 = {}
            kwargs2 = {}

   
        # if p_value_mask_power, plot as 'o', else as 'x'. If p_value_mask, plot as red else as blue

        ax[i].axhline(y = 0, color='black', linestyle='--', alpha=0.5, lw = 1.5)
        ax[i].errorbar(Rmax_list[p_mask1], fitted_params_arr[i, p_mask1, 0], \
                        yerr=fitted_params_arr[i, p_mask1, 1], fmt = 'x', color='red', **kwargs1, **kwargs)
        ax[i].errorbar(Rmax_list[p_mask2], fitted_params_arr[i, p_mask2, 0], \
                        yerr=fitted_params_arr[i, p_mask2, 1], fmt = 'x', color = colors[0], **kwargs2, **kwargs)
        ax[i].errorbar(Rmax_list[p_mask3], fitted_params_arr[i, p_mask3, 0], \
                        yerr=fitted_params_arr[i, p_mask3, 1], fmt = '.', color = 'red', **kwargs2, **kwargs)
        ax[i].errorbar(Rmax_list[p_mask4], fitted_params_arr[i, p_mask4, 0], \
                        yerr=fitted_params_arr[i, p_mask4, 1], fmt = '.', color = colors[1], **kwargs1, **kwargs)

        ax[i].text(0.01, 0.9, f'N = {N}, Nexp = {Nexp_list[i]}', transform=ax[i].transAxes, fontsize=10, verticalalignment='bottom')
     #   ax[i].set_ylim(ylim)
        ax[i].yaxis.set_ticks(yticks)
        ax[i].xaxis.set_ticks(xticks)

        if i % 3 != 0:
            ax[i].yaxis.set_ticklabels([])
        

    ax[-1].axis('off')
    # only add legend once to avoid duplicates
  #  fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.96), ncol=2, fancybox=True, shadow=True)

    fig.suptitle(r"Fit parameter $\alpha$ vs. $R_{max}$")
    fig.supxlabel(r"$R_{max}$")
    fig.supylabel(r"$\alpha$")
    fig.tight_layout()

    # save figures in 420 dpi

    fig.savefig(f"figs/alpha_vs_Rmax_{N_center_points_fraction}_ntot{Ntot}.png", dpi=420, format='png')
    fig2.savefig(f"figs/pval_vs_Rmax_{N_center_points_fraction}_ntot{Ntot}.png", dpi=420, bbox_inches='tight', format='png')



    plt.show()




   


if __name__ == '__main__':
    main()