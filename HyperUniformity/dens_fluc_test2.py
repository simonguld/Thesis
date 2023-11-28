# Author: Simon Guldager & Patrizio Cugia di Sant'Orsola
# Date (latest update): 

### SETUP ------------------------------------------------------------------------------------

## Imports:

import os
import sys
import pickle
import glob
import time

import numpy as np
import seaborn as sns
import warnings
import pandas as pd
from sklearn.neighbors import KDTree

from iminuit import Minuit

import matplotlib.pyplot as plt
from matplotlib import rcParams
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


def calc_density_fluctuations(points_arr, window_sizes, N_center_points = None, Ndof = 1, x_boundaries = None, y_boundaries = None, normalize = False):
    """
    This function is a modification of a number of functions written by Patrizio Cugia di Sant'Orsola, who has kindly
    lend me his code.

    Calculates the density fluctuations for a set of points in a 2D plane for different window sizes. 
    For each window_size (i.e. radius), the density fluctuations are calculated by choosing N_center_points random points
    and calculating the number of points within a circle of radius R for each of these points, from which
    the number and density variance can be calculated.

    Parameters:
    -----------
    points_arr : (numpy array) - Array of points in 2D plane
    window_sizes : (numpy array or list) - Array of window sizes (i.e. radii) for which to calculate density fluctuations
    N_center_points : (int) - Number of center points to use for each window size. If None, all points are used.
    Ndof : (int) - Number of degrees of freedom to use for variance calculation
    x_boundaries : (list) - List of x boundaries with the format [x_min, x_max]. If None, no boundaries are used.
    y_boundaries : (list) - List of y boundaries with the format [y_min, y_max]. If None, no boundaries are used.
    normalize : (bool) - If True, the density fluctuations are normalized by the square of the average density of the system.

    Returns:
    --------
    var_counts : (numpy array) - Array containing the number variance for each window size
    var_densities : (numpy array) - Array containing the density variance for each window size
   
    """

    # If N is not given, use all points
    Npoints = len(points_arr)
    N_center_points = len(points_arr) if N_center_points is None else N_center_points

    # Initialize KDTree
    tree = KDTree(points_arr)

    # Initialize density array, density variance array and counts variance arrays
    var_counts = np.empty_like(window_sizes, dtype=float)
    var_densities = np.empty_like(var_counts)
    av_counts = np.empty_like(var_counts)

    for i, radius in enumerate(window_sizes):
        ## Choose N random points
        # If boundaries are given, only consider points whose distance to the boundary is larger than R
        if x_boundaries is not None or y_boundaries is not None:
            # Initialize index of allowed center points
            indices = np.arange(Npoints)
            if x_boundaries is not None:
                mask_x = (points_arr[:,0] - radius >= x_boundaries[0]) & (points_arr[:,0] + radius <= x_boundaries[1])
                indices = indices[mask_x]
            if y_boundaries is not None:
                mask_y = (points_arr[indices, 1] - radius >= y_boundaries[0]) & (points_arr[indices, 1] + radius <= y_boundaries[1])
                indices = indices[mask_y]
            # Choose N random points
            indices = np.random.choice(indices, min(N_center_points, len(indices)), replace=False)
        else:
            # Choose N random points
            indices = np.random.choice(range(Npoints), N_center_points, replace=False)

        if len(indices) == 0:
            var_counts[i] = np.nan
            var_densities[i] = np.nan
            av_counts[i] = np.nan
            continue

        # Initalize arrays of points to consider as center points
        center_points = points_arr[indices]

        # Calculate no. of points within circle for each point
        counts = tree.query_radius(center_points, r=radius, count_only=True)

        # Calculate average counts
        av_counts[i] = np.mean(counts)

        # Calculate number and density variance
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            var_counts[i] = np.var(counts, ddof = Ndof)
            var_densities[i] = var_counts[i] / (np.pi * radius**2)**2

    if normalize:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            av_densities = np.nanmean(av_counts / (np.pi * window_sizes**2))
            var_densities /= av_densities**2

    return var_counts, var_densities

def calc_density_fluctuations_modp(points_arr, window_sizes, N_center_points=None, Ndof=1, dist_to_boundaries=None, normalize=False):
    """
    Calculates the density fluctuations for a set of points in a 2D plane for different window sizes.
    For each window_size (i.e. radius), the density fluctuations are calculated by choosing N_center_points random points
    inside a region determined by dist_to_boundaries and calculating the number of points within a circle of radius R for each
    of these points, from which the number and density variance can be calculated.

    Parameters:
    -----------
    points_arr : (numpy array) - Array of points in 2D plane
    window_sizes : (numpy array or list) - Array of window sizes (i.e. radii) for which to calculate density fluctuations
    N_center_points : (int) - Number of center points to use for each window size. If None, all points are used.
    Ndof : (int) - Number of degrees of freedom to use for variance calculation
    dist_to_boundaries : (float) - Maximum distance to the boundaries. Centers will be chosen within this region.
    normalize : (bool) - If True, the density fluctuations are normalized by the square of the average density of the system.

    Returns:
    --------
    var_counts : (numpy array) - Array containing the number variance for each window size
    var_densities : (numpy array) - Array containing the density variance for each window size
    """

    # If N is not given, use all points
    Npoints = len(points_arr)
    N_center_points = Npoints if N_center_points is None else N_center_points

    # Initialize KDTree
    tree = KDTree(points_arr)

    # Initialize density array, density variance array, and counts variance arrays
    var_counts = np.empty_like(window_sizes, dtype=float)
    var_densities = np.empty_like(var_counts)
    av_counts = np.empty_like(var_counts)

    for i, radius in enumerate(window_sizes):
        indices = []
        min_x, max_x = np.min(points_arr[:, 0]) + dist_to_boundaries, np.max(points_arr[:, 0]) - dist_to_boundaries
        min_y, max_y = np.min(points_arr[:, 1]) + dist_to_boundaries, np.max(points_arr[:, 1]) - dist_to_boundaries

        while len(indices) < N_center_points:
            random_index = np.random.choice(Npoints)
            random_point = points_arr[random_index]
            x, y = random_point[0], random_point[1]
            if min_x <= x <= max_x and min_y <= y <= max_y:
                distance_to_boundaries = np.min([x - min_x, max_x - x, y - min_y, max_y - y])
                if distance_to_boundaries >= dist_to_boundaries:
                    indices.append(random_index)

        indices = np.array(indices)

        # Calculate no. of points within circle for each point
        counts = tree.query_radius(points_arr[indices], r=radius, count_only=True)

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

    return var_counts, var_densities

def calc_density_fluctuations_mod2(points_arr, window_sizes, N_center_points=None, Ndof=1, dist_to_boundaries=None, normalize=False):
    """
    Calculates the density fluctuations for a set of points in a 2D plane for different window sizes.
    For each window_size (i.e. radius), the density fluctuations are calculated by choosing N_center_points random points
    inside a region determined by dist_to_boundaries and calculating the number of points within a circle of radius R for each
    of these points, from which the number and density variance can be calculated.

    Parameters:
    -----------
    points_arr : (numpy array) - Array of points in 2D plane
    window_sizes : (numpy array or list) - Array of window sizes (i.e. radii) for which to calculate density fluctuations
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

    # Construct mask for points within boundaries
    center_mask_x = (points_arr[:, 0] - dist_to_boundaries >= np.min(points_arr[:, 0])) & (points_arr[:, 0] + dist_to_boundaries <= np.max(points_arr[:, 0]))
    center_mask_y = (points_arr[:, 1] - dist_to_boundaries >= np.min(points_arr[:, 1])) & (points_arr[:, 1] + dist_to_boundaries <= np.max(points_arr[:, 1]))
    center_mask = center_mask_x & center_mask_y

    # If N is not given, use all points within boundaries
    Npoints = len(points_arr)
    Npoints_within_boundaries = center_mask.sum()
    N_center_points = Npoints_within_boundaries if N_center_points is None else N_center_points

    print("Number of points within boundaries: ", Npoints_within_boundaries)
    print("Number of points to use: ", N_center_points)

    if N_center_points > Npoints_within_boundaries:
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

    return var_counts, var_densities

def calc_density_fluctuations_mod(points_arr, window_sizes, boundaries = None, N_center_points=None, Ndof=1, dist_to_boundaries=None, normalize=False):
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

    print("Number of points within boundaries: ", Npoints_within_boundaries)
    print("Number of points to use: ", N_center_points)

    if N_center_points > Npoints_within_boundaries:
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


### TODO:

# Test all relevant Ns (100-750) for many exp across different Rmax (0.05-0.1). Plot chi2 vs. Rmax. Find optimal Rmax.
# Do combos of N and Nexp to get same statistics
### .... maybe the goal is actually to find the alpha and its uncertainty so that we can subtract it later or add it to uncertainty

def main():
    # create mock data
    N = 800 #15_000
    Nexp = 50 # 5000 #1000 #00 #30 #5
    N_center_points = None
    normalize = False

    x_boundaries = [0, 1]
    y_boundaries = [0, 1] 

    R = np.linspace(0.1,2, 20)
    R = np.logspace(np.log10(x_boundaries[-1] / 100), np.log10(x_boundaries[-1] * 2 / 10), 20)

    boundaries = [x_boundaries, y_boundaries]   
    field = np.random.uniform(x_boundaries[0], x_boundaries[1], (N, 2))
    center_mask_x = (field[:, 0] - R[-1] >= x_boundaries[0]) & (field[:, 0] + R[-1] <= x_boundaries[1])
    center_mask_y = (field[:, 1] - R[-1] >= y_boundaries[0]) & (field[:, 1] + R[-1] <= y_boundaries[1])

    center_mask = center_mask_x & center_mask_y
    N_center_points = center_mask.sum()
    N_center_points = int(N_center_points)

    t0 = time.time()

    # calculate density fluctuations N times for statistics

    counts_var = np.empty((Nexp, len(R)), dtype=float)
    density_var = np.empty_like(counts_var)

    counts_var_mod = np.empty_like(counts_var)
    density_var_mod = np.empty_like(counts_var)

    for i in range(Nexp):
        print(f"\nExperiment {i+1}/{Nexp}")
        t0 = time.time()
        # initialize field
        field = np.random.uniform(x_boundaries[0], x_boundaries[1], (N, 2))
        counts_var[i], density_var[i] = calc_density_fluctuations_modp(field, R, N_center_points, dist_to_boundaries=R[-1], normalize=normalize)
       # counts_var[i], density_var[i], = calc_density_fluctuations(field, R,  \
        #                                N_center_points = N_center_points, x_boundaries = boundaries[0], y_boundaries = boundaries[1], normalize=False)
        t1 = time.time()
        print("Time elapsed: ", t1-t0)

        counts_var_mod[i], density_var_mod[i], _ = calc_density_fluctuations_mod(field, R, boundaries = boundaries, N_center_points = N_center_points, dist_to_boundaries=R[-1], normalize=normalize)
        t2 = time.time()
        print("Time elapsed: ", t2-t1)


    # save data
    np.save("data/counts_var_orig.npy", counts_var)
    np.save("data/counts_var_mod.npy", counts_var_mod)

    counts = [counts_var, counts_var_mod]

    #counts = [counts_var_mod]

    def fit_func(x, alpha, beta):
        return beta * (2 - alpha) + (2 - alpha) * x

    def power_func(x, a, b): #, c, d, e):
                return  a + b * x ** 2
    
    def fit_func(x, alpha, beta):
        return beta * (2 - alpha) + (2 - alpha) * x

    def power_func(x, b): #, c, d, e):
                return  b * x ** 2


    param_guess_lin = np.array([0.1, 3])
    param_guess_power = np.array([2400])
    fit_objects = []

    # plot results
    fig1, ax1 = plt.subplots(ncols = 2, figsize = (10,6))
    fig2, ax2 = plt.subplots(ncols = 2, figsize = (10,6))
    ax1 = ax1.flatten()
    ax2 = ax2.flatten()


    colors = ["blue", "red",]
    labels = ["Original", "Modified",]
    d = {}
    d_power = {}


    for i, count in enumerate(counts):
        count_var_av = np.mean(count, axis=0)
        count_var_std = np.std(count, axis=0, ddof=1) / np.sqrt(Nexp)
        
        count_var_av_log = np.log(count_var_av)
        count_var_std_log = count_var_std / count_var_av

        fit = do_chi2_fit(fit_func, np.log(R), count_var_av_log, count_var_std_log, param_guess_lin, verbose = True)
        fit_objects.append(fit)
        Ndof, chi2, prop = get_statistics_from_fit(fit, len(R), subtract_1dof_for_binning = False)

        d0 = {'Ndefects': N, 'Nexp': Nexp}
        d0.update(generate_dictionary(fit, len(R), chi2_fit = True, chi2_suffix = None, subtract_1dof_for_binning = False))
  
        text = nice_string_output(d0, extra_spacing=4, decimals=3)
        add_text_to_ax(0.02, 0.96, text, ax1[i], fontsize=12)

        print(f"Chi2: {chi2}, Ndof: {Ndof}, prop: {prop}")

        # plot log results
        ax1[i].errorbar(R, count_var_av_log, yerr=count_var_std_log, fmt='.', label=labels[i], color=colors[i], alpha=.5, elinewidth=1, capsize=2, capthick=1,)
        ax1[i].plot(R, fit_func(np.log(R), *fit.values[:]), \
                 label=rf"Fit {i+1}: $y = (2 - \alpha) \beta + (2 - \alpha) x$", color=colors[i], linestyle="--", alpha=.5)
        ax1[i].set_xscale('log')
        ax1[i].set_xticks([0.01, 0.025, 0.05, 0.1])
        ax1[i].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax1[i].legend(fontsize = 12, loc = 'lower right')

        mval = np.log(np.nanmean(counts[-1], axis = 0))
        print(mval)
        ax1[i].set_ylim(np.min(mval) - 0.6 * np.abs(np.min(mval)), np.max(mval) * 1.3)
        

   
        power_fit = do_chi2_fit(power_func, R, count_var_av, count_var_std, param_guess_power, verbose = True)

        d0 = {'Ndefects': N, 'Nexp': Nexp}
        d0.update(generate_dictionary(fit, len(R), chi2_fit = True, chi2_suffix = None, subtract_1dof_for_binning = False))
        text = nice_string_output(d0, extra_spacing=2, decimals=3)
        add_text_to_ax(0.02, 0.96, text, ax2[i], fontsize=12)
        d_power.update(d0)
        # plot results
        ax2[i].errorbar(R, count_var_av, yerr=count_var_std, fmt='.', label=labels[i], color=colors[i], alpha=.5, elinewidth=1, capsize=2, capthick=1,)
        fit_vals = np.exp(fit_func(np.log(R), *fit.values[:]))
       # ax2.plot(R, fit_vals, label=rf"Fit {i+1}: $y = e^\beta R^2 /R^\alpha $", color=colors[i], linestyle="-.", alpha=.5)

        ax2[i].plot(R, power_func(R, *power_fit.values[:]), label=f"Fit {i+1}: $y = b R^2$", color=colors[i], linestyle="--", alpha=.5)
        ax2[i].legend(fontsize = 12, loc = 'lower right')
        mval = np.nanmean(counts[-1], axis = 0)
        ax2[i].set_ylim(- 0.2 * np.max(mval), np.max(mval) * 1.3)

        # only include xticks on left plot
        if i == 1:
            ax1[i].set_yticklabels([])
            ax2[i].set_yticklabels([])


    fig2.supxlabel("Window size (1/Sys. size)",)
    fig2.supylabel(rf"$ \sigma_N^2$",)
    fig2.suptitle("Count variance vs. window size",)

    fig1.supxlabel("Window size (1/Sys. size) [logscale]",)
    fig1.supylabel(rf"ln $ \sigma_N^2$", )
    fig1.suptitle("Log of count variance vs. window size",)


    if 0:
        ax2.set_xlabel("Window size (1/Sys. size)")
        ax2.set_ylabel("Counts variance")
        ax2.legend(fontsize = 12, loc = 'lower right')
        ax2.set_title("Counts variance vs. window size")

    
        ax1.set_xlabel(rf"$ln R$")
        ax1.set_ylabel(rf"$ln \sigma_N^2$")
        ax1.legend(loc='lower right')
        ax1.set_title("Logplot of counts variance vs. window size")
        text = nice_string_output(d, extra_spacing=4, decimals=3)
        add_text_to_ax(0.02, 0.96, text, ax1, fontsize=12)

        text = nice_string_output(d_power, extra_spacing=4, decimals=3)
        add_text_to_ax(0.02, 0.96, text, ax2, fontsize=12)

    plt.show()



    if 0:

        # plot results
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()


        colors = ["blue", "red",]
        labels = ["Original", "Modified",]




        vars = [density_var_av, density_var_mod_av]
        vars_std = [density_var_std, density_var_mod_std]
        counts = [counts_var, counts_var_mod]
        counts_std = [count_var_std, count_var_mod_std]



        # plot results
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()


        colors = ["blue", "red",]
        labels = ["Original", "Modified",]

        def power_func(x, a, b): #, c, d, e):
                return  a + b * x ** 2
        def power_func_sum(a, b, Y, X): #, c, d, e):
                return np.sum(np.abs(Y - power_func(X, a, b)))


        for i in range(len(vars)):
            param_guess = np.array([counts[i][0] / R[0] ** 2])
            power_fit = Minuit(lambda b: power_func_sum(0, b, counts[i], R), *param_guess)
            power_fit.errordef = Minuit.LEAST_SQUARES
            print(power_fit.migrad())
            ax1.plot(R, vars[i], '.-', label=labels[i], color=colors[i], alpha=.5)
            ax2.plot(R, counts[i],'.-', label=labels[i], color=colors[i], alpha=.5)
            #ax2.plot(R, counts[i][0] * R**2 / R[0] ** 2, label=f"{labels[i]} R^2", color=colors[i], linestyle="--", alpha=.5)
            ax2.plot(R, power_func(R, 0, *power_fit.values[:]), label=f"Fit: y = {np.round(power_fit.values[0],2)} R^2", color=colors[i], linestyle="--", alpha=.5)



        ax1.set_xlabel("Radius")
        ax1.set_ylabel("Density variance")
        ax1.legend()
        ax2.set_xlabel("Radius")
        ax2.set_ylabel("Counts variance")
        ax2.legend()




        plt.show()



if __name__ == '__main__':
    main()