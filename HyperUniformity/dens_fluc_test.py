# Author: Simon Guldager & Patrizio Cugia di Sant'Orsola
# Date (latest update): 

### SETUP ------------------------------------------------------------------------------------

## Imports:

from calendar import c
import os
import sys
import pickle
import glob
import time

import numpy as np
import seaborn as sns
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from matplotlib import rcParams
from sklearn.neighbors import KDTree
from cycler import cycler
from sympy import use
from iminuit import Minuit

sys.path.append('Appstat2022\\External_Functions')
from ExternalFunctions import Chi2Regression, BinnedLH, UnbinnedLH
from ExternalFunctions import nice_string_output, add_text_to_ax    # Useful functions to print fit results on figure


## Change directory to current one
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
sys.path.append(os.getcwd())

## Set plotting style and print options
sns.set_theme()
sns.set_style("darkgrid")
sns.set_context("paper") #Possible are paper, notebook, talk and poster


d = {'lines.linewidth': 2, 'axes.titlesize': 18, 'axes.labelsize': 18, 'xtick.labelsize': 12, 'ytick.labelsize': 12,\
     'legend.fontsize': 15, 'font.family': 'serif', 'figure.figsize': (9,6)}
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

def calc_density_fluctuations_mod(points_arr, window_sizes, N_center_points=None, Ndof=1, dist_to_boundaries=None, normalize=False):
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


def main():
    # create mock data
    N = 150_000
    N_center_points = None
    R = np.linspace(0.1,2, 20)

    x_boundaries = [0, 10]
    y_boundaries = [0, 10] 
    field = np.random.uniform(x_boundaries[0], x_boundaries[1], (N, 2))
    center_mask_x = (field[:, 0] - R[-1] >= x_boundaries[0]) & (field[:, 0] + R[-1] <= x_boundaries[1])
    center_mask_y = (field[:, 1] - R[-1] >= y_boundaries[0]) & (field[:, 1] + R[-1] <= y_boundaries[1])

    center_mask = center_mask_x & center_mask_y
    N_center_points = center_mask.sum()
    N_center_points = int(N/3)

    t0 = time.time()

    counts_var, density_var = calc_density_fluctuations(field, R, N_center_points, x_boundaries = x_boundaries, y_boundaries = y_boundaries, normalize=True)

   # counts_var_mod, density_var_mod = calc_density_fluctuations_mod(field, R, N_center_points = N_center_points, dist_to_boundaries=R[-1], normalize=True)
    counts_var_mod2, density_var_mod2 = calc_density_fluctuations_mod2(field, R, N_center_points = N_center_points, dist_to_boundaries=R[-1], normalize=True)
    t1 = time.time()
    print("Time elapsed: ", t1-t0)

    # plot results
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    vars = [density_var,  density_var_mod2]
    counts = [counts_var,  counts_var_mod2]
    colors = ["blue", "red", "green"]
    labels = ["Original", "Modified", "Modified2"]

    def power_func(x, a, b): #, c, d, e):
            return  a + b * x ** 2
    def power_func_sum(a, b, Y, X): #, c, d, e):
            return np.sum(np.abs(Y - power_func(X, a, b)))


    if 0:
        power_fit = Minuit(power_func_sum, *param_guess)
        power_fit.errordef = 1.0
        print(power_fit.migrad())
        x_vals = np.linspace(GDP_ranked[0], GDP_ranked[-1], 500)
        y_vals = power_func(x_vals, *power_fit.values[:])
        ax0.plot(x_vals, y_vals, label = r"Fit (no uncertainty)")
        
        ## calc residuals and typical uncertainty
        residuals = happiness_ranked - power_func(GDP_ranked, *power_fit.values[:])
        std = residuals.std(ddof = 1) 
        print("typical happiness-index uncertainty: ", std)
        ax0.errorbar(GDP_ranked, happiness_ranked, std, fmt = 'k.', elinewidth=.7, capsize=.7, capthick=.7)

        err_fit = do_chi2_fit(power_func, GDP_ranked, happiness_ranked, std, power_fit.values)
        ax0.plot(x_vals, power_func(x_vals, *err_fit.values), label = 'Fit (with uncertainty)')
        d = generate_dictionary(err_fit, Ndatapoints = len(GDP_ranked))
        # Plot figure text
        text = nice_string_output(d, extra_spacing=0, decimals=5)
        add_text_to_ax(0.27, 0.53, text, ax0, fontsize=13)


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