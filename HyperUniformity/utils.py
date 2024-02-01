# Author: Simon Guldager Andersen
# Date(last edit): 05-01-2024

## Imports:
import os
import sys
import warnings
import time
import shutil

import numpy as np
import pandas as pd
from iminuit import Minuit
from scipy import stats
from sklearn.neighbors import KDTree

import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller

import massPy as mp

sys.path.append('C:\\Users\\Simon Andersen\\Projects\\Projects\\Appstat2022\\External_Functions')
from ExternalFunctions import Chi2Regression, BinnedLH, UnbinnedLH
from ExternalFunctions import nice_string_output, add_text_to_ax    # Useful functions to print fit results on figure


# Helper functions -------------------------------------------------------------------

def move_files(old_path, new_path = None):
    if new_path is None:
        new_path = old_path.replace('_sfac', '')
    act_dirs = os.listdir(old_path)

    for i, dir in enumerate(act_dirs):
        act_dir_new = os.path.join(new_path, dir)
        act_dir_old = os.path.join(old_path, dir)

        exp_dirs = os.listdir(act_dir_old)

        for j, exp_dir in enumerate(exp_dirs):
            act_exp_dir_new = os.path.join(act_dir_new, exp_dir)
            act_exp_dir_old = os.path.join(act_dir_old, exp_dir)

            for file in os.listdir(act_exp_dir_old):
                src_path = os.path.join(act_exp_dir_old, file)
                dest_path = os.path.join(act_exp_dir_new, file)

                if os.path.isfile(src_path):
                    shutil.copy2(src_path, dest_path)
    return

# Functions for nematic analysis  -----------------------------------------------------

def get_dir(Qxx, Qyx, return_S=False):
    """
    This function has been provided by Lasse Frederik Bonn:

    get director nx, ny from Order parameter Qxx, Qyx
    """
    S = np.sqrt(Qxx**2+Qyx**2)
    dx = np.abs(np.sqrt((np.ones_like(S) + Qxx/S)/2))
    dy = np.sqrt((np.ones_like(S)-Qxx/S)/2)*np.sign(Qyx)
    
    if return_S:
        return dx, dy, S
    else:
        return dx, dy

def get_defect_list(archive, LX, LY, idx_first_frame=0, Nframes = None, verbose=False):
    """
    Get list of topological defects for each frame in archive
    Parameters:
        archive: massPy archive object
        LX, LY: system size
        verbose: print time to get defect list
    Returns:
        top_defects: list of lists of dictionaries holding defect charge and position for each frame 
    """
    # Initialize list of top. defects
    top_defects = []

    Nframes = archive.__dict__['num_frames'] if Nframes is None else Nframes
    if verbose:
        t_start = time.time()

    # Loop over frames
    for i in range(idx_first_frame, Nframes - idx_first_frame):
        # Load frame
        frame = archive._read_frame(i)
        Qxx_dat = frame.QQxx.reshape(LX, LY)
        Qyx_dat = frame.QQyx.reshape(LX, LY)
        # Get defects
        defects = mp.nematic.nematicPy.get_defects(Qxx_dat, Qyx_dat, LX, LY)
        # Add to list
        top_defects.append(defects)

    if verbose:
        t_end = time.time() - t_start
        print('Time to get defect list: %.2f s' % t_end)

    return top_defects

def get_defect_arr_from_frame(defect_dict):
    """
    Convert dictionary of defects to array of defect positions
    """
    Ndefects = len(defect_dict)
    if Ndefects == 0:
        return None
    defect_positions = np.empty([Ndefects, 2])
    for i, defect in enumerate(defect_dict):
        defect_positions[i] = defect['pos']
    return defect_positions

def get_defect_density(defect_list, area, return_charges=False, save = False, save_path = None,):
        """
        Get defect density for each frame in archive
        parameters:
            defect_list: list of lists of dictionaries holding defect charge and position for each frame 
            area: Area of system
            return_charges: if True, return list of densities of positive and negative defects
        returns:
            dens_defects: list of defect densities
        """

        if return_charges:
            # Initialize list of defect densities
            dens_pos_defects = []
            dens_neg_defects = []
            for defects in defect_list:
                # Get no. of defects
                nposdef = len([d for d in defects if d['charge'] == 0.5])
                nnegdef = len([d for d in defects if d['charge'] == -0.5])

                dens_pos_defects.append(nposdef / area)
                dens_neg_defects.append(nnegdef / area)
            return dens_pos_defects, dens_neg_defects
        else:
            dens_defects = []
            for defects in defect_list:
                # Get no. of defects
                ndef = len(defects)
                dens_defects.append(ndef / area)
            if save:
                np.savetxt(save_path, dens_defects)
            return dens_defects

def calc_density_fluctuations(points_arr, window_sizes, boundaries = None, N_center_points=None, Ndof=1, dist_to_boundaries=None, normalize=False):
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

    #logging.info(f"Number of points within boundaries: {Npoints_within_boundaries}")

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
    av_counts = np.zeros_like(var_counts)

    if N_center_points == 0:
        print(f"No points within boundaries. Returning NaNs")
        return np.nan * var_counts, np.nan * var_densities, av_counts

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

def get_density_fluctuations(top_defect_list, window_sizes, boundaries = None, N_center_points = None, Ndof = 1, \
                             dist_to_boundaries = None, normalize = False, save = False, save_path_av_counts = None, save_path_var_counts = None):
    """
    Calculate defect density fluctuations for different window sizes
    Parameters:
    -----------
    top_defect_list: list of dictionaries, each dictionary contains defect positions and charges for one frame
    window_sizes: array of window sizes (i.e. radii) for which to calculate density fluctuations
    N_center_points: number of center points to use for each window size. If None, all points are used.
    Ndof: number of degrees of freedom to use for variance calculation
    dist_to_boundaries: maximum distance to the boundaries. Centers will be chosen within this region.
    normalize: if True, the density fluctuations are normalized by the square of the average density of the system.
    save: if True, save density fluctuations to file
    save_path_av_counts: path to file to save average counts
    save_path_var_counts: path to file to save count fluctuations
    Returns:
    --------
    
    defect_densities: array of defect densities for different window sizes

    """
    Nframes = len(top_defect_list)
    Nwindows = len(window_sizes)

    # Intialize array of count fluctuations and average counts
    count_fluctuation_arr = np.zeros([Nframes, len(window_sizes)])
    av_count_arr = np.zeros_like(count_fluctuation_arr)

    for frame, defects in enumerate(top_defect_list):
        # Step 1: Convert list of dictionaries to array of defect positions
        Ndefects = len(defects)
        if Ndefects == 0:
            count_fluctuation_arr[frame] = np.nan
            av_count_arr[frame] = 0
            continue

        defect_positions = np.empty([Ndefects, 2])
        for i, defect in enumerate(defects):
            defect_positions[i] = defect['pos']
        #logging.info(f"Frame {frame} has {Ndefects} defects")

        # Calculate density fluctuations
        count_fluctuation_arr[frame], _, av_count_arr[frame] = calc_density_fluctuations(defect_positions, window_sizes,\
                                         boundaries = boundaries,N_center_points=N_center_points, Ndof=Ndof, \
                                            dist_to_boundaries=dist_to_boundaries, normalize=normalize)
    if save:
        np.savetxt(save_path_var_counts, count_fluctuation_arr)
        np.savetxt(save_path_av_counts, av_count_arr)

    return count_fluctuation_arr, av_count_arr


### Functions for statistical analysis ------------------------------------------------

def get_statistics_from_fit(fitting_object, Ndatapoints, subtract_1dof_for_binning = False):
    
    Nparameters = len(fitting_object.values[:])
    if subtract_1dof_for_binning:
        Ndof = Ndatapoints - Nparameters - 1
    else:
        Ndof = Ndatapoints - Nparameters
    chi2 = fitting_object.fval
    prop = stats.chi2.sf(chi2, Ndof)
    return Ndof, chi2, prop

def do_chi2_fit(fit_function, x, y, dy, parameter_guesses, verbose = True):

    chi2_object = Chi2Regression(fit_function, x, y, dy)
    fit = Minuit(chi2_object, *parameter_guesses)
    fit.errordef = Minuit.LEAST_SQUARES

    if verbose:
        print(fit.migrad())
    else:
        fit.migrad()
    return fit

def generate_dictionary(fitting_object, Ndatapoints, chi2_fit = True, chi2_suffix = None, subtract_1dof_for_binning = False):

    Nparameters = len(fitting_object.values[:])
    if chi2_suffix is None:
        chi2_suffix = ''
    else:
        chi2_suffix = f'({chi2_suffix})'
   
    dictionary = {f'{chi2_suffix} Npoints': Ndatapoints}


    for i in range(Nparameters):
        dict_new = {f'{chi2_suffix} {fitting_object.parameters[i]}': [fitting_object.values[i], fitting_object.errors[i]]}
        dictionary.update(dict_new)
    if subtract_1dof_for_binning:
        Ndof = Ndatapoints - Nparameters - 1
    else:
        Ndof = Ndatapoints - Nparameters

    dictionary.update({f'{chi2_suffix} Ndof': Ndof})

    if chi2_fit:
        chi2 = fitting_object.fval
        p = stats.chi2.sf(chi2, Ndof)   
        dictionary.update({f'{chi2_suffix} chi2': chi2, f'{chi2_suffix} pval': p})

    return dictionary

def runstest(residuals):
   
    N = len(residuals)

    indices_above = np.argwhere(residuals > 0.0).flatten()
    N_above = len(indices_above)
    N_below = N - N_above

    print(N_above)
    print("bel", N_below)
    # calculate no. of runs
    runs = 1
    for i in range(1, len(residuals)):
        if np.sign(residuals[i]) != np.sign(residuals[i-1]):
            runs += 1

    # calculate expected number of runs assuming the two samples are drawn from the same distribution
    runs_expected = 1 + 2 * N_above * N_below / N
    runs_expected_err = np.sqrt((2 * N_above * N_below) * (2 * N_above * N_below - N) / (N ** 2 * (N-1)))

    # calc test statistic
    test_statistic = (runs - runs_expected) / runs_expected_err

    print("Expected runs and std: ", runs_expected, " ", runs_expected_err)
    print("Actual no. of runs: ", runs)
    # use t or z depending on sample size (2 sided so x2)
    if N < 50:
        p_val = 2 * stats.t.sf(np.abs(test_statistic), df = N - 2)
    else:
        p_val = 2 * stats.norm.sf(np.abs(test_statistic))

    return test_statistic, p_val

def calc_weighted_mean(x, dx, axis = -1):
    """
    returns: weighted mean, error on mean,
    """
    assert(len(x) > 1)
    assert(len(x) == len(dx))
    
    var = 1 / np.sum(1 / dx ** 2, axis = axis)
    mean = np.sum(x / dx ** 2, axis = axis) * var

    return mean, np.sqrt(var)

def calc_weighted_mean_vec(x, dx):
    """
    returns: weighted mean, error on mean, Ndof, Chi2, p_val
    """
    assert(len(x) > 1)
    assert(len(x) == len(dx))
    
    var = 1 / np.sum(1 / dx ** 2)
    mean = np.sum(x / dx ** 2) * var

    # Calculate statistics
    Ndof = len(x) - 1
    chi2 = np.sum((x - mean) ** 2 / dx ** 2)
    p_val = stats.chi2.sf(chi2, Ndof)

    return mean, np.sqrt(var), Ndof, chi2, p_val

def calc_corr_matrix(x):
    """assuming that each column of x represents a separate variable"""
   
    data = x.astype('float')
    rows, cols = data.shape
    corr_matrix = np.empty([cols, cols])
 
    for i in range(cols):
        for j in range(i, cols):
                corr_matrix[i,j] = (np.mean(data[:,i] * data[:,j]) - data[:,i].mean() * data[:,j].mean()) / (data[:,i].std(ddof = 0) * data[:,j].std(ddof = 0))

        corr_matrix[j,i] = corr_matrix[i,j]
    return corr_matrix

def prop_err(dzdx, dzdy, x, y, dx, dy, correlation = 0):
    """ derivatives must takes arguments (x,y)
    """
    var_from_x = dzdx(x,y) ** 2 * dx ** 2
    var_from_y = dzdy (x, y) ** 2 * dy ** 2
    interaction = 2 * correlation * dzdx(x, y) * dzdy (x, y) * dx * dy

    prop_err = np.sqrt(var_from_x + var_from_y + interaction)

    if correlation == 0:
        return prop_err, np.sqrt(var_from_x), np.sqrt(var_from_y)
    else:
        return prop_err

def do_adf_test(time_series, maxlag = None, autolag = 'AIC', regression = 'c', verbose = True):
    """
    Performs the augmented Dickey-Fuller test on a time series.
    """
    result = adfuller(time_series, maxlag = maxlag, autolag = autolag, regression = regression)
    if verbose:
        print(f'ADF Statistic: {result[0]}')
        print(f'p-value: {result[1]}')
        print(f'nobs used: {result[3]}')
        print(f'lags used: {result[2]}')
        print(f'Critical Values:')
        
        for key, value in result[4].items():
            print(f'\t{key}: {value}')

    return result