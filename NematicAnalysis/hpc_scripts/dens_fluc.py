# Author:  Simon Guldager
# Date (latest update): 

### SETUP ------------------------------------------------------------------------------------

## Imports:
import os
import sys
import pickle
import warnings
import time
import argparse
import logging

import numpy as np


sys.path.append('/groups/astro/kpr279/')
import massPy as mp

#from sklearn.neighbors import KDTree

development_mode = False
if development_mode:
    num_frames = 5


### FUNCTIONS ----------------------------------------------------------------------------------


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

def get_defect_list(archive, LX, LY, idx_first_frame=0, verbose=False):
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

    if verbose:
        t_start = time.time()
    if not development_mode:
        Nframes = archive.__dict__['num_frames']
    else:  
        Nframes = num_frames

    # Loop over frames
    for i in range(idx_first_frame, Nframes):
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
        # print 2 with 2 decimals
        print('Time to get defect list: %.2f s' % t_end)

    return top_defects

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

def calc_distance_matrix(center_point, position_arr, L, periodic = True):
    """
    Compute a distance matrix between a center point and an array of positions.
    
    Parameters:
    - Center point: A tuple of coordinates (x, y, ... Ndim).
    - A: Arr of points in format (Npoints, Ndim)
    - L: Length of the square domain in each dimension.
    - periodic: If True, use periodic boundary conditions.
    
    Returns:
    - distance_matrix: A numpy array of distances with shape (Npoints).
    """

    if periodic:
        min_func_vectorized = lambda x: np.minimum(x, L - x)
        dR = np.abs(position_arr - center_point)
        dr = np.apply_along_axis(min_func_vectorized, axis = 1, arr = dR)   
    else:  
        dr = position_arr - center_point

    distance_arr = np.sqrt((dr**2).sum(axis = 1))   
    return distance_arr

def calc_density_fluctuations(points_arr, window_sizes, side_length, periodic = True, N_center_points=1,):
    """
    Calculates the density fluctuations for a set of points in a 2D plane for different window sizes.
    For each window_size (i.e. radius), the density fluctuations are calculated by choosing N_center_points random points
    inside a region determined by dist_to_boundaries and calculating the number of points within a circle of radius R for each
    of these points, from which the number and density variance can be calculated.

    Parameters:
    -----------
    points_arr : (numpy array) - Array of points in 2D plane in format (Npoints, 2) located within the square domain [0, side_length] x [0, side_length]
    window_sizes : (numpy array) - Array of window sizes (i.e. radii) for which to calculate density fluctuations
    side_length : (float) - Length of the square domain in each dimension   
    periodic : (bool) - If True, use periodic boundary conditions
    N_center_points : (int) - Number of center points to use for each window size. If None, all points are used.

    Returns:
    --------
    av_counts : (numpy array) - Array containing the number of points for each window and windowsize in format (Nwindows, Ncenterpoints)
    """

    # Initialize density array, density variance array, and counts variance arrays
    counts = np.zeros((len(window_sizes), N_center_points))
    rmax = window_sizes[-1]

    for j in range (N_center_points):

        # generate random center point
        if periodic:
            center_point = np.random.rand(2) * side_length
        else:
            center_point = rmax + np.random.rand(2) * (side_length - 2 * rmax)

        # Calculate distance matrix
        distance_matrix = calc_distance_matrix(center_point, points_arr, L = side_length, periodic = periodic)

        # Calculate number of points within each window size
        counts[:,j] = (distance_matrix < window_sizes[:,None]).sum(axis=1)
    return counts

def get_density_fluctuations(top_defect_list, window_sizes, side_length,  \
                                    N_center_points = 1, periodic = True, save_path = None):
    """
    Calculate defect density fluctuations for different window sizes
    Parameters:
    -----------
    top_defect_list: list of dictionaries, each dictionary contains defect positions and charges for one frame
    window_sizes: array of window sizes (i.e. radii) for which to calculate density fluctuations
    side_length: length of the square domain in each dimension
    N_center_points: number of center points to use for each window size. If None, all points are used.
    periodic: if True, use periodic boundary conditions for distance calculations
    save_path: path to file to save count arr. If None, do not save.

    Returns:
    --------
    count_arr: array of number of points for each window and windowsize in format (Nframes, Nwindows, Ncenterpoints)
    """

    # Intialize array of count fluctuations and average counts
    count_arr = np.zeros([len(top_defect_list), len(window_sizes), N_center_points])

    for frame, defects in enumerate(top_defect_list):

        Ndefects = len(defects)
        if Ndefects == 0:
            count_arr[frame] = 0
            continue

        # Convert dictionary of defects to array of defect positions
        defect_positions =  get_defect_arr_from_frame(defects)   

        count_arr[frame] = calc_density_fluctuations(defect_positions, window_sizes, side_length, \
                                                    periodic = periodic, N_center_points = N_center_points)
    
    if save_path is not None:
        np.save(save_path, count_arr)
    return count_arr

def est_stationarity(time_series, interval_len, Njump, Nconverged, max_sigma_dist = 2):
 
    # Estimate the stationarity of a time series by calculating the mean and standard deviation
    # of the time series in intervals of length interval_len. If the mean of a block is sufficiently
    # close to the mean of the entire time series, then the block is considered stationary.

    Nframes = len(time_series)
    Nblocks = int(Nframes / interval_len)
    converged_mean = np.mean(time_series[Nconverged:])
    global_std = np.std(time_series[Nconverged:], ddof = 1)

    it = 0
    while it * Njump < Nframes - interval_len:
        mean_block = np.mean(time_series[it * Njump: it * Njump + interval_len])
        dist_from_mean = np.abs(mean_block - converged_mean) / global_std

        if np.abs(mean_block - converged_mean) > max_sigma_dist * global_std:
            it += 1
        else:
            return it * Njump, True
    return it * Njump, False

def gen_status_txt(message = '', log_path = None):
    """
    Generate txt file with message and no.
    """
    with open(log_path, 'w') as f:
        f.write(message)
    return

def get_windows(Nwindows, min_window_size, max_window_size, logspace = False):
    """
    Get window sizes
    """
    if logspace:
        window_sizes = np.logspace(np.log10(min_window_size), np.log10(max_window_size), Nwindows)
    else:
        window_sizes = np.linspace(min_window_size, max_window_size, Nwindows)
    return window_sizes



### MAIN ---------------------------------------------------------------------------------------



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str)
    parser.add_argument('--output_folder', type=str)
    parser.add_argument('--defect_list_folder', type=str, default=None)
    parser.add_argument('--periodic', type=bool, default=False) 
    parser.add_argument('--rmax_fraction', type=float, default=0.1)
    args = parser.parse_args()

    archive_path = args.input_folder
    output_path = args.output_folder
    defect_list_folder = args.defect_list_folder

    defect_dir = output_path if defect_list_folder is None else defect_list_folder
    defect_position_path = os.path.join(defect_dir, f'defect_positions.pkl')

    # Load data archive
    t1 = time.perf_counter()
    ar = mp.archive.loadarchive(archive_path)
    LX, LY = ar.__dict__['LX'], ar.__dict__['LY']
    exp = int(output_path.split('_')[-1])
    act = float(output_path.split('_')[-3])

    msg = f"\nAnalyzing experiment {exp} and activity {act}"
    print(msg)

    # Define window sizes as fraction of system size
    periodic = args.periodic
    suffix = '_periodic' if periodic else ''
    Ncenter_points = 1
    Nwindows = 50
    min_window_size_fraction = 0.01
    max_window_size_fraction = args.rmax_fraction
    logspace = False
    window_sizes = get_windows(Nwindows, min_window_size_fraction * LX, \
                               max_window_size_fraction * LX, logspace = logspace)

    print(f"rmax: {window_sizes[-1]}")

    # Define paths 
    av_defects_path = os.path.join(output_path, f'Ndefects.txt')
    counts_arr_path = os.path.join(output_path, f'av_counts{suffix}_rm{max_window_size_fraction}.npy')
    window_sizes_path = os.path.join(output_path, f'window_sizes.txt')
    model_params_path = os.path.join(output_path, f'model_params.pkl')

    if os.path.exists(defect_position_path):
        with open(defect_position_path, 'rb') as f:
            top_defects = pickle.load(f)
    else:
        # Get defect list
        top_defects = get_defect_list(ar, LX, LY,)

        # save top_defects
        with open(os.path.join(output_path, 'defect_positions.pkl'), 'wb') as f:
            pickle.dump(top_defects, f)
        print(f"Time to calculate defect positions for experiment {exp} and activity {act}: ", np.round(time.perf_counter()-t1,2), "s")

    t2 = time.perf_counter()
    print(f"Calculating defect densities for experiment {exp} and activity {act}")
    
    # save parameters
    np.savetxt(window_sizes_path, window_sizes)
    model_params = ar.__dict__.copy()
    # save model_params
    with open(model_params_path, 'wb') as fp:
        pickle.dump(model_params, fp)

    # Get total no. of defects
    _ = get_defect_density(top_defects, area = 1, save = True, save_path = av_defects_path)

    # Get density fluctuations
    _ = get_density_fluctuations(top_defects, window_sizes, \
                                side_length = LX, N_center_points=Ncenter_points, \
                                periodic=periodic, save_path = counts_arr_path)

    msg = f"Time to calculate density fluctuations for experiment {exp} and activity {act}: {np.round(time.perf_counter()-t2,2)} s\n"
    print(msg)

    gen_status_txt(msg, os.path.join(output_path, 'dens_analysis_completed.txt'))


if __name__ == '__main__':
    main()
