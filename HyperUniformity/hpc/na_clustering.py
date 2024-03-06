# Author:  Simon Guldager
# Date (latest update): 

### SETUP ------------------------------------------------------------------------------------


import os
import sys
import pickle
import warnings
import time
import argparse
import logging

import numpy as np
from sklearn.neighbors import KDTree
from sklearn.cluster import AgglomerativeClustering

from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
rpy2_logger.setLevel(logging.ERROR)   # will display errors, but not warnings

from structure_factor.point_pattern import PointPattern
from structure_factor.spatial_windows import BoxWindow
from structure_factor.hyperuniformity import bin_data
from structure_factor.structure_factor import StructureFactor
import structure_factor.pair_correlation_function as pcf

sys.path.append('/groups/astro/kpr279/')
import massPy as mp


## Change directory to current one
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
sys.path.append(os.getcwd())


development_mode = False
check_for_convergence = False

# if development_mode, use only few frames
if development_mode:
    num_frames = 3


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

def get_structure_factor(top_defect_list, box_window, kmax = 1, debiased = True, direct = True, nbins = 50, \
                         corr_func_method = "fv", method_kwargs = dict(method="b", spar=0.2, nknots = 40), rmax = 10):
    """
    Calculate structure factor for the frames in frame_interval
    """

    # Get number of frames
    Nframes = len(top_defect_list)

    # Initialize structure factor
    sf_arr = None

    for i, defects in enumerate(top_defect_list):

        # Get defect array for frame
        defect_positions = get_defect_arr_from_frame(defects)

        if defect_positions is None:
            continue

        # Initialize point pattern
        point_pattern = PointPattern(defect_positions, box_window)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            sf = StructureFactor(point_pattern)
            k, sf_estimated = sf.scattering_intensity(k_max=kmax, debiased=debiased, direct=direct)
            pcf_estimated = pcf.estimate(point_pattern, method=corr_func_method, \
                                 Kest=dict(rmax=rmax), fv=method_kwargs)

        # Bin data
        knorms = np.linalg.norm(k, axis=1)
        kbins, smeans, sstds = bin_data(knorms, sf_estimated, bins=nbins,)

        # Store results
        if sf_arr is None:
            kbins_arr = kbins.astype('float')
            rad_arr = pcf_estimated.r.values

            sf_arr = np.zeros([Nframes, len(kbins_arr), 2]) * np.nan
            pcf_arr = np.zeros([Nframes, len(rad_arr)]) * np.nan
   
        sf_arr[i, :, 0] = smeans
        sf_arr[i, :, 1] = sstds
        pcf_arr[i] = pcf_estimated.pcf

    return kbins_arr, sf_arr, rad_arr, pcf_arr

def get_clustering(top_defect_list, method, method_kwargs, save = False, save_path = None):
    """
    
    Parameters:
    -----------
    
    Returns:
    --------
    

    """
  
    labels_list = []
    cst = method(**method_kwargs)

    for frame, defects in enumerate(top_defect_list):

        # Get defect array for frame
        defect_positions = get_defect_arr_from_frame(defects)

        if defect_positions is None:
            labels_list.append(None)
            continue

        labels = cst.fit_predict(defect_positions)
        labels_list.append(labels)

    if save:
        # save labels list
        save_path = save_path if save_path is not None else 'labels_list.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump(labels_list, f)

    return labels_list

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
    parser.add_argument('--mode', type=str)
    args = parser.parse_args()

    mode = args.mode
    folder_path = args.input_folder
    output_path = args.output_folder
    read_path = os.path.join(output_path, f'defect_positions.pkl')

    exp = int(output_path.split('_')[-1])
    act = float(output_path.split('_')[-3])

    t1 = time.time()
    msg = f"\nAnalyzing experiment {exp} and activity {act}"
    print(msg)

    
    with open(read_path, 'rb') as f:
        top_defects = pickle.load(f)

    print("\nSucsessfully loaded defect positions")

    dist_max = np.arange(31,50)

    for dist in dist_max:
        save_path = os.path.join(output_path, f'labels_rm{dist}.pkl')
        method_kwargs = dict(n_clusters=None, linkage = 'single', distance_threshold=dist)
        _ = get_clustering(top_defects, AgglomerativeClustering, method_kwargs, save = True, save_path=save_path)

    t2 = time.time()
    msg = f"Time to cluster defects for experiment {exp} and activity {act}: {np.round(t2-t1,2)} s"
    print(msg)



if __name__ == '__main__':
    main()
