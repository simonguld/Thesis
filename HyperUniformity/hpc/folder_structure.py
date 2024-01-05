# Author:  Simon Guldager
# Date (latest update): 

### SETUP ------------------------------------------------------------------------------------

## Imports:

import os
import sys
import pickle
import warnings
import time
import logging
import glob

import numpy as np
from sklearn.neighbors import KDTree

sys.path.append('/groups/astro/kpr279/')
import massPy as mp


## Change directory to current one

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
sys.path.append(os.getcwd())


# Set number of experiments to run
Nexperiments = 10

# Set cutoff for activity
zeta_lower = 0.0
zeta_upper = 0.023

overwrite_log = False
empty_output_folder = False
development_mode = False
run_locally = False

# if development_mode, use only few frames
if development_mode:
    num_frames = 5

if run_locally:
    folder_path = "C:\\Users\\Simon Andersen\\Documents\\Uni\\Speciale\\Hyperuniformity\\nematic_data"
    output_path = 'C:\\Users\\Simon Andersen\\Documents\\Uni\\Speciale\\Hyperuniformity\\output_local'
else:
    ## Define external paths
    folder_path = '/lustre/astro/rsx187/mmout/active_sample_forperp'
    #folder_path = '/lustre/astro/kpr279/nematic_simulation_512'
    output_path = '/lustre/astro/kpr279/output_oldv2'
    log_path = '/lustre/astro/kpr279/output_oldv2.txt'
if not os.path.isdir(output_path):
    os.mkdir(output_path)

### FUNCTIONS ----------------------------------------------------------------------------------

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
        # Get no. of frames
        Nframes = len(defect_list)

        if return_charges:
            # Initialize list of defect densities
            dens_pos_defects = []
            dens_neg_defects = []
            for i, defects in enumerate(defect_list):
                # Get no. of defects
                nposdef = len([d for d in defects if d['charge'] == 0.5])
                nnegdef = len([d for d in defects if d['charge'] == -0.5])

                dens_pos_defects.append(nposdef / area)
                dens_neg_defects.append(nnegdef / area)

            return dens_pos_defects, dens_neg_defects
        else:
            dens_defects = []
            for i, defects in enumerate(defect_list):
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
        logging.info(f"No points within boundaries. Returning NaNs")
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

def gen_status_txt(message = '', exp_path = None):
    """
    Generate txt file with message and no.
    """
    with open(os.path.join(f'{exp_path}', 'analysis_completed.txt'), 'w') as f:
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

def get_relevant_dirs(dirs_all, exp, zeta_lower, zeta_upper):
    """
    Get relevant dirs for a given experiment
    """
    dirs = [dir for dir in dirs_all if int(dir[-1]) == exp]

    keep_list = []
    pop_idx_list = []

    for i, dir in enumerate(dirs):
        split_name = dir.split('z')[-1]
        zeta = float(split_name.split('_')[0])
        if zeta < zeta_lower or zeta > zeta_upper:
            pop_idx_list.append(i)
        else:
            keep_list.append(zeta)

    # Remove folders with activities below the cutoff
    for i in reversed(pop_idx_list):
        dirs.pop(i)

    # Sort dirs according to activity
    dirs = [x for _, x in sorted(zip(keep_list, dirs))]

    # Sort keep_list according to activity
    keep_list = sorted(keep_list)

    return dirs, keep_list

### MAIN ---------------------------------------------------------------------------------------

def main():

    # Get list of folders
    dirs_all = os.listdir(folder_path)

    if overwrite_log:
        if os.path.isfile(log_path):
            os.remove(log_path)
    if empty_output_folder:
        for folder in os.listdir(output_path):
            f_path = os.path.join(f'{output_path}', f'{folder}')
            if os.path.isdir(f_path):
                for file in os.listdir(f_path):
                    file_path = os.path.join(f'{f_path}', f'{file}')
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                os.rmdir(f_path)


    # Initialize logger
    logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info('\n\n\n ------------------ NEW RUN ------------------ \n\n\n')

    # Decide which experiment to analyze [from 0 to 9]
    experiment_list = np.arange(Nexperiments)

    # Define window sizes as fraction of system size
    Nwindows = 30
    min_window_size_fraction = 0.1 / 20
    max_window_size_fraction = 0.1
    logspace = True


    for j, exp in enumerate(experiment_list):
        exp_path = os.path.join(f'{output_path}', f'exp{exp}')
        if not os.path.isdir(exp_path):
            os.mkdir(exp_path)

        completed_path = os.path.join(f'{exp_path}', 'analysis_completed.txt')
        model_params_path = os.path.join(f'{exp_path}', 'model_params.pkl')
 
        if os.path.isfile(completed_path):
            msg = f"Experiment {exp} has already been analyzed. Continuing to next experiment"
            logging.info(msg)
            continue

        msg = f"Analyzing experiment {exp}"
        logging.info(msg)
        t_start = time.time()

        # Find all relevant dirs for a given experiment
        dirs, activity_list = get_relevant_dirs(dirs_all, exp, zeta_lower, zeta_upper)
        logging.info(f"Activity list: {activity_list}")
        logging.info(f"Dirs: {dirs}")

        # Get top. defects for each activity
        for i, dir in enumerate(dirs):

            archive_path = os.path.join(f'{folder_path}', f'{dir}')
            av_defects_path = os.path.join(f'{exp_path}', f'Ndefects_act{activity_list[i]}.txt')
            var_counts_path = os.path.join(f'{exp_path}', f'count_fluctuations_act{activity_list[i]}.txt')
            av_counts_path = os.path.join(f'{exp_path}', f'av_counts_act{activity_list[i]}.txt')
            window_sizes_path = os.path.join(f'{exp_path}', f'window_sizes.txt')

            # check if analysis has already been done
            if os.path.isfile(av_defects_path) and os.path.isfile(var_counts_path) and os.path.isfile(av_counts_path):
                msg = f"Analysis for experiment {exp} and activity {activity_list[i]} has already been done. Continuing to next activity"
                logging.info(msg)
                continue
                
            msg = f"\nAnalyzing experiment {exp} and activity {activity_list[i]}"
            logging.info(msg)
            t1 = time.time()

            # Load data archive
            ar = mp.archive.loadarchive(archive_path)
            LX, LY = ar.__dict__['LX'], ar.__dict__['LY']
 
            if not activity_list[i] == ar.__dict__['zeta']:
                err_msg = f"Activity list and zeta in archive do not match for experiment {exp}. Exiting..."
                logging.error(err_msg)
                raise ValueError(err_msg)
            if not os.path.isfile(model_params_path):
                model_params = ar.__dict__.copy()
                # save model_params
                with open(model_params_path, 'wb') as fp:
                    pickle.dump(model_params, fp)

            window_sizes = get_windows(Nwindows, min_window_size_fraction * LX, max_window_size_fraction * LX, logspace = logspace)
            # save window sizes if not already saved
            if not os.path.isfile(window_sizes_path):
                np.savetxt(window_sizes_path, window_sizes)

            # Get defect list
            top_defects = get_defect_list(ar, LX, LY,)
        
            # Get total no. of defects
            dens_defects = get_defect_density(top_defects, area = 1, save = True, save_path = av_defects_path)

            if development_mode:
                idx_first_frame = 0
            else:
                idx_first_frame, converged = est_stationarity(dens_defects, interval_len = 10, Njump = 25, Nconverged = 100, max_sigma_dist=2)
                logging.info(f"Av. no. of defects CONVERGED after {idx_first_frame} frames \n")
                if not converged:
                    print("Experiment ", exp, " act. ", activity_list[i], " did not converge")
                    logging.info(f"Experiment {exp} act. {activity_list[i]} DID NOT CONVERGE\n")
                    

            # Get count fluctuations and average counts for each frame
            boundaries = [[0, LX], [0, LY]]
            _, _ = get_density_fluctuations(top_defects[idx_first_frame:], window_sizes, boundaries = boundaries, N_center_points= None, Ndof=1, \
                                            save = True, save_path_av_counts=av_counts_path, save_path_var_counts=var_counts_path)

            t2 = time.time()
            msg = f"Time to analyze experiment {exp} and activity {activity_list[i]}: {t2-t1} s"
            logging.info(msg)


        # save txt to indicate exp done
        t_end = time.time() - t_start
        msg = f"Time to analyze experiment {exp}: {t_end} s \n\n"
        logging.info(msg)
        gen_status_txt(msg, exp_path=exp_path)
        
    logging.shutdown()


if __name__ == '__main__':
    main()
