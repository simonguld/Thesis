# Author: Simon Guldager Andersen
# Date(last edit): June 2 2024

## Imports:
import os
import sys
import warnings
import time
import shutil
import io
import lz4
import json
import pickle

import numpy as np
from iminuit import Minuit
from scipy import stats
from sklearn.neighbors import KDTree
from sklearn.cluster import AgglomerativeClustering
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

def compress_file(input_file, output_file):
    # Open the input file in binary mode for reading
    with open(input_file, 'rb') as f_in:
        # Read the data from the input file
        data = f_in.read()

        # Compress the data using lz4 compression
        compressed_data = lz4.frame.compress(data)

        # Write the compressed data to the output file
        with open(output_file, 'wb') as f_out:
            f_out.write(compressed_data)
    return

def print_size(input_file, output_file):
    # Get the size of the input file
    input_size = os.path.getsize(input_file)

    # Get the size of the output file
    output_size = os.path.getsize(output_file)

    # Print the size of the input and output files
    print(f'Input file size: {input_size / 1024 ** 2:.2f} MB')
    print(f'Output file size: {output_size / 1024 ** 2:.2f} MB')

    # Calculate the compression ratio
    ratio = input_size / output_size
    print(f'Compression ratio: {ratio:.2f}x')
    return

def estimate_size_reduction(input_file, output_file, Nframes = 1, verbose=True):
    # Get the size of the input file
    input_size = os.path.getsize(input_file)

    # Get the size of the output file
    output_size = os.path.getsize(output_file)

    # Calculate the compression ratio
    ratio = input_size / output_size

    if verbose:
        print(f'Uncompressed archive size: {input_size / 1024 ** 2 * Nframes:.2f} MB')
        print(f'Compressed archive size: {output_size / 1024 ** 2 * Nframes:.2f} MB')
        print(f'Compression ratio: {ratio:.2f}x\n')

    return input_size * Nframes, output_size * Nframes, ratio

def decompress_and_convert(input_file, out_format = 'json'):
    # Open the input file in binary mode for reading
    with open(input_file, 'rb') as f_in:
        # Read the compressed data from the input file
        compressed_data = f_in.read()

        # Decompress the data using lz4 decompression
        decompressed_data = lz4.frame.decompress(compressed_data)    

        if out_format == 'json':
            # Decode the bytes to string
            decoded_data = decompressed_data.decode('utf-8')
            json_data = json.loads(decoded_data)
            return json_data       
        elif out_format == 'npz':
            # Decode the bytes to string and parse JSON
            npz_data = npz_data = np.load(io.BytesIO(decompressed_data), allow_pickle=True)
            return npz_data      
        else:
            print('Invalid output format. Please use "json" or "npz"')
            return
        
def unpack_arrays(json_dict, dtype_out = 'float64', exclude_keys=[]):
    keys = list(json_dict['data'].keys())
    arr_dict = {}
    arr_dict = {key: np.array(json_dict['data'][key]['value'],dtype=dtype_out) for key in keys if key not in exclude_keys}
   # for key in keys:
    #    arr_dict[key] = np.array(json_dict['data'][key]['value'],dtype=dtype_out)
    return arr_dict

def unpack_nematic_json_dict(json_dict, dtype_out = 'float64', exclude_keys=[], calc_velocities = False):
    keys = list(json_dict['data'].keys())
    arr_dict = {}
    arr_dict = {key: np.array(json_dict['data'][key]['value'],dtype=dtype_out) for key in keys if key not in exclude_keys}

    if calc_velocities:
        ff = np.array(json_dict['data']['ff']['value'],dtype=dtype_out)
        arr_dict['vx'], arr_dict['vy'] = mp.base_modules.flow.velocity(ff,)
    return arr_dict

def find_missing_frames(archive_path):

    ar = mp.archive.loadarchive(archive_path)

    dir_list = os.listdir(archive_path)
    frame_list = []

    for item in dir_list:
        if item.startswith("frame"):
            frame_num = int(item.split('.')[0].split('frame')[-1])
            frame_list.append(frame_num)

    if len(frame_list) == ar.num_frames:
        return np.arange(ar.nstart, ar.nsteps + 1, ar.ninfo)
    else:
        frame_list.sort()
        return frame_list
    
def convert_json_to_npz(json_path, out_path, compress = True, dtype_out = 'float64', exclude_keys=[]):

    with open(json_path, 'r') as f:
        data = json.load(f)

    arr_dict = unpack_arrays(data,  dtype_out = dtype_out, exclude_keys=exclude_keys)
    if compress:
        np.savez_compressed(out_path, **arr_dict)
    else:
        np.savez(out_path, **arr_dict)
    return

def create_npz_folder(archive_path, output_folder = None, check_for_missing_frames = False, compress = True, \
                      dtype_out= 'float32', exclude_keys=[], verbose = 1):
    """
    verbose = 0: no output
    verbose = 1: print time to process entire archive
    verbose = 2: print time to process each frame
    """
    # Create the output folder if it does not exist

    output_folder = archive_path + '_npz' if output_folder is None else output_folder

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # copy parameters.json to output folder
    parameters_path = os.path.join(archive_path, 'parameters.json')
    shutil.copy(parameters_path, output_folder)

    # Load the archive and get the list of frames
    ar = mp.archive.loadarchive(archive_path)
    frame_list = find_missing_frames(archive_path) if check_for_missing_frames else np.arange(ar.nstart, ar.nsteps + 1, ar.ninfo)

    # initialize failed conversions list
    failed_conversions = []

    if verbose > 0:
        start = time.perf_counter()

    for i, frame in enumerate(frame_list):
        frame_input_path = os.path.join(archive_path, f'frame{frame}.json')
        frame_output_path = os.path.join(output_folder, f'frame{frame}.npz')
        try:
            if verbose == 2:
                start_frame = time.perf_counter()
            convert_json_to_npz(frame_input_path, frame_output_path, compress = compress, dtype_out = dtype_out, exclude_keys = exclude_keys)
            if verbose == 2:
                print(f'Frame {frame} processed in {time.perf_counter() - start_frame:.2f} seconds')
        except:
            print(f'Error processing frame {frame}. Skipping...')
            failed_conversions.append(frame)

    if verbose > 0:
        print(f'Archive processed in {time.perf_counter() - start:.2f} seconds with {len(failed_conversions)} failed conversions')
        if len(failed_conversions) > 0:
            print(f'Frames for which conversion to npz failed: {failed_conversions}')
        print('\nEstimated (from first frame) size reduction of archive: ')

        frame_input_path = os.path.join(archive_path, f'frame{frame_list[0]}.json')
        frame_output_path = os.path.join(output_folder, f'frame{frame_list[0]}.npz')
        input_size, output_size, ratio = estimate_size_reduction(frame_input_path, frame_output_path, Nframes = len(frame_list))
    return


# Functions for nematic analysis  -----------------------------------------------------

def get_frame_number(idx, path, ninfo):
    """
    Assuming equal spacing between frames, get the frame number from the index
    """
    frame_list = []
    for f in os.listdir(path):
        if f.startswith('frame'):
            frame_list.append(int(f.split('.')[0].split('frame')[-1]))
    frame_list.sort()
    frame_interval = frame_list[1]-frame_list[0]

    return int(frame_interval / ninfo) * idx

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

def get_defect_list(archive, idx_first_frame=0, Nframes = None, verbose=False, archive_path = None):
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

    LX = archive.LX
    LY = archive.LY

    Nframes = archive.__dict__['num_frames'] if Nframes is None else Nframes
    if verbose:
        t_start = time.time()

    # Loop over frames
    for i in range(idx_first_frame, Nframes - idx_first_frame):
        # Load frame

        frame_num = i if archive_path is None else get_frame_number(i, archive_path, archive.__dict__['ninfo'])

        frame = archive._read_frame(frame_num)
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

def gen_clustering_metadata(path,):
    """
    Given a path to a directory containing the defect clustering data, it returns Nexp_list, act_list, act_dir_list
    """

    Nexp_list = []
    act_list = []
    act_dir_list = []
 
    for i, dir in enumerate(os.listdir(path)):
        if dir.startswith('analysis'):
            act_list.append(float(dir.split('_')[-1]))
            act_dir_list.append(os.path.join(path, dir))
    for i, file in enumerate(os.listdir(os.path.join(path, os.listdir(path)[0]))):
        if file.startswith('zeta'):
            Nexp_list.append(int(file.split('_')[-1]))

    return Nexp_list, act_list, act_dir_list

def do_poisson_clustering(Nlist, L, Ntrial, Ncmin = 2, method_kwargs = dict(n_clusters=None, linkage = 'single', distance_threshold=33)):
    """
    Simulate points uniformly and do Agglomerative clustering for a given set of parameters

    Nlist: list of number of defects for each activity
    L: length of the square box
    Ntrial: number of trials per N in Nlist
    Ncmin: minimum number of defects in a cluster
    method_kwargs: dictionary of arguments for AgglomerativeClustering

    Returns: cluster_arr, cl_mean, cl_std

    """

    cluster_arr = np.nan * np.zeros([4, len(Nlist), Ntrial])

    for i, N in enumerate(Nlist):

        p_arr = np.random.rand(N, 2, Ntrial) * L   
        cst = AgglomerativeClustering(**method_kwargs)

        for j in range(Ntrial):

            # Get defect array for frame
            defect_positions = p_arr[:, :, j]

            labels = cst.fit_predict(defect_positions)

            unique, counts = np.unique(labels, return_counts=True)

            # Only count clusters with more than Ncmin defects
            mask = (counts >= Ncmin)
            counts_above_min = counts[mask]

            # store the total number of defects
            cluster_arr[0, i, j] = N

            # store the fraction of clustered defects
            cluster_arr[1, i, j] = counts_above_min.sum() / N

            # store the number of clusters
            cluster_arr[2, i, j] = len(counts_above_min)
    
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)

                # store the average cluster size
                cluster_arr[3, i, j] = np.mean(counts_above_min)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        cl_mean = np.nanmean(cluster_arr, axis = -1)
        cl_std = np.nanstd(cluster_arr, axis = -1)
        cl_std /= np.sqrt(Ntrial)

    return cluster_arr, cl_mean, cl_std

def do_poisson_clustering_improved(def_arr, L, Ntrial, Ncmin = 2, method_kwargs = dict(n_clusters=None, linkage = 'single', distance_threshold=33)):
    """
    This function is an improved version of do_poisson_clustering. It allows for Ndefects to vary for each activity,
    and also allows for Ntrial to be bigger than the number of defect entries in def_arr. For each run, it randomly selects
    Ndefects from the given column in def_arr, simulates points uniformly and performs the clustering.

    Parameters:
        def_arr: array of defect entries for each activity. Format (Ndefect_entries, Nactivities)
        L: length of the square box
        Ntrial: number of trials per column in def_arr
        Ncmin: minimum number of defects in a cluster
        method_kwargs: dictionary of arguments for AgglomerativeClustering

    Returns: cluster_arr, cl_mean, cl_std

    """
    
    _, Nact = def_arr.shape
    cluster_arr = np.nan * np.zeros([4, Nact, Ntrial])

    cst = AgglomerativeClustering(**method_kwargs)

    for i in range(Nact):

        Nlist = np.random.choice(def_arr[:, i], size = Ntrial)
        

        for j in range(Ntrial):

           # N = np.random.choice(def_arr[:, i], size = 1)[0]
            N = Nlist[j]

            if np.isnan(N):
                continue

            # generate points
            N = int(N)
            defect_positions = np.random.rand(N, 2) * L   

            # cluster
            labels = cst.fit_predict(defect_positions)

            counts = np.unique(labels, return_counts=True)[1]

            # Only count clusters with more than Ncmin defects
            mask = (counts >= Ncmin)
            counts_above_min = counts[mask]

            # store the total number of defects
            cluster_arr[0, i, j] = N

            # store the fraction of clustered defects
            cluster_arr[1, i, j] = counts_above_min.sum() / N

            # store the number of clusters
            cluster_arr[2, i, j] = len(counts_above_min)
    
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)

                # store the average cluster size
                cluster_arr[3, i, j] = np.mean(counts_above_min)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        cl_mean = np.nanmean(cluster_arr, axis = -1)
        cl_std = np.nanstd(cluster_arr, axis = -1)
        cl_std /= np.sqrt(Ntrial)

    return cluster_arr, cl_mean, cl_std

def extract_clustering_results(Nframes, act_list, act_dir_list, Nexp, Ncmin=2, save = False, save_path = None):
    """
    Analyse the defects for all the input folders
    """
    for data_set in range(1):
        # create arrays to store the clustering data
        cluster_arr = np.nan * np.zeros([Nframes, 4, len(act_list), Nexp])
        
        # print('Analysing clustering data for input folder {}'.format(self.input_paths[N]))
        for i, (act, act_dir) in enumerate(zip(act_list, act_dir_list)):

            exp_list = []
            exp_dir_list = []

            for file in os.listdir(act_dir):
                exp_count = file.split('_')[-1]
                exp_list.append(int(exp_count))
                exp_dir_list.append(os.path.join(act_dir, file))

            # sort the activity list and the activity directory list
            exp_list, exp_dir_list = zip(*sorted(zip(exp_list, exp_dir_list)))

            for j, (exp, exp_dir) in enumerate(zip(exp_list, exp_dir_list)):

                with open(os.path.join(exp_dir, 'labels_rm33.pkl'), 'rb') as f:
                        labels = pickle.load(f)
                nan_counter = 0
                for k, frame in enumerate(labels[:Nframes]):
                        
                        if frame is None:
                            nan_counter += 1
                            continue

                        Ndefects = len(frame)
                             
                        # store the number of defects 
                        cluster_arr[k, 0, i, j] = Ndefects

                        counts = np.unique(frame, return_counts=True)[1]

                        # Only count clusters with more than Ncmin defects
                        mask = (counts >= Ncmin)
                        counts_above_min = counts[mask]

                        # store the fraction of clustered defects
                        cluster_arr[k, 1, i, j] = counts_above_min.sum() / Ndefects
                        # store the number of clusters
                        cluster_arr[k, 2, i, j] = len(counts_above_min)
                
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=RuntimeWarning)
                            # store the average cluster size
                            cluster_arr[k, 3, i, j] = np.nanmean(counts_above_min)

        # average over experiments and frames
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            cl_mean = np.nanmean(cluster_arr, axis = (0,-1))
            cl_std = np.nanstd(cluster_arr, axis = (0,-1))
            cl_std /= np.sqrt(Nframes * Nexp)
        if save:
            pass
    return cluster_arr, cl_mean, cl_std


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
    if not len(x) > 1:
        print('Length of x must be greater than 1')
        return
    if not len(x) == len(dx):
        print('Length of x and dx must be equal')
        return
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        var = 1 / np.nansum(1 / dx ** 2, axis = axis)
        mean = np.nansum(x / dx ** 2, axis = axis) * var

    return mean, np.sqrt(var)

def calc_weighted_mean_vec(x, dx, replace_null_uncertainties = True):
    """
    returns: weighted mean, error on mean,
    """
  
    if not len(x) > 1:
        print('Length of x must be greater than 1')
        return
    if not len(x) == len(dx):
        print('Length of x and dx must be equal')
        return

    if replace_null_uncertainties:
          dx[dx == 0] = np.nanmean(dx[dx != 0])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        var_vec_inv = (1 / dx ** 2)
        var = 1 / np.nansum(var_vec_inv)
        mean = np.nansum(x / dx ** 2,) * var

        # Calculate statistics
        Ndof = len(x) - 1
        chi2 = np.nansum((x - mean) ** 2 / dx ** 2)
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

def generate_moments(order_param, LX, conv_list):

    moments = np.nan * np.zeros((4, order_param.shape[1]))

    for i in range(order_param.shape[1]):

        cum1 = stats.kstat(order_param[conv_list[i]:, i, :], n = 1, axis = None, nan_policy = 'omit')
        cum2 = stats.kstat(order_param[conv_list[i]:, i, :], n = 2, axis = None, nan_policy = 'omit')
        cum3 = stats.kstat(order_param[conv_list[i]:, i, :], n = 3, axis = None, nan_policy = 'omit')
        cum4 = stats.kstat(order_param[conv_list[i]:, i, :], n = 4, axis = None, nan_policy = 'omit')

        moments[0, i] = cum1 * LX
        moments[1, i] = cum2 * LX + cum1 ** 2 * LX ** 2
        moments[2, i] = cum3 * LX + 3 * cum1 * cum2 * LX ** 2 + cum1 ** 3 * LX ** 3
        moments[3, i] = cum4 * LX + (3 * cum2 ** 2 + 4 * cum1 * cum3) * LX ** 2 + 6 * cum1 ** 2 * cum2 * LX ** 3 + cum1 ** 4 * LX ** 4

    return moments

def one_sample_test(sample_array, exp_value, error_on_mean = None, one_sided = False, small_statistics = False):
    """ Assuming that the errors to be used are the standard error on the mean as calculated by the sample std 
    Returns test-statistic, p_val
    If a scalar sample is passed, the error on the mean must be passed as well, and large statistics is assumed
    """
    if np.size(sample_array) == 1:
        assert(error_on_mean is not None)
        assert(np.size(error_on_mean) == 1)
        assert(small_statistics == False)
        SEM = error_on_mean
        x = sample_array
    else:
        x = sample_array.astype('float')
        Npoints = np.size(x)
        SEM = x.std(ddof = 1) / np.sqrt(Npoints)
    
    test_statistic = (np.mean(x) - exp_value) / SEM

    if small_statistics:
        p_val = stats.t.sf(np.abs(test_statistic), df = Npoints - 1)
    else:
        p_val = stats.norm.sf(np.abs(test_statistic))

    if one_sided:
        return test_statistic, p_val
    else:
        return test_statistic, 2 * p_val

def two_sample_test(x, y, x_err = None, y_err = None, one_sided = False, small_statistics = False):
    """
    x,y must be 1d arrays of the same length. 
    If x and y are scalars, the errors on the means x_rr and y_rr must be passed as well, and small_statistics must be False
    If x and y are arrays, the standard errors on the mean will be used to perform the test

    Returns: test_statistics, p_val
    """
    Npoints = np.size(x)
    assert(np.size(x) == np.size(y))

    if x_err == None:
        SEM_x = x.std(ddof = 1) / np.sqrt(Npoints)
    else:
        assert(small_statistics == False)
        assert(np.size(x_err) == 1)
        SEM_x = x_err
        
    if y_err == None:
        SEM_y = y.std(ddof = 1) / np.sqrt(Npoints)
    else:
        assert(small_statistics == False)
        assert(np.size(y_err) == 1)
        SEM_y = y_err
        

    test_statistic = (np.mean(x) - np.mean(y)) / (np.sqrt(SEM_x ** 2 + SEM_y ** 2)) 

    if small_statistics:
        p_val = stats.t.sf(np.abs(test_statistic), df = 2 * (Npoints - 1))
    else:
        p_val = stats.norm.sf(np.abs(test_statistic))
    if one_sided:
        return test_statistic, p_val
    else:
        return test_statistic, 2 * p_val
