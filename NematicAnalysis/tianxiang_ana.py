
import enum
import os
import pickle as pkl
import random
import sys
import logging
from itertools import tee
from time import perf_counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker

from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
rpy2_logger.setLevel(logging.ERROR)   # will display errors, but not warnings

from structure_factor.point_pattern import PointPattern
from structure_factor.spatial_windows import BoxWindow
from structure_factor.hyperuniformity import bin_data
from structure_factor.structure_factor import StructureFactor
import structure_factor.pair_correlation_function as pcf

import massPy as mp

from utils import *
from plot_utils import *
from AnalyseDefects_dev import AnalyseDefects
from AnalyseDefectsAll import AnalyseDefectsAll

plt.style.use('sg_article')
#plt.rcParams.update({"text.usetex": True})

## Change directory to current one
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

## ---------------------------------------------------------


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

def get_density_fluctuations(top_defect_list, window_sizes, side_length, lattice_space_scaling = 1,  \
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
        defect_positions *= lattice_space_scaling 

        count_arr[frame] = calc_density_fluctuations(defect_positions, window_sizes, side_length, \
                                                    periodic = periodic, N_center_points = N_center_points)
    
    if save_path is not None:
        np.save(save_path, count_arr)
    return count_arr

def get_structure_factor(top_defect_list, box_window, lattice_space_scaling = 1,
                         kmax = 1, nbins = 50,):
    """
    Calculate structure factor for the frames in frame_interval
    """

    # Get number of frames
    Nframes = len(top_defect_list)

    # Initialize structure factor
    sf_arr_init = False

    for i, defects in enumerate(top_defect_list):

        # Get defect array for frame
        defect_positions = get_defect_arr_from_frame(defects)

        if defect_positions is None:
            continue

        defect_positions *= lattice_space_scaling

        # Initialize point pattern
        point_pattern = PointPattern(defect_positions, box_window)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            sf = StructureFactor(point_pattern)
            k, sf_estimated = sf.scattering_intensity(k_max=kmax,)
    
        # Bin data
        knorms = np.linalg.norm(k, axis=1)
        kbins, smeans, sstds = bin_data(knorms, sf_estimated, bins=nbins,)

        # Store results
        if not sf_arr_init:
            kbins_arr = kbins.astype('float')
            sf_arr = np.zeros([Nframes, len(kbins_arr), 2]) * np.nan
            sf_arr_init = True
   
        sf_arr[i, :, 0] = smeans
        sf_arr[i, :, 1] = sstds

    if sf_arr_init:
        return kbins_arr, sf_arr
    else:
        return None, None

def get_pair_corr_from_defect_list(defect_list, ball_window, frame_idx_interval = None, 
                        Npoints_min = 30, method = "fv", \
                        kest_kwargs = {'rmax': 10, 'correction': 'best', 'var.approx': False},\
                        smoothing_kwargs = dict(method="b", spar=0.85, nknots=25), 
                        lattice_space_scaling = 1, save=False, save_dir=None, save_suffix=''):
    """
    Calculate pair correlation function for the frames in frame_interval
    """

    # Get number of frames
    Nframes = len(defect_list) if frame_idx_interval is None else frame_idx_interval[1] - frame_idx_interval[0]
    frame_interval = [0, Nframes] if frame_idx_interval is None else frame_idx_interval

    arrays_initialized = False

    for i, frame in enumerate(range(frame_interval[0], frame_interval[1])):

        if i % 10 == 0:
            print(f"Processing frame {i+1}/{Nframes}...")

        # Get defect array for frame
        defect_positions = get_defect_arr_from_frame(defect_list[frame])

        # Skip if less than Npoints_min defects
        if defect_positions is None:
            continue
        if len(defect_positions) < Npoints_min:
            continue

        defect_positions *= lattice_space_scaling  # Scale defect positions by lattice spacing

        # Initialize point pattern
        point_pattern = PointPattern(defect_positions, ball_window)

        # Calculate pair correlation function
        pcf_estimated = pcf.estimate(point_pattern, method=method, \
                                 Kest=kest_kwargs, fv=smoothing_kwargs)
   
        # Store results
        if not arrays_initialized:
            rad_arr = pcf_estimated.r.values
            pcf_arr = np.nan * np.zeros([Nframes, len(rad_arr)])
            arrays_initialized = True
        pcf_arr[i] = pcf_estimated.pcf

    if arrays_initialized:
        if save:
            rad_arr_path = os.path.join(save_dir, f'rad_arr{save_suffix}.npy')
            pcf_arr_path = os.path.join(save_dir, f'pcf_arr{save_suffix}.npy')
            np.save(rad_arr_path, rad_arr)
            np.save(pcf_arr_path, pcf_arr)
    return

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

## ---------------------------------------------------------

def main():

    use_defect_boundaries = True # otherwise, use director boundaries
    extract_defects, calc_gnf, calc_sfac, calc_pcf = False, False, False, True

    frame_idx_interval = [100, 150]

    data_dirs = 'C:\\Users\\Simon Andersen\\OneDrive - University of Copenhagen\\PhD\\Active Nematic Defect Transition\\Tianxiang data\\data_new'
    save_path_params = os.path.join(data_dirs, f'parameters.pkl')
    save_path_full = os.path.join(data_dirs, f'defect_positions.pkl')
    save_path_fields = os.path.join(data_dirs, 'fields.npz')

    data_dict = {}
    params_dict = {}

    lattice_spacing = 40

    periodic = True
    gnf_dict = {'suffix': '_periodic' if periodic else '',
                'Ncenter_points': 1,
                'Nwindows': 50,
                'min_window_size_fraction': 0.01,
                'max_window_size_fraction': .125,
                'logspace': False}
    
    data_names = ['frame1-50.csv', 'frame51-100.csv', 'frame101-150.csv']

    t1 = perf_counter()
    for i, file in enumerate(data_names):
        key_name = file.split('.')[0]
        data_dict[key_name] = pd.read_csv(os.path.join(data_dirs, file),) 

    key_list = list(data_dict.keys())
    print(key_list)

    print(f'Data loaded: {len(key_list)} files in {perf_counter() - t1:.2f} seconds.')

    if extract_defects:
        for Ndata, key in enumerate(key_list):
            print(f'Processing data file: {key} ({Ndata+1}/{len(key_list)})')
            t2 = perf_counter()

            save_path = os.path.join(data_dirs, f'defect_list_{key_list[Ndata]}.pkl')
            
            # Select a specific data file by its key
            df = data_dict[key_list[Ndata]]  # Change index to select different data file

            df['X']  *= 2
            df['Y']  *= 2
            df['DX'] *= 2
            df['DY'] *= 2

            # sorted unique coords & times
            x_coords = np.sort(df['X'].unique())
            y_coords = np.sort(df['Y'].unique())
            times    = np.sort(df['Slice'].unique())

            nx, ny, nt = len(x_coords), len(y_coords), len(times)

            params_dict[key] = {'LX_bounds': [x_coords.min(), x_coords.max()],
                                'LY_bounds': [y_coords.min(), y_coords.max()],
                                'LX_bounds_defects': [0, nx * lattice_spacing],
                                'LY_bounds_defects': [0, nx * lattice_spacing],
                                'Nx_grid_points': nx,
                                'Ny_grid_points': ny,
                                'Nframes': nt}

            # build index maps
            ix_map = {x:i for i, x in enumerate(x_coords)}
            iy_map = {y:j for j, y in enumerate(y_coords)}
            t_map  = {t:k for k, t in enumerate(times)}

            # allocate U,V
            U = np.zeros((nx, ny, nt))
            V = np.zeros((nx, ny, nt))

            # fill
            for _, row in df.iterrows():
                i = ix_map[row['X']]
                j = iy_map[row['Y']]
                k = t_map[row['Slice']]
                U[i, j, k] = row['DX']
                V[i, j, k] = row['DY']

            # meshgrid for plotting (shape ny×nx)
            Xg, Yg = np.meshgrid(x_coords, y_coords)

            plus_defects  = []   # list of (n_i,2) arrays of X,Y
            #minus_defects = []
            #defect_list = []

            ix = np.arange(nx)
            iy = np.arange(ny)

            if Ndata == 0:
                director_x = np.nan * np.ones((nx, ny, 3 * nt), dtype=float)
                director_y = np.nan * np.ones_like(director_x, dtype=float)
            else:
                director_x[:, :, Ndata * nt : (Ndata + 1) * nt] = U.astype(float)
                director_y[:, :, Ndata * nt : (Ndata + 1) * nt] = V.astype(float)

            if os.path.exists(save_path):
                print(f'Skipping {key_list[Ndata]} as defect list already exists.')
                continue

            for k in range(nt):
                print(f'Processing time slice {k+1}/{nt}...')
                raw = mp.base_modules.defects.get_defects(U[:,:,k], V[:,:,k], nx, ny, threshold=0.4)
                if not raw:
                    print(f'No defects found at time slice {k}.')
                    #plus_defects.append(np.empty((0,2)))
                    #minus_defects.append(np.empty((0,2)))
                    continue
                print(f'Found {len(raw)} defects at time slice {k}.')
                defect_list.append(raw)

            if Ndata == 0:
                defect_list_full = defect_list
            else:
                defect_list_full.extend(defect_list)

            # save defect_list at pkl file
            with open(save_path, 'wb') as f:
                pkl.dump(defect_list, f)
            print(f'Processed {key_list[Ndata]} in {perf_counter() - t2:.2f} seconds.')
        

        if not os.path.exists(save_path_fields):
            print('saving npz')
            np.savez(save_path_fields, director_x=director_x, director_y=director_y,)
        if not os.path.exists(save_path_full):
            with open(save_path_full, 'wb') as f:
                pkl.dump(defect_list_full, f)
        if not os.path.exists(save_path_params):
            with open(save_path_params, 'wb') as f:
                    pkl.dump(params_dict, f)

    if os.path.isfile(save_path_full):
        with open(save_path_full, 'rb') as f:
            defect_list_full = pkl.load(f)
    else:
        for i, key in enumerate(key_list):
            save_path = os.path.join(data_dirs, f'defect_list_{key_list[i]}.pkl')
            with open(save_path, 'rb') as f:
                defect_list = pkl.load(f)
            if i == 0:
                defect_list_full = defect_list
            else:
                defect_list_full.extend(defect_list)
        with open(save_path_full, 'wb') as f:
            pkl.dump(defect_list_full, f)
    # load parameters
    with open(save_path_params, 'rb') as f:
        params_dict = pkl.load(f)
    # load fields
    director_x = np.load(save_path_fields, allow_pickle=True)['director_x']
    director_y = np.load(save_path_fields, allow_pickle=True)['director_y']

    param_key = list(params_dict.keys())[-1]

    if use_defect_boundaries:
            x_bounds = params_dict[param_key]['LX_bounds_defects']
            y_bounds = params_dict[param_key]['LY_bounds_defects']
    else:
        x_bounds, y_bounds = params_dict[param_key]['LX_bounds'], params_dict[param_key]['LY_bounds']   
    LX, LY = x_bounds[1] - x_bounds[0], y_bounds[1] - y_bounds[0]

    if calc_gnf:
        # Define paths 
        av_defects_path = os.path.join(data_dirs, f'Ndefects.txt')
        counts_arr_path = os.path.join(data_dirs, f"av_counts{gnf_dict['suffix']}_rm{gnf_dict['max_window_size_fraction']}.npy")
        window_sizes_path = os.path.join(data_dirs, f'window_sizes.txt')

        window_sizes = get_windows(gnf_dict['Nwindows'], gnf_dict['min_window_size_fraction'] * LX, \
                                gnf_dict['max_window_size_fraction'] * LX, gnf_dict['logspace'])
        # save window sizes
        np.savetxt(window_sizes_path, window_sizes)

        # Get total no. of defects
        t1 = perf_counter()
        _ = get_defect_density(defect_list_full, area = 1, save = True, save_path = av_defects_path)

        # Get density fluctuations
        _ = get_density_fluctuations(defect_list_full, window_sizes, lattice_space_scaling = lattice_spacing, \
                                    side_length = LX, N_center_points=gnf_dict['Ncenter_points'], \
                                    periodic=periodic, save_path = counts_arr_path)
        print(f"Time to calculate density and density fluctuations: ", np.round(time.perf_counter()-t1,2), "s")

    if calc_sfac:
        sfac_path = os.path.join(data_dirs, f'sfac.npy')
        kbins_path = os.path.join(data_dirs, f'kbins.txt')

        # Define sfac parameters
        box_window = BoxWindow(bounds=[x_bounds, y_bounds])  
        kmax = 256 / LX

        # Get structure factor
        t2 = time.perf_counter()
        kbins, sfac = get_structure_factor(defect_list_full, box_window, 
                                           lattice_space_scaling = lattice_spacing,
                                            kmax = kmax, nbins = 50,)

        print(f"Time to calculate structure factor: ", np.round(time.perf_counter()-t2,2), "s")

        if sfac is None:
            print(f"No defects in data. Skipping...")
        else:
            # Save structure factor
            np.save(sfac_path, sfac)
            np.savetxt(kbins_path, kbins)

    if calc_pcf:

        # set parameters
        rmax = int((x_bounds[-1] - x_bounds[0])/4 - 1) 
        nknots = int(rmax / (lattice_spacing / 2))
        method = 'fv'
        spar = 1.2
        smoothing_kwargs = dict(method="b", spar=spar, nknots=nknots)
        kest_kwargs = {'rmax': rmax, 'correction': 'good', 'nlarge': 3000, 'var.approx': False}

        box_window = BoxWindow(bounds=[x_bounds, x_bounds])  

        print([x_bounds, y_bounds], rmax)
        print(box_window)
        print(kest_kwargs)
        print(smoothing_kwargs)
        print(method)

        t3 = time.perf_counter()
        get_pair_corr_from_defect_list(defect_list_full, box_window, \
                                    frame_idx_interval = frame_idx_interval, \
                                    method = method, lattice_space_scaling=lattice_spacing, \
                                    kest_kwargs = kest_kwargs, smoothing_kwargs = smoothing_kwargs, \
                                    save=True, save_dir=data_dirs,)
        
        print(f"Time to calculate pcf: ", np.round(time.perf_counter()-t3,2), "s")
    
if __name__ == "__main__":
    main()