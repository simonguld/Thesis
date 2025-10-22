# Author: Simon Guldager Andersen


## Imports:
import os
import pickle as pkl
import warnings
import time
import glob

from functools import wraps
from multiprocessing.pool import Pool as Pool

import numpy as np
from scipy.stats import moment


# Helper functions -------------------------------------------------------------------

def timeit(func):
    @wraps(func)   # keeps function name/docs intact
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)   # call the real function
        end = time.perf_counter()
        print(f"{func.__name__} runtime: {end - start:.3f} s")
        return result
    return wrapper


# CID related functions -------------------------------------------------------------------

def gen_conv_list(conv_list, output_suffix, save_path):
    """
    Convert conv_list to cube indices and save to file.
    """

    with open(os.path.join(save_path, f'cid_params{output_suffix}.pkl'), 'rb') as f:
                cid_params = pkl.load(f)

    first_frame_idx = cid_params['first_frame_idx']
    njumps_between_frames = cid_params['njumps_between_frames'] 
    time_subinterval = cid_params['time_subinterval']
    
    conv_list_cubes = np.zeros_like(conv_list, dtype=int)
    mask = (conv_list > first_frame_idx)
    conv_list_cubes[mask] = np.ceil((conv_list[mask] // njumps_between_frames - first_frame_idx) / time_subinterval).astype(int)

    np.save(os.path.join(save_path, f'conv_list_cubes{output_suffix}.npy'), conv_list_cubes)
    return

def extract_cid_results(info_dict, verbose=True):
    """
    Extracts CID results from multiple experiment directories and compiles them into a single dataset.
    """
    
    base_path = info_dict['base_path']
    save_path = info_dict['save_path']
    output_suffix = info_dict['output_suffix']
    
    LX = info_dict['LX']
    nexp = info_dict['nexp']
    act_exclude_list = info_dict['act_exclude_list']

    act_dir_list = glob.glob(os.path.join(base_path, '*'))
    act_list = [float(act_dir.split('_')[-1]) for act_dir in act_dir_list]
    # exclude activities in act_exclude_list
    act_dir_list = [act_dir for i, act_dir in enumerate(act_dir_list) if act_list[i] not in act_exclude_list]
    act_list = [act for act in act_list if act not in act_exclude_list]

    # extract parameter dict for first run
    exp_dirs = [x[0] for x in os.walk(act_dir_list[0])][1:]
    
    for exp_dir in exp_dirs:
        try:
            with open(os.path.join(exp_dir, f'cid_params{output_suffix}.pkl'), 'rb') as f:
                cid_params = pkl.load(f)
            with open(os.path.join(save_path, f'cid_params{output_suffix}.pkl'), 'wb') as f:
                pkl.dump(cid_params, f)

            break
        except:
            continue

    ncubes = cid_params['ncubes']
    npartitions = cid_params['npartitions']

    cid_arr = np.nan * np.zeros((ncubes, npartitions, len(act_list), nexp, 2))
    cid_shuffle_arr = np.nan * np.zeros((ncubes, npartitions, len(act_list), nexp, 2))
    cid_frac_arr = np.nan * np.zeros((ncubes, npartitions, len(act_list), nexp, 2))

    for i, act_dir in enumerate(act_dir_list):
        exp_dir_list =  [x[0] for x in os.walk(act_dir)][1:]
        for j, exp_dir in enumerate(exp_dir_list):
            try:
                data_npz = np.load(os.path.join(exp_dir, f'cid{output_suffix}.npz'), allow_pickle=True)
            except:
                if verbose: print(f'cid{output_suffix}.npz not found in {exp_dir}, skipping...')
                continue

            nframes = data_npz['cid'].shape[0]
            cid_arr[-nframes:, :, i, j, :] = data_npz['cid']
            cid_shuffle_arr[-nframes:, :, i, j, :] = data_npz['cid_shuffle']

    cid_frac_arr[:, :, :, :, 0] = cid_arr[:, :, :, :, 0] / cid_shuffle_arr[:, :, :, :, 0]
    cid_frac_arr[:, :, :, :, 1] = cid_frac_arr[:, :, :, :, 0] * np.sqrt( (cid_arr[:, :, :, :, 1]/cid_arr[:, :, :, :, 0])**2 + (cid_shuffle_arr[:, :, :, :, 1]/cid_shuffle_arr[:, :, :, :, 0])**2 )

    # save cid_arr, cid_shuffle_arr, cid_frac_arr
    np.savez_compressed(os.path.join(save_path, f'cid_data{output_suffix}.npz'), cid=cid_arr, cid_shuffle=cid_shuffle_arr, cid_frac=cid_frac_arr, act_list=act_list)
    if verbose: print(f'cid data saved to {os.path.join(save_path, f"cid_data{output_suffix}.npz")}')
    return


def get_allowed_time_intervals(system_size, nbits_max = 8):
    """
    Get allowed intervals for CID calculation based on system size and max bits.
    """
    allowed_intervals = []
    
    # system size must be divisible by 2^n

    if np.log2(system_size) % 1 != 0:
        warnings.warn("System size must be a power of 2 for exact interval calculation.")
        return
    if not type(nbits_max) == int:
        raise ValueError("nbits_max must be an integer.")

    for nbits in range(1, nbits_max + 1):
        interval_exp = 3 * nbits - 2 * np.log2(system_size)

        if interval_exp < 0:
            continue

        allowed_intervals.append({'time_interval': int(2 ** interval_exp), 'nbits': nbits})
    return allowed_intervals


def block_flatten(array, m, k):
    """
    Efficiently flatten a 2D array into m x k blocks traversed horizontally.
    
    Parameters:
        array (np.ndarray): Input 2D array of shape (M, N)
        m (int): Number of rows per block
        k (int): Number of columns per block
        
    Returns:
        np.ndarray: Flattened 1D array of blocks
    """
    M, N = array.shape

    # Check divisibility
    if M % m != 0:
        raise ValueError(f"Number of rows {M} is not divisible by block row size {m}.")
    if N % k != 0:
        raise ValueError(f"Number of columns {N} is not divisible by block column size {k}.")
    
    # Reshape array into blocks
    reshaped = array.reshape(M//m, m, N//k, k)
    # Transpose to bring blocks into row-major order: (block_row, block_col, m, k)
    reshaped = reshaped.transpose(0, 2, 1, 3)
    # Flatten all blocks
    return reshaped.reshape(-1)

def calc_time_av_ind_samples(data_arr, conv_list, unc_multiplier = 1, ddof = 1,):
    """
    data_arr must have shape (Nframes, Nsomething, Nact, Nexp)
    returns an array of shape (Nact, 2)
    """

    Nact = data_arr.shape[2]
    time_av = np.nan * np.zeros((Nact, 2))
    
    for i in range(Nact):
        ff_idx = conv_list[i]
        Nsamples = np.sum(~np.isnan(data_arr[ff_idx:,:,i,:])) 
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            time_av[i, 0]  = np.nanmean(data_arr[ff_idx:, :, i, :], axis = (0, 1, -1))
            time_av[i, 1] = np.nanstd(data_arr[ff_idx:, :, i, :], axis = (0, 1, -1), ddof = ddof) / np.sqrt(Nsamples / unc_multiplier)
    return time_av

def calc_time_avs_ind_samples(data_arr, conv_list, unc_multiplier = 1, ddof = 1,):
    """
    data_arr must have shape (Nframes, Nsomething, Nact, Nexp)
    returns time_av, var_av, var_per_exp
    """

    Nact = data_arr.shape[2]
    time_av = np.nan * np.zeros((Nact, 2))
    var_av = np.nan * np.zeros((Nact))
    var_per_exp = np.nan * np.zeros((Nact, data_arr.shape[-1]))
    
    for i in range(Nact):
        ff_idx = conv_list[i]
        Nsamples = np.sum(~np.isnan(data_arr[ff_idx:,:,i,:])) 
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            time_av[i, 0]  = np.nanmean(data_arr[ff_idx:, :, i, :], axis = (0, 1, -1))

            var_val = np.nanvar(data_arr[ff_idx:, :, i, :], axis = (0, 1, -1), ddof = ddof)
            var_av[i] = var_val
            time_av[i, 1] = np.sqrt(var_val) / np.sqrt(Nsamples / unc_multiplier)
            var_per_exp[i,:] = np.nanvar(data_arr[ff_idx:, :, i, :], axis = (0, 1), ddof = ddof)
    return time_av, var_av, var_per_exp

def calc_moments(data_arr, conv_list, center=None, norm_factor=None):
    """
    Calculate 1st–4th moments over the frame(s) and experiment dimensions
    for each act, allowing variable intermediate dimensions (e.g. partitions).

    Parameters
    ----------
    data_arr : np.ndarray
        Array of shape (Nframes, ..., Nact, Nexp)
    conv_list : list[int]
        List of starting frame indices for each act.
    center : float, optional
        Central value for moment calculation.
    norm_factor : float or None, optional
        Normalization factor for data_arr.

    Returns
    -------
    moments : np.ndarray
        Array of shape (4, Nact)
    """

    Nact = data_arr.shape[-2]
    moments = np.zeros((4, Nact))
    
    normalization = norm_factor if norm_factor is not None else 1.0
    defects = data_arr / normalization

    for i in range(Nact):
        data_slice = defects[conv_list[i]:, ..., i, :]
        
        # Compute 1st–4th moments
        for j in range(4):
            moments[j, i]  = moment(data_slice,moment=j + 1, \
                            axis=tuple(range(data_slice.ndim)), \
                            center=0 if j==0 else center, nan_policy='omit')
    return moments
