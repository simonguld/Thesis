# Author:  Simon Guldager
# Date (latest update): 

### SETUP ------------------------------------------------------------------------------------
import os
import sys
import pickle
import time
import argparse

import numpy as np
from scipy.stats import binned_statistic

sys.path.append('/groups/astro/kpr279/')
import massPy as mp

development_mode = False
num_frames = 5 if development_mode else None
njump_between_frames = 1 if development_mode else 5

### FUNCTIONS ----------------------------------------------------------------------------------

def get_defect_list(archive, LX, LY, idx_first_frame=0,):
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

    Nframes = archive.__dict__['num_frames']

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
    return top_defects

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

def get_defect_arr_from_frame(defect_dict, return_charge = False):
    """
    Convert dictionary of defects to array of defect positions
    Parameters:
    -----------
    defect_dict : dict
        Dictionary of defects positions and charges
    return_charge : bool
        If True, return defect charges as well

    Returns:
    --------
    defect_positions : np.ndarray
        Array of defect positions
    defect_charges : np.ndarray
    """

    Ndefects = len(defect_dict)
    if Ndefects == 0:
        return None
    
    defect_positions = np.empty([Ndefects, 3 if return_charge else 2])

    for i, defect in enumerate(defect_dict):
        defect_positions[i] = *defect['pos'], defect['charge'] if return_charge else defect['pos']
    return defect_positions

def calc_autocorr(nx, ny=None, shift=0, abs_val=False, normalize=False):

    n = nx.shape[0]
    if not n % 2 == 0:
        raise ValueError("The length of the array must be even.")
    
    # initialize arrays
    corr_arr = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if ny is not None:
                dot_arr = nx * np.roll(np.roll(nx, i, axis=0), j, axis=1) + ny * np.roll(np.roll(ny, i, axis=0), j, axis=1)
            else:
                dot_arr = nx * np.roll(np.roll(nx, i, axis=0), j, axis=1)
            if abs_val:
                dot_arr = np.abs(dot_arr)
            dot_arr -= shift
            corr_arr[i, j] = dot_arr.sum()
    
    if normalize:
        corr_arr /= corr_arr[0, 0]
    return corr_arr

def rdf2d_mod(Fx, Fy=None, corr_func = None, corr_func_kwargs = {}, \
               origin=False, step=1.,):
    """ Radial distribution function (2D) \n
    Args:
        F: two-dimensional array containing samples of a scalar function.
        step: disctrization of radial displacement/seperation (default = 1.0).
    Returns:
        Radial distribution function and r-values.
    """
    if corr_func is None:
        C = mp.base_modules.correlation.autocov(Fx) if Fy is None \
            else mp.base_modules.correlation.autocov2(Fx, Fy)
    else:
        C = corr_func(Fx, Fy, **corr_func_kwargs)

    C = np.divide(C, C[0,0], where=C[0,0]!=0)   # normalize
    L = C.shape[0]  # linear system size
    
    r_max = L//2
    r_bins = np.arange(0.5, r_max, step)        # list of bin edges
    r_vals = .5 * (r_bins[1:] + r_bins[:-1])    # list of bin midpoints
    
    # two-dimensional array containing the radial distance 
    # w.r.t the top left corner on a periodic square domain.
    r = np.arange(0, L, 1)
    r = np.minimum(r%L, -r%L)
    r_nrm = np.abs(r[:,None] + 1J*r[None,:])
    
    # bin the autocovariance in radial-space.
    rdf, _, _ = binned_statistic(r_nrm.flatten(), C.flatten(), 'mean', r_bins)
    
    if origin: # insert rdf(r=0) = 1.
        rdf = np.insert(rdf, 0, 1.)
        r_vals = np.insert(r_vals, 0, .0)
    
    return rdf, r_vals

def get_radial_director_correlations(archive, corr_func = None, corr_func_kwargs = {}, \
               origin=False, step=1., save_path=None):
    
    Nframes = archive.num_frames if not development_mode else num_frames
    L = archive.LX

    r_max = L//2
    r_bins = np.arange(0.5, r_max, step)        # list of bin edges
    r_vals = .5 * (r_bins[1:] + r_bins[:-1])    # list of bin midpoints
    r_vals = np.insert(r_vals, 0, .0) if origin else r_vals
    rdf_arr = np.nan * np.zeros((Nframes, len(r_vals))) # +1 for r=0

    for i in range(0, Nframes, njump_between_frames):
        frame = archive._read_frame(i)
        Qxx_dat = frame.QQxx.reshape(L, L)
        Qyx_dat = frame.QQyx.reshape(L, L)
        _, nx, ny = mp.nematic.nematicPy.get_director(Qxx_dat, Qyx_dat)

        rdf, _ = rdf2d_mod(nx, ny, corr_func = corr_func, corr_func_kwargs = corr_func_kwargs, \
                                 origin=origin, step=step)
        rdf_arr[i] = rdf  
    
    if save_path is not None:
        np.save(os.path.join(save_path, 'rdf_arr.npy'), rdf_arr)
        np.save(os.path.join(save_path, 'rdf_rad_arr.npy'), r_vals)  
    return rdf_arr, r_vals

def gen_status_txt(message = '', log_path = None):
    """
    Generate txt file with message and no.
    """
    with open(log_path, 'w') as f:
        f.write(message)
    return


### MAIN ---------------------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str)
    parser.add_argument('--output_folder', type=str)
    args = parser.parse_args()

    input_folder = args.input_folder
    output_path = args.output_folder
    save_path = output_path

    # Load data archive
    ar = mp.archive.loadarchive(input_folder)

    # Set correlation parameters
    abs_val = True # when True, the degeneracy in the director is resolved
    auto_corr_kwargs = {'shift': 2/np.pi if abs_val is True else 9/(7*np.pi),
                        'abs_val': abs_val, 
                        'normalize': True}
 
    t1 = time.perf_counter()
    msg = f"\nAnalyzing data from input_folder: {input_folder}"
    print(msg)

    _, _ = get_radial_director_correlations(ar, corr_func = calc_autocorr, \
                                            corr_func_kwargs = auto_corr_kwargs, 
                                            origin=True, step=1., save_path=save_path)
 
    msg = f"Time to calculate director correlation: {np.round(time.perf_counter()-t1,2)} s"
    print(msg)
    gen_status_txt(msg, os.path.join(output_path, 'dir_correlation_completed.txt'))

if __name__ == '__main__':
    main()
