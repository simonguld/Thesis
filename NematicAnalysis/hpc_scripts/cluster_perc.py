# Author:  Simon Guldager
# Date (latest update): 

### SETUP ------------------------------------------------------------------------------------
import os
import sys
import pickle
import time
import argparse

import numpy as np
from sklearn.cluster import AgglomerativeClustering

sys.path.append('/groups/astro/kpr279/')
import massPy as mp


development_mode = False
num_frames = 5 if development_mode else None

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

def calc_distance_matrix(points, L=None, periodic=False):
    """
    Calculate the distance matrix between points in an N-dimensional space.

    Parameters:
    -----------
    points : np.ndarray
        Array of points in N-dimensional space. Shape (N, D) where N is the number of points and D is the dimensionality.
    L : float
        Length of the square domain in each dimension. If periodic is True, this is required.
    periodic : bool
        If True, use periodic boundary conditions.

    Returns:
    --------
    distance_matrix : np.ndarray
        Array of distances between points. Shape (N, N).
    """ 

    displacement = points[:, None] - points[None, :]
    if periodic:
        min_func_vectorized = lambda x: np.minimum(x, L - x)
        dr = np.apply_along_axis(min_func_vectorized, axis = -1, arr = np.abs(displacement))
    else:
        dr = displacement
    return np.sqrt(np.sum(dr**2, axis=-1))

def calc_mean_nearest_neighbor_dist(distance_matrix,):
    """
    Calculate the mean nearest neighbor distance for each point in a set of points.

    Parameters:
    -----------
    distance_matrix : np.ndarray
        Array of distances between points. Shape (N, N).
    
    Returns:
    --------
    mean_nearest_neighbour_dist : float
        Mean nearest neighbor distance.
    std_nearest_neighbour_dist : float
        Standard deviation of the mean nearest neighbor distance.
    """

    dist  = distance_matrix.astype('float64')
    np.fill_diagonal(dist, np.inf)
    nearest_neighbours_all = np.min(dist, axis = 1)
    return np.mean(nearest_neighbours_all), np.std(nearest_neighbours_all) / np.sqrt(len(nearest_neighbours_all))

def get_clustering_signed(top_defect_list, method, L, rmax_list, method_kwargs, periodic = False, \
                          save = False, save_dir = None):
    """
    
    Parameters:
    -----------
    Returns:
    --------
    """
  
    Nframes = len(top_defect_list)
    Nwindows = len(rmax_list)

    cl_arr = np.nan * np.ones([Nframes, Nwindows, 3])
    nearest_neighbours_arr = np.nan * np.ones([Nframes, 2])
   
    for frame, defects in enumerate(top_defect_list):
        # Get defect array for frame
        defect_arr = get_defect_arr_from_frame(defects, return_charge = True)

        if defect_arr is None:
            continue

        defect_positions = defect_arr[:, :-1]
        defect_charges = defect_arr[:, -1] 

        distance_matrix = calc_distance_matrix(defect_positions, L = L, periodic = periodic)
        nearest_neighbours_arr[frame] = calc_mean_nearest_neighbor_dist(distance_matrix)

        for i, rmax in enumerate(rmax_list):
            cst = method(distance_threshold = rmax, **method_kwargs, metric='precomputed')
            labels = cst.fit_predict(distance_matrix)

            Ncl = np.max(labels) + 1
            Qc_arr = np.zeros(Ncl)

            for N in range(Ncl):
                mask = (labels == N)
                Qc = np.sum(defect_charges[mask])
                Qc_arr[N] = Qc
 
            all_neutral = float(np.all(Qc_arr == 0))
            Qcl = np.sum(np.abs(Qc_arr)) / Ncl
            cl_arr[frame, i] = [Ncl, Qcl, all_neutral]

            if Ncl == 1:
                break

    if save:
        # save labels list
        np.save(os.path.join(save_dir, 'cl_arr.npy') if save_dir is not None else 'cl_arr.npy', cl_arr)
        np.save(os.path.join(save_dir, 'nn_arr.npy') if save_dir is not None else 'nn_arr.npy', nearest_neighbours_arr)
    return cl_arr, nearest_neighbours_arr

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
    parser.add_argument('--defect_list_folder', type=str, default=None)
    args = parser.parse_args()

    input_folder = args.input_folder
    output_path = args.output_folder
    save_path = output_path
    defect_list_folder = args.defect_list_folder
    
    if defect_list_folder is not None:
        defect_position_path = os.path.join(defect_list_folder, f'defect_positions.pkl')
    else:
        defect_position_path = os.path.join(output_path, f'defect_positions.pkl')


    # Load data archive
    ar = mp.archive.loadarchive(input_folder)
    act = ar.__dict__['zeta']
    LX, LY = ar.__dict__['LX'], ar.__dict__['LY']

    # Set clustering parameters
    rmax_list = np.arange(1, LX / 2)
    method_kwargs = dict(n_clusters=None, linkage = 'single',)
    periodic = True

    t1 = time.perf_counter()
    msg = f"\nAnalyzing data from input_folder: {input_folder}"
    print(msg)

    # Get defect list if provided
    if os.path.exists(defect_position_path):
        with open(defect_position_path, 'rb') as f:
            top_defects = pickle.load(f)
    else:    
        # Get defect list
        top_defects = get_defect_list(ar, LX, LY,)

        # save top_defects
        with open(os.path.join(output_path, 'defect_positions.pkl'), 'wb') as f:
            pickle.dump(top_defects, f)
 
        print("Time to calculate defect positions: ", np.round(time.perf_counter()-t1,2), "s")

    
    t2 = time.perf_counter()
    _, _ = get_clustering_signed(top_defects[:num_frames], AgglomerativeClustering, 
                            L = LX, periodic = periodic,
                            rmax_list=rmax_list, method_kwargs=method_kwargs, 
                            save = True, save_dir=save_path)

    msg = f"Time to do clustering/percolation: {np.round(time.perf_counter()-t2,2)} s"
    print(msg)

    gen_status_txt(msg, os.path.join(output_path, 'percolation_completed.txt'))

if __name__ == '__main__':
    main()
