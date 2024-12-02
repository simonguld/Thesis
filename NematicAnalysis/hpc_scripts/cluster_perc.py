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

def get_clustering_signed(top_defect_list, method, rmax_list, method_kwargs, save = False, save_path = None):
    """
    
    Parameters:
    -----------
    Returns:
    --------
    """
  
    Nframes = len(top_defect_list)
    Nwindows = len(rmax_list)

    cl_arr = np.nan * np.ones([Nframes, Nwindows, 3])
   
    for frame, defects in enumerate(top_defect_list):
        # Get defect array for frame
        defect_arr = get_defect_arr_from_frame(defects, return_charge = True)
        defect_positions = defect_arr[:, :-1]
        defect_charges = defect_arr[:, -1] 

        if defect_positions is None:
            continue

        for i, rmax in enumerate(rmax_list):
            cst = method(distance_threshold = rmax, **method_kwargs)
            labels = cst.fit_predict(defect_positions)

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
        save_path = save_path if save_path is not None else 'cl_arr.npy'
        np.save(save_path, cl_arr)
    return cl_arr

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
    save_path = os.path.join(output_path, f'cl_arr.npy')
    defect_list_folder = args.defect_list_folder
    
    if defect_list_folder is not None:
        defect_position_path = os.path.join(defect_list_folder, f'defect_positions.pkl')
    else:
        defect_position_path = os.path.join(output_path, f'defect_positions.pkl')

    exp = int(output_path.split('_')[-1])
    act = float(output_path.split('_')[-3])

    # Load data archive
    ar = mp.archive.loadarchive(input_folder)
    LX, LY = ar.__dict__['LX'], ar.__dict__['LY']

    t1 = time.perf_counter()
    msg = f"\nAnalyzing experiment {exp} and activity {act}"
    print(msg)

    # Get defect list if provided
    if os.path.exists(defect_position_path):
        with open(defect_position_path, 'rb') as f:
            top_defects = pickle.load(f)
    else:
        if not act == ar.__dict__['zeta']:
            err_msg = f"Activity list and zeta in archive do not match for experiment {exp}. Exiting..."
            print(err_msg)
            raise ValueError(err_msg)
        
        # Get defect list
        top_defects = get_defect_list(ar, LX, LY,)

        # save top_defects
        with open(os.path.join(output_path, 'defect_positions.pkl'), 'wb') as f:
            pickle.dump(top_defects, f)
 
        print("Time to calculate defect positions: ", np.round(time.perf_counter()-t1,2), "s")

    rmax_list = np.arange(10, 500)
    method_kwargs = dict(n_clusters=None, linkage = 'single',)

    t2 = time.perf_counter()
    _ = get_clustering_signed(top_defects[:num_frames], AgglomerativeClustering, 
                            rmax_list=rmax_list, method_kwargs=method_kwargs, 
                            save = True, save_path=save_path)

    msg = f"Time to do clustering/percolation: {np.round(time.perf_counter()-t2,2)} s"
    print(msg)

    gen_status_txt(msg, os.path.join(output_path, 'percolation_completed.txt'))

if __name__ == '__main__':
    main()
