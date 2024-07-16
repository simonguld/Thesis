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

def gen_status_txt(message = '', log_path = None):
    """
    Generate txt file with message and no.
    """
    with open(log_path, 'w') as f:
        f.write(message)
    return


### MAIN ---------------------------------------------------------------------------------------


def main():

    Rmax = 33
    method_kwargs = dict(n_clusters=None, linkage = 'single', distance_threshold=Rmax)

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str)
    parser.add_argument('--output_folder', type=str)
    args = parser.parse_args()

    input_folder = args.input_folder
    output_path = args.output_folder
    save_path = os.path.join(output_path, f'labels_rm{Rmax}.pkl')
    defect_position_path = os.path.join(output_path, f'defect_positions.pkl')

    exp = int(output_path.split('_')[-1])
    act = float(output_path.split('_')[-3])

    t1 = time.perf_counter()
    msg = f"\nAnalyzing experiment {exp} and activity {act}"
    print(msg)

    if os.path.exists(defect_position_path):
        with open(defect_position_path, 'rb') as f:
            top_defects = pickle.load(f)
    else:
        # Load data archive
        ar = mp.archive.loadarchive(input_folder)
        LX, LY = ar.__dict__['LX'], ar.__dict__['LY']

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

    t2 = time.perf_counter()
    _ = get_clustering(top_defects, AgglomerativeClustering, method_kwargs, save = True, save_path=save_path)

    msg = f"Time to cluster defects: {np.round(time.perf_counter()-t2,2)} s"
    print(msg)

    gen_status_txt(msg, os.path.join(output_path, 'clustering_completed.txt'))

if __name__ == '__main__':
    main()
