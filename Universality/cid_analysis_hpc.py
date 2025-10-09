
# Author:  Simon Guldager
# Date (latest update): 

### SETUP ------------------------------------------------------------------------------------


import sys
import os
import pickle
import warnings
import time
import argparse
import logging

#from functools import wraps, partial
#from pathlib import Path
#from multiprocessing import cpu_count
from multiprocessing.pool import Pool as Pool

import numpy as np
import matplotlib.pyplot as plt


sys.path.append('/groups/astro/kpr279/')
#sys.path.append('/groups/astro/kpr279/ComputableInformationDensity')
import massPy as mp


from ComputableInformationDensity.cid import interlaced_time, cid2d
from ComputableInformationDensity.computable_information_density import cid

#sys.path.insert(0,'/groups/astro/robinboe/mass_analysis')
#sys.path.insert(0,'/groups/astro/robinboe/mass_analysis/computable-information-density')
#from joblib import Parallel, delayed

development_mode = False
if development_mode:
    num_frames = 2


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


def gen_status_txt(message = '', log_path = None):
    """
    Generate txt file with message and no.
    """
    with open(log_path, 'w') as f:
        f.write(message)
    return

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    

### MAIN ---------------------------------------------------------------------------------------



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str)
    parser.add_argument('--output_folder', type=str)
    parser.add_argument('--defect_list_folder', type=str, default=None)
    #parser.add_argument('--periodic', type=str2bool, default=False) 
    #parser.add_argument('--rmax_fraction', type=float, default=0.1)
    args = parser.parse_args()

    archive_path = args.input_folder
    output_path = args.output_folder
    defect_list_folder = args.defect_list_folder

    defect_dir = output_path if defect_list_folder is None else defect_list_folder
    defect_position_path = os.path.join(defect_dir, f'defect_positions.pkl')

    # --------------------------------------------

    # Load data
    t1 = time.perf_counter()
    ar = mp.archive.loadarchive(archive_path)
    LX, LY = ar.__dict__['LX'], ar.__dict__['LY']
    exp = int(output_path.split('_')[-1])
    act = float(output_path.split('_')[-3])

    msg = f"\nAnalyzing experiment {exp} and activity {act}"
    print(msg)

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
    # --------------------------------------------
    # Set params

    compression_factor = 4
    nshuffle = 4
    nframes = 16 if not development_mode else num_frames

    observation_window_bounds = [(0, int(LX / compression_factor)), (0, int(LY / compression_factor))]
    lx_window = observation_window_bounds[0][1] - observation_window_bounds[0][0]
    ly_window = observation_window_bounds[1][1] - observation_window_bounds[1][0]

    nbits_frame = int(np.log2(lx_window))

    if not lx_window == ly_window:
        raise ValueError("Observation window must be square.")

    allowed_intervals_list = get_allowed_time_intervals(system_size = lx_window, nbits_max=8)

    # check that nframes is in allowed intervals
    if nframes not in [ai['time_interval'] for ai in allowed_intervals_list]:
        raise ValueError(f"nframes {nframes} is not in allowed intervals {allowed_intervals_list}")
    else:
        # get nbits for nframes
        nbits = [ai['nbits'] for ai in allowed_intervals_list if ai['time_interval'] == nframes][0]

    print(f"Using nbits = {nbits} (size {1 << nbits}) for nframes = {nframes} and window size {lx_window}x{ly_window}")

    # --------------------------------------------
    # Create defect grid and compute CID

    defect_grid = np.zeros((nframes, lx_window, ly_window), dtype=int)
    defect_count = []

    for i, defect in enumerate(top_defects[-nframes:]):
        def_arr = get_defect_arr_from_frame(defect).astype(int)

        def_arr_xmask = (observation_window_bounds[0][0] < def_arr[:,0]) & (def_arr[:,0] < observation_window_bounds[0][1])
        def_arr_ymask = (observation_window_bounds[0][0] < def_arr[:,1]) & (def_arr[:,1] < observation_window_bounds[0][1])
        def_arr = def_arr[def_arr_xmask & def_arr_ymask]

        defect_grid[i, def_arr[:,0], def_arr[:,1]] = 1
        defect_count.append(defect_grid[i,:,:].sum())

    print(f"Defect positions loaded and defect grid created. Time: ", np.round(time.perf_counter()-t2,2), "s")
    t3 = time.perf_counter()

    # instantiate CID object:
    CID = interlaced_time(nbits=nbits, nshuff=nshuffle)
    cid_, cid_shuff = CID(defect_grid)

    print(f"CID calculated. Time: ", np.round(time.perf_counter()-t3,2), "s")

    res_cid = {
        'zeta' : ar.zeta,
        'density' : defect_count,
        'cid' : cid_,
        'cid_shuffle' : cid_shuff,
        'lambda' : 1. - cid_/cid_shuff
    }

    with open(os.path.join(output_path, f'cid.pkl'), 'wb') as f:
        pickle.dump(res_cid, f)

    gen_status_txt(msg, os.path.join(output_path, 'cid_analysis_completed.txt'))


if __name__ == '__main__':
    main()


