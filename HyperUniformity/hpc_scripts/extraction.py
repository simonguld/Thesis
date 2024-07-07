# Author:  Simon Guldager
# Date (latest update): 

### SETUP ------------------------------------------------------------------------------------

## Imports:

import os
import sys
import pickle
import time
import argparse

import numpy as np

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
    parser.add_argument('--mode', type=str)
    args = parser.parse_args()

    archive_path = args.input_folder
    output_path = args.output_folder

    exp = int(output_path.split('_')[-1])
    act = float(output_path.split('_')[-3])

    t1 = time.time()
    msg = f"\nAnalyzing experiment {exp} and activity {act}"
    print(msg)

    # Load data archive
    ar = mp.archive.loadarchive(archive_path)
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

    t2 = time.time()
    msg = f"Time to get and save defect list: {(t2 - t1)/60:.2f} minutes"
    print(msg)

    gen_status_txt(msg, os.path.join(output_path, 'extraction_analysis_completed.txt'))


if __name__ == '__main__':
    main()
