# Author:  Simon Guldager
# Date (latest update): 

### SETUP ------------------------------------------------------------------------------------

## Imports:
import os
import sys
import pickle
import warnings
import time
import argparse
import logging
import signal
import subprocess

import numpy as np

from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
rpy2_logger.setLevel(logging.ERROR)   # will display errors, but not warnings

from structure_factor.point_pattern import PointPattern
from structure_factor.spatial_windows import BoxWindow
import structure_factor.pair_correlation_function as pcf

sys.path.append('/groups/astro/kpr279/')
import massPy as mp


## SET WHETHER TO USE DEVELOPMENT MODE --------------------------------------------------------
development_mode = False
# if development_mode, use only few frames
if development_mode:
    num_frames = 15
## ---------------------------------------------------------------------------------------------


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

def get_pair_corr_from_defect_list(defect_list, ball_window, frame_idx_interval = None, Npoints_min = 30, method = "fv", \
                            kest_kwargs = {'rmax': 10, 'correction': 'best', 'var.approx': False},\
                                smoothing_kwargs = dict(method="b", spar=0.85, nknots=25), save=False, save_dir=None, save_suffix=''):
    """
    Calculate pair correlation function for the frames in frame_interval
    """

    # Get number of frames
    Nframes = len(defect_list) if frame_idx_interval is None else frame_idx_interval[1] - frame_idx_interval[0]
    frame_interval = [0, Nframes] if frame_idx_interval is None else frame_idx_interval

    arrays_initialized = False

    for i, frame in enumerate(range(frame_interval[0], frame_interval[1])):

        # Get defect array for frame
        defect_positions = get_defect_arr_from_frame(defect_list[frame])

        # Skip if less than Npoints_min defects
        if defect_positions is None:
            continue
        if len(defect_positions) < Npoints_min:
            continue

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

def gen_status_txt(message = '', log_path = None):
    """
    Generate txt file with message and no.
    """
    with open(log_path, 'w') as f:
        f.write(message)
    return

def kill_r_processes():
    try:
        job_id = os.getenv('SLURM_JOB_ID')  # Get the current Slurm job ID
        if job_id:
            # Get the PIDs of R processes within the Slurm job
            result = subprocess.run(['scontrol', 'listpids', job_id], stdout=subprocess.PIPE, text=True)
            if result.stdout:
                for pid in result.stdout.splitlines():
                    if 'R' in subprocess.run(['ps', '-p', pid, '-o', 'comm='], stdout=subprocess.PIPE, text=True).stdout.strip():
                        os.kill(int(pid), signal.SIGTERM)  # Gracefully terminate the R process
                        print(f"Killed R process with PID: {pid} for job: {job_id}")
    except Exception as e:
        print(f"Failed to kill R processes: {e}")
    return

def kill_r_processes():
    try:
        # Get current Slurm job ID
        job_id = os.getenv('SLURM_JOB_ID')
        if job_id:
            # Get PIDs of all processes associated with this Slurm job
            result = subprocess.run(['scontrol', 'listpids', job_id], stdout=subprocess.PIPE, text=True)
            if result.stdout:
                for pid in result.stdout.splitlines():
                    # Use 'ps' to get the full command for the process
                    try:
                        cmd_result = subprocess.run(['ps', '-p', pid, '-o', 'args='], stdout=subprocess.PIPE, text=True)
                        if 'R' in cmd_result.stdout:
                            os.kill(int(pid), signal.SIGTERM)  # Terminate the R process
                            print(f"Killed R process with PID: {pid} for job: {job_id}")
                    except Exception as e:
                        print(f"Failed to inspect or kill process {pid}: {e}")
    except Exception as e:
        print(f"Failed to kill R processes: {e}")

### MAIN ---------------------------------------------------------------------------------------



def main():

    ## Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str)
    parser.add_argument('--output_folder', type=str)
    parser.add_argument('--defect_list_folder', type=str, default=None)
    parser.add_argument('--mode', type=str, default='pcf')
    parser.add_argument('--sbatch_count', type=int, default=0)
    args = parser.parse_args()
    input_path = args.input_folder
    output_path = args.output_folder
    defect_list_folder = args.defect_list_folder
    sbatch_count = args.sbatch_count    

    # Get experiment no, and activity
    exp = int(output_path.split('_')[-1])
    act = float(output_path.split('_')[-3])

    job_id = os.getenv('SLURM_JOB_ID') + f'.{sbatch_count}'

    print("ID: ", job_id)

    ## Define parameters
    ar = mp.archive.loadarchive(input_path)
    LX, LY = ar.__dict__['LX'], ar.__dict__['LY']
    LXB, LYB = 25, 25
    window = BoxWindow(bounds=[[0+LXB, LX-LXB], [0+LYB, LY-LYB]])  
    

    frame_interval = [0, num_frames] if development_mode else None

    rmax = ((LX-LXB)/4 - 1) #60 if act > 0.024 else ((LX-LXB)/4 - 1)
    nknots = int(rmax) #min(300, int(rmax))
    method = 'fv'
    spar = 1.2
    smoothing_kwargs = dict(method="b", spar=spar, nknots=nknots)
    kest_kwargs = {'rmax': rmax, 'correction': 'good', 'nlarge': 3000, 'var.approx': False}

    t1 = time.perf_counter()

    if defect_list_folder is not None:
        defect_position_path = os.path.join(defect_list_folder, f'defect_positions.pkl')
        defect_list_params_path = os.path.join(defect_list_folder, f'parameters.json')
       
        if not os.path.exists(defect_list_params_path):
            params_path = os.path.join(input_path, f'parameters.json')
            # Copy parameters.json to defect_list_folder
            os.system(f'cp {params_path} {defect_list_folder}')
        with open(defect_position_path, 'rb') as f:
            top_defects = pickle.load(f)
    elif os.path.exists(os.path.join(output_path, f'defect_positions.pkl')):
        defect_position_path = os.path.join(output_path, f'defect_positions.pkl')
        with open(defect_position_path, 'rb') as f:
            top_defects = pickle.load(f)
    elif os.path.exists(os.path.join(input_path, f'defect_positions.pkl')):
        defect_position_path = os.path.join(input_path, f'defect_positions.pkl')
        with open(defect_position_path, 'rb') as f:
            top_defects = pickle.load(f)
    else:
        # Load data archive
        ar = mp.archive.loadarchive(input_path)
        LX, LY = ar.__dict__['LX'], ar.__dict__['LY']

        if not act == ar.__dict__['zeta']:
            err_msg = f"Activity list and zeta in archive do not match for experiment {exp}. Exiting..."
            print(err_msg)
            raise ValueError(err_msg)
        
        # Get defect list
        top_defects = get_defect_list(ar, LX, LY,)

        # save top_defects
        if not development_mode:
            with open(os.path.join(output_path, 'defect_positions.pkl'), 'wb') as f:
                pickle.dump(top_defects, f)
    
        print(f"Time to calculate defect positions for experiment {exp} and activity {act}: ", np.round(time.perf_counter()-t1,2), "s")

    t1 = time.perf_counter()
    # save pcf params
    pcf_params_path = os.path.join(output_path, 'pcf_params.pkl')
    pcf_params = {'L': LX, 'LXB': LXB, 'act': act, 'kest_kwargs': kest_kwargs, 'smoothing_kwargs': smoothing_kwargs}
    with open(pcf_params_path, 'wb') as f:
        pickle.dump(pcf_params, f)

    get_pair_corr_from_defect_list(top_defects, window, frame_idx_interval = frame_interval, method = method, \
                    kest_kwargs = kest_kwargs, smoothing_kwargs = smoothing_kwargs, save=True, save_dir=output_path,)

    print(f"Time to calculate pcf for experiment {exp} and activity {act}: ", np.round(time.perf_counter()-t1,2), "s")
    msg = f"PCF analysis completed for experiment {exp} and activity {act}."
    gen_status_txt(msg, os.path.join(output_path, 'pcf_analysis_completed.txt'))

if __name__ == '__main__':
    main()
