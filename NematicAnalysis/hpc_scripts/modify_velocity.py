# Author:  Simon Guldager
# Date (latest update): 

### SETUP ------------------------------------------------------------------------------------

## Imports:
import os
import json
import shutil
import argparse

from multiprocessing.pool import Pool as Pool
from glob import glob
from time import perf_counter

import numpy as np

### FUNCTIONS ----------------------------------------------------------------------------------

def sort_files(files):
    return sorted(files, key = lambda x: int(x.split('frame')[-1].split('.')[0]))

def gen_status_txt(message = '', log_path = None):
    """
    Generate txt file with message and no.
    """
    with open(log_path, 'w') as f:
        f.write(message)
    return

def modify_velocity(file, output_dir, AA_LBF = 0.5):

    outpath = os.path.join(output_dir, os.path.basename(file))
    npz = dict(np.load(file, allow_pickle=True))

    npz['vx'] += AA_LBF * npz['FFx'] / npz['density']
    npz['vy'] += AA_LBF * npz['FFy'] / npz['density']
    np.savez_compressed(outpath, **npz)

### MAIN ---------------------------------------------------------------------------------------


if __name__ == '__main__':

    local = False

    call_cluster_cmd = False if local else True
    ncores = os.cpu_count() if local else int(os.environ['SRUN_CPUS_PER_TASK'])

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str)
    parser.add_argument('--output_folder', type=str)
    parser.add_argument('--num_frames', type=int, default=0)
    #parser.add_argument('--delete_if_successful', type=int, default=0)
    parser.add_argument('--test_mode', type=int, default=0)
    args = parser.parse_args()

    archive_path = args.input_folder
    output_dir = args.output_folder
    #delete_original_archive = bool(args.delete_if_successful)
    test_mode = bool(args.test_mode)
    num_frames_max = args.num_frames if args.num_frames > 0 else 2000
    
    files = sort_files(glob(os.path.join(archive_path, 'frame*')))
    Nfiles = min(10, len(files)) if test_mode else min(num_frames_max, len(files))
    files = files[-Nfiles:] 
    first_frame_num = int(files[0].split('frame')[-1].split('.')[0])

    exp = int(output_dir.split('_')[-1])
    act = float(output_dir.split('_')[-3])

    ntasks = min(Nfiles, ncores)
    csize = 10

    params_path = os.path.join(archive_path, 'parameters.json')
    shutil.copy(params_path, output_dir)

    with open(params_path, 'r') as f:
        simulation_params = json.load(f)['data']

    isGuo = True
    if 'isGuo' in simulation_params.keys():
        isGuo = simulation_params['isGuo']['value']
    # set AA_LBF term according to isGuo
    AA_LBF = .5 if isGuo else simulation_params['tau']['value'] 

    t_start = perf_counter()
    with Pool(ntasks) as p:
            p.starmap(modify_velocity, [(file, output_dir, AA_LBF) for file in files], chunksize=csize)
    print(f'Time to convert all files for exp. {exp} and activity {act} using {ntasks} cpus: {perf_counter() - t_start:.2f} s')
