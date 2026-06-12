# Author:  Simon Guldager
# Date (latest update): 

### SETUP ------------------------------------------------------------------------------------

import io
import sys
import os
import pickle
import warnings
import time
import argparse
from multiprocessing.pool import Pool as Pool

import numpy as np

sys.path.append('/groups/astro/kpr279/')
sys.path.append('/groups/astro/kpr279/ComputableInformationDensity/')

import massPy as mp
#import ComputableInformationDensity 

from ComputableInformationDensity.cid import CID, CID_old
from ComputableInformationDensity.computable_information_density import cid

#plt.style.use('sg_article')
#plt.rcParams.update({"text.usetex": True,})

# Define number of frames to use in debug mode and max frames
num_frames_debug = 64

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

def get_defect_list(archive, LX, LY, idx_first_frame=0, debug=False, verbose=False):
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
    if not debug:
        Nframes = archive.__dict__['num_frames']
    else:  
        Nframes = num_frames_debug

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

## considerations for implementation

## allowing for window sizes which are not powers of two, restrict 
# system accordingly. does current imp work for thaT?
## incorp multi-value through field values and z-dim
# incorp for scalar fields

# we are saving ndefects needlessly now. just one per activity.
# pass verbose as argument so we can silence in batch jobs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str)
    parser.add_argument('--output_folder', type=str)
    parser.add_argument('--defect_list_folder', type=str, default=None)
    parser.add_argument('--nframes_per_cube', type=int)
    parser.add_argument('--window_size', type=int)
    parser.add_argument('--cg_box_length', type=int, default=1)
    parser.add_argument('--nshuffle', type=int, default=16)
    parser.add_argument('--nframes_max', type=int, default=400)
    parser.add_argument('--system_size', type=int, default=None)
    parser.add_argument('--activity', type=float, default=None)
    parser.add_argument('--experiment', type=int, default=None)
    parser.add_argument('--debug', type=str2bool, default=False)
    args = parser.parse_args()


    user = os.environ.get("USERNAME")
    home_dir = os.path.expanduser("~")
    projects_dir = os.environ['Projects']
    project_dir = os.path.join(projects_dir, 'MonolayerTransition',)

    working_dir = os.path.join(project_dir, 'DataExtraction')
    data_dir = os.path.join(project_dir, 'data',)
    data_folders = os.listdir(data_dir)

    archive_path = args.input_folder
    output_path = args.output_folder
    defect_list_folder = args.defect_list_folder

    debug = args.debug
    system_size = args.system_size
    act = args.activity
    exp = args.experiment

    exp_verbose = None if exp is None else 1

    defect_dir = output_path if defect_list_folder is None else defect_list_folder
    defect_position_path = os.path.join(defect_dir, f'defect_positions.pkl')

    if not debug:
        # Redirect stdout to a buffer
        buffer = io.StringIO()
        sys.stdout = buffer

    # --------------------------------------------
    # Load data
    t1 = time.perf_counter()
    if system_size is None:
        ar = mp.archive.loadarchive(archive_path)
        LX, LY = ar.__dict__['LX'], ar.__dict__['LY']
    else:
        LX, LY = system_size, system_size

    if exp==exp_verbose: print(f"\nAnalyzing experiment {exp} and activity {act}")

    if os.path.exists(defect_position_path):
        with open(defect_position_path, 'rb') as f:
            top_defects = pickle.load(f)
    else:
        if system_size is not None:
            print("Error: defect positions not found, but system size specified. Please provide defect list folder.")
            return
        
        # Get defect list
        top_defects = get_defect_list(ar, LX, LY, debug=debug)
        # save top_defects
        with open(defect_position_path, 'wb') as f:
            pickle.dump(top_defects, f)
        print(f"Time to calculate defect positions for experiment {exp} and activity {act}: ", np.round(time.perf_counter()-t1,2), "s")

    # --------------------------------------------
    
    # Set params
    num_frames_max = args.nframes_max
    nframes_per_cube = args.nframes_per_cube
    window_size = args.window_size
    coarse_graining_box_length = args.cg_box_length
    sequential = True if nframes_per_cube == 1 else False

    defect_dtype = int
    dtype = np.uint8
    cid_mode = 'lz77'
    verbose = True
    compare_hilbert = False

    overlap = 0 # overlap between cubes in time direction
    njumps_between_frames = 1 # use every nth frame to reduce temporal correlation
    top_defects = top_defects[::njumps_between_frames]

    Nframes = len(top_defects)
    nframes_to_analyze = min(num_frames_max, Nframes) if not debug else num_frames_debug
    
    LX_cg = LX // coarse_graining_box_length
    compression_factor = LX // window_size # number of partitions along one dimension
    npartitions = compression_factor**2
    lx_window_cg = window_size // coarse_graining_box_length

    hyperwindow_shape = (nframes_per_cube, window_size, window_size) if not sequential else (window_size, window_size)
    hyperwindow_shape_cg = (nframes_per_cube, lx_window_cg, lx_window_cg) if not sequential else (lx_window_cg, lx_window_cg)
    hyperwindow_dim = len(hyperwindow_shape_cg)
    nzorder_permutations = int(np.math.factorial(hyperwindow_dim))
    nshuffle = max(args.nshuffle, nzorder_permutations)

    save_suffix = f'_nx{window_size}nt{nframes_per_cube}cg{coarse_graining_box_length}'
    
    time_subinterval = nframes_per_cube - overlap
    ncubes = 1 + int(((nframes_to_analyze - nframes_per_cube) / time_subinterval))
    first_frame_idx = (Nframes - nframes_to_analyze) + (nframes_to_analyze - nframes_per_cube - ((ncubes - 1) * time_subinterval))
 
    param_dict = {
        'cid_mode' : cid_mode,
        'dtype' : dtype,
        'nshuffle' : nshuffle,
        'hyperwindow_shape' : hyperwindow_shape,
        'hyperwindow_shape_cg' : hyperwindow_shape_cg,
        'compression_factor' : compression_factor,
        'coarse_graining_box_length' : coarse_graining_box_length,
        'window_size' : window_size,
        'npartitions' : npartitions,
        'num_frames' : Nframes,
        'njumps_between_frames' : njumps_between_frames,
        'nframes_to_analyze' : nframes_to_analyze,
        'ncubes' : ncubes,
        'nframes_per_cube' : nframes_per_cube,
        'overlap' : overlap,
        'time_subinterval' : time_subinterval,
        'first_frame_idx' : first_frame_idx,     
    }

    if verbose and exp==exp_verbose:
        print(f"compression_factor, npartitions, nshuffle: {compression_factor}, {npartitions}, {nshuffle}")
        print(f"Hyperwindow shape: {hyperwindow_shape}.")
        print(f"Hyperwindow shape (coarse-grained): {hyperwindow_shape_cg}.")
        print(f"Ncubes, Nframes_per, overlap, time subinterval: {ncubes}, {nframes_per_cube}, {overlap}, {time_subinterval}\n")

    if not debug:
        with open(os.path.join(output_path, f'cid_params{save_suffix}.pkl'), 'wb') as f:
                pickle.dump(param_dict, f)

    # --------------------------------------------
    # Create defect grid and compute CID

    defect_grid = np.zeros((Nframes - first_frame_idx, LX_cg, LX_cg), dtype=dtype)
    defect_count_full = np.zeros(Nframes - first_frame_idx, dtype=int)
    defect_count = np.zeros_like(defect_count_full)


    for i, defect in enumerate(top_defects[first_frame_idx:]):
        if isinstance(defect, np.ndarray):
            def_arr = defect
        else:
            def_arr = get_defect_arr_from_frame(defect) #.astype(int)
        if def_arr is None:
            continue

        if defect_dtype == int:
            def_arr = np.round(def_arr)

        def_arr = def_arr.astype(defect_dtype) % LX
        defect_count_full[i] = len(def_arr)

        # Coarse-grain: map positions to coarse grid indices
        coarse_x = (def_arr[:, 0] // coarse_graining_box_length).astype(int)
        coarse_y = (def_arr[:, 1] // coarse_graining_box_length).astype(int)

        # Set coarse cells to 1 if any defect is inside
        defect_grid[i, coarse_x, coarse_y] = 1
        defect_count[i] = defect_grid[i,:,:].sum()

    if verbose and exp==exp_verbose: 
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            print(f"Average number of defects per frame before/after coarse graining: {np.mean(defect_count_full):.2f}, {np.mean(defect_count):.2f}")
    if debug:
        pass
    t3 = time.perf_counter()

    CID_obj = CID(dim=hyperwindow_dim, 
                  data_shape=hyperwindow_shape_cg, 
                  nshuff=nshuffle, 
                  mode=cid_mode, 
                  ordering='zcurve', 
                  verbose=False)
    if compare_hilbert:
        CID_hil = CID(dim=hyperwindow_dim, 
                    nbits=(nframes_per_cube - 1).bit_length(), 
                    nshuff=nshuffle, 
                    mode=cid_mode, 
                    ordering='hilbert', 
                    verbose=False)
        CID_hil_old = CID_old(dim=hyperwindow_dim, 
                    nbits=(nframes_per_cube - 1).bit_length(), 
                    nshuff=nshuffle, 
                    mode=cid_mode, 
                    verbose=False)
        cid_arr_hil = np.nan * np.ones((ncubes, npartitions, 2))
        cid_arr_old = np.nan * np.ones((ncubes, npartitions, 2))

    cid_arr_full = np.nan * np.ones((ncubes, npartitions, nzorder_permutations))
    cid_arr = np.nan * np.ones((ncubes, npartitions, 2))
    cid_shuffle_arr = np.nan * np.ones_like(cid_arr)
    cid_frac_arr = np.nan * np.ones_like(cid_arr)

    # Compute CID for null cube
    null_cube = np.zeros(hyperwindow_shape_cg, dtype=dtype)
    cid_min = cid(null_cube.flatten())

    for j in range(ncubes):
        for i in range(npartitions):
            x_start = (i % compression_factor) * lx_window_cg
            y_start = (i // compression_factor) * lx_window_cg
            data = defect_grid[j * time_subinterval:(j+1)*time_subinterval+overlap, x_start:x_start+lx_window_cg, y_start:y_start+lx_window_cg]
 
            if data.sum() == 0:
                cid_av, cid_std = cid_min, 0
                cid_shuff = cid_min, 0
                cid_vals = cid_min * np.ones(cid_arr_full.shape[-1])
            else:
                cid_av, cid_std, cid_shuff, cid_vals = CID_obj(data)          
            
            cid_arr[j, i, 0] = cid_av
            cid_arr[j, i, 1] = cid_std
            cid_shuffle_arr[j, i, :] = cid_shuff
            cid_arr_full[j, i, :] = cid_vals

            if compare_hilbert:
                cid_av_hil, cid_std_hil, _, _ = CID_hil(data)
                cid_arr_hil[j, i, 0] = cid_av_hil
                cid_arr_hil[j, i, 1] = cid_std_hil
        
                cid_av_old, cid_std_old, _, _ = CID_hil_old(data)
                cid_arr_old[j, i, 0] = cid_av_old
                cid_arr_old[j, i, 1] = cid_std_old

        if debug: 
            print(f" for act, exp {act}, {exp}: av cid, frac for time cube {j+1}/{ncubes}: {np.nanmean(cid_arr[j,:,0]):.5f}, {np.nanmean(cid_arr[j,:,0]/cid_shuffle_arr[j,:,0]):.5f}")
            if compare_hilbert:
                print(f" for act, exp {act}, {exp}: av cid_hil, frac for time cube {j+1}/{ncubes}: {np.nanmean(cid_arr_hil[j,:,0]):.5f}, {np.nanmean(cid_arr_hil[j,:,0]/cid_shuffle_arr[j,:,0]):.5f}")
                print(f" for act, exp {act}, {exp}: av cid_old, frac for time cube {j+1}/{ncubes}: {np.nanmean(cid_arr_old[j,:,0]):.5f}, {np.nanmean(cid_arr_old[j,:,0]/cid_shuffle_arr[j,:,0]):.5f}")

    cid_frac_arr[..., 0] = cid_arr[..., 0] / cid_shuffle_arr[..., 0]
    cid_frac_arr[..., 1] = cid_frac_arr[..., 0] * np.sqrt( (cid_arr[..., 1]/cid_arr[..., 0])**2 + (cid_shuffle_arr[..., 1]/cid_shuffle_arr[..., 0])**2 )

    if verbose and exp==exp_verbose: 
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            print(f"For act,exp {act,exp}: av CID, CID_shuff, frac: {np.nanmean(cid_arr[:,:,0]):.5f}, {np.nanmean(cid_shuffle_arr[:,:,0]):.5f}, {np.nanmean(cid_arr[:,:,0] /cid_shuffle_arr[:,:,0]):.3f} Time: ", np.round(time.perf_counter()-t3,2), "s")

    if not debug:
        np.savez_compressed(os.path.join(output_path, f'cid{save_suffix}.npz'), cid=cid_arr, cid_shuffle=cid_shuffle_arr, cid_frac=cid_frac_arr,
                            cid_full=cid_arr_full, defect_count_full=defect_count_full, defect_count=defect_count,)
        # Restore stdout and print everything at once
        sys.stdout = sys.__stdout__
        print(buffer.getvalue())

if __name__ == '__main__':
    main()

   