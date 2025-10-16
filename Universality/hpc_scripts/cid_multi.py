
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

from ComputableInformationDensity.cid import interlaced_time
from ComputableInformationDensity.computable_information_density import cid, cid_linear, cid_shuffle

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

    # Redirect stdout to a buffer
    buffer = io.StringIO()
    sys.stdout = buffer

    # --------------------------------------------
    # Load data
    t1 = time.perf_counter()
    ar = mp.archive.loadarchive(archive_path)
    LX, LY = ar.__dict__['LX'], ar.__dict__['LY']
    #Nframes = ar.__dict__['num_frames'] if not development_mode else num_frames
    exp = int(output_path.split('_')[-1])
    act = float(output_path.split('_')[-3])

    if exp==1: print(f"\nAnalyzing experiment {exp} and activity {act}")
    #print(msg)

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

    # --------------------------------------------
    
    # Set params
    dtype = np.uint8
    cid_mode = 'lz77'
    verbose = True

    # use every nth frame to reduce temporal correlation
    njumps_between_frames = 1 #4 if len(top_defects) > 200 else 2
    top_defects = top_defects[::njumps_between_frames]

    num_frames = len(top_defects)
    nframes_to_analyze = min(400, num_frames)
    nshuffle = 16

    nbits = 5 # side length of each hypercube will be 2^nbits
    coarse_graining_box_length = 4
    LX_cg = LX // coarse_graining_box_length
    
    compression_factor = LX_cg // (1 << nbits)  # number of partitions along one dimension 
    npartitions = compression_factor**2
    lx_window = LX // compression_factor
    lx_window_cg = lx_window // coarse_graining_box_length

    overlap = 0 # overlap between cubes
    nframes_per_cube = get_allowed_time_intervals(system_size = lx_window_cg, nbits_max=nbits)[-1]['time_interval']
    time_subinterval = nframes_per_cube - overlap
    ncubes = 1 + int(((nframes_to_analyze - nframes_per_cube) / time_subinterval))
    first_frame_idx = (num_frames - nframes_to_analyze) + (nframes_to_analyze - nframes_per_cube - ((ncubes - 1) * time_subinterval))

    save_suffix = f'_{cid_mode}_c{compression_factor}_nb{nbits}_o{overlap}_cg{coarse_graining_box_length}' # suffix for saving files
    save_suffix = f'_nb{nbits}cg{coarse_graining_box_length}' 
    param_dict = {
        'cid_mode' : cid_mode,
        'dtype' : dtype,
        'nshuffle' : nshuffle,
        'nbits' : nbits,
        'compression_factor' : compression_factor,
        'coarse_graining_box_length' : coarse_graining_box_length,
        'window_size' : lx_window,
        'npartitions' : npartitions,
        'num_frames' : num_frames,
        'njumps_between_frames' : njumps_between_frames,
        'nframes_to_analyze' : nframes_to_analyze,
        'ncubes' : ncubes,
        'nframes_per_cube' : nframes_per_cube,
        'overlap' : overlap,
        'time_subinterval' : time_subinterval,
        'first_frame_idx' : first_frame_idx,     
    }

    if verbose and exp==1:
        print(f"compression_factor, npartitions: {compression_factor}, {npartitions}")
        print(f"Using window size {lx_window}x{lx_window}.")
        print(f"nf_cube,lx_cg,ly_cg= {nframes_per_cube}x{lx_window_cg}x{lx_window_cg}")
        print(f"Ncubes, Nframes_per, overlap, time subinterval: {ncubes}, {nframes_per_cube}, {overlap}, {time_subinterval}")


    with open(os.path.join(output_path, f'cid_params{save_suffix}.pkl'), 'wb') as f:
            pickle.dump(param_dict, f)

    # --------------------------------------------
    # Create defect grid and compute CID

    defect_grid = np.zeros((num_frames - first_frame_idx, LX_cg, LX_cg), dtype=dtype)
    defect_count_full = []
    defect_count = []

    for i, defect in enumerate(top_defects[first_frame_idx:]):
        def_arr = get_defect_arr_from_frame(defect) #.astype(int)
        if def_arr is None:
            defect_count.append(0)
            continue

        def_arr = def_arr.astype(int)
        defect_count_full.append(len(def_arr))

        # Coarse-grain: map positions to coarse grid indices
        coarse_x = (def_arr[:, 0] // coarse_graining_box_length).astype(int)
        coarse_y = (def_arr[:, 1] // coarse_graining_box_length).astype(int)

        # Set coarse cells to 1 if any defect is inside
        defect_grid[i, coarse_x, coarse_y] = 1
        defect_count.append(defect_grid[i,:,:].sum())

    if verbose: print(f"Average number of defects in window before/after coarse graining: {np.mean(defect_count_full):.2f}, {np.mean(defect_count):.2f}")


    t3 = time.perf_counter()
    CID = interlaced_time(nbits=nbits, nshuff=nshuffle,mode=cid_mode, verbose=False)

    cid_arr = np.nan * np.ones((ncubes, npartitions, 2))
    cid_shuffle_arr = np.nan * np.ones_like(cid_arr)
    cid_frac_arr = np.nan * np.ones_like(cid_arr)

    for j in range(ncubes):
        for i in range(npartitions):
            x_start = (i % compression_factor) * lx_window_cg
            y_start = (i // compression_factor) * lx_window_cg
            data = defect_grid[j * time_subinterval:(j+1)*time_subinterval+overlap, x_start:x_start+lx_window_cg, y_start:y_start+lx_window_cg]
 
            cid_av, cid_std, cid_shuff = CID(data)
            cid_arr[j, i, 0] = cid_av
            cid_arr[j, i, 1] = cid_std
            cid_shuffle_arr[j, i, :] = cid_shuff


    cid_frac_arr[:, :, 0] = cid_arr[:, :, 0] / cid_shuffle_arr[:, :, 0]
    cid_frac_arr[:, :, 1] = cid_frac_arr[:, :, 0] * np.sqrt( (cid_arr[:, :, 1]/cid_arr[:, :, 0])**2 + (cid_shuffle_arr[:, :, 1]/cid_shuffle_arr[:, :, 0])**2 )

    if verbose: print(f"For act,exp {act,exp}: av CID, CID_shuff, frac: {np.nanmean(cid_arr[:,:,0]):.5f}, {np.nanmean(cid_shuffle_arr[:,:,0]):.5f}, {np.nanmean(cid_arr[:,:,0] /cid_shuffle_arr[:,:,0]):.3f} Time: ", np.round(time.perf_counter()-t3,2), "s")

    # save a .npz file with cid results
    np.savez_compressed(os.path.join(output_path, f'cid{save_suffix}.npz'), cid=cid_arr, cid_shuffle=cid_shuffle_arr, cid_frac=cid_frac_arr)

    # Restore stdout and print everything at once
    sys.stdout = sys.__stdout__
    print(buffer.getvalue())

if __name__ == '__main__':
    main()

   