# Author:  Simon Guldager
# Date (latest update): 

### SETUP ------------------------------------------------------------------------------------
import os
import sys
import pickle
import time
import argparse

import numpy as np

sys.path.append('/groups/astro/kpr279/')
import massPy as mp


development_mode = False
num_frames = 5 if development_mode else None

### FUNCTIONS ----------------------------------------------------------------------------------


def calc_scalar_order_param(director_x, director_y, use_2d_form = True):
   
    nx = director_x.astype(np.float64)  
    ny = director_y.astype(np.float64)
    n_norm = np.sqrt(nx**2 + ny**2)

    if not np.allclose(n_norm, 1.0):
        nx /= n_norm
        ny /= n_norm

    direction = np.arctan(ny/nx).flatten()
    av_direction = np.arctan(ny.mean() / nx.mean())
 
    rel_angle = av_direction - direction
    order_param = (2* np.cos(rel_angle)**2 - 1) if use_2d_form else .5 * (3* np.cos(rel_angle)**2 - 1)
    return order_param


def calc_order_param_block(director_x, director_y, block_size = None, use_2d_form = True):

    nx = director_x.astype(np.float64)  
    ny = director_y.astype(np.float64)
    
    n_norm = np.sqrt(nx**2 + ny**2)
    N = nx.shape[0]

    if not np.allclose(n_norm, 1.0):
        nx /= n_norm
        ny /= n_norm
    if block_size is None or block_size == N:
        if block_size is None:
            print("Block size not specified, using full array")
        return calc_scalar_order_param(nx, ny, use_2d_form)
    if not N % block_size == 0:
        print("N must be divisable by block_size for this to work")
        return
    if block_size == 1:
        print("Block size must be greater than 1")
        return

    direction = np.arctan(ny/nx) #.flatten()

    block_arr_dir = direction.reshape(N//block_size, block_size, N//block_size, block_size)
    block_arr_nx = nx.reshape(N//block_size, block_size, N//block_size, block_size)
    block_arr_ny = ny.reshape(N//block_size, block_size, N//block_size, block_size)

    block_arr_dir_av = np.arctan(block_arr_ny.mean(axis = (1, 3)) / block_arr_nx.mean(axis = (1, 3)))
    block_arr_diff = (block_arr_dir - block_arr_dir_av[:, None, :, None]).flatten()

    order_param = (2* np.cos(block_arr_diff)**2 - 1) if use_2d_form else .5 * (3* np.cos(block_arr_diff)**2 - 1)
    return order_param

def get_scalar_order_param(archive, frame_idx_range = None, ddof = 1, save_dir = None):
    """
    
    Parameters:
    -----------
    Returns: order param array shape (Nframes, 4), where the first index is the frame number, 
             the second index is mean and std of S using eq. in 2d, then the mean and std of S using eq. in 3d
    --------
    """

    Nframes = archive.num_frames if frame_idx_range is None else frame_idx_range[1] - frame_idx_range[0]
    if frame_idx_range is None:
        frame_idx_range = [0, Nframes]
    LX = archive.LX  
    LY = archive.LY

    order_param_arr = np.nan * np.ones([Nframes, 4])

    for i in np.arange(frame_idx_range[0], frame_idx_range[1]):
        frame = archive._read_frame(i)

        Qxx_dat = frame.QQxx.reshape(LX, LY)
        Qyx_dat = frame.QQyx.reshape(LX, LY)

        _, nx, ny = mp.nematic.nematicPy.get_director(Qxx_dat, Qyx_dat)

        S_2d = calc_scalar_order_param(nx, ny, use_2d_form = True)
        S_3d = calc_scalar_order_param(nx, ny, use_2d_form = False)

        order_param_arr[i] = S_2d.mean(), S_2d.std(ddof = ddof) / np.sqrt(LX*LY), S_3d.mean(), S_3d.std(ddof = ddof)/ np.sqrt(LX*LY)

    if save_dir is not None:
        np.save(os.path.join(save_dir, 'order_param_arr.npy'), order_param_arr)
    return order_param_arr


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

    t1 = time.perf_counter()
    msg = f"\nCalculating scalar order param for input_folder: {input_folder}"
    print(msg)

    _ = get_scalar_order_param(ar, frame_idx_range = [0, num_frames], save_dir = save_path)

    msg = f"Time to do analysis: {np.round(time.perf_counter()-t1,2)} s"
    print(msg)

    gen_status_txt(msg, os.path.join(output_path, 'order_param_analysis_completed.txt'))

if __name__ == '__main__':
    main()
