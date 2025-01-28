# Author:  Simon Guldager
# Date (latest update): 

### SETUP ------------------------------------------------------------------------------------

## Imports:
import os
import sys
import pickle
import time
import argparse
import cv2

import numpy as np
import matplotlib.pyplot as plt
from itertools import product


sys.path.append('/groups/astro/kpr279/')
import massPy as mp

## Change directory to current one
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
sys.path.append(os.getcwd())

make_plots = True
make_movies = True
development_mode = False
check_for_convergence = False

# if development_mode, use only few frames
if development_mode:
    num_frames = 20

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

def director(frame, Qxx=None, Qyx=None, engine=plt, scale=False, avg=4, ms = 1, alpha=1, lw=.5):
    """ Plot the director field """
    # get nematic field tensor
    if Qxx is None:
        Qxx = frame.QQxx.reshape(frame.LX, frame.LY)
        Qyx = frame.QQyx.reshape(frame.LX, frame.LY)

    # get order S and director (nx, ny)
    S, nx, ny = mp.nematic.nematicPy.get_director(Qxx, Qyx)
    # plot using engine
    x, y = [], []
    for i, j in product(np.arange(frame.LX, step=avg), np.arange(frame.LY, step=avg)):
        f = avg*(S[i,j] if scale else 1.)
        x.append(i - f*nx[i,j]/2.)
        x.append(i + f*nx[i,j]/2.)
        x.append(None)
        y.append(j - f*ny[i,j]/2.)
        y.append(j + f*ny[i,j]/2.)
        y.append(None)
    engine.plot(x, y, color='k', linestyle='-', markersize=ms, linewidth=lw, alpha=alpha)
    return

def plot_defects(frame, defect_dict = None, defect_ms = 1,
                director_dict = {'scale': False, 'avg': 4, 'ms':  1, 'alpha': 1, 'lw': .5}, 
                dpi = 420, save_path = None):
    LX = frame.LX
    LY = frame.LY 
    Qxx_dat = frame.QQxx.reshape(LX, LY)
    Qyx_dat = frame.QQyx.reshape(LX, LY)

    if defect_dict is None:
        defect_dict = mp.nematic.nematicPy.get_defects(Qxx_dat, Qyx_dat, LX, LY)
    
    def_arr = get_defect_arr_from_frame(defect_dict)
    charge_arr = np.array([defect_dict[i]['charge'] for i in np.arange(len(defect_dict))])
    plus_mask = (charge_arr > 0)


    fig, ax = plt.subplots(figsize=(6,6))     
    director(frame, Qxx_dat, Qyx_dat, engine=ax, **director_dict)

    if def_arr is not None:
        ax.scatter(def_arr[plus_mask, 0], def_arr[plus_mask, 1], c='g', alpha=1,marker='^', s=defect_ms)
        ax.scatter(def_arr[~plus_mask, 0], def_arr[~plus_mask, 1], c='b', alpha=1,marker='v', s=defect_ms)   
    ax.set_aspect('equal')
    ax.set(xlim=[0,LX], ylim=[0,LY]);
    ax.xaxis.set_ticks_position('none') 
    ax.yaxis.set_ticks_position('none')

    if save_path is None:
        save_path = 'defects.png'

    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    return

def make_movie(image_folder, Nexp, act, output_folder=None, resize_factor = 1, fps = 10):
    """
    Make a movie from the images in image_folder
    """
  
    output_folder = output_folder if output_folder is not None else image_folder
    video_name = os.path.join(output_folder, f"zeta_{act}_counter_{Nexp}.mp4")

    # Get all image files from the folder
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images = sort_files(images)  # Ensure images are in the correct order

    if len(images) == 0:
        print(f"No images found in {image_folder}. Exiting...")
        return

    # Read the first image to get the dimensions
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, _ = frame.shape
    width = int(width*resize_factor)
    height = int(height*resize_factor)

    # Define the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
    video = cv2.VideoWriter(video_name, fourcc, 10, (width, height))

    # Loop through images and add them to the video
    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        if resize_factor != 1:
            frame = cv2.resize(frame, (width, height))
        video.write(frame)

    # Release the video writer
    video.release()
    print(f"Video saved as {video_name}")
    return


def gen_status_txt(message = '', log_path = None):
    """
    Generate txt file with message and no.
    """
    with open(log_path, 'w') as f:
        f.write(message)
    return

def sort_files(files):
    return sorted(files, key = lambda x: int(x.split('_')[-1].split('.')[0]))

### MAIN ---------------------------------------------------------------------------------------




def main():

    ## Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str)
    parser.add_argument('--output_folder', type=str)
    parser.add_argument('--defect_list_folder', type=str, default=None)

    args = parser.parse_args()
    input_path = args.input_folder
    output_path = args.output_folder
    defect_list_folder = args.defect_list_folder

    # Get experiment no, and activity
    exp = int(output_path.split('_')[-1])
    act = float(output_path.split('_')[-3])

    ## Define parameters
    ar = mp.archive.loadarchive(input_path)
    LX, LY = ar.__dict__['LX'], ar.__dict__['LY']
  
    # use only a subset of frames if development_mode
    Njump = 1
    first_frame_idx = 0 #301 if act <= 0.022 else 81
    frame_interval = [first_frame_idx, first_frame_idx + num_frames] if development_mode else [first_frame_idx, ar.num_frames]     

    # set plotting parameters
    director_dict = {'scale': False, 'avg': 4, 'ms':  .5, 'alpha': .3, 'lw': .4}
    
    dpi = 320 #420
    defect_ms = 1 if act < 0.024 else .6
    # set video parameters
    video_dict = {'resize_factor': .5, 'fps': 5}


    # Get defect list if provided
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
        # If defect list is not provided, calculate it
        ar = mp.archive.loadarchive(input_path)
        LX, LY = ar.__dict__['LX'], ar.__dict__['LY']
        t1 = time.perf_counter()


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

    if make_plots:
        for frame_idx in range(frame_interval[0], frame_interval[1], Njump):
            frame = ar._read_frame(frame_idx)
            defects = top_defects[frame_idx]
            save_path = os.path.join(output_path, f'defect_director_{frame_idx}.png') 
            plot_defects(frame, defect_dict=defects, defect_ms = defect_ms, director_dict=director_dict, save_path=save_path, dpi = dpi)
    if make_movies:
        # Make movie
        make_movie(output_path, exp, act, output_folder=output_path, **video_dict)

    print(f"Time to create plots for experiment {exp} and activity {act}: ", np.round(time.perf_counter()-t1,2), "s")
    msg = f"Analysis completed for experiment {exp} and activity {act}."
    gen_status_txt(msg, os.path.join(output_path, 'analysis_completed.txt'))
  

if __name__ == '__main__':
    main()
