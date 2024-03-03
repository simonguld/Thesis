from multiprocessing.pool import Pool as Pool

import os
import numpy as np
import json
import pickle as pkl
import shutil

from glob import glob
from time import perf_counter




import numpy as np


import massPy as mp



def convert_json_to_npz_parallel(frame_input_path, compress_archive_instance):
    compress_archive_instance.convert_json_to_npz_parallel(frame_input_path)
    return




def convert_json_to_npz(frame_input_path, compress = True, dtype_out = 'float32', exclude_keys=[]):


    frame_output_path = os.path.dirname(frame_input_path) + '_npz'
    frame_output_path = os.path.join(frame_output_path, os.path.basename(frame_input_path).replace('.json', '.npz'))

    with open(frame_input_path, 'r') as f:
        data = json.load(f)

    keys = list(data['data'].keys())
    arr_dict = {}
    arr_dict = {key: np.array(data['data'][key]['value'],dtype=dtype_out) for key in keys if key not in exclude_keys}

    if compress:
        np.savez_compressed(frame_output_path, **arr_dict)
    else:
        np.savez(frame_output_path, **arr_dict)
    return



class CompressArchive:
    def __init__(self, archive_dir, output_dir = None, dtype_out = 'float64', delete_archive_if_successful = False):

        self.archive_dir = archive_dir
        self.output_dir = archive_dir + '_compressed' if output_dir is None else output_dir

        self.archive = mp.archive.loadarchive(archive_dir)
        self.frame_list = self.__find_missing_frames()
        self.failed_conversion_list = []

        self.num_frames = len(self.frame_list)
        self.LX = self.archive.LX
        self.LY = self.archive.LY


        self.dtype_out = dtype_out
        self.delete_archive_if_successful = delete_archive_if_successful

        self.time_to_open_json = np.nan
        self.time_to_open_npz = np.nan
        self.compression_ratio = np.nan
        self.json_frame_size = np.nan
        self.npz_frame_size = np.nan

    def __calc_density(self, ff):
        return np.sum(ff, axis=1)

    def __calc_velocity(self, ff, density = None):
        d = self.__calc_density(ff) if density is None else density
        return np.asarray([ (ff.T[1] - ff.T[2] + ff.T[5] - ff.T[6] - ff.T[7] + ff.T[8]) / d,
                            (ff.T[3] - ff.T[4] + ff.T[5] - ff.T[6] + ff.T[7] - ff.T[8]) / d
                        ]).reshape(2, self.LX, self.LY)

    def __estimate_size_reduction(self,):

        input_file = os.path.join(self.archive_dir, f'frame{self.frame_list[0]}.json')
        output_file = os.path.join(self.output_dir, f'frame{self.frame_list[0]}.npz')

        if not os.path.exists(output_file):
            print(f'File {output_file} does not exist')
            return

        # Get the sizes in bytes of the input and output archives
        self.json_frame_size = os.path.getsize(input_file)
        self.npz_frame_size = os.path.getsize(output_file)

        # Calculate the compression ratio
        self.compression_ratio = self.json_frame_size / self.npz_frame_size
        return 
  
    def __get_time_to_open_npz(self, frame_number = None):

        if np.isnan(self.time_to_open_npz):
            frame_number = self.frame_list[0] if frame_number is None else frame_number
            npz_file_path = os.path.join(self.output_dir, f'frame{frame_number}.npz')
            if not os.path.exists(npz_file_path):
                print(f'File {npz_file_path} does not exist')
                return
            try:
                t_start = perf_counter()
                npz = np.load(npz_file_path, allow_pickle=True)
                self.time_to_open_npz = perf_counter() - t_start
            except:
                print(f'Error opening file {npz_file_path}')
        else:
            pass
        return

    def __find_missing_frames(self):
        dir_list = os.listdir(self.archive_dir)
        frame_list = []

        for item in dir_list:
            if item.startswith("frame"):
                frame_num = int(item.split('.')[0].split('frame')[-1])
                frame_list.append(frame_num)

        if len(frame_list) == self.archive.num_frames:
            return np.arange(self.archive.nstart, self.archive.nsteps + 1, self.archive.ninfo)
        else:
            frame_list.sort()
            return frame_list
        
    def __unpack_json_dict(self, json_dict, exclude_keys=[], calc_velocities = False):

        keys = list(json_dict['data'].keys())
        arr_dict = {key: np.array(json_dict['data'][key]['value'],dtype=self.dtype_out) for key in keys if key not in exclude_keys}

        if calc_velocities:
            ff = np.array(json_dict['data']['ff']['value'],dtype='float64')
            arr_dict['density'] = self.__calc_density(ff).astype(self.dtype_out)         
            v = self.__calc_velocity(ff, arr_dict['density'])
            arr_dict['vx'] = v[0].flatten().astype(self.dtype_out)
            arr_dict['vy'] = v[1].flatten().astype(self.dtype_out)
        return arr_dict

    def __convert_json_to_npz(self, frame_number, overwrite_existing_npz_files = False, compress = True, exclude_keys=[], calc_velocities = False):

        frame_input_path = os.path.join(self.archive_dir, f'frame{frame_number}.json')
        frame_output_path = os.path.join(self.output_dir, f'frame{frame_number}.npz')

        if os.path.exists(frame_output_path) and not overwrite_existing_npz_files:
            print(f'File {frame_output_path} already exists. Skipping...')
            return

        if np.isnan(self.time_to_open_json):
            t_start = perf_counter()
            with open(frame_input_path, 'r') as f:
                data = json.load(f)
            self.time_to_open_json = perf_counter() - t_start
        else:
            with open(frame_input_path, 'r') as f:
                data = json.load(f)

        arr_dict = self.__unpack_json_dict(data, exclude_keys=exclude_keys, calc_velocities = calc_velocities)
        if compress:
            np.savez_compressed(frame_output_path, **arr_dict)
        else:
            np.savez(frame_output_path, **arr_dict)
        return

    def convert_archive_to_npz(self, compress = True, exclude_keys=[], calc_velocities = False, overwrite_existing_npz_files = False, verbose = 1,):
        # Create the output folder if it does not exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # copy parameters.json to output folder
        parameters_path = os.path.join(self.archive_dir, 'parameters.json')
        shutil.copy(parameters_path, self.output_dir)

        if verbose > 0:
            start = perf_counter()

        for i, frame in enumerate(self.frame_list):
            try:
                if verbose == 2:
                    start_frame = perf_counter()
                self.__convert_json_to_npz(frame, overwrite_existing_npz_files = overwrite_existing_npz_files,\
                                            compress = compress, exclude_keys = exclude_keys, calc_velocities = calc_velocities)
                if verbose == 2:
                    print(f'Frame {frame} processed in {perf_counter() - start_frame:.2f} seconds')
            except:
                print(f'Error processing frame {frame}. Skipping...')
                self.failed_conversion_list.append(frame)

        if verbose > 0:
            print(f'Archive processed in {perf_counter() - start:.2f} seconds with {len(self.failed_conversion_list)} failed conversions')
            if len(self.failed_conversion_list) > 0:
                print(f'Frames for which conversion to npz failed: {self.failed_conversion_list}')
            self.print_conversion_info()

        if self.delete_archive_if_successful and len(self.failed_conversion_list) == 0:
            shutil.rmtree(self.archive_dir)
            print(f'Archive {self.archive_dir} deleted')
        return

    def print_conversion_info(self):

        self.__estimate_size_reduction()
        self.__get_time_to_open_npz()
    
        print(f'\nEstimated size reduction for entire archive ({self.num_frames} frames):')
        print(f'Uncompressed archive size: {self.json_frame_size / 1024 ** 2 * self.num_frames:.2f} MB')
        print(f'Compressed archive size: {self.npz_frame_size / 1024 ** 2 * self.num_frames:.2f} MB')
        print(f'Compression ratio: {self.compression_ratio:.2f}x\n')

        print(f'Time to open json frame: {self.time_to_open_json:.2f} seconds')
        print(f'Time to open npz frame: {self.time_to_open_npz:.2f} seconds')
        print(f'Speedup: {self.time_to_open_json / self.time_to_open_npz:.2f}x')
        return

    def convert_json_to_npz_parallel(self, frame_input_path, compress = True, dtype_out = 'float32', exclude_keys=[]):


        frame_output_path = os.path.dirname(frame_input_path) + '_npz'
        frame_output_path = os.path.join(frame_output_path, os.path.basename(frame_input_path).replace('.json', '.npz'))

        with open(frame_input_path, 'r') as f:
            data = json.load(f)

        keys = list(data['data'].keys())
        arr_dict = {}
        arr_dict = {key: np.array(data['data'][key]['value'],dtype=dtype_out) for key in keys if key not in exclude_keys}

        if compress:
            np.savez_compressed(frame_output_path, **arr_dict)
        else:
            np.savez(frame_output_path, **arr_dict)
        return



if __name__ == '__main__':
    main_path = 'C:\\Users\\Simon Andersen\\Dokumenter\\Uni\\Speciale\\Hyperuniformity\\nematic_data'
    data_dirs = [os.path.join(main_path, d) for d in os.listdir(main_path)]
    N = -3
    dir = data_dirs[N]


    frame_list= list(np.arange(10000, 11000, 500))


    archive_path = dir  
    output_folder = dir + '_npz'
    compress = True
    dtype_out= 'float32'
    exclude_keys=[]

    for file in glob(os.path.join(output_folder, 'frame*.npz')):
        os.remove(file)

    files = glob(os.path.join(dir, 'frame*'))
    ntasks = min(len(files), os.cpu_count())
    csize = 5 # max(10, len(frame_list) // ntasks)
    print(f'Number of tasks: {ntasks}, chunksize: {csize}')

    ca = CompressArchive(archive_path, output_folder)
    convert_json_to_npz_parallel(files[0], ca)

    if 0:
        t_start = perf_counter()
        with Pool(ntasks) as p:
            p.map(convert_json_to_npz, files, chunksize=csize)
            #   p.close()
            #  p.join()

        print(f'Elapsed time: {perf_counter() - t_start:.2f} s')
        
