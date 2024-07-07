
import os
import json
import shutil
from multiprocessing.pool import Pool as Pool
from time import perf_counter

import numpy as np



class CompressArchive:
    def __init__(self, archive_dir, output_dir = None, overwrite_existing_npz_files = False, \
                    conversion_kwargs = {'dtype_out': 'float64', 'compress': True, 'exclude_keys': [], 'calc_velocities': False},):

        self.archive_dir = archive_dir
        self.output_dir = archive_dir + '_npz' if output_dir is None else output_dir

        if not os.path.exists(self.archive_dir):
            raise ValueError(f'Archive directory {self.archive_dir} does not exist')
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

        params_path = os.path.join(self.archive_dir, 'parameters.json')
        shutil.copy(params_path, self.output_dir)

        with open(params_path, 'r') as f:
            self.simulation_params = json.load(f)['data']

        self.frame_list = self.__find_missing_frames()
        self.failed_conversion_list = []

        self.LX = self.simulation_params['LX']['value']
        self.LY = self.simulation_params['LY']['value']
        self.num_frames = len(self.frame_list)

        self.overwrite_existing_npz_files = overwrite_existing_npz_files

        self.dtype_out = conversion_kwargs['dtype_out']
        self.compress = conversion_kwargs['compress']
        self.exclude_keys = conversion_kwargs['exclude_keys']
        self.calc_velocities = conversion_kwargs['calc_velocities']
        self.include_keys = []

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
        return

    def __find_missing_frames(self):
        dir_list = os.listdir(self.archive_dir)
        frame_list = []

        for item in dir_list:
            if item.startswith("frame"):
                frame_num = int(item.split('.')[0].split('frame')[-1])
                frame_list.append(frame_num)

        sp = self.simulation_params
        frame_range = np.arange(sp['nstart']['value'], sp['nsteps']['value'] + 1, sp['ninfo']['value'])

        if len(frame_list) == len(frame_range):
            return frame_range
        else:
            frame_list.sort()
            return frame_list
        
    def __unpack_json_dict(self, json_dict,):

        keys = list(json_dict['data'].keys())
        if self.include_keys == []:
            self.include_keys = [key for key in keys if key not in self.exclude_keys]

        arr_dict = {key: np.array(json_dict['data'][key]['value'],dtype=self.dtype_out) for key in keys if key not in self.exclude_keys}

        if self.calc_velocities:
            ff = np.array(json_dict['data']['ff']['value'],dtype='float64')
            arr_dict['density'] = self.__calc_density(ff).astype(self.dtype_out)         
            v = self.__calc_velocity(ff, arr_dict['density'])
            arr_dict['vx'] = v[0].flatten().astype(self.dtype_out)
            arr_dict['vy'] = v[1].flatten().astype(self.dtype_out)
        return arr_dict

    def __convert_json_to_npz(self, frame_number):

        frame_input_path = os.path.join(self.archive_dir, f'frame{frame_number}.json')
        frame_output_path = os.path.join(self.output_dir, f'frame{frame_number}.npz')

        if os.path.exists(frame_output_path) and not self.overwrite_existing_npz_files:
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

        arr_dict = self.__unpack_json_dict(data)
        if self.compress:
            np.savez_compressed(frame_output_path, **arr_dict)
        else:
            np.savez(frame_output_path, **arr_dict)
        return

    def convert_archive_to_npz(self, verbose = 1,):

        # copy parameters.json to output folder
        parameters_path = os.path.join(self.archive_dir, 'parameters.json')
        shutil.copy(parameters_path, self.output_dir)

        if verbose > 0:
            start = perf_counter()

        for i, frame in enumerate(self.frame_list):
            try:
                if verbose == 2:
                    start_frame = perf_counter()
                self.__convert_json_to_npz(frame)
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

    def delete_original_archive(self, only_if_successful = True, call_cluster_cmd = False,  verbose = 1):
        if only_if_successful:
            succesful = self.check_conversion_success(verbose = verbose)
            if not succesful:
                print(f'Unsuccesful conversion: Archive {self.archive_dir} not deleted')
                return        
        if call_cluster_cmd:
            os.system(f'rm -r {self.archive_dir}') 
        else:
            shutil.rmtree(self.archive_dir)
        if verbose > 0:
            print(f'\nArchive {self.archive_dir} deleted\n')
        return

    def print_conversion_info(self):

        self.__estimate_size_reduction()
        self.__get_time_to_open_npz()

        if np.isnan(self.time_to_open_json):
            frame_input_path = os.path.join(self.archive_dir, f'frame{self.frame_list[0]}.json')
            t_start = perf_counter()
            with open(frame_input_path, 'r') as f:
                data = json.load(f)
            self.time_to_open_json = perf_counter() - t_start
    
        print(f'\nEstimated size reduction for entire archive ({self.num_frames} frames):')
        print(f'Uncompressed archive size: {self.json_frame_size / 1024 ** 2 * self.num_frames:.2f} MB')
        print(f'Compressed archive size: {self.npz_frame_size / 1024 ** 2 * self.num_frames:.2f} MB')
        print(f'Compression ratio: {self.compression_ratio:.2f}x\n')

        print(f'Time to open json frame: {self.time_to_open_json:.5f} seconds')
        print(f'Time to open npz frame: {self.time_to_open_npz:.5f} seconds')
        print(f'Speedup: {self.time_to_open_json / self.time_to_open_npz:.2f}x')
        return

    def convert_json_to_npz_parallel(self, frame_input_path, verbose=0):

        frame_output_path = os.path.join(self.output_dir, os.path.basename(frame_input_path).replace('.json', '.npz'))

        with open(frame_input_path, 'r') as f:
            data = json.load(f)

        arr_dict = self.__unpack_json_dict(data)
        if self.compress:
            np.savez_compressed(frame_output_path, **arr_dict)
        else:
            np.savez(frame_output_path, **arr_dict)
        if verbose > 1:
            print(f'File {frame_output_path} created')
        return

    def check_conversion_success(self, verbose = 1):
        if len(self.failed_conversion_list) == 0:

            # check that converted files can be opened and the self.include_keys key are present
            for frame in self.frame_list:
                npz_path = os.path.join(self.output_dir, f'frame{frame}.npz')
                arr_dict = np.load(npz_path, allow_pickle=True)
                for key in self.include_keys:
                    if key not in arr_dict.files:
                        print(f'Key {key} not found in frame {frame}')
                        return False
            if verbose > 0:
                print(f'All {self.num_frames} frames successfully converted to npz')
            return True
        else:
            if verbose > 0:
                print(f'{len(self.failed_conversion_list)} frames failed conversion to npz')
            return False



def conversion_wrapper(frame_input_path, compressor):
    compressor.convert_json_to_npz_parallel(frame_input_path)




