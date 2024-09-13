from  subprocess import check_output
from pathlib import Path
import numpy as np
import json
import os

class Archive:
    """ Import archive :
    Automatically import the parameters and allows to extract the individual frame files as well.
    """
    def __init__(self, path):
        """ Reads parameters from archive """
        # declare archive path as instance attribute:
        self._path = Path(path)
        
        # check if we use a compressed archive or a directory:
        if self._path.suffix == '.zip':
            self._compress_full = True
        else:
            self._compress_full = False

        # check for single file compression:
        if (self._path / 'parameters.json.zip').is_file():
            self._compress = True
            self._ext = '.json.zip'
        if (self._path / 'parameters.json').is_file():
            self._compress = False
            self._ext = '.json'
        
        # load archive parameters as a dictionary:
        parameters = self._extract_and_read('parameters')

        # check if we have npz files:
        path_first_frame = os.path.join(self._path, f'frame{parameters["nstart"]}.npz')
        if os.path.isfile(path_first_frame):
            self._ext = '.npz'

        parameters['_ext'] = self._ext

        # add forcing scheme parameter if not present
        self.isGuo = parameters['isGuo'] if 'isGuo' in parameters.keys() else True

        # and add them as instance attributes on the Archive object:
        self.__dict__.update(parameters)
        # total number of frames (fencepost problem):
        self.num_frames = int((self.nsteps-self.nstart)/self.ninfo) + 1
    
    def __getitem__(self, n):
        """ Return state file from archive """
        return self._read_frame(n % self.num_frames)
    
    def _extract_and_read(self, fname):
        """Extract json file from archive."""
        # extract
        if self._compress_full:
            raise NotImplementedError("Jeez! It seems that you try to load a zipped archive; this feature hasn't been implemented yet!")
        elif self._compress:
            # get output from stdin (directly pipe from unzip):
            output = check_output(['unzip', '-pj', self._path.joinpath(fname).with_suffix(self._ext)]).decode()
            data = json.loads(output)['data']
        elif self._ext == '.npz':
            data = np.load(self._path.joinpath(fname).with_suffix('.npz'), allow_pickle=True)
            return {key: data[key] for key in data.files}
        else:
            # read content of file:
            output = open(self._path.joinpath(fname).with_suffix(self._ext))
            data = json.load(output)['data']
        # convert to dictionary:
        return {entry : self._get_value(data[entry]['value'], data[entry]['type']) for entry in data}
    
    def _get_value(self, v, t):
        """Convert string to value with correct type handling."""
        if v=="nan" or v=="-nan":
          raise ValueError(f"Nan found while converting to {t}")

        if   t=='double' or t=='float':
            return float(v)
        elif t=='int' or t=='unsigned':
            return int(v)
        elif t=='long' or t=='unsigned long':
            return int(v)
        elif t=='bool':
            return bool(v)
        elif t=='string':
            # quotes are already erased for string types
            return v
        elif t[:5]=='array':
            return np.array([ self._get_value(i, t[6:-1]) for i in v ])
        else:
            raise ValueError(f"Unrecognized type {t}")

    def _read_frame(self, nframe):
        """ Read state file from archive """
        if nframe >= self.num_frames:
            raise ValueError('Frame does not exist.')
        # frame timestamp:
        timestamp = self.nstart + nframe * self.ninfo
        # return a dummy object that holds the data:
        class FrameHolder:
            """ dummy class to hold frame data """
            def __init__(self, timestamp): self.time = timestamp
        # instantiate dummy object and forward Archive instance attributes:
        frame = FrameHolder(timestamp)
        frame.__dict__.update(self.__dict__)
        # load frame data as a dictionary:
        data = self._extract_and_read(f"frame{timestamp}")
        # and add it as instance attributes:
        frame.__dict__.update(data)
        return frame

    def read_frames(self):
        """ Generates all frames successively """
        for n in range(self.num_frames):
            yield self._read_frame(n)

def loadarchive(path):
    return Archive(path)
