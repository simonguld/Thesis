from .computable_information_density import cid, cid78, cid_linear, cid_hybrid, cid_shuffle
from .hilbert_curve import hilbert_curve, precompute_hcurves, precompute_zcurves
from abc import ABC, abstractmethod
from multiprocessing import Pool
import numpy as np
import time

class ComputableInformationDensity(ABC):
    """ Abstract base class for the Computable Information Density """
    def __init__(self, dim, nbits, nshuff, mode='lz77', verbose=False):
        
        self.size = 1 << nbits
        self.nshuff = nshuff
        self.dim = dim
        self.mode = mode
        self.verbose = verbose

        self.principal_hcurve = hilbert_curve(dim, nbits)
        self.hcurves = precompute_hcurves(self.hamiltonian_cycle, self.principal_hcurve, nbits)

        # select CID implementation
        cid_dict = {'lz77': cid,
                'lz78': cid78,
                    'linear': cid_linear,
                    'hybrid': cid_hybrid}
        self.cid = cid_dict[mode]
    
    @abstractmethod
    def itter_hscan(self, data): pass

    @abstractmethod
    def itter_hscan2(self, data): pass

    def precompute_hcurves(self):
        """Precompute all hcurve coordinate variants exactly once."""
        hcurves = []

        h = self.principal_hcurve.copy()

        for k in self.hamiltonian_cycle:
            if k == 0: 
                h[0] = (self.size - 1) - h[0]
            if k == 1:
                h[1] = (self.size - 1) - h[1]
            if k == 2:
                h[[0, 1]] = h[[1, 0]]

            hcurves.append(h.copy())
        return hcurves

    def hscan(self, data, hcurve):
        """ Hilbert scaned view of data. """
        return data[tuple( hcurve )].T.ravel()
    
    def cid_shuffle(self, data):
        """ CID of randomly shuffled data """
        # flattened- and hilbert scanned- view are
        # statistically equivalent after shuffling: 
        t1 = time.perf_counter()
        cid_shuff_val = cid_shuffle(data.ravel(), self.nshuff, cid_mode=self.mode)
        t2 = time.perf_counter()
        if self.verbose: print(f"shuffling took {t2 - t1:.2f} seconds")
        return cid_shuff_val
    
    def __call__(self, data, n_workers):
        # reshape data to be compatible with Hilbert scan:
        data = np.transpose(data).reshape((-1, ) + (self.size, ) * self.dim).T
        # create and configure a pool of workers:
        t1 = time.perf_counter()
        with Pool(n_workers) as pool:
            cid_pool = pool.map_async(self.cid, self.itter_hscan(data))
            pool.close()    # close the process pool
            pool.join()     # wait for all workers to complete
        t2 = time.perf_counter()
        if self.verbose: print(f"CID took {t2 - t1:.2f} seconds")
        cid_vals = cid_pool.get()
        return np.mean(cid_vals), np.std(cid_vals, ddof=1), self.cid_shuffle(data)
    
    def __call2__(self, data, n_workers):
        # reshape data to be compatible with Hilbert scan:
        data = np.transpose(data).reshape((-1, ) + (self.size, ) * self.dim).T
        # create and configure a pool of workers:
        t1 = time.perf_counter()
        with Pool(n_workers) as pool:
            cid_pool = pool.map_async(self.cid, self.itter_hscan2(data))
            pool.close()    # close the process pool
            pool.join()     # wait for all workers to complete
        t2 = time.perf_counter()
        if self.verbose: print(f"CID took {t2 - t1:.2f} seconds")
        cid_vals = cid_pool.get()
        return np.mean(cid_vals), np.std(cid_vals, ddof=1), self.cid_shuffle(data)
