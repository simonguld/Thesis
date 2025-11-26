
import time
from abc import ABC, abstractmethod
from multiprocessing import Pool

import numpy as np

from .computable_information_density import cid, cid78, cid_linear, cid_hybrid, cid_shuffle
from .hilbert_curve import hilbert_curve, precompute_hcurves

class ComputableInformationDensity(ABC):

    """ Abstract base class for the Computable Information Density """
    def __init__(self, dim, nshuff, mode='lz77', verbose=False):
        
        self.size = 1 << self.nbits
        self.nshuff = nshuff
        self.dim = dim
        self.mode = mode
        self.verbose = verbose

        # select CID implementation
        cid_dict = {'lz77': cid,
                'lz78': cid78,
                    'linear': cid_linear,
                    'hybrid': cid_hybrid}
        self.cid = cid_dict[mode]
    
    @abstractmethod
    def scan(self, data):
        pass

    def itter_scan(self, data):
        """Yield scanned views using precomputed curves (no recomputation)."""
        for curve in self.curves:
            yield self.scan(data, curve)

    def cid_shuffle(self, data):
        """ CID of randomly shuffled data """
        t1 = time.perf_counter()
        cid_shuff_val = cid_shuffle(data.ravel(), self.nshuff, cid_mode=self.mode)
        t2 = time.perf_counter()
        if self.verbose: print(f"shuffling took {t2 - t1:.2f} seconds")
        return cid_shuff_val
    
    def calc_cid(self, data,):
        if self.ordering == 'hilbert':
            data = np.transpose(data).reshape((-1, ) + (self.size, ) * self.dim).T
        else:
            data = data.ravel()
        with Pool(self.ncpus) as pool:
            cid_pool = pool.map_async(self.cid, self.itter_scan(data))
            pool.close() 
            pool.join()
        cid_vals = cid_pool.get()
        if self.verbose: print(cid_vals)
        return np.mean(cid_vals), np.std(cid_vals, ddof=1), self.cid_shuffle(data), cid_vals




class ComputableInformationDensity_old(ABC):

    """ Abstract base class for the Computable Information Density """
    def __init__(self, dim, nbits, nshuff, mode='lz77', verbose=False):
        
        self.size = 1 << nbits
        self.nshuff = nshuff
        self.dim = dim
        self.mode = mode
        self.verbose = verbose

        self.principal_hcurve = hilbert_curve(dim, nbits)
        self.hcurves = precompute_hcurves(self.hamiltonian_cycle, self.principal_hcurve, nbits)
        #self.zcurves = precompute_zcurves( (self.size, ) * dim, nbits)

        # select CID implementation
        cid_dict = {'lz77': cid,
                'lz78': cid78,
                    'linear': cid_linear,
                    'hybrid': cid_hybrid}
        self.cid = cid_dict[mode]
    
    def itter_hscan(self, data):
        """Yield Hilbert scanned views using precomputed hcurves (no recomputation)."""
        for hcurve in self.hcurves:
            yield self.hscan(data, hcurve)

    def itter_zscan(self, data_flattened):
        """Yield zscan views using precomputed zcurves (no recomputation)."""
        for zcurve in self.zcurves:
            yield self.zscan(data_flattened, zcurve)

    def hscan(self, data, hcurve):
        """ Hilbert scaned view of data. """
        return data[tuple( hcurve )].T.ravel()

    def zscan(self, data_flattened, zcurve):
        """ zscan view of data. """
        return data_flattened[zcurve]

    def cid_shuffle(self, data):
        """ CID of randomly shuffled data """
        t1 = time.perf_counter()
        cid_shuff_val = cid_shuffle(data.ravel(), self.nshuff, cid_mode=self.mode)
        t2 = time.perf_counter()
        if self.verbose: print(f"shuffling took {t2 - t1:.2f} seconds")
        return cid_shuff_val
    
    def calc_zorder_cid(self, data, n_workers):
        data_flattened = data.ravel()
        t1 = time.perf_counter()
        with Pool(n_workers) as pool:
            cid_pool = pool.map_async(self.cid, self.itter_zscan(data_flattened))
            pool.close() 
            pool.join()
        t2 = time.perf_counter()
        if self.verbose: print(f"CID took {t2 - t1:.2f} seconds")
        cid_vals = cid_pool.get()
        return np.mean(cid_vals), np.std(cid_vals, ddof=1), self.cid_shuffle(data)
    
    def calc_horder_cid(self, data, n_workers):
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
        return np.mean(cid_vals), np.std(cid_vals, ddof=1), self.cid_shuffle(data), cid_vals
    
