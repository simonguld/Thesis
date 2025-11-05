import os
from multiprocessing import cpu_count

import numpy as np
from .cid_abc import ComputableInformationDensity

class CID(ComputableInformationDensity):
    
    hamiltonian_cycle = [0, 1, 0, 2, 1, 0, 1, 2]
    length = len(hamiltonian_cycle)

    slurm_cpus = os.getenv("SLURM_CPUS_PER_TASK")
    ncpus = min(int(slurm_cpus) if slurm_cpus else cpu_count(), length)
    
    def __init__(self, dim, nbits, nshuff, mode='lz77', verbose=False):
        super().__init__(dim, nbits, nshuff, mode, verbose)
        if verbose:
            print(f"Using {self.ncpus} workers for CID calculations")

    def itter_hscan2(self, data):
        """ yields all 8 distinct Hilbert scanned views of the data. 
        Since a view is returned, this operation is O(1). """
        for k in self.hamiltonian_cycle:
            hcurve = self.principal_hcurve  # view of principal_hcurve i.e. O(1)
            if k == 0: hcurve[0] = (self.size - 1) - hcurve[0]
            if k == 1: hcurve[1] = (self.size - 1) - hcurve[1]
            if k == 2: hcurve[[0,1]] = hcurve[[1,0]]
            yield self.hscan(data, hcurve)  # view of data, i.e. O(1)

    def itter_hscan(self, data):
        """Yield Hilbert scanned views using precomputed hcurves (no recomputation)."""
        for hcurve in self.hcurves:
            yield self.hscan(data, hcurve)
        
    def __call__(self, data):
        cid_av, cid_std, cid_shuffle = super().__call__(data, n_workers=self.ncpus)
        cid_sem = cid_std / np.sqrt(self.length)
        return cid_av, cid_sem, cid_shuffle
    
    def __call2__(self, data):
        cid_av, cid_std, cid_shuffle = super().__call2__(data, n_workers=self.ncpus)
        cid_sem = cid_std / np.sqrt(self.length)
        return cid_av, cid_sem, cid_shuffle

def cid2d(nbits, nshuff, mode='lz77', verbose=False):
    """ Two-dimensional CID Analysis \n
    Args:
        order: linear system size (log2) (int).
        nshuff: number of radnom shuffles of data.
    Returns: instance of the CID class """
    return CID(2, nbits, nshuff, mode, verbose)

def sequential_time(nbits, nshuff, mode='lz77', verbose=False):
    """ Spatiotemporal CID Analysis \n
    Args:
        order: linear system size (log2) (int).
        nshuff: number of radnom shuffles of data.
    Returns: instance of the CID class """
    return CID(2, nbits, nshuff, mode, verbose)

def interlaced_time(nbits, nshuff, mode='lz77', verbose=False):
    """ Spatiotemporal CID Analysis \n
    Args:
        order: linear system size (log2) (int).
        nshuff: number of radnom shuffles of data.
    Returns: instance of the CID class """
    return CID(3, nbits, nshuff, mode, verbose)
