import os
from multiprocessing import cpu_count

import numpy as np

from .cid_abc import ComputableInformationDensity, ComputableInformationDensity_old
from .hilbert_curve import hilbert_curve, precompute_hcurves, precompute_zcurves

class CID(ComputableInformationDensity):
    def __init__(self, dim, nshuff, nbits=None, data_shape=None,
                ordering='hilbert', mode='lz77', verbose=False):

        self.ordering = ordering

        if ordering == 'hilbert':
            self.__init_hilbert(dim, nbits)
        elif ordering == 'zcurve':
            self.__init_zcurve(dim, data_shape)
        else:
            raise ValueError("ordering must be 'hilbert' or 'zcurve'")

        # CPU allocation depending on Slurm or local environment
        slurm_cpus = os.getenv("SLURM_CPUS_PER_TASK")
        available_cpus = int(slurm_cpus) if slurm_cpus else cpu_count()
        self.ncpus = min(available_cpus, self.length)

        super().__init__(dim, nshuff, mode=mode, verbose=verbose)

    def __init_hilbert(self, dim, nbits):
        if nbits is None:
            raise ValueError("nbits must be provided for hilbert ordering")

        self.nbits = nbits
        self.size = 1 << self.nbits

        # Precompute hilbert curves
        self.hamiltonian_cycle = [0, 1, 0, 2, 1, 0, 1, 2]
        self.principal_hcurve = hilbert_curve(dim, self.nbits)
        self.length = len(self.hamiltonian_cycle)
        self.curves = precompute_hcurves(
            self.hamiltonian_cycle,
            self.principal_hcurve,
            self.nbits)

    def __init_zcurve(self, dim, data_shape):
        if data_shape is None:
            raise ValueError("data_shape must be provided for zcurve ordering")
        if len(data_shape) != dim:
            raise ValueError("data_shape length must match dim for zcurve ordering")

        # Determine required bit-depth based on largest dimension
        self.nbits = max((s - 1).bit_length() for s in data_shape)
        self.size = 1 << self.nbits

        self.length = np.math.factorial(dim)
        self.curves = precompute_zcurves(data_shape, self.nbits)

    def __hscan(self, data, hcurve):
        """ Hilbert scaned view of data. """
        return data[tuple( hcurve )].T.ravel()

    def __zscan(self, data_flattened, zcurve):
        """ zscan view of data. """
        return data_flattened[zcurve]

    def scan(self, data, curve):
        if self.ordering == 'hilbert':
            return self.__hscan(data, curve)
        else:
            return self.__zscan(data, curve)

    def __call__(self, data):
        cid_av, cid_std, cid_shuffle, cid_vals = super().calc_cid(data,)
        cid_sem = cid_std / np.sqrt(self.length)
        return cid_av, cid_sem, cid_shuffle, cid_vals


class CID_old(ComputableInformationDensity_old):
    
    hamiltonian_cycle = [0, 1, 0, 2, 1, 0, 1, 2]
    length = len(hamiltonian_cycle)

    slurm_cpus = os.getenv("SLURM_CPUS_PER_TASK")
    ncpus = min(int(slurm_cpus) if slurm_cpus else cpu_count(), length)
    
    def __init__(self, dim, nbits, nshuff, mode='lz77', verbose=False):
        super().__init__(dim, nbits, nshuff, mode, verbose)
        if verbose:
            print(f"Using {self.ncpus} workers for CID calculations")
    
    def __call__(self, data):
        cid_av, cid_std, cid_shuffle, cid_vals = super().__call__(data, n_workers=self.ncpus)
        cid_sem = cid_std / np.sqrt(self.length)
        return cid_av, cid_sem, cid_shuffle, cid_vals

    def __call2__(self, data):
        cid_av, cid_std, cid_shuffle = super().__call2__(data, n_workers=self.ncpus)
        cid_sem = cid_std / np.sqrt(self.length)
        return cid_av, cid_sem, cid_shuffle

def cid2d(nshuff, nbits=None, data_shape=None, ordering='hilbert',mode='lz77', verbose=False):
    """ Two-dimensional CID Analysis \n
    Args:
        order: linear system size (log2) (int).
        nshuff: number of radnom shuffles of data.
    Returns: instance of the CID class """
    return CID(dim=2, nbits=nbits, data_shape=data_shape, nshuff=nshuff, ordering=ordering, mode=mode, verbose=verbose)

def sequential_time(nbits, nshuff, data_shape=None, ordering='hilbert', mode='lz77', verbose=False):
    """ Spatiotemporal CID Analysis \n
    Args:
        order: linear system size (log2) (int).
        nshuff: number of radnom shuffles of data.
    Returns: instance of the CID class """
    return CID(dim=2, nbits=nbits, data_shape=data_shape, nshuff=nshuff, ordering=ordering, mode=mode, verbose=verbose)

def interlaced_time(nbits, nshuff, data_shape=None, ordering='hilbert', mode='lz77', verbose=False):
    """ Spatiotemporal CID Analysis \n
    Args:
        order: linear system size (log2) (int).
        nshuff: number of radnom shuffles of data.
    Returns: instance of the CID class """
    return CID(dim=3, nbits=nbits, data_shape=data_shape, nshuff=nshuff, ordering=ordering, mode=mode, verbose=verbose)
