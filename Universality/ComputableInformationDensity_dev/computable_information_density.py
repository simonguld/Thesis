import os
from multiprocessing import Pool, cpu_count

import numpy as np

from lempel_ziv_complexity.main import lz77, lz78
from lempel_ziv_complexity.LempelZiv import lz_hybrid, lz77_complexity_linear

def cid(sequence, version='C', implementation='lz77'):
    """ Computable Information Density \n
    Args: one-dimensional data array.
    Returns: CID measure of the sequence.
    """
    cid_func = lz77 if implementation == 'lz77' else lz78
    C, L = cid_func(sequence, version), len(sequence)
    return C*(np.log2(C) + 2*np.log2(L/C)) / L

def cid78(sequence, version='python'):
    """ Computable Information Density \n
    Args: one-dimensional data array.
    Returns: CID measure of the sequence.
    """
    C, L = lz78(sequence, version), len(sequence)
    return C*(np.log2(C) + 2*np.log2(L/C)) / L

def cid_linear(sequence):
    """ Computable Information Density \n
    Args: one-dimensional data array.
    Returns: CID measure of the sequence.
    """
    C, L = lz77_complexity_linear(sequence), len(sequence)
    return C*(np.log2(C) + 2*np.log2(L/C)) / L

def cid_hybrid(sequence, window_size=1024):
    """ Computable Information Density \n
    Args: one-dimensional data array.
    Returns: CID measure of the sequence.
    """
    C, L = lz_hybrid(sequence, window_size=window_size), len(sequence)
    return C*(np.log2(C) + 2*np.log2(L/C)) / L

def cid_shuffle(sequence, nshuff, cid_mode='lz77', ddof = 1):
    """Computable Information Density via random shuffling.
    
    Args:
        sequence : np.ndarray
            One-dimensional data array.
        nshuff   : int
            Number of shuffles.
    Returns:
        float : Mean CID value over shuffled sequences.
    """

    slurm_cpus = os.getenv("SLURM_CPUS_PER_TASK")
    ncpus = min(int(slurm_cpus) if slurm_cpus else cpu_count(), nshuff)

    cid_dict = {'lz77': cid,
                'lz78': cid78,
                    'linear': cid_linear,
                    'hybrid': cid_hybrid}
    cid_func = cid_dict[cid_mode]

    rng = np.random.default_rng()  # independent RNG
    seq = sequence.copy()

    # generator yielding shuffled copy of sequence:
    def shuffle():
        for _ in range(nshuff):
            rng.shuffle(seq)
            # yield copy so not to share memory
            # among process-pool of workers:
            yield seq.copy()
    
    # create and configure a pool of workers:
    with Pool(ncpus) as pool:
        cid_pool = pool.map_async(cid_func, shuffle())
        pool.close()    # close the process pool
        pool.join()     # wait for all workers to complete
    cid_values = cid_pool.get()
    return np.mean(cid_values), np.std(cid_values, ddof=ddof if nshuff > 1 else 0)

def cid_correlation(sequence):
    """ CID Correlation Length \n
    Args: one-dimensional data array.
    Returns: correlation length using CID measure.
    """
    # Implement metthod from:
    # https://doi.org/10.1103/PhysRevLett.125.170601
    raise NotImplementedError('cid_correlation')

