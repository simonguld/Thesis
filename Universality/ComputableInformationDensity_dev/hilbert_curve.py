import numpy as np
import zCurve as zC
from itertools import permutations

def hilbert_curve(n, p):
    """ Principal Hilbert Curve. \n
    
    Vectorized implementation of the principal Hilbert curve by J. Skilling.
    
    Reference:
    J. Skilling, "Programming the Hilbert curve". AIP Conf. Proc., 2004, 707, 
    381-387.
    
    Algorithm:
    A single global Gray code is being applied to the np-bit binary rep. of the 
    Hilbert distance/index H. This overtransforms the distance H and the excess 
    work is undone by a single traverse through the np-bit Gray code rep. of H.
    
    Args:
        n: dimensionality of the curve (int).
        p: number of bits in each dimension (int).
    
    Returns:
        Two-dimensional numpy array with shape (n, 2**(n * p)) Entry (i,j) contains the the i'th component
        of the coordinate to the j'th point along the principal Hilbert curve.
    """
    
    H = np.arange(1 << n*p) # distance along the Hilbert curve
    H ^= H >> 1             # Gray code / reflected binary code
    
    # unpackbits:
    H = H & 1 << np.arange(n*p)[:, None] != 0
    
    # collect each np-bit integer into n preliminary p-bit integers:
    H = H.reshape(p, n, -1)
    
    # packbits:
    H = np.sum(H.T * 1 << np.arange(p), -1)
    
    H = H.T # this makes life less awkward
    
    # undo excess work:
    for q in 2 << np.arange(p - 1):
        for m in range(n):
            # if bit q of of coordinate m is OFF
            mask = H[m] & q == 0
            # then exchange low bits of coordinate n and m:
            H[:, mask] ^= (H[-1, mask] ^ H[m, mask]) & q - 1
            # else invert low bits of coordinate n:
            H[-1, np.logical_not(mask)] ^= q - 1
    
    return H

def itter_hscan(data_arr, dim, nbits):
    """ yields all 8 distinct Hilbert scanned views of the data. 
    Since a view is returned, this operation is O(1). """
    
    data = np.transpose(data_arr).reshape((-1, ) + (1<<nbits, ) * dim).T
    hamiltonian_cycle = [0, 1, 0, 2, 1, 0, 1, 2]
    principal_curve = hilbert_curve(dim, nbits)
    size = 1 << nbits  # 2**nbits
    hcurves_list = []
    for k in hamiltonian_cycle:
        hcurve = principal_curve  # view of principal_curve i.e. O(1)

        if k == 0: hcurve[0] = (size - 1) - hcurve[0]
        if k == 1: hcurve[1] = (size - 1) - hcurve[1]
        if k == 2: hcurve[[0,1]] = hcurve[[1,0]]
        hcurves_list.append(data[tuple( hcurve )].T.ravel())  # view of data, i.e. O(1)
    return hcurves_list

def itter_hscan_gen(data_arr, dim, nbits):
    """ yields all 8 distinct Hilbert scanned views of the data. 
    Since a view is returned, this operation is O(1). """
    
    data = np.transpose(data_arr).reshape((-1, ) + (1<<nbits, ) * dim).T
    hamiltonian_cycle = [0, 1, 0, 2, 1, 0, 1, 2]
    principal_curve = hilbert_curve(dim, nbits)
    size = 1 << nbits  # 2**nbits
    for k in hamiltonian_cycle:
        hcurve = principal_curve  # view of principal_curve i.e. O(1)
        
        if k == 0: hcurve[0] = (size - 1) - hcurve[0]
        if k == 1: hcurve[1] = (size - 1) - hcurve[1]
        if k == 2: hcurve[[0,1]] = hcurve[[1,0]]
        yield data[tuple( hcurve )].T.ravel()  # view of data, i.e. O(1)

def precompute_hcurves_standalone(dim, nbits):
    """Precompute all hcurve coordinate variants exactly once."""
    hamiltonian_cycle = [0, 1, 0, 2, 1, 0, 1, 2]
    principal_curve = hilbert_curve(dim, nbits)
    size = 1 << nbits  # 2**nbits
    hcurves = []

    idx_arr = np.arange((1 << nbits) ** dim).reshape( (1 << nbits, ) * dim )
    h = principal_curve.copy()

    for k in hamiltonian_cycle:
        if k == 0:
            h[0] = (size - 1) - h[0]
        if k == 1:
            h[1] = (size - 1) - h[1]
        if k == 2:
            h[[0, 1]] = h[[1, 0]]
        hcurves.append(idx_arr[tuple( h )].T.ravel())
    return hcurves

def precompute_hcurves(hamiltonian_cycle, principal_curve, nbits):
    """Precompute all hcurve coordinate variants exactly once."""

    size = 1 << nbits  # 2**nbits
    hcurves = []

    h = principal_curve.copy()

    for k in hamiltonian_cycle:
        if k == 0:
            h[0] = (size - 1) - h[0]
        if k == 1:
            h[1] = (size - 1) - h[1]
        if k == 2:
            h[[0, 1]] = h[[1, 0]]
            
        hcurves.append(h.copy())
    return hcurves

def precompute_hcurves_gen(dim, nbits):
    """Precompute all hcurve coordinate variants exactly once."""

    hamiltonian_cycle = [0, 1, 0, 2, 1, 0, 1, 2]
    principal_curve = hilbert_curve(dim, nbits)
    size = 1 << nbits  # 2**nbits
    hcurves = []

    h = principal_curve.copy()

    for k in hamiltonian_cycle:
        if k == 0:
            h[0] = (size - 1) - h[0]
        if k == 1:
            h[1] = (size - 1) - h[1]
        if k == 2:
            h[[0, 1]] = h[[1, 0]]
            
        hcurves.append(h.copy())
    return hcurves

def precompute_zcurves(data_shape, bits_per_dim,):
    """
    Compute Morton codes for every voxel coordinate in D dimensions,
    for all D! permutations of coordinate ordering.

    Parameters
    ----------
    data_Shape : tuple of ints
        The side lengths along each dimension, e.g. (Nx, Ny, Nz) or (N0, N1, ..., N_{D-1}).
    bits_per_dim : int
        Number of bits allocated per dimension for Morton encoding.
    Returns
    -------
    orders : list of 1D numpy arrays 
        For each permutation of coordinate axes, an ordering (argsort) of Morton codes.
    """
    Ndims = data_shape
    D = len(Ndims)
    perms = list(permutations(range(D)))
    Nperm = len(perms)

    # find total number of points
    Npoints = np.prod(Ndims)

    # set dtype based on Npoints
    if Npoints <= 2**8: dtype = np.uint8
    elif Npoints <= 2**16: dtype = np.uint16
    elif Npoints <= 2**32: dtype = np.uint32
    elif Npoints <= 2**64: dtype = np.uint64
    elif Npoints <= 2**128: dtype = np.uint128
    else:
        raise ValueError("Too many points for Morton encoding.")

    # Create a coordinate grid
    grids = np.meshgrid(*[np.arange(n, dtype=dtype) for n in Ndims], indexing='ij')
    # Stack into shape (Npoints, D)
    coords = np.stack(grids, axis=-1).reshape(-1, D)

    # Convert to tuple list once, because par_interlace expects list[(coord tuple)...]
    coord_list = [tuple(int(c[d]) for d in range(D)) for c in coords]

    # Allocate Morton codes array: one column per permutation
    codes_flat = np.empty((len(coord_list), Nperm), dtype=np.int64)

    # Compute Morton codes for each permutation
    for i, p in enumerate(perms):
        permuted_list = [tuple(c[pj] for pj in p) for c in coord_list]
        codes_flat[:, i] = zC.par_interlace(permuted_list, dims=D, bits_per_dim=bits_per_dim)

    # Convert codes to sort orders
    orders = [np.argsort(codes_flat[:, i]).astype(dtype) for i in range(Nperm)]
    return orders

def zscan(data, zcurve):
    """ zscan view of data. """
    return data.ravel()[zcurve]