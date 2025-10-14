import numpy as np

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
        for k in hamiltonian_cycle:
            hcurve = principal_curve  # view of principal_curve i.e. O(1)
            
            if k == 0: hcurve[0] = (size - 1) - hcurve[0]
            if k == 1: hcurve[1] = (size - 1) - hcurve[1]
            if k == 2: hcurve[[0,1]] = hcurve[[1,0]]
            yield data[tuple( hcurve )].T.ravel()  # view of data, i.e. O(1)