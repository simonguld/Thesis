from scipy.stats import binned_statistic
import numpy as np

def xcorr(F, G):
    """ Cross-correlation \n
    Args:
        F: two-dimensional array containing samples of a scalar function.
        G: two-dimensional array containing samples of a scalar function.
    Returns:
        The cross-correlation of F and G.
    """
    C = np.fft.rfft2(F)
    C = np.conj(C)
    C *= np.fft.rfft2(G)
    return np.fft.irfft2(C)


def xcov(F, G):
    """ Cross-covariance \n
    Args:
        F: two-dimensional array containing samples of a scalar function.
        G: two-dimensional array containing samples of a scalar function.
    Returns:
        The cross-covariance of F and G.
    """
    return xcorr(
        F - np.mean(F),
        G - np.mean(G),
    )


def autocorr(F):
    """ Autocorrelation \n
    Args:
        F: two-dimensional array containing samples of a scalar function.
    Returns:
        The autocorrelation of F.
    """
    return xcorr(F, F)


def autocov(F):
    """ Autocovariance \n
    Args:
        F: two-dimensional array containing samples of a scalar function.
    Returns:
        The autocovariance of F.
    """
    return xcov(F, F)


def autocorr2(Fx, Fy):
    """ Autocorrelation of a two-component vector valued function. \n
    Args:
        Fx, Fy: the component of a vector valued function.
    Returns:
        The autocorrelation of a two-component vector valued function.
    """
    return autocorr(Fx) + autocorr(Fy)


def autocov2(Fx, Fy):
    """ Autocovariance of a two-component vector valued function. \n
    Args:
        Fx, Fy: the component of a vector valued function.
    Returns:
        The autocovariance of a two-component vector valued function.
    """
    return autocov(Fx) + autocov(Fy)


def rdf2d(Fx, Fy=None, origin=False, step=1.):
    """ Radial distribution function (2D) \n
    Args:
        F: two-dimensional array containing samples of a scalar function.
        step: disctrization of radial displacement/seperation (default = 1.0).
    Returns:
        Radial distribution function and r-values.
    """
    if Fy is None: C = autocov(Fx)
    else: C = autocov2(Fx, Fy)
    
    C = np.divide(C, C[0,0], where=C[0,0]!=0)   # normalize

    L = C.shape[0]  # linear system size
    
    r_max = L//2
    r_bins = np.arange(0.5, r_max, step)        # list of bin edges
    r_vals = .5 * (r_bins[1:] + r_bins[:-1])    # list of bin midpoints
    
    # two-dimensional array containing the radial distance 
    # w.r.t the top left corner on a periodic square domain.
    r = np.arange(0, L, 1)
    r = np.minimum(r%L, -r%L)
    r_nrm = np.abs(r[:,None] + 1J*r[None,:])
    
    # bin the autocovariance in radial-space.
    rdf, _, _ = binned_statistic(r_nrm.flatten(), C.flatten(), 'mean', r_bins)
    
    if origin: # insert rdf(r=0) = 1.
        rdf = np.insert(rdf, 0, 1.)
        r_vals = np.insert(r_vals, 0, .0)
    
    return rdf, r_vals

