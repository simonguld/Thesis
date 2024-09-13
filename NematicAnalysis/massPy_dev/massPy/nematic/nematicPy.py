from ..base_modules import defects
import numpy as np

def get_director(Qxx, Qyx):
    """
    Compute the director field (S, nx, ny) from the Qij = 2*S(ni*nj - d_ij/2) tensor \n
    Args:
        Qxx, Qxy: The components of the nematic-tensor field.
    Returns:
        S, nx, ny all of the same shape as Qxx and Qyx.
    """
    S  = np.sqrt(Qxx**2 + Qyx**2)
    nx = np.sqrt(.5*(1 + Qxx/S))
    ny = np.sign(Qyx)*np.sqrt(.5*(1 - Qxx/S))
    return S, nx, ny


def get_charge(Qxx, Qyx, LX, LY):
    """
    Compute the charge array associated with the nematic-tensor field Q. Defects
    then show up as small regions of non-zero charge (typically 2x2). \n
    Args:
        Qxx, Qxy: The components of the nematic-tensor field.
        LX, LY: Size of domain.
    Returns:
        Charge field with shape (LX, LY).
    """
    # geet the director field
    _, nx, ny = get_director(Qxx, Qyx)
    
    return defects.get_charge(nx, ny, LX, LY)


def get_diffusive_charge_density(Qxx, Qyx, LX, LY):
    """ whatever """
    return defects.get_diffusive_charge_density(Qxx, Qyx, LX, LY)


def get_defect_polarity(Qxx, Qyx, LX, LY, d):
    """ 
    Compute the polarity angle for half-integer defects. \n
    Args:
        Qxx, Qxy: The components of the nematic field.
        LX, LY: Size of domain.
        d: defect dictionary {'pos': (x, y), 'charge': w}
    Returns:
        The polarity angle for half-integer defects.
    """
    # reshape the nematic field components
    Qxx, Qyx = Qxx.reshape(LX, LY), Qyx.reshape(LX, LY)
    
    # unwrap defect dictionary
    s = np.sign( d['charge'] )
    x, y = d['pos']
    
    # compute polarity angle, see doi:10.1039/c6sm01146b
    num, den = 0, 0
    for (dx, dy) in [(0, 0), (0, 1), (1, 1), (1, 0)]:
        # coordinates of nodes around the defect
        kk = (int(x) + LX + dx) % LX
        ll = (int(y) + LY + dy) % LY
        # derivative (3 point stencil) at these points
        dxQxx = .5*(Qxx[(kk+1) % LX, ll] - Qxx[(kk-1+LX) % LX, ll])
        dxQxy = .5*(Qyx[(kk+1) % LX, ll] - Qyx[(kk-1+LX) % LX, ll])
        dyQxx = .5*(Qxx[kk, (ll+1) % LY] - Qxx[kk, (ll-1+LY) % LY])
        dyQxy = .5*(Qyx[kk, (ll+1) % LY] - Qyx[kk, (ll-1+LY) % LY])
        # accumulate numerator and denominator
        num += s*dxQxy - dyQxx
        den += dxQxx + s*dyQxy
    
    return s/(2.-s)*np.arctan2(num, den)


def get_defects(Qxx, Qyx, LX, LY, get_polarity=False):
    """
    Returns list of defects from the nematic-tensor field. \n
    Args:
        Qxx, Qxy: The components of the nematic-tensor field.
        LX, LY: Size of domain.
        get_polarity: If True, compute polarity angle.
    Returns:
        List of the form [ {'pos': (x, y), 'charge': w, 'angle': psi} ].
    """
    # get the director field
    _, nx, ny = get_director(Qxx, Qyx)
    
    # get list of defects
    list_of_defects = defects.get_defects(nx, ny, LX, LY, threshold=.4)
    
    # compute polarity angle
    if get_polarity:
        for d in list_of_defects:
            d['angle'] = get_defect_polarity(Qxx, Qyx, LX, LY, d)
    
    return list_of_defects

