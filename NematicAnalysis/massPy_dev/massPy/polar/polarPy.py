from ..base_modules import defects, correlation

def get_charge(Px, Py, LX, LY):
    """
    Compute the charge array associated with the polarity field P. Defects
    then show up as small regions of non-zero charge (typically 2x2). \n
    Args:
        Px, Py: Components of the polarity field.
        LX, LY: Size of domain.
    Returns:
        Charge field with shape (LX, LY).
    """
    return defects.get_charge(Px, Py, LX, LY)


def get_polarity_tensor(Px, Py):
    """ whatever """
    return .5*(Px**2 - Py**2), Px*Py


def get_diffusive_charge_density(Px, Py, LX, LY):
    """ whatever """
    # Qxx, Qyx = get_polarity_tensor(Px, Py)
    # return defects.get_diffusive_charge_density(Qxx, Qyx, LX, LY)
    return defects.get_diffusive_charge_density(Px, Py, LX, LY)


def get_defects(Px, Py, LX, LY):
    """
    Returns list of defects from the polarity field. \n
    Args:
        Px, Py: Components of the polarity field.
        LX, LY: Size of domain.
    Returns:
        List of the form [ {'pos': (x, y), 'charge': w} ].
    """
    return defects.get_defects(Px, Py, LX, LY, threshold=.8)


def pair_correlation(Px, Py, LX, LY, radial=True):
    """ Pair correlation function \n
    Args:
        Px, Py: Components of the polarity field.
        LX, LY: Size of domain.
        radial: Whether or not to sort by radial distance (boolean).
    Returns:
        The autoconvolution of the polarity field.
    """
    if radial: return correlation.rdf2(Px.reshape(LX, LY), Py.reshape(LX, LY))
    else: return correlation.autocov2(Px.reshape(LX, LY), Py.reshape(LX, LY))

