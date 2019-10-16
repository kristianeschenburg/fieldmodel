import numpy as np
from scipy.spatial.distance import cdist

def correlation(sfield, prob):

    """
    Compute the correlation distance between the estimated density
    and the scalar field on which to fit the density.

    Correlation distance here is defined as 1 - correlation.
    Since the codomain of a correlation function is [-1, 1],
    the codomain of the correlation distance [0, 2].

    Parameter:
    - - - - -
    field: float, array
        Scalar field on which to compute the density

    density: float, array
        estimated density of current model
    """

    prob = prob.squeeze()
    sfield = sfield.squeeze()

    p = np.ma.masked_invalid(prob)
    s = np.ma.masked_invalid(sfield)
    c = np.ma.corrcoef(p, s)

    cmat = c.data
    merror = 1 - cmat[0, 1]

    return merror

def lsq(sfield, prob):

    """
    Compute the least square error between the estimated density
    and the scalar field on which to compute the density.

    Parameter:
    - - - - -
    field: float, array
        Scalar field on which to compute the density

    density: float, array
        estimated density of current model
    """

    prob = prob.squeeze()
    sfield = sfield.squeeze()

    p = np.ma.masked_invalid(prob)
    s = np.ma.masked_invalid(sfield)
    merror = s-p
    merror = merror**2

    merror = merror.sum()

    return merror