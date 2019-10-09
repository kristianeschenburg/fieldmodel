import numpy as np
from scipy.spatial.distance import cdist

def correlation(sfield, prob):

    """
    Compute the correlation distance between the estimated density
    and the scalar field on which to fit the density.

    Parameter:
    - - - - -
    field: float, array
        Scalar field on which to compute the density

    density: float, array
        estimated density of current model
    """

    prob = prob.squeeze()
    sfield = sfield.squeeze()

    merror = cdist(sfield[None, :], prob[None, :], metric='correlation')
    merror = merror.squeeze()

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

    merror = sfield - prob
    merror = merror**2
    merror = merror.sum()

    return merror