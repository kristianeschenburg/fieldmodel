import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import kendalltau
from scipy.stats.mstats import spearmanr

def pearson(sfield, prob):

    """
    Compute Pearson's correlation distance between the estimated density
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

    p = prob.squeeze()[None, :]
    s = sfield.squeeze()[None, :]

    # compute Pearson distance
    merror = cdist(s, p, metric='correlation')

    return merror

def kendall(sfield, prob):

    """
    Compute Pearson's correlation distance between the estimated density
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

    p = prob.squeeze()[None, :]
    s = sfield.squeeze()[None, :]

    # compute Kendall distance
    K = kendalltau(s, p)
    merror = 1-K[0]

    return merror

def spearman(sfield, prob):

    """
    Compute Pearson's correlation distance between the estimated density
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

    p = prob.squeeze()[None, :]
    s = sfield.squeeze()[None, :]

    # Compute Spearman distance
    S = spearmanr(s, p)
    merror = 1-S[0]

    return merror

def L2(sfield, prob):

    """
    Compute the L2-norm error between the estimated density
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

    # Compute L2 distance
    merror = s-p
    merror = (merror**2).sum()

    return merror

def L1(sfield, prob):

    """
    Compute the L1-norm error between the estimated density
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

    # Compute L1 distance
    merror = np.abs(s-p)
    merror = merror.sum()

    return merror
