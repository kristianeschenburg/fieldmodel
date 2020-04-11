import numpy as np

from . import models

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

from scipy.stats import ks_2samp


##########
# Methods for finding, filtering, and smoothing local maxima in scalar field.
##########

def peak_neighborhood(apsp, peak, n_size):

    """
    Find vertices in neighborhood of peak vertices.

    Parameters:
    apsp: float / int, array
        all-pairs shortest path matrix between all samples
    peaks: list
        indices of local maxima
    n_size: float / int
        maximum geodesic distance from peaks
    """

    dpeaks = apsp[peak, :]
    nhood = np.where(dpeaks < n_size)[0]

    return nhood

def find_peaks(dist, sfield, n_size):

    """
    Find the local maxima of a dataset.
    
    Parameters:
    - - - - -
    dist: int, array
        all-pairs shortest path matrix between all samples
    sfield: float, array
        scalar map from which to compute local maxima
    n_size: int
        minimum geodesic distance between local maxima
    """

    inds = np.arange(sfield.shape[0])

    # for each index in scalar field
    # find all neighbors within given distance
    # and store in dictionary
    neighbs = {k: None for k in inds}
    for k in inds:
        h = np.where(np.asarray(dist[k, :]) <= n_size)[0]
        neighbs[k] = h

    # identify most-correlated signal in neighborhood
    maxsignal = {k: None for k in inds}
    for k in inds:

        temp = np.ma.masked_invalid(sfield[neighbs[k]])
        temp = temp.argmax()
        maxsignal[k] = neighbs[k][temp]

    # get unique local maxima in scalar field
    # sort them in order of signal strength
    up = np.unique(list(maxsignal.values()))
    up = np.asarray(up[np.argsort((-1*sfield)[up])])

    passed = np.zeros(len(dist[0, :]))
    passed[up] = 1

    # identify those peaks that pass neighborhood size threshold
    for peak in up:
        # if current peak passes
        if passed[peak]:

            # find points that are farther than n_size away
            zinds = np.where((dist[peak, :] < n_size) & (dist[peak, :] > 0))[0]
            zinds = list(set(up).intersection(zinds))
            passed[zinds] = 0

    peaks = np.where(passed)[0]

    return peaks


def global_peak(apsp, sfield, peaks, n_size=5):

    """
    Get local maxima with largest (in magnitude) neighborhood.
    
    Parameters:
    - - - - -
    apsp: int, array
        all-pairs shortest path matrix between all samples
    sfield: float, array
        scalar map from which to compute local maxima
    peaks: list
        local maxima in scalar field
    """

    peak_map = {p: None for p in peaks}
    corr_map = {p: None for p in peaks}

    for p in peaks:

        idx = (apsp[p, :]<=n_size)
        peak_map[p] = sfield[idx].mean()
        corr_map[p] = sfield[p]

    maxima = max(peak_map, key=peak_map.get)

    return [maxima, peak_map]


def peak_KS(peaks, sfield, x, y, field_model):

    """
    Compute Kolmogorov–Smirnov test beteen scalar field and predicted field.
    
    Parameters:
    - - - - -
    peaks: list
        list of local maxima
    sfield: float, array
        scalar field
    x, y: float, array
        coordinates of data
    field_model: ConnectiveFieldModel
        parameterized CFM model
    """

    ks_map = {p: {'pval': None, 'coefs': None, 'amp': None} for p in peaks}
    for p in peaks:

        p0 = [x[p], y[p], 4.5]
        field_model.fit(data=sfield, x=x, y=y, p0=p0)
        diff = field_model.difference()
        K = ks_2samp(sfield, diff)
        ks_map[p]['pval'] = K[1]
        ks_map[p]['coefs'] = field_model.coefs_
        ks_map[p]['amp'] = field_model.amp_

    return ks_map


##########
# Methods for computing kernel density estimates of fieldmodel mappings.
##########


def kde(sregion, tregion, tdist, mapping, index_map, sigma=1.5):
    
    """
    Compute the Kernel Density Estimate, in target coordinate space,
    of the mapped vertices.  Each target vertex will be smoothed 
    using an isotropic Gaussian kernel of width ```sigma```.  We
    compute, for each target vertex, the number of mapped source 
    vertices.  The transformed value of each target is the convolution 
    of the isotropic Gaussian, centered at itself, with the count map.
    
    Parameters:
    - - - - -
    sregion / tregion: string
        names of source and target regions
    tdist: float, array
        target geodesic pairwise distance matrix
    mapping: DataFrame
        output from fieldmodel, containing mapping
        of each source verte to a target vertex
    index_map: dictionary
        mapping of region names to region indices
    sigma: float
        isotropic Gaussian standard deviation
        larger values will smooth out a point source more
        
    Returns:
    - - - -
    density: float, array
        kernel density estimate of target counts
    """

    tinds = index_map[tregion]
    
    # mapping of target indices to 0 : # targets
    t2i = dict(zip(tinds, np.arange(len(tinds))))
    
    # determine number of source vertices mapping to each target
    counts = np.zeros((len(tinds),))
    for i in mapping.index:
        mu = mapping.loc[i, 'mu']
        counts[t2i[mu]] += 1

    # iterate over target vertices, and convolve count map 
    # with isotropic Gaussian kernel
    density = np.zeros((counts.shape[0],))
    for i in np.arange(len(tinds)):
        
        pdf = models.geodesic(tdist[i, :], [sigma])
        
        d = (pdf*counts).sum()
        density[i] = d
        
    return density
