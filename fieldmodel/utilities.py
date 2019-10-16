import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

from scipy.stats import ks_2samp

def peak_neighborhood(apsp, peaks, n_size):

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

    dpeaks = apsp[peaks, :]
    idx = np.where(dpeaks < n_size)

    nhood = np.unique(idx)

    return nhood

def find_peaks(apsp, sfield, n_size):

    """
    Find the local maxima of a dataset.
    
    Parameters:
    - - - - -
    apsp: int, array
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
        h = np.where(np.asarray(apsp[k, :]) <= n_size)[0]
        neighbs[k] = h

    # for each index in scalar field
    # identify most correlated signal
    maxcorr = {k: None for k in inds}
    for k in inds:

        temp = np.ma.masked_invalid(sfield[neighbs[k]])
        temp = temp.argmax()
        maxcorr[k] = neighbs[k][temp]

    keys = np.asarray(list(maxcorr.keys()))
    values = np.asarray(list(maxcorr.values()))

    peaks = np.where(keys == values)[0]

    return peaks


def plot_peaks(x, y, sfield, peaks):

    """
    Plot peaks of scalar field.
    
    Parameters:
    - - - - -
    x, y: float, array
        coordinates of data
    sfield: float, array
        scalar field
    peaks: list
        indices of local maxima
    """

    cmap = mpl.cm.jet
    norm = mpl.colors.Normalize(vmin=sfield.min(), vmax=sfield.max())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.scatter(x, y, marker='.')
    for p in peaks:
        ax1.scatter(x[p], y[p], c='r', s=50,)
        ax1.annotate('%i' %(p), (x[p], y[p]), fontsize=15)
    ax1.set_title('Peaks', fontsize=15)

    img = ax2.scatter(x, y, c=sfield, marker='.', cmap='jet', norm=norm)
    ax2.set_title('Scalar Field', fontsize=15)
    plt.tight_layout()
    plt.colorbar(img, ax=ax2)
    plt.show()


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