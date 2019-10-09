import pandas as pd
import numpy as np
from scipy.optimize import minimize, Bounds
from scipy.spatial.distance import cdist

import fieldmodel.fielderrors as fe
import fieldmodel.models as models
import fieldmodel.utilities as util

import matplotlib as mpl
import matplotlib.pyplot as plt

class FieldModel(object):

    """
    Class to perform Connective Field Modeling using a geodesic-based
    distance function.

    Compute the field model for a given source-to-target mapping.
    The source is decribed by a point source feature vector, while a target
    is described by a set of point source feature vectors.  For a given 
    source-to-target mapping, a scalar field model is fit to the similarity
    between the source and target feature vectors.
    
    For a given scalar field, we fit an, isotropic geodesic Gaussian
    to the scalar field.  The fit is performed by minimizing the
    correlation distance between field and mixture density.

    Parameters:
    - - - - -
    amplitude: bool
        whether to fit amplitude or not
    r: float
        intial estimate of sigma
    peak_size: float / int
        minimum geodesic distance between peaks
        larger values results in fewer peaks
    hood_size: float / int
        maximum distance away from peaks to include in search
        large values between larger neighborhood
    """

    def __init__(self, r=10, amplitude=False,
                 peak_size=15, hood_size=20):

        self.r = 10
        self.amplitude = amplitude
        self.peak_size = peak_size
        self.hood_size = hood_size
    
    def fit(self, distances, data, x, y):

        """
        Parameters:
        - - - - -
        distance, float, array
            distance matrix between all pairs of points in a region
        data: float, array
            scalar field to fit parameters to
        """

        self.data = data
        self.x = x
        self.y = y

        [n, _] = distances.shape

        if self.amplitude:
            params = np.zeros((n, 3))
        else:
            params = np.zeros((n, 2))
        [n, d] = params.shape

        peaks = util.find_peaks(distances, data, n_size=self.peak_size)
        [gmax, _] = util.global_peak(distances, data, peaks, n_size=10)
        nhood = util.peak_neighborhood(distances, [gmax], n_size=self.hood_size)

        idx = np.zeros((n,))
        idx[nhood] = 1
        idx = ~idx.astype(np.bool)
        params[idx, :] = np.nan

        print('Searching over %.2f%% samples.' % (100*len(nhood) / data.shape[0]))
        
        for idx in nhood:
            [p, c] = self.mini(distances[idx, :])
            params[idx, :-1] = p
            params[idx, -1] = c

        self.params = params
        self.mu_ = np.nanargmin(params[:, -1])
        self.sigma_ = params[self.mu_, -1*(d)]
        self.dist_ = distances[self.mu_, :]

        self.fitted = True

    def mini(self, dist):

        a0 = [self.r*2]
        bds = Bounds(0.001, np.min([self.x.std(), self.y.std()]))

        T = minimize(self.error, a0, 
                 args=(dist, self.data),
                 bounds=(bds))

        p = T.x
        c = T.fun

        return [p, c]

    def error(self, params, dist, field):

        """
        
        """
    
        density = models.geodesic(dists=dist, params=params)
        merror = fe.correlation(field, density)
            
        return merror

    def plot(self):

        """
        Plot scalar field and density, given fitted parameters:.
        """

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        dcmap = mpl.cm.jet
        dnorm = mpl.colors.Normalize(vmin=self.data.min(), vmax=self.data.max())

        img1 = ax1.scatter(self.x, self.y, c=self.data, marker='.', cmap='jet', norm=dnorm)
        ax1.set_title('Scalar Field', fontsize=15)
        ax1.set_xlabel('X', fontsize=15)
        ax1.set_ylabel('Y', fontsize=15)
        plt.colorbar(img1, ax=ax1)

        L = models.geodesic(self.dist_, [self.sigma_])

        lcmap = mpl.cm.jet
        lnorm = mpl.colors.Normalize(vmin=L.min(), vmax=L.max())

        img2 = ax2.scatter(self.x, self.y, c=L, marker='.', cmap='jet', norm=lnorm)
        ax2.set_title('Estimated Density', fontsize=15)
        ax2.set_xlabel('X', fontsize=15)
        ax2.set_ylabel('Y', fontsize=15)
        plt.colorbar(img2, ax=ax2)

        plt.tight_layout()
        plt.show()

    def get_params(self):

        """
        Return parameters as dictionary.
        """

        assert self.fitted, 'Must fit model before writing coefficients.'

        param_names=['x_mean', 'y_mean', 'sigma',
                     'nu', 'amplitude', 'density',
                     'opterror']

        param_vals = [self.coefs_[0], self.coefs_[1], self.coefs_[2],
                      self.nu, self.amp_, self.density,
                      self.opterror]
        
        param_map = dict(zip(param_names, param_vals))

        return param_map

    def write(self, outfile):

        """
        Write fitted model to CSV file.

        Parameters:
        - - - - -

        outfile: string
            name of fitted parameter file
        """

        pmap = self.get_params()
        df = pd.DataFrame(pmap)
        df.to_csv(outfile)