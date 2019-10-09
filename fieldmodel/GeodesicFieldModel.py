import pandas as pd
import numpy as np
from scipy.optimize import fmin
from scipy.spatial.distance import cdist

import models
import fielderrors as fe

import matplotlib as mpl
import matplotlib.pyplot as plt

from multiprocessing import Pool
from multiprocessing.pool import ThreadPool

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
    """

    def __init__(self, r=10, amplitude=True):

        """
        
        """

        self.r = 10
        self.amplitude = amplitude
    
    def fit(self, distances, data):

        """
        Parameters:
        - - - - -
        distance, float, array
            distance matrix between all pairs of points in a region
        data: float, array
            scalar field to fit parameters to
        p0: list
            must be of length (n_components) * (n_params)
            where n_params is the number of parameters in
            the Gaussian model
        """

        pool = ThreadPool()

        [n,_] = distances.shape

        A = [distances[i, :] for i in np.arange(10)]
        P = pool.map(self.minimize, A, data)

        self.cost_ = P
        self.mu_ = np.argmin(self.cost_)

        self.fitted = True

    def minimize(self, dist, field):

        if self.amplitude:
            a0 = [1, self.r*2]
        else:
            a0 = [self.r*2]

        T = fmin(self.error, a0, 
                 args=(dist, field), maxiter=10000,
                 disp=False)

        print(T)
        cost = T

        return cost

    def error(self, params, dist, field):

        """
        
        """
    
        density = models.geodesic(dists=dist, params=params)
        merror = fe.correlation(field, density)
            
        return

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

        if self.density == 'gaussian':
            L = models.gaussian(self.x, self.y, self.coefs_).squeeze()
        else:
            L = models.student(self.x, self.y, self.nu, self.coefs_).squeeze()
        
        if self.fit_amplitude:
            L = self.amp_ * L
        
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