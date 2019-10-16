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
                 peak_size=15, hood_size=20,
                 verbose=False):

        self.r = 10
        self.peak_size = peak_size
        self.hood_size = hood_size
        self.verbose = verbose
        self.amplitude = amplitude

    def fit(self, distances, data, x, y):

        """
        Estimate the location and scale parameters of the geodesic Gaussian.
        Finds a list of local maxima, estimates the global maxima, and then
        searches the neighborhood of this global maximum to best fit the
        location and scale parameters.

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
        self.up = distances.max()/2

        [n, _] = distances.shape

        if self.amplitude:
            params = np.zeros((n, 2))
        else:
            params = np.zeros((n, 1))

        costs = np.repeat(np.nan, n)

        # find local maxima in scalar field
        peaks = util.find_peaks(distances, data, n_size=self.peak_size)

        # estimate global maximum
        # based on highest mean signal in neighborhood of each local maxima
        [gmax, _] = util.global_peak(distances, data, peaks, n_size=10)

        # restrict location search space to neighborhood of global max
        nhood = util.peak_neighborhood(distances, [gmax], n_size=self.hood_size)

        idx = np.zeros((n,))
        idx[nhood] = 1
        idx = ~idx.astype(np.bool)
        params[idx, :] = np.nan

        if self.verbose:
            print('Searching over %.2f%% samples.' % (100*len(nhood) / data.shape[0]))

        # iterate over neighborhood points
        # compute cost for each point
        for idx in nhood:
            tempopt = self.mini(distances[idx, :])
            params[idx, :] = tempopt['params']
            costs[idx] = tempopt['cost']

        self.nhood_ = nhood
        self.params_ = params

        mu = np.nanargmin(costs)

        costs[np.isnan(costs)] = 0
        self.costs_ = costs

        if self.amplitude:
            sigma = params[mu, 1]
            amplitude = params[mu, 0]
        else:
            sigma = params[mu, 0]
            amplitude = 1

        self.mu_ = mu
        self.optimal_ = [amplitude, sigma]
        self.amplitude_ = amplitude
        self.sigma_ = sigma
        self.dist_ = distances[mu, :]

        self.fitted = True

    def mini(self, dist):

        if self.amplitude:
            a0 = [1, self.r*2]
        else:
            a0 = [self.r*2]

        bds = Bounds(0.001, self.up)

        T = minimize(self.error, a0,
                 args=(dist, self.data),
                 bounds=(bds))

        optimal = {'params': T.x,
                   'cost': T.fun}

        return optimal

    def error(self, params, dist, field):

        """
        Compute error of current density estimate.

        Parameters:
        - - - - -
        params: list
            current parameter estimates
        dist: float, array
            geodesic distance vector
        field: float, array
            scalar field on which to fit density
        """

        density = models.geodesic(dists=dist, params=params)
        merror = fe.correlation(field, density)

        return merror

    def pdf(self):

        """
        Return density of fitted model.
        """

        prob = models.geodesic(self.dist_, self.optimal_)
        prob = prob/prob.sum()

        return prob

    def weight(self):

        """
        Computed weighted signal average of estimated density.
        """

        pdf = self.pdf()
        l = (pdf * self.data).sum()

        return l

    def plot(self):

        """
        Plot scalar field and density, given fitted parameters:
        """

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        dnorm = mpl.colors.Normalize(vmin=self.data.min(), vmax=self.data.max())

        img1 = ax1.scatter(self.x, self.y, c=self.data, cmap='jet', norm=dnorm)
        ax1.set_title('Scalar Field', fontsize=15)
        ax1.set_xlabel('X', fontsize=15)
        ax1.set_ylabel('Y', fontsize=15)
        plt.colorbar(img1, ax=ax1)

        L = self.pdf()

        lnorm = mpl.colors.Normalize(vmin=L.min(), vmax=L.max())

        img2 = ax2.scatter(self.x, self.y, c=L, cmap='jet', norm=lnorm)
        title2 = 'Estimated Density\nSigma: %.2f (mm)' % (self.sigma_)
        ax2.set_title(title2, fontsize=15)
        ax2.set_xlabel('X', fontsize=15)
        ax2.set_ylabel('Y', fontsize=15)
        plt.colorbar(img2, ax=ax2)

        plt.tight_layout()
        plt.show()

    def plot_costs(self):

        """
        Plot scalar field and cost of possible neighborhood options.
        """

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        dnorm = mpl.colors.Normalize(vmin=self.data.min(),
                                     vmax=self.data.max())

        img1 = ax1.scatter(self.x, self.y, c=self.data, cmap='jet', norm=dnorm)
        ax1.set_title('Scalar Field', fontsize=15)
        ax1.set_xlabel('X', fontsize=15)
        ax1.set_ylabel('Y', fontsize=15)
        plt.colorbar(img1, ax=ax1)

        costs = self.costs_
        cnorm = mpl.colors.Normalize(vmin=0,
                                     vmax=np.nanmax(costs))

        img2 = ax2.scatter(self.x, self.y, c=costs, cmap='jet', norm=cnorm)
        ax2.set_title('Costs', fontsize=15)
        ax2.set_xlabel('X', fontsize=15)
        ax2.set_ylabel('Y', fontsize=15)
        plt.colorbar(img2, ax=ax2)

        plt.tight_layout()
        plt.show()


    def plot_sigmas(self):

        """
        Plot scalar field and sigmas of possible neighborhood.
        """

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        dnorm = mpl.colors.Normalize(vmin=self.data.min(),
                                     vmax=self.data.max())

        img1 = ax1.scatter(self.x, self.y, c=self.data, cmap='jet', norm=dnorm)
        ax1.set_title('Scalar Field', fontsize=15)
        ax1.set_xlabel('X', fontsize=15)
        ax1.set_ylabel('Y', fontsize=15)
        plt.colorbar(img1, ax=ax1)

        if self.amplitude:
            sigmas = self.params_[:, 1]
        else:
            sigmas = self.params_[:, 0]
        sigmas[np.isnan(sigmas)] = 0

        cnorm = mpl.colors.Normalize(vmin=np.nanmin(sigmas), vmax=np.nanmax(sigmas))

        img2 = ax2.scatter(self.x, self.y, c=sigmas, cmap='jet', norm=cnorm)
        ax2.set_title('Sigmas (mm)', fontsize=15)
        ax2.set_xlabel('X', fontsize=15)
        ax2.set_ylabel('Y', fontsize=15)
        plt.colorbar(img2, ax=ax2)

        plt.tight_layout()
        plt.show()


    def plot_amplitudes(self):

        """
        Plot scalar field and amplitude of possible neighborhood.
        """

        if not self.amplitude:
            print('Amplitudes were not fit.')
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        dnorm = mpl.colors.Normalize(vmin=self.data.min(),
                                     vmax=self.data.max())

        img1 = ax1.scatter(self.x, self.y, c=self.data, cmap='jet', norm=dnorm)
        ax1.set_title('Scalar Field', fontsize=15)
        ax1.set_xlabel('X', fontsize=15)
        ax1.set_ylabel('Y', fontsize=15)
        plt.colorbar(img1, ax=ax1)

        amplitudes = self.params_[:, 0]
        amplitudes[np.isnan(amplitudes)] = 0

        cnorm = mpl.colors.Normalize(vmin=np.nanmin(amplitudes), vmax=np.nanmax(amplitudes))

        img2 = ax2.scatter(self.x, self.y, c=amplitudes, cmap='jet', norm=cnorm)
        ax2.set_title('Amplitudes', fontsize=15)
        ax2.set_xlabel('X', fontsize=15)
        ax2.set_ylabel('Y', fontsize=15)
        plt.colorbar(img2, ax=ax2)

        plt.tight_layout()
        plt.show()

    def get_params(self):

        """
        Return parameters as dictionary.

        Returns:
        - - - -
        df: Pandas DataFrame
            Includes field location ('mu'), scale ('sigma'),
            cost ('cost') and weighted signal average ('signal')
        """

        assert self.fitted, 'Must fit model before writing coefficients.'

        param_names=['mu', 'sigma', 'cost', 'signal']

        param_vals = [self.mu_, self.sigma_, self.costs_[self.mu_], self.weight()]
        param_vals = [[x] for x in param_vals]

        param_map = dict(zip(param_names, param_vals))
        df = pd.DataFrame(param_map)

        return df

    def write(self, outfile):

        """
        Write fitted model to CSV file.

        Parameters:
        - - - - -

        outfile: string
            name of fitted parameter file
        """

        pmap = self.get_params()
        pmap.to_csv(outfile, index_label=False)