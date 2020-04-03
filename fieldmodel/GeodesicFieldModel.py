import pandas as pd
import numpy as np
from scipy.optimize import minimize, Bounds
from scipy.spatial.distance import cdist

import fieldmodel.fielderrors as fe
import fieldmodel.models as models
import fieldmodel.utilities as util
import fieldmodel.plotting as plotting

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
    
    For a given scalar field, we fit an isotropic geodesic Gaussian
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
                 verbose=False, metric='pearson'):

        self.r = 10
        self.peak_size = peak_size
        self.hood_size = hood_size
        self.verbose = verbose
        self.amplitude = amplitude
        self.metric = metric

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
 
        # for each local maximum, to mitigate finding peaks that are noise
        # apply Laplacian smoothing (at each peak)
        # global max is peak with highest smoothed signal
        [gmax, _] = util.global_peak(distances, data, peaks, n_size=10)
        self.gmax = gmax
        self.ring = np.where(distances[gmax, :] == self.hood_size)[0]

        # restrict location search space to neighborhood of global max
        nhood = util.peak_neighborhood(distances, gmax, n_size=self.hood_size)

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


        """
        Optimization sub-method.
        """


        if self.amplitude:
            a0 = [1, self.r*2]
        else:
            a0 = [self.r*2]

        bds = Bounds(0.001, self.up)

        T = minimize(self.error, a0,
                 args=(dist, self.data, self.metric),
                 bounds=(bds))

        optimal = {'params': T.x,
                   'cost': T.fun}

        return optimal

    def fit_amplitude(self):

        """
        Fit post-hoc amplitude, after fitting sigma and mean parameters.
        """

        a0 = [1, self.r*2]

        bds = Bounds(0.001, self.up)

        T = minimize(self.error, a0,
                args=(self.dist_, self.data, self.metric),
                bounds=(bds))

        optimal = {'params': T.x,
                'cost': T.fun}

        self.amplitude_ = optimal['params'][0]


    def error(self, params, dist, field, metric='pearson'):

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
        if metric == 'pearson':
            merror = fe.pearson(field, density)
        if metric == 'kendall':
            merror = fe.kendall(field, density)
        if metric == 'spearman':
            merror = fe.spearman(field, density)
        elif metric == 'L2':
            merror = fe.L2(field, density)
        elif metric == 'L1':
            merror = fe.L1(field, density)

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

    def plot(self, cmap='jet', field='pdf'):

        """
        Plot scalar field and density, given fitted parameters.

        Parameters:
        - - - - -
        cmap: string
            colormap to use
        field: string
            scalar map to plot
            choices: ['pdf', 'amplitude', 'sigma', 'cost']
        """

        if field == 'amplitude' and not self.amplitude:
            print('Amplitudes were not fit.')
            return

        field_func = {
            'pdf': self.pdf(),
            'cost': self.costs_,
            'amplitude': self.params_[:, 0],
            'sigma': self.params_[:, 1]
            }

        field_titles = {
            'pdf': 'Estimated Density\nSigma: %.2f (mm)' % (self.sigma_),
            'cost': 'Costs',
            'amplitude': 'Amplitudes',
            'sigma': 'Sigmas (mm)'
            }

        fig = plt.figure(constrained_layout=False)
        gs = fig.add_gridspec(nrows=1, ncols=2, 
                                wspace=0.50, hspace=0.3)

        dnorm = mpl.colors.Normalize(vmin=np.min([self.data.min(), 0]),
                                     vmax=self.data.max())

        ax1 = fig.add_subplot(gs[0, 0])
        img1 = ax1.scatter(self.x, self.y, 
                            c=self.data, 
                            cmap='jet', 
                            norm=dnorm)
        ax1.scatter(self.x[self.ring], self.y[self.ring],
                    c='k',
                    s=5, alpha=0.75)
        ax1.scatter(self.x[self.mu_], 
                    self.y[self.mu_], 
                    c='k', 
                    s=50)
        ax1.scatter(self.x[self.gmax], 
                    self.y[self.gmax], 
                    c='k', 
                    s=50,
                    marker='^')
        ax1.annotate('Mu', (self.x[self.mu_], self.y[self.mu_]), fontsize=15)
        ax1.annotate('Peak', (self.x[self.gmax], self.y[self.gmax]), fontsize=15)

        ax1.set_title('Scalar Field', fontsize=15)
        ax1.set_xlabel('X', fontsize=15)
        ax1.set_ylabel('Y', fontsize=15)
        plt.colorbar(img1, ax=ax1)

        sfield = field_func[field]
        stitle = field_titles[field]

        snorm = mpl.colors.Normalize(vmin=sfield.min(), vmax=sfield.max())

        ax2 = fig.add_subplot(gs[0, 1])
        img2 = ax2.scatter(self.x, self.y, 
                            c=sfield, 
                            cmap='jet', 
                            norm=snorm)
        ax2.set_title(stitle, fontsize=15)
        ax2.set_xlabel('X', fontsize=15)
        ax2.set_ylabel('Y', fontsize=15)
        plt.colorbar(img2, ax=ax2)

        return fig

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

        param_names=['mu', 'amplitude', 'sigma', 'cost', 'w_signal', 'signal']

        param_vals = [self.mu_, self.amplitude_, self.sigma_,
                      self.costs_[self.mu_], self.weight(), self.data[self.mu_]]

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
