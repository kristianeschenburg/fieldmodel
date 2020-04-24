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
    amplitude: boolean
        whether or not to fit amplitude during optimization procedure
    peak_size: float / int
        minimum geodesic distance between local maxima
        larger values result in selecting few local maxima
    hood_size: float / int
        maximum radial distance from local maxima in which to search for
        field model mean location
        larger values means more candidates are considered
    metric: string
        cost function to use
        options are: ['pearson', 'kendall', 'spearman', 'L2', 'L1']
    """

    def __init__(self, r=10, peak_size=15, hood_size=20, metric='pearson'):

        self.r = 10
        self.peak_size = peak_size
        self.hood_size = hood_size
        self.metric = metric

    def fit(self, distances, data, x, y):

        """
        Estimate the location and scale parameters of the geodesic Gaussian.
        Finds a list of local maxima, estimates the global maxima, and then
        searches the neighborhood of this global maximum to best fit the
        location and scale parameters.

        Parameters:
        - - - - -
        distances: float, array
            distance matrix between all pairs of points in target region
        data: float, array
            scalar field to fit parameters to (i.e. correlation map)
        x, y: float, array
            flattened surface coordinates, used for plotting fitted parameters

        Returns:
        - - - -
        mu_: int
            lowest cost index
        sigma_: float
            sigma of fitted model
        dist_: int, array
            distance from mean location to all other indices
        optimal_: list, float
            array of optimal amplitude and sigma
        """

        self.data = data
        self.x = x
        self.y = y

        # define upper bound on field model variance
        self.up = distances.max(0).min()

        [n, _] = distances.shape

        # initialize parameter vectors
        params = np.zeros((n, 2))

        # intiailize cost vector array
        opt_cost = np.repeat(np.nan, n)
        lsq_cost = np.repeat(np.nan, n)


        ##### BEGIN PEAK FINDING #####

        # find local maxima in scalar field
        peaks = util.find_peaks(distances, data, n_size=self.peak_size)
        self.peaks = peaks

        # restrict location search space to neighborhood of global max
        nhood = util.peak_neighborhood(distances, peaks, h_size=self.hood_size)
        self.nhood = nhood

        ##### END PEAK FINDING #####



        ##### BEGIN FITTING PROCEDURE #####

        # set exclusion index parameters to NAN
        # we won't be searching over these indices
        exclude = list(set(np.arange(n)).difference(set(nhood)))
        params[exclude, :] = np.nan

        # iterate over all indices in neighborhood of global maximum
        # compute cost and optimized parameters for each index
        
        for idx in nhood:

            # we optimize first to find areas that look Gaussian
            tempopt = self.optimize(distances[idx, :])
            params[idx, 1] = tempopt['params']

            # then we fit the amplitudes to minimize the least-squares distance
            # of the distribution to the field
            coefs = self.fit_amplitude(distances[idx, :], 
                                       tempopt['params'], 
                                       include_intercept=True)

            params[idx, 0] = coefs[1]

            opt_cost[idx] = self.error([params[idx, 1]], distances[idx, :], self.data, metric='pearson')
            lsq_cost[idx] = self.error(params[idx, :], distances[idx, :], self.data, metric='L2')

        ##### END FITTING PROCEDURE #####


        ##### BEGIN PARAMETER SELECTION #####

        self.params_ = params

        # identify index with lowest cost
        self.mu_ = {'opt': np.nanargmin(opt_cost),
                    'lsq': np.nanargmin(lsq_cost),
                    'amp': np.nanargmax(params[:, 0])}
        
        self.signal_ = {'opt': self.data[self.mu_['opt']],
                        'lsq': self.data[self.mu_['lsq']],
                        'amp': self.data[self.mu_['amp']]}

        # store cost array for all indices in search space
        self.cost_ = {'opt': opt_cost,
                      'lsq': lsq_cost}

        # store fieldmodel sigma and amplitude parameters of lowest cost index
        self.amplitude_ = {'opt': params[self.mu_['opt'], 0],
                           'lsq': params[self.mu_['lsq'], 0],
                           'amp': params[self.mu_['amp'], 0]}
        
        self.sigma_ = {'opt': params[self.mu_['opt'], 1],
                       'lsq': params[self.mu_['lsq'], 1],
                       'amp': params[self.mu_['amp'], 1]}
        
        self.optimal_ = {'opt': [self.amplitude_['opt'], self.sigma_['opt']],
                         'lsq': [self.amplitude_['lsq'], self.sigma_['lsq']],
                         'lsq': [self.amplitude_['amp'], self.sigma_['amp']]}
        
        self.dist_ = {'opt': distances[self.mu_['opt']],
                      'lsq': distances[self.mu_['lsq']],
                      'amp': distances[self.mu_['amp']]}

        self.fitted = True

    def optimize(self, dist):


        """
        Optimization sub-method.

        Parameters:
        - - - - -
        dist: int, array
            distance of current index to all other indices
        """

        a0 = [self.r*2]

        bds = Bounds(0.5, self.up)

        T = minimize(self.error, a0,
                 args=(dist, self.data, self.metric),
                 bounds=(bds))

        optimal = {'params': T.x,
                   'cost': T.fun}

        return optimal

    def fit_amplitude(self, dist, sigma, include_intercept=True):

        """
        Fit amplitude after fitting sigma, incluing 

        Parameters:
        - - - - -
        dist: int, array
            distance of current index to all other indices
        sigma: float
            estimated sigma value
        """

        X = models.geodesic(dist, [sigma])
        X = (X-X.min()) / (X.max() - X.min())

        if include_intercept:
            X = np.column_stack([np.ones(X.shape[0]), X])

        Y = self.data

        if X.ndim == 1:
            A = (1 / (X.T.dot(X))) * X.T.dot(Y)
        else:
            A = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(Y))

        return A

    def error(self, params, dist, field, metric='pearson'):

        """
        Compute error of current density estimate.

        Parameters:
        - - - - -
        params: list
            current parameter estimates
            [amplitude, sigma]
            or
            [sigma]
        dist: float, array
            geodesic distance vector
        field: float, array
            scalar field on which to fit density
        metric: string
            cost function to use
            choices include ['pearson', 'kendall', 'spearman', 'L2', 'L1']
        """

        error_map = {'pearson': fe.pearson,
                     'kendall': fe.kendall,
                     'spearman': fe.spearman,
                     'L2': fe.L2,
                     'L1': fe.L1}

        density = models.geodesic(dists=dist, params=params)
        merror = error_map[metric](field, density)

        return merror

    def pdf(self):

        """
        Return density of fitted model, normalized to density 1.
        """

        prob = models.geodesic(self.dist_['opt'], self.optimal_['opt'])
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
            choices: ['pdf', 'amplitude', 'sigma', 'cost', 'data']
        """

        field_func = {
            'pdf': self.pdf(),
            'cost': self.cost_['opt'],
            'amplitude': self.params_[:, 0],
            'sigma': self.params_[:, 1]
            }

        field_titles = {
            'pdf': 'Estimated Density\nSigma: %.2f (mm)' % (self.sigma_['opt']),
            'cost': 'Fitted Search-Space Costs',
            'amplitude': 'Fitted Search-Space Amplitudes',
            'sigma': 'Fitted Search-Space Sigmas'
            }

        field_labels = {
            'pdf': 'Density',
            'cost': 'Cost',
            'amplitude': 'Amplitude',
            'sigma': 'Sigma (mm)'
            }

        if field == 'data':
            fig = plt.figure(constrained_layout=False, figsize=(6, 4))

            gs = fig.add_gridspec(nrows=1, 
                                ncols=2,
                                wspace=0.50,
                                hspace=0.3)
        else:
            fig = plt.figure(constrained_layout=False, figsize=(10, 4))

            gs = fig.add_gridspec(nrows=1, 
                                ncols=2,
                                wspace=0.50,
                                hspace=0.3)

        dnorm = mpl.colors.Normalize(vmin=np.nanmin([self.data.min(), 0]),
                                     vmax=np.nanmax(self.data))

        ax1 = fig.add_subplot(gs[0, 0])
        img1 = ax1.scatter(self.x, self.y, 
                            c=self.data, 
                            cmap='jet', 
                            norm=dnorm)

        mu_symbols = ['o', '*', 'P']
        mu_names = [r'Pearson $\rho$', 'LSQ', 'MaxAmp']
        mean_map = {list(self.mu_.keys())[i]: {'Symbol': mu_symbols[i], 
                                               'Name': mu_names[i]} for i in np.arange(3)}

        for j, kind in enumerate(self.mu_.keys()):
            ax1.scatter(self.x[self.mu_[kind]], self.y[self.mu_[kind]], 
                        c='k', s=70, 
                        marker=mean_map[kind]['Symbol'], 
                        label=mean_map[kind]['Name'])
        
        ax1.legend(loc='upper left', fontsize=10, fancybox=True, framealpha=0)


        ax1.set_title('Fitted Scalar Field', fontsize=15)
        ax1.set_xlabel('X', fontsize=15)
        ax1.set_ylabel('Y', fontsize=15)
        plt.colorbar(img1, ax=ax1)

        if field != 'data':
            sfield = field_func[field]
            stitle = field_titles[field]

            snorm = mpl.colors.Normalize(vmin=np.nanmin(sfield), 
                                        vmax=np.nanmax(sfield))

            ax2 = fig.add_subplot(gs[0, 1])
            img2 = ax2.scatter(self.x, self.y, 
                                c=sfield, 
                                cmap='jet', 
                                norm=snorm)

            ax2.set_title(stitle, fontsize=15)
            ax2.set_xlabel('X', fontsize=15)
            ax2.set_ylabel('Y', fontsize=15)
            plt.colorbar(img2, ax=ax2, label=field_labels[field])

        return fig

    def get_params(self, param_names=['mu_lsq', 'sd_lsq', 'a_lsq', 'r_lsq']):

        """
        Return requested parameters as Pandas DataFrame.

        Fitted values include: 
            mean ('m')
            sigma ('sd')
            signal ('r')
            amplitude ('a')

        along with the three cost functions:
            minimum correlation cost ('opt')
            minimized least-squared ('lsq')
            maximum amplitude ('amp')
        
        Supplied parameter names must be a combination of value and cost:
            e.g. 'm_lsq', 'sd_amp', 'r_opt', etc.


        Returns:
        - - - -
        df: Pandas DataFrame
            Includes field location ('mu'), scale ('sigma'),
            cost ('cost') and weighted signal average ('signal')
        """

        assert self.fitted, 'Must fit model before writing coefficients.'

        param_vars = ['mu_opt', 'mu_lsq', 'mu_amp', 
                       'sd_opt', 'sd_lsq', 'sd_amp',
                       'a_opt', 'a_lsq', 'a_amp', 
                       'r_opt', 'r_lsq', 'r_amp']
        
        param_vals = [self.mu_['opt'], self.mu_['lsq'], self.mu_['amp'],
                      self.sigma_['opt'], self.sigma_['lsq'], self.sigma_['amp'],
                      self.amplitude_['opt'], self.amplitude_['lsq'], self.amplitude_['amp'],
                      self.signal_['opt'], self.signal_['lsq'], self.signal_['amp']]

        param_map = dict(zip(param_vars, param_vals))

        params = {pn: [param_map[pn]] for pn in param_names if pn in param_vars}
        df = pd.DataFrame(params)

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
