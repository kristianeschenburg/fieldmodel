import pandas as pd
import numpy as np
from scipy.optimize import fmin, minimize
from scipy.spatial.distance import cdist

import fieldmodel.fielderrors as fe
import fieldmodel.models as models

import matplotlib as mpl
import matplotlib.pyplot as plt

class FieldModel(object):

    """
    Class to perform Connective Field Modeling.

    Compute the field model for a given source-to-target mapping.
    The source is decribed by a point source feature vector, while a target
    is described by a set of point source feature vectors.  For a given 
    source-to-target mapping, a scalar field model is fit to the similarity
    between the source and target feature vectors.
    
    For a given scalar field, we fit an isotropic Gaussian or T distribution
    to the scalar field.  The fit is performed by minimizing one of two metrics:

        1) Correlation distance between field and mixture density
        2) SSQ distance between field and mixture density

    Parameters:
    - - - - -
    density: string
        'gaussian' or 'students'
    opterror: string
        optimization measure
        'correlation' or 'lsq'
    nu: float > 0
        degrees of freedom
    amplitude: boolean
        flag to optimize amplitude
    """

    def __init__(self, density='geodesic', opterror='correlation', nu=None, fit_amplitude=False):

        """
        For now, n_components must be set to 1.
        """

        assert density in ['gaussian', 'students',]
        assert opterror in ['lsq', 'correlation']
        assert isinstance(fit_amplitude, bool)

        if density == 'students':
            assert nu
            assert nu >= 1

        self.opterror = opterror
        self.density = density
        self.fit_amplitude = fit_amplitude
        self.nu = nu
    
    def fit(self, x, y, data, p0):

        """
        Parameters:
        - - - - -
        x, y: float, array
            coordinates of data
        data: float, array
            scalar field to fit parameters to
        p0: list
            must be of length (n_components) * (n_params)
            where n_params is the number of parameters in
            the Gaussian model
        """

        n = len(x)
        opterror = self.opterror
        density = self.density

        coefs = fmin(self.error, p0,
                args=(x, y, data, opterror, density),
                maxiter=5000, disp=False)
        
        self.x = x
        self.y = y
        self.data = data

        self.coefs_ = coefs

        self.p0 = p0
        self.means_ = self.coefs_[0:2]
        self.sigmas_ = self.coefs_[2]

        self.fitted = True

    def error(self, params, x, y, sfield, opterror, density):

        """
        Compute error of current fit.

        Parameters:
        - - - - -
        params: list
            current parameter estimates
        x, y: float, array
            spatial coordinates of data
        sfield: float, array
            data to fit
        opterror: string
            error function to use
        density: string
            parameterized distribution of coordinates
        """

        error_map = {'correlation': fe.correlation,
                     'lsq': fe.lsq}

        nu = self.nu
        sfield = sfield.squeeze()

        # estimte density given current parameters
        if density == 'gaussian':
            L = models.gaussian(x, y, params).squeeze()
        elif density == 'students':
            L = models.student(x, y, nu, params).squeeze()

        # estimate error between scalar field and current density
        merror = error_map[opterror](sfield, L)

        return merror

    def _error_amplitude(self, params, x, y, sfield, constants, density):

        """
        Compute error of current amplitude-scaled fit.

        Parameters:
        - - - - -
        params: list
            current amplitude estimate
        x, y: float, array
            coordinates of data
        sfield: float, array
            scalar field
        constants: list
            estimated distributional parameters
        density: string
            parameterized distribution of coordinates
        """

        nu = self.nu
        sfield = sfield.squeeze()

        if density == 'gaussian':
            L = models.gaussian(x, y, constants).squeeze()
        else:
            L = models.student(x, y, nu, constants).squeeze()

        L = params[0] * L
        merror = fe.lsq(sfield, L)

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