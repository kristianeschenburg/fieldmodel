import numpy as np
from scipy import special

def density(x, y, pmap, amplitude=True):

    """
    Return density of fitted model.

    Parameters:
    - - - - -
    x, y: float, array
        coordinates of data
    pmap: dictionary
        mapping of parameter names to parameter values
    """

    d = pmap['density']
    params = [pmap['x_mean'], pmap['y_mean'], pmap['sigma']]

    if d == 'gaussian':
        pdf = gaussian(x, y, params)
    
    elif d == 'students':
        pdf = student(x, y, pmap['nu'], params)
    
    if amplitude:
        pdf = pmap['amplitude'] * pdf

    return pdf


def gaussian(x, y, params):

        """
        Compute the Multivariate Normal density of samples, given the current parameters.

        Parameters:
        - - - - -
        samples: float, array
            coordinates of data
        params: list
            current parameter estimates
        """

        n = len(x)
        mix = np.zeros((n,))

        ex = (((x - params[0])**2) / (2.*params[2]**2))
        ey = (((y - params[1])**2) / (2.*params[2]**2))
        mix = ex + ey

        pdf = np.sqrt((1. / (2*np.pi*params[2]**2))) * np.exp(-1*mix)

        return pdf

def geodesic(dists, params):

    """
    Compute the surface-based normal density, given the current parameters.

    Parameters:
    - - - - -
    dists: float, array
        distance from mean location to all other surface locations
    params: list
        current parameter estimates
    """

    if len(params) == 1:
        g = np.exp(-1*(dists**2) / (2*params[0]**2))
    
    else:
        g = params[0] * np.exp(-1*(dists**2) / (2*params[1]**2))
    
    g = g/g.sum()

    return g


def student(x, y, nu, params):

        """
        Compute the Multivariate T density of samples, given the current parameters.

        Parameters:
        - - - - -
        x, y: float, array
            coordinates of data
        params: list
            current parameter estimates
        """

        numer = special.gamma((nu + 2.) / 2.)
        denom = special.gamma(nu / 2.) * nu * np.pi * (params[2]**2)

        C = numer / denom

        ex = ((x - params[0])**2) / params[2]**2
        ey = ((y - params[1])**2) / params[2]**2
        mix = ex + ey
        body = (1. + (1./nu) * mix) ** (-1. * (nu + 2.)/2.)

        pdf = C * body

        return pdf