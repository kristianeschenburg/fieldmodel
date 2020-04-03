import matplotlib as mpl
import matplotlib.pyplot as plt

import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.patches import ConnectionPatch
import matplotlib.gridspec as gridspec
from matplotlib import textpath

import seaborn as sns
sns.set()
sns.set_style("darkgrid", {"axes.facecolor": "0.75"})

import numpy as np


class FieldModelGraph(object):

    """
    Class to visualize the fitted field modeling mapping.

    Parameters:
    - - - - -
    vertices: float, array
        spatial coordinates of data
    index_map: dictionary
        mapping of region ID to spatial coordinates
    nsamples: int
        number of samples to plot
    doPCA: bool
        whether to perform PCA on spatial coordinates
        of each region
    colormap: string
        colormap of scalar fields
    figsize: tuple
        size of subplot figure
    """

    def __init__(self, vertices, index_map, nSamples=20, doPCA=True,
                 colormap='coolwarm',
                 figsize=(15, 8)):

        self.vertices = vertices
        self.indices = index_map
        self.nSamples = nSamples
        self.doPCA = doPCA
        self.colormap = colormap
        self.figsize=figsize

    def plot(self, C, sregion, tregion, mapping):

        """
        Plot the source-to-target mappings for a set of source vertices.
        Returns at least 4 plots.

        Parameters:
        - - - - -
        C: float, array
            source x target scalar field matrix
            generally the cross-correlation matrix\
        s/t basis: float, array
            source and target region coordinate bases
        mapping: DataFrame
            fitted source-to-target mappings
        sregion / tregion: list, string
            source / target regions
        """

        # preprocess spatial coordinates of data
        s_inds = self.indices[sregion]
        t_inds = self.indices[tregion]
        sverts = self.vertices[s_inds, :2]
        tverts = self.vertices[t_inds, :2]

        if self.doPCA:
            sverts = self.PCA(self.vertices[s_inds, :2])
            tverts = self.PCA(self.vertices[t_inds, :2])
        
        self.sverts = sverts
        self.tverts = tverts

        include = (np.isnan(C).sum(1) != C.shape[1])
        [X, Y] = self.sample_coordinates(self.sverts,
                                         self.nSamples,
                                         include=include)

        sMap = dict(zip(s_inds, np.arange(len(s_inds))))
        tMap = dict(zip(t_inds, np.arange(len(t_inds))))

        # Initialize figure
        fig = plt.figure(constrained_layout=False, 
                            figsize=self.figsize)
        gs = fig.add_gridspec(nrows=2, ncols=3, 
                                wspace=0.50, hspace=0.3)

        # Plot source and target coordinates for first PCA axis
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.scatter(self.sverts[:, 0], self.sverts[:, 1], 
                        marker='.', alpha=0.5)
        ax1.set_title('Source Coordinates: Axis 1\n%s' % (sregion))
        ax1.set_ylabel('PCA Axis 2')

        ax4 = fig.add_subplot(gs[1, 0])
        ax4.scatter(self.tverts[:, 0], self.tverts[:, 1], 
                        marker='.', alpha=0.5)
        ax4.set_title('Target Coordinates\n%s' % (tregion))
        ax4.set_xlabel('PCA Axis 1')
        ax4.set_ylabel('PCA Axis 2')

        # Plot source and target coordinates for second PCA axis
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.scatter(self.sverts[:, 0], self.sverts[:, 1], 
                        marker='.', alpha=0.5)
        ax2.set_title('Source Coordinates: Axis 2\n%s' % (sregion))

        ax3 = fig.add_subplot(gs[0, 2])
        ax3.scatter(self.tverts[:, 0], self.tverts[:, 1], 
                        marker='.', alpha=0.5)
        ax3.set_title('Target Coordinates\n%s' % (tregion))




        # AXIS 1
        tc = [tMap[mapping.iloc[c]['mu']] for c in X]
        norm = colors.Normalize(vmin=self.sverts[X, 0].min(),
                                vmax=self.sverts[X, 0].max())
        
        cmap = cm.plasma
        m = cm.ScalarMappable(norm=norm, cmap=cmap)
        for j, (s, t) in enumerate(zip(X, tc)):

                c = m.to_rgba(np.asarray(self.sverts[X[j], 0]))

                patch = ConnectionPatch(xyA=tuple(self.tverts[t, :]),
                                        xyB=tuple(self.sverts[s, :]),
                                        coordsA='data', coordsB='data',
                                        axesA=ax4, axesB=ax1,
                                        lw=2,
                                        linestyle='--',
                                        alpha=0.75,
                                        color=c)

                ax4.add_artist(patch)
                ax1.scatter(self.sverts[s, 0],
                               self.sverts[s, 1],
                               c='red',
                               s=20,
                               alpha=0.5)

        # AXIS 2
        tc = [tMap[mapping.iloc[c]['mu']] for c in Y]
        norm = colors.Normalize(vmin=self.sverts[Y, 1].min(),
                                vmax=self.sverts[Y, 1].max())

        cmap = cm.plasma
        m = cm.ScalarMappable(norm=norm, cmap=cmap)
        for j, (s, t) in enumerate(zip(Y, tc)):

                c = m.to_rgba(np.asarray(self.sverts[Y[j], 1]))

                patch = ConnectionPatch(xyA=tuple(self.tverts[t, :]),
                                        xyB=tuple(self.sverts[s, :]),
                                        coordsA='data', coordsB='data',
                                        axesA=ax3, axesB=ax2,
                                        lw=2,
                                        linestyle='--',
                                        alpha=0.75,
                                        color=c)

                ax3.add_artist(patch)
                ax2.scatter(self.sverts[s, 0],
                               self.sverts[s, 1],
                               c='red',
                               s=20,
                               alpha=0.5)
        
        norm4 = mpl.colors.Normalize(vmin=0, vmax=mapping['sigma'].max())
        ax4 = fig.add_subplot(gs[1, 1])
        img4 = ax4.scatter(sverts[:, 0], sverts[:, 1], 
                            c=mapping['sigma'],
                            norm=norm4,
                            cmap='jet')
        ax4.set_title('Mapping Variance')
        plt.colorbar(img4, ax=ax4)


        norm5 = mpl.colors.DivergingNorm(vmin=-1,
                                         vcenter=np.tanh(mapping['signal']).mean(),
                                         vmax=1)
        ax5 = fig.add_subplot(gs[1, 2])
        img5 = ax5.scatter(sverts[:, 0], sverts[:, 1], 
                            c=np.tanh(mapping['signal']), 
                            norm=norm5,
                            cmap='jet')
        ax5.set_title('Mapping Signal')
        plt.colorbar(img5, ax=ax5)


    def sample_coordinates(self, coordinates, nsamples, include):

        """
        Sample indices unformly in the X and Y directions, independently.
        
        Parameters:
        - - - - -
        index_map: dictionary
            mapping of region name to indices
        vertices: float, array
            original spatial coordinates of data
        sreg: string
            region name
        nsamples: int
            number of samples to return
        
        Returns:
        - - - -
        x/y coords: list, int
            samples in the X and Y directions.
        """

        V = coordinates[include, :]
        [xwin, ywin] = np.sqrt(V.std(0))

        yinds = np.where((V[:, 0] < xwin) & (V[:, 0] > -xwin))[0]
        Y = np.random.choice(yinds, np.min([len(yinds), nsamples]), replace=False)

        xinds = np.where((V[:, 1] < ywin) & (V[:, 1] > -ywin))[0]
        X = np.random.choice(xinds, np.min([len(xinds), nsamples]), replace=False)

        return [X, Y]


    def PCA(self, coordinates):

        """
        Compute PCA-transformed coordinate system.

        Parameters:
        - - - - -
        coordinates: float, array
            spatial coordinates of data
        """

        [u, s, v] = np.linalg.svd(np.cov(coordinates.T))
        coordinates = coordinates.dot(u)
        coordinates = coordinates - np.median(coordinates, 0)[None, :]

        return coordinates


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
    
    norm = mpl.colors.Normalize(vmin=np.min([sfield.min(), 0]),
                                vmax=sfield.max())

    fig = plt.figure(constrained_layout=False)
    gs = fig.add_gridspec(nrows=1, ncols=2,
                            wspace=0.50, hspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(x, y, marker='.')
    for p in peaks:
        ax1.scatter(x[p], y[p], c='r', s=50,)
        ax1.annotate('%i' %(p), (x[p], y[p]), fontsize=15)
    ax1.set_title('Peaks', fontsize=15)

    ax2 = fig.add_subplot(gs[0, 1])
    img = ax2.scatter(x, y, c=sfield, marker='.', cmap='jet', norm=norm)
    ax2.set_title('Scalar Field', fontsize=15)
    plt.tight_layout()
    plt.colorbar(img, ax=ax2)
    plt.show()

def kde()