"""
The classes in this file do preprocessing on data and monte carlo to be used
to do a point source analysis.
"""

__author__ = 'John Evans'
__copyright__ = ''
__credits__ = ['John Evans', 'Jason Fan', 'Michael Larson']
__license__ = 'Apache License 2.0'
__version__ = '0.0.1'
__maintainer__ = 'John Evans'
__email__ = 'john.evans@icecube.wisc.edu'
__status__ = 'Development'

from typing import List, Union

import numpy as np

import scipy.interpolate

class EventModel:
    """Stores the events and pre-processed parameters used in analyses.

    Currently, this class uses internal data and monte-carlo datasets. This
    will be updated before the first release to use the upcoming public data
    release.

    Attributes:
        data (np.ndarray): Real neutrino event data.
        sim (np.ndarray): Simulated neutrino events.
        grl (np.ndarray): A list of runs/times when the detector was working
            properly.
        sin_dec_bins (np.array): An array of sin(dec) bin edges for the energy
            maps.
        log_energy_bins (np.array): An array of log(energy) bin edges for the
            energy maps.
        background_dec_spline (scipy.interpolate.UnivariateSpline): A spline
            fit of neutrino flux vs. sin(dec).
        log_sob_gamma_splines (List[List[scipy.interpolate.UnivariateSpline]]):
            A 2D list of spline fits of the log(signal-over-background) vs.
            gamma at a binned energy and sin(dec).
    """
    def __init__(self, # For an initial release, this implementation is fine... pylint: disable=too-many-arguments
                 data: np.ndarray,
                 sim: np.ndarray,
                 grl: np.ndarray,
                 background_sin_dec_bins: Union[np.array, int] = 500,
                 signal_sin_dec_bins: Union[np.array, int] = 50,
                 log_energy_bins: Union[np.array, int] = 50,
                 gamma_bins: Union[np.array, int] = 50,
                 verbose: bool = False) -> None:
        """Initializes EventModel and calculates energy sob maps.

        Args:
            background_sin_dec_bins: If an int, then the number of bins
                spanning -1 -> 1, otherwise, a numpy array of bin edges.
            signal_sin_dec_bins: If an int, then the number of bins spanning
                -1 -> 1, otherwise, a numpy array of bin edges.
            log_energy_bins: if an int, then the number of bins spanning
                1 -> 8, otherwise, a numpy array of bin edges.
            gamma_bins: If an int, then the number of bins spanning
                -4.25 -> -0.5, otherwise, a numpy array of bin edges.
            verbose: A flag to print progress.
        """
        self._data = data
        self._sim = sim

        min_mjd = np.min(data['time'])
        max_mjd = np.max(data['time'])
        self._grl = grl[(grl['start'] < max_mjd) & (grl['stop'] > min_mjd)]

        if isinstance(background_sin_dec_bins, int):
            background_sin_dec_bins = np.linspace(-1, 1,
                                                  1+background_sin_dec_bins)

        self._background_dec_spline = self._create_background_dec_spline(
            background_sin_dec_bins)

        if isinstance(signal_sin_dec_bins, int):
            signal_sin_dec_bins = np.linspace(-1, 1, 1+signal_sin_dec_bins)
        self._sin_dec_bins = signal_sin_dec_bins

        if isinstance(log_energy_bins, int):
            log_energy_bins = np.linspace(1, 8, 1+log_energy_bins)
        self._log_energy_bins = log_energy_bins

        if isinstance(gamma_bins, int):
            gamma_bins = np.linspace(-4.25, -0.5, 1+gamma_bins)

        self._log_sob_gamma_splines = self._create_log_sob_gamma_splines(
            gamma_bins, verbose)

    def _create_background_dec_spline(self, sin_dec_bins: np.array, *args,
                                      **kwargs,
    ) -> scipy.interpolate.UnivariateSpline:
        """Builds a histogram of neutrino flux vs. sin(dec) and splines it.

        The UnivariateSpline function call uses these default arguments:
        bbox=[-1.0, 1.0], s=1.5e-5, ext=1. To replace any of these defaults, or
        to pass any other args/kwargs to UnivariateSpline, just pass them to
        this function.

        Args:
            sin_dec_bins: A numpy array of bin edges to use to build the
                histogram to spline.

        Returns:
            A spline function of the neutrino flux vs. sin(dec) histogram.
        """
        # Our background PDF only depends on declination.
        # In order for us to capture the dec-dependent
        # behavior, we first take a look at the dec values
        # in the data. We can do this by histogramming them.
        sin_dec = np.sin(self._data['dec'])

        # Make the background histogram. Note that we do NOT
        # want to use density=True here, since that would mean
        # that our spline depends on the original bin widths!
        weights = np.ones_like(self._data['dec'])/len(self._data['dec'])
        hist, bins = np.histogram(sin_dec, bins=sin_dec_bins, weights=weights)
        bin_centers = bins[:-1] + np.diff(bins)/2

        # These values have a lot of "noise": they jump
        # up and down quite a lot. We could use fewer
        # bins, but that may hide some features that
        # we care about. We want something that captures
        # the right behavior, but is smooth and continuous.
        # The best way to do that is to use a "spline",
        # which will fit a continuous and differentiable
        # piecewise polynomial function to our data.
        # We can set a smoothing factor (s) to control
        # how smooth our spline is.

        if 'bbox' not in kwargs:
            kwargs['bbox'] = [-1.0, 1.0]
        if 's' not in kwargs:
            kwargs['s'] = 1.5e-5
        if 'ext' not in kwargs:
            kwargs['ext'] = 1

        # Normalize over the right ascension, assuming
        # uniform background response. This works for
        # longer time windows and is roughly correct for
        # shorter windows as well. This is necessary in
        # order to get identical units as the 2d gaussian
        # used for the signal spatial PDF
        hist /= (2 * np.pi)

        return scipy.interpolate.UnivariateSpline(bin_centers, hist, *args,
                                                  *kwargs)

    def _create_sob_map(self, gamma: float, *args, verbose: bool = False, # I don't think it's necessary to reduce the number of variables... pylint: disable=too-many-locals
                        **kwargs) -> np.array:
        """Creates sob histogram for a given spectral index (gamma).

        The UnivariateSpline function call uses these default arguments:
        k=1, s=0, ext=3. To replace any of these defaults, or to pass any other
        args/kwargs to UnivariateSpline, just pass them to this function.
        Spline is used here to smooth over the energies to get the values for
        each bin.

        Args:
            gamma: The gamma value to use to weight the signal.
            verbose: A flag to print progress.

        Returns:
            An array of signal-over-background values binned in sin(dec) and
            log(energy) for a given gamma.
        """
        if verbose:
            print(f'Building map for gamma = {gamma}...', end='')
        bins = np.array([self._sin_dec_bins, self._log_energy_bins])
        bin_centers = bins[1,:-1] + np.diff(bins[1])/2

        # background
        bg_h, _, _ = np.histogram2d(np.sin(self._data['dec']),
                                    self._data['logE'], bins=bins, density=True)

        # signal
        sig_w = self._sim['ow'] * self._sim['trueE']**gamma
        sig_h, _, _ = np.histogram2d(np.sin(self._sim['dec']),
                                     self._sim['logE'], bins=bins,
                                     weights=sig_w, density=True)

        # Normalize histograms by dec band
        bg_h /= np.sum(bg_h,axis=1)[:,None]
        sig_h /= np.sum(sig_h,axis=1)[:,None]

        ratio = sig_h / bg_h

        if 'k' not in kwargs:
            kwargs['k'] = 1
        if 's' not in kwargs:
            kwargs['s'] = 0
        if 'ext' not in kwargs:
            kwargs['ext'] = 3

        for i in range(ratio.shape[0]):
            # Pick out the values we want to use.
            # We explicitly want to avoid NaNs and infinities
            good = np.isfinite(ratio[i]) & (ratio[i]>0)
            good_bins, good_vals = bins[1][:-1][good], ratio[i][good]

            # Do a linear interpolation across the energy range
            spline = scipy.interpolate.UnivariateSpline(good_bins, good_vals,
                                                        *args, **kwargs)

            # And store the interpolated values
            ratio[i] = spline(bin_centers)
        if verbose:
            print('done')
        return ratio

    def _create_log_sob_gamma_splines(self, gamma_bins: np.array, *args,
                                      verbose: bool = False, **kwargs,
    ) -> List[List[scipy.interpolate.UnivariateSpline]]:
        """Builds a 3D hist of sob vs. sin(dec), log(energy), and gamma, then
            returns splines of sob vs. gamma.

        The UnivariateSpline function call uses these default arguments:
        k=3, s=0, ext='raise'. To replace any of these defaults, or to pass any
        other args/kwargs to UnivariateSpline, just pass them to this function.
        Spline is used here to smooth over the energies to get the values for
        each bin.

        Args:
            gamma_bins: The spectral indicies at which to build the histograms.
            verbose: A flag to print progress.

        Returns: A Nested spline list of shape (sin_dec_bins, log_energy_bins).
        """
        if verbose:
            print('Building signal-over-background maps...')
        sob_maps = np.array([self._create_sob_map(gamma, verbose)
                             for gamma in gamma_bins])
        if verbose:
            print('done.')

        if 'k' not in kwargs:
            kwargs['k'] = 3
        if 's' not in kwargs:
            kwargs['s'] = 0
        if 'ext' not in kwargs:
            kwargs['ext'] = 'raise'

        transposed_log_sob_maps = np.log(sob_maps.transpose(1, 2, 0))

        if verbose:
            print('Fitting log(signal-over-background vs. gamma splines)...',
                  end='')

        def spline(log_ratios):
            return scipy.interpolate.UnivariateSpline(gamma_bins, log_ratios,
                                                      *args, **kwargs)

        splines = [[spline(log_ratios) for log_ratios in dec_bin]
                   for dec_bin in transposed_log_sob_maps]
        if verbose:
            print('done')

        return splines

    def get_log_sob_gamma_splines(self, events: np.ndarray,
    ) -> List[scipy.interpolate.UnivariateSpline]:
        """Gets the splines of sob vs. gamma required for each event.

        Args:
            events: An array of events including their positional data.

        Returns:
            A list of splines of sob vs gamma for each event.
        """
        # Get the bin that each event belongs to
        sin_dec_idx = np.searchsorted(self._sin_dec_bins[:-1],
                                      np.sin(events['dec']))
        log_energy_idx = np.searchsorted(self._log_energy_bins[:-1],
                                         events['logE'])

        return [self._log_sob_gamma_splines[i][j]
                for i,j in zip(sin_dec_idx, log_energy_idx)]

    @property
    def data(self) -> np.ndarray:
        """Getter for data."""
        return self._data

    @property
    def sim(self) -> np.ndarray:
        """Getter for sim."""
        return self._sim

    @property
    def grl(self) -> np.ndarray:
        """Getter for GRL."""
        return self._grl

    @property
    def background_dec_spline(self) -> scipy.interpolate.UnivariateSpline:
        """Getter for background_dec_spline."""
        return self._background_dec_spline
