__author__ = 'John Evans'
__copyright__ = ''
__credits__ = ['John Evans', 'Jason Fan', 'Michael Larson']
__license__ = 'Apache License 2.0'
__version__ = '0.0.1'
__maintainer__ = 'John Evans'
__email__ = 'john.evans@icecube.wisc.edu'
__status__ = 'Development'

"""
Docstring
"""

from typing import Dict, List, Union

import numpy as np
from i3pubtools import tools

import numpy.lib.recfunctions as rf
import scipy.interpolate

class EventModel:
    """Stores the events and pre-processed parameters used in analyses.
    
    More class info...
    
    Attributes:
        data (np.ndarray): Real neutrino event data.
        sim (np.ndarray): Simulated neutrino events.
        grl (np.ndarray): A list of runs/times when the detector was working properly.
        sin_dec_bins (np.array): An array of sin(dec) bin edges for the energy maps.
        log_energy_bins (np.array): An array of log(energy) bin edges for the energy maps.
        background_dec_spline (scipy.interpolate.UnivariateSpline): A spline fit of neutrino flux
            vs. sin(dec).
        log_sob_gamma_splines (List[List[scipy.interpolate.UnivariateSpline]]): A 2D list of spline
            fits of the log(signal-over-background) vs. gamma at a binned energy and sin(dec).
    """
    def __init__(self,
                 data: np.ndarray,
                 sim: np.ndarray,
                 grl: np.ndarray,
                 background_sin_dec_bins: Union[np.array, int] = 500,
                 signal_sin_dec_bins: Union[np.array, int] = 50,
                 log_energy_bins: Union[np.array, int] = 50,
                 gamma_bins: Union[np.array, int] = 50,
                 verbose: bool = False) -> None:
        """Initializes EventModel and calculates energy signal-over-background (sob) maps.
        
        More function info...
        
        Args:
            background_sin_dec_bins: If an int, then the number of bins spanning -1 -> 1,
                otherwise, a numpy array of bin edges.
            signal_sin_dec_bins: If an int, then the number of bins spanning -1 -> 1,
                otherwise, a numpy array of bin edges.
            log_energy_bins: if an int, then the number of bins spanning 1 -> 8,
                otherwise, a numpy array of bin edges.
            gamma_bins: If an int, then the number of bins spanning -4.25 -> -0.5,
                otherwise, a numpy array of bin edges.
            verbose: A flag to print progress.
        """
        self._data = data
        self._sim = sim
        
        min_mjd = np.min(data['time'])
        max_mjd = np.max(data['time'])
        self._grl = grl[(grl['start'] < max_mjd) & (grl['stop'] > min_mjd)]
        
        if isinstance(background_sin_dec_bins, int):
            background_sin_dec_bins = np.linspace(-1, 1, 1+background_sin_dec_bins)
        self._background_dec_spline = self._create_background_dec_spline(background_sin_dec_bins)
            
        if isinstance(signal_sin_dec_bins, int):
            signal_sin_dec_bins = np.linspace(-1, 1, 1+signal_sin_dec_bins)
        self._sin_dec_bins = signal_sin_dec_bins
            
        if isinstance(log_energy_bins, int):
            log_energy_bins = np.linspace(1, 8, 1+log_energy_bins)
        self._log_energy_bins = log_energy_bins
        
        if isinstance(gamma_bins, int):
            gamma_bins = np.linspace(-4.25, -0.5, 1+gamma_bins)
        self._log_sob_gamma_splines = self._create_log_sob_gamma_splines(gamma_bins, verbose)
    
    def _create_background_dec_spline(self, sin_dec_bins: np.array,
                                      *args,
                                      **kwargs) -> scipy.interpolate.UnivariateSpline:
        """Builds a histogram of neutrino flux vs. sin(dec) and returns a spline of it.

        More function info...

        Args:
            sin_dec_bins: A numpy array of bin edges to use to build the histogram to spline.

        Returns:
            A spline function representing the neutrino flux vs. sin(dec) histogram.
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
        
        if 'bbox' not in kwargs: kwargs['bbox'] = [-1.0, 1.0]
        if 's' not in kwargs: kwargs['s'] = 1.5e-5
        if 'ext' not in kwargs: kwargs['ext'] = 1
        
        return scipy.interpolate.UnivariateSpline(bin_centers, hist, *args, *kwargs)
    
    def _create_sob_map(self, gamma: float, *args, verbose: bool = False, **kwargs) -> np.array:
        """Function info...

        More function info...

        Args:
            gamma: The gamma value to use to weight the signal.
            verbose: A flag to print progress.

        Returns:
            An array of signal-over-background values binned in sin(dec) and log(energy) for a given gamma."""
        if verbose: print(f'Building map for gamma = {gamma}...', end='')
        bins = np.array([self._sin_dec_bins, self._log_energy_bins])
        bin_centers = bins[1,:-1] + np.diff(bins[1])/2
        
        # background
        bg_h, xedges, yedges = np.histogram2d(
            np.sin(self._data['dec']),
            self._data['logE'],
            bins=bins,
            density=True)

        # signal
        sig_w = self._sim['ow'] * self._sim['trueE']**gamma
        sig_h, xedges, yedges = np.histogram2d(
            np.sin(self._sim['dec']),
            self._sim['logE'],
            bins=bins,
            weights=sig_w,
            density=True)
        
        # Normalize histograms by dec band
        bg_h /= np.sum(bg_h,axis=1)[:,None] 
        sig_h /= np.sum(sig_h,axis=1)[:,None]
        
        ratio = sig_h / bg_h
        
        if 'k' not in kwargs: kwargs['k'] = 1
        if 's' not in kwargs: kwargs['s'] = 0
        if 'ext' not in kwargs: kwargs['ext'] = 3

        for i in range(ratio.shape[0]):
            # Pick out the values we want to use.
            # We explicitly want to avoid NaNs and infinities
            values = ratio[i]
            good = np.isfinite(values) & (values>0)
            x, y = bins[1][:-1][good], values[good]

            # Do a linear interpolation across the energy range
            spline = scipy.interpolate.UnivariateSpline(x, y, *args, **kwargs)

            # And store the interpolated values
            ratio[i] = spline(bin_centers)
        if verbose: print('done')
        return ratio
    
    def _create_log_sob_gamma_splines(self,
                                      gamma_bins: np.array,
                                      verbose: bool = False,
                                      *args,
                                      **kwargs) -> List[List[scipy.interpolate.UnivariateSpline]]:
        """Builds a 3D hist of sob vs. sin(dec), log(energy), and gamma, then returns splines of sob vs. gamma.

        More function info...

        Args:
            gamma_bins:
            verbose:

        Returns: A Nested list of splines of shape (sin_dec_bins, log_energy_bins).
        """
        if verbose: print('Building signal-over-background maps...')
        sob_maps = np.array([self._create_sob_map(gamma, verbose) for gamma in gamma_bins])
        if verbose: print('done.')

        if 'k' not in kwargs: kwargs['k'] = 3
        if 's' not in kwargs: kwargs['s'] = 0
        if 'ext' not in kwargs: kwargs['ext'] = 'raise'
            
        transposed_log_sob_maps = np.log(sob_maps.transpose(1, 2, 0))

        if verbose: print('Fitting log(signal-over-background vs. gamma splines)...', end='')
        splines = [[scipy.interpolate.UnivariateSpline(gamma_bins, log_ratios, *args, **kwargs)
                    for log_ratios in dec_bin] for dec_bin in transposed_log_sob_maps]
        if verbose: print('done')
        
        return splines

    def get_log_sob_gamma_splines(self, events: np.ndarray) -> List[scipy.interpolate.UnivariateSpline]:
        """Gets the splines of signal-over-background vs. gamma required for each event.
        
        More function info...
        
        Args:
            events: An array of events including their positional data.
            
        Returns:
            A list of splines of signal-over-background vs gamma for each event.
        """
        # Get the bin that each event belongs to
        sin_dec_idx = np.searchsorted(self._sin_dec_bins[:-1], np.sin(events['dec']))
        log_energy_idx = np.searchsorted(self._log_energy_bins[:-1], events['logE'])
        
        return [self._log_sob_gamma_splines[i][j] for i,j in zip(sin_dec_idx, log_energy_idx)]
    
    @property
    def data(self) -> np.ndarray:
        return self._data
    
    @property
    def sim(self) -> np.ndarray:
        return self._sim
    
    @property
    def grl(self) -> np.ndarray:
        return self._grl
    
    @property
    def background_dec_spline(self) -> scipy.interpolate.UnivariateSpline:
        return self._background_dec_spline