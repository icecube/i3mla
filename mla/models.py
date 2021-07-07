"""
The classes in this file do preprocessing on data and monte carlo to be used
to do a point source analysis.
"""

__author__ = 'John Evans'
__copyright__ = 'Copyright 2020 John Evans'
__credits__ = ['John Evans', 'Jason Fan', 'Michael Larson']
__license__ = 'Apache License 2.0'
__version__ = '0.0.1'
__maintainer__ = 'John Evans'
__email__ = 'john.evans@icecube.wisc.edu'
__status__ = 'Development'

from typing import List, Optional, Tuple, Union

import numpy as np
import numpy.lib.recfunctions as rf
from scipy.interpolate import UnivariateSpline as Spline

from dataclasses import dataclass
from dataclasses import field
from dataclasses import InitVar

from . import sources
from . import _models


@dataclass
class _I3EventModelBase(_models.EventModelBase):
    """Docstring

    Attributes:
        sin_dec_bins (np.array): An array of sin(dec) bin edges for the energy
            maps.
        log_energy_bins (np.array): An array of log(energy) bin edges for the
            energy maps.
        log_sob_gamma_splines (List[List[scipy.interpolate.UnivariateSpline]]):
            A 2D list of spline fits of the log(signal-over-background) vs.
            gamma at a binned energy and sin(dec).
        """
    _sin_dec_bins: np.array = field(init=False)
    _log_energy_bins: np.array = field(init=False)
    _log_sob_gamma_splines: List[List[Spline]] = field(init=False)


@dataclass
class _I3EventModelDefaultsBase(_models.TdEventModelDefaultsBase):
    """Docstring"""
    signal_sin_dec_bins: InitVar[Union[np.array, int]] = field(default=50)
    log_energy_bins: InitVar[Union[np.array, int]] = field(default=50)
    gamma_bins: InitVar[Union[np.array, int]] = field(default=50)
    verbose: InitVar[bool] = field(default=False)


@dataclass
class I3EventModel(
    _models.TdEventModel,
    _I3EventModelDefaultsBase,
    _I3EventModelBase,
):
    """Docstring"""
    def __post_init__(
        self,
        source: sources.Source,
        data: np.ndarray,
        sim: np.ndarray,
        grl: np.ndarray,
        gamma: float,
        sampling_width: Optional[float],
        background_sin_dec_bins: Union[np.array, int],
        background_window: float,
        withinwindow: bool,
        signal_sin_dec_bins: Union[np.array, int],
        log_energy_bins: Union[np.array, int],
        gamma_bins: Union[np.array, int],
        verbose: bool,
    ) -> None:
        """Docstring"""
        super().__post_init__(
            source,
            data,
            sim,
            grl,
            gamma,
            sampling_width,
            background_sin_dec_bins,
            background_window,
            withinwindow,
        )

        if isinstance(signal_sin_dec_bins, int):
            signal_sin_dec_bins = np.linspace(-1, 1, 1 + signal_sin_dec_bins)
        self._sin_dec_bins = signal_sin_dec_bins

        if isinstance(log_energy_bins, int):
            log_energy_bins = np.linspace(1, 8, 1 + log_energy_bins)
        self._log_energy_bins = log_energy_bins

        if isinstance(gamma_bins, int):
            gamma_bins = np.linspace(-4.25, -0.5, 1 + gamma_bins)

        self._log_sob_gamma_splines = self._init_log_sob_gamma_splines(
            gamma_bins, verbose=verbose)

    def _init_sob_map(self, gamma: float, *args, verbose: bool = False,
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
        bin_centers = bins[1, :-1] + np.diff(bins[1]) / 2

        # background
        bg_h, _, _ = np.histogram2d(self._data['sindec'], self._data['logE'],
                                    bins=bins, density=True)

        # signal
        sig_w = self._sim['ow'] * self._sim['trueE']**gamma
        sig_h, _, _ = np.histogram2d(self._sim['sindec'], self._sim['logE'],
                                     bins=bins, weights=sig_w, density=True)

        # Normalize histograms by dec band
        bg_h /= np.sum(bg_h, axis=1)[:, None]
        sig_h /= np.sum(sig_h, axis=1)[:, None]

        # div-0 okay here
        with np.errstate(divide='ignore', invalid='ignore'):
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
            good = np.isfinite(ratio[i]) & (ratio[i] > 0)
            good_bins, good_vals = bin_centers[good], ratio[i][good]

            # Do a linear interpolation across the energy range
            spline = Spline(good_bins, good_vals, *args, **kwargs)

            # And store the interpolated values
            ratio[i] = spline(bin_centers)
        if verbose:
            print('done')
        return ratio

    def _init_log_sob_gamma_splines(self, gamma_bins: np.array, *args,
                                    verbose: bool = False,
                                    **kwargs) -> List[List[Spline]]:
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
        sob_maps = np.array([self._init_sob_map(gamma, verbose=verbose)
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

        splines = [[
            Spline(gamma_bins, log_ratios, *args, **kwargs)
            for log_ratios in dec_bin
        ] for dec_bin in transposed_log_sob_maps]

        if verbose:
            print('done')

        return splines

    def log_sob_spline_prepro(
        self,
        events: np.ndarray,
    ) -> Tuple[np.ndarray, List]:
        """Docstring"""
        # Get the bin that each event belongs to
        sin_dec_idx = np.searchsorted(self._sin_dec_bins[:-1],
                                      events['sindec'])

        log_energy_idx = np.searchsorted(self._log_energy_bins[:-1],
                                         events['logE'])

        spline_idxs, event_spline_idxs = np.unique(
            [sin_dec_idx - 1, log_energy_idx - 1],
            return_inverse=True,
            axis=1
        )

        splines = [
            self._log_sob_gamma_splines[i][j]
            for i, j in spline_idxs.T
        ]

        return np.array(event_spline_idxs, dtype=int), splines

    def get_sob_energy(
        self,
        gamma: float,
        splines: List[Spline],
        event_spline_idxs: np.ndarray,
    ) -> np.array:
        """Docstring"""
        spline_evals = np.exp([spline(gamma) for spline in splines])
        return spline_evals[event_spline_idxs]


class MutliEventModel(object):
    """Construct multi-sample event models.Only works for uniform time profile."""
    def __init__(
        self,
        listofmodels: List[EventModel],
    ) -> None:
    """Docstring"""
        self.listofmodels = listofmodels
        self.livetime_ratio = [] #livetime ratio between sample
        self._log_energy_bins = self.listofmodels[0]._log_energy_bins
        self._sin_dec_bins = self.listofmodels[0]._sin_dec_bins
        self._p = []
        self.mc_index = []

        for i,sample in enumerate(self.listofmodels):         
            self.livetime_ratio.append(sample.signal_contained_livetime)

        self.livetime_ratio = np.array(self.livetime_ratio)
        self.livetime_ratio /= np.sum(self.livetime_ratio)

        for i,sample in enumerate(self.listofmodels):
            mc_array = rf.append_fields(
                            np.empty((sample.prob)),
                            'p',
                            sample.prob * self.livetime_ratio[i],
                            usemask=False,
            )
            mc_array = rf.append_fields(
                            mc_array,
                            'index',
                            np.arange(len(mc_array)),
                            usemask=False,
            )
            mc_array = rf.append_fields(
                            mc_array,
                            'sample',
                            i,
                            usemask=False,
            )
            self.mc_index.append(mc_array)

        self.mc_index = np.array(self.mc_index)

        
    def inject_background_events(self) -> np.ndarray:
        """Injects background events for a trial.

        Args:
            event_model: Preprocessed data and simulation.

        Returns:
            An array of injected background events.
        """
        background_list = []
        for i,sample in enumerate(self.listofmodels):
            background = sample.inject_background_events()
            background = rf.append_fields(
                         background,
                         'sample_index',
                         i,
                         usemask=False,
            )
            background_list.append(background)
        
        return np.vstack(background_list)

    def inject_signal_events(
        self,
        flux_norm: float,
        n_signal_observed: Optional[int] = None,
    ) -> np.ndarray:
        """Injects signal events for a trial.

        Args:
            flux_norm:
            n_signal_observed:

        Returns:
            An array of injected signal events.
        """

        # Pick the signal events
        signal_list = []
        if n_signal_observed is not None:
            sample_index = np.random.choice(
                self.mc_index,
                n_signal_observed,
                p=self.mc_index['p'],
                replace=False,
            ).copy()
            for i,sample in enumerate(self.listofmodels):
                mc_sampled = sample_index[sample_index['sample'] == i]
                signal = sample.reduced_sim[mc_sampled['index']].copy()
                signal_list.append(signal)

        else:
            for i,sample in enumerate(self.listofmodels):
                flux = flux_norm * self.livetime_ratio[i] #scale the flux norm
                signal = sample.inject_signal_events(flux)
                signal = rf.append_fields(
                             signal,
                             'sample_index',
                             i,
                             usemask=False,
                )            
                signal_list.append(signal)
        signal = np.vstack(signal_list)
        return signal
    
    def cal_ns(self, flux_norm):
        """Docstring"""
        ns = 0
        for i,sample in enumerate(self.listofmodels):
            flux = flux_norm * self.livetime_ratio[i] #scale the flux norm
            ns += sample.cal_ns(flux)
        return ns
    
    def cal_flux_norm(self, ns):
        """Docstring"""
        flux_norm = 0
        for i,sample in enumerate(self.listofmodels):
            ns_sample = ns * self.livetime_ratio[i] #scale the flux norm
            flux_norm += sample.cal_flux_norm(ns_sample)
            
        return flux_norm


    def log_sob_spline_prepro(
        self,
        events: np.ndarray,
    ) -> Tuple[np.ndarray, List]:
        """Docstring"""
        # Get the bin that each event belongs to
        sin_dec_idx = np.searchsorted(self.listofmodels[0]._sin_dec_bins[:-1],
                                      events['sindec'])

        log_energy_idx = np.searchsorted(
            self.listofmodels[0]._log_energy_bins[:-1],
            events['logE']
        )

        spline_idxs, event_spline_idxs = np.unique(
            [sin_dec_idx - 1, log_energy_idx - 1],
            return_inverse=True,
            axis=1
        )
        splines = []
        for k,sample in enumerate(self.listofmodels):
            splines = [
                self._log_sob_gamma_splines[i][j]
                for i, j in spline_idxs.T
            ]

        return np.array(event_spline_idxs, dtype=int), splines

    def get_sob_energy(
        self,
        gamma: float,
        splines: List[Spline],
        event_spline_idxs: np.ndarray,
    ) -> np.array:
        """Docstring"""
        spline_evals = np.exp([spline(gamma) for spline in splines])
        return spline_evals[event_spline_idxs]