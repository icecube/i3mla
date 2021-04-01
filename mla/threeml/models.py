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

from typing import Optional, Tuple, Union
import numpy as np
import numpy.lib.recfunctions as rf
from scipy.interpolate import UnivariateSpline as Spline

from dataclasses import dataclass
from dataclasses import field
from dataclasses import InitVar

from .. import sources
from .. import _models
from . import spectral


@dataclass
class _ThreeMLEventModelBase(_models.EventModelBase):
    """Docstring"""
    _sin_dec_bins: np.array = field(init=False)
    _log_energy_bins: np.array = field(init=False)
    _edge_point: Tuple[float, float] = field(init=False)
    _background_sob_map: np.ndarray = field(init=False)
    _ratio: np.ndarray = field(init=False)
    _reduced_sim_reconstructed: np.ndarray = field(init=False)


@dataclass
class _ThreeMLEventModelDefaultsBase(_models.TdEventModelDefaultsBase):
    """Docstring"""
    signal_sin_dec_bins: InitVar[Union[np.array, int]] = field(default=50)
    log_energy_bins: InitVar[Union[np.array, int]] = field(default=50)
    _spectrum: spectral.BaseSpectrum = field(
        default=spectral.PowerLaw(1e3, 1e-14, -2))


@dataclass
class ThreeMLEventModel(
    _models.TdEventModel,
    _ThreeMLEventModelDefaultsBase,
    _ThreeMLEventModelBase,
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
    ) -> None:
        """
        Args:
            source:
            grl:
            background_sin_dec_bins: If an int, then the number of bins
                spanning -1 -> 1, otherwise, a numpy array of bin edges.
            background_window:
            withinwindow:
        """
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
        self._background_sob_map = self._init_background_sob_map()
        self._init_reduced_sim_reconstructed(source)
        self._ratio = self._init_sob_ratio()

    def _init_background_sob_map(self) -> None:
        """Create the backgroub SOB map
        """
        # background
        bins = np.array([self._sin_dec_bins, self._log_energy_bins])
        bg_h, _, _ = np.histogram2d(self._data['sindec'], self._data['logE'],
                                    bins=bins, density=True)
        bg_h /= np.sum(bg_h, axis=1)[:, None]
        return bg_h

    def _init_sob_ratio(self, *args, **kwargs) -> None:
        """Create the SOB map with a spectrum
        """
        bins = np.array([self._sin_dec_bins, self._log_energy_bins])
        bin_centers = bins[1, :-1] + np.diff(bins[1]) / 2
        sig_w = self._reduced_sim_reconstructed['ow'] * self._spectrum(
            self._reduced_sim_reconstructed['trueE'])
        sig_h, _, _ = np.histogram2d(self._reduced_sim_reconstructed['sindec'],
                                     self._reduced_sim_reconstructed['logE'],
                                     bins=bins, weights=sig_w, density=True)

        # Normalize histograms by dec band
        with np.errstate(divide='ignore'):  # divide zero warnings
            sig_h /= np.sum(sig_h, axis=1)[:, None]

        if 'k' not in kwargs:
            kwargs['k'] = 1
        if 's' not in kwargs:
            kwargs['s'] = 0
        if 'ext' not in kwargs:
            kwargs['ext'] = 3

        with np.errstate(divide='ignore'):  # divide zero warnings
            ratio = sig_h / self._background_sob_map

        with np.errstate(divide='ignore'):  # divide zero warnings
            for i in range(ratio.shape[0]):
                # Pick out the values we want to use.
                # We explicitly want to avoid NaNs and infinities
                values = ratio[i]
                good = np.isfinite(values) & (values > 0)
                x_good, y_good = bin_centers[good], values[good]

                # Do a linear interpolation across the energy range
                if len(x_good) > 1:
                    spline = Spline(x_good, y_good, *args, **kwargs)
                    ratio[i] = spline(bin_centers)
                elif len(x_good) == 1:
                    ratio[i] = y_good
                else:
                    ratio[i] = 0
        return ratio

    def _init_reduced_sim_reconstructed(self, source: sources.Source) -> None:
        """Gets a small simulation dataset to use for injecting signal.

        Prunes the simulation set to only events close to a given source and
        calculate the weight for each event. Adds the weights as a new column
        to the simulation set.

        Args:
            source:

        Returns:
            A reweighted simulation set around the source declination.
        """
        if self._sampling_width is not None:
            self._cut_sim_reconstructed(source)
        else:
            self._reduced_sim_reconstructed = self._sim

    def _cut_sim_reconstructed(self, source: sources.Source) -> np.ndarray:
        """Select simulation events in a reconstruction dec band

        Args:
            source:
        """
        if self._sampling_width is not None:
            self._edge_point = (np.searchsorted(
                                self._sin_dec_bins,
                                np.sin(source.dec - self._sampling_width)) - 1,
                                np.searchsorted(
                                self._sin_dec_bins,
                                np.sin(source.dec + self._sampling_width)) - 1)
        else:
            self._edge_point = (self._sin_dec_bins[0], self._sin_dec_bins[-1])
        sindec_dist = np.abs(source.dec - self._sim['dec'])
        close = sindec_dist < self._sampling_width
        self._reduced_sim_reconstructed = self._sim[close].copy()

    def _weight_reduced_sim(self, reduced_sim: np.ndarray) -> np.ndarray:
        """Docstring"""
        try:
            reduced_sim = rf.append_fields(reduced_sim, 'weight',
                                           np.zeros(len(reduced_sim)),
                                           dtypes=np.float32)

        except ValueError:  # weight already exist
            pass

        # Assign the weights using the newly defined "time profile"
        # classes above. If you want to make this a more complicated
        # shape, talk to me and we can work it out.
        reduced_sim['weight'] = reduced_sim['ow'] * self._spectrum(
            reduced_sim['trueE'])
        return reduced_sim

    def reweight_reduced_sim(self, spectrum: spectral.BaseSpectrum):
        """Docstring"""
        self._reduced_sim['weight'] = self._reduced_sim['ow'] * spectrum(
            self._reduced_sim['trueE'])

    def prepro_index(self, events: np.ndarray) -> np.ndarray:
        """Find the sindec index and energy index for events

        More function info...

        Args:
            events: An array of events including their positional data.

        Returns:
            A list of index
        """
        # Get the bin that each event belongs to
        try:
            sin_dec_idx = np.searchsorted(self._sin_dec_bins[:-1],
                                          events['sindec']) - 1
        except ValueError:
            sin_dec_idx = np.searchsorted(self._sin_dec_bins[:-1],
                                          np.sin(events['dec'])) - 1

        log_energy_idx = np.searchsorted(self._log_energy_bins[:-1],
                                         events['logE']) - 1

        sin_dec_idx[sin_dec_idx < self._edge_point[0]] = self._edge_point[0]
        # If events fall outside the sampling width, just gonna approxiamte the
        # weight using the nearest non-zero sinDec bin.
        sin_dec_idx[sin_dec_idx > self._edge_point[1]] = self._edge_point[1]
        return sin_dec_idx, log_energy_idx

    def _energy_sob(
        self,
        sin_dec_idx: np.ndarray,
        log_energy_idx: np.ndarray
    ) -> np.ndarray:
        """Gets the sob vs. gamma required for each event and specific .

        More function info...

        Args:
            sin_dec_idx: An array of sin dec index of events
            log_energy_idx: An array of log energy index of events

        Returns:
            signal-over-background for each event.
        """
        return self._ratio[sin_dec_idx, log_energy_idx]

    def get_sob_energy(
        self,
        sin_dec_idx: np.ndarray,
        log_energy_idx: np.ndarray,
    ) -> np.ndarray:
        """Docstring"""
        return self._energy_sob(sin_dec_idx, log_energy_idx)

    @property
    def edge_point(self) -> Tuple[float, float]:
        """Docstring"""
        return self._edge_point

    @property
    def spectrum(self) -> spectral.BaseSpectrum:
        """Docstring"""
        return self._spectrum

    @spectrum.setter
    def spectrum(self, spectrum: spectral.BaseSpectrum):
        """Docstring"""
        self._spectrum = spectrum
