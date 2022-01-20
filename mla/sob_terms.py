"""Docstring"""

__author__ = 'John Evans'
__copyright__ = 'Copyright 2021 John Evans'
__credits__ = ['John Evans', 'Jason Fan', 'Michael Larson']
__license__ = 'Apache License 2.0'
__version__ = '0.0.1'
__maintainer__ = 'John Evans'
__email__ = 'john.evans@icecube.wisc.edu'
__status__ = 'Development'

from typing import List

import abc
import copy
import dataclasses
import warnings

import numpy as np
from scipy.interpolate import UnivariateSpline as Spline

from . import configurable
from .params import Params
from .sources import PointSource
from .data_handlers import DataHandler
from .time_profiles import GenericProfile


@dataclasses.dataclass
class SoBTerm:
    """Docstring"""
    __metaclass__ = abc.ABCMeta
    _params: Params
    _sob: np.ndarray

    @property
    @abc.abstractmethod
    def params(self) -> Params:
        """Docstring"""

    @params.setter
    @abc.abstractmethod
    def params(self, params: Params) -> None:
        """Docstring"""

    @property
    @abc.abstractmethod
    def sob(self) -> np.ndarray:
        """Docstring"""


@dataclasses.dataclass
class SoBTermFactory(configurable.Configurable):
    """Docstring"""
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __call__(self, params: Params, events: np.ndarray) -> SoBTerm:
        """Docstring"""

    @abc.abstractmethod
    def calculate_drop_mask(self, events: np.ndarray) -> np.ndarray:
        """Docstring"""

    @abc.abstractmethod
    def generate_params(self) -> tuple:
        """Docstring"""


@dataclasses.dataclass
class SpatialTerm(SoBTerm):
    """Docstring"""

    @property
    def params(self) -> Params:
        """Docstring"""
        return self._params

    @params.setter
    def params(self, params: Params) -> None:
        """Docstring"""
        self._params = params

    @property
    def sob(self) -> np.ndarray:
        """Docstring"""
        return self._sob


@dataclasses.dataclass
class SpatialTermFactory(SoBTermFactory):
    """Docstring"""
    data_handler: DataHandler
    source: PointSource

    def __call__(self, params: Params, events: np.ndarray) -> SoBTerm:
        """Docstring"""
        sob_spatial = self.source.spatial_pdf(events)
        sob_spatial /= self.data_handler.evaluate_background_sindec_pdf(events)
        return SpatialTerm(_params=params, _sob=sob_spatial)

    def calculate_drop_mask(self, events: np.ndarray) -> np.ndarray:
        """Docstring"""
        return self.source.spatial_pdf(events) != 0

    def generate_params(self) -> tuple:
        return {}, {}


@dataclasses.dataclass
class TimeTerm(SoBTerm):
    """Docstring"""
    _times: np.ndarray
    _signal_time_profile: GenericProfile

    @property
    def params(self) -> Params:
        """Docstring"""
        return self._params

    @params.setter
    def params(self, params: Params) -> None:
        """Docstring"""
        self._signal_time_profile.params = params
        self._params = params

    @property
    def sob(self) -> np.ndarray:
        """Docstring"""
        return self._sob * self._signal_time_profile.pdf(self._times)


@dataclasses.dataclass
class TimeTermFactory(SoBTermFactory):
    """Docstring"""
    background_time_profile: GenericProfile
    signal_time_profile: GenericProfile

    def __call__(self, params: Params, events: np.ndarray) -> SoBTerm:
        """Docstring"""
        times = np.empty(len(events), dtype=events['time'].dtype)
        times[:] = events['time'][:]
        signal_time_profile = copy.deepcopy(self.signal_time_profile)
        signal_time_profile.params = params
        sob_bg = 1 / self.background_time_profile.pdf(times)

        if np.logical_not(np.all(np.isfinite(sob_bg))):
            warnings.warn(
                'Warning, events outside background time profile',
                RuntimeWarning
            )

        return TimeTerm(
            _params=params,
            _sob=sob_bg,
            _times=times,
            _signal_time_profile=signal_time_profile,
        )

    def calculate_drop_mask(self, events: np.ndarray) -> np.ndarray:
        """Docstring"""
        return 1 / self.background_time_profile.pdf(events['time']) != 0

    def generate_params(self) -> tuple:
        return self.signal_time_profile.params, self.signal_time_profile.param_bounds


@dataclasses.dataclass
class SplineMapEnergyTerm(SoBTerm):
    """Docstring"""
    gamma: float
    _splines: List[Spline]
    _event_spline_idxs: np.ndarray

    @property
    def params(self) -> Params:
        """Docstring"""
        return self._params

    @params.setter
    def params(self, params: Params) -> None:
        """Docstring"""
        if 'gamma' in params:
            self.gamma = params['gamma']
        self._params = params

    @property
    def sob(self) -> np.ndarray:
        """Docstring"""
        spline_evals = np.exp([spline(self.gamma) for spline in self._splines])
        return spline_evals[self._event_spline_idxs]


@dataclasses.dataclass
class SplineMapEnergyTermFactory(SoBTermFactory):
    """Docstring"""
    data_handler: DataHandler
    _sin_dec_bins: np.ndarray = dataclasses.field(init=False, repr=False)
    _log_energy_bins: np.ndarray = dataclasses.field(init=False, repr=False)
    _gamma_bins: np.ndarray = dataclasses.field(init=False, repr=False)
    _spline_map: List[List[Spline]] = dataclasses.field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Docstring"""
        self._sin_dec_bins = np.linspace(-1, 1, 1 + self.config['sin_dec_bins'])
        self._log_energy_bins = np.linspace(
            *self.config['log_energy_bounds'], 1 + self.config['log_energy_bins'])
        self._gamma_bins = np.linspace(
            *self.config['gamma_bounds'], 1 + self.config['gamma_bins'])
        self._spline_map = self._init_spline_map()

    def __call__(self, params: Params, events: np.ndarray) -> SoBTerm:
        """Docstring"""
        sin_dec_idx = np.searchsorted(self._sin_dec_bins[:-1], events['sindec'])
        log_energy_idx = np.searchsorted(self._log_energy_bins[:-1], events['logE'])

        spline_idxs, event_spline_idxs = np.unique(
            [sin_dec_idx - 1, log_energy_idx - 1],
            return_inverse=True,
            axis=1
        )

        splines = [self._spline_map[i][j] for i, j in spline_idxs.T]

        return SplineMapEnergyTerm(
            _params=params,
            _sob=np.empty(1),
            gamma=self.config['initial_gamma'],
            _splines=splines,
            _event_spline_idxs=event_spline_idxs,
        )

    def calculate_drop_mask(self, events: np.ndarray) -> np.ndarray:
        """Docstring"""
        return np.ones(len(events), dtype=bool)

    def generate_params(self) -> tuple:
        return {'gamma': -2}, {'gamma': (-4, -1)}

    def _init_sob_map(
        self,
        gamma: float,
        bins: np.ndarray,
        bin_centers: np.ndarray,
        bg_h: np.ndarray,
    ) -> np.ndarray:
        """Creates sob histogram for a given spectral index (gamma).

        Args:
            gamma: The gamma value to use to weight the signal.

        Returns:
            An array of signal-over-background values binned in sin(dec) and
            log(energy) for a given gamma.
        """
        sig_h = self.data_handler.build_signal_sindec_logenergy_histogram(gamma, bins)

        # Normalize histogram by dec band
        sig_h /= np.sum(sig_h, axis=1)[:, None]

        # div-0 okay here
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = sig_h / bg_h

        for i in range(ratio.shape[0]):
            # Pick out the values we want to use.
            # We explicitly want to avoid NaNs and infinities
            good = np.isfinite(ratio[i]) & (ratio[i] > 0)
            good_bins, good_vals = bin_centers[good], ratio[i][good]

            # Do a linear interpolation across the energy range
            spline = Spline(
                good_bins,
                good_vals,
                k=self.config['energy_spline_k'],
                s=self.config['energy_spline_s'],
                ext=self.config['energy_spline_ext'],
            )

            # And store the interpolated values
            ratio[i] = spline(bin_centers)
        return ratio

    def _init_spline_map(self) -> List[List[Spline]]:
        """Builds a 3D hist of sob vs. sin(dec), log(energy), and gamma, then
            returns splines of sob vs. gamma.

        Returns: A Nested spline list of shape (sin_dec_bins, log_energy_bins).
        """
        bins = np.array([self._sin_dec_bins, self._log_energy_bins])
        bin_centers = bins[1, :-1] + np.diff(bins[1]) / 2
        bg_h = self.data_handler.build_background_sindec_logenergy_histogram(bins)

        # Normalize histogram by dec band
        bg_h /= np.sum(bg_h, axis=1)[:, None]

        sob_maps = np.array([
            self._init_sob_map(gamma, bins, bin_centers, bg_h)
            for gamma in self._gamma_bins
        ])

        transposed_log_sob_maps = np.log(sob_maps.transpose(1, 2, 0))

        splines = [[
            Spline(
                self._gamma_bins,
                log_ratios,
                k=self.config['sob_spline_k'],
                s=self.config['sob_spline_s'],
                ext=self.config['sob_spline_ext'],
            )
            for log_ratios in dec_bin
        ] for dec_bin in transposed_log_sob_maps]

        return splines

    @classmethod
    def generate_config(cls):
        """Docstring"""
        config = super().generate_config()
        config['initial_gamma'] = -2
        config['sin_dec_bins'] = 50
        config['log_energy_bins'] = 50
        config['log_energy_bounds'] = (1, 8)
        config['gamma_bins'] = 50
        config['gamma_bounds'] = (-4.25, -0.5)
        config['sob_spline_k'] = 3
        config['sob_spline_s'] = 0
        config['sob_spline_ext'] = 'raise'
        config['energy_spline_k'] = 1
        config['energy_spline_s'] = 0
        config['energy_spline_ext'] = 3
        return config
