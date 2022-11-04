"""Docstring"""

__author__ = 'John Evans'
__copyright__ = 'Copyright 2021 John Evans'
__credits__ = ['John Evans', 'Jason Fan', 'Michael Larson']
__license__ = 'Apache License 2.0'
__version__ = '0.0.1'
__maintainer__ = 'John Evans'
__email__ = 'john.evans@icecube.wisc.edu'
__status__ = 'Development'

from typing import List, Tuple

import abc
import copy
import dataclasses
import warnings

import numpy as np
from scipy.interpolate import UnivariateSpline as Spline

from .configurable import Configurable
from .params import Params
from .sources import PointSource
from .data_handlers import DataHandler, Injector
from .time_profiles import GenericProfile
from .events import Events


@dataclasses.dataclass(kw_only=True)
class SoBTerm(metaclass=abc.ABCMeta):
    """Docstring"""
    name: str
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


@dataclasses.dataclass(kw_only=True)
class SoBTermFactory(Configurable, metaclass=abc.ABCMeta):
    """
        name: Term Name
    """
    name: str = 'SoBTerm'

    @abc.abstractmethod
    def __call__(self, params: Params, events: Events) -> SoBTerm:
        """Docstring"""

    @abc.abstractmethod
    def calculate_drop_mask(self, events: Events) -> np.ndarray:
        """Docstring"""

    @abc.abstractmethod
    def generate_params(self) -> tuple:
        """Docstring"""


@dataclasses.dataclass(kw_only=True)
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


@dataclasses.dataclass(kw_only=True)
class SpatialTermFactory(SoBTermFactory, Configurable):
    """Docstring"""
    name: str = 'SpatialTerm'
    injector: Injector
    source: PointSource

    def __call__(self, params: Params, events: Events) -> SoBTerm:
        """Docstring"""
        sob_spatial = self.source.spatial_pdf(events)
        sob_spatial /= self.injector.evaluate_background_sindec_pdf(events)
        return SpatialTerm(
            name=self.name,
            _params=params,
            _sob=sob_spatial,
        )

    def calculate_drop_mask(self, events: Events) -> np.ndarray:
        """Docstring"""
        return self.source.spatial_pdf(events) != 0

    def generate_params(self) -> tuple:
        return {}, {}


@dataclasses.dataclass(kw_only=True)
class TimeTerm(SoBTerm):
    """Docstring"""
    times: np.ndarray
    signal_time_profile: GenericProfile
    injector: Injector

    @property
    def params(self) -> Params:
        """Docstring"""
        return self._params

    @params.setter
    def params(self, params: Params) -> None:
        """Docstring"""
        self.signal_time_profile.params = params
        self._params = params

    @property
    def sob(self) -> np.ndarray:
        """Docstring"""
        # assume sorted by times
        idxs = np.searchsorted(
            self.times, self.signal_time_profile.range)

        # the flare edges are event times that should be included
        idxs[1] = idxs[1] + 1

        sob = np.empty(self._sob.shape)
        signal_window = (
            max(self.times[0], self.signal_time_profile.range[0]),
            min(self.times[-1], self.signal_time_profile.range[1]),
        )

        livetime_weight = (
            signal_window[1] - signal_window[0]
        ) / self.injector.contained_livetime(*signal_window)

        sob[:idxs[0]] = 0

        sob[idxs[0]:idxs[1]] = self._sob[
            idxs[0]:idxs[1]
        ] * self.signal_time_profile.pdf_inrange(
            self.times[idxs[0]:idxs[1]]
        ) * livetime_weight

        sob[idxs[1]:] = 0

        return sob


@dataclasses.dataclass(kw_only=True)
class TimeTermFactory(SoBTermFactory, Configurable):
    """Docstring"""
    name: str = 'TimeTerm'
    background_time_profile: GenericProfile
    signal_time_profile: GenericProfile
    injector: Injector

    def __call__(self, params: Params, events: Events) -> SoBTerm:
        """Docstring"""
        times = np.empty(len(events), dtype=events.time.dtype)
        times[:] = events.time[:]
        signal_time_profile = copy.deepcopy(self.signal_time_profile)
        signal_time_profile.params = params
        background_window_width = np.min([
            self.background_time_profile.range[1] - self.background_time_profile.range[0],
            times[-1] - times[0],
        ])
        sob_bg = 1 / self.background_time_profile.pdf(times)
        sob_bg /= self.injector.contained_livetime(*self.background_time_profile.range)
        sob_bg *= background_window_width

        if np.logical_not(np.all(np.isfinite(sob_bg))):
            warnings.warn(
                'Warning, events outside background time profile',
                RuntimeWarning
            )

        return TimeTerm(
            name=self.name,
            _params=params,
            _sob=sob_bg,
            times=times,
            signal_time_profile=signal_time_profile,
            injector=self.injector,
        )

    def calculate_drop_mask(self, events: Events) -> np.ndarray:
        """Docstring"""
        return 1 / self.background_time_profile.pdf(events.time) != 0

    def generate_params(self) -> tuple:
        return self.signal_time_profile.params, self.signal_time_profile.param_bounds


@dataclasses.dataclass(kw_only=True)
class SplineMapEnergyTerm(SoBTerm):
    """Docstring"""
    gamma: float
    splines: List[Spline]
    event_spline_idxs: np.ndarray

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
        spline_evals = np.exp([spline(self.gamma) for spline in self.splines])
        return spline_evals[self.event_spline_idxs]


@dataclasses.dataclass(kw_only=True)
class SplineMapEnergyTermFactory(SoBTermFactory, Configurable):
    """
        init_gamma: Initial Gamma
        sin_dec_bins: sin(Declination) Bins
        log_energy_bins: log_10(Energy) Bins
        log_energy_bounds: log_10(Energy) Bounds (log_10(GeV))
        gamma_bins: Gamma Bins
        gamma_bounds: Gamma Bounds
        norm_bghist_by_dec_band: Normalize Background Histogram By Declination Band
        sob_spline_kwargs: Signal-over-background Spline Keyword Arguments
        energy_spline_kwargs: Energy Spline Keyword Arguments
    """
    data_handler: dataclasses.InitVar[DataHandler]

    init_gamma: float = -2
    sin_dec_bins: int = 50
    log_energy_bins: int = 50
    log_energy_bounds: Tuple[float, float] = (1, 8)
    gamma_bins: int = 50
    gamma_bounds: Tuple[float, float] = (-3.75, -0.5)
    norm_bghist_by_dec_band: bool = False
    sob_spline_kwargs: dict = dataclasses.field(
        default_factory=lambda: {'k': 3, 's': 0, 'ext': 'raise'})
    energy_spline_kwargs: dict = dataclasses.field(
        default_factory=lambda: {'k': 1, 's': 0, 'ext': 3})

    _sob_maps: np.ndarray = dataclasses.field(init=False, repr=False)
    _sin_dec_bins: np.ndarray = dataclasses.field(init=False, repr=False)
    _log_energy_bins: np.ndarray = dataclasses.field(init=False, repr=False)
    _gamma_bins: np.ndarray = dataclasses.field(init=False, repr=False)
    _spline_map: List[List[Spline]] = dataclasses.field(init=False, repr=False)

    def __post_init__(self, data_handler: DataHandler) -> None:
        """Docstring"""
        self._sin_dec_bins = np.linspace(-1, 1, 1 + self.sin_dec_bins)
        self._log_energy_bins = np.linspace(
            *self.log_energy_bounds, 1 + self.log_energy_bins)
        self._gamma_bins = np.linspace(*self.gamma_bounds, 1 + self.gamma_bins)
        self._sob_maps, self._spline_map = self._init_spline_map(data_handler)

    def __call__(self, params: Params, events: Events) -> SoBTerm:
        """Docstring"""
        sin_dec_idx = np.searchsorted(self._sin_dec_bins[:-1], events.sinDec)
        log_energy_idx = np.searchsorted(self._log_energy_bins[:-1], events.logE)

        spline_idxs, event_spline_idxs = np.unique(
            [sin_dec_idx - 1, log_energy_idx - 1],
            return_inverse=True,
            axis=1
        )

        splines = [self._spline_map[i][j] for i, j in np.squeeze(spline_idxs).T]

        return SplineMapEnergyTerm(
            name=self.name,
            _params=params,
            _sob=np.empty(1),
            gamma=self.init_gamma,
            splines=splines,
            event_spline_idxs=event_spline_idxs,
        )

    def calculate_drop_mask(self, events: Events) -> np.ndarray:
        """Docstring"""
        return np.ones(len(events), dtype=bool)

    def generate_params(self) -> tuple:
        return {'gamma': self.init_gamma}, {'gamma': self.gamma_bounds}

    def _init_sob_map(
        self,
        data_handler: DataHandler,
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
        sig_h = data_handler.build_signal_sindec_logenergy_histogram(gamma, bins)

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
            spline = Spline(good_bins, good_vals, **self.energy_spline_kwargs)

            # And store the interpolated values
            ratio[i] = spline(bin_centers)
        return ratio

    def _init_spline_map(self, data_handler: DataHandler) -> List[List[Spline]]:
        """Builds a 3D hist of sob vs. sin(dec), log(energy), and gamma, then
            returns splines of sob vs. gamma.

        Returns: A Nested spline list of shape (sin_dec_bins, log_energy_bins).
        """
        bins = np.array([self._sin_dec_bins, self._log_energy_bins])
        bin_centers = bins[1, :-1] + np.diff(bins[1]) / 2
        bg_h = data_handler.build_background_sindec_logenergy_histogram(bins)

        # Normalize histogram by dec band
        bg_h /= np.sum(bg_h, axis=1)[:, None]
        if self.norm_bghist_by_dec_band:
            bg_h[bg_h <= 0] = np.min(bg_h[bg_h > 0])

        sob_maps = np.array([
            self._init_sob_map(data_handler, gamma, bins, bin_centers, bg_h)
            for gamma in self._gamma_bins
        ])



        transposed_log_sob_maps = np.log(sob_maps.transpose(1, 2, 0))

        splines = [[
            Spline(self._gamma_bins, log_ratios, **self.sob_spline_kwargs)
            for log_ratios in dec_bin
        ] for dec_bin in transposed_log_sob_maps]

        return sob_maps, splines

    @property
    def sob_maps(self) -> np.ndarray:
        return self._sob_maps
