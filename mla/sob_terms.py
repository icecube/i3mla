"""Docstring"""

__author__ = 'John Evans'
__copyright__ = 'Copyright 2021 John Evans'
__credits__ = ['John Evans', 'Jason Fan', 'Michael Larson']
__license__ = 'Apache License 2.0'
__version__ = '0.0.1'
__maintainer__ = 'John Evans'
__email__ = 'john.evans@icecube.wisc.edu'
__status__ = 'Development'

from typing import Callable, List, Optional, Sequence, Tuple
from typing import TYPE_CHECKING

import abc
import copy
import dataclasses
import warnings
import numpy as np

if TYPE_CHECKING:
    from . import sources
    from . import _models
    from . import models
    from . import time_profiles
    from threeml import models as threeml_models
else:
    sources = object  # pylint: disable=invalid-name
    _models = object  # pylint: disable=invalid-name
    models = object  # pylint: disable=invalid-name
    time_profiles = object  # pylint: disable=invalid-name
    threeml_models = object  # pylint: disable=invalid-name


Bounds = Optional[Sequence[Tuple[float, float]]]
SpatialPDF = Callable[[np.ndarray, sources.Source], np.ndarray]


def angular_distance(src_ra: float, src_dec: float, r_a: float,
                     dec: float) -> float:
    """Computes angular distance between source and location.

    Args:
        src_ra: The right ascension of the first point (radians).
        src_dec: The declination of the first point (radians).
        r_a: The right ascension of the second point (radians).
        dec: The declination of the second point (radians).

    Returns:
        The distance, in radians, between the two points.
    """
    sin_dec = np.sin(dec)

    cos_dec = np.sqrt(1. - sin_dec**2)

    cos_dist = (
        np.cos(src_ra - r_a) * np.cos(src_dec) * cos_dec
    ) + np.sin(src_dec) * sin_dec
    # handle possible floating precision errors
    cos_dist = np.clip(cos_dist, -1, 1)

    return np.arccos(cos_dist)


def gauassian_spatial_pdf(
    events: np.ndarray,
    source: sources.Source,
) -> np.ndarray:
    """Docstring"""
    r_a, dec = source.get_location()
    sigma_sq = events['angErr']**2 + source.get_sigma()**2
    dist = angular_distance(events['ra'], events['dec'], r_a,
                            dec)
    norm = 1 / (2 * np.pi * sigma_sq)
    return norm * np.exp(-dist**2 / (2 * sigma_sq))


@dataclasses.dataclass
class SoBTerm:
    """Docstring"""
    __metaclass__ = abc.ABCMeta
    _params: np.ndarray
    _sob: np.ndarray

    @property
    def params(self) -> np.ndarray:
        """Docstring"""
        return self._params

    @abc.abstractmethod
    @params.setter
    def params(self, params: np.ndarray) -> None:
        """Docstring"""

    @abc.abstractmethod
    @property
    def sob(self) -> np.ndarray:
        """Docstring"""


@dataclasses.dataclass
class SoBTermFactory:
    """Docstring"""
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __call__(
        self,
        params: np.ndarray,
        bounds: Bounds,
        events: np.ndarray,
        event_model: _models.EventModel,
        source: sources.Source,
    ) -> SoBTerm:
        """Docstring"""

    @abc.abstractmethod
    def update_bounds(self, bounds: Bounds) -> Bounds:
        """Docstring"""

    @abc.abstractmethod
    def calculate_drop_mask(
        self,
        events: np.ndarray,
        source: sources.Source,
    ) -> np.ndarray:
        """Docstring"""


@dataclasses.dataclass
class SpatialTerm(SoBTerm):
    """Docstring"""

    @params.setter
    def params(self, params: np.ndarray) -> None:
        """Docstring"""
        self._params = params

    @property
    def sob(self) -> np.ndarray:
        """Docstring"""
        return self._sob


@dataclasses.dataclass
class SpatialTermFactory(SoBTermFactory):
    """Docstring"""
    spatial_pdf: SpatialPDF

    def __call__(self,
        params: np.ndarray,
        bounds: Bounds,
        events: np.ndarray,
        event_model: _models.EventModel,
        source: sources.Source,
    ) -> SpatialTerm:
        """Docstring"""
        sob_spatial = self.spatial_pdf(events, source)
        sob_spatial /= event_model.background_spatial_pdf(events)
        return SpatialTerm(_params=params, _sob=sob_spatial)

    def update_bounds(self, bounds: Bounds) -> Bounds:
        """Docstring"""
        return bounds

    def calculate_drop_mask(
        self,
        events: np.ndarray,
        source: sources.Source,
    ) -> np.ndarray:
        """Docstring"""
        return self.spatial_pdf(events, source) != 0


@dataclasses.dataclass
class TimeTerm(SoBTerm):
    """Docstring"""
    _times: np.ndarray
    _background_time_profile: time_profiles.GenericProfile
    _signal_time_profile: time_profiles.GenericProfile

    @params.setter
    def params(self, params: np.ndarray) -> None:
        """Docstring"""
        self._signal_time_profile.params = params

    @property
    def sob(self) -> np.ndarray:
        """Docstring"""
        return self._sob * self._signal_time_profile.pdf(self._times)


@dataclasses.dataclass
class TimeTermFactory(SoBTermFactory):
    """Docstring"""
    background_time_profile: time_profiles.GenericProfile
    signal_time_profile: time_profiles.GenericProfile

    def __call__(
        self,
        params: np.ndarray,
        bounds: Bounds,
        events: np.ndarray,
        event_model: _models.EventModel,
        source: sources.Source,
    ) -> SoBTerm:
        """Docstring"""
        times = np.empty(len(events), dtype=events['time'].dtype)
        times[:] = events['time'][:]
        signal_time_profile = copy.deepcopy(self.signal_time_profile)
        background_time_profile = copy.deepcopy(self.background_time_profile)
        signal_time_profile.params = params
        sob_bg = 1 / background_time_profile.pdf(times)

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
            _background_time_profile=background_time_profile,
        )

    def update_bounds(self, bounds: Bounds) -> Bounds:
        """Docstring"""
        return bounds

    def calculate_drop_mask(
        self,
        events: np.ndarray,
        source: sources.Source,
    ) -> np.ndarray:
        """Docstring"""
        return 1 / self.background_time_profile.pdf(events['time']) != 0


@dataclasses.dataclass
class SplineMapEnergyTerm(SoBTerm):
    """Docstring"""
    _gamma: float
    _sob_func: Callable
    _spline_idxs: np.ndarray
    _splines: list

    def __post_init__(self) -> None:
        """Docstring"""
        self.params = self._params

    @params.setter
    def params(self, params: np.ndarray) -> None:
        """Docstring"""
        if 'gamma' in params.dtype.names:
            self._gamma = params['gamma']
        self._params = params

    @property
    def sob(self) -> np.ndarray:
        """Docstring"""
        return self._sob_func(self._gamma, self._splines, self._spline_idxs)


@dataclasses.dataclass
class SplineMapEnergyTermFactory(SoBTermFactory):
    """Docstring"""
    gamma: float = -2

    def __call__(
        self,
        params: np.ndarray,
        bounds: Bounds,
        events: np.ndarray,
        event_model: models.SplineMapEventModel,
        source: sources.Source,
    ) -> SoBTerm:
        """Docstring"""
        spline_idxs, splines = event_model.map_splines_to_events(events)
        return SplineMapEnergyTerm(
            _params=params,
            _sob=np.empty(1),
            _gamma=self.gamma,
            _sob_func=event_model.get_sob_energy,
            _spline_idxs=spline_idxs,
            _splines=splines
        )

    def update_bounds(self, bounds: Bounds) -> Bounds:
        """Docstring"""
        return bounds

    def calculate_drop_mask(
        self,
        events: np.ndarray,
        source: sources.Source,
    ) -> np.ndarray:
        """Docstring"""
        return np.ones(len(events), dtype=bool)


@dataclasses.dataclass
class ThreeMLEnergyTerm(SoBTerm):
    """Docstring"""
    _sob_func: Callable
    _sin_dec_idx: np.ndarray
    _log_energy_idx: List

    @params.setter
    def params(self, params: np.ndarray) -> None:
        """Docstring"""
        self._params = params

    @property
    def sob(self) -> np.ndarray:
        """Docstring"""
        return self._sob_func(self._sin_dec_idx, self._log_energy_idx)


@dataclasses.dataclass
class ThreeMLEnergyTermFactory(SoBTermFactory):
    """Docstring"""

    def __call__(
        self,
        params: np.ndarray,
        bounds: Bounds,
        events: np.ndarray,
        event_model: threeml_models.ThreeMLEventModel,
        source: sources.Source,
    ) -> SoBTerm:
        """Docstring"""
        sin_dec_idx, log_energy_idx = event_model.map_dec_energy_to_events(
            events)
        return ThreeMLEnergyTerm(
            _params=params,
            _sob=np.empty(1),
            _sob_func=event_model.get_sob_energy,
            _sin_dec_idx=sin_dec_idx,
            _log_energy_idx=log_energy_idx,
        )

    def update_bounds(self, bounds: Bounds) -> Bounds:
        """Docstring"""
        return bounds

    def calculate_drop_mask(
        self,
        events: np.ndarray,
        source: sources.Source,
    ) -> np.ndarray:
        """Docstring"""
        return np.ones(len(events), dtype=bool)
