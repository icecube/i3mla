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

from . import utility_functions as uf

import abc
import copy
import dataclasses
import warnings
import numpy as np

if TYPE_CHECKING:
    from .sources import Source
    from .background_models import BaseBackgroundModel
    from .energy_models import BaseEnergyModel
    from .time_profiles import GenericProfile
else:
    Source = object  # pylint: disable=invalid-name
    BaseBackgroundModel = object  # pylint: disable=invalid-name
    BaseEnergyModel = object  # pylint: disable=invalid-name
    GenericProfile = object  # pylint: disable=invalid-name


Bounds = Optional[Sequence[Tuple[float, float]]]


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
    def __call__(self, params: np.ndarray, bounds: Bounds, events: np.ndarray) -> SoBTerm:
        """Docstring"""

    @abc.abstractmethod
    def update_bounds(self, bounds: Bounds) -> Bounds:
        """Docstring"""

    @abc.abstractmethod
    def calculate_drop_mask(self, events: np.ndarray) -> np.ndarray:
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
    background_model: BaseBackgroundModel
    source: Source

    def __call__(self, params: np.ndarray, bounds: Bounds, events: np.ndarray) -> SoBTerm:
        """Docstring"""
        sob_spatial = self.source.pdf(events)
        sob_spatial /= self.background_model.pdf(events)
        return SpatialTerm(_params=params, _sob=sob_spatial)

    def update_bounds(self, bounds: Bounds) -> Bounds:
        """Docstring"""
        return bounds

    def calculate_drop_mask(self, events: np.ndarray) -> np.ndarray:
        """Docstring"""
        return self.source.pdf(events) != 0


@dataclasses.dataclass
class TimeTerm(SoBTerm):
    """Docstring"""
    _times: np.ndarray
    _signal_time_profile: GenericProfile

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
    background_time_profile: GenericProfile
    signal_time_profile: GenericProfile

    def __call__(self, params: np.ndarray, bounds: Bounds, events: np.ndarray) -> SoBTerm:
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

    def update_bounds(self, bounds: Bounds) -> Bounds:
        """Docstring"""
        return bounds

    def calculate_drop_mask(self, events: np.ndarray) -> np.ndarray:
        """Docstring"""
        return 1 / self.background_time_profile.pdf(events['time']) != 0


@dataclasses.dataclass
class EnergyTerm(SoBTerm):
    """Docstring"""
    _energy_model: BaseEnergyModel
    _event_energy_map: dict

    @params.setter
    def params(self, params: np.ndarray) -> None:
        """Docstring"""
        self._energy_model.params = params

    @property
    def sob(self) -> np.ndarray:
        """Docstring"""
        return self._energy_model.evaluate_sob(**_event_energy_map)


@dataclasses.dataclass
class EnergyTermFactory(SoBTermFactory):
    """Docstring"""
    energy_model: BaseEnergyModel

    def __call__(self, params: np.ndarray, bounds: Bounds, events: np.ndarray) -> SoBTerm:
        """Docstring"""
        event_energy_map = energy_model.build_event_map(events)
        return EnergyTerm(
            _params=params,
            _sob=np.empty(1),
            _energy_model=self.energy_model,
            _event_energy_map=event_energy_map,
        )

    def update_bounds(self, bounds: Bounds) -> Bounds:
        """Docstring"""
        return bounds

    def calculate_drop_mask(self, events: np.ndarray) -> np.ndarray:
        """Docstring"""
        return np.ones(len(events), dtype=bool)
