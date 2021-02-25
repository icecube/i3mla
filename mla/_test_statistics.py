"""Docstring"""

__author__ = 'John Evans'
__copyright__ = 'Copyright 2020 John Evans'
__credits__ = ['John Evans', 'Jason Fan', 'Michael Larson']
__license__ = 'Apache License 2.0'
__version__ = '0.0.1'
__maintainer__ = 'John Evans'
__email__ = 'john.evans@icecube.wisc.edu'
__status__ = 'Development'

from typing import ClassVar, List, Optional, Sequence, Tuple

import warnings
import dataclasses
import numpy as np

from . import sources
from . import _models
from . import time_profiles


Bounds = Optional[Sequence[Tuple[float, float]]]


@dataclasses.dataclass
class Preprocessing:
    """Docstring"""
    params: np.ndarray
    _bounds: Bounds
    events: np.ndarray
    n_events: Optional[int] = None
    n_dropped: Optional[int] = None
    sob_spatial: Optional[np.array] = None
    drop_index: Optional[np.array] = None

    def _fix_bounds(
        self,
        bnds: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """Docstring"""
        if 'ns' in self.params.dtype.names:
            i = self.params.dtype.names.index('ns')
            bnds[i] = (
                max(bnds[i][0], -self.n_events + self.n_dropped),
                min(bnds[i][1], self.n_events - self.n_dropped),
            )
        return bnds

    @property
    def bounds(self) -> Bounds:
        """Docstring"""
        if self._bounds is None:
            self._bounds = [(-np.inf, np.inf)] * len(self.params.dtype.names)
        return self._fix_bounds(self._bounds)


@dataclasses.dataclass
class Preprocessor:
    """Docstring"""
    factory_type: ClassVar = Preprocessing

    def _preprocess(self, event_model: _models.EventModel,
                    source: sources.Source, events: np.ndarray) -> dict:
        """Contains all of the calculations for the ts that can be done once.

        Separated from the main test-statisic functions to improve performance.

        Args:
            event_model: An object containing data and preprocessed parameters.
            source:
            events:
        """

        n_events = len(events)

        if n_events == 0:
            return {
                'drop_index': np.array([]),
                'n_events': 0,
                'n_dropped': 0,
                'sob_spatial': np.array([]),
            }

        sob_spatial = event_model.signal_spatial_pdf(source, events)

        # Drop events with zero spatial or time llh
        # The contribution of those llh will be accounts in
        # n_dropped*np.log(1-n_signal/n_events)
        drop_index = sob_spatial != 0

        n_dropped = len(events) - np.sum(drop_index)
        sob_spatial = sob_spatial[drop_index]
        sob_spatial /= event_model.background_spatial_pdf(events[drop_index])

        return {'drop_index': drop_index, 'n_events': n_events,
                'n_dropped': n_dropped, 'sob_spatial': sob_spatial}

    def __call__(self, event_model: _models.EventModel,
                 source: sources.Source, events: np.ndarray,
                 params: np.ndarray, bounds: Bounds) -> Preprocessing:
        """Docstring"""
        prepro_dict = self._preprocess(event_model, source, events)
        basic_keys = {'drop_index', 'n_events', 'n_dropped', 'sob_spatial'}
        keys = [key for key in prepro_dict if key not in basic_keys]

        # astuple returns a deepcopy of the instance attributes.
        return self.factory_type(
            params,
            bounds,
            events,
            **prepro_dict,
        )


@dataclasses.dataclass
class TdPreprocessing(Preprocessing):
    """Docstring"""
    sig_time_profile: Optional[time_profiles.GenericProfile] = None
    bg_time_profile: Optional[time_profiles.GenericProfile] = None
    sob_time: Optional[np.array] = None
    times: Optional[np.array] = None


@dataclasses.dataclass
class TdPreprocessor(Preprocessor):
    """Docstring"""
    sig_time_profile: time_profiles.GenericProfile
    bg_time_profile: time_profiles.GenericProfile

    factory_type: ClassVar = TdPreprocessing

    def _preprocess(self, event_model: _models.EventModel,
                    source: sources.Source, events: np.ndarray) -> dict:
        """Contains all of the calculations for the ts that can be done once.

        Separated from the main test-statisic functions to improve performance.

        Args:
            event_model: An object containing data and preprocessed parameters.
            source:
            events:
        """
        super_prepro_dict = super()._preprocess(event_model, source, events)
        times = np.empty(
            super_prepro_dict['drop_index'].sum(),
            dtype=events['time'].dtype,
        )
        times = events[super_prepro_dict['drop_index']]['time']
        sob_time = 1 / self.bg_time_profile.pdf(times)

        if np.logical_not(np.all(np.isfinite(sob_time))):
            warnings.warn('Warning, events outside background time profile',
                          RuntimeWarning)
        return {
            **super_prepro_dict,
            'sig_time_profile': self.sig_time_profile,
            'bg_time_profile': self.bg_time_profile,
            'sob_time': sob_time,
            'times': times,
        }


def get_i3_llh(sob: np.ndarray, ns_ratio: float) -> np.array:
    """Docstring"""
    return np.log(ns_ratio * (sob - 1) + 1), np.log(1 - ns_ratio)


def get_ns_ratio(sob: np.array, prepro: Preprocessing, params: np.ndarray,
                 iterations: int) -> float:
    """Docstring

    Args:
        sob:
        iterations:

    Returns:

    """
    if 'ns' in params.dtype.names:
        return params['ns'] / prepro.n_events

    eps = 1e-5
    k = 1 / (sob - 1)
    lo = max(-1 + eps, 1 / (1 - np.max(sob)))
    x = [0] * iterations

    for i in range(iterations - 1):
        # get next iteration and clamp
        terms = 1 / (x[i] + k)
        zero_term = 1 / (x[i] - 1)
        first_derivative = np.sum(terms) + prepro.n_dropped * zero_term
        second_derivative = np.sum(
            terms**2) + prepro.n_dropped * zero_term**2
        x[i + 1] = min(1 - eps, max(
            lo,
            x[i] + first_derivative / second_derivative,
        ))

    return x[-1]


def get_sob_time(params: np.ndarray, prepro: TdPreprocessing) -> np.array:
    """Docstring"""
    time_params = prepro.sig_time_profile.param_dtype.names

    if set(time_params).issubset(set(params.dtype.names)):
        prepro.sig_time_profile.update_params(params[list(time_params)])

    sob_time = prepro.sob_time * prepro.sig_time_profile.pdf(prepro.times)

    return sob_time