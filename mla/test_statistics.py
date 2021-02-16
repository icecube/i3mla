"""Docstring"""

__author__ = 'John Evans'
__copyright__ = 'Copyright 2020 John Evans'
__credits__ = ['John Evans', 'Jason Fan', 'Michael Larson']
__license__ = 'Apache License 2.0'
__version__ = '0.0.1'
__maintainer__ = 'John Evans'
__email__ = 'john.evans@icecube.wisc.edu'
__status__ = 'Development'

from typing import Callable, ClassVar, List, Optional

import warnings
import dataclasses
import numpy as np
import numpy.lib.recfunctions as rf
import scipy.optimize

from . import sources
from . import models
from . import time_profiles


@dataclasses.dataclass
class Preprocessing:
    """Docstring"""
    params: np.ndarray
    events: np.ndarray
    n_events: int
    n_dropped: int
    sob_spatial: np.array
    drop_index: np.array


@dataclasses.dataclass
class Preprocessor:
    """Docstring"""
    factory_type: ClassVar = Preprocessing

    def _preprocess(self, event_model: models.EventModel,
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
            return 0, 0, [], np.array([]), np.array([])

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

    def __call__(self, event_model: models.EventModel,
                 source: sources.Source, events: np.ndarray,
                 params: np.ndarray) -> Preprocessing:
        """Docstring"""
        prepro = self._preprocess(event_model, source, events)
        basic_keys = {'drop_index', 'n_events', 'n_dropped', 'sob_spatial'}
        keys = [key for key in prepro if key not in basic_keys]

        if keys:
            return self.factory_type(
                params,
                events,
                prepro['n_events'],
                prepro['n_dropped'],
                prepro['sob_spatial'],
                prepro['drop_index'],
                *dataclasses.astuple(self),
                *[prepro[key] for key in keys],
            )

        # astuple returns a deepcopy of the instance attributes.
        return self.factory_type(
            params,
            prepro['n_events'],
            prepro['n_dropped'],
            prepro['sob_spatial'],
            prepro['drop_index'],
            *dataclasses.astuple(self),
        )


def _calculate_ns_ratio(sob: np.array, iterations: int = 5) -> float:
    """Docstring

    Args:
        sob:
        iterations:

    Returns:

    """
    k = 1 / (sob - 1)
    lo = -min(1, np.min(k))
    x = [0] * iterations

    for i in range(iterations - 1):
        # get next iteration and clamp
        x[i + 1] = min(1, max(
            lo,
            x[i] + np.sum(1 / (x[i] + k)) / np.sum(1 / (x[i] + k)**2),
        ))

    return x[-1]


def _i3_ts(sob: np.ndarray, prepro: Preprocessing,
           return_ns: bool, ns_ratio: Optional[float] = None) -> float:
    """Docstring"""
    if prepro.n_events == 0:
        return 0

    if ns_ratio is None:
        ns_ratio = _calculate_ns_ratio(sob)

    if return_ns:
        return ns_ratio * prepro.n_events

    return -2 * np.sum(
        np.log(ns_ratio * (sob - 1)) + 1
    ) + prepro.n_dropped * np.log(1 - ns_ratio)


TestStatistic = Callable[[np.ndarray, Preprocessing], float]


@dataclasses.dataclass
class _TdPreprocessing(Preprocessing):
    """Docstring"""
    sig_time_profile: time_profiles.GenericProfile
    bg_time_profile: time_profiles.GenericProfile
    sob_time: np.array


@dataclasses.dataclass
class _TdPreprocessor(Preprocessor):
    """Docstring"""
    sig_time_profile: time_profiles.GenericProfile
    bg_time_profile: time_profiles.GenericProfile

    factory_type: ClassVar = _TdPreprocessing

    def _preprocess(self, event_model: models.EventModel,
                    source: sources.Source, events: np.ndarray) -> dict:
        """Contains all of the calculations for the ts that can be done once.

        Separated from the main test-statisic functions to improve performance.

        Args:
            event_model: An object containing data and preprocessed parameters.
            source:
            events:
        """
        super_prepro_dict = super()._preprocess(event_model, source, events)

        sob_time = 1 / self.bg_time_profile.pdf(
            events[super_prepro_dict['drop_index']]['time'])

        if np.logical_not(np.all(np.isfinite(sob_time))):
            warnings.warn('Warning, events outside background time profile',
                          RuntimeWarning)
        return {**super_prepro_dict, 'sob_time': sob_time}


def _sob_time(params: np.ndarray, prepro: _TdPreprocessing) -> float:
    """Docstring"""
    time_params = prepro.sig_time_profile.param_dtype.names

    if set(time_params).issubset(set(params.dtype.names)):
        prepro.sig_time_profile.update_params(params[time_params])

    sob_time = prepro.sob_time * prepro.sig_time_profile.pdf(
        prepro.events[prepro.drop_index]['time'])

    return sob_time


@dataclasses.dataclass
class I3Preprocessing(_TdPreprocessing):
    """Docstring"""
    splines: List[scipy.interpolate.UnivariateSpline]


@dataclasses.dataclass
class I3Preprocessor(_TdPreprocessor):
    """Docstring"""
    factory_type: ClassVar = I3Preprocessing

    def _preprocess(self, event_model: models.EventModel,
                    source: sources.Source, events: np.ndarray) -> dict:
        """Contains all of the calculations for the ts that can be done once.

        Separated from the main test-statisic functions to improve performance.

        Args:
            event_model: An object containing data and preprocessed parameters.
            source:
            events:
        """
        super_prepro_dict = super()._preprocess(event_model, source, events)

        splines = event_model.log_sob_gamma_splines(
            events[super_prepro_dict['drop_index']])

        return {**super_prepro_dict, 'splines': splines}


def i3_test_statistic(params: np.ndarray,
                      prepro: I3Preprocessing,
                      return_ns: bool = False) -> float:
    """Evaluates the test-statistic for the given events and parameters

    Calculates the test-statistic using a given event model, n_signal, and
    gamma. This function does not attempt to fit n_signal or gamma.

    Args:
        params: An array containing (*time_params, gamma).
        prepro:
        return_ns:

    Returns:
        The overall test-statistic value for the given events and
        parameters.
    """
    temp_params = rf.unstructured_to_structured(
        params, dtype=prepro.params.dtype, copy=True)
    sob = prepro.sob_spatial * _sob_time(temp_params, prepro) * np.exp(
        [spline(temp_params['gamma']) for spline in prepro.splines])

    ns_ratio = None
    if 'ns' in temp_params.dtype.names:
        ns_ratio = temp_params['ns'] / prepro.n_events
    return _i3_ts(sob, prepro, return_ns, ns_ratio)


@dataclasses.dataclass
class ThreeMLPreprocessing(_TdPreprocessing):
    """Docstring"""
    event_model: models.ThreeMLEventModel


@dataclasses.dataclass
class ThreeMLPreprocessor(_TdPreprocessor):
    """Docstring"""
    factory_type: ClassVar = ThreeMLPreprocessing


def threeml_ps_test_statistic(params: np.ndarray,
                              prepro: ThreeMLPreprocessing,
                              return_ns: bool = False) -> float:

    """(ThreeML version) Evaluates the ts for the given events and parameters

    Calculates the test-statistic using a given event model, n_signal, and
    gamma. This function does not attempt to fit n_signal or gamma.

    Args:
        params:
        event_model:
        events:
        prepro:
        return_ns:

    Returns:
        The overall test-statistic value for the given events and
        parameters.
    """
    temp_params = rf.unstructured_to_structured(
        params, dtype=prepro.params.dtype, copy=True)

    sob_energy = prepro.event_model.get_energy_sob(
        prepro.events[prepro['drop_index']])

    sob = prepro.sob_spatial * _sob_time(params, prepro) * sob_energy

    sob = prepro.sob_spatial * _sob_time(
        temp_params, prepro) * sob_energy

    ns_ratio = None
    if 'ns' in temp_params.dtype.names:
        ns_ratio = temp_params['ns'] / prepro.n_events
    return _i3_ts(sob, prepro, return_ns, ns_ratio)
