"""Docstring"""

__author__ = 'John Evans'
__copyright__ = 'Copyright 2020 John Evans'
__credits__ = ['John Evans', 'Jason Fan', 'Michael Larson']
__license__ = 'Apache License 2.0'
__version__ = '0.0.1'
__maintainer__ = 'John Evans'
__email__ = 'john.evans@icecube.wisc.edu'
__status__ = 'Development'

from typing import Callable, ClassVar, List

import warnings
import dataclasses
import numpy as np
import scipy.optimize

from . import sources
from . import models
from . import time_profiles


@dataclasses.dataclass
class Preprocessing:
    """Docstring"""
    params: np.ndarray
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
        sob_spatial /= event_model.background_spatial_pdf(
            events[drop_index], event_model)

        return {'drop_index': drop_index, 'n_events: ': n_events,
                'n_dropped': n_dropped, 'sob_spatial': sob_spatial}

    def __call__(self, event_model: models.EventModel,
                 source: sources.Source, events: np.ndarray,
                 params: np.ndarray) -> Preprocessing:
        """Docstring"""
        prepro = self._preprocess(event_model, source, events)
        basic_keys = {'drop_index', 'n_events', 'n_dropped', 'sob_spatial'}
        keys = [key for key, _ in prepro if key not in basic_keys]

        if keys:
            return self.factory_type(
                params,
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


def _calculate_ns_ratio(sob: np.array, iterations: int = 3) -> float:
    """Docstring

    Args:
        sob:
        iterations:

    Returns:

    """
    k = 1 / (1 - sob)
    x = [0]

    for _ in range(iterations):
        x.append(x[-1] + np.sum(1 / (x[-1] + k)) / np.sum(1 / (x[-1] + k)**2))

    return x[-1]


def _i3_ts(sob: np.ndarray, prepro: Preprocessing,
           ns: Optional[float] = None, return_ns: bool) -> float:
    """Docstring"""
    if prepro.n_events == 0:
        return 0

    if ns is not None:
        ns_ratio = ns / prepro.n_events
    else:
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
    time_params = [name for name, _ in prepro.sig_time_profile.param_dtype]
    param_names = [name for name, _ in params.dtype]

    if set(time_params).issubset(set(param_names)):
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

        splines = event_model.get_log_sob_gamma_splines(
            events[super_prepro_dict['drop_index']])

        return {**super_prepro_dict, 'splines': splines}


def i3_test_statistic(params: np.ndarray, prepro: I3Preprocessing,
                      ns: Optional[float] = None, return_ns: bool = False) -> float:
                      ) -> float:
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

    sob = prepro.sob_spatial * _sob_time(params, prepro) * np.exp(
        [spline(params['gamma']) for spline in prepro.splines])
    return _i3_ts(sob, prepro, ns, return_ns)


@dataclasses.dataclass
class ThreeMLPreprocessing(_TdPreprocessing):
    """Docstring"""
    sob_energy: np.array = dataclasses.field(init=False)


@dataclasses.dataclass
class ThreeMLPreprocessor(_TdPreprocessor):
    """Docstring"""
    factory_type: ClassVar = ThreeMLPreprocessing

    def _preprocess(self, event_model: models.ThreeMLEventModel,
                    source: sources.Source, events: np.ndarray) -> dict:
        """ThreeML version of TdPreprocess

        Args:

        """
        super_prepro_dict = super()._preprocess(event_model, source, events)

        sob_energy = event_model.get_energy_sob(
            events[super_prepro_dict['drop_index']])

        return {**super_prepro_dict, 'sob_energy': sob_energy}


def threeml_ps_test_statistic(params: np.ndarray,
                              prepro: ThreeMLPreprocessing,
                              ns: Optional[float] = None,
                              return_ns: bool = False) -> float:

    """(ThreeML version) Evaluates the ts for the given events and parameters

    Calculates the test-statistic using a given event model, n_signal, and
    gamma. This function does not attempt to fit n_signal or gamma.

    Args:
        params:
        prepro:
        ns: fixing the ns
        return_ns:

    Returns:
        The overall test-statistic value for the given events and
        parameters.
    """
    sob = prepro.sob_spatial * _sob_time(params, prepro) * prepro.sob_energy
    return _i3_ts(sob, prepro, ns, return_ns)
