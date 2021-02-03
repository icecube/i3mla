"""Docstring"""

__author__ = 'John Evans'
__copyright__ = 'Copyright 2020 John Evans'
__credits__ = ['John Evans', 'Jason Fan', 'Michael Larson']
__license__ = 'Apache License 2.0'
__version__ = '0.0.1'
__maintainer__ = 'John Evans'
__email__ = 'john.evans@icecube.wisc.edu'
__status__ = 'Development'

from typing import Callable, ClassVar, List, Optional, Sequence, Tuple

import warnings
import dataclasses
import numpy as np
import scipy.optimize

from . import sources
from . import models
from . import injectors
from . import time_profiles


Bounds = Sequence[Tuple[Optional[float], Optional[float]]]


@dataclasses.dataclass
class Preprocessing:
    """Docstring"""
    _params: Tuple[float, float]
    n_events: int
    n_dropped: int
    splines: List[scipy.interpolate.UnivariateSpline]
    sob_spatial: np.array
    drop_index: np.array

    _bounds: Bounds


@dataclasses.dataclass
class Preprocessor:
    """Docstring"""
    bounds: Bounds
    factory_type: ClassVar = Preprocessing

    def _preprocess(self, event_model: models.EventModel,
                    injector: injectors.PsInjector, source: sources.Source,
                    events: np.ndarray) -> Tuple:
        """Contains all of the calculations for the ts that can be done once.

        Separated from the main test-statisic functions to improve performance.

        Args:
            event_model: An object containing data and preprocessed parameters.
            injector:
            source:
            events:

        Raises:
            RuntimeWarning: There should be at least one event.
        """

        n_events = len(events)

        if n_events == 0:
            warnings.warn(''.join(['You are trying to preprocess zero events. ',
                                   'This will likely result in unexpected ',
                                   'behavior']), RuntimeWarning)

        sob_spatial = injector.signal_spatial_pdf(source, events)

        # Drop events with zero spatial or time llh
        # The contribution of those llh will be accounts in
        # n_dropped*np.log(1-n_signal/n_events)
        drop_index = sob_spatial != 0

        n_dropped = len(events) - np.sum(drop_index)
        sob_spatial = sob_spatial[drop_index]
        sob_spatial /= injector.background_spatial_pdf(
            events[drop_index], event_model)
        splines = event_model.get_log_sob_gamma_splines(
            events[drop_index])

        return n_events, n_dropped, splines, sob_spatial, drop_index

    def __call__(self, event_model: models.EventModel,
                 injector: injectors.PsInjector, source: sources.Source,
                 events: np.ndarray, params: Tuple) -> Preprocessing:
        """Docstring"""
        prepro = self._preprocess(event_model, injector, source, events)

        if len(prepro) >= 5:
            return self.factory_type(
                params,
                *prepro[:5],
                *dataclasses.astuple(self),
                *prepro[5:]
            )

        return self.factory_type(params, *prepro, *dataclasses.astuple(self))


TestStatistic = Callable[[np.ndarray, Preprocessing], float]


def ps_test_statistic(params: np.ndarray, prepro: Preprocessing) -> float:
    """Evaluates the test-statistic for the given events and parameters

    Calculates the test-statistic using a given event model, n_signal, and
    gamma. This function does not attempt to fit n_signal or gamma.

    Args:
        params: A two item array containing (n_signal, gamma).
        prepro:

    Returns:
        The overall test-statistic value for the given events and
        parameters.
    """

    sob_new = prepro.sob_spatial * np.exp(
        [spline(params[1]) for spline in prepro.splines])
    return -2 * np.sum(
        np.log((params[0] / prepro.n_events * (sob_new - 1)) + 1)
    ) + prepro.n_dropped * np.log(1 - params[0] / prepro.n_events)


@dataclasses.dataclass
class TdPreprocessing(Preprocessing):
    """Docstring"""
    sig_time_profile: time_profiles.GenericProfile
    bg_time_profile: time_profiles.GenericProfile
    sob_time: np.array


@dataclasses.dataclass
class TdPreprocessor(Preprocessor):
    """Docstring"""
    sig_time_profile: time_profiles.GenericProfile
    bg_time_profile: time_profiles.GenericProfile

    factory_type: ClassVar = TdPreprocessing

    def _preprocess(self, event_model: models.EventModel,
                    injector: injectors.TimeDependentPsInjector,
                    source: sources.Source, events: np.ndarray) -> Tuple:
        """Contains all of the calculations for the ts that can be done once.

        Separated from the main test-statisic functions to improve performance.

        Args:

        Raises:
            RuntimeWarning:
        """

        super_prepro = super()._preprocess(
            event_model, injector, source, events)
        # drop_index == super_prepro[4]
        sob_time = self.sig_time_profile.pdf(events[super_prepro[4]]['time'])
        sob_time /= self.bg_time_profile.pdf(events[super_prepro[4]]['time'])

        if np.logical_not(np.all(np.isfinite(sob_time))):
            warnings.warn('Warning, events outside background time profile',
                          RuntimeWarning)

        return (*super_prepro, sob_time)


def td_ps_test_statistic(params: np.ndarray, prepro: TdPreprocessing) -> float:
    """Evaluates the test-statistic for the given events and parameters

    Calculates the test-statistic using a given event model, n_signal, and
    gamma. This function does not attempt to fit n_signal or gamma.

    Args:
        params: A two item array containing (n_signal, gamma).
        prepro:

    Returns:
        The overall test-statistic value for the given events and
        parameters.
    """

    sob_new = prepro.sob_spatial * prepro.sob_time * np.exp(
        [spline(params[1]) for spline in prepro.splines])
    return -2 * np.sum(
        np.log((params[0] / prepro.n_events * (sob_new - 1)) + 1)
    ) + prepro.n_dropped * np.log(1 - params[0] / prepro.n_events)


@dataclasses.dataclass
class ThreeMLPreprocessing(TdPreprocessing):
    """Docstring"""
    sob_energy: np.array = dataclasses.field(init=False)


@dataclasses.dataclass
class ThreeMLPreprocessor(TdPreprocessor):
    """Docstring"""
    factory_type: ClassVar = ThreeMLPreprocessing

    def _preprocess(self, event_model: models.ThreeMLEventModel,
                    injector: injectors.TimeDependentPsInjector,
                    source: sources.Source, events: np.ndarray) -> Tuple:
        """ThreeML version of TdPreprocess

        Args:

        """
        super_prepro = super()._preprocess(
            event_model, injector, source, events)

        # drop_index == super_prepro[4]
        sob_energy = event_model.get_energy_sob(events[super_prepro[4]])
        return (*super_prepro, sob_energy)


def threeml_ps_test_statistic(params: float,
                              prepro: ThreeMLPreprocessing) -> float:
    """(ThreeML version) Evaluates the ts for the given events and parameters

    Calculates the test-statistic using a given event model, n_signal, and
    gamma. This function does not attempt to fit n_signal or gamma.

    Args:
        params: n_signal
        prepro:

    Returns:
        The overall test-statistic value for the given events and
        parameters.
    """

    sob_new = prepro.sob_spatial * prepro.sob_time * prepro.sob_energy
    return -2 * np.sum(
        np.log((params / prepro.n_events * (sob_new - 1)) + 1)
    ) + prepro.n_dropped * np.log(1 - params / prepro.n_events)
