"""Docstring"""

__author__ = 'John Evans'
__copyright__ = 'Copyright 2020 John Evans'
__credits__ = ['John Evans', 'Jason Fan', 'Michael Larson']
__license__ = 'Apache License 2.0'
__version__ = '0.0.1'
__maintainer__ = 'John Evans'
__email__ = 'john.evans@icecube.wisc.edu'
__status__ = 'Development'

from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import warnings
import dataclasses
import numpy as np
import scipy.optimize

from . import core
from . import models
from . import injectors
from . import spectral
from . import time_profiles


@dataclasses.dataclass
class PsPreprocess:
    """Docstring"""
    event_model: models.EventModel
    injector: injectors.PsInjector
    source: core.Source
    events: np.ndarray
    n_events: int = dataclasses.field(init=False)
    n_dropped: int = dataclasses.field(init=False)
    splines: List[scipy.interpolate.UnivariateSpline] = dataclasses.field(
        init=False)
    sob_spatial: np.array = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        """Contains all of the calculations for the ts that can be done once.

        Separated from the main test-statisic functions to improve performance.

        Args:
            event_model: An object containing data and preprocessed parameters.
            injector:
            source:
            events: An array of events to calculate the test-statistic for.
            n_events:
            n_dropped:
            splines:
            sob_spatial:

        Raises:
            ValueError: There must be at least one event.
        """

        self.n_events = len(self.events)

        if self.n_events == 0:
            return None

        self.sob_spatial = self.injector.signal_spatial_pdf(self.source,
                                                            self.events)

        # Drop events with zero spatial or time llh
        # The contribution of those llh will be accounts in
        # n_dropped*np.log(1-n_signal/n_events)
        drop_index = self.sob_spatial != 0

        self.n_dropped = len(self.events) - np.sum(drop_index)
        self.events = self.events[drop_index]
        self.sob_spatial = self.sob_spatial[drop_index]
        self.sob_spatial /= self.injector.background_spatial_pdf(
            self.events, self.event_model)
        self.splines = self.event_model.get_log_sob_gamma_splines(self.events)


TestStatistic = Callable[[np.ndarray, PsPreprocess], float]


def ps_test_statistic(params: np.ndarray, pp: PsPreprocess) -> float:
    """Evaluates the test-statistic for the given events and parameters

    Calculates the test-statistic using a given event model, n_signal, and
    gamma. This function does not attempt to fit n_signal or gamma.

    Args:
        params: A two item array containing (n_signal, gamma).
        pp:

    Returns:
        The overall test-statistic value for the given events and
        parameters.
    """

    sob_new = pp.sob_spatial * np.exp(
        [spline(params[1]) for spline in pp.splines])
    return -2 * np.sum(
        np.log((params[0] / pp.n_events * (sob_new - 1)) + 1)
    ) + pp.n_dropped * np.log(1 - params[0] / pp.n_events)


@dataclasses.dataclass
class TdPsPreprocess(PsPreprocess):
    """Docstring"""
    sig_time_profile: time_profiles.GenericProfile
    bg_time_profile: time_profiles.GenericProfile
    sob_time: np.array = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        """Contains all of the calculations for the ts that can be done once.

        Separated from the main test-statisic functions to improve performance.

        Args:
            sig_time_profile:
            bg_time_profile:

        Raises:
            RuntimeWarning:
        """

        super().__post_init__()

        self.sob_time = self.sig_time_profile.pdf(self.events['time'])
        self.sob_time /= self.bg_time_profile.pdf(self.events['time'])

        if np.logical_not(np.all(np.isfinite(self.sob_time))):
            warnings.warn("Warning, events outside background time profile",
                          RuntimeWarning)


def td_ps_test_statistic(params: np.ndarray, pp: TdPsPreprocess) -> float:
    """Evaluates the test-statistic for the given events and parameters

    Calculates the test-statistic using a given event model, n_signal, and
    gamma. This function does not attempt to fit n_signal or gamma.

    Args:
        params: A two item array containing (n_signal, gamma).
        pp:

    Returns:
        The overall test-statistic value for the given events and
        parameters.
    """

    sob_new = pp.sob_spatial * pp.sob_time * np.exp(
        [spline(params[1]) for spline in pp.splines])
    return -2 * np.sum(
        np.log((params[0] / pp.n_events * (sob_new - 1)) + 1)
    ) + pp.n_dropped * np.log(1 - params[0] / pp.n_events)



@dataclasses.dataclass
class ThreeMLPsPreprocess(PsPreprocess):
    """Docstring"""
    sob_energy: np.array = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        """ThreeML version of TdPsPreprocess

        Args:

        """

        super().__post_init__()

        self.sob_energy = self.event_model.get_energy_sob(self.events)


def ThreeML_ps_test_statistic(params: float, pp: ThreeMLPsPreprocess) -> float:
    """(ThreeML version)Evaluates the test-statistic for the given events and parameters

    Calculates the test-statistic using a given event model, n_signal, and
    gamma. This function does not attempt to fit n_signal or gamma.

    Args:
        params: n_signal
        pp:

    Returns:
        The overall test-statistic value for the given events and
        parameters.
    """

    sob_new = pp.sob_spatial * pp.sob_time * pp.sob_energy
    return -2 * np.sum(
        np.log((params / pp.n_events * (sob_new - 1)) + 1)
    ) + pp.n_dropped * np.log(1 - params / pp.n_events)
