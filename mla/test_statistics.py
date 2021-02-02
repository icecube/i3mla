"""Docstring"""

__author__ = 'John Evans'
__copyright__ = 'Copyright 2020 John Evans'
__credits__ = ['John Evans', 'Jason Fan', 'Michael Larson']
__license__ = 'Apache License 2.0'
__version__ = '0.0.1'
__maintainer__ = 'John Evans'
__email__ = 'john.evans@icecube.wisc.edu'
__status__ = 'Development'

from typing import Callable, ClassVar, Dict, List, Optional, Sequence, Tuple

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
class PsPreprocess:
    """Docstring"""
    event_model: dataclasses.InitVar[models.EventModel]
    injector: dataclasses.InitVar[injectors.PsInjector]
    source: dataclasses.InitVar[sources.Source]
    events: dataclasses.InitVar[np.ndarray]

    _params: Tuple[float, float]
    _bounds: Bounds

    n_events: int = dataclasses.field(init=False)
    n_dropped: int = dataclasses.field(init=False)
    splines: List[scipy.interpolate.UnivariateSpline] = dataclasses.field(
        init=False)
    sob_spatial: np.array = dataclasses.field(init=False)
    drop_index: np.array = dataclasses.field(init=False)

    def __post_init__(self, event_model: models.EventModel,
                      injector: injectors.PsInjector, source: sources.Source,
                      events: np.ndarray) -> None:
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

        self.n_events = len(events)

        if self.n_events == 0:
            warnings.warn(''.join(['You are trying to preprocess zero events. ',
                                   'This will likely result in unexpected ',
                                   'behavior']), RuntimeWarning)

        self.sob_spatial = injector.signal_spatial_pdf(source, events)

        # Drop events with zero spatial or time llh
        # The contribution of those llh will be accounts in
        # n_dropped*np.log(1-n_signal/n_events)
        self.drop_index = self.sob_spatial != 0

        self.n_dropped = len(events) - np.sum(self.drop_index)
        self.sob_spatial = self.sob_spatial[self.drop_index]
        self.sob_spatial /= injector.background_spatial_pdf(
            events[self.drop_index], event_model)
        self.splines = event_model.get_log_sob_gamma_splines(
            events[self.drop_index])

    @property
    def params(self) -> Dict[str, float]:
        """Docstring"""
        return {'n_signal': min(self._params[0], self.n_events - 1e-5),
                'gamma': self._params[1]}

    @property
    def bounds(self) -> Bounds:
        """Docstring"""
        new_bounds = self._bounds
        new_bounds[0][1] = min(self._bounds[0][1], self.params['n_signal'])
        return new_bounds


@dataclasses.dataclass
class PsPreprocessFactory:
    """Docstring"""
    bounds: Bounds
    factory_type: ClassVar = PsPreprocess

    def __call__(self, event_model: models.EventModel,
                 injector: injectors.PsInjector, source: sources.Source,
                 events: np.ndarray, params: Tuple) -> PsPreprocess:
        """Docstring"""
        return self.factory_type(event_model, injector, source, events, params,
                                 **dataclasses.asdict(self))


TestStatistic = Callable[[np.ndarray, PsPreprocess], float]


def ps_test_statistic(params: np.ndarray, pre_pro: PsPreprocess) -> float:
    """Evaluates the test-statistic for the given events and parameters

    Calculates the test-statistic using a given event model, n_signal, and
    gamma. This function does not attempt to fit n_signal or gamma.

    Args:
        params: A two item array containing (n_signal, gamma).
        pre_pro:

    Returns:
        The overall test-statistic value for the given events and
        parameters.
    """

    sob_new = pre_pro.sob_spatial * np.exp(
        [spline(params[1]) for spline in pre_pro.splines])
    return -2 * np.sum(
        np.log((params[0] / pre_pro.n_events * (sob_new - 1)) + 1)
    ) + pre_pro.n_dropped * np.log(1 - params[0] / pre_pro.n_events)


@dataclasses.dataclass
class TdPsPreprocess(PsPreprocess):
    """Docstring"""
    sig_time_profile: time_profiles.GenericProfile
    bg_time_profile: time_profiles.GenericProfile

    sob_time: np.array = dataclasses.field(init=False)

    def __post_init__(self, event_model: models.EventModel,
                      injector: injectors.TimeDependentPsInjector,
                      source: sources.Source, events: np.ndarray) -> None:
        """Contains all of the calculations for the ts that can be done once.

        Separated from the main test-statisic functions to improve performance.

        Args:
            sig_time_profile:
            bg_time_profile:

        Raises:
            RuntimeWarning:
        """

        super().__post_init__(event_model, injector, source, events)

        self.sob_time = self.sig_time_profile.pdf(
            events[self.drop_index]['time'])
        self.sob_time /= self.bg_time_profile.pdf(
            events[self.drop_index]['time'])

        if np.logical_not(np.all(np.isfinite(self.sob_time))):
            warnings.warn('Warning, events outside background time profile',
                          RuntimeWarning)


@dataclasses.dataclass
class TdPsPreprocessFactory(PsPreprocessFactory):
    """Docstring"""
    sig_time_profile: time_profiles.GenericProfile
    bg_time_profile: time_profiles.GenericProfile

    factory_type: ClassVar = TdPsPreprocess


def td_ps_test_statistic(params: np.ndarray, pre_pro: TdPsPreprocess) -> float:
    """Evaluates the test-statistic for the given events and parameters

    Calculates the test-statistic using a given event model, n_signal, and
    gamma. This function does not attempt to fit n_signal or gamma.

    Args:
        params: A two item array containing (n_signal, gamma).
        pre_pro:

    Returns:
        The overall test-statistic value for the given events and
        parameters.
    """

    sob_new = pre_pro.sob_spatial * pre_pro.sob_time * np.exp(
        [spline(params[1]) for spline in pre_pro.splines])
    return -2 * np.sum(
        np.log((params[0] / pre_pro.n_events * (sob_new - 1)) + 1)
    ) + pre_pro.n_dropped * np.log(1 - params[0] / pre_pro.n_events)


@dataclasses.dataclass
class ThreeMLPsPreprocess(TdPsPreprocess):
    """Docstring"""
    sob_energy: np.array = dataclasses.field(init=False)

    def __post_init__(self, event_model: models.ThreeMLEventModel,
                      injector: injectors.TimeDependentPsInjector,
                      source: sources.Source, events: np.ndarray) -> None:
        """ThreeML version of TdPsPreprocess

        Args:

        """

        super().__post_init__(event_model, injector, source, events)

        self.sob_energy = event_model.get_energy_sob(events[self.drop_index])


@dataclasses.dataclass
class ThreeMLPsPreprocessFactory(TdPsPreprocessFactory):
    """Docstring"""
    factory_type: ClassVar = ThreeMLPsPreprocess


def threeml_ps_test_statistic(params: float,
                              pre_pro: ThreeMLPsPreprocess) -> float:
    """(ThreeML version) Evaluates the ts for the given events and parameters

    Calculates the test-statistic using a given event model, n_signal, and
    gamma. This function does not attempt to fit n_signal or gamma.

    Args:
        params: n_signal
        pre_pro:

    Returns:
        The overall test-statistic value for the given events and
        parameters.
    """

    sob_new = pre_pro.sob_spatial * pre_pro.sob_time * pre_pro.sob_energy
    return -2 * np.sum(
        np.log((params / pre_pro.n_events * (sob_new - 1)) + 1)
    ) + pre_pro.n_dropped * np.log(1 - params / pre_pro.n_events)
