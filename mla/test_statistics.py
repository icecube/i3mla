"""Docstring"""

__author__ = 'John Evans'
__copyright__ = 'Copyright 2021 John Evans'
__credits__ = ['John Evans', 'Jason Fan', 'Michael Larson']
__license__ = 'Apache License 2.0'
__version__ = '0.0.1'
__maintainer__ = 'John Evans'
__email__ = 'john.evans@icecube.wisc.edu'
__status__ = 'Development'

from typing import ClassVar, Dict, List, Optional, Tuple

import dataclasses
import math
import numpy as np
from numba import njit

from .configurable import Configurable
from .sob_terms import SoBTerm, SoBTermFactory
from .events import Events
from .params import Params
from .data_handlers import Injector, TimeDependentNuSourcesInjector


@njit(parallel=True, fastmath=True)
def _newton_ns_ratio(
    sob: np.ndarray,
    precision: float,
    iterations: int,
    n_dropped: int,
) -> float:
    """Docstring

    Args:
        sob:

    Returns:

    """
    precision += 1
    eps = 1e-5
    k = 1 / (sob - 1)
    x = [0.] * iterations

    for i in range(iterations - 1):
        # get next iteration and clamp
        inv_terms = x[i] + k
        inv_terms[inv_terms == 0] = eps
        terms = 1 / inv_terms
        drop_term = 1 / (x[i] - 1)
        d1 = np.sum(terms) + n_dropped * drop_term
        d2 = np.sum(terms**2) + n_dropped * drop_term**2
        x[i + 1] = min(1 - eps, max(0, x[i] + d1 / d2))

        if x[i] == x[i + 1] or (
            x[i] < x[i + 1] and x[i + 1] <= x[i] * precision
        ) or (x[i + 1] < x[i] and x[i] <= x[i + 1] * precision):
            break
    return x[i + 1]


@njit(parallel=True, fastmath=True)
def _calculate_llh(ns_ratio: float, sob: np.ndarray) -> np.ndarray:
    return np.sum(np.sign(ns_ratio) * np.log(np.abs(ns_ratio) * (sob - 1) + 1))


@njit(fastmath=True)
def _calculate_dropterm(ns_ratio: float, n_dropped: int) -> float:
    return n_dropped * np.sign(ns_ratio) * np.log(1 - np.abs(ns_ratio))


@dataclasses.dataclass(kw_only=True)
class LLHTestStatistic():
    """Docstring"""
    sob_terms: Dict[str, SoBTerm]
    _n_events: int
    _n_kept: int
    _events: Events
    _params: Params
    _newton_precision: float
    _newton_iterations: int
    _best_ts: float = dataclasses.field(init=False, default=0)
    _best_ns: float = dataclasses.field(init=False, default=0)

    def __call__(
        self,
        param_values: Optional[np.ndarray] = None,
        fitting_ns: bool = False,
    ) -> float:
        """Evaluates the test-statistic for the given events and parameters

        Calculates the test-statistic using a given event model, n_signal, and
        gamma. This function does not attempt to fit n_signal or gamma.

        Returns:
            The overall test-statistic value for the given events and
            parameters.
        """
        if param_values is not None:
            self.params.value_array = param_values
            self._update_term_params()

        if self._n_events == 0:
            return 0

        sob = self._calculate_sob()

        if fitting_ns:
            ns_ratio = self._params['ns'] / self._n_events
        else:
            ns_ratio = None

        if ns_ratio == 0:
            ts = 0
        else:
            ns_ratio, ts = self._calculate_ts(ns_ratio, sob)

        if ts < self._best_ts:
            self._best_ts = ts
            self._best_ns = ns_ratio * self._n_events

        return ts

    def _calculate_ts(self, ns_ratio: float, sob: np.ndarray) -> Tuple[float, float]:
        """Docstring"""
        if ns_ratio is None:
            ns_ratio = _newton_ns_ratio(
                sob, self._newton_precision, self._newton_iterations, self.n_dropped)

        return (
            ns_ratio,
            -2 * (_calculate_dropterm(
                    ns_ratio, self.n_dropped) + _calculate_llh(ns_ratio, sob)),
        )

    def _calculate_sob(self) -> np.ndarray:
        """Docstring"""
        sob = np.ones(self.n_kept)
        for name, term in self.sob_terms.items():
            sob *= term.sob.reshape((-1,))
        return sob

    @property
    def params(self) -> Params:
        """Docstring"""
        return self._params

    @params.setter
    def params(self, params: Params) -> None:
        """Docstring"""
        if params == self._params:
            return
        if 'ns' in params:
            params.bounds['ns'] = (0, min(params.bounds['ns'], self.n_kept))
        self._params = params
        self._update_term_params()
        self._best_ns = self._best_ts = 0

    def _update_term_params(self) -> None:
        """Docstring"""
        for _, term in self.sob_terms.items():
            term.params = self.params

    @property
    def best_ns(self) -> float:
        """Docstring"""
        return self._best_ns

    @property
    def best_ts(self) -> float:
        """Docstring"""
        return self._best_ts

    @property
    def n_events(self) -> int:
        """Docstring"""
        return self._n_events

    @property
    def n_kept(self) -> int:
        """Docstring"""
        return self._n_kept

    @property
    def n_dropped(self) -> int:
        """Docstring"""
        return self._n_events - self._n_kept


@dataclasses.dataclass(kw_only=True)
class LLHTestStatisticFactory(Configurable):
    """
        newton_precision: Newton Method n_s Precision
        newton_iterations: Newton Method n_s Iterations
    """
    factory_of: ClassVar = LLHTestStatistic
    sob_term_factories: List[SoBTermFactory]

    newton_precision: float = 0
    newton_iterations: int = 20

    def __call__(self, params: Params, events: Events) -> LLHTestStatistic:
        """Docstring"""
        drop_mask = np.logical_and.reduce(np.array([
            term_factory.calculate_drop_mask(events)
            for term_factory in self.sob_term_factories
        ]))

        n_kept = drop_mask.sum()
        pruned_events = events.from_idx(np.nonzero(drop_mask))

        sob_terms = {
            term_factory.name: term_factory(params, pruned_events)
            for term_factory in self.sob_term_factories
        }

        return self.__class__.factory_of(
            sob_terms=sob_terms,
            _n_events=len(events),
            _n_kept=n_kept,
            _events=pruned_events,
            _params=params,
            **self._factory_kwargs(),
        )

    def _factory_kwargs(self) -> dict:
        """Docstring"""
        return {
            '_newton_iterations': self.newton_iterations,
            '_newton_precision': self.newton_precision,
        }

    def generate_params(self) -> Params:
        """Docstring"""
        param_values = {'ns': 0}
        param_bounds = {'ns': (0, np.inf)}

        for term in self.sob_term_factories:
            vals, bounds = term.generate_params()
            param_values = dict(param_values, **vals)
            param_bounds = dict(param_bounds, **bounds)

        return Params.from_dict(param_values, param_bounds)


@dataclasses.dataclass(kw_only=True)
class FlareStackLLHTestStatistic(LLHTestStatistic):
    """Docstring"""
    _min_sob: float
    _time_term_name: str
    _window_start: float
    _window_length: float
    _injector: TimeDependentNuSourcesInjector

    _best_ts_dict: dict[tuple[float, float], dict] = dataclasses.field(
        init=False, default_factory=dict)
    _best_time_params: dict[str, float] = dataclasses.field(
        init=False, default_factory=dict)

    def _calculate_sob(self) -> np.ndarray:
        sob = np.ones(self.n_kept)
        for name, term in self.sob_terms.items():
            if name == self._time_term_name:
                continue
            sob *= term.sob.reshape((-1,))
        return sob

    def _calculate_ts(
            self, ns_ratio: Optional[float], sob: np.ndarray) -> Tuple[float, float]:
        """Docstring"""
        ts_dict = {}
        term_dict = {}
        flares = []

        time_params = self.sob_terms[self._time_term_name].params
        if 'start' not in time_params or 'length' not in time_params:
            raise TypeError('Only mla.UniformProfile is currently supported')

        edges = self._events.time[sob >= self._min_sob]

        if len(edges) == 0:
            return 0, 0

        for i, start in enumerate(edges[:-1]):
            for end in edges[i + 1:]:
                flares.append((start, end - start))

        if len(flares) == 0:
            return 0, 0

        log_bg_livetime = math.log(self._injector.contained_livetime(
            self._window_start, self._window_start + self._window_length))

        signal_livetimes = np.empty(len(flares))
        for i, (start, length) in enumerate(flares):
            signal_livetimes[i] = self._injector.contained_livetime(start, start + length)
        time_corrections = np.log(signal_livetimes) - log_bg_livetime

        for i, (start, length) in enumerate(flares):
            time_params['start'], time_params['length'] = start, length
            self.sob_terms[self._time_term_name].params = time_params
            combined_sob = sob * self.sob_terms[self._time_term_name].sob.reshape((-1,))

            if ns_ratio is None:
                flare_ns_ratio = _newton_ns_ratio(
                    combined_sob,
                    self._newton_precision,
                    self._newton_iterations,
                    self.n_dropped,
                )
            else:
                flare_ns_ratio = ns_ratio

            drop_term = _calculate_dropterm(flare_ns_ratio, self.n_dropped)
            llh = _calculate_llh(flare_ns_ratio, combined_sob)
            ts_dict[(start, length)] = -2 * (llh + drop_term + time_corrections[i])

            term_dict[(start, length)] = {
                'ts': ts_dict[(start, length)],
                'llh': llh,
                'drop term': drop_term,
                'time_correction': time_corrections[i],
                'ns_ratio': flare_ns_ratio,
            }

        ts_pair = min(ts_dict.items(), key=lambda x: x[1])
        if ts_pair[1] < self._best_ts:
            self._best_ts_dict = term_dict
            self._best_time_params['start'], self._best_time_params['length'] = ts_pair[0]
        return term_dict[ts_pair[0]]['ns_ratio'], ts_pair[1]

    @property
    def best_ts_dict(self) -> dict:
        return self._best_ts_dict

    @property
    def best_time_params(self) -> dict:
        return self._best_time_params

@dataclasses.dataclass(kw_only=True)
class FlareStackLLHTestStatisticFactory(LLHTestStatisticFactory, Configurable):
    """
        min_sob: Minimum Signal-over-background Ratio For Flare
        time_term_name: Time Term Name
        window_start: Full Time Window Start (MJD)
        window_length: Full Time Window Length (days)
    """
    factory_of: ClassVar = FlareStackLLHTestStatistic
    injector: Injector

    min_sob: float = 1
    time_term_name: str = 'TimeTerm'
    window_start: float = 1
    window_length: float = 1

    def _factory_kwargs(self) -> dict:
        """Docstring"""
        return {
            **super()._factory_kwargs(),
            '_min_sob': self.min_sob,
            '_time_term_name': self.time_term_name,
            '_window_start': self.window_start,
            '_window_length': self.window_length,
            '_injector': self.injector,
        }
