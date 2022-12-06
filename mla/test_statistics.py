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


class MinimizingTestStatistic():
    """Docstring"""
    def minimize(self, fitting_params: Optional[List[str]] = None) -> dict:
        """Docstring"""
        raise NotImplementedError


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
class NonMinimizingLLHTestStatistic():
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

    def evaluate(self) -> float:
        """Docstring"""
        return self._calculate_ts_wrapper()

    def _calculate_ts_wrapper(
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
class NonMinimizingLLHTestStatisticFactory(Configurable):
    """
        newton_precision: Newton Method n_s Precision
        newton_iterations: Newton Method n_s Iterations
    """
    factory_of: ClassVar = NonMinimizingLLHTestStatistic
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
class LLHTestStatistic(NonMinimizingLLHTestStatistic, MinimizingTestStatistic):
    """Docstring"""
    _gridsearch_size: int
    _minimization_algorithm: str

    def minimize(self, fitting_params: Optional[List[str]] = None) -> dict:
        if fitting_params is None:
            fitting_params = list(self.params.key_idx_map)

        fitting_key_idx_map = {
            key: val for key, val in self.params.key_idx_map.items()
            if key in fitting_params
        }

        fitting_bounds = {
            key: val for key, val in self.params.bounds.items()
            if key in fitting_params
        }

        if self.n_kept == 0:
            return 0, np.array([(0,)], dtype=[('ns', np.float64)])

        grid = [
            np.linspace(lo, hi, self._gridsearch_size)
            for lo, hi in fitting_bounds.values()
        ]

        points = np.array(np.meshgrid(*grid)).T

        grid_ts_values = np.array([
            self._minimize_ts_wrapper(point, fitting_key_idx_map)
            for point in points
        ])

        return self._minimizer_wrapper(
            points[grid_ts_values.argmin()], fitting_key_idx_map, fitting_bounds)

    def _param_values(self, point: np.ndarray, fitting_key_idx_map: dict) -> np.ndarray:
        """Docstring"""
        param_values = self.params.value_array.copy()

        for i, j in enumerate(fitting_key_idx_map.values()):
            param_values[j] = point[i]

        return param_values

    def _minimize_ts_wrapper(self, point: np.ndarray, fitting_key_idx_map: dict) -> float:
        """Docstring"""
        return self._calculate_ts_wrapper(
            self._param_values(point, fitting_key_idx_map),
            fitting_ns='ns' in fitting_key_idx_map,
        )

    def _minimizer_wrapper(
        self,
        point: np.ndarray
        fitting_key_idx_map: dict,
        fitting_bounds: dict,
        fitting_ns: bool,
    ) -> dict:
        """Docstring"""
        result = scipy.optimize.minimize(
            self._minimize_ts_wrapper,
            x0=point,
            args=(fitting_key_idx_map,),
            bounds=fitting_bounds.values(),
            method=self.min_method,
        )

        best_ts_value = result.fun
        best_param_values = self._param_values(result.x, fitting_key_idx_map)

        if 'ns' not in fitting_key_idx_map:
            idx = self.params.key_idx_map['ns']
            best_param_values[idx] = self.best_ns

        return {
            'ts': best_ts_value,
            **{
                key: best_param_values[idx]
                for key, idx in self.params.key_idx_map.items()
            },
        }


@dataclasses.dataclass(kw_only=True)
class LLHTestStatisticFactory(NonMinimizingLLHTestStatisticFactory):
    """Docstring"""
    factory_of: ClassVar = LLHTestStatistic
    gridsearch_size: int = 5
    minimization_algorithm: str = 'L-BFGS-B'

    def _factory_kwargs(self) -> dict:
        """Docstring"""
        return {
            **super()._factory_kwargs(),
            '_gridsearch_size': self.gridsearch_size,
            '_minimization_algorithm': self.minimization_algorithm,
        }


@dataclasses.dataclass(kw_only=True)
class FlareStackLLHTestStatistic(LLHTestStatistic):
    """Docstring"""
    _min_sob: float
    _time_term_name: str
    _window_start: float
    _window_length: float
    _injector: TimeDependentNuSourcesInjector
    _return_perflare_ts: bool

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

    def _minimizer_wrapper(
        self,
        point: np.ndarray,
        fitting_key_idx_map: dict,
        fitting_bounds: dict,
    ) -> dict:
        best_dict = super()._minimizer_wrapper(point, fitting_key_idx_map, fitting_bounds)
        for key, val in self.best_time_params.items():
            best_dict[key] = val
        if self._return_perflare_ts:
            best_dict['perflare_ts'] = self.best_ts_dict
        return best_dict

    @property
    def best_ts_dict(self) -> dict:
        return self._best_ts_dict

    @property
    def best_time_params(self) -> dict:
        return self._best_time_params


@dataclasses.dataclass(kw_only=True)
class FlareStackLLHTestStatisticFactory(LLHTestStatisticFactory):
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
    return_perflare_ts: bool = False

    def _factory_kwargs(self) -> dict:
        """Docstring"""
        return {
            **super()._factory_kwargs(),
            '_min_sob': self.min_sob,
            '_time_term_name': self.time_term_name,
            '_window_start': self.window_start,
            '_window_length': self.window_length,
            '_injector': self.injector,
            '_return_perflare_ts': self.return_perflare_ts,
        }


@njit
def _expectation_maximization(
    sob_se: np.ndarray,
    times: np.ndarray,
    sob_t_bg: np.ndarray,
    livetime: float,
    n_dropped: int,
    n_events: int,
    mu: float,
    sigma: float,
    ns_ratio: float,
    iterations: int,
    precision: float,
) -> Tuple[float, float, float, float]:
    """Docstring"""
    precision += 1
    one_over_sigma = 1 / sigma
    bos_se = 1 / sob_se
    resp_mat, llh = _expectation_llh(
        sob_se,
        bos_se,
        times,
        sob_t_bg,
        livetime,
        mu,
        one_over_sigma,
        ns_ratio,
        n_dropped,
        n_events,
    )

    for _ in range(iterations):
        old_llh = llh
        mu, one_over_sigma, ns_ratio = _calculate_maximization(resp_mat, times, n_events)
        resp_mat, llh = _expectation_llh(
            sob_se,
            bos_se,
            times,
            sob_t_bg,
            livetime,
            mu,
            one_over_sigma,
            ns_ratio,
            n_dropped,
            n_events,
        )

        if (old_llh == llh) or (
            llh < old_llh and old_llh <= llh * precision
        ) or (
            old_llh < llh and llh <= old_llh * precision
        ):
            break

    return llh, mu, 1 / one_over_sigma, ns_ratio


ONE_OVER_ROOT_TWO_PI = 1 / np.sqrt(2 * np.pi)


@njit
def _expectation_llh(
    sob_se: np.ndarray,
    bos_se: np.ndarray,
    times: np.ndarray,
    sob_t_bg: np.ndarray,
    livetime: float,
    mu: float,
    one_over_sigma: float,
    ns_ratio: float,
    n_dropped: int,
    n_events: int,
) -> Tuple[np.ndarray, float]:
    """Docstring"""
    sob_t_sig = ONE_OVER_ROOT_TWO_PI * one_over_sigma * np.exp(
            (-0.5) * ((times - mu) * one_over_sigma)**2)
    sob_t = sob_t_sig * sob_t_bg

    sob = sob_se * sob_t
    bg_term = (1 - ns_ratio) / (livetime * ns_ratio)

    drop_term = n_dropped * (np.log(bg_term) - 1)
    ns_term = n_events * np.log(ns_ratio)
    sob_term = np.sum(np.log(sob_t + bg_term * bos_se))

    return sob / (sob + bg_term), ns_term + drop_term + sob_term


@njit
def _calculate_maximization(
    responsibility_matrix: np.ndarray,
    times: np.ndarray,
    n_events: int,
) -> Tuple[float, float, float]:
    """Docstring"""
    ns_fit = np.sum(responsibility_matrix)
    mu_fit = np.sum(responsibility_matrix * times) / ns_fit
    one_over_sigma_fit = ns_fit / np.sum(responsibility_matrix * (times - mu_fit)**2)
    ns_ratio_fit = ns_fit / n_events
    return mu_fit, one_over_sigma_fit, ns_ratio_fit


class FlareExpMaxLLHTestStatistic(NonMinimizingLLHTestStatistic):
    """Docstring"""
    _time_term_name: str
    _window_start: float
    _window_length: float
    _expmax_precision: float
    _expmax_iterations: int
    _injector: TimeDependentNuSourcesInjector

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
        time_params = self.sob_terms[self._time_term_name].params
        if 'mean' not in time_params or 'sigma' not in time_params:
            raise TypeError('Only mla.GaussProfile is currently supported')

        livetime = self._injector.contained_livetime(
            self._window_start, self._window_start + self._window_length)

        if ns_ratio is None:
            ns_ratio = 0

        llh, mu, sigma, ns_ratio = _expectation_maximization(
            sob,
            self.sob_terms[self._time_term_name].times,
            self.sob_terms[self._time_term_name]._sob,
            livetime,
            self.n_dropped,
            self.n_events,
            self.params['mean'],
            self.params['sigma'],
            ns_ratio,
            self._expmax_precision,
            self._expmax_iterations,
        )

        ts = -2 * llh

        if ts < self._best_ts:
            self._best_time_params['mean'] = mu
            self._best_time_params['sigma'] = sigma

        return ns_ratio, ts


class FlareExpMaxLLHTestStatisticFactory(NonMinimizingLLHTestStatisticFactory):
    """Docstring"""
    factory_of: ClassVar = FlareExpMaxLLHTestStatistic
    injector: TimeDependentNuSourcesInjector

    time_term_name: str = 'TimeTerm'
    window_start: float = 1
    window_length: float = 1
    expmax_precision: float = 1e-5
    expmax_iterations: int = 50

    def _factory_kwargs(self) -> dict:
        """Docstring"""
        return {
            **super()._factory_kwargs(),
            '_time_term_name': self.time_term_name,
            '_window_start': self.window_start,
            '_window_length': self.window_length,
            '_injector': self.injector,
            '_expmax_precision': self.expmax_precision,
            '_expmax_iterations': self.expmax_iterations,
        }
