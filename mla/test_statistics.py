"""Docstring"""

__author__ = 'John Evans'
__copyright__ = 'Copyright 2021 John Evans'
__credits__ = ['John Evans', 'Jason Fan', 'Michael Larson']
__license__ = 'Apache License 2.0'
__version__ = '0.0.1'
__maintainer__ = 'John Evans'
__email__ = 'john.evans@icecube.wisc.edu'
__status__ = 'Development'

from typing import ClassVar, Dict, List, Optional

import dataclasses
import numpy as np

from .configurable import Configurable
from .sob_terms import SoBTerm, SoBTermFactory
from .events import Events
from .params import Params


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
            ns_ratio = self._newton_ns_ratio(sob)

        ts = self._calculate_ts(ns_ratio, sob)

        if ts < self._best_ts:
            self._best_ts = ts
            self._best_ns = ns_ratio * self._n_events

        return ts

    def _calculate_ts(ns_ratio: float, sob: np.ndarray) -> float:
        """Docstring"""
        llh = np.sign(ns_ratio) * np.log(np.abs(ns_ratio) * (sob - 1) + 1)
        drop_term = np.sign(ns_ratio) * np.log(1 - np.abs(ns_ratio))
        return -2 * (llh.sum() + self.n_dropped * drop_term)

    def _calculate_sob(self) -> np.ndarray:
        """Docstring"""
        sob = np.ones(self.n_kept)
        for _, term in self.sob_terms.items():
            sob *= term.sob.reshape((-1,))
        return sob

    def _newton_ns_ratio(self, sob: np.ndarray) -> float:
        """Docstring

        Args:
            sob:

        Returns:

        """
        precision = self._newton_precision + 1
        eps = 1e-5
        k = 1 / (sob - 1)
        x = [0.] * self._newton_iterations

        for i in range(self._newton_iterations - 1):
            # get next iteration and clamp
            inv_terms = x[i] + k
            inv_terms[inv_terms == 0] = eps
            terms = 1 / inv_terms
            drop_term = 1 / (x[i] - 1)
            d1 = np.sum(terms) + self.n_dropped * drop_term
            d2 = np.sum(terms**2) + self.n_dropped * drop_term**2
            x[i + 1] = min(1 - eps, max(0, x[i] + d1 / d2))

            if x[i] == x[i + 1] or (
                x[i] < x[i + 1] and x[i + 1] <= x[i] * precision
            ) or (x[i + 1] < x[i] and x[i] <= x[i + 1] * precision):
                break
        return x[i + 1]

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
    """Docstring"""
    sob_term_factories: List[SoBTermFactory]

    _factory_of: ClassVar = LLHTestStatistic
    _config_map: ClassVar[dict] = {
        '_newton_precision': ('Newton Method n_s Precision', 0),
        '_newton_iterations': ('Newton Method n_s Iterations', 20),
    }

    _newton_precision: float = 0
    _newton_iterations: int = 20

    @classmethod
    def from_config(
        cls,
        config: dict,
        sob_term_factories: List[SoBTermFactory],
    ) -> 'LLHTestStatisticFactory':
        """Docstring"""
        return cls(sob_term_factories=sob_term_factories, **cls._map_kwargs(config))

    def __call__(self, params: Params, events: Events) -> LLHTestStatistic:
        """Docstring"""
        drop_mask = np.logical_and.reduce(np.array([
            term_factory.calculate_drop_mask(events)
            for term_factory in self.sob_term_factories
        ]))

        n_kept = drop_mask.sum()
        pruned_events = events.from_idx(drop_mask)

        sob_terms = {
            term_factory.name: term_factory(params, pruned_events)
            for term_factory in self.sob_term_factories
        }

        return self.__class__._factory_of(
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
            '_newton_iterations': self._newton_iterations,
            '_newton_precision': self._newton_precision,
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
    _min_length: float
    _time_term_name: str

    _ts_dict: dict[tuple[float, float], float] = dataclasses.field(
        init=False, default_factory=dict())

    def _calculate_ts(ns_ratio: float, sob: np.ndarray) -> float:
        """Docstring"""
        self._ts_dict = {}
        time_params = self.sob_terms[self._time_term_name].params
        if 'start' not in time_params or 'length' not in time_params:
            raise TypeError('Only mla.UniformProfile is currently supported')
        
        edges = self._events.time[sob >= self._min_sob]
        
        for i, start in enumerate(edges):
            for end in edges[i+1:]:
                length = end - start
                if length >= self._min_length:
                    self._ts_dict[(start, length)] = np.nan

        for start, length in self._ts_dict:
            time_params['start'] = start
            time_params['length'] = length
            self.sob_terms[self._time_term_name].params = time_params
            self._ts_dict[(start, length)] = super()._calculate_ts(ns_ratio, sob)

        return max(self._ts_dict.values())

@dataclasses.dataclass(kw_only=True)
class FlareStackLLHTestStatisticFactory(LLHTestStatisticFactory, Configurable):
    """Docstring"""
    _factory_of: ClassVar = FlareStackLLHTestStatistic
    _config_map: ClassVar[dict] = {
        **LLHTestStatisticFactory._config_map,
        '_min_sob': ('Minimum Signal-over-background Ratio For Flare', 1),
        '_min_length': ('Minimum Flare Duration (days)', 1),
        '_time_term_name': ('Time Term Name', 'TimeTerm'),
    }

    _min_sob: float = 1
    _min_length: float = 1
    _time_term_name: str = 'TimeTerm'

    def _factory_kwargs() -> dict:
        """Docstring"""
        return {
            **super()._factory_kwargs(),
            '_min_sob': self._min_sob,
            '_min_length': self._min_length,
            '_time_term_name': self._time_term_name,
        }
