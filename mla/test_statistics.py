"""Docstring"""

__author__ = 'John Evans'
__copyright__ = 'Copyright 2021 John Evans'
__credits__ = ['John Evans', 'Jason Fan', 'Michael Larson']
__license__ = 'Apache License 2.0'
__version__ = '0.0.1'
__maintainer__ = 'John Evans'
__email__ = 'john.evans@icecube.wisc.edu'
__status__ = 'Development'

from typing import Dict, List, Optional

import dataclasses
import numpy as np

from . import configurable
from .sob_terms import SoBTerm, SoBTermFactory
from .params import Params


@dataclasses.dataclass
class LLHTestStatistic():
    """Docstring"""
    sob_terms: Dict[str, SoBTerm]
    _n_events: int
    _n_kept: int
    _events: np.ndarray
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

        llh = np.sign(ns_ratio) * np.log(np.abs(ns_ratio) * (sob - 1) + 1)
        drop_term = np.sign(ns_ratio) * np.log(1 - np.abs(ns_ratio))
        ts = -2 * (llh.sum() + self.n_dropped * drop_term)

        if ts < self._best_ts:
            self._best_ts = ts
            self._best_ns = ns_ratio * self._n_events

        return ts

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
        return x[i+1]

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
    def n_kept(self) -> int:
        """Docstring"""
        return self._n_kept

    @property
    def n_dropped(self) -> int:
        """Docstring"""
        return self._n_events - self._n_kept


@dataclasses.dataclass
class LLHTestStatisticFactory(configurable.Configurable):
    """Docstring"""
    sob_term_factories: List[SoBTermFactory]

    def __call__(self, params: Params, events: np.ndarray) -> LLHTestStatistic:
        """Docstring"""
        drop_mask = np.logical_and.reduce(np.array([
            term_factory.calculate_drop_mask(events)
            for term_factory in self.sob_term_factories
        ]))

        n_kept = drop_mask.sum()
        pruned_events = np.empty(n_kept, dtype=events.dtype)
        pruned_events[:] = events[drop_mask]

        sob_terms = {
            term_factory.config['name']: term_factory(params, pruned_events)
            for term_factory in self.sob_term_factories
        }

        return LLHTestStatistic(
            sob_terms=sob_terms,
            _n_events=len(events),
            _n_kept=n_kept,
            _events=pruned_events,
            _params=params,
            _newton_precision=self.config['newton_precision'],
            _newton_iterations=self.config['newton_iterations'],
        )

    def generate_params(self) -> Params:
        """Docstring"""
        param_values = {'ns': 0}
        param_bounds = {'ns': (0, np.inf)}

        for term in self.sob_term_factories:
            vals, bounds = term.generate_params()
            param_values = dict(param_values, **vals)
            param_bounds = dict(param_bounds, **bounds)

        return Params.from_dict(param_values, param_bounds)

    @classmethod
    def generate_config(cls) -> dict:
        """Docstring"""
        config = super().generate_config()
        config['newton_precision'] = 0
        config['newton_iterations'] = 20
        return config
