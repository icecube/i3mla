"""Docstring"""

__author__ = 'John Evans'
__copyright__ = 'Copyright 2021 John Evans'
__credits__ = ['John Evans', 'Jason Fan', 'Michael Larson']
__license__ = 'Apache License 2.0'
__version__ = '0.0.1'
__maintainer__ = 'John Evans'
__email__ = 'john.evans@icecube.wisc.edu'
__status__ = 'Development'

from typing import List, Tuple
from typing import TYPE_CHECKING

import dataclasses
import numpy as np

if TYPE_CHECKING:
    from . import sources
    from . import _models
    from . import sob_terms
else:
    sources = object  # pylint: disable=invalid-name
    _models = object  # pylint: disable=invalid-name
    sob_terms = object  # pylint: disable=invalid-name


@dataclasses.dataclass
class LLHTestStatistic:
    """Docstring"""
    _sob_terms: List[sob_terms.SoBTerm]
    _n_events: int
    _n_kept: int
    _events: np.ndarray
    _params: np.ndarray
    _bounds: sob_terms.Bounds
    _best_ts: float = dataclasses.field(init=False, default=0)
    _best_ns: float = dataclasses.field(init=False, default=0)

    def __call__(self, **kwargs) -> float:
        """Evaluates the test-statistic for the given events and parameters

        Calculates the test-statistic using a given event model, n_signal, and
        gamma. This function does not attempt to fit n_signal or gamma.

        Returns:
            The overall test-statistic value for the given events and
            parameters.
        """
        if self._n_events == 0:
            return 0

        sob = self._calculate_sob()

        if 'ns' in self.params.dtype.names:
            ns_ratio = self.params['ns'] / self._n_events
        else:
            ns_ratio = self._newton_ns_ratio(sob, **kwargs)

        llh, drop_term = self._calculate_llh(sob, ns_ratio)
        ts = -2 * (llh.sum() + self.n_dropped * drop_term)

        if ts < self._best_ts:
            self._best_ts = ts
            self._best_ns = ns_ratio * self._n_events

        return ts

    def _calculate_sob(self) -> np.ndarray:
        """Docstring"""
        sob = np.ones(self.n_kept)
        for term in self._sob_terms:
            sob *= term.sob.reshape((-1,))
        return sob

    def _newton_ns_ratio(
        self,
        sob: np.ndarray,
        precision: float = 0,
        newton_iterations: int = 20,
        **kwargs,
    ) -> float:
        """Docstring

        Args:
            sob:
            precision:
            newton_iterations:

        Returns:

        """
        # kwargs no-op
        len(kwargs)

        precision += 1
        eps = 1e-5
        k = 1 / (sob - 1)
        x = [0] * newton_iterations

        for i in range(newton_iterations - 1):
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
        return x[-1]

    @staticmethod
    def _calculate_llh(
        sob: np.ndarray,
        ns_ratio: float,
    ) -> Tuple[np.ndarray, float]:
        """Docstring"""
        return (
            np.sign(ns_ratio) * np.log(np.abs(ns_ratio) * (sob - 1) + 1),
            np.sign(ns_ratio) * np.log(1 - np.abs(ns_ratio)),
        )

    @property
    def params(self) -> np.ndarray:
        """Docstring"""
        return self._params

    @params.setter
    def params(self, params: np.ndarray) -> None:
        """Docstring"""
        if params == self._params:
            return
        for term in self._sob_terms:
            term.params = params
        self._params = params
        self._best_ns = self._best_ts = 0

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

    def _fix_bounds(
        self,
        bnds: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """Docstring"""
        if 'ns' in self._params.dtype.names:
            i = self._params.dtype.names.index('ns')
            bnds[i] = (0, min(bnds[i][1], self._n_kept))
        return bnds

    @property
    def bounds(self) -> sob_terms.Bounds:
        """Docstring"""
        if self._bounds is None:
            self._bounds = [(-np.inf, np.inf)] * len(self._params.dtype.names)
        return self._fix_bounds(self._bounds)


@dataclasses.dataclass
class LLHTestStatisticFactory:
    """Docstring"""
    sob_term_factories: List[sob_terms.SoBTermFactory]

    def __call__(
        self,
        params: np.ndarray,
        events: np.ndarray,
        event_model: _models.EventModel,
        source: sources.Source,
        bounds: sob_terms.Bounds = None,
    ) -> LLHTestStatistic:
        """Docstring"""
        drop_mask = np.logical_and.reduce(np.array([
            term_factory.calculate_drop_mask(events, source)
            for term_factory in self.sob_term_factories
        ]))

        for term_factory in self.sob_term_factories:
            bounds = term_factory.update_bounds(bounds)

        n_kept = drop_mask.sum()
        pruned_events = np.empty(n_kept, dtype=events.dtype)
        pruned_events[:] = events[drop_mask]

        sob_terms = [
            term_factory(params, bounds, pruned_events, event_model, source)
            for term_factory in self.sob_term_factories
        ]

        return LLHTestStatistic(
            _sob_terms=sob_terms,
            _n_events=len(events),
            _n_kept=n_kept,
            _events=pruned_events,
            _params=params,
            _bounds=bounds,
        )
