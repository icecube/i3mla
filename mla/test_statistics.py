"""Docstring"""

__author__ = 'John Evans'
__copyright__ = 'Copyright 2020 John Evans'
__credits__ = ['John Evans', 'Jason Fan', 'Michael Larson']
__license__ = 'Apache License 2.0'
__version__ = '0.0.1'
__maintainer__ = 'John Evans'
__email__ = 'john.evans@icecube.wisc.edu'
__status__ = 'Development'

from typing import Callable, List, Optional, Sequence, Tuple

import abc
import dataclasses
import warnings
import numpy as np
import numpy.lib.recfunctions as rf

from . import sources
from . import _models
from . import models
from . import time_profiles


Bounds = Optional[Sequence[Tuple[float, float]]]


@dataclasses.dataclass
class LLHTestStatistic:
    """Docstring"""
    _sob_terms: List['SoBTerm']
    _n_events: int = dataclasses.field(init=False)
    _n_dropped: int = dataclasses.field(init=False)
    _events: np.ndarray = dataclasses.field(init=False)
    _params: np.ndarray = dataclasses.field(init=False)
    _drop_index: np.ndarray = dataclasses.field(init=False)
    _best_ts: float = dataclasses.field(init=False)
    _best_ns: float = dataclasses.field(init=False)
    _bounds: Bounds = dataclasses.field(init=False)

    def preprocess(
        self,
        params: np.ndarray,
        events: np.ndarray,
        event_model: _models.EventModel,
        source: sources.Source,
        bounds: Bounds = None,
    ) -> None:
        """Docstring"""
        drop_index = np.empty(len(events))
        for term in self.sob_terms:
            drop_index, bounds = term.preprocess(
                params,
                bounds,
                events,
                event_model,
                source,
                drop_index,
            )

        self._bounds = bounds
        self._params = params
        self._drop_index = drop_index
        self._n_events = len(events)
        self._n_dropped = drop_index.sum()
        self._best_ns = 0
        self._best_ts = 0

    def __call__(self, unstructured_params: np.ndarray, **kwargs) -> float:
        """Docstring"""
        if self._n_events == 0:
            return 0

        structured_params = rf.unstructured_to_structured(
            unstructured_params,
            dtype=self._params.dtype,
            copy=True,
        )

        return self._evaluate(structured_params, **kwargs)

    def _evaluate(self, params: np.ndarray, **kwargs) -> float:
        """Evaluates the test-statistic for the given events and parameters

        Calculates the test-statistic using a given event model, n_signal, and
        gamma. This function does not attempt to fit n_signal or gamma.

        Args:
            params: An array containing (*time_params, gamma).

        Returns:
            The overall test-statistic value for the given events and
            parameters.
        """
        sob = self._sob(params)

        if 'ns' in params.dtype.names:
            ns_ratio = params['ns'] / self._n_events
        else:
            ns_ratio = self._newton_ns_ratio(sob, **kwargs)

        llh, drop_term = self._llh(sob, ns_ratio)
        ts = -2 * (llh.sum() + self._n_dropped * drop_term)

        if ts < self._best_ts:
            self._best_ts = ts
            self._best_ns = ns_ratio * self._n_events

        return ts

    def _sob(self, params: np.ndarray) -> np.ndarray:
        """Docstring"""
        sob = np.ones(self._n_events - self._n_dropped)

        for term in self._sob_terms:
            sob *= term(params, self._events, self._drop_index)

        return sob

    def _newton_ns_ratio(
        self,
        sob: np.ndarray,
        newton_iterations: int = 20,
        ns_lower_bound: Optional[float] = None,
        **kwargs,
    ) -> float:
        """Docstring

        Args:
            sob:
            iterations:

        Returns:

        """
        # kwargs no-op
        len(kwargs)

        eps = 1e-5
        k = 1 / (sob - 1)
        x = [0] * newton_iterations

        if ns_lower_bound is not None:
            lo = ns_lower_bound / self._n_events
        else:
            lo = max(-1 + eps, 1 / (1 - np.max(sob)))

        for i in range(newton_iterations - 1):
            # get next iteration and clamp
            inv_terms = x[i] + k
            inv_terms[inv_terms == 0] = eps
            terms = 1 / inv_terms
            drop_term = 1 / (x[i] - 1)
            d1 = terms.sum() + self._n_dropped * drop_term
            d2 = terms.sum()**2 + self._n_dropped * drop_term**2
            x[i + 1] = min(1 - eps, max(lo, x[i] + d1 / d2))

        return x[-1]

    def _llh(
        self,
        sob: np.ndarray,
        ns_ratio: float,
    ) -> Tuple[np.ndarray, float]:
        """Docstring"""
        return (
            np.sign(ns_ratio) * np.log(np.abs(ns_ratio) * (sob - 1) + 1),
            np.sign(ns_ratio) * np.log(1 - np.abs(ns_ratio)),
        )

    @property
    def best_ns(self) -> float:
        """Docstring"""
        return self._best_ns

    @property
    def n_kept(self) -> float:
        """Docstring"""
        return self._n_events - self._n_dropped

    def _fix_bounds(
        self,
        bnds: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """Docstring"""
        if 'ns' in self._params.dtype.names:
            i = self._params.dtype.names.index('ns')
            bnds[i] = (
                max(bnds[i][0], -self._n_events + self._n_dropped),
                min(bnds[i][1], self._n_events - self._n_dropped),
            )
        return bnds

    @property
    def bounds(self) -> Bounds:
        """Docstring"""
        if self._bounds is None:
            self._bounds = [(-np.inf, np.inf)] * len(self._params.dtype.names)
        return self._fix_bounds(self._bounds)


@dataclasses.dataclass
class SoBTerm:
    """Docstring"""
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def preprocess(
        self,
        params: np.ndarray,
        bounds: Bounds,
        events: np.ndarray,
        event_model: _models.EventModel,
        source: sources.Source,
        drop_index: np.ndarray,
    ) -> Tuple[np.ndarray, Bounds]:
        """Docstring"""

    @abc.abstractmethod
    def __call__(
        self,
        params: np.ndarray,
        events: np.ndarray,
        drop_index: np.ndarray,
    ) -> np.ndarray:
        """Docstring"""


@dataclasses.dataclass
class SpatialTerm(SoBTerm):
    """Docstring"""
    _sob_spatial: np.ndarray = dataclasses.field(init=False)

    def preprocess(
        self,
        params: np.ndarray,
        bounds: Bounds,
        events: np.ndarray,
        event_model: _models.EventModel,
        source: sources.Source,
        drop_index: np.ndarray,
    ) -> Tuple[np.ndarray, Bounds]:
        """Docstring"""
        self._sob_spatial = event_model.signal_spatial_pdf(source, events)
        drop_index *= self._sob_spatial != 0

        self._sob_spatial[drop_index] /= event_model.background_spatial_pdf(
            events[drop_index],
        )

        return drop_index, bounds

    def __call__(
        self,
        params: np.ndarray,
        events: np.ndarray,
        drop_index: np.ndarray,
    ) -> np.ndarray:
        """Docstring"""
        return self._sob_spatial[drop_index]


@dataclasses.dataclass
class TimeTerm(SoBTerm):
    """Docstring"""
    background_time_profile: time_profiles.GenericProfile
    signal_time_profile: time_profiles.GenericProfile
    _sob_time: np.ndarray = dataclasses.field(init=False)
    _sob_time_complete: np.ndarray = dataclasses.field(init=False)
    _times: np.ndarray = dataclasses.field(init=False)

    def preprocess(
        self,
        params: np.ndarray,
        bounds: Bounds,
        events: np.ndarray,
        event_model: _models.EventModel,
        source: sources.Source,
        drop_index: np.ndarray,
    ) -> Tuple[np.ndarray, Bounds]:
        """Docstring"""
        self._times = np.empty(drop_index.sum(), dtype=events['time'].dtype)
        self._sob_time = np.zeros(len(events))

        self._sob_time[drop_index] = 1 / self.bg_time_profile.pdf(
            self._times[drop_index],
        )

        if np.logical_not(np.all(np.isfinite(self._sob_time))):
            warnings.warn(
                'Warning, events outside background time profile',
                RuntimeWarning
            )

        self._sob_time_complete = self._sob_time.copy()

        self._sob_time_complete[drop_index] *= self.signal_time_profile.pdf(
            self._times[drop_index]
        )

        return drop_index, bounds

    def __call__(
        self,
        params: np.ndarray,
        events: np.ndarray,
        drop_index: np.ndarray,
    ) -> np.ndarray:
        """Docstring"""
        if self.signal_time_profile.update_params(params):
            return self._sob_time[drop_index] * self.signal_time_profile.pdf(
                self._times[drop_index],
            )
        else:
            return self._sob_time_complete[drop_index]


@dataclasses.dataclass
class I3EnergyTerm(SoBTerm):
    """Docstring"""
    _spline_idxs: np.ndarray = dataclasses.field(init=False)
    _splines: np.ndarray = dataclasses.field(init=False)
    _energy_sob: Callable = dataclasses.field(init=False)
    gamma: float = -2

    def preprocess(
        self,
        params: np.ndarray,
        bounds: Bounds,
        events: np.ndarray,
        event_model: models.I3EventModel,
        source: sources.Source,
        drop_index: np.ndarray,
    ) -> Tuple[np.ndarray, Bounds]:
        """Docstring"""
        self._energy_sob = event_model.get_sob_energy

        self._spline_idxs, self._splines = event_model.log_sob_spline_prepro(
            events,
        )

        return drop_index, bounds

    def __call__(
        self,
        params: np.ndarray,
        events: np.ndarray,
        drop_index: np.ndarray,
    ) -> np.ndarray:
        """Docstring"""
        if 'gamma' in params.dtype.names:
            gamma = params['gamma']
        else:
            gamma = self.gamma

        return self._energy_sob(
            gamma,
            drop_index,
            self._splines,
            self._spline_idxs,
        )
