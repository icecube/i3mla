"""Docstring"""

__author__ = 'John Evans'
__copyright__ = 'Copyright 2021 John Evans'
__credits__ = ['John Evans', 'Jason Fan', 'Michael Larson']
__license__ = 'Apache License 2.0'
__version__ = '0.0.1'
__maintainer__ = 'John Evans'
__email__ = 'john.evans@icecube.wisc.edu'
__status__ = 'Development'

from typing import List, Optional
from dataclasses import dataclass, field, InitVar

import numpy as np

from .configurable import Configurable
from .params import Params
from .sources import PointSource
from .test_statistics import LLHTestStatisticFactory, NonMinimizingLLHTestStatisticFactory
from .trial_generators import SingleSourceTrialGenerator, StackedTrialGenerator
from .events import Events


@dataclass(kw_only=True)
class SingleSourceLLHAnalysis:
    """Docstring"""
    test_statistic_factory: LLHTestStatisticFactory
    trial_generator: SingleSourceTrialGenerator

    def produce_and_minimize(
        self,
        params: Params,
        fitting_params: Optional[List[str]] = None,
        n_signal: float = 0,
    ) -> dict:
        """Docstring"""
        return self.minimize_trial(
            self.produce_trial(n_signal=n_signal), params, fitting_params)

    def produce_trial(self, n_signal: float = 0) -> Events:
        return self.trial_generator(n_signal=n_signal)

    def minimize_trial(
        self,
        trial: Events,
        params: Params,
        fitting_params: Optional[List[str]] = None,
    ) -> dict:
        test_statistic = self.test_statistic_factory(params, trial)
        return test_statistic.minimize(fitting_params)

    def generate_params(self) -> Params:
        """Docstring"""
        return self.test_statistic_factory.generate_params()


@dataclass(kw_only=True)
class StackingLLHAnalysis(Configurable):
    """Docstring"""
    test_statistic_factories: dict[str, NonMinimizingLLHTestStatisticFactory]
    sources: dict[str, PointSource]
    trial_generator: StackedTrialGenerator
    gammas: InitVar[list[float]]
    _weights: dict[float, dict[str, float]] = field(init=False, repr=False)
    weight_generation_trials: int = 100
    n_signal_for_weights: float = 1e-12

    def __post_init__(self, gammas: list[float]) -> None:
        """Docstring"""
        self._generate_weights(gammas)

    def produce_trial(self, n_signals: dict[str, float]) -> Events:
        """Docstring"""
        return self.trial_generator(n_signals)

    def evaluate_trial(self, trial: Events, gamma: float) -> dict:
        """Docstring"""
        params_dict = self.generate_params_dict(gamma)

        test_statistics = {
            key: ts_factory(params_dict[key], trial)
            for key, ts_factory in self.test_statistic_factories.items()
        }

        ts_dicts = {key: ts.minimize() for key, ts in test_statistics.items()}
        overall_ts = 0
        total_ns = 0

        for key, ts_dict in ts_dicts.items():
            overall_ts += ts_dict['ts'] * self._weights[gamma][key]
            total_ns += ts_dict['ns']

        return {'overall ts': overall_ts, 'total ns': total_ns, 'individual ts': ts_dicts}

    def generate_params_dict(self, gamma: float) -> dict[str, Params]:
        """Docstring"""
        params = {
            key: ts_factory.generate_params()
            for key, ts_factory in self.test_statistic_factories.items()
        }

        for key in self.test_statistic_factories:
            params[key]['gamma'] = gamma

        return params

    def _generate_weights(self, gammas: list[float]) -> None:
        """Docstring"""
        self._weights = {}
        for gamma in gammas:
            average_ns_dict = {
                    key: self._single_source_n_trials_average_ns(ts_factory, gamma)
                for key, ts_factory in self.test_statistic_factories.items()
            }
            total_average_ns = sum(list(average_ns_dict.values()))
            weights_per_gamma = {
                key: average_ns / total_average_ns
                for key, average_ns in average_ns_dict.items()
            }
            self._weights[gamma] = weights_per_gamma

    def _single_source_n_trials_average_ns(
        self,
        ts_factory: NonMinimizingLLHTestStatisticFactory,
        gamma: float,
    ) -> float:
        """Docstring"""
        params = ts_factory.generate_params()
        params['gamma'] = gamma
        ns = np.empty(self.weight_generation_trials)

        n_signals = {
            key: self.n_signal_for_weights
            for key in self.test_statistic_factories
        }

        for i in range(self.weight_generation_trials):
            events = self.produce_trial(n_signals)
            ts = ts_factory(params, events)
            ts_dict = ts.minimize()
            ns[i] = ts_dict['ns']

        return float(np.average(ns))
