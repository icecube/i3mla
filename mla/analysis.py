"""Docstring"""

__author__ = 'John Evans'
__copyright__ = 'Copyright 2021 John Evans'
__credits__ = ['John Evans', 'Jason Fan', 'Michael Larson']
__license__ = 'Apache License 2.0'
__version__ = '0.0.1'
__maintainer__ = 'John Evans'
__email__ = 'john.evans@icecube.wisc.edu'
__status__ = 'Development'

from typing import List, Optional, Tuple
from dataclasses import InitVar, dataclass, field

from .data_handlers import DataHandler, Injector
from .params import Params
from .sources import PointSource
from .test_statistics import LLHTestStatisticFactory
from .trial_generators import SingleSourceTrialGenerator
from .events import Events


@dataclass(kw_only=True)
class SingleSourceLLHAnalysis:
    """Docstring"""
    test_statistic_factory: LLHTestStatisticFactory
    injector: Injector
    source: PointSource
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
